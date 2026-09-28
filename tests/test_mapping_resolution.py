"""resolve_setup_from_mapping: the data-free twin of from_mapping + setup()."""

import copy

import pytest

from captest import resolve_setup_from_mapping
from captest.captest import TEST_SETUPS, resolve_test_setup

FDR = {"power": {"group": "real_pwr_mtr", "agg": "mean"}}


class TestResolveSetupFromMapping:
    @pytest.mark.parametrize("name", sorted(TEST_SETUPS))
    def test_bare_preset_is_the_preset(self, name):
        assert resolve_setup_from_mapping({"test_setup": name}) is TEST_SETUPS[name]

    def test_empty_overrides_keep_the_preset(self):
        # The pfcli new-project template writes `overrides: {rep_conditions: {}}`.
        sub = {"test_setup": "e2848_default", "overrides": {"rep_conditions": {}}}
        assert resolve_setup_from_mapping(sub) is TEST_SETUPS["e2848_default"]

    def test_reg_cols_override_merges_key_by_key(self):
        sub = {"test_setup": "e2848_default", "overrides": {"reg_cols_meas": FDR}}
        got = resolve_setup_from_mapping(sub)
        assert got == resolve_test_setup("e2848_default", {"reg_cols_meas": FDR})
        assert got.semantic_digest() != TEST_SETUPS["e2848_default"].semantic_digest()

    def test_top_level_keys_match_overrides(self):
        via_overrides = {
            "test_setup": "e2848_default",
            "overrides": {"reg_cols_meas": FDR},
        }
        top_level = {"test_setup": "e2848_default", "reg_cols_meas": FDR}
        assert resolve_setup_from_mapping(top_level) == resolve_setup_from_mapping(
            via_overrides
        )

    def test_null_test_setup_uses_default(self):
        from captest import CapTest

        default = CapTest.param["test_setup"].default
        assert resolve_setup_from_mapping({"test_setup": None}) is TEST_SETUPS[default]

    def test_scatter_plots_override(self):
        sub = {"test_setup": "e2848_default", "overrides": {"scatter_plots": "etotal"}}
        assert resolve_setup_from_mapping(sub).scatter_plots == "etotal"

    def test_custom(self):
        doc = TEST_SETUPS["e2848_default"].to_dict()
        sub = {
            "test_setup": "custom",
            "overrides": {
                "reg_fml": doc["reg_fml"],
                "reg_cols_meas": doc["meas"]["reg_cols"],
                "reg_cols_sim": doc["sim"]["reg_cols"],
            },
        }
        got = resolve_setup_from_mapping(sub)
        assert got.name == "custom"
        assert got.meas == TEST_SETUPS["e2848_default"].meas

    def test_opens_no_data(self):
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "/nonexistent/meas.csv",
            "sim_path": "relative/sim.csv",
            "meas_prep": [{"type": "Scale", "group": "real_pwr_mtr", "factor": 2.0}],
        }
        assert resolve_setup_from_mapping(sub) is TEST_SETUPS["e2848_default"]

    def test_does_not_mutate_input(self):
        sub = {"test_setup": "e2848_default", "overrides": {"reg_cols_meas": FDR}}
        before = copy.deepcopy(sub)
        resolve_setup_from_mapping(sub)
        assert sub == before

    def test_matches_captest_setup(self, ct_default):
        ct_default.reg_cols_meas = FDR
        ct_default.setup(verbose=False)
        mapping = ct_default.to_mapping()
        assert resolve_setup_from_mapping(mapping) == ct_default.resolved_setup

    @pytest.mark.parametrize(
        ("sub", "exc"),
        [
            ({"test_setup": "nope"}, KeyError),
            ({"test_setup": "e2848_default", "bogus": 1}, ValueError),
            ({"meas_path": "x.csv"}, ValueError),
            (
                {
                    "test_setup": "e2848_default",
                    "reg_fml": "power ~ poa - 1",
                    "overrides": {"reg_fml": "power ~ poa - 1"},
                },
                ValueError,
            ),
            (
                {
                    "test_setup": "e2848_default",
                    "overrides": {"reg_cols_meas": {"power": {"group": 1}}},
                },
                ValueError,
            ),
            ({"test_setup": "custom", "overrides": {"reg_fml": "a ~ b"}}, ValueError),
        ],
    )
    def test_errors(self, sub, exc):
        with pytest.raises(exc):
            resolve_setup_from_mapping(sub)
