"""Tests for captest.captest module.

Covers the TEST_SETUPS registry, the validate_test_setup / resolve_test_setup
/ load_config helpers, the three shipped scatter-plot callables, and the
``CapTest`` orchestrator class itself (Units 4 and 6 of the implementation
plan).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from captest import CapTest, captest as ct
from captest.calcparams import (
    apparent_zenith_pvsyst,
    cell_temp,
    e_total,
    poa_spec_corrected,
    power_temp_correct,
    spectral_factor_firstsolar,
)

# Presets that the shipped meas_cd_default / sim_cd_default fixtures cover
# end-to-end through CapTest.setup(). Extend this list when you add a preset
# whose required column_groups and site attributes are satisfied by the
# default fixtures; otherwise add a dedicated fixture and a targeted test.
_DEFAULT_FIXTURE_PRESETS = [
    p
    for p in ct.TEST_SETUPS.keys()
    if p not in {"e2848_spec_corrected_poa", "bifi_power_tc_meas_tbom"}
]


class TestTestSetupsRegistry:
    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_shipped_preset_validates(self, preset):
        """Each shipped preset dict passes validate_test_setup as-is."""
        ct.validate_test_setup(ct.TEST_SETUPS[preset])

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_preset_lhs_is_power(self, preset):
        """Naming convention: lhs regression variable is always 'power'."""
        entry = ct.TEST_SETUPS[preset]
        lhs = entry["reg_fml"].split("~")[0].strip()
        assert lhs == "power", f"preset {preset!r} lhs is {lhs!r}, expected 'power'"

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_preset_has_rep_conditions_dict(self, preset):
        rc = ct.TEST_SETUPS[preset]["rep_conditions"]
        assert isinstance(rc, dict)
        assert "func" in rc
        assert isinstance(rc["func"], dict)

    def test_e2848_default_shape(self):
        """Sanity-check the default preset's reg_cols keys."""
        entry = ct.TEST_SETUPS["e2848_default"]
        assert set(entry["reg_cols_meas"].keys()) == {"power", "poa", "t_amb", "w_vel"}
        assert set(entry["reg_cols_sim"].keys()) == {"power", "poa", "t_amb", "w_vel"}

    def test_bifi_e2848_etotal_rear_shade_sim_uses_e_total(self):
        """bifi_e2848_etotal_rear_shade_sim preset wraps poa in an e_total calc-tuple."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim"]
        meas_poa = entry["reg_cols_meas"]["poa"]
        assert isinstance(meas_poa, tuple)
        assert meas_poa[0] is e_total

    def test_bifi_power_tc_calc_tbom_uses_power_temp_correct(self):
        entry = ct.TEST_SETUPS["bifi_power_tc_calc_tbom"]
        meas_power = entry["reg_cols_meas"]["power"]
        assert isinstance(meas_power, tuple)
        assert meas_power[0] is power_temp_correct

    def test_bifi_power_tc_meas_tbom_uses_measured_bom(self):
        """meas-side BOM temp is a direct column-group ref, not a calc tuple."""
        entry = ct.TEST_SETUPS["bifi_power_tc_meas_tbom"]
        meas_power = entry["reg_cols_meas"]["power"]
        assert isinstance(meas_power, tuple)
        assert meas_power[0] is power_temp_correct
        cell_temp_node = meas_power[1]["cell_temp"]
        assert isinstance(cell_temp_node, tuple)
        assert cell_temp_node[0] is cell_temp
        # BOM temperature comes from field measurements (direct column-group ref).
        bom_spec = cell_temp_node[1]["bom"]
        assert bom_spec == ("temp_bom", "mean")

    def test_e2848_spec_corrected_poa_meas_tree_uses_spectral_factor(self):
        """The meas-side poa tree ends with spectral_factor_firstsolar."""
        entry = ct.TEST_SETUPS["e2848_spec_corrected_poa"]
        meas_poa = entry["reg_cols_meas"]["poa"]
        assert isinstance(meas_poa, tuple)
        assert meas_poa[0] is poa_spec_corrected
        spec_node = meas_poa[1]["spectral_correction"]
        assert isinstance(spec_node, tuple)
        assert spec_node[0] is spectral_factor_firstsolar

    def test_e2848_spec_corrected_poa_sim_uses_apparent_zenith_pvsyst(self):
        """The sim-side tree routes through apparent_zenith_pvsyst."""
        entry = ct.TEST_SETUPS["e2848_spec_corrected_poa"]
        sim_poa = entry["reg_cols_sim"]["poa"]
        abs_airmass_node = sim_poa[1]["spectral_correction"][1]["absolute_airmass"]
        zenith_node = abs_airmass_node[1]["apparent_zenith"]
        assert isinstance(zenith_node, tuple)
        assert zenith_node[0] is apparent_zenith_pvsyst

    def test_validate_rejects_unknown_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["bogus"] = 42
        with pytest.raises(KeyError, match="unknown keys"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_missing_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad.pop("rep_conditions")
        with pytest.raises(KeyError, match="missing required keys"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_non_callable_scatter_plots(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["scatter_plots"] = "not-a-callable"
        with pytest.raises(ValueError, match="scatter_plots"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_formula_vars_missing_from_reg_cols(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["reg_cols_meas"] = {
            k: v for k, v in bad["reg_cols_meas"].items() if k != "w_vel"
        }
        with pytest.raises(ValueError, match="missing keys required by reg_fml"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_non_dict_rep_conditions(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["rep_conditions"] = "not-a-dict"
        with pytest.raises(ValueError, match="'rep_conditions' must be a dict"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_func_with_non_rhs_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        # Reconstruct rep_conditions with an extra func key.
        rc = dict(bad["rep_conditions"])
        rc["func"] = dict(rc["func"])
        rc["func"]["bogus"] = "mean"
        bad["rep_conditions"] = rc
        with pytest.raises(ValueError, match="rhs variables"):
            ct.validate_test_setup(bad)


class TestResolveTestSetup:
    def test_named_preset_no_overrides(self):
        resolved = ct.resolve_test_setup("e2848_default")
        assert resolved["reg_fml"] == ct.TEST_SETUPS["e2848_default"]["reg_fml"]

    def test_named_preset_with_reg_fml_override(self):
        # Keep the same rhs variables (otherwise rep_conditions.func keys no
        # longer align with rhs and validate_test_setup rejects the result).
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={
                "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel)"
            },
        )
        assert resolved["reg_fml"].startswith("power ~ poa")

    def test_named_preset_with_rep_conditions_partial_merge(self):
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={"rep_conditions": {"percent_filter": 10}},
        )
        # Top-level percent_filter replaced, other keys preserved.
        assert resolved["rep_conditions"]["percent_filter"] == 10
        assert resolved["rep_conditions"]["irr_bal"] is False
        # func dict untouched.
        assert set(resolved["rep_conditions"]["func"].keys()) == {
            "poa",
            "t_amb",
            "w_vel",
        }

    def test_named_preset_with_rep_conditions_func_partial_merge(self):
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={"rep_conditions": {"func": {"poa": ct.perc_wrap(55)}}},
        )
        # POA entry swapped; others preserved.
        assert resolved["rep_conditions"]["func"]["t_amb"] == "mean"
        assert resolved["rep_conditions"]["func"]["w_vel"] == "mean"
        assert callable(resolved["rep_conditions"]["func"]["poa"])

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown test_setup"):
            ct.resolve_test_setup("nonexistent")

    def test_custom_requires_all_three_overrides(self):
        with pytest.raises(ValueError, match="test_setup='custom'"):
            ct.resolve_test_setup("custom", overrides={"reg_fml": "y ~ x"})

    def test_custom_with_minimal_overrides(self):
        resolved = ct.resolve_test_setup(
            "custom",
            overrides={
                "reg_cols_meas": {"power": "p", "poa": "i"},
                "reg_cols_sim": {"power": "p", "poa": "i"},
                "reg_fml": "power ~ poa",
            },
        )
        assert resolved["reg_fml"] == "power ~ poa"
        # scatter_plots falls back to scatter_default.
        assert resolved["scatter_plots"] is ct.scatter_default
        # rep_conditions defaults to empty dict.
        assert resolved["rep_conditions"] == {}


class TestLoadConfig:
    def test_happy_path(self, tmp_path):
        yaml_text = "captest:\n  test_setup: e2848_default\n  ac_nameplate: 125000\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p)
        assert sub["test_setup"] == "e2848_default"
        assert sub["ac_nameplate"] == 125000

    def test_missing_key_raises_with_suggestion(self, tmp_path):
        yaml_text = "captset:\n  test_setup: e2848_default\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        with pytest.raises(KeyError, match="captset"):
            ct.load_config(p)

    def test_custom_key(self, tmp_path):
        yaml_text = "captest_bifi:\n  test_setup: bifi_e2848_etotal_rear_shade_sim\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p, key="captest_bifi")
        assert sub["test_setup"] == "bifi_e2848_etotal_rear_shade_sim"

    def test_top_level_not_a_mapping_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("- just a list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            ct.load_config(p)

    def test_perc_N_string_resolved_in_overrides(self, tmp_path):
        yaml_text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_55\n"
            "        t_amb: mean\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p)
        func = sub["overrides"]["rep_conditions"]["func"]
        # 'mean' passes through.
        assert func["t_amb"] == "mean"
        # 'perc_55' resolves to perc_wrap(55).
        sample = pd.Series(np.arange(100))
        expected = ct.perc_wrap(55)(sample)
        actual = func["poa"](sample)
        assert actual == expected

    def test_malformed_perc_string_raises(self, tmp_path):
        yaml_text = (
            "captest:\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_xx\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        with pytest.raises(ValueError, match="perc_<int>"):
            ct.load_config(p)


class TestPercWrap:
    def test_returns_callable(self):
        f = ct.perc_wrap(60)
        assert callable(f)

    def test_computes_percentile(self):
        f = ct.perc_wrap(60)
        sample = pd.Series(np.arange(100))
        # method='nearest' returns the nearest existing value; compare against
        # np.percentile with the same method to avoid rounding ambiguity.
        expected = np.percentile(sample, 60, method="nearest")
        assert f(sample) == expected

    def test_name_encodes_percentile(self):
        f = ct.perc_wrap(55)
        assert f.__name__ == "perc_wrap(55)"


class TestScatterCallables:
    """Smoke tests for the shipped scatter callables using a synthetic CapData."""

    def _make_synthetic_cd(self, formula, columns):
        from captest.capdata import CapData

        cd = CapData("test")
        idx = pd.date_range("2024-01-01", periods=50, freq="1min")
        data = {col: np.linspace(1, 50, 50) for col in columns}
        cd.data = pd.DataFrame(data, index=idx)
        cd.data_filtered = cd.data.copy()
        cd.column_groups = {col: [col] for col in columns}
        cd.regression_cols = {col: col for col in columns}
        cd.regression_formula = formula
        return cd

    def test_scatter_default_returns_layout(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa - 1", ["power", "poa"])
        layout = ct.scatter_default(cd)
        assert isinstance(layout, hv.Layout)

    def test_scatter_etotal_returns_layout(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa + rpoa", ["power", "poa", "rpoa"])
        layout = ct.scatter_etotal(cd)
        assert isinstance(layout, hv.Layout)

    def test_scatter_bifi_power_tc_has_two_panels(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa + rpoa", ["power", "poa", "rpoa"])
        layout = ct.scatter_bifi_power_tc(cd)
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2


class TestConstruction:
    """Construction paths for CapTest: bare init, from_params, and kwargs."""

    def test_bare_init_has_defaults(self):
        capt = CapTest()
        assert capt.test_setup == "e2848_default"
        assert capt.meas is None
        assert capt.sim is None
        assert capt.test_tolerance == "- 4"
        assert capt.bifaciality == 0.0
        assert capt.rear_shade == 0.0
        assert capt.power_temp_coeff == -0.32
        assert capt.base_temp == 25
        assert capt.bifacial_frac == 1.0
        assert capt.module_type == "glass_cell_poly"
        assert capt.racking == "open_rack"
        assert capt.airmass_model == "kastenyoung1989"
        assert capt.altitude_override == 0
        assert capt._resolved_setup is None

    def test_bare_init_accepts_kwargs(self):
        capt = CapTest(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            ac_nameplate=125_000,
            bifaciality=0.15,
        )
        assert capt.test_setup == "bifi_e2848_etotal_rear_shade_sim"
        assert capt.ac_nameplate == 125_000
        assert capt.bifaciality == 0.15

    def test_bare_init_rejects_unknown_kwarg(self):
        with pytest.raises(TypeError):
            CapTest(bogus_kwarg=1)

    def test_class_level_downstream_attrs(self):
        assert CapTest._downstream_attrs == (
            "bifaciality",
            "bifacial_frac",
            "rear_shade",
            "power_temp_coeff",
            "base_temp",
            "module_type",
            "racking",
            "spectral_module_type",
            "airmass_model",
            "altitude_override",
        )
        assert CapTest._downstream_attrs_meas_only == ("rear_shade",)

    def test_downstream_attrs_meas_only_is_subset(self):
        """meas-only attrs must be a subset of _downstream_attrs; setup()
        iterates _downstream_attrs, so an orphaned meas-only name would never
        propagate to either CapData instance."""
        assert set(CapTest._downstream_attrs_meas_only).issubset(
            CapTest._downstream_attrs
        )

    def test_from_params_with_capdata_instances_triggers_setup(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
        )
        assert capt._resolved_setup is not None
        # process_regression_columns should have resolved the aggregated poa
        # column name on both CapData instances.
        assert capt.meas.regression_cols["poa"] == "irr_poa_mean_agg"
        assert capt.sim.regression_cols["poa"] == "GlobInc"

    def test_from_params_partial_leaves_unset_defers_setup(self, meas_cd_default):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
        )
        assert capt._resolved_setup is None
        assert capt.meas is meas_cd_default
        assert capt.sim is None

    def test_from_params_pre_built_meas_wins_over_path(
        self, meas_cd_default, sim_cd_default
    ):
        with pytest.warns(UserWarning, match="pre-built"):
            capt = CapTest.from_params(
                test_setup="e2848_default",
                meas=meas_cd_default,
                meas_path="/nonexistent/should/not/be/opened.csv",
                sim=sim_cd_default,
            )
        assert capt.meas is meas_cd_default

    def test_from_params_loads_data_via_custom_loader(self, meas_cd_default):
        meas_loader = MagicMock(return_value=meas_cd_default)
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas_path="/fake/path",
            meas_loader=meas_loader,
        )
        meas_loader.assert_called_once_with("/fake/path")
        assert capt.meas is meas_cd_default

    def test_rep_irr_filter_low_default(self):
        """rep_irr_filter_low == 1 - rep_irr_filter with the default value."""
        capt = CapTest()
        assert capt.rep_irr_filter_low == pytest.approx(1 - capt.rep_irr_filter)
        assert capt.rep_irr_filter_low == pytest.approx(0.8)

    def test_rep_irr_filter_high_default(self):
        """rep_irr_filter_high == 1 + rep_irr_filter with the default value."""
        capt = CapTest()
        assert capt.rep_irr_filter_high == pytest.approx(1 + capt.rep_irr_filter)
        assert capt.rep_irr_filter_high == pytest.approx(1.2)

    def test_rep_irr_filter_low_high_custom_at_construction(self):
        """Low and high fractions reflect a non-default rep_irr_filter."""
        capt = CapTest(rep_irr_filter=0.1)
        assert capt.rep_irr_filter_low == pytest.approx(0.9)
        assert capt.rep_irr_filter_high == pytest.approx(1.1)

    def test_rep_irr_filter_low_high_update_on_reassignment(self):
        """Reassigning rep_irr_filter updates the derived properties."""
        capt = CapTest(rep_irr_filter=0.2)
        capt.rep_irr_filter = 0.15
        assert capt.rep_irr_filter_low == pytest.approx(0.85)
        assert capt.rep_irr_filter_high == pytest.approx(1.15)


class TestFromYaml:
    """Yaml-driven construction."""

    def _write(self, tmp_path, text, name="cfg.yaml"):
        p = tmp_path / name
        p.write_text(text)
        return p

    def test_happy_path(self, captest_yaml):
        capt = CapTest.from_yaml(captest_yaml)
        assert capt.test_setup == "e2848_default"
        assert capt.ac_nameplate == 6_000_000
        assert capt.test_tolerance == "- 4"

    def test_unknown_key_raises_with_suggestion(self, tmp_path):
        p = self._write(
            tmp_path,
            "captest:\n  test_setup: e2848_default\n  ac_namplate: 1\n",
        )
        with pytest.raises(ValueError, match="ac_namplate"):
            CapTest.from_yaml(p)

    def test_missing_test_setup_raises(self, tmp_path):
        p = self._write(tmp_path, "captest:\n  ac_nameplate: 100\n")
        with pytest.raises(ValueError, match="test_setup"):
            CapTest.from_yaml(p)

    def test_custom_setup_requires_overrides(self, tmp_path):
        p = self._write(tmp_path, "captest:\n  test_setup: custom\n")
        with pytest.raises(ValueError, match="custom"):
            CapTest.from_yaml(p)

    def test_conflicting_reg_fml_raises(self, tmp_path):
        text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  reg_fml: power ~ poa\n"
            "  overrides:\n"
            "    reg_fml: power ~ poa + t_amb\n"
        )
        p = self._write(tmp_path, text)
        with pytest.raises(ValueError, match="reg_fml"):
            CapTest.from_yaml(p)

    def test_null_values_equivalent_to_absence(self, tmp_path):
        text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  ac_nameplate: null\n"
            "  reg_fml: null\n"
        )
        p = self._write(tmp_path, text)
        capt = CapTest.from_yaml(p)
        assert capt.ac_nameplate is None
        assert capt.reg_fml is None

    def test_relative_paths_resolve_to_yaml_dir(
        self, tmp_path, monkeypatch, meas_cd_default
    ):
        """Relative meas_path / sim_path in yaml resolve against yaml dir."""
        text = "captest:\n  test_setup: e2848_default\n  meas_path: ./subdir/meas.csv\n"
        p = self._write(tmp_path, text, name="cfg.yaml")
        received = {}

        def fake_load_data(path, **kwargs):
            received["path"] = path
            return meas_cd_default

        # from_yaml pulls the default loader via _default_meas_loader, which
        # returns captest.io.load_data. Patch that here.
        monkeypatch.setattr("captest.io.load_data", fake_load_data)
        CapTest.from_yaml(p)
        assert received["path"] == str(tmp_path / "subdir" / "meas.csv")

    def test_from_yaml_forwards_meas_loader_kwarg(self, tmp_path, meas_cd_default):
        """A programmatic meas_loader passed to from_yaml wins over the default.

        Downstream wrappers drive yaml-based construction but need to inject
        their own measured-data loader because loader callables cannot be
        represented in yaml.
        """
        text = "captest:\n  test_setup: e2848_default\n  meas_path: ./meas.csv\n"
        p = self._write(tmp_path, text)
        meas_loader = MagicMock(return_value=meas_cd_default)
        capt = CapTest.from_yaml(p, meas_loader=meas_loader)
        meas_loader.assert_called_once_with(str(tmp_path / "meas.csv"))
        assert capt.meas is meas_cd_default

    def test_from_yaml_forwards_sim_loader_kwarg(self, tmp_path, sim_cd_default):
        """A programmatic sim_loader passed to from_yaml wins over the default."""
        text = "captest:\n  test_setup: e2848_default\n  sim_path: ./sim.csv\n"
        p = self._write(tmp_path, text)
        sim_loader = MagicMock(return_value=sim_cd_default)
        capt = CapTest.from_yaml(p, sim_loader=sim_loader)
        sim_loader.assert_called_once_with(str(tmp_path / "sim.csv"))
        assert capt.sim is sim_cd_default

    def test_from_yaml_without_loader_kwargs_uses_default_resolution(
        self, tmp_path, monkeypatch, meas_cd_default
    ):
        """Regression guard: omitting the new kwargs preserves prior behavior."""
        text = "captest:\n  test_setup: e2848_default\n  meas_path: ./meas.csv\n"
        p = self._write(tmp_path, text)
        called = []

        def fake_load_data(path, **kwargs):
            called.append(path)
            return meas_cd_default

        monkeypatch.setattr("captest.io.load_data", fake_load_data)
        CapTest.from_yaml(p)  # no meas_loader/sim_loader kwargs
        assert called == [str(tmp_path / "meas.csv")]


class TestFromMapping:
    """Direct construction from an already-parsed sub-mapping dict.

    Downstream wrappers mutate the captest sub-mapping in memory --
    applying project-specific defaults, promoting fields, absolutizing
    paths -- before constructing the ``CapTest``.
    ``from_mapping`` exposes this handoff directly, avoiding a tempfile
    round-trip through ``from_yaml``. The post-parse pipeline shared with
    ``from_yaml`` (unknown-key detection, overrides flattening,
    test_setup required, reg_fml collision, custom-requires-overrides,
    None-stripping, loader injection, raw-path preservation) lives in
    ``from_mapping`` after this refactor; ``from_yaml`` is a thin file-read
    wrapper that delegates.
    """

    def test_happy_path_returns_capt(self, tmp_path, meas_cd_default, sim_cd_default):
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("")
        sim_file = tmp_path / "sim.csv"
        sim_file.write_text("")
        sub = {
            "test_setup": "e2848_default",
            "ac_nameplate": 100,
            "meas_path": str(meas_file),
            "sim_path": str(sim_file),
        }
        meas_loader = MagicMock(return_value=meas_cd_default)
        sim_loader = MagicMock(return_value=sim_cd_default)
        capt = CapTest.from_mapping(sub, meas_loader=meas_loader, sim_loader=sim_loader)
        assert capt.ac_nameplate == 100
        meas_loader.assert_called_once_with(str(meas_file))
        sim_loader.assert_called_once_with(str(sim_file))

    def test_requires_test_setup(self):
        with pytest.raises(ValueError, match="test_setup"):
            CapTest.from_mapping({"ac_nameplate": 100})

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="ac_namplate"):
            CapTest.from_mapping({"test_setup": "e2848_default", "ac_namplate": 1})

    def test_rejects_non_mapping_sub(self):
        with pytest.raises(TypeError, match="sub"):
            CapTest.from_mapping([("test_setup", "e2848_default")])

    def test_relative_path_without_base_dir_raises(self):
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "./rel/meas.csv",
        }
        with pytest.raises(ValueError, match="base_dir"):
            CapTest.from_mapping(sub)

    def test_relative_path_resolves_against_local_base_dir(
        self, tmp_path, meas_cd_default
    ):
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("")
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "./meas.csv",
        }
        meas_loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_mapping(sub, base_dir=tmp_path, meas_loader=meas_loader)
        meas_loader.assert_called_once_with(str(tmp_path / "meas.csv"))

    def test_uri_scheme_path_skips_resolution(self, meas_cd_default):
        """``s3://...`` passes through without being joined to base_dir.

        ``Path("s3://bucket/key").is_absolute()`` returns False on posix,
        so a naive Path-based check would mangle the URI. The explicit
        ``"://"`` check in ``_is_uri_or_absolute_path`` handles it.
        """
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "s3://bucket/clients/signal/data/parquet_clean",
        }
        meas_loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_mapping(
            sub,
            base_dir="/irrelevant/local/dir",  # should be ignored
            meas_loader=meas_loader,
        )
        meas_loader.assert_called_once_with(
            "s3://bucket/clients/signal/data/parquet_clean"
        )

    def test_uri_scheme_base_dir_string_concats(self, meas_cd_default):
        """URI-scheme base_dir + relative filename preserves the scheme.

        ``Path("s3://bucket/prefix") / "file"`` would collapse the
        double slash to ``s3:/bucket/prefix/file``; the string-concat
        branch in ``_join_base_and_relative`` avoids this.
        """
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "data/parquet_clean",
        }
        meas_loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_mapping(
            sub,
            base_dir="s3://bucket/clients/signal",
            meas_loader=meas_loader,
        )
        meas_loader.assert_called_once_with(
            "s3://bucket/clients/signal/data/parquet_clean"
        )

    def test_forwards_meas_loader_kwarg(self, tmp_path, meas_cd_default):
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("")
        sub = {"test_setup": "e2848_default", "meas_path": str(meas_file)}
        meas_loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_mapping(sub, meas_loader=meas_loader)
        meas_loader.assert_called_once_with(str(meas_file))

    def test_forwards_sim_loader_kwarg(self, tmp_path, sim_cd_default):
        sim_file = tmp_path / "sim.csv"
        sim_file.write_text("")
        sub = {"test_setup": "e2848_default", "sim_path": str(sim_file)}
        sim_loader = MagicMock(return_value=sim_cd_default)
        CapTest.from_mapping(sub, sim_loader=sim_loader)
        sim_loader.assert_called_once_with(str(sim_file))

    def test_does_not_mutate_sub(self, tmp_path, meas_cd_default):
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("")
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "./meas.csv",
        }
        snapshot = dict(sub)
        meas_loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_mapping(sub, base_dir=tmp_path, meas_loader=meas_loader)
        assert sub == snapshot, "from_mapping must not mutate the sub-mapping"

    def test_preserves_raw_meas_path_for_round_trip(self, tmp_path, meas_cd_default):
        """The raw (possibly relative) meas_path is stored on _meas_path so
        a later to_yaml emits what the user originally wrote."""
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("")
        sub = {
            "test_setup": "e2848_default",
            "meas_path": "./meas.csv",
        }
        meas_loader = MagicMock(return_value=meas_cd_default)
        capt = CapTest.from_mapping(sub, base_dir=tmp_path, meas_loader=meas_loader)
        assert capt._meas_path == "./meas.csv"

    def test_from_yaml_delegates_to_from_mapping(
        self, tmp_path, monkeypatch, meas_cd_default
    ):
        """``from_yaml`` is now a thin wrapper around ``from_mapping``.

        Verifies that after load_config parses the file, ``from_yaml``
        forwards the sub-mapping, key, base_dir, and loader kwargs
        unchanged to ``from_mapping``.
        """
        p = tmp_path / "cfg.yaml"
        p.write_text("captest:\n  test_setup: e2848_default\n  ac_nameplate: 42\n")
        recorded = {}

        real_from_mapping = CapTest.from_mapping

        def spy(sub, *, key, base_dir, meas_loader, sim_loader):
            recorded["sub"] = sub
            recorded["key"] = key
            recorded["base_dir"] = base_dir
            recorded["meas_loader"] = meas_loader
            recorded["sim_loader"] = sim_loader
            return real_from_mapping.__func__(
                CapTest,
                sub,
                key=key,
                base_dir=base_dir,
                meas_loader=meas_loader,
                sim_loader=sim_loader,
            )

        monkeypatch.setattr(CapTest, "from_mapping", spy)
        CapTest.from_yaml(p)

        assert recorded["key"] == "captest"
        assert recorded["base_dir"] == p.parent
        assert recorded["sub"]["test_setup"] == "e2848_default"
        assert recorded["sub"]["ac_nameplate"] == 42
        assert recorded["meas_loader"] is None
        assert recorded["sim_loader"] is None


class TestLoadConfigPublicExport:
    """``load_config`` is a public helper on the ``captest`` package root.

    Downstream wrappers parse the captest sub-mapping via this helper,
    apply project-specific defaults, and then hand off to
    ``CapTest.from_params``. Keeping this helper
    publicly importable from the package root (not just the submodule) is
    part of the public API contract.
    """

    def test_importable_from_package_root(self):
        import captest

        assert hasattr(captest, "load_config")
        assert captest.load_config is ct.load_config

    def test_returns_sub_mapping(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "captest:\n  test_setup: bifi_e2848_etotal_rear_shade_sim\n  ac_nameplate: 1234\n"
        )
        import captest

        sub = captest.load_config(p)
        assert sub["test_setup"] == "bifi_e2848_etotal_rear_shade_sim"
        assert sub["ac_nameplate"] == 1234

    def test_missing_key_raises_keyerror_listing_available_keys(self, tmp_path):
        """Missing top-level key surfaces a helpful error for downstream wrappers."""
        p = tmp_path / "cfg.yaml"
        p.write_text("client: barnhart\nsystem: {}\n")
        import captest

        with pytest.raises(KeyError, match="captest"):
            captest.load_config(p)

    def test_absent_optional_keys_are_absent_not_none(self, tmp_path):
        """Post-condition for downstream ``dict.setdefault``: absent keys stay absent.

        Downstream wrappers rely on ``setdefault`` to inject conventional
        values; that breaks if ``load_config`` stamps missing optional keys
        as ``None``.
        """
        p = tmp_path / "cfg.yaml"
        p.write_text("captest:\n  test_setup: e2848_default\n  ac_nameplate: 1000\n")
        import captest

        sub = captest.load_config(p)
        for absent_key in (
            "meas_path",
            "sim_path",
            "meas_load_kwargs",
            "sim_load_kwargs",
            "rep_conditions",
            "overrides",
        ):
            assert absent_key not in sub, (
                f"Expected {absent_key!r} to be absent, got {sub[absent_key]!r}"
            )


class TestSetup:
    """Behavior of CapTest.setup()."""

    def test_setup_requires_meas(self, sim_cd_default):
        capt = CapTest(sim=sim_cd_default)
        with pytest.raises(RuntimeError, match="meas"):
            capt.setup()

    def test_setup_requires_sim(self, meas_cd_default):
        capt = CapTest(meas=meas_cd_default)
        with pytest.raises(RuntimeError, match="sim"):
            capt.setup()

    def test_setup_returns_self(self, ct_default):
        # from_params already invoked setup(); re-run returns self.
        assert ct_default.setup(verbose=False) is ct_default

    def test_setup_propagates_downstream_attrs_to_both_cd(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.22,
            power_temp_coeff=-0.41,
            base_temp=20,
        )
        for attr in CapTest._downstream_attrs:
            assert getattr(capt.meas, attr) == getattr(capt, attr)
            if attr in CapTest._downstream_attrs_meas_only:
                # meas-only attrs must NOT be propagated to sim.
                assert not hasattr(capt.sim, attr)
            else:
                assert getattr(capt.sim, attr) == getattr(capt, attr)

    @pytest.mark.parametrize("preset", _DEFAULT_FIXTURE_PRESETS)
    def test_setup_wires_regression_formula(
        self, preset, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup=preset,
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
        )
        expected_fml = ct.TEST_SETUPS[preset]["reg_fml"]
        assert capt.meas.regression_formula == expected_fml
        assert capt.sim.regression_formula == expected_fml

    def test_setup_wires_tolerance(self, ct_default):
        assert ct_default.meas.tolerance == "- 4"
        assert ct_default.sim.tolerance == "- 4"

    def test_setup_assigns_resolved_setup(self, ct_default):
        resolved = ct_default._resolved_setup
        assert resolved is not None
        assert set(resolved.keys()) == {
            "description",
            "reg_cols_meas",
            "reg_cols_sim",
            "reg_fml",
            "scatter_plots",
            "rep_conditions",
        }

    def test_setup_rerun_resets_data_filtered(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )
        # Simulate a filter step: shrink data_filtered.
        capt.meas.data_filtered = capt.meas.data_filtered.iloc[:100].copy()
        assert capt.meas.data_filtered.shape[0] == 100
        capt.setup(verbose=False)
        # process_regression_columns resets data_filtered = data.copy().
        assert capt.meas.data_filtered.shape[0] == capt.meas.data.shape[0]

    def test_setup_verbose_prints_aggregations(
        self, meas_cd_default, sim_cd_default, capsys
    ):
        CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )  # from_params uses verbose default (True).
        captured = capsys.readouterr()
        assert "Aggregating the below" in captured.out

    def test_setup_verbose_false_silent(self, meas_cd_default, sim_cd_default, capsys):
        capt = CapTest(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )
        capt.setup(verbose=False)
        captured = capsys.readouterr()
        assert "Aggregating the below" not in captured.out

    def test_setup_applies_rep_conditions_override(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            rep_conditions={"percent_filter": 10},
        )
        resolved_rc = capt._resolved_setup["rep_conditions"]
        assert resolved_rc["percent_filter"] == 10
        # Non-overridden preset keys are preserved.
        assert resolved_rc["irr_bal"] is False
        assert set(resolved_rc["func"].keys()) == {"poa", "t_amb", "w_vel"}


class TestDownstreamPropagation:
    """Calc-params scalars on CapTest flow through to calcparams functions."""

    def test_bifaciality_flows_into_e_total(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
        )
        # Sanity: e_total = poa + rpoa * bifaciality (bifacial_frac=1,
        # rear_shade=0 by default). Extract the first non-zero row.
        meas_df = capt.meas.data
        mask = meas_df["irr_poa_mean_agg"] > 0
        first = meas_df.loc[mask].iloc[0]
        expected = first["irr_poa_mean_agg"] + first["irr_rpoa_mean_agg"] * 0.5
        assert first["e_total"] == pytest.approx(expected)

    def test_rear_shade_flows_into_e_total(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
            rear_shade=0.12,
        )
        # Sanity: e_total = poa + rpoa * bifaciality * (1 - rear_shade)
        meas_df = capt.meas.data
        mask = meas_df["irr_poa_mean_agg"] > 0
        first = meas_df.loc[mask].iloc[0]
        expected = first["irr_poa_mean_agg"] + first["irr_rpoa_mean_agg"] * 0.5 * (
            1 - 0.12
        )
        assert first["e_total"] == pytest.approx(expected)

    def test_rear_shade_not_propagated_to_sim(self, meas_cd_default, sim_cd_default):
        """rear_shade is meas-only: set on meas, absent on sim, no sim discount."""
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
            rear_shade=0.12,
        )
        assert capt.meas.rear_shade == 0.12
        assert not hasattr(capt.sim, "rear_shade")
        # Sim e_total uses the e_total default rear_shade=0 (no shade discount).
        # sim rpoa = rpoa_pvsyst(GlobBak, BackShd) = GlobBak (BackShd is 0).
        sim_df = capt.sim.data
        first = sim_df.loc[sim_df["GlobInc"] > 0].iloc[0]
        expected_sim = first["GlobInc"] + first["GlobBak"] * 0.5
        assert first["e_total"] == pytest.approx(expected_sim)

    def test_meas_shade_setup_applies_shade_to_meas_only(
        self, meas_cd_default, sim_cd_default
    ):
        """bifi_e2848_etotal_rear_shade_meas discounts the measured rear by
        rear_shade while mapping sim rpoa directly to GlobBak (no discount).
        """
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_meas",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
            rear_shade=0.12,
        )
        # Meas: e_total = poa + rpoa * bifaciality * (1 - rear_shade).
        meas_df = capt.meas.data
        m = meas_df.loc[meas_df["irr_poa_mean_agg"] > 0].iloc[0]
        assert m["e_total"] == pytest.approx(
            m["irr_poa_mean_agg"] + m["irr_rpoa_mean_agg"] * 0.5 * (1 - 0.12)
        )
        # Sim: rpoa maps directly to GlobBak with no rear_shade discount.
        assert not hasattr(capt.sim, "rear_shade")
        sim_df = capt.sim.data
        s = sim_df.loc[sim_df["GlobInc"] > 0].iloc[0]
        assert s["e_total"] == pytest.approx(s["GlobInc"] + s["GlobBak"] * 0.5)

    def test_bifacial_frac_flows_into_e_total(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
            bifacial_frac=0.8,
        )
        # e_total = poa + rpoa * bifaciality * bifacial_frac (rear_shade=0).
        meas_df = capt.meas.data
        first = meas_df.loc[meas_df["irr_poa_mean_agg"] > 0].iloc[0]
        expected = first["irr_poa_mean_agg"] + first["irr_rpoa_mean_agg"] * 0.5 * 0.8
        assert first["e_total"] == pytest.approx(expected)

    def test_module_type_and_racking_flow_into_temp_model(
        self, meas_cd_default, sim_cd_default
    ):
        """module_type/racking propagate into the Sandia temp model so a
        non-default racking changes the calculated cell_temp column."""
        ct_open = CapTest.from_params(
            test_setup="bifi_power_tc_calc_tbom",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
        )
        cell_temp_open = ct_open.meas.data["cell_temp"].copy()
        ct_ins = CapTest.from_params(
            test_setup="bifi_power_tc_calc_tbom",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
            racking="insulated_back",
        )
        assert ct_ins.meas.racking == "insulated_back"
        assert ct_ins.sim.racking == "insulated_back"
        # open_rack del_tcnd=3 vs insulated_back del_tcnd=0 -> cell_temp differs.
        assert not np.allclose(
            cell_temp_open.to_numpy(), ct_ins.meas.data["cell_temp"].to_numpy()
        )

    def test_airmass_model_flows_into_absolute_airmass(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        """A non-default airmass_model changes the absolute_airmass column."""
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            ct_default_model = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
            )
            am_default = ct_default_model.meas.data["absolute_airmass"].copy()
            ct_alt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                airmass_model="simple",
            )
        assert ct_alt.meas.airmass_model == "simple"
        assert ct_alt.sim.airmass_model == "simple"
        assert not np.allclose(
            am_default.to_numpy(),
            ct_alt.meas.data["absolute_airmass"].to_numpy(),
            equal_nan=True,
        )

    def test_altitude_override_flows_into_apparent_zenith(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        """A non-zero altitude_override changes the apparent_zenith column."""
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            ct_sea = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
            )
            zen_sea = ct_sea.meas.data["apparent_zenith"].copy()
            ct_alt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                altitude_override=5000,
            )
        assert ct_alt.meas.altitude_override == 5000
        assert not np.allclose(
            zen_sea.to_numpy(),
            ct_alt.meas.data["apparent_zenith"].to_numpy(),
            equal_nan=True,
        )

    def test_power_temp_coeff_flows_into_power_temp_correct(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_power_tc_calc_tbom",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
            power_temp_coeff=-0.5,
            base_temp=25,
        )
        # The power_temp_correct column lives on sim.data; its formula is
        # power / (1 + (coeff/100) * (cell_temp - base_temp)).
        sim_df = capt.sim.data
        first = sim_df.iloc[10]  # avoid the nighttime zeros at the top
        if first["E_Grid"] == 0:
            # grab a daytime row
            first = sim_df.loc[sim_df["E_Grid"] > 0].iloc[0]
        expected = first["E_Grid"] / (1 + (-0.5 / 100) * (first["TArray"] - 25))
        assert first["power_temp_correct"] == pytest.approx(expected)

    def test_base_temp_flows_into_power_temp_correct(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_power_tc_calc_tbom",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
            base_temp=35,
        )
        sim_df = capt.sim.data
        first = sim_df.loc[sim_df["E_Grid"] > 0].iloc[0]
        expected = first["E_Grid"] / (1 + (-0.32 / 100) * (first["TArray"] - 35))
        assert first["power_temp_correct"] == pytest.approx(expected)

    def test_power_temp_coeff_flows_into_power_temp_correct_meas_tbom(
        self, meas_cd_bom_temp, sim_cd_default
    ):
        """For meas_tbom preset, power_temp_coeff flows through to the sim side."""
        capt = CapTest.from_params(
            test_setup="bifi_power_tc_meas_tbom",
            meas=meas_cd_bom_temp,
            sim=sim_cd_default,
            bifaciality=0.15,
            power_temp_coeff=-0.5,
            base_temp=25,
        )
        sim_df = capt.sim.data
        first = sim_df.loc[sim_df["E_Grid"] > 0].iloc[0]
        expected = first["E_Grid"] / (1 + (-0.5 / 100) * (first["TArray"] - 25))
        assert first["power_temp_correct"] == pytest.approx(expected)


class TestCapTestSpectralCorrection:
    """End-to-end behavior of the e2848_spec_corrected_poa preset."""

    def test_meas_poa_spec_corrected_column_exists(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            capt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                ac_nameplate=6_000_000,
                test_tolerance="- 4",
            )
        assert "poa_spec_corrected" in capt.meas.data.columns
        # Daytime values should be finite and non-zero.
        daytime = capt.meas.data["poa_spec_corrected"].dropna()
        assert len(daytime) > 0
        assert (daytime[daytime > 0] > 0).all()

    def test_sim_poa_spec_corrected_column_exists(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            capt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                ac_nameplate=6_000_000,
            )
        assert "poa_spec_corrected" in capt.sim.data.columns

    def test_sim_site_auto_propagated_with_fixed_offset_tz(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        assert getattr(sim_cd_spec_corrected, "site", None) is None
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            capt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
            )
        assert capt.sim.site is not None
        tz = capt.sim.site["loc"]["tz"]
        assert tz.startswith("Etc/GMT")
        # America/Chicago standard offset is UTC-6 -> Etc/GMT+6.
        assert tz == "Etc/GMT+6"
        # Caller's meas.site tz is not mutated.
        assert meas_cd_spec_corrected.site["loc"]["tz"] == "America/Chicago"

    def test_user_set_sim_site_is_not_overwritten(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        user_site = {
            "loc": {
                "latitude": 33.0,
                "longitude": -99.5,
                "altitude": 0,
                "tz": "Etc/GMT+5",
            },
            "sys": {"surface_tilt": 0, "surface_azimuth": 180, "albedo": 0.2},
        }
        sim_cd_spec_corrected.site = user_site
        capt = CapTest.from_params(
            test_setup="e2848_spec_corrected_poa",
            meas=meas_cd_spec_corrected,
            sim=sim_cd_spec_corrected,
        )
        assert capt.sim.site is user_site

    def test_spectral_module_type_propagates_to_both_cd(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            capt = CapTest.from_params(
                test_setup="e2848_spec_corrected_poa",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                spectral_module_type="monosi",
            )
        assert capt.meas.spectral_module_type == "monosi"
        assert capt.sim.spectral_module_type == "monosi"


class TestLoaderInjection:
    """Loader callable defaults, overrides, and kwarg splatting."""

    def test_default_meas_loader_is_load_data(self):
        from captest.io import load_data

        assert ct._default_meas_loader() is load_data

    def test_default_sim_loader_is_load_pvsyst(self):
        from captest.io import load_pvsyst

        assert ct._default_sim_loader() is load_pvsyst

    def test_custom_meas_loader_called_with_path_and_kwargs(self, meas_cd_default):
        loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_params(
            test_setup="e2848_default",
            meas_path="/fake/path",
            meas_loader=loader,
            meas_load_kwargs={"period": "2024-01", "groups": ["a", "b"]},
        )
        loader.assert_called_once_with(
            "/fake/path", period="2024-01", groups=["a", "b"]
        )

    def test_custom_sim_loader_called_with_path_and_kwargs(self, sim_cd_default):
        loader = MagicMock(return_value=sim_cd_default)
        CapTest.from_params(
            test_setup="e2848_default",
            sim_path="/fake/sim/path",
            sim_loader=loader,
            sim_load_kwargs={"egrid_unit_adj_factor": 1000},
        )
        loader.assert_called_once_with("/fake/sim/path", egrid_unit_adj_factor=1000)


class TestRepCondConvenience:
    """CapTest.rep_cond and CapTest.scatter_plots convenience methods."""

    def test_rep_cond_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.rep_cond()

    def test_rep_cond_calls_cd_rep_cond_with_resolved_defaults(self, ct_default):
        # Patch cd.rep_cond to capture the kwargs passed through.
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond()
        preset_rc = ct.TEST_SETUPS["e2848_default"]["rep_conditions"]
        assert received["percent_filter"] == preset_rc["percent_filter"]
        assert received["irr_bal"] == preset_rc["irr_bal"]
        assert set(received["func"].keys()) == set(preset_rc["func"].keys())

    def test_rep_cond_partial_merge_overrides(self, ct_default):
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond(percent_filter=10)
        assert received["percent_filter"] == 10
        # Preset keys preserved.
        assert received["irr_bal"] is False
        assert set(received["func"].keys()) == {"poa", "t_amb", "w_vel"}

    def test_rep_cond_func_partial_merge(self, ct_default):
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        new_poa_fn = ct.perc_wrap(55)
        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond(func={"poa": new_poa_fn})
        assert received["func"]["poa"] is new_poa_fn
        # Preserved from preset.
        assert received["func"]["t_amb"] == "mean"
        assert received["func"]["w_vel"] == "mean"

    def test_rep_cond_which_sim(self, ct_default):
        received = {}

        def fake_sim_rep_cond(**kwargs):
            received["target"] = "sim"
            received.update(kwargs)

        def fake_meas_rep_cond(**kwargs):
            received["target"] = "meas"
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_meas_rep_cond
        ct_default.sim.rep_cond = fake_sim_rep_cond
        ct_default.rep_cond(which="sim")
        assert received["target"] == "sim"

    def test_rep_cond_which_invalid(self, ct_default):
        with pytest.raises(ValueError, match="must be 'meas' or 'sim'"):
            ct_default.rep_cond(which="bogus")

    def test_rep_conditions_override_from_init_partial_merges(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            rep_conditions={"percent_filter": 10},
        )
        resolved_rc = capt._resolved_setup["rep_conditions"]
        assert resolved_rc["percent_filter"] == 10
        assert resolved_rc["irr_bal"] is False
        assert "func" in resolved_rc

    @pytest.mark.parametrize("preset", _DEFAULT_FIXTURE_PRESETS)
    def test_each_preset_rep_conditions_round_trips_through_rep_cond(
        self, preset, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup=preset,
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
        )
        # Should not raise.
        capt.rep_cond()
        assert isinstance(capt.meas.rc, pd.DataFrame)

    def test_scatter_plots_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.scatter_plots()

    def test_scatter_plots_dispatches_to_resolved_callable(self, ct_default):
        import holoviews as hv

        layout = ct_default.scatter_plots()
        assert isinstance(layout, hv.Layout)

    def test_scatter_plots_which_sim(self, ct_default):
        import holoviews as hv

        layout = ct_default.scatter_plots(which="sim")
        assert isinstance(layout, hv.Layout)


class TestResolvedSetupProperty:
    def test_property_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.resolved_setup

    def test_property_returns_resolved_dict(self, ct_default):
        resolved = ct_default.resolved_setup
        assert resolved is ct_default._resolved_setup


# --- Cross-CapData methods ported from capdata module-level functions ----


class TestPortedMethods:
    """Tests for CapTest methods ported from the former module-level
    ``capdata.captest_results``, ``captest_results_check_pvalues``,
    ``determine_pass_or_fail``, ``get_summary``, ``overlay_scatters``, and
    ``plotting.residual_plot``.
    """

    def _build_capdata_with_fit(self):
        """Return a (meas, sim) pair with fitted regression results and rc."""
        import statsmodels.formula.api as smf

        np.random.seed(9876789)
        nsample = 100
        e = np.random.normal(size=nsample)
        a = np.linspace(0, 10, nsample)
        b = a / 2.0
        c = a + 3.0

        das_y = a + (a**2) + (a * b) + (a * c) + e
        sim_y = a + (a**2 * 0.9) + (a * b * 1.1) + (a * c * 0.8) + e
        das_df = pd.DataFrame({"power": das_y, "poa": a, "t_amb": b, "w_vel": c})
        sim_df = pd.DataFrame({"power": sim_y, "poa": a, "t_amb": b, "w_vel": c})

        from captest.capdata import CapData

        meas = CapData("meas")
        sim = CapData("sim")
        meas.data = das_df
        sim.data = sim_df
        meas.data_filtered = das_df.copy()
        sim.data_filtered = sim_df.copy()
        meas.rc = pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]})
        sim.rc = pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]})

        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        meas.regression_results = smf.ols(formula=fml, data=das_df).fit()
        sim.regression_results = smf.ols(formula=fml, data=sim_df).fit()
        return meas, sim

    def _build_ct(self):
        meas, sim = self._build_capdata_with_fit()
        capt = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        capt.meas = meas
        capt.sim = sim
        return capt

    def test_determine_pass_or_fail_uses_ct_attrs(self):
        capt = CapTest(test_tolerance="+/- 4", ac_nameplate=100)
        passed, bounds = capt.determine_pass_or_fail(0.96)
        assert passed is True or passed == np.True_
        assert bounds == "96.0, 104.0"

    def test_determine_pass_or_fail_minus_sign(self):
        capt = CapTest(test_tolerance="- 4", ac_nameplate=100)
        passed, bounds = capt.determine_pass_or_fail(0.95)
        # 0.95 is below 1 - 0.04 = 0.96, so this is a fail.
        assert passed is False or passed == np.False_
        assert bounds == "96.0, None"

    def test_determine_pass_or_fail_warns_on_bad_sign(self):
        capt = CapTest(test_tolerance="+ 4", ac_nameplate=100)
        with pytest.warns(UserWarning, match=r"Sign must be"):
            result = capt.determine_pass_or_fail(1.04)
        assert result is None

    def test_captest_results_requires_meas_and_sim(self):
        capt = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        with pytest.raises(RuntimeError, match="meas"):
            capt.captest_results(print_res=False)

    def test_captest_results_matches_direct_prediction(self):
        capt = self._build_ct()
        expected_actual = capt.meas.regression_results.predict(capt.meas.rc)[0]
        expected_expected = capt.sim.regression_results.predict(capt.meas.rc)[0]
        expected_ratio = expected_actual / expected_expected

        cp_rat = capt.captest_results(print_res=False)

        assert cp_rat == pytest.approx(expected_ratio, rel=1e-10)

    def test_captest_results_uses_rep_cond_source_meas_by_default(self):
        capt = self._build_ct()
        # Set sim.rc to a different value; default rep_cond_source="meas"
        # should keep the meas rc.
        capt.sim.rc = pd.DataFrame({"poa": [99], "t_amb": [99], "w_vel": [99]})
        expected_actual = capt.meas.regression_results.predict(capt.meas.rc)[0]
        expected_expected = capt.sim.regression_results.predict(capt.meas.rc)[0]
        expected_ratio = expected_actual / expected_expected

        cp_rat = capt.captest_results(print_res=False)
        assert cp_rat == pytest.approx(expected_ratio, rel=1e-10)

    def test_captest_results_uses_rep_cond_source_sim(self):
        capt = self._build_ct()
        capt.rep_cond_source = "sim"
        capt.sim.rc = pd.DataFrame({"poa": [8], "t_amb": [4], "w_vel": [2]})
        expected_actual = capt.meas.regression_results.predict(capt.sim.rc)[0]
        expected_expected = capt.sim.regression_results.predict(capt.sim.rc)[0]
        expected_ratio = expected_actual / expected_expected

        cp_rat = capt.captest_results(print_res=False)
        assert cp_rat == pytest.approx(expected_ratio, rel=1e-10)

    def test_captest_results_warns_on_mismatched_formulas(self):
        capt = self._build_ct()
        capt.sim.regression_formula = "power ~ poa + t_amb"
        with pytest.warns(UserWarning, match="regression formula"):
            capt.captest_results(print_res=False)

    def test_captest_results_check_pvalues_returns_styled_df(self):
        capt = self._build_ct()
        styled = capt.captest_results_check_pvalues(print_res=False)
        # Styler objects expose the underlying data via the .data attribute.
        underlying = styled.data
        assert isinstance(underlying, pd.DataFrame)
        assert set(underlying.columns) == {
            "das_pvals",
            "sim_pvals",
            "das_params",
            "sim_params",
        }

    def test_get_summary_concatenates_meas_and_sim(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
        )
        # Apply a filter to each CapData so get_summary has rows.
        capt.meas.filter_irr(200, 800)
        capt.sim.filter_irr(200, 800)
        combined = capt.get_summary()
        assert isinstance(combined, pd.DataFrame)
        # Combined summary has rows from both CapData instances.
        names = combined.index.get_level_values(0).unique().tolist()
        assert capt.meas.name in names
        assert capt.sim.name in names

    def test_get_summary_requires_meas_and_sim(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="meas"):
            capt.get_summary()

    def test_overlay_scatters_requires_setup(self, meas_cd_default, sim_cd_default):
        capt = CapTest()
        capt.meas = meas_cd_default
        capt.sim = sim_cd_default
        with pytest.raises(RuntimeError, match="setup"):
            capt.overlay_scatters()

    def test_overlay_scatters_returns_overlay(self, ct_default):
        import holoviews as hv

        overlay = ct_default.overlay_scatters()
        assert isinstance(overlay, hv.Overlay)

    def test_overlay_scatters_uses_expected_label(self, ct_default):
        overlay = ct_default.overlay_scatters(expected_label="My PVsyst")
        # Both children of the overlay get labeled via .relabel; verify the
        # second one (sim) got the custom label.
        labels = [child.label for child in overlay]
        assert "My PVsyst" in labels
        assert "Measured" in labels

    def test_residual_plot_returns_layout(self):
        import holoviews as hv

        capt = self._build_ct()
        # residual_plot needs a .data_filtered index aligned with the fitted
        # model's exog; the _build_capdata_with_fit helper uses the full data
        # as data_filtered, so alignment holds.
        layout = capt.residual_plot()
        assert isinstance(layout, hv.Layout)

    def test_residual_plot_requires_meas_and_sim(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="meas"):
            capt.residual_plot()


# --- to_yaml / round-trip / key parametrization --------------------------


class TestToYamlAndRoundTrip:
    """Serialization of CapTest state to yaml and round-trip through yaml."""

    def _load(self, path):
        with open(path, "r") as fh:
            return yaml.safe_load(fh)

    def test_to_yaml_writes_curated_scalar_set(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(
            test_setup="e2848_default",
            ac_nameplate=125_000,
            test_tolerance="- 4",
            bifaciality=0.12,
        )
        capt.to_yaml(p, merge_into_existing=False)
        doc = self._load(p)
        sub = doc["captest"]
        # Scalar params round-trip.
        assert sub["test_setup"] == "e2848_default"
        assert sub["ac_nameplate"] == 125_000
        assert sub["test_tolerance"] == "- 4"
        assert sub["bifaciality"] == 0.12
        # ``meas``, ``sim``, and loader callables are never written.
        for forbidden in ("meas", "sim", "meas_loader", "sim_loader"):
            assert forbidden not in sub

    def test_to_yaml_omits_paths_when_not_constructed_from_paths(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(test_setup="e2848_default")
        capt.to_yaml(p, merge_into_existing=False)
        sub = self._load(p)["captest"]
        assert "meas_path" not in sub
        assert "sim_path" not in sub

    def test_to_yaml_writes_paths_when_constructed_from_paths(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        def fake_loader(path, **kwargs):
            # Any pre-built CapData is fine; we're only testing path
            # round-trip.
            return meas_cd_default

        p = tmp_path / "cfg.yaml"
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas_path="/some/meas/path.csv",
            meas_loader=fake_loader,
            sim_path="/some/sim/path.csv",
            sim_loader=fake_loader,
        )
        with pytest.warns(UserWarning, match=r"meas_loader"):
            capt.to_yaml(p, merge_into_existing=False)
        sub = self._load(p)["captest"]
        assert sub["meas_path"] == "/some/meas/path.csv"
        assert sub["sim_path"] == "/some/sim/path.csv"

    def test_to_yaml_omits_overrides_when_user_did_not_set_them(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(test_setup="e2848_default")
        capt.to_yaml(p, merge_into_existing=False)
        sub = self._load(p)["captest"]
        assert "overrides" not in sub

    def test_to_yaml_writes_overrides_when_user_set_them(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(
            test_setup="e2848_default",
            reg_fml="power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel)",
        )
        capt.to_yaml(p, merge_into_existing=False)
        sub = self._load(p)["captest"]
        assert "overrides" in sub
        assert sub["overrides"]["reg_fml"].startswith("power ~ poa")

    def test_to_yaml_omits_reg_fml_override_equal_to_preset(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        preset_fml = ct.TEST_SETUPS["e2848_default"]["reg_fml"]
        capt = CapTest(test_setup="e2848_default", reg_fml=preset_fml)
        capt.to_yaml(p, merge_into_existing=False)
        sub = self._load(p)["captest"]
        assert "overrides" not in sub

    def test_to_yaml_writes_load_kwargs_only_when_non_empty(self, tmp_path):
        p_empty = tmp_path / "empty.yaml"
        capt_empty = CapTest(test_setup="e2848_default")
        capt_empty.to_yaml(p_empty, merge_into_existing=False)
        sub_empty = self._load(p_empty)["captest"]
        assert "meas_load_kwargs" not in sub_empty
        assert "sim_load_kwargs" not in sub_empty

        p_full = tmp_path / "full.yaml"
        capt_full = CapTest(
            test_setup="e2848_default",
            meas_load_kwargs={"sep": ";"},
            sim_load_kwargs={"egrid_unit_adj_factor": 1000},
        )
        capt_full.to_yaml(p_full, merge_into_existing=False)
        sub_full = self._load(p_full)["captest"]
        assert sub_full["meas_load_kwargs"] == {"sep": ";"}
        assert sub_full["sim_load_kwargs"] == {"egrid_unit_adj_factor": 1000}

    def test_to_yaml_round_trip(self, tmp_path):
        yaml_src = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  ac_nameplate: 6000000\n"
            "  test_tolerance: '- 4'\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      percent_filter: 15\n"
            "      func:\n"
            "        poa: perc_55\n"
            "        t_amb: mean\n"
            "        w_vel: mean\n"
        )
        p1 = tmp_path / "cfg1.yaml"
        p1.write_text(yaml_src)
        capt = CapTest.from_yaml(p1)
        p2 = tmp_path / "cfg2.yaml"
        capt.to_yaml(p2, merge_into_existing=False)
        doc1 = self._load(p1)["captest"]
        doc2 = self._load(p2)["captest"]
        # Round-trip preserves test_setup and the nameplate/tolerance scalars.
        assert doc2["test_setup"] == doc1["test_setup"]
        assert doc2["ac_nameplate"] == doc1["ac_nameplate"]
        assert doc2["test_tolerance"] == doc1["test_tolerance"]
        # Override rep_conditions round-trips; perc_wrap(55) -> 'perc_55'.
        assert (
            doc2["overrides"]["rep_conditions"]["percent_filter"]
            == doc1["overrides"]["rep_conditions"]["percent_filter"]
        )
        assert (
            doc2["overrides"]["rep_conditions"]["func"]["poa"]
            == doc1["overrides"]["rep_conditions"]["func"]["poa"]
            == "perc_55"
        )
        # A second from_yaml on the written file succeeds.
        capt2 = CapTest.from_yaml(p2)
        assert capt2.ac_nameplate == 6_000_000

    def test_to_yaml_round_trips_calc_param_scalars(self, tmp_path):
        """New calc-param scalars + rear_shade survive a to_yaml/from_yaml trip."""
        p = tmp_path / "cfg.yaml"
        capt = CapTest(
            test_setup="bifi_e2848_etotal_rear_shade_sim",
            bifaciality=0.3,
            bifacial_frac=0.8,
            rear_shade=0.12,
            module_type="glass_cell_glass",
            racking="close_roof_mount",
            airmass_model="kasten1966",
            altitude_override=500,
        )
        capt.to_yaml(p, merge_into_existing=False)
        loaded = CapTest.from_yaml(p)
        assert loaded.bifaciality == 0.3
        assert loaded.bifacial_frac == 0.8
        assert loaded.rear_shade == 0.12
        assert loaded.module_type == "glass_cell_glass"
        assert loaded.racking == "close_roof_mount"
        assert loaded.airmass_model == "kasten1966"
        assert loaded.altitude_override == 500

    def test_to_yaml_round_trips_altitude_override_none(self, tmp_path):
        """altitude_override=None (respect site altitude) survives the round
        trip as None rather than being coerced to the default of 0."""
        p = tmp_path / "cfg.yaml"
        capt = CapTest(test_setup="e2848_default", altitude_override=None)
        capt.to_yaml(p, merge_into_existing=False)
        loaded = CapTest.from_yaml(p)
        assert loaded.altitude_override is None

    def test_to_yaml_warns_when_scatter_plots_is_user_mutated(
        self, tmp_path, ct_default
    ):
        p = tmp_path / "cfg.yaml"

        def my_custom_scatter(cd, **kwargs):
            return None

        ct_default._resolved_setup["scatter_plots"] = my_custom_scatter
        with pytest.warns(UserWarning, match="scatter_plots"):
            ct_default.to_yaml(p, merge_into_existing=False)
        # Written yaml contains no scatter_plots key.
        sub = self._load(p)["captest"]
        assert "scatter_plots" not in sub
        assert "overrides" not in sub or "scatter_plots" not in sub.get("overrides", {})

    def test_to_yaml_warns_when_loader_callable_set(self, tmp_path):
        p = tmp_path / "cfg.yaml"

        def fake_loader(path, **kwargs):
            return None

        capt = CapTest(test_setup="e2848_default", meas_loader=fake_loader)
        with pytest.warns(UserWarning, match="meas_loader"):
            capt.to_yaml(p, merge_into_existing=False)


class TestYamlPercShorthand:
    """perc_N string shorthand round-trips through from_yaml / to_yaml."""

    def test_perc_N_string_converts_to_perc_wrap_on_from_yaml(self, tmp_path):
        yaml_src = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_55\n"
            "        t_amb: mean\n"
            "        w_vel: mean\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_src)
        capt = CapTest.from_yaml(p)
        func = capt.rep_conditions["func"]
        assert func["t_amb"] == "mean"
        # The resolved poa value is a callable equivalent to perc_wrap(55).
        sample = pd.Series(np.arange(100))
        assert func["poa"](sample) == ct.perc_wrap(55)(sample)

    def test_to_yaml_emits_perc_N_for_perc_wrap(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(
            test_setup="e2848_default",
            rep_conditions={
                "percent_filter": 20,
                "func": {
                    "poa": ct.perc_wrap(65),
                    "t_amb": "mean",
                    "w_vel": "mean",
                },
            },
        )
        capt.to_yaml(p, merge_into_existing=False)
        doc = yaml.safe_load(p.read_text())
        func_dict = doc["captest"]["overrides"]["rep_conditions"]["func"]
        assert func_dict["poa"] == "perc_65"
        assert func_dict["t_amb"] == "mean"
        assert func_dict["w_vel"] == "mean"

    def test_invalid_perc_string_raises_on_from_yaml(self, tmp_path):
        yaml_src = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_bogus\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_src)
        with pytest.raises(ValueError, match="perc_<int>"):
            CapTest.from_yaml(p)


class TestYamlKeyParametrization:
    """``key=`` parametrizes the top-level mapping for multi-flavor yaml files."""

    def test_from_yaml_reads_under_captest_key_by_default(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "captest:\n  test_setup: bifi_e2848_etotal_rear_shade_sim\n  ac_nameplate: 1\n"
        )
        capt = CapTest.from_yaml(p)
        assert capt.test_setup == "bifi_e2848_etotal_rear_shade_sim"
        assert capt.ac_nameplate == 1

    def test_from_yaml_reads_under_custom_key(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "captest_bifi:\n  test_setup: bifi_e2848_etotal_rear_shade_sim\n  ac_nameplate: 2\n"
        )
        capt = CapTest.from_yaml(p, key="captest_bifi")
        assert capt.test_setup == "bifi_e2848_etotal_rear_shade_sim"
        assert capt.ac_nameplate == 2

    def test_from_yaml_missing_key_lists_available_keys(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("captest_alt:\n  test_setup: e2848_default\n")
        with pytest.raises(KeyError, match="captest_alt"):
            CapTest.from_yaml(p)  # default key='captest' not present

    def test_to_yaml_writes_under_custom_key(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        capt = CapTest(test_setup="e2848_default")
        capt.to_yaml(p, key="captest_bifi", merge_into_existing=False)
        doc = yaml.safe_load(p.read_text())
        assert "captest_bifi" in doc
        assert "captest" not in doc
        assert doc["captest_bifi"]["test_setup"] == "e2848_default"

    def test_to_yaml_merge_into_existing_preserves_other_top_level_keys(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        # Pre-populate with project-level keys plus an orthogonal captest
        # sub-map.
        p.write_text(
            "client: barnhart\n"
            "loc:\n"
            "  latitude: 42.28\n"
            "  longitude: -84.65\n"
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  ac_nameplate: 10\n"
        )
        capt = CapTest(test_setup="bifi_e2848_etotal_rear_shade_sim", ac_nameplate=20)
        capt.to_yaml(p, key="captest_bifi", merge_into_existing=True)
        doc = yaml.safe_load(p.read_text())
        # Other top-level keys preserved.
        assert doc["client"] == "barnhart"
        assert doc["loc"] == {"latitude": 42.28, "longitude": -84.65}
        # Original captest sub-map untouched.
        assert doc["captest"]["test_setup"] == "e2848_default"
        assert doc["captest"]["ac_nameplate"] == 10
        # New captest_bifi sub-map written.
        assert doc["captest_bifi"]["test_setup"] == "bifi_e2848_etotal_rear_shade_sim"
        assert doc["captest_bifi"]["ac_nameplate"] == 20

    def test_to_yaml_merge_false_overwrites_existing_file(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("unrelated_key: keep_me\n")
        capt = CapTest(test_setup="e2848_default")
        capt.to_yaml(p, merge_into_existing=False)
        doc = yaml.safe_load(p.read_text())
        assert "unrelated_key" not in doc
        assert "captest" in doc


# --- End-to-end integration tests (Unit 9) -------------------------------

# Time window covering the example measured data's 5-day range. ``filter_time``
# is applied to the simulated CapData so its one-year PVsyst export is
# narrowed to the same span before ``rep_cond`` is computed.
_SIM_WINDOW = ("1990-10-09", "1990-10-13 23:55:00")


class TestIntegration:
    """End-to-end runs through each shipped ``TEST_SETUPS`` preset.

    Each test exercises the canonical filter-sequence ->
    reporting-conditions -> fit -> ``captest_results`` path using the
    ``ct_*`` fixtures from ``conftest.py``. Asserts a plausible capacity
    ratio and that any preset-specific calculated columns were materialized
    by ``setup()``.
    """

    @staticmethod
    def _run_canonical_sequence(capt):
        """Apply filter_irr (both), filter_shade+filter_time (sim), rep_cond,
        fit_regression. Mirrors the canonical capacity-test workflow a user
        drives manually between ``setup()`` and ``captest_results()``.
        """
        capt.meas.filter_irr(capt.min_irr, capt.max_irr)
        capt.sim.filter_irr(capt.min_irr, capt.max_irr)
        # ``filter_shade`` keys off PVsyst's ``FShdBm`` column, so it is only
        # applicable to the sim CapData.
        capt.sim.filter_shade(fshdbm=capt.fshdbm)
        capt.sim.filter_time(start=_SIM_WINDOW[0], end=_SIM_WINDOW[1])
        capt.rep_cond()
        capt.rep_cond(which="sim")
        capt.meas.fit_regression(summary=False)
        capt.sim.fit_regression(summary=False)

    def test_end_to_end_e2848_default(self, ct_default):
        """Default ASTM E2848 preset runs end-to-end to a plausible cap ratio."""
        self._run_canonical_sequence(ct_default)
        cap_ratio = ct_default.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2
        # Regression-column resolution on setup() wires the aggregated names.
        assert ct_default.meas.regression_cols["poa"] == "irr_poa_mean_agg"
        assert ct_default.sim.regression_cols["poa"] == "GlobInc"

    def test_end_to_end_bifi_e2848_etotal_rear_shade_sim(self, ct_etotal):
        """Bifacial e_total preset runs end-to-end; e_total column materialized."""
        # setup() (called by from_params) already ran process_regression_columns
        # which adds the e_total column to both CapData instances.
        assert "e_total" in ct_etotal.meas.data.columns
        assert "e_total" in ct_etotal.sim.data.columns

        self._run_canonical_sequence(ct_etotal)
        cap_ratio = ct_etotal.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2
        # The regression uses e_total as the "poa" column for both sides.
        assert ct_etotal.meas.regression_cols["poa"] == "e_total"
        assert ct_etotal.sim.regression_cols["poa"] == "e_total"

    def test_end_to_end_bifi_power_tc(self, ct_bifi_power_tc):
        """Bifacial power-temp-corrected preset runs end-to-end.

        Verifies that the ``power_temp_correct`` calculated column was added to
        both CapData instances during ``setup()``, that the scatter layout has
        two panels (one per rhs variable: ``poa`` and ``rpoa``), and that the
        cap ratio is plausible.
        """
        assert "power_temp_correct" in ct_bifi_power_tc.meas.data.columns
        assert "power_temp_correct" in ct_bifi_power_tc.sim.data.columns

        # scatter_plots delegates to the preset's scatter_bifi_power_tc
        # callable, which returns a 2-panel Layout (one for each rhs term).
        layout = ct_bifi_power_tc.scatter_plots()
        import holoviews as hv

        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

        self._run_canonical_sequence(ct_bifi_power_tc)
        cap_ratio = ct_bifi_power_tc.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2

    def test_end_to_end_bifi_power_tc_meas_tbom(self, ct_bifi_power_tc_meas_tbom):
        """bifi_power_tc_meas_tbom preset runs end-to-end.

        Verifies that the ``power_temp_correct`` calculated column was added to
        both CapData instances during ``setup()`` using the measured BOM
        temperature on the meas side, and that the scatter layout has two
        panels. Cap ratio is checked for plausibility.
        """
        assert "power_temp_correct" in ct_bifi_power_tc_meas_tbom.meas.data.columns
        assert "power_temp_correct" in ct_bifi_power_tc_meas_tbom.sim.data.columns

        layout = ct_bifi_power_tc_meas_tbom.scatter_plots()
        import holoviews as hv

        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

        self._run_canonical_sequence(ct_bifi_power_tc_meas_tbom)
        cap_ratio = ct_bifi_power_tc_meas_tbom.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2
