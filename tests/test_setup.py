"""Tests for ``captest.setup``: node models, documents, derive, tiers."""

import json

import pytest
import yaml
from pydantic import ValidationError

from captest import setup
from captest.setup import Calc, Column, Group, Side, TestSetup

E2848_FML = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"


def e2848_doc(**changes):
    """A complete e2848_default-shaped document as plain data."""
    doc = {
        "name": "e2848_default",
        "reg_fml": E2848_FML,
        "meas": {
            "reg_cols": {
                "power": {"group": "real_pwr_mtr", "agg": "sum"},
                "poa": {"group": "irr_poa"},
                "t_amb": {"group": "temp_amb"},
                "w_vel": {"group": "wind_speed"},
            }
        },
        "sim": {
            "reg_cols": {
                "power": {"column": "E_Grid"},
                "poa": {"column": "GlobInc"},
                "t_amb": {"column": "T_Amb"},
                "w_vel": {"column": "WindVel"},
            }
        },
        "rep_conditions": {
            "func": {"poa": "perc_60", "t_amb": "mean", "w_vel": "mean"}
        },
    }
    doc.update(changes)
    return doc


def _paths(exc):
    return [".".join(str(p) for p in err["loc"]) for err in exc.value.errors()]


@pytest.fixture
def probe_calc(monkeypatch):
    """Register a calculation whose signature accepts the literal kinds under test."""
    from captest.calcparams import CALC_REGISTRY, CalcEntry

    def probe(data, col=None, factor=1.0, flag=None, xs=None):
        return data[col] * factor

    monkeypatch.setitem(CALC_REGISTRY, "probe", CalcEntry(probe, (), ()))
    return probe


class TestNodes:
    def test_group_defaults_agg_to_mean(self):
        assert Group(group="irr_poa").agg == "mean"

    def test_group_is_hashable_and_equal_by_value(self):
        assert hash(Group(group="a")) == hash(Group(group="a", agg="mean"))
        assert Group(group="a") == Group(group="a", agg="mean")

    def test_group_rejects_unknown_agg(self):
        with pytest.raises(ValidationError):
            Group(group="a", agg="product")

    def test_node_kind_is_decided_by_the_tag(self):
        side = Side.model_validate(
            {"reg_cols": {"x": {"group": "g"}, "y": {"column": "c"}}}
        )
        assert isinstance(side.reg_cols["x"], Group)
        assert isinstance(side.reg_cols["y"], Column)

    def test_mapping_without_a_tag_is_an_error_not_a_literal(self):
        with pytest.raises(ValidationError) as exc:
            Calc(calc="e_total", args={"poa": {"grp": "irr_poa"}})
        assert any("args.poa" in p for p in _paths(exc))

    def test_mapping_with_two_tags_is_an_error(self):
        with pytest.raises(ValidationError):
            Calc(calc="e_total", args={"poa": {"group": "a", "column": "b"}})

    def test_literals_pass_through_inside_args(self, probe_calc):
        node = Calc(
            calc="probe",
            args={
                "col": {"column": "PrecWat"},
                "factor": 100,
                "flag": None,
                "xs": [1, 2],
            },
        )
        assert node.args["factor"] == 100
        assert node.args["flag"] is None
        assert node.args["xs"] == [1, 2]

    def test_non_finite_float_literal_is_rejected(self, probe_calc):
        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": float("nan")})

    def test_dict_literal_is_rejected(self, probe_calc):
        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": {"a": 1}})

    def test_callable_argument_is_rejected(self, probe_calc):
        import numpy as np

        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": np.mean})

    def test_unknown_calc_names_the_path_and_suggests(self):
        with pytest.raises(ValidationError) as exc:
            Side.model_validate({"reg_cols": {"poa": {"calc": "e_totl", "args": {}}}})
        assert any("reg_cols.poa" in p for p in _paths(exc))
        assert "e_total" in str(exc.value)

    def test_unknown_arg_key_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown argument"):
            Calc(
                calc="e_total",
                args={"poa": {"group": "a"}, "rpoa": {"group": "b"}, "x": 1},
            )

    def test_missing_required_arg_is_rejected(self):
        with pytest.raises(ValidationError, match="missing"):
            Calc(calc="e_total", args={"poa": {"group": "a"}})

    def test_injected_param_may_be_given_explicitly(self):
        node = Calc(
            calc="power_temp_correct",
            args={
                "power": {"group": "p", "agg": "sum"},
                "cell_temp": {"column": "T"},
                "base_temp": 20,
            },
        )
        assert node.args["base_temp"] == 20

    def test_calc_with_a_reserved_output_parameter_is_rejected(self, monkeypatch):
        from captest.calcparams import CALC_REGISTRY, CalcEntry

        def clash(data, col=None, output=None):
            return data[col]

        monkeypatch.setitem(CALC_REGISTRY, "clash", CalcEntry(clash, (), ()))
        with pytest.raises(ValidationError, match="reserved"):
            Calc(calc="clash", args={"col": {"column": "a"}})

    def test_extra_keys_are_forbidden(self):
        with pytest.raises(ValidationError):
            Group(group="a", aggr="mean")


class TestOutputNames:
    def test_group_writes_agg_column(self):
        assert setup.agg_column_name("irr_poa", "mean") == "irr_poa_mean_agg"

    def test_calc_writes_registry_name(self):
        assert (
            setup.calc_output_name(
                Calc(
                    calc="e_total", args={"poa": {"group": "a"}, "rpoa": {"group": "b"}}
                )
            )
            == "e_total"
        )

    def test_same_calc_different_args_on_one_side_is_rejected(self):
        with pytest.raises(ValidationError, match="e_total"):
            Side.model_validate(
                {
                    "reg_cols": {
                        "poa": {
                            "calc": "e_total",
                            "args": {"poa": {"group": "a"}, "rpoa": {"group": "b"}},
                        },
                        "poa2": {
                            "calc": "e_total",
                            "args": {"poa": {"group": "a"}, "rpoa": {"group": "c"}},
                        },
                    }
                }
            )

    def test_identical_calc_nodes_are_permitted(self):
        node = {
            "calc": "e_total",
            "args": {"poa": {"group": "a"}, "rpoa": {"group": "b"}},
        }
        Side.model_validate({"reg_cols": {"poa": node, "poa2": node}})

    def test_top_level_literal_is_rejected(self):
        with pytest.raises(ValidationError, match="literal"):
            Side.model_validate({"reg_cols": {"poa": "irr_poa"}})


class TestTestSetupDocument:
    def test_loads_a_complete_document(self):
        tsd = TestSetup.model_validate(e2848_doc())
        assert isinstance(tsd.meas.reg_cols["power"], Group)
        assert tsd.rep_conditions.percent_filter == 20

    def test_formula_variable_missing_from_a_side_is_rejected(self):
        doc = e2848_doc()
        del doc["sim"]["reg_cols"]["w_vel"]
        with pytest.raises(ValidationError) as exc:
            TestSetup.model_validate(doc)
        assert any(p.startswith("sim") for p in _paths(exc))
        assert "w_vel" in str(exc.value)

    def test_unparseable_formula_is_a_located_validation_error(self):
        with pytest.raises(ValidationError) as exc:
            TestSetup.model_validate(e2848_doc(reg_fml="power ~ poa +"))
        assert any("reg_fml" in p for p in _paths(exc))

    def test_rep_conditions_func_key_not_in_rhs_is_rejected(self):
        doc = e2848_doc(rep_conditions={"func": {"ghi": "mean"}})
        with pytest.raises(ValidationError, match="ghi"):
            TestSetup.model_validate(doc)

    def test_rep_conditions_func_value_must_be_mean_median_or_perc(self):
        doc = e2848_doc(rep_conditions={"func": {"poa": "p60"}})
        with pytest.raises(ValidationError, match="perc_N"):
            TestSetup.model_validate(doc)

    def test_percent_filter_is_numeric_only(self):
        doc = e2848_doc(rep_conditions={"percent_filter": [10, 20]})
        with pytest.raises(ValidationError):
            TestSetup.model_validate(doc)

    def test_params_keys_must_be_downstream_params(self):
        with pytest.raises(ValidationError, match="params"):
            TestSetup.model_validate(e2848_doc(params={"ac_nameplate": 1}))

    def test_params_accepts_downstream_param(self):
        assert TestSetup.model_validate(e2848_doc(params={"rear_shade": 0})).params == {
            "rear_shade": 0
        }

    def test_setup_is_equal_by_value_and_not_hashable(self):
        a, b = (
            TestSetup.model_validate(e2848_doc()),
            TestSetup.model_validate(e2848_doc()),
        )
        assert a == b
        with pytest.raises(TypeError):
            hash(a)


class TestNormalisationAndIdentity:
    def test_to_dict_materialises_defaults_and_keeps_nulls(self):
        d = TestSetup.model_validate(e2848_doc()).to_dict()
        assert d["meas"]["reg_cols"]["poa"] == {"group": "irr_poa", "agg": "mean"}
        assert d["rep_conditions"]["w_vel"] is None
        assert d["derived_from"] is None

    def test_to_dict_is_plain_json_types(self):
        d = TestSetup.model_validate(e2848_doc()).to_dict()
        json.dumps(d, allow_nan=False)

    def test_normalisation_is_idempotent(self):
        first = TestSetup.model_validate(e2848_doc()).to_dict()
        second = TestSetup.model_validate(first).to_dict()
        assert first == second

    def test_explicit_null_and_omitted_field_digest_equal(self):
        with_null = e2848_doc(
            rep_conditions={
                "func": {"poa": "perc_60", "t_amb": "mean", "w_vel": "mean"},
                "w_vel": None,
            }
        )
        assert (
            TestSetup.model_validate(with_null).content_digest()
            == TestSetup.model_validate(e2848_doc()).content_digest()
        )

    def test_content_digest_changes_with_content(self):
        base = TestSetup.model_validate(e2848_doc()).content_digest()
        doc = e2848_doc()
        doc["meas"]["reg_cols"]["power"] = {"group": "real_pwr_inv", "agg": "sum"}
        assert TestSetup.model_validate(doc).content_digest() != base

    def test_canonical_json_rejects_nan_and_non_string_keys(self):
        with pytest.raises(ValueError):
            setup.canonical_json({"x": float("nan")})
        with pytest.raises(TypeError):
            setup.canonical_json({1: "x"})

    def test_load_from_yaml_path_json_path_text_and_mapping(self, tmp_path):
        doc = e2848_doc()
        y = tmp_path / "s.yaml"
        y.write_text(yaml.safe_dump(doc))
        j = tmp_path / "s.json"
        j.write_text(json.dumps(doc))
        from_yaml = TestSetup.load(y)
        assert TestSetup.load(j) == from_yaml
        assert TestSetup.load(str(j)) == from_yaml
        assert TestSetup.loads(y.read_text()) == from_yaml
        assert TestSetup.loads(json.dumps(doc)) == from_yaml  # long text, never a path
        assert TestSetup.load(doc) == from_yaml

    def test_to_yaml_and_to_json_round_trip(self, tmp_path):
        tsd = TestSetup.model_validate(e2848_doc())
        tsd.to_yaml(tmp_path / "out.yaml")
        tsd.to_json(tmp_path / "out.json")
        assert TestSetup.load(tmp_path / "out.yaml") == tsd
        assert TestSetup.load(tmp_path / "out.json") == tsd

    def test_json_schema_marks_extra_keys_forbidden(self):
        schema = TestSetup.json_schema()
        assert schema["additionalProperties"] is False
        assert "Group" in schema["$defs"]


class TestParseRegressionFormula:
    def test_reexported_from_util(self):
        from captest import util

        assert util.parse_regression_formula is setup.parse_regression_formula


class TestDerive:
    def _base(self):
        return TestSetup.model_validate(e2848_doc())

    def test_replaces_one_term_and_keeps_the_others(self):
        out = setup.derive(
            self._base(),
            reg_cols_meas={"power": {"group": "real_pwr_inv", "agg": "sum"}},
        )
        assert out.meas.reg_cols["power"] == Group(group="real_pwr_inv", agg="sum")
        assert out.meas.reg_cols["poa"] == Group(group="irr_poa")
        assert out.sim == self._base().sim

    def test_accepts_model_nodes_as_well_as_mappings(self):
        out = setup.derive(self._base(), reg_cols_meas={"poa": Group(group="irr_ghi")})
        assert out.meas.reg_cols["poa"].group == "irr_ghi"

    def test_records_provenance_and_keeps_name(self):
        out = setup.derive(self._base(), reg_fml="power ~ poa")
        assert out.derived_from == "e2848_default"
        assert out.name == "e2848_default"

    def test_null_removes_a_term_and_prunes_rep_conditions_func(self):
        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) - 1"
        out = setup.derive(
            self._base(),
            reg_fml=fml,
            reg_cols_meas={"w_vel": None},
            reg_cols_sim={"w_vel": None},
        )
        assert "w_vel" not in out.meas.reg_cols
        assert "w_vel" not in out.sim.reg_cols
        assert set(out.rep_conditions.func) == {"poa", "t_amb"}
        assert out.rep_conditions.func["poa"] == "perc_60"

    def test_pruning_only_removes(self):
        out = setup.derive(self._base(), rep_conditions={"func": {"poa": "perc_55"}})
        assert out.rep_conditions.func == {"poa": "perc_55"}

    def test_null_for_a_term_the_base_lacks_is_rejected(self):
        with pytest.raises(setup.DerivationError, match="ghi"):
            setup.derive(self._base(), reg_cols_meas={"ghi": None})

    def test_override_key_that_is_not_a_formula_variable_is_rejected(self):
        with pytest.raises(setup.DerivationError, match="ghi"):
            setup.derive(self._base(), reg_cols_meas={"ghi": {"group": "irr_ghi"}})

    def test_removing_a_term_the_formula_still_uses_is_rejected(self):
        with pytest.raises(ValidationError, match="w_vel"):
            setup.derive(self._base(), reg_cols_meas={"w_vel": None})

    def test_other_fields_replace_wholesale(self):
        out = setup.derive(
            self._base(), params={"rear_shade": 0}, scatter_plots="etotal"
        )
        assert out.params == {"rear_shade": 0}
        assert out.scatter_plots == "etotal"

    def test_result_is_complete_and_digests_like_a_fresh_document(self):
        out = setup.derive(
            self._base(),
            reg_cols_meas={"power": {"group": "real_pwr_inv", "agg": "sum"}},
        )
        doc = out.to_dict()
        doc.pop("derived_from")
        fresh = e2848_doc()
        fresh["meas"]["reg_cols"]["power"] = {"group": "real_pwr_inv", "agg": "sum"}
        expected = TestSetup.model_validate(fresh).to_dict()
        expected.pop("derived_from")
        assert doc == expected

    def test_staticmethod_delegates_to_module_function(self):
        base = self._base()
        assert TestSetup.derive(base, reg_fml="power ~ poa") == setup.derive(
            base, reg_fml="power ~ poa"
        )


class _Cd:
    """Duck-typed CapData for tier-2 tests."""

    def __init__(self, groups, columns=(), **attrs):
        import pandas as pd

        self.column_groups = groups
        all_cols = [c for cols in groups.values() for c in cols] + list(columns)
        self.data = pd.DataFrame(columns=all_cols)
        for key, value in attrs.items():
            setattr(self, key, value)


def _fit(doc, side="meas", **cd_kwargs):
    return setup.check_project_fit(
        TestSetup.model_validate(doc), side, _Cd(**cd_kwargs)
    )


MEAS_GROUPS = {
    "real_pwr_mtr": ["meter_power"],
    "irr_poa": ["poa1", "poa2"],
    "temp_amb": ["ta1"],
    "wind_speed": ["ws1"],
}


class TestCheckProjectFit:
    def test_clean_project_has_no_errors(self):
        assert _fit(e2848_doc(), groups=MEAS_GROUPS) == []

    def test_missing_group_is_reported_with_path(self):
        groups = {k: v for k, v in MEAS_GROUPS.items() if k != "wind_speed"}
        errors = _fit(e2848_doc(), groups=groups)
        assert [e.path for e in errors] == ["meas.reg_cols.w_vel.group"]
        assert "wind_speed" in errors[0].message

    def test_missing_column_on_sim_side(self):
        errors = _fit(e2848_doc(), side="sim", groups={}, columns=["E_Grid", "GlobInc"])
        assert {e.path for e in errors} == {
            "sim.reg_cols.t_amb.column",
            "sim.reg_cols.w_vel.column",
        }

    def _tc_doc(self, **args):
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {
                "calc": "power_temp_correct",
                "args": {
                    "power": {"group": "real_pwr_mtr", "agg": "sum"},
                    "cell_temp": {"column": "bom"},
                    **args,
                },
            },
            "poa": {"group": "irr_poa"},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        return doc

    def test_requires_param_none_by_every_route_is_reported(self):
        # 1. CapData attribute None (power_temp_correct's own default is None)
        errors = _fit(
            self._tc_doc(),
            groups=MEAS_GROUPS,
            columns=["bom"],
            power_temp_coeff=None,
            base_temp=25,
        )
        assert [e.path for e in errors] == ["meas.reg_cols.power"]
        assert "power_temp_coeff" in errors[0].message
        # 2. explicit null in args
        errors = _fit(
            self._tc_doc(power_temp_coeff=None),
            groups=MEAS_GROUPS,
            columns=["bom"],
            power_temp_coeff=-0.3,
            base_temp=25,
        )
        assert [e.path for e in errors] == ["meas.reg_cols.power"]
        assert "power_temp_coeff" in errors[0].message
        # 3. no attribute at all
        errors = _fit(self._tc_doc(), groups=MEAS_GROUPS, columns=["bom"])
        assert any("power_temp_coeff" in e.message for e in errors)

    def test_requires_param_message_distinguishes_no_value_from_none(self, monkeypatch):
        from captest.calcparams import CALC_REGISTRY, CalcEntry

        def broken(data, real=None, verbose=True):
            return data[real]

        # requires_params names a parameter the function itself does not
        # declare, so effective_value can only resolve it to
        # inspect.Parameter.empty, not None.
        monkeypatch.setitem(
            CALC_REGISTRY, "broken_probe", CalcEntry(broken, ("missing_param",), ())
        )
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {"calc": "broken_probe", "args": {"real": {"group": "irr_poa"}}},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        errors = _fit(doc, groups=MEAS_GROUPS)
        assert any("has no value" in e.message for e in errors)
        assert not any("resolves to None" in e.message for e in errors)

    def test_explicit_arg_satisfies_requires_param(self):
        errors = _fit(
            self._tc_doc(power_temp_coeff=-0.3),
            groups=MEAS_GROUPS,
            columns=["bom"],
            base_temp=25,
        )
        assert errors == []

    def test_function_default_satisfies_requires_param(self):
        # base_temp has a real default (25); no attribute needed.
        errors = _fit(
            self._tc_doc(power_temp_coeff=-0.3), groups=MEAS_GROUPS, columns=["bom"]
        )
        assert errors == []

    def test_requires_import_missing_is_reported(self, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda n: None if n == "pvlib" else real(n)
        )
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {
                "calc": "absolute_airmass",
                "args": {"apparent_zenith": {"column": "z"}, "pressure": None},
            },
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        errors = _fit(
            doc, groups=MEAS_GROUPS, columns=["z"], airmass_model="kastenyoung1989"
        )
        assert any("pvlib" in e.message for e in errors)

    def test_output_shadowing_a_group_id_or_sensor_column(self):
        groups = {**MEAS_GROUPS, "e_total": ["e_total"]}
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {
                "calc": "e_total",
                "args": {"poa": {"group": "irr_poa"}, "rpoa": {"column": "poa2"}},
            },
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        errors = _fit(
            doc, groups=groups, bifaciality=0.7, bifacial_frac=1, rear_shade=0
        )
        assert [e.path for e in errors] == ["meas.reg_cols.poa"]
        assert "e_total" in errors[0].message

    def test_group_output_shadowing_a_sensor_column_is_reported(self):
        groups = {**MEAS_GROUPS, "shadow": ["real_pwr_mtr_sum_agg"]}
        errors = _fit(e2848_doc(), groups=groups)
        assert [e.path for e in errors] == ["meas.reg_cols.power"]
        assert "real_pwr_mtr_sum_agg" in errors[0].message

    def test_generated_aggregate_bookkeeping_does_not_shadow_a_group_node(self):
        # agg_group appends every aggregate it writes to column_groups["agg"];
        # expand_agg_map adds "<key>_aggs" groups of pre-rename subgroup
        # columns. Neither is a real sensor group, so a second setup() of the
        # same instance -- with this bookkeeping now present -- must not
        # report a Group node's own output as shadowing it.
        groups = {
            **MEAS_GROUPS,
            "agg": [
                "real_pwr_mtr_sum_agg",
                "irr_poa_mean_agg",
                "temp_amb_mean_agg",
                "wind_speed_mean_agg",
            ],
            "irr_poa_aggs": ["irr_poa_met1_mean_agg", "irr_poa_met2_mean_agg"],
        }
        assert _fit(e2848_doc(), groups=groups) == []

    def test_calc_argument_colliding_with_a_column_group_id_is_reported(self):
        # CapData.custom_param raises when a parameter absent from args is
        # also a column-group id (the call is ambiguous); tier 2 must report
        # this rather than silently predicting a value evaluation would never
        # actually produce.
        groups = {**MEAS_GROUPS, "base_temp": ["some_column"]}
        errors = _fit(
            self._tc_doc(power_temp_coeff=-0.3), groups=groups, columns=["bom"]
        )
        assert [e.path for e in errors] == ["meas.reg_cols.power"]
        assert "base_temp" in errors[0].message

    def test_explicit_arg_clears_a_column_group_id_collision(self):
        groups = {**MEAS_GROUPS, "base_temp": ["some_column"]}
        errors = _fit(
            self._tc_doc(power_temp_coeff=-0.3, base_temp=20),
            groups=groups,
            columns=["bom"],
        )
        assert errors == []

    def test_collision_check_uses_the_raw_column_groups_including_bookkeeping(
        self, monkeypatch
    ):
        # The collision check must consult the raw cd.column_groups, the same
        # mapping CapData.custom_param raises against -- not the agg/_aggs
        # -filtered groups the shadow checks use -- so a parameter literally
        # named "agg" is still caught even though "agg" is bookkeeping.
        from captest.calcparams import CALC_REGISTRY, CalcEntry

        def probe_agg(data, poa=None, agg=None, verbose=True):
            return data[poa]

        monkeypatch.setitem(CALC_REGISTRY, "probe_agg", CalcEntry(probe_agg, (), ()))
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {"calc": "probe_agg", "args": {"poa": {"group": "irr_poa"}}},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        groups = {**MEAS_GROUPS, "agg": ["real_pwr_mtr_sum_agg"]}

        errors = _fit(doc, groups=groups)
        assert [e.path for e in errors] == ["meas.reg_cols.poa"]
        assert "agg" in errors[0].message

        doc["meas"]["reg_cols"]["poa"]["args"]["agg"] = "mean"
        assert _fit(doc, groups=groups) == []

    def test_params_constraint_checks_effective_value_per_side(self):
        doc = e2848_doc(reg_fml="power ~ poa", params={"rear_shade": 0})
        etotal = {
            "calc": "e_total",
            "args": {"poa": {"group": "irr_poa"}, "rpoa": {"column": "poa2"}},
        }
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": etotal,
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        # meas with rear_shade 0.2 -> violation
        errors = _fit(
            doc, groups=MEAS_GROUPS, bifaciality=0.7, bifacial_frac=1, rear_shade=0.2
        )
        assert any(e.path == "params.rear_shade" for e in errors)
        # meas with rear_shade 0 -> fine
        assert (
            _fit(
                doc, groups=MEAS_GROUPS, bifaciality=0.7, bifacial_frac=1, rear_shade=0
            )
            == []
        )
        # sim side has no rear_shade attribute; e_total's default 0 satisfies it
        doc["sim"]["reg_cols"] = {
            "power": {"column": "E_Grid"},
            "poa": {
                "calc": "e_total",
                "args": {"poa": {"column": "GlobInc"}, "rpoa": {"column": "GlobBak"}},
            },
        }
        assert (
            _fit(
                doc,
                side="sim",
                groups={},
                columns=["E_Grid", "GlobInc", "GlobBak"],
                bifaciality=0.7,
                bifacial_frac=1,
            )
            == []
        )

    def test_side_without_the_param_is_not_checked(self):
        doc = e2848_doc(params={"rear_shade": 0})
        assert _fit(doc, groups=MEAS_GROUPS, rear_shade=0.5) == []

    def test_setup_fit_error_lists_every_path(self):
        errors = [setup.FitError("a", "x"), setup.FitError("b", "y")]
        exc = setup.SetupFitError(errors)
        assert exc.errors == errors
        assert "a: x" in str(exc) and "b: y" in str(exc)
