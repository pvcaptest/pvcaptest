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
