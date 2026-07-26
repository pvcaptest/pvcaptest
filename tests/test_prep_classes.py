"""Tests for the prep step classes in captest.prep."""

import numpy as np
import pandas as pd
import pytest

from captest import capdata as pvc
from captest import prep
from captest.columngroups import ColumnGroups


@pytest.fixture
def cd():
    """CapData with two temperature columns, wind, and power."""
    index = pd.date_range("1/1/2021 12:00", freq="5min", periods=4)
    data = pd.DataFrame(
        {
            "temp_amb_1": [32.0, 41.0, 50.0, 212.0],
            "temp_bom_1": [77.0, 86.0, 95.0, 104.0],
            "wind_1": [10.0, 20.0, 30.0, 40.0],
            "power_1": [1000.0, 2000.0, 3000.0, 4000.0],
        },
        index=index,
    )
    out = pvc.CapData("test")
    out.data = data
    out.column_groups = ColumnGroups(
        {
            "temp_amb": ["temp_amb_1"],
            "temp_bom": ["temp_bom_1"],
            "wind": ["wind_1"],
            "real_pwr": ["power_1"],
        }
    )
    return out


class TestScale:
    def test_scales_named_columns_in_place(self, cd):
        prep.Scale(columns=["power_1"], factor=0.001).run(cd)
        assert cd.data["power_1"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_offset_applied_after_factor(self, cd):
        prep.Scale(columns=["wind_1"], factor=2.0, offset=1.0).run(cd)
        assert cd.data["wind_1"].tolist() == [21.0, 41.0, 61.0, 81.0]

    def test_appends_step_to_prep(self, cd):
        step = prep.Scale(columns=["power_1"], factor=0.001)
        step.run(cd)
        assert cd.prep == [step]
        assert step.columns_resolved == ["power_1"]

    def test_leaves_other_columns_untouched(self, cd):
        prep.Scale(columns=["power_1"], factor=0.001).run(cd)
        assert cd.data["wind_1"].tolist() == [10.0, 20.0, 30.0, 40.0]

    def test_config_round_trip(self, cd):
        step = prep.Scale(columns=["power_1"], factor=0.001, custom_name="W to kW")
        config = step.to_config()
        assert config["type"] == "Scale"
        rebuilt = prep.prep_step_from_config(config)
        assert isinstance(rebuilt, prep.Scale)
        assert rebuilt.columns == ["power_1"]
        assert rebuilt.factor == 0.001
        assert rebuilt.custom_name == "W to kW"

    def test_args_repr_and_explanation(self, cd):
        step = prep.Scale(columns=["power_1"], factor=0.001)
        step.run(cd)
        assert "factor=0.001" in step.args_repr
        assert "power_1" in step.explanation

    def test_non_numeric_column_raises_type_error(self, cd):
        cd.data["power_1"] = ["a", "b", "c", "d"]
        with pytest.raises(TypeError, match="power_1"):
            prep.Scale(columns=["power_1"], factor=0.001).run(cd)


class TestSelectors:
    def test_group_selector(self, cd):
        prep.Scale(group="wind", factor=2.0).run(cd)
        assert cd.data["wind_1"].tolist() == [20.0, 40.0, 60.0, 80.0]

    def test_list_valued_group_selector(self, cd):
        step = prep.Scale(group=["temp_amb", "temp_bom"], factor=1.0)
        step.run(cd)
        assert step.columns_resolved == ["temp_amb_1", "temp_bom_1"]

    def test_group_regex_spans_several_ids(self, cd):
        step = prep.Scale(group_regex="^temp", factor=1.0)
        step.run(cd)
        assert step.columns_resolved == ["temp_amb_1", "temp_bom_1"]

    def test_two_selectors_raises(self, cd):
        with pytest.raises(ValueError, match="exactly one"):
            prep.Scale(columns=["wind_1"], group="wind", factor=1.0).run(cd)

    def test_no_selector_raises(self, cd):
        with pytest.raises(ValueError, match="exactly one"):
            prep.Scale(factor=1.0).run(cd)

    def test_unknown_group_raises_with_close_match(self, cd):
        with pytest.raises(ValueError, match="Did you mean 'wind'"):
            prep.Scale(group="wnd", factor=1.0).run(cd)

    def test_group_regex_matching_nothing_raises(self, cd):
        with pytest.raises(ValueError, match="nomatch"):
            prep.Scale(group_regex="nomatch", factor=1.0).run(cd)

    def test_column_missing_from_data_raises(self, cd):
        with pytest.raises(ValueError, match="absent"):
            prep.Scale(columns=["not_a_column"], factor=1.0).run(cd)


class TestAtomicFailure:
    def test_failed_step_restores_data_and_records_nothing(self, cd):
        cd.data["power_1"] = ["a", "b", "c", "d"]
        before_bad = cd.data.copy()
        with pytest.raises(TypeError):
            prep.Scale(columns=["power_1", "wind_1"], factor=0.001).run(cd)
        assert cd.prep == []
        pd.testing.assert_frame_equal(cd.data, before_bad)

    def test_failed_step_restores_column_groups_too(self, cd):
        """A step that reaches column_groups before failing rolls both back."""

        def drop_then_fail(capdata):
            capdata.drop_cols(["power_1"], record=False)
            raise RuntimeError("boom")

        before = cd.data.copy()
        before_groups = {k: list(v) for k, v in cd.column_groups.items()}
        with pytest.raises(RuntimeError, match="boom"):
            prep.Custom(drop_then_fail).run(cd)
        assert cd.prep == []
        pd.testing.assert_frame_equal(cd.data, before)
        assert {k: list(v) for k, v in cd.column_groups.items()} == before_groups


class TestOrderingGuard:
    def test_value_mutating_step_after_filtering_raises(self, cd):
        cd.data["poa"] = [500.0, 600.0, 700.0, 800.0]
        cd.filter_custom(pd.DataFrame.head, 3)
        with pytest.raises(RuntimeError, match="reset_filter"):
            prep.Scale(columns=["power_1"], factor=0.001).run(cd)

    def test_succeeds_after_reset_filter(self, cd):
        cd.data["poa"] = [500.0, 600.0, 700.0, 800.0]
        cd.filter_custom(pd.DataFrame.head, 3)
        cd.reset_filter()
        prep.Scale(columns=["power_1"], factor=0.001).run(cd)
        assert cd.data["power_1"].tolist() == [1.0, 2.0, 3.0, 4.0]


class TestStrictAttrsOnPrepSteps:
    def test_mistyped_param_raises(self):
        step = prep.Scale(columns=["a"], factor=1.0)
        with pytest.raises(AttributeError, match="Did you mean 'factor'"):
            step.factr = 2.0


class TestRegistry:
    def test_scale_is_registered(self):
        assert prep.PREP_REGISTRY["Scale"] is prep.Scale

    def test_unknown_type_raises_with_close_match(self):
        with pytest.raises(ValueError, match="Did you mean 'Scale'"):
            prep.prep_step_from_config({"type": "Scaled"})


class TestConvertUnits:
    def test_fahrenheit_to_celsius_is_affine(self, cd):
        prep.ConvertUnits(columns=["temp_amb_1"], from_units="F", to_units="C").run(cd)
        assert cd.data["temp_amb_1"].tolist() == pytest.approx([0.0, 5.0, 10.0, 100.0])

    def test_scale_alone_cannot_express_it(self, cd):
        """32 F is 0 C, not 17.8 — the case that disproves calcparams.scale."""
        prep.Scale(columns=["temp_amb_1"], factor=5 / 9).run(cd)
        assert cd.data["temp_amb_1"].iloc[0] == pytest.approx(17.78, abs=0.01)

    def test_mph_to_meters_per_second(self, cd):
        prep.ConvertUnits(group="wind", from_units="mph", to_units="m/s").run(cd)
        assert cd.data["wind_1"].iloc[0] == pytest.approx(4.4704)

    def test_group_regex_converts_every_temperature_group(self, cd):
        prep.ConvertUnits(group_regex="^temp", from_units="F", to_units="C").run(cd)
        assert cd.data["temp_amb_1"].iloc[0] == pytest.approx(0.0)
        assert cd.data["temp_bom_1"].iloc[0] == pytest.approx(25.0)

    def test_round_trip_within_tolerance(self, cd):
        original = cd.data["temp_amb_1"].tolist()
        prep.ConvertUnits(columns=["temp_amb_1"], from_units="F", to_units="C").run(cd)
        cd.prep = []
        prep.ConvertUnits(columns=["temp_amb_1"], from_units="C", to_units="F").run(cd)
        assert cd.data["temp_amb_1"].tolist() == pytest.approx(original)

    def test_aliases_are_case_insensitive(self, cd):
        prep.ConvertUnits(
            columns=["temp_amb_1"], from_units="degF", to_units="Celsius"
        ).run(cd)
        assert cd.data["temp_amb_1"].iloc[0] == pytest.approx(0.0)

    def test_unknown_pair_raises_listing_supported(self, cd):
        with pytest.raises(ValueError, match="supported"):
            prep.ConvertUnits(
                columns=["wind_1"], from_units="furlongs", to_units="m/s"
            ).run(cd)

    def test_unsupported_pair_hint_targets_the_wrong_unit(self, cd):
        """A valid from_units must not be echoed back; the typo'd to_units is."""
        with pytest.raises(ValueError) as exc:
            prep.ConvertUnits(columns=["wind_1"], from_units="mph", to_units="m/z").run(
                cd
            )
        message = str(exc.value)
        assert "for from_units" not in message
        assert "'m/s' for to_units" in message

    def test_identical_units_raise(self, cd):
        with pytest.raises(ValueError, match="identical"):
            prep.ConvertUnits(columns=["wind_1"], from_units="C", to_units="C").run(cd)

    def test_missing_units_raise_clear_message(self, cd):
        with pytest.raises(ValueError, match="both required"):
            prep.ConvertUnits(columns=["wind_1"], to_units="m/s").run(cd)

    def test_config_round_trip(self):
        step = prep.ConvertUnits(group_regex="^temp", from_units="F", to_units="C")
        rebuilt = prep.prep_step_from_config(step.to_config())
        assert isinstance(rebuilt, prep.ConvertUnits)
        assert rebuilt.group_regex == "^temp"
        assert rebuilt.from_units == "F"

    def test_explanation_names_the_units(self, cd):
        step = prep.ConvertUnits(group="wind", from_units="mph", to_units="m/s")
        step.run(cd)
        assert "mph" in step.explanation and "m/s" in step.explanation

    def test_inverse_is_derived_not_tabulated(self):
        forward = prep.conversion_factors("F", "C")
        inverse = prep.conversion_factors("C", "F")
        value = 68.0
        celsius = value * forward[0] + forward[1]
        assert celsius * inverse[0] + inverse[1] == pytest.approx(value)

    def test_registered(self):
        assert prep.PREP_REGISTRY["ConvertUnits"] is prep.ConvertUnits


class TestAsType:
    def test_casts_selected_columns(self, cd):
        cd.data["power_1"] = ["1000", "2000", "3000", "4000"]
        prep.AsType(columns=["power_1"], dtype="float64").run(cd)
        assert cd.data["power_1"].dtype == np.dtype("float64")

    def test_config_round_trip(self):
        step = prep.AsType(columns=["a"], dtype="float64")
        rebuilt = prep.prep_step_from_config(step.to_config())
        assert rebuilt.dtype == "float64"

    def test_uncastable_value_raises_and_restores(self, cd):
        cd.data["power_1"] = ["1000", "not_a_number", "3000", "4000"]
        before = cd.data.copy()
        with pytest.raises(ValueError):
            prep.AsType(columns=["power_1"], dtype="float64").run(cd)
        pd.testing.assert_frame_equal(cd.data, before)
        assert cd.prep == []


class TestDropColumns:
    def test_drops_from_data_and_column_groups(self, cd):
        prep.DropColumns(columns=["wind_1"]).run(cd)
        assert "wind_1" not in cd.data.columns
        assert cd.column_groups["wind"] == []

    def test_permitted_after_filtering(self, cd):
        cd.data["poa"] = [500.0, 600.0, 700.0, 800.0]
        cd.filter_custom(pd.DataFrame.head, 3)
        prep.DropColumns(columns=["wind_1"]).run(cd)
        assert "wind_1" not in cd.data.columns

    def test_does_not_mutate_values(self):
        assert prep.DropColumns._mutates_values is False


class TestRenameColumns:
    def test_renames_in_data_and_column_groups(self, cd):
        prep.RenameColumns(column_map={"wind_1": "wind_speed"}).run(cd)
        assert "wind_speed" in cd.data.columns
        assert cd.column_groups["wind"] == ["wind_speed"]

    def test_selector_is_rejected(self, cd):
        with pytest.raises(ValueError, match="column_map"):
            prep.RenameColumns(columns=["wind_1"], column_map={"wind_1": "x"}).run(cd)

    def test_unknown_key_raises(self, cd):
        with pytest.raises(ValueError, match="absent"):
            prep.RenameColumns(column_map={"nope": "x"}).run(cd)

    def test_config_round_trip(self):
        step = prep.RenameColumns(column_map={"a": "b"})
        rebuilt = prep.prep_step_from_config(step.to_config())
        assert rebuilt.column_map == {"a": "b"}


def blank_before(capdata, columns, hour):
    """Module-level prep function used by the Custom tests."""
    mask = capdata.data.index.hour < hour
    capdata.data.loc[mask, columns] = np.nan


class TestCustomPrep:
    def test_calls_func_with_capdata(self, cd):
        prep.Custom(blank_before, ["power_1"], 13).run(cd)
        assert cd.data["power_1"].isna().all()

    def test_no_selector_required(self, cd):
        step = prep.Custom(blank_before, ["power_1"], 13)
        step.run(cd)
        assert step.columns_resolved == []

    def test_config_round_trip(self, cd):
        step = prep.Custom(blank_before, ["power_1"], 13)
        config = step.to_config()
        assert config["func"] == "tests.test_prep_classes:blank_before"
        rebuilt = prep.prep_step_from_config(config)
        assert rebuilt.func is blank_before
        assert rebuilt.args == (["power_1"], 13)

    def test_lambda_raises_at_serialization(self, cd):
        step = prep.Custom(lambda capdata: None)
        with pytest.raises(ValueError):
            step.to_config()

    def test_failure_restores_data(self, cd):
        def boom(capdata):
            capdata.data["power_1"] = 0.0
            raise RuntimeError("boom")

        before = cd.data.copy()
        with pytest.raises(RuntimeError, match="boom"):
            prep.Custom(boom).run(cd)
        pd.testing.assert_frame_equal(cd.data, before)
        assert cd.prep == []


class TestRegistryCoverage:
    @pytest.mark.parametrize("name", list(prep.PREP_REGISTRY))
    def test_every_step_round_trips_its_defaults(self, name):
        cls = prep.PREP_REGISTRY[name]
        assert issubclass(cls, prep.BasePrepStep)
        assert cls.__name__ == name
