"""Tests for the prep step classes in captest.prep."""

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
