"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import numpy as np
import pandas as pd
import pytest

from captest.capdata import CapData
from captest.filters import BaseSummaryStep, BaseFilter


def _make_capdata(n=5):
    """CapData with a tiny integer-indexed DataFrame and data_filtered set."""
    cd = CapData("test")
    cd.data = pd.DataFrame({"power": np.arange(n), "poa": np.arange(n) * 10.0})
    cd.data_filtered = cd.data.copy()
    return cd


class _DropFirstRow(BaseFilter):
    """Test-only filter: drops the first remaining row."""

    def _execute(self, capdata):
        return capdata.data_filtered.index[1:]


class TestBaseSummaryStep:
    def test_base_filter_is_summary_step(self):
        assert issubclass(BaseFilter, BaseSummaryStep)

    def test_custom_name_defaults_to_none(self):
        assert _DropFirstRow().custom_name is None

    def test_execute_not_implemented_on_base(self):
        cd = _make_capdata()
        with pytest.raises(NotImplementedError):
            BaseSummaryStep().run(cd)

    def test_run_records_runtime_state(self):
        cd = _make_capdata(n=5)
        step = _DropFirstRow()
        step.run(cd)
        assert step.pts_before == 5
        assert step.pts_after == 4
        assert step.pts_removed == 1
        assert list(step.ix_before) == [0, 1, 2, 3, 4]
        assert list(step.ix_after) == [1, 2, 3, 4]

    def test_run_appends_to_filters(self):
        cd = _make_capdata()
        step = _DropFirstRow()
        step.run(cd)
        assert cd.filters == [step]

    def test_run_reassigns_filters_list(self):
        """run() must reassign (not in-place append) so param watchers fire."""
        cd = _make_capdata()
        original = cd.filters
        _DropFirstRow().run(cd)
        assert cd.filters is not original

    def test_run_updates_data_filtered_transitionally(self):
        cd = _make_capdata(n=5)
        _DropFirstRow().run(cd)
        assert list(cd.data_filtered.index) == [1, 2, 3, 4]

    def test_run_warns_when_all_data_removed(self):
        cd = _make_capdata(n=1)
        with pytest.warns(UserWarning, match="removed all data"):
            _DropFirstRow().run(cd)

    def test_args_repr_default(self):
        assert _DropFirstRow().args_repr == "Default arguments"


class TestCapDataFiltersParam:
    def test_capdata_is_parameterized(self):
        import param

        assert isinstance(CapData("x"), param.Parameterized)

    def test_filters_is_a_param_list(self):
        cd = CapData("x")
        assert "filters" in cd.param
        assert cd.filters == []

    def test_default_filters_distinct_per_instance(self):
        a, b = CapData("a"), CapData("b")
        assert a.filters is not b.filters

    def test_name_preserved_and_constant(self):
        cd = CapData("system_a")
        assert cd.name == "system_a"
        with pytest.raises(TypeError):
            cd.name = "other"

    def test_filters_item_type_enforced(self):
        cd = CapData("x")
        with pytest.raises(TypeError):
            cd.filters = ["not a step"]

    def test_run_triggers_filters_watcher(self):
        cd = _make_capdata()
        events = []
        cd.param.watch(lambda e: events.append(e), "filters")
        _DropFirstRow().run(cd)
        assert len(events) == 1
        assert isinstance(events[0].new[-1], _DropFirstRow)

    def test_copy_copies_filters_and_name(self):
        cd = _make_capdata()
        _DropFirstRow().run(cd)
        cd_c = cd.copy()
        assert len(cd_c.filters) == 1
        assert cd_c.filters is not cd.filters
        assert cd_c.name == cd.name

    def test_reset_filter_clears_filters(self):
        cd = _make_capdata()
        _DropFirstRow().run(cd)
        assert len(cd.filters) == 1
        cd.reset_filter()
        assert cd.filters == []
