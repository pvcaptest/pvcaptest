"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import numpy as np
import pandas as pd
import param
import pytest

from captest.capdata import CapData
from captest.filters import BaseSummaryStep, BaseFilter, FilterIrr


@pytest.fixture
def make_capdata():
    """Factory fixture: a CapData with an n-row power+poa frame.

    ``data_filtered`` is initialized to a copy of ``data``.
    """

    def _make(n=5):
        cd = CapData("test")
        cd.data = pd.DataFrame({"power": np.arange(n), "poa": np.arange(n) * 10.0})
        cd.data_filtered = cd.data.copy()
        return cd

    return _make


@pytest.fixture
def cd_irr():
    """A CapData with a 5-row poa frame and regression_cols set, for FilterIrr."""
    cd = CapData("irr")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 300.0, 500.0, 700.0, 900.0]},
        index=pd.RangeIndex(5),
    )
    cd.data_filtered = cd.data.copy()
    cd.regression_cols = {"poa": "poa"}
    return cd


class _DropFirstRow(BaseFilter):
    """Test-only filter: drops the first remaining row."""

    def _execute(self, capdata):
        return capdata.data_filtered.index[1:]


class _ConfiguredFilter(BaseFilter):
    """Test-only filter with a non-None default and a None-default param."""

    threshold = param.Number(default=100.0)
    ref_val = param.Number(default=None, allow_None=True)

    def _execute(self, capdata):
        return capdata.data_filtered.index


class TestBaseSummaryStep:
    def test_base_filter_is_summary_step(self):
        assert issubclass(BaseFilter, BaseSummaryStep)

    def test_custom_name_defaults_to_none(self):
        assert _DropFirstRow().custom_name is None

    def test_execute_not_implemented_on_base(self, make_capdata):
        with pytest.raises(NotImplementedError):
            BaseSummaryStep().run(make_capdata())

    def test_run_records_runtime_state(self, make_capdata):
        cd = make_capdata(n=5)
        step = _DropFirstRow()
        step.run(cd)
        assert step.pts_before == 5
        assert step.pts_after == 4
        assert step.pts_removed == 1
        assert list(step.ix_before) == [0, 1, 2, 3, 4]
        assert list(step.ix_after) == [1, 2, 3, 4]

    def test_run_appends_to_filters(self, make_capdata):
        cd = make_capdata()
        step = _DropFirstRow()
        step.run(cd)
        assert cd.filters == [step]

    def test_run_reassigns_filters_list(self, make_capdata):
        """run() must reassign (not in-place append) so param watchers fire."""
        cd = make_capdata()
        original = cd.filters
        _DropFirstRow().run(cd)
        assert cd.filters is not original

    def test_run_updates_data_filtered_transitionally(self, make_capdata):
        cd = make_capdata(n=5)
        _DropFirstRow().run(cd)
        assert list(cd.data_filtered.index) == [1, 2, 3, 4]

    def test_run_warns_when_all_data_removed(self, make_capdata):
        with pytest.warns(UserWarning, match="removed all data"):
            _DropFirstRow().run(make_capdata(n=1))

    def test_args_repr_default(self):
        assert _DropFirstRow().args_repr == "Default arguments"

    def test_args_repr_includes_defaults_and_skips_none(self):
        assert _ConfiguredFilter().args_repr == "threshold=100.0"

    def test_args_repr_includes_overridden_none_param(self):
        assert (
            _ConfiguredFilter(ref_val=5.0).args_repr == "ref_val=5.0, threshold=100.0"
        )


class TestCapDataFiltersParam:
    def test_capdata_is_parameterized(self):
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

    def test_run_triggers_filters_watcher(self, make_capdata):
        cd = make_capdata()
        events = []
        cd.param.watch(lambda e: events.append(e), "filters")
        _DropFirstRow().run(cd)
        assert len(events) == 1
        assert isinstance(events[0].new[-1], _DropFirstRow)

    def test_copy_copies_filters_and_name(self, make_capdata):
        cd = make_capdata()
        _DropFirstRow().run(cd)
        cd_c = cd.copy()
        assert len(cd_c.filters) == 1
        assert cd_c.filters is not cd.filters
        assert cd_c.name == cd.name

    def test_reset_filter_clears_filters(self, make_capdata):
        cd = make_capdata()
        _DropFirstRow().run(cd)
        assert len(cd.filters) == 1
        cd.reset_filter()
        assert cd.filters == []


class TestFilterIrr:
    def test_execute_absolute_bounds(self, cd_irr):
        f = FilterIrr(low=200, high=800)
        assert list(f._execute(cd_irr)) == [1, 2, 3]

    def test_execute_uses_explicit_col_name(self, cd_irr):
        cd_irr.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
        cd_irr.data_filtered = cd_irr.data.copy()
        f = FilterIrr(low=500, high=2000, col_name="ghi")
        assert list(f._execute(cd_irr)) == [4]

    def test_execute_fraction_with_ref_val(self, cd_irr):
        # low/high are fractions of ref_val 500 -> [400, 600]
        f = FilterIrr(low=0.8, high=1.2, ref_val=500)
        assert list(f._execute(cd_irr)) == [2]

    def test_execute_resolves_rep_irr_into_runtime_attr(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")
        f._execute(cd_irr)
        # intent preserved on the param; resolved value on the runtime attr
        assert f.ref_val == "rep_irr"
        assert f.ref_val_resolved == 500.0
        assert isinstance(f.ref_val_resolved, float)

    def test_execute_resolves_self_val(self, cd_irr):
        # 'self_val' is translated to 'rep_irr' and resolved the same way;
        # the param keeps the original 'self_val' intent.
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = FilterIrr(low=0.8, high=1.2, ref_val="self_val")
        f._execute(cd_irr)
        assert f.ref_val == "self_val"
        assert f.ref_val_resolved == 500.0

    def test_args_repr_shows_resolved_ref_val_not_sentinel(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")
        f._execute(cd_irr)
        args = f.args_repr
        assert "rep_irr" not in args
        assert "np." not in args
        assert "ref_val=500.0" in args

    def test_args_repr_numeric_ref_val_unchanged(self):
        f = FilterIrr(low=0.8, high=1.2, ref_val=500)
        # no resolution happened; ref_val_resolved is unset, param value shown
        assert "ref_val=500" in f.args_repr

    def test_explanation_absolute_bounds(self, cd_irr):
        f = FilterIrr(low=200, high=800)
        f.run(cd_irr)
        assert f.explanation == (
            "Intervals where poa is below 200 or above 800 W/m^2 were removed."
        )

    def test_explanation_uses_effective_bounds_with_ref_val(self, cd_irr):
        f = FilterIrr(low=0.8, high=1.2, ref_val=500)
        f.run(cd_irr)
        # effective bounds = fraction * ref_val -> 400 / 600
        assert "below 400.0 or above 600.0" in f.explanation
        assert "poa" in f.explanation

    def test_explanation_uses_resolved_col_name(self, cd_irr):
        cd_irr.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
        cd_irr.data_filtered = cd_irr.data.copy()
        f = FilterIrr(low=500, high=2000, col_name="ghi")
        f.run(cd_irr)
        assert f.explanation.startswith("Intervals where ghi is below 500")

    def test_execute_rep_irr_without_rc_raises(self, cd_irr):
        cd_irr.rc = None
        with pytest.raises(ValueError, match="Call rep_cond"):
            FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd_irr)

    def test_execute_rep_irr_without_poa_col_raises(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"irr": [500.0]})
        with pytest.raises(ValueError, match="does not have a 'poa' column"):
            FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd_irr)


class TestRunLegacyMirroring:
    def test_run_populates_legacy_summary(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        assert cd_irr.summary_ix == [("irr", "filter_irr")]
        assert cd_irr.summary[0]["pts_after_filter"] == 3
        assert cd_irr.summary[0]["pts_removed"] == 2
        assert "low=200" in cd_irr.summary[0]["filter_arguments"]

    def test_run_populates_removed_and_kept(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        assert list(cd_irr.removed[0]["index"]) == [0, 4]
        assert list(cd_irr.kept[0]["index"]) == [1, 2, 3]
        assert cd_irr.removed[0]["name"] == "filter_irr"

    def test_run_enumerates_repeated_filters(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert [ix[1] for ix in cd_irr.summary_ix] == ["filter_irr", "filter_irr-1"]

    def test_run_summary_shows_resolved_ref_val(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        FilterIrr(low=0.8, high=1.2, ref_val="rep_irr").run(cd_irr)
        args = cd_irr.summary[0]["filter_arguments"]
        assert "rep_irr" not in args
        assert "np." not in args
        assert "500" in args


class TestFilterIrrWrapper:
    def test_wrapper_records_filterirr_step(self, cd_irr):
        # New: the wrapper now appends a FilterIrr to cd.filters (nothing in
        # the existing suite touches cd.filters).
        cd_irr.filter_irr(200, 800)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], FilterIrr)

    def test_wrapper_inplace_false_records_no_step(self, cd_irr):
        # New/behavior change: the legacy @update_summary decorator recorded a
        # step even when inplace=False; the wrapper must NOT record one.
        cd_irr.filter_irr(200, 800, inplace=False)
        assert cd_irr.filters == []


class TestDescribeFilters:
    def test_describe_empty_is_blank(self, cd_irr):
        assert cd_irr.describe_filters() == ""

    def test_describe_single_filter(self, cd_irr):
        cd_irr.filter_irr(200, 800)
        assert cd_irr.describe_filters() == (
            "Intervals where poa is below 200 or above 800 W/m^2 were removed."
        )

    def test_describe_multiple_steps(self, cd_irr):
        cd_irr.filter_irr(200, 800)
        cd_irr.filter_irr(400, 800)
        lines = cd_irr.describe_filters().splitlines()
        assert len(lines) == 2
        assert lines[0] == (
            "Intervals where poa is below 200 or above 800 W/m^2 were removed."
        )
        assert lines[1] == (
            "Intervals where poa is below 400 or above 800 W/m^2 were removed."
        )
