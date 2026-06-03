"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import numpy as np
import pandas as pd
import param
import pytest

from captest.capdata import CapData
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterCustom,
    FilterIrr,
    FilterSensors,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)


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


@pytest.fixture
def cd_time():
    """A CapData with a 90-day daily DatetimeIndex for time-window tests."""
    cd = CapData("time")
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    cd.data = pd.DataFrame({"power": range(90)}, index=idx)
    cd.data_filtered = cd.data.copy()
    return cd


def _drop_first(df):
    return df.iloc[1:]


def _gt_threshold(df, threshold=0, col="poa"):
    return df[df[col] > threshold]


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

    def test_execute_one_sided_high_only(self, cd_irr):
        # low=None means no lower bound; keep everything <= 500.
        f = FilterIrr(low=None, high=500)
        assert list(f._execute(cd_irr)) == [0, 1, 2]

    def test_execute_one_sided_with_ref_val_does_not_raise(self, cd_irr):
        # A None bound combined with a numeric ref_val must not raise.
        f = FilterIrr(low=None, high=1.2, ref_val=500)
        assert list(f._execute(cd_irr)) == [0, 1, 2]  # <= 600

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
        cd_irr.filter_irr(200, 800)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], FilterIrr)

    def test_wrapper_inplace_false_records_no_step(self, cd_irr):
        result = cd_irr.filter_irr(200, 800, inplace=False)
        assert cd_irr.filters == []
        assert result.shape[0] == 3
        assert cd_irr.data_filtered.shape[0] == 5


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


class TestFilterSensors:
    def test_execute_default_perc_diff_resolves(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = FilterSensors()
        kept = f._execute(capdata_irr)
        # tightly-clustered random data (876-900) is within the 5% default,
        # so no rows are removed
        assert list(kept) == list(capdata_irr.data_filtered.index)
        assert f.perc_diff_resolved == {"poa": 0.05}

    def test_execute_explicit_row_filter_drops_outliers(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        capdata_irr.data_filtered = capdata_irr.data.copy()
        f = FilterSensors(perc_diff={"poa": 25}, row_filter=abs_diff_from_average)
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_row_filter_defaults_to_check_all_perc_diff_comb(self):
        assert FilterSensors().row_filter is check_all_perc_diff_comb

    def test_args_repr_renders_row_filter_by_name(self):
        f = FilterSensors(perc_diff={"poa": 0.05})
        args = f.args_repr
        assert "row_filter=check_all_perc_diff_comb" in args
        assert "<function" not in args

    def test_explanation_names_group_and_row_filter(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = FilterSensors()
        f.run(capdata_irr)
        exp = f.explanation
        assert "poa" in exp
        assert "check_all_perc_diff_comb" in exp
        assert exp.endswith("were removed.")

    def test_execute_empty_perc_diff_raises(self, capdata_irr):
        f = FilterSensors(perc_diff={})
        with pytest.raises(ValueError, match="must not be empty"):
            f._execute(capdata_irr)

    def test_explanation_before_run_returns_none(self):
        # explanation is post-run; reading it before run() must not raise
        assert FilterSensors().explanation is None
        # FilterIrr has the same property by virtue of the base-class guard
        assert FilterIrr(low=0, high=1).explanation is None


class TestFilterSensorsWrapper:
    def test_wrapper_records_filtersensors_step(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        capdata_irr.filter_sensors()
        assert len(capdata_irr.filters) == 1
        assert isinstance(capdata_irr.filters[0], FilterSensors)

    def test_wrapper_inplace_false_records_no_step(self, capdata_irr):
        result = capdata_irr.filter_sensors(perc_diff={"poa": 0.05}, inplace=False)
        assert capdata_irr.filters == []
        assert result.shape[0] == capdata_irr.data_filtered.shape[0]
        assert capdata_irr.data_filtered.shape[0] == capdata_irr.data.shape[0]


class TestFilterTime:
    def test_execute_start_end(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_start_end_drop(self, cd_time):
        n_before = len(cd_time.data_filtered)
        f = FilterTime(start="2023-02-01", end="2023-02-15", drop=True)
        kept = f._execute(cd_time)
        assert len(kept) == n_before - 15

    def test_execute_start_days(self, cd_time):
        f = FilterTime(start="2023-02-01", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-11")

    def test_execute_end_days(self, cd_time):
        f = FilterTime(end="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-05")
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_test_date(self, cd_time):
        f = FilterTime(test_date="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-10")
        assert kept[-1] == pd.Timestamp("2023-02-20")

    def test_execute_start_only_defaults_to_last(self, cd_time):
        f = FilterTime(start="2023-02-01")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == cd_time.data_filtered.index[-1]

    def test_execute_end_only_defaults_to_first(self, cd_time):
        f = FilterTime(end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == cd_time.data_filtered.index[0]
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_no_args_raises(self, cd_time):
        with pytest.raises(ValueError, match="at least one of"):
            FilterTime()._execute(cd_time)

    def test_execute_test_date_no_days_warns_and_keeps_all(self, cd_time):
        n_before = len(cd_time.data_filtered)
        f = FilterTime(test_date="2023-02-15")
        with pytest.warns(UserWarning, match="Must specify days"):
            kept = f._execute(cd_time)
        assert len(kept) == n_before

    def test_execute_drop_with_start_days(self, cd_time):
        # start+days resolves end=02-11; df.loc[02-01:02-11] inclusive is
        # 11 rows; drop=True keeps the complement.
        n = len(cd_time.data_filtered)
        kept = FilterTime(start="2023-02-01", days=10, drop=True)._execute(cd_time)
        assert len(kept) == n - 11
        assert pd.Timestamp("2023-02-05") not in kept

    def test_execute_drop_with_test_date(self, cd_time):
        # test_date+days resolves window 02-10..02-20 (11 rows inclusive).
        n = len(cd_time.data_filtered)
        kept = FilterTime(test_date="2023-02-15", days=10, drop=True)._execute(cd_time)
        assert len(kept) == n - 11
        assert pd.Timestamp("2023-02-15") not in kept

    def test_execute_drop_start_only(self, cd_time):
        # start-only resolves end=last row; drop keeps rows BEFORE start.
        # cd_time spans 2023-01-01..2023-03-31 daily (90 rows); window
        # 02-01..03-31 is 59 rows; complement is 31 (Jan 1..Jan 31).
        kept = FilterTime(start="2023-02-01", drop=True)._execute(cd_time)
        assert len(kept) == 31
        assert kept[-1] < pd.Timestamp("2023-02-01")

    def test_execute_drop_end_only(self, cd_time):
        # end-only resolves start=first row; drop keeps rows AFTER end.
        # Window 01-01..02-15 = 46 rows; complement = 44.
        kept = FilterTime(end="2023-02-15", drop=True)._execute(cd_time)
        assert len(kept) == 44
        assert kept[0] > pd.Timestamp("2023-02-15")

    def test_explanation_start_end(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15")
        f.run(cd_time)
        assert "outside" in f.explanation
        assert "2023-02-01" in f.explanation
        assert "2023-02-15" in f.explanation

    def test_explanation_drop(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15", drop=True)
        f.run(cd_time)
        assert f.explanation.startswith("Data between")
        assert f.explanation.endswith("was removed.")

    def test_explanation_test_date(self, cd_time):
        f = FilterTime(test_date="2023-02-15", days=10)
        f.run(cd_time)
        assert "centered" in f.explanation
        assert "10-day" in f.explanation

    def test_explanation_drop_with_start_days(self, cd_time):
        f = FilterTime(start="2023-02-01", days=10, drop=True)
        f.run(cd_time)
        exp = f.explanation
        assert "within" in exp
        assert "10-day" in exp
        assert "2023-02-01" in exp
        assert "2023-02-11" in exp  # resolved end

    def test_explanation_drop_with_test_date(self, cd_time):
        f = FilterTime(test_date="2023-02-15", days=10, drop=True)
        f.run(cd_time)
        exp = f.explanation
        assert "within" in exp
        assert "2023-02-10" in exp  # resolved start
        assert "2023-02-20" in exp  # resolved end

    def test_explanation_drop_start_only(self, cd_time):
        f = FilterTime(start="2023-02-01", drop=True)
        f.run(cd_time)
        assert f.explanation == "Data from 2023-02-01 onward was removed."

    def test_explanation_drop_end_only(self, cd_time):
        f = FilterTime(end="2023-02-15", drop=True)
        f.run(cd_time)
        assert f.explanation == "Data up to 2023-02-15 was removed."

    def test_explanation_start_only(self, cd_time):
        f = FilterTime(start="2023-02-01")
        f.run(cd_time)
        assert f.explanation == "Data before 2023-02-01 was removed."

    def test_explanation_end_only(self, cd_time):
        f = FilterTime(end="2023-02-15")
        f.run(cd_time)
        assert f.explanation == "Data after 2023-02-15 was removed."

    def test_wrap_year_kwarg_is_rejected(self):
        with pytest.raises(TypeError, match="wrap_year"):
            FilterTime(wrap_year=True)

    def test_helpers_importable_from_filters(self):
        from captest.filters import spans_year, wrap_year_end

        assert callable(spans_year)
        assert callable(wrap_year_end)

    def test_capdata_still_exposes_wrap_year_end(self):
        from captest import capdata

        assert callable(capdata.wrap_year_end)


class TestFilterTimeWrapper:
    def test_wrapper_records_filtertime_step(self, cd_time):
        cd_time.filter_time(start="2023-02-01", end="2023-02-15")
        assert len(cd_time.filters) == 1
        assert isinstance(cd_time.filters[0], FilterTime)

    def test_wrapper_inplace_false_records_no_step(self, cd_time):
        n_before = cd_time.data_filtered.shape[0]
        result = cd_time.filter_time(
            start="2023-02-01", end="2023-02-15", inplace=False
        )
        assert cd_time.filters == []
        assert cd_time.data_filtered.shape[0] == n_before
        assert result.shape[0] == 15


class TestFilterCustom:
    def test_execute_applies_func(self, cd_irr):
        kept = FilterCustom(_drop_first)._execute(cd_irr)
        assert list(kept) == [1, 2, 3, 4]

    def test_execute_passes_args_and_kwargs(self, cd_irr):
        # poa values [100, 300, 500, 700, 900] -> > 400 -> indices [2, 3, 4]
        f = FilterCustom(_gt_threshold, threshold=400)
        assert list(f._execute(cd_irr)) == [2, 3, 4]

    def test_execute_with_pandas_method_dropna(self):
        cd = CapData("c")
        cd.data = pd.DataFrame(
            {"power": [1.0, np.nan, 3.0, np.nan, 5.0]},
            index=pd.RangeIndex(5),
        )
        cd.data_filtered = cd.data.copy()
        kept = FilterCustom(pd.DataFrame.dropna)._execute(cd)
        assert list(kept) == [0, 2, 4]

    def test_args_repr_renders_func_name(self):
        f = FilterCustom(_gt_threshold, threshold=400)
        args = f.args_repr
        assert "_gt_threshold" in args
        assert "threshold=400" in args
        assert "<function" not in args

    def test_args_repr_with_positional_args(self):
        f = FilterCustom(pd.DataFrame.between_time, "9:00", "17:00")
        args = f.args_repr
        assert "between_time" in args
        assert "'9:00'" in args
        assert "'17:00'" in args

    def test_args_repr_handles_callable_without_dunder_name(self):
        # functools.partial has no __name__; args_repr must fall back rather
        # than raising AttributeError mid-run() and leaving the step
        # half-applied.
        import functools

        f = FilterCustom(functools.partial(pd.DataFrame.dropna))
        args = f.args_repr  # must not raise
        assert isinstance(args, str)

    def test_explanation_reuses_call(self, cd_irr):
        f = FilterCustom(_drop_first)
        # Pre-run: BaseSummaryStep.explanation returns None until ix_after is
        # set by run(). Pinning this keeps the guard a tested contract.
        assert f.explanation is None
        f.run(cd_irr)
        exp = f.explanation
        assert "_drop_first" in exp
        assert exp.endswith("was applied.")

    def test_custom_name_passes_through(self):
        f = FilterCustom(_drop_first, custom_name="prune")
        assert f.custom_name == "prune"
