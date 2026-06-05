"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import unittest.mock

import numpy as np
import pandas as pd
import param
import pytest

from captest.capdata import CapData
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterClearsky,
    FilterCustom,
    FilterDays,
    FilterIrr,
    FilterMissing,
    FilterOutliers,
    FilterPf,
    FilterPower,
    FilterPvsyst,
    FilterRegression,
    FilterSensors,
    FilterShade,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)


@pytest.fixture
def make_capdata():
    """Factory fixture: a CapData with an n-row power+poa frame."""

    def _make(n=5):
        cd = CapData("test")
        cd.data = pd.DataFrame({"power": np.arange(n), "poa": np.arange(n) * 10.0})
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
    cd.regression_cols = {"poa": "poa"}
    return cd


@pytest.fixture
def cd_time():
    """A CapData with a 90-day daily DatetimeIndex for time-window tests."""
    cd = CapData("time")
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    cd.data = pd.DataFrame({"power": range(90)}, index=idx)
    return cd


def _drop_first(df):
    return df.iloc[1:]


def _gt_threshold(df, threshold=0, col="poa"):
    return df[df[col] > threshold]


@pytest.fixture
def cd_reg():
    """A CapData with a clean linear power~poa relationship + one outlier."""
    n = 40
    poa = np.linspace(100, 1000, n)
    rng = np.random.default_rng(0)
    power = poa * 0.5 + rng.normal(0, 3, n)
    power[10] += 400  # gross residual outlier at index 10
    cd = CapData("reg")
    cd.data = pd.DataFrame({"poa": poa, "power": power}, index=pd.RangeIndex(n))
    cd.column_groups = {"irr-poa-": ["poa"], "real_pwr--": ["power"]}
    cd.regression_cols = {"power": "power", "poa": "poa"}
    cd.regression_formula = "power ~ poa"
    return cd


@pytest.fixture
def cd_pp():
    """A CapData with poa+power columns and three injected outliers."""
    np.random.seed(0)
    n = 50
    poa = np.linspace(100.0, 1000.0, n)
    power = poa * 0.2 + np.random.normal(0, 5, n)
    power[5] = 500.0
    power[20] = 0.0
    power[40] = -100.0
    cd = CapData("pp")
    cd.data = pd.DataFrame({"poa": poa, "power": power}, index=pd.RangeIndex(n))
    cd.regression_cols = {"poa": "poa", "power": "power"}
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

    def test_data_filtered_no_filter_is_a_defensive_copy(self, make_capdata):
        # With no filters the property still returns a copy, so mutating the
        # returned frame must not corrupt self.data.
        cd = make_capdata(n=5)
        df = cd.data_filtered
        assert df is not cd.data
        df.iloc[0, df.columns.get_loc("power")] = 99999
        assert cd.data.iloc[0, cd.data.columns.get_loc("power")] != 99999

    def test_data_filtered_with_filter_is_a_defensive_copy(self, make_capdata):
        cd = make_capdata(n=5)
        _DropFirstRow().run(cd)
        df = cd.data_filtered
        df.iloc[0, df.columns.get_loc("power")] = 99999
        assert (cd.data["power"] == 99999).sum() == 0


class TestFilterIrr:
    def test_execute_absolute_bounds(self, cd_irr):
        f = FilterIrr(low=200, high=800)
        assert list(f._execute(cd_irr)) == [1, 2, 3]

    def test_execute_uses_explicit_col_name(self, cd_irr):
        cd_irr.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
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


class TestRunSummary:
    def test_run_populates_summary(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        gs = cd_irr.get_summary()
        assert list(gs.index) == [("irr", "FilterIrr")]
        assert gs["pts_after_filter"].iloc[0] == 3
        assert gs["pts_removed"].iloc[0] == 2
        assert "low=200" in gs["filter_arguments"].iloc[0]

    def test_run_enumerates_repeated_filters(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert [ix[1] for ix in cd_irr.get_summary().index] == [
            "FilterIrr",
            "FilterIrr-1",
        ]

    def test_run_summary_shows_resolved_ref_val(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        FilterIrr(low=0.8, high=1.2, ref_val="rep_irr").run(cd_irr)
        args = cd_irr.get_summary()["filter_arguments"].iloc[0]
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


class TestFilterCustomWrapper:
    def test_wrapper_records_filtercustom_step(self, cd_irr):
        cd_irr.filter_custom(_drop_first)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], FilterCustom)

    def test_wrapper_passes_args_kwargs_to_func(self, cd_irr):
        cd_irr.filter_custom(_gt_threshold, threshold=400)
        assert list(cd_irr.data_filtered.index) == [2, 3, 4]

    def test_wrapper_custom_name_kwarg_is_kwonly(self, cd_irr):
        cd_irr.filter_custom(_drop_first, custom_name="prune")
        assert cd_irr.filters[0].custom_name == "prune"


class TestFilterOutliers:
    def test_execute_removes_outliers(self, cd_pp):
        # Default contamination=0.04 on n=50 removes 2 points; verified
        # empirically that indices 5 and 40 (the two most extreme injected
        # outliers) go and index 20 stays at the default contamination.
        n_before = len(cd_pp.data_filtered)
        kept = FilterOutliers()._execute(cd_pp)
        assert len(kept) < n_before
        assert 5 not in kept
        assert 40 not in kept

    def test_execute_higher_contamination_removes_more(self, cd_pp):
        f = FilterOutliers(envelope_kwargs={"contamination": 0.10})
        kept = f._execute(cd_pp)
        for ix in (5, 20, 40):
            assert ix not in kept

    def test_execute_resolves_default_kwargs(self, cd_pp):
        f = FilterOutliers()
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved == {
            "support_fraction": 0.9,
            "contamination": 0.04,
        }

    def test_execute_user_kwargs_override(self, cd_pp):
        f = FilterOutliers(envelope_kwargs={"contamination": 0.10})
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved["contamination"] == 0.10
        assert f.envelope_kwargs_resolved["support_fraction"] == 0.9

    def test_execute_too_many_columns_warns_and_keeps_all(self, cd_pp):
        cd_pp.data["poa2"] = cd_pp.data["poa"]
        cd_pp.column_groups = {"poa": ["poa", "poa2"], "power": ["power"]}
        n_before = len(cd_pp.data_filtered)
        f = FilterOutliers()
        with pytest.warns(UserWarning, match="aggregate_sensors"):
            kept = f._execute(cd_pp)
        assert len(kept) == n_before

    def test_execute_nan_calls_filter_missing(self, cd_pp):
        cd_pp.data.iloc[0, cd_pp.data.columns.get_loc("poa")] = np.nan
        with pytest.warns(UserWarning, match="missing values"):
            kept = FilterOutliers()._execute(cd_pp)
        assert 0 not in kept
        # The nested filter_missing is recorded as its own step in the chain.
        gs = cd_pp.get_summary()
        assert len(gs) == 1
        assert gs.index[0][1] == "FilterMissing"

    def test_pts_removed_excludes_nan_drop(self, cd_pp):
        cd_pp.data.iloc[1, cd_pp.data.columns.get_loc("poa")] = np.nan
        pre_run_pts = len(cd_pp.data_filtered)  # includes the NaN row
        f = FilterOutliers()
        with pytest.warns(UserWarning):
            f.run(cd_pp)
        gs = cd_pp.get_summary()
        assert gs.index[-2][1] == "FilterMissing"
        assert gs.index[-1][1] == "FilterOutliers"
        assert f.pts_before == gs["pts_after_filter"].iloc[-2]
        assert (
            gs["pts_removed"].iloc[-1]
            == gs["pts_after_filter"].iloc[-2] - gs["pts_after_filter"].iloc[-1]
        )
        assert gs["pts_removed"].iloc[-1] < (
            pre_run_pts - gs["pts_after_filter"].iloc[-1]
        )

    def test_args_repr_renders_envelope_call(self):
        # Pre-_execute: param dict is None, so the resolved attr isn't set yet
        # and args_repr falls back to the base "Default arguments" rendering.
        f = FilterOutliers()
        assert "EllipticEnvelope" not in f.args_repr
        cd = CapData("x")
        cd.data = pd.DataFrame(
            {"poa": np.linspace(100, 1000, 30), "power": np.linspace(20, 200, 30)},
            index=pd.RangeIndex(30),
        )
        cd.regression_cols = {"poa": "poa", "power": "power"}
        f._execute(cd)
        post = f.args_repr
        assert "EllipticEnvelope(" in post
        assert "contamination=0.04" in post
        assert "support_fraction=0.9" in post

    def test_explanation_describes_envelope(self, cd_pp):
        f = FilterOutliers()
        f.run(cd_pp)
        exp = f.explanation
        assert "EllipticEnvelope" in exp
        assert exp.endswith("were removed.")


class TestFilterOutliersWrapper:
    def test_wrapper_records_filteroutliers_step(self, cd_pp):
        cd_pp.filter_outliers()
        assert len(cd_pp.filters) == 1
        assert isinstance(cd_pp.filters[0], FilterOutliers)

    def test_wrapper_passes_kwargs(self, cd_pp):
        cd_pp.filter_outliers(contamination=0.10)
        assert cd_pp.filters[0].envelope_kwargs_resolved["contamination"] == 0.10

    def test_wrapper_inplace_false_records_no_step(self, cd_pp):
        result = cd_pp.filter_outliers(inplace=False)
        assert cd_pp.filters == []
        assert result is not None
        assert result.shape[0] < cd_pp.data.shape[0]


class TestFilterClearsky:
    def test_execute_keeps_clear_periods(self, nrel_clear_sky):
        n_before = nrel_clear_sky.data_filtered.shape[0]
        kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) < n_before
        assert nrel_clear_sky.data_filtered.shape[0] == n_before

    def test_execute_keep_clear_false_inverts_mask(self, nrel_clear_sky):
        clear_kept = FilterClearsky()._execute(nrel_clear_sky)
        cloudy_kept = FilterClearsky(keep_clear=False)._execute(nrel_clear_sky)
        full = nrel_clear_sky.data_filtered.index
        assert clear_kept.union(cloudy_kept).equals(full)
        assert clear_kept.intersection(cloudy_kept).empty

    def test_execute_resolves_default_detect_kwargs(self, nrel_clear_sky):
        f = FilterClearsky()
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved == {"infer_limits": True}

    def test_execute_user_kwargs_override(self, nrel_clear_sky):
        f = FilterClearsky(detect_kwargs={"infer_limits": False, "window_length": 30})
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved["infer_limits"] is False
        assert f.detect_kwargs_resolved["window_length"] == 30

    def test_execute_no_ghi_mod_csky_warns_and_keeps_all(self, nrel_clear_sky):
        nrel_clear_sky.drop_cols("ghi_mod_csky")
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Modeled clear sky"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_no_measured_ghi_group_warns_and_keeps_all(self, nrel_clear_sky):
        # ghi_mod_csky present (so the first guard passes) but no measured
        # GHI column group at all — must warn and no-op rather than IndexError.
        del nrel_clear_sky.column_groups["irr-ghi-"]
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="No measured GHI"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_too_many_ghi_categories_warns_and_keeps_all(self, nrel_clear_sky):
        # Add a second measured GHI category alongside the existing irr-ghi-
        # (irr-ghi-clear_sky is excluded by the filter, so two real groups
        # remain and trigger the "Too many ghi categories" guard).
        nrel_clear_sky.column_groups["irr-ghi-pyran"] = ["some_pyran_col"]
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Too many ghi"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_multi_column_ghi_averages_with_warning(self, nrel_clear_sky):
        # Append a second column to the single irr-ghi- group; the auto-detect
        # path then averages the columns and warns. Use .squeeze() so the
        # assignment yields a Series regardless of pandas-version quirks.
        nrel_clear_sky.data["ws 2 ghi W/m^2"] = (
            nrel_clear_sky.loc["irr-ghi-"].squeeze() * 1.05
        )
        nrel_clear_sky.column_groups["irr-ghi-"].append("ws 2 ghi W/m^2")
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Averaging"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        # Filtering still runs after averaging — some rows should be removed.
        assert len(kept) < n_before

    def test_execute_no_clear_periods_warns_and_keeps_all(self, nrel_clear_sky):
        # Stub detect_clearsky to return an all-False mask so the
        # `not any(clear_per)` guard fires.
        n_before = nrel_clear_sky.data_filtered.shape[0]
        all_false = pd.Series(False, index=nrel_clear_sky.data_filtered.index)
        with unittest.mock.patch(
            "captest.filters.detect_clearsky", return_value=all_false
        ):
            with pytest.warns(UserWarning, match="No clear periods"):
                kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_specify_ghi_col(self, nrel_clear_sky):
        # .squeeze() makes the right-hand side an explicit Series rather than
        # a single-column DataFrame — defensive against future pandas changes
        # where the DataFrame-to-Series unwrap might align on column names
        # and silently yield NaN.
        nrel_clear_sky.data["ws 2 ghi W/m^2"] = (
            nrel_clear_sky.loc["irr-ghi-"].squeeze() * 1.05
        )
        nrel_clear_sky.column_groups["irr-ghi-"].append("ws 2 ghi W/m^2")
        f = FilterClearsky(ghi_col="ws 2 ghi W/m^2")
        kept = f._execute(nrel_clear_sky)
        assert len(kept) < nrel_clear_sky.data_filtered.shape[0]

    def test_args_repr_renders_detect_call(self, nrel_clear_sky):
        f = FilterClearsky()
        assert "detect_clearsky" not in f.args_repr
        f._execute(nrel_clear_sky)
        assert "detect_clearsky(" in f.args_repr
        assert "infer_limits=True" in f.args_repr

    def test_explanation_default_says_cloudy(self, nrel_clear_sky):
        f = FilterClearsky()
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Cloudy intervals")
        assert "detect_clearsky" in f.explanation
        assert f.explanation.endswith("were removed.")

    def test_explanation_keep_clear_false_says_clear(self, nrel_clear_sky):
        f = FilterClearsky(keep_clear=False)
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Clear intervals")


class TestFilterClearskyWrapper:
    def test_wrapper_records_filterclearsky_step(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky()
        assert len(nrel_clear_sky.filters) == 1
        assert isinstance(nrel_clear_sky.filters[0], FilterClearsky)

    def test_wrapper_passes_kwargs(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky(infer_limits=False, window_length=30)
        resolved = nrel_clear_sky.filters[0].detect_kwargs_resolved
        assert resolved["infer_limits"] is False
        assert resolved["window_length"] == 30

    def test_wrapper_inplace_false_records_no_step(self, nrel_clear_sky):
        original = nrel_clear_sky.data_filtered.copy()
        result = nrel_clear_sky.filter_clearsky(inplace=False)
        assert nrel_clear_sky.filters == []
        pd.testing.assert_frame_equal(nrel_clear_sky.data_filtered, original)
        assert result.shape[0] < original.shape[0]


class TestFilterPvsyst:
    def _cd(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame(
            {"IL Pmin": [0.0, 1.0, 0.0, 2.0], "power": [10.0, 20.0, 30.0, 40.0]},
            index=pd.RangeIndex(4),
        )
        return cd

    def test_execute_drops_positive_rows(self):
        kept = FilterPvsyst()._execute(self._cd())
        assert list(kept) == [0, 2]

    def test_execute_underscored_column_names(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame({"IL_Pmax": [0.0, 5.0, 0.0]}, index=pd.RangeIndex(3))
        assert list(FilterPvsyst()._execute(cd)) == [0, 2]

    def test_execute_missing_column_warns(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        with pytest.warns(UserWarning, match="not a column"):
            kept = FilterPvsyst()._execute(cd)
        assert list(kept) == [0, 1]

    def test_explanation(self):
        f = FilterPvsyst()
        f.run(self._cd())
        assert "maximum power point" in f.explanation
        assert f.explanation.endswith("were removed.")


class TestFilterShade:
    def _cd(self):
        cd = CapData("sh")
        cd.data = pd.DataFrame(
            {"FShdBm": [1.0, 0.5, 1.0, 0.8], "ShdLoss": [0.0, 50.0, 0.0, 130.0]},
            index=pd.RangeIndex(4),
        )
        return cd

    def test_execute_default_fshdbm(self):
        assert list(FilterShade()._execute(self._cd())) == [0, 2]

    def test_execute_custom_fshdbm(self):
        assert list(FilterShade(fshdbm=0.6)._execute(self._cd())) == [0, 2, 3]

    def test_execute_query_str(self):
        assert list(FilterShade(query_str="ShdLoss<=125")._execute(self._cd())) == [
            0,
            1,
            2,
        ]

    def test_explanation_default_shows_resolved_threshold(self):
        f = FilterShade()
        f.run(self._cd())
        assert "shad" in f.explanation.lower()
        # resolved value, not the literal @fshdbm placeholder
        assert "FShdBm>=1.0" in f.explanation
        assert "@fshdbm" not in f.explanation
        assert f.explanation.endswith("were removed.")

    def test_explanation_custom_query(self):
        f = FilterShade(query_str="ShdLoss<=125")
        f.run(self._cd())
        assert "ShdLoss<=125" in f.explanation


class TestFilterDaysClass:
    def _cd(self):
        # Hourly index so a day-string selects many rows (DataFrame), matching
        # how filter_days is used in production. A daily index would make
        # df.loc["day"] return a single-row Series.
        cd = CapData("d")
        idx = pd.date_range("1990-10-01", periods=72, freq="h")  # 3 days
        cd.data = pd.DataFrame({"power": range(72)}, index=idx)
        return cd

    def test_execute_keep_days(self):
        kept = FilterDays(days=["10/1/1990", "10/2/1990"])._execute(self._cd())
        assert len(kept) == 48
        assert set(ts.day for ts in kept) == {1, 2}

    def test_execute_drop_days(self):
        kept = FilterDays(days=["10/1/1990"], drop=True)._execute(self._cd())
        assert len(kept) == 48  # 72 - 24
        assert all(ts.day != 1 for ts in kept)

    def test_explanation_keep(self):
        f = FilterDays(days=["10/2/1990"])
        f.run(self._cd())
        assert "except" in f.explanation.lower()

    def test_explanation_drop(self):
        f = FilterDays(days=["10/2/1990"], drop=True)
        f.run(self._cd())
        assert "removed" in f.explanation.lower()


class TestFilterPf:
    def _cd(self):
        cd = CapData("pf")
        cd.data = pd.DataFrame(
            {"inv1 pf": [1.0, 0.5, 0.99], "inv2 pf": [0.999, 0.9, 1.0]},
            index=pd.RangeIndex(3),
        )
        cd.column_groups = {"pf--": ["inv1 pf", "inv2 pf"]}
        return cd

    def test_execute_keeps_high_pf(self):
        assert list(FilterPf(pf=0.95)._execute(self._cd())) == [0, 2]

    def test_execute_no_pf_group_warns(self):
        cd = CapData("pf")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        cd.column_groups = {"real_pwr--": ["power"]}
        with pytest.warns(UserWarning, match="power factor"):
            kept = FilterPf(pf=0.99)._execute(cd)
        assert list(kept) == [0, 1]

    def test_explanation(self):
        f = FilterPf(pf=0.95)
        f.run(self._cd())
        assert "0.95" in f.explanation
        assert "power factor" in f.explanation.lower()


class TestFilterPower:
    def _cd(self):
        cd = CapData("pw")
        cd.data = pd.DataFrame(
            {"meter_power": [100.0, 600.0, 300.0, 900.0]},
            index=pd.RangeIndex(4),
        )
        cd.regression_cols = {"power": "meter_power"}
        return cd

    def test_execute_threshold(self):
        assert list(FilterPower(power=500)._execute(self._cd())) == [0, 2]

    def test_execute_percent(self):
        assert list(FilterPower(power=1000, percent=0.5)._execute(self._cd())) == [
            0,
            2,
        ]

    def test_execute_named_column(self):
        assert list(
            FilterPower(power=500, columns="meter_power")._execute(self._cd())
        ) == [0, 2]

    def test_execute_bad_columns_warns(self):
        cd = self._cd()
        f = FilterPower(power=500, columns=1)
        with pytest.warns(UserWarning, match="None or a string"):
            kept = f._execute(cd)
        assert len(kept) == cd.data_filtered.shape[0]

    def test_explanation(self):
        f = FilterPower(power=500)
        f.run(self._cd())
        assert "power" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")


class TestFilterMissingClass:
    def _cd(self):
        cd = CapData("m")
        cd.data = pd.DataFrame(
            {"poa": [1.0, np.nan, 3.0], "power": [10.0, 20.0, np.nan]},
            index=pd.RangeIndex(3),
        )
        cd.regression_cols = {"poa": "poa", "power": "power"}
        return cd

    def test_execute_default_regcols(self):
        assert list(FilterMissing()._execute(self._cd())) == [0]

    def test_execute_subset_columns(self):
        assert list(FilterMissing(columns=["poa"])._execute(self._cd())) == [0, 2]

    def test_explanation(self):
        f = FilterMissing()
        f.run(self._cd())
        assert "missing" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")


class TestFilterRegression:
    def test_n_std_default_is_2(self):
        assert FilterRegression().n_std == 2

    def test_execute_exposes_fitted_model(self, cd_reg):
        f = FilterRegression()
        f._execute(cd_reg)
        assert hasattr(f, "regression_model")
        assert hasattr(f.regression_model, "resid")

    def test_execute_removes_the_outlier(self, cd_reg):
        kept = FilterRegression(n_std=2)._execute(cd_reg)
        assert 10 not in kept  # the injected outlier is dropped
        assert len(kept) == cd_reg.data_filtered.shape[0] - 1

    def test_execute_kept_rows_within_threshold(self, cd_reg):
        f = FilterRegression(n_std=2)
        kept = f._execute(cd_reg)
        reg = f.regression_model
        threshold = 2 * reg.scale**0.5
        assert (reg.resid.loc[kept].abs() < threshold).all()

    def test_larger_n_std_keeps_more(self, cd_reg):
        kept_2 = FilterRegression(n_std=2)._execute(cd_reg)
        kept_4 = FilterRegression(n_std=4)._execute(cd_reg)
        assert len(kept_4) >= len(kept_2)

    def test_execute_nan_calls_filter_missing(self, cd_reg):
        # NaN in a regression column: must run filter_missing (recording its
        # own step) rather than raise an unalignable-boolean error.
        cd_reg.data.iloc[5, cd_reg.data.columns.get_loc("power")] = np.nan
        with pytest.warns(UserWarning, match="missing values"):
            kept = FilterRegression(n_std=2)._execute(cd_reg)
        assert 5 not in kept  # NaN row removed
        assert cd_reg.get_summary().index[-1][1] == "FilterMissing"

    def test_explanation(self, cd_reg):
        f = FilterRegression(n_std=2)
        f.run(cd_reg)
        assert "residual" in f.explanation.lower()
        assert "2" in f.explanation
        assert f.explanation.endswith("were removed.")


class TestFitRegressionWrapper:
    def test_filter_true_records_filterregression_step(self, cd_reg):
        cd_reg.fit_regression(filter=True, summary=False)
        assert len(cd_reg.filters) == 1
        assert isinstance(cd_reg.filters[0], FilterRegression)

    def test_filter_true_not_inplace_records_no_step(self, cd_reg):
        n_before = cd_reg.data_filtered.shape[0]
        out = cd_reg.fit_regression(filter=True, inplace=False, summary=False)
        assert cd_reg.filters == []
        assert cd_reg.data_filtered.shape[0] == n_before
        assert out.shape[0] < n_before  # the outlier is removed

    def test_filter_false_stores_regression_results(self, cd_reg):
        cd_reg.fit_regression(filter=False, summary=False)
        assert cd_reg.regression_results is not None
        assert cd_reg.filters == []  # plain fit records no filter step


class TestChainDerivedHelpers:
    def test_step_labels_enumerate_repeats(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert cd_irr._step_labels() == ["FilterIrr", "FilterIrr-1"]

    def test_step_labels_use_custom_name(self, cd_irr):
        FilterIrr(low=200, high=800, custom_name="Irradiance bounds").run(cd_irr)
        assert cd_irr._step_labels() == ["Irradiance bounds"]

    def test_pts_before_first_step_is_full_data(self, cd_irr):
        full = cd_irr.data.shape[0]
        FilterIrr(low=200, high=800).run(cd_irr)
        assert cd_irr._pts_before(0) == full

    def test_ix_before_second_step_is_prior_ix_after(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert list(cd_irr._ix_before(1)) == list(cd_irr.filters[0].ix_after)


class TestRemovedByStep:
    def test_no_filters_returns_empty(self, cd_irr):
        assert cd_irr._removed_by_step() == []

    def test_single_filter(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        result = cd_irr._removed_by_step()
        assert len(result) == 1
        i, label, removed_ix = result[0]
        assert i == 0
        assert label == "FilterIrr"
        assert list(removed_ix) == [0, 4]

    def test_two_filters_indices_and_labels(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)  # keeps [1,2,3]
        FilterIrr(low=400, high=800).run(cd_irr)  # keeps [2,3], removes [1]
        result = cd_irr._removed_by_step()
        assert [(i, label) for i, label, _ in result] == [
            (0, "FilterIrr"),
            (1, "FilterIrr-1"),
        ]
        assert list(result[1][2]) == [1]

    def test_skips_zero_removal_step(self, cd_irr):
        FilterIrr(low=0, high=10000).run(cd_irr)  # keeps all 5 -> removes nothing
        assert cd_irr._removed_by_step() == []

    def test_zero_removal_step_between_real_filters(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)  # removes [0,4]
        FilterIrr(low=0, high=10000).run(cd_irr)  # removes nothing -> skipped
        FilterIrr(low=400, high=800).run(cd_irr)  # removes [1]
        result = cd_irr._removed_by_step()
        # The no-op middle filter is skipped; the third filter keeps its real
        # index i=2 (skipping affects only which entries are returned, not the
        # chain math).
        assert [(i, label) for i, label, _ in result] == [
            (0, "FilterIrr"),
            (2, "FilterIrr-2"),
        ]
        assert list(result[1][2]) == [1]
