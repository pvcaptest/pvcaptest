"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import math
import unittest.mock
import warnings

import numpy as np
import pandas as pd
import param
import pytest

from captest import util
from captest.capdata import CapData
from captest.filters import (
    AbsDiffPrev,
    BaseSummaryStep,
    BaseFilter,
    Backtracking,
    BooleanFlag,
    Clearsky,
    Custom,
    Days,
    Irradiance,
    Missing,
    Outliers,
    PowerFactor,
    Power,
    Pvsyst,
    Regression,
    RollingStd,
    Sensors,
    Shade,
    Time,
    RepCond,
    _backtracking_geometry_error,
    abs_diff_from_average,
    backtracking_active,
)
from captest.filters import FILTER_REGISTRY, step_from_config


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
    """A CapData with a 5-row poa frame and regression_cols set, for Irradiance."""
    cd = CapData("irr")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 300.0, 500.0, 700.0, 900.0]},
        index=pd.RangeIndex(5),
    )
    cd.regression_cols = {"poa": "poa"}
    return cd


@pytest.fixture
def cd_roll():
    """A CapData with a poa column that has a stable stretch and a spike."""
    cd = CapData("roll")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 100.0, 100.0, 500.0, 100.0, 100.0]},
        index=pd.RangeIndex(6),
    )
    cd.regression_cols = {"poa": "poa"}
    return cd


@pytest.fixture
def cd_step():
    """A CapData with a poa column that has one large step change."""
    cd = CapData("step")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 102.0, 300.0, 305.0, 310.0]},
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


@pytest.fixture
def cd_flag():
    """A CapData with a boolean flag column (e.g. tracker backtracking)."""
    cd = CapData("flag")
    cd.data = pd.DataFrame(
        {
            "power": [1.0, 2.0, 3.0, 4.0, 5.0],
            "backtrack_on": [False, True, False, True, False],
        },
        index=pd.RangeIndex(5),
    )
    return cd


@pytest.fixture
def cd_thresh():
    """A CapData with an availability column and a temperature column."""
    cd = CapData("thresh")
    cd.data = pd.DataFrame(
        {
            "avail": [95.0, 97.4, 98.0, 99.0, 100.0],
            "temp": [30.0, 40.0, 45.0, 50.0, 35.0],
        },
        index=pd.RangeIndex(5),
    )
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
        # run() stores only ix_after/pts_after; before/removed are chain-derived.
        assert step.pts_after == 4
        assert list(step.ix_after) == [1, 2, 3, 4]
        assert cd._pts_before(0) == 5
        assert list(cd._ix_before(0)) == [0, 1, 2, 3, 4]
        assert cd._pts_before(0) - step.pts_after == 1

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
    def test_wrapper_custom_name_sets_step_label(self, cd_irr):
        cd_irr.filter_irr(200, 800, custom_name="my irr step")
        assert cd_irr.filters[-1].custom_name == "my irr step"

    def test_execute_absolute_bounds(self, cd_irr):
        f = Irradiance(low=200, high=800)
        assert list(f._execute(cd_irr)) == [1, 2, 3]

    def test_execute_uses_explicit_col_name(self, cd_irr):
        cd_irr.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
        f = Irradiance(low=500, high=2000, col_name="ghi")
        assert list(f._execute(cd_irr)) == [4]

    def test_execute_fraction_with_ref_val(self, cd_irr):
        # low/high are fractions of ref_val 500 -> [400, 600]
        f = Irradiance(low=0.8, high=1.2, ref_val=500)
        assert list(f._execute(cd_irr)) == [2]

    def test_execute_one_sided_high_only(self, cd_irr):
        # low=None means no lower bound; keep everything <= 500.
        f = Irradiance(low=None, high=500)
        assert list(f._execute(cd_irr)) == [0, 1, 2]

    def test_execute_one_sided_with_ref_val_does_not_raise(self, cd_irr):
        # A None bound combined with a numeric ref_val must not raise.
        f = Irradiance(low=None, high=1.2, ref_val=500)
        assert list(f._execute(cd_irr)) == [0, 1, 2]  # <= 600

    def test_execute_resolves_rep_irr_into_runtime_attr(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = Irradiance(low=0.8, high=1.2, ref_val="rep_irr")
        f._execute(cd_irr)
        # intent preserved on the param; resolved value on the runtime attr
        assert f.ref_val == "rep_irr"
        assert f.ref_val_resolved == 500.0
        assert isinstance(f.ref_val_resolved, float)

    def test_execute_resolves_self_val(self, cd_irr):
        # 'self_val' is translated to 'rep_irr' and resolved the same way;
        # the param keeps the original 'self_val' intent.
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = Irradiance(low=0.8, high=1.2, ref_val="self_val")
        f._execute(cd_irr)
        assert f.ref_val == "self_val"
        assert f.ref_val_resolved == 500.0

    def test_args_repr_shows_resolved_ref_val_not_sentinel(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        f = Irradiance(low=0.8, high=1.2, ref_val="rep_irr")
        f._execute(cd_irr)
        args = f.args_repr
        assert "rep_irr" not in args
        assert "np." not in args
        assert "ref_val=500.0" in args

    def test_args_repr_numeric_ref_val_unchanged(self):
        f = Irradiance(low=0.8, high=1.2, ref_val=500)
        # no resolution happened; ref_val_resolved is unset, param value shown
        assert "ref_val=500" in f.args_repr

    def test_explanation_absolute_bounds(self, cd_irr):
        f = Irradiance(low=200, high=800)
        f.run(cd_irr)
        assert f.explanation == (
            "Intervals where poa is below 200 or above 800 W/m^2 were removed."
        )

    def test_explanation_uses_effective_bounds_with_ref_val(self, cd_irr):
        f = Irradiance(low=0.8, high=1.2, ref_val=500)
        f.run(cd_irr)
        # effective bounds = fraction * ref_val -> 400 / 600
        assert "below 400.0 or above 600.0" in f.explanation
        assert "poa" in f.explanation

    def test_explanation_uses_resolved_col_name(self, cd_irr):
        cd_irr.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
        f = Irradiance(low=500, high=2000, col_name="ghi")
        f.run(cd_irr)
        assert f.explanation.startswith("Intervals where ghi is below 500")

    def test_execute_rep_irr_without_rc_raises(self, cd_irr):
        cd_irr.rc = None
        with pytest.raises(ValueError, match="Call rep_cond"):
            Irradiance(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd_irr)

    def test_execute_rep_irr_without_poa_col_raises(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"irr": [500.0]})
        with pytest.raises(ValueError, match="requires a 'poa' column"):
            Irradiance(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd_irr)


class TestRunSummary:
    def test_run_populates_summary(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)
        gs = cd_irr.get_summary()
        assert list(gs.index) == [("irr", "Irradiance")]
        assert gs["pts_after_filter"].iloc[0] == 3
        assert gs["pts_removed"].iloc[0] == 2
        assert "low=200" in gs["filter_arguments"].iloc[0]

    def test_run_enumerates_repeated_filters(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)
        Irradiance(low=400, high=800).run(cd_irr)
        assert [ix[1] for ix in cd_irr.get_summary().index] == [
            "Irradiance",
            "Irradiance-1",
        ]

    def test_run_summary_shows_resolved_ref_val(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        Irradiance(low=0.8, high=1.2, ref_val="rep_irr").run(cd_irr)
        args = cd_irr.get_summary()["filter_arguments"].iloc[0]
        assert "rep_irr" not in args
        assert "np." not in args
        assert "500" in args


class TestFilterIrrWrapper:
    def test_wrapper_records_filterirr_step(self, cd_irr):
        cd_irr.filter_irr(200, 800)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], Irradiance)

    def test_wrapper_rejects_inplace_kwarg(self, cd_irr):
        with pytest.raises(TypeError):
            cd_irr.filter_irr(200, 800, inplace=False)


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
    def test_execute_default_thresholds_resolves(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors()
        kept = f._execute(capdata_irr)
        # tightly-clustered random data (876-900) is within the 5% default,
        # so no rows are removed
        assert list(kept) == list(capdata_irr.data_filtered.index)
        assert f.thresholds_resolved == {"poa": 0.05}

    def test_method_defaults_to_percent_diff(self):
        assert Sensors().method == "percent_diff"

    def test_execute_abs_diff_method_drops_outliers(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        f = Sensors(method="abs_diff", thresholds={"poa": 25})
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_execute_custom_callable_method(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        f = Sensors(method=abs_diff_from_average, thresholds={"poa": 25})
        assert f._resolve_comparison() is abs_diff_from_average
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_execute_abs_diff_without_thresholds_raises(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors(method="abs_diff")
        with pytest.raises(ValueError, match="thresholds is required"):
            f._execute(capdata_irr)

    def test_execute_empty_thresholds_raises(self, capdata_irr):
        f = Sensors(thresholds={})
        with pytest.raises(ValueError, match="must not be empty"):
            f._execute(capdata_irr)

    def test_args_repr_renders_method_name(self):
        f = Sensors(thresholds={"poa": 0.05})
        args = f.args_repr
        assert "method=percent_diff" in args
        assert "<function" not in args

    def test_explanation_names_group_and_method(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors()
        f.run(capdata_irr)
        exp = f.explanation
        assert "poa" in exp
        assert "percent_diff" in exp
        assert exp.endswith("were removed.")

    def test_explanation_before_run_returns_none(self):
        # explanation is post-run; reading it before run() must not raise
        assert Sensors().explanation is None
        # Irradiance has the same property by virtue of the base-class guard
        assert Irradiance(low=0, high=1).explanation is None


class TestFilterSensorsWrapper:
    def test_wrapper_records_filtersensors_step(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        capdata_irr.filter_sensors()
        assert len(capdata_irr.filters) == 1
        assert isinstance(capdata_irr.filters[0], Sensors)


class TestFilterTime:
    def test_execute_start_end(self, cd_time):
        f = Time(start="2023-02-01", end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_start_end_drop(self, cd_time):
        n_before = len(cd_time.data_filtered)
        f = Time(start="2023-02-01", end="2023-02-15", drop=True)
        kept = f._execute(cd_time)
        assert len(kept) == n_before - 15

    def test_execute_start_days(self, cd_time):
        f = Time(start="2023-02-01", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-11")

    def test_execute_end_days(self, cd_time):
        f = Time(end="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-05")
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_test_date(self, cd_time):
        f = Time(test_date="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-10")
        assert kept[-1] == pd.Timestamp("2023-02-20")

    def test_execute_start_only_defaults_to_last(self, cd_time):
        f = Time(start="2023-02-01")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == cd_time.data_filtered.index[-1]

    def test_execute_end_only_defaults_to_first(self, cd_time):
        f = Time(end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == cd_time.data_filtered.index[0]
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_no_args_raises(self, cd_time):
        with pytest.raises(ValueError, match="at least one of"):
            Time()._execute(cd_time)

    def test_execute_test_date_no_days_warns_and_keeps_all(self, cd_time):
        n_before = len(cd_time.data_filtered)
        f = Time(test_date="2023-02-15")
        with pytest.warns(UserWarning, match="Must specify days"):
            kept = f._execute(cd_time)
        assert len(kept) == n_before

    def test_execute_drop_with_start_days(self, cd_time):
        # start+days resolves end=02-11; df.loc[02-01:02-11] inclusive is
        # 11 rows; drop=True keeps the complement.
        n = len(cd_time.data_filtered)
        kept = Time(start="2023-02-01", days=10, drop=True)._execute(cd_time)
        assert len(kept) == n - 11
        assert pd.Timestamp("2023-02-05") not in kept

    def test_execute_drop_with_test_date(self, cd_time):
        # test_date+days resolves window 02-10..02-20 (11 rows inclusive).
        n = len(cd_time.data_filtered)
        kept = Time(test_date="2023-02-15", days=10, drop=True)._execute(cd_time)
        assert len(kept) == n - 11
        assert pd.Timestamp("2023-02-15") not in kept

    def test_execute_drop_start_only(self, cd_time):
        # start-only resolves end=last row; drop keeps rows BEFORE start.
        # cd_time spans 2023-01-01..2023-03-31 daily (90 rows); window
        # 02-01..03-31 is 59 rows; complement is 31 (Jan 1..Jan 31).
        kept = Time(start="2023-02-01", drop=True)._execute(cd_time)
        assert len(kept) == 31
        assert kept[-1] < pd.Timestamp("2023-02-01")

    def test_execute_drop_end_only(self, cd_time):
        # end-only resolves start=first row; drop keeps rows AFTER end.
        # Window 01-01..02-15 = 46 rows; complement = 44.
        kept = Time(end="2023-02-15", drop=True)._execute(cd_time)
        assert len(kept) == 44
        assert kept[0] > pd.Timestamp("2023-02-15")

    def test_explanation_start_end(self, cd_time):
        f = Time(start="2023-02-01", end="2023-02-15")
        f.run(cd_time)
        assert "outside" in f.explanation
        assert "2023-02-01" in f.explanation
        assert "2023-02-15" in f.explanation

    def test_explanation_drop(self, cd_time):
        f = Time(start="2023-02-01", end="2023-02-15", drop=True)
        f.run(cd_time)
        assert f.explanation.startswith("Data between")
        assert f.explanation.endswith("was removed.")

    def test_explanation_test_date(self, cd_time):
        f = Time(test_date="2023-02-15", days=10)
        f.run(cd_time)
        assert "centered" in f.explanation
        assert "10-day" in f.explanation

    def test_explanation_drop_with_start_days(self, cd_time):
        f = Time(start="2023-02-01", days=10, drop=True)
        f.run(cd_time)
        exp = f.explanation
        assert "within" in exp
        assert "10-day" in exp
        assert "2023-02-01" in exp
        assert "2023-02-11" in exp  # resolved end

    def test_explanation_drop_with_test_date(self, cd_time):
        f = Time(test_date="2023-02-15", days=10, drop=True)
        f.run(cd_time)
        exp = f.explanation
        assert "within" in exp
        assert "2023-02-10" in exp  # resolved start
        assert "2023-02-20" in exp  # resolved end

    def test_explanation_drop_start_only(self, cd_time):
        f = Time(start="2023-02-01", drop=True)
        f.run(cd_time)
        assert f.explanation == "Data from 2023-02-01 onward was removed."

    def test_explanation_drop_end_only(self, cd_time):
        f = Time(end="2023-02-15", drop=True)
        f.run(cd_time)
        assert f.explanation == "Data up to 2023-02-15 was removed."

    def test_explanation_start_only(self, cd_time):
        f = Time(start="2023-02-01")
        f.run(cd_time)
        assert f.explanation == "Data before 2023-02-01 was removed."

    def test_explanation_end_only(self, cd_time):
        f = Time(end="2023-02-15")
        f.run(cd_time)
        assert f.explanation == "Data after 2023-02-15 was removed."

    def test_wrap_year_kwarg_is_rejected(self):
        with pytest.raises(TypeError, match="wrap_year"):
            Time(wrap_year=True)

    def test_helpers_importable_from_filters(self):
        from captest.filters import spans_year, wrap_year_end

        assert callable(spans_year)
        assert callable(wrap_year_end)

    def test_capdata_still_exposes_wrap_year_end(self):
        from captest import capdata

        assert callable(capdata.wrap_year_end)

    def test_wrapper_custom_name_sets_step_label(self, cd_time):
        cd_time.filter_time(start="2023-02-01", end="2023-02-15", custom_name="window")
        assert cd_time.filters[-1].custom_name == "window"


class TestFilterTimeWrapper:
    def test_wrapper_records_filtertime_step(self, cd_time):
        cd_time.filter_time(start="2023-02-01", end="2023-02-15")
        assert len(cd_time.filters) == 1
        assert isinstance(cd_time.filters[0], Time)


class TestFilterCustom:
    def test_execute_applies_func(self, cd_irr):
        kept = Custom(_drop_first)._execute(cd_irr)
        assert list(kept) == [1, 2, 3, 4]

    def test_execute_passes_args_and_kwargs(self, cd_irr):
        # poa values [100, 300, 500, 700, 900] -> > 400 -> indices [2, 3, 4]
        f = Custom(_gt_threshold, threshold=400)
        assert list(f._execute(cd_irr)) == [2, 3, 4]

    def test_execute_with_pandas_method_dropna(self):
        cd = CapData("c")
        cd.data = pd.DataFrame(
            {"power": [1.0, np.nan, 3.0, np.nan, 5.0]},
            index=pd.RangeIndex(5),
        )
        kept = Custom(pd.DataFrame.dropna)._execute(cd)
        assert list(kept) == [0, 2, 4]

    def test_args_repr_renders_func_name(self):
        f = Custom(_gt_threshold, threshold=400)
        args = f.args_repr
        assert "_gt_threshold" in args
        assert "threshold=400" in args
        assert "<function" not in args

    def test_args_repr_with_positional_args(self):
        f = Custom(pd.DataFrame.between_time, "9:00", "17:00")
        args = f.args_repr
        assert "between_time" in args
        assert "'9:00'" in args
        assert "'17:00'" in args

    def test_args_repr_handles_callable_without_dunder_name(self):
        # functools.partial has no __name__; args_repr must fall back rather
        # than raising AttributeError mid-run() and leaving the step
        # half-applied.
        import functools

        f = Custom(functools.partial(pd.DataFrame.dropna))
        args = f.args_repr  # must not raise
        assert isinstance(args, str)

    def test_explanation_reuses_call(self, cd_irr):
        f = Custom(_drop_first)
        # Pre-run: BaseSummaryStep.explanation returns None until ix_after is
        # set by run(). Pinning this keeps the guard a tested contract.
        assert f.explanation is None
        f.run(cd_irr)
        exp = f.explanation
        assert "_drop_first" in exp
        assert exp.endswith("was applied.")

    def test_custom_name_passes_through(self):
        f = Custom(_drop_first, custom_name="prune")
        assert f.custom_name == "prune"


class TestFilterCustomWrapper:
    def test_wrapper_records_filtercustom_step(self, cd_irr):
        cd_irr.filter_custom(_drop_first)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], Custom)

    def test_wrapper_passes_args_kwargs_to_func(self, cd_irr):
        cd_irr.filter_custom(_gt_threshold, threshold=400)
        assert list(cd_irr.data_filtered.index) == [2, 3, 4]

    def test_wrapper_custom_name_kwarg_is_kwonly(self, cd_irr):
        cd_irr.filter_custom(_drop_first, custom_name="prune")
        assert cd_irr.filters[0].custom_name == "prune"


class TestRollingStd:
    def test_execute_removes_unstable_and_leading_nan(self, cd_roll):
        # window=2: rolling std is NaN at row 0 (dropped), 0 on the stable
        # rows, and large where the spike enters/leaves (rows 3, 4 dropped).
        f = RollingStd(window=2, threshold=50, column="poa")
        assert list(f._execute(cd_roll)) == [1, 2, 5]

    def test_execute_matches_oracle(self, cd_roll):
        def unstable_irr_filter(df, irr_col, window, threshold):
            std = df[irr_col].rolling(window).std()
            return df[std < threshold]

        f = RollingStd(window=2, threshold=50, column="poa")
        oracle = unstable_irr_filter(cd_roll.data, "poa", 2, 50)
        assert list(f._execute(cd_roll)) == list(oracle.index)

    def test_execute_defaults_column_to_poa(self, cd_roll):
        f = RollingStd(window=2, threshold=50)  # column=None -> poa
        assert list(f._execute(cd_roll)) == [1, 2, 5]

    def test_execute_requires_window_and_threshold(self, cd_roll):
        with pytest.raises(ValueError, match="window and threshold"):
            RollingStd(threshold=50, column="poa")._execute(cd_roll)
        with pytest.raises(ValueError, match="window and threshold"):
            RollingStd(window=2, column="poa")._execute(cd_roll)

    def test_config_round_trips(self):
        f = RollingStd(window="10min", threshold=20, column="poa")
        cfg = f.to_config()
        assert cfg["type"] == "RollingStd"
        f2 = step_from_config(cfg)
        assert isinstance(f2, RollingStd)
        assert f2.window == "10min"
        assert f2.threshold == 20
        assert f2.column == "poa"

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["RollingStd"] is RollingStd

    def test_explanation_reports_resolved_column(self, cd_roll):
        f = RollingStd(window=2, threshold=50)
        f.run(cd_roll)
        assert f.explanation == (
            "Intervals where the rolling std (window=2) of poa was at or "
            "above 50 were removed."
        )


class TestRollingStdWrapper:
    def test_wrapper_records_step(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50)
        assert len(cd_roll.filters) == 1
        assert isinstance(cd_roll.filters[0], RollingStd)

    def test_wrapper_filters_data(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50)
        assert list(cd_roll.data_filtered.index) == [1, 2, 5]

    def test_wrapper_custom_name_sets_step_label(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50, custom_name="stability")
        assert cd_roll.filters[-1].custom_name == "stability"

    def test_wrapper_explicit_column(self, cd_roll):
        cd_roll.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cd_roll.filter_rolling_std(2, 50, column="ghi")
        assert list(cd_roll.data_filtered.index) == [1, 2, 3, 4, 5]


class TestAbsDiffPrev:
    def test_execute_removes_step_and_leading_nan(self, cd_step):
        # abs(diff/col): row0 NaN (dropped), row2 ~0.66 (dropped), the rest
        # are well under 0.05.
        f = AbsDiffPrev(threshold=0.05, column="poa")
        assert list(f._execute(cd_step)) == [1, 3, 4]

    def test_execute_matches_oracle(self, cd_step):
        def filter_abs_perc_diff_prev_interval(data, column, threshold=0.05):
            return (
                data.assign(diff=lambda x: x[column].diff())
                .assign(abs_diff=lambda x: abs(x["diff"] / x[column]))
                .loc[lambda x: x["abs_diff"] <= threshold]
            )

        f = AbsDiffPrev(threshold=0.05, column="poa")
        oracle = filter_abs_perc_diff_prev_interval(cd_step.data, "poa", 0.05)
        assert list(f._execute(cd_step)) == list(oracle.index)

    def test_execute_defaults_column_to_poa(self, cd_step):
        f = AbsDiffPrev(threshold=0.05)  # column=None -> poa
        assert list(f._execute(cd_step)) == [1, 3, 4]

    def test_threshold_defaults_to_005(self):
        assert AbsDiffPrev().threshold == 0.05

    def test_execute_larger_threshold_keeps_more(self, cd_step):
        # threshold 0.7 keeps the ~0.66 step row too; only the NaN row drops.
        f = AbsDiffPrev(threshold=0.7, column="poa")
        assert list(f._execute(cd_step)) == [1, 2, 3, 4]

    def test_config_round_trips(self):
        f = AbsDiffPrev(threshold=0.1, column="poa")
        cfg = f.to_config()
        assert cfg["type"] == "AbsDiffPrev"
        f2 = step_from_config(cfg)
        assert isinstance(f2, AbsDiffPrev)
        assert f2.threshold == 0.1
        assert f2.column == "poa"

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["AbsDiffPrev"] is AbsDiffPrev

    def test_explanation_reports_resolved_column(self, cd_step):
        f = AbsDiffPrev(threshold=0.05)
        f.run(cd_step)
        assert f.explanation == (
            "Intervals where poa changed by more than 0.05 (fractional) from "
            "the previous interval were removed."
        )


class TestAbsDiffPrevWrapper:
    def test_wrapper_records_step(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05)
        assert len(cd_step.filters) == 1
        assert isinstance(cd_step.filters[0], AbsDiffPrev)

    def test_wrapper_filters_data(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05)
        assert list(cd_step.data_filtered.index) == [1, 3, 4]

    def test_wrapper_default_threshold(self, cd_step):
        cd_step.filter_abs_diff_prev()  # default 0.05
        assert list(cd_step.data_filtered.index) == [1, 3, 4]

    def test_wrapper_custom_name_sets_step_label(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05, custom_name="stability")
        assert cd_step.filters[-1].custom_name == "stability"

    def test_wrapper_explicit_column(self, cd_step):
        cd_step.data["ghi"] = [500.0, 500.0, 500.0, 500.0, 500.0]
        cd_step.filter_abs_diff_prev(0.05, column="ghi")
        # ghi is constant -> only the leading-NaN row drops.
        assert list(cd_step.data_filtered.index) == [1, 2, 3, 4]


class TestBooleanFlag:
    def test_execute_drops_truthy_rows(self, cd_flag):
        f = BooleanFlag(column="backtrack_on")
        assert list(f._execute(cd_flag)) == [0, 2, 4]

    def test_execute_matches_oracle(self, cd_flag):
        def remove_inter_row_shading(data, boolean_column="backtrack_on"):
            return data[~data[boolean_column].astype(bool)]

        f = BooleanFlag(column="backtrack_on")
        oracle = remove_inter_row_shading(cd_flag.data, "backtrack_on")
        assert list(f._execute(cd_flag)) == list(oracle.index)

    def test_execute_invert_keeps_truthy(self, cd_flag):
        f = BooleanFlag(column="backtrack_on", invert=True)
        assert list(f._execute(cd_flag)) == [1, 3]

    def test_execute_coerces_int_and_nan(self, cd_flag):
        # astype(bool): 0->False, nonzero->True, NaN->True.
        cd_flag.data["mixed"] = [0, 1, 0, np.nan, 2]
        f = BooleanFlag(column="mixed")
        assert list(f._execute(cd_flag)) == [0, 2]

    def test_execute_requires_column(self, cd_flag):
        with pytest.raises(ValueError, match="requires a column"):
            BooleanFlag()._execute(cd_flag)

    def test_config_round_trips(self):
        f = BooleanFlag(column="backtrack_on", invert=True)
        cfg = f.to_config()
        assert cfg["type"] == "BooleanFlag"
        f2 = step_from_config(cfg)
        assert isinstance(f2, BooleanFlag)
        assert f2.column == "backtrack_on"
        assert f2.invert is True

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["BooleanFlag"] is BooleanFlag

    def test_explanation_default(self, cd_flag):
        f = BooleanFlag(column="backtrack_on")
        f.run(cd_flag)
        assert f.explanation == ("Intervals flagged True in backtrack_on were removed.")

    def test_explanation_invert(self, cd_flag):
        f = BooleanFlag(column="backtrack_on", invert=True)
        f.run(cd_flag)
        assert f.explanation == (
            "Intervals flagged False in backtrack_on were removed."
        )

    def test_explanation_none_before_run(self):
        assert BooleanFlag(column="backtrack_on").explanation is None


class TestBooleanFlagWrapper:
    def test_wrapper_records_step(self, cd_flag):
        cd_flag.filter_flag("backtrack_on")
        assert len(cd_flag.filters) == 1
        assert isinstance(cd_flag.filters[0], BooleanFlag)

    def test_wrapper_filters_data(self, cd_flag):
        cd_flag.filter_flag("backtrack_on")
        assert list(cd_flag.data_filtered.index) == [0, 2, 4]

    def test_wrapper_invert(self, cd_flag):
        cd_flag.filter_flag("backtrack_on", invert=True)
        assert list(cd_flag.data_filtered.index) == [1, 3]

    def test_wrapper_custom_name_sets_step_label(self, cd_flag):
        cd_flag.filter_flag("backtrack_on", custom_name="no backtracking")
        assert cd_flag.filters[-1].custom_name == "no backtracking"


class TestFilterThreshold:
    def test_wrapper_records_irradiance_step(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        assert len(cd_thresh.filters) == 1
        step = cd_thresh.filters[0]
        assert isinstance(step, Irradiance)
        assert step.col_name == "avail"

    def test_low_only_keeps_at_or_above(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        assert list(cd_thresh.data_filtered.index) == [1, 2, 3, 4]

    def test_high_only_keeps_at_or_below(self, cd_thresh):
        cd_thresh.filter_threshold("temp", high=40)
        assert list(cd_thresh.data_filtered.index) == [0, 1, 4]

    def test_both_bounds_keep_band(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4, high=99.0)
        assert list(cd_thresh.data_filtered.index) == [1, 2, 3]

    def test_custom_name_sets_step_label(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4, custom_name="availability")
        assert cd_thresh.filters[-1].custom_name == "availability"

    def test_explanation_omits_wm2_units(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4, high=99.0)
        exp = cd_thresh.filters[-1].explanation
        assert "W/m^2" not in exp
        assert exp == (
            "Intervals where avail is below 97.4 or above 99.0 were removed."
        )

    def test_serializes_and_replays_as_irradiance(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        config = cd_thresh.filters_to_config()
        assert config[0]["type"] == "Irradiance"
        assert config[0]["col_name"] == "avail"

        fresh = CapData("fresh")
        fresh.data = cd_thresh.data.copy()
        fresh.run_pipeline(config)
        assert list(fresh.data_filtered.index) == [1, 2, 3, 4]


class TestFilterOutliers:
    def test_execute_removes_outliers(self, cd_pp):
        # Default contamination=0.04 on n=50 removes 2 points; verified
        # empirically that indices 5 and 40 (the two most extreme injected
        # outliers) go and index 20 stays at the default contamination.
        n_before = len(cd_pp.data_filtered)
        kept = Outliers()._execute(cd_pp)
        assert len(kept) < n_before
        assert 5 not in kept
        assert 40 not in kept

    def test_execute_higher_contamination_removes_more(self, cd_pp):
        f = Outliers(envelope_kwargs={"contamination": 0.10})
        kept = f._execute(cd_pp)
        for ix in (5, 20, 40):
            assert ix not in kept

    def test_execute_resolves_default_kwargs(self, cd_pp):
        f = Outliers()
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved == {
            "support_fraction": 0.9,
            "contamination": 0.04,
        }

    def test_execute_user_kwargs_override(self, cd_pp):
        f = Outliers(envelope_kwargs={"contamination": 0.10})
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved["contamination"] == 0.10
        assert f.envelope_kwargs_resolved["support_fraction"] == 0.9

    def test_execute_too_many_columns_warns_and_keeps_all(self, cd_pp):
        cd_pp.data["poa2"] = cd_pp.data["poa"]
        cd_pp.column_groups = {"poa": ["poa", "poa2"], "power": ["power"]}
        n_before = len(cd_pp.data_filtered)
        f = Outliers()
        with pytest.warns(UserWarning, match="aggregate_sensors"):
            kept = f._execute(cd_pp)
        assert len(kept) == n_before

    def test_execute_nan_calls_filter_missing(self, cd_pp):
        cd_pp.data.iloc[0, cd_pp.data.columns.get_loc("poa")] = np.nan
        with pytest.warns(UserWarning, match="missing values"):
            kept = Outliers()._execute(cd_pp)
        assert 0 not in kept
        # The nested filter_missing is recorded as its own step in the chain.
        gs = cd_pp.get_summary()
        assert len(gs) == 1
        assert gs.index[0][1] == "Missing"

    def test_pts_removed_excludes_nan_drop(self, cd_pp):
        cd_pp.data.iloc[1, cd_pp.data.columns.get_loc("poa")] = np.nan
        pre_run_pts = len(cd_pp.data_filtered)  # includes the NaN row
        f = Outliers()
        with pytest.warns(UserWarning):
            f.run(cd_pp)
        gs = cd_pp.get_summary()
        assert gs.index[-2][1] == "Missing"
        assert gs.index[-1][1] == "Outliers"
        # f's chain-derived "before" count == the prior step's surviving count.
        f_index = len(cd_pp.filters) - 1
        assert cd_pp._pts_before(f_index) == gs["pts_after_filter"].iloc[-2]
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
        f = Outliers()
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
        f = Outliers()
        f.run(cd_pp)
        exp = f.explanation
        assert "EllipticEnvelope" in exp
        assert exp.endswith("were removed.")


class TestFilterOutliersWrapper:
    def test_wrapper_records_filteroutliers_step(self, cd_pp):
        cd_pp.filter_outliers()
        assert len(cd_pp.filters) == 1
        assert isinstance(cd_pp.filters[0], Outliers)

    def test_wrapper_passes_kwargs(self, cd_pp):
        cd_pp.filter_outliers(contamination=0.10)
        assert cd_pp.filters[0].envelope_kwargs_resolved["contamination"] == 0.10


class TestFilterClearsky:
    def test_execute_keeps_clear_periods(self, nrel_clear_sky):
        n_before = nrel_clear_sky.data_filtered.shape[0]
        kept = Clearsky()._execute(nrel_clear_sky)
        assert len(kept) < n_before
        assert nrel_clear_sky.data_filtered.shape[0] == n_before

    def test_execute_keep_clear_false_inverts_mask(self, nrel_clear_sky):
        clear_kept = Clearsky()._execute(nrel_clear_sky)
        cloudy_kept = Clearsky(keep_clear=False)._execute(nrel_clear_sky)
        full = nrel_clear_sky.data_filtered.index
        assert clear_kept.union(cloudy_kept).equals(full)
        assert clear_kept.intersection(cloudy_kept).empty

    def test_execute_resolves_default_detect_kwargs(self, nrel_clear_sky):
        f = Clearsky()
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved == {"infer_limits": True}

    def test_execute_user_kwargs_override(self, nrel_clear_sky):
        f = Clearsky(detect_kwargs={"infer_limits": False, "window_length": 30})
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved["infer_limits"] is False
        assert f.detect_kwargs_resolved["window_length"] == 30

    def test_execute_no_ghi_mod_csky_warns_and_keeps_all(self, nrel_clear_sky):
        nrel_clear_sky.drop_cols("ghi_mod_csky")
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Modeled clear sky"):
            kept = Clearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_no_measured_ghi_group_warns_and_keeps_all(self, nrel_clear_sky):
        # ghi_mod_csky present (so the first guard passes) but no measured
        # GHI column group at all — must warn and no-op rather than IndexError.
        del nrel_clear_sky.column_groups["irr-ghi-"]
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="No measured GHI"):
            kept = Clearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_too_many_ghi_categories_warns_and_keeps_all(self, nrel_clear_sky):
        # Add a second measured GHI category alongside the existing irr-ghi-
        # (irr-ghi-clear_sky is excluded by the filter, so two real groups
        # remain and trigger the "Too many ghi categories" guard).
        nrel_clear_sky.column_groups["irr-ghi-pyran"] = ["some_pyran_col"]
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Too many ghi"):
            kept = Clearsky()._execute(nrel_clear_sky)
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
            kept = Clearsky()._execute(nrel_clear_sky)
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
                kept = Clearsky()._execute(nrel_clear_sky)
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
        f = Clearsky(ghi_col="ws 2 ghi W/m^2")
        kept = f._execute(nrel_clear_sky)
        assert len(kept) < nrel_clear_sky.data_filtered.shape[0]

    def test_args_repr_renders_detect_call(self, nrel_clear_sky):
        f = Clearsky()
        assert "detect_clearsky" not in f.args_repr
        f._execute(nrel_clear_sky)
        assert "detect_clearsky(" in f.args_repr
        assert "infer_limits=True" in f.args_repr

    def test_explanation_default_says_cloudy(self, nrel_clear_sky):
        f = Clearsky()
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Cloudy intervals")
        assert "detect_clearsky" in f.explanation
        assert f.explanation.endswith("were removed.")

    def test_explanation_keep_clear_false_says_clear(self, nrel_clear_sky):
        f = Clearsky(keep_clear=False)
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Clear intervals")


class TestFilterClearskyWrapper:
    def test_wrapper_records_filterclearsky_step(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky()
        assert len(nrel_clear_sky.filters) == 1
        assert isinstance(nrel_clear_sky.filters[0], Clearsky)

    def test_wrapper_passes_kwargs(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky(infer_limits=False, window_length=30)
        resolved = nrel_clear_sky.filters[0].detect_kwargs_resolved
        assert resolved["infer_limits"] is False
        assert resolved["window_length"] == 30


@pytest.fixture
def cd_backtrack():
    """A tracker CapData with a clear-day index and a tracker site dict.

    Solar position is computed by the filter from ``site['loc']``; geometry
    defaults come from ``site['sys']``.
    """
    from pvlib.location import Location

    idx = pd.date_range(
        "2023-06-15 04:00", "2023-06-15 20:00", freq="5min", tz="Etc/GMT+7"
    )
    cd = CapData("backtrack")
    # A single poa column is enough; the filter reads geometry/solpos, not poa.
    loc = Location(35.0, -100.0, altitude=300, tz="Etc/GMT+7")
    poa = loc.get_solarposition(idx)["apparent_zenith"].to_numpy()
    cd.data = pd.DataFrame({"poa": poa}, index=idx.tz_localize(None))
    cd.regression_cols = {"poa": "poa"}
    cd.site = {
        "loc": {
            "latitude": 35.0,
            "longitude": -100.0,
            "altitude": 300,
            "tz": "Etc/GMT+7",
        },
        "sys": {
            "axis_tilt": 0,
            "axis_azimuth": 180,
            "gcr": 0.4,
            "max_angle": 60,
            "backtrack": True,
            "albedo": 0.2,
        },
    }
    return cd


class TestFilterBacktracking:
    def test_removes_backtracking_keeps_true_tracking(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) < n_before
        # data_filtered is not mutated by _execute alone.
        assert cd_backtrack.data_filtered.shape[0] == n_before

    def test_keep_backtracking_inverts_mask(self, cd_backtrack):
        removed_default = Backtracking()._execute(cd_backtrack)
        kept_backtracking = Backtracking(keep_backtracking=True)._execute(cd_backtrack)
        full = cd_backtrack.data_filtered.index
        assert removed_default.union(kept_backtracking).equals(full)
        assert removed_default.intersection(kept_backtracking).empty

    def test_resolves_geometry_from_site(self, cd_backtrack):
        f = Backtracking()
        f._execute(cd_backtrack)
        assert f.gcr_resolved == 0.4
        assert f.axis_tilt_resolved == 0
        assert f.axis_azimuth_resolved == 180

    def test_explicit_params_override_site(self, cd_backtrack):
        f = Backtracking(gcr=0.25)
        f._execute(cd_backtrack)
        assert f.gcr_resolved == 0.25

    def test_resolves_cross_axis_tilt_from_site(self, cd_backtrack):
        # A site-provided cross_axis_tilt must be honored (not silently 0).
        f_flat = Backtracking()
        f_flat._execute(cd_backtrack)

        cd_backtrack.site["sys"]["cross_axis_tilt"] = 20
        f_sloped = Backtracking()
        sloped_kept = f_sloped._execute(cd_backtrack)
        assert f_sloped.cross_axis_tilt_resolved == 20
        # The resolved slope changes the classification vs. the flat default.
        f_flat_again = Backtracking(cross_axis_tilt=0)
        assert not sloped_kept.equals(f_flat_again._execute(cd_backtrack))

    def test_cross_axis_tilt_defaults_to_zero_when_absent(self, cd_backtrack):
        # No cross_axis_tilt in site sys and none passed -> resolves to 0.
        assert "cross_axis_tilt" not in cd_backtrack.site["sys"]
        f = Backtracking()
        f._execute(cd_backtrack)
        assert f.cross_axis_tilt_resolved == 0

    def test_no_site_warns_and_keeps_all(self, cd_backtrack):
        cd_backtrack.site = None
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="site"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_gcr_zero_in_site_warns_and_keeps_all(self, cd_backtrack):
        cd_backtrack.site["sys"]["gcr"] = 0
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="gcr"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_missing_gcr_key_warns_and_keeps_all(self, cd_backtrack):
        del cd_backtrack.site["sys"]["gcr"]
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="gcr"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_invalid_cross_axis_tilt_warns_and_keeps_all(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="cross_axis_tilt"):
            kept = Backtracking(cross_axis_tilt=90)._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_naive_fall_dst_index_does_not_crash(self, cd_backtrack):
        # A tz-naive index spanning the fall-back DST transition with a lone
        # ambiguous 01:00-01:59 local hour would make ambiguous="infer" raise.
        # The filter's ambiguous=True policy must localize it and run normally.
        idx = pd.date_range(
            "2023-11-05 00:00", "2023-11-05 05:00", freq="30min"
        )  # America/Chicago fall-back at 02:00; tz-naive
        cd_backtrack.data = pd.DataFrame({"poa": range(len(idx))}, index=idx)
        cd_backtrack.regression_cols = {"poa": "poa"}
        cd_backtrack.site["loc"]["tz"] = "America/Chicago"
        n_before = cd_backtrack.data_filtered.shape[0]
        # Night-time rows: predicate is False everywhere, so nothing is removed,
        # but crucially _execute must not raise, and must not silently degrade
        # to the warn-and-no-op path (which would also happen to keep the full
        # index if ambiguous="infer" raised and got swallowed).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_malformed_location_warns_and_keeps_all(self, cd_backtrack):
        # An unknown tz raises ZoneInfoNotFoundError (a KeyError subclass) from
        # get_solarposition; the filter must warn-and-no-op, not crash.
        cd_backtrack.site["loc"]["tz"] = "Not/AZone"
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="solar position"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_missing_location_key_warns_and_keeps_all(self, cd_backtrack):
        # A missing required loc key makes Location(**loc) raise TypeError;
        # the filter must warn-and-no-op.
        del cd_backtrack.site["loc"]["longitude"]
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="solar position"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["Backtracking"] is Backtracking

    def test_config_round_trips(self):
        f = Backtracking(
            gcr=0.3, axis_tilt=5, cross_axis_tilt=10, keep_backtracking=True
        )
        cfg = f.to_config()
        assert cfg["type"] == "Backtracking"
        f2 = step_from_config(cfg)
        assert isinstance(f2, Backtracking)
        assert f2.gcr == 0.3
        assert f2.axis_tilt == 5
        assert f2.cross_axis_tilt == 10
        assert f2.keep_backtracking is True

    def test_config_preserves_none_geometry(self):
        # None geometry (resolve-from-site intent) must survive serialization,
        # including cross_axis_tilt.
        cfg = Backtracking().to_config()
        assert cfg["gcr"] is None
        assert cfg["axis_tilt"] is None
        assert cfg["cross_axis_tilt"] is None
        rebuilt = step_from_config(cfg)
        assert rebuilt.gcr is None
        assert rebuilt.cross_axis_tilt is None

    def test_explanation_reports_resolved_geometry(self, cd_backtrack):
        f = Backtracking()
        f.run(cd_backtrack)
        assert "0.4" in f.explanation
        assert "removed" in f.explanation


class TestBacktrackingWrapper:
    def test_wrapper_records_step(self, cd_backtrack):
        cd_backtrack.filter_backtracking()
        assert len(cd_backtrack.filters) == 1
        assert isinstance(cd_backtrack.filters[0], Backtracking)

    def test_wrapper_filters_data(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        cd_backtrack.filter_backtracking()
        assert cd_backtrack.data_filtered.shape[0] < n_before

    def test_wrapper_custom_name_sets_step_label(self, cd_backtrack):
        cd_backtrack.filter_backtracking(custom_name="no backtrack")
        assert cd_backtrack.filters[-1].custom_name == "no backtrack"

    def test_wrapper_keep_backtracking(self, cd_backtrack):
        cd_default = cd_backtrack
        n_full = cd_default.data.shape[0]
        cd_default.filter_backtracking(keep_backtracking=True)
        # Keeping only backtracking removes the true-tracking midday rows.
        assert cd_default.data_filtered.shape[0] < n_full

    def test_serializes_and_replays(self, cd_backtrack):
        cd_backtrack.filter_backtracking()
        expected_index = list(cd_backtrack.data_filtered.index)
        config = cd_backtrack.filters_to_config()
        assert config[0]["type"] == "Backtracking"

        fresh = CapData("fresh")
        fresh.data = cd_backtrack.data.copy()
        fresh.site = cd_backtrack.site
        fresh.regression_cols = dict(cd_backtrack.regression_cols)
        fresh.run_pipeline(config)
        # None geometry re-resolves from site on replay -> identical result.
        assert list(fresh.data_filtered.index) == expected_index


class TestFilterPvsyst:
    def _cd(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame(
            {"IL Pmin": [0.0, 1.0, 0.0, 2.0], "power": [10.0, 20.0, 30.0, 40.0]},
            index=pd.RangeIndex(4),
        )
        return cd

    def test_execute_drops_positive_rows(self):
        kept = Pvsyst()._execute(self._cd())
        assert list(kept) == [0, 2]

    def test_execute_underscored_column_names(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame({"IL_Pmax": [0.0, 5.0, 0.0]}, index=pd.RangeIndex(3))
        assert list(Pvsyst()._execute(cd)) == [0, 2]

    def test_execute_missing_column_warns(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        with pytest.warns(UserWarning, match="not a column"):
            kept = Pvsyst()._execute(cd)
        assert list(kept) == [0, 1]

    def test_explanation(self):
        f = Pvsyst()
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
        assert list(Shade()._execute(self._cd())) == [0, 2]

    def test_execute_custom_fshdbm(self):
        assert list(Shade(fshdbm=0.6)._execute(self._cd())) == [0, 2, 3]

    def test_execute_query_str(self):
        assert list(Shade(query_str="ShdLoss<=125")._execute(self._cd())) == [
            0,
            1,
            2,
        ]

    def test_explanation_default_shows_resolved_threshold(self):
        f = Shade()
        f.run(self._cd())
        assert "shad" in f.explanation.lower()
        # resolved value, not the literal @fshdbm placeholder
        assert "FShdBm>=1.0" in f.explanation
        assert "@fshdbm" not in f.explanation
        assert f.explanation.endswith("were removed.")

    def test_explanation_custom_query(self):
        f = Shade(query_str="ShdLoss<=125")
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
        kept = Days(days=["10/1/1990", "10/2/1990"])._execute(self._cd())
        assert len(kept) == 48
        assert set(ts.day for ts in kept) == {1, 2}

    def test_execute_drop_days(self):
        kept = Days(days=["10/1/1990"], drop=True)._execute(self._cd())
        assert len(kept) == 48  # 72 - 24
        assert all(ts.day != 1 for ts in kept)

    def test_explanation_keep(self):
        f = Days(days=["10/2/1990"])
        f.run(self._cd())
        assert "except" in f.explanation.lower()

    def test_explanation_drop(self):
        f = Days(days=["10/2/1990"], drop=True)
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
        assert list(PowerFactor(pf=0.95)._execute(self._cd())) == [0, 2]

    def test_execute_no_pf_group_warns(self):
        cd = CapData("pf")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        cd.column_groups = {"real_pwr--": ["power"]}
        with pytest.warns(UserWarning, match="power factor"):
            kept = PowerFactor(pf=0.99)._execute(cd)
        assert list(kept) == [0, 1]

    def test_explanation(self):
        f = PowerFactor(pf=0.95)
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
        assert list(Power(power=500)._execute(self._cd())) == [0, 2]

    def test_execute_percent(self):
        assert list(Power(power=1000, percent=0.5)._execute(self._cd())) == [
            0,
            2,
        ]

    def test_execute_named_column(self):
        assert list(Power(power=500, columns="meter_power")._execute(self._cd())) == [
            0,
            2,
        ]

    def test_execute_bad_columns_warns(self):
        cd = self._cd()
        f = Power(power=500, columns=1)
        with pytest.warns(UserWarning, match="None or a string"):
            kept = f._execute(cd)
        assert len(kept) == cd.data_filtered.shape[0]

    def test_explanation(self):
        f = Power(power=500)
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
        assert list(Missing()._execute(self._cd())) == [0]

    def test_execute_subset_columns(self):
        assert list(Missing(columns=["poa"])._execute(self._cd())) == [0, 2]

    def test_explanation(self):
        f = Missing()
        f.run(self._cd())
        assert "missing" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")


class TestFilterRegression:
    def test_n_std_default_is_2(self):
        assert Regression().n_std == 2

    def test_execute_exposes_fitted_model(self, cd_reg):
        f = Regression()
        f._execute(cd_reg)
        assert hasattr(f, "regression_model")
        assert hasattr(f.regression_model, "resid")

    def test_execute_removes_the_outlier(self, cd_reg):
        kept = Regression(n_std=2)._execute(cd_reg)
        assert 10 not in kept  # the injected outlier is dropped
        assert len(kept) == cd_reg.data_filtered.shape[0] - 1

    def test_execute_kept_rows_within_threshold(self, cd_reg):
        f = Regression(n_std=2)
        kept = f._execute(cd_reg)
        reg = f.regression_model
        threshold = 2 * reg.scale**0.5
        assert (reg.resid.loc[kept].abs() < threshold).all()

    def test_larger_n_std_keeps_more(self, cd_reg):
        kept_2 = Regression(n_std=2)._execute(cd_reg)
        kept_4 = Regression(n_std=4)._execute(cd_reg)
        assert len(kept_4) >= len(kept_2)

    def test_execute_nan_calls_filter_missing(self, cd_reg):
        # NaN in a regression column: must run filter_missing (recording its
        # own step) rather than raise an unalignable-boolean error.
        cd_reg.data.iloc[5, cd_reg.data.columns.get_loc("power")] = np.nan
        with pytest.warns(UserWarning, match="missing values"):
            kept = Regression(n_std=2)._execute(cd_reg)
        assert 5 not in kept  # NaN row removed
        assert cd_reg.get_summary().index[-1][1] == "Missing"

    def test_explanation(self, cd_reg):
        f = Regression(n_std=2)
        f.run(cd_reg)
        assert "residual" in f.explanation.lower()
        assert "2" in f.explanation
        assert f.explanation.endswith("were removed.")


class TestFitRegressionWrapper:
    def test_filter_true_records_filterregression_step(self, cd_reg):
        cd_reg.fit_regression(filter=True, summary=False)
        assert len(cd_reg.filters) == 1
        assert isinstance(cd_reg.filters[0], Regression)

    def test_fit_regression_rejects_inplace_kwarg(self, cd_reg):
        with pytest.raises(TypeError):
            cd_reg.fit_regression(filter=True, inplace=False, summary=False)
        assert cd_reg.filters == []

    def test_filter_false_stores_regression_results(self, cd_reg):
        cd_reg.fit_regression(filter=False, summary=False)
        assert cd_reg.regression_results is not None
        assert cd_reg.filters == []  # plain fit records no filter step


class TestChainDerivedHelpers:
    def test_step_labels_enumerate_repeats(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)
        Irradiance(low=400, high=800).run(cd_irr)
        assert cd_irr._step_labels() == ["Irradiance", "Irradiance-1"]

    def test_step_labels_use_custom_name(self, cd_irr):
        Irradiance(low=200, high=800, custom_name="Irradiance bounds").run(cd_irr)
        assert cd_irr._step_labels() == ["Irradiance bounds"]

    def test_pts_before_first_step_is_full_data(self, cd_irr):
        full = cd_irr.data.shape[0]
        Irradiance(low=200, high=800).run(cd_irr)
        assert cd_irr._pts_before(0) == full

    def test_ix_before_second_step_is_prior_ix_after(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)
        Irradiance(low=400, high=800).run(cd_irr)
        assert list(cd_irr._ix_before(1)) == list(cd_irr.filters[0].ix_after)


class TestRemovedByStep:
    def test_no_filters_returns_empty(self, cd_irr):
        assert cd_irr._removed_by_step() == []

    def test_single_filter(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)
        result = cd_irr._removed_by_step()
        assert len(result) == 1
        i, label, removed_ix = result[0]
        assert i == 0
        assert label == "Irradiance"
        assert list(removed_ix) == [0, 4]

    def test_two_filters_indices_and_labels(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)  # keeps [1,2,3]
        Irradiance(low=400, high=800).run(cd_irr)  # keeps [2,3], removes [1]
        result = cd_irr._removed_by_step()
        assert [(i, label) for i, label, _ in result] == [
            (0, "Irradiance"),
            (1, "Irradiance-1"),
        ]
        assert list(result[1][2]) == [1]

    def test_skips_zero_removal_step(self, cd_irr):
        Irradiance(low=0, high=10000).run(cd_irr)  # keeps all 5 -> removes nothing
        assert cd_irr._removed_by_step() == []

    def test_zero_removal_step_between_real_filters(self, cd_irr):
        Irradiance(low=200, high=800).run(cd_irr)  # removes [0,4]
        Irradiance(low=0, high=10000).run(cd_irr)  # removes nothing -> skipped
        Irradiance(low=400, high=800).run(cd_irr)  # removes [1]
        result = cd_irr._removed_by_step()
        # The no-op middle filter is skipped; the third filter keeps its real
        # index i=2 (skipping affects only which entries are returned, not the
        # chain math).
        assert [(i, label) for i, label, _ in result] == [
            (0, "Irradiance"),
            (2, "Irradiance-2"),
        ]
        assert list(result[1][2]) == [1]


class TestFilterConfigRoundTrip:
    def test_base_to_config_includes_all_params(self):
        cfg = Irradiance(low=200, high=800).to_config()
        assert cfg == {
            "type": "Irradiance",
            "low": 200,
            "high": 800,
            "ref_val": None,
            "col_name": None,
            "units": "W/m^2",
            "custom_name": None,
        }

    def test_base_roundtrip(self):
        cfg = Irradiance(low=200, high=800, custom_name="bounds").to_config()
        step = step_from_config(cfg)
        assert isinstance(step, Irradiance)
        assert step.to_config() == cfg

    def test_rep_cond_func_dict_roundtrips_perc(self):
        rc = RepCond(func={"poa": util.perc_wrap(60), "t_amb": "mean"})
        cfg = rc.to_config()
        assert cfg["func"] == {"poa": "perc_60", "t_amb": "mean"}
        rebuilt = step_from_config(cfg)
        assert rebuilt.func["poa"].__name__ == "perc_wrap(60)"
        assert rebuilt.func["t_amb"] == "mean"

    def test_rep_cond_none_func_roundtrips(self):
        cfg = RepCond().to_config()
        assert cfg["func"] is None
        assert step_from_config(cfg).func is None

    def test_rep_cond_str_func_roundtrips(self):
        cfg = RepCond(func="mean").to_config()
        assert cfg["func"] == "mean"
        assert step_from_config(cfg).func == "mean"

    def test_rep_cond_bare_perc_wrap_func_roundtrips(self):
        cfg = RepCond(func=util.perc_wrap(60)).to_config()
        assert cfg["func"] == "perc_60"
        assert step_from_config(cfg).func.__name__ == "perc_wrap(60)"

    def test_rep_cond_bare_named_callable_func_roundtrips(self):
        import numpy as np

        cfg = RepCond(func=np.mean).to_config()
        assert isinstance(cfg["func"], str) and ":" in cfg["func"]
        assert step_from_config(cfg).func is np.mean

    def test_rep_cond_dict_with_named_callable_roundtrips(self):
        import numpy as np

        cfg = RepCond(func={"poa": np.mean, "t_amb": "mean"}).to_config()
        assert cfg["func"]["t_amb"] == "mean"
        assert ":" in cfg["func"]["poa"]
        rebuilt = step_from_config(cfg).func
        assert rebuilt["poa"] is np.mean
        assert rebuilt["t_amb"] == "mean"

    def test_filter_custom_named_func_roundtrips(self):
        cfg = Custom(pd.DataFrame.head, 3).to_config()
        assert cfg["func"] == "pandas.core.generic:NDFrame.head"
        assert cfg["args"] == [3]
        step = step_from_config(cfg)
        assert step.func is pd.DataFrame.head
        assert step.args == (3,)

    def test_filter_custom_lambda_raises(self):
        with pytest.raises(ValueError, match="lambdas and closures"):
            Custom(lambda df: df).to_config()

    def test_filter_sensors_method_roundtrips(self):
        cfg = Sensors(thresholds={"irr-poa-": 0.05}).to_config()
        assert cfg["method"] == "percent_diff"
        step = step_from_config(cfg)
        assert step.method == "percent_diff"
        assert step.thresholds == {"irr-poa-": 0.05}

    def test_filter_sensors_custom_callable_roundtrips(self):
        cfg = Sensors(method=abs_diff_from_average, thresholds={"poa": 25}).to_config()
        assert cfg["method"] == "captest.filters:abs_diff_from_average"
        step = step_from_config(cfg)
        assert step.method is abs_diff_from_average
        assert step.thresholds == {"poa": 25}

    def test_unknown_type_suggests_closest(self):
        with pytest.raises(ValueError, match="Did you mean 'Irradiance'"):
            step_from_config({"type": "FilterIrradiance"})

    def test_from_config_direct_tolerates_type_key(self):
        # Cls.from_config(Cls(...).to_config()) must round-trip directly, not
        # only via step_from_config — to_config emits a "type" key, so the
        # splatting from_config overrides must drop it.
        base = Irradiance.from_config(Irradiance(low=200, high=800).to_config())
        assert isinstance(base, Irradiance) and base.low == 200

        sensors = Sensors.from_config(
            Sensors(thresholds={"irr-poa-": 0.05}).to_config()
        )
        assert sensors.thresholds == {"irr-poa-": 0.05}
        assert sensors.method == "percent_diff"

        rep = RepCond.from_config(RepCond(func={"poa": util.perc_wrap(60)}).to_config())
        assert rep.func["poa"].__name__ == "perc_wrap(60)"

    def test_registry_covers_all_step_classes(self):
        # Every concrete step class defined in captest.filters must be
        # registered, so a future filter can't be added without a registry
        # entry (which would silently break its YAML round-trip).
        import inspect

        from captest import filters as filters_mod

        concrete = {
            name
            for name, obj in inspect.getmembers(filters_mod, inspect.isclass)
            if issubclass(obj, BaseSummaryStep)
            and obj not in (BaseSummaryStep, BaseFilter)
            and obj.__module__ == "captest.filters"
        }
        assert concrete == set(FILTER_REGISTRY)
        assert FILTER_REGISTRY["RepCond"] is RepCond
        assert FILTER_REGISTRY["Custom"] is Custom


class TestBacktrackingGeometryError:
    def test_valid_geometry_returns_none(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 0) is None

    def test_none_value_reports_param_name(self):
        assert "gcr" in _backtracking_geometry_error(0, 180, None, 0)
        assert "axis_tilt" in _backtracking_geometry_error(None, 180, 0.3, 0)

    def test_gcr_zero_is_invalid(self):
        reason = _backtracking_geometry_error(0, 180, 0, 0)
        assert reason is not None
        assert "gcr" in reason

    def test_gcr_negative_is_invalid(self):
        assert "gcr" in _backtracking_geometry_error(0, 180, -0.3, 0)

    def test_non_finite_values_are_invalid(self):
        assert _backtracking_geometry_error(0, 180, math.nan, 0) is not None
        assert _backtracking_geometry_error(0, 180, math.inf, 0) is not None
        assert _backtracking_geometry_error(math.nan, 180, 0.3, 0) is not None

    def test_string_value_is_invalid_without_typeerror(self):
        # Must not raise TypeError from math.isfinite on a str.
        reason = _backtracking_geometry_error(0, 180, "0.3", 0)
        assert reason is not None
        assert "gcr" in reason

    def test_pd_na_is_invalid_without_typeerror(self):
        reason = _backtracking_geometry_error(0, 180, pd.NA, 0)
        assert reason is not None

    def test_bool_is_invalid(self):
        # bool is a numbers.Real subtype; must be rejected, not coerced to 1/0.
        assert _backtracking_geometry_error(0, 180, True, 0) is not None

    def test_cross_axis_tilt_at_90_is_invalid(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 90) is not None
        assert _backtracking_geometry_error(0, 180, 0.3, -90) is not None

    def test_cross_axis_tilt_within_range_is_valid(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 45) is None
        assert _backtracking_geometry_error(0, 180, 0.3, -45) is None


class TestBacktrackingActiveHelper:
    @pytest.fixture
    def clear_day_solpos(self):
        """Solar position over a clear June day at a mid-latitude site."""
        from pvlib.location import Location

        loc = Location(35.0, -100.0, altitude=300, tz="Etc/GMT+7")
        times = pd.date_range(
            "2023-06-15 04:00", "2023-06-15 20:00", freq="5min", tz="Etc/GMT+7"
        )
        sp = loc.get_solarposition(times)
        return sp["apparent_zenith"], sp["azimuth"]

    def test_matches_pvlib_singleaxis_at_sun_up(self, clear_day_solpos):
        from pvlib import tracking

        zen, azi = clear_day_solpos
        axis_tilt, axis_azimuth, gcr = 0, 180, 0.4
        # Oracle: singleaxis with max_angle high enough to avoid clipping so the
        # backtrack on/off difference isolates the backtracking decision.
        tracked = tracking.singleaxis(
            apparent_zenith=zen,
            solar_azimuth=azi,
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            max_angle=90,
            backtrack=True,
            gcr=gcr,
            cross_axis_tilt=0,
        )
        true_track = tracking.singleaxis(
            apparent_zenith=zen,
            solar_azimuth=azi,
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            max_angle=90,
            backtrack=False,
            gcr=gcr,
            cross_axis_tilt=0,
        )
        # pvlib backtracks exactly where the tracked angle differs from the
        # true-tracking angle (both non-NaN, i.e. sun up).
        sun_up = tracked["tracker_theta"].notna() & true_track["tracker_theta"].notna()
        pvlib_backtracking = (
            (tracked["tracker_theta"] - true_track["tracker_theta"]).abs() > 1e-6
        ) & sun_up

        mask = backtracking_active(zen, azi, axis_tilt, axis_azimuth, gcr)
        # Compare only where the sun is up (the helper's <=90 term and pvlib's
        # NaN handling agree there).
        assert mask[sun_up].equals(pvlib_backtracking[sun_up])

    def test_sun_down_intervals_are_false(self, clear_day_solpos):
        zen, azi = clear_day_solpos
        mask = backtracking_active(zen, azi, 0, 180, 0.4)
        assert not mask[zen > 90].any()

    def test_cross_axis_tilt_changes_result(self, clear_day_solpos):
        zen, azi = clear_day_solpos
        flat = backtracking_active(zen, azi, 0, 180, 0.4, cross_axis_tilt=0)
        sloped = backtracking_active(zen, azi, 0, 180, 0.4, cross_axis_tilt=20)
        assert not flat.equals(sloped)

    def test_invalid_gcr_raises(self):
        zen = pd.Series([30.0, 45.0])
        azi = pd.Series([90.0, 100.0])
        with pytest.raises(ValueError, match="gcr"):
            backtracking_active(zen, azi, 0, 180, 0)

    def test_non_numeric_geometry_raises_valueerror_not_typeerror(self):
        zen = pd.Series([30.0, 45.0])
        azi = pd.Series([90.0, 100.0])
        with pytest.raises(ValueError):
            backtracking_active(zen, azi, 0, 180, "0.4")
