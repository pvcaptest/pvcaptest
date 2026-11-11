import numpy as np
import pandas as pd
import pytest
import yaml

from captest import checks


class FakeCapData:
    """Minimal CapData stand-in: data, data_filtered and regression_cols."""

    def __init__(self, data, regression_cols, filtered=None):
        self.data = data
        self.regression_cols = regression_cols
        self.data_filtered = data if filtered is None else filtered


def make_data(n=120, irr=500.0, start="2025-06-01 10:00"):
    index = pd.date_range(start, periods=n, freq="15min")
    return pd.DataFrame(
        {
            "poa_col": irr,
            "front_col": irr,
            "raw_col": irr,
        },
        index=index,
    )


ROLES = {"poa": "poa_col", "front_poa": "front_col"}


@pytest.fixture
def cd():
    return FakeCapData(make_data(), dict(ROLES))


def ctx(cd, **kw):
    return checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index, **kw)


class TestDuration:
    def test_pass_at_exact_hours(self, cd):
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, ctx(cd))
        assert res.passed
        assert res.detail["hours"].iloc[0] == 30.0
        assert "30 h at or above 400" in res.summary

    def test_fail_one_short(self):
        cd = FakeCapData(make_data(119), dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, ctx(cd))
        assert not res.passed
        assert res.detail["hours"].iloc[0] == 29.75

    def test_counts_only_pr_rows(self, cd):
        context = checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index[:100])
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, context)
        assert not res.passed
        assert res.detail["hours"].iloc[0] == 25.0

    def test_nan_irradiance_never_qualifies(self):
        data = make_data(120)
        data.iloc[:10, :] = np.nan
        cd = FakeCapData(data, dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, ctx(cd))
        assert res.detail["hours"].iloc[0] == 27.5

    def test_inf_irradiance_never_qualifies(self):
        data = make_data(120)
        data.iloc[:4, :] = np.inf
        cd = FakeCapData(data, dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, ctx(cd))
        assert res.detail["hours"].iloc[0] == 29.0


class TestIntervalCount:
    def test_boundary_is_at_or_above(self):
        cd = FakeCapData(make_data(10, irr=400.0), dict(ROLES))
        ok = checks.IntervalCount(threshold=400, min_count=10).evaluate(cd, ctx(cd))
        assert ok.passed
        assert ok.detail["count"].iloc[0] == 10
        bad = checks.IntervalCount(threshold=400.1, min_count=1).evaluate(cd, ctx(cd))
        assert not bad.passed


class TestValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ValueError):
            checks.Duration(threshold=bad, min_hours=1)
        with pytest.raises(ValueError):
            checks.Duration(threshold=1, min_hours=bad)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            checks.Duration(threshold=1, min_hours=-1)
        with pytest.raises(ValueError):
            checks.IntervalCount(threshold=1, min_count=-1)

    def test_missing_required(self):
        with pytest.raises(ValueError):
            checks.Duration(threshold=1)


class TestIrradiance:
    def test_default_reads_front_poa(self):
        data = make_data()
        data["front_col"] = 100.0
        cd = FakeCapData(data, dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, ctx(cd))
        assert res.detail["hours"].iloc[0] == 0.0

    def test_falls_back_to_poa(self):
        cd = FakeCapData(make_data(), {"poa": "poa_col"})
        assert checks.resolve_irradiance(cd, "front_poa") == "poa_col"

    def test_poa_role(self, cd):
        assert checks.resolve_irradiance(cd, "poa") == "poa_col"

    def test_raw_column(self, cd):
        assert checks.resolve_irradiance(cd, "raw_col") == "raw_col"

    def test_unknown(self, cd):
        with pytest.raises(ValueError, match="nope"):
            checks.resolve_irradiance(cd, "nope")


class TestSelection:
    def test_no_rows(self, cd):
        context = checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index[:0])
        res = checks.Duration(threshold=400, min_hours=1).evaluate(cd, context)
        assert res.passed is False
        assert res.summary == "no data in the selected rows"

    def test_pr_rows_none_falls_back_to_filtered(self):
        data = make_data()
        cd = FakeCapData(data, dict(ROLES), filtered=data.iloc[:4])
        context = checks.CheckContext(dt_hours=0.25)
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, context)
        assert res.detail["hours"].iloc[0] == 1.0

    def test_filtered_mode(self):
        data = make_data()
        cd = FakeCapData(data, dict(ROLES), filtered=data.iloc[:8])
        chk = checks.Duration(threshold=400, min_hours=0, data="filtered")
        assert chk.evaluate(cd, ctx(cd)).detail["hours"].iloc[0] == 2.0

    def test_unfiltered_period(self):
        index = pd.date_range("2025-06-01", "2025-06-05 23:45", freq="15min")
        data = pd.DataFrame({"front_col": 500.0, "poa_col": 500.0}, index=index)
        kept = data.loc["2025-06-02 10:00":"2025-06-03 08:00"]
        cd = FakeCapData(data, dict(ROLES), filtered=kept)
        chk = checks.IntervalCount(threshold=0, min_count=0, data="unfiltered_period")
        # fallback: whole days 2 and 3
        sel = checks.select_rows(cd, chk, ctx(cd))
        assert sel.index.min() == pd.Timestamp("2025-06-02 00:00")
        assert sel.index.max() == pd.Timestamp("2025-06-03 23:45")
        # context.period beats the fallback
        period = (pd.Timestamp("2025-06-04 12:00"), pd.Timestamp("2025-06-05 01:00"))
        sel = checks.select_rows(cd, chk, ctx(cd, period=period))
        assert sel.index.min() == pd.Timestamp("2025-06-04 00:00")
        assert sel.index.max() == pd.Timestamp("2025-06-05 23:45")
        # own start/end beats context.period
        chk.start, chk.end = "2025-06-01", "2025-06-01"
        sel = checks.select_rows(cd, chk, ctx(cd, period=period))
        assert len(sel) == 96
        assert sel.index.min() == pd.Timestamp("2025-06-01 00:00")


class TestConfig:
    def test_round_trip_omits_defaults(self):
        chk = checks.Duration(threshold=400, min_hours=30)
        assert chk.to_config() == {
            "type": "Duration",
            "threshold": 400,
            "min_hours": 30,
        }
        assert checks.check_from_config(chk.to_config()).to_config() == chk.to_config()

    def test_round_trip_non_defaults(self):
        chk = checks.IntervalCount(
            threshold=400,
            min_count=5,
            irr="poa",
            data="unfiltered_period",
            start="2025-06-01",
            end="2025-06-02",
            custom_name="enough",
        )
        config = chk.to_config()
        assert config["irr"] == "poa" and config["custom_name"] == "enough"
        rebuilt = checks.check_from_config(config)
        assert rebuilt.to_config() == config
        assert rebuilt.label == "enough"

    def test_unknown_type_hint(self):
        with pytest.raises(ValueError, match="Did you mean 'Duration'"):
            checks.check_from_config({"type": "Duraton"})

    def test_strict_attrs(self):
        chk = checks.Duration(threshold=400, min_hours=30)
        with pytest.raises(AttributeError):
            chk.min_hour = 3


class TestExtra:
    def test_half_set_period_rejected(self):
        with pytest.raises(ValueError, match="start and end"):
            checks.Duration(threshold=1, min_hours=1, start="2025-06-01")

    def test_missing_type(self):
        cfg = {"threshold": 1}
        with pytest.raises(ValueError, match="missing the 'type'"):
            checks.check_from_config(cfg)
        assert cfg == {"threshold": 1}

    def test_numpy_numbers_accepted(self):
        chk = checks.IntervalCount(threshold=np.float64(4), min_count=np.int64(3))
        assert chk.min_count == 3

    def test_role_with_missing_column(self):
        cd = FakeCapData(make_data(), {"poa": "gone"})
        with pytest.raises(ValueError, match="'poa'.*'gone'"):
            checks.resolve_irradiance(cd, "poa")

    def test_empty_filtered_unfiltered_period(self):
        data = make_data()
        cd = FakeCapData(data, dict(ROLES), filtered=data.iloc[:0])
        chk = checks.Duration(threshold=1, min_hours=0, data="unfiltered_period")
        res = chk.evaluate(cd, checks.CheckContext(dt_hours=0.25))
        assert res.passed is False
        assert res.summary == "no data in the selected rows"

    def test_tz_aware_unfiltered_period(self):
        index = pd.date_range(
            "2025-06-01", "2025-06-03 23:45", freq="15min", tz="Etc/GMT+7"
        )
        data = pd.DataFrame({"front_col": 500.0, "poa_col": 500.0}, index=index)
        cd = FakeCapData(data, dict(ROLES))
        chk = checks.Duration(
            threshold=1,
            min_hours=0,
            data="unfiltered_period",
            start="2025-06-02",
            end="2025-06-02",
        )
        sel = checks.select_rows(cd, chk, ctx(cd))
        assert len(sel) == 96
        assert sel.index.min() == pd.Timestamp("2025-06-02", tz="Etc/GMT+7")

    def test_non_default_data_serialized(self):
        chk = checks.Duration(threshold=1, min_hours=1, data="filtered")
        assert chk.to_config()["data"] == "filtered"
        assert checks.check_from_config(chk.to_config()).data == "filtered"

    def test_validate_reruns_in_evaluate(self, cd):
        chk = checks.Duration(threshold=1, min_hours=1)
        chk.threshold = float("nan")
        with pytest.raises(ValueError):
            chk.evaluate(cd, ctx(cd))


def day_frame(runs, dt="15min", tz=None, days=1, start="2025-06-01"):
    """Build a one-column frame of 500 W/m2 rows with ``runs`` overriding values."""
    index = pd.date_range(start, periods=96 * days, freq=dt, tz=tz)
    data = pd.DataFrame({"front_col": 0.0, "poa_col": 0.0}, index=index)
    for first, last, value in runs:
        data.iloc[first:last, :] = value
    return data


def day_ctx(cd, **kw):
    return checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index, **kw)


def run_check(chk, data, filtered=None, **kw):
    cd = FakeCapData(data, dict(ROLES), filtered=filtered)
    return chk.evaluate(cd, day_ctx(cd, **kw))


class TestContinuousHoursDays:
    def make(self, **kw):
        return checks.ContinuousHoursDays(threshold=600, hours=3, min_days=1, **kw)

    def test_default_data_mode(self):
        assert self.make().data == "unfiltered_period"

    def test_three_hour_run_qualifies(self):
        res = run_check(self.make(), day_frame([(40, 52, 700.0)]))
        assert res.passed
        assert res.detail["longest_run_hours"].iloc[0] == 3.0
        assert bool(res.detail["qualifies"].iloc[0])

    def test_eleven_rows_do_not(self):
        res = run_check(self.make(), day_frame([(40, 51, 700.0)]))
        assert not res.passed
        assert res.detail["longest_run_hours"].iloc[0] == 2.75

    def test_boundary_is_at_or_above(self):
        res = run_check(self.make(), day_frame([(40, 52, 600.0)]))
        assert res.passed

    def test_nan_breaks_run(self):
        data = day_frame([(40, 52, 700.0)])
        data.iloc[46, :] = np.nan
        res = run_check(self.make(), data)
        assert not res.passed
        assert res.detail["longest_run_hours"].iloc[0] == 1.5

    def test_inf_breaks_run(self):
        data = day_frame([(40, 52, 700.0)])
        data.iloc[46, :] = np.inf
        res = run_check(self.make(), data)
        assert not res.passed

    def test_missing_timestamp_breaks_run(self):
        data = day_frame([(40, 52, 700.0)]).drop(pd.Timestamp("2025-06-01 11:30"))
        res = run_check(self.make(), data)
        assert not res.passed
        assert res.detail["longest_run_hours"].iloc[0] == 1.5

    def test_days_are_local_dates(self):
        data = day_frame([(88, 96, 700.0)], tz="America/Chicago", days=2)
        data.iloc[96:100, :] = 700.0
        res = run_check(self.make(), data)
        assert len(res.detail) == 2
        assert not res.detail["qualifies"].any()
        assert not res.passed
        assert list(res.detail["longest_run_hours"]) == [2.0, 1.0]

    def test_min_days(self):
        data = day_frame([(40, 52, 700.0), (136, 148, 700.0)], days=3)
        chk = checks.ContinuousHoursDays(threshold=600, hours=3, min_days=2)
        res = run_check(chk, data)
        assert res.passed
        assert list(res.detail["qualifies"]) == [True, True, False]
        chk3 = checks.ContinuousHoursDays(threshold=600, hours=3, min_days=3)
        assert not run_check(chk3, data).passed

    def test_filter_does_not_break_run(self):
        data = day_frame([(40, 52, 700.0)])
        filtered = data.drop(data.index[44:48])
        res = run_check(self.make(), data, filtered=filtered)
        assert res.passed

    def test_start_end_restrict_days(self):
        data = day_frame([(40, 52, 700.0), (136, 148, 700.0)], days=3)
        chk = self.make(start="2025-06-02", end="2025-06-02")
        res = run_check(chk, data)
        assert len(res.detail) == 1
        assert res.detail.index[0] == pd.Timestamp("2025-06-02")

    def test_round_trip_keeps_period(self):
        chk = self.make(start="2025-06-01", end="2025-06-03")
        config = chk.to_config()
        assert config["start"] == "2025-06-01"
        rebuilt = checks.check_from_config(config)
        assert rebuilt.to_config() == config
        assert isinstance(rebuilt, checks.ContinuousHoursDays)

    @pytest.mark.parametrize(
        "kw",
        [
            {"threshold": float("nan"), "hours": 3, "min_days": 1},
            {"threshold": 600, "hours": -1, "min_days": 1},
            {"threshold": 600, "hours": 3, "min_days": -1},
            {"threshold": 600, "hours": 3},
        ],
    )
    def test_validation(self, kw):
        with pytest.raises(ValueError):
            checks.ContinuousHoursDays(**kw)


class TestDailyInsolationDays:
    def test_sums_with_dt(self):
        data = day_frame([(0, 8, 500.0)])
        chk = checks.DailyInsolationDays(threshold_wh_m2=1000, min_days=1)
        res = run_check(chk, data)
        assert res.detail["insolation_wh_m2"].iloc[0] == 1000.0
        assert res.passed

    def test_nan_and_inf_contribute_zero(self):
        data = day_frame([(0, 8, 500.0)])
        data.iloc[0, :] = np.nan
        data.iloc[1, :] = np.inf
        chk = checks.DailyInsolationDays(threshold_wh_m2=0, min_days=1)
        res = run_check(chk, data)
        assert res.detail["insolation_wh_m2"].iloc[0] == 750.0

    def test_boundary_and_min_days(self):
        data = day_frame([(0, 8, 500.0), (96, 104, 499.0)], days=2)
        chk = checks.DailyInsolationDays(threshold_wh_m2=1000, min_days=2)
        res = run_check(chk, data)
        assert list(res.detail["qualifies"]) == [True, False]
        assert not res.passed

    def test_default_data_mode_and_validation(self):
        chk = checks.DailyInsolationDays(threshold_wh_m2=1, min_days=1)
        assert chk.data == "unfiltered_period"
        with pytest.raises(ValueError):
            checks.DailyInsolationDays(threshold_wh_m2=float("inf"), min_days=1)
        with pytest.raises(ValueError):
            checks.DailyInsolationDays(threshold_wh_m2=1, min_days=-2)

    def test_round_trip(self):
        chk = checks.DailyInsolationDays(
            threshold_wh_m2=1000, min_days=2, start="2025-06-01", end="2025-06-04"
        )
        rebuilt = checks.check_from_config(chk.to_config())
        assert rebuilt.to_config() == chk.to_config()


def nrel_data(spec):
    """One day per ``spec`` entry of (insolation_high, has_run)."""
    days = []
    for n, (high, run) in enumerate(spec):
        day = day_frame(
            [], days=1, start=pd.Timestamp("2025-06-01") + pd.Timedelta(days=n)
        )
        if high:
            day.iloc[8:48, :] = 100.0
            day.iloc[48:60, :] = 700.0 if run else 590.0
            day.iloc[60:96, :] = 100.0
        days.append(day)
    return pd.concat(days)


class TestNrelMinimumIrradiance:
    def test_defaults(self):
        chk = checks.NrelMinimumIrradiance()
        assert (chk.irr_threshold, chk.hours, chk.insolation_wh_m2) == (600, 3, 3000)
        assert (chk.min_insolation_days, chk.min_irr_days) == (5, 3)
        assert chk.data == "unfiltered_period"

    def test_five_days_pass(self):
        data = nrel_data([(True, True)] * 3 + [(True, False)] * 2)
        res = run_check(checks.NrelMinimumIrradiance(), data)
        assert res.passed
        assert res.detail["insolation_ok"].all()
        assert int(res.detail["irradiance_ok"].sum()) == 3

    def test_four_insolation_days_fail(self):
        data = nrel_data([(True, True)] * 3 + [(True, False), (False, False)])
        res = run_check(checks.NrelMinimumIrradiance(), data)
        assert not res.passed

    def test_two_runs_fail(self):
        data = nrel_data([(True, True)] * 2 + [(True, False)] * 3)
        res = run_check(checks.NrelMinimumIrradiance(), data)
        assert not res.passed

    def test_seven_days_runs_counted_among_insolation_days(self):
        data = nrel_data(
            [(True, True)] * 3 + [(True, False)] * 2 + [(False, False)] * 2
        )
        assert run_check(checks.NrelMinimumIrradiance(), data).passed

    def test_run_on_low_insolation_day_does_not_count(self):
        data = nrel_data([(True, True)] * 2 + [(True, False)] * 3)
        extra = day_frame([(40, 52, 700.0)], start="2025-06-06")
        res = run_check(checks.NrelMinimumIrradiance(), pd.concat([data, extra]))
        assert not res.passed
        assert not bool(res.detail["insolation_ok"].iloc[-1])
        assert bool(res.detail["irradiance_ok"].iloc[-1]) is False

    def test_detail_columns_and_round_trip(self):
        data = nrel_data([(True, True)] * 5)
        res = run_check(checks.NrelMinimumIrradiance(), data)
        assert {
            "insolation_wh_m2",
            "longest_run_hours",
            "insolation_ok",
            "irradiance_ok",
        } <= set(res.detail.columns)
        chk = checks.NrelMinimumIrradiance(start="2025-06-01", end="2025-06-05")
        config = chk.to_config()
        assert config["start"] == "2025-06-01" and config["end"] == "2025-06-05"
        assert checks.check_from_config(config).to_config() == config

    def test_validation(self):
        with pytest.raises(ValueError):
            checks.NrelMinimumIrradiance(min_irr_days=-1)
        with pytest.raises(ValueError):
            checks.NrelMinimumIrradiance(hours=float("nan"))


class TestDayCheckEdges:
    def make(self, **kw):
        return checks.ContinuousHoursDays(threshold=600, hours=3, min_days=1, **kw)

    def test_inexact_float_product_run_qualifies(self):
        # 3 rows * 0.3 h == 0.8999999999999999 < 0.9
        assert 3 * 0.3 < 0.9
        index = pd.date_range("2025-06-01", periods=80, freq="18min")
        data = pd.DataFrame({"front_col": 0.0, "poa_col": 0.0}, index=index)
        data.iloc[10:13, :] = 700.0
        cd = FakeCapData(data, dict(ROLES))
        context = checks.CheckContext(dt_hours=0.3, pr_rows=data.index)
        chk = checks.ContinuousHoursDays(threshold=600, hours=0.9, min_days=1)
        res = chk.evaluate(cd, context)
        assert res.passed
        assert bool(res.detail["qualifies"].iloc[0])

    @pytest.mark.parametrize("freq, dt", [("10min", 1 / 6), ("1min", 1 / 60)])
    def test_exact_run_on_other_steps(self, freq, dt):
        rows = round(3 / dt)
        index = pd.date_range("2025-06-01", periods=24 * rows, freq=freq)
        data = pd.DataFrame({"front_col": 0.0, "poa_col": 0.0}, index=index)
        data.iloc[5 : 5 + rows, :] = 700.0
        cd = FakeCapData(data, dict(ROLES))
        context = checks.CheckContext(dt_hours=dt, pr_rows=data.index)
        assert self.make().evaluate(cd, context).passed
        data.iloc[5, :] = 0.0
        assert not self.make().evaluate(cd, context).passed

    def test_unsorted_index_matches_sorted(self):
        data = day_frame([(40, 52, 700.0)], days=2)
        shuffled = data.sample(frac=1, random_state=0)
        for chk in (
            self.make(),
            checks.DailyInsolationDays(threshold_wh_m2=1000, min_days=1),
            checks.NrelMinimumIrradiance(),
        ):
            expected = run_check(chk, data)
            actual = run_check(chk, shuffled)
            assert actual.passed == expected.passed
            pd.testing.assert_frame_equal(actual.detail, expected.detail)

    def test_filtered_mode_sees_filtered_rows(self):
        data = day_frame([(40, 52, 700.0)])
        filtered = data.drop(data.index[44:48])
        assert run_check(self.make(), data, filtered=filtered).passed
        chk = self.make(data="filtered")
        assert not run_check(chk, data, filtered=filtered).passed
        ins = checks.DailyInsolationDays(
            threshold_wh_m2=2000, min_days=1, data="filtered"
        )
        res = run_check(ins, data, filtered=filtered)
        assert res.detail["insolation_wh_m2"].iloc[0] == 8 * 700 * 0.25

    def test_dst_spring_forward_day(self):
        index = pd.date_range(
            "2025-03-09", "2025-03-09 23:45", freq="15min", tz="America/Chicago"
        )
        assert len(index) == 92
        data = pd.DataFrame({"front_col": 0.0, "poa_col": 0.0}, index=index)
        local = index.tz_localize(None)
        qualifying = (local >= "2025-03-09 01:00") & (local < "2025-03-09 04:45")
        data.loc[qualifying, :] = 700.0
        res = run_check(self.make(), data)
        assert len(res.detail) == 1
        assert res.detail["longest_run_hours"].iloc[0] == 2.75
        assert not res.passed
        data.loc[local == pd.Timestamp("2025-03-09 04:45"), :] = 700.0
        res = run_check(self.make(), data)
        assert res.detail["longest_run_hours"].iloc[0] == 3.0
        assert res.passed

    def test_duplicate_timestamp_breaks_run(self):
        data = day_frame([(40, 52, 700.0)])
        duplicated = pd.concat([data, data.iloc[[45]]]).sort_index(kind="stable")
        res = run_check(self.make(), duplicated)
        assert res.detail["longest_run_hours"].iloc[0] == 1.75
        assert not res.passed

    def test_trivial_thresholds_pass_on_day_count(self):
        data = day_frame([], days=3)
        runs = checks.ContinuousHoursDays(threshold=600, hours=0, min_days=3)
        res = run_check(runs, data)
        assert res.passed
        assert res.detail["qualifies"].all()
        ins = checks.DailyInsolationDays(threshold_wh_m2=0, min_days=3)
        res = run_check(ins, data)
        assert res.passed
        assert res.detail["qualifies"].all()

    def test_nrel_min_irr_days_cannot_exceed_insolation_days(self):
        with pytest.raises(ValueError, match="min_irr_days"):
            checks.NrelMinimumIrradiance(min_insolation_days=2, min_irr_days=3)
        checks.NrelMinimumIrradiance(min_insolation_days=3, min_irr_days=3)

    @pytest.mark.parametrize(
        "cls, kw",
        [
            (checks.ContinuousHoursDays, {"threshold": 1, "hours": 1, "min_days": 1}),
            (checks.DailyInsolationDays, {"threshold_wh_m2": 1, "min_days": 1}),
            (checks.NrelMinimumIrradiance, {}),
        ],
    )
    def test_invalid_data_mode_rejected(self, cls, kw):
        with pytest.raises(ValueError):
            cls(data="bogus", **kw)


def enough_rows(frame, minimum=1):
    """Module-level custom check (importable by qualified name)."""
    return len(frame) >= minimum


def returns_frame(frame):
    return False, frame.head(2)


class TestAnyOf:
    def tiers(self):
        return checks.AnyOf(
            checks.IntervalCount(400, 60),
            checks.IntervalCount(600, 40),
            custom_name="insolation tiers",
        )

    def tier_data(self, high, low):
        data = make_data(100, irr=450.0)
        data.iloc[:high] = 700.0
        data.iloc[high : high + low] = 450.0
        data.iloc[high + low :] = 100.0
        return FakeCapData(data, dict(ROLES))

    def test_passes_when_only_second_tier_passes(self):
        cd = self.tier_data(high=45, low=5)
        res = self.tiers().evaluate(cd, ctx(cd))
        assert res.passed
        assert res.name == "insolation tiers"
        assert list(res.detail["passed"]) == [False, True]
        assert list(res.detail["member"]) == ["IntervalCount", "IntervalCount"]

    def test_fails_when_no_tier_passes(self):
        cd = self.tier_data(high=30, low=10)
        res = self.tiers().evaluate(cd, ctx(cd))
        assert not res.passed
        assert list(res.detail["passed"]) == [False, False]

    def test_default_label(self):
        chk = checks.AnyOf(checks.Duration(threshold=1, min_hours=1))
        assert chk.label == "AnyOf"
        assert "custom_name" not in chk.to_config()

    def test_requires_members(self):
        with pytest.raises(ValueError, match="at least one"):
            checks.AnyOf()

    def test_members_must_be_checks(self):
        with pytest.raises(TypeError, match="BaseCheck"):
            checks.AnyOf("nope")

    def test_nested_round_trip(self):
        inner = checks.AnyOf(
            checks.Duration(threshold=1, min_hours=2), custom_name="inner"
        )
        outer = checks.AnyOf(inner, checks.IntervalCount(threshold=3, min_count=4))
        config = outer.to_config()
        assert config["type"] == "AnyOf"
        assert config["checks"][0]["type"] == "AnyOf"
        assert set(config) == {"type", "checks"}
        rebuilt = checks.check_from_config(config)
        assert rebuilt.to_config() == config
        assert rebuilt.checks[0].label == "inner"

    def test_empty_selection_fails_each_member(self):
        cd = FakeCapData(make_data(), dict(ROLES), filtered=make_data(0))
        chk = checks.AnyOf(
            checks.IntervalCount(threshold=1, min_count=1, data="filtered")
        )
        assert not chk.evaluate(cd, ctx(cd)).passed


class TestCustomCheck:
    def test_bool_return(self, cd):
        chk = checks.CustomCheck(enough_rows, minimum=10)
        res = chk.evaluate(cd, ctx(cd))
        assert res.passed and res.detail.empty
        assert res.name == "CustomCheck"

    def test_numpy_bool_return(self, cd):
        chk = checks.CustomCheck(lambda frame: np.bool_(False))
        assert not chk.evaluate(cd, ctx(cd)).passed

    def test_tuple_return_carries_detail(self, cd):
        res = checks.CustomCheck(returns_frame, custom_name="mine").evaluate(
            cd, ctx(cd)
        )
        assert not res.passed
        assert len(res.detail) == 2
        assert res.name == "mine"

    def test_bad_return_type(self, cd):
        chk = checks.CustomCheck(lambda frame: "yes", custom_name="mine")
        with pytest.raises(TypeError, match="mine"):
            chk.evaluate(cd, ctx(cd))

    def test_bad_tuple_frame(self, cd):
        chk = checks.CustomCheck(lambda frame: (True, "x"))
        with pytest.raises(TypeError, match="CustomCheck"):
            chk.evaluate(cd, ctx(cd))

    def test_runs_without_irradiance_column(self):
        cd = FakeCapData(pd.DataFrame({"other": [1.0, 2.0]}), {})
        context = checks.CheckContext(dt_hours=0.25)
        res = checks.CustomCheck(enough_rows, minimum=2, data="filtered").evaluate(
            cd, context
        )
        assert res.passed

    def test_empty_selection_fails(self):
        cd = FakeCapData(make_data(), dict(ROLES), filtered=make_data(0))
        res = checks.CustomCheck(enough_rows).evaluate(cd, ctx(cd))
        assert not res.passed and res.summary == checks.NO_DATA_SUMMARY

    def test_exception_propagates_with_note(self, cd):
        def boom(frame):
            raise KeyError("missing")

        chk = checks.CustomCheck(boom, custom_name="mine")
        with pytest.raises(KeyError) as info:
            chk.evaluate(cd, ctx(cd))
        if hasattr(info.value, "__notes__"):
            assert any("mine" in note for note in info.value.__notes__)

    def test_round_trip(self, cd):
        chk = checks.CustomCheck(enough_rows, minimum=5, custom_name="rows")
        config = chk.to_config()
        assert config == {
            "type": "CustomCheck",
            "func": f"{enough_rows.__module__}:enough_rows",
            "args": [],
            "kwargs": {"minimum": 5},
            "custom_name": "rows",
        }
        assert "custom_name" not in checks.CustomCheck(enough_rows).to_config()
        rebuilt = checks.check_from_config(config)
        assert rebuilt.to_config() == config
        assert rebuilt.evaluate(cd, ctx(cd)).passed

    def test_positional_args_round_trip(self):
        config = checks.CustomCheck(enough_rows, 7).to_config()
        assert config["args"] == [7]
        assert checks.check_from_config(config).args == [7]

    def test_default_data_is_filtered(self):
        assert checks.CustomCheck(enough_rows).data == "filtered"

    def test_lambda_to_config_raises(self):
        with pytest.raises(ValueError, match="lambda"):
            checks.CustomCheck(lambda frame: True).to_config()

    def test_inside_any_of(self, cd):
        chk = checks.AnyOf(
            checks.CustomCheck(enough_rows, minimum=10**6),
            checks.CustomCheck(enough_rows),
        )
        assert chk.evaluate(cd, ctx(cd)).passed
        assert checks.check_from_config(chk.to_config()).evaluate(cd, ctx(cd)).passed

    def test_numpy_scalars_in_args_encode_for_yaml(self):
        chk = checks.CustomCheck(
            enough_rows,
            np.int64(3),
            [np.float64(1.5), (np.int64(2),)],
            opts={"x": np.float64(2.5)},
        )
        config = chk.to_config()
        text = yaml.safe_dump(config)
        loaded = checks.check_from_config(yaml.safe_load(text))
        assert loaded.args == [3, [1.5, [2]]]
        assert loaded.kwargs == {"opts": {"x": 2.5}}
        assert type(config["args"][0]) is int

    def test_non_default_data_emitted_and_restored(self):
        chk = checks.CustomCheck(enough_rows, data="pr_rows")
        config = chk.to_config()
        assert config["data"] == "pr_rows"
        assert checks.check_from_config(config).data == "pr_rows"

    def test_from_config_without_args_or_kwargs(self):
        func = f"{enough_rows.__module__}:enough_rows"
        chk = checks.check_from_config({"type": "CustomCheck", "func": func})
        assert chk.args == [] and chk.kwargs == {}


class TestAnyOfUnusedParams:
    @pytest.mark.parametrize(
        "kw",
        [
            {"irr": "poa"},
            {"data": "filtered"},
            {"start": "2025-06-01", "end": "2025-06-02"},
        ],
    )
    def test_inherited_params_rejected(self, kw):
        member = checks.Duration(1, 1)
        with pytest.raises(ValueError, match="does not use"):
            checks.AnyOf(member, **kw)


class TestPositionalConstructors:
    def test_each_positional_form_matches_keywords(self):
        cases = [
            (checks.IntervalCount(400, 60), {"threshold": 400, "min_count": 60}),
            (checks.Duration(400, 30), {"threshold": 400, "min_hours": 30}),
            (
                checks.ContinuousHoursDays(600, 3, 4),
                {"threshold": 600, "hours": 3, "min_days": 4},
            ),
            (
                checks.DailyInsolationDays(3000, 5),
                {"threshold_wh_m2": 3000, "min_days": 5},
            ),
        ]
        for positional, keywords in cases:
            same = type(positional)(**keywords)
            assert positional.to_config() == same.to_config()
            rebuilt = checks.check_from_config(positional.to_config())
            assert rebuilt.to_config() == positional.to_config()

    def test_positional_with_keyword_extras(self):
        chk = checks.IntervalCount(400, 60, custom_name="tier", data="filtered")
        assert chk.label == "tier" and chk.data == "filtered"

    def test_missing_required_still_raises(self):
        with pytest.raises(ValueError):
            checks.IntervalCount(400)


class TestInsolationTolerance:
    def irregular(self, n):
        index = pd.date_range("2025-06-01 10:00", periods=n, freq="1min")
        return pd.DataFrame({"front_col": 700.0, "poa_col": 700.0}, index=index)

    def run(self, chk, n):
        data = self.irregular(n)
        cd = FakeCapData(data, dict(ROLES))
        return chk.evaluate(cd, checks.CheckContext(dt_hours=0.7, pr_rows=data.index))

    def test_daily_insolation_equal_to_threshold_through_float_product(self):
        raw = checks.daily_insolation(self.irregular(3)["front_col"], 0.7).iloc[0]
        assert raw < 1470.0
        res = self.run(checks.DailyInsolationDays(1470, 1), 3)
        assert res.passed

    def test_just_below_threshold_still_fails(self):
        assert not self.run(checks.DailyInsolationDays(1470.001, 1), 3).passed

    def test_nrel_insolation_equal_to_threshold_through_float_product(self):
        chk = checks.NrelMinimumIrradiance(
            insolation_wh_m2=1470,
            min_insolation_days=1,
            min_irr_days=1,
            irr_threshold=600,
            hours=0,
        )
        res = self.run(chk, 3)
        assert res.passed and res.detail["insolation_ok"].all()

    def test_nrel_exact_length_run_on_inexact_dt(self):
        index = pd.date_range("2025-06-01 08:00", periods=3, freq="18min")
        data = pd.DataFrame({"front_col": 700.0, "poa_col": 700.0}, index=index)
        cd = FakeCapData(data, dict(ROLES))
        chk = checks.NrelMinimumIrradiance(
            insolation_wh_m2=0, hours=0.9, min_insolation_days=1, min_irr_days=1
        )
        res = chk.evaluate(cd, checks.CheckContext(dt_hours=0.3, pr_rows=data.index))
        assert bool(res.detail["irradiance_ok"].iloc[0])
        assert res.passed


class TestFinalReviewFixes:
    def test_to_config_validates_after_post_construction_edit(self):
        dur = checks.Duration(400, 5)
        dur.min_hours = -1
        with pytest.raises(ValueError, match="min_hours"):
            dur.to_config()
        half = checks.Duration(400, 5, data="unfiltered_period")
        half.start = "2025-06-01"
        with pytest.raises(ValueError, match="start and end"):
            half.to_config()
        any_of = checks.AnyOf(checks.Duration(400, 5))
        any_of.irr = "poa"
        with pytest.raises(ValueError, match="does not use"):
            any_of.to_config()
        custom = checks.CustomCheck(enough_rows)
        custom.data = "unfiltered_period"
        custom.start = "2025-06-01"
        with pytest.raises(ValueError, match="start and end"):
            custom.to_config()

    @pytest.mark.parametrize("data", ["pr_rows", "filtered"])
    def test_period_requires_unfiltered_period_mode(self, data):
        with pytest.raises(ValueError, match="unfiltered_period"):
            checks.Duration(400, 5, data=data, start="2025-06-01", end="2025-06-02")

    def test_custom_check_period(self, cd):
        chk = checks.CustomCheck(
            len_is, 24, data="unfiltered_period", start="2025-06-01", end="2025-06-01"
        )
        assert chk.evaluate(cd, ctx(cd)).passed is False
        config = chk.to_config()
        assert config["start"] == "2025-06-01" and config["end"] == "2025-06-01"
        rebuilt = checks.check_from_config(config)
        assert (rebuilt.start, rebuilt.end) == ("2025-06-01", "2025-06-01")
        assert rebuilt.kwargs == {}

    def test_custom_check_start_not_forwarded_to_func(self, cd):
        chk = checks.CustomCheck(
            enough_rows, data="unfiltered_period", start="2025-06-01", end="2025-06-01"
        )
        assert chk.kwargs == {}

    def test_custom_check_period_needs_mode(self):
        with pytest.raises(ValueError, match="unfiltered_period"):
            checks.CustomCheck(enough_rows, start="2025-06-01", end="2025-06-01")

    def test_duration_hours_tolerance(self):
        index = pd.date_range("2025-06-01", periods=444, freq="30s")
        data = pd.DataFrame({"poa_col": 500.0, "front_col": 500.0}, index=index)
        cd = FakeCapData(data, dict(ROLES))
        dt = 30 / 3600
        assert 444 * dt < 3.7
        res = checks.Duration(400, 3.7).evaluate(
            cd, checks.CheckContext(dt_hours=dt, pr_rows=index)
        )
        assert res.passed

    def test_period_in_other_timezone_is_converted(self):
        index = pd.date_range(
            "2025-06-01", periods=96 * 3, freq="15min", tz="America/Chicago"
        )
        data = pd.DataFrame({"poa_col": 500.0, "front_col": 500.0}, index=index)
        cd = FakeCapData(data, dict(ROLES))
        tokyo = "2025-06-02T00:00:00+09:00"  # 2025-06-01 10:00 in Chicago
        chk = checks.ContinuousHoursDays(400, 1, 1, start=tokyo, end=tokyo)
        sel = checks.select_rows(cd, chk, checks.CheckContext(0.25))
        assert len(sel) == 96
        assert sel.index.min() == pd.Timestamp("2025-06-01", tz="America/Chicago")

    def test_aware_period_on_naive_index_raises(self, cd):
        chk = checks.ContinuousHoursDays(
            400,
            1,
            1,
            start="2025-06-01T00:00:00+00:00",
            end="2025-06-01T00:00:00+00:00",
        )
        with pytest.raises(ValueError, match="naive"):
            checks.select_rows(cd, chk, checks.CheckContext(0.25))

    def test_unprocessed_regression_cols_raise_value_error(self, cd):
        cd.regression_cols = {"poa": {"group": "irr_poa", "agg": "mean"}}
        with pytest.raises(ValueError, match="process_regression_columns"):
            checks.IntervalCount(400, 1).evaluate(cd, ctx(cd))

    def test_bad_period_strings_and_order(self):
        with pytest.raises(ValueError, match="comparable timestamps"):
            checks.Duration(
                1, 1, data="unfiltered_period", start="not a date", end="2025-06-01"
            )
        with pytest.raises(ValueError, match="after end"):
            checks.Duration(
                1, 1, data="unfiltered_period", start="2025-06-02", end="2025-06-01"
            )
        with pytest.raises(ValueError, match="comparable timestamps"):
            checks.Duration(
                1,
                1,
                data="unfiltered_period",
                start="2025-06-01T00:00:00+00:00",
                end="2025-06-02",
            )

    def test_base_from_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="did you mean 'min_hours'"):
            checks.check_from_config(
                {"type": "Duration", "threshold": 400, "min_houar": 5}
            )

    def test_from_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="did you mean 'checks'"):
            checks.AnyOf.from_config({"type": "AnyOf", "chekcs": []})
        with pytest.raises(ValueError, match="'checks'"):
            checks.AnyOf.from_config({"type": "AnyOf"})
        with pytest.raises(ValueError, match="did you mean 'kwargs'"):
            checks.check_from_config(
                {
                    "type": "CustomCheck",
                    "func": "tests.test_checks:enough_rows",
                    "kwarg": {},
                }
            )


def len_is(frame, n):
    return len(frame) == n
