import numpy as np
import pandas as pd
import pytest

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
