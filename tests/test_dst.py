"""Naive local timestamps across DST, and the sim year wrap they run into."""

import numpy as np
import pandas as pd
import pytest

from captest import CapTest, calcparams, util
from captest.capdata import CapData
from captest.clearsky import get_tz_index, localize_index
from captest.filters import Time

TZ = "America/Chicago"
CDT = pd.Timedelta(hours=-5)
CST = pd.Timedelta(hours=-6)


def fall_back_day(repeat_hour=False):
    """Naive 1-min Nov 1 2026, with or without the repeated 01:00 hour."""
    day = pd.date_range("2026-11-01 00:00", "2026-11-01 23:59", freq="1min")
    if not repeat_hour:
        return day
    first = day[day < "2026-11-01 02:00"]
    second = day[(day >= "2026-11-01 01:00") & (day < "2026-11-01 02:00")]
    return first.append(second).append(day[day >= "2026-11-01 02:00"])


class TestLocalizeIndex:
    def test_infers_both_offsets_when_hour_repeats(self):
        localized = localize_index(fall_back_day(repeat_hour=True), TZ)
        offsets = pd.Series(localized).map(lambda ts: ts.utcoffset())
        assert (offsets.iloc[60:120] == CDT).all()
        assert (offsets.iloc[120:180] == CST).all()

    def test_repeated_hour_stays_unique(self):
        localized = localize_index(fall_back_day(repeat_hour=True), TZ)
        assert localized.is_unique

    def test_lone_hour_is_assumed_dst(self):
        localized = localize_index(fall_back_day(), TZ)
        offsets = pd.Series(localized).map(lambda ts: ts.utcoffset())
        assert len(localized) == 1440
        assert (offsets.iloc[60:120] == CDT).all()
        assert (offsets.iloc[120:] == CST).all()

    def test_spring_gap_defaults_to_nat(self):
        naive = pd.DatetimeIndex(["2026-03-08 01:59", "2026-03-08 02:30"])
        localized = localize_index(naive, TZ)
        assert localized.isna().tolist() == [False, True]

    def test_spring_gap_shifts_forward_on_request(self):
        naive = pd.DatetimeIndex(["2026-03-08 01:59", "2026-03-08 02:30"])
        localized = localize_index(naive, TZ, nonexistent="shift_forward")
        assert not localized.isna().any()
        assert localized[1].hour == 3

    def test_get_tz_index_handles_lone_hour(self):
        times = get_tz_index(fall_back_day(), {"tz": TZ})
        assert times.tz is not None
        assert len(times) == 1440


class TestApparentZenithFallBack:
    def test_lone_ambiguous_hour_does_not_raise(self):
        site = {"loc": {"latitude": 32.3, "longitude": -86.4, "tz": TZ}}
        df = pd.DataFrame({"irr": 0.0}, index=fall_back_day())
        zenith = calcparams.apparent_zenith(df, site=site, verbose=False)
        assert zenith.index.equals(df.index)
        assert zenith.notna().any()


class TestReindexDatetimeRepeatedHour:
    def test_first_occurrence_survives(self):
        index = fall_back_day(repeat_hour=True)
        passes = np.zeros(len(index))
        passes[120:180] = 1
        df = pd.DataFrame({"pass": passes}, index=index)
        with pytest.warns(UserWarning, match="Dropping duplicate indices"):
            reindexed, _, _ = util.reindex_datetime(df)
        assert len(reindexed) == 1440
        assert (reindexed["pass"] == 0).all()


def wrapped_sim():
    """A typical-year sim wrapped by a measured window reaching Nov 1."""
    tst = CapTest()
    meas_index = pd.date_range("2026-10-29", "2026-11-01 23:59", freq="1min")
    sim_index = pd.date_range("1990-01-01", periods=8760, freq="h")
    tst.meas = CapData("meas")
    tst.meas.data = pd.DataFrame({"poa": 0.0}, index=meas_index)
    tst.sim = CapData("sim")
    tst.sim.data = pd.DataFrame({"poa": 1.0}, index=sim_index)
    tst._maybe_wrap_sim_year_end()
    assert tst.sim.data.index[0] == pd.Timestamp("1989-07-01")
    return tst.sim


class TestTimeOnWrappedSim:
    def test_pinned_test_date_moves_into_wrapped_year(self):
        kept = Time(test_date="1990-10-22", days=30)._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1989-10-07")
        assert kept[-1] == pd.Timestamp("1989-11-06")

    def test_pinned_test_date_in_early_wrapped_year_moves_forward(self):
        kept = Time(test_date="1989-02-15", days=30)._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1990-01-31")
        assert kept[-1] == pd.Timestamp("1990-03-02")

    def test_pinned_start_end_move_into_wrapped_year(self):
        kept = Time(start="1990-10-07", end="1990-11-06")._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1989-10-07")
        assert kept[-1] == pd.Timestamp("1989-11-06")

    def test_window_across_new_year_stays_contiguous(self):
        kept = Time(test_date="1990-12-25", days=30)._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1989-12-10")
        assert kept[-1] == pd.Timestamp("1990-01-09")

    def test_date_already_in_wrapped_range_is_unchanged(self):
        kept = Time(test_date="1990-03-15", days=30)._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1990-02-28")

    def test_stored_config_is_not_rewritten(self):
        step = Time(test_date="1990-10-22", days=30)
        step._execute(wrapped_sim())
        assert step.to_config()["test_date"] == "1990-10-22"

    def test_unwrapped_data_keeps_out_of_range_dates_empty(self):
        cd = CapData("meas")
        index = pd.date_range("2026-10-01", periods=48, freq="h")
        cd.data = pd.DataFrame({"poa": 0.0}, index=index)
        assert len(Time(test_date="2025-10-01", days=2)._execute(cd)) == 0

    def test_window_straddling_wrapped_end_keeps_its_overlap(self):
        kept = Time(start="1990-06-20", end="1990-07-10")._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1990-06-20")
        assert kept[-1] == pd.Timestamp("1990-06-30 23:00")

    def test_start_alone_moves_and_runs_to_end_of_data(self):
        kept = Time(start="1990-10-07")._execute(wrapped_sim())
        assert kept[0] == pd.Timestamp("1989-10-07")
        assert kept[-1] == pd.Timestamp("1990-06-30 23:00")
