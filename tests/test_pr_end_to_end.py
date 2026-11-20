"""End-to-end PR runs on synthetic data shaped like two contract test types.

One run is a substantial-completion style test (bifacial, temperature
corrected, reference derived from the modeled side); the other a contract
monthly test (fixed monthly reference, per-interval availability, degradation).
Every expected value is computed here from the synthetic definitions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from captest import prtest
from captest.captest import CapTest
from captest.io import loader_id
from tests.pr_fixtures import DC_NAMEPLATE, make_meas_pr, make_sim_pr

COEFF = -0.35
BIFACIALITY = 0.7


def keep_available(frame, column, minimum):
    """Row filter keeping rows whose ``column`` is at least ``minimum``."""
    return frame[frame[column] >= minimum]


def _plain_pr(frame, irradiance):
    """Plain PR of hourly modeled rows: sum(E) / sum(nameplate * G / 1000)."""
    return frame["E_Grid"].sum() / (DC_NAMEPLATE * irradiance.sum() / 1000.0)


def _total_irradiance(sim_frame):
    rear = sim_frame["GlobBak"] + sim_frame["BackShd"]
    return sim_frame["GlobInc"] + BIFACIALITY * rear


class TestHaynevilleShaped:
    DESIGN_PR = 0.80

    def test_hayneville_shaped_run(self, tmp_path):
        meas_index = pd.date_range("2024-06-10", "2024-06-14 23:45", freq="15min")
        sim_index = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
        pr_by_month = dict(zip(range(1, 13), np.linspace(0.76, 0.82, 12)))
        availability = pd.Series(1.0, index=meas_index)
        availability[(meas_index.day == 11) & (meas_index.hour == 12)] = 0.0

        @loader_id("test.pr_e2e_hayneville_meas")
        def meas_loader(path, **kwargs):
            return make_meas_pr(
                meas_index,
                dc_nameplate=DC_NAMEPLATE,
                pr=self.DESIGN_PR,
                t_ref=45.0,
                coeff=COEFF,
                bifacial=True,
                availability=availability,
                bifaciality=BIFACIALITY,
            )

        @loader_id("test.pr_e2e_hayneville_sim")
        def sim_loader(path, **kwargs):
            return make_sim_pr(
                sim_index,
                pr_by_month=pr_by_month,
                bifacial=True,
                bifaciality=BIFACIALITY,
                t_array=45.0,
            )

        tst = CapTest.from_params(
            test_setup="pr_tc_meas_bom_bifi",
            meas_path="meas.csv",
            meas_loader=meas_loader,
            sim_path="sim.csv",
            sim_loader=sim_loader,
            dc_nameplate=DC_NAMEPLATE,
            power_temp_coeff=COEFF,
            bifaciality=BIFACIALITY,
            test_tolerance="- 5",
            verbose=False,
        )

        tst.sim.filter_time(days=30, test_date="2024-06-12")
        tst.sim.filter_irr(200, 2000, col_name="GlobInc")
        rows_before_clipping = len(tst.sim.data_filtered)
        clip_at = 0.97 * tst.sim.data_filtered["E_Grid"].max()
        tst.sim.filter_power(clip_at)
        front = tst.meas.regression_cols["front_poa"]
        tst.meas.filter_irr(300, 1200, col_name=front)
        tst.meas.filter_custom(keep_available, "avail", 1.0)

        kept = tst.meas.data_filtered
        n_high = int((kept[front] >= 900).sum())
        n_low = int((kept[front] >= 400).sum())
        tst.meas.check_any_of(
            tst.meas.check_interval_count(900, n_high + 1, custom_name="high tier"),
            tst.meas.check_interval_count(400, n_low, custom_name="low tier"),
        )

        results = tst.run_test()

        sim_kept = tst.sim.data_filtered
        assert len(sim_kept) < rows_before_clipping
        assert sim_kept["E_Grid"].max() < clip_at
        assert sim_kept["GlobInc"].between(200, 2000).all()
        assert set(sim_kept.index.month) == {5, 6}
        expected_pr = _plain_pr(sim_kept, _total_irradiance(sim_kept))
        assert results.expected_ratio == pytest.approx(expected_pr, rel=1e-9)
        assert expected_pr != pytest.approx(pr_by_month[6], abs=1e-4)
        assert results.t_ref == pytest.approx(45.0)
        assert results.ratio == pytest.approx(self.DESIGN_PR, rel=1e-9)
        assert results.points_used["meas"] == len(kept)
        assert (kept["avail"] == 1.0).all()
        assert results.threshold == pytest.approx(0.95 * expected_pr, rel=1e-9)
        assert results.passed is True

        tiers = results.checks["meas: AnyOf"]
        assert tiers.passed
        assert tiers.detail["passed"].tolist() == [False, True]
        assert results.valid is True

        path = tmp_path / "hayneville.yaml"
        with pytest.warns(UserWarning, match="programmatic-only"):
            tst.to_yaml(path)
        reloaded = CapTest.from_yaml(
            path, meas_loader=meas_loader, sim_loader=sim_loader
        )
        again = reloaded.run_test()
        assert again.ratio == pytest.approx(results.ratio, rel=1e-12)
        assert again.expected_ratio == pytest.approx(results.expected_ratio, rel=1e-12)
        assert again.t_ref == pytest.approx(results.t_ref)
        assert again.passed is True
        assert again.points_used == results.points_used
        assert again.checks["meas: AnyOf"].detail["passed"].tolist() == [False, True]
        assert reloaded.run_fingerprint == tst.run_fingerprint


class TestP3bShaped:
    DESIGN_PR = 0.81
    JAN, FEB = 38.0, 41.0
    EPR_MONTHLY = [
        0.82,
        0.84,
        0.85,
        0.86,
        0.86,
        0.85,
        0.84,
        0.84,
        0.85,
        0.86,
        0.84,
        0.82,
    ]
    T_REF_MONTHLY = [
        38.0,
        41.0,
        44.0,
        47.0,
        50.0,
        53.0,
        55.0,
        54.0,
        50.0,
        46.0,
        42.0,
        39.0,
    ]

    def test_p3b_shaped_run(self):
        index = pd.date_range("2025-01-29", "2025-02-03 23:45", freq="15min")
        availability = pd.Series(1.0, index=index)
        availability[(index.day == 30) & (index.hour >= 10) & (index.hour < 14)] = 0.9
        availability[(index.day == 2) & (index.hour >= 8) & (index.hour < 12)] = 0.5
        meas = make_meas_pr(
            index,
            dc_nameplate=DC_NAMEPLATE,
            pr=self.DESIGN_PR,
            t_ref={1: self.JAN, 2: self.FEB},
            coeff=COEFF,
            availability=availability,
            temp_delta=0.5,
        )
        tst = CapTest(
            meas=meas,
            test_setup="pr_tc_meas_bom",
            reg_cols_meas={
                "cell_temp": {
                    "calc": "cell_temp",
                    "args": {
                        "bom": {"group": "temp_bom", "agg": "mean"},
                        "poa": {"group": "irr_poa", "agg": "mean"},
                        "temp_delta": 0.5,
                    },
                },
                "availability": {"column": "avail"},
            },
            dc_nameplate=DC_NAMEPLATE,
            power_temp_coeff=COEFF,
            reference_source="fixed",
            reference_granularity="monthly",
            expected_pr_monthly=self.EPR_MONTHLY,
            t_ref_monthly=self.T_REF_MONTHLY,
            degradation_rate=0.4,
            degradation_years=1.5,
            test_tolerance="- 3",
        )
        tst.setup(verbose=False)
        tst.meas.check_duration(400, 30)

        results = tst.run_test()

        data = meas.data
        dt = 0.25
        poa = data["poa_1"]
        cell = data["bom_1"] + poa / 1000.0 * 0.5
        t_ref = pd.Series(np.where(index.month == 1, self.JAN, self.FEB), index=index)
        expected_full = (
            DC_NAMEPLATE * poa / 1000.0 * dt * (1 + COEFF / 100.0 * (cell - t_ref))
        )
        expected = expected_full * data["avail"]
        energy = data["meter_w"] * dt

        assert energy.sum() / expected.sum() == pytest.approx(self.DESIGN_PR, rel=1e-9)
        assert results.ratio == pytest.approx(self.DESIGN_PR, rel=1e-9)
        assert results.ratio_without_availability == pytest.approx(
            energy.sum() / expected_full.sum(), rel=1e-9
        )
        assert results.ratio_without_availability < results.ratio
        intervals = results.intervals
        assert (intervals.loc[intervals.index.month == 1, "t_ref"] == self.JAN).all()
        assert (intervals.loc[intervals.index.month == 2, "t_ref"] == self.FEB).all()
        assert (intervals["availability"] == data["avail"]).all()

        h_jan = poa[index.month == 1].sum() * dt
        h_feb = poa[index.month == 2].sum() * dt
        undegraded = (h_jan * 0.82 + h_feb * 0.84) / (h_jan + h_feb)
        gpr = undegraded * 0.994
        assert results.degradation["factor"] == pytest.approx(0.994)
        assert results.expected_ratio_undegraded == pytest.approx(undegraded, rel=1e-9)
        assert results.expected_ratio == pytest.approx(gpr, rel=1e-9)
        assert results.threshold == pytest.approx(0.97 * gpr, rel=1e-9)
        assert results.passed is True
        assert results.passed == (0.97 * gpr <= self.DESIGN_PR)
        assert results.t_ref is None
        assert results.monthly["insolation_wh_m2"].tolist() == pytest.approx(
            [h_jan, h_feb]
        )
        assert results.monthly["expected_pr"].tolist() == [0.82, 0.84]
        assert results.monthly["t_ref"].tolist() == [self.JAN, self.FEB]

        hours = int((poa >= 400).sum()) * dt
        duration = results.checks["meas: Duration"]
        assert hours > 30
        assert duration.passed
        assert duration.detail["hours"].iloc[0] == pytest.approx(hours)
        assert results.valid is True


def _month_pair_inputs():
    sim_index = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h")
    meas_index = pd.date_range("2025-01-29", "2025-02-03 23:45", freq="15min")
    return meas_index, sim_index


def _derived_pair(meas, sim, *, granularity):
    tst = CapTest(
        meas=meas,
        sim=sim,
        test_setup="pr_tc_meas_bom",
        dc_nameplate=DC_NAMEPLATE,
        power_temp_coeff=COEFF,
        reference_source="derived",
        reference_granularity=granularity,
        auto_wrap_sim=False,
        test_tolerance="- 5",
    )
    tst.setup(verbose=False)
    return tst


class TestDerivedAcrossMonths:
    DESIGN_PR = 0.82
    T_ARRAY = {1: 30.0, 2: 40.0}

    def test_single_pair_across_month_boundary(self):
        meas_index = pd.date_range("2025-01-31", "2025-02-01 23:45", freq="15min")
        sim_index = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h")
        sim = make_sim_pr(
            sim_index, pr_by_month={1: 0.78, 2: 0.80}, t_array=self.T_ARRAY
        )
        frame = sim.data
        weighted_t = (frame["GlobInc"] * frame["TArray"]).sum() / frame["GlobInc"].sum()
        meas = make_meas_pr(
            meas_index,
            dc_nameplate=DC_NAMEPLATE,
            pr=self.DESIGN_PR,
            t_ref=weighted_t,
            coeff=COEFF,
        )
        tst = _derived_pair(meas, sim, granularity="single")

        results = tst.run_test()

        assert 30.0 < weighted_t < 40.0
        assert results.granularity == "single"
        assert results.monthly is None
        assert results.t_ref == pytest.approx(weighted_t, rel=1e-12)
        assert set(results.intervals.index.month) == {1, 2}
        assert results.intervals["t_ref"].nunique() == 1
        assert results.intervals["t_ref"].to_numpy() == pytest.approx(weighted_t)
        assert results.expected_ratio == pytest.approx(
            _plain_pr(frame, frame["GlobInc"]), rel=1e-9
        )
        assert results.ratio == pytest.approx(self.DESIGN_PR, rel=1e-9)

    def _monthly_pair(self):
        meas_index, sim_index = _month_pair_inputs()
        sim = make_sim_pr(
            sim_index, pr_by_month={1: 0.78, 2: 0.84}, t_array=self.T_ARRAY
        )
        sim.data.loc["2025-01-01":"2025-01-10", "E_Grid"] *= 0.9
        meas = make_meas_pr(
            meas_index,
            dc_nameplate=DC_NAMEPLATE,
            pr=self.DESIGN_PR,
            t_ref=self.T_ARRAY,
            coeff=COEFF,
        )
        return meas, _derived_pair(meas, sim, granularity="monthly")

    def test_derived_monthly_equals_per_month_plain_pr(self):
        meas, tst = self._monthly_pair()
        tst.sim.filter_irr(200, 2000, col_name="GlobInc")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = tst.run_test()

        kept = tst.sim.data_filtered
        by_month = dict(list(kept.groupby(kept.index.month)))
        assert sorted(by_month) == [1, 2]
        for month, rows in by_month.items():
            assert results.monthly.loc[month, "expected_pr"] == pytest.approx(
                _plain_pr(rows, rows["GlobInc"]), rel=1e-9
            )
            assert results.monthly.loc[month, "t_ref"] == pytest.approx(
                self.T_ARRAY[month]
            )
        assert results.monthly.loc[1, "expected_pr"] < 0.78
        assert results.monthly.loc[2, "expected_pr"] == pytest.approx(0.84)
        poa = meas.data["poa_1"]
        h = [poa[poa.index.month == m].sum() * 0.25 for m in (1, 2)]
        weighted = (
            h[0] * results.monthly.loc[1, "expected_pr"]
            + h[1] * results.monthly.loc[2, "expected_pr"]
        ) / sum(h)
        assert results.expected_ratio == pytest.approx(weighted, rel=1e-9)
        assert results.t_ref is None
        assert results.ratio == pytest.approx(self.DESIGN_PR, rel=1e-9)

    def test_derived_monthly_warns_on_half_a_month(self):
        _, tst = self._monthly_pair()
        tst.sim.filter_time(start="2025-01-16", end="2025-01-31 23:59", drop=True)

        with pytest.warns(UserWarning, match="partly cover January"):
            results = tst.run_test()

        kept = tst.sim.data_filtered
        january = kept[kept.index.month == 1]
        assert january.index.max() < pd.Timestamp("2025-01-16")
        assert results.monthly.loc[1, "expected_pr"] == pytest.approx(
            _plain_pr(january, january["GlobInc"]), rel=1e-9
        )
        assert isinstance(results, prtest.PrTestResults)
