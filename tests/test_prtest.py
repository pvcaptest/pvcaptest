import warnings

import numpy as np
import pandas as pd
import pytest

from captest import prtest
from captest.calcparams import register_to_interval_energy


class TestUnits:
    def test_constants(self):
        assert prtest.G_STC == 1000.0
        assert prtest.POWER_UNITS == {"W": 1.0, "kW": 1e3, "MW": 1e6}
        assert prtest.ENERGY_UNITS == {"Wh": 1.0, "kWh": 1e3, "MWh": 1e6}

    def test_dt_hours(self):
        index = pd.date_range("2024-01-01", periods=5, freq="15min")
        assert prtest.dt_hours(index) == 0.25

    def test_power_to_energy_wh(self):
        out = prtest.to_energy_wh(
            pd.Series([0.5]), kind="power", units="MW", dt_hours=0.25
        )
        assert out.tolist() == [125000.0]

    def test_energy_to_wh(self):
        out = prtest.to_energy_wh(
            pd.Series([125.0]), kind="energy", units="kWh", dt_hours=0.25
        )
        assert out.tolist() == [125000.0]

    @pytest.mark.parametrize(
        "kind, units",
        [("power", "kWh"), ("energy", "kW"), ("power", "GW"), ("energy", "J")],
    )
    def test_bad_units_raise_listing_allowed(self, kind, units):
        with pytest.raises(ValueError, match="W, kW, MW|Wh, kWh, MWh"):
            prtest.to_energy_wh(pd.Series([1.0]), kind=kind, units=units, dt_hours=1)

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            prtest.to_energy_wh(
                pd.Series([1.0]), kind="other", units="kW", dt_hours=1.0
            )

    def test_power_from_energy(self):
        out = prtest.power_from_energy(
            pd.Series([25.0]),
            energy_units="kWh",
            power_units="MW",
            dt_hours=0.25,
        )
        assert out.tolist() == pytest.approx([0.1])

    def test_power_from_energy_bad_units(self):
        with pytest.raises(ValueError, match="Wh, kWh, MWh"):
            prtest.power_from_energy(
                pd.Series([1.0]), energy_units="kW", power_units="MW", dt_hours=1.0
            )
        with pytest.raises(ValueError, match="W, kW, MW"):
            prtest.power_from_energy(
                pd.Series([1.0]), energy_units="kWh", power_units="MWh", dt_hours=1.0
            )


class TestExpectedEnergy:
    def test_plain(self):
        out = prtest.expected_energy_wh(
            pd.Series([1000.0, 500.0]), dc_nameplate=1e6, dt_hours=0.25
        )
        assert out.tolist() == pytest.approx([250000.0, 125000.0])

    def test_corrected_multiplies(self):
        out = prtest.expected_energy_wh(
            pd.Series([1000.0, 500.0]),
            dc_nameplate=1e6,
            dt_hours=0.25,
            cell_temp=pd.Series([45.0, 25.0]),
            t_ref=25.0,
            power_temp_coeff=-0.4,
        )
        assert out.tolist() == pytest.approx([230000.0, 125000.0])

    def test_t_ref_series_applied_per_row(self):
        out = prtest.expected_energy_wh(
            pd.Series([1000.0, 1000.0]),
            dc_nameplate=1e6,
            dt_hours=0.25,
            cell_temp=pd.Series([45.0, 45.0]),
            t_ref=pd.Series([25.0, 45.0]),
            power_temp_coeff=-0.4,
        )
        assert out.tolist() == pytest.approx([230000.0, 250000.0])

    def test_availability_scalar_and_series(self):
        poa = pd.Series([1000.0, 1000.0])
        half = prtest.expected_energy_wh(
            poa, dc_nameplate=1e6, dt_hours=0.25, availability=0.5
        )
        assert half.tolist() == pytest.approx([125000.0, 125000.0])
        per_row = prtest.expected_energy_wh(
            poa,
            dc_nameplate=1e6,
            dt_hours=0.25,
            availability=pd.Series([1.0, 0.5]),
        )
        assert per_row.tolist() == pytest.approx([250000.0, 125000.0])

    def test_corrected_requires_t_ref_and_coeff(self):
        kwargs = {
            "poa": pd.Series([1000.0]),
            "dc_nameplate": 1e6,
            "dt_hours": 0.25,
            "cell_temp": pd.Series([45.0]),
        }
        with pytest.raises(ValueError, match="t_ref"):
            prtest.expected_energy_wh(**kwargs, power_temp_coeff=-0.4)
        with pytest.raises(ValueError, match="power_temp_coeff"):
            prtest.expected_energy_wh(**kwargs, t_ref=25.0)

    def test_t_ref_without_cell_temp_raises(self):
        with pytest.raises(ValueError, match="cell_temp"):
            prtest.expected_energy_wh(
                pd.Series([1000.0]), dc_nameplate=1e6, dt_hours=0.25, t_ref=25.0
            )

    def test_matches_pvwatts_dc(self):
        pvlib = pytest.importorskip("pvlib")
        rng = np.random.default_rng(0)
        g = pd.Series(rng.uniform(0, 1100, 50))
        t = pd.Series(rng.uniform(5, 65, 50))
        out = prtest.expected_energy_wh(
            g,
            dc_nameplate=1e6,
            dt_hours=0.25,
            cell_temp=t,
            t_ref=31.7,
            power_temp_coeff=-0.35,
        )
        expected = pvlib.pvsystem.pvwatts_dc(g, t, 1e6, -0.0035, 31.7) * 0.25
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_nameplate_and_timestep_validated(self):
        poa = pd.Series([1000.0])
        for bad in (0, -1e6, np.nan):
            with pytest.raises(ValueError, match="dc_nameplate"):
                prtest.expected_energy_wh(poa, dc_nameplate=bad, dt_hours=0.25)
        for bad in (0, -0.25, np.nan):
            with pytest.raises(ValueError, match="dt_hours"):
                prtest.expected_energy_wh(poa, dc_nameplate=1e6, dt_hours=bad)
            with pytest.raises(ValueError, match="dt_hours"):
                prtest.to_energy_wh(poa, kind="power", units="kW", dt_hours=bad)
            with pytest.raises(ValueError, match="dt_hours"):
                prtest.power_from_energy(
                    poa, energy_units="kWh", power_units="kW", dt_hours=bad
                )


class TestValidators:
    def test_positive_coeff_raises_with_nrel_message(self):
        with pytest.raises(ValueError) as exc:
            prtest.validate_power_temp_coeff(0.32)
        message = str(exc.value)
        assert "power_temp_coeff" in message
        assert "eq. 8" in message
        assert "-0.32" in message

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_coeff_raises(self, bad):
        with pytest.raises(ValueError, match="power_temp_coeff"):
            prtest.validate_power_temp_coeff(bad)

    @pytest.mark.parametrize("coeff", [-0.0032, -1.5])
    def test_coeff_magnitude_warns(self, coeff):
        with pytest.warns(UserWarning, match="%/°C"):
            prtest.validate_power_temp_coeff(coeff)

    def test_typical_coeff_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prtest.validate_power_temp_coeff(-0.32)

    @pytest.mark.parametrize(
        "bad",
        [95, np.nan, np.inf, pd.Series([1.0, 1.2]), pd.Series([0.5, -0.1])],
    )
    def test_availability_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="0 ≤ A ≤ 1"):
            prtest.validate_availability(bad)

    def test_availability_valid(self):
        prtest.validate_availability(1.0)
        prtest.validate_availability(0.0)
        prtest.validate_availability(pd.Series([0.2, 1.0, np.nan]))


class TestRowSet:
    def test_drops_counted_and_warned(self):
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, np.inf, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        with pytest.warns(UserWarning, match=r"meas: dropped 1 .*NaN.* 1 .*inf"):
            rows = prtest.complete_rows(frame, label="meas")
        assert list(rows.index) == [0, 3, 4]
        assert rows.dropped_nan == 1
        assert rows.dropped_nonfinite == 1

    def test_nan_and_inf_in_one_row_counts_as_nan(self):
        frame = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.inf, 1.0]})
        with pytest.warns(UserWarning):
            rows = prtest.complete_rows(frame)
        assert (rows.dropped_nan, rows.dropped_nonfinite) == (1, 0)

    def test_no_warning_when_nothing_dropped(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rows = prtest.complete_rows(pd.DataFrame({"a": [1.0, 2.0]}))
        assert list(rows.index) == [0, 1]

    def test_inf_row_excluded_and_counted(self):
        frame = pd.DataFrame(
            {
                "meas": [200000.0, np.inf, 100000.0],
                "exp": [250000.0, 250000.0, 125000.0],
            }
        )
        with pytest.warns(UserWarning):
            rows = prtest.complete_rows(frame)
        kept = frame.loc[rows.index]
        ratio, _ = prtest.pr_ratio(kept["meas"], kept["exp"])
        assert ratio == pytest.approx(0.8)
        assert rows.dropped_nonfinite == 1


class TestRatio:
    def test_ratio_and_interval(self):
        ratio, interval = prtest.pr_ratio(
            pd.Series([200000.0, 100000.0]), pd.Series([250000.0, 125000.0])
        )
        assert ratio == pytest.approx(0.8)
        assert interval.tolist() == pytest.approx([0.8, 0.8])

    def test_zero_expected_row_is_nan_and_total_unchanged(self):
        ratio, interval = prtest.pr_ratio(
            pd.Series([200000.0, 100000.0, 0.0]),
            pd.Series([250000.0, 125000.0, 0.0]),
        )
        assert ratio == pytest.approx(0.8)
        assert np.isnan(interval.iloc[2])

    def test_total_expected_zero_raises(self):
        with pytest.raises(ValueError, match="total expected energy is zero"):
            prtest.pr_ratio(pd.Series([1.0]), pd.Series([0.0]))


class TestWeightedCellTemp:
    def test_weighted(self):
        out = prtest.weighted_cell_temp(
            pd.Series([1000.0, 500.0]), pd.Series([40.0, 25.0])
        )
        assert out == pytest.approx(35.0)

    def test_zero_poa_raises(self):
        with pytest.raises(ValueError, match="irradiance"):
            prtest.weighted_cell_temp(pd.Series([0.0, 0.0]), pd.Series([40.0, 25.0]))

    @pytest.mark.parametrize("bifacial", [False, True])
    def test_plain_equals_corrected_at_weighted_tref(self, bifacial):
        rng = np.random.default_rng(7)
        n = 200
        front = pd.Series(rng.uniform(50, 1100, n))
        rear = pd.Series(rng.uniform(5, 150, n))
        poa = front + 0.15 * rear if bifacial else front
        temp = pd.Series(rng.uniform(5, 70, n))
        measured = pd.Series(rng.uniform(1e4, 2e5, n))
        kwargs = {"dc_nameplate": 1e6, "dt_hours": 0.25}
        plain, _ = prtest.pr_ratio(measured, prtest.expected_energy_wh(poa, **kwargs))
        corrected, _ = prtest.pr_ratio(
            measured,
            prtest.expected_energy_wh(
                poa,
                cell_temp=temp,
                t_ref=prtest.weighted_cell_temp(poa, temp),
                power_temp_coeff=-0.35,
                **kwargs,
            ),
        )
        assert plain == pytest.approx(corrected, rel=1e-12)

    def test_pvanalytics_plain_cross_check(self):
        metrics = pytest.importorskip("pvanalytics.metrics")
        rng = np.random.default_rng(3)
        index = pd.date_range("2024-06-01", periods=96, freq="15min")
        poa = pd.Series(rng.uniform(100, 1000, 96), index=index)
        temp_air = pd.Series(rng.uniform(10, 30, 96), index=index)
        wind = pd.Series(rng.uniform(0, 5, 96), index=index)
        pdc0 = 1e6
        pac = pd.Series(rng.uniform(0.7, 0.9, 96), index=index) * pdc0 * poa / 1000
        expected = prtest.expected_energy_wh(poa, dc_nameplate=pdc0, dt_hours=0.25)
        measured = prtest.to_energy_wh(pac, kind="power", units="W", dt_hours=0.25)
        ours, _ = prtest.pr_ratio(measured, expected)
        theirs = metrics.performance_ratio_nrel(poa, temp_air, wind, pac, pdc0)
        assert ours == pytest.approx(theirs, rel=1e-9)


def _register(minutes):
    index = pd.date_range("2024-01-01", periods=minutes + 1, freq="1min")
    power = np.arange(minutes) * 10.0
    values = np.concatenate([[0.0], np.cumsum(power / 60.0)])
    return pd.DataFrame({"reg": values}, index=index)


def _interval(frame):
    return register_to_interval_energy(frame, "reg", label="start", verbose=False)


class TestResampleIntervalEnergy:
    def test_native_equals_boundary_route(self):
        frame = _register(60)
        native = prtest.resample_interval_energy(_interval(frame), "15min").dropna()
        boundary = (
            frame["reg"]
            .asfreq("15min")
            .diff()
            .shift(freq=-pd.Timedelta("15min"))
            .dropna()
        )
        pd.testing.assert_series_equal(
            native, boundary, check_names=False, check_freq=False
        )
        hand = sum(i * 10.0 / 60 for i in range(15))
        assert native.iloc[0] == pytest.approx(hand)

    def test_missing_interior_reading_voids_bin(self):
        frame = _register(60)
        gap = frame.drop(frame.index[20])
        out = prtest.resample_interval_energy(_interval(gap), "15min")
        assert np.isnan(out.iloc[1])
        assert not np.isnan(out.iloc[0])

    def test_reset_inside_bin_voids_bin(self):
        frame = _register(60)
        frame.loc[frame.index[20:], "reg"] -= frame["reg"].iloc[20] - 1.0
        with pytest.warns(UserWarning, match="decreases"):
            interval = _interval(frame)
        out = prtest.resample_interval_energy(interval, "15min")
        assert np.isnan(out.iloc[1])
        assert not np.isnan(out.iloc[0])

    def test_incomplete_edge_bins_are_nan(self):
        interval = _interval(_register(60)).iloc[5:-5]
        out = prtest.resample_interval_energy(interval, "15min")
        assert np.isnan(out.iloc[0])
        assert np.isnan(out.iloc[-1])
        assert out.iloc[1:-1].notna().all()

    def test_freq_not_multiple_raises(self):
        interval = _interval(_register(60))
        with pytest.raises(ValueError, match="whole multiple"):
            prtest.resample_interval_energy(interval, "90s")

    def test_duplicate_timestamps_raise(self):
        index = list(pd.date_range("2024-01-01", periods=14, freq="1min"))
        index.append(index[-1])
        series = pd.Series(1.0, index=pd.DatetimeIndex(index))
        with pytest.raises(ValueError, match="duplicate"):
            prtest.resample_interval_energy(series, "15min")

    def test_off_grid_timestamps_raise(self):
        index = pd.date_range("2024-01-01 00:00:30", periods=30, freq="1min")
        with pytest.raises(ValueError, match="aligned"):
            prtest.resample_interval_energy(pd.Series(1.0, index=index), "15min")

    def test_grid_continues_across_midnight(self):
        index = pd.date_range("2024-01-01 22:59", periods=30, freq="7min")
        out = prtest.resample_interval_energy(pd.Series(1.0, index=index), "14min")
        complete = out.dropna()
        assert (complete == 2.0).all()
        assert len(complete) >= 10

    @pytest.mark.filterwarnings("ignore:.*whole multiple of the dominant")
    def test_grid_restarting_at_midnight_raises(self):
        first = pd.date_range("2024-01-01 22:59", "2024-01-01 23:55", freq="7min")
        second = pd.date_range("2024-01-02 00:00", periods=10, freq="7min")
        series = pd.Series(1.0, index=first.append(second))
        with pytest.raises(ValueError, match="aligned"):
            prtest.resample_interval_energy(series, "14min")


class TestDegradation:
    def test_factor_linear(self):
        assert prtest.degradation_factor(0.4, 1.5) == pytest.approx(0.994)

    def test_factor_zero_rate_is_one(self):
        assert prtest.degradation_factor(0, 3) == 1.0

    def test_factor_non_positive_raises(self):
        with pytest.raises(ValueError, match="factor"):
            prtest.degradation_factor(50, 3)

    @pytest.mark.parametrize(
        "rate, years",
        [(-0.4, 1), (0.4, -1), (np.nan, 1), (0.4, np.inf), (np.inf, 1)],
    )
    def test_factor_bad_inputs_raise(self, rate, years):
        with pytest.raises(ValueError, match="rate|years"):
            prtest.degradation_factor(rate, years)

    def test_years_midpoint_of_test_period(self):
        years = prtest.degradation_years("2026-01-01", "2027-01-01", "2027-01-31")
        assert years == pytest.approx(380 / 365)

    def test_years_accepts_timestamps(self):
        years = prtest.degradation_years(
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2027-01-01"),
            pd.Timestamp("2027-01-31"),
        )
        assert years == pytest.approx(380 / 365)

    def test_years_end_before_start_raises(self):
        with pytest.raises(ValueError, match="test_end"):
            prtest.degradation_years("2026-01-01", "2027-01-31", "2027-01-01")

    def test_years_start_before_warranty_raises(self):
        with pytest.raises(ValueError, match="warranty_start"):
            prtest.degradation_years("2026-01-01", "2025-12-01", "2026-01-31")

    def test_degradation_placement_equivalence(self):
        ratio, expected, tol, factor = 0.80, 0.83, 0.03, 0.994
        assert (ratio >= (1 - tol) * expected * factor) == (
            ratio / factor >= (1 - tol) * expected
        )
        scaled = prtest.judge(ratio, expected * factor, tol)
        corrected = prtest.judge(ratio / factor, expected, tol)
        assert scaled.ratio_to_threshold == pytest.approx(corrected.ratio_to_threshold)
        assert scaled.passed == corrected.passed


class TestTolerance:
    @pytest.mark.parametrize(
        "text, expected", [("- 3", 0.03), ("- 2.5", 0.025), ("- 0", 0.0)]
    )
    def test_parses_one_sided(self, text, expected):
        assert prtest.parse_pr_tolerance(text) == pytest.approx(expected)

    def test_two_sided_raises(self):
        with pytest.raises(ValueError, match="one-sided"):
            prtest.parse_pr_tolerance("+/- 5")

    @pytest.mark.parametrize(
        "text", ["- 100", "-3", "- abc", "- nan", "- -1", "+ 3", "", 3]
    )
    def test_malformed_raises(self, text):
        with pytest.raises(ValueError, match="test_tolerance"):
            prtest.parse_pr_tolerance(text)


class TestJudge:
    def test_pass(self):
        result = prtest.judge(0.80, 0.82, 0.03)
        assert result.threshold == pytest.approx(0.7954)
        assert result.margin == pytest.approx(0.0046)
        assert result.ratio_to_threshold == pytest.approx(0.80 / 0.7954)
        assert result.passed is True

    def test_fail(self):
        result = prtest.judge(0.79, 0.82, 0.03)
        assert result.passed is False
        assert result.margin < 0

    def test_exactly_at_threshold_passes(self):
        assert prtest.judge(0.5, 0.5, 0.0).passed is True

    @pytest.mark.parametrize(
        "ratio, expected, name",
        [
            (np.nan, 0.82, "ratio"),
            (np.inf, 0.82, "ratio"),
            (0.8, np.nan, "expected_ratio"),
            (0.8, np.inf, "expected_ratio"),
            (0.8, 0.0, "expected_ratio"),
            (0.8, -0.5, "expected_ratio"),
        ],
    )
    def test_invalid_raises_never_passes(self, ratio, expected, name):
        with pytest.raises(ValueError, match=name):
            prtest.judge(ratio, expected, 0.03)


class TestInputContracts:
    @pytest.fixture
    def poa(self):
        index = pd.date_range("2025-01-01", periods=4, freq="1h")
        return pd.Series([800.0, 900.0, 1000.0, 700.0], index=index)

    @pytest.mark.parametrize("name", ["cell_temp", "t_ref", "availability"])
    def test_expected_energy_rejects_shifted_series(self, poa, name):
        shifted = pd.Series(
            [25.0, 26.0, 27.0, 28.0] if name != "availability" else [1.0] * 4,
            index=poa.index + pd.Timedelta("1h"),
        )
        kwargs = {"cell_temp": poa * 0 + 30.0, "t_ref": 25.0, "power_temp_coeff": -0.3}
        kwargs[name] = shifted
        with pytest.raises(ValueError, match=name):
            prtest.expected_energy_wh(poa, dc_nameplate=1e6, dt_hours=1.0, **kwargs)

    def test_pr_ratio_rejects_misaligned(self, poa):
        expected = pd.Series([1.0, 2.0, 3.0, 4.0], index=poa.index + pd.Timedelta("1h"))
        with pytest.raises(ValueError, match="expected_wh"):
            prtest.pr_ratio(poa, expected)

    def test_pr_ratio_rejects_negative_total(self):
        index = pd.date_range("2025-01-01", periods=2, freq="1h")
        measured = pd.Series([1.0, 1.0], index=index)
        expected = pd.Series([-100.0, 50.0], index=index)
        with pytest.raises(ValueError, match="not positive"):
            prtest.pr_ratio(measured, expected)

    def test_pr_ratio_rejects_nonfinite_total(self):
        index = pd.date_range("2025-01-01", periods=2, freq="1h")
        measured = pd.Series([1.0, 1.0], index=index)
        expected = pd.Series([np.inf, 50.0], index=index)
        with pytest.raises(ValueError):
            prtest.pr_ratio(measured, expected)

    def test_weighted_cell_temp_rejects_misaligned(self, poa):
        cell_temp = pd.Series([30.0] * 4, index=poa.index + pd.Timedelta("1h"))
        with pytest.raises(ValueError, match="cell_temp"):
            prtest.weighted_cell_temp(poa, cell_temp)

    def test_weighted_cell_temp_rejects_all_nan(self, poa):
        cell_temp = pd.Series(np.nan, index=poa.index)
        with pytest.raises(ValueError):
            prtest.weighted_cell_temp(poa, cell_temp)

    def test_resample_rejects_bad_label(self):
        energy = pd.Series(
            1.0, index=pd.date_range("2025-01-01", periods=8, freq="5min")
        )
        with pytest.raises(ValueError, match="label"):
            prtest.resample_interval_energy(energy, "15min", label="center")

    def test_resample_rejects_bad_closed(self):
        energy = pd.Series(
            1.0, index=pd.date_range("2025-01-01", periods=8, freq="5min")
        )
        with pytest.raises(ValueError, match="closed"):
            prtest.resample_interval_energy(energy, "15min", closed="both")

    @pytest.mark.parametrize("freq", ["MS", "M"])
    def test_resample_rejects_non_fixed_freq(self, freq):
        energy = pd.Series(
            1.0, index=pd.date_range("2025-01-01", periods=8, freq="5min")
        )
        with pytest.raises(ValueError, match="freq"):
            prtest.resample_interval_energy(energy, freq)

    def test_resample_rejects_empty(self):
        energy = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        with pytest.raises(ValueError, match="empty"):
            prtest.resample_interval_energy(energy, "15min")

    @pytest.mark.parametrize("tolerance", [1.0, 1.5, float("nan"), -0.01])
    def test_judge_rejects_invalid_tolerance(self, tolerance):
        with pytest.raises(ValueError, match="tolerance"):
            prtest.judge(0.8, 0.82, tolerance)
