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

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_scalar_t_ref_raises(self, bad):
        poa = pd.Series([800.0, 900.0])
        with pytest.raises(ValueError, match="t_ref"):
            prtest.expected_energy_wh(
                poa,
                dc_nameplate=1e6,
                dt_hours=1.0,
                cell_temp=poa * 0 + 40.0,
                t_ref=bad,
                power_temp_coeff=-0.35,
            )

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

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("name", ["measured_wh", "expected_wh"])
    def test_nonfinite_value_raises_instead_of_skipping(self, name, bad):
        series = {
            "measured_wh": pd.Series([75.0, 75.0, 75.0, 75.0]),
            "expected_wh": pd.Series([100.0, 100.0, 100.0, 100.0]),
        }
        series[name].iloc[1] = bad
        with pytest.raises(ValueError, match=rf"{name}.*complete_rows"):
            prtest.pr_ratio(series["measured_wh"], series["expected_wh"])


class TestWeightedCellTemp:
    def test_weighted(self):
        out = prtest.weighted_cell_temp(
            pd.Series([1000.0, 500.0]), pd.Series([40.0, 25.0])
        )
        assert out == pytest.approx(35.0)

    def test_zero_poa_raises(self):
        with pytest.raises(ValueError, match="irradiance"):
            prtest.weighted_cell_temp(pd.Series([0.0, 0.0]), pd.Series([40.0, 25.0]))

    def test_nan_cell_temp_row_raises(self):
        with pytest.raises(ValueError, match=r"cell_temp.*complete_rows"):
            prtest.weighted_cell_temp(
                pd.Series([800.0] * 4), pd.Series([50.0, np.nan, 50.0, 50.0])
            )

    def test_nan_poa_row_raises(self):
        with pytest.raises(ValueError, match=r"poa.*complete_rows"):
            prtest.weighted_cell_temp(
                pd.Series([800.0, np.nan, 800.0, 800.0]), pd.Series([50.0, 10.0] * 2)
            )

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
        series = pd.Series(1.0, index=first.union(second))
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
        with pytest.raises(ValueError, match="finite"):
            prtest.pr_ratio(measured, expected)

    def test_weighted_cell_temp_rejects_misaligned(self, poa):
        cell_temp = pd.Series([30.0] * 4, index=poa.index + pd.Timedelta("1h"))
        with pytest.raises(ValueError, match="cell_temp"):
            prtest.weighted_cell_temp(poa, cell_temp)

    def test_weighted_cell_temp_rejects_all_nan(self, poa):
        cell_temp = pd.Series(np.nan, index=poa.index)
        with pytest.raises(ValueError, match="finite"):
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


def _hourly(start, periods):
    return pd.date_range(start, periods=periods, freq="1h")


class TestFixedReference:
    def test_plain_single_needs_only_expected_pr(self):
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.8)
        assert (ref.pr, ref.t_ref) == (0.8, None)
        assert (ref.source, ref.granularity) == ("fixed", "single")

    def test_corrected_single_requires_explicit_t_ref(self):
        with pytest.raises(ValueError, match="explicit t_ref"):
            prtest.fixed_reference(temp_corrected=True, expected_pr=0.8)

    def test_corrected_single(self):
        ref = prtest.fixed_reference(temp_corrected=True, expected_pr=0.8, t_ref=35.0)
        assert ref.t_ref_for(_hourly("2025-01-01", 2)) == 35.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_ref": 25.0},
            {
                "granularity": "monthly",
                "expected_pr_monthly": [0.8] * 12,
                "t_ref_monthly": [25.0] * 12,
            },
        ],
    )
    def test_plain_rejects_any_t_ref(self, kwargs):
        kwargs.setdefault("expected_pr", 0.8)
        with pytest.raises(ValueError, match="t_ref"):
            prtest.fixed_reference(temp_corrected=False, **kwargs)

    @pytest.mark.parametrize(
        "table",
        [[0.8] * 11, [0.8] * 11 + [np.nan], [0.8] * 11 + [np.inf]],
    )
    def test_bad_table_rejected(self, table):
        with pytest.raises(ValueError, match="expected_pr_monthly"):
            prtest.fixed_reference(
                temp_corrected=False,
                granularity="monthly",
                expected_pr_monthly=table,
            )

    @pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf])
    def test_bad_expected_pr_rejected(self, value):
        with pytest.raises(ValueError, match="expected_pr"):
            prtest.fixed_reference(temp_corrected=False, expected_pr=value)

    def test_bad_granularity_rejected(self):
        with pytest.raises(ValueError, match="granularity"):
            prtest.fixed_reference(
                temp_corrected=False, granularity="daily", expected_pr=0.8
            )

    def test_nonfinite_t_ref_rejected(self):
        with pytest.raises(ValueError, match="t_ref"):
            prtest.fixed_reference(
                temp_corrected=True, expected_pr=0.8, t_ref=float("nan")
            )

    def test_missing_month_names_month_and_table(self):
        table = [0.8, None] + [0.8] * 10
        ref = prtest.fixed_reference(
            temp_corrected=False, granularity="monthly", expected_pr_monthly=table
        )
        with pytest.raises(ValueError, match=r"month 2.*expected_pr_monthly"):
            ref.require_months([1, 2], temp_corrected=False)

    def test_missing_t_ref_month_names_t_ref_table_when_corrected(self):
        ref = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
            t_ref_monthly=[30.0, None] + [30.0] * 10,
        )
        ref.require_months([1], temp_corrected=False)
        with pytest.raises(ValueError, match=r"month 2.*t_ref_monthly"):
            ref.require_months([1, 2], temp_corrected=True)

    def test_monthly_t_ref_for_maps_rows_to_months(self):
        ref = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
            t_ref_monthly=[30.0, 20.0] + [25.0] * 10,
        )
        index = pd.DatetimeIndex(["2025-01-31 23:00", "2025-02-01 00:00"])
        assert ref.t_ref_for(index).tolist() == [30.0, 20.0]

    def test_monthly_dict_read_by_month_number(self):
        table = {month: 40.0 + month for month in range(1, 13)}
        ref = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
            t_ref_monthly=table,
        )
        assert ref.t_ref == {month: 40.0 + month for month in range(1, 13)}

    def test_monthly_dict_missing_months_are_unstated(self):
        ref = prtest.fixed_reference(
            temp_corrected=False,
            granularity="monthly",
            expected_pr_monthly={2: 0.82, 1: 0.81, 3: None},
        )
        assert ref.pr == {1: 0.81, 2: 0.82}
        with pytest.raises(ValueError, match=r"month 3.*expected_pr_monthly"):
            ref.require_months([1, 3], temp_corrected=False)

    def test_series_indexed_by_month_read_by_month_number(self):
        ref = prtest.fixed_reference(
            temp_corrected=False,
            granularity="monthly",
            expected_pr_monthly=pd.Series({6: 0.81, 7: 0.83}),
        )
        assert ref.pr == {6: 0.81, 7: 0.83}

    def test_series_with_default_index_read_by_position(self):
        values = pd.Series([0.80 + 0.01 * i for i in range(12)])
        ref = prtest.fixed_reference(
            temp_corrected=False, granularity="monthly", expected_pr_monthly=values
        )
        assert ref.pr == {i + 1: pytest.approx(0.80 + 0.01 * i) for i in range(12)}

    @pytest.mark.parametrize("key", [0, 13, 1.0, "1", True])
    def test_monthly_dict_bad_key_raises(self, key):
        with pytest.raises(ValueError, match=r"expected_pr_monthly.*month number"):
            prtest.fixed_reference(
                temp_corrected=False,
                granularity="monthly",
                expected_pr_monthly={key: 0.8},
            )

    @pytest.mark.parametrize("table", ["0.8" * 12, 0.8, {0.8, 0.81}])
    def test_monthly_table_of_unsupported_type_raises(self, table):
        with pytest.raises(TypeError, match="expected_pr_monthly"):
            prtest.fixed_reference(
                temp_corrected=False,
                granularity="monthly",
                expected_pr_monthly=table,
            )


class TestDerivedReference:
    @pytest.fixture
    def rows(self):
        index = _hourly("2025-01-31 08:00", 4).union(_hourly("2025-02-01 08:00", 4))
        poa = pd.Series([200.0, 400.0, 600.0, 800.0] * 2, index=index)
        cell = pd.Series([10.0, 20.0, 30.0, 40.0, 5.0, 10.0, 15.0, 20.0], index=index)
        energy = pd.Series(
            [90.0, 180.0, 270.0, 360.0, 100.0, 200.0, 300.0, 400.0], index=index
        )
        return energy, poa, cell

    def test_single_matches_pr_ratio_and_eq5(self, rows):
        energy, poa, cell = rows
        ref = prtest.derive_reference(
            energy, poa, dc_nameplate=1000.0, dt_hours=1.0, cell_temp=cell
        )
        expected = prtest.expected_energy_wh(poa, dc_nameplate=1000.0, dt_hours=1.0)
        assert ref.pr == prtest.pr_ratio(energy, expected)[0]
        assert ref.t_ref == prtest.weighted_cell_temp(poa, cell)
        assert (ref.source, ref.granularity) == ("derived", "single")

    def test_plain_derives_no_t_ref(self, rows):
        energy, poa, _ = rows
        ref = prtest.derive_reference(energy, poa, dc_nameplate=1000.0, dt_hours=1.0)
        assert ref.t_ref is None

    def test_monthly_equals_per_month_plain_pr(self, rows):
        energy, poa, cell = rows
        ref = prtest.derive_reference(
            energy,
            poa,
            dc_nameplate=1000.0,
            dt_hours=1.0,
            cell_temp=cell,
            monthly=True,
        )
        assert ref.granularity == "monthly"
        for month in (1, 2):
            sel = energy.index.month == month
            expected = prtest.expected_energy_wh(
                poa[sel], dc_nameplate=1000.0, dt_hours=1.0
            )
            assert ref.pr[month] == pytest.approx(
                prtest.pr_ratio(energy[sel], expected)[0]
            )
            assert ref.t_ref[month] == pytest.approx(
                prtest.weighted_cell_temp(poa[sel], cell[sel])
            )

    def test_month_number_pools_across_years(self):
        index = _hourly("1990-01-05 10:00", 2).union(_hourly("2027-01-05 10:00", 2))
        poa = pd.Series([500.0] * 4, index=index)
        energy = pd.Series([400.0] * 4, index=index)
        ref = prtest.derive_reference(
            energy, poa, dc_nameplate=1000.0, dt_hours=1.0, monthly=True
        )
        assert list(ref.pr) == [1]
        assert ref.pr[1] == pytest.approx(0.8)

    def test_nonfinite_pr_raises(self, rows):
        energy, poa, _ = rows
        energy = energy.copy()
        energy.iloc[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            prtest.derive_reference(energy, poa, dc_nameplate=1000.0, dt_hours=1.0)

    def test_nonfinite_monthly_pr_raises(self, rows):
        energy, poa, _ = rows
        energy = energy.copy()
        energy.iloc[-1] = np.inf
        with pytest.raises(ValueError, match="finite"):
            prtest.derive_reference(
                energy, poa, dc_nameplate=1000.0, dt_hours=1.0, monthly=True
            )

    def test_nonfinite_t_ref_raises(self, rows):
        energy, poa, cell = rows
        cell = cell.copy()
        cell.iloc[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            prtest.derive_reference(
                energy, poa, dc_nameplate=1000.0, dt_hours=1.0, cell_temp=cell
            )

    @pytest.mark.parametrize("name", ["energy_wh", "poa", "cell_temp"])
    @pytest.mark.parametrize("monthly", [False, True])
    def test_nan_input_raises_instead_of_skipping(self, rows, name, monthly):
        inputs = dict(zip(("energy_wh", "poa", "cell_temp"), rows, strict=True))
        inputs[name] = inputs[name].copy()
        inputs[name].iloc[1] = np.nan
        with pytest.raises(ValueError, match=rf"{name}.*complete_rows"):
            prtest.derive_reference(
                inputs["energy_wh"],
                inputs["poa"],
                cell_temp=inputs["cell_temp"],
                dc_nameplate=1000.0,
                dt_hours=1.0,
                monthly=monthly,
            )

    def test_monthly_round_trips_through_fixed_reference(self, rows):
        energy, poa, cell = rows
        derived = prtest.derive_reference(
            energy,
            poa,
            dc_nameplate=1000.0,
            dt_hours=1.0,
            cell_temp=cell,
            monthly=True,
        )
        fixed = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=derived.pr,
            t_ref_monthly=derived.t_ref,
        )
        assert fixed.pr == derived.pr
        assert fixed.t_ref == derived.t_ref
        assert sorted(fixed.pr) == [1, 2]

    def test_derived_missing_month_raises_naming_month(self, rows):
        energy, poa, _ = rows
        ref = prtest.derive_reference(
            energy, poa, dc_nameplate=1000.0, dt_hours=1.0, monthly=True
        )
        with pytest.raises(ValueError, match="month 3"):
            ref.require_months([1, 3], temp_corrected=False)


class TestPartialModeledMonth:
    @staticmethod
    def _derive(kept_days):
        unfiltered = pd.date_range("2025-01-01", "2025-01-31 23:00", freq="1h")
        index = unfiltered[unfiltered.day.isin(kept_days)]
        poa = pd.Series(500.0, index=index)
        energy = pd.Series(400.0, index=index)
        return prtest.derive_reference(
            energy,
            poa,
            dc_nameplate=1000.0,
            dt_hours=1.0,
            monthly=True,
            unfiltered_index=unfiltered,
        )

    def test_partial_month_warns_naming_january(self):
        with pytest.warns(UserWarning, match="January"):
            self._derive(range(1, 11))

    def test_scattered_gaps_do_not_warn(self):
        kept = [d for d in range(1, 32) if d % 3 != 0 or d > 31]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._derive(kept)

    def test_seven_day_run_warns(self):
        with pytest.warns(UserWarning, match="January"):
            self._derive(range(8, 32))

    def test_six_day_run_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._derive(range(7, 32))

    @staticmethod
    def _derive_two_months(gap_start, gap_end):
        unfiltered = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="1h")
        gap = (unfiltered >= gap_start) & (unfiltered < pd.Timestamp(gap_end))
        index = unfiltered[~gap]
        return prtest.derive_reference(
            pd.Series(400.0, index=index),
            pd.Series(500.0, index=index),
            dc_nameplate=1000.0,
            dt_hours=1.0,
            monthly=True,
            unfiltered_index=unfiltered,
        )

    def test_seven_day_run_across_month_boundary_warns(self):
        with pytest.warns(UserWarning, match="January.*February") as record:
            self._derive_two_months("2025-01-29", "2025-02-05")
        assert "7 consecutive days" in str(record[0].message)

    def test_six_day_run_across_month_boundary_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._derive_two_months("2025-01-29", "2025-02-04")

    def test_run_through_month_without_kept_rows_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._derive_two_months("2025-01-01", "2025-02-01")


class TestMeasuredPr:
    @pytest.fixture
    def rows(self):
        index = _hourly("2025-01-31 22:00", 2).union(_hourly("2025-02-01 00:00", 2))
        poa = pd.Series([500.0] * 4, index=index)
        cell = pd.Series([40.0] * 4, index=index)
        energy = pd.Series([300.0, 310.0, 320.0, 330.0], index=index)
        return energy, poa, cell

    def test_monthly_t_ref_applied_per_interval(self, rows):
        energy, poa, cell = rows
        ref = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
            t_ref_monthly=[30.0, 20.0] + [25.0] * 10,
        )
        out = prtest.measured_pr(
            energy,
            poa,
            dc_nameplate=1000.0,
            dt_hours=1.0,
            reference=ref,
            cell_temp=cell,
            power_temp_coeff=-0.4,
        )
        assert out.intervals["t_ref"].tolist() == [30.0, 30.0, 20.0, 20.0]
        base = 1000.0 * 0.5
        en = [base * (1 - 0.004 * (40 - t)) for t in (30.0, 30.0, 20.0, 20.0)]
        assert out.ratio == pytest.approx(energy.sum() / sum(en))
        assert list(out.intervals.columns) == [
            "energy_wh",
            "expected_wh",
            "poa",
            "cell_temp",
            "t_ref",
            "availability",
            "pr",
        ]
        assert list(out.monthly.index) == [1, 2]
        assert out.monthly["intervals"].tolist() == [2, 2]
        assert out.monthly["measured_pr"].iloc[0] == pytest.approx(
            energy.iloc[:2].sum() / sum(en[:2])
        )

    def test_availability_scales_expected(self, rows):
        energy, poa, _ = rows
        energy, poa = energy.iloc[:2], poa.iloc[:2]
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.8)
        out = prtest.measured_pr(
            energy,
            poa,
            dc_nameplate=1000.0,
            dt_hours=1.0,
            reference=ref,
            availability=pd.Series([1.0, 0.5], index=energy.index),
        )
        assert out.ratio == pytest.approx(energy.sum() / (500.0 * 1.0 + 500.0 * 0.5))
        assert out.ratio_without_availability == pytest.approx(energy.sum() / 1000.0)
        assert out.intervals["availability"].tolist() == [1.0, 0.5]
        assert out.intervals["cell_temp"].isna().all()
        assert out.intervals["t_ref"].isna().all()

    def test_insolation_weights_from_kept_rows_only(self, rows):
        energy, poa, _ = rows
        poa = poa.copy()
        poa.iloc[0] = 100.0
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.8)
        out = prtest.measured_pr(
            energy, poa, dc_nameplate=1000.0, dt_hours=0.5, reference=ref
        )
        assert out.monthly["insolation_wh_m2"].tolist() == [300.0, 500.0]

    def test_nan_availability_raises_instead_of_skipping(self, rows):
        energy, poa, _ = rows
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.8)
        availability = pd.Series([1.0, np.nan, 1.0, 1.0], index=energy.index)
        with pytest.raises(ValueError, match=r"availability.*complete_rows"):
            prtest.measured_pr(
                energy,
                poa,
                dc_nameplate=1000.0,
                dt_hours=1.0,
                reference=ref,
                availability=availability,
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize("name", ["energy_wh", "poa", "cell_temp"])
    def test_nonfinite_input_raises_instead_of_skipping(self, rows, name, bad):
        inputs = dict(zip(("energy_wh", "poa", "cell_temp"), rows, strict=True))
        inputs[name] = inputs[name].copy()
        inputs[name].iloc[2] = bad
        ref = prtest.fixed_reference(temp_corrected=True, expected_pr=0.8, t_ref=25.0)
        with pytest.raises(ValueError, match=rf"{name}.*complete_rows"):
            prtest.measured_pr(
                inputs["energy_wh"],
                inputs["poa"],
                cell_temp=inputs["cell_temp"],
                power_temp_coeff=-0.4,
                dc_nameplate=1000.0,
                dt_hours=1.0,
                reference=ref,
            )

    def test_month_without_expected_energy_warns(self, rows):
        energy, poa, _ = rows
        poa = poa.copy()
        poa.iloc[:2] = 0.0
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.8)
        with pytest.warns(UserWarning, match="January") as record:
            out = prtest.measured_pr(
                energy, poa, dc_nameplate=1000.0, dt_hours=1.0, reference=ref
            )
        assert "February" not in str(record[0].message)
        assert np.isnan(out.monthly.loc[1, "measured_pr"])
        assert out.monthly.loc[2, "measured_pr"] == pytest.approx(650.0 / 1000.0)

    def test_missing_reference_month_raises(self, rows):
        energy, poa, cell = rows
        ref = prtest.fixed_reference(
            temp_corrected=True,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
            t_ref_monthly=[30.0] + [None] * 11,
        )
        with pytest.raises(ValueError, match="month 2"):
            prtest.measured_pr(
                energy,
                poa,
                dc_nameplate=1000.0,
                dt_hours=1.0,
                reference=ref,
                cell_temp=cell,
                power_temp_coeff=-0.4,
            )


class TestReferenceShape:
    @pytest.mark.parametrize(
        "pr, t_ref, granularity",
        [
            ({1: 0.8}, None, "single"),
            (0.8, {1: 25.0}, "single"),
            (0.8, None, "monthly"),
            ({1: 0.8}, 25.0, "monthly"),
            ([0.8] * 12, None, "monthly"),
            (0.8, None, "daily"),
        ],
    )
    def test_values_must_match_granularity(self, pr, t_ref, granularity):
        with pytest.raises(ValueError, match="granularity"):
            prtest.Reference(pr, t_ref, "fixed", granularity)

    @pytest.mark.parametrize(
        "pr, t_ref, granularity",
        [
            (0.8, 25.0, "single"),
            (None, 25, "single"),
            (np.float64(0.8), None, "single"),
            ({1: 0.8}, {1: 25.0}, "monthly"),
            ({1: 0.8}, None, "monthly"),
        ],
    )
    def test_matching_values_accepted(self, pr, t_ref, granularity):
        ref = prtest.Reference(pr, t_ref, "fixed", granularity)
        assert (ref.pr, ref.t_ref) == (pr, t_ref)


class TestWeightedExpectedPr:
    def test_monthly_weights_by_insolation(self):
        ref = prtest.fixed_reference(
            temp_corrected=False,
            granularity="monthly",
            expected_pr_monthly=[0.80, 0.84] + [0.9] * 10,
        )
        h = pd.Series({1: 3000.0, 2: 1000.0})
        assert prtest.weighted_expected_pr(ref, h) == pytest.approx(0.81)

    def test_zero_insolation_month_needs_no_value(self):
        ref = prtest.fixed_reference(
            temp_corrected=False,
            granularity="monthly",
            expected_pr_monthly=[0.80, None] + [0.9] * 10,
        )
        h = pd.Series({1: 3000.0, 2: 0.0})
        assert prtest.weighted_expected_pr(ref, h) == pytest.approx(0.80)

    def test_single_returns_pr(self):
        ref = prtest.fixed_reference(temp_corrected=False, expected_pr=0.77)
        assert prtest.weighted_expected_pr(ref, pd.Series({1: 5.0})) == 0.77

    def test_no_insolation_raises(self):
        ref = prtest.fixed_reference(
            temp_corrected=False,
            granularity="monthly",
            expected_pr_monthly=[0.8] * 12,
        )
        with pytest.raises(ValueError, match="insolation"):
            prtest.weighted_expected_pr(ref, pd.Series({1: 0.0}))


def _pr_frame(periods=8, freq="15min"):
    idx = pd.date_range("2025-06-01 10:00", periods=periods, freq=freq)
    rng = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "power": 400.0 + 10 * rng,
            "poa": 600.0 + 20 * rng,
            "tcell": 40.0 + rng,
            "bom": 35.0 + rng,
            "amb": 25.0 + 0.5 * rng,
            "wind": 2.0 + 0.1 * rng,
        },
        index=idx,
    )


class TestPrTestResults:
    def _full(self, **changes):
        fields = {
            "ratio": 0.8123,
            "ratio_without_availability": 0.8123,
            "temp_corrected": False,
            "dc_nameplate": 1000.0,
            "power_temp_coeff": None,
            "intervals": pd.DataFrame(),
            "points_used": {"meas": 4},
            "dropped_nan": {"meas": 0},
            "dropped_nonfinite": {"meas": 0},
            "dt_hours": {"meas": 0.25},
            "numerator": {"meas": {"kind": "power", "units": "W"}},
            "expected_ratio": 0.80,
            "threshold": 0.78,
            "margin": 0.0323,
            "passed": True,
        }
        fields.update(changes)
        return prtest.PrTestResults(**fields)

    def test_kind_is_not_a_field(self):
        import dataclasses

        assert prtest.PrTestResults.kind == "pr"
        assert "kind" not in [f.name for f in dataclasses.fields(prtest.PrTestResults)]

    def test_summary_full(self):
        text = self._full().summary()
        assert "PASS" in text
        assert "81.23" in text
        assert "80.00" in text
        assert "78.00" in text
        assert "+3.23" in text
        assert str(self._full()) == text

    def test_summary_fail(self):
        assert "FAIL" in self._full(passed=False, margin=-0.01).summary()

    def test_summary_standalone(self):
        text = self._full(
            expected_ratio=None, threshold=None, margin=None, passed=None
        ).summary()
        assert "81.23" in text
        assert "PASS" not in text and "FAIL" not in text
        assert "Threshold" not in text and "Margin" not in text


class TestPerformanceRatio:
    kwargs = {"numerator": "power", "poa": "poa", "dc_nameplate": 1000.0}

    def test_plain_matches_hand_computation(self):
        df = _pr_frame()
        res = prtest.performance_ratio(df, **self.kwargs)
        expected = (df["power"] * 0.25).sum() / (1000.0 * df["poa"] / 1000 * 0.25).sum()
        assert res.ratio == pytest.approx(expected)
        assert res.temp_corrected is False
        assert res.passed is None and res.valid is None
        assert res.expected_ratio is None
        assert res.dt_hours == {"meas": 0.25}
        assert res.points_used == {"meas": 8}
        assert res.numerator == {"meas": {"kind": "power", "units": "W"}}
        assert res.reference_source == "standalone"
        assert res.granularity == "single"

    def _corrected_expected(self, df, tcell, coeff=-0.4, base=25.0):
        exp = 1000.0 * df["poa"] / 1000 * 0.25 * (1 + coeff / 100 * (tcell - base))
        return (df["power"] * 0.25).sum() / exp.sum()

    def test_corrected_via_cell_temp(self):
        df = _pr_frame()
        res = prtest.performance_ratio(
            df, cell_temp="tcell", power_temp_coeff=-0.4, **self.kwargs
        )
        assert res.ratio == pytest.approx(self._corrected_expected(df, df["tcell"]))
        assert res.temp_corrected is True
        assert res.t_ref == 25.0
        assert res.power_temp_coeff == -0.4

    def test_corrected_via_bom(self):
        from captest import calcparams

        df = _pr_frame()
        tcell = calcparams.cell_temp(df, "bom", "poa", temp_delta=3.0, verbose=False)
        res = prtest.performance_ratio(
            df, bom="bom", temp_delta=3.0, power_temp_coeff=-0.4, **self.kwargs
        )
        assert res.ratio == pytest.approx(self._corrected_expected(df, tcell))

    def test_corrected_via_amb_and_wind(self):
        from captest import calcparams

        df = _pr_frame()
        bom = calcparams.bom_temp(
            df, poa="poa", temp_amb="amb", wind_speed="wind", verbose=False
        )
        tcell = calcparams.cell_temp(df.assign(b=bom), "b", "poa", verbose=False)
        res = prtest.performance_ratio(
            df,
            temp_amb="amb",
            wind_speed="wind",
            power_temp_coeff=-0.4,
            base_temp=20.0,
            **self.kwargs,
        )
        assert res.t_ref == 20.0
        assert res.ratio == pytest.approx(
            self._corrected_expected(df, tcell, base=20.0)
        )

    def test_multiple_temperature_sources_raise(self):
        with pytest.raises(ValueError, match="one"):
            prtest.performance_ratio(
                _pr_frame(),
                cell_temp="tcell",
                bom="bom",
                power_temp_coeff=-0.4,
                **self.kwargs,
            )

    def test_incomplete_amb_wind_raises(self):
        with pytest.raises(ValueError, match="temp_amb"):
            prtest.performance_ratio(
                _pr_frame(), temp_amb="amb", power_temp_coeff=-0.4, **self.kwargs
            )

    def test_source_without_coefficient_raises(self):
        with pytest.raises(ValueError, match="power_temp_coeff"):
            prtest.performance_ratio(_pr_frame(), cell_temp="tcell", **self.kwargs)

    def test_coefficient_without_source_raises(self):
        with pytest.raises(ValueError, match="power_temp_coeff"):
            prtest.performance_ratio(_pr_frame(), power_temp_coeff=-0.4, **self.kwargs)

    def test_dt_hours_inferred_and_explicit(self):
        df = _pr_frame(freq="30min")
        assert prtest.performance_ratio(df, **self.kwargs).dt_hours == {"meas": 0.5}
        res = prtest.performance_ratio(df, dt_hours=0.25, **self.kwargs)
        assert res.dt_hours == {"meas": 0.25}

    @pytest.mark.parametrize("bad", [0, -0.25])
    def test_bad_dt_hours_raises(self, bad):
        with pytest.raises(ValueError, match="dt_hours"):
            prtest.performance_ratio(_pr_frame(), dt_hours=bad, **self.kwargs)

    @pytest.mark.parametrize("bad", [0, -5.0])
    def test_bad_nameplate_raises(self, bad):
        with pytest.raises(ValueError, match="dc_nameplate"):
            prtest.performance_ratio(
                _pr_frame(), numerator="power", poa="poa", dc_nameplate=bad
            )

    def test_energy_numerator_and_units(self):
        df = _pr_frame()
        df["energy"] = df["power"] * 0.25 / 1000
        res = prtest.performance_ratio(
            df, kind="energy", units="kWh", **{**self.kwargs, "numerator": "energy"}
        )
        plain = prtest.performance_ratio(df, **self.kwargs)
        assert res.ratio == pytest.approx(plain.ratio)

    def test_nan_rows_dropped_and_counted(self):
        df = _pr_frame()
        df.iloc[2, df.columns.get_loc("poa")] = np.nan
        df.iloc[3, df.columns.get_loc("power")] = np.inf
        with pytest.warns(UserWarning, match="dropped"):
            res = prtest.performance_ratio(df, **self.kwargs)
        assert res.points_used == {"meas": 6}
        assert res.dropped_nan == {"meas": 1}
        assert res.dropped_nonfinite == {"meas": 1}

    def test_nan_row_leaves_complete_row_pr(self):
        df = _pr_frame()
        df.iloc[2, df.columns.get_loc("poa")] = np.nan
        with pytest.warns(UserWarning, match="dropped 1"):
            res = prtest.performance_ratio(df, **self.kwargs)
        kept = df.drop(df.index[2])
        expected = (kept["power"] * 0.25).sum() / (kept["poa"] * 0.25).sum()
        assert res.ratio == pytest.approx(expected)
        assert res.points_used == {"meas": 7}

    def test_no_complete_rows_raises(self):
        df = _pr_frame()
        df["poa"] = np.nan
        with (
            pytest.warns(UserWarning, match="dropped"),
            pytest.raises(ValueError, match=r"no complete rows.*meas.*8.*0"),
        ):
            prtest.performance_ratio(df, **self.kwargs)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_base_temp_raises(self, bad):
        with pytest.raises(ValueError, match="base_temp"):
            prtest.performance_ratio(
                _pr_frame(),
                cell_temp="tcell",
                power_temp_coeff=-0.4,
                base_temp=bad,
                **self.kwargs,
            )

    @pytest.mark.parametrize(
        "changes",
        [
            {"temp_delta": 3.0},
            {"module_type": "glass_cell_glass"},
            {"racking": "insulated_back"},
            {"base_temp": 20.0},
        ],
    )
    def test_temperature_options_without_source_raise(self, changes):
        (name,) = changes
        with pytest.raises(ValueError, match=name):
            prtest.performance_ratio(_pr_frame(), **changes, **self.kwargs)

    @pytest.mark.parametrize(
        "changes",
        [
            {"temp_delta": 3.0},
            {"module_type": "glass_cell_glass"},
            {"racking": "insulated_back"},
        ],
    )
    def test_sandia_options_with_measured_cell_temp_raise(self, changes):
        (name,) = changes
        with pytest.raises(ValueError, match=name):
            prtest.performance_ratio(
                _pr_frame(),
                cell_temp="tcell",
                power_temp_coeff=-0.4,
                **changes,
                **self.kwargs,
            )

    def test_unknown_column_raises_key_error(self):
        with pytest.raises(KeyError, match="powr"):
            prtest.performance_ratio(
                _pr_frame(), **{**self.kwargs, "numerator": "powr"}
            )

    def test_standalone_identity_note(self):
        df = _pr_frame()
        plain = prtest.performance_ratio(df, **self.kwargs)
        weighted = prtest.weighted_cell_temp(df["poa"], df["tcell"])
        corrected = prtest.performance_ratio(
            df,
            cell_temp="tcell",
            power_temp_coeff=-0.4,
            base_temp=weighted,
            **self.kwargs,
        )
        assert corrected.ratio == pytest.approx(plain.ratio)
