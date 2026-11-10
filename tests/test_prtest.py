import warnings

import numpy as np
import pandas as pd
import pytest

from captest import prtest


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
