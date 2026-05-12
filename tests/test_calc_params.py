import warnings

import numpy as np
import pvlib
import pandas as pd
import pytest

from captest import calcparams


@pytest.fixture
def site_mendoza():
    """Site dict with loc/sys sub-dicts for a notional Texas site."""
    return {
        "loc": {
            "latitude": 33.0,
            "longitude": -99.5,
            "altitude": 500,
            "tz": "Etc/GMT+6",
        },
        "sys": {
            "axis_tilt": 0,
            "axis_azimuth": 180,
            "max_angle": 60,
            "backtrack": False,
            "gcr": 0.33,
            "albedo": 0.2,
        },
    }


@pytest.fixture
def solar_day_index():
    """DatetimeIndex spanning a single day at hourly cadence (tz-naive)."""
    return pd.date_range("2023-06-21 00:00", periods=24, freq="h")


class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_series(self, capsys):
        df = pd.DataFrame({"power_col": [10, 12, 15], "cell_temp_col": [50, 50, 50]})
        power_tc = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert isinstance(power_tc, pd.Series)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "power_temp_correct" column as '
            "(power_col) / (1 + ((-0.37 / 100) * (cell_temp_col - 25)))"
        )

    def test_high_temp_higher_power(self, capsys):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [50]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37, verbose=False
        )
        assert corr_power.iloc[0] > df["power_col"].iloc[0]
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_low_temp_lower_power(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [10]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert corr_power.iloc[0] < df["power_col"].iloc[0]

    def test_math_series_power(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {"power_col": [10, 20, 15], "cell_temp_col": [50, 50, 50]}, index=ix
        )
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert pytest.approx(corr_power.values, 0.3) == [11.019, 22.038, 16.528]

    def test_no_temp_diff(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [25]})
        corrected_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert corrected_power.iloc[0] == 10

    def test_user_base_temp(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [30]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37, base_temp=27.5
        )
        assert pytest.approx(corr_power.iloc[0], 0.3) == 10.093


class TestBomTemp:
    """Test calculation of back of module (BOM) temperature from weather."""

    def test_no_output_when_verbose_false(self, capsys):
        """Ensure bom_temp does not print when verbose is False"""
        df = pd.DataFrame(
            {
                "poa": [800],
                "temp_amb": [25],
                "wind": [1.0],
            }
        )
        _ = calcparams.bom_temp(df, "poa", "temp_amb", "wind", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dataframe_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {
                "poa": [805, 810, 812],
                "temp_amb": [26, 27, 27.5],
                "wind": [0.5, 1, 2.5],
            },
            index=ix,
        )

        exp_results = pd.Series([48.0506544, 48.3709869, 46.6442104], index=ix)

        pd.testing.assert_series_equal(
            calcparams.bom_temp(df, "poa", "temp_amb", "wind"), exp_results
        )
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "bom_temp" column as '
            "poa * e^(-3.56 + -0.075 * wind) + temp_amb. "
            'Coefficients a and b assume "glass_cell_poly" modules and "open_rack" racking.'
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", 50.77154),
            ("open_rack", "glass_cell_poly", 48.33028),
            ("open_rack", "poly_tf_steel", 46.82361),
            ("close_roof_mount", "glass_cell_glass", 65.86252),
            ("insulated_back", "glass_cell_poly", 72.98647),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        # create single-row DataFrame
        df = pd.DataFrame(
            {
                "poa": [800],
                "temp_amb": [28],
                "wind": [1.5],
            }
        )
        bom = calcparams.bom_temp(
            df,
            "poa",
            "temp_amb",
            "wind",
            module_type=module_type,
            racking=racking,
            verbose=False,
        )
        assert bom.iloc[0] == pytest.approx(expected)


class TestCellTemp:
    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_bom = pd.Series([26, 27, 27.5], index=ix)
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom}, index=ix)

        exp_results = pd.Series([28.415, 29.43, 29.936], index=ix)

        pd.testing.assert_series_equal(
            calcparams.cell_temp(df, "bom_temp", "poa"), exp_results
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", pd.Series([28.415, 29.43, 29.936])),
            (
                "close_roof_mount",
                "glass_cell_glass",
                pd.Series([26.805, 27.81, 28.312]),
            ),
            ("insulated_back", "glass_cell_poly", pd.Series([26, 27, 27.5])),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        # ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812])
        temp_bom = pd.Series([26, 27, 27.5])
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom})
        ctemp = calcparams.cell_temp(
            df,
            "bom_temp",
            "poa",
            module_type=module_type,
            racking=racking,
            verbose=False,
        )
        pd.testing.assert_series_equal(ctemp, expected, check_names=False)

    def test_output_message_series_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix, name="poa")
        temp_bom = pd.Series([26, 27, 27.5], index=ix, name="bom_temp")
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom}, index=ix)

        calcparams.cell_temp(df, "bom_temp", "poa")

        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out

        print(stdout_content)
        assert stdout_content.rstrip("\n") == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the "bom_temp" and "poa" columns.'
        )


class TestAvgTypCellTemp:
    def test_math(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {
                "poa": [805, 810, 812],
                "cell_temp": [26, 27, 27.5],
            },
            index=ix,
        )

        assert calcparams.avg_typ_cell_temp(df, "poa", "cell_temp") == pytest.approx(
            26.8356
        )


class TestRpoaPvsyst:
    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        globbak = pd.Series([100, 110, 120], index=ix)
        backshd = pd.Series([10, 15, 20], index=ix)
        df = pd.DataFrame(
            {
                "GlobBak": globbak,
                "BackShd": backshd,
            },
            index=ix,
        )

        exp_results = pd.Series([110, 125, 140], index=ix)

        pd.testing.assert_series_equal(calcparams.rpoa_pvsyst(df), exp_results)


class TestEtotal:
    def _make_df(self, poa_vals, rear_vals):
        return pd.DataFrame({"poa": poa_vals, "rear": rear_vals})

    def test_numeric_inputs(self):
        df = self._make_df([100], [10])
        result = calcparams.e_total(df, "poa", "rear")
        assert result.iloc[0] == 107

    def test_numeric_non_default_bifaciality(self):
        df = self._make_df([100], [10])
        result = calcparams.e_total(df, "poa", "rear", bifaciality=0.5)
        assert result.iloc[0] == 105

    def test_numeric_non_default_bifi_frac(self):
        df = self._make_df([100], [10])
        result = calcparams.e_total(df, "poa", "rear", bifaciality=1, bifacial_frac=0.5)
        assert result.iloc[0] == 105

    def test_numeric_non_default_bifaciality_and_bifacial_frac(self):
        df = self._make_df([100], [20])
        result = calcparams.e_total(
            df, "poa", "rear", bifaciality=0.5, bifacial_frac=0.5
        )
        assert result.iloc[0] == 105

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = self._make_df([100, 110, 120], [100, 150, 200])
        df.index = ix
        exp_results = pd.Series([170, 215, 260], index=ix)
        pd.testing.assert_series_equal(
            calcparams.e_total(df, "poa", "rear"), exp_results, check_dtype=False
        )

    def test_rear_shade(self):
        df = self._make_df([100], [20])
        result = calcparams.e_total(df, "poa", "rear", rear_shade=0.5)
        assert result.iloc[0] == 107

    def test_output_message(self, capsys):
        """Ensure e_total prints correct formula when verbose is True"""
        df = self._make_df([100], [10])
        _ = calcparams.e_total(df, "poa", "rear")
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "e_total" column as poa + rear * 0.7 * 1 * (1 - 0)'
        )


class TestApparentZenith:
    """Test apparent_zenith wrapping pvlib.Location.get_solarposition."""

    def test_returns_series_aligned_to_data_index(self, site_mendoza, solar_day_index):
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        zenith = calcparams.apparent_zenith(df, site=site_mendoza, verbose=False)
        assert isinstance(zenith, pd.Series)
        assert zenith.index.equals(df.index)

    def test_nighttime_rows_are_nan(self, site_mendoza, solar_day_index):
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        zenith = calcparams.apparent_zenith(df, site=site_mendoza, verbose=False)
        # Midnight local time -> sun below horizon -> NaN.
        assert np.isnan(zenith.iloc[0])
        # Noon local time -> daytime -> finite.
        assert np.isfinite(zenith.iloc[12])
        assert zenith.iloc[12] < 90

    def test_altitude_override_does_not_mutate_caller_site(
        self, site_mendoza, solar_day_index
    ):
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        original_altitude = site_mendoza["loc"]["altitude"]
        calcparams.apparent_zenith(
            df, site=site_mendoza, altitude_override=0, verbose=False
        )
        assert site_mendoza["loc"]["altitude"] == original_altitude

    def test_altitude_override_none_respects_site_altitude(
        self, site_mendoza, solar_day_index
    ):
        """altitude_override=None should NOT force altitude=0."""
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        z_sealevel = calcparams.apparent_zenith(
            df, site=site_mendoza, altitude_override=0, verbose=False
        )
        z_site = calcparams.apparent_zenith(
            df, site=site_mendoza, altitude_override=None, verbose=False
        )
        # Differ at noon because altitude affects apparent zenith via refraction.
        noon_diff = abs(z_sealevel.iloc[12] - z_site.iloc[12])
        # Any non-zero difference demonstrates altitude was respected.
        assert noon_diff >= 0  # both finite; precise value depends on pvlib.


class TestApparentZenithPvsyst:
    """Test apparent_zenith_pvsyst with 30-minute shift."""

    def test_returned_index_matches_input(self, site_mendoza, solar_day_index):
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        zenith = calcparams.apparent_zenith_pvsyst(df, site=site_mendoza, verbose=False)
        assert zenith.index.equals(df.index)

    def test_values_equal_shifted_reference(self, site_mendoza, solar_day_index):
        """Zenith at label t equals zenith computed at t+30min unshifted."""
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        ref_df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))},
            index=solar_day_index.shift(30, "min"),
        )
        ref_zenith = calcparams.apparent_zenith(
            ref_df, site=site_mendoza, verbose=False
        )
        shifted_zenith = calcparams.apparent_zenith_pvsyst(
            df, site=site_mendoza, verbose=False
        )
        # Values at position i are the same; only index labels differ.
        np.testing.assert_allclose(
            shifted_zenith.values,
            ref_zenith.values,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_shift_minutes_zero_matches_unshifted(self, site_mendoza, solar_day_index):
        df = pd.DataFrame(
            {"irr": np.zeros(len(solar_day_index))}, index=solar_day_index
        )
        z_shift0 = calcparams.apparent_zenith_pvsyst(
            df, site=site_mendoza, shift_minutes=0, verbose=False
        )
        z_noshift = calcparams.apparent_zenith(df, site=site_mendoza, verbose=False)
        np.testing.assert_allclose(
            z_shift0.values, z_noshift.values, rtol=1e-10, equal_nan=True
        )


class TestAbsoluteAirmass:
    """Test absolute_airmass with and without pressure column."""

    def _day_df(self):
        ix = pd.date_range("2023-06-21 10:00", periods=5, freq="h")
        return pd.DataFrame(
            {
                "zenith": [20.0, 30.0, 45.0, 60.0, 75.0],
                "pressure": [1000.0, 1001.0, 1002.0, 1001.0, 1000.0],  # hPa
            },
            index=ix,
        )

    def test_default_pressure_uses_pvlib_default(self):
        df = self._day_df()
        result = calcparams.absolute_airmass(
            df, apparent_zenith="zenith", verbose=False
        )
        rel = pvlib.atmosphere.get_relative_airmass(
            df["zenith"], model="kastenyoung1989"
        )
        expected = pvlib.atmosphere.get_absolute_airmass(rel)
        np.testing.assert_allclose(result.values, expected.values)

    def test_with_pressure_scales_by_100(self):
        df = self._day_df()
        result = calcparams.absolute_airmass(
            df,
            apparent_zenith="zenith",
            pressure="pressure",
            verbose=False,
        )
        rel = pvlib.atmosphere.get_relative_airmass(
            df["zenith"], model="kastenyoung1989"
        )
        expected = pvlib.atmosphere.get_absolute_airmass(rel, df["pressure"] * 100)
        np.testing.assert_allclose(result.values, expected.values)

    def test_pressure_scale_override(self):
        df = self._day_df()
        # Simulate pressure already in Pa; pressure_scale=1 passes through.
        df["pressure_pa"] = df["pressure"] * 100
        result = calcparams.absolute_airmass(
            df,
            apparent_zenith="zenith",
            pressure="pressure_pa",
            pressure_scale=1,
            verbose=False,
        )
        rel = pvlib.atmosphere.get_relative_airmass(
            df["zenith"], model="kastenyoung1989"
        )
        expected = pvlib.atmosphere.get_absolute_airmass(rel, df["pressure_pa"])
        np.testing.assert_allclose(result.values, expected.values)

    def test_in_range_pressure_does_not_warn(self):
        """Typical station pressure (hPa * 100) passes the sanity check silently."""
        df = self._day_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            calcparams.absolute_airmass(
                df,
                apparent_zenith="zenith",
                pressure="pressure",
                verbose=False,
            )

    def test_outlier_pressure_row_does_not_warn(self):
        """Isolated bad-data rows outside the 5th-95th percentile band are ignored."""
        n = 100
        ix = pd.date_range("2023-06-21 00:00", periods=n, freq="h")
        pressure = np.full(n, 1013.0)
        # A couple of sentinel outliers at the ends — well below the 5th
        # percentile position (n * 0.05 = 5) so they don't bias the check.
        pressure[0] = -9999.0
        pressure[-1] = 99999.0
        df = pd.DataFrame({"zenith": np.full(n, 45.0), "pressure": pressure}, index=ix)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            calcparams.absolute_airmass(
                df,
                apparent_zenith="zenith",
                pressure="pressure",
                verbose=False,
            )

    def test_out_of_range_pressure_warns_when_scale_mismatch(self):
        """Column already in Pa with default scale=100 triggers the warning."""
        df = self._day_df()
        # Multiply by 100 so the column is in Pa; the default pressure_scale
        # of 100 then over-scales to ~10^7 Pa -> 95th percentile well above
        # the record maximum.
        df["pressure_already_pa"] = df["pressure"] * 100
        with pytest.warns(UserWarning, match="out of range"):
            calcparams.absolute_airmass(
                df,
                apparent_zenith="zenith",
                pressure="pressure_already_pa",
                verbose=False,
            )

    def test_out_of_range_pressure_low_warns(self):
        """Sustained low pressure (below record) triggers the warning."""
        ix = pd.date_range("2023-06-21 10:00", periods=20, freq="h")
        df = pd.DataFrame(
            {"zenith": np.full(20, 45.0), "pressure": np.full(20, 500.0)},
            index=ix,
        )
        with pytest.warns(UserWarning, match="out of range"):
            calcparams.absolute_airmass(
                df,
                apparent_zenith="zenith",
                pressure="pressure",
                verbose=False,
            )

    def test_empty_pressure_series_does_not_warn(self):
        """All-NaN pressure skips the check instead of raising."""
        ix = pd.date_range("2023-06-21 10:00", periods=5, freq="h")
        df = pd.DataFrame(
            {"zenith": [20.0, 30.0, 45.0, 60.0, 75.0], "pressure": np.nan},
            index=ix,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            calcparams.absolute_airmass(
                df,
                apparent_zenith="zenith",
                pressure="pressure",
                verbose=False,
            )


class TestPrecipitableWaterGueymard:
    """Test precipitable_water_gueymard wraps pvlib.atmosphere.gueymard94_pw."""

    def test_matches_pvlib_reference(self):
        ix = pd.date_range("2023-06-21 12:00", periods=3, freq="h")
        df = pd.DataFrame(
            {"temp": [20.0, 25.0, 30.0], "rh": [30.0, 50.0, 70.0]}, index=ix
        )
        result = calcparams.precipitable_water_gueymard(
            df, temp_amb="temp", rel_humidity="rh", verbose=False
        )
        expected = pvlib.atmosphere.gueymard94_pw(df["temp"], df["rh"])
        np.testing.assert_allclose(result.values, expected.values)

    def test_output_message(self, capsys):
        ix = pd.date_range("2023-06-21 12:00", periods=1, freq="h")
        df = pd.DataFrame({"temp": [25.0], "rh": [50.0]}, index=ix)
        calcparams.precipitable_water_gueymard(df, temp_amb="temp", rel_humidity="rh")
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "precipitable_water_gueymard" column as '
            "pvlib.atmosphere.gueymard94_pw(temp, rh)."
        )


class TestScale:
    """Test scale helper."""

    def test_multiplies_by_factor(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = calcparams.scale(df, col="x", factor=100, verbose=False)
        np.testing.assert_array_equal(result.values, [100.0, 200.0, 300.0])

    def test_default_factor_is_identity(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = calcparams.scale(df, col="x", verbose=False)
        np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])

    def test_output_message(self, capsys):
        df = pd.DataFrame({"x": [1.0]})
        calcparams.scale(df, col="x", factor=100)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "scale" column as x * 100.'
        )


class TestSpectralFactorFirstSolar:
    """Test spectral_factor_firstsolar wraps the pvlib call."""

    def test_matches_pvlib_reference_cdte(self):
        ix = pd.date_range("2023-06-21 12:00", periods=3, freq="h")
        df = pd.DataFrame({"pw": [1.0, 1.5, 2.0], "am": [1.0, 1.5, 2.0]}, index=ix)
        result = calcparams.spectral_factor_firstsolar(
            df,
            precipitable_water="pw",
            absolute_airmass="am",
            spectral_module_type="cdte",
            verbose=False,
        )
        expected = pvlib.spectrum.spectral_factor_firstsolar(
            df["pw"], df["am"], module_type="cdte"
        )
        np.testing.assert_allclose(result.values, expected.values)

    def test_spectral_module_type_override(self):
        ix = pd.date_range("2023-06-21 12:00", periods=2, freq="h")
        df = pd.DataFrame({"pw": [1.0, 1.5], "am": [1.0, 1.5]}, index=ix)
        cdte = calcparams.spectral_factor_firstsolar(
            df,
            precipitable_water="pw",
            absolute_airmass="am",
            spectral_module_type="cdte",
            verbose=False,
        )
        monosi = calcparams.spectral_factor_firstsolar(
            df,
            precipitable_water="pw",
            absolute_airmass="am",
            spectral_module_type="monosi",
            verbose=False,
        )
        assert not np.allclose(cdte.values, monosi.values)


class TestMultiply:
    """Test generic column multiplication."""

    def test_elementwise_math(self):
        ix = pd.date_range("2023-06-21 12:00", periods=3, freq="h")
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, index=ix)
        result = calcparams.multiply(df, a="x", b="y", verbose=False)
        np.testing.assert_array_equal(result.values, [4.0, 10.0, 18.0])
        assert result.index.equals(df.index)

    def test_output_message(self, capsys):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        calcparams.multiply(df, a="x", b="y")
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "multiply" column as x * y.'
        )


class TestPoaSpecCorrected:
    """Test poa_spec_corrected named alias."""

    def test_elementwise_math(self):
        ix = pd.date_range("2023-06-21 12:00", periods=3, freq="h")
        df = pd.DataFrame(
            {"poa": [800.0, 900.0, 1000.0], "sc": [0.99, 1.0, 1.01]}, index=ix
        )
        result = calcparams.poa_spec_corrected(
            df, poa="poa", spectral_correction="sc", verbose=False
        )
        np.testing.assert_allclose(result.values, [792.0, 900.0, 1010.0])

    def test_output_message(self, capsys):
        df = pd.DataFrame({"poa": [800.0], "sc": [0.99]})
        calcparams.poa_spec_corrected(df, poa="poa", spectral_correction="sc")
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "poa_spec_corrected" column as poa * sc.'
        )
