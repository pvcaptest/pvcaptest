import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from captest import prtest as pr

"""
Run tests using pytest use the following from project root.
To run a class of tests
pytest tests/test_CapData.py::TestCapDataEmpty

To run a specific test:
pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty
"""

ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)

ix_5min = pd.date_range(start="1/1/2021 12:00", freq="5min", periods=3)


class TestGetCommonTimestep:
    def test_output_type_str(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = pr.get_common_timestep(df, units="h", string_output=True)
        assert isinstance(time_step, str)

    def test_output_type_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = pr.get_common_timestep(df, units="h", string_output=False)
        assert isinstance(time_step, np.float64)

    def test_hours_string(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = pr.get_common_timestep(df, units="h", string_output=True)
        assert time_step == "1.0 hours"

    def test_hours_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = pr.get_common_timestep(df, units="h", string_output=False)
        assert time_step == 1.0

    def test_minutes_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = pr.get_common_timestep(df, units="m", string_output=False)
        assert time_step == 60.0


class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_numeric(self):
        assert isinstance(pr.temp_correct_power(10, -0.37, 50), float)

    def test_output_type_series(self):
        assert isinstance(
            pr.temp_correct_power(pd.Series([10, 12, 15]), -0.37, 50), pd.Series
        )

    def test_high_temp_higher_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert corr_power > power

    def test_low_temp_lower_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 10)
        assert corr_power < power

    def test_math_numeric_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert pytest.approx(corr_power, 0.3) == 11.019

    def test_math_series_power(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)
        power = pd.Series([10, 20, 15], index=ix)
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert pytest.approx(corr_power.values, 0.3) == [11.019, 22.038, 16.528]

    def test_no_temp_diff(self):
        assert pr.temp_correct_power(10, -0.37, 25) == 10

    def test_user_base_temp(self):
        corr_power = pr.temp_correct_power(10, -0.37, 30, base_temp=27.5)
        assert pytest.approx(corr_power, 0.3) == 10.093


class TestBackOfModuleTemp:
    """Test calculation of back of module (BOM) temperature from weather."""

    def test_float_inputs(self):
        assert pr.back_of_module_temp(800, 30, 3) == pytest.approx(48.1671)

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_amb = pd.Series([26, 27, 27.5], index=ix)
        wind = pd.Series([0.5, 1, 2.5], index=ix)

        exp_results = pd.Series([48.0506544, 48.3709869, 46.6442104], index=ix)

        assert (
            pd.testing.assert_series_equal(
                pr.back_of_module_temp(poa, temp_amb, wind), exp_results
            )
            is None
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
        bom = pr.back_of_module_temp(
            800, 28, 1.5, module_type=module_type, racking=racking
        )
        assert bom == pytest.approx(expected)


class TestCellTemp:
    def test_float_inputs(self):
        assert pr.cell_temp(30, 850) == pytest.approx(32.55)

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_bom = pd.Series([26, 27, 27.5], index=ix)

        exp_results = pd.Series([28.415, 29.43, 29.936], index=ix)

        assert (
            pd.testing.assert_series_equal(pr.cell_temp(temp_bom, poa), exp_results)
            is None
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", 30.4),
            ("open_rack", "glass_cell_poly", 30.4),
            ("open_rack", "poly_tf_steel", 30.4),
            ("close_roof_mount", "glass_cell_glass", 28.8),
            ("insulated_back", "glass_cell_poly", 28),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        bom = pr.cell_temp(28, 800, module_type=module_type, racking=racking)
        assert bom == pytest.approx(expected)


class TestAvgTypCellTemp:
    def test_math(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        cell_temp = pd.Series([26, 27, 27.5], index=ix)

        assert pr.avg_typ_cell_temp(poa, cell_temp) == pytest.approx(26.8356)


class TestCheckPerfRatioInputs:
    def test_ok_inputs(self):
        ac_energy = pd.Series({"energy": [90, 95, 97]}, index=ix)
        poa = pd.Series([805, 810, 812], index=ix)
        input_ok = pr.perf_ratio_inputs_ok(ac_energy, 110, poa)
        assert input_ok is True

    def test_warn_ac_energy_type(self):
        """Raise warning if `ac_energy` is not a Pandas Series."""
        ac_energy = pd.DataFrame({"energy": [90, 95, 97]}, index=ix)
        poa = pd.Series([805, 810, 812], index=ix)
        with pytest.warns(UserWarning):
            input_ok = pr.perf_ratio_inputs_ok(ac_energy, 110, poa)
        assert input_ok is False

    def test_warn_poa_type(self):
        """Raise warning if `poa` is not a Pandas Series."""
        ac_energy = pd.Series([90, 95, 97], index=ix)
        poa = pd.DataFrame({"poa": [805, 810, 812]}, index=ix)
        with pytest.warns(UserWarning):
            input_ok = pr.perf_ratio_inputs_ok(ac_energy, 110, poa)
        assert input_ok is False

    def test_poa_ac_energy_index_match(self):
        """Raise warning if indices of poa and ac_energy do not match."""
        ix_poa = pd.date_range(start="1/1/2021 13:00", freq="H", periods=3)
        ac_energy = pd.Series([90, 95, 97], index=ix)
        poa = pd.Series([805, 810, 812], index=ix_poa)
        with pytest.warns(UserWarning):
            input_ok = pr.perf_ratio_inputs_ok(ac_energy, 110, poa)
        assert input_ok is False

    def test_avail_index_match(self):
        """Raise warning if index of availability does not match poa."""
        ix_availability = pd.date_range(start="1/1/2021 13:00", freq="H", periods=3)
        ac_energy = pd.Series([90, 95, 97], index=ix)
        poa = pd.Series([805, 810, 812], index=ix)
        avail = pd.Series([0.9, 1, 0.95], index=ix_availability)
        with pytest.warns(UserWarning):
            pr.perf_ratio(ac_energy, 110, poa, availability=avail)


class TestPerfRatio:
    def test_simple_pr_hourly(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC

        perf_ratio = pr.perf_ratio(ac_energy, dc_nameplate, poa)
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr > 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)

    def test_simple_pr_hourly_unit_adj(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # kWh for 3 hours
        ac_energy = pd.Series([80, 90, 95], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC

        perf_ratio = pr.perf_ratio(ac_energy, dc_nameplate, poa, unit_adj=1000)
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr >= 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)

    def test_simple_pr_5min(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix_5min)
        ac_energy = ac_energy / 12  # convert to Wh for 5min intervals
        poa = pd.Series([850, 900, 1000], index=ix_5min)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC

        perf_ratio = pr.perf_ratio(ac_energy, dc_nameplate, poa)
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr > 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)

    def test_simple_pr_hourly_int_avail(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC

        perf_ratio = pr.perf_ratio(ac_energy, dc_nameplate, poa, availability=0.9)
        assert perf_ratio.pr == pytest.approx(0.8922556)

    def test_simple_pr_hourly_series_avail(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC
        avail = pd.Series([0.9, 1, 0.95], index=ix)

        perf_ratio = pr.perf_ratio(ac_energy, dc_nameplate, poa, availability=avail)
        assert perf_ratio.pr == pytest.approx(0.844487)

    @pytest.mark.parametrize(
        "degrad, year, expected",
        [
            (0.5, 1, 0.807065),
            (0.5, 2, 0.811121),
            (0.5, 3, 0.815197),
            (0.7, 1, 0.808691),
            (0.7, 2, 0.814392),
            (0.7, 3, 0.820132),
        ],
    )
    def test_simple_pr_hourly_degrad(self, degrad, year, expected):
        """Test degradation applied to PR denominator.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC

        perf_ratio = pr.perf_ratio(
            ac_energy, dc_nameplate, poa, year=year, degradation=degrad
        )
        assert perf_ratio.pr == pytest.approx(expected)


class TestPerfRatioTempCorrNREL:
    def test_simple_pr_hourly(self):
        """Test a short series of data for a hypothetical system.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC
        temp_amb = pd.Series([30, 32, 34], index=ix)
        wind_speed = pd.Series([1, 1.5, 0.8], index=ix)
        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.37,
            temp_amb=temp_amb,
            wind_speed=wind_speed,
        )
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr > 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)


class TestPrResults:
    """Test the print statements of the print_pr_result method of the PerfRatio class."""
    def test_passing_test(self, capsys):
        """Test that the print statement is correct for a passing test."""
        perf_ratio = pr.PrResults(pr=0.8, expected_pr=0.78)
        perf_ratio.print_pr_result()
        captured = capsys.readouterr()
        assert captured.out == (
                "The test is PASSING with a measured PR of 80.00, "
                "which is 2.00 above the expected PR of 78.00\n"
        )

    def test_failing_test(self, capsys):
        """Test that the print statement is correct for a passing test."""
        perf_ratio = pr.PrResults(pr=0.78, expected_pr=0.8)
        perf_ratio.print_pr_result()
        captured = capsys.readouterr()
        assert captured.out == (
                "The test is FAILING with a measured PR of 78.00, "
                "which is 2.00 below the expected PR of 80.00\n"
        )