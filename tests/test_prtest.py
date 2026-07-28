import numpy as np
import pandas as pd
import pytest

from captest import prtest as pr

"""
Run tests using pytest use the following from project root.
To run a class of tests
pytest tests/test_CapData.py::TestCapDataEmpty

To run a specific test:
pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty
"""

ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)

ix_5min = pd.date_range(start="1/1/2021 12:00", freq="5min", periods=3)


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
        ix_poa = pd.date_range(start="1/1/2021 13:00", freq="h", periods=3)
        ac_energy = pd.Series([90, 95, 97], index=ix)
        poa = pd.Series([805, 810, 812], index=ix_poa)
        with pytest.warns(UserWarning):
            input_ok = pr.perf_ratio_inputs_ok(ac_energy, 110, poa)
        assert input_ok is False

    def test_avail_index_match(self):
        """Raise warning if index of availability does not match poa."""
        ix_availability = pd.date_range(start="1/1/2021 13:00", freq="h", periods=3)
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
        assert perf_ratio.pr == pytest.approx(0.8030, rel=1e-2)

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
        assert perf_ratio.pr == pytest.approx(0.8030, rel=1e-2)

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
        assert perf_ratio.pr == pytest.approx(0.8030, rel=1e-2)

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

    def test_pr_meas_bom(self):
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
        temp_bom = pd.Series([30, 32, 34], index=ix)
        wind_speed = pd.Series([1, 1.5, 0.8], index=ix)
        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.37,
            temp_bom=temp_bom,
            wind_speed=wind_speed,
        )
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr > 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)

    def test_irr_weighted_cell_temp_matches_per_interval_correction(self):
        """Verify per-interval correction equals correcting at the weighted mean temp.

        This is the property that made the removed `single_irr_weighted_temp` option
        redundant: the correction is linear in cell temperature and each interval is
        weighted by its POA irradiance, so collapsing the per-interval cell
        temperatures to their irradiance-weighted mean leaves the summed expected DC,
        and therefore the reported PR, unchanged.
        """
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)
        dc_nameplate = 120_000
        temp_bom = pd.Series([30, 32, 34], index=ix)
        power_temp_coeff = -0.37

        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=power_temp_coeff,
            temp_bom=temp_bom,
        )

        # Correct once at the irradiance-weighted mean cell temperature instead.
        temp_cell = temp_bom + (poa / 1000) * 3  # del_tcnd of the default module
        temp_cell_weighted = (poa * temp_cell).sum() / poa.sum()
        nameplate_weighted = dc_nameplate * (
            1 + (power_temp_coeff / 100) * (temp_cell_weighted - 25)
        )
        expected_dc_weighted = nameplate_weighted * poa / 1000
        pr_weighted = ac_energy.sum() / expected_dc_weighted.sum()

        assert perf_ratio.pr == pytest.approx(pr_weighted)

    def test_single_irr_weighted_temp_argument_removed(self):
        """Verify the removed `single_irr_weighted_temp` option is rejected."""
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)
        temp_bom = pd.Series([30, 32, 34], index=ix)

        with pytest.raises(TypeError):
            pr.perf_ratio_temp_corr_nrel(
                ac_energy,
                120_000,
                poa,
                power_temp_coeff=-0.37,
                temp_bom=temp_bom,
                single_irr_weighted_temp=True,
            )

    def test_pr_meas_bom_and_amb(self):
        """Test a short series of data for a hypothetical system.

        Providing too many inputs - BOM temp and amb temp plus wind speed.

        System specs:
        - ac nameplate: 100 kW
        - dc/ac ratio: 1.2
        - dc nameplate: 120 kW-DC
        """
        # Wh for 3 hours
        ac_energy = pd.Series([80_000, 90_000, 95_000], index=ix)
        poa = pd.Series([850, 900, 1000], index=ix)  # poa W/m^2
        dc_nameplate = 120_000  # W-DC
        temp_bom = pd.Series([30, 32, 34], index=ix)
        temp_amb = pd.Series([30, 32, 34], index=ix)
        wind_speed = pd.Series([1, 1.5, 0.8], index=ix)
        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.37,
            temp_bom=temp_bom,
            temp_amb=temp_amb,
            wind_speed=wind_speed,
        )
        assert perf_ratio.pr <= 1
        assert perf_ratio.pr > 0
        assert isinstance(perf_ratio.timestep[0], np.float64)
        assert isinstance(perf_ratio.timestep[1], str)
        assert perf_ratio.dc_nameplate == dc_nameplate
        assert isinstance(perf_ratio.results_data, pd.DataFrame)

    def test_expected_dc_derated_when_cells_above_base_temp(self):
        """Verify hot cells reduce the temperature-corrected nameplate.

        Constant 1000 W/m^2 POA and a 45 C BOM temp give a cell temp of
        45 + (1000 / 1000) * 3 = 48 C for the default open rack, glass/cell/poly
        module. With a -0.40 %/C coefficient the correction factor is
        1 + (-0.0040 * (48 - 25)) = 0.908, so the 100 kW-DC nameplate corrects
        down to 90.8 kW-DC and each hourly interval expects 90,800 Wh.
        """
        ac_energy = pd.Series([80_000, 82_000, 84_000], index=ix)
        poa = pd.Series([1000, 1000, 1000], index=ix)
        dc_nameplate = 100_000
        temp_bom = pd.Series([45, 45, 45], index=ix)

        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.40,
            temp_bom=temp_bom,
        )

        np.testing.assert_allclose(
            perf_ratio.results_data["expected_dc"].values, [90_800.0] * 3
        )
        assert perf_ratio.pr == pytest.approx(246_000 / 272_400)

    def test_expected_dc_uprated_when_cells_below_base_temp(self):
        """Verify cold cells increase the temperature-corrected nameplate.

        Constant 500 W/m^2 POA and a 5 C BOM temp give a cell temp of
        5 + (500 / 1000) * 3 = 6.5 C. With a -0.40 %/C coefficient the correction
        factor is 1 + (-0.0040 * (6.5 - 25)) = 1.074, so the 100 kW-DC nameplate
        corrects up to 107.4 kW-DC and each hourly interval expects 53,700 Wh.
        """
        ac_energy = pd.Series([40_000, 41_000, 42_000], index=ix)
        poa = pd.Series([500, 500, 500], index=ix)
        dc_nameplate = 100_000
        temp_bom = pd.Series([5, 5, 5], index=ix)

        perf_ratio = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.40,
            temp_bom=temp_bom,
        )

        np.testing.assert_allclose(
            perf_ratio.results_data["expected_dc"].values, [53_700.0] * 3
        )
        assert perf_ratio.pr == pytest.approx(123_000 / 161_100)

    def test_hot_cells_raise_pr_above_uncorrected_pr(self):
        """Verify weather correcting removes the thermal penalty from the PR.

        Cell temperatures above the base temperature lower the expected DC, so
        the temperature-corrected PR must exceed the uncorrected PR for the same
        measured energy.
        """
        ac_energy = pd.Series([80_000, 82_000, 84_000], index=ix)
        poa = pd.Series([1000, 1000, 1000], index=ix)
        dc_nameplate = 100_000
        temp_bom = pd.Series([45, 45, 45], index=ix)

        pr_corrected = pr.perf_ratio_temp_corr_nrel(
            ac_energy,
            dc_nameplate,
            poa,
            power_temp_coeff=-0.40,
            temp_bom=temp_bom,
        )
        pr_uncorrected = pr.perf_ratio(ac_energy, dc_nameplate, poa)

        assert pr_corrected.pr > pr_uncorrected.pr


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
