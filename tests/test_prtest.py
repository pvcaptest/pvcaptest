import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from .context import prtest as pr

"""
Run tests using pytest use the following from project root.
To run a class of tests
pytest tests/test_CapData.py::TestCapDataEmpty

To run a specific test:
pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty
"""
class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_numeric(self):
        assert isinstance(pr.temp_correct_power(10, -0.37, 50), float)

    def test_output_type_series(self):
        assert isinstance(
            pr.temp_correct_power(pd.Series([10, 12, 15]), -0.37, 50),
            pd.Series
        )

    def test_high_temp_lower_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert corr_power < power

    def test_low_temp_lower_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 10)
        assert corr_power > power

    def test_math_numeric_power(self):
        power = 10
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert corr_power == 9.075

    def test_math_series_power(self):
        ix = pd.date_range(
            start='1/1/2021 12:00',
            freq='H',
            periods=3
        )
        power = pd.Series([10, 20, 15], index=ix)
        corr_power = pr.temp_correct_power(power, -0.37, 50)
        assert pd.testing.assert_series_equal(
            corr_power,
            pd.Series([9.075, 18.15, 13.6125], index=ix)
        ) is None

    def test_no_temp_diff(self):
        assert pr.temp_correct_power(10, -0.37, 25) == 10

    def test_user_base_temp(self):
        corr_power = pr.temp_correct_power(10, -0.37, 30, base_temp=27.5)
        assert corr_power == 9.9075

class TestBackOfModuleTemp:
    """Test calculation of back of module (BOM) temperature from weather."""
    def test_float_inputs(self):
        assert pr.back_of_module_temp(800, 30, 3) == pytest.approx(48.1671)

    def test_series_inputs(self):
        ix = pd.date_range(
            start='1/1/2021 12:00',
            freq='H',
            periods=3
        )
        poa = pd.Series([805, 810, 812], index=ix)
        temp_amb = pd.Series([26, 27, 27.5], index=ix)
        wind = pd.Series([0.5, 1, 2.5], index=ix)

        exp_results = pd.Series(
            [48.0506544,
             48.3709869,
             46.6442104],
             index=ix
        )

        assert pd.testing.assert_series_equal(
            pr.back_of_module_temp(poa, temp_amb, wind),
            exp_results
        ) is None

    @pytest.mark.parametrize(
        'racking, module_type, expected',
        [
            ('open_rack', 'glass_cell_glass', 50.77154),
            ('open_rack', 'glass_cell_poly', 48.33028),
            ('open_rack', 'poly_tf_steel', 46.82361),
            ('close_roof_mount', 'glass_cell_glass', 65.86252),
            ('insulated_back', 'glass_cell_poly', 72.98647)
        ]
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        bom = pr.back_of_module_temp(
            800,
            28,
            1.5,
            module_type=module_type,
            racking=racking
        )
        assert bom == pytest.approx(expected)
