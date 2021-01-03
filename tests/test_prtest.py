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
