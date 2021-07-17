import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from .context import util

ix = pd.date_range(
    start='1/1/2021 12:00',
    freq='H',
    periods=3
)

ix_5min = pd.date_range(
    start='1/1/2021 12:00',
    freq='5min',
    periods=3
)

class TestGetCommonTimestep():
    def test_output_type_str(self):
        df = pd.DataFrame({'a':[1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units='h', string_output=True)
        assert isinstance(time_step, str)

    def test_output_type_numeric(self):
        df = pd.DataFrame({'a':[1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units='h', string_output=False)
        assert isinstance(time_step, np.float64)

    def test_hours_string(self):
        df = pd.DataFrame({'a':[1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units='h', string_output=True)
        assert time_step == '1H'

    def test_hours_numeric(self):
        df = pd.DataFrame({'a':[1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units='h', string_output=False)
        assert time_step == 1.0

    def test_minutes_numeric(self):
        df = pd.DataFrame({'a':[1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units='m', string_output=False)
        assert time_step == 60.0

    def test_mixed_intervals(self):
        df = pd.DataFrame(
            {'a':np.ones(120)},
            index=pd.date_range(start='1/1/21', freq='1min', periods=120)
        )
        assert df.index.is_monotonic_increasing
        print(df)
        df_gaps = pd.concat([
            df.loc['1/1/21':'1/1/21 00:10', :],
            df.loc['1/1/21 00:15':'1/1/21 00:29', :],
            df.loc['1/1/21 00:31':, :],
        ])
        assert df_gaps.shape[0] < df.shape[0]
        time_step = util.get_common_timestep(df_gaps, units='m', string_output=False)
        assert time_step == 1
        time_step = util.get_common_timestep(df_gaps, units='m', string_output=True)
        assert time_step == '1min'
