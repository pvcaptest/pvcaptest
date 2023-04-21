import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from captest import util

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

@pytest.fixture
def reindex_dfs():
    df1 = pd.DataFrame(
        {'a': np.full(4, 5)},
        index=pd.date_range(start='1/1/21', end='1/1/21 00:15', freq='5min'),
    )
    df2 = pd.DataFrame(
        {'a': np.full(4, 5)},
        index=pd.date_range(start='1/1/21 00:30', end='1/1/21 00:45', freq='5min'),
    )
    return (df1, df2)

class TestReindexDatetime():
    def test_adds_missing_intervals(self, reindex_dfs):
        """Check that missing intervals in the index are added to the dataframe."""
        df1, df2 = reindex_dfs
        df = pd.concat([df1, df2])
        (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(
            df, add_index_col=False
        )
        assert df_reindexed.shape[0] == 10

        df = pd.concat([df2, df1]) # reverse order, check sorting
        (df_reindexed, missing_intervals, str) = util.reindex_datetime(
            df, add_index_col=False
        )
        assert df_reindexed.shape[0] == 10

    def test_adds_index_column(self, reindex_dfs):
        """Check that a string representation of the datetime index is added."""
        df1, df2 = reindex_dfs
        df = pd.concat([df2, df1])
        # df_reindexed = util.reindex_datetime(df, add_index_col=True)
        (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(
            df, add_index_col=True
        )
        assert df_reindexed.shape[1] == 2
        assert isinstance((df_reindexed.loc[:, 'index'][0]), str)
        datetime_str = df_reindexed.iloc[0, 1]
        date_str = datetime_str.split(' ')[0]
        assert len(date_str.split('/')[0]) == 2
        assert len(date_str.split('/')[1]) == 2
        assert len(date_str.split('/')[2]) == 4

    def test_isinstance_str(self):
        assert isinstance('test string', str)

    def test_report_output(self, reindex_dfs):
        """Check that a string representation of the datetime index is added."""
        df1, df2 = reindex_dfs
        df = pd.concat([df2, df1])
        df_reindexed, missing_int, freq = util.reindex_datetime(
            df, report=True, add_index_col=True
        )
        assert df_reindexed.shape[0] == 10
        assert missing_int == 2
        assert freq == '5min'
