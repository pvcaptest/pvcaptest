import os
import sys
import unittest
import numpy as np
import pandas as pd

from .context import captest as pvc

data = np.arange(0, 1300, 54.167)
index = pd.DatetimeIndex(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

capdata = pvc.CapData()
capdata.df = df

"""
Run test from project root with 'python -m tests.test_CapTest'


update_summary
CapData
    set_reg_trans
    copy
    empty
    load_das
    load_pvsyst
    load_data
    __series_type
    __set_trans
    drop_cols
    view
    rview
    plot
CapTest
    summary
    scatter
    reg_scatter_matrix
    sim_apply_losses- blank
    pred_rcs- future
    rep_cond
    agg_sensors
    reg_data
    __flt_setup
    reset_flt
    filter_outliers
    filter_pf
    filter_irr
    filter_op_state
    filter_missing
    __std_filter
    __sensor_filter
    filter_sensors
    reg_cpt
    cp_results
    equip_counts- not used
"""

test_files = ['test1.csv', 'test2.csv', 'test3.CSV', 'test4.txt',
              'pvsyst.csv', 'pvsyst_data.csv']


class TestCapDataLoadMethods(unittest.TestCase):
    """Tests for load_data method."""

    def setUp(self):
        os.mkdir('test_csvs')
        for fname in test_files:
            with open('test_csvs/' + fname, 'a') as f:
                f.write('Date, val\n11/21/2017, 1')

        self.capdata = pvc.CapData()
        self.capdata.load_data(directory='test_csvs/', set_trans=False)

    def tearDown(self):
        for fname in test_files:
            os.remove('test_csvs/' + fname)
        os.rmdir('test_csvs')

    def test_read_csvs(self):
        self.assertEqual(self.capdata.df.shape[0], 3,
                         'imported a non csv or pvsyst file')


# class TestFilterIrr(unittest.TestCase):
#     """Tests for CapTest class."""
#
#     def test_col_count(self):
#         # column count shouldn't change
#         # min val after should be >= min
#         # max val after should be <= max
#         data = np.ndarray()
#         pass
#
#     def test_min_val(self):
#         pass
#
#     def test_max_val(self):
#         pass


if __name__ == '__main__':
    unittest.main()
