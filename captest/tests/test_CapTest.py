import unittest
import captest as pvc
import numpy as np
import pandas as pd


data = np.arange(0, 1300, 54.167)
index = pd.DatetimeIndex(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

capdata = pvc.CapData()
capdata.df = df

"""
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

class TestFilterIrr(unittest.TestCase):
    """Tests for CapTest class."""

    def test_col_count(self):
        # column count shouldn't change
        # min val after should be >= min
        # max val after should be <= max
        data = np.ndarray()
        pass

    def test_min_val(self):
        pass

    def test_max_val(self):
        pass


if __name__ == '__main__':
    unittest.main()
