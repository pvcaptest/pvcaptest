import os
import collections
import unittest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .context import captest as pvc

data = np.arange(0, 1300, 54.167)
index = pd.DatetimeIndex(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

capdata = pvc.CapData()
capdata.df = df

"""
Run all tests from project root:
'python -m tests.test_CapTest'

Run individual tests:
'python -m unittest tests.test_CapTest.Class.Method'

-m flag imports unittest as module rather than running as script

update_summary
x  perc_wrap
x irrRC_balanced
spans_year
cntg_eoy
x flt_irr
x fit_model
x predict
x pred_summary

CapTest
    summary
    scatter
    reg_scatter_matrix
    sim_apply_losses- blank
    pred_rcs- future
    rep_cond
    x rep_cond(pred=True)
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


class Test_CapTest_cp_results_single_coeff(unittest.TestCase):
    """Tests for the capactiy test results method using a regression formula
    with a single coefficient."""

    def setUp(self):
        np.random.seed(9876789)

        meas = pvc.CapData()
        sim = pvc.CapData()
        self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.cptest.rc = {'x': [6]}

        nsample = 100
        e = np.random.normal(size=nsample)

        x = np.linspace(0, 10, 100)
        das_y = x * 2
        sim_y = x * 2 + 1

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({'y': das_y, 'x': x})
        sim_df = pd.DataFrame({'y': sim_y, 'x': x})

        das_model = smf.ols(formula='y ~ x - 1', data=das_df)
        sim_model = smf.ols(formula='y ~ x - 1', data=sim_df)

        self.cptest.ols_model_das = das_model.fit()
        self.cptest.ols_model_sim = sim_model.fit()

    def test_return(self):
        res = self.cptest.cp_results(100, print_res=False)

        self.assertIsInstance(res,
                              float,
                              'Returned value is not a tuple')


class Test_CapTest_cp_results_mult_coeff(unittest.TestCase):
    """Tests for the capactiy test results method using a regression formula
    with multiple coefficients."""

    def setUp(self):
        np.random.seed(9876789)

        meas = pvc.CapData()
        sim = pvc.CapData()
        self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.cptest.rc = pd.DataFrame({'poa': [6], 't_amb': [5], 'w_vel': [3]})

        nsample = 100
        e = np.random.normal(size=nsample)

        a = np.linspace(0, 10, 100)
        b = np.linspace(0, 10, 100) / 2.0
        c = np.linspace(0, 10, 100) + 3.0

        das_y = a + (a ** 2) + (a * b) + (a * c)
        sim_y = a + (a ** 2 * 0.9) + (a * b * 1.1) + (a * c * 0.8)

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({'power': das_y, 'poa': a, 't_amb': b, 'w_vel': c})
        sim_df = pd.DataFrame({'power': sim_y, 'poa': a, 't_amb': b, 'w_vel': c})

        meas.df = das_df
        meas.set_reg_trans(power='power', poa='poa', t_amb='t_amb', w_vel='w_vel')

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        self.cptest.ols_model_das = das_model.fit()
        self.cptest.ols_model_sim = sim_model.fit()

    def test_pvals_default_false(self):
        self.cptest.cp_results(100, print_res=False)

        self.assertTrue(all(self.cptest.ols_model_das.params.values != 0),
                        'Coefficient was set to zero that should not be zero.')
        self.assertTrue(all(self.cptest.ols_model_sim.params.values != 0),
                        'Coefficient was set to zero that should not be zero.')

    def test_pvals_true_all_below_pval(self):
        self.cptest.cp_results(100, check_pvalues=True, print_res=False)

        self.assertTrue(all(self.cptest.ols_model_das.params.values != 0),
                        'Coefficient was set to zero that should not be zero.')
        self.assertTrue(all(self.cptest.ols_model_sim.params.values != 0),
                        'Coefficient was set to zero that should not be zero.')

    def test_pvals_true(self):
        self.cptest.cp_results(100, check_pvalues=True, pval=1e-15,
                               print_res=False)

        self.assertEqual(self.cptest.ols_model_das.params.values[0],
                         0.0,
                         'Coefficient that should be set to zero was not.')
        self.assertEqual(self.cptest.ols_model_sim.params.values[0],
                         0.0,
                         'Coefficient that should be set to zero was not.')


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
