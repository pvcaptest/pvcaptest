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

class Test_top_level_funcs(unittest.TestCase):
    def test_perc_wrap(self):
        """Test percent wrap function."""
        rng = np.arange(1, 100, 1)
        rng_cpy = rng.copy()
        df = pd.DataFrame({'vals': rng})
        df_cpy = df.copy()
        bool_array = []
        for val in rng:
            np_perc = np.percentile(rng, val, interpolation='nearest')
            wrap_perc = df.agg(pvc.perc_wrap(val)).values[0]
            bool_array.append(np_perc == wrap_perc)
        self.assertTrue(all(bool_array),
                        'np.percentile wrapper gives different value than np perc')
        self.assertTrue(all(df == df_cpy), 'perc_wrap function modified input df')

    def test_flt_irr(self):
        rng = np.arange(0, 1000)
        df = pd.DataFrame(np.array([rng, rng+100, rng+200]).T,
                          columns = ['weather_station irr poa W/m^2',
                                     'col_1', 'col_2'])
        df_flt = pvc.flt_irr(df, 'weather_station irr poa W/m^2', 50, 100)

        self.assertEqual(df_flt.shape[0], 51,
                         'Incorrect number of rows returned from filter.')
        self.assertEqual(df_flt.shape[1], 3,
                         'Incorrect number of columns returned from filter.')
        self.assertEqual(df_flt.columns[0], 'weather_station irr poa W/m^2',
                      'Filter column name inadverdently modified by method.')
        self.assertEqual(df_flt.iloc[0, 0], 50,
                         'Minimum value in returned data in filter column is'
                         'not equal to low argument.')
        self.assertEqual(df_flt.iloc[-1, 0], 100,
                         'Maximum value in returned data in filter column is'
                         'not equal to high argument.')

    def test_fit_model(self):
        """Test fit model func which wraps statsmodels ols.fit for dataframe."""
        rng = np.random.RandomState(1)
        x = 50 * abs(rng.rand(50))
        y = 2 * x - 5 + 5 * rng.randn(50)
        df = pd.DataFrame({'x': x, 'y': y})
        fml = 'y ~ x - 1'
        passed_ind_vars = fml.split('~')[1].split()[::2]
        try:
            passed_ind_vars.remove('1')
        except ValueError:
            pass

        reg = pvc.fit_model(df, fml=fml)

        for var in passed_ind_vars:
            self.assertIn(var, reg.params.index,
                          '{} ind variable in formula argument not in model'
                          'parameters'.format(var))

    def test_predict(self):
        x = np.arange(0, 50)
        y1 = x
        y2 = x * 2
        y3 = x * 10

        dfs = [pd.DataFrame({'x': x, 'y': y1}),
               pd.DataFrame({'x': x, 'y': y2}),
               pd.DataFrame({'x': x, 'y': y3})]

        reg_lst = []
        for df in dfs:
            reg_lst.append(pvc.fit_model(df, fml='y ~ x'))
        reg_ser = pd.Series(reg_lst)

        for regs in [reg_lst, reg_ser]:
            preds = pvc.predict(regs, pd.DataFrame({'x': [10, 10, 10]}))
            self.assertAlmostEqual(preds.iloc[0], 10, 7, 'Pred for x = y wrong.')
            self.assertAlmostEqual(preds.iloc[1], 20, 7, 'Pred for x = y * 2 wrong.')
            self.assertAlmostEqual(preds.iloc[2], 100, 7, 'Pred for x = y * 10 wrong.')
            self.assertEqual(3, preds.shape[0], 'Each of the three input'
                                                'regressions should have a'
                                                'prediction')

    def test_pred_summary(self):
        """Test aggregation of reporting conditions and predicted results."""
        """
        grpby -> df of regressions
        regs -> series of predicted values
        df of reg parameters
        """
        pvsyst = pvc.CapData()
        pvsyst.load_data(path='./tests/data/', load_pvsyst=True)

        df_regs = pvsyst.df.loc[:, ['E_Grid', 'GlobInc', 'TAmb', 'WindVel']]
        df_regs_day = df_regs.query('GlobInc > 0')
        grps = df_regs_day.groupby(pd.Grouper(freq='M', label='right'))

        ones = np.ones(12)
        irr_rc = ones * 500
        temp_rc = ones * 20
        w_vel = ones
        rcs = pd.DataFrame({'GlobInc': irr_rc, 'TAmb': temp_rc, 'WindVel': w_vel})

        results = pvc.pred_summary(grps, rcs, 0.05,
                                   fml='E_Grid ~ GlobInc +'
                                                 'I(GlobInc * GlobInc) +'
                                                 'I(GlobInc * TAmb) +'
                                                 'I(GlobInc * WindVel) - 1')

        self.assertEqual(results.shape[0], 12, 'Not all months in results.')
        self.assertEqual(results.shape[1], 10, 'Not all cols in results.')

        self.assertIsInstance(results.index,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Index is not pandas DatetimeIndex')

        col_length = len(results.columns.values)
        col_set_length = len(set(results.columns.values))
        self.assertEqual(col_set_length, col_length,
                         'There is a duplicate column name in the results df.')

        pt_qty_exp = [341, 330, 392, 390, 403, 406,
                           456, 386, 390, 346, 331, 341]
        gaur_cap_exp = [3089550.4039329495, 3103610.4635679387,
                        3107035.251399103, 3090681.1145782764,
                        3058186.270209293, 3059784.2309170915,
                        3088294.50827525, 3087081.0026879036,
                        3075251.990424683, 3093287.331878834,
                        3097089.7852036236, 3084318.093294242]
        for i, mnth in enumerate(results.index):
            self.assertLess(results.loc[mnth, 'guaranteedCap'],
                            results.loc[mnth, 'PredCap'],
                            'Gauranteed capacity is greater than predicted in'
                            'month {}'.format(mnth))
            self.assertGreater(results.loc[mnth, 'guaranteedCap'], 0,
                               'Gauranteed capacity is less than 0 in'
                               'month {}'.format(mnth))
            self.assertAlmostEqual(results.loc[mnth, 'guaranteedCap'],
                                   gaur_cap_exp[i], 7,
                                   'Gauranted capacity not equal to expected'
                                   'value in {}'.format(mnth))
            self.assertEqual(results.loc[mnth, 'pt_qty'], pt_qty_exp[i],
                               'Point quantity not equal to expected values in'
                               '{}'.format(mnth))




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


class Test_CapTest_filters(unittest.TestCase):
    """
    Tests for filtering methods.
    """
    def test_filter_clearsky(self):
        pvsyst = pvc.CapData()
        meas = pvc.CapData()
        loc = {'latitude': 39.742, 'longitude': -105.18,
               'altitude': 1828.8, 'tz': 'Etc/GMT+7'}
        sys = {'surface_tilt': 40, 'surface_azimuth': 180,
               'albedo': 0.2}
        meas.load_data(path='./tests/data/', fname='nrel_data.csv',
                       source='AlsoEnergy', clear_sky=True, loc=loc, sys=sys)
        self.cptest = pvc.CapTest(meas, pvsyst, '+/- 5')

        self.cptest.filter_clearsky('das')

        self.assertLess(self.cptest.flt_das.df.shape[0],
                        self.cptest.das.df.shape[0],
                        'Filtered dataframe should have less rows.')
        self.assertEqual(self.cptest.flt_das.df.shape[1],
                         self.cptest.das.df.shape[1],
                         'Filtered dataframe should have equal number of cols.')
        for i, col in enumerate(self.cptest.flt_das.df.columns):
            self.assertEqual(col, self.cptest.das.df.columns[i],
                             'Filter changed column {} to '
                             '{}'.format(self.cptest.das.df.columns[i], col))

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
