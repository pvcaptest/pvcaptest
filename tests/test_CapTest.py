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
x  perc_wrap
irrRC_balanced
spans_year
cntg_eoy
flt_irr
fit_model
predict
pred_summary

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


class Test_top_level_funcs(unittest.TestCase):
    """Test percent wrap function."""
    def test_perc_wrap(self):
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


class Test_CapData_methods_sim(unittest.TestCase):
    """Test for top level irrRC_balanced function."""

    def setUp(self):
        self.pvsyst = pvc.CapData()
        self.pvsyst.load_data(directory='./tests/data/', load_pvsyst=True)
        # self.jun = self.pvsyst.df.loc['06/1990']
        # self.jun_cpy = self.jun.copy()
        # self.low = 0.5
        # self.high = 1.5
        # (self.irr_RC, self.jun_flt) = pvc.irrRC_balanced(self.jun, self.low,
        #                                                  self.high)
        # self.jun_flt_irr = self.jun_flt['GlobInc']

    def test_irrRC_balanced(self):
        jun = self.pvsyst.df.loc['06/1990']
        jun_cpy = jun.copy()
        low = 0.5
        high = 1.5
        (irr_RC, jun_flt) = pvc.irrRC_balanced(jun, low, high)
        jun_flt_irr = jun_flt['GlobInc']
        self.assertTrue(all(jun_flt.columns == jun.columns),
                        'Columns of input df missing in filtered ouput df.')
        self.assertGreater(jun_flt.shape[0], 0,
                           'Returned df has no rows')
        self.assertLess(jun_flt.shape[0], jun.shape[0],
                        'No rows removed from filtered df.')
        self.assertTrue(jun.equals(jun_cpy),
                        'Input dataframe modified by function.')
        self.assertGreater(irr_RC, jun[jun['GlobInc'] > 0]['GlobInc'].min(),
                           'Reporting irr not greater than min irr in input data')
        self.assertLess(irr_RC, jun['GlobInc'].max(),
                        'Reporting irr no less than max irr in input data')

        pts_below_irr = jun_flt_irr[jun_flt_irr.between(0, irr_RC)].shape[0]
        perc_below = pts_below_irr / jun_flt_irr.shape[0]
        self.assertLess(perc_below, 0.6,
                        'More than 60 percent of points below reporting irr')
        self.assertGreaterEqual(perc_below, 0.5,
                                'Less than 50 percent of points below rep irr')

        pts_above_irr = jun_flt_irr[jun_flt_irr.between(irr_RC, 1500)].shape[0]
        perc_above = pts_above_irr / jun_flt_irr.shape[0]
        self.assertGreater(perc_above, 0.4,
                           'Less than 40 percent of points above reporting irr')
        self.assertLessEqual(perc_above, 0.5,
                             'More than 50 percent of points above reportin irr')

    def test_rep_cond_pred(self):
        """Test prediction option of reporting conditions method."""
        vals = self.pvsyst.trans['--']
        self.pvsyst.trans['--'] = vals[:-1]
        self.pvsyst.trans['irr-ghi-'] = ['GlobInc']
        self.pvsyst.set_reg_trans(poa='irr-ghi-', power='real_pwr--',
                                  w_vel='wind--', t_amb='temp-amb-')
        meas = pvc.CapData()
        cptest = pvc.CapTest(meas, self.pvsyst, 0.5)
        results = cptest.rep_cond('sim', 0.8, 1.2, inplace=False, freq='M',
                                  pred=True)

        self.assertEqual(results.shape[0], 12, 'Not all months in results.')
        self.assertEqual(results.shape[1], 10, 'Not all cols in results.')

        col_names = ['poa', 'w_vel', 't_amb', 'PredCap', 'poa_coef',
                     'I(poa * poa)', 'I(poa * t_amb)', 'I(poa * w_vel)',
                     'guaranteedCap', 'pt_qty']
        for name in col_names:
            self.assertIn(name, results.columns,
                          '{} column is not in results.'.format(name))
        self.assertIsInstance(results.index,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Index is not pandas DatetimeIndex')

        # Check irradiance values for each month
        for val in results.index:
            mnth_str = val.strftime('%m/%Y')
            df_irr = self.pvsyst.df['GlobInc']
            irr_result = results['poa'].loc[mnth_str][0]
            np_result = np.percentile(df_irr.loc[mnth_str], 60,
                                      interpolation='nearest')
            self.assertEqual(np_result, irr_result,
                             'The 60th percentile from function does not match '
                             'numpy percentile for {}'.format(mnth_str))

            df_w_vel = self.pvsyst.df['WindVel']
            w_result = results['w_vel'].loc[mnth_str][0]
            w_result_pd = df_w_vel.loc[mnth_str].mean()
            self.assertEqual(w_result_pd, w_result,
                             'The average wind speed result does not match '
                             'pandas aveage for {}'.format(mnth_str))

            df_t_amb = self.pvsyst.df['TAmb']
            t_amb_result = results['t_amb'].loc[mnth_str][0]
            t_amb_result_pd = df_t_amb.loc[mnth_str].mean()
            self.assertEqual(t_amb_result_pd, t_amb_result,
                             'The average amb temp result does not match '
                             'pandas aveage for {}'.format(mnth_str))




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
