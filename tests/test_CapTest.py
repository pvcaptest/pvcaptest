import os
import collections
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

CapData
    set_reg_trans- no test needed
    x copy
    empty
    x load_das
    x load_pvsyst
    x load_data
    x __series_type
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

test_files = ['test1.csv', 'test2.csv', 'test3.CSV', 'test4.txt',
              'pvsyst.csv', 'pvsyst_data.csv']

class TestLoadDataMethods(unittest.TestCase):
    """Test for load data methods without setup."""

    def test_load_pvsyst(self):
        pvsyst = pvc.CapData()
        pvsyst = pvsyst.load_pvsyst('./tests/data/',
                                    'pvsyst_example_HourlyRes_2.CSV')
        self.assertEqual(8760, pvsyst.shape[0],
                         'Not the correct number of rows in imported data.')
        self.assertIsInstance(pvsyst.index,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Index is not a datetime index.')
        self.assertIsInstance(pvsyst.columns,
                              pd.core.indexes.base.Index,
                              'Columns might be MultiIndex; should be base index')

    def test_load_das(self):
        das = pvc.CapData()
        das = das.load_das('./tests/data/',
                           'example_meas_data.csv')
        self.assertEqual(1440, das.shape[0],
                         'Not the correct number of rows in imported data.')
        self.assertIsInstance(das.index,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Index is not a datetime index.')
        self.assertIsInstance(das.columns,
                              pd.core.indexes.base.Index,
                              'Columns might be MultiIndex; should be base index')


class TestCapDataLoadMethods(unittest.TestCase):
    """Tests for load_data method."""

    def setUp(self):
        os.mkdir('test_csvs')
        for fname in test_files:
            with open('test_csvs/' + fname, 'a') as f:
                f.write('Date, val\n11/21/2017, 1')

        self.capdata = pvc.CapData()
        self.capdata.load_data(path='test_csvs/', set_trans=False)

    def tearDown(self):
        for fname in test_files:
            os.remove('test_csvs/' + fname)
        os.rmdir('test_csvs')

    def test_read_csvs(self):
        self.assertEqual(self.capdata.df.shape[0], 3,
                         'imported a non csv or pvsyst file')


class TestCapDataSeriesTypes(unittest.TestCase):
    """Test CapData private methods assignment of type to each series of data."""

    def setUp(self):
        self.cdata = pvc.CapData()

    def test_series_type(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        out = self.cdata._CapData__series_type(test_series, pvc.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

    def test_series_type_caps_in_type_def(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        type_def = collections.OrderedDict([
                     ('irr', [['IRRADIANCE', 'IRR', 'PLANE OF ARRAY', 'POA',
                               'GHI', 'GLOBAL', 'GLOB', 'W/M^2', 'W/M2', 'W/M',
                               'W/'],
                              (-10, 1500)])])
        out = self.cdata._CapData__series_type(test_series, type_def)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

    def test_series_type_repeatable(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        out = []
        i = 0
        while i < 100:
            out.append(self.cdata._CapData__series_type(test_series, pvc.type_defs))
            i += 1
        out_np = np.array(out)

        self.assertTrue(all(out_np == 'irr'),
                        'Result is not consistent after repeated runs.')

    def test_series_type_valErr(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(name=name)
        out = self.cdata._CapData__series_type(test_series, pvc.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr-valuesError',
                         'Returned object is not "irr-valuesError".')

    def test_series_type_no_str(self):
        name = 'should not return key string'
        test_series = pd.Series(name=name)
        out = self.cdata._CapData__series_type(test_series, pvc.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertIs(out, '',
                      'Returned object is not empty string.')

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
        df = pd.DataFrame({'weather_station irr poa W/m^2':rng,
                           'col_1':rng,
                           'col_2':rng})
        df_flt = pvc.flt_irr(df, 'weather_station irr poa W/m^2', 50, 100)

        self.assertEqual(df_flt.shape[0], 51,
                         'Incorrect number of rows returned from filter.')
        self.assertEqual(df_flt.shape[1], 3,
                         'Incorrect number of columns returned from filter.')
        self.assertIs(df.columns[2], 'weather_station irr poa W/m^2',
                      'Filter column name inadverdently modified by method.')
        self.assertEqual(df_flt.iloc[0, 2], 50,
                         'Minimum value in returned data in filter column is'
                         'not equal to low argument.')
        self.assertEqual(df_flt.iloc[-1, 2], 100,
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
        grps = df_regs_day.groupby(by=pd.TimeGrouper('M'))

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

class Test_CapData_methods_sim(unittest.TestCase):
    """Test for top level irrRC_balanced function."""

    def setUp(self):
        self.pvsyst = pvc.CapData()
        self.pvsyst.load_data(path='./tests/data/', load_pvsyst=True)
        # self.jun = self.pvsyst.df.loc['06/1990']
        # self.jun_cpy = self.jun.copy()
        # self.low = 0.5
        # self.high = 1.5
        # (self.irr_RC, self.jun_flt) = pvc.irrRC_balanced(self.jun, self.low,
        #                                                  self.high)
        # self.jun_flt_irr = self.jun_flt['GlobInc']

    def test_copy(self):
        self.pvsyst.set_reg_trans(power='real_pwr--', poa='irr-ghi-',
                                  t_amb='temp-amb-', w_vel='wind--')
        pvsyst_copy = self.pvsyst.copy()
        df_equality = pvsyst_copy.df.equals(self.pvsyst.df)

        self.assertTrue(df_equality,
                        'Dataframe of copy not equal to original')
        self.assertEqual(pvsyst_copy.trans, self.pvsyst.trans,
                         'Trans dict of copy is not equal to original')
        self.assertEqual(pvsyst_copy.trans_keys, self.pvsyst.trans_keys,
                         'Trans dict keys are not equal to original.')
        self.assertEqual(pvsyst_copy.reg_trans, self.pvsyst.reg_trans,
                         'Regression trans dict copy is not equal to orig.')

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

        # Tests for typical monthly predictions
        results = cptest.rep_cond('sim', 0.8, 1.2, inplace=False, freq='M',
                                  pred=True)

        self.assertEqual(results.shape[0], 12,
                         'freq=M should give 12 rows in results dataframe')
        self.assertEqual(results.index[0].month, 1,
                         'First row should be for January.')
        self.assertEqual(results.index[11].month, 12,
                         'Last row should be for January.')

        for val in results.index:
            mnth_str = val.strftime('%m/%Y')

            # Check irradiance values for each month
            df_irr = self.pvsyst.df['GlobInc']
            irr_result = results['poa'].loc[mnth_str][0]
            np_result = np.percentile(df_irr.loc[mnth_str], 60,
                                      interpolation='nearest')
            self.assertAlmostEqual(np_result, irr_result, 7,
                             'The 60th percentile from function does not match '
                             'numpy percentile for {}'.format(mnth_str))

            # Check wind speed values for each month
            df_w_vel = self.pvsyst.df['WindVel']
            w_result = results['w_vel'].loc[mnth_str][0]
            w_result_pd = df_w_vel.loc[mnth_str].mean()
            self.assertAlmostEqual(w_result_pd, w_result, 7,
                             'The average wind speed result does not match '
                             'pandas aveage for {}'.format(mnth_str))

            # Check ambient temperature values for each month
            df_t_amb = self.pvsyst.df['TAmb']
            t_amb_result = results['t_amb'].loc[mnth_str][0]
            t_amb_result_pd = df_t_amb.loc[mnth_str].mean()
            self.assertAlmostEqual(t_amb_result_pd, t_amb_result, 7,
                             'The average amb temp result does not match '
                             'pandas aveage for {}'.format(mnth_str))

        # Tests for seasonal predictions
        check_freqs = ['BQ-JAN', 'BQ-FEB', 'BQ-MAR',
                       'BQ-APR', 'BQ-MAY', 'BQ-JUN',
                       'BQ-JUL', 'BQ-AUG', 'BQ-SEP',
                       'BQ-OCT', 'BQ-NOV', 'BQ-DEC']
        for freq in check_freqs:
            results_qtr = cptest.rep_cond('sim', 0.8, 1.2, inplace=False,
                                          freq=freq, pred=True)

            self.assertEqual(results_qtr.shape[0], 4,
                             '{} seasonal freq not aligned with input '
                             'data'.format(freq))

        cptest.rep_cond('sim', 0.8, 1.2, inplace=True, freq='M',
                                  pred=True)
        self.assertIsInstance(cptest.rc,
                              pd.core.frame.DataFrame,
                              'Results not saved to CapTest rc attribute')


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
