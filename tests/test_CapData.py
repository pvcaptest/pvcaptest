import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import pvlib

from .context import capdata as pvc

data = np.arange(0, 1300, 54.167)
index = pd.date_range(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

# capdata = pvc.CapData('capdata')
# capdata.df = df

"""
Run all tests from project root:
'python -m tests.test_CapData'

Run individual tests:
'python -m unittest tests.test_CapData.Class.Method'

-m flag imports unittest as module rather than running as script
"""

test_files = ['test1.csv', 'test2.csv', 'test3.CSV', 'test4.txt',
              'pvsyst.csv', 'pvsyst_data.csv']





class TestTopLevelFuncs(unittest.TestCase):
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

    def test_filter_irr(self):
        rng = np.arange(0, 1000)
        df = pd.DataFrame(np.array([rng, rng+100, rng+200]).T,
                          columns = ['weather_station irr poa W/m^2',
                                     'col_1', 'col_2'])
        df_flt = pvc.filter_irr(df, 'weather_station irr poa W/m^2', 50, 100)

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
        """
        Test fit model func which wraps statsmodels ols.fit for dataframe.
        """
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
        pvsyst = pvc.CapData('pvsyst')
        pvsyst.load_data(path='./tests/data/', load_pvsyst=True)

        df_regs = pvsyst.data.loc[:, ['E_Grid', 'GlobInc', 'TAmb', 'WindVel']]
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

    def test_perc_bounds_perc(self):
        bounds = pvc.perc_bounds(20)
        self.assertEqual(bounds[0], 0.8,
                         '{} for 20 perc is not 0.8'.format(bounds[0]))
        self.assertEqual(bounds[1], 1.2,
                         '{} for 20 perc is not 1.2'.format(bounds[1]))

    def test_perc_bounds_tuple(self):
        bounds = pvc.perc_bounds((15, 40))
        self.assertEqual(bounds[0], 0.85,
                         '{} for 15 perc is not 0.85'.format(bounds[0]))
        self.assertEqual(bounds[1], 1.4,
                         '{} for 40 perc is not 1.4'.format(bounds[1]))

    def test_filter_grps(self):
        pvsyst = pvc.CapData('pvsyst')
        pvsyst.load_data(path='./tests/data/',
                         fname='pvsyst_example_HourlyRes_2.CSV',
                         load_pvsyst=True)
        pvsyst.set_reg_trans(power='real_pwr--', poa='irr-poa-',
                             t_amb='temp-amb-', w_vel='wind--')
        pvsyst.filter_irr(200, 800)
        pvsyst.rep_cond(freq='MS')
        grps = pvsyst.data_filtered.groupby(pd.Grouper(freq='MS', label='left'))
        poa_col = pvsyst.trans[pvsyst.reg_trans['poa']][0]

        grps_flt = pvc.filter_grps(grps, pvsyst.rc, poa_col, 0.8, 1.2)

        self.assertIsInstance(grps_flt,
                              pd.core.groupby.generic.DataFrameGroupBy,
                              'Returned object is not a dataframe groupby.')

        self.assertEqual(grps.ngroups, grps_flt.ngroups,
                         'Returned groubpy does not have the same number of\
                          groups as passed groupby.')

        cnts_before_flt = grps.count()[poa_col]
        cnts_after_flt = grps_flt.count()[poa_col]
        less_than = all(cnts_after_flt < cnts_before_flt)
        self.assertTrue(less_than, 'Points were not removed for each group.')

    def test_perc_difference(self):
        result = pvc.perc_difference(9, 10)
        self.assertAlmostEqual(result, 0.105263158)

        result = pvc.perc_difference(10, 9)
        self.assertAlmostEqual(result, 0.105263158)

        result = pvc.perc_difference(10, 10)
        self.assertAlmostEqual(result, 0)

        result = pvc.perc_difference(0, 0)
        self.assertAlmostEqual(result, 0)

    def test_check_all_perc_diff_comb(self):
        ser = pd.Series([10.1, 10.2])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertTrue(val,
                        'Failed on two values within 5 percent.')

        ser = pd.Series([10.1, 10.2, 10.15, 10.22, 10.19])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertTrue(val,
                        'Failed with 5 values within 5 percent.')

        ser = pd.Series([10.1, 10.2, 3])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertFalse(val,
                         'Returned True for value outside of 5 percent.')

    def test_sensor_filter_three_cols(self):
        rng = np.zeros(10)
        df = pd.DataFrame({'a':rng, 'b':rng, 'c':rng})
        df['a'] = df['a'] + 4.1
        df['b'] = df['b'] + 4
        df['c'] = df['c'] + 4.2
        df.iloc[0, 0] = 1200
        df.iloc[4, 1] = 100
        df.iloc[7, 2] = 150
        ix = pvc.sensor_filter(df, 0.05)
        self.assertEqual(ix.shape[0], 7,
                         'Filter should have droppe three rows.')

    def test_sensor_filter_one_col(self):
        rng = np.zeros(10)
        df = pd.DataFrame({'a':rng})
        df['a'] = df['a'] + 4.1
        df.iloc[0, 0] = 1200
        ix = pvc.sensor_filter(df, 0.05)
        self.assertEqual(ix.shape[0], 10,
                         'Should be no filtering for single column df.')

    def test_determine_pass_or_fail(self):
        'Tolerance band around 100%'
        self.assertTrue(pvc.determine_pass_or_fail(.96, '+/- 4', 100)[0],
                        'Should pass, cp ratio equals bottom of tolerance.')
        self.assertTrue(pvc.determine_pass_or_fail(.97, '+/- 4', 100)[0],
                        'Should pass, cp ratio above bottom of tolerance.')
        self.assertTrue(pvc.determine_pass_or_fail(1.03, '+/- 4', 100)[0],
                        'Should pass, cp ratio below top of tolerance.')
        self.assertTrue(pvc.determine_pass_or_fail(1.04, '+/- 4', 100)[0],
                        'Should pass, cp ratio equals top of tolerance.')
        self.assertFalse(pvc.determine_pass_or_fail(.959, '+/- 4', 100)[0],
                         'Should fail, cp ratio below bottom of tolerance.')
        self.assertFalse(pvc.determine_pass_or_fail(1.041, '+/- 4', 100)[0],
                         'Should fail, cp ratio above top of tolerance.')
        'Tolerance below 100%'
        self.assertTrue(pvc.determine_pass_or_fail(0.96, '- 4', 100)[0],
                        'Should pass, cp ratio equals bottom of tolerance.')
        self.assertTrue(pvc.determine_pass_or_fail(.97, '- 4', 100)[0],
                        'Should pass, cp ratio above bottom of tolerance.')
        self.assertTrue(pvc.determine_pass_or_fail(1.04, '- 4', 100)[0],
                        'Should pass, cp ratio above bottom of tolerance.')
        self.assertFalse(pvc.determine_pass_or_fail(.959, '- 4', 100)[0],
                         'Should fail, cp ratio below bottom of tolerance.')
        'warn on incorrect tolerance spec'
        with self.assertWarns(UserWarning):
            pvc.determine_pass_or_fail(1.04, '+ 4', 100)

    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test_print_results_pass(self):
        """
        This test uses the pytest autouse fixture defined above to
        capture the print to stdout and test it, so it must be run
        using pytest 'pytest tests/
        test_CapData.py::TestTopLevelFuncs::test_print_results_pass'
        """
        test_passed = (True, '950, 1050')
        pvc.print_results(test_passed, 1000, 970, 0.97, 970, test_passed[1])
        captured = self.capsys.readouterr()

        results_str = ('Capacity Test Result:         PASS\n'
                       'Modeled test output:          1000.000\n'
                       'Actual test output:           970.000\n'
                       'Tested output ratio:          0.970\n'
                       'Tested Capacity:              970.000\n'
                       'Bounds:                       950, 1050\n\n\n')

        self.assertEqual(results_str, captured.out)

    def test_print_results_fail(self):
        """
        This test uses the pytest autouse fixture defined above to
        capture the print to stdout and test it, so it must be run
        using pytest 'pytest tests/
        test_CapData.py::TestTopLevelFuncs::test_print_results_pass'
        """
        test_passed = (False, '950, 1050')
        pvc.print_results(test_passed, 1000, 940, 0.94, 940, test_passed[1])
        captured = self.capsys.readouterr()

        results_str = ('Capacity Test Result:    FAIL\n'
                       'Modeled test output:          1000.000\n'
                       'Actual test output:           940.000\n'
                       'Tested output ratio:          0.940\n'
                       'Tested Capacity:              940.000\n'
                       'Bounds:                       950, 1050\n\n\n')

        self.assertEqual(results_str, captured.out)

class TestLoadDataMethods(unittest.TestCase):
    """Test for load data methods without setup."""

    def test_load_pvsyst(self):
        pvsyst = pvc.CapData('pvsyst')
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


    def test_source_alsoenergy(self):
        das_1 = pvc.CapData('das_1')
        das_1.load_data(path='./tests/data/col_naming_examples/',
                      fname='ae_site1.csv', source='AlsoEnergy')
        col_names1 = ['Elkor Production Meter PowerFactor, ',
                      'Elkor Production Meter KW, kW',
                      'Weather Station 1 TempF, °F', 'Weather Station 2 Sun2, W/m²',
                      'Weather Station 1 Sun, W/m²', 'Weather Station 1 WindSpeed, mph',
                      'index']
        self.assertTrue(all(das_1.data.columns == col_names1),
                        'Column names are not expected value for ae_site1')

        das_2 = pvc.CapData('das_2')
        das_2.load_data(path='./tests/data/col_naming_examples/',
                      fname='ae_site2.csv', source='AlsoEnergy')
        col_names2 = ['Acuvim II Meter PowerFactor, PF', 'Acuvim II Meter KW, kW',
                      'Weather Station 1 TempF, °F', 'Weather Station 3 TempF, °F',
                      'Weather Station 2 Sun2, W/m²', 'Weather Station 4 Sun2, W/m²',
                      'Weather Station 1 Sun, W/m²', 'Weather Station 3 Sun, W/m²',
                      'Weather Station 1 WindSpeed, mph',
                      'Weather Station 3 WindSpeed, mph',
                      'index']
        self.assertTrue(all(das_2.data.columns == col_names2),
                        'Column names are not expected value for ae_site1')

    def test_load_das(self):
        das = pvc.CapData('das')
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

        self.capdata = pvc.CapData('capdata')
        self.capdata.load_data(path='test_csvs/', set_trans=False)

    def tearDown(self):
        for fname in test_files:
            os.remove('test_csvs/' + fname)
        os.rmdir('test_csvs')

    def test_read_csvs(self):
        self.assertEqual(self.capdata.data.shape[0], 3,
                         'imported a non csv or pvsyst file')


class TestCapDataSeriesTypes(unittest.TestCase):
    """Test CapData private methods assignment of type to each series of data."""

    def setUp(self):
        self.cdata = pvc.CapData('cdata')

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
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

    def test_series_type_no_str(self):
        name = 'should not return key string'
        test_series = pd.Series(name=name)
        out = self.cdata._CapData__series_type(test_series, pvc.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertIs(out, '',
                      'Returned object is not empty string.')


class Test_CapData_methods_sim(unittest.TestCase):
    """Test for top level irrRC_balanced function."""

    def setUp(self):
        self.pvsyst = pvc.CapData('pvsyst')
        self.pvsyst.load_data(path='./tests/data/', load_pvsyst=True)
        # self.jun = self.pvsyst.data.loc['06/1990']
        # self.jun_cpy = self.jun.copy()
        # self.low = 0.5
        # self.high = 1.5
        # (self.irr_RC, self.jun_flt) = pvc.irrRC_balanced(self.jun, self.low,
        #                                                  self.high)
        # self.jun_filter_irr = self.jun_flt['GlobInc']

    def test_copy(self):
        self.pvsyst.set_reg_trans(power='real_pwr--', poa='irr-ghi-',
                                  t_amb='temp-amb-', w_vel='wind--')
        pvsyst_copy = self.pvsyst.copy()
        df_equality = pvsyst_copy.data.equals(self.pvsyst.data)

        self.assertTrue(df_equality,
                        'Dataframe of copy not equal to original')
        self.assertEqual(pvsyst_copy.trans, self.pvsyst.trans,
                         'Trans dict of copy is not equal to original')
        self.assertEqual(pvsyst_copy.trans_keys, self.pvsyst.trans_keys,
                         'Trans dict keys are not equal to original.')
        self.assertEqual(pvsyst_copy.reg_trans, self.pvsyst.reg_trans,
                         'Regression trans dict copy is not equal to orig.')

    def test_irrRC_balanced(self):
        jun = self.pvsyst.data.loc['06/1990']
        jun_cpy = jun.copy()
        low = 0.5
        high = 1.5
        (irr_RC, jun_flt) = pvc.irrRC_balanced(jun, low, high)
        jun_filter_irr = jun_flt['GlobInc']
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

        pts_below_irr = jun_filter_irr[jun_filter_irr.between(0, irr_RC)].shape[0]
        perc_below = pts_below_irr / jun_filter_irr.shape[0]
        self.assertLess(perc_below, 0.6,
                        'More than 60 percent of points below reporting irr')
        self.assertGreaterEqual(perc_below, 0.5,
                                'Less than 50 percent of points below rep irr')

        pts_above_irr = jun_filter_irr[jun_filter_irr.between(irr_RC, 1500)].shape[0]
        perc_above = pts_above_irr / jun_filter_irr.shape[0]
        self.assertGreater(perc_above, 0.4,
                           'Less than 40 percent of points above reporting irr')
        self.assertLessEqual(perc_above, 0.5,
                             'More than 50 percent of points above reportin irr')

    def test_filter_pvsyst_default(self):
        self.pvsyst.filter_pvsyst()
        self.assertEqual(self.pvsyst.data_filtered.shape[0], 8670,
                         'Data should contain 8670 points after removing any\
                          of IL Pmin, IL Pmax, IL Vmin, IL Vmax that are\
                          greater than zero.')

    def test_filter_pvsyst_not_inplace(self):
        df = self.pvsyst.filter_pvsyst(inplace=False)
        self.assertIsInstance(df, pd.core.frame.DataFrame,
                              'Did not return DataFrame object.')
        self.assertEqual(df.shape[0], 8670,
                         'Data should contain 8670 points after removing any\
                          of IL Pmin, IL Pmax, IL Vmin, IL Vmax that are\
                          greater than zero.')

    def test_filter_pvsyst_missing_column(self):
        self.pvsyst.drop_cols('IL Pmin')
        self.pvsyst.filter_pvsyst()

    def test_filter_pvsyst_missing_all_columns(self):
        self.pvsyst.drop_cols(['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax'])
        self.pvsyst.filter_pvsyst()

    def test_filter_shade_default(self):
        self.pvsyst.filter_shade()
        self.assertEqual(self.pvsyst.data_filtered.shape[0], 8645,
                         'Data should contain 8645 time periods\
                          without shade.')

    def test_filter_shade_default_not_inplace(self):
        df = self.pvsyst.filter_shade(inplace=False)
        self.assertIsInstance(df, pd.core.frame.DataFrame,
                              'Did not return DataFrame object.')
        self.assertEqual(df.shape[0], 8645,
                         'Returned dataframe should contain 8645 time periods\
                          without shade.')

    def test_filter_shade_query(self):
        # create PVsyst ShdLoss type values for testing query string
        self.pvsyst.data.loc[self.pvsyst.data['FShdBm'] == 1.0, 'ShdLoss'] = 0
        is_shaded = self.pvsyst.data['ShdLoss'].isna()
        shdloss_values = 1 / self.pvsyst.data.loc[is_shaded, 'FShdBm'] * 100
        self.pvsyst.data.loc[is_shaded, 'ShdLoss'] = shdloss_values
        self.pvsyst.data_filtered = self.pvsyst.data.copy()

        self.pvsyst.filter_shade(query_str='ShdLoss<=125')
        self.assertEqual(self.pvsyst.data_filtered.shape[0], 8671,
                         'Filtered data should contain have 8671 periods with\
                          shade losses less than 125.')


class Test_pvlib_loc_sys(unittest.TestCase):
    """ Test function wrapping pvlib get_clearsky method of Location."""
    def test_pvlib_location(self):
        loc = {'latitude': 30.274583,
               'longitude': -97.740352,
               'altitude': 500,
               'tz': 'America/Chicago'}

        loc_obj = pvc.pvlib_location(loc)

        self.assertIsInstance(loc_obj,
                              pvlib.location.Location,
                              'Did not return instance of\
                               pvlib Location')

    def test_pvlib_system(self):
        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        tracker_sys2 = {'max_angle': 52, 'gcr': 0.3}

        fx_sys = pvc.pvlib_system(fixed_sys)
        trck_sys1 = pvc.pvlib_system(tracker_sys1)
        trck_sys2 = pvc.pvlib_system(tracker_sys1)

        self.assertIsInstance(fx_sys,
                              pvlib.pvsystem.PVSystem,
                              'Did not return instance of\
                               pvlib PVSystem')

        self.assertIsInstance(trck_sys1,
                              pvlib.tracking.SingleAxisTracker,
                              'Did not return instance of\
                               pvlib SingleAxisTracker')

        self.assertIsInstance(trck_sys2,
                              pvlib.tracking.SingleAxisTracker,
                              'Did not return instance of\
                               pvlib SingleAxisTracker')


# possible assertions for method returning ghi
        # self.assertIsInstance(ghi,
        #                       pd.core.series.Series,
        #                       'Second returned object is not an instance of\
        #                        pandas Series.')
        # self.assertEqual(ghi.name, 'ghi',
        #                  'Series data returned is not named ghi')
        # self.assertEqual(ghi.shape[0], df.shape[0],
        #                  'Returned ghi does not have the same number of rows\
        #                   as the passed dataframe.')
        # self.assertEqual(df.index.tz, ghi.index.tz,
        #                  'Returned series index has different timezone from\
        #                   passed dataframe.')


class Test_csky(unittest.TestCase):
    """Test clear sky function which returns pvlib ghi and poa clear sky."""
    def setUp(self):
        self.loc = {'latitude': 30.274583,
                    'longitude': -97.740352,
                    'altitude': 500,
                    'tz': 'America/Chicago'}

        self.sys = {'surface_tilt': 20,
                    'surface_azimuth': 180,
                    'albedo': 0.2}

        self.meas = pvc.CapData('meas')
        self.df = self.meas.load_das('./tests/data/', 'example_meas_data.csv')

    def test_get_tz_index_df(self):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe without a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/Chicago')
        ix_2days = pd.date_range(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/Chicago')
        ix_dst = ix_3days.append(ix_2days)
        ix_dst = ix_dst.tz_localize(None)
        self.df.index = ix_dst

        self.tz_ix = pvc.get_tz_index(self.df, self.loc)

        self.assertIsInstance(self.tz_ix,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Returned object is not a pandas DatetimeIndex.')
        self.assertEqual(self.tz_ix.tz,
                         pytz.timezone(self.loc['tz']),
                         'Returned index does not have same timezone as\
                          the passed location dictionary.')

    def test_get_tz_index_df_tz(self):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe with a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/Chicago')
        ix_2days = pd.date_range(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/Chicago')
        ix_dst = ix_3days.append(ix_2days)
        self.df.index = ix_dst

        self.tz_ix = pvc.get_tz_index(self.df, self.loc)

        self.assertIsInstance(self.tz_ix,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Returned object is not a pandas DatetimeIndex.')
        self.assertEqual(self.tz_ix.tz,
                         pytz.timezone(self.loc['tz']),
                         'Returned index does not have same timezone as\
                          the passed location dictionary.')

    def test_get_tz_index_df_tz_warn(self):
        """Test that get_tz_index function returns warns when datetime index\
           of dataframe does not match loc dic timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/New_York')
        ix_2days = pd.date_range(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/New_York')
        ix_dst = ix_3days.append(ix_2days)
        self.df.index = ix_dst

        with self.assertWarns(UserWarning):
            self.tz_ix = pvc.get_tz_index(self.df, self.loc)

    def test_get_tz_index_ix_tz(self):
        """Test that get_tz_index function returns a datetime index
           with a timezone when passed a datetime index with a timezone."""
        self.ix = pd.date_range(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/Chicago')
        self.tz_ix = pvc.get_tz_index(self.ix, self.loc)

        self.assertIsInstance(self.tz_ix,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Returned object is not a pandas DatetimeIndex.')
        # If passing an index with a timezone use that timezone rather than
        # the timezone in the location dictionary if there is one.
        self.assertEqual(self.tz_ix.tz,
                         self.ix.tz,
                         'Returned index does not have same timezone as\
                          the passed index.')

    def test_get_tz_index_ix_tz_warn(self):
        """Test that get_tz_index function warns when DatetimeIndex timezone
           does not match the location dic timezone.
        """
        self.ix = pd.date_range(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/New_York')

        with self.assertWarns(UserWarning):
            self.tz_ix = pvc.get_tz_index(self.ix, self.loc)

    def test_get_tz_index_ix(self):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a datetime index without a timezone."""
        self.ix = pd.date_range(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/Chicago')
        # remove timezone info but keep missing  hour and extra hour due to DST
        self.ix = self.ix.tz_localize(None)
        self.tz_ix = pvc.get_tz_index(self.ix, self.loc)

        self.assertIsInstance(self.tz_ix,
                              pd.core.indexes.datetimes.DatetimeIndex,
                              'Returned object is not a pandas DatetimeIndex.')
        # If passing an index without a timezone use returned index should have
        # the timezone of the passed location dictionary.
        self.assertEqual(self.tz_ix.tz,
                         pytz.timezone(self.loc['tz']),
                         'Returned index does not have same timezone as\
                          the passed location dictionary.')

    def test_csky_concat(self):
        # concat=True by default
        csky_ghi_poa = pvc.csky(self.df, loc=self.loc, sys=self.sys)

        self.assertIsInstance(csky_ghi_poa, pd.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertEqual(csky_ghi_poa.shape[1],
                         self.df.shape[1] + 2,
                         'Returned dataframe does not have 2 new columns.')
        self.assertIn('ghi_mod_csky', csky_ghi_poa.columns,
                      'Modeled clear sky ghi not in returned dataframe columns')
        self.assertIn('poa_mod_csky', csky_ghi_poa.columns,
                      'Modeled clear sky poa not in returned dataframe columns')
        # assumes typical orientation is used to calculate the poa irradiance
        self.assertGreater(csky_ghi_poa.loc['10/9/1990 12:30',
                                            'poa_mod_csky'],
                           csky_ghi_poa.loc['10/9/1990 12:30',
                                            'ghi_mod_csky'],
                           'POA is not greater than GHI at 12:30.')
        self.assertEqual(csky_ghi_poa.index.tz,
                         self.df.index.tz,
                         'Returned dataframe index timezone is not the same as\
                          passed dataframe.')

    def test_csky_not_concat(self):
        csky_ghi_poa = pvc.csky(self.df, loc=self.loc, sys=self.sys,
                                     concat=False)

        self.assertIsInstance(csky_ghi_poa, pd.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertEqual(csky_ghi_poa.shape[1], 2,
                         'Returned dataframe does not have 2 columns.')
        self.assertIn('ghi_mod_csky', csky_ghi_poa.columns,
                      'Modeled clear sky ghi not in returned dataframe columns')
        self.assertIn('poa_mod_csky', csky_ghi_poa.columns,
                      'Modeled clear sky poa not in returned dataframe columns')
        # assumes typical orientation is used to calculate the poa irradiance
        self.assertGreater(csky_ghi_poa.loc['10/9/1990 12:30',
                                            'poa_mod_csky'],
                           csky_ghi_poa.loc['10/9/1990 12:30',
                                            'ghi_mod_csky'],
                           'POA is not greater than GHI at 12:30.')
        self.assertEqual(csky_ghi_poa.index.tz,
                         self.df.index.tz,
                         'Returned dataframe index timezone is not the same as\
                          passed dataframe.')

    def test_csky_not_concat_poa_all(self):
        csky_ghi_poa = pvc.csky(self.df, loc=self.loc, sys=self.sys,
                                     concat=False, output='poa_all')

        self.assertIsInstance(csky_ghi_poa, pd.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertEqual(csky_ghi_poa.shape[1], 5,
                         'Returned dataframe does not have 5 columns.')
        cols = ['poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', 'poa_ground_diffuse', 'poa_ground_diffuse']
        for col in cols:
            self.assertIn(col, csky_ghi_poa.columns,
                          '{} not in the columns of returned\
                           dataframe'.format(col))
        # assumes typical orientation is used to calculate the poa irradiance
        self.assertEqual(csky_ghi_poa.index.tz,
                         self.df.index.tz,
                         'Returned dataframe index timezone is not the same as\
                          passed dataframe.')

    def test_csky_not_concat_ghi_all(self):
        csky_ghi_poa = pvc.csky(self.df, loc=self.loc, sys=self.sys,
                                concat=False, output='ghi_all')

        self.assertIsInstance(csky_ghi_poa, pd.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertEqual(csky_ghi_poa.shape[1], 3,
                         'Returned dataframe does not have 5 columns.')
        cols = ['ghi', 'dni', 'dhi']
        for col in cols:
            self.assertIn(col, csky_ghi_poa.columns,
                          '{} not in the columns of returned\
                           dataframe'.format(col))
        # assumes typical orientation is used to calculate the poa irradiance
        self.assertEqual(csky_ghi_poa.index.tz,
                         self.df.index.tz,
                         'Returned dataframe index timezone is not the same as\
                          passed dataframe.')

    def test_csky_not_concat_all(self):
        csky_ghi_poa = pvc.csky(self.df, loc=self.loc, sys=self.sys,
                                concat=False, output='all')

        self.assertIsInstance(csky_ghi_poa, pd.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertEqual(csky_ghi_poa.shape[1], 8,
                         'Returned dataframe does not have 5 columns.')
        cols = ['ghi', 'dni', 'dhi', 'poa_global', 'poa_direct', 'poa_diffuse',
                'poa_sky_diffuse', 'poa_ground_diffuse', 'poa_ground_diffuse']
        for col in cols:
            self.assertIn(col, csky_ghi_poa.columns,
                          '{} not in the columns of returned\
                           dataframe'.format(col))
        # assumes typical orientation is used to calculate the poa irradiance
        self.assertEqual(csky_ghi_poa.index.tz,
                         self.df.index.tz,
                         'Returned dataframe index timezone is not the same as\
                          passed dataframe.')

"""
Change csky to two functions for creating pvlib location and system objects.
Separate function calling location and system to calculate POA
- concat add columns to passed df or return just ghi and poa option
load_data calls final function with in place to get ghi and poa
"""

class TestGetRegCols(unittest.TestCase):
    def setUp(self):
        self.das = pvc.CapData('das')
        self.das.load_data(path='./tests/data/',
                           fname='example_meas_data_aeheaders.csv',
                           source='AlsoEnergy')
        self.das.set_reg_trans(power='-mtr-', poa='irr-poa-',
                               t_amb='temp-amb-', w_vel='wind--')

    def test_not_aggregated(self):
        with self.assertWarns(UserWarning):
            self.das.get_reg_cols()

    def test_all_coeffs(self):
        self.das.agg_sensors()
        cols = ['power', 'poa', 't_amb', 'w_vel']
        df = self.das.get_reg_cols()
        self.assertEqual(len(df.columns), 4,
                         'Returned number of columns is incorrect.')
        self.assertEqual(df.columns.to_list(), cols,
                         'Columns are not renamed properly.')
        self.assertEqual(self.das.data['-mtr-sum-agg'].iloc[100],
                         df['power'].iloc[100],
                         'Data in column labeled power is not power.')
        self.assertEqual(self.das.data['irr-poa-mean-agg'].iloc[100],
                         df['poa'].iloc[100],
                         'Data in column labeled poa is not poa.')
        self.assertEqual(self.das.data['temp-amb-mean-agg'].iloc[100],
                         df['t_amb'].iloc[100],
                         'Data in column labeled t_amb is not t_amb.')
        self.assertEqual(self.das.data['wind--mean-agg'].iloc[100],
                         df['w_vel'].iloc[100],
                         'Data in column labeled w_vel is not w_vel.')

    def test_poa_power(self):
        self.das.agg_sensors()
        cols = ['poa', 'power']
        df = self.das.get_reg_cols(reg_vars=cols)
        self.assertEqual(len(df.columns), 2,
                         'Returned number of columns is incorrect.')
        self.assertEqual(df.columns.to_list(), cols,
                         'Columns are not renamed properly.')
        self.assertEqual(self.das.data['-mtr-sum-agg'].iloc[100],
                         df['power'].iloc[100],
                         'Data in column labeled power is not power.')
        self.assertEqual(self.das.data['irr-poa-mean-agg'].iloc[100],
                         df['poa'].iloc[100],
                         'Data in column labeled poa is not poa.')

    def test_agg_sensors_mix(self):
        """
        Test when agg_sensors resets reg_trans values to a mix of trans keys
        and column names.
        """
        self.das.agg_sensors(agg_map={'-inv-': 'sum', 'irr-poa-': 'mean',
                                      'temp-amb-': 'mean', 'wind--': 'mean'})
        cols = ['poa', 'power']
        df = self.das.get_reg_cols(reg_vars=cols)
        mtr_col = self.das.trans[self.das.reg_trans['power']][0]
        self.assertEqual(len(df.columns), 2,
                         'Returned number of columns is incorrect.')
        self.assertEqual(df.columns.to_list(), cols,
                         'Columns are not renamed properly.')
        self.assertEqual(self.das.data[mtr_col].iloc[100],
                         df['power'].iloc[100],
                         'Data in column labeled power is not power.')
        self.assertEqual(self.das.data['irr-poa-mean-agg'].iloc[100],
                         df['poa'].iloc[100],
                         'Data in column labeled poa is not poa.')


class TestAggSensors(unittest.TestCase):
    def setUp(self):
        self.das = pvc.CapData('das')
        self.das.load_data(path='./tests/data/',
                           fname='example_meas_data_aeheaders.csv',
                           source='AlsoEnergy')
        self.das.set_reg_trans(power='-mtr-', poa='irr-poa-',
                               t_amb='temp-amb-', w_vel='wind--')

    def test_agg_map_none(self):
        self.das.agg_sensors()
        self.assertEqual(self.das.data_filtered.shape[1], self.das.data.shape[1],
                         'df and data_filtered should have same number of rows.')
        self.assertEqual(self.das.data_filtered.shape[0], self.das.data.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum-agg', self.das.data_filtered.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean-agg', self.das.data_filtered.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean-agg', self.das.data_filtered.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean-agg', self.das.data_filtered.columns,
                      'Mean of wind trans group not in aggregated df.')

    def test_agg_map_none_inplace_false(self):
        df_flt_copy = self.das.data_filtered.copy()
        df = self.das.agg_sensors(inplace=False)
        self.assertEqual(df.shape[1], self.das.data.shape[1] + 4,
                         'Returned df does not include 4 additional cols.')
        self.assertEqual(df.shape[0], self.das.data.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum-agg', df.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean-agg', df.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean-agg', df.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean-agg', df.columns,
                      'Mean of wind trans group not in aggregated df.')
        self.assertTrue(df_flt_copy.equals(self.das.data_filtered),
                        'Method with inplace false changed data_filtered attribute.')

    def test_agg_map_none_keep_false(self):
        self.das.agg_sensors(keep=False)
        self.assertEqual(self.das.data_filtered.shape[1], 4,
                         'Returned dataframe does not have 4 columns.')
        self.assertEqual(self.das.data_filtered.shape[0], self.das.data.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum-agg', self.das.data_filtered.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean-agg', self.das.data_filtered.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean-agg', self.das.data_filtered.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean-agg', self.das.data_filtered.columns,
                      'Mean of wind trans group not in aggregated df.')

    def test_agg_map_non_str_func(self):
        self.das.agg_sensors(agg_map={'irr-poa-': np.mean})
        self.assertEqual(self.das.data_filtered.shape[1], self.das.data.shape[1],
                         'df and data_filtered should have same number of rows.')
        self.assertEqual(self.das.data_filtered.shape[0], self.das.data.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('irr-poa-mean-agg', self.das.data_filtered.columns,
                      'Mean of poa trans group not in aggregated df.')

    def test_agg_map_mix_funcs(self):
        self.das.agg_sensors(agg_map={'irr-poa-': [np.mean, 'sum']})
        self.assertEqual(self.das.data_filtered.shape[1], self.das.data.shape[1],
                         'df and data_filtered should have same number of rows.')
        self.assertEqual(self.das.data_filtered.shape[0], self.das.data.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('irr-poa-mean-agg', self.das.data_filtered.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('irr-poa-sum-agg', self.das.data_filtered.columns,
                      'Sum of poa trans group not in aggregated df.')

    def test_agg_map_update_reg_trans(self):
        self.das.agg_sensors()
        self.assertEqual(self.das.reg_trans['power'], '-mtr-sum-agg',
                         'Power reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['poa'], 'irr-poa-mean-agg',
                         'POA reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['t_amb'], 'temp-amb-mean-agg',
                         'Amb temp reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['w_vel'], 'wind--mean-agg',
                         'Wind velocity reg_trans not updated to agg column.')

    def test_reset_summary(self):
        self.das.agg_sensors()
        self.assertEqual(len(self.das.summary), 0,
                         'Summary data not reset.')
        self.assertEqual(len(self.das.summary_ix), 0,
                         'Summary index not reset.')

    def test_reset_agg_method(self):
        orig_df = self.das.data.copy()
        orig_trans = self.das.trans.copy()
        orig_reg_trans = self.das.reg_trans.copy()

        self.das.agg_sensors()
        self.das.filter_irr(200, 500)
        self.das.reset_agg()

        self.assertTrue(self.das.data.equals(orig_df),
                        'df attribute does not match pre-agg df after reset.')
        self.assertTrue(all(self.das.data_filtered.columns == orig_df.columns),
                        'Filtered dataframe does not have same columns as'
                        'original dataframe after resetting agg.')
        self.assertLess(self.das.data_filtered.shape[0], orig_df.shape[0],
                        'Filtering overwritten by reset agg method.')

    def test_warn_if_filters_already_run(self):
        """
        Warn if method is writing over filtering already applied to data_filtered.
        """
        poa_key = self.das.reg_trans['poa']
        self.das.trans[poa_key] = [self.das.trans[poa_key][0]]
        self.das.filter_irr(200, 800)
        with self.assertWarns(UserWarning):
            self.das.agg_sensors()


class TestFilterSensors(unittest.TestCase):
    def setUp(self):
        self.das = pvc.CapData('das')
        self.das.load_data(path='./tests/data/',
                           fname='example_meas_data.csv',
                           trans_report=False)
        self.das.set_reg_trans(power='-mtr-', poa='irr-poa-ref_cell',
                               t_amb='temp-amb-', w_vel='wind--')

    def test_perc_diff_none(self):
        rows_before_flt = self.das.data_filtered.shape[0]
        self.das.filter_sensors(perc_diff=None, inplace=True)
        self.assertIsInstance(self.das.data_filtered, pd.core.frame.DataFrame,
                              'Did not dave a dataframe to data_filtered.')
        self.assertLess(self.das.data_filtered.shape[0], rows_before_flt,
                        'No rows removed.')

    def test_perc_diff(self):
        rows_before_flt = self.das.data_filtered.shape[0]
        self.das.filter_sensors(perc_diff={'irr-poa-ref_cell': 0.05,
                                           'temp-amb-': 0.1},
                                inplace=True)
        self.assertIsInstance(self.das.data_filtered, pd.core.frame.DataFrame,
                              'Did not dave a dataframe to data_filtered.')
        self.assertLess(self.das.data_filtered.shape[0], rows_before_flt,
                        'No rows removed.')

    def test_after_agg_sensors(self):
        rows_before_flt = self.das.data_filtered.shape[0]
        self.das.agg_sensors(agg_map={'-inv-': 'sum',
                                      'irr-poa-ref_cell': 'mean',
                                      'wind--': 'mean',
                                      'temp-amb-': 'mean'})
        self.das.filter_sensors(perc_diff={'irr-poa-ref_cell': 0.05,
                                           'temp-amb-': 0.1},
                                inplace=True)
        self.assertIsInstance(self.das.data_filtered, pd.core.frame.DataFrame,
                              'Did not dave a dataframe to data_filtered.')
        self.assertLess(self.das.data_filtered.shape[0], rows_before_flt,
                        'No rows removed.')
        self.assertIn('-inv-sum-agg', self.das.data_filtered.columns,
                      'filter_sensors did not retain aggregation columns.')


class TestRepCondNoFreq(unittest.TestCase):
    def setUp(self):
        self.meas = pvc.CapData('meas')
        self.meas.load_data(path='./tests/data/', fname='nrel_data.csv',
                            source='AlsoEnergy')
        self.meas.set_reg_trans(power='', poa='irr-poa-',
                                t_amb='temp--', w_vel='wind--')

    def test_defaults(self):
        self.meas.rep_cond()
        self.assertIsInstance(self.meas.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')

    def test_defaults_wvel(self):
        self.meas.rep_cond(w_vel=50)
        self.assertEqual(self.meas.rc['w_vel'][0], 50,
                         'Wind velocity not overwritten by user value')

    def test_defaults_not_inplace(self):
        df = self.meas.rep_cond(inplace=False)
        self.assertIsNone(self.meas.rc,
                          'Method result stored instead of returned.')
        self.assertIsInstance(df, pd.core.frame.DataFrame,
                              'No dataframe returned from method.')

    def test_irr_bal_inplace(self):
        self.meas.filter_irr(0.1, 2000)
        meas2 = self.meas.copy()
        meas2.rep_cond()
        self.meas.rep_cond(irr_bal=True, percent_filter=20)
        self.assertIsInstance(self.meas.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')
        self.assertNotEqual(self.meas.rc['poa'][0], meas2.rc['poa'][0],
                            'Irr_bal function returned same result\
                             as w/o irr_bal')

    def test_irr_bal_inplace_wvel(self):
        self.meas.rep_cond(irr_bal=True, percent_filter=20, w_vel=50)
        self.assertEqual(self.meas.rc['w_vel'][0], 50,
                         'Wind velocity not overwritten by user value')

    def test_irr_bal_inplace_no_percent_filter(self):
        with self.assertWarns(UserWarning):
            self.meas.rep_cond(irr_bal=True, percent_filter=None)


class TestRepCondFreq(unittest.TestCase):
    def setUp(self):
        self.pvsyst = pvc.CapData('pvsyst')
        self.pvsyst.load_data(path='./tests/data/',
                              fname='pvsyst_example_HourlyRes_2.CSV',
                              load_pvsyst=True)
        self.pvsyst.set_reg_trans(power='real_pwr--', poa='irr-poa-',
                                 t_amb='temp-amb-', w_vel='wind--')

    def test_monthly_no_irr_bal(self):
        self.pvsyst.rep_cond(freq='M')
        self.assertIsInstance(self.pvsyst.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')
        self.assertEqual(self.pvsyst.rc.shape[0], 12,
                         'Rep conditions dataframe does not have 12 rows.')

    def test_monthly_irr_bal(self):
        self.pvsyst.rep_cond(freq='M', irr_bal=True, percent_filter=20)
        self.assertIsInstance(self.pvsyst.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')
        self.assertEqual(self.pvsyst.rc.shape[0], 12,
                         'Rep conditions dataframe does not have 12 rows.')

    def test_seas_no_irr_bal(self):
        self.pvsyst.rep_cond(freq='BQ-NOV', irr_bal=False)
        self.assertIsInstance(self.pvsyst.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')
        self.assertEqual(self.pvsyst.rc.shape[0], 4,
                         'Rep conditions dataframe does not have 4 rows.')


class TestPredictCapacities(unittest.TestCase):
    def setUp(self):
        self.pvsyst = pvc.CapData('pvsyst')
        self.pvsyst.load_data(path='./tests/data/',
                              fname='pvsyst_example_HourlyRes_2.CSV',
                              load_pvsyst=True)
        self.pvsyst.set_reg_trans(power='real_pwr--', poa='irr-poa-',
                                 t_amb='temp-amb-', w_vel='wind--')
        self.pvsyst.filter_irr(200, 800)
        self.pvsyst.tolerance = '+/- 5'

    def test_monthly(self):
        self.pvsyst.rep_cond(freq='MS')
        pred_caps = self.pvsyst.predict_capacities(irr_filter=True, percent_filter=20)
        july_grpby = pred_caps.loc['1990-07-01', 'PredCap']

        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is not a Dataframe.')
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

        self.pvsyst.data_filtered = self.pvsyst.data_filtered.loc['7/1/90':'7/31/90', :]
        self.pvsyst.rep_cond()
        self.pvsyst.filter_irr(0.8, 1.2, ref_val=self.pvsyst.rc['poa'][0])
        df = self.pvsyst.rview(['power', 'poa', 't_amb', 'w_vel'],
                               filtered_data=True)
        rename = {df.columns[0]: 'power',
                  df.columns[1]: 'poa',
                  df.columns[2]: 't_amb',
                  df.columns[3]: 'w_vel'}
        df = df.rename(columns=rename)
        reg = pvc.fit_model(df)
        july_manual = reg.predict(self.pvsyst.rc)[0]
        self.assertEqual(july_manual, july_grpby,
                         'Manual prediction for July {} is not equal'
                         'to the predict_capacites groupby'
                         'prediction {}'.format(july_manual, july_grpby))

    def test_no_irr_filter(self):
        self.pvsyst.rep_cond(freq='M')
        pred_caps = self.pvsyst.predict_capacities(irr_filter=False)
        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is not a Dataframe.')
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

    def test_rc_from_irrBal(self):
        self.pvsyst.rep_cond(freq='M', irr_bal=True, percent_filter=20)
        pred_caps = self.pvsyst.predict_capacities(irr_filter=False)
        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is {} not a\
                               Dataframe.'.format(type(pred_caps)))
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

    def test_seasonal_freq(self):
        self.pvsyst.rep_cond(freq='BQ-NOV')
        pred_caps = self.pvsyst.predict_capacities(irr_filter=True, percent_filter=20)
        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is {} not a\
                               Dataframe.'.format(type(pred_caps)))
        self.assertEqual(pred_caps.shape[0], 4,
                         'Predicted capacities has {} rows instead of 4\
                          rows.'.format(pred_caps.shape[0]))


class TestFilterIrr(unittest.TestCase):
    def setUp(self):
        self.meas = pvc.CapData('meas')
        self.meas.load_data('./tests/data/', 'nrel_data.csv',
                            source='AlsoEnergy')
        self.meas.set_reg_trans(power='', poa='irr-poa-',
                                t_amb='temp--', w_vel='wind--')

    def test_get_poa_col(self):
        col = self.meas._CapData__get_poa_col()
        self.assertEqual(col, 'POA 40-South CMP11 [W/m^2]',
                         'POA column not returned')

    def test_get_poa_col_multcols(self):
        self.meas.data['POA second column'] = self.meas.rview('poa').values
        self.meas.set_translation()
        with self.assertWarns(UserWarning):
            col = self.meas._CapData__get_poa_col()

    def test_lowhigh_nocol(self):
        pts_before = self.meas.data_filtered.shape[0]
        self.meas.filter_irr(500, 600, ref_val=None, col_name=None,
                             inplace=True)
        self.assertLess(self.meas.data_filtered.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_lowhigh_colname(self):
        pts_before = self.meas.data_filtered.shape[0]
        self.meas.data['POA second column'] = self.meas.rview('poa').values
        self.meas.set_translation()
        self.meas.data_filtered = self.meas.data.copy()
        self.meas.filter_irr(500, 600, ref_val=None,
                             col_name='POA second column', inplace=True)
        self.assertLess(self.meas.data_filtered.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_nocol(self):
        pts_before = self.meas.data_filtered.shape[0]
        self.meas.filter_irr(0.8, 1.2, ref_val=500, col_name=None,
                             inplace=True)
        self.assertLess(self.meas.data_filtered.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_withcol(self):
        pts_before = self.meas.data_filtered.shape[0]
        self.meas.data['POA second column'] = self.meas.rview('poa').values
        self.meas.set_translation()
        self.meas.data_filtered = self.meas.data.copy()
        self.meas.filter_irr(0.8, 1.2, ref_val=500,
                             col_name='POA second column', inplace=True)
        self.assertLess(self.meas.data_filtered.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_withcol_notinplace(self):
        pts_before = self.meas.data_filtered.shape[0]
        df = self.meas.filter_irr(500, 600, ref_val=None, col_name=None,
                                  inplace=False)
        self.assertEqual(self.meas.data_filtered.shape[0], pts_before,
                         'Filter removed points from data_filtered.')
        self.assertIsInstance(df, pd.core.frame.DataFrame,
                              'Did not return DataFrame object.')
        self.assertLess(df.shape[0], pts_before,
                        'Filter did not remove points from returned DataFrame.')


class TestGetSummary(unittest.TestCase):
    def setUp(self):
        self.meas = pvc.CapData('meas')
        self.meas.load_data('./tests/data/', 'nrel_data.csv',
                            source='AlsoEnergy')
        self.meas.set_reg_trans(power='', poa='irr-poa-',
                                t_amb='temp--', w_vel='wind--')

    def test_col_names(self):
        self.meas.filter_irr(200, 500)
        smry = self.meas.get_summary()
        self.assertEqual(smry.columns[0], 'pts_after_filter',
                         'First column of summary data is not labeled '
                         'pts_after_filter.')
        self.assertEqual(smry.columns[1], 'pts_removed',
                         'First column of summary data is not labeled '
                         'pts_removed.')
        self.assertEqual(smry.columns[2], 'filter_arguments',
                         'First column of summary data is not labeled '
                         'filter_arguments.')


class TestFilterTime(unittest.TestCase):
    def setUp(self):
        self.pvsyst = pvc.CapData('pvsyst')
        self.pvsyst.load_data(path='./tests/data/',
                              fname='pvsyst_example_HourlyRes_2.CSV',
                              load_pvsyst=True)
        self.pvsyst.set_reg_trans(power='real_pwr--', poa='irr-poa-',
                                  t_amb='temp-amb-', w_vel='wind--')

    def test_start_end(self):
        self.pvsyst.filter_time(start='2/1/90', end='2/15/90')
        self.assertEqual(self.pvsyst.data_filtered.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.data_filtered.index[-1],
                         pd.Timestamp(year=1990, month=2, day=15, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_start_days(self):
        self.pvsyst.filter_time(start='2/1/90', days=15)
        self.assertEqual(self.pvsyst.data_filtered.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.data_filtered.index[-1],
                         pd.Timestamp(year=1990, month=2, day=16, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_end_days(self):
        self.pvsyst.filter_time(end='2/16/90', days=15)
        self.assertEqual(self.pvsyst.data_filtered.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.data_filtered.index[-1],
                         pd.Timestamp(year=1990, month=2, day=16, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_test_date(self):
        self.pvsyst.filter_time(test_date='2/16/90', days=30)
        self.assertEqual(self.pvsyst.data_filtered.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.data_filtered.index[-1],
                         pd.Timestamp(year=1990, month=3, day=3, hour=00),
                         'Last timestamp should be 3/2/1990 00:00')

    def test_start_end_not_inplace(self):
        df = self.pvsyst.filter_time(start='2/1/90', end='2/15/90',
                                     inplace=False)
        self.assertEqual(df.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(df.index[-1],
                         pd.Timestamp(year=1990, month=2, day=15, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_start_no_days(self):
        with self.assertWarns(UserWarning):
            self.pvsyst.filter_time(start='2/1/90')

    def test_end_no_days(self):
        with self.assertWarns(UserWarning):
            self.pvsyst.filter_time(end='2/1/90')

    def test_test_date_no_days(self):
        with self.assertWarns(UserWarning):
            self.pvsyst.filter_time(test_date='2/1/90')


class TestFilterPF(unittest.TestCase):
    def setUp(self):
        self.meas = pvc.CapData('meas')
        self.meas.load_data(path='./tests/data/', fname='nrel_data.csv',
                            source='AlsoEnergy')
        self.meas.set_reg_trans(power='', poa='irr-poa-',
                                t_amb='temp--', w_vel='wind--')

    def test_pf(self):
        pf = np.ones(5)
        pf = np.append(pf, np.ones(5) * -1)
        pf = np.append(pf, np.arange(0, 1, 0.1))
        self.meas.data['pf'] = np.tile(pf, 576)
        self.meas.data_filtered = self.meas.data.copy()
        self.meas.set_translation()
        self.meas.filter_pf(1)
        self.assertEqual(self.meas.data_filtered.shape[0], 5760,
                         'Incorrect number of points removed.')


class TestFilterOutliers(unittest.TestCase):
    def setUp(self):
        self.das = pvc.CapData('das')
        self.das.load_data(path='./tests/data/',
                           fname='example_meas_data_aeheaders.csv',
                           source='AlsoEnergy')
        self.das.set_reg_trans(power='-mtr-', poa='irr-poa-',
                               t_amb='temp-amb-', w_vel='wind--')

    def test_not_aggregated(self):
        with self.assertWarns(UserWarning):
            self.das.filter_outliers()


class Test_Csky_Filter(unittest.TestCase):
    """
    Tests for filter_clearsky method.
    """
    def setUp(self):
        self.meas = pvc.CapData('meas')
        loc = {'latitude': 39.742, 'longitude': -105.18,
               'altitude': 1828.8, 'tz': 'Etc/GMT+7'}
        sys = {'surface_tilt': 40, 'surface_azimuth': 180,
               'albedo': 0.2}
        self.meas.load_data(path='./tests/data/', fname='nrel_data.csv',
                       source='AlsoEnergy', clear_sky=True, loc=loc, sys=sys)

    def test_default(self):
        self.meas.filter_clearsky()

        self.assertLess(self.meas.data_filtered.shape[0],
                        self.meas.data.shape[0],
                        'Filtered dataframe should have less rows.')
        self.assertEqual(self.meas.data_filtered.shape[1],
                         self.meas.data.shape[1],
                         'Filtered dataframe should have equal number of cols.')
        for i, col in enumerate(self.meas.data_filtered.columns):
            self.assertEqual(col, self.meas.data.columns[i],
                             'Filter changed column {} to '
                             '{}'.format(self.meas.data.columns[i], col))

    def test_two_ghi_cols(self):
        self.meas.data['ws 2 ghi W/m^2'] = self.meas.view('irr-ghi-') * 1.05
        self.meas.set_translation()

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_mult_ghi_categories(self):
        cn = 'irrad ghi pyranometer W/m^2'
        self.meas.data[cn] = self.meas.view('irr-ghi-') * 1.05
        self.meas.set_translation()

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_no_clear_ghi(self):
        self.meas.drop_cols('ghi_mod_csky')

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_specify_ghi_col(self):
        self.meas.data['ws 2 ghi W/m^2'] = self.meas.view('irr-ghi-') * 1.05
        self.meas.set_translation
        self.meas.data_filtered = self.meas.data.copy()

        self.meas.filter_clearsky(ghi_col='ws 2 ghi W/m^2')

        self.assertLess(self.meas.data_filtered.shape[0],
                        self.meas.data.shape[0],
                        'Filtered dataframe should have less rows.')
        self.assertEqual(self.meas.data_filtered.shape[1],
                         self.meas.data.shape[1],
                         'Filtered dataframe should have equal number of cols.')
        for i, col in enumerate(self.meas.data_filtered.columns):
            self.assertEqual(col, self.meas.data.columns[i],
                             'Filter changed column {} to '
                             '{}'.format(self.meas.data.columns[i], col))

    def test_no_clear_sky(self):
        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky(window_length=2)


class TestCapTestCpResultsSingleCoeff(unittest.TestCase):
    """Tests for the capactiy test results method using a regression formula
    with a single coefficient."""

    def setUp(self):
        np.random.seed(9876789)

        self.meas = pvc.CapData('meas')
        self.sim = pvc.CapData('sim')
        # self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.meas.rc = {'x': [6]}

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

        self.meas.ols_model = das_model.fit()
        self.sim.ols_model = sim_model.fit()
        self.meas.data_filtered = pd.DataFrame()
        self.sim.data_filtered = pd.DataFrame()

    def test_return(self):
        res = pvc.captest_results(self.sim, self.meas, 100, '+/- 5',
                             print_res=False)

        self.assertIsInstance(res,
                              float,
                              'Returned value is not a tuple')


class TestCapTestCpResultsMultCoeffKwVsW(unittest.TestCase):
    """
    Setup and test to check automatic adjustment for kW vs W.
    """
    def test_pvals_default_false_kw_vs_w(self):
        np.random.seed(9876789)

        meas = pvc.CapData('meas')
        sim = pvc.CapData('sim')
        # cptest = pvc.CapTest(meas, sim, '+/- 5')
        meas.rc = pd.DataFrame({'poa': [6], 't_amb': [5], 'w_vel': [3]})

        nsample = 100
        e = np.random.normal(size=nsample)

        a = np.linspace(0, 10, 100)
        b = np.linspace(0, 10, 100) / 2.0
        c = np.linspace(0, 10, 100) + 3.0

        das_y = a + (a ** 2) + (a * b) + (a * c)
        sim_y = a + (a ** 2 * 0.9) + (a * b * 1.1) + (a * c * 0.8)

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({'power': das_y, 'poa': a,
                               't_amb': b, 'w_vel': c})
        sim_df = pd.DataFrame({'power': sim_y, 'poa': a,
                               't_amb': b, 'w_vel': c})

        meas.data = das_df
        meas.data['power'] /= 1000
        meas.set_reg_trans(power='power', poa='poa',
                           t_amb='t_amb', w_vel='w_vel')

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        meas.ols_model = das_model.fit()
        sim.ols_model = sim_model.fit()
        meas.data_filtered = pd.DataFrame()
        sim.data_filtered = pd.DataFrame()

        actual = meas.ols_model.predict(meas.rc)[0] * 1000
        expected = sim.ols_model.predict(meas.rc)[0]
        cp_rat_test_val = actual / expected

        with self.assertWarns(UserWarning):
            cp_rat = pvc.captest_results(sim, meas, 100, '+/- 5',
                                    check_pvalues=False, print_res=False)

        self.assertAlmostEqual(cp_rat, cp_rat_test_val, 6,
                               'captest_results did not return expected value.')

class TestCapTestCpResultsMultCoeff(unittest.TestCase):
    """
    Test captest_results function using a regression formula with multiple coef.
    """

    def setUp(self):
        np.random.seed(9876789)

        self.meas = pvc.CapData('meas')
        self.sim = pvc.CapData('sim')
        # self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.meas.rc = pd.DataFrame({'poa': [6], 't_amb': [5], 'w_vel': [3]})

        nsample = 100
        e = np.random.normal(size=nsample)

        a = np.linspace(0, 10, 100)
        b = np.linspace(0, 10, 100) / 2.0
        c = np.linspace(0, 10, 100) + 3.0

        das_y = a + (a ** 2) + (a * b) + (a * c)
        sim_y = a + (a ** 2 * 0.9) + (a * b * 1.1) + (a * c * 0.8)

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({'power': das_y, 'poa': a,
                               't_amb': b, 'w_vel': c})
        sim_df = pd.DataFrame({'power': sim_y, 'poa': a,
                               't_amb': b, 'w_vel': c})

        self.meas.data = das_df
        self.meas.set_reg_trans(power='power', poa='poa',
                                t_amb='t_amb', w_vel='w_vel')

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        self.meas.ols_model = das_model.fit()
        self.sim.ols_model = sim_model.fit()
        self.meas.data_filtered = pd.DataFrame()
        self.sim.data_filtered = pd.DataFrame()

    def test_pvals_default_false(self):
        actual = self.meas.ols_model.predict(self.meas.rc)[0]
        expected = self.sim.ols_model.predict(self.meas.rc)[0]
        cp_rat_test_val = actual / expected

        cp_rat = pvc.captest_results(self.sim, self.meas, 100, '+/- 5',
                                check_pvalues=False, print_res=False)

        self.assertEqual(cp_rat, cp_rat_test_val,
                         'captest_results did not return expected value.')

    def test_pvals_true(self):
        self.meas.ols_model.params['poa'] = 0
        self.sim.ols_model.params['poa'] = 0
        actual_pval_check = self.meas.ols_model.predict(self.meas.rc)[0]
        expected_pval_check = self.sim.ols_model.predict(self.meas.rc)[0]
        cp_rat_pval_check = actual_pval_check / expected_pval_check

        cp_rat = pvc.captest_results(self.sim, self.meas, 100, '+/- 5',
                                check_pvalues=True, pval=1e-15,
                                print_res=False)

        self.assertEqual(cp_rat, cp_rat_pval_check,
                         'captest_results did not return expected value.')

    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test_pvals_true_print(self):
        """
        This test uses the pytest autouse fixture defined above to
        capture the print to stdout and test it, so it must be run
        using pytest.  Run just this test using 'pytest tests/
        test_CapData.py::TestCapTestCpResultsMultCoeff::test_pvals_true_print'
        """
        self.meas.ols_model.params['poa'] = 0
        self.sim.ols_model.params['poa'] = 0

        pvc.captest_results(self.sim, self.meas, 100, '+/- 5',
                                check_pvalues=True, pval=1e-15,
                                print_res=True)

        captured = self.capsys.readouterr()

        results_str = ('Using reporting conditions from das. \n\n'

                       'Capacity Test Result:    FAIL\n'
                       'Modeled test output:          66.451\n'
                       'Actual test output:           72.429\n'
                       'Tested output ratio:          1.090\n'
                       'Tested Capacity:              108.996\n'
                       'Bounds:                       95.0, 105.0\n\n\n')

        self.assertEqual(results_str, captured.out)

    def test_formulas_match(self):
        sim = pvc.CapData('sim')
        sim.data_filtered = pd.DataFrame()
        das = pvc.CapData('das')
        das.data_filtered = pd.DataFrame()

        sim.reg_fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) - 1'

        with self.assertWarns(UserWarning):
            pvc.captest_results(sim, das, 100, '+/- 5', check_pvalues=True)


if __name__ == '__main__':
    unittest.main()
