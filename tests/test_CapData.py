import os
import collections
import unittest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import pvlib

from .context import captest as pvc
from .context import capdata as cpd

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


    def test_source_alsoenergy(self):
        das_1 = pvc.CapData()
        das_1.load_data(path='./tests/data/col_naming_examples/',
                      fname='ae_site1.csv', source='AlsoEnergy')
        col_names1 = ['Elkor Production Meter PowerFactor, ',
                      'Elkor Production Meter KW, kW',
                      'Weather Station 1 TempF, °F', 'Weather Station 2 Sun2, W/m²',
                      'Weather Station 1 Sun, W/m²', 'Weather Station 1 WindSpeed, mph',
                      'index']
        self.assertTrue(all(das_1.df.columns == col_names1),
                        'Column names are not expected value for ae_site1')

        das_2 = pvc.CapData()
        das_2.load_data(path='./tests/data/col_naming_examples/',
                      fname='ae_site2.csv', source='AlsoEnergy')
        col_names2 = ['Acuvim II Meter PowerFactor, PF', 'Acuvim II Meter KW, kW',
                      'Weather Station 1 TempF, °F', 'Weather Station 3 TempF, °F',
                      'Weather Station 2 Sun2, W/m²', 'Weather Station 4 Sun2, W/m²',
                      'Weather Station 1 Sun, W/m²', 'Weather Station 3 Sun, W/m²',
                      'Weather Station 1 WindSpeed, mph',
                      'Weather Station 3 WindSpeed, mph',
                      'index']
        self.assertTrue(all(das_2.df.columns == col_names2),
                        'Column names are not expected value for ae_site1')

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
        out = self.cdata._CapData__series_type(test_series, cpd.type_defs)

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
            out.append(self.cdata._CapData__series_type(test_series, cpd.type_defs))
            i += 1
        out_np = np.array(out)

        self.assertTrue(all(out_np == 'irr'),
                        'Result is not consistent after repeated runs.')

    def test_series_type_valErr(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(name=name)
        out = self.cdata._CapData__series_type(test_series, cpd.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr-valuesError',
                         'Returned object is not "irr-valuesError".')

    def test_series_type_no_str(self):
        name = 'should not return key string'
        test_series = pd.Series(name=name)
        out = self.cdata._CapData__series_type(test_series, cpd.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertIs(out, '',
                      'Returned object is not empty string.')


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
        cptest = pvc.CapTest(meas, self.pvsyst, '+/- 5')

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

class Test_csky(unittest.TestCase):
    """ Test function wrapping pvlib get_clearsky method of Location."""
    def test_pvlib_location(self):
        loc = {'latitude': 30.274583,
               'longitude': -97.740352,
               'altitude': 500,
               'tz': 'America/Chicago'}

        loc_obj = cpd.pvlib_location(loc)

        self.assertIsInstance(loc_obj,
                              pvlib.location.Location,
                              'Did not return instance of\
                               pvlib Location')


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

    def test_csky(self):
        loc = {'latitude': 30.274583,
               'longitude': -97.740352,
               'altitude': 500,
               'tz': 'America/Chicago'}

        sys = {'surface_tilt': 20,
               'surface_azimuth': 180,
               'albedo': 0.2}

        meas = pvc.CapData()
        df = meas.load_das('./tests/data/', 'example_meas_data.csv')
        # meas.df.index = meas.df.index.localize

        csky_ghi_poa = cpd.csky(df, loc=loc, sys=sys)

        self.assertIsInstance(cksy_ghi_poa, pandas.core.frame.DataFrame,
                              'Did not return a pandas dataframe.')
        self.assertGreater(csky_ghi_poa.loc['10/9/1990 12:30', 'poa'],
                           csky_ghi_poa.loc['10/9/1990 12:30', 'ghi'],
                           'POA is not greater than GHI at 12:30.')

"""
Change csky to two functions for creating pvlib location and system objects.
Separate function calling location function to calculate GHI
Separate function calling location and system to calculate POA
- inplace gives concat or return option
load_data calls final funtion with in place to get ghi and poa
"""


if __name__ == '__main__':
    unittest.main()
