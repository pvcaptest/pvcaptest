import os
import collections
import unittest
import pytz
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import pvlib

from .context import captest as pvc
from .context import capdata as cpd

data = np.arange(0, 1300, 54.167)
index = pd.DatetimeIndex(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

capdata = pvc.CapData('capdata')
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
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

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

class Test_pvlib_loc_sys(unittest.TestCase):
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

    def test_pvlib_system(self):
        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        tracker_sys2 = {'max_angle': 52, 'gcr': 0.3}

        fx_sys = cpd.pvlib_system(fixed_sys)
        trck_sys1 = cpd.pvlib_system(tracker_sys1)
        trck_sys2 = cpd.pvlib_system(tracker_sys1)

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

        self.meas = pvc.CapData()
        self.df = self.meas.load_das('./tests/data/', 'example_meas_data.csv')

    def test_get_tz_index_df(self):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe without a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.DatetimeIndex(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/Chicago')
        ix_2days = pd.DatetimeIndex(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/Chicago')
        ix_dst = ix_3days.append(ix_2days)
        ix_dst = ix_dst.tz_localize(None)
        self.df.index = ix_dst

        self.tz_ix = cpd.get_tz_index(self.df, self.loc)

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
        ix_3days = pd.DatetimeIndex(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/Chicago')
        ix_2days = pd.DatetimeIndex(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/Chicago')
        ix_dst = ix_3days.append(ix_2days)
        self.df.index = ix_dst

        self.tz_ix = cpd.get_tz_index(self.df, self.loc)

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
        ix_3days = pd.DatetimeIndex(start='11/3/2018', periods=864, freq='5min',
                                    tz='America/New_York')
        ix_2days = pd.DatetimeIndex(start='3/9/2019', periods=576, freq='5min',
                                    tz='America/New_York')
        ix_dst = ix_3days.append(ix_2days)
        self.df.index = ix_dst

        with self.assertWarns(UserWarning):
            self.tz_ix = cpd.get_tz_index(self.df, self.loc)

    def test_get_tz_index_ix_tz(self):
        """Test that get_tz_index function returns a datetime index
           with a timezone when passed a datetime index with a timezone."""
        self.ix = pd.DatetimeIndex(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/Chicago')
        self.tz_ix = cpd.get_tz_index(self.ix, self.loc)

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
        self.ix = pd.DatetimeIndex(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/New_York')

        with self.assertWarns(UserWarning):
            self.tz_ix = cpd.get_tz_index(self.ix, self.loc)

    def test_get_tz_index_ix(self):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a datetime index without a timezone."""
        self.ix = pd.DatetimeIndex(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/Chicago')
        # remove timezone info but keep missing  hour and extra hour due to DST
        self.ix = self.ix.tz_localize(None)
        self.tz_ix = cpd.get_tz_index(self.ix, self.loc)

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
        csky_ghi_poa = cpd.csky(self.df, loc=self.loc, sys=self.sys)

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
        csky_ghi_poa = cpd.csky(self.df, loc=self.loc, sys=self.sys,
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
        csky_ghi_poa = cpd.csky(self.df, loc=self.loc, sys=self.sys,
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
        csky_ghi_poa = cpd.csky(self.df, loc=self.loc, sys=self.sys,
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
        csky_ghi_poa = cpd.csky(self.df, loc=self.loc, sys=self.sys,
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
class TestAggSensors(unittest.TestCase):
    def setUp(self):
        self.das = pvc.CapData('das')
        self.das.load_data(path='./examples/data/',
                           fname='example_meas_data.csv', source='AlsoEnergy')
        self.das.set_reg_trans(power='-mtr-', poa='irr-poa-',
                               t_amb='temp-amb-', w_vel='wind--')

    def test_agg_map_none(self):
        self.das.agg_sensors()
        self.assertEqual(self.das.df_flt.shape[1], self.das.df.shape[1] + 4,
                         'Returned df does not include 4 additional cols.')
        self.assertEqual(self.das.df_flt.shape[0], self.das.df.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum', self.das.df_flt.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean', self.das.df_flt.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean', self.das.df_flt.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean', self.das.df_flt.columns,
                      'Mean of wind trans group not in aggregated df.')

    def test_agg_map_none_inplace_false(self):
        df_flt_copy = self.das.df_flt.copy()
        df = self.das.agg_sensors(inplace=False)
        self.assertEqual(df.shape[1], self.das.df.shape[1] + 4,
                         'Returned df does not include 4 additional cols.')
        self.assertEqual(df.shape[0], self.das.df.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum', df.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean', df.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean', df.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean', df.columns,
                      'Mean of wind trans group not in aggregated df.')
        self.assertTrue(df_flt_copy.equals(self.das.df_flt),
                        'Method with inplace false changed df_flt attribute.')

    def test_agg_map_none_keep_false(self):
        self.das.agg_sensors(keep=False)
        self.assertEqual(self.das.df_flt.shape[1], 4,
                         'Returned dataframe does not have 4 columns.')
        self.assertEqual(self.das.df_flt.shape[0], self.das.df.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('-mtr-sum', self.das.df_flt.columns,
                      'Sum of power trans group not in aggregated df.')
        self.assertIn('irr-poa-mean', self.das.df_flt.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('temp-amb-mean', self.das.df_flt.columns,
                      'Mean of amb temp trans group not in aggregated df.')
        self.assertIn('wind--mean', self.das.df_flt.columns,
                      'Mean of wind trans group not in aggregated df.')

    def test_agg_map_non_str_func(self):
        self.das.agg_sensors(agg_map={'irr-poa-': np.mean})
        self.assertEqual(self.das.df_flt.shape[1], self.das.df.shape[1] + 1,
                         'Returned df does not include 1 additional col.')
        self.assertEqual(self.das.df_flt.shape[0], self.das.df.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('irr-poa-mean', self.das.df_flt.columns,
                      'Mean of poa trans group not in aggregated df.')

    def test_agg_map_mix_funcs(self):
        self.das.agg_sensors(agg_map={'irr-poa-': [np.mean, 'sum']})
        self.assertEqual(self.das.df_flt.shape[1], self.das.df.shape[1] + 2,
                         'Returned df does not include 2 additional col.')
        self.assertEqual(self.das.df_flt.shape[0], self.das.df.shape[0],
                         'Agg method inadverdently changed number of rows.')
        self.assertIn('irr-poa-mean', self.das.df_flt.columns,
                      'Mean of poa trans group not in aggregated df.')
        self.assertIn('irr-poa-sum', self.das.df_flt.columns,
                      'Sum of poa trans group not in aggregated df.')

    def test_agg_map_update_reg_trans(self):
        self.das.agg_sensors()
        self.assertEqual(self.das.reg_trans['power'], '-mtr-sum',
                         'Power reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['poa'], 'irr-poa-mean',
                         'POA reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['t_amb'], 'temp-amb-mean',
                         'Amb temp reg_trans not updated to agg column.')
        self.assertEqual(self.das.reg_trans['w_vel'], 'wind--mean',
                         'Wind velocity reg_trans not updated to agg column.')
    # test for updates to reg_trans and trans



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
        self.meas.rep_cond(irr_bal=True, perc_flt=20)
        self.assertIsInstance(self.meas.rc, pd.core.frame.DataFrame,
                              'No dataframe stored in the rc attribute.')
        self.assertNotEqual(self.meas.rc['poa'][0], meas2.rc['poa'][0],
                            'Irr_bal function returned same result\
                             as w/o irr_bal')

    def test_irr_bal_inplace_wvel(self):
        self.meas.rep_cond(irr_bal=True, perc_flt=20, w_vel=50)
        self.assertEqual(self.meas.rc['w_vel'][0], 50,
                         'Wind velocity not overwritten by user value')

    def test_irr_bal_inplace_no_perc_flt(self):
        with self.assertWarns(UserWarning):
            self.meas.rep_cond(irr_bal=True, perc_flt=None)


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
        self.pvsyst.rep_cond(freq='M', irr_bal=True, perc_flt=20)
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
        pred_caps = self.pvsyst.predict_capacities(irr_flt=True, perc_flt=20)
        july_grpby = pred_caps.loc['1990-07-01', 'PredCap']

        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is not a Dataframe.')
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

        self.pvsyst.df_flt = self.pvsyst.df_flt.loc['7/1/90':'7/31/90', :]
        self.pvsyst.rep_cond()
        self.pvsyst.filter_irr(0.8, 1.2, ref_val=self.pvsyst.rc['poa'][0])
        df = self.pvsyst.rview(['power', 'poa', 't_amb', 'w_vel'],
                               filtered_data=True)
        rename = {df.columns[0]: 'power',
                  df.columns[1]: 'poa',
                  df.columns[2]: 't_amb',
                  df.columns[3]: 'w_vel'}
        df = df.rename(columns=rename)
        reg = cpd.fit_model(df)
        july_manual = reg.predict(self.pvsyst.rc)[0]
        self.assertEqual(july_manual, july_grpby,
                         'Manual prediction for July {} is not equal'
                         'to the predict_capacites groupby'
                         'prediction {}'.format(july_manual, july_grpby))

    def test_no_irr_flt(self):
        self.pvsyst.rep_cond(freq='M')
        pred_caps = self.pvsyst.predict_capacities(irr_flt=False)
        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is not a Dataframe.')
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

    def test_rc_from_irrBal(self):
        self.pvsyst.rep_cond(freq='M', irr_bal=True, perc_flt=20)
        pred_caps = self.pvsyst.predict_capacities(irr_flt=False)
        self.assertIsInstance(pred_caps, pd.core.frame.DataFrame,
                              'Returned object is {} not a\
                               Dataframe.'.format(type(pred_caps)))
        self.assertEqual(pred_caps.shape[0], 12,
                         'Predicted capacities does not have 12 rows.')

    def test_seasonal_freq(self):
        self.pvsyst.rep_cond(freq='BQ-NOV')
        pred_caps = self.pvsyst.predict_capacities(irr_flt=True, perc_flt=20)
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
        self.meas.df['POA second column'] = self.meas.rview('poa').values
        self.meas._CapData__set_trans()
        with self.assertWarns(UserWarning):
            col = self.meas._CapData__get_poa_col()

    def test_lowhigh_nocol(self):
        pts_before = self.meas.df_flt.shape[0]
        self.meas.filter_irr(500, 600, ref_val=None, col_name=None,
                             inplace=True)
        self.assertLess(self.meas.df_flt.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_lowhigh_colname(self):
        pts_before = self.meas.df_flt.shape[0]
        self.meas.df['POA second column'] = self.meas.rview('poa').values
        self.meas._CapData__set_trans()
        self.meas.df_flt = self.meas.df.copy()
        self.meas.filter_irr(500, 600, ref_val=None,
                             col_name='POA second column', inplace=True)
        self.assertLess(self.meas.df_flt.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_nocol(self):
        pts_before = self.meas.df_flt.shape[0]
        self.meas.filter_irr(0.8, 1.2, ref_val=500, col_name=None,
                             inplace=True)
        self.assertLess(self.meas.df_flt.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_withcol(self):
        pts_before = self.meas.df_flt.shape[0]
        self.meas.df['POA second column'] = self.meas.rview('poa').values
        self.meas._CapData__set_trans()
        self.meas.df_flt = self.meas.df.copy()
        self.meas.filter_irr(0.8, 1.2, ref_val=500,
                             col_name='POA second column', inplace=True)
        self.assertLess(self.meas.df_flt.shape[0], pts_before,
                        'Filter did not remove points.')

    def test_refval_withcol_notinplace(self):
        pts_before = self.meas.df_flt.shape[0]
        df = self.meas.filter_irr(500, 600, ref_val=None, col_name=None,
                                  inplace=False)
        self.assertEqual(self.meas.df_flt.shape[0], pts_before,
                         'Filter removed points from df_flt.')
        self.assertIsInstance(df, pd.core.frame.DataFrame,
                              'Did not return DataFrame object.')
        self.assertLess(df.shape[0], pts_before,
                        'Filter did not remove points from returned DataFrame.')


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
        self.assertEqual(self.pvsyst.df_flt.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.df_flt.index[-1],
                         pd.Timestamp(year=1990, month=2, day=15, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_start_days(self):
        self.pvsyst.filter_time(start='2/1/90', days=15)
        self.assertEqual(self.pvsyst.df_flt.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.df_flt.index[-1],
                         pd.Timestamp(year=1990, month=2, day=16, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_end_days(self):
        self.pvsyst.filter_time(end='2/16/90', days=15)
        self.assertEqual(self.pvsyst.df_flt.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.df_flt.index[-1],
                         pd.Timestamp(year=1990, month=2, day=16, hour=00),
                         'Last timestamp should be 2/15/1990 00:00')

    def test_test_date(self):
        self.pvsyst.filter_time(test_date='2/16/90', days=30)
        self.assertEqual(self.pvsyst.df_flt.index[0],
                         pd.Timestamp(year=1990, month=2, day=1, hour=0),
                         'First timestamp should be 2/1/1990')
        self.assertEqual(self.pvsyst.df_flt.index[-1],
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
        self.meas.df['pf'] = np.tile(pf, 576)
        self.meas.df_flt = self.meas.df.copy()
        self.meas._CapData__set_trans()
        self.meas.filter_pf(1)
        self.assertEqual(self.meas.df_flt.shape[0], 5760,
                         'Incorrect number of points removed.')


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

        self.assertLess(self.meas.df_flt.shape[0],
                        self.meas.df.shape[0],
                        'Filtered dataframe should have less rows.')
        self.assertEqual(self.meas.df_flt.shape[1],
                         self.meas.df.shape[1],
                         'Filtered dataframe should have equal number of cols.')
        for i, col in enumerate(self.meas.df_flt.columns):
            self.assertEqual(col, self.meas.df.columns[i],
                             'Filter changed column {} to '
                             '{}'.format(self.meas.df.columns[i], col))

    def test_two_ghi_cols(self):
        self.meas.df['ws 2 ghi W/m^2'] = self.meas.view('irr-ghi-') * 1.05
        self.meas._CapData__set_trans()

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_mult_ghi_categories(self):
        cn = 'irrad ghi pyranometer W/m^2'
        self.meas.df[cn] = self.meas.view('irr-ghi-') * 1.05
        self.meas._CapData__set_trans()

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_no_clear_ghi(self):
        self.meas.drop_cols('ghi_mod_csky')

        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky()

    def test_specify_ghi_col(self):
        self.meas.df['ws 2 ghi W/m^2'] = self.meas.view('irr-ghi-') * 1.05
        self.meas._CapData__set_trans()
        self.meas.df_flt = self.meas.df.copy()

        self.meas.filter_clearsky(ghi_col='ws 2 ghi W/m^2')

        self.assertLess(self.meas.df_flt.shape[0],
                        self.meas.df.shape[0],
                        'Filtered dataframe should have less rows.')
        self.assertEqual(self.meas.df_flt.shape[1],
                         self.meas.df.shape[1],
                         'Filtered dataframe should have equal number of cols.')
        for i, col in enumerate(self.meas.df_flt.columns):
            self.assertEqual(col, self.meas.df.columns[i],
                             'Filter changed column {} to '
                             '{}'.format(self.meas.df.columns[i], col))

    def test_no_clear_sky(self):
        with self.assertWarns(UserWarning):
            self.meas.filter_clearsky(window_length=2)


class TestTopLevelFuncs(unittest.TestCase):
    def test_perc_bounds_perc(self):
        bounds = cpd.perc_bounds(20)
        self.assertEqual(bounds[0], 0.8,
                         '{} for 20 perc is not 0.8'.format(bounds[0]))
        self.assertEqual(bounds[1], 1.2,
                         '{} for 20 perc is not 1.2'.format(bounds[1]))
    def test_perc_bounds_tuple(self):
        bounds = cpd.perc_bounds((15, 40))
        self.assertEqual(bounds[0], 0.85,
                         '{} for 15 perc is not 0.85'.format(bounds[0]))
        self.assertEqual(bounds[1], 1.4,
                         '{} for 40 perc is not 1.4'.format(bounds[1]))

    def test_flt_irr(self):
        rng = np.arange(0, 1000)
        df = pd.DataFrame(np.array([rng, rng+100, rng+200]).T,
                          columns = ['weather_station irr poa W/m^2',
                                     'col_1', 'col_2'])
        df_flt = cpd.flt_irr(df, 'weather_station irr poa W/m^2', 50, 100)

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

    def test_filter_grps(self):
        pvsyst = pvc.CapData('pvsyst')
        pvsyst.load_data(path='./tests/data/',
                         fname='pvsyst_example_HourlyRes_2.CSV',
                         load_pvsyst=True)
        pvsyst.set_reg_trans(power='real_pwr--', poa='irr-poa-',
                             t_amb='temp-amb-', w_vel='wind--')
        pvsyst.filter_irr(200, 800)
        pvsyst.rep_cond(freq='M')
        grps = pvsyst.df_flt.groupby(pd.Grouper(freq='M', label='left'))
        poa_col = pvsyst.trans[pvsyst.reg_trans['poa']][0]

        grps_flt = cpd.filter_grps(grps, pvsyst.rc, poa_col, 0.8, 1.2)

        self.assertIsInstance(grps_flt, pd.core.groupby.groupby.DataFrameGroupBy,
                              'Returned object is not a dataframe groupby.')

        self.assertEqual(grps.ngroups, grps_flt.ngroups,
                         'Returned groubpy does not have the same number of\
                          groups as passed groupby.')

        cnts_before_flt = grps.count()[poa_col]
        cnts_after_flt = grps_flt.count()[poa_col]
        less_than = all(cnts_after_flt < cnts_before_flt)
        self.assertTrue(less_than, 'Points were not removed for each group.')

if __name__ == '__main__':
    unittest.main()
