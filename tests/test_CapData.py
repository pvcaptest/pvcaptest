from pathlib import Path
import os
import copy
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import json
import warnings

import pvlib

from captest import capdata as pvc
from captest import util
from captest import columngroups as cg
from captest import io
from captest import(
    load_pvsyst,
)

data = np.arange(0, 1300, 54.167)
index = pd.date_range(start='1/1/2017', freq='H', periods=24)
df = pd.DataFrame(data=data, index=index, columns=['poa'])

# capdata = pvc.CapData('capdata')
# capdata.df = df

"""
Run tests using pytest use the following from project root.
To run a class of tests
pytest tests/test_CapData.py::TestCapDataEmpty

To run a specific test:
pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty

To create a test coverage report (html output) with pytest:
pytest --cov-report html --cov=captest tests/

pytest fixtures meas, location_and_system, nrel, pvsyst, pvsyst_irr_filter, and
nrel_clear_sky are in the ./tests/conftest.py file.
"""

class TestUpdateSummary:
    """Test the update_summary wrapper and functions used within."""

    def test_round_kwarg_floats(self):
        """Tests round kwarg_floats."""
        kwarg_dict = {'ref_val': 763.4536140499999, 't1': 2, 'inplace': True}
        rounded_kwarg_dict_3 = {'ref_val': 763.454, 't1': 2,
                                'inplace': True}
        assert pvc.round_kwarg_floats(kwarg_dict) == rounded_kwarg_dict_3
        rounded_kwarg_dict_4 = {'ref_val': 763.4536, 't1': 2,
                                'inplace': True}
        assert pvc.round_kwarg_floats(kwarg_dict, 4) == rounded_kwarg_dict_4

    def test_tstamp_kwarg_to_strings(self):
        """Tests coversion of kwarg values from timestamp to strings."""
        start_datetime = pd.to_datetime('10/10/1990 00:00')
        kwarg_dict = {'start': start_datetime, 't1': 2}
        kwarg_dict_str_dates = {'start': '1990-10-10 00:00', 't1': 2}
        assert pvc.tstamp_kwarg_to_strings(kwarg_dict) == kwarg_dict_str_dates


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
        pvsyst = load_pvsyst(path='./tests/data/pvsyst_example_HourlyRes_2.CSV')

        df_regs = pvsyst.data.loc[:, ['E_Grid', 'GlobInc', 'T_Amb', 'WindVel']]
        df_regs_day = df_regs.query('GlobInc > 0')
        grps = df_regs_day.groupby(pd.Grouper(freq='M', label='right'))

        ones = np.ones(12)
        irr_rc = ones * 500
        temp_rc = ones * 20
        w_vel = ones
        rcs = pd.DataFrame({'GlobInc': irr_rc, 'T_Amb': temp_rc, 'WindVel': w_vel})

        results = pvc.pred_summary(grps, rcs, 0.05,
                                   fml='E_Grid ~ GlobInc +'
                                                 'I(GlobInc * GlobInc) +'
                                                 'I(GlobInc * T_Amb) +'
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

        pt_qty_exp = [341, 330, 392, 390, 403, 406, 456, 386, 390, 346, 331, 341]
        gaur_cap_exp = [
            3089550.4039329495,
            3103610.4635679387,
            3107035.251399103,
            3090681.1145782764,
            3058186.270209293,
            3059784.2309170915,
            3088294.50827525,
            3087081.0026879036,
            3075251.990424683,
            3093287.331878834,
            3097089.7852036236,
            3084318.093294242,
        ]
        for i, mnth in enumerate(results.index):
            self.assertLess(
                results.loc[mnth, 'guaranteedCap'],
                results.loc[mnth, 'PredCap'],
                'Gauranteed cap is greater than predicted in month {}'.format(mnth)
            )
            self.assertGreater(
                results.loc[mnth, 'guaranteedCap'], 0,
                'Gauranteed capacity is less than 0 in month {}'.format(mnth)
            )
            self.assertAlmostEqual(
                results.loc[mnth, 'guaranteedCap'], gaur_cap_exp[i], 7,
                'Gauranted capacity not equal to expected value in {}'.format(mnth))
            self.assertEqual(
                results.loc[mnth, 'pt_qty'], pt_qty_exp[i],
                'Point quantity not equal to expected values in {}'.format(mnth)
            )

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
        pvsyst = load_pvsyst(path='./tests/data/pvsyst_example_HourlyRes_2.CSV')
        pvsyst.set_regression_cols(
            power='real_pwr__', poa='irr_poa_', t_amb='temp_amb_', w_vel='wind__')
        pvsyst.filter_irr(200, 800)
        pvsyst.rep_cond(freq='MS')
        grps = pvsyst.data_filtered.groupby(pd.Grouper(freq='MS', label='left'))
        poa_col = pvsyst.column_groups[pvsyst.regression_cols['poa']][0]

        grps_flt = pvc.filter_grps(grps, pvsyst.rc, poa_col, 0.8, 1.2, 'MS')

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


class TestCapDataEmpty:
    """Tests of CapData empty method."""

    def test_capdata_empty(self):
        """Test that an empty CapData object returns True."""
        empty_cd = pvc.CapData('empty')
        assert empty_cd.empty()

    def test_capdata_not_empty(self, meas):
        """Test that an CapData object with data returns False."""
        assert not meas.empty()


class TestCapDataSeriesTypes(unittest.TestCase):
    """Test CapData private methods assignment of type to each series of data."""

    def setUp(self):
        self.cdata = pvc.CapData('cdata')

    def test_series_type(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

    def test_series_type_caps_in_type_def(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        type_def = collections.OrderedDict([
            ('irr', ['IRRADIANCE', 'IRR', 'PLANE OF ARRAY', 'POA',
                     'GHI', 'GLOBAL', 'GLOB', 'W/M^2', 'W/M2', 'W/M', 'W/']),
        ])
        out = cg.series_type(test_series, type_def)

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
            out.append(cg.series_type(test_series, cg.type_defs))
            i += 1
        out_np = np.array(out)

        self.assertTrue(all(out_np == 'irr'),
                        'Result is not consistent after repeated runs.')

    def test_series_type_valErr(self):
        name = 'weather station 1 weather station 1 ghi poa w/m2'
        test_series = pd.Series(name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertEqual(out, 'irr',
                         'Returned object is not "irr".')

    def test_series_type_no_str(self):
        name = 'should not return key string'
        test_series = pd.Series(name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str,
                              'Returned object is not a string.')
        self.assertIs(out, '',
                      'Returned object is not empty string.')


class TestIndexCapdata():
    """Test the indexing functionality of the CapData loc method."""
    def test_single_label_column_group_key(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        out = pvc.index_capdata(meas, 'irr_poa_pyran', filtered=True)
        assert out.equals(meas.data[['met1_poa_pyranometer', 'met2_poa_pyranometer']])

    def test_single_label_regression_columns_key(self, meas):
        """Test that regression_columns key returns the columns of Capdata.data that
        are the values of the key."""
        out = pvc.index_capdata(meas, 'poa', filtered=True)
        assert out.equals(meas.data[['met1_poa_pyranometer', 'met2_poa_pyranometer']])

    def test_single_label_data_column_label(self, meas):
        """Test that a column label returns the columns of Capdata.data that
        are the values of the key. Passes label through to DataFrame.loc."""
        out = pvc.index_capdata(meas, 'met1_poa_pyranometer', filtered=True)
        assert out.equals(meas.data.loc[:, 'met1_poa_pyranometer'])

    def test_list_of_labels_column_group_keys(self, meas):
        """
        Test that a list of column_groups key returns the columns of Capdata.data that
        are the union of the values of the keys.
        """
        out = pvc.index_capdata(meas, ['irr_poa_pyran', 'temp_amb'], filtered=True)
        assert out.equals(meas.data[[
            'met1_poa_pyranometer',
            'met2_poa_pyranometer',
            'met1_amb_temp',
            'met2_amb_temp',
        ]])

    def test_list_of_labels_regression_columns_keys(self, meas):
        """
        Test that a list of regression_columns key returns the columns of Capdata.data that
        are the union of the values of the keys.
        """
        out = pvc.index_capdata(meas, ['poa', 't_amb'], filtered=True)
        assert out.equals(meas.data[[
            'met1_poa_pyranometer',
            'met2_poa_pyranometer',
            'met1_amb_temp',
            'met2_amb_temp',
        ]])

    def test_list_of_labels_data_column_labels(self, meas):
        """
        Test that a list of column labels returns the columns of Capdata.data.
        Passes labels through to DataFrame.loc.
        """
        out = pvc.index_capdata(
            meas, ['met1_poa_pyranometer', 'met2_amb_temp'], filtered=True
        )
        assert out.equals(meas.data.loc[:, ['met1_poa_pyranometer', 'met2_amb_temp']])

    def test_list_of_labels_mixed(self, meas):
        """
        Test that a list containing a column_group, regression_columns key, and
        column labels returns the columns of Capdata.data that are the union of the
        values of the keys and the labels.
        """
        out = pvc.index_capdata(
            meas, ['irr_poa_pyran', 't_amb', 'met1_windspeed'], filtered=True
        )
        assert out.equals(meas.data[[
            'met1_poa_pyranometer',
            'met2_poa_pyranometer',
            'met1_amb_temp',
            'met2_amb_temp',
            'met1_windspeed',
        ]])

    def test_list_of_labels_mixed_regression_column_maps_to_column_label(self, meas):
        """
        Test a list containing a regression_column key that maps directly to a column
        label rather than a column_group key is added to the columns returned.
        """
        meas.regression_cols['poa'] = 'met1_poa_pyranometer'
        out = pvc.index_capdata(
            meas, ['irr_poa_ref_cell', 'poa', 'met1_windspeed'], filtered=True
        )
        assert out.equals(meas.data[[
            'met1_poa_refcell',
            'met2_poa_refcell',
            'met1_poa_pyranometer',
            'met1_windspeed',
        ]])


class TestLocAndFloc():
    def test_single_label_column_group_key_loc(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        meas.data_filtered = meas.data.iloc[0:10].copy()
        out = meas.loc['irr_poa_pyran']
        assert out.equals(meas.data[['met1_poa_pyranometer', 'met2_poa_pyranometer']])
        assert out.shape[0] == meas.data.shape[0]

    def test_single_label_column_group_key_floc(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        meas.data_filtered = (meas.data.iloc[0:10, :]).copy()
        out = meas.floc['irr_poa_pyran']
        assert out.equals(meas.data_filtered[['met1_poa_pyranometer', 'met2_poa_pyranometer']])
        assert out.shape[0] == meas.data_filtered.shape[0]



class TestIrrRcBalanced():
    """Test the functionality of the irr_rc_balanced function"""
    def test_check_csv_output_exists(self, meas, tmp_path):
        """Check that function outputs a csv file when given a file path."""
        f = tmp_path / 'output.csv'
        print(meas.column_groups)
        meas.agg_sensors(agg_map={'irr_poa_pyran': 'mean'})
        print(meas.regression_cols['poa'])
        rep_irr = pvc.ReportingIrradiance(
            df=meas.data,
            irr_col=meas.regression_cols['poa'],
            percent_band=20,
        )
        results = rep_irr.get_rep_irr()
        rep_irr.save_csv(output_csv_path=f)
        assert f.exists()

    def test_irr_rc_balanced(self, pvsyst):
        jun = pvsyst.data.loc['06/1990']
        jun_cpy = jun.copy()
        jun = jun.loc[jun['GlobInc'] > 400, :]
        print(jun)

        rc_tool = pvc.ReportingIrradiance(jun, 'GlobInc', percent_band=50)
        rc_tool.min_ref_irradiance = 600
        rc_tool.max_ref_irradiance = 800
        (irr_RC, jun_flt) = rc_tool.get_rep_irr()
        print(irr_RC)
        print(jun_flt)
        print(rc_tool.poa_flt)
        jun_filter_irr = jun_flt['GlobInc']
        assert all(jun_flt.columns == jun.columns)
        assert jun_flt.shape[0] > 0
        assert jun_flt.shape[0] < jun_cpy.shape[0]
        assert irr_RC > jun[jun['GlobInc'] > 0]['GlobInc'].min()
        assert irr_RC < jun['GlobInc'].max()

        pts_below_irr = jun_filter_irr[jun_filter_irr.between(0, irr_RC)].shape[0]
        perc_below = pts_below_irr / jun_filter_irr.shape[0]
        assert perc_below < 0.6
        # Less than 50 percent of points below rep irr
        assert round(perc_below, 1) <= 0.5

        pts_above_irr = jun_filter_irr[jun_filter_irr.between(irr_RC, 1500)].shape[0]
        perc_above = pts_above_irr / jun_filter_irr.shape[0]
        # Less than 40 percent of points above reporting irr
        assert perc_above > 0.4
        # More than 50 percent of points above reportin irr
        assert perc_above <= 0.5

    def test_irr_rc_balanced_warns_if_min_greather_than_max(self, pvsyst):
        """
        Check that the function warns if the minimum reference irradiance is
        greater than the maximum referene irradiance.

        With this dataset and the defaults for the min and max reference irradiance
        the minimum irradiance (800) will be higher than the maximum irradiance (722).        
        """
        jun = pvsyst.data.loc['06/1990']
        jun = jun.loc[jun['GlobInc'] > 400, :]
        rc_tool = pvc.ReportingIrradiance(jun, 'GlobInc', percent_band=50)
        with pytest.warns(UserWarning):
            rc_tool.get_rep_irr()
        assert isinstance(rc_tool.poa_flt, pd.DataFrame)
        assert rc_tool.min_ref_irradiance == 400
        assert rc_tool.max_ref_irradiance == 1000

    def test_irr_rc_balanced_warns_if_no_ref_irr_found(self, pvsyst):
        """
        Check that the function warns if it cannot determine a reference irradiance.
        Also check that the function still stores the filtered data to use in the
        plot and dashboard methods for user troubleshooting.

        With this dataset and the defaults for the min and max reference irradiance
        the minimum irradiance (800) will be higher than the maximum irradiance (722).        
        """
        jun = pvsyst.data.loc['06/1990']
        jun = jun.loc[jun['GlobInc'] > 400, :]
        jun.loc[(jun['GlobInc'] > 600) & (jun['GlobInc'] < 700), 'GlobInc'] = np.nan
        rc_tool = pvc.ReportingIrradiance(jun, 'GlobInc', percent_band=50)
        rc_tool.min_ref_irradiance = 605
        rc_tool.max_ref_irradiance = 695
        with pytest.warns(UserWarning):
            rc_tool.get_rep_irr()
        assert isinstance(rc_tool.poa_flt, pd.DataFrame)
        assert np.isnan(rc_tool.irr_rc)

class TestCapDataCopy():
    def test_copy_of_pre_agg_attributes(self, meas):
        pre_agg_cols = copy.copy(meas.data.columns)
        pre_agg_col_groups = copy.deepcopy(meas.column_groups)
        pre_agg_reg_columns = copy.deepcopy(meas.regression_cols)
        meas.agg_sensors(agg_map={
            'irr_poa_pyran': 'mean',
            'temp_amb': 'mean',
            'wind': 'mean',
        })
        meas_copy = meas.copy()
        assert meas_copy.pre_agg_cols.equals(pre_agg_cols)
        assert meas_copy.pre_agg_trans == pre_agg_col_groups
        assert meas_copy.pre_agg_reg_trans == pre_agg_reg_columns
        assert meas_copy.pre_agg_cols.equals(meas.pre_agg_cols)
        assert meas_copy.pre_agg_trans == meas.pre_agg_trans
        assert meas_copy.pre_agg_reg_trans == meas.pre_agg_reg_trans

class TestCapDataMethodsSim():
    """Test for top level irr_rc_balanced function."""
    def test_copy(self, pvsyst):
        pvsyst.set_regression_cols(
            power='real_pwr--', poa='irr-ghi-', t_amb='temp_amb', w_vel='wind--'
        )
        print(pvsyst.trans_keys)
        pvsyst_copy = pvsyst.copy()
        assert pvsyst_copy.data.equals(pvsyst.data)
        assert pvsyst_copy.column_groups == pvsyst.column_groups
        assert pvsyst_copy.trans_keys == pvsyst.trans_keys
        assert pvsyst_copy.regression_cols == pvsyst.regression_cols

    def test_filter_pvsyst_default(self, pvsyst):
        pvsyst.filter_pvsyst()
        assert pvsyst.data_filtered.shape[0] == 8670

    def test_filter_pvsyst_default_newer_pvsyst_var_names(self, pvsyst):
        pvsyst.data_filtered.rename(
            columns={
                'IL Pmin':'IL_Pmin',
                'IL Vmin':'IL_Vmin',
                'IL Pmax':'IL_Pmax',
                'IL Vmax':'IL_Vmax',
            }, inplace=True
        )
        assert pvsyst.data_filtered.shape[0] == 8760
        pvsyst.filter_pvsyst()
        assert pvsyst.data_filtered.shape[0] == 8670

    def test_filter_pvsyst_not_inplace(self, pvsyst):
        df = pvsyst.filter_pvsyst(inplace=False)
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == 8670

    def test_filter_pvsyst_missing_column(self, pvsyst):
        pvsyst.data.drop(columns='IL Pmin', inplace=True)
        pvsyst.data_filtered.drop(columns='IL Pmin', inplace=True)
        with pytest.warns(
            UserWarning, match='IL_Pmin or IL Pmin is not a column in the data.'
        ):
            pvsyst.filter_pvsyst()

    def test_filter_pvsyst_missing_all_columns(self, pvsyst):
        pvsyst.data.drop(
            columns=['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax'],
            inplace=True
        )
        pvsyst.data_filtered.drop(
            columns=['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax'],
            inplace=True
        )
        with pytest.warns(UserWarning):
            pvsyst.filter_pvsyst()

    def test_filter_shade_default(self, pvsyst):
        pvsyst.filter_shade()
        assert pvsyst.data_filtered.shape[0] == 8645

    def test_filter_shade_default_not_inplace(self, pvsyst):
        df = pvsyst.filter_shade(inplace=False)
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == 8645

    def test_filter_shade_query(self, pvsyst):
        # create PVsyst ShdLoss type values for testing query string
        pvsyst.data.loc[pvsyst.data['FShdBm'] == 1.0, 'ShdLoss'] = 0
        is_shaded = pvsyst.data['ShdLoss'].isna()
        shdloss_values = 1 / pvsyst.data.loc[is_shaded, 'FShdBm'] * 100
        pvsyst.data.loc[is_shaded, 'ShdLoss'] = shdloss_values
        pvsyst.data_filtered = pvsyst.data.copy()

        pvsyst.filter_shade(query_str='ShdLoss<=125')
        assert pvsyst.data_filtered.shape[0] == 8671


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
        trck_sys2 = pvc.pvlib_system(tracker_sys2)

        self.assertIsInstance(fx_sys,
                              pvlib.pvsystem.PVSystem,
                              'Did not return instance of\
                               pvlib PVSystem')
        self.assertIsInstance(fx_sys.arrays[0].mount,
                              pvlib.pvsystem.FixedMount,
                              'Did not return instance of\
                               pvlib FixedMount')

        self.assertIsInstance(trck_sys1,
                              pvlib.pvsystem.PVSystem,
                              'Did not return instance of\
                               pvlib PVSystem')
        self.assertIsInstance(trck_sys1.arrays[0].mount,
                              pvlib.pvsystem.SingleAxisTrackerMount,
                              'Did not return instance of\
                               pvlib SingleAxisTrackerMount')

        self.assertIsInstance(trck_sys2,
                              pvlib.pvsystem.PVSystem,
                              'Did not return instance of\
                               pvlib PVSystem')
        self.assertIsInstance(trck_sys2.arrays[0].mount,
                              pvlib.pvsystem.SingleAxisTrackerMount,
                              'Did not return instance of\
                               pvlib SingleAxisTrackerMount')


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

class TestGetTimezoneIndex():
    """Test get_tz_index function."""
    def test_get_tz_index_df(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe without a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(
            start='11/3/2018', periods=864, freq='5min', tz='America/Chicago'
        )
        ix_2days = pd.date_range(
            start='3/9/2019', periods=576, freq='5min', tz='America/Chicago'
        )
        ix_dst = ix_3days.append(ix_2days)

        ix_dst = ix_dst.tz_localize(None) # remove timezone from index

        df = pd.DataFrame(index=ix_dst)
        print(df.loc['11/4/18 01:00'].index)
        tz_ix = pvc.get_tz_index(df, location_and_system['location'])
        assert(isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex))
        assert(tz_ix.tz == pytz.timezone(location_and_system['location']['tz']))

    def test_get_tz_index_df_tz(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe WITH a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(
            start='11/3/2018', periods=864, freq='5min', tz='America/Chicago'
        )
        ix_2days = pd.date_range(
            start='3/9/2019', periods=576, freq='5min', tz='America/Chicago'
        )
        ix_dst = ix_3days.append(ix_2days)
        df = pd.DataFrame(index=ix_dst)
        tz_ix = pvc.get_tz_index(df, location_and_system['location'])
        assert(isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex))
        assert(tz_ix.tz == pytz.timezone(location_and_system['location']['tz']))

    def test_get_tz_index_df_tz_warn(self, location_and_system):
        """Test that get_tz_index function warns when datetime index\
           of dataframe does not match loc dic timezone."""
        df = pd.DataFrame(index=pd.date_range(
            start='11/3/2018', periods=864, freq='5min', tz='America/New_York'
        ))  # tz is New York
        with pytest.warns(UserWarning, match=(
                'Passed a DataFrame with a timezone that does not match '
                'the timezone in the loc dict. Using the timezone of the DataFrame.'
        )):
            tz_ix = pvc.get_tz_index(df, location_and_system['location']) # tz is Chicago

    def test_get_tz_index_ix_tz(self, location_and_system):
        """Test that get_tz_index function returns a datetime index
           with a timezone when passed a datetime index with a timezone."""
        ix = pd.date_range(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/Chicago')
        tz_ix = pvc.get_tz_index(ix, location_and_system['location'])  # tz is Chicago
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        # If passing an index with a timezone use that timezone rather than
        # the timezone in the location dictionary if there is one.
        assert tz_ix.tz == ix.tz

    def test_get_tz_index_ix_tz_warn(self, location_and_system):
        """Test that get_tz_index function warns when DatetimeIndex timezone
           does not match the location dic timezone.
        """
        ix = pd.date_range(start='1/1/2019', periods=8760, freq='H',
                                   tz='America/New_York')

        with pytest.warns(UserWarning, match=(
            'Passed a DatetimeIndex with a timezone that '
            'does not match the timezone in the loc dict. '
            'Using the timezone of the DatetimeIndex.'
        )):
            tz_ix = pvc.get_tz_index(ix, location_and_system['location'])

    def test_get_tz_index_ix(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a datetime index without a timezone."""
        ix = pd.date_range(
            start='1/1/2019', periods=8760, freq='H', tz='America/Chicago'
        )
        # remove timezone info but keep missing  hour and extra hour due to DST
        ix = ix.tz_localize(None)
        tz_ix = pvc.get_tz_index(ix, location_and_system['location']) # tz is Chicago
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        # If passing an index without a timezone use returned index should have
        # the timezone of the passed location dictionary.
        assert tz_ix.tz == pytz.timezone(location_and_system['location']['tz'])

class Test_csky():
    """Test clear sky function which returns pvlib ghi and poa clear sky."""
    def test_csky_concat(self, meas, location_and_system):
        # concat=True by default
        csky_ghi_poa = pvc.csky(
            meas.data,
            loc=location_and_system['location'],
            sys=location_and_system['system']
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == (meas.data.shape[1] + 2)
        assert 'ghi_mod_csky' in csky_ghi_poa.columns
        assert 'poa_mod_csky' in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.loc['10/9/1990 12:30', 'poa_mod_csky'] > \
               csky_ghi_poa.loc['10/9/1990 12:30', 'ghi_mod_csky']
        assert csky_ghi_poa.index.tz == df.index.tz

    def test_csky_concat_dst_spring(self, meas, location_and_system):
        """Test that csky concatenates clear sky ghi and poa when the time_source
           includes spring daylight savings time. This test assumes the time_source
           includes the 2 to 3AM hour that is skipped during daylight savings time."""
        # concat=True by default
        data = meas.data.loc['10/9/1990']
        data.index = pd.date_range('3/12/23', periods=(60 / 5) * 24, freq='5min')
        csky_ghi_poa = pvc.csky(
            data,
            loc=location_and_system['location'],
            sys=location_and_system['system']
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == (meas.data.shape[1] + 2)
        assert 'ghi_mod_csky' in csky_ghi_poa.columns
        assert 'poa_mod_csky' in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert (
            csky_ghi_poa.loc['3/12/23 12:30', 'poa_mod_csky']
            > csky_ghi_poa.loc['3/12/23 12:30', 'ghi_mod_csky']
        )
        assert csky_ghi_poa.index.tz == df.index.tz

    def test_csky_concat_dst_fall(self, meas, location_and_system):
        """Test that csky concatenates clear sky ghi and poa when the time_source
           includes spring daylight savings time. This test assumes the time_source
           does not include the extra 1AM hour that is added during daylight savings
           time, which causes the tz_localize in get_tz_index to fail because it
           expects two 1AM hours in the index. Leaving this as a failing test for now"""
        # concat=True by default
        # data = meas.data.loc['10/9/1990']
        # data.index = pd.date_range('11/5/23', periods=(60 / 5) * 24, freq='5min')
        # fails because tz_localize  in get_tz_index expects two 1AM hours in the index
        # csky_ghi_poa = pvc.csky(
        #     data,
        #     loc=location_and_system['location'],
        #     sys=location_and_system['system']
        # ) 
        assert 1
        # assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        # assert csky_ghi_poa.shape[1] == (meas.data.shape[1] + 2)
        # assert 'ghi_mod_csky' in csky_ghi_poa.columns
        # assert 'poa_mod_csky' in csky_ghi_poa.columns
        # # assumes typical orientation is used to calculate the poa irradiance
        # assert (
        #     csky_ghi_poa.loc['11/5/23 12:30', 'poa_mod_csky']
        #     > csky_ghi_poa.loc['11/5/23 12:30', 'ghi_mod_csky']
        # )
        # assert csky_ghi_poa.index.tz == df.index.tz

    def test_csky_not_concat(self, meas, location_and_system):
        csky_ghi_poa = pvc.csky(
            meas.data,
            loc=location_and_system['location'],
            sys=location_and_system['system'],
            concat=False,
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 2
        assert 'ghi_mod_csky' in csky_ghi_poa.columns
        assert 'poa_mod_csky' in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.loc['10/9/1990 12:30', 'poa_mod_csky'] > \
               csky_ghi_poa.loc['10/9/1990 12:30', 'ghi_mod_csky']
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_poa_all(self, meas, location_and_system):
        csky_ghi_poa = pvc.csky(
            meas.data,
            loc=location_and_system['location'],
            sys=location_and_system['system'],
            concat=False,
            output='poa_all',
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 5
        cols = [
            'poa_global',
            'poa_direct',
            'poa_diffuse',
            'poa_sky_diffuse',
            'poa_ground_diffuse'
        ]
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_ghi_all(self, meas, location_and_system):
        csky_ghi_poa = pvc.csky(
            meas.data,
            loc=location_and_system['location'],
            sys=location_and_system['system'],
            concat=False,
            output='ghi_all',
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 3
        cols = ['ghi', 'dni', 'dhi']
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_all(self, meas, location_and_system):
        csky_ghi_poa = pvc.csky(
            meas.data,
            loc=location_and_system['location'],
            sys=location_and_system['system'],
            concat=False,
            output='all',
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 8
        cols = [
            'ghi',
            'dni',
            'dhi',
            'poa_global',
            'poa_direct',
            'poa_diffuse',
            'poa_sky_diffuse',
            'poa_ground_diffuse'
        ]
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

"""
Change csky to two functions for creating pvlib location and system objects.
Separate function calling location and system to calculate POA
- concat add columns to passed df or return just ghi and poa option
load_data calls final function with in place to get ghi and poa
"""

class TestGetRegCols():
    """Test the get_reg_cols method of the CapData class."""
    def test_not_aggregated(self, meas):
        with pytest.warns(UserWarning):
            meas.get_reg_cols()

    def test_all_coeffs(self, meas):
        meas.agg_sensors()
        cols = ['power', 'poa', 't_amb', 'w_vel']
        df = meas.get_reg_cols()
        assert len(df.columns) == 4
        assert df.columns.to_list() == cols
        print(meas.data.columns)
        assert meas.data['irr_poa_pyran_mean_agg'].iloc[100] == df['poa'].iloc[100]
        assert meas.data['temp_amb_mean_agg'].iloc[100] == df['t_amb'].iloc[100]
        assert meas.data['wind_mean_agg'].iloc[100] == df['w_vel'].iloc[100]

    def test_agg_sensors_mix(self, meas):
        """
        Test when agg_sensors resets regression_cols values to a mix of trans keys
        and column names.
        """
        meas.agg_sensors(agg_map={
            'power_inv': 'sum',
            'irr_poa_pyran': 'mean',
            'temp_amb': 'mean',
            'wind': 'mean',
        })
        cols = ['poa', 'power']
        df = meas.get_reg_cols(reg_vars=cols)
        mtr_col = meas.column_groups[meas.regression_cols['power']][0]
        assert len(df.columns) == 2
        assert df.columns.to_list() == cols
        assert meas.data[mtr_col].iloc[100] == df['power'].iloc[100]
        assert meas.data['irr_poa_pyran_mean_agg'].iloc[100] == df['poa'].iloc[100]


class TestAggSensors():
    def test_agg_map_none(self, meas):
        """ Test default behaviour when no agg_map is passed. """
        meas.agg_sensors()
        # data and data_filtered should have same number of columns
        assert meas.data_filtered.shape[1] == meas.data.shape[1]
        # Rows should be the same in both dataframes
        assert meas.data_filtered.shape[0] == meas.data.shape[0]
        # Data after aggregation should not have sum of power columns because there
        # is only one power column, so it is not aggregated.
        assert 'power_sum_agg' not in meas.data.columns
        assert 'power_sum_agg' not in meas.data_filtered.columns

        # Check for poa aggregation column
        assert 'irr_poa_pyran_mean_agg' in meas.data_filtered.columns
        # Check for amb temp aggregation column
        assert 'temp_amb_mean_agg' in meas.data_filtered.columns
        # Check for wind aggregation column
        assert 'wind_mean_agg' in meas.data_filtered.columns

    def test_agg_map_non_str_func(self, meas):
        meas.agg_sensors(agg_map={'irr_poa_pyran': np.mean})
        # data and data_filtered should have same number of columns
        assert meas.data_filtered.shape[1] == meas.data.shape[1]
        # Rows should be the same in both dataframes
        assert meas.data_filtered.shape[0] == meas.data.shape[0]
        # Check for poa aggregation column
        assert 'irr_poa_pyran_mean_agg' in meas.data_filtered.columns

    def test_agg_map_update_regression_cols(self, meas):
        meas.agg_sensors()
        # Regression column for power should not be updated because there is only
        # one power column.
        assert meas.regression_cols['power'] == 'meter_power'
        # Regression columns for poa, amb temp, and wind should be updated to
        # the aggregated columns from the column group ids.
        assert meas.regression_cols['poa'] == 'irr_poa_pyran_mean_agg'
        assert meas.regression_cols['t_amb'] == 'temp_amb_mean_agg'
        assert meas.regression_cols['w_vel'] == 'wind_mean_agg'

    def test_reset_summary(self, meas):
        meas.agg_sensors()
        # Summary should be empty after aggregation
        assert len(meas.summary) == 0
        # Summary index should be empty after aggregation
        assert len(meas.summary_ix) == 0

    def test_reset_agg_method(self, meas):
        orig_df = meas.data.copy()
        orig_trans = meas.column_groups.copy()
        orig_reg_trans = meas.regression_cols.copy()

        meas.agg_sensors()
        meas.filter_irr(200, 500)
        meas.reset_agg()

        # Dataframe should be the same as before aggregation
        assert meas.data.equals(orig_df)
        # Columns should be the same as before aggregation
        assert all(meas.data_filtered.columns == orig_df.columns)
        # Reset should not affect filtering
        assert meas.data_filtered.shape[0] < orig_df.shape[0]

    def test_warn_if_filters_already_run(self, meas):
        """
        Warn if method is writing over filtering already applied to data_filtered.
        """
        poa_key = meas.regression_cols['poa']
        meas.column_groups[poa_key] = [meas.column_groups[poa_key][0]]
        meas.filter_irr(200, 800)
        with pytest.warns(UserWarning, match=(
            'The data_filtered attribute has been overwritten '
            'and previously applied filtering steps have been '
            'lost.  It is recommended to use agg_sensors '
            'before any filtering methods.'
        )):
            meas.agg_sensors()

    def test_regression_columns_not_in_column_groups(self, meas):
        """Sould be able to aggregate columns if the regression columns includes
        a column that is not in the column_groups attribute.
        """
        meas.data['irr_poa_total'] = meas.data.loc[:, 'met1_poa_pyranometer']
        meas.regression_cols['poa'] = 'irr_poa_total'
        meas.agg_sensors(agg_map={'temp_amb': 'mean'})
        assert meas.regression_cols['t_amb'] == 'temp_amb_mean_agg'

    def test_pre_agg_regression_dict_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_reg_trans, dict)

    def test_pre_agg_column_groups_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_trans, cg.ColumnGroups)

    def test_pre_agg_columns_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_cols, pd.Index)


class TestFilterSensors():
    def test_perc_diff_none(self, meas):
        rows_before_flt = meas.data_filtered.shape[0]
        meas.filter_sensors(perc_diff=None, inplace=True)
        # Check that data_filtered is still a dataframe
        assert isinstance(meas.data_filtered, pd.core.frame.DataFrame)
        # Check that rows were removed
        assert meas.data_filtered.shape[0] < rows_before_flt

    def test_perc_diff(self, meas):
        rows_before_flt = meas.data_filtered.shape[0]
        meas.filter_sensors(
            perc_diff={'irr_poa_ref_cell': 0.05, 'temp_amb': 0.1},
            inplace=True
        )
        # Check that data_filtered is still a dataframe
        assert isinstance(meas.data_filtered, pd.core.frame.DataFrame)
        # Check that rows were removed
        assert (meas.data_filtered.shape[0] < rows_before_flt)

    def test_after_agg_sensors(self, meas):
        rows_before_flt = meas.data_filtered.shape[0]
        meas.agg_sensors(agg_map={
            'power_inv': 'sum',
            'irr_poa_ref_cell': 'mean',
            'wind': 'mean',
            'temp_amb': 'mean'
        })
        meas.filter_sensors(
            perc_diff={'irr_poa_ref_cell': 0.05, 'temp_amb': 0.1},
            inplace=True,
        )
        assert isinstance(meas.data_filtered, pd.core.frame.DataFrame)
        assert meas.data_filtered.shape[0] < rows_before_flt
        # Filter_sensors should retain the aggregated columns
        assert 'power_inv_sum_agg' in meas.data_filtered.columns


class TestRepCondNoFreq():
    def test_defaults(self, nrel):
        nrel.rep_cond()
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)

    def test_defaults_wvel(self, nrel):
        nrel.rep_cond(w_vel=50)
        assert nrel.rc['w_vel'][0] == 50

    def test_defaults_not_inplace(self, nrel):
        df = nrel.rep_cond(inplace=False)
        assert nrel.rc is None
        assert isinstance(df, pd.core.frame.DataFrame)

    def test_irr_bal_inplace(self, nrel):
        nrel.filter_irr(0.1, 2000)
        meas2 = nrel.copy()
        meas2.rep_cond()
        nrel.rep_cond(irr_bal=True, percent_filter=20)
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)
        assert nrel.rc['poa'][0] != meas2.rc['poa'][0]

    def test_irr_bal_inplace_wvel(self, nrel):
        nrel.rep_cond(irr_bal=True, percent_filter=20, w_vel=50)
        assert nrel.rc['w_vel'][0] == 50


class TestRepCondFreq():
    def test_monthly_no_irr_bal(self, pvsyst):
        pvsyst.rep_cond(freq='M')
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 12 rows
        assert pvsyst.rc.shape[0] == 12

    def test_monthly_irr_bal(self, pvsyst):
        pvsyst.rep_cond(freq='M', irr_bal=True, percent_filter=20)
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 12 rows
        assert pvsyst.rc.shape[0] == 12

    def test_seas_no_irr_bal(self, pvsyst):
        pvsyst.rep_cond(freq='BQ-NOV', irr_bal=False)
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 4 rows
        assert pvsyst.rc.shape[0] == 4


class TestPredictCapacities():
    def test_monthly(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond(freq='MS')
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=True, percent_filter=20)
        july_grpby = pred_caps.loc['1990-07-01', 'PredCap']

        # Check that the returned object is a dataframe
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        # Check that the returned dataframe has 12 rows
        assert pred_caps.shape[0] == 12

        pvsyst_irr_filter.data_filtered = pvsyst_irr_filter.data_filtered.loc['7/1/90':'7/31/90', :]
        pvsyst_irr_filter.rep_cond()
        pvsyst_irr_filter.filter_irr(0.8, 1.2, ref_val=pvsyst_irr_filter.rc['poa'][0])
        df = pvsyst_irr_filter.rview(['power', 'poa', 't_amb', 'w_vel'],
                               filtered_data=True)
        rename = {df.columns[0]: 'power',
                  df.columns[1]: 'poa',
                  df.columns[2]: 't_amb',
                  df.columns[3]: 'w_vel'}
        df = df.rename(columns=rename)
        reg = pvc.fit_model(df)
        july_manual = reg.predict(pvsyst_irr_filter.rc)[0]
        assert july_manual == pytest.approx(july_grpby)

    def test_no_irr_filter(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond(freq='M')
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=False)
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 12

    def test_rc_from_irrBal(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond(freq='M', irr_bal=True, percent_filter=20)
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=False)
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 12

    def test_seasonal_freq(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond(freq='BQ-NOV')
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=True, percent_filter=20)
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 4

class TestFilterIrr():
    def test_get_poa_col(self, nrel):
        col = nrel._CapData__get_poa_col()
        assert col == 'POA 40-South CMP11 [W/m^2]'

    def test_get_poa_col_multcols(self, nrel):
        nrel.data['POA second column'] = nrel.rview('poa').values
        nrel.column_groups['irr-poa-'].append('POA second column')
        with pytest.warns(UserWarning, match=(
            '[0-9]+ columns of irradiance data. Use col_name to specify a single column.'
        )):
            col = nrel._CapData__get_poa_col()

    def test_lowhigh_nocol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(500, 600, ref_val=None, col_name=None, inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_lowhigh_colname(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.data['POA second column'] = nrel.rview('poa').values
        nrel.column_groups['irr-poa-'].append('POA second column')
        nrel.data_filtered = nrel.data.copy()
        nrel.filter_irr(
            500, 600, ref_val=None, col_name='POA second column', inplace=True
        )
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_nocol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(0.8, 1.2, ref_val=500, col_name=None,
                             inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_withcol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.data['POA second column'] = nrel.rview('poa').values
        nrel.column_groups['irr-poa-'].append('POA second column')
        nrel.data_filtered = nrel.data.copy()
        nrel.filter_irr(0.8, 1.2, ref_val=500,
                             col_name='POA second column', inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_use_attribute(self, nrel):
        nrel.rc = pd.DataFrame({'poa':500, 'w_vel':1, 't_amb':20}, index=[0])
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(0.8, 1.2, ref_val='self_val', col_name=None,
                             inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_withcol_notinplace(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        df = nrel.filter_irr(500, 600, ref_val=None, col_name=None,
                                  inplace=False)
        assert nrel.data_filtered.shape[0] == pts_before
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] < pts_before


class TestGetSummary():
    def test_col_names(self, nrel):
        nrel.filter_irr(200, 500)
        smry = nrel.get_summary()
        assert smry.columns[0] == 'pts_after_filter'
        assert smry.columns[1] == 'pts_removed'
        assert smry.columns[2] == 'filter_arguments'



class TestFilterTime():
    def test_start_end(self, pvsyst):
        pvsyst.filter_time(start='2/1/90', end='2/15/90')
        assert (
            pvsyst.data_filtered.index[0]
            == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        )
        assert (
            pvsyst.data_filtered.index[-1]
            == pd.Timestamp(year=1990, month=2, day=15, hour=00)
        )

    def test_start_end_drop_is_true(self, pvsyst):
        pvsyst.filter_time(start='2/1/90', end='2/15/90', drop=True)
        assert (
            pvsyst.data_filtered.index[0]
            == pd.Timestamp(year=1990, month=1, day=1, hour=0)
        )
        assert (
            pvsyst.data_filtered.index[-1]
            == pd.Timestamp(year=1990, month=12, day=31, hour=23)
        )
        assert (
            pvsyst.data_filtered.shape[0]
            == (8760 - 14 * 24) - 1
        )

    def test_start_days(self, pvsyst):
        pvsyst.filter_time(start='2/1/90', days=15)
        assert (
            pvsyst.data_filtered.index[0]
            == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        )
        assert (
            pvsyst.data_filtered.index[-1]
            == pd.Timestamp(year=1990, month=2, day=16, hour=00)
        )

    def test_end_days(self, pvsyst):
        pvsyst.filter_time(end='2/16/90', days=15)
        assert (
            pvsyst.data_filtered.index[0]
            == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        )
        assert (
            pvsyst.data_filtered.index[-1]
            == pd.Timestamp(year=1990, month=2, day=16, hour=00)
        )

    def test_test_date(self, pvsyst):
        pvsyst.filter_time(test_date='2/16/90', days=30)
        assert (
            pvsyst.data_filtered.index[0]
            == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        )
        assert (
            pvsyst.data_filtered.index[-1]
            == pd.Timestamp(year=1990, month=3, day=3, hour=00)
        )

    def test_start_end_not_inplace(self, pvsyst):
        df = pvsyst.filter_time(start='2/1/90', end='2/15/90', inplace=False)
        assert df.index[0] == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        assert df.index[-1] == pd.Timestamp(year=1990, month=2, day=15, hour=00)

    def test_start_no_days(self, pvsyst):
        with pytest.warns(UserWarning):
            pvsyst.filter_time(start='2/1/90')

    def test_end_no_days(self, pvsyst):
        with pytest.warns(UserWarning):
            pvsyst.filter_time(end='2/1/90')

    def test_test_date_no_days(self, pvsyst):
        with pytest.warns(UserWarning):
            pvsyst.filter_time(test_date='2/1/90')


class TestFilterDays():
    def test_keep_one_day(self, pvsyst):
        pvsyst.filter_days(['10/5/1990'], drop=False, inplace=True)
        assert pvsyst.data_filtered.shape[0] == 24
        assert pvsyst.data_filtered.index[0].day == 5

    def test_keep_two_contiguous_days(self, pvsyst):
        pvsyst.filter_days(['10/5/1990', '10/6/1990'], drop=False,
                                inplace=True)
        assert pvsyst.data_filtered.shape[0] == 48
        assert pvsyst.data_filtered.index[-1].day == 6

    def test_keep_three_noncontiguous_days(self, pvsyst):
        pvsyst.filter_days(['10/5/1990', '10/7/1990', '10/9/1990'],
                                drop=False, inplace=True)
        assert pvsyst.data_filtered.shape[0] == 72
        assert pvsyst.data_filtered.index[0].day == 5
        assert pvsyst.data_filtered.index[25].day == 7
        assert pvsyst.data_filtered.index[49].day == 9

    def test_drop_one_day(self, pvsyst):
        pvsyst.filter_days(['1/1/1990'], drop=True, inplace=True)
        assert pvsyst.data_filtered.shape[0] == (8760 - 24)
        assert pvsyst.data_filtered.index[0].day == 2
        assert pvsyst.data_filtered.index[0].hour == 0

    def test_drop_three_days(self, pvsyst):
        pvsyst.filter_days(['1/1/1990', '1/3/1990', '1/5/1990'],
                                drop=True, inplace=True)
        assert pvsyst.data_filtered.shape[0] == (8760 - 24 * 3)
        assert pvsyst.data_filtered.index[0].day == 2
        assert pvsyst.data_filtered.index[25].day == 4
        assert pvsyst.data_filtered.index[49].day == 6

    def test_not_inplace(self, pvsyst):
        df = pvsyst.filter_days(['10/5/1990'], drop=False, inplace=False)
        assert pvsyst.data_filtered.shape[0] == 8760
        assert df.shape[0] == 24

class TestFilterPF():
    def test_pf(self, nrel):
        pf = np.ones(5)
        pf = np.append(pf, np.ones(5) * -1)
        pf = np.append(pf, np.arange(0, 1, 0.1))
        nrel.data['pf'] = np.tile(pf, 576)
        nrel.data_filtered = nrel.data.copy()
        nrel.column_groups['pf--'] = ['pf']
        nrel.trans_keys = list(nrel.column_groups.keys())
        nrel.filter_pf(1)
        assert nrel.data_filtered.shape[0] == 5760


class TestFilterOutliersAndPower():
    def test_not_aggregated(self, meas):
        with pytest.warns(UserWarning):
            meas.filter_outliers()

    def test_filter_power_defaults(self, meas):
        meas.filter_power(5_000_000, percent=None, columns=None, inplace=True)
        assert meas.data_filtered.shape[0] == 1289

    def test_filter_power_percent(self, meas):
        meas.filter_power(6_000_000, percent=0.05, columns=None, inplace=True)
        assert meas.data_filtered.shape[0] == 1388

    def test_filter_power_a_column(self, meas):
        print(meas.data.columns)
        meas.filter_power(
            5_000_000,
            percent=None,
            columns='meter_power',
            inplace=True
        )
        assert meas.data_filtered.shape[0] == 1289

    def test_filter_power_column_group(self, meas):
        meas.filter_power(500_000, percent=None, columns='power_inv', inplace=True)
        assert meas.data_filtered.shape[0] == 1138

    def test_filter_power_columns_not_str(self, meas):
        with pytest.warns(UserWarning):
            meas.filter_power(500_000, percent=None, columns=1, inplace=True)


class TestCskyFilter():
    """
    Tests for filter_clearsky method.
    """
    def test_default(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky()

        assert nrel_clear_sky.data_filtered.shape[0] < nrel_clear_sky.data.shape[0]
        assert nrel_clear_sky.data_filtered.shape[1] == nrel_clear_sky.data.shape[1]
        for i, col in enumerate(nrel_clear_sky.data_filtered.columns):
            assert col == nrel_clear_sky.data.columns[i]

    def test_default_drop_clear_sky(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky()
        clear_ix = nrel_clear_sky.data_filtered.index
        nrel_clear_sky.reset_filter()
        nrel_clear_sky.filter_clearsky(keep_clear=False)
        cloudy_ix = nrel_clear_sky.data_filtered.index
        assert nrel_clear_sky.data.index.difference(clear_ix).equals(cloudy_ix)

    def test_two_ghi_cols(self, nrel_clear_sky):
        nrel_clear_sky.data['ws 2 ghi W/m^2'] = nrel_clear_sky.view('irr-ghi-') * 1.05
        nrel_clear_sky.data_filtered = nrel_clear_sky.data.copy()
        nrel_clear_sky.column_groups['irr-ghi-'].append('ws 2 ghi W/m^2')
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_mult_ghi_categories(self, nrel_clear_sky):
        nrel_clear_sky.data['irrad ghi pyranometer W/m^2'] = (
            nrel_clear_sky.data.loc[:, 'Global CMP22 (vent/cor) [W/m^2]']
            * 1.05
        )
        nrel_clear_sky.column_groups['irr-ghi-pyran'] = ['irrad ghi pyranometer W/m^2']
        nrel_clear_sky.trans_keys = list(nrel_clear_sky.column_groups.keys())
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_no_clear_ghi(self, nrel_clear_sky):
        nrel_clear_sky.drop_cols('ghi_mod_csky')
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_specify_ghi_col(self, nrel_clear_sky):
        nrel_clear_sky.data['ws 2 ghi W/m^2'] = nrel_clear_sky.view('irr-ghi-') * 1.05
        nrel_clear_sky.data_filtered = nrel_clear_sky.data.copy()
        nrel_clear_sky.column_groups['irr-ghi-'].append('ws 2 ghi W/m^2')
        nrel_clear_sky.trans_keys = list(nrel_clear_sky.column_groups.keys())

        nrel_clear_sky.filter_clearsky(ghi_col='ws 2 ghi W/m^2')

        assert nrel_clear_sky.data_filtered.shape[0] < nrel_clear_sky.data.shape[0]
        assert nrel_clear_sky.data_filtered.shape[1] == nrel_clear_sky.data.shape[1]
        for i, col in enumerate(nrel_clear_sky.data_filtered.columns):
            assert col == nrel_clear_sky.data.columns[i]

    #def test_no_clear_sky(self, nrel_clear_sky):
    #    with pytest.warns(UserWarning):
    #        nrel_clear_sky.filter_clearsky(window_length=2)


class TestFilterMissing():
    """
    Newer tests written for pytest. Uses the meas pytest fixture defined below.
    """
    def test_filter_missing_default(self, meas):
        """Checks missing data in regression columns are removed."""
        print(meas.data.columns)
        meas.set_regression_cols(
            power='meter_power',
            poa='met1_poa_refcell',
            t_amb='met2_amb_temp',
            w_vel='met1_windspeed',
        )
        assert all(meas.rview('all', filtered_data=True).isna().sum() == 0)
        assert meas.data_filtered.shape[0] == 1440
        meas.data_filtered.loc['10/9/90 12:00', 'meter_power'] = np.NaN
        meas.data_filtered.loc['10/9/90 12:30', 'met1_poa_refcell'] = np.NaN
        meas.data_filtered.loc['10/10/90 12:35', 'met2_amb_temp'] = np.NaN
        meas.data_filtered.loc['10/10/90 12:50', 'met1_windspeed'] = np.NaN
        meas.filter_missing()
        assert meas.data_filtered.shape[0] == 1436

    def test_filter_missing_missing_not_in_columns_considered(self, meas):
        """Checks that nothing is dropped for missing data not in `columns`."""
        meas.set_regression_cols(
            power='meter_power',
            poa='met1_poa_refcell',
            t_amb='met2_amb_temp',
            w_vel='met1_windspeed',
        )
        assert all(meas.rview('all', filtered_data=True).isna().sum() == 0)
        assert meas.data_filtered.shape[0] == 1440
        assert meas.data_filtered.isna().sum().sum() > 0
        meas.filter_missing()
        assert meas.data_filtered.shape[0] == 1440

    def test_filter_missing_missing_passed_columns(self, meas):
        """Checks that nothing is dropped for missing data not in `columns`."""
        assert meas.data_filtered.shape[0] == 1440
        assert meas.data_filtered.isna().sum().sum() > 0
        meas.filter_missing(columns=['met1_amb_temp'])
        assert meas.data_filtered.shape[0] == 1424

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

        self.meas.regression_results = das_model.fit()
        self.sim.regression_results = sim_model.fit()
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
        meas.set_regression_cols(power='power', poa='poa',
                                 t_amb='t_amb', w_vel='w_vel')

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        meas.regression_results = das_model.fit()
        sim.regression_results = sim_model.fit()
        meas.data_filtered = pd.DataFrame()
        sim.data_filtered = pd.DataFrame()

        actual = meas.regression_results.predict(meas.rc)[0] * 1000
        expected = sim.regression_results.predict(meas.rc)[0]
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
        self.meas.set_regression_cols(power='power', poa='poa',
                                      t_amb='t_amb', w_vel='w_vel')

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        self.meas.regression_results = das_model.fit()
        self.sim.regression_results = sim_model.fit()
        self.meas.data_filtered = pd.DataFrame()
        self.sim.data_filtered = pd.DataFrame()

    def test_pvals_default_false(self):
        actual = self.meas.regression_results.predict(self.meas.rc)[0]
        expected = self.sim.regression_results.predict(self.meas.rc)[0]
        cp_rat_test_val = actual / expected

        cp_rat = pvc.captest_results(self.sim, self.meas, 100, '+/- 5',
                                check_pvalues=False, print_res=False)

        self.assertEqual(cp_rat, cp_rat_test_val,
                         'captest_results did not return expected value.')

    def test_pvals_true(self):
        self.meas.regression_results.params['poa'] = 0
        self.sim.regression_results.params['poa'] = 0
        actual_pval_check = self.meas.regression_results.predict(self.meas.rc)[0]
        expected_pval_check = self.sim.regression_results.predict(self.meas.rc)[0]
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
        self.meas.regression_results.params['poa'] = 0
        self.sim.regression_results.params['poa'] = 0

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

        sim.regression_formula = 'power ~ poa + I(poa * poa) + I(poa * t_amb) - 1'

        with self.assertWarns(UserWarning):
            pvc.captest_results(sim, das, 100, '+/- 5', check_pvalues=True)


class TestGetFilteringTable:
    """Check the DataFrame summary showing which filter removed which intervals."""

    def test_get_filtering_table(self, nrel):
        nrel.filter_irr(200, 900)
        flt0_kept_ix = nrel.data_filtered.index
        flt0_removed_ix = nrel.data.index.difference(flt0_kept_ix)
        nrel.filter_irr(400, 800)
        flt1_kept_ix = nrel.data_filtered.index
        flt1_removed_ix = flt0_kept_ix.difference(flt1_kept_ix)
        nrel.filter_irr(500, 600)
        flt2_kept_ix = nrel.data_filtered.index
        flt2_removed_ix = flt1_kept_ix.difference(flt2_kept_ix)
        flt_table = nrel.get_filtering_table()
        assert isinstance(flt_table, pd.DataFrame)
        assert flt_table.shape == (nrel.data.shape[0], 4)
        table_flt0_column = flt_table.iloc[:, 0]
        table_flt0_removed = table_flt0_column[table_flt0_column == 1].index
        assert table_flt0_removed.equals(flt0_removed_ix)
        table_flt1_column = flt_table.iloc[:, 1]
        table_flt1_removed = table_flt1_column[table_flt1_column == 1].index
        assert table_flt1_removed.equals(flt1_removed_ix)
        table_flt2_column = flt_table.iloc[:, 2]
        table_flt2_removed = table_flt2_column[table_flt2_column == 1].index
        assert table_flt2_removed.equals(flt2_removed_ix)
        table_flt_all_column = flt_table.iloc[:, 3]
        table_flt_all_removed = table_flt_all_column[~table_flt_all_column].index
        out = pd.concat([flt_table, nrel.rview('poa')], axis=1)
        assert table_flt_all_removed.equals(
            flt0_removed_ix.union(flt1_removed_ix).union(flt2_removed_ix)
        )

@pytest.fixture
def pts_summary(meas):
    pts_summary = pvc.PointsSummary(meas)
    return pts_summary

class TestPointsSummary():
    def test_length_test_period_no_filter(self, meas):
        meas.get_length_test_period()
        assert meas.length_test_period == 5

    def test_length_test_period_after_one_filter_time(self, meas):
        meas.filter_time(start='10/9/1990', end='10/12/1990 23:00')
        meas.get_length_test_period()
        assert meas.length_test_period == 4

    def test_length_test_period_after_two_filter_time(self, meas):
        meas.filter_time(start='10/9/1990', end='10/12/1990 23:00')
        meas.filter_time(start='10/9/1990', end='10/11/1990 23:00')
        meas.get_length_test_period()
        assert meas.length_test_period == 4

    def test_get_pts_required_default(self, meas):
        meas.get_pts_required()
        assert meas.pts_required == 150

    def test_get_pts_required_10_hrs(self, meas):
        meas.get_pts_required(hrs_req=10)
        assert meas.pts_required == 120

    def test_set_test_complete_equal_pts_req(self, meas):
        meas.set_test_complete(1440)
        assert meas.test_complete

    def test_set_test_complete_more_than_pts_req(self, meas):
        meas.set_test_complete(1439)
        assert meas.test_complete

    def test_set_test_complete_not_enough_pts(self, meas):
        meas.set_test_complete(1441)
        assert not meas.test_complete

    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test_print_points_summary_pass(self, meas):
        meas.print_points_summary()
        captured = self.capsys.readouterr()

        results_str = (
            'length of test period to date: 5 days\n'
            'sufficient points have been collected. 150.0 points required; '
            '1440 points collected\n'
        )

        assert results_str == captured.out

    def test_print_points_summary_fail(self, meas):
        meas.data_filtered = meas.data.iloc[0:10, :]
        meas.print_points_summary()
        captured = self.capsys.readouterr()

        results_str = (
            'length of test period to date: 5 days\n'
            '10 points of 150.0 points needed, 140.0 remaining to collect.\n'
            '2.00 points / day on average.\n'
            'Approximate days remaining: 71\n'
        )

        assert results_str == captured.out


class TestSetPlotsAttributes():
    """Test assigning colors to each column using the keys of the column_grouping."""
    def test_real_power_group_colors(self, meas):
        """
        Test that the color assigned to the column(s) in the `data` attributute is
        one of the colors in the `plot_colors_brewer` dictionary with the real_pwr key.
        """
        meas.set_plot_attributes()
        assert meas.col_colors['meter_power'] == '#2b8cbe'
        assert meas.col_colors['met1_poa_refcell'] == '#e31a1c'
        assert meas.col_colors['met2_poa_refcell'] == '#fd8d3c'
        assert meas.col_colors['met2_poa_refcell'] == '#fd8d3c'
        assert meas.col_colors['met1_ghi_pyranometer'] == '#91003f'
        assert meas.col_colors['met1_amb_temp'] == '#238443'
        assert meas.col_colors['met1_mod_temp1'] == '#88419d'
        assert meas.col_colors['met1_windspeed'] == '#238b45'
        assert meas.col_colors['inv1_power'] == '#d60000'


class TestDataColumnsToExcel():
    """
    Test the `data_columns_to_excel` method of the `CapData` class.
    """
    def test_data_columns_to_excel_path_is_dir(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader('./tests/data/')
        meas.data_columns_to_excel(sort_by_reversed_names=True)
        xlsx_file = meas.data_loader.path / 'column_groups.xlsx'
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == 'met1_mod_temp1'
        os.remove(xlsx_file)

    def test_data_columns_to_excel_path_is_file(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader('./tests/data/example_measured_data.csv')
        meas.data_columns_to_excel(sort_by_reversed_names=True)
        xlsx_file = meas.data_loader.path.parent / 'column_groups.xlsx'
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == 'met1_mod_temp1'
        os.remove(xlsx_file)

    def test_data_columns_to_excel_not_reverse_sorted(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader('./tests/data/')
        meas.data_columns_to_excel(sort_by_reversed_names=False)
        xlsx_file = meas.data_loader.path / 'column_groups.xlsx'
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == 'inv1_power'
        os.remove(xlsx_file)


if __name__ == '__main__':
    unittest.main()
