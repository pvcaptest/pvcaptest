import json
import os
import copy
import collections
import unittest
import pytest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import holoviews as hv
from patsy import dmatrix

import pvlib

import panel as pn

from captest import capdata as pvc
from captest import captest as captest_module
from captest import clearsky
from captest import columngroups as cg
from captest import filters
from captest import io
from captest import (
    CapTest,
    load_pvsyst,
    calcparams,
)

data = np.arange(0, 1300, 54.167)
index = pd.date_range(start="1/1/2017", freq="h", periods=24)
df = pd.DataFrame(data=data, index=index, columns=["poa"])

# capdata = pvc.CapData('capdata')
# capdata.df = df

"""
Run tests using pytest use the following from project root.
To run a class of tests
pytest tests/test_CapData.py::TestCapDataEmpty

To run a specific test:
pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty

To create a test coverage report (html output) with pytest:
pytest --cov-report html --cov=src/captest tests/

pytest fixtures meas, location_and_system, nrel, pvsyst, pvsyst_irr_filter, and
nrel_clear_sky are in the ./tests/conftest.py file.
"""


class TestUpdateSummary:
    """Test the utility functions used for argument formatting."""

    def test_round_kwarg_floats(self):
        """Tests round kwarg_floats."""
        kwarg_dict = {"ref_val": 763.4536140499999, "t1": 2, "inplace": True}
        rounded_kwarg_dict_3 = {"ref_val": 763.454, "t1": 2, "inplace": True}
        assert pvc.round_kwarg_floats(kwarg_dict) == rounded_kwarg_dict_3
        rounded_kwarg_dict_4 = {"ref_val": 763.4536, "t1": 2, "inplace": True}
        assert pvc.round_kwarg_floats(kwarg_dict, 4) == rounded_kwarg_dict_4

    def test_tstamp_kwarg_to_strings(self):
        """Tests coversion of kwarg values from timestamp to strings."""
        start_datetime = pd.to_datetime("10/10/1990 00:00")
        kwarg_dict = {"start": start_datetime, "t1": 2}
        kwarg_dict_str_dates = {"start": "1990-10-10 00:00", "t1": 2}
        assert pvc.tstamp_kwarg_to_strings(kwarg_dict) == kwarg_dict_str_dates


class TestTopLevelFuncs(unittest.TestCase):
    def test_perc_wrap(self):
        """Test percent wrap function."""
        rng = np.arange(1, 100, 1)
        df = pd.DataFrame({"vals": rng})
        df_cpy = df.copy()
        bool_array = []
        for val in rng:
            np_perc = np.percentile(rng, val, method="nearest")
            wrap_perc = df.agg(captest_module.perc_wrap(val)).values[0]
            bool_array.append(np_perc == wrap_perc)
        self.assertTrue(
            all(bool_array), "np.percentile wrapper gives different value than np perc"
        )
        self.assertTrue(all(df == df_cpy), "perc_wrap function modified input df")

    def test_filter_irr(self):
        rng = np.arange(0, 1000)
        df = pd.DataFrame(
            np.array([rng, rng + 100, rng + 200]).T,
            columns=["weather_station irr poa W/m^2", "col_1", "col_2"],
        )
        df_flt = pvc.filter_irr(df, "weather_station irr poa W/m^2", 50, 100)

        self.assertEqual(
            df_flt.shape[0], 51, "Incorrect number of rows returned from filter."
        )
        self.assertEqual(
            df_flt.shape[1], 3, "Incorrect number of columns returned from filter."
        )
        self.assertEqual(
            df_flt.columns[0],
            "weather_station irr poa W/m^2",
            "Filter column name inadverdently modified by method.",
        )
        self.assertEqual(
            df_flt.iloc[0, 0],
            50,
            "Minimum value in returned data in filter column is"
            "not equal to low argument.",
        )
        self.assertEqual(
            df_flt.iloc[-1, 0],
            100,
            "Maximum value in returned data in filter column is"
            "not equal to high argument.",
        )

    def test_fit_model(self):
        """
        Test fit model func which wraps statsmodels ols.fit for dataframe.
        """
        rng = np.random.RandomState(1)
        x = 50 * abs(rng.rand(50))
        y = 2 * x - 5 + 5 * rng.randn(50)
        df = pd.DataFrame({"x": x, "y": y})
        fml = "y ~ x - 1"
        passed_ind_vars = fml.split("~")[1].split()[::2]
        try:
            passed_ind_vars.remove("1")
        except ValueError:
            pass

        reg = pvc.fit_model(df, fml=fml)

        for var in passed_ind_vars:
            self.assertIn(
                var,
                reg.params.index,
                "{} ind variable in formula argument not in modelparameters".format(
                    var
                ),
            )

    def test_predict(self):
        x = np.arange(0, 50)
        y1 = x
        y2 = x * 2
        y3 = x * 10

        dfs = [
            pd.DataFrame({"x": x, "y": y1}),
            pd.DataFrame({"x": x, "y": y2}),
            pd.DataFrame({"x": x, "y": y3}),
        ]

        reg_lst = []
        for df in dfs:
            reg_lst.append(pvc.fit_model(df, fml="y ~ x"))
        reg_ser = pd.Series(reg_lst)

        for regs in [reg_lst, reg_ser]:
            preds = pvc.predict(regs, pd.DataFrame({"x": [10, 10, 10]}))
            self.assertAlmostEqual(preds.iloc[0], 10, 7, "Pred for x = y wrong.")
            self.assertAlmostEqual(preds.iloc[1], 20, 7, "Pred for x = y * 2 wrong.")
            self.assertAlmostEqual(preds.iloc[2], 100, 7, "Pred for x = y * 10 wrong.")
            self.assertEqual(
                3,
                preds.shape[0],
                "Each of the three inputregressions should have aprediction",
            )

    def test_pred_summary(self):
        """Test aggregation of reporting conditions and predicted results."""
        """
        grpby -> df of regressions
        regs -> series of predicted values
        df of reg parameters
        """
        pvsyst = load_pvsyst(path="./tests/data/pvsyst_example_HourlyRes_2.CSV")

        df_regs = pvsyst.data.loc[:, ["E_Grid", "GlobInc", "T_Amb", "WindVel"]]
        df_regs_day = df_regs.query("GlobInc > 0")
        grps = df_regs_day.groupby(pd.Grouper(freq="ME", label="right"))

        ones = np.ones(12)
        irr_rc = ones * 500
        temp_rc = ones * 20
        w_vel = ones
        rcs = pd.DataFrame({"GlobInc": irr_rc, "T_Amb": temp_rc, "WindVel": w_vel})

        results = pvc.pred_summary(
            grps,
            rcs,
            0.05,
            fml="E_Grid ~ GlobInc +"
            "I(GlobInc * GlobInc) +"
            "I(GlobInc * T_Amb) +"
            "I(GlobInc * WindVel) - 1",
        )

        self.assertEqual(results.shape[0], 12, "Not all months in results.")
        self.assertEqual(results.shape[1], 10, "Not all cols in results.")

        self.assertIsInstance(
            results.index,
            pd.core.indexes.datetimes.DatetimeIndex,
            "Index is not pandas DatetimeIndex",
        )

        col_length = len(results.columns.values)
        col_set_length = len(set(results.columns.values))
        self.assertEqual(
            col_set_length,
            col_length,
            "There is a duplicate column name in the results df.",
        )

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
                results.loc[mnth, "guaranteedCap"],
                results.loc[mnth, "PredCap"],
                "Gauranteed cap is greater than predicted in month {}".format(mnth),
            )
            self.assertGreater(
                results.loc[mnth, "guaranteedCap"],
                0,
                "Gauranteed capacity is less than 0 in month {}".format(mnth),
            )
            self.assertAlmostEqual(
                results.loc[mnth, "guaranteedCap"],
                gaur_cap_exp[i],
                7,
                "Gauranted capacity not equal to expected value in {}".format(mnth),
            )
            self.assertEqual(
                results.loc[mnth, "pt_qty"],
                pt_qty_exp[i],
                "Point quantity not equal to expected values in {}".format(mnth),
            )

    def test_perc_bounds_perc(self):
        bounds = pvc.perc_bounds(20)
        self.assertEqual(bounds[0], 0.8, "{} for 20 perc is not 0.8".format(bounds[0]))
        self.assertEqual(bounds[1], 1.2, "{} for 20 perc is not 1.2".format(bounds[1]))

    def test_perc_bounds_tuple(self):
        bounds = pvc.perc_bounds((15, 40))
        self.assertEqual(
            bounds[0], 0.85, "{} for 15 perc is not 0.85".format(bounds[0])
        )
        self.assertEqual(bounds[1], 1.4, "{} for 40 perc is not 1.4".format(bounds[1]))

    def test_filter_grps(self):
        pvsyst = load_pvsyst(path="./tests/data/pvsyst_example_HourlyRes_2.CSV")
        pvsyst.set_regression_cols(
            power="real_pwr__", poa="irr_poa_", t_amb="temp_amb_", w_vel="wind__"
        )
        pvsyst.filter_irr(200, 800)
        pvsyst.rep_cond_freq(freq="MS")
        grps = pvsyst.data_filtered.groupby(pd.Grouper(freq="MS", label="left"))
        poa_col = pvsyst.column_groups[pvsyst.regression_cols["poa"]][0]

        grps_flt = pvc.filter_grps(grps, pvsyst.rc, poa_col, 0.8, 1.2, "MS")

        self.assertIsInstance(
            grps_flt,
            pd.core.groupby.generic.DataFrameGroupBy,
            "Returned object is not a dataframe groupby.",
        )

        self.assertEqual(
            grps.ngroups,
            grps_flt.ngroups,
            "Returned groubpy does not have the same number of\
                          groups as passed groupby.",
        )

        cnts_before_flt = grps.count()[poa_col]
        cnts_after_flt = grps_flt.count()[poa_col]
        less_than = all(cnts_after_flt < cnts_before_flt)
        self.assertTrue(less_than, "Points were not removed for each group.")

    def test_perc_difference(self):
        result = filters.perc_difference(9, 10)
        self.assertAlmostEqual(result, 0.105263158)

        result = filters.perc_difference(10, 9)
        self.assertAlmostEqual(result, 0.105263158)

        result = filters.perc_difference(10, 10)
        self.assertAlmostEqual(result, 0)

        result = filters.perc_difference(0, 0)
        self.assertAlmostEqual(result, 0)

    def test_check_all_perc_diff_comb(self):
        ser = pd.Series([10.1, 10.2])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertTrue(val, "Failed on two values within 5 percent.")

        ser = pd.Series([10.1, 10.2, 10.15, 10.22, 10.19])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertTrue(val, "Failed with 5 values within 5 percent.")

        ser = pd.Series([10.1, 10.2, 3])
        val = pvc.check_all_perc_diff_comb(ser, 0.05)
        self.assertFalse(val, "Returned True for value outside of 5 percent.")

    def test_sensor_filter_three_cols(self):
        rng = np.zeros(10)
        df = pd.DataFrame({"a": rng, "b": rng, "c": rng})
        df["a"] = df["a"] + 4.1
        df["b"] = df["b"] + 4
        df["c"] = df["c"] + 4.2
        df.iloc[0, 0] = 1200
        df.iloc[4, 1] = 100
        df.iloc[7, 2] = 150
        ix = filters.sensor_filter(df, 0.05)
        self.assertEqual(ix.shape[0], 7, "Filter should have droppe three rows.")

    def test_sensor_filter_one_col(self):
        rng = np.zeros(10)
        df = pd.DataFrame({"a": rng})
        df["a"] = df["a"] + 4.1
        df.iloc[0, 0] = 1200
        ix = filters.sensor_filter(df, 0.05)
        self.assertEqual(
            ix.shape[0], 10, "Should be no filtering for single column df."
        )

    def test_determine_pass_or_fail(self):
        # Tolerance band around 100%
        ct_pm = CapTest(test_tolerance="+/- 4", ac_nameplate=100)
        self.assertTrue(
            ct_pm.determine_pass_or_fail(0.96)[0],
            "Should pass, cp ratio equals bottom of tolerance.",
        )
        self.assertTrue(
            ct_pm.determine_pass_or_fail(0.97)[0],
            "Should pass, cp ratio above bottom of tolerance.",
        )
        self.assertTrue(
            ct_pm.determine_pass_or_fail(1.03)[0],
            "Should pass, cp ratio below top of tolerance.",
        )
        self.assertTrue(
            ct_pm.determine_pass_or_fail(1.04)[0],
            "Should pass, cp ratio equals top of tolerance.",
        )
        self.assertFalse(
            ct_pm.determine_pass_or_fail(0.959)[0],
            "Should fail, cp ratio below bottom of tolerance.",
        )
        self.assertFalse(
            ct_pm.determine_pass_or_fail(1.041)[0],
            "Should fail, cp ratio above top of tolerance.",
        )
        # Tolerance below 100%
        ct_minus = CapTest(test_tolerance="- 4", ac_nameplate=100)
        self.assertTrue(
            ct_minus.determine_pass_or_fail(0.96)[0],
            "Should pass, cp ratio equals bottom of tolerance.",
        )
        self.assertTrue(
            ct_minus.determine_pass_or_fail(0.97)[0],
            "Should pass, cp ratio above bottom of tolerance.",
        )
        self.assertTrue(
            ct_minus.determine_pass_or_fail(1.04)[0],
            "Should pass, cp ratio above bottom of tolerance.",
        )
        self.assertFalse(
            ct_minus.determine_pass_or_fail(0.959)[0],
            "Should fail, cp ratio below bottom of tolerance.",
        )
        # test fractional tolerance
        ct_frac = CapTest(test_tolerance="- 4.5", ac_nameplate=100)
        self.assertTrue(
            ct_frac.determine_pass_or_fail(0.956)[0],
            "Should pass, cp ratio above bottom of tolerance.",
        )
        # warn on incorrect tolerance spec
        ct_bad = CapTest(test_tolerance="+ 4", ac_nameplate=100)
        with self.assertWarns(UserWarning):
            ct_bad.determine_pass_or_fail(1.04)

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
        test_passed = (True, "950, 1050")
        captest_module.print_results(test_passed, 1000, 970, 0.97, 970, test_passed[1])
        captured = self.capsys.readouterr()

        results_str = (
            "Capacity Test Result:         PASS\n"
            "Modeled test output:          1000.000\n"
            "Actual test output:           970.000\n"
            "Tested output ratio:          0.970\n"
            "Tested Capacity:              970.000\n"
            "Bounds:                       950, 1050\n\n\n"
        )

        self.assertEqual(results_str, captured.out)

    def test_print_results_fail(self):
        """
        This test uses the pytest autouse fixture defined above to
        capture the print to stdout and test it, so it must be run
        using pytest 'pytest tests/
        test_CapData.py::TestTopLevelFuncs::test_print_results_pass'
        """
        test_passed = (False, "950, 1050")
        captest_module.print_results(test_passed, 1000, 940, 0.94, 940, test_passed[1])
        captured = self.capsys.readouterr()

        results_str = (
            "Capacity Test Result:    FAIL\n"
            "Modeled test output:          1000.000\n"
            "Actual test output:           940.000\n"
            "Tested output ratio:          0.940\n"
            "Tested Capacity:              940.000\n"
            "Bounds:                       950, 1050\n\n\n"
        )

        self.assertEqual(results_str, captured.out)


class TestCapDataEmpty:
    """Tests of CapData empty method."""

    def test_capdata_empty(self):
        """Test that an empty CapData object returns True."""
        empty_cd = pvc.CapData("empty")
        assert empty_cd.empty()

    def test_capdata_not_empty(self, meas):
        """Test that an CapData object with data returns False."""
        assert not meas.empty()


class TestCapDataSeriesTypes(unittest.TestCase):
    """Test CapData private methods assignment of type to each series of data."""

    def setUp(self):
        self.cdata = pvc.CapData("cdata")

    def test_series_type(self):
        name = "weather station 1 weather station 1 ghi poa w/m2"
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str, "Returned object is not a string.")
        self.assertEqual(out, "irr", 'Returned object is not "irr".')

    def test_series_type_caps_in_type_def(self):
        name = "weather station 1 weather station 1 ghi poa w/m2"
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        type_def = collections.OrderedDict(
            [
                (
                    "irr",
                    [
                        "IRRADIANCE",
                        "IRR",
                        "PLANE OF ARRAY",
                        "POA",
                        "GHI",
                        "GLOBAL",
                        "GLOB",
                        "W/M^2",
                        "W/M2",
                        "W/M",
                        "W/",
                    ],
                ),
            ]
        )
        out = cg.series_type(test_series, type_def)

        self.assertIsInstance(out, str, "Returned object is not a string.")
        self.assertEqual(out, "irr", 'Returned object is not "irr".')

    def test_series_type_repeatable(self):
        name = "weather station 1 weather station 1 ghi poa w/m2"
        test_series = pd.Series(np.arange(0, 900, 100), name=name)
        out = []
        i = 0
        while i < 100:
            out.append(cg.series_type(test_series, cg.type_defs))
            i += 1
        out_np = np.array(out)

        self.assertTrue(
            all(out_np == "irr"), "Result is not consistent after repeated runs."
        )

    def test_series_type_valErr(self):
        name = "weather station 1 weather station 1 ghi poa w/m2"
        test_series = pd.Series(name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str, "Returned object is not a string.")
        self.assertEqual(out, "irr", 'Returned object is not "irr".')

    def test_series_type_no_str(self):
        name = "should not return key string"
        test_series = pd.Series(name=name)
        out = cg.series_type(test_series, cg.type_defs)

        self.assertIsInstance(out, str, "Returned object is not a string.")
        self.assertIs(out, "", "Returned object is not empty string.")


class TestIndexCapdata:
    """Test the indexing functionality of the CapData loc method."""

    """All below tests are for the filter=True option."""

    def test_single_label_column_group_key_filtered(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key from data_filtered."""
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "irr_poa_pyran", filtered=True)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data_filtered[["met1_poa_pyranometer", "met2_poa_pyranometer"]]
        )
        assert out.shape[0] == 10

    def test_list_of_labels_column_group_keys_filtered(self, meas):
        """
        Test that a list of column_groups key returns the columns of Capdata.data that
        are the union of the values of the keys from data_filtered.
        """
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, ["irr_poa_pyran", "temp_amb"], filtered=True)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data_filtered[
                [
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                ]
            ]
        )
        assert out.shape[0] == 10

    def test_list_of_labels_reg_col_keys_filtered(self, meas):
        """
        Test that a list of regression_col keys returns the columns that
        are the regression_col and column_groups maps to in `data_filtered`.
        """
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, ["poa", "t_amb"], filtered=True)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data_filtered[
                [
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                ]
            ]
        )
        assert out.shape[0] == 10

    def test_list_of_labels_reg_col_keys_filtered_pvsyst(self, pvsyst_irr_filter):
        """
        Test that a list of regression_col keys returns the columns that
        are the regression_col and column_groups maps to in `data_filtered`.
        """
        pvsyst_irr_filter.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(
            pvsyst_irr_filter, ["poa", "t_amb", "w_vel"], filtered=True
        )
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            pvsyst_irr_filter.data_filtered[
                [
                    "GlobInc",
                    "T_Amb",
                    "WindVel",
                ]
            ]
        )
        assert out.shape[0] == 10

    def test_regcols_label_filtered(self, meas):
        """
        Test that passing the label `regcols` returns the columns of
        Capdata.data_filtered that are identified in `regression_cols`.
        """
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "regcols", filtered=True)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data_filtered[
                [
                    "meter_power",
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                    "met1_windspeed",
                    "met2_windspeed",
                ]
            ]
        )
        assert out.shape[0] == 10

    def test_regcols_label_after_agg_cols_filtered(self, meas):
        """
        Test that passing the label `regcols` returns the columns of
        Capdata.data_filtered that are identified in `regression_cols`.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "temp_amb": ("temp_amb", "mean"),
            "wind": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "regcols", filtered=True)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data_filtered[
                [
                    "meter_power",
                    "irr_poa_pyran_mean_agg",
                    "temp_amb_mean_agg",
                    "wind_mean_agg",
                ]
            ]
        )
        assert out.shape[0] == 10

    """#################################################"""
    """All below tests are for the filtered=False option."""
    """#################################################"""

    def test_single_label_column_group_key(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "irr_poa_pyran", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(meas.data[["met1_poa_pyranometer", "met2_poa_pyranometer"]])
        assert out.shape[0] == 1440

    def test_single_label_regression_columns_key(self, meas):
        """Test that regression_columns key returns the columns of Capdata.data that
        are the values of the key."""
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "poa", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(meas.data[["met1_poa_pyranometer", "met2_poa_pyranometer"]])
        assert out.shape[0] == 1440

    def test_single_label_regression_columns_after_agg(self, meas):
        """Test that regression_columns key returns the columns of Capdata.data that
        are the values of the key after agg_sensors has reset regression_columns
        to map to the new aggregated column."""
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        meas.regression_cols = {"poa": ("irr_poa_pyran", "mean")}
        meas.process_regression_columns()
        out = pvc.index_capdata(meas, "poa", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(meas.data["irr_poa_pyran_mean_agg"].to_frame())
        assert out.shape[0] == 1440

    def test_single_label_data_column_label(self, meas):
        """Test that a column label returns the columns of Capdata.data that
        are the values of the key. Passes label through to DataFrame.loc."""
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "met1_poa_pyranometer", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(meas.data.loc[:, "met1_poa_pyranometer"].to_frame())
        assert out.shape[0] == 1440

    def test_list_of_labels_column_group_keys(self, meas):
        """
        Test that a list of column_groups key returns the columns of Capdata.data that
        are the union of the values of the keys.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, ["irr_poa_pyran", "temp_amb"], filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_list_of_labels_regression_columns_keys(self, meas):
        """
        Test that a list of regression_columns key returns the columns of Capdata.data that
        are the union of the values of the keys.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, ["poa", "t_amb"], filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_list_of_labels_regression_columns_keys_after_agg(self, meas):
        """
        Test that a list of regression_columns key returns the columns of Capdata.data that
        are the new aggregated columns after agg_sensors has been run.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        meas.regression_cols = {
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
        }
        meas.process_regression_columns()
        out = pvc.index_capdata(meas, ["poa", "t_amb"], filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "irr_poa_pyran_mean_agg",
                    "temp_amb_mean_agg",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_list_of_labels_regression_columns_keys_after_partial_agg(self, meas):
        """
        Test that a list of regression_columns key returns the columns of Capdata.data
        that are the union of the values of the keys and the aggregated column.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"})
        meas.regression_cols = {"poa": "irr_poa_pyran_mean_agg", "t_amb": "temp_amb"}
        out = pvc.index_capdata(meas, ["poa", "t_amb"], filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "irr_poa_pyran_mean_agg",
                    "met1_amb_temp",
                    "met2_amb_temp",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_list_of_labels_data_column_labels(self, meas):
        """
        Test that a list of column labels returns the columns of Capdata.data.
        Passes labels through to DataFrame.loc.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(
            meas, ["met1_poa_pyranometer", "met2_amb_temp"], filtered=False
        )
        assert isinstance(out, pd.DataFrame)
        assert out.equals(meas.data.loc[:, ["met1_poa_pyranometer", "met2_amb_temp"]])
        assert out.shape[0] == 1440

    def test_list_of_labels_mixed(self, meas):
        """
        Test that a list containing a column_group, regression_columns key, and
        column labels returns the columns of Capdata.data that are the union of the
        values of the keys and the labels.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(
            meas, ["irr_poa_pyran", "t_amb", "met1_windspeed"], filtered=False
        )
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                    "met1_windspeed",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_list_of_labels_mixed_regression_column_maps_to_column_label(self, meas):
        """
        Test a list containing a regression_column key that maps directly to a column
        label rather than a column_group key is added to the columns returned.
        """
        # filter data_filtered to make check of row count for filtered=False meaningful
        meas.filter_custom(pd.DataFrame.head, 10)
        meas.regression_cols["poa"] = "met1_poa_pyranometer"
        out = pvc.index_capdata(
            meas, ["irr_poa_ref_cell", "poa", "met1_windspeed"], filtered=False
        )
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "met1_poa_refcell",
                    "met2_poa_refcell",
                    "met1_poa_pyranometer",
                    "met1_windspeed",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_regcols_label(self, meas):
        """
        Test that passing the label `regcols` returns the columns of
        Capdata.data that are identified in `regression_cols`.
        """
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "regcols", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "meter_power",
                    "met1_poa_pyranometer",
                    "met2_poa_pyranometer",
                    "met1_amb_temp",
                    "met2_amb_temp",
                    "met1_windspeed",
                    "met2_windspeed",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_regcols_label_after_agg_cols(self, meas):
        """
        Test that passing the label `regcols` returns the columns of
        Capdata that are identified in `regression_cols`.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "temp_amb": ("temp_amb", "mean"),
            "wind": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_custom(pd.DataFrame.head, 10)
        out = pvc.index_capdata(meas, "regcols", filtered=False)
        assert isinstance(out, pd.DataFrame)
        assert out.equals(
            meas.data[
                [
                    "meter_power",
                    "irr_poa_pyran_mean_agg",
                    "temp_amb_mean_agg",
                    "wind_mean_agg",
                ]
            ]
        )
        assert out.shape[0] == 1440

    def test_single_label_missing_key_warns(self, meas):
        """Warn with the missing label when it is not found anywhere in the CapData.

        A label that is not in `column_groups`, `regression_cols`, or the columns of
        `data` should emit a warning that names the offending label rather than
        raising `UnboundLocalError`.
        """
        meas.filter_custom(pd.DataFrame.head, 10)
        missing_label = "nonexistent_column_group"
        with pytest.warns(UserWarning, match=missing_label):
            out = pvc.index_capdata(meas, missing_label, filtered=True)
        assert out is None


class TestLocAndFloc:
    def test_single_label_column_group_key_loc(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        meas.filter_custom(pd.DataFrame.head, 10)
        out = meas.loc["irr_poa_pyran"]
        assert out.equals(meas.data[["met1_poa_pyranometer", "met2_poa_pyranometer"]])
        assert out.shape[0] == meas.data.shape[0]

    def test_single_label_column_group_key_floc(self, meas):
        """Test that column_groups key returns the columns of Capdata.data that
        are the values of the key."""
        meas.filter_custom(pd.DataFrame.head, 10)
        out = meas.floc["irr_poa_pyran"]
        assert out.equals(
            meas.data_filtered[["met1_poa_pyranometer", "met2_poa_pyranometer"]]
        )
        assert out.shape[0] == meas.data_filtered.shape[0]


class TestIrrRcBalanced:
    """Test the functionality of the irr_rc_balanced function"""

    def test_check_csv_output_exists(self, meas, tmp_path):
        """Check that function outputs a csv file when given a file path."""
        f = tmp_path / "output.csv"
        meas.regression_cols = {"poa": ("irr_poa_pyran", "mean")}
        meas.process_regression_columns()
        rep_irr = pvc.ReportingIrradiance(
            df=meas.data,
            irr_col=meas.regression_cols["poa"],
            percent_band=20,
        )
        rep_irr.get_rep_irr()
        rep_irr.save_csv(output_csv_path=f)
        assert f.exists()

    def test_irr_rc_balanced(self, pvsyst):
        jun = pvsyst.data.loc["06/1990"]
        jun_cpy = jun.copy()
        jun = jun.loc[jun["GlobInc"] > 400, :]
        print(jun)

        rc_tool = pvc.ReportingIrradiance(jun, "GlobInc", percent_band=50)
        rc_tool.min_ref_irradiance = 600
        rc_tool.max_ref_irradiance = 800
        (irr_RC, jun_flt) = rc_tool.get_rep_irr()
        print(irr_RC)
        print(jun_flt)
        print(rc_tool.poa_flt)
        jun_filter_irr = jun_flt["GlobInc"]
        assert all(jun_flt.columns == jun.columns)
        assert jun_flt.shape[0] > 0
        assert jun_flt.shape[0] < jun_cpy.shape[0]
        assert irr_RC > jun[jun["GlobInc"] > 0]["GlobInc"].min()
        assert irr_RC < jun["GlobInc"].max()

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
        jun = pvsyst.data.loc["06/1990"]
        jun = jun.loc[jun["GlobInc"] > 400, :]
        rc_tool = pvc.ReportingIrradiance(jun, "GlobInc", percent_band=50)
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
        jun = pvsyst.data.loc["06/1990"]
        jun = jun.loc[jun["GlobInc"] > 400, :]
        jun.loc[(jun["GlobInc"] > 600) & (jun["GlobInc"] < 700), "GlobInc"] = np.nan
        rc_tool = pvc.ReportingIrradiance(jun, "GlobInc", percent_band=50)
        rc_tool.min_ref_irradiance = 605
        rc_tool.max_ref_irradiance = 695
        with pytest.warns(UserWarning):
            rc_tool.get_rep_irr()
        assert isinstance(rc_tool.poa_flt, pd.DataFrame)
        assert np.isnan(rc_tool.irr_rc)


class TestCapDataCopy:
    def test_copy_of_pre_agg_attributes(self, meas):
        pre_agg_cols = copy.copy(meas.data.columns)
        pre_agg_col_groups = copy.deepcopy(meas.column_groups)
        pre_agg_reg_columns = copy.deepcopy(meas.regression_cols)
        meas.agg_sensors(
            agg_map={
                "irr_poa_pyran": "mean",
                "temp_amb": "mean",
                "wind": "mean",
            }
        )
        meas_copy = meas.copy()
        assert meas_copy.pre_agg_cols.equals(pre_agg_cols)
        assert meas_copy.pre_agg_trans == pre_agg_col_groups
        assert meas_copy.pre_agg_reg_trans == pre_agg_reg_columns
        assert meas_copy.pre_agg_cols.equals(meas.pre_agg_cols)
        assert meas_copy.pre_agg_trans == meas.pre_agg_trans
        assert meas_copy.pre_agg_reg_trans == meas.pre_agg_reg_trans

    def test_copy_column_groups_is_independent(self, pvsyst):
        """Verify mutating column_groups on the copy does not affect the original."""
        pvsyst_copy = pvsyst.copy()
        first_key = next(iter(pvsyst_copy.column_groups))
        pvsyst_copy.column_groups[first_key].append("new_col")
        assert "new_col" not in pvsyst.column_groups[first_key]

    def test_drop_cols_on_copy_does_not_affect_original(self, pvsyst):
        """Verify drop_cols on the copy leaves the original column_groups unchanged."""
        col_to_drop = "IL Pmax"
        original_group = list(pvsyst.column_groups["pvsyt_losses--"])
        pvsyst_copy = pvsyst.copy()
        pvsyst_copy.drop_cols([col_to_drop])
        assert col_to_drop not in pvsyst_copy.column_groups["pvsyt_losses--"]
        assert pvsyst.column_groups["pvsyt_losses--"] == original_group


class TestCapDataMethodsSim:
    """Test for top level irr_rc_balanced function."""

    def test_copy(self, pvsyst):
        pvsyst.set_regression_cols(
            power="real_pwr--", poa="irr-ghi-", t_amb="temp_amb", w_vel="wind--"
        )
        pvsyst_copy = pvsyst.copy()
        assert pvsyst_copy.data.equals(pvsyst.data)
        assert pvsyst_copy.column_groups == pvsyst.column_groups
        assert pvsyst_copy.regression_cols == pvsyst.regression_cols

    def test_filter_pvsyst_default(self, pvsyst):
        pvsyst.filter_pvsyst()
        assert pvsyst.data_filtered.shape[0] == 8670

    def test_filter_pvsyst_default_newer_pvsyst_var_names(self, pvsyst):
        pvsyst.data.rename(
            columns={
                "IL Pmin": "IL_Pmin",
                "IL Vmin": "IL_Vmin",
                "IL Pmax": "IL_Pmax",
                "IL Vmax": "IL_Vmax",
            },
            inplace=True,
        )
        assert pvsyst.data_filtered.shape[0] == 8760
        pvsyst.filter_pvsyst()
        assert pvsyst.data_filtered.shape[0] == 8670

    def test_filter_pvsyst_not_inplace(self, pvsyst):
        df = pvsyst.filter_pvsyst(inplace=False)
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == 8670

    def test_filter_pvsyst_missing_column(self, pvsyst):
        pvsyst.data.drop(columns="IL Pmin", inplace=True)
        with pytest.warns(
            UserWarning, match="IL_Pmin or IL Pmin is not a column in the data."
        ):
            pvsyst.filter_pvsyst()

    def test_filter_pvsyst_missing_all_columns(self, pvsyst):
        pvsyst.data.drop(
            columns=["IL Pmin", "IL Vmin", "IL Pmax", "IL Vmax"], inplace=True
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
        pvsyst.data.loc[pvsyst.data["FShdBm"] == 1.0, "ShdLoss"] = 0
        is_shaded = pvsyst.data["ShdLoss"].isna()
        shdloss_values = 1 / pvsyst.data.loc[is_shaded, "FShdBm"] * 100
        pvsyst.data.loc[is_shaded, "ShdLoss"] = shdloss_values

        pvsyst.filter_shade(query_str="ShdLoss<=125")
        assert pvsyst.data_filtered.shape[0] == 8671


class Test_pvlib_loc_sys(unittest.TestCase):
    """Test function wrapping pvlib get_clearsky method of Location."""

    def test_pvlib_location(self):
        loc = {
            "latitude": 30.274583,
            "longitude": -97.740352,
            "altitude": 500,
            "tz": "America/Chicago",
        }

        loc_obj = clearsky.pvlib_location(loc)

        self.assertIsInstance(
            loc_obj,
            pvlib.location.Location,
            "Did not return instance of\
                               pvlib Location",
        )

    def test_pvlib_system(self):
        fixed_sys = {"surface_tilt": 20, "surface_azimuth": 180, "albedo": 0.2}

        tracker_sys1 = {
            "axis_tilt": 0,
            "axis_azimuth": 0,
            "max_angle": 90,
            "backtrack": True,
            "gcr": 0.2,
            "albedo": 0.2,
        }

        tracker_sys2 = {"max_angle": 52, "gcr": 0.3}

        fx_sys = clearsky.pvlib_system(fixed_sys)
        trck_sys1 = clearsky.pvlib_system(tracker_sys1)
        trck_sys2 = clearsky.pvlib_system(tracker_sys2)

        self.assertIsInstance(
            fx_sys,
            pvlib.pvsystem.PVSystem,
            "Did not return instance of\
                               pvlib PVSystem",
        )
        self.assertIsInstance(
            fx_sys.arrays[0].mount,
            pvlib.pvsystem.FixedMount,
            "Did not return instance of\
                               pvlib FixedMount",
        )

        self.assertIsInstance(
            trck_sys1,
            pvlib.pvsystem.PVSystem,
            "Did not return instance of\
                               pvlib PVSystem",
        )
        self.assertIsInstance(
            trck_sys1.arrays[0].mount,
            pvlib.pvsystem.SingleAxisTrackerMount,
            "Did not return instance of\
                               pvlib SingleAxisTrackerMount",
        )

        self.assertIsInstance(
            trck_sys2,
            pvlib.pvsystem.PVSystem,
            "Did not return instance of\
                               pvlib PVSystem",
        )
        self.assertIsInstance(
            trck_sys2.arrays[0].mount,
            pvlib.pvsystem.SingleAxisTrackerMount,
            "Did not return instance of\
                               pvlib SingleAxisTrackerMount",
        )


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


class TestGetTimezoneIndex:
    """Test get_tz_index function."""

    def test_get_tz_index_df(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe without a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(
            start="11/3/2018", periods=864, freq="5min", tz="America/Chicago"
        )
        ix_2days = pd.date_range(
            start="3/9/2019", periods=576, freq="5min", tz="America/Chicago"
        )
        ix_dst = ix_3days.append(ix_2days)

        ix_dst = ix_dst.tz_localize(None)  # remove timezone from index

        df = pd.DataFrame(index=ix_dst)
        print(df.loc["11/4/18 01:00"].index)
        tz_ix = clearsky.get_tz_index(df, location_and_system["location"])
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        assert str(tz_ix.tz) == location_and_system["location"]["tz"]

    def test_get_tz_index_df_tz(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a dataframe WITH a timezone."""
        # reindex test dataset to cover DST in the fall and spring
        ix_3days = pd.date_range(
            start="11/3/2018", periods=864, freq="5min", tz="America/Chicago"
        )
        ix_2days = pd.date_range(
            start="3/9/2019", periods=576, freq="5min", tz="America/Chicago"
        )
        ix_dst = ix_3days.append(ix_2days)
        df = pd.DataFrame(index=ix_dst)
        tz_ix = clearsky.get_tz_index(df, location_and_system["location"])
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        assert str(tz_ix.tz) == location_and_system["location"]["tz"]

    def test_get_tz_index_df_tz_warn(self, location_and_system):
        """Test that get_tz_index function warns when datetime index\
           of dataframe does not match loc dic timezone."""
        df = pd.DataFrame(
            index=pd.date_range(
                start="11/3/2018", periods=864, freq="5min", tz="America/New_York"
            )
        )  # tz is New York
        with pytest.warns(
            UserWarning,
            match=(
                "The DatetimeIndex of time_source has a timezone that "
                "does not match the timezone in the loc dict. "
                "Using the timezone of the time_source DatetimeIndex."
            ),
        ):
            clearsky.get_tz_index(df, location_and_system["location"])  # tz is Chicago

    def test_get_tz_index_ix_tz(self, location_and_system):
        """Test that get_tz_index function returns a datetime index
        with a timezone when passed a datetime index with a timezone."""
        ix = pd.date_range(
            start="1/1/2019", periods=8760, freq="h", tz="America/Chicago"
        )
        tz_ix = clearsky.get_tz_index(
            ix, location_and_system["location"]
        )  # tz is Chicago
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        # If passing an index with a timezone use that timezone rather than
        # the timezone in the location dictionary if there is one.
        assert tz_ix.tz == ix.tz

    def test_get_tz_index_ix_tz_warn(self, location_and_system):
        """Test that get_tz_index function warns when DatetimeIndex timezone
        does not match the location dic timezone.
        """
        ix = pd.date_range(
            start="1/1/2019", periods=8760, freq="h", tz="America/New_York"
        )

        with pytest.warns(
            UserWarning,
            match=(
                "The DatetimeIndex of time_source has a timezone that "
                "does not match the timezone in the loc dict. "
                "Using the timezone of the time_source DatetimeIndex."
            ),
        ):
            clearsky.get_tz_index(ix, location_and_system["location"])

    def test_get_tz_index_ix(self, location_and_system):
        """Test that get_tz_index function returns a datetime index\
           with a timezone when passed a datetime index without a timezone."""
        ix = pd.date_range(
            start="1/1/2019", periods=8760, freq="h", tz="America/Chicago"
        )
        # remove timezone info but keep missing  hour and extra hour due to DST
        ix = ix.tz_localize(None)
        tz_ix = clearsky.get_tz_index(
            ix, location_and_system["location"]
        )  # tz is Chicago
        assert isinstance(tz_ix, pd.core.indexes.datetimes.DatetimeIndex)
        # If passing an index without a timezone use returned index should have
        # the timezone of the passed location dictionary.
        assert str(tz_ix.tz) == location_and_system["location"]["tz"]


class Test_csky:
    """Test clear sky function which returns pvlib ghi and poa clear sky."""

    def test_csky_concat(self, meas, location_and_system):
        # concat=True by default
        csky_ghi_poa = clearsky.csky(
            meas.data,
            loc=location_and_system["location"],
            sys=location_and_system["system"],
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == (meas.data.shape[1] + 2)
        assert "ghi_mod_csky" in csky_ghi_poa.columns
        assert "poa_mod_csky" in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert (
            csky_ghi_poa.loc["10/9/1990 12:30", "poa_mod_csky"]
            > csky_ghi_poa.loc["10/9/1990 12:30", "ghi_mod_csky"]
        )
        assert csky_ghi_poa.index.tz == df.index.tz

    def test_csky_concat_dst_spring(self, meas, location_and_system):
        """Test that csky concatenates clear sky ghi and poa when the time_source
        includes spring daylight savings time. This test assumes the time_source
        includes the 2 to 3AM hour that is skipped during daylight savings time."""
        # concat=True by default
        data = meas.data.loc["10/9/1990"]
        data.index = pd.date_range("3/12/23", periods=int((60 / 5) * 24), freq="5min")
        csky_ghi_poa = clearsky.csky(
            data, loc=location_and_system["location"], sys=location_and_system["system"]
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == (meas.data.shape[1] + 2)
        assert "ghi_mod_csky" in csky_ghi_poa.columns
        assert "poa_mod_csky" in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert (
            csky_ghi_poa.loc["3/12/23 12:30", "poa_mod_csky"]
            > csky_ghi_poa.loc["3/12/23 12:30", "ghi_mod_csky"]
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
        # csky_ghi_poa = clearsky.csky(
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
        csky_ghi_poa = clearsky.csky(
            meas.data,
            loc=location_and_system["location"],
            sys=location_and_system["system"],
            concat=False,
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 2
        assert "ghi_mod_csky" in csky_ghi_poa.columns
        assert "poa_mod_csky" in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert (
            csky_ghi_poa.loc["10/9/1990 12:30", "poa_mod_csky"]
            > csky_ghi_poa.loc["10/9/1990 12:30", "ghi_mod_csky"]
        )
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_poa_all(self, meas, location_and_system):
        csky_ghi_poa = clearsky.csky(
            meas.data,
            loc=location_and_system["location"],
            sys=location_and_system["system"],
            concat=False,
            output="poa_all",
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 5
        cols = [
            "poa_global",
            "poa_direct",
            "poa_diffuse",
            "poa_sky_diffuse",
            "poa_ground_diffuse",
        ]
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_ghi_all(self, meas, location_and_system):
        csky_ghi_poa = clearsky.csky(
            meas.data,
            loc=location_and_system["location"],
            sys=location_and_system["system"],
            concat=False,
            output="ghi_all",
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 3
        cols = ["ghi", "dni", "dhi"]
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_not_concat_all(self, meas, location_and_system):
        csky_ghi_poa = clearsky.csky(
            meas.data,
            loc=location_and_system["location"],
            sys=location_and_system["system"],
            concat=False,
            output="all",
        )
        assert isinstance(csky_ghi_poa, pd.core.frame.DataFrame)
        assert csky_ghi_poa.shape[1] == 8
        cols = [
            "ghi",
            "dni",
            "dhi",
            "poa_global",
            "poa_direct",
            "poa_diffuse",
            "poa_sky_diffuse",
            "poa_ground_diffuse",
        ]
        for col in cols:
            assert col in csky_ghi_poa.columns
        # assumes typical orientation is used to calculate the poa irradiance
        assert csky_ghi_poa.index.tz == meas.data.index.tz

    def test_csky_invalid_output_raises(self, meas, location_and_system):
        with pytest.raises(ValueError, match="Unrecognized output"):
            clearsky.csky(
                meas.data,
                loc=location_and_system["location"],
                sys=location_and_system["system"],
                concat=False,
                output="bad",
            )


"""
Change csky to two functions for creating pvlib location and system objects.
Separate function calling location and system to calculate POA
- concat add columns to passed df or return just ghi and poa option
load_data calls final function with in place to get ghi and poa
"""


class TestGetRegCols:
    """Test the get_reg_cols method of the CapData class."""

    def test_not_aggregated(self, meas):
        with pytest.warns(UserWarning):
            meas.get_reg_cols()

    def test_all_coeffs(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        cols = ["power", "poa", "t_amb", "w_vel"]
        df = meas.get_reg_cols()
        assert len(df.columns) == 4
        assert df.columns.to_list() == cols
        print(meas.data.columns)
        assert meas.data["irr_poa_pyran_mean_agg"].iloc[100] == df["poa"].iloc[100]
        assert meas.data["temp_amb_mean_agg"].iloc[100] == df["t_amb"].iloc[100]
        assert meas.data["wind_mean_agg"].iloc[100] == df["w_vel"].iloc[100]

    def test_agg_sensors_mix(self, meas):
        """
        Test when agg_sensors resets regression_cols values to a mix of trans keys
        and column names.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.agg_sensors(
            agg_map={
                "power_inv": "sum",
            }
        )
        cols = ["poa", "power"]
        df = meas.get_reg_cols(reg_vars=cols)
        mtr_col = meas.column_groups[meas.regression_cols["power"]][0]
        assert len(df.columns) == 2
        assert df.columns.to_list() == cols
        assert meas.data[mtr_col].iloc[100] == df["power"].iloc[100]
        assert meas.data["irr_poa_pyran_mean_agg"].iloc[100] == df["poa"].iloc[100]


class TestCapDataHelperMethods:
    def test_rename_cols(self, meas):
        meas.rename_cols({"met1_poa_pyranometer": "poa_new_name"})
        assert "poa_new_name" in meas.data.columns
        assert "poa_new_name" in meas.data_filtered.columns
        assert "poa_new_name" in meas.column_groups["irr_poa_pyran"]


class TestExpandAggMap:
    """Test the expand_agg_map method of the CapData class."""

    def test_expand_agg_map(self, cd_nested_col_groups):
        """Test the example from the docstring showing nested aggregation map expansion."""
        cd = cd_nested_col_groups

        # Input aggregation map with nested structure
        agg_map = {
            "irr_ghi": "mean",
            "irr_poa": {"irr_poa_met1": "mean", "irr_poa_met2": "mean"},
        }

        # Expected expanded map
        expected_expanded_map = {
            "irr_ghi": "mean",
            "irr_poa_met1": "mean",
            "irr_poa_met2": "mean",
            "irr_poa_aggs": "mean",
        }

        # Call the method
        expanded_map, rename_map, subgroup_rename_map = cd.expand_agg_map(agg_map)

        # Verify the expanded map matches expected (order independent)
        assert set(expanded_map.keys()) == set(expected_expanded_map.keys())
        for key in expected_expanded_map:
            assert expanded_map[key] == expected_expanded_map[key]

        # Verify the column groups were updated correctly
        assert cd.column_groups["irr_poa_aggs"] == [
            "irr_poa_met1_mean_agg",
            "irr_poa_met2_mean_agg",
        ]


class TestAggSensors:
    def test_get_group_valid_group_id_returns_dataframe(self, meas):
        """_get_group returns the expected DataFrame for a valid group_id."""
        result = meas._get_group("irr_poa_pyran")
        expected = meas.data[meas.column_groups["irr_poa_pyran"]]
        assert isinstance(result, pd.DataFrame)
        assert result.equals(expected)

    def test_get_group_invalid_group_id_raises_key_error(self, meas):
        """_get_group raises a KeyError that names the missing group_id."""
        missing_id = "irr_poa_pyran_typo"
        with pytest.raises(KeyError, match=missing_id):
            meas._get_group(missing_id)

    def test_agg_group_invalid_group_id_raises_key_error(self, meas):
        """KeyError naming the missing group_id is raised when it is not found."""
        missing_id = "irr_poa_pyran_typo"
        with pytest.raises(KeyError, match=missing_id):
            meas.agg_group(missing_id, "mean")

    def test_agg_sensors_invalid_group_id_raises_key_error(self, meas):
        """KeyError naming the missing group_id is raised when it is not found in agg_map."""
        missing_id = "irr_poa_pyran_typo"
        with pytest.raises(KeyError, match=missing_id):
            meas.agg_sensors(agg_map={missing_id: "mean"})

    def test_agg_group(self, meas):
        agg_result, col_name = meas.agg_group("irr_poa_pyran", "mean", inplace=False)
        assert "irr_poa_pyran_mean_agg" == col_name
        assert isinstance(agg_result, pd.DataFrame)
        exp_mean = (
            meas.data[meas.column_groups["irr_poa_pyran"]]
            .mean(axis=1)
            .rename("irr_poa_pyran_mean_agg")
            .to_frame()
        )
        assert exp_mean.shape == agg_result.shape
        assert exp_mean.equals(agg_result)

    def test_agg_group_groups_with_one_tag(self, meas_groups_with_one_tag):
        """
        Test when all groups passed in the agg_map only have one tag, so no
        aggregation occurs.
        """
        meas_groups_with_one_tag.agg_sensors(
            agg_map={"irr_poa_pyran": "mean", "irr_ghi_pyran": "mean"}
        )
        assert "agg" not in meas_groups_with_one_tag.column_groups

    def test_agg_map_none(self, meas):
        """Test default behaviour when no agg_map is passed."""
        meas.agg_sensors()
        # data and data_filtered should have same number of columns
        assert meas.data_filtered.shape[1] == meas.data.shape[1]
        # Rows should be the same in both dataframes
        assert meas.data_filtered.shape[0] == meas.data.shape[0]
        # Data after aggregation should not have sum of power columns because there
        # is only one power column, so it is not aggregated.
        assert "power_sum_agg" not in meas.data.columns
        assert "power_sum_agg" not in meas.data_filtered.columns

        # Check for poa aggregation column
        assert "irr_poa_pyran_mean_agg" in meas.data_filtered.columns
        # Check for amb temp aggregation column
        assert "temp_amb_mean_agg" in meas.data_filtered.columns
        # Check for wind aggregation column
        assert "wind_mean_agg" in meas.data_filtered.columns

    def test_agg_map_non_str_func(self, meas):
        meas.agg_sensors(agg_map={"irr_poa_pyran": np.mean})
        # data and data_filtered should have same number of columns
        assert meas.data_filtered.shape[1] == meas.data.shape[1]
        # Rows should be the same in both dataframes
        assert meas.data_filtered.shape[0] == meas.data.shape[0]
        # Check for poa aggregation column
        assert "irr_poa_pyran_mean_agg" in meas.data_filtered.columns

    def test_reset_summary(self, meas):
        meas.agg_sensors()
        # Aggregation clears the filter chain (and thus the summary).
        assert len(meas.filters) == 0

    def test_reset_agg_method(self, meas):
        orig_df = meas.data.copy()

        meas.agg_sensors()
        meas.filter_irr(200, 500, col_name="irr_poa_pyran_mean_agg")
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
        poa_key = meas.regression_cols["poa"]
        meas.column_groups[poa_key] = [meas.column_groups[poa_key][0]]
        meas.filter_irr(200, 800)
        with pytest.warns(
            UserWarning,
            match=(
                "The data_filtered attribute has been overwritten "
                "and previously applied filtering steps have been "
                "lost.  It is recommended to use agg_sensors "
                "before any filtering methods."
            ),
        ):
            meas.agg_sensors()

    def test_pre_agg_regression_dict_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_reg_trans, dict)

    def test_pre_agg_column_groups_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_trans, cg.ColumnGroups)

    def test_pre_agg_columns_exists(self, meas):
        meas.agg_sensors()
        assert isinstance(meas.pre_agg_cols, pd.Index)

    def test_verbose_false_no_aggregation_output(self, meas, capsys):
        """Default verbose=False should not print aggregation details."""
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"})
        captured = capsys.readouterr()
        assert "Aggregating the below" not in captured.out

    def test_verbose_prints_columns_when_lte_10(self, meas, capsys):
        """Verbose should print each column name when group has <= 10 columns."""
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"}, verbose=True)
        captured = capsys.readouterr()
        assert (
            "Aggregating the below 2 columns of the irr_poa_pyran group using the mean function"
            in captured.out
        )
        assert "irr_poa_pyran_mean_agg" in captured.out
        assert "    met1_poa_pyranometer" in captured.out
        assert "    met2_poa_pyranometer" in captured.out

    def test_verbose_prints_group_name_when_gt_10(self, meas, capsys):
        """Verbose shows first 3 and last 3 column names with ellipsis when > 10 cols."""
        extra_cols = [f"extra_poa_{i}" for i in range(10)]
        for col in extra_cols:
            meas.data[col] = meas.data["met1_poa_pyranometer"]
        meas.column_groups["irr_poa_pyran"] = (
            meas.column_groups["irr_poa_pyran"] + extra_cols
        )
        all_cols = meas.column_groups["irr_poa_pyran"]  # 12 total
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"}, verbose=True)
        captured = capsys.readouterr()
        assert (
            "OUTPUT TRUNCATED - Aggregating the below 12 columns of the irr_poa_pyran group using the mean function"
            in captured.out
        )
        # First 3 columns should be listed
        for col in all_cols[:3]:
            assert f"    {col}" in captured.out
        # Ellipsis separates the head from the tail
        assert "    ..." in captured.out
        # Last 3 columns should be listed
        for col in all_cols[-3:]:
            assert f"    {col}" in captured.out
        # A middle column should not appear
        assert f"    {all_cols[5]}" not in captured.out
        # Old single-line group message should not appear
        assert "Aggregating all columns of the irr_poa_pyran group" not in captured.out

    def test_agg_subgroups_expanded_map(self, cd_nested_col_groups):
        """
        Proof of concept test of idea to implement aggregating sub
        groups of sensors with agg_sensors by recursively traversing
        the original agg map, expanding it, sorting it, and adding
        the intermediate column groupings to the column groups.
        """
        cd = cd_nested_col_groups
        cd.column_groups["irr_poa_aggs"] = [
            "irr_poa_met1_mean_agg",
            "irr_poa_met2_mean_agg",
        ]
        cd.agg_sensors(
            agg_map={
                "irr_poa": "mean",
                "irr_poa_met1": "mean",
                "irr_poa_met2": "mean",
                "irr_poa_aggs": "mean",
            }
        )
        cd.rename_cols({"irr_poa_aggs_mean_agg": "irr_poa_mean_agg"})
        # test that all aggregated columns exist
        for agg_col in [
            "irr_poa_mean_agg",
            "irr_poa_met1_mean_agg",
            "irr_poa_met2_mean_agg",
        ]:
            assert agg_col in cd.data.columns

    def test_agg_subgroups(self, cd_nested_col_groups, capsys):
        cd = cd_nested_col_groups
        cd.regression_cols["poa"] = "irr_poa"
        cd.agg_sensors(
            agg_map={
                "irr_poa": {"irr_poa_met1": "mean", "irr_poa_met2": "mean"},
                "irr_rpoa": {"irr_rpoa_met1": "mean", "irr_rpoa_met2": "mean"},
            },
            verbose=True,
        )

        # Check stdout from agg_sensors: verify each group header and its columns appear.
        captured = capsys.readouterr()
        for group_id, col_name, cols in [
            (
                "irr_poa_met1",
                "irr_poa_met1_mean_agg",
                ["met1_poa1_pyranometer", "met1_poa2_pyranometer"],
            ),
            (
                "irr_poa_met2",
                "irr_poa_met2_mean_agg",
                ["met2_poa1_pyranometer", "met2_poa2_pyranometer"],
            ),
            (
                "irr_poa_aggs",
                "irr_poa_mean_agg",
                ["irr_poa_met1_mean_agg", "irr_poa_met2_mean_agg"],
            ),
            (
                "irr_rpoa_met1",
                "irr_rpoa_met1_mean_agg",
                [
                    "met1_rpoa1_pyranometer",
                    "met1_rpoa2_pyranometer",
                    "met1_rpoa3_pyranometer",
                ],
            ),
            (
                "irr_rpoa_met2",
                "irr_rpoa_met2_mean_agg",
                [
                    "met2_rpoa1_pyranometer",
                    "met2_rpoa2_pyranometer",
                    "met2_rpoa3_pyranometer",
                ],
            ),
            (
                "irr_rpoa_aggs",
                "irr_rpoa_mean_agg",
                ["irr_rpoa_met1_mean_agg", "irr_rpoa_met2_mean_agg"],
            ),
        ]:
            assert group_id in captured.out
            assert col_name in captured.out
            for col in cols:
                assert f"    {col}" in captured.out
        # Check that the expected columns exist
        expected_columns = [
            "irr_poa_mean_agg",
            "irr_poa_met1_mean_agg",
            "irr_poa_met2_mean_agg",
            "irr_rpoa_mean_agg",
            "irr_rpoa_met1_mean_agg",
            "irr_rpoa_met2_mean_agg",
        ]
        for agg_col in expected_columns:
            assert agg_col in cd.data.columns

        # Check that average of subgroup averages is as expected
        expected_mean = (
            cd.data["irr_poa_met1_mean_agg"] + cd.data["irr_poa_met2_mean_agg"]
        ) / 2
        assert expected_mean.equals(cd.data["irr_poa_mean_agg"])

        # Check that column_groups were updated correctly
        assert "agg" in cd.column_groups.keys()
        assert cd.column_groups["agg"] == [
            "irr_poa_met1_mean_agg",
            "irr_poa_met2_mean_agg",
            "irr_poa_mean_agg",
            "irr_rpoa_met1_mean_agg",
            "irr_rpoa_met2_mean_agg",
            "irr_rpoa_mean_agg",
        ]

        # check that aggs CapData attributes are created
        for agg_col in expected_columns:
            assert hasattr(cd, "aggs_" + agg_col)
            assert getattr(cd, "aggs_" + agg_col).equals(cd.data[agg_col].to_frame())

    def test_no_duplicating_agg_column(self, meas):
        """Test that agg_sensors does not create duplicate aggregation columns
        in CapData.data. Aggregation columns may already exist if
        process_regression_columns was run first.
        """
        meas.data["irr_poa_pyran_mean_agg"] = meas.data[
            [
                "met1_poa_pyranometer",
                "met2_poa_pyranometer",
            ]
        ].mean(axis=1)
        meas.column_groups["agg"] = ["irr_poa_pyran_mean_agg"]
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"})
        assert "irr_poa_pyran_mean_agg" in meas.data.columns
        # Should be one column which will return a series not a DataFrame
        assert len(set(meas.data.columns)) == len(meas.data.columns)
        assert isinstance(meas.data["irr_poa_pyran_mean_agg"], pd.Series)


class TestAggGroupCutoff:
    """Tests for the cutoff parameter of agg_group verbose output."""

    def _add_extra_cols_to_group(self, meas, n, group_id="irr_poa_pyran"):
        """Add n synthetic columns to a group and return the full updated column list."""
        extra_cols = [f"extra_poa_{i}" for i in range(n)]
        for col in extra_cols:
            meas.data[col] = meas.data["met1_poa_pyranometer"]
        meas.column_groups[group_id] = meas.column_groups[group_id] + extra_cols
        return meas.column_groups[group_id]

    def test_verbose_exactly_at_default_cutoff_shows_all_columns(self, meas, capsys):
        """All columns are listed when count equals the default cutoff of 10."""
        all_cols = self._add_extra_cols_to_group(meas, 8)  # 2 original + 8 extra = 10
        meas.agg_group("irr_poa_pyran", "mean", verbose=True)
        captured = capsys.readouterr()
        for col in all_cols:
            assert f"    {col}" in captured.out
        assert "    ..." not in captured.out

    def test_verbose_above_default_cutoff_shows_first_and_last_three(
        self, meas, capsys
    ):
        """First 3 and last 3 column names appear with ellipsis when count > 10."""
        all_cols = self._add_extra_cols_to_group(meas, 10)  # 2 + 10 = 12
        meas.agg_group("irr_poa_pyran", "mean", verbose=True)
        captured = capsys.readouterr()
        for col in all_cols[:3]:
            assert f"    {col}" in captured.out
        assert "    ..." in captured.out
        for col in all_cols[-3:]:
            assert f"    {col}" in captured.out

    def test_verbose_above_default_cutoff_excludes_middle_columns(self, meas, capsys):
        """Columns between the first 3 and last 3 are not printed when truncated."""
        all_cols = self._add_extra_cols_to_group(meas, 10)  # 2 + 10 = 12
        meas.agg_group("irr_poa_pyran", "mean", verbose=True)
        captured = capsys.readouterr()
        assert f"    {all_cols[5]}" not in captured.out

    def test_verbose_custom_cutoff_shows_all_at_boundary(self, meas, capsys):
        """All columns are listed when count equals a custom cutoff value."""
        all_cols = self._add_extra_cols_to_group(meas, 2)  # 2 + 2 = 4
        meas.agg_group("irr_poa_pyran", "mean", verbose=True, cutoff=4)
        captured = capsys.readouterr()
        for col in all_cols:
            assert f"    {col}" in captured.out
        assert "    ..." not in captured.out

    def test_verbose_custom_cutoff_truncates_above_boundary(self, meas, capsys):
        """Truncation triggers at the custom cutoff, not the default of 10."""
        all_cols = self._add_extra_cols_to_group(meas, 4)  # 2 + 4 = 6, cutoff=4
        meas.agg_group("irr_poa_pyran", "mean", verbose=True, cutoff=4)
        captured = capsys.readouterr()
        for col in all_cols[:3]:
            assert f"    {col}" in captured.out
        assert "    ..." in captured.out
        for col in all_cols[-3:]:
            assert f"    {col}" in captured.out


class TestFilterSensors:
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
            perc_diff={"irr_poa_ref_cell": 0.05, "temp_amb": 0.1}, inplace=True
        )
        # Check that data_filtered is still a dataframe
        assert isinstance(meas.data_filtered, pd.core.frame.DataFrame)
        # Check that rows were removed
        assert meas.data_filtered.shape[0] < rows_before_flt

    def test_after_agg_sensors(self, meas):
        rows_before_flt = meas.data_filtered.shape[0]
        meas.agg_sensors(
            agg_map={
                "power_inv": "sum",
                "irr_poa_ref_cell": "mean",
                "wind": "mean",
                "temp_amb": "mean",
            }
        )
        meas.filter_sensors(
            perc_diff={"irr_poa_ref_cell": 0.05, "temp_amb": 0.1},
            inplace=True,
        )
        assert isinstance(meas.data_filtered, pd.core.frame.DataFrame)
        assert meas.data_filtered.shape[0] < rows_before_flt
        # Filter_sensors should retain the aggregated columns
        assert "power_inv_sum_agg" in meas.data_filtered.columns


class TestAbsDiffFromAverage:
    """Test the abs_diff_from_average method of the CapData class."""

    def test_doesnt_meet_theshold(self):
        """Test that the method returns False when the absolute difference
        between at least one value in the Series and the average of the other values
        is greater than the threshold.
        """
        s = pd.Series([800, 805, 806, 840], index=["poa1", "poa2", "poa3", "poa4"])
        meets_threshold = filters.abs_diff_from_average(s, 25)
        assert meets_threshold is False

    def test_meets_theshold(self):
        """Test that the method returns True when the absolute difference
        between all values in the Series and the average of the other values
        is less than the threshold.
        """
        s = pd.Series([800, 805, 806, 801], index=["poa1", "poa2", "poa3", "poa4"])
        meets_threshold = filters.abs_diff_from_average(s, 25)
        assert meets_threshold is True

    def test_meets_theshold_with_nan(self):
        """Test that the method returns True when the absolute difference
        between all values in the Series and the average of the other values
        is less than the threshold.
        """
        s = pd.Series([800, 805, 806, np.nan], index=["poa1", "poa2", "poa3", "poa4"])
        meets_threshold = filters.abs_diff_from_average(s, 25)
        assert meets_threshold is True

    def test_equals_threshold(self):
        """Test that the method returns True when the absolute difference
        between all values in the Series and the average of the other values
        equals the threshold.
        """
        s = pd.Series([800, 800, 800, 825], index=["poa1", "poa2", "poa3", "poa4"])
        meets_threshold = filters.abs_diff_from_average(s, 25)
        assert meets_threshold is True

    def test_only_1_value(self):
        """Test that the method returns True when there is only one value in the
        Series. Check that method warns that there is only one value in the Series.
        """
        s = pd.Series([800], index=["poa1"])
        meets_threshold = filters.abs_diff_from_average(s, 25)
        assert meets_threshold is True


class TestFilterSensorsWithAbsDiffFromAverage:
    "Test filter_sensors method of CapData when row_filter is abs_diff_from_average."

    def test_does_not_drop_rows_when_no_outliers(self, capdata_irr):
        capdata_irr.filter_sensors(
            perc_diff={"poa": 25}, row_filter=filters.abs_diff_from_average
        )
        assert (capdata_irr.data.max(axis=1) - capdata_irr.data.min(axis=1) < 25).all()
        assert capdata_irr.data_filtered.shape[0] == capdata_irr.data.shape[0]

    def test_drops_rows_with_outliers(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        capdata_irr.filter_sensors(
            perc_diff={"poa": 25}, row_filter=filters.abs_diff_from_average
        )
        assert (capdata_irr.data.max(axis=1) - capdata_irr.data.min(axis=1) >= 25).any()
        assert capdata_irr.data_filtered.shape[0] == capdata_irr.data.shape[0] - 2


class TestRepCondNoFreq:
    def test_defaults(self, nrel):
        nrel.rep_cond()
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)

    def test_defaults_wvel(self, nrel):
        nrel.rep_cond(w_vel=50)
        assert nrel.rc["w_vel"][0] == 50

    def test_irr_bal(self, nrel):
        nrel.filter_irr(0.1, 2000)
        meas2 = nrel.copy()
        meas2.rep_cond()
        nrel.rep_cond(irr_bal=True, percent_filter=20)
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)
        assert nrel.rc["poa"].iloc[0] != meas2.rc["poa"].iloc[0]

    def test_irr_bal_wvel(self, nrel):
        nrel.rep_cond(irr_bal=True, percent_filter=20, w_vel=50)
        assert nrel.rc["w_vel"].iloc[0] == 50

    def test_custom_func_dict(self, nrel):
        """Passing a func dict overrides the mean default per rhs variable."""
        nrel.rep_cond(
            func={
                "poa": captest_module.perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            }
        )
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)

    def test_appends_zero_removal_step(self, nrel):
        """rep_cond records a RepCond step in the chain that removes nothing."""
        pts_before = nrel.data_filtered.shape[0]
        nrel.rep_cond()
        assert isinstance(nrel.filters[-1], filters.RepCond)
        assert nrel.filters[-1].pts_removed == 0
        assert nrel.data_filtered.shape[0] == pts_before
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)


class TestRepCondFreq:
    def test_monthly_no_irr_bal(self, pvsyst):
        pvsyst.rep_cond_freq(freq="ME")
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 12 rows
        assert pvsyst.rc.shape[0] == 12

    def test_monthly_irr_bal(self, pvsyst):
        pvsyst.rep_cond_freq(freq="ME", irr_bal=True, percent_filter=20)
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 12 rows
        assert pvsyst.rc.shape[0] == 12

    def test_seas_no_irr_bal(self, pvsyst):
        pvsyst.rep_cond_freq(freq="QE-DEC", irr_bal=False)
        # Check that the rc attribute is a dataframe
        assert isinstance(pvsyst.rc, pd.core.frame.DataFrame)
        # Rep conditions dataframe should have 4 rows
        assert pvsyst.rc.shape[0] == 4


class TestPredictCapacities:
    def test_monthly(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond_freq(freq="MS")
        pred_caps = pvsyst_irr_filter.predict_capacities(
            irr_filter=True, percent_filter=20
        )
        july_grpby = pred_caps.loc["1990-07-01", "PredCap"]

        # Check that the returned object is a dataframe
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        # Check that the returned dataframe has 12 rows
        assert pred_caps.shape[0] == 12

        pvsyst_irr_filter.filter_time(start="7/1/90", end="7/31/90 23:00")
        pvsyst_irr_filter.rep_cond()
        pvsyst_irr_filter.filter_irr(
            0.8, 1.2, ref_val=pvsyst_irr_filter.rc["poa"].iloc[0]
        )
        df = pvsyst_irr_filter.floc["regcols"]
        rename = {
            df.columns[0]: "power",
            df.columns[1]: "poa",
            df.columns[2]: "t_amb",
            df.columns[3]: "w_vel",
        }
        df = df.rename(columns=rename)
        reg = pvc.fit_model(df)
        july_manual = reg.predict(pvsyst_irr_filter.rc)[0]
        assert july_manual == pytest.approx(july_grpby)

    def test_no_irr_filter(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond_freq(freq="ME")
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=False)
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 12

    def test_rc_from_irrBal(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond_freq(freq="ME", irr_bal=True, percent_filter=20)
        pred_caps = pvsyst_irr_filter.predict_capacities(irr_filter=False)
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 12

    def test_seasonal_freq(self, pvsyst_irr_filter):
        pvsyst_irr_filter.rep_cond_freq(freq="QE-DEC")
        pred_caps = pvsyst_irr_filter.predict_capacities(
            irr_filter=True, percent_filter=20
        )
        assert isinstance(pred_caps, pd.core.frame.DataFrame)
        assert pred_caps.shape[0] == 4


class TestFilterIrr:
    def test_get_poa_col(self, nrel):
        col = nrel._get_poa_col()
        assert col == "POA 40-South CMP11 [W/m^2]"

    def test_get_poa_col_multcols(self, nrel):
        nrel.data["POA second column"] = nrel.loc["poa"].values
        nrel.column_groups["irr-poa-"].append("POA second column")
        with pytest.warns(
            UserWarning,
            match=(
                "[0-9]+ columns of irradiance data. Use col_name to specify a single column."
            ),
        ):
            nrel._get_poa_col()

    def test_lowhigh_nocol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(500, 600, ref_val=None, col_name=None, inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_lowhigh_colname(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.data["POA second column"] = nrel.loc["poa"].values
        nrel.column_groups["irr-poa-"].append("POA second column")
        nrel.filter_irr(
            500, 600, ref_val=None, col_name="POA second column", inplace=True
        )
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_nocol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(0.8, 1.2, ref_val=500, col_name=None, inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_withcol(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        nrel.data["POA second column"] = nrel.loc["poa"].values
        nrel.column_groups["irr-poa-"].append("POA second column")
        nrel.filter_irr(
            0.8, 1.2, ref_val=500, col_name="POA second column", inplace=True
        )
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_use_attribute(self, nrel):
        nrel.rc = pd.DataFrame({"poa": 500, "w_vel": 1, "t_amb": 20}, index=[0])
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(0.8, 1.2, ref_val="rep_irr", col_name=None, inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_self_val_translation(self, nrel):
        """'self_val' is silently translated to 'rep_irr' and filters correctly."""
        nrel.rc = pd.DataFrame({"poa": 500, "w_vel": 1, "t_amb": 20}, index=[0])
        pts_before = nrel.data_filtered.shape[0]
        nrel.filter_irr(0.8, 1.2, ref_val="self_val", col_name=None, inplace=True)
        assert nrel.data_filtered.shape[0] < pts_before

    def test_refval_rep_irr_shows_in_summary(self, nrel):
        """Resolved numeric value appears in summary, not the sentinel string."""
        nrel.rc = pd.DataFrame({"poa": 500.0, "w_vel": 1, "t_amb": 20}, index=[0])
        nrel.filter_irr(0.8, 1.2, ref_val="rep_irr")
        summary = nrel.get_summary()
        filter_args = summary["filter_arguments"].iloc[0]
        assert "rep_irr" not in filter_args
        assert "np." not in filter_args
        assert "500" in filter_args

    def test_refval_rep_irr_rc_none_raises(self, nrel):
        """ValueError is raised when ref_val='rep_irr' and self.rc is None."""
        with pytest.raises(ValueError, match="Call rep_cond\\(\\) before"):
            nrel.filter_irr(0.8, 1.2, ref_val="rep_irr")

    def test_refval_rep_irr_no_poa_col_raises(self, nrel):
        """ValueError when self.rc exists but has no 'poa' column."""
        nrel.rc = pd.DataFrame({"irr": 500, "w_vel": 1, "t_amb": 20}, index=[0])
        with pytest.raises(ValueError, match="does not have a 'poa' column"):
            nrel.filter_irr(0.8, 1.2, ref_val="rep_irr")

    def test_refval_withcol_notinplace(self, nrel):
        pts_before = nrel.data_filtered.shape[0]
        df = nrel.filter_irr(500, 600, ref_val=None, col_name=None, inplace=False)
        assert nrel.data_filtered.shape[0] == pts_before
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] < pts_before


class TestGetSummary:
    def test_col_names(self, nrel):
        nrel.filter_irr(200, 500)
        smry = nrel.get_summary()
        assert smry.columns[0] == "function_name"
        assert smry.columns[1] == "pts_after_filter"
        assert smry.columns[2] == "pts_removed"
        assert smry.columns[3] == "filter_arguments"

    def test_empty_when_no_filters(self, nrel):
        """No filters -> empty DataFrame with the standard columns, not None.

        This is the path CapTest.get_summary relies on when it concatenates the
        meas and sim summaries; the prior None return would have raised there.
        """
        nrel.reset_filter()
        smry = nrel.get_summary()
        assert smry.empty
        assert list(smry.columns) == pvc.columns

    def test_function_name_is_class_name(self, nrel):
        nrel.filter_irr(200, 500)
        smry = nrel.get_summary()
        assert smry["function_name"].iloc[0] == "FilterIrr"

    def test_function_name_stays_class_name_with_custom_label(self, nrel):
        """function_name is always the class name even when the index label is a
        custom_name; the two are intentionally allowed to diverge."""
        nrel.filter_custom(pd.DataFrame.head, 5, custom_name="My custom step")
        smry = nrel.get_summary()
        assert smry.index[0][1] == "My custom step"
        assert smry["function_name"].iloc[0] == "FilterCustom"


class TestFilterTime:
    def test_start_end(self, pvsyst):
        pvsyst.filter_time(start="2/1/90", end="2/15/90")
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=2, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=2, day=15, hour=00
        )

    def test_start_end_drop_is_true(self, pvsyst):
        pvsyst.filter_time(start="2/1/90", end="2/15/90", drop=True)
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=1, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=12, day=31, hour=23
        )
        assert pvsyst.data_filtered.shape[0] == (8760 - 14 * 24) - 1

    def test_start_days(self, pvsyst):
        pvsyst.filter_time(start="2/1/90", days=15)
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=2, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=2, day=16, hour=00
        )

    def test_end_days(self, pvsyst):
        pvsyst.filter_time(end="2/16/90", days=15)
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=2, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=2, day=16, hour=00
        )

    def test_test_date(self, pvsyst):
        pvsyst.filter_time(test_date="2/16/90", days=30)
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=2, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=3, day=3, hour=00
        )

    def test_start_end_not_inplace(self, pvsyst):
        df = pvsyst.filter_time(start="2/1/90", end="2/15/90", inplace=False)
        assert df.index[0] == pd.Timestamp(year=1990, month=2, day=1, hour=0)
        assert df.index[-1] == pd.Timestamp(year=1990, month=2, day=15, hour=00)

    def test_start_no_end_uses_last_timestamp(self, pvsyst):
        """Verify that omitting end defaults to the last timestamp of data_filtered."""
        last_ts = pvsyst.data_filtered.index[-1]
        pvsyst.filter_time(start="2/1/90")
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=2, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == last_ts

    def test_end_no_start_uses_first_timestamp(self, pvsyst):
        """Verify that omitting start defaults to the first timestamp of data_filtered."""
        first_ts = pvsyst.data_filtered.index[0]
        pvsyst.filter_time(end="11/1/90")
        assert pvsyst.data_filtered.index[0] == first_ts
        assert pvsyst.data_filtered.index[-1] == pd.Timestamp(
            year=1990, month=11, day=1, hour=0
        )

    def test_start_no_end_respects_prefilterd_boundary(self, pvsyst):
        """Verify end default reflects data_filtered boundary after a prior filter."""
        pvsyst.filter_time(start="1/1/90", end="6/30/90")
        last_ts = pvsyst.data_filtered.index[-1]
        pvsyst.filter_time(start="3/1/90")
        assert pvsyst.data_filtered.index[0] == pd.Timestamp(
            year=1990, month=3, day=1, hour=0
        )
        assert pvsyst.data_filtered.index[-1] == last_ts

    def test_test_date_no_days(self, pvsyst):
        with pytest.warns(UserWarning):
            pvsyst.filter_time(test_date="2/1/90")


class TestFilterDays:
    def test_keep_one_day(self, pvsyst):
        pvsyst.filter_days(["10/5/1990"], drop=False, inplace=True)
        assert pvsyst.data_filtered.shape[0] == 24
        assert pvsyst.data_filtered.index[0].day == 5

    def test_keep_two_contiguous_days(self, pvsyst):
        pvsyst.filter_days(["10/5/1990", "10/6/1990"], drop=False, inplace=True)
        assert pvsyst.data_filtered.shape[0] == 48
        assert pvsyst.data_filtered.index[-1].day == 6

    def test_keep_three_noncontiguous_days(self, pvsyst):
        pvsyst.filter_days(
            ["10/5/1990", "10/7/1990", "10/9/1990"], drop=False, inplace=True
        )
        assert pvsyst.data_filtered.shape[0] == 72
        assert pvsyst.data_filtered.index[0].day == 5
        assert pvsyst.data_filtered.index[25].day == 7
        assert pvsyst.data_filtered.index[49].day == 9

    def test_drop_one_day(self, pvsyst):
        pvsyst.filter_days(["1/1/1990"], drop=True, inplace=True)
        assert pvsyst.data_filtered.shape[0] == (8760 - 24)
        assert pvsyst.data_filtered.index[0].day == 2
        assert pvsyst.data_filtered.index[0].hour == 0

    def test_drop_three_days(self, pvsyst):
        pvsyst.filter_days(
            ["1/1/1990", "1/3/1990", "1/5/1990"], drop=True, inplace=True
        )
        assert pvsyst.data_filtered.shape[0] == (8760 - 24 * 3)
        assert pvsyst.data_filtered.index[0].day == 2
        assert pvsyst.data_filtered.index[25].day == 4
        assert pvsyst.data_filtered.index[49].day == 6

    def test_not_inplace(self, pvsyst):
        df = pvsyst.filter_days(["10/5/1990"], drop=False, inplace=False)
        assert pvsyst.data_filtered.shape[0] == 8760
        assert df.shape[0] == 24


class TestFilterPF:
    def test_pf(self, nrel):
        pf = np.ones(5)
        pf = np.append(pf, np.ones(5) * -1)
        pf = np.append(pf, np.arange(0, 1, 0.1))
        nrel.data["pf"] = np.tile(pf, 576)
        nrel.column_groups["pf--"] = ["pf"]
        nrel.trans_keys = list(nrel.column_groups.keys())
        nrel.filter_pf(1)
        assert nrel.data_filtered.shape[0] == 5760


class TestFilterOutliersAndPower:
    def test_not_aggregated(self, meas):
        with pytest.warns(UserWarning):
            meas.filter_outliers()

    def test_filter_outliers_warns_and_succeeds_when_nans_present(self, pvsyst):
        """filter_outliers warns and auto-removes NaN rows in poa/power before fitting."""
        pvsyst.data.iloc[0, pvsyst.data.columns.get_loc("GlobInc")] = np.nan
        pvsyst.data.iloc[1, pvsyst.data.columns.get_loc("E_Grid")] = np.nan
        initial_rows = pvsyst.data_filtered.shape[0]

        with pytest.warns(UserWarning, match="missing values"):
            pvsyst.filter_outliers()

        assert pvsyst.data_filtered.shape[0] < initial_rows
        assert pvsyst.data_filtered[["GlobInc", "E_Grid"]].isna().sum().sum() == 0

    def test_filter_outliers_nan_records_filter_missing_in_summary(self, pvsyst):
        """When filter_outliers auto-calls filter_missing, both are recorded in summary."""
        pvsyst.data.iloc[0, pvsyst.data.columns.get_loc("GlobInc")] = np.nan

        with pytest.warns(UserWarning):
            pvsyst.filter_outliers()

        filter_names = [ix[1] for ix in pvsyst.get_summary().index]
        assert filter_names.index("FilterMissing") < filter_names.index(
            "FilterOutliers"
        )

    def test_filter_power_defaults(self, meas):
        meas.filter_power(5_000_000, percent=None, columns=None, inplace=True)
        assert meas.data_filtered.shape[0] == 1289

    def test_filter_power_percent(self, meas):
        meas.filter_power(6_000_000, percent=0.05, columns=None, inplace=True)
        assert meas.data_filtered.shape[0] == 1388

    def test_filter_power_a_column(self, meas):
        print(meas.data.columns)
        meas.filter_power(5_000_000, percent=None, columns="meter_power", inplace=True)
        assert meas.data_filtered.shape[0] == 1289

    def test_filter_power_column_group(self, meas):
        meas.filter_power(500_000, percent=None, columns="power_inv", inplace=True)
        assert meas.data_filtered.shape[0] == 1138

    def test_filter_power_column_group_with_nan(self, meas):
        """NaN values in a multi-column power group should not cause row removal."""
        # Introduce NaN in one inverter column
        meas.data.iloc[0, meas.data.columns.get_loc("inv1_power")] = np.nan
        meas.data.iloc[1, meas.data.columns.get_loc("inv2_power")] = np.nan
        meas.filter_power(500_000, percent=None, columns="power_inv", inplace=True)
        # Rows with NaN should still be present (NaN < threshold is treated as True)
        assert meas.data_filtered.shape[0] == 1138

    def test_filter_power_columns_not_str(self, meas):
        with pytest.warns(UserWarning):
            meas.filter_power(500_000, percent=None, columns=1, inplace=True)


class TestCskyFilter:
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
        nrel_clear_sky.data["ws 2 ghi W/m^2"] = nrel_clear_sky.loc["irr-ghi-"] * 1.05
        nrel_clear_sky.column_groups["irr-ghi-"].append("ws 2 ghi W/m^2")
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_mult_ghi_categories(self, nrel_clear_sky):
        nrel_clear_sky.data["irrad ghi pyranometer W/m^2"] = (
            nrel_clear_sky.data.loc[:, "Global CMP22 (vent/cor) [W/m^2]"] * 1.05
        )
        nrel_clear_sky.column_groups["irr-ghi-pyran"] = ["irrad ghi pyranometer W/m^2"]
        nrel_clear_sky.trans_keys = list(nrel_clear_sky.column_groups.keys())
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_no_clear_ghi(self, nrel_clear_sky):
        nrel_clear_sky.drop_cols("ghi_mod_csky")
        with pytest.warns(UserWarning):
            nrel_clear_sky.filter_clearsky()

    def test_specify_ghi_col(self, nrel_clear_sky):
        nrel_clear_sky.data["ws 2 ghi W/m^2"] = nrel_clear_sky.loc["irr-ghi-"] * 1.05
        nrel_clear_sky.column_groups["irr-ghi-"].append("ws 2 ghi W/m^2")
        nrel_clear_sky.trans_keys = list(nrel_clear_sky.column_groups.keys())

        nrel_clear_sky.filter_clearsky(ghi_col="ws 2 ghi W/m^2")

        assert nrel_clear_sky.data_filtered.shape[0] < nrel_clear_sky.data.shape[0]
        assert nrel_clear_sky.data_filtered.shape[1] == nrel_clear_sky.data.shape[1]
        for i, col in enumerate(nrel_clear_sky.data_filtered.columns):
            assert col == nrel_clear_sky.data.columns[i]

    def test_infer_limits_default(self, nrel_clear_sky):
        """Verify infer_limits=True is passed to detect_clearsky by default."""
        with unittest.mock.patch(
            "captest.filters.detect_clearsky", wraps=filters.detect_clearsky
        ) as mock_detect:
            nrel_clear_sky.filter_clearsky()
            mock_detect.assert_called_once()
            _, call_kwargs = mock_detect.call_args
            assert call_kwargs["infer_limits"] is True

    def test_kwargs_passed_to_detect_clearsky(self, nrel_clear_sky):
        """Verify user can override infer_limits and pass window_length."""
        with unittest.mock.patch(
            "captest.filters.detect_clearsky", wraps=filters.detect_clearsky
        ) as mock_detect:
            nrel_clear_sky.filter_clearsky(infer_limits=False, window_length=30)
            mock_detect.assert_called_once()
            _, call_kwargs = mock_detect.call_args
            assert call_kwargs["infer_limits"] is False
            assert call_kwargs["window_length"] == 30


class TestFilterMissing:
    """
    Newer tests written for pytest. Uses the meas pytest fixture defined below.
    """

    def test_filter_missing_default(self, meas):
        """Checks missing data in regression columns are removed."""
        print(meas.data.columns)
        meas.set_regression_cols(
            power="meter_power",
            poa="met1_poa_refcell",
            t_amb="met2_amb_temp",
            w_vel="met1_windspeed",
        )
        assert all(meas.floc["regcols"].isna().sum() == 0)
        assert meas.data_filtered.shape[0] == 1440
        meas.data.loc["10/9/90 12:00", "meter_power"] = np.nan
        meas.data.loc["10/9/90 12:30", "met1_poa_refcell"] = np.nan
        meas.data.loc["10/10/90 12:35", "met2_amb_temp"] = np.nan
        meas.data.loc["10/10/90 12:50", "met1_windspeed"] = np.nan
        meas.filter_missing()
        assert meas.data_filtered.shape[0] == 1436

    def test_filter_missing_missing_not_in_columns_considered(self, meas):
        """Checks that nothing is dropped for missing data not in `columns`."""
        meas.set_regression_cols(
            power="meter_power",
            poa="met1_poa_refcell",
            t_amb="met2_amb_temp",
            w_vel="met1_windspeed",
        )
        assert all(meas.floc["regcols"].isna().sum() == 0)
        assert meas.data_filtered.shape[0] == 1440
        assert meas.data_filtered.isna().sum().sum() > 0
        meas.filter_missing()
        assert meas.data_filtered.shape[0] == 1440

    def test_filter_missing_missing_passed_columns(self, meas):
        """Checks that nothing is dropped for missing data not in `columns`."""
        assert meas.data_filtered.shape[0] == 1440
        assert meas.data_filtered.isna().sum().sum() > 0
        meas.filter_missing(columns=["met1_amb_temp"])
        assert meas.data_filtered.shape[0] == 1424


class TestStatsmodelsParamModification:
    """
    Tests documenting statsmodels parameter modification behavior.

    Using `model.predict(modified_params, exog)` works consistently across
    pandas versions. Direct assignment to `.params` has inconsistent behavior:
    - pandas 2.x: Direct modification DOES affect predictions
    - pandas 3.0+: Direct modification does NOT affect predictions (CoW)

    This behavior is important for pandas 3.0 Copy-on-Write compatibility.
    """

    @pytest.fixture
    def regression_model(self):
        """Create a simple regression model for testing."""
        np.random.seed(42)
        df = pd.DataFrame({"y": np.random.randn(100), "x": np.random.randn(100)})
        results = smf.ols("y ~ x", data=df).fit()
        test_df = pd.DataFrame({"x": [1]})  # i.e. reporting conditions
        return results, test_df

    def test_model_predict_with_custom_params(self, regression_model):
        """
        Verify that model.predict(modified_params, exog) correctly uses
        the modified parameters for prediction.

        This approach works consistently across pandas 2.x and 3.0+.
        """
        results, test_df = regression_model

        # Get original prediction
        orig_pred = results.predict(test_df)[0]

        # Create modified params with x coefficient set to 0
        modified_params = results.params.copy()
        modified_params["x"] = 0

        # Create design matrix for prediction
        design_info = results.model.data.design_info
        exog = dmatrix(design_info, test_df)

        # Predict with modified params
        pred_with_modified = results.model.predict(modified_params, exog)[0]

        # With x=0 coefficient, prediction should equal intercept
        expected = results.params["Intercept"]

        assert pred_with_modified == pytest.approx(expected, abs=1e-10), (
            "Prediction with x=0 should equal intercept"
        )
        assert orig_pred != pytest.approx(pred_with_modified, abs=1e-5), (
            "Modified prediction should differ from original"
        )


class TestPredictWithPvalueCheck:
    """
    Tests for predict_with_pvalue_check function.

    This function makes predictions using model.predict() with custom params
    for consistent behavior across pandas versions.
    """

    @pytest.fixture
    def capdata_with_regression(self):
        """
        Create a CapData object with fitted regression and reporting conditions.

        Uses data where some coefficients will have high p-values (insignificant)
        to test the p-value filtering functionality.
        """
        np.random.seed(42)

        cd = pvc.CapData("test")

        # Create data where 'x' is a strong predictor but 'noise' is not
        n = 100
        x = np.linspace(0, 10, n)
        noise_var = np.random.randn(n) * 0.01  # Very weak relationship
        y = 2 * x + 5 + np.random.randn(n) * 0.5  # Strong relationship with x

        df = pd.DataFrame({"y": y, "x": x, "noise": noise_var})
        cd.data = df

        # Fit model with both significant (x) and insignificant (noise) predictors
        fml = "y ~ x + noise"
        model = smf.ols(formula=fml, data=df)
        cd.regression_results = model.fit()

        # Set reporting conditions
        cd.rc = pd.DataFrame({"x": [5.0], "noise": [0.0]})

        return cd

    def test_predict_no_pvalue_threshold(self, capdata_with_regression):
        """Test prediction without p-value filtering (threshold=None)."""
        cd = capdata_with_regression

        # Prediction without filtering should match standard predict
        expected = cd.regression_results.predict(cd.rc)[0]
        actual = pvc.predict_with_pvalue_check(cd, pval_threshold=None)

        assert actual == pytest.approx(expected, rel=1e-10)

    def test_predict_with_pvalue_threshold_zeros_insignificant(
        self, capdata_with_regression
    ):
        """Test that insignificant coefficients are zeroed with p-value threshold."""
        cd = capdata_with_regression

        # Check that 'noise' coefficient has high p-value (should be filtered)
        noise_pval = cd.regression_results.pvalues["noise"]
        assert noise_pval > 0.05, "Test setup error: noise should be insignificant"

        # Get prediction with p-value filtering
        pred_with_filter = pvc.predict_with_pvalue_check(cd, pval_threshold=0.05)

        # The key test is that the function runs without error and returns a float
        assert isinstance(pred_with_filter, (float, np.floating))

    def test_predict_with_very_low_threshold_zeros_all(self, capdata_with_regression):
        """Test with very low threshold that zeros out all coefficients."""
        cd = capdata_with_regression

        # With threshold of 0, all coefficients should be zeroed
        # (no p-value is exactly 0)
        pred = pvc.predict_with_pvalue_check(cd, pval_threshold=0)

        # Result should be close to 0 (all coeffs zeroed, including intercept)
        # or equal to intercept if intercept p-value is 0
        assert isinstance(pred, (float, np.floating))

    def test_predict_with_high_threshold_keeps_all(self, capdata_with_regression):
        """Test with threshold of 1.0 keeps all coefficients."""
        cd = capdata_with_regression

        # With threshold of 1.0, no coefficients should be zeroed
        pred_high_threshold = pvc.predict_with_pvalue_check(cd, pval_threshold=1.0)
        pred_no_filter = pvc.predict_with_pvalue_check(cd, pval_threshold=None)

        assert pred_high_threshold == pytest.approx(pred_no_filter, rel=1e-10)


class TestCapTestCpResultsSingleCoeff(unittest.TestCase):
    """Tests for the capactiy test results method using a regression formula
    with a single coefficient."""

    def setUp(self):
        np.random.seed(9876789)

        self.meas = pvc.CapData("meas")
        self.sim = pvc.CapData("sim")
        # self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.meas.rc = {"x": [6]}

        nsample = 100
        e = np.random.normal(size=nsample)

        x = np.linspace(0, 10, 100)
        das_y = x * 2
        sim_y = x * 2 + 1

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({"y": das_y, "x": x})
        sim_df = pd.DataFrame({"y": sim_y, "x": x})

        das_model = smf.ols(formula="y ~ x - 1", data=das_df)
        sim_model = smf.ols(formula="y ~ x - 1", data=sim_df)

        self.meas.regression_results = das_model.fit()
        self.sim.regression_results = sim_model.fit()

    def test_return(self):
        ct = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        ct.meas = self.meas
        ct.sim = self.sim
        res = ct.captest_results(print_res=False)

        self.assertIsInstance(res, float, "Returned value is not a tuple")


class TestCapTestCpResultsMultCoeffKwVsW(unittest.TestCase):
    """
    Setup and test to check automatic adjustment for kW vs W.
    """

    def test_pvals_default_false_kw_vs_w(self):
        np.random.seed(9876789)

        meas = pvc.CapData("meas")
        sim = pvc.CapData("sim")
        # cptest = pvc.CapTest(meas, sim, '+/- 5')
        meas.rc = pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]})

        nsample = 100
        e = np.random.normal(size=nsample)

        a = np.linspace(0, 10, 100)
        b = np.linspace(0, 10, 100) / 2.0
        c = np.linspace(0, 10, 100) + 3.0

        das_y = a + (a**2) + (a * b) + (a * c)
        sim_y = a + (a**2 * 0.9) + (a * b * 1.1) + (a * c * 0.8)

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({"power": das_y, "poa": a, "t_amb": b, "w_vel": c})
        sim_df = pd.DataFrame({"power": sim_y, "poa": a, "t_amb": b, "w_vel": c})

        meas.data = das_df
        meas.data["power"] /= 1000
        meas.set_regression_cols(power="power", poa="poa", t_amb="t_amb", w_vel="w_vel")

        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        meas.regression_results = das_model.fit()
        sim.regression_results = sim_model.fit()

        actual = meas.regression_results.predict(meas.rc)[0] * 1000
        expected = sim.regression_results.predict(meas.rc)[0]
        cp_rat_test_val = actual / expected

        ct = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        ct.meas = meas
        ct.sim = sim

        with self.assertWarns(UserWarning):
            cp_rat = ct.captest_results(check_pvalues=False, print_res=False)

        self.assertAlmostEqual(
            cp_rat, cp_rat_test_val, 6, "captest_results did not return expected value."
        )


class TestCapTestCpResultsMultCoeff(unittest.TestCase):
    """
    Test captest_results function using a regression formula with multiple coef.
    """

    def setUp(self):
        np.random.seed(9876789)

        self.meas = pvc.CapData("meas")
        self.sim = pvc.CapData("sim")
        # self.cptest = pvc.CapTest(meas, sim, '+/- 5')
        self.meas.rc = pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]})

        nsample = 100
        e = np.random.normal(size=nsample)

        a = np.linspace(0, 10, 100)
        b = np.linspace(0, 10, 100) / 2.0
        c = np.linspace(0, 10, 100) + 3.0

        das_y = a + (a**2) + (a * b) + (a * c)
        sim_y = a + (a**2 * 0.9) + (a * b * 1.1) + (a * c * 0.8)

        das_y = das_y + e
        sim_y = sim_y + e

        das_df = pd.DataFrame({"power": das_y, "poa": a, "t_amb": b, "w_vel": c})
        sim_df = pd.DataFrame({"power": sim_y, "poa": a, "t_amb": b, "w_vel": c})

        self.meas.data = das_df
        self.meas.set_regression_cols(
            power="power", poa="poa", t_amb="t_amb", w_vel="w_vel"
        )

        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        das_model = smf.ols(formula=fml, data=das_df)
        sim_model = smf.ols(formula=fml, data=sim_df)

        self.meas.regression_results = das_model.fit()
        self.sim.regression_results = sim_model.fit()

    def test_model_predict_with_modified_params(self):
        """Test that model.predict() with modified params works correctly.

        This test verifies the approach used by predict_with_pvalue_check:
        using model.predict(modified_params, exog) to make predictions with
        modified coefficients, which works consistently across pandas versions.
        """
        das_results = self.meas.regression_results
        sim_results = self.sim.regression_results
        rc = self.meas.rc

        # Get predictions before param modifications
        actual_before = das_results.predict(rc)[0]
        expected_before = sim_results.predict(rc)[0]

        # Create modified params (copy to avoid CoW issues)
        das_params_modified = das_results.params.copy()
        das_params_modified["poa"] = 0

        sim_params_modified = sim_results.params.copy()
        sim_params_modified["poa"] = 0

        # Use model.predict() with custom params
        das_design_info = das_results.model.data.design_info
        sim_design_info = sim_results.model.data.design_info

        das_exog = dmatrix(das_design_info, rc)
        sim_exog = dmatrix(sim_design_info, rc)

        # Predictions with modified params
        actual_after = das_results.model.predict(das_params_modified, das_exog)[0]
        expected_after = sim_results.model.predict(sim_params_modified, sim_exog)[0]

        # Both predictions should change when poa coefficient is zeroed
        self.assertNotAlmostEqual(
            actual_before,
            actual_after,
            3,
            "DAS prediction should change when poa coeff set to 0",
        )
        self.assertNotAlmostEqual(
            expected_before,
            expected_after,
            3,
            "SIM prediction should change when poa coeff set to 0",
        )

    def _build_ct(self):
        """Helper returning a CapTest configured with the setUp meas/sim."""
        ct = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        ct.meas = self.meas
        ct.sim = self.sim
        return ct

    def test_pvals_default_false(self):
        actual = self.meas.regression_results.predict(self.meas.rc)[0]
        expected = self.sim.regression_results.predict(self.meas.rc)[0]
        cp_rat_test_val = actual / expected

        ct = self._build_ct()
        cp_rat = ct.captest_results(check_pvalues=False, print_res=False)

        self.assertEqual(
            cp_rat, cp_rat_test_val, "captest_results did not return expected value."
        )

    def test_pvals_true(self):
        """Test that check_pvalues=True returns different result than False.

        With pval=1e-15, the 'poa' coefficient should be zeroed because its
        p-value is > 1e-15, which changes the prediction.
        """
        ct = self._build_ct()
        # Get ratio without p-value check
        cp_rat_no_check = ct.captest_results(
            check_pvalues=False,
            print_res=False,
        )

        # Get ratio with p-value check (pval=1e-15 zeros poa coefficient)
        cp_rat_with_check = ct.captest_results(
            check_pvalues=True,
            pval=1e-15,
            print_res=False,
        )

        # The ratios should be different because poa coefficient is zeroed
        self.assertNotEqual(
            cp_rat_no_check,
            cp_rat_with_check,
            "check_pvalues=True should produce different result than False",
        )

    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test_pvals_true_print(self):
        """
        Test that captest_results with check_pvalues=True prints expected output.

        This test uses the pytest autouse fixture defined above to
        capture the print to stdout and test it, so it must be run
        using pytest.  Run just this test using 'pytest tests/
        test_CapData.py::TestCapTestCpResultsMultCoeff::test_pvals_true_print'
        """
        self.maxDiff = 10_000

        ct = self._build_ct()
        ct.captest_results(
            check_pvalues=True,
            pval=1e-15,
            print_res=True,
        )
        captured = self.capsys.readouterr()

        # Expected output when poa coefficient is zeroed due to p-value check.
        # CapTest picks reporting conditions from ``rep_cond_source`` (default
        # 'meas'), so the output says "from meas." instead of the legacy
        # module-level "from das.".
        results_str = (
            "Using reporting conditions from meas. \n\n"
            "Capacity Test Result:    FAIL\n"
            "Modeled test output:          66.451\n"
            "Actual test output:           72.429\n"
            "Tested output ratio:          1.090\n"
            "Tested Capacity:              108.996\n"
            "Bounds:                       95.0, 105.0\n\n\n"
        )

        self.assertEqual(results_str, captured.out)

    def test_formulas_match(self):
        sim = pvc.CapData("sim")
        das = pvc.CapData("das")

        sim.regression_formula = "power ~ poa + I(poa * poa) + I(poa * t_amb) - 1"

        ct = CapTest(test_tolerance="+/- 5", ac_nameplate=100)
        ct.meas = das
        ct.sim = sim

        with self.assertWarns(UserWarning):
            ct.captest_results(check_pvalues=True)


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
        assert table_flt_all_removed.equals(
            flt0_removed_ix.union(flt1_removed_ix).union(flt2_removed_ix)
        )

    def test_zero_removal_step_gets_no_column(self, nrel):
        nrel.filter_irr(200, 900)
        nrel.filter_irr(400, 800)
        nrel.rep_cond()  # RepCond: zero-removal -> no column
        flt_table = nrel.get_filtering_table()
        # Pin the column-per-removing-step contract by label and order: one
        # column per removing filter (named via _step_labels) then all_filters;
        # the zero-removal RepCond step gets no column.
        assert list(flt_table.columns) == ["FilterIrr", "FilterIrr-1", "all_filters"]


@pytest.fixture
def pts_summary(meas):
    pts_summary = pvc.PointsSummary(meas)
    return pts_summary


class TestPointsSummary:
    def test_length_test_period_no_filter(self, meas):
        meas.get_length_test_period()
        assert meas.length_test_period == 5

    def test_length_test_period_after_one_filter_time(self, meas):
        meas.filter_time(start="10/9/1990", end="10/12/1990 23:00")
        meas.get_length_test_period()
        assert meas.length_test_period == 4

    def test_length_test_period_after_two_filter_time(self, meas):
        meas.filter_time(start="10/9/1990", end="10/12/1990 23:00")
        meas.filter_time(start="10/9/1990", end="10/11/1990 23:00")
        meas.get_length_test_period()
        assert meas.length_test_period == 4

    def test_length_test_period_custom_name_filter_time(self, meas):
        # A custom_name'd FilterTime must still be found: the period comes from
        # isinstance(step, FilterTime), not a label-string match.
        filters.FilterTime(
            start="10/9/1990", end="10/12/1990 23:00", custom_name="window"
        ).run(meas)
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
            "length of test period to date: 5 days\n"
            "sufficient points have been collected. 150.0 points required; "
            "1440 points collected\n"
        )
        assert results_str == captured.out

    def test_print_points_summary_fail(self, meas):
        meas.filter_custom(pd.DataFrame.head, 10)
        meas.print_points_summary()
        captured = self.capsys.readouterr()

        results_str = (
            "length of test period to date: 5 days\n"
            "10 points of 150.0 points needed, 140.0 remaining to collect.\n"
            "2.00 points / day on average.\n"
            "Approximate days remaining: 71\n"
        )

        assert results_str == captured.out


class TestDataColumnsToExcel:
    """
    Test the `data_columns_to_excel` method of the `CapData` class.
    """

    def test_data_columns_to_excel_path_is_dir(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader("./tests/data/")
        meas.data_columns_to_excel(sort_by_reversed_names=True)
        xlsx_file = meas.data_loader.path / "column_groups.xlsx"
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == "met1_mod_temp1"
        os.remove(xlsx_file)

    def test_data_columns_to_excel_path_is_file(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader("./tests/data/example_measured_data.csv")
        meas.data_columns_to_excel(sort_by_reversed_names=True)
        xlsx_file = meas.data_loader.path.parent / "column_groups.xlsx"
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == "met1_mod_temp1"
        os.remove(xlsx_file)

    def test_data_columns_to_excel_not_reverse_sorted(self, meas):
        """
        Test that the `data_columns_to_excel` method of the `CapData` class
        saves an excel file with a blank first column and the second column is the
        column names of the `data` attribute.
        """
        meas.data_loader = io.DataLoader("./tests/data/")
        meas.data_columns_to_excel(sort_by_reversed_names=False)
        xlsx_file = meas.data_loader.path / "column_groups.xlsx"
        assert xlsx_file.is_file()
        df = pd.read_excel(xlsx_file, header=None)
        assert df.iloc[0, 1] == "inv1_power"
        os.remove(xlsx_file)


class TestScatterHv:
    """Test scatter_hv method of CapData class."""

    def test_no_index_str_column_in_data(self, meas):
        "Check that plot function works when there is no index column in the data."
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        assert "index" not in meas.data.columns
        plot = meas.scatter_hv()
        assert isinstance(plot, hv.element.chart.Scatter)

    def test_curve_timeseries(self, meas):
        """Test that the curve_timeseries method works. Shouldn't require `data` index
        to have a specific name.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        assert "index" not in meas.data.columns
        assert meas.data.index.name is None
        plot = meas.scatter_hv(timeseries=True)
        assert isinstance(plot, hv.core.layout.Layout)


class TestScatterFilters:
    """Test the scatter_filters method of the CapData class."""

    def test_returns_overlay(self, meas):
        """
        Test that the scatter_filters method of the CapData class returns a
        holoviews overlay object.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.scatter_filters()
        assert "index" not in meas.data.columns
        assert "index" not in meas.data_filtered.columns
        assert isinstance(overlay, hv.core.overlay.Overlay)

    def test_layer_count_is_retained_plus_removing_filters(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.scatter_filters()
        # 1 retained baseline + 2 removing filters
        assert len(overlay) == 3

    def test_zero_removal_step_adds_no_layer(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.rep_cond()  # RepCond: zero-removal -> no layer
        overlay = meas.scatter_filters()
        # 1 retained baseline + 1 removing filter; RepCond contributes nothing
        assert len(overlay) == 2

    def test_layers_carry_the_right_rows(self, meas):
        """Pin the row-selection glue: the retained baseline holds the survivors
        and each removed layer holds exactly that filter's removed rows (a
        retained/removed swap would still pass the count assertions above)."""
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.scatter_filters()
        # Ordered leaf elements: retained baseline first, then one per removing
        # filter. The Scatter's backing frame carries an "index" column set to
        # the original data index, so we can check which rows landed in a layer.
        layers = list(overlay)
        assert list(layers[0].data["index"]) == list(meas.filters[-1].ix_after)
        _i, _label, removed_ix = meas._removed_by_step()[0]
        assert list(layers[1].data["index"]) == list(removed_ix)


class TestTimeseriesFilters:
    """Test the timeseries_filters method of the CapData class."""

    def test_returns_overlay(self, meas):
        """
        Test that the timeseries_filters method of the CapData class returns a
        holoviews overlay object.
        """
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.timeseries_filters()
        assert "index" not in meas.data.columns
        assert "index" not in meas.data_filtered.columns
        assert isinstance(overlay, hv.core.overlay.Overlay)

    def test_layer_count_is_curve_plus_removing_filters(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.timeseries_filters()
        # 1 full-data curve + 2 removing-filter scatters
        assert len(overlay) == 3

    def test_removed_layer_carries_the_right_rows(self, meas):
        """Pin that a removed-filter scatter layer holds exactly that filter's
        removed rows (the full-data Curve baseline is layer 0)."""
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.timeseries_filters()
        layers = list(overlay)  # curve baseline first, then one scatter per remover
        _i, _label, removed_ix = meas._removed_by_step()[0]
        assert list(layers[1].data["Timestamp"]) == list(removed_ix)


class TestPlotDashboard:
    def test_plot(self, meas):
        """Check types of returned dashboard and tabs."""
        dboard = meas.plot()
        assert isinstance(dboard, pn.layout.tabs.Tabs)
        assert isinstance(dboard[0], pn.pane.holoviews.HoloViews)
        assert isinstance(dboard[1], pn.layout.base.Column)
        assert isinstance(dboard[2], pn.layout.base.Column)

    def test_plot_defaults_path_with_valid_columns(self, meas, tmp_path, recwarn):
        """Defaults file with valid columns is loaded without column warnings."""
        some_columns = list(meas.data.columns[:2])
        defaults_file = tmp_path / "defaults.json"
        defaults_file.write_text(json.dumps([some_columns]))
        dboard = meas.plot(plot_defaults_path=defaults_file)
        column_warnings = [
            w
            for w in recwarn.list
            if "not found in the data" in str(w.message)
            or "No valid columns" in str(w.message)
        ]
        assert len(column_warnings) == 0
        assert isinstance(dboard, pn.layout.tabs.Tabs)

    def test_plot_defaults_path_warns_on_missing_columns(self, meas, tmp_path):
        """Defaults file with some missing columns emits a warning, plots valid ones."""
        valid_col = list(meas.data.columns[:1])
        defaults_file = tmp_path / "defaults.json"
        defaults_file.write_text(json.dumps([valid_col + ["nonexistent_column"]]))
        with pytest.warns(UserWarning, match="nonexistent_column"):
            dboard = meas.plot(plot_defaults_path=defaults_file)
        assert isinstance(dboard, pn.layout.tabs.Tabs)

    def test_plot_defaults_path_all_missing_falls_back_to_default_groups(
        self, meas, tmp_path
    ):
        """Defaults file with only missing columns warns and falls back to default groups."""
        defaults_file = tmp_path / "defaults.json"
        defaults_file.write_text(
            json.dumps([["nonexistent_col_1", "nonexistent_col_2"]])
        )
        with pytest.warns(UserWarning):
            dboard = meas.plot(plot_defaults_path=defaults_file)
        assert isinstance(dboard, pn.layout.tabs.Tabs)

    def test_per_capdata_defaults_file_uses_cd_name(self, meas, tmp_path, monkeypatch):
        """CapData.plot() reads from plot_defaults_{name}.json in the CWD."""
        monkeypatch.chdir(tmp_path)
        some_columns = list(meas.data.columns[:2])
        (tmp_path / "plot_defaults_meas.json").write_text(json.dumps([some_columns]))
        dboard = meas.plot()
        assert isinstance(dboard, pn.layout.tabs.Tabs)


class TestCreateColumnGroupAttributes:
    """
    Test the create_column_group_attributes method of the CapData class.

    Checks the following:
    - an attribute is created for each key of self.column_groups
    - the attributes return the correct view of the data
    """

    def test_column_group_attributes(self, meas):
        """Test that column group attributes are created and return correct data."""
        # Create the column group attributes
        meas.create_column_group_attributes()

        # Check that an attribute exists for each key in column_groups
        for group_key in meas.column_groups.keys():
            assert hasattr(meas, group_key), f"Attribute {group_key} not created"

            # Get the data view using the attribute
            attr_data = getattr(meas, group_key)
            # Get the expected data using loc indexer
            expected_data = meas.loc[group_key]

            # Check that the attribute returns the correct data
            pd.testing.assert_frame_equal(attr_data, expected_data)


class TestProcessRegressionColumns:
    def test_e_total_reg_cols(self, meas):
        meas.regression_cols = {
            "e_total": (
                calcparams.e_total,
                {
                    "poa": ("irr_poa_pyran", "mean"),
                    "rpoa": ("irr_poa_pyran", "mean"),
                },
            )
        }
        meas.process_regression_columns()
        assert "e_total" in meas.data.columns
        assert "e_total" in meas.data_filtered.columns
        assert meas.regression_cols == {"e_total": "e_total"}

    def test_agg_amb_temp(self, meas):
        meas.regression_cols = {"temp_amb": ("temp_amb", "mean")}
        meas.process_regression_columns()
        assert "temp_amb_mean_agg" in meas.data.columns
        assert "temp_amb_mean_agg" in meas.data_filtered.columns
        assert meas.regression_cols == {"temp_amb": "temp_amb_mean_agg"}

    def test_power_tc_from_amb_temp(self, meas):
        meas.power_temp_coeff = -0.32
        meas.regression_cols = {
            "power_tc": (
                calcparams.power_temp_correct,
                {
                    "power": "meter_power",
                    "cell_temp": (
                        calcparams.cell_temp,
                        {
                            "bom": (
                                calcparams.bom_temp,
                                {
                                    "poa": ("irr_poa_pyran", "mean"),
                                    "temp_amb": ("temp_amb", "mean"),
                                    "wind_speed": ("wind", "mean"),
                                },
                            ),
                            "poa": ("irr_poa_pyran", "mean"),
                        },
                    ),
                },
            )
        }
        meas.process_regression_columns()
        assert "power_temp_correct" in meas.data.columns
        assert "power_temp_correct" in meas.data_filtered.columns
        assert meas.regression_cols == {"power_tc": "power_temp_correct"}

    def test_col_grp_id_conflict(self, meas):
        """Expect ValueError when a kwarg name conflicts with a column-group ID.

        Ommitted the temp_amb kwarg to calcparams.bom_temp here, which is the
        overlapping kwarg and column group id. The column grouping used in this
        test does not include a 'wind_speed' group, it is just 'wind'.
        """
        meas.regression_cols = {
            "bom": (
                calcparams.bom_temp,
                {
                    "poa": ("irr_poa_pyran", "mean"),
                    "wind_speed": ("wind", "mean"),
                },
            )
        }
        with pytest.raises(
            ValueError,
            match=r"kwarg temp_amb.*bom_temp.*column groups id.*Change the name of.*",
        ):
            meas.process_regression_columns()

    def test_pass_kwarg_value_in_regression_columns(self, meas):
        """Check that a kwarg for a calcparams function passes through correctly"""
        meas.regression_cols = {
            "power_tc": (
                calcparams.power_temp_correct,
                {
                    "power": "meter_power",
                    "cell_temp": ("temp_amb", "mean"),
                    "power_temp_coeff": -0.38,
                },
            )
        }
        meas.process_regression_columns()
        assert "power_temp_correct" in meas.data.columns
        assert "power_temp_correct" in meas.data_filtered.columns
        assert meas.regression_cols == {"power_tc": "power_temp_correct"}


if __name__ == "__main__":
    unittest.main()
