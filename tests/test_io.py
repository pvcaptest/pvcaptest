import os
import shutil
from io import StringIO
import collections
import unittest
import pytest
import warnings
import pytz
from pathlib import Path
import numpy as np
import pandas as pd

from captest.io import file_reader

from .context import capdata as pvc
from .context import util
from .context import columngroups as cg
from .context import io
from .context import (
    load_pvsyst,
    load_data,
    DataLoader,
)


class TestFlattenMultiIndex:
    def test_flatten_multi_index_2levels(self):
        m_ix = pd.MultiIndex.from_tuples(
            [
                ("met1", "poa1"),
                ("met1", "ghi1"),
            ]
        )
        print(m_ix)
        flattened_ix = io.flatten_multi_index(m_ix)
        assert flattened_ix == ["met1_poa1", "met1_ghi1"]

    def test_flatten_multi_index_3levels(self):
        m_ix = pd.MultiIndex.from_tuples(
            [
                ("met1", "kipp and zonnen", "poa1"),
                ("met1", "kipp and zonnen", "ghi1"),
            ]
        )
        flattened_ix = io.flatten_multi_index(m_ix)
        assert flattened_ix == [
            "met1_kipp and zonnen_poa1",
            "met1_kipp and zonnen_ghi1",
        ]


class TestFileReader:
    """
    Mock different files to test file reading.
    """

    def test_well_formatted_file(self, tmp_path):
        """
        Test loading a well formatted csv with headers on the first row.

        No blank rows between column headings and the beginning of the data.
        """
        csv_path = tmp_path / "simple_data.csv"
        pd.DataFrame(
            {
                "met1_poa1": np.arange(0, 20),
                "met1_poa2": np.arange(20, 40),
            },
            index=pd.date_range(start="8/1/22", periods=20, freq="1min"),
        ).to_csv(csv_path)
        loaded_data = io.file_reader(csv_path)
        assert isinstance(loaded_data, pd.DataFrame)

    def test_double_headers(self, tmp_path):
        """
        Test loading a well formatted csv with two rows of headers.

        Two rows of headers that should be concatenated in the loaded data.
        No blank rows between the last header row and the data.
        """
        csv_path = tmp_path / "double_headers.csv"
        pd.DataFrame(
            np.column_stack((np.arange(0, 20), np.arange(20, 40))),
            index=pd.date_range(start="8/1/22", periods=20, freq="1min"),
            columns=pd.MultiIndex.from_tuples([("met1", "poa"), ("met2", "poa")]),
        ).to_csv(csv_path)
        loaded_data = io.file_reader(csv_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.columns[0] == "met1_poa"
        assert isinstance(loaded_data.index, pd.DatetimeIndex)

    def test_empty_rows_at_start(self, tmp_path):
        """
        Test loading a well csv with empty rows between headers and data.

        Single row of headers and a single blank row between header and data.
        """
        test_csv = StringIO()
        csv_path = tmp_path / "empty_rows.csv"
        pd.DataFrame(
            {
                "met1_poa1": np.arange(0, 20),
                "met1_poa2": np.arange(20, 40),
            },
            index=pd.date_range(start="8/1/22", periods=20, freq="1min"),
        ).to_csv(test_csv)
        test_csv.seek(0)
        df_str = test_csv.getvalue()
        df_with_blank_row = df_str[0:21] + ",,\n" + df_str[21:]
        with open(csv_path, "w") as f:
            f.write(df_with_blank_row)
        loaded_data = io.file_reader(csv_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.columns[0] == "met1_poa1"
        assert isinstance(loaded_data.index, pd.DatetimeIndex)

    def test_double_headers_with_blank(self, tmp_path):
        """Two header rows followed by a blank line."""
        test_csv = StringIO()
        csv_path = tmp_path / "double_headers_with_blank.csv"
        pd.DataFrame(
            np.column_stack((np.arange(0, 20), np.arange(20, 40))),
            index=pd.date_range(start="8/1/22", periods=20, freq="1min"),
            columns=pd.MultiIndex.from_tuples([("met1", "poa"), ("met2", "poa")]),
        ).to_csv(test_csv)
        test_csv.seek(0)
        df_str = test_csv.getvalue()
        df_with_blank_row = df_str[0:20] + ",,\n" + df_str[20:]
        with open(csv_path, "w") as f:
            f.write(df_with_blank_row)
        loaded_data = io.file_reader(csv_path)
        print(loaded_data)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.columns[0] == "met1_poa"
        assert isinstance(loaded_data.index, pd.DatetimeIndex)

    def test_ae_headers(self):
        """Four rows of headers, no blank rows, header in first column on third row."""
        loaded_data = io.file_reader("./tests/data/example_meas_data_aeheaders.csv")
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.columns[0] == (
            "Example Project_Weather Station 1 (Standard w/ POA GHI)_Weather Station 1 (Standard w/ POA GHI), Sun_W/m^2"
        )
        assert isinstance(loaded_data.index, pd.DatetimeIndex)

    def test_load_das(self):
        das = io.file_reader("./tests/data/example_meas_data.csv")
        assert 1440 == das.shape[0]
        assert isinstance(das.index, pd.DatetimeIndex)
        assert isinstance(das.columns, pd.Index)


class TestLoadPVsyst:
    def test_load_pvsyst(self):
        pvsyst = load_pvsyst("./tests/data/pvsyst_example_HourlyRes_2.CSV")
        assert isinstance(pvsyst, pvc.CapData)
        assert 8760 == pvsyst.data.shape[0]
        assert isinstance(pvsyst.data.index, pd.DatetimeIndex)
        assert isinstance(pvsyst.data.columns, pd.Index)
        assert pvsyst.data.loc["1/1/90 12:00", "E_Grid"] == 5_469_083
        assert pvsyst.regression_cols == {
            "power": "E_Grid",
            "poa": "GlobInc",
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        }

    def test_scale_egrid(self):
        pvsyst = load_pvsyst(
            "./tests/data/pvsyst_example_HourlyRes_2.CSV", egrid_unit_adj_factor=1_000
        )
        assert pvsyst.data.loc["1/1/90 12:00", "E_Grid"] == 5_469.083

    def test_dont_set_reg_cols(self):
        pvsyst = load_pvsyst(
            "./tests/data/pvsyst_example_HourlyRes_2.CSV",
            set_regression_columns=False,
        )
        assert pvsyst.regression_cols == {}


class TestDataLoader:
    """
    Tests of the data loader class.
    """

    def test_default_data_path(self):
        dl = DataLoader()
        assert isinstance(dl.path, Path)
        assert dl.path == Path("./data/")

    def test_user_path(self):
        dl = DataLoader("./data/data_for_yyyy-mm-dd.csv")
        assert isinstance(dl.path, Path)
        assert dl.path == Path("./data/data_for_yyyy-mm-dd.csv")

    def test_reindex_loaded_files(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"a": np.arange(24 * 6)},
            index=pd.date_range(start="1/1/22", freq="5 min", periods=24 * 6),
        )
        day3 = day1.copy()
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
            "day3": day3,
        }
        reix_dfs, common_freq, file_frequencies = dl._reindex_loaded_files()
        assert common_freq == "60min"
        assert file_frequencies == ["60min", "5min", "60min"]

    def test_join_files_same_headers(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24), "b": np.arange(24, 48, 1)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"a": np.arange(24, 48, 1), "b": np.arange(24)},
            index=pd.date_range(start="1/2/22", freq="60 min", periods=24),
        )
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
        }
        dl.common_freq = "60min"
        data = dl._join_files()
        print(data)
        assert data.shape == (48, 2)
        assert data.index.is_monotonic_increasing

    def test_join_files_same_headers_same_index_warning(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24), "b": np.arange(24, 48, 1)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"a": np.arange(24, 48, 1), "b": np.arange(24)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
        }
        dl.common_freq = "60min"
        with pytest.warns(UserWarning):
            data = dl._join_files()
            warnings.warn("Some columns contain overlapping indices.", UserWarning)
        assert data.shape == (48, 2)

    def test_join_files_different_headers(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24), "b": np.arange(24, 48, 1)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"c": np.arange(24, 48, 1), "d": np.arange(24)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
        }
        dl.common_freq = "60min"
        data = dl._join_files()
        assert data.shape == (24, 4)
        assert data.isna().sum().sum() == 0

    def test_join_files_different_headers_and_days(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24), "b": np.arange(24, 48, 1)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"c": np.arange(24, 48, 1), "d": np.arange(24)},
            index=pd.date_range(start="1/2/22", freq="60 min", periods=24),
        )
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
        }
        dl.common_freq = "60min"
        data = dl._join_files()
        assert data.shape == (48, 4)
        assert data["1/2/22"][["a", "b"]].isna().sum().sum() == 48
        assert data["1/1/22"][["c", "d"]].isna().sum().sum() == 48
        assert data.index.is_monotonic_increasing

    def test_join_files_overlapping_headers(self):
        day1 = pd.DataFrame(
            {"a": np.arange(24), "b": np.arange(24, 48, 1)},
            index=pd.date_range(start="1/1/22", freq="60 min", periods=24),
        )
        day2 = pd.DataFrame(
            {"b": np.arange(24, 48, 1), "c": np.arange(24)},
            index=pd.date_range(start="1/2/22", freq="60 min", periods=24),
        )
        dl = io.DataLoader()
        dl.loaded_files = {
            "day1": day1,
            "day2": day2,
        }
        dl.common_freq = "60min"
        data = dl._join_files()
        assert data.shape == (48, 3)
        assert data.index[0] == pd.to_datetime("1/1/22")
        assert data.index[-1] == pd.to_datetime("1/2/22 23:00")
        assert all(data["1/1/22"]["c"].isna())
        assert all(data["1/2/22"]["a"].isna())
        assert data.index.is_monotonic_increasing

    def test_load_single_file(self, tmp_path):
        csv_path = tmp_path / "single_file.csv"
        pd.DataFrame(
            {
                "met1_poa1": np.arange(0, 20),
                "met1_poa2": np.arange(20, 40),
            },
            index=pd.date_range(start="8/1/22", periods=20, freq="1min"),
        ).to_csv(csv_path)
        dl = DataLoader(csv_path)
        cd = dl.load()
        assert isinstance(cd.data, pd.DataFrame)

class TestLoadDataMethods(unittest.TestCase):
    """Test for load data methods without setup."""

    def test_source_alsoenergy(self):
        das_1 = load_data(path="./tests/data/col_naming_examples/ae_site1.csv")
        col_names1 = [
            "ae_site_1_Elkor Production Meter_Elkor Production Meter, PowerFactor_Unnamed: 1_level_3",
            "ae_site_1_Elkor Production Meter_Elkor Production Meter, KW_kW",
            "ae_site_1_Weather Station 1 (Standard w/ Kipp POA)_Weather Station 1 (Standard w/ Kipp POA), TempF_°F",
            "ae_site_1_Weather Station 2 (Kipp GHI)_Weather Station 2 (Kipp GHI), Sun2_W/m²",
            "ae_site_1_Weather Station 1 (Standard w/ Kipp POA)_Weather Station 1 (Standard w/ Kipp POA), Sun_W/m²",
            "ae_site_1_Weather Station 1 (Standard w/ Kipp POA)_Weather Station 1 (Standard w/ Kipp POA), WindSpeed_mph",
            "index",
        ]
        self.assertTrue(
            all(das_1.data.columns == col_names1),
            "Column names are not expected value for ae_site1",
        )

        das_2 = load_data(path="./tests/data/col_naming_examples/ae_site2.csv")
        print(das_2.data.columns)
        col_names2 = [
            "ae_site_2_Acuvim II Meter (Customer)_Acuvim II Meter (Customer), PowerFactor_PF",
            "ae_site_2_Acuvim II Meter (Customer)_Acuvim II Meter (Customer), KW_kW",
            "ae_site_2_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA)_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA), TempF_°F",
            "ae_site_2_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA)_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA), TempF_°F",
            "ae_site_2_Weather Station 2 (Pad 1) (Hukse GHI)_Weather Station 2 (Pad 1) (Hukse GHI), Sun2_W/m²",
            "ae_site_2_Weather Station 4 (Pad 3 - Main) (Hukse GHI)_Weather Station 4 (Pad 3 - Main) (Hukse GHI), Sun2_W/m²",
            "ae_site_2_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA)_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA), Sun_W/m²",
            "ae_site_2_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA)_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA), Sun_W/m²",
            "ae_site_2_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA)_Weather Station 1 (Pad 1) (Standard w/o Mod, w/ Hukse POA), WindSpeed_mph",
            "ae_site_2_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA)_Weather Station 3 (Pad 3 - Main) (Standard w/o Mod, w/ Hukse POA), WindSpeed_mph",
            "index",
        ]
        self.assertTrue(
            all(das_2.data.columns == col_names2),
            "Column names are not expected value for ae_site2",
        )


test_files = [
    "test1.csv",
    "test2.csv",
    "test3.CSV",
    "test4.txt",
    "pvsyst.csv",
    "pvsyst_data.csv",
]


# class TestCapDataLoadMethods(unittest.TestCase):
#     """Tests for load_data method."""
#
#     def setUp(self):
#         os.mkdir("test_csvs")
#         for fname in test_files:
#             with open("test_csvs/" + fname, "a") as f:
#                 f.write("Date, val\n11/21/2017, 1")
#
#         self.capdata = load_data(path="test_csvs/")
#
#     def tearDown(self):
#         for fname in test_files:
#             os.remove("test_csvs/" + fname)
#         os.rmdir("test_csvs")
#
#     def test_read_csvs(self):
#         self.assertEqual(
#             self.capdata.data.shape[0], 3, "imported a non csv or pvsyst file"
#         )
