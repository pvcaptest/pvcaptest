# this file is formatted with black
from distutils.log import warn
import os
import dateutil
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from captest.capdata import CapData
from captest.capdata import csky
from captest import columngroups as cg
from captest import util


def flatten_multi_index(columns):
    return ["_".join(col_name) for col_name in columns.to_list()]


def load_pvsyst(path, filename, **kwargs):
    """
    Load data from a PVsyst energy production model.

    Parameters
    ----------
    path : str
        Path to file to import.
    filename : str
        Name of file to import.
    **kwargs
        Use to pass additional kwargs to pandas read_csv.

    Returns
    -------
    pandas dataframe
    """
    dirName = os.path.normpath(path + filename)

    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for encoding in encodings:
        try:
            # pvraw = pd.read_csv(dirName, skiprows=10, encoding=encoding,
            #                     header=[0, 1], parse_dates=[0],
            #                     infer_datetime_format=True, **kwargs)
            pvraw = pd.read_csv(
                dirName, skiprows=10, encoding=encoding, header=[0, 1], **kwargs
            )
        except UnicodeDecodeError:
            continue
        else:
            break

    pvraw.columns = pvraw.columns.droplevel(1)
    dates = pvraw.loc[:, "date"]
    try:
        dt_index = pd.to_datetime(dates, format="%m/%d/%y %H:%M")
    except ValueError:
        dt_index = pd.to_datetime(dates)
    pvraw.index = dt_index
    pvraw.drop("date", axis=1, inplace=True)
    pvraw = pvraw.rename(columns={"T Amb": "TAmb"})
    return pvraw


def file_reader(path, **kwargs):
    """
    Read measured solar data from a csv file.

    Utilizes pandas read_csv to import measure solar data from a csv file.
    Attempts a few diferent encodings, trys to determine the header end
    by looking for a date in the first column, and concantenates column
    headings to a single string.

    Parameters
    ----------
    path : Path
        Path to file to import.
    **kwargs
        Use to pass additional kwargs to pandas read_csv.

    Returns
    -------
    pandas DataFrame
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for encoding in encodings:
        try:
            data_file = pd.read_csv(
                path,
                encoding=encoding,
                index_col=0,
                parse_dates=True,
                skip_blank_lines=True,
                low_memory=False,
                **kwargs,
            )
        except UnicodeDecodeError:
            continue
        else:
            break
    data_file.dropna(how='all', axis=0, inplace=True)
    if not isinstance(data_file.index[0], pd.Timestamp):
        for i, _indice in enumerate(data_file.index):
            try:
                isinstance(
                    dateutil.parser.parse(str(data_file.index[i])), datetime.date
                )
                header_end = i + 1
                break
            except ValueError:
                continue
        header = list(np.arange(header_end))
        data_file = pd.read_csv(
            path,
            encoding=encoding,
            header=header,
            index_col=0,
            parse_dates=True,
            skip_blank_lines=True,
            low_memory=False,
            **kwargs,
        )

    data_file = data_file.apply(pd.to_numeric, errors="coerce")
    if isinstance(data_file.columns, pd.MultiIndex):
        data_file.columns = flatten_multi_index(data_file.columns)
    data_file = data_file.rename(columns=(lambda x: x.strip()))
    return data_file




@dataclass
class DataLoader:
    """
    Class to load SCADA data and return a CapData object.
    """

    path: str = "./data/"
    loc: Optional[dict] = field(default=None)
    sys: Optional[dict] = field(default=None)
    group_columns: object = cg.group_columns
    file_reader: object = file_reader
    name: str = "meas"
    files_to_load: Optional[list] = field(default=None)

    def __setattr__(self, key, value):
        if key == "path":
            value = Path(value)
        super().__setattr__(key, value)

    def set_files_to_load(self, extension="csv"):
        """
        Set `files_to_load` attribute to a list of filepaths.
        """
        self.files_to_load = [file for file in self.path.glob("*." + extension)]
        if len(self.files_to_load) == 0:
            return warnings.warn(
                "No files with .{} extension were found in the directory: {}".format(
                    extension,
                    self.path,
                )
            )

    def _reindex_loaded_files(self):
        """Reindex files to ensure no missing indices and find frequency for each file.

        Returns
        -------
        reindexed_dfs : dict
            Filenames mapped to reindexed DataFrames.
        common_freq : str
            The index frequency most common across the reindexed DataFrames.
        file_frequencies : list
            The index frequencies for each file.
        """
        reindexed_dfs = {}
        file_frequencies = []
        for name, file in self.loaded_files.items():
            current_file, missing_intervals, freq_str = util.reindex_datetime(
                file,
                report=False,
                add_index_col=True,
            )
            reindexed_dfs[name] = current_file
            file_frequencies.append(freq_str)

        unique_freq = np.unique(
            np.array([freq for freq in file_frequencies]),
            return_counts=True,
        )
        common_freq = unique_freq[0][np.argmax(unique_freq[1])]

        return reindexed_dfs, common_freq, file_frequencies

    def _join_files(self):
        """Combine the DataFrames of `loaded_files` into a single DataFrame.

        Checks if the columns of each DataFrame in `loaded_files` matches. If they do
        all match, then they will be combined along vertically along the index.

        If they do not match, then they will be combined by creating a datetime index
        that begins with the earliest datetime in all the indices to the latest datetime
        in all the indices using the most common frequency across all the indices. The
        columns will be a set of all the columns.

        Returns
        -------
        data : DataFrame
            The combined data.
        """
        all_columns = [df.columns for df in self.loaded_files.values()]
        columns_match = all(
            [pair[0].equals(pair[1]) for pair in combinations(all_columns, 2)]
        )
        all_indices = [df.index for df in self.loaded_files.values()]
        indices_match = all(
            [pair[0].equals(pair[1]) for pair in combinations(all_indices, 2)]
        )
        if columns_match and not indices_match:
            data = pd.concat(self.loaded_files.values(), axis='index')
        elif columns_match and indices_match:
            warnings.warn('Some columns contain overlapping indices.')
            data = pd.concat(self.loaded_files.values(), axis='index')
        else:
            joined_columns = pd.Index(
                set([item for cols in all_columns for item in cols])
            ).sort_values()
            data = pd.DataFrame(
                index=pd.date_range(
                    start=min([df.index.min() for df in self.loaded_files.values()]),
                    end=max([df.index.max() for df in self.loaded_files.values()]),
                    freq=self.common_freq,
                ),
                columns=joined_columns,
            )
            for file in self.loaded_files.values():
                data.loc[file.index, file.columns] = file.values
        return data

    def load(self, sort=True, drop_duplicates=True, reindex=True, extension="csv"):
        """
        Load file(s) of timeseries data from SCADA / DAS systems.

        This is a convience function to generate an instance of DataLoader
        and call the `load` method.

        A single file or multiple files can be loaded. Multiple files will be joined together
        and may include files with different column headings.

        Parameters
        ----------
        sort : bool, default True
            By default sorts the data by the datetime index from old to new.
        drop_duplicates : bool, default True
            By default drops rows of the joined data where all the columns are duplicats
            of another row. Keeps the first instance of the duplicated values. This is
            helpful if individual datafiles have overlaping rows with the same data.
        reindex : bool, default True
            By default will create a new index for the data using the earliest datetime,
            latest datetime, and the most frequent time interval ensuring there are no
            missing intervals.
        extension : str, default "csv"
            Change the extension to allow loading different filetypes. Must also set
            the `file_reader` attribute to a function that will read that type of file.
        """
        if self.path.is_file():
            data = self.file_reader(self.path)
        elif self.path.is_dir():
            if self.files_to_load is not None:
                self.loaded_files = {
                    file.stem: self.file_reader(file) for file in self.files_to_load
                }
            else:
                self.set_files_to_load(extension=extension)
                self.loaded_files = {
                    file.stem: self.file_reader(file) for file in self.files_to_load
                }
            (
                self.loaded_files,
                self.common_freq,
                self.file_frequencies,
            ) = self._reindex_loaded_files()
            data = join_files()

        # try:
        cd = CapData(self.name)
        data.index.name = "Timestamp"
        cd.data = data.copy()

        if sort:
            cd.data.sort_index(inplace=True)
        if drop_duplicates:
            cd.data.drop_duplicates(inplace=True)
        if reindex:
            if not self.path.is_dir():
                cd.data, missing_intervals, freq_str = util.reindex_datetime(
                    cd.data,
                    report=False,
                )
                self.missing_intervals = missing_intervals
                self.freq_str = freq_str

        cd.data_loader = self
        cd.data_filtered = cd.data.copy()

        # group columns
        if callable(self.group_columns):
            cd.column_groups = self.group_columns(cd.data)
        elif isinstance(self.group_columns, str):
            p = Path(self.group_columns)
            if p.suffix == ".json":
                cd.column_groups = cg.ColumnGroups(
                    util.read_json(self.group_columns)
                )

        cd.trans_keys = list(cd.column_groups.keys())
        return cd
        # except:
        #     print(type(cd.data.index))
        #     print(cd.data.columns[0:5])
        #     print(cd.data.head())
        #     cd.data_loader = self
        #     return cd
        #     raise


def load_data(
    path,
    group_columns=cg.group_columns,
    file_reader=file_reader,
    name="meas",
    **kwargs,
):
    """
    Load file(s) of timeseries data from SCADA / DAS systems.

    This is a convience function to generate an instance of DataLoader
    and call the `load` method.

    A single file or multiple files can be loaded. Multiple files will be joined together
    and may include files with different column headings.

    Parameters
    ----------
    path : str
        Path to either a single file to load or a directory of files to load.
    group_columns : function or str, default columngroups.group_columns
        Function to use to group the columns of the loaded data. Function should accept
        a DataFrame and return a dictionary with keys that are ids and valeus that are
        lists of column names. Will be set to the `group_columns` attribute of the
        CapData.DataLoader object.
        Provide a string to load column grouping from a json file.
    file_reader : function, default io.file_reader
        Function to use to load an individual file. By default will use the built in
        `file_reader` function to try to load csv files. If passing a function to read
        other filetypes, the kwargs should include the filetype extension e.g. 'parquet'.
    name : str
        Identifier that will be assigned to the returned CapData instance.
    **kwargs
        Passed to `DataLoader.load` Options include: sort, drop_duplicates, reindex,
        extension. See `DataLoader` for complete documentation.
    """
    dl = DataLoader(
        path=path,
        group_columns=group_columns,
        file_reader=file_reader,
        name=name,
    )
    cd = dl.load(**kwargs)
    return cd



#     loc=None,
#     sys=None,
#     """
#     Import data from csv files.
#
#     The intent of the default behavior is to combine csv files that have
#     the same columns and rows of data from different times. For example,
#     combining daily files of 5 minute measurements from the same sensors
#     for each day.
#
#     Use the path and fname arguments to specify a single file to import.
#
#     Parameters
#     ----------
#     loc : dict
#         See the csky function for details on dictionary options.
#     sys : dict
#         See the csky function for details on dictionary options.
#   """
#         if clear_sky:
#             if loc is None:
#                 warnings.warn(
#                     "Must provide loc and sys dictionary\
#                               when clear_sky is True.  Loc dict missing."
#                 )
#             if sys is None:
#                 warnings.warn(
#                     "Must provide loc and sys dictionary\
#                               when clear_sky is True.  Sys dict missing."
#                 )
#             cd.data = csky(cd.data, loc=loc, sys=sys, concat=True, output="both")
#
#     if callable(group_columns):
#         cd.column_groups = group_columns(cd.data)
#     elif isinstance(group_columns, str):
#         p = Path(group_columns)
#         if p.suffix == ".json":
#             cd.column_groups = cg.ColumnGroups(util.read_json(group_columns))
#         # elif p.suffix == '.xlsx':
#         #     cd.column_groups = "read excel file"
#
#     cd.data_filtered = cd.data.copy()
#     cd.trans_keys = list(cd.column_groups.keys())
#     return cd