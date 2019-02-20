# standard library imports
import os
import datetime
import re
import math
import copy
import collections
from functools import wraps
import warnings

# anaconda distribution defaults
import dateutil
import numpy as np
import pandas as pd

# anaconda distribution defaults
# visualization library imports
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.palettes import Category10, Category20c, Category20b
from bokeh.layouts import gridplot
from bokeh.models import Legend, HoverTool, tools, ColumnDataSource

plot_colors_brewer = {'real_pwr': ['#2b8cbe', '#7bccc4', '#bae4bc', '#f0f9e8'],
                      'irr-poa': ['#e31a1c', '#fd8d3c', '#fecc5c', '#ffffb2'],
                      'irr-ghi': ['#91003f', '#e7298a', '#c994c7', '#e7e1ef'],
                      'temp-amb': ['#238443', '#78c679', '#c2e699', '#ffffcc'],
                      'temp-mod': ['#88419d', '#8c96c6', '#b3cde3', '#edf8fb'],
                      'wind': ['#238b45', '#66c2a4', '#b2e2e2', '#edf8fb']}

met_keys = ['poa', 't_amb', 'w_vel', 'power']

# The search strings for types cannot be duplicated across types.
type_defs = collections.OrderedDict([
             ('irr', [['irradiance', 'irr', 'plane of array', 'poa', 'ghi',
                       'global', 'glob', 'w/m^2', 'w/m2', 'w/m', 'w/'],
                      (-10, 1500)]),
             ('temp', [['temperature', 'temp', 'degrees', 'deg', 'ambient',
                        'amb', 'cell temperature'],
                       (-49, 127)]),
             ('wind', [['wind', 'speed'],
                       (0, 18)]),
             ('pf', [['power factor', 'factor', 'pf'],
                     (-1, 1)]),
             ('op_state', [['operating state', 'state', 'op', 'status'],
                           (0, 10)]),
             ('real_pwr', [['real power', 'ac power', 'e_grid'],
                           (-1000000, 1000000000000)]),  # set to very lax bounds
             ('shade', [['fshdbm', 'shd', 'shade'], (0, 1)]),
             ('index', [['index'], ('', 'z')])])

sub_type_defs = collections.OrderedDict([
                 ('ghi', [['sun2', 'global horizontal', 'ghi', 'global', 'glob']]),
                 ('poa', [['sun', 'plane of array', 'poa']]),
                 ('amb', [['TempF', 'ambient', 'amb']]),
                 ('mod', [['Temp1', 'module', 'mod']]),
                 ('mtr', [['revenue meter', 'rev meter', 'billing meter', 'meter']]),
                 ('inv', [['inverter', 'inv']])])

irr_sensors_defs = {'ref_cell': [['reference cell', 'reference', 'ref',
                                  'referance', 'pvel']],
                    'pyran': [['pyranometer', 'pyran']]}


class CapData(object):
    """
    Class to store capacity test data and translation of column names.

    CapData objects store a pandas dataframe of measured or simulated data
    and a translation dictionary used to translate and group the raw column
    names provided in the data.

    The translation dictionary allows maintaining the column names in the raw
    data while also grouping measurements of the same type from different
    sensors.

    Parameters
    ----------

    df : pandas dataframe
        Used to store measured or simulated data imported from csv.
    trans : dictionary
        A dictionary with keys that are algorithimically determined based on
        the data of each imported column in the dataframe and values that are
        the column labels in the raw data.
    trans_keys : list
        Simply a list of the translation dictionary (trans) keys.
    reg_trans : dictionary
        Dictionary that is manually set to link abbreviations for
        for the independent variables of the ASTM Capacity test regression
        equation to the translation dictionary keys.
    trans_abrev : dictionary
        Enumerated translation dict keys mapped to original column names.
        Enumerated translation dict keys are used in plot hover tooltip.
    col_colors : dictionary
        Original column names mapped to a color for use in plot function.
    """

    def __init__(self):
        super(CapData, self).__init__()
        self.df = pd.DataFrame()
        self.trans = {}
        self.trans_keys = []
        self.reg_trans = {}
        self.trans_abrev = {}
        self.col_colors = {}

    def set_reg_trans(self, power='', poa='', t_amb='', w_vel=''):
        """
        Create a dictionary linking the regression variables to trans_keys.

        Links the independent regression variables to the appropriate
        translation keys.  Sets attribute and returns nothing.

        Parameters
        ----------
        power : str
            Translation key for the power variable.
        poa : str
            Translation key for the plane of array (poa) irradiance variable.
        t_amb : str
            Translation key for the ambient temperature variable.
        w_vel : str
            Translation key for the wind velocity key.
        """
        self.reg_trans = {'power': power,
                          'poa': poa,
                          't_amb': t_amb,
                          'w_vel': w_vel}

    def copy(self):
        """Creates and returns a copy of self."""
        cd_c = CapData()
        cd_c.df = self.df.copy()
        cd_c.trans = copy.copy(self.trans)
        cd_c.trans_keys = copy.copy(self.trans_keys)
        cd_c.reg_trans = copy.copy(self.reg_trans)
        cd_c.trans_abrev = copy.copy(self.trans_abrev)
        cd_c.col_colors = copy.copy(self.col_colors)
        return cd_c

    def empty(self):
        """Returns a boolean indicating if the CapData object contains data."""
        if self.df.empty and len(self.trans_keys) == 0 and len(self.trans) == 0:
            return True
        else:
            return False

    def load_das(self, path, filename, source=None, **kwargs):
        """
        Reads measured solar data from a csv file.

        Utilizes pandas read_csv to import measure solar data from a csv file.
        Attempts a few diferent encodings, trys to determine the header end
        by looking for a date in the first column, and concantenates column
        headings to a single string.

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

        data = os.path.normpath(path + filename)

        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                all_data = pd.read_csv(data, encoding=encoding, index_col=0,
                                       parse_dates=True, skip_blank_lines=True,
                                       low_memory=False, **kwargs)
            except UnicodeDecodeError:
                continue
            else:
                break

        if not isinstance(all_data.index[0], pd.Timestamp):
            for i, indice in enumerate(all_data.index):
                try:
                    isinstance(dateutil.parser.parse(str(all_data.index[i])),
                               datetime.date)
                    header_end = i + 1
                    break
                except ValueError:
                    continue

            if source == 'AlsoEnergy':
                header = 'infer'
            else:
                header = list(np.arange(header_end))

            for encoding in encodings:
                try:
                    all_data = pd.read_csv(data, encoding=encoding,
                                           header=header, index_col=0,
                                           parse_dates=True, skip_blank_lines=True,
                                           low_memory=False, **kwargs)
                except UnicodeDecodeError:
                    continue
                else:
                    break

            if source == 'AlsoEnergy':
                row0 = all_data.iloc[0, :]
                row1 = all_data.iloc[1, :]
                row2 = all_data.iloc[2, :]

                row0_noparen = []
                for val in row0:
                    if type(val) is str:
                        row0_noparen.append(val.split('(')[0].strip())
                    else:
                        row0_noparen.append(val)

                row1_nocomm = []
                for val in row1:
                    if type(val) is str:
                        strings = val.split(',')
                        if len(strings) == 1:
                            row1_nocomm.append(val)
                        else:
                            row1_nocomm.append(strings[-1].strip())
                    else:
                        row1_nocomm.append(val)

                row2_noNan = []
                for val in row2:
                    if val is pd.np.nan:
                        row2_noNan.append('')
                    else:
                        row2_noNan.append(val)

                new_cols = []
                for one, two, three in zip(row0_noparen, row1_nocomm, row2_noNan):
                    new_cols.append(str(one) + ' ' + str(two) + ', ' + str(three))

                all_data.columns = new_cols

        all_data = all_data.apply(pd.to_numeric, errors='coerce')
        all_data.dropna(axis=1, how='all', inplace=True)
        all_data.dropna(how='all', inplace=True)

        if source is not 'AlsoEnergy':
            all_data.columns = [' '.join(col).strip() for col in all_data.columns.values]
        else:
            all_data.index = pd.to_datetime(all_data.index)

        return all_data

    def load_pvsyst(self, path, filename, **kwargs):
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

        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                # pvraw = pd.read_csv(dirName, skiprows=10, encoding=encoding,
                #                     header=[0, 1], parse_dates=[0],
                #                     infer_datetime_format=True, **kwargs)
                pvraw = pd.read_csv(dirName, skiprows=10, encoding=encoding,
                                    header=[0, 1], **kwargs)
            except UnicodeDecodeError:
                continue
            else:
                break

        pvraw.columns = pvraw.columns.droplevel(1)
        dates = pvraw.loc[:, 'date']
        try:
            dt_index = pd.to_datetime(dates, format='%m/%d/%y %H:%M')
        except ValueError:
            dt_index = pd.to_datetime(dates)
        pvraw.index = dt_index
        pvraw.drop('date', axis=1, inplace=True)
        pvraw = pvraw.rename(columns={"T Amb": "TAmb"})
        return pvraw

    def load_data(self, path='./data/', fname=None, set_trans=True, source=None,
                  load_pvsyst=False, **kwargs):
        """
        Import data from csv files.

        Parameters
        ----------
        path : str, default './data/'
            Path to directory containing csv files to load.
        fname: str, default None
            Filename of specific file to load. If filename is none method will
            load all csv files into one dataframe.
        set_trans : bool, default True
            Generates translation dicitionary for column names after loading
            data.
        source : str, default None
            Default of None uses general approach that concatenates header data.
            Set to 'AlsoEnergy' to use column heading parsing specific to
            downloads from AlsoEnergy.
        load_pvsyst : bool, default False
            By default skips any csv file that has 'pvsyst' in the name.  Is
            not case sensitive.  Set to true to import a csv with 'pvsyst' in
            the name and skip all other files.
        **kwargs
            Will pass kwargs onto load_pvsyst or load_das, which will pass to
            Pandas.read_csv.  Useful to adjust the separator (Ex. sep=';').

        Returns
        -------
        None
        """
        if fname is None:
            files_to_read = []
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    files_to_read.append(file)
                elif file.endswith('.CSV'):
                    files_to_read.append(file)

            all_sensors = pd.DataFrame()

            if not load_pvsyst:
                for filename in files_to_read:
                    if filename.lower().find('pvsyst') != -1:
                        print("Skipped file: " + filename)
                        continue
                    nextData = self.load_das(path, filename, source=source,
                                             **kwargs)
                    all_sensors = pd.concat([all_sensors, nextData], axis=0)
                    print("Read: " + filename)
            elif load_pvsyst:
                for filename in files_to_read:
                    if filename.lower().find('pvsyst') == -1:
                        print("Skipped file: " + filename)
                        continue
                    nextData = self.load_pvsyst(path, filename, **kwargs)
                    all_sensors = pd.concat([all_sensors, nextData], axis=0)
                    print("Read: " + filename)
        else:
            if not load_pvsyst:
                all_sensors = self.load_das(path, fname, source=source, **kwargs)
            elif load_pvsyst:
                all_sensors = self.load_pvsyst(path, fname, **kwargs)

        ix_ser = all_sensors.index.to_series()
        all_sensors['index'] = ix_ser.apply(lambda x: x.strftime('%m/%d/%Y %H %M'))
        self.df = all_sensors

        if set_trans:
            self.__set_trans()

    def __series_type(self, series, type_defs, bounds_check=True,
                      warnings=False):
        """
        Assigns columns to a category by analyzing the column names.

        The type_defs parameter is a dictionary which defines search strings
        and value limits for each key, where the key is a categorical name
        and the search strings are possible related names.  For example an
        irradiance sensor has the key 'irr' with search strings 'irradiance'
        'plane of array', 'poa', etc.

        Parameters
        ----------
        series : pandas series
            Pandas series, row or column of dataframe passed by pandas.df.apply.
        type_defs : dictionary
            Dictionary with the following structure.  See type_defs
            {'category abbreviation': [[category search strings],
                                       (min val, max val)]}
        bounds_check : bool, default True
            When true checks series values against min and max values in the
            type_defs dictionary.
        warnings : bool, default False
            When true prints warning that values in series are outside expected
            range and adds '-valuesError' to returned str.

        Returns
        -------
        string
            Returns a string representing the category for the series.
            Concatenates '-valuesError' if bounds_check and warnings are both
            True and values within the series are outside the expected range.
        """
        for key in type_defs.keys():
            # print('################')
            # print(key)
            for search_str in type_defs[key][0]:
                # print(search_str)
                if series.name.lower().find(search_str.lower()) == -1:
                    continue
                else:
                    if bounds_check:
                        min_bool = series.min() >= type_defs[key][1][0]
                        max_bool = series.max() <= type_defs[key][1][1]
                        if min_bool and max_bool:
                            return key
                        else:
                            if warnings:
                                if not min_bool:
                                    print('Values in {} exceed min values for {}'.format(series.name, key))
                                elif not max_bool:
                                    print('Values in {} exceed max values for {}'.format(series.name, key))
                            return key + '-valuesError'
                    else:
                        return key
        return ''

    def set_plot_attributes(self):
        dframe = self.df

        for key in self.trans_keys:
            df = dframe[self.trans[key]]
            cols = df.columns.tolist()
            for i, col in enumerate(cols):
                abbrev_col_name = key + str(i)
                self.trans_abrev[abbrev_col_name] = col

                col_key0 = key.split('-')[0]
                col_key1 = key.split('-')[1]
                if col_key0 in ('irr', 'temp'):
                    col_key = col_key0 + '-' + col_key1
                else:
                    col_key = col_key0

                try:
                    j = i % 4
                    self.col_colors[col] = plot_colors_brewer[col_key][j]
                except KeyError:
                    j = i % 10
                    self.col_colors[col] = Category10[10][j]

    def __set_trans(self):
        """
        Creates a dict of raw column names paired to categorical column names.

        Uses multiple type_def formatted dictionaries to determine the type,
        sub-type, and equipment type for data series of a dataframe.  The determined
        types are concatenated to a string used as a dictionary key with a list
        of one or more oringal column names as the paried value.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets attributes self.trans and self.trans_keys

        Todo
        ----
        type_defs parameter
            Consider refactoring to have a list of type_def dictionaries as an
            input and loop over each dict in the list.
        """
        col_types = self.df.apply(self.__series_type, args=(type_defs,)).tolist()
        sub_types = self.df.apply(self.__series_type, args=(sub_type_defs,),
                                  bounds_check=False).tolist()
        irr_types = self.df.apply(self.__series_type, args=(irr_sensors_defs,),
                                  bounds_check=False).tolist()

        col_indices = []
        for typ, sub_typ, irr_typ in zip(col_types, sub_types, irr_types):
            col_indices.append('-'.join([typ, sub_typ, irr_typ]))

        names = []
        for new_name, old_name in zip(col_indices, self.df.columns.tolist()):
            names.append((new_name, old_name))
        names.sort()
        orig_names_sorted = [name_pair[1] for name_pair in names]

        trans = {}
        col_indices.sort()
        cols = list(set(col_indices))
        cols.sort()
        for name in set(cols):
            start = col_indices.index(name)
            count = col_indices.count(name)
            trans[name] = orig_names_sorted[start:start + count]

        self.trans = trans

        trans_keys = list(self.trans.keys())
        if 'index--' in trans_keys:
            trans_keys.remove('index--')
        trans_keys.sort()
        self.trans_keys = trans_keys

        self.set_plot_attributes()

    def drop_cols(self, columns):
        """
        Drops columns from CapData dataframe and translation dictionary.

        Parameters
        ----------
        Columns (list) List of columns to drop.

        Todo
        ----
        Change to accept a string column name or list of strings
        """
        for key, value in self.trans.items():
            for col in columns:
                try:
                    value.remove(col)
                    self.trans[key] = value
                except ValueError:
                    continue
        self.df.drop(columns, axis=1, inplace=True)

    def view(self, tkey):
        """
        Convience function returns columns using translation dictionary names.

        Parameters
        ----------
        tkey: int or str or list of int or strs
            String or list of strings from self.trans_keys or int postion or
            list of int postitions of value in self.trans_keys.
        """

        if isinstance(tkey, int):
            keys = self.trans[self.trans_keys[tkey]]
        elif isinstance(tkey, list) and len(tkey) > 1:
            keys = []
            for key in tkey:
                if isinstance(key, str):
                    keys.extend(self.trans[key])
                elif isinstance(key, int):
                    keys.extend(self.trans[self.trans_keys[key]])
        elif tkey in self.trans_keys:
            keys = self.trans[tkey]

        return self.df[keys]

    def rview(self, ind_var):
        """
        Convience fucntion to return regression independent variable.

        Parameters
        ----------
        ind_var: string or list of strings
            may be 'power', 'poa', 't_amb', 'w_vel', a list of some subset of
            the previous four strings or 'all'
        """

        if ind_var == 'all':
            keys = list(self.reg_trans.values())
        elif isinstance(ind_var, list) and len(ind_var) > 1:
            keys = [self.reg_trans[key] for key in ind_var]
        elif ind_var in met_keys:
            ind_var = [ind_var]
            keys = [self.reg_trans[key] for key in ind_var]

        lst = []
        for key in keys:
            lst.extend(self.trans[key])
        return self.df[lst]

    def __comb_trans_keys(self, grp):
        comb_keys = []

        for key in self.trans_keys:
            if key.find(grp) != -1:
                comb_keys.append(key)

        cols = []
        for key in comb_keys:
            cols.extend(self.trans[key])

        grp_comb = grp + '_comb'
        if grp_comb not in self.trans_keys:
            self.trans[grp_comb] = cols
            self.trans_keys.extend([grp_comb])
            print('Added new group: ' + grp_comb)

    def plot(self, reindex=False, freq=None, marker='line', ncols=2,
             width=400, height=350, legends=False, merge_grps=['irr', 'temp'],
             subset=None, **kwargs):
        """
        Plots a Bokeh line graph for each group of sensors in self.trans.

        Function returns a Bokeh grid of figures.  A figure is generated for each
        key in the translation dictionary and a line is plotted for each raw
        column name paired with that key.

        For example, if there are multiple plane of array irradiance sensors,
        the data from each one will be plotted on a single figure.

        Figures are not generated for categories that would plot more than 10
        lines.

        Parameters
        ----------
        reindex : Boolean, default False
            Use with the freq argument to reset index of dataframe, which will
            shows the gaps were data was removed by filtering steps.
        freq : str
            Pandas offset alias to use for frequency of new index, when reindex
            is set to True.
        marker : str, default 'line'
            Accepts 'line', 'circle', 'line-circle'.  These are bokeh marker
            options.
        ncols : int, default 2
            Number of columns in the bokeh gridplot.
        width : int, default 400
            Width of individual plots in gridplot.
        height: int, default 350
            Height of individual plots in gridplot.
        legends : bool, default False
            Turn on or off legends for individual plots.
        merge_grps : list, default ['irr', 'temp']
            List of strings to search for in the translation dictionary keys.
            A new key and group is created in the translation dictionary for
            each group.  By default will combine all irradiance measurements
            into a group and temperature measurements into a group.
            Pass empty list to not merge any plots.
        subset : list, default None
            List of the translation dictionary keys to use to control order of
            plots or to plot only a subset of the plots.
        kwargs
            Pass additional options to bokeh gridplot.  Merge_tools=False will
            shows the hover tool icon, so it can be turned off.

        Returns
        -------
        show(grid)
            Command to show grid of figures.  Intended for use in jupyter
            notebook.
        """
        for str_val in merge_grps:
            self.__comb_trans_keys(str_val)

        dframe = self.df
        dframe.index.name = 'Timestamp'

        names_to_abrev = {val: key for key, val in self.trans_abrev.items()}

        if reindex:
            index = pd.DatetimeIndex(freq=freq, start=self.df.index[0],
                                     end=self.df.index[-1])
            dframe = self.df.reindex(index=index)

        index = dframe.index.tolist()
        plots = []
        x_axis = None

        source = ColumnDataSource(dframe)

        hover = HoverTool()
        hover.tooltips = [
            ("Name", "$name"),
            ("Datetime", "@Timestamp{%D %H:%M}"),
            ("Value", "$y"),
        ]
        hover.formatters = {"Timestamp": "datetime"}

        if isinstance(subset, list):
            plot_keys = subset
        else:
            plot_keys = self.trans_keys

        for j, key in enumerate(plot_keys):
            df = dframe[self.trans[key]]
            cols = df.columns.tolist()

            if x_axis is None:
                p = figure(title=key, plot_width=width, plot_height=height,
                           x_axis_type='datetime', tools='pan, xwheel_pan, xwheel_zoom, box_zoom, save, reset')
                p.tools.append(hover)
                x_axis = p.x_range
            if j > 0:
                p = figure(title=key, plot_width=width, plot_height=height,
                           x_axis_type='datetime', x_range=x_axis, tools='pan, xwheel_pan, xwheel_zoom, box_zoom, save, reset')
                p.tools.append(hover)
            legend_items = []
            for i, col in enumerate(cols):
                abbrev_col_name = key + str(i)
                if marker == 'line':
                    series = p.line('Timestamp', col, source=source,
                                    line_color=self.col_colors[col],
                                    name=names_to_abrev[col])
                elif marker == 'circle':
                    series = p.circle('Timestamp', col,
                                      source=source,
                                      line_color=self.col_colors[col],
                                      size=2, fill_color="white",
                                      name=names_to_abrev[col])
                if marker == 'line-circle':
                    series = p.line('Timestamp', col, source=source,
                                    line_color=self.col_colors[col],
                                    name=names_to_abrev[col])
                    series = p.circle('Timestamp', col,
                                      source=source,
                                      line_color=self.col_colors[col],
                                      size=2, fill_color="white",
                                      name=names_to_abrev[col])
                legend_items.append((col, [series, ]))

            legend = Legend(items=legend_items, location=(40, -5))
            legend.label_text_font_size = '8pt'
            if legends:
                p.add_layout(legend, 'below')

            plots.append(p)

        grid = gridplot(plots, ncols=ncols, **kwargs)
        return show(grid)
