import os
import numpy as np
import pandas as pd
import dateutil
import datetime
import re
import matplotlib.pyplot as plt
import math
import copy
from functools import wraps

import statsmodels.formula.api as smf

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from bokeh.models import Legend, HoverTool, tools

import pecos  # remove?


met_keys = ['poa', 't_amb', 'w_vel', 'power']

aux_load = 100
ac_nameplate = 21040

# The search strings for types cannot be duplicated across types.
type_defs = {'irr': [['irradiance', 'irr', 'plane of array', 'poa', 'ghi',
                     'global', 'glob', 'w/m^2', 'w/m2', 'w/m', 'w/'],
                     (-10, 1500)],
             'temp': [['temperature', 'temp', 'degrees', 'deg', 'ambient',
                       'amb', 'cell temperature'],
                      (-49, 127)],
             'wind': [['wind', 'speed'],
                      (0, 18)],
             'pf': [['power factor', 'factor', 'pf'],
                    (-1, 1)],
             'op_state': [['operating state', 'state', 'op', 'status'],
                          (0, 10)],
             'real_pwr': [['real power', 'ac power', 'e_grid'],
                          (aux_load, ac_nameplate * 1.05)],
             'shade': [['fshdbm', 'shd', 'shade'], (0, 1)],
             'index': [['index'], ('', 'z')]}

sub_type_defs = {'poa': [['plane of array', 'poa']],
                 'ghi': [['global horizontal', 'ghi', 'global', 'glob']],
                 'amb': [['ambient', 'amb']],
                 'mod': [['module', 'mod']],
                 'mtr': [['revenue meter', 'rev meter', 'billing meter', ' meter']],
                 'inv': [['inverter', 'inv']]}

irr_sensors_defs = {'ref_cell': [['reference cell', 'reference', 'ref',
                                  'referance', 'pvel']],
                    'pyran': [['pyranometer', 'pyran']]}


columns = ['Timestamps', 'Timestamps_filtered', 'Filter_arguments']


def update_summary(func):
    """
    ToDo:
    Check if summary is updated when function is called with inplace=False. It
    should not be.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if 'sim' in args:
            pts_before = self.flt_sim.df.shape[0]
            if pts_before == 0:
                pts_before = self.sim.df.shape[0]
                self.sim_mindex.append(('sim', 'sim_count'))
                self.sim_summ_data.append({columns[0]: pts_before,
                                           columns[1]: 0,
                                           columns[2]: 'no filters'})
        if 'das' in args:
            pts_before = self.flt_das.df.shape[0]
            if pts_before == 0:
                pts_before = self.das.df.shape[0]
                self.das_mindex.append(('das', 'das_count'))
                self.das_summ_data.append({columns[0]: pts_before,
                                           columns[1]: 0,
                                           columns[2]: 'no filters'})

        ret_val = func(self, *args, **kwargs)

        arg_str = args.__repr__() + kwargs.__repr__()

        if 'sim' in args:
            pts_after = self.flt_sim.df.shape[0]
            pts_removed = pts_before - pts_after
            self.sim_mindex.append(('sim', func.__name__))
            self.sim_summ_data.append({columns[0]: pts_after,
                                       columns[1]: pts_removed,
                                       columns[2]: arg_str})
        if 'das' in args:
            pts_after = self.flt_das.df.shape[0]
            pts_removed = pts_before - pts_after
            self.das_mindex.append(('das', func.__name__))
            self.das_summ_data.append({columns[0]: pts_after,
                                       columns[1]: pts_removed,
                                       columns[2]: arg_str})

        return ret_val
    return wrapper


class CapData(object):
    """docstring for CapData."""
    def __init__(self):
        super(CapData, self).__init__()
        self.df = pd.DataFrame()
        self.trans = {}
        self.trans_keys = []
        self.reg_trans = {}

    def set_reg_trans(self, power='', poa='', t_amb='', w_vel=''):
        self.reg_trans = {'power': power,
                          'poa': poa,
                          't_amb': t_amb,
                          'w_vel': w_vel}

    def copy(self):
        cd_c = CapData()
        cd_c.df = self.df.copy()
        cd_c.trans = copy.copy(self.trans)
        cd_c.trans_keys = copy.copy(self.trans_keys)
        cd_c.reg_trans = copy.copy(self.reg_trans)
        return cd_c

    def empty(self):
        if self.df.empty and len(self.trans_keys) == 0 and len(self.trans) == 0:
            return True
        else:
            return False

    def load_das(self, path, filename, **kwargs):
        header_end = 1

        data = os.path.normpath(path + filename)

        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                all_data = pd.read_csv(data, encoding=encoding,
                                       header=[0, header_end], index_col=0,
                                       parse_dates=True, skip_blank_lines=True,
                                       low_memory=False, **kwargs)
            except UnicodeDecodeError:
                continue
            else:
                break

        if not isinstance(all_data.index[0], pd.Timestamp):
            for i, indice in enumerate(all_data.index):
                try:
                    isinstance(dateutil.parser.parse(all_data.index[i]), datetime.date)
                    header_end = i + 1
                    break
                except ValueError:
                    continue

        for encoding in encodings:
            try:
                all_data = pd.read_csv(data, encoding=encoding,
                                       header=[0, header_end], index_col=0,
                                       parse_dates=True, skip_blank_lines=True,
                                       low_memory=False, **kwargs)
            except UnicodeDecodeError:
                continue
            else:
                break

        all_data = all_data.apply(pd.to_numeric, errors='coerce')
        all_data.dropna(axis=1, how='all', inplace=True)
        all_data.dropna(how='all', inplace=True)
        all_data.columns = [' '.join(col).strip() for col in all_data.columns.values]

        return all_data

    def load_pvsyst(self, path, filename, **kwargs):
        """
        Load sim data and add assign to attribute sim.
        """
        dirName = os.path.normpath(path + filename)

        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                pvraw = pd.read_csv(dirName, skiprows=10, encoding=encoding,
                                    header=[0, 1], parse_dates=[0],
                                    infer_datetime_format=True, **kwargs)
            except UnicodeDecodeError:
                continue
            else:
                break

        pvraw.columns = pvraw.columns.droplevel(1)
        # pvraw['dateString'] = pvraw['date'].apply(lambda x: x.strftime('%m/%d/%Y %H'))
        pvraw.set_index('date', drop=True, inplace=True)
        pvraw = pvraw.rename(columns={"T Amb": "TAmb"})
        return pvraw

    def load_data(self, directory='./data/', set_trans=True, load_pvsyst=False,
                  **kwargs):
        """
        Import data from csv files.

        Parameters
        ----------
        directory: str, default './data/'
            Path to directory containing csv files to load.
        set_trans: bool, default True
            Generates translation dicitionary for column names after loading data.
        load_pvsyst: bool, default False
            By default skips any csv file that has 'pvsyst' in the name.  Is not
            case sensitive.  Set to true to import a csv with 'pvsyst' in the name
            and skip all other files.
        **kwargs
            Will pass kwargs onto the inner call to Pandas.read_csv.  Useful to
            adjust the separator (Ex. sep=';').
        """

        files_to_read = []
        for file in os.listdir(directory):
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
                nextData = self.load_das(directory, filename, **kwargs)
                all_sensors = pd.concat([all_sensors, nextData], axis=0)
                print("Read: " + filename)
        elif load_pvsyst:
            for filename in files_to_read:
                if filename.lower().find('pvsyst') == -1:
                    print("Skipped file: " + filename)
                    continue
                nextData = self.load_pvsyst(directory, filename, **kwargs)
                all_sensors = pd.concat([all_sensors, nextData], axis=0)
                print("Read: " + filename)

        ix_ser = all_sensors.index.to_series()
        all_sensors['index'] = ix_ser.apply(lambda x: x.strftime('%m/%d/%Y %H %M'))
        self.df = all_sensors

        if set_trans:
            self.__set_trans()

    def __series_type(self, series, type_defs, bounds_check=True,
                      warnings=False):
        for key in type_defs.keys():
            # print('################')
            # print(key)
            for search_str in type_defs[key][0]:
                # print(search_str)
                if series.name.lower().find(search_str) == -1:
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

    def __set_trans(self):
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

    def drop_cols(self, columns):
        """
        Drops columns from CapData dataframe and translation dictionary.

        Parameters
        ----------
        columns (list) List of columns to drop.
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
        Convience function to return dataframe columns using translation
        dictionary names.

        Parameters
        --------------
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

        Paremeters
        --------------
        ind_var: string or list of strings
            may be 'power', 'poa', 't_amb', 'w_vel', a list of some subset of
            the previous four strings or 'all'
        ToDo:
        -rename to view?
        -split into two methods view for trans keys and rview for reg_trans keys
        -expand to all values in trans_keys? if var is an integer then use that
         integer as an index in the trans_keys list
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


class CapTest(object):
    """
    CapTest provides methods to facilitate solar PV capacity testing.
    """

    def __init__(self, das, sim):
        self.das = das
        self.flt_das = CapData()
        self.das_mindex = []
        self.das_summ_data = []
        self.sim = sim
        self.flt_sim = CapData()
        self.sim_mindex = []
        self.sim_summ_data = []
        self.rc = dict()
        self.ols_model_das = None
        self.ols_model_sim = None

    def summary(self):
        summ_data, mindex = [], []
        if len(self.das_summ_data) != 0 and len(self.sim_summ_data) != 0:
            summ_data.extend(self.das_summ_data)
            summ_data.extend(self.sim_summ_data)
            mindex.extend(self.das_mindex)
            mindex.extend(self.sim_mindex)
        elif len(self.das_summ_data) != 0:
            summ_data.extend(self.das_summ_data)
            mindex.extend(self.das_mindex)
        else:
            summ_data.extend(self.sim_summ_data)
            mindex.extend(self.sim_mindex)
        try:
            df = pd.DataFrame(data=summ_data,
                              index=pd.MultiIndex.from_tuples(mindex),
                              columns=columns)
            return df
        except TypeError:
            print('No filters have been run.')

    def plot(self, capdata):
        index = capdata.df.index.tolist()
        colors = Category10[10]
        plots = []
        for j, key in enumerate(capdata.trans_keys):
            df = capdata.df[capdata.trans[key]]
            cols = df.columns.tolist()
            if len(cols) > len(colors):
                print('Skipped {} because there are more than 10   columns.'.format(key))
                continue

            if j == 0:
                p = figure(title=key, plot_width=400, plot_height=225,
                           x_axis_type='datetime')
                x_axis = p.x_range
            if j > 0:
                p = figure(title=key, plot_width=400, plot_height=225,
                           x_axis_type='datetime', x_range=x_axis)
            legend_items = []
            for i, col in enumerate(cols):
                line = p.line(index, df[col], line_color=colors[i])
                legend_items.append((col, [line, ]))

            legend = Legend(items=legend_items, location=(40, -5))
            legend.label_text_font_size = '8pt'
            p.add_layout(legend, 'below')

            plots.append(p)

        grid = gridplot(plots, ncols=2)
        return show(grid)

    def scatter(self, data):
        """
        Create scatter plot of irradiance vs power.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if plot is of sim or das data.

        ToDo:
        -add nans for filtered time stamps, so it is clear what has been removed
        """
        flt_cd = self.__flt_setup(data)

        df = flt_cd.rview(['power', 'poa'])

        if df.shape[1] != 2:
            print('Aggregate sensors before using this method.')
            return None

        df = df.rename(columns={df.columns[0]: 'power', df.columns[1]: 'poa'})
        plt = df.plot(kind='scatter', x='poa', y='power',
                      title=data, alpha=0.2)
        return(plt)

    def reg_scatter_matrix(self, data):
        """
        Create pandas scatter matrix of regression variables.

        Parameters
        ----------
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        """
        cd_obj = self.__flt_setup(data)

        df = cd_obj.rview(['poa', 't_amb', 'w_vel'])
        rename = {df.columns[0]: 'poa',
                  df.columns[1]: 't_amb',
                  df.columns[2]: 'w_vel'}
        df = df.rename(columns=rename)
        df['poa_poa'] = df['poa'] * df['poa']
        df['poa_t_amb'] = df['poa'] * df['t_amb']
        df['poa_w_vel'] = df['poa'] * df['w_vel']
        df.drop(['t_amb', 'w_vel'], axis=1, inplace=True)
        return(pd.plotting.scatter_matrix(df))

    def sim_apply_losses(self):
        """
        Apply post sim losses to sim data.
        xfmr loss, mv voltage drop, availability
        """
        pass

    def pred_rcs(self,):
        """
        Parameters
        ----------
        data: str, 'sim' or 'das'
            'sim' or 'das' determines if filter is on sim or das data
        test_date: str, 'mm/dd/yyyy', optional
            Date to center reporting conditions aggregation functions around.
            When not used specified reporting conditions for all data passed
            are returned grouped by the freq provided.  freq='90D' give seasonal
            reporting conditions and freq='30D' give monthly reproting conditions.
        freq: str, default '60D'
            String representing number of days to aggregate for reporting
            condition calculation.  Ex '60D' for 60 Days.  Typical '30D', '60D',
            '90D'.
        """
        # if data = 'sim' and test_date is not None:
        #     date = pd.to_datetime(test_date)
        #     offset = pd.DateOffset(days=int(freq[:2]) / 2)
        #     start = date - offset
        #
        #     # this is only useful for simulated data
        #     # need differnt approach for real data across end of year
        #     tail = df.loc[start:, :]
        #     head = df.loc[:start, :]
        #     head = head.iloc[:head.shape[0] - 1, :]
        #     head_shifted = head.shift(8760, freq='H')
        #     dfnewstart = pd.concat([tail, head_shifted])
        #
        #     temp_wind = dfnewstart[['t_amb', 'w_vel']]
        #     irr = dfnewstart['GlobInc']
        #
        #     RCs = temp_wind.groupby(pd.Grouper(freq=freq, label='right')).mean()
        #     RCs['GlobInc'] = irr.groupby(pd.TimeGrouper(freq=freq,
        #                                     label='right')).quantile(.6)
        #     RCs = RCs.iloc[0, :]
        pass

    @update_summary
    def rep_cond(self, data, test_date=None, days=60, inplace=True, mean=False):
        """
        Calculate reporting conditons.

        Parameters
        ----------
        data: str, 'sim' or 'das'
            'sim' or 'das' determines if filter is on sim or das data
        test_date: str, 'mm/dd/yyyy', optional
            Date to center reporting conditions aggregation functions around.
            When not used specified reporting conditions for all data passed
            are returned grouped by the freq provided.  freq='90D' give seasonal
            reporting conditions and freq='30D' give monthly reproting conditions.
        days: int, default 60
            Number of days to use when calculating reporting conditons.  Typically
            no less than 30 and no more than 90.
        inplace: bool, True by default
            When true updates object rc parameter, when false returns dicitionary
            of reporting conditions.
        mean: bool, False by default
            Calculates irradiance reporting conditions by mean rather than default
            of 60th percentile.  Default follows ASTM standard.
        """
        flt_cd = self.__flt_setup(data)
        df = flt_cd.rview(['poa', 't_amb', 'w_vel'])
        df = df.rename(columns={df.columns[0]: 'poa',
                                df.columns[1]: 't_amb',
                                df.columns[2]: 'w_vel'})

        date = pd.to_datetime(test_date)
        offset = pd.DateOffset(days=days)
        start = date - offset
        end = date + offset
        if start < df.index[0]:
            start = df.index[0]
        if end > df.index[-1]:
            end = df.index[-1]
        df = df.loc[start:end, :]

        if mean:
            RCs = {'poa': [df['poa'].mean()],
                   't_amb': [df['t_amb'].mean()],
                   'w_vel': [df['w_vel'].mean()]}
        else:
            RCs = {'poa': [df['poa'].mean()],
                   't_amb': [df['t_amb'].mean()],
                   'w_vel': [df['w_vel'].quantile(0.6)]}
        print(RCs)

        if inplace:
            self.rc = RCs
        else:
            return RCs

    def agg_sensors(self, data, irr='median', temp='mean', wind='mean',
                    real_pwr='sum', inplace=True, keep=True):
        """
        Aggregate measurments of the same variable from different sensors.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
        irr: str, default 'median'
            Aggregates irradiance columns using the specified method.
        temp: str, default 'mean'
            Aggregates temperature columns using the specified method.
        wind: str, default 'mean'
            Aggregates wind speed columns using the specified method.
        real_pwr: str, default 'sum'
            Aggregates real power columns using the specified method.
        inplace: bool, default True
            True writes over current filtered dataframe.
            False returns an aggregated dataframe.
        keep: bool, default True
            Keeps non regression columns in returned dataframe.
        """
        # met_keys = ['poa', 't_amb', 'w_vel', 'power']
        cd_obj = self.__flt_setup(data)

        agg_series = []
        agg_series.append((cd_obj.rview('poa')).agg(irr, axis=1))
        agg_series.append((cd_obj.rview('t_amb')).agg(temp, axis=1))
        agg_series.append((cd_obj.rview('w_vel')).agg(wind, axis=1))
        agg_series.append((cd_obj.rview('power')).agg(real_pwr, axis=1))

        comb_names = []
        for key in met_keys:
            comb_name = ('AGG-' + ', '.join(cd_obj.trans[cd_obj.reg_trans[key]]))
            comb_names.append(comb_name)
            if inplace:
                cd_obj.trans[cd_obj.reg_trans[key]] = [comb_name, ]

        temp_dict = {key: val for key, val in zip(comb_names, agg_series)}
        df = pd.DataFrame(temp_dict)

        if keep:
            lst = []
            for value in cd_obj.reg_trans.values():
                lst.extend(cd_obj.trans[value])
            sel = [i for i, name in enumerate(cd_obj.df) if name not in lst]
            df = pd.concat([df, cd_obj.df.iloc[:, sel]], axis=1)

        cd_obj.df = df

        if inplace:
            if data == 'das':
                self.flt_das = cd_obj
            elif data == 'sim':
                self.flt_sim = cd_obj
        else:
            return cd_obj

    def drop_non_reg_cols(self, arg):
        """
        Is this needed?  easily done with pandas, why drop when you can select
        """
        pass

    """
    Filtering methods must do the following:
    -add name of filter, pts before, and pts after to a self.DataFrame
    -possibly also add argument values filter function is called with
    -check if this is the first filter function run, if True copy raw_data
    -determine if filter methods return new object (copy data) or modify df
    """

    def __flt_setup(self, data):
        if data == 'das':
            if self.flt_das.empty():
                self.flt_das = self.das.copy()
            return self.flt_das
        if data == 'sim':
            if self.flt_sim.empty():
                self.flt_sim = self.sim.copy()
            return self.flt_sim

    def reset_flt(self, data):
        """
        Copies over filtered dataframe with raw data.
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        Removes all summary history.

        Todo:

        """
        if data == 'das':
            self.flt_das = self.das.copy()
            self.das_mindex = []
            self.das_summ_data = []
        elif data == 'sim':
            self.flt_sim = self.sim.copy()
            self.sim_mindex = []
            self.sim_summ_data = []
        else:
            print("'data must be 'das' or 'sim'")

    @update_summary
    def filter_outliers(self, data, inplace=True):
        """
        Apply eliptic envelope from scikit-learn to remove outliers.

        Parameters
        ----------
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        """
        flt_cd = self.__flt_setup(data)

        XandY = flt_cd.rview(['poa', 'power'])
        X1 = XandY.values

        clf_1 = EllipticEnvelope(contamination=0.04)
        clf_1.fit(X1)

        flt_cd.df = flt_cd.df[clf_1.predict(X1) == 1]

        if inplace:
            if data == 'das':
                self.flt_das = flt_cd
            if data == 'sim':
                self.flt_sim = flt_cd
        else:
            return flt_cd

    @update_summary
    def filter_pf(self, data, pf):
        """
        Keep timestamps where all power factors are greater than or equal to pf.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
        pf: float
            0.999 or similar to remove timestamps with lower PF values
        """
        flt_cd = self.__flt_setup(data)

        for key in flt_cd.trans_keys:
            if key.find('pf') == 0:
                selection = key

        df = flt_cd.df[flt_cd.trans[selection]]
        flt_cd.df = flt_cd.df[(df >= pf).all(axis=1)]

        if data == 'das':
            self.flt_das = flt_cd
        if data == 'sim':
            self.flt_sim = flt_cd

    @update_summary
    def filter_irr(self, data, low, high, ref_val=None, inplace=True):
        """
        Filter on irradiance values.

        Parameters
        ----------
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        low (float/int) - minimum value as fraction (0.8) or absolute 200 (W/m^2)
        high (float/int) - max value as fraction (1.2) or absolute 800 (W/m^2)
        ref_val (float/ing) - Must provide arg when min/max are fractions
        inplace (bool) - Default true write back to CapTest.flt_sim or flt_das
        """
        flt_cd = self.__flt_setup(data)

        if ref_val is not None:
            low *= ref_val
            high *= ref_val

        df = flt_cd.rview('poa')
        df = df.rename(columns={df.columns[0]: 'poa'})
        df.query('@low <= poa <= @high', inplace=True)

        flt_cd.df = flt_cd.df.loc[df.index, :]

        if inplace:
            if data == 'das':
                self.flt_das = flt_cd
            if data == 'sim':
                self.flt_sim = flt_cd
        else:
            return flt_cd

    @update_summary
    def filter_op_state(self, data, op_state, mult_inv=None, inplace=True):
        """
        Filter on inverter operation state.

        Parameters
        ----------
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        op_state (integer) - integer inverter operating state to keep
        mult_inv (list of tuples) - [(start, stop, op_state), ...] list of tuples
                    where start is the first column of an type of inverter, stop
                    is the last column and op_state is the operating state for the
                    inverter type.
        inplace (bool) - default True writes over current filtered dataframe
                         False returns CapData object
        """
        if data == 'sim':
            print('Method not implemented for pvsyst data.')
            return None

        flt_cd = self.__flt_setup(data)

        for key in flt_cd.trans_keys:
            if key.find('op') == 0:
                selection = key

        df = flt_cd.df[flt_cd.trans[selection]]
        # print('df shape: {}'.format(df.shape))

        if mult_inv is not None:
            return_index = flt_cd.df.index
            for pos_tup in mult_inv:
                # print('pos_tup: {}'.format(pos_tup))
                inverters = df.iloc[:, pos_tup[0]:pos_tup[1]]
                # print('inv shape: {}'.format(inverters.shape))
                df_temp = flt_cd.df[(inverters == pos_tup[2]).all(axis=1)]
                # print('df_temp shape: {}'.format(df_temp.shape))
                return_index = return_index.intersection(df_temp.index)
            flt_cd.df = flt_cd.df.loc[return_index, :]
        else:
            flt_cd.df = flt_cd.df[(df == op_state).all(axis=1)]

        if inplace:
            if data == 'das':
                self.flt_das = flt_cd
            if data == 'sim':
                # should not run as 'sim' is not implemented
                self.flt_sim = flt_cd
        else:
            return flt_cd

    def filter_clipping(self, arg):
        """
        May not be needed as can be accomplished through filter_irr
        """
        pass

    @update_summary
    def filter_missing(self, data):
        """
        Remove timestamps with missing data.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
        """
        flt_cd = self.__flt_setup(data)
        flt_cd.df = flt_cd.df.dropna(axis=0, how='all', inplace=False)
        if data == 'das':
            self.flt_das = flt_cd
        if data == 'sim':
            self.flt_sim = flt_cd

    def __std_filter(self, series, std_devs=3):
        mean = series.mean()
        std = series.std()
        min_bound = mean - std * std_devs
        max_bound = mean + std * std_devs
        return all(series.apply(lambda x: min_bound < x < max_bound))

    def __sensor_filter(self, df, perc_diff):
        if df.shape[1] > 2:
            return df[df.apply(self.__std_filter, axis=1)].index
        elif df.shape[1] == 1:
            return df.index
        else:
            sens_1 = df.iloc[:, 0]
            sens_2 = df.iloc[:, 1]
            return df[abs((sens_1 - sens_2) / sens_1) <= perc_diff].index

    @update_summary
    def filter_sensors(self, data, skip_strs=[], perc_diff=0.05, inplace=True,
                       reg_trans=True):
        """
        Drop suspicious measurments by comparing values from different sensors.

        Parameters
        ----------
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        skip_strs (list like) - strings to search for in column label.
                                If found skip column.
        perc_diff (float) - Percent difference cutoff for readings of the same
                            measurement from different sensors.
        inplace (bool) - default True writes over current filtered dataframe
                         False returns CapData object

        TODO:
        -perc_diff can be dict of sensor type keys paired with per_diff values
        """

        cd_obj = self.__flt_setup(data)

        if reg_trans:
            cols = cd_obj.reg_trans.values()
        else:
            cols = cd_obj.trans_keys

        # labels = list(set(df.columns.get_level_values(col_level).tolist()))
        for i, label in enumerate(cols):
            # print(i)
            skip_col = False
            if len(skip_strs) != 0:
                # print('skip strings: {}'.format(len(skip_strs)))
                for string in skip_strs:
                    # print(string)
                    if label.find(string) != -1:
                        skip_col = True
            if skip_col:
                continue
            if 'index' in locals():
                # if index has been assigned then take intersection
                # print(label)
                # print(pm.df[pm.trans[label]].head(1))
                df = cd_obj.df[cd_obj.trans[label]]
                next_index = self.__sensor_filter(df, perc_diff)
                index = index.intersection(next_index)
            else:
                # if index has not been assigned then assign it
                # print(label)
                # print(pm.df[pm.trans[label]].head(1))
                df = cd_obj.df[cd_obj.trans[label]]
                index = self.__sensor_filter(df, perc_diff)

        cd_obj.df = cd_obj.df.loc[index, :]

        if inplace:
            if data == 'das':
                self.flt_das = cd_obj
            elif data == 'sim':
                self.flt_sim = cd_obj
        else:
            return cd_obj

    @update_summary
    def reg_cpt(self, data, filter=False, inplace=True):
        """
        Performs regression with statsmodels on filtered data.

        Parameters
        ----------
        data: str, 'sim' or 'das'
            'sim' or 'das' determines if filter is on sim or das data
        filter: bool, default False
            When true removes timestamps where the residuals are greater than
            two standard deviations.  When false just calcualtes ordinary least
            squares regression.
        inplace: bool, default True
            If filter is true and inplace is true, then function overwrites the
            filtered data for sim or das.  If false returns a CapData object.
        """
        cd_obj = self.__flt_setup(data)

        df = cd_obj.rview(['power', 'poa', 't_amb', 'w_vel'])
        rename = {df.columns[0]: 'power',
                  df.columns[1]: 'poa',
                  df.columns[2]: 't_amb',
                  df.columns[3]: 'w_vel'}
        df = df.rename(columns=rename)

        fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        mod = smf.ols(formula=fml, data=df)
        reg = mod.fit()

        if filter:
            print('NOTE: Regression used to filter outlying points.\n\n')
            print(reg.summary())
            df = df[np.abs(reg.resid) < 2 * np.sqrt(reg.scale)]
            cd_obj.df = cd_obj.df.loc[df.index, :]
            if inplace:
                if data == 'das':
                    self.flt_das = cd_obj
                elif data == 'sim':
                    self.flt_sim = cd_obj
            else:
                return cd_obj
        else:
            print(reg.summary())
            if data == 'das':
                self.ols_model_das = reg
            elif data == 'sim':
                self.ols_model_sim = reg

    def predict(self, arg):
        """
        Calculate prediction from regression.
        """
        pass

    def cap_test(self, arg):
        """
        Apply methods to run a standard cap test following the ASTM standard.
        """
        pass


def equip_counts(df):
    equip_counts = {}
    eq_cnt_lst = []
    col_names = df.columns.tolist()
    for i, col_name in enumerate(col_names):
        # print('################')
        # print('loop: {}'.format(i))
        # print(col_name)
        if i == 0:
            equip_counts[col_name] = 1
            eq_cnt_lst.append(equip_counts[col_name])
            continue
        if col_name not in equip_counts.keys():
            equip_counts[col_name] = 1
            eq_cnt_lst.append(equip_counts[col_name])
        else:
            equip_counts[col_name] += 1
            eq_cnt_lst.append(equip_counts[col_name])
#         print(eq_cnt_lst[i])
    return eq_cnt_lst
