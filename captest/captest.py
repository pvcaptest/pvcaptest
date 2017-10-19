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

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from bokeh.models import Legend, HoverTool, tools

import pecos


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


columns = ['Timestamps', 'Timestamps_filtered', 'filter_arguments']


def update_summary(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if 'pvsyst' in args:
            pts_before = self.flt_sim.df.shape[0]
            if pts_before == 0:
                pts_before = self.sim.df.shape[0]
        if 'das' in args:
            pts_before = self.flt_das.df.shape[0]
            if pts_before == 0:
                pts_before = self.das.df.shape[0]

        func(self, *args, **kwargs)

        if 'pvsyst' in args:
            pts_after = self.flt_sim.df.shape[0]
            self.mindex.append(('pvsyst', func.__name__))
        if 'das' in args:
            pts_after = self.flt_das.df.shape[0]
            self.mindex.append(('das', func.__name__))

        pts_removed = pts_before - pts_after

        arg_str = args.__repr__() + kwargs.__repr__()
        self.summ_data.append({columns[0]: pts_after,
                               columns[1]: pts_removed,
                               columns[2]: arg_str})

    return wrapper


class CapData(object):
    """docstring for CapData."""
    def __init__(self):
        super(CapData, self).__init__()
        self.df = pd.DataFrame()
        self.trans = {}
        self.trans_keys = []

    def copy(self):
        cd_c = CapData()
        cd_c.df = self.df.copy()
        cd_c.trans = copy.copy(self.trans)
        cd_c.trans_keys = copy.copy(self.trans_keys)
        return cd_c

    def empty(self):
        if self.df.empty and len(self.trans_keys) == 0 and len(self.trans) == 0:
            return True
        else:
            return False

    def load_das_file(self, path, filename):
        header_end = 1

        data = os.path.normpath(path + filename)
        all_data = pd.read_csv(data, encoding="UTF-8", header=[0, header_end],
                               index_col=0, parse_dates=True, skip_blank_lines=True,
                               low_memory=False)

        if not isinstance(all_data.index[0], pd.Timestamp):
            for i, indice in enumerate(all_data.index):
                try:
                    isinstance(dateutil.parser.parse(all_data.index[i]), datetime.date)
                    header_end = i + 1
                    break
                except ValueError:
                    continue

            all_data = pd.read_csv(data, encoding="UTF-8", header=[0, header_end],
                                   index_col=0, parse_dates=True,
                                   skip_blank_lines=True, low_memory=False)

        all_data = all_data.apply(pd.to_numeric, errors='coerce')
        all_data.dropna(axis=1, how='all', inplace=True)
        all_data.dropna(how='all', inplace=True)
        all_data.columns = [' '.join(col).strip() for col in all_data.columns.values]

        return all_data

    def load_pvsyst(self, path, filename):
        """
        Load pvsyst data and add assign to attribute pvsyst.
        """
        dirName = os.path.normpath(path + filename)
        pvraw = pd.read_csv(dirName, skiprows=10, header=[0, 1],
                            parse_dates=[0], infer_datetime_format=True)
        pvraw.columns = pvraw.columns.droplevel(1)
        # pvraw['dateString'] = pvraw['date'].apply(lambda x: x.strftime('%m/%d/%Y %H'))
        pvraw.set_index('date', drop=True, inplace=True)
        pvraw = pvraw.rename(columns={"T Amb": "TAmb"})
        return pvraw

    def load_data(self, directory='./data/', set_trans=True, load_pvsyst=False):
        """
        Import data from csv files.
        directory (string) - default is to import from './data/'
        directoy='./path/to/data.csv'
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
                nextData = self.load_das_file(directory, filename)
                all_sensors = pd.concat([all_sensors, nextData], axis=0)
                print("Read: " + filename)
        elif load_pvsyst:
            for filename in files_to_read:
                if filename.lower().find('pvsyst') == -1:
                    print("Skipped file: " + filename)
                    continue
                nextData = self.load_pvsyst(directory, filename)
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


class CapTest(object):
    """
    CapTest provides methods to facilitate solar PV capacity testing.
    """

    def __init__(self, das):
        self.das = das
        self.flt_das = CapData()
        self.sim = CapData()  # should initialize with sim also
        self.flt_sim = CapData()
        self.reg_trans = {}
        self.mindex = []
        self.summ_data = []

    def set_reg_trans(self, power='', poa='', t_amb='', w_vel=''):
        self.reg_trans = {'power': power,
                          'poa': poa,
                          't_amb': t_amb,
                          'w_vel': w_vel}

    def var(self, var, capdata):
        """
        Convience fucntion to return regression independent variable.
        var (string) may be 'power', 'poa', 't_amb', 'w_vel' or 'all'
        capdata (CapData object)
        """

        if var == 'all':
            lst = []
            for value in self.reg_trans.values():
                lst.extend(capdata.trans[value])
            return capdata.df[lst]
        return capdata.df[capdata.trans[self.reg_trans[var]]]

    def summary(self):
        df = pd.DataFrame(data=self.summ_data,
                          index=pd.MultiIndex.from_tuples(self.mindex),
                          columns=columns)
        return df

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
        return grid

    def scatter(self, arg):
        """
        Create scatter plot of irradiance vs power.
        """
        pass

    def pvsyst_apply_losses(self):
        """
        Apply post pvsyst losses to pvsyst data.
        xfmr loss, mv voltage drop, availability
        """
        pass

    def rep_cond(self, arg):
        """
        Calculate reporting conditons.
        mnth, year, season
        from pvsyst, from actual data (at which filter step)
        """
        pass

    def agg_sensors(self, cd_obj, irr='median', temp='mean', wind='mean',
                    real_pwr='sum', inplace=True, keep=True):
        """
        Aggregate measurments of the same variable from different sensors.
        Optional keyword argument for each measurment:
        cd_obj (CapData) -  CapData object usually CapTest.raw_data or
                            CapTest.flt_data
        irr (string) - default 'median'
        temp (string) - default 'mean'
        wind (string) - default 'mean'
        real_pwr (string) - default 'mean'
        inplace (bool) - default True writes over current filtered dataframe
                         False returns dataframe
        keep (bool) - keeps non regression columns in output dataframe

        TODO:
        -Should function like a filter and act on filtered data if there
        is already filtered data or create filtered data
        """
        # met_keys = ['poa', 't_amb', 'w_vel', 'power']
        agg_series = []
        agg_series.append(self.var('poa', cd_obj).agg(irr, axis=1))
        agg_series.append(self.var('t_amb', cd_obj).agg(temp, axis=1))
        agg_series.append(self.var('w_vel', cd_obj).agg(wind, axis=1))
        agg_series.append(self.var('power', cd_obj).agg(real_pwr, axis=1))

        comb_names = []
        for key in met_keys:
            comb_name = ('AGG-' + ', '.join(cd_obj.trans[self.reg_trans[key]]))
            comb_names.append(comb_name)
            if inplace:
                cd_obj.trans[self.reg_trans[key]] = [comb_name, ]

        temp_dict = {key: val for key, val in zip(comb_names, agg_series)}
        df = pd.DataFrame(temp_dict)

        if keep:
            lst = []
            for value in self.reg_trans.values():
                lst.extend(cd_obj.trans[value])
            sel = [i for i, name in enumerate(cd_obj.df) if name not in lst]
            df = pd.concat([df, cd_obj.df.iloc[:, sel]])

        if inplace:
            cd_obj.df = df
        else:
            return(df)

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
        if data == 'pvsyst':
            if self.flt_sim.empty():
                self.flt_sim = self.sim.copy()
            return self.flt_sim

    def reset_flt(self, data):
        if data == 'das':
            self.flt_das = self.das.copy()
        if data == 'pvsyst':
            self.flt_sim = self.sim.copy()

    def filter_outliers(self, arg):
        """
        Apply eliptic envelope from scikit-learn to remove outliers.
        """
        pass

    @update_summary
    def filter_pf(self, data, pf):
        """
        Filter based on power factor values.
        data (str) - 'sim' or 'das' determines if filter is on sim or das data
        pf (float) - 0.999 or similar to remove timestamps with lower PF values
        """
        flt_cd = self.__flt_setup(data)

        for key in flt_cd.trans_keys:
            if key.find('pf') == 0:
                selection = key

        df = flt_cd.df[flt_cd.trans[selection]]
        flt_cd.df = flt_cd.df[(df >= pf).all(axis=1)]

        if data == 'das':
            self.flt_das = flt_cd
        if data == 'pvsyst':
            self.flt_sim = flt_cd

    def filter_irr(self, low, high, percent=True):
        """
        Filter on irradiance values.
        allow for a plus or minus percent argument or
        a min and max value
        if percent is false then low and high are min and max
        """
        pass

    def filter_op_state(self, arg):
        """
        Filter on inverter operation state.
        agrument needs to include operating state definition for specific inv
        This is inherently an inverter mppt filter.
        """
        pass

    def filter_clipping(self, arg):
        """
        May not be needed as can be accomplished through filter_irr
        """
        pass

    def filter_missing(self, arg):
        """
        Remove timestamps with missing data.
        """
        pass

    def filter_sensors(self, arg):
        """
        Drop suspicious measurments by comparing values from different sensors.
        """
        pass

    def regression(self, filter=False):
        """
        Performs the regression on the current data.
        Argument to use regression as filter, default is False.
        If used as filter must provide filter udpate.
        statsmodesl patsy formulas cannot have spaces in column var/col names
        -possibly create temporary dataframe within method from the var method
        """
        pass

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


def std_filter(series, std_devs=3):
    mean = series.mean()
    std = series.std()
    min_bound = mean - std * std_devs
    max_bound = mean + std * std_devs
    return all(series.apply(lambda x: min_bound < x < max_bound))


def sensor_filter(df, perc_diff):
    if df.shape[1] > 2:
        return df[df.apply(std_filter, axis=1)].index
    elif df.shape[1] == 1:
        return df.index
    else:
        sens_1 = df.iloc[:, 0]
        sens_2 = df.iloc[:, 1]
        return df[abs((sens_1 - sens_2) / sens_1) < perc_diff].index


def apply_filter(pm, skip_strs=[], perc_diff=0.05):
    """
    pm - pecos object
    skip_strs - (list) strings to search for in column label; if found skip col
    """
    trans_keys = list(pm.trans.keys())
    trans_keys.sort()
    trans_keys
    # labels = list(set(df.columns.get_level_values(col_level).tolist()))
    for i, label in enumerate(trans_keys):
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
            # print(label)
            # print(pm.df[pm.trans[label]].head(1))
            next_index = sensor_filter(pm.df[pm.trans[label]], perc_diff)
            index = index.intersection(next_index)
        else:
            # print(label)
            # print(pm.df[pm.trans[label]].head(1))
            index = sensor_filter(pm.df[pm.trans[label]], perc_diff)
    return pm.df.loc[index, :]
