import os
import numpy as np
import pandas as pd
import dateutil
import datetime
import re


def load_das_file(path, filename):
    header_end = 1

    data = os.path.normpath(path + filename)
    all_data = pd.read_csv(data, encoding="UTF-8", header=[0, header_end],
                           index_col=0, parse_dates=True, skip_blank_lines=True,
                           low_memory=False)

    if not isinstance(all_data.index[0], pd.tslib.Timestamp):
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

    return(all_data)


def load_data(dir='./data/'):
    files_to_read = []
    for file in os.listdir(dir):
        if file.endswith('.csv'):
            files_to_read.append(file)

    all_sensors = pd.DataFrame()
    for filename in files_to_read:
        print("Read: " + filename)
        nextData = load_das_file(dir, filename)
        all_sensors = pd.concat([all_sensors, nextData], axis=0)
    return all_sensors


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
             'real_pwr': [['real power', 'ac power', 'power'],
                          (aux_load, ac_nameplate * 1.05)]}

sub_type_defs = {'poa': [['plane of array', 'poa']],
                 'ghi': [['global horizontal', 'ghi', 'global', 'glob']],
                 'amb': [['ambient', 'amb']],
                 'mod': [['module', 'mod']],
                 'mtr': [['revenue meter', 'rev meter', 'billing meter', ' meter']],
                 'inv': [['inverter', 'inv']]}

irr_sensors_defs = {'ref_cell': [['reference cell', 'reference', 'ref',
                                  'referance', 'pvel']],
                    'pyran': [['pyranometer', 'pyran']]}


def series_type(series, type_defs, bounds_check=True, warnings=False):
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


def trans_dict(df):
    col_types = df.apply(series_type, args=(type_defs,)).tolist()
    sub_types = df.apply(series_type, args=(sub_type_defs,),
                         bounds_check=False).tolist()
    irr_types = df.apply(series_type, args=(irr_sensors_defs,),
                         bounds_check=False).tolist()

    col_indices = []
    for typ, sub_typ, irr_typ in zip(col_types, sub_types, irr_types):
        col_indices.append('-'.join([typ, sub_typ, irr_typ]))

    names = []
    for new_name, old_name in zip(col_indices, df.columns.tolist()):
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

    return trans


def std_filter(series):
    mean = series.mean()
    std = series.std()
    min_bound = mean - std * 2
    max_bound = mean + std * 2
    return all(series.apply(lambda x: min_bound < x < max_bound))


def sensor_filter(df, perc_diff):
    if df.shape[1] > 2:
        return df[df.apply(std_filter, axis=1)].index
    elif df.shape[1] == 1:
        return df.index
    else:
        sens_1 = df.iloc[:, 0]
        sens_2 = df.iloc[:, 1]
        return df[(sens_1 - sens_2) / sens_1 < perc_diff].index


def apply_filter(df, col_level, skip_strs=[]):
    """
    df - dataframe to apply filter to
    col_level - (str) label of multiindex level to get column names from
    skip_strs - (list) strings to search for in column label; if found skip col
    """
    labels = list(set(df.columns.get_level_values(col_level).tolist()))
    for i, label in enumerate(labels):
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
            next_index = sensor_filter(df[label], 0.05)
            index = index.intersection(next_index)
        else:
            # print(label)
            index = sensor_filter(df[label], 0.05)
    return df.loc[index, :]
