import os
import numpy as np
import pandas as pd
import dateutil
import datetime
import re


def load_das_data(path, filename):
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
