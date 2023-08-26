import json
import yaml
import numpy as np
import pandas as pd
from scipy import stats


def read_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_common_timestep(data, units='m', string_output=True):
    """
    Get the most commonly occuring timestep of data as frequency string.

    Parameters
    ----------
    data : Series or DataFrame
        Data with a DateTimeIndex.
    units : str, default 'm'
        String representing date/time unit, such as (D)ay, (M)onth, (Y)ear,
        (h)ours, (m)inutes, or (s)econds.
    string_output : bool, default True
        Set to False to return a numeric value.

    Returns
    -------
    str or numeric
        If the `string_output` is True and the most common timestep is an integer
        in the specified units then a valid pandas frequency or offset alias is
        returned.
        If `string_output` is false, then a numeric value is returned.
    """
    units_abbrev = {
        'D': 'D',
        'M': 'M',
        'Y': 'Y',
        'h': 'H',
        'm': 'min',
        's': 'S'
    }
    common_timestep = data.index.to_series().diff().mode().values[0]
    common_timestep_tdelta = common_timestep.astype('timedelta64[m]')
    freq = common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        try:
            return str(int(freq)) + units_abbrev[units]
        except:
            return str(freq) + units_abbrev[units]
    else:
        return freq

def reindex_datetime(data, report=False, add_index_col=True):
    """
    Find dataframe index frequency and reindex to add any missing intervals.

    Sorts index of passed dataframe before reindexing.

    Parameters
    ----------
    data : DataFrame
        DataFrame to be reindexed.

    Returns
    -------
    Reindexed DataFrame
    """
    data_index_length = data.shape[0]
    df = data.copy()
    df.sort_index(inplace=True)

    freq_str = get_common_timestep(data, string_output=True)
    full_ix = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq_str)
    df = df.reindex(index=full_ix)
    df_index_length = df.shape[0]
    missing_intervals = df_index_length - data_index_length

    if add_index_col:
        ix_ser = df.index.to_series()
        df['index'] = ix_ser.apply(lambda x: x.strftime('%m/%d/%Y %H %M'))

    if report:
        print('Frequency determined to be ' + freq_str + ' minutes.')
        print('{:,} intervals added to index.'.format(missing_intervals))
        print('')

    return df, missing_intervals, freq_str

def generate_irr_distribution(
    lowest_irr,
    highest_irr,
    rng=np.random.default_rng(82)
):
    """
    Create a list of increasing values similar to POA irradiance data.

    Default parameters result in increasing values where the difference
    between each subsquent value is randomly chosen from the typical range
    of steps for a POA tracker.

    Parameters
    ----------
    lowest_irr : numeric
        Lowest value in the list of values returned.
    highest_irr : numeric
        Highest value in the list of values returned.
    rng : Numpy Random Generator
        Instance of the default Generator.

    Returns
    -------
    irr_values : list
    """
    irr_values = [lowest_irr, ]
    possible_steps = (
        rng.integers(1, high=8, size=10000)
        + rng.random(size=10000)
        - 1
    )
    below_max = True
    while below_max:
        next_val = irr_values[-1] + rng.choice(possible_steps, replace=False)
        if next_val >= highest_irr:
            below_max = False
        else:
            irr_values.append(next_val)
    return irr_values
