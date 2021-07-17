import numpy as np
import pandas as pd
from scipy import stats


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
    common_timestep = stats.mode(np.diff(data.index.values))[0][0]
    common_timestep_tdelta = common_timestep.astype('timedelta64[m]')
    freq = common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        try:
            return str(int(freq)) + units_abbrev[units]
        except:
            return str(freq) + units_abbrev[units]
    else:
        return freq
