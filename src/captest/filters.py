"""Filter step classes and row-filter helper functions.

This module is imported one-way by ``capdata.py``; it never imports
``capdata``. Filter steps touch a ``CapData`` instance only through the
runtime ``capdata`` argument to ``run``/``_execute``.
"""

from itertools import combinations

import pandas as pd


def perc_difference(x, y):
    """Calculate percent difference of two values."""
    if x == y == 0:
        return 0
    else:
        if x + y == 0:
            return 1
        else:
            return abs(x - y) / ((x + y) / 2)


def check_all_perc_diff_comb(series, perc_diff):
    """
    Check series for pairs of values with percent difference above perc_diff.

    Calculates the percent difference between all combinations of two values in
    the passed series and checks if all of them are below the passed perc_diff.

    Parameters
    ----------
    series : pd.Series
        Pandas series of values to check.
    perc_diff : float
        Percent difference threshold value as decimal i.e. 5% is 0.05.

    Returns
    -------
    bool
    """
    c = combinations(series.__iter__(), 2)
    return all([perc_difference(x, y) < perc_diff for x, y in c])


def abs_diff_from_average(series, threshold):
    """Check each value in series <= average of other values.

    Drops NaNs from series before calculating difference from average for each value.

    Returns True if there is only one value in the series.

    Parameters
    ----------
    series : pd.Series
        Pandas series of values to check.
    threshold : numeric
        Threshold value for absolute difference from average.

    Returns
    -------
    bool
    """
    series = series.dropna()
    if len(series) == 1:
        return True
    abs_diffs = []
    for i, val in enumerate(series):
        abs_diffs.append(abs(val - series.drop(series.index[i]).mean()) <= threshold)
    return all(abs_diffs)


def sensor_filter(df, threshold, row_filter=check_all_perc_diff_comb):
    """
    Check dataframe for rows with inconsistent values.

    Applies check_all_perc_diff_comb function along rows of passed dataframe.

    Parameters
    ----------
    df : pandas DataFrame
    perc_diff : float
        Percent difference as decimal.
    """
    if df.shape[1] >= 2:
        bool_ser = df.apply(row_filter, args=(threshold,), axis=1)
        return df[bool_ser].index
    elif df.shape[1] == 1:
        return df.index


def filter_irr(df, irr_col, low, high, ref_val=None):
    """
    Top level filter on irradiance values.

    Parameters
    ----------
    df : DataFrame
        Dataframe to be filtered.
    irr_col : str
        String that is the name of the column with the irradiance data.
    low : float or int
        Minimum value as fraction (0.8) or absolute 200 (W/m^2)
    high : float or int
        Max value as fraction (1.2) or absolute 800 (W/m^2)
    ref_val : float or int
        Must provide arg when low/high are fractions

    Returns
    -------
    DataFrame
    """
    if ref_val is not None:
        low *= ref_val
        high *= ref_val

    return df.loc[(df[irr_col] >= low) & (df[irr_col] <= high), :]


def filter_grps(grps, rcs, irr_col, low, high, freq, **kwargs):
    """
    Apply irradiance filter around passsed reporting irradiances to groupby.

    For each group in the grps argument the irradiance is filtered by a
    percentage around the reporting irradiance provided in rcs.

    Parameters
    ----------
    grps : pandas groupby
        Groupby object with time groups (months, seasons, etc.).
    rcs : pandas DataFrame
        Dataframe of reporting conditions.  Use the rep_cond method to generate
        a dataframe for this argument.
    irr_col : str
        String that is the name of the column with the irradiance data.
    low : float
        Minimum value as fraction e.g. 0.8.
    high : float
        Max value as fraction e.g. 1.2.
    freq : str
        Frequency to groupby e.g. 'MS' for month start.
    **kwargs
        Passed to pandas Grouper to control label and closed side of intervals.
        See pandas Grouper doucmentation for details. Default is left labeled
        and left closed.

    Returns
    -------
    pandas groupby
    """
    flt_dfs = []
    for grp_name, grp_df in grps:
        ref_val = rcs.loc[grp_name, "poa"]
        grp_df_flt = filter_irr(grp_df, irr_col, low, high, ref_val=ref_val)
        flt_dfs.append(grp_df_flt)
    df_flt = pd.concat(flt_dfs)
    df_flt_grpby = df_flt.groupby(pd.Grouper(freq=freq, **kwargs))
    return df_flt_grpby
