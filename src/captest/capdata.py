"""
Provides the CapData class and supporting functions.

The CapData class provides methods for loading, filtering, and regressing solar
data.  A capacity test following the ASTM standard can be performed using a
CapData object for the measured data and a seperate CapData object for the
modeled data. The get_summary and captest_results functions accept two CapData
objects as arguments and provide a summary of the data filtering steps and
the results of the capacity test, respectively.
"""
# standard library imports
import os
from pathlib import Path
import re
import datetime
import copy
from functools import wraps
from itertools import combinations
import warnings
import pytz
import importlib

# anaconda distribution defaults
import dateutil
import numpy as np
import pandas as pd

# anaconda distribution defaults
# statistics and machine learning imports
import statsmodels.formula.api as smf
from scipy import stats
# from sklearn.covariance import EllipticEnvelope
import sklearn.covariance as sk_cv

# anaconda distribution defaults
# visualization library imports
import matplotlib.pyplot as plt
import colorcet as cc
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from bokeh.models import Legend, HoverTool, ColumnDataSource

import param

# visualization library imports
hv_spec = importlib.util.find_spec('holoviews')
if hv_spec is not None:
    import holoviews as hv
    from holoviews.plotting.links import DataLink
    from holoviews import opts
    hv.extension('bokeh')
else:
    warnings.warn('Some plotting functions will not work without the '
                  'holoviews package.')

pn_spec = importlib.util.find_spec('panel')
if pn_spec  is not None:
    import panel as pn
    pn.extension()
else:
    warnings.warn(
        'The ReportingIrradiance.dashboard method will not work without '
        'the panel package.'
    )

xlsx_spec = importlib.util.find_spec('openpyxl')
if xlsx_spec is None:
    warnings.warn(
        'Specifying a column grouping in an excel file will not work without '
        'the openpyxl package.'
    )

# pvlib imports
pvlib_spec = importlib.util.find_spec('pvlib')
if pvlib_spec is not None:
    from pvlib.location import Location
    from pvlib.pvsystem import (
        PVSystem, Array, FixedMount, SingleAxisTrackerMount
    )
    from pvlib.pvsystem import retrieve_sam
    from pvlib.modelchain import ModelChain
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn('Clear sky functions will not work without the '
                  'pvlib package.')

from captest import util

plot_colors_brewer = {'real_pwr': ['#2b8cbe', '#7bccc4', '#bae4bc', '#f0f9e8'],
                      'irr_poa': ['#e31a1c', '#fd8d3c', '#fecc5c', '#ffffb2'],
                      'irr_ghi': ['#91003f', '#e7298a', '#c994c7', '#e7e1ef'],
                      'temp_amb': ['#238443', '#78c679', '#c2e699', '#ffffcc'],
                      'temp_mod': ['#88419d', '#8c96c6', '#b3cde3', '#edf8fb'],
                      'wind': ['#238b45', '#66c2a4', '#b2e2e2', '#edf8fb']}

met_keys = ['poa', 't_amb', 'w_vel', 'power']


columns = ['pts_after_filter', 'pts_removed', 'filter_arguments']


def round_kwarg_floats(kwarg_dict, decimals=3):
    """
    Round float values in a dictionary.

    Parameters
    ----------
    kwarg_dict : dict
    decimals : int, default 3
        Number of decimal places to round to.

    Returns
    -------
    dict
        Dictionary with rounded floats.
    """
    rounded_vals = []
    for val in kwarg_dict.values():
        if isinstance(val, float):
            rounded_vals.append(round(val, decimals))
        else:
            rounded_vals.append(val)
    return {key: val for key, val in zip(kwarg_dict.keys(), rounded_vals)}


def tstamp_kwarg_to_strings(kwarg_dict):
    """
    Convert timestamp values in dictionary to strings.

    Parameters
    ----------
    kwarg_dict : dict

    Returns
    -------
    dict
    """
    output_vals = []
    for val in kwarg_dict.values():
        if isinstance(val, pd.Timestamp):
            output_vals.append(val.strftime('%Y-%m-%d %H:%M'))
        else:
            output_vals.append(val)
    return {key: val for key, val in zip(kwarg_dict.keys(), output_vals)}


def update_summary(func):
    """
    Decoratates the CapData class filter methods.

    Updates the CapData.summary and CapData.summary_ix attributes, which
    are used to generate summary data by the CapData.get_summary method.

    Todo
    ----
    not in place
        Check if summary is updated when function is called with inplace=False.
        It should not be.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        pts_before = self.data_filtered.shape[0]
        ix_before = self.data_filtered.index
        if pts_before == 0:
            pts_before = self.data.shape[0]
            self.summary_ix.append((self.name, 'count'))
            self.summary.append({columns[0]: pts_before,
                                 columns[1]: 0,
                                 columns[2]: 'no filters'})

        ret_val = func(self, *args, **kwargs)

        arg_str = args.__repr__()
        lst = arg_str.split(',')
        arg_lst = [item.strip("()") for item in lst]
        arg_lst_one = arg_lst[0]
        if arg_lst_one == 'das' or arg_lst_one == 'sim':
            arg_lst = arg_lst[1:]
        arg_str = ', '.join(arg_lst)

        func_re = re.compile('<function (.*) at', re.IGNORECASE)
        if func_re.search(arg_str) is not None:
            custom_func_name = func_re.search(arg_str).group(1)
            arg_str = re.sub("<function.*>", custom_func_name, arg_str)

        kwargs = round_kwarg_floats(kwargs)
        kwargs = tstamp_kwarg_to_strings(kwargs)
        kwarg_str = kwargs.__repr__()
        kwarg_str = kwarg_str.strip('{}')
        kwarg_str = kwarg_str.replace("'", "")

        if len(arg_str) == 0 and len(kwarg_str) == 0:
            arg_str = 'Default arguments'
        elif len(arg_str) == 0:
            arg_str = kwarg_str
        else:
            arg_str = arg_str + ', ' + kwarg_str

        filter_name = func.__name__
        if filter_name in self.filter_counts.keys():
            filter_name_enum = filter_name + '-' + str(self.filter_counts[filter_name])
            self.filter_counts[filter_name] += 1
        else:
            self.filter_counts[filter_name] = 1
            filter_name_enum = filter_name

        pts_after = self.data_filtered.shape[0]
        pts_removed = pts_before - pts_after
        self.summary_ix.append((self.name, filter_name_enum))
        self.summary.append({columns[0]: pts_after,
                             columns[1]: pts_removed,
                             columns[2]: arg_str})

        ix_after = self.data_filtered.index
        self.removed.append({
            'name': filter_name_enum,
            'index': ix_before.difference(ix_after)
        })
        self.kept.append({
            'name': filter_name_enum,
            'index': ix_after
        })

        if pts_after == 0:
            warnings.warn('The last filter removed all data! '
                          'Calling additional filtering or visualization '
                          'methods that reference the data_filtered attribute '
                          'will raise an error.')

        return ret_val
    return wrapper


def wrap_year_end(df, start, end):
    """
    Shifts data before or after new year to form a contigous time period.

    This function shifts data from the end of the year a year back or data from
    the begining of the year a year forward, to create a contiguous time
    period. Intended to be used on historical typical year data.

    If start date is in dataframe, then data at the beginning of the year will
    be moved ahead one year.  If end date is in dataframe, then data at the end
    of the year will be moved back one year.

    cntg (contiguous); eoy (end of year)

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe to be adjusted.
    start: pandas Timestamp
        Start date for time period.
    end: pandas Timestamp
        End date for time period.

    Todo
    ----
    Need to test and debug this for years not matching.
    """
    if df.index[0].year == start.year:
        df_start = df.loc[start:, :]

        df_end = df.copy()
        df_end.index = df_end.index + pd.DateOffset(days=365)
        df_end = df_end.loc[:end, :]

    elif df.index[0].year == end.year:
        df_end = df.loc[:end, :]

        df_start = df.copy()
        df_start.index = df_start.index - pd.DateOffset(days=365)
        df_start = df_start.loc[start:, :]

    df_return = pd.concat([df_start, df_end], axis=0)
    ix_series = df_return.index.to_series()
    df_return['index'] = ix_series.apply(lambda x: x.strftime('%m/%d/%Y %H %M'))  # noqa E501
    return df_return


def spans_year(start_date, end_date):
    """
    Determine if dates passed are in the same year.

    Parameters
    ----------
    start_date: pandas Timestamp
    end_date: pandas Timestamp

    Returns
    -------
    bool
    """
    if start_date.year != end_date.year:
        return True
    else:
        return False


def wrap_seasons(df, freq):
    """
    Rearrange an 8760 so a quarterly groupby will result in seasonal groups.

    Parameters
    ----------
    df : DataFrame
        Dataframe to be rearranged.
    freq : str
        String pandas offset alias to specify aggregattion frequency
        for reporting condition calculation.

    Returns
    -------
    DataFrame

    Todo
    ----
    Write unit test
    BQ-NOV vs BQS vs QS
        Need to review if BQ is the correct offset alias vs BQS or QS.
    """
    check_freqs = ['BQ-JAN', 'BQ-FEB', 'BQ-APR', 'BQ-MAY', 'BQ-JUL',
                   'BQ-AUG', 'BQ-OCT', 'BQ-NOV']
    month_int = {'JAN': 1, 'FEB': 2, 'APR': 4, 'MAY': 5, 'JUL': 7,
                 'AUG': 8, 'OCT': 10, 'NOV': 11}

    if freq in check_freqs:
        warnings.warn('DataFrame index adjusted to be continous through new'
                      'year, but not returned or set to attribute for user.'
                      'This is not an issue if using RCs with'
                      'predict_capacities.')
        if isinstance(freq, str):
            month = month_int[freq.split('-')[1]]
        else:
            month = freq.startingMonth
        year = df.index[0].year
        months_year_end = 12 - month
        months_year_start = 3 - months_year_end
        if int(month) >= 10:
            str_date = str(months_year_start) + '/' + str(year)
        else:
            str_date = str(month) + '/' + str(year)
        tdelta = df.index[1] - df.index[0]
        date_to_offset = df.loc[str_date].index[-1].to_pydatetime()
        start = date_to_offset + tdelta
        end = date_to_offset + pd.DateOffset(years=1)
        if month < 8 or month >= 10:
            df = wrap_year_end(df, start, end)
        else:
            df = wrap_year_end(df, end, start)
        return df
    else:
        return df


def perc_wrap(p):
    """Wrap numpy percentile function for use in rep_cond method."""
    def numpy_percentile(x):
        return np.percentile(x.T, p, interpolation='nearest')
    return numpy_percentile


def perc_bounds(percent_filter):
    """
    Convert +/- percentage to decimals to be used to determine bounds.

    Parameters
    ----------
    percent_filter : float or tuple, default None
        Percentage or tuple of percentages used to filter around reporting
        irradiance in the irr_rc_balanced function.  Required argument when
        irr_bal is True.

    Returns
    -------
    tuple
        Decimal versions of the percent irradiance filter. 0.8 and 1.2 would be
        returned when passing 20 to the input.
    """
    if isinstance(percent_filter, tuple):
        perc_low = percent_filter[0] / 100
        perc_high = percent_filter[1] / 100
    else:
        perc_low = percent_filter / 100
        perc_high = percent_filter / 100
    low = 1 - (perc_low)
    high = 1 + (perc_high)
    return (low, high)


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


def sensor_filter(df, perc_diff):
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
        bool_ser = df.apply(check_all_perc_diff_comb, perc_diff=perc_diff,
                            axis=1)
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

    df_renamed = df.rename(columns={irr_col: 'poa'})

    flt_str = '@low <= ' + 'poa' + ' <= @high'
    indx = df_renamed.query(flt_str).index

    return df.loc[indx, :]


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
        ref_val = rcs.loc[grp_name, 'poa']
        grp_df_flt = filter_irr(grp_df, irr_col, low, high, ref_val=ref_val)
        flt_dfs.append(grp_df_flt)
    df_flt = pd.concat(flt_dfs)
    df_flt_grpby = df_flt.groupby(pd.Grouper(freq=freq, **kwargs))
    return df_flt_grpby


class ReportingIrradiance(param.Parameterized):
    df = param.DataFrame(
        doc='Data to use to calculate reporting irradiance.',
        precedence=-1)
    irr_col = param.String(
        default='GlobInc',
        doc="Name of column in `df` containing irradiance data.",
        precedence=-1)
    irr_rc = param.Number(precedence=-1)
    poa_flt = param.DataFrame(precedence=-1)
    total_pts = param.Number(precedence=-1)
    rc_irr_60th_perc = param.Number(precedence=-1)
    percent_band = param.Integer(20, softbounds=(2, 50), step=1)
    min_percent_below = param.Integer(
        default=40,
        doc='Minimum number of points as a percentage allowed below the \
        reporting irradiance.')
    max_percent_above = param.Integer(
        default=60,
        doc='Maximum number of points as a percentage allowed above the \
        reporting irradiance.')
    min_ref_irradiance = param.Integer(
        default=None,
        doc='Minimum value allowed for the reference irradiance.')
    max_ref_irradiance = param.Integer(None,
        doc='Maximum value allowed for the reference irradiance. By default this\
        maximum is calculated by dividing the highest irradiance value in `df`\
        by `high`.')
    points_required = param.Integer(
        default=750,
        doc='This is value is only used in the plot to overlay a horizontal \
        line on the plot of the total points.')

    def __init__(self, df, irr_col, **param):
        super().__init__(**param)
        self.df = df
        self.irr_col = irr_col
        self.rc_irr_60th_perc = np.percentile(self.df[self.irr_col], 60)

    def get_rep_irr(self):
        """
        Calculates the reporting irradiance.

        Returns
        -------
        Tuple
            Float reporting irradiance and filtered dataframe.
        """
        low, high = perc_bounds(self.percent_band)
        poa_flt = self.df.copy()

        poa_flt.sort_values(self.irr_col, inplace=True)

        poa_flt['plus_perc'] = poa_flt[self.irr_col] * high
        poa_flt['minus_perc'] = poa_flt[self.irr_col] * low


        poa_flt['below_count'] = [
            poa_flt[self.irr_col].between(low, ref).sum() for low, ref
            in zip(poa_flt['minus_perc'], poa_flt[self.irr_col])
        ]
        poa_flt['above_count'] = [
            poa_flt[self.irr_col].between(ref, high).sum() for ref, high
            in zip(poa_flt[self.irr_col], poa_flt['plus_perc'])
        ]

        poa_flt['total_pts'] = poa_flt['above_count'] + poa_flt['below_count']
        poa_flt['perc_above'] = (poa_flt['above_count'] / poa_flt['total_pts']) * 100
        poa_flt['perc_below'] =  (poa_flt['below_count'] / poa_flt['total_pts']) * 100

        # set index to the poa irradiance
        poa_flt.set_index(self.irr_col, inplace=True)

        if self.max_ref_irradiance is None:
            self.max_ref_irradiance = int(poa_flt.index[-1] / high)
        if self.min_ref_irradiance is None:
            self.min_ref_irradiance = int(poa_flt.index[0] / low)
        if self.min_ref_irradiance > self.max_ref_irradiance:
            warnings.warn(
                'The minimum reference irradiance ({:.2f}) is greater than the maximum '
                'reference irradiance ({:.2f}). Setting the minimum to 400 and the '
                'maximum to 1000.'.format(self.min_ref_irradiance, self.max_ref_irradiance)
            )
            self.min_ref_irradiance = 400
            self.max_ref_irradiance = 1000

        # determine ref irradiance by finding 50/50 irradiance in upper group of data
        poa_flt['valid'] = (
            poa_flt['perc_below'].between(
                self.min_percent_below, self.max_percent_above) &
            poa_flt.index.to_series().between(
                self.min_ref_irradiance, self.max_ref_irradiance)
        )
        if poa_flt['valid'].sum() == 0:
            self.poa_flt = poa_flt
            self.irr_rc = np.NaN
            warnings.warn(
                'No valid reference irradiance found. Try reviewing the min and max '
                'reference irradiance values and the min and max percent below and '
                'above values. The dashboard method will show these values with '
                'related plots and allow you to adjust them.'
            )
            return None
        poa_flt['perc_below_minus_50_abs'] = (poa_flt['perc_below'] - 50).abs()
        valid_df = poa_flt[poa_flt['valid']].copy()
        valid_df.sort_values('perc_below_minus_50_abs', inplace=True)
        # if there are more than one points that are exactly 50 points above and
        # 50 above then pick the one that results in the most points
        self.valid_df = valid_df
        fifty_fifty_points = valid_df['perc_below_minus_50_abs'] == 0
        if (fifty_fifty_points).sum() > 1:
            possible_points = poa_flt.loc[
                fifty_fifty_points[fifty_fifty_points].index,
                'total_pts'
            ]
            possible_points.sort_values(ascending=False, inplace=True)
            irr_RC = possible_points.index[0]
        else:
            irr_RC = valid_df.index[0]
        flt_df = filter_irr(self.df, self.irr_col, low, high, ref_val=irr_RC)
        self.irr_rc = irr_RC
        self.poa_flt = poa_flt
        self.total_pts = poa_flt.loc[self.irr_rc, 'total_pts']

        return (irr_RC, flt_df)


    def save_plot(self, output_plot_path=None):
        """
        Save a plot of the possible reporting irradiances and time intervals.

        Saves plot as an html file at path given.

        output_plot_path : str or Path
            Path to save plot to.
        """
        hv.save(
            self.plot(),
            output_plot_path,
            fmt='html',
            toolbar=True
        )

    def save_csv(self, output_csv_path):
        """
        Save possible reporting irradiance data to csv file at given path.
        """
        self.poa_flt.to_csv(output_csv_path)

    @param.depends('percent_band', 'min_percent_below', 'max_percent_above', 'min_ref_irradiance', 'points_required', 'max_ref_irradiance')
    def plot(self):
        self.get_rep_irr()
        below_count_scatter = hv.Scatter(
            self.poa_flt['below_count'].reset_index(), ['poa'], ['below_count'],
            label='Count pts below',
        )
        above_count_scatter = hv.Scatter(
            self.poa_flt['above_count'].reset_index(), ['poa'], ['above_count'],
            label='Count pts above',
        )
        if self.irr_rc is not np.NaN:
            count_ellipse = hv.Ellipse(
                self.irr_rc,
                self.poa_flt.loc[self.irr_rc, 'below_count'],
                (20, 50)
            )
        perc_below_scatter = (
            hv.Scatter(
                self.poa_flt['perc_below'].reset_index(), ['poa'], ['perc_below']
            ) *
            hv.HLine(self.min_percent_below) *
            hv.HLine(self.max_percent_above) *
            hv.VLine(self.min_ref_irradiance) *
            hv.VLine(self.max_ref_irradiance)
        )
        if self.irr_rc is not np.NaN:
            perc_ellipse = hv.Ellipse(
                self.irr_rc,
                self.poa_flt.loc[self.irr_rc, 'perc_below'],
                (20, 10)
            )
        total_points_scatter = (
            hv.Scatter(
                self.poa_flt['total_pts'].reset_index(), ['poa'], ['total_pts']
            ) *
            hv.HLine(self.points_required)
        )
        if self.irr_rc is not np.NaN:
            total_points_ellipse = hv.Ellipse(
                self.irr_rc,
                self.poa_flt.loc[self.irr_rc, 'total_pts'],
                (20, 50)
            )

        ylim_bottom = self.poa_flt['total_pts'].min() - 20
        if self.total_pts < self.points_required:
            ylim_top =  self.points_required + 20
        else:
            ylim_top = self.total_pts + 50
        vl = hv.VLine(self.rc_irr_60th_perc).opts(line_color='gray')
        if self.irr_rc is not np.NaN:
            rep_cond_plot = (
                (below_count_scatter * above_count_scatter * count_ellipse * vl).opts(ylabel='count points') +
                (perc_below_scatter * perc_ellipse).opts(ylim=(0, 100)) +
                (total_points_scatter * total_points_ellipse).opts(
                    ylim=(ylim_bottom, ylim_top))
            ).opts(
                opts.HLine(line_width=1),
                opts.VLine(line_width=1),
                opts.Scatter(
                     size=4, show_legend=True, legend_position='right', tools=['hover']
                ),
               opts.Overlay(width=700),
                opts.Layout(
                    title='Reporting Irradiance: {:0.2f}, Total Points {}'.format(
                        self.irr_rc,
                        self.total_pts)),
            ).cols(1)
        else:
            rep_cond_plot = (
                (below_count_scatter * above_count_scatter * vl).opts(ylabel='count points') +
                perc_below_scatter.opts(ylim=(0, 100)) +
                total_points_scatter.opts(
                    ylim=(ylim_bottom, ylim_top))
            ).opts(
                opts.HLine(line_width=1),
                opts.VLine(line_width=1),
                opts.Scatter(
                    size=4, show_legend=True, legend_position='right', tools=['hover']
                ),
                opts.Overlay(width=700),
                opts.Layout(
                    title='Reporting Irradiance: None identified, Total Points {}'.format(
                        self.total_pts)),
            ).cols(1)
        return rep_cond_plot

    def dashboard(self):
        return pn.Row(self.param, self.plot)


def fit_model(df, fml='power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'):  # noqa E501
    """
    Fits linear regression using statsmodels to dataframe passed.

    Dataframe must be first argument for use with pandas groupby object
    apply method.

    Parameters
    ----------
    df : pandas dataframe
    fml : str
        Formula to fit refer to statsmodels and patsy documentation for format.
        Default is the formula in ASTM E2848.

    Returns
    -------
    Statsmodels linear model regression results wrapper object.
    """
    mod = smf.ols(formula=fml, data=df)
    reg = mod.fit()
    return reg


def predict(regs, rcs):
    """
    Calculate predicted values for given linear models and predictor values.

    Evaluates the first linear model in the iterable with the first row of the
    predictor values in the dataframe.  Passed arguments must be aligned.

    Parameters
    ----------
    regs : iterable of statsmodels regression results wrappers
    rcs : pandas dataframe
        Dataframe of predictor values used to evaluate each linear model.
        The column names must match the strings used in the regression
        formuala.

    Returns
    -------
    Pandas series of predicted values.
    """
    pred_cap = list()
    for i, mod in enumerate(regs):
        RC_df = pd.DataFrame(rcs.iloc[i, :]).T
        pred_cap.append(mod.predict(RC_df).values[0])
    return pd.Series(pred_cap)


def pred_summary(grps, rcs, allowance, **kwargs):
    """
    Summarize reporting conditions, predicted cap, and gauranteed cap.

    This method does not calculate reporting conditions.

    Parameters
    ----------
    grps : pandas groupby object
        Solar data grouped by season or month used to calculate reporting
        conditions.  This argument is used to fit models for each group.
    rcs : pandas dataframe
        Dataframe of reporting conditions used to predict capacities.
    allowance : float
        Percent allowance to calculate gauranteed capacity from predicted
        capacity.

    Returns
    -------
    Dataframe of reporting conditions, model coefficients, predicted capacities
    gauranteed capacities, and points in each grouping.
    """
    regs = grps.apply(fit_model, **kwargs)
    predictions = predict(regs, rcs)
    params = regs.apply(lambda x: x.params.transpose())
    pt_qty = grps.agg('count').iloc[:, 0]
    predictions.index = pt_qty.index

    params.index = pt_qty.index
    rcs.index = pt_qty.index
    predictions.name = 'PredCap'

    for rc_col_name in rcs.columns:
        for param_col_name in params.columns:
            if rc_col_name == param_col_name:
                new_col_name = param_col_name + '-param'
                params.rename(columns={param_col_name: new_col_name},
                              inplace=True)

    results = pd.concat([rcs, predictions, params], axis=1)

    results['guaranteedCap'] = results['PredCap'] * (1 - allowance)
    results['pt_qty'] = pt_qty.values

    return results


def pvlib_location(loc):
    """
    Create a pvlib location object.

    Parameters
    ----------
    loc : dict
        Dictionary of values required to instantiate a pvlib Location object.

        loc = {'latitude': float,
               'longitude': float,
               'altitude': float/int,
               'tz': str, int, float, or pytz.timezone, default 'UTC'}
        See
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        pytz.timezone objects will be converted to strings.
        ints and floats must be in hours from UTC.

    Returns
    -------
    pvlib location object.
    """
    return Location(**loc)


def pvlib_system(sys):
    """
    Create a pvlib :py:class:`~pvlib.pvsystem.PVSystem` object.

    The :py:class:`~pvlib.pvsystem.PVSystem` will have either a
    :py:class:`~pvlib.pvsystem.FixedMount` or a
    :py:class:`~pvlib.pvsystem.SingleAxisTrackerMount` depending on
    the keys of the passed dictionary.

    Parameters
    ----------
    sys : dict
        Dictionary of keywords required to create a pvlib
        ``SingleAxisTrackerMount`` or ``FixedMount``, plus ``albedo``.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.

    Returns
    -------
    pvlib PVSystem object.
    """
    sandia_modules = retrieve_sam('SandiaMod')
    cec_inverters = retrieve_sam('cecinverter')
    sandia_module = sandia_modules.iloc[:, 0]
    cec_inverter = cec_inverters.iloc[:, 0]

    albedo = sys.pop('albedo', None)
    trck_kwords = ['axis_tilt', 'axis_azimuth', 'max_angle', 'backtrack', 'gcr']  # noqa: E501
    if any(kword in sys.keys() for kword in trck_kwords):
        mount = SingleAxisTrackerMount(**sys)
    else:
        mount = FixedMount(**sys)
    array = Array(mount, albedo=albedo, module_parameters=sandia_module,
                  temperature_model_parameters={'u_c': 29.0, 'u_v': 0.0})
    system = PVSystem(arrays=[array], inverter_parameters=cec_inverter)

    return system


def get_tz_index(time_source, loc):
    """
    Create DatetimeIndex with timezone aligned with location dictionary.

    Handles generating a DatetimeIndex with a timezone for use as an agrument
    to pvlib ModelChain prepare_inputs method or pvlib Location get_clearsky
    method.

    Parameters
    ----------
    time_source : dataframe or DatetimeIndex
        If passing a dataframe the index of the dataframe will be used.  If the
        index does not have a timezone the timezone will be set using the
        timezone in the passed loc dictionary. If passing a DatetimeIndex with
        a timezone it will be returned directly. If passing a DatetimeIndex
        without a timezone the timezone in the timezone dictionary will be
        used.

    Returns
    -------
    DatetimeIndex with timezone
    """
    if isinstance(time_source, pd.core.indexes.datetimes.DatetimeIndex):
        if time_source.tz is None:
            time_source = time_source.tz_localize(
                loc['tz'], ambiguous='infer', nonexistent='NaT'
            )
            return time_source
        else:
            if pytz.timezone(loc['tz']) != time_source.tz:
                warnings.warn('Passed a DatetimeIndex with a timezone that '
                              'does not match the timezone in the loc dict. '
                              'Using the timezone of the DatetimeIndex.')
            return time_source
    elif isinstance(time_source, pd.core.frame.DataFrame):
        if time_source.index.tz is None:
            return time_source.index.tz_localize(
                loc['tz'], ambiguous='infer', nonexistent='NaT'
            )
        else:
            if pytz.timezone(loc['tz']) != time_source.index.tz:
                warnings.warn('Passed a DataFrame with a timezone that '
                              'does not match the timezone in the loc dict. '
                              'Using the timezone of the DataFrame.')
            return time_source.index


def csky(time_source, loc=None, sys=None, concat=True, output='both'):
    """
    Calculate clear sky poa and ghi.

    Parameters
    ----------
    time_source : dataframe or DatetimeIndex
        If passing a dataframe the index of the dataframe will be used.  If the
        index does not have a timezone the timezone will be set using the
        timezone in the passed loc dictionary. If passing a DatetimeIndex with
        a timezone it will be returned directly. If passing a DatetimeIndex
        without a timezone the timezone in the timezone dictionary will
        be used.
    loc : dict
        Dictionary of values required to instantiate a pvlib Location object.

        loc = {'latitude': float,
               'longitude': float,
               'altitude': float/int,
               'tz': str, int, float, or pytz.timezone, default 'UTC'}
        See
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        pytz.timezone objects will be converted to strings.
        ints and floats must be in hours from UTC.
    sys : dict
        Dictionary of keywords required to create a pvlib
        :py:class:`~pvlib.pvsystem.SingleAxisTrackerMount` or
        :py:class:`~pvlib.pvsystem.FixedMount`.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.
    concat : bool, default True
        If concat is True then returns columns as defined by return argument
        added to passed dataframe, otherwise returns just clear sky data.
    output : str, default 'both'
        both - returns only total poa and ghi
        poa_all - returns all components of poa
        ghi_all - returns all components of ghi
        all - returns all components of poa and ghi
    """
    location = pvlib_location(loc)
    system = pvlib_system(sys)
    mc = ModelChain(system, location)
    times = get_tz_index(time_source, loc)
    ghi = location.get_clearsky(times=times)
    # pvlib get_Clearsky also returns 'wind_speed' and 'temp_air'
    mc.prepare_inputs(weather=ghi)
    cols = ['poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse',
            'poa_ground_diffuse']

    if output == 'both':
        csky_df = pd.DataFrame({
            'poa_mod_csky': mc.results.total_irrad['poa_global'],
            'ghi_mod_csky': ghi['ghi']
        })
    if output == 'poa_all':
        csky_df = mc.results.total_irrad[cols]
    if output == 'ghi_all':
        csky_df = ghi[['ghi', 'dni', 'dhi']]
    if output == 'all':
        csky_df = pd.concat([mc.results.total_irrad[cols],
                             ghi[['ghi', 'dni', 'dhi']]],
                            axis=1)

    ix_no_tz = csky_df.index.tz_localize(None, ambiguous='infer',
                                         nonexistent='NaT')
    csky_df.index = ix_no_tz

    if concat:
        if isinstance(time_source, pd.core.frame.DataFrame):
            try:
                df_with_csky = pd.concat([time_source, csky_df], axis=1)
            except pd.errors.InvalidIndexError:
                # Drop NaT that occur for March DST shift in US data
                df_with_csky = pd.concat(
                    [time_source, csky_df.loc[csky_df.index.dropna(), :]], axis=1
                )
            return df_with_csky
        else:
            warnings.warn('time_source is not a dataframe; only clear sky data\
                           returned')
            return csky_df
    else:
        return csky_df


def get_summary(*args):
    """
    Return summary dataframe of filtering steps for multiple CapData objects.

    See documentation for the CapData.get_summary method for additional
    details.
    """
    summaries = [cd.get_summary() for cd in args]
    return pd.concat(summaries)


def pick_attr(sim, das, name):
    """Check for conflict between attributes of two CapData objects."""
    sim_attr = getattr(sim, name)
    das_attr = getattr(das, name)
    if sim_attr is None and das_attr is None:
        warn_str = '{} must be set for either sim or das'.format(name)
        return warnings.warn(warn_str)
    elif sim_attr is None and das_attr is not None:
        return (das_attr, 'das')
    elif sim_attr is not None and das_attr is None:
        return (sim_attr, 'sim')
    elif sim_attr is not None and das_attr is not None:
        warn_str = ('{} found for sim and das set {} to None for one of '
                    'the two'.format(name, name))
        return warnings.warn(warn_str)


def determine_pass_or_fail(cap_ratio, tolerance, nameplate):
    """
    Determine a pass/fail result from a capacity ratio and test tolerance.

    Parameters
    ----------
    cap_ratio : float
        Ratio of the measured data regression result to the simulated data
        regression result.
    tolerance : str
        String representing error band.  Ex. '+/- 3' or '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    nameplate : numeric
        Nameplate rating of the PV plant.

    Returns
    -------
    tuple of boolean and string
        True for a passing test and false for a failing test.
        Limits for passing and failing test.
    """
    sign = tolerance.split(sep=' ')[0]
    error = int(tolerance.split(sep=' ')[1]) / 100

    nameplate_plus_error = nameplate * (1 + error)
    nameplate_minus_error = nameplate * (1 - error)

    if sign == '+/-' or sign == '-/+':
        return (round(np.abs(1 - cap_ratio), ndigits=6) <= error,
                str(nameplate_minus_error) + ', ' + str(nameplate_plus_error))
    elif sign == '-':
        return (cap_ratio >= 1 - error,
                str(nameplate_minus_error) + ', None')
    else:
        warnings.warn("Sign must be '-', '+/-', or '-/+'.")


def captest_results(sim, das, nameplate, tolerance, check_pvalues=False,
                    pval=0.05, print_res=True):
    """
    Print a summary indicating if system passed or failed capacity test.

    NOTE: Method will try to adjust for 1000x differences in units.

    Parameters
    ----------
    sim : CapData
        CapData object for simulated data.
    das : CapData
        CapData object for measured data.
    nameplate : numeric
        Nameplate rating of the PV plant.
    tolerance : str
        String representing error band.  Ex. +/- 3', '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    check_pvalues : boolean, default False
        Set to true to check p values for each coefficient.  If p values is
        greater than pval, then the coefficient is set to zero.
    pval : float, default 0.05
        p value to use as cutoff.  Regresion coefficients with a p value
        greater than pval will be set to zero.
    print_res : boolean, default True
        Set to False to prevent printing results.

    Returns
    -------
    Capacity test ratio - the capacity calculated from the reporting conditions
    and the measured data divided by the capacity calculated from the reporting
    conditions and the simulated data.
    """
    sim_int = sim.copy()
    das_int = das.copy()

    if sim_int.regression_formula != das_int.regression_formula:
        return warnings.warn('CapData objects do not have the same'
                             'regression formula.')

    if check_pvalues:
        for cd in [sim_int, das_int]:
            for key, val in cd.regression_results.pvalues.items():
                if val > pval:
                    cd.regression_results.params[key] = 0

    rc = pick_attr(sim_int, das_int, 'rc')
    if print_res:
        print('Using reporting conditions from {}. \n'.format(rc[1]))
    rc = rc[0]

    actual = das_int.regression_results.predict(rc)[0]
    expected = sim_int.regression_results.predict(rc)[0]
    cap_ratio = actual / expected
    if cap_ratio < 0.01:
        cap_ratio *= 1000
        actual *= 1000
        warnings.warn('Capacity ratio and actual capacity multiplied by 1000'
                      ' because the capacity ratio was less than 0.01.')
    capacity = nameplate * cap_ratio

    if print_res:
        test_passed = determine_pass_or_fail(cap_ratio, tolerance, nameplate)
        print_results(test_passed, expected, actual, cap_ratio, capacity,
                      test_passed[1])

    return(cap_ratio)


def print_results(test_passed, expected, actual, cap_ratio, capacity, bounds):
    """Print formatted results of capacity test."""
    if test_passed[0]:
        print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
    else:
        print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))

    print("{:<30s}{:0.3f}".format("Modeled test output:",
                                  expected) + "\n" +
          "{:<30s}{:0.3f}".format("Actual test output:",
                                  actual) + "\n" +
          "{:<30s}{:0.3f}".format("Tested output ratio:",
                                  cap_ratio) + "\n" +
          "{:<30s}{:0.3f}".format("Tested Capacity:",
                                  capacity)
          )

    print("{:<30s}{}\n\n".format("Bounds:", test_passed[1]))


def highlight_pvals(s):
    """Highlight vals greater than or equal to 0.05 in a Series yellow."""
    is_greaterthan = s >= 0.05
    return ['background-color: yellow' if v else '' for v in is_greaterthan]


def captest_results_check_pvalues(sim, das, nameplate, tolerance,
                                  print_res=False, **kwargs):
    """
    Print a summary of the capacity test results.

    Capacity ratio is the capacity calculated from the reporting conditions
    and the measured data divided by the capacity calculated from the reporting
    conditions and the simulated data.

    The tolerance is applied to the capacity test ratio to determine if the
    test passes or fails.

    Parameters
    ----------
    sim : CapData
        CapData object for simulated data.
    das : CapData
        CapData object for measured data.
    nameplate : numeric
        Nameplate rating of the PV plant.
    tolerance : str
        String representing error band.  Ex. '+ 3', '+/- 3', '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    print_res : boolean, default True
        Set to False to prevent printing results.
    **kwargs
        kwargs are passed to captest_results.  See documentation for
        captest_results for options. check_pvalues is set in this method,
        so do not pass again.

    Prints:
    Capacity ratio without setting parameters with high p-values to zero.
    Capacity ratio after setting paramters with high p-values to zero.
    P-values for simulated and measured regression coefficients.
    Regression coefficients (parameters) for simulated and measured data.
    """
    das_pvals = das.regression_results.pvalues
    sim_pvals = sim.regression_results.pvalues
    das_params = das.regression_results.params
    sim_params = sim.regression_results.params

    df_pvals = pd.DataFrame([das_pvals, sim_pvals, das_params, sim_params])
    df_pvals = df_pvals.transpose()
    df_pvals.rename(columns={0: 'das_pvals', 1: 'sim_pvals',
                             2: 'das_params', 3: 'sim_params'}, inplace=True)

    cap_ratio = captest_results(sim, das, nameplate, tolerance,
                                print_res=print_res, check_pvalues=False,
                                **kwargs)
    cap_ratio_check_pvalues = captest_results(sim, das, nameplate, tolerance,
                                              print_res=print_res,
                                              check_pvalues=True, **kwargs)

    cap_ratio_rounded = np.round(cap_ratio, decimals=4) * 100
    cap_ratio_check_pvalues_rounded = np.round(cap_ratio_check_pvalues,
                                               decimals=4) * 100

    result_str = '{:.3f}% - Cap Ratio'
    print(result_str.format(cap_ratio_rounded))

    result_str_pval_check = '{:.3f}% - Cap Ratio after pval check'
    print(result_str_pval_check.format(cap_ratio_check_pvalues_rounded))

    return(df_pvals.style.format('{:20,.5f}').apply(highlight_pvals,
                                                    subset=['das_pvals',
                                                            'sim_pvals']))


def run_test(cd, steps):
    """
    Apply a list of capacity test steps to a given CapData object.

    A list of CapData methods is applied sequentially with the passed
    parameters.  This method allows succintly defining a capacity test,
    which facilitates parametric and automatic testing.

    Parameters
    ----------
    cd : CapData
        The CapData methods will be applied to this instance of the pvcaptest
        CapData class.
    steps : list of tuples
        A list of the methods to be applied and the arguments to be used.
        Each item in the list should be a tuple of the CapData method followed
        by a tuple of arguments and a dictionary of keyword arguments. If
        there are not args or kwargs an empty tuple or dict should be included.
        Example: [(CapData.filter_irr, (400, 1500), {})]
    """
    for step in steps:
        step[0](cd, *step[1], **step[2])


def overlay_scatters(measured, expected, expected_label='PVsyst'):
    """
    Plot labeled overlay scatter of final filtered measured and simulated data.

    Parameters
    ----------
    measured : Overlay
        Holoviews overlay scatter plot produced from CapData object used to
        calculate reporting conditions.
    expected : Overlay
        Holoviews overlay scatter plot produced from CapData object not used to
        calculate reporting conditions.
    rcs_from_meas : bool
        If rest was run calculating reporting conditions from measured or
        simulated data.

    Returns
    -------
    Overlay scatter plot of remaining data after filtering from measured and
    simulated data.
    """
    meas_last_filter_scatter = getattr(
        measured.Scatter,
        measured.Scatter.children[-1]
    ).relabel('Measured')
    exp_last_filter_scatter = getattr(
        expected.Scatter,
        expected.Scatter.children[-1]
    ).relabel(expected_label)
    overlay = (
        meas_last_filter_scatter * exp_last_filter_scatter
    ).opts(hv.opts.Overlay(legend_position='right'))
    return overlay


def index_capdata(capdata, label, filtered=True):
    if filtered:
        data = capdata.data_filtered
    else:
        data = capdata.data
    if isinstance(label, str):
        if label in capdata.column_groups.keys():
            return data[capdata.column_groups[label]]
        elif label in capdata.regression_cols.keys():
            return data[capdata.column_groups[capdata.regression_cols[label]]]
        elif label in data.columns:
            return data.loc[:, label]
    elif isinstance(label, list):
        cols_to_return = []
        for l in label:
            if l in capdata.column_groups.keys():
                cols_to_return.extend(capdata.column_groups[l])
            elif l in capdata.regression_cols.keys():
                col_or_grp = capdata.regression_cols[l]
                if col_or_grp in capdata.column_groups.keys():
                    cols_to_return.extend(capdata.column_groups[col_or_grp])
                elif col_or_grp in data.columns:
                    cols_to_return.append(col_or_grp)
            elif l in data.columns:
                cols_to_return.append(l)
        return data[cols_to_return]


class LocIndexer(object):
    """
    Class to implement __getitem__ for indexing the CapData.data dataframe.

    Allows passing a column_groups key, a list of column_groups keys, or a column or
    list of columns of the CapData.data dataframe.
    """
    def __init__(self, _capdata):
        self._capdata = _capdata

    def __getitem__(self, label):
        return index_capdata(self._capdata, label, filtered=False)


class FilteredLocIndexer(object):
    """
    Class to implement __getitem__ for indexing the CapData.data_filtered dataframe.

    Allows passing a column_groups key, a list of column_groups keys, or a column or
    list of columns of the CapData.data_filtered dataframe.
    """
    def __init__(self, _capdata):
        self._capdata = _capdata

    def __getitem__(self, label):
        return index_capdata(self._capdata, label, filtered=True)


class CapData(object):
    """
    Class to store capacity test data and translation of column names.

    CapData objects store a pandas dataframe of measured or simulated data
    and a dictionary used grouping columns by type of measurement.

    The `column_groups` dictionary allows maintaining the original column names
    while also grouping measurements of the same type from different
    sensors.  Many of the methods for plotting and filtering data rely on the
    column groupings to streamline user interaction.

    Parameters
    ----------
    name : str
        Name for the CapData object.
    data : pandas dataframe
        Used to store measured or simulated data imported from csv.
    data_filtered : pandas dataframe
        Holds filtered data.  Filtering methods act on and write to this
        attribute.
    column_groups : dictionary
        Assigned by the `group_columns` method, which attempts to infer the
        type of measurement recorded in each column of the dataframe stored in
        the `data` attribute.  For each inferred measurement type,
        `group_columns` creates an abbreviated name and a list of columns that
        contain measurements of that type. The abbreviated names are the keys
        and the corresponding values are the lists of columns.
    trans_keys : list
        Simply a list of the `column_groups` keys.
    regression_cols : dictionary
        Dictionary identifying which columns in `data` or groups of columns as
        identified by the keys of `column_groups` are the independent variables
        of the ASTM Capacity test regression equation. Set using
        `set_regression_cols` or by directly assigning a dictionary.
    trans_abrev : dictionary
        Enumerated translation dict keys mapped to original column names.
        Enumerated translation dict keys are used in plot hover tooltip.
    col_colors : dictionary
        Original column names mapped to a color for use in plot function.
    summary_ix : list of tuples
        Holds the row index data modified by the update_summary decorator
        function.
    summary : list of dicts
        Holds the data modified by the update_summary decorator function.
    rc : DataFrame
        Dataframe for the reporting conditions (poa, t_amb, and w_vel).
    regression_results : statsmodels linear regression model
        Holds the linear regression model object.
    regression_formula : str
        Regression formula to be fit to measured and simulated data.  Must
        follow the requirements of statsmodels use of patsy.
    tolerance : str
        String representing error band.  Ex. '+ 3', '+/- 3', '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    """

    def __init__(self, name):  # noqa: D107
        super(CapData, self).__init__()
        self.name = name
        self.data = pd.DataFrame()
        self.data_filtered = None
        self.column_groups = {}
        self.trans_keys = []
        self.regression_cols = {}
        self.trans_abrev = {}
        self.col_colors = {}
        self.summary_ix = []
        self.summary = []
        self.removed = []
        self.kept = []
        self.filter_counts = {}
        self.rc = None
        self.regression_results = None
        self.regression_formula = ('power ~ poa + I(poa * poa)'
                                   '+ I(poa * t_amb) + I(poa * w_vel) - 1')
        self.tolerance = None
        self.pre_agg_cols = None
        self.pre_agg_trans = None
        self.pre_agg_reg_trans = None
        self.loc = LocIndexer(self)
        self.floc = FilteredLocIndexer(self)

    def set_regression_cols(self, power='', poa='', t_amb='', w_vel=''):
        """
        Create a dictionary linking the regression variables to data.

        Links the independent regression variables to the appropriate
        translation keys or a column name may be used to specify a
        single column of data.

        Sets attribute and returns nothing.

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
        self.regression_cols = {'power': power,
                                'poa': poa,
                                't_amb': t_amb,
                                'w_vel': w_vel}

    def copy(self):
        """Create and returns a copy of self."""
        cd_c = CapData('')
        cd_c.name = copy.copy(self.name)
        cd_c.data = self.data.copy()
        cd_c.data_filtered = self.data_filtered.copy()
        cd_c.column_groups = copy.copy(self.column_groups)
        cd_c.trans_keys = copy.copy(self.trans_keys)
        cd_c.regression_cols = copy.copy(self.regression_cols)
        cd_c.trans_abrev = copy.copy(self.trans_abrev)
        cd_c.col_colors = copy.copy(self.col_colors)
        cd_c.col_colors = copy.copy(self.col_colors)
        cd_c.summary_ix = copy.copy(self.summary_ix)
        cd_c.summary = copy.copy(self.summary)
        cd_c.rc = copy.copy(self.rc)
        cd_c.regression_results = copy.deepcopy(self.regression_results)
        cd_c.regression_formula = copy.copy(self.regression_formula)
        cd_c.pre_agg_cols = copy.copy(self.pre_agg_cols)
        cd_c.pre_agg_trans = copy.deepcopy(self.pre_agg_trans)
        cd_c.pre_agg_reg_trans = copy.deepcopy(self.pre_agg_reg_trans)
        return cd_c

    def empty(self):
        """Return a boolean indicating if the CapData object contains data."""
        tests_indicating_empty = [self.data.empty, len(self.trans_keys) == 0,
                                  len(self.column_groups) == 0]
        return all(tests_indicating_empty)

    def set_plot_attributes(self):
        """Set column colors used in plot method."""
        # dframe = self.data

        group_id_regex = {
            'real_pwr': re.compile(r'real_pwr|pwr|meter_power|active_pwr|active_power', re.IGNORECASE),
            'irr_poa': re.compile(r'poa|irr_poa|poa_irr', re.IGNORECASE),
            'irr_ghi': re.compile(r'ghi|irr_ghi|ghi_irr', re.IGNORECASE),
            'temp_amb': re.compile(r'amb|temp.*amb', re.IGNORECASE),
            'temp_mod': re.compile(r'bom|temp.*bom|module.*temp.*|temp.*mod.*', re.IGNORECASE),
            'wind': re.compile(r'wind|w_vel|wspd|wind__', re.IGNORECASE),
        }

        for group_id, cols_in_group in self.column_groups.items():
            col_key = None
            for plot_colors_group_key, regex in group_id_regex.items():
                if regex.match(group_id):
                    col_key = plot_colors_group_key
                    break
            for i, col in enumerate(cols_in_group):
                try:
                    j = i % 4
                    self.col_colors[col] = plot_colors_brewer[col_key][j]
                except KeyError:
                    j = i % 256
                    self.col_colors[col] = cc.glasbey_dark[j]

    def drop_cols(self, columns):
        """
        Drop columns from CapData `data` and `column_groups`.

        Parameters
        ----------
        Columns : list
            List of columns to drop.

        Todo
        ----
        Change to accept a string column name or list of strings
        """
        for key, value in self.column_groups.items():
            for col in columns:
                try:
                    value.remove(col)
                    self.column_groups[key] = value
                except ValueError:
                    continue
        self.data.drop(columns, axis=1, inplace=True)
        self.data_filtered.drop(columns, axis=1, inplace=True)

    def get_reg_cols(self, reg_vars=None, filtered_data=True):
        """
        Get regression columns renamed with keys from `regression_cols`.

        Parameters
        ----------
        reg_vars : list or str, default None
            By default returns all columns identified in `regression_cols`.
            A list with any combination of the keys of `regression_cols` is valid
            or pass a single key as a string.
        filtered_data : bool, default true
            Return filtered or unfiltered data.

        Returns
        -------
        DataFrame
        """
        if reg_vars is None:
            reg_vars = list(self.regression_cols.keys())
        df = self.rview(reg_vars, filtered_data=filtered_data).copy()
        rename = {df.columns[0]: reg_vars}

        if isinstance(reg_vars, list):
            for reg_var in reg_vars:
                if self.regression_cols[reg_var] in self.data_filtered.columns:
                    continue
                else:
                    columns = self.column_groups[self.regression_cols[reg_var]]
                    if len(columns) != 1:
                        return warnings.warn('Multiple columns per translation '
                                             'dictionary group. Run agg_sensors '
                                             'before this method.')
            rename = {old: new for old, new in zip(df.columns, reg_vars)}

        df.rename(columns=rename, inplace=True)
        return df

    def view(self, tkey, filtered_data=False):
        """
        Convience function returns columns using `column_groups` names.

        Parameters
        ----------
        tkey: int or str or list of int or strs
            String or list of strings from self.trans_keys or int postion or
            list of int postitions of value in self.trans_keys.
        """
        if isinstance(tkey, int):
            keys = self.column_groups[self.trans_keys[tkey]]
        elif isinstance(tkey, list) and len(tkey) > 1:
            keys = []
            for key in tkey:
                if isinstance(key, str):
                    keys.extend(self.column_groups[key])
                elif isinstance(key, int):
                    keys.extend(self.column_groups[self.trans_keys[key]])
        elif tkey in self.trans_keys:
            keys = self.column_groups[tkey]

        if filtered_data:
            return self.data_filtered[keys]
        else:
            return self.data[keys]

    def rview(self, ind_var, filtered_data=False):
        """
        Convience fucntion to return regression independent variable.

        Parameters
        ----------
        ind_var: string or list of strings
            may be 'power', 'poa', 't_amb', 'w_vel', a list of some subset of
            the previous four strings or 'all'
        """
        if ind_var == 'all':
            keys = list(self.regression_cols.values())
        elif isinstance(ind_var, list) and len(ind_var) > 1:
            keys = [self.regression_cols[key] for key in ind_var]
        elif ind_var in met_keys:
            ind_var = [ind_var]
            keys = [self.regression_cols[key] for key in ind_var]

        lst = []
        for key in keys:
            if key in self.data.columns:
                lst.extend([key])
            else:
                lst.extend(self.column_groups[key])
        if filtered_data:
            return self.data_filtered[lst]
        else:
            return self.data[lst]

    def __comb_trans_keys(self, grp):
        comb_keys = []

        for key in self.trans_keys:
            if key.find(grp) != -1:
                comb_keys.append(key)

        cols = []
        for key in comb_keys:
            cols.extend(self.column_groups[key])

        grp_comb = grp + '_comb'
        if grp_comb not in self.trans_keys:
            self.column_groups[grp_comb] = cols
            self.trans_keys.extend([grp_comb])
            print('Added new group: ' + grp_comb)

    def review_column_groups(self):
        """Print `column_groups` with nice formatting."""
        if len(self.column_groups) == 0:
            return 'column_groups attribute is empty.'
        else:
            for trans_grp, col_list in self.column_groups.items():
                print(trans_grp)
                for col in col_list:
                    print('    ' + col)

    # PLOTTING METHODS
    def reg_scatter_matrix(self):
        """Create pandas scatter matrix of regression variables."""
        df = self.get_reg_cols(reg_vars=['poa', 't_amb', 'w_vel'])
        df['poa_poa'] = df['poa'] * df['poa']
        df['poa_t_amb'] = df['poa'] * df['t_amb']
        df['poa_w_vel'] = df['poa'] * df['w_vel']
        df.drop(['t_amb', 'w_vel'], axis=1, inplace=True)
        return(pd.plotting.scatter_matrix(df))

    def scatter(self, filtered=True):
        """
        Create scatter plot of irradiance vs power.

        Parameters
        ----------
        filtered : bool, default true
            Plots filtered data when true and all data when false.
        """
        if filtered:
            df = self.rview(['power', 'poa'], filtered_data=True)
        else:
            df = self.rview(['power', 'poa'], filtered_data=False)

        if df.shape[1] != 2:
            return warnings.warn('Aggregate sensors before using this '
                                 'method.')

        df = df.rename(columns={df.columns[0]: 'power', df.columns[1]: 'poa'})
        plt = df.plot(kind='scatter', x='poa', y='power',
                      title=self.name, alpha=0.2)
        return(plt)

    def scatter_hv(self, timeseries=False, all_reg_columns=False):
        """
        Create holoviews scatter plot of irradiance vs power.

        Use holoviews opts magics in notebook cell before calling method to
        adjust height and width of plots:

        %%opts Scatter [height=200, width=400]
        %%opts Curve [height=200, width=400]

        Parameters
        ----------
        timeseries : boolean, default False
            True adds timeseries plot of the data linked to the scatter plot.
            Points selected in teh scatter plot will be highlighted in the
            timeseries plot.
        all_reg_columns : boolean, default False
            Set to True to include the data used in the regression in addition
            to poa irradiance and power in the hover tooltip.
        """
        df = self.get_reg_cols(filtered_data=True)
        df.index.name = 'index'
        df.reset_index(inplace=True)
        vdims = ['power', 'index']
        if all_reg_columns:
            vdims.extend(list(df.columns.difference(vdims)))
        poa_vs_kw = hv.Scatter(df, 'poa', vdims).opts(
            size=5,
            tools=['hover', 'lasso_select', 'box_select'],
            legend_position='right',
            height=400,
            width=400,
        )
        # layout_scatter = (poa_vs_kw).opts(opt_dict)
        if timeseries:
            poa_vs_time = hv.Curve(df, 'index', ['power', 'poa']).opts(
                tools=['hover', 'lasso_select', 'box_select'],
                height=400,
                width=800,
            )
            layout_timeseries = (poa_vs_kw + poa_vs_time)
            DataLink(poa_vs_kw, poa_vs_time)
            return(layout_timeseries.cols(1))
        else:
            return(poa_vs_kw)

    def plot(self, marker='line', ncols=1, width=1500, height=250,
             legends=False, merge_grps=['irr', 'temp'], subset=None,
             filtered=False, use_abrev_name=False, **kwargs):
        """
        Create a plot for each group of sensors in self.column_groups.

        Function returns a Bokeh grid of figures.  A figure is generated for
        each type of measurement identified by the keys in `column_groups` and
        a line is plotted on the figure for each column of measurements of
        that type.

        For example, if there are multiple plane of array irradiance sensors,
        the data from each one will be plotted on a single figure.

        Figures are not generated for categories that would plot more than 10
        lines.

        Parameters
        ----------
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
            List of strings to search for in the `column_groups` keys.
            A new entry is added to `column_groups` with keys following the
            format 'search str_comb' and the value is a list of column names
            that contain the search string. The default will combine all
            irradiance measurements into a group and temperature measurements
            into a group.

            Pass an empty list to not merge any plots.

            Use 'irr-poa' and 'irr-ghi' to plot clear sky modeled with measured
            data.
        subset : list, default None
            List of the keys of `column_groups` to control the order of to plot
            only a subset of the plots or control the order of plots.
        filtered : bool, default False
            Set to true to plot the filtered data.
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

        if filtered:
            dframe = self.data_filtered
        else:
            dframe = self.data
        dframe.index.name = 'Timestamp'

        names_to_abrev = {val: key for key, val in self.trans_abrev.items()}

        plots = []
        x_axis = None

        source = ColumnDataSource(dframe)

        hover = HoverTool()
        hover.tooltips = [
            ("Name", "$name"),
            ("Datetime", "@Timestamp{%F %H:%M}"),
            ("Value", "$y{0,0.00}"),
        ]
        hover.formatters = {"@Timestamp": "datetime"}

        tools = 'pan, xwheel_pan, xwheel_zoom, box_zoom, save, reset'

        if isinstance(subset, list):
            plot_keys = subset
        else:
            plot_keys = self.trans_keys

        for j, key in enumerate(plot_keys):
            df = dframe[self.column_groups[key]]
            cols = df.columns.tolist()

            if x_axis is None:
                p = figure(title=key, width=width, height=height,
                           x_axis_type='datetime', tools=tools)
                p.tools.append(hover)
                x_axis = p.x_range
            if j > 0:
                p = figure(title=key, width=width, height=height,
                           x_axis_type='datetime', x_range=x_axis, tools=tools)
                p.tools.append(hover)
            legend_items = []
            for i, col in enumerate(cols):
                if use_abrev_name:
                    name = names_to_abrev[col]
                else:
                    name = col

                if col.find('csky') == -1:
                    line_dash = 'solid'
                else:
                    line_dash = (5, 2)

                if marker == 'line':
                    try:
                        series = p.line('Timestamp', col, source=source,
                                        line_color=self.col_colors[col],
                                        line_dash=line_dash,
                                        name=name)
                    except KeyError:
                            series = p.line('Timestamp', col, source=source,
                                            line_dash=line_dash,
                                            name=name)
                elif marker == 'circle':
                    series = p.circle('Timestamp', col,
                                      source=source,
                                      line_color=self.col_colors[col],
                                      size=2, fill_color="white",
                                      name=name)
                if marker == 'line-circle':
                    series = p.line('Timestamp', col, source=source,
                                    line_color=self.col_colors[col],
                                    name=name)
                    series = p.circle('Timestamp', col,
                                      source=source,
                                      line_color=self.col_colors[col],
                                      size=2, fill_color="white",
                                      name=name)
                legend_items.append((col, [series, ]))

            legend = Legend(items=legend_items, location=(40, -5))
            legend.label_text_font_size = '8pt'
            if legends:
                p.add_layout(legend, 'below')

            plots.append(p)

        grid = gridplot(plots, ncols=ncols, **kwargs)
        return show(grid)

    def scatter_filters(self):
        """
        Returns an overlay of scatter plots of intervals removed for each filter.

        A scatter plot of power vs irradiance is generated for the time intervals
        removed for each filtering step. Each of these plots is labeled and
        overlayed.
        """
        scatters = []

        data = self.get_reg_cols(reg_vars=['power', 'poa'], filtered_data=False)
        data['index'] = self.data.loc[:, 'index']
        plt_no_filtering = hv.Scatter(data, 'poa', ['power', 'index']).relabel('all')
        scatters.append(plt_no_filtering)

        d1 = data.loc[self.removed[0]['index'], :]
        plt_first_filter = hv.Scatter(d1, 'poa', ['power', 'index']).relabel(
            self.removed[0]['name']
        )
        scatters.append(plt_first_filter)

        for i, filtering_step in enumerate(self.kept):
            if i >= len(self.kept) - 1:
                break
            else:
                flt_legend = self.kept[i + 1]['name']
            d_flt = data.loc[filtering_step['index'], :]
            plt = hv.Scatter(d_flt, 'poa', ['power', 'index']).relabel(flt_legend)
            scatters.append(plt)

        scatter_overlay = hv.Overlay(scatters)
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                width=650,
                height=500,
                muted_fill_alpha=0,
                fill_alpha=0.4,
                line_width=0,
                tools=['hover'],
            ),
            hv.opts.Overlay(
                legend_position='right',
                toolbar='above'
            ),
        )
        return scatter_overlay

    def timeseries_filters(self):
        """
        Returns an overlay of scatter plots of intervals removed for each filter.

        A scatter plot of power vs irradiance is generated for the time intervals
        removed for each filtering step. Each of these plots is labeled and
        overlayed.
        """
        plots = []

        data = self.get_reg_cols(reg_vars='power', filtered_data=False)
        data.reset_index(inplace=True)
        plt_no_filtering  = hv.Curve(data, ['Timestamp'], ['power'], label='all')
        plt_no_filtering.opts(
            line_color='black',
            line_width=1,
            width=1500,
            height=450,
        )
        plots.append(plt_no_filtering)

        d1 = self.rview('power').loc[self.removed[0]['index'], :]
        plt_first_filter = hv.Scatter(
            (d1.index, d1.iloc[:, 0]),
            label=self.removed[0]['name'])
        plots.append(plt_first_filter)

        for i, filtering_step in enumerate(self.kept):
            if i >= len(self.kept) - 1:
                break
            else:
                flt_legend = self.kept[i + 1]['name']
            d_flt = self.rview('power').loc[filtering_step['index'], :]
            plt = hv.Scatter((d_flt.index, d_flt.iloc[:, 0]), label=flt_legend)
            plots.append(plt)

        scatter_overlay = hv.Overlay(plots)
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                muted_fill_alpha=0,
                fill_alpha=1,
                line_width=0,
                tools=['hover'],
            ),
            hv.opts.Overlay(
                legend_position='bottom',
                toolbar='right',
            ),
        )
        return scatter_overlay

    def reset_filter(self):
        """
        Set `data_filtered` to `data` and reset filtering summary.

        Parameters
        ----------
        data : str
            'sim' or 'das' determines if filter is on sim or das data.
        """
        self.data_filtered = self.data.copy()
        self.summary_ix = []
        self.summary = []
        self.filter_counts = {}
        self.removed = []
        self.kept = []

    def reset_agg(self):
        """
        Remove aggregation columns from data and data_filtered attributes.

        Does not reset filtering of data or data_filtered.
        """
        if self.pre_agg_cols is None:
            return warnings.warn('Nothing to reset; agg_sensors has not been'
                                 'used.')
        else:
            self.data = self.data[self.pre_agg_cols].copy()
            self.data_filtered = self.data_filtered[self.pre_agg_cols].copy()

            self.column_groups = self.pre_agg_trans.copy()
            self.regression_cols = self.pre_agg_reg_trans.copy()

    def __get_poa_col(self):
        """
        Return poa column name from `column_groups`.

        Also, issues warning if there are more than one poa columns in
        `column_groups`.
        """
        poa_trans_key = self.regression_cols['poa']
        if poa_trans_key in self.data.columns:
            return poa_trans_key
        else:
            poa_cols = self.column_groups[poa_trans_key]
        if len(poa_cols) > 1:
            return warnings.warn('{} columns of irradiance data. '
                                 'Use col_name to specify a single '
                                 'column.'.format(len(poa_cols)))
        else:
            return poa_cols[0]

    def agg_sensors(self, agg_map=None):
        """
        Aggregate measurments of the same variable from different sensors.

        Parameters
        ----------
        agg_map : dict, default None
            Dictionary specifying aggregations to be performed on
            the specified groups from the `column_groups` attribute.  The dictionary
            keys should be keys from the `column_gruops` attribute. The
            dictionary values should be aggregation functions. See pandas API
            documentation of Computations / descriptive statistics for a list of all
            options. 
            By default the groups of columns assigned to the 'power', 'poa', 't_amb',
            and 'w_vel' keys in the `regression_cols` attribute are aggregated:
            - sum power
            - mean of poa, t_amb, w_vel

        Returns
        -------
        None
            Acts in place on the data, data_filtered, and regression_cols attributes.
            
        Notes
        -----
        This method is intended to be used before any filtering methods are applied.
        Filtering steps applied when this method is used will be lost.
        """
        if not len(self.summary) == 0:
            warnings.warn('The data_filtered attribute has been overwritten '
                          'and previously applied filtering steps have been '
                          'lost.  It is recommended to use agg_sensors '
                          'before any filtering methods.')
        # reset summary data
        self.summary_ix = []
        self.summary = []

        self.pre_agg_cols = self.data.columns.copy()
        self.pre_agg_trans = copy.deepcopy(self.column_groups)
        self.pre_agg_reg_trans = copy.deepcopy(self.regression_cols)

        if agg_map is None:
            agg_map = {self.regression_cols['power']: 'sum',
                       self.regression_cols['poa']: 'mean',
                       self.regression_cols['t_amb']: 'mean',
                       self.regression_cols['w_vel']: 'mean'}

        dfs_to_concat = []
        for group_id, agg_func in agg_map.items():
            columns_to_aggregate = self.view(group_id, filtered_data=False)
            if columns_to_aggregate.shape[1] == 1:
                continue
            agg_result = columns_to_aggregate.agg(agg_func, axis=1).to_frame()
            if isinstance(agg_func, str):
                col_name = group_id + '_' + agg_func + '_agg'
            else:
                col_name = group_id + '_' + agg_func.__name__ + '_agg'
            agg_result.rename(columns={agg_result.columns[0]: col_name}, inplace=True)
            dfs_to_concat.append(agg_result)

        dfs_to_concat.append(self.data)
        # write over data and data_filtered attributes
        self.data = pd.concat(dfs_to_concat, axis=1)
        self.data_filtered = self.data.copy()

        # update regression_cols attribute 
        for reg_var, trans_group in self.regression_cols.items():
            if self.rview(reg_var).shape[1] == 1:
                continue
            if trans_group in agg_map.keys():
                try:
                    agg_col = trans_group + '_' + agg_map[trans_group] + '_agg'  # noqa: E501
                except TypeError:
                    agg_col = trans_group + '_' + col_name + '_agg'
                print(agg_col)
                self.regression_cols[reg_var] = agg_col

    def data_columns_to_excel(self, sort_by_reversed_names=True):
        """
        Write the columns of data to an excel file as a template for a column grouping.

        Parameters
        ----------
        sort_by_inverted_names : bool, default False
            If true sort column names after reversing them.

        Returns
        -------
        None
            Writes to excel file at self.data_loader.path / 'column_groups.xlsx'.
        """
        df = self.data.columns.to_frame().reset_index(drop=True)
        df['a'] = ""
        df = df[['a', 0]]
        # print(df)
        df.sort_values(by=0, inplace=True, ascending=True)
        if sort_by_reversed_names:
            df['reversed'] = df[0].str[::-1]
            df.sort_values(by='reversed', inplace=True, ascending=True)
            df = df[['a', 0]]
        if self.data_loader.path.is_dir():
            df.to_excel(
                self.data_loader.path / 'column_groups.xlsx', index=False, header=False
            )
        elif self.data_loader.path.is_file():
            print(self.data_loader.path.parent)
            df.to_excel(
                self.data_loader.path.parent / 'column_groups.xlsx',
                index=False,
                header=False,
            )

    @update_summary
    def filter_irr(self, low, high, ref_val=None, col_name=None, inplace=True):
        """
        Filter on irradiance values.

        Parameters
        ----------
        low : float or int
            Minimum value as fraction (0.8) or absolute 200 (W/m^2).
        high : float or int
            Max value as fraction (1.2) or absolute 800 (W/m^2).
        ref_val : float or int or `self_val`
            Must provide arg when `low` and `high` are fractions.
            Pass `self_val` to use the value in `self.rc`.
        col_name : str, default None
            Column name of irradiance data to filter.  By default uses the POA
            irradiance set in regression_cols attribute or average of the POA
            columns.
        inplace : bool, default True
            Default true write back to data_filtered or return filtered
            dataframe.

        Returns
        -------
        DataFrame
            Filtered dataframe if inplace is False.
        """
        if col_name is None:
            irr_col = self.__get_poa_col()
        else:
            irr_col = col_name

        if ref_val == 'self_val':
            ref_val = self.rc['poa'][0]

        df_flt = filter_irr(self.data_filtered, irr_col, low, high,
                            ref_val=ref_val)
        if inplace:
            self.data_filtered = df_flt
        else:
            return df_flt

    @update_summary
    def filter_pvsyst(self, inplace=True):
        """
        Filter pvsyst data for off max power point tracking operation.

        This function is only applicable to simulated data generated by PVsyst.
        Filters the 'IL Pmin', IL Vmin', 'IL Pmax', 'IL Vmax' values if they
        are greater than 0.

        Parameters
        ----------
        inplace: bool, default True
            If inplace is true, then function overwrites the filtered data.  If
            false returns a CapData object.

        Returns
        -------
        CapData object if inplace is set to False.
        """
        df = self.data_filtered

        columns = ['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax']
        index = df.index

        for column in columns:
            if column not in df.columns:
                column = column.replace(' ', '_')
            if column in df.columns:
                indices_to_drop = df[df[column] > 0].index
                if not index.equals(indices_to_drop):
                    index = index.difference(indices_to_drop)
            else:
                warnings.warn('{} or {} is not a column in the '
                              'data.'.format(column, column.replace('_', ' ')))

        if inplace:
            self.data_filtered = self.data_filtered.loc[index, :]
        else:
            return self.data_filtered.loc[index, :]

    @update_summary
    def filter_shade(self, fshdbm=1.0, query_str=None, inplace=True):
        """
        Remove data during periods of array shading.

        The default behavior assumes the filter is applied to data output from
        PVsyst and removes all periods where values in the column 'FShdBm' are
        less than 1.0.

        Use the query_str parameter when shading losses (power) rather than a
        shading fraction are available.

        Parameters
        ----------
        fshdbm : float, default 1.0
            The value for fractional shading of beam irradiance as given by the
            PVsyst output parameter FShdBm. Data is removed when the shading
            fraction is less than the value passed to fshdbm. By default all
            periods of shading are removed.
        query_str : str
            Query string to pass to pd.DataFrame.query method. The query string
            should be a boolean expression comparing a column name to a numeric
            filter value, like 'ShdLoss<=50'.  The column name must not contain
            spaces.
        inplace: bool, default True
            If inplace is true, then function overwrites the filtered
            dataframe. If false returns a DataFrame.

        Returns
        -------
        pd.DataFrame
            If inplace is false returns a dataframe.
        """
        df = self.data_filtered

        if query_str is None:
            query_str = "FShdBm>=@fshdbm"

        index_shd = df.query(query_str).index

        if inplace:
            self.data_filtered = self.data_filtered.loc[index_shd, :]
        else:
            return self.data_filtered.loc[index_shd, :]

    @update_summary
    def filter_time(self, start=None, end=None, drop=False, days=None, test_date=None,
                    inplace=True, wrap_year=False):
        """
        Select data for a specified time period.

        Parameters
        ----------
        start : str or pd.Timestamp or None, default None
            Start date for data to be returned.  If a string is passed it must
            be in format that can be converted by pandas.to_datetime.  Not
            required if test_date and days arguments are passed.
        end : str or pd.Timestamp or None, default None
            End date for data to be returned.  If a string is passed it must
            be in format that can be converted by pandas.to_datetime.  Not
            required if test_date and days arguments are passed.
        drop : bool, default False
            Set to true to drop time period between `start` and `end` rather
            than keep it. Must supply `start` and `end` and `wrap_year` must
            be false.
        days : int or None, default None
            Days in time period to be returned.  Not required if `start` and
            `end` are specified.
        test_date : str or pd.Timestamp or None, default None
            Must be format that can be converted by pandas.to_datetime.  Not
            required if `start` and `end` are specified.  Requires `days`
            argument. Time period returned will be centered on this date.
        inplace : bool, default True
            If inplace is true, then function overwrites the filtered
            dataframe. If false returns a DataFrame.
        wrap_year : bool, default False
            If true calls the wrap_year_end function.  See wrap_year_end
            docstring for details. wrap_year_end was cntg_eoy prior to v0.7.0.

        Todo
        ----
        Add inverse options to remove time between start end rather than return
        it.
        """
        if start is not None and end is not None:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if wrap_year and spans_year(start, end):
                df_temp = wrap_year_end(self.data_filtered, start, end)
            else:
                df_temp = self.data_filtered.loc[start:end, :]
                if drop:
                    keep_ix = self.data_filtered.index.difference(df_temp.index)
                    df_temp = self.data_filtered.loc[keep_ix, :]

        if start is not None and end is None:
            if days is None:
                return warnings.warn("Must specify end date or days.")
            else:
                start = pd.to_datetime(start)
                end = start + pd.DateOffset(days=days)
                if wrap_year and spans_year(start, end):
                    df_temp = wrap_year_end(self.data_filtered, start, end)
                else:
                    df_temp = self.data_filtered.loc[start:end, :]

        if start is None and end is not None:
            if days is None:
                return warnings.warn("Must specify end date or days.")
            else:
                end = pd.to_datetime(end)
                start = end - pd.DateOffset(days=days)
                if wrap_year and spans_year(start, end):
                    df_temp = wrap_year_end(self.data_filtered, start, end)
                else:
                    df_temp = self.data_filtered.loc[start:end, :]

        if test_date is not None:
            test_date = pd.to_datetime(test_date)
            if days is None:
                return warnings.warn("Must specify days")
            else:
                offset = pd.DateOffset(days=days // 2)
                start = test_date - offset
                end = test_date + offset
                if wrap_year and spans_year(start, end):
                    df_temp = wrap_year_end(self.data_filtered, start, end)
                else:
                    df_temp = self.data_filtered.loc[start:end, :]

        if inplace:
            self.data_filtered = df_temp
        else:
            return df_temp

    @update_summary
    def filter_days(self, days, drop=False, inplace=True):
        """
        Select or drop timestamps for days passed.

        Parameters
        ----------
        days : list
            List of days to select or drop.
        drop : bool, default False
            Set to true to drop the timestamps for the days passed instead of
            keeping only those days.
        inplace : bool, default True
            If inplace is true, then function overwrites the filtered
            dataframe. If false returns a DataFrame.
        """
        ix_all_days = None
        for day in days:
            ix_day = self.data_filtered.loc[day].index
            if ix_all_days is None:
                ix_all_days = ix_day
            else:
                ix_all_days = ix_all_days.union(ix_day)

        if drop:
            ix_wo_days = self.data_filtered.index.difference(ix_all_days)
            filtered_data = self.data_filtered.loc[ix_wo_days, :]
        else:
            filtered_data = self.data_filtered.loc[ix_all_days, :]

        if inplace:
            self.data_filtered = filtered_data
        else:
            return filtered_data

    @update_summary
    def filter_outliers(self, inplace=True, **kwargs):
        """
        Apply eliptic envelope from scikit-learn to remove outliers.

        Parameters
        ----------
        inplace : bool
            Default of true writes filtered dataframe back to data_filtered
            attribute.
        **kwargs
            Passed to sklearn EllipticEnvelope.  Contamination keyword
            is useful to adjust proportion of outliers in dataset.
            Default is 0.04.
        Todo
        ----
        Add plot option
            Add option to return plot showing envelope with points not removed
            alpha decreased.
        """
        XandY = self.rview(['poa', 'power'], filtered_data=True)
        if XandY.shape[1] > 2:
            return warnings.warn('Too many columns. Try running '
                                 'aggregate_sensors before using '
                                 'filter_outliers.')
        X1 = XandY.values

        if 'support_fraction' not in kwargs.keys():
            kwargs['support_fraction'] = 0.9
        if 'contamination' not in kwargs.keys():
            kwargs['contamination'] = 0.04

        clf_1 = sk_cv.EllipticEnvelope(**kwargs)
        clf_1.fit(X1)

        if inplace:
            self.data_filtered = self.data_filtered[clf_1.predict(X1) == 1]
        else:
            return self.data_filtered[clf_1.predict(X1) == 1]

    @update_summary
    def filter_pf(self, pf, inplace=True):
        """
        Filter data on the power factor.

        Parameters
        ----------
        pf: float
            0.999 or similar to remove timestamps with lower power factor
            values.  Values greater than or equal to `pf` are kept.
        inplace : bool
            Default of true writes filtered dataframe back to data_filtered
            attribute.

        Returns
        -------
        Dataframe when inplace is False.

        Todo
        ----
        Spec pf column
            Increase options to specify which columns are used in the filter.
        """
        for key in self.trans_keys:
            if key.find('pf') == 0:
                selection = key

        df = self.data_filtered[self.column_groups[selection]]

        df_flt = self.data_filtered[(np.abs(df) >= pf).all(axis=1)]

        if inplace:
            self.data_filtered = df_flt
        else:
            return df_flt

    @update_summary
    def filter_power(self, power, percent=None, columns=None, inplace=True):
        """
        Remove data above the specified power threshold.

        Parameters
        ----------
        power : numeric
            If `percent` is none, all data equal to or greater than `power`
            is removed.
            If `percent` is not None, then power should be the nameplate power.
        percent : None, or numeric, default None
            Data greater than or equal to `percent` of `power` is removed.
            Specify percentage as decimal i.e. 1% is passed as 0.01.
        columns : None or str, default None
            By default filter is applied to the power data identified in the
            `regression_cols` attribute.
            Pass a column name or column group to filter on. When passing a
            column group the power filter is applied to each column in the
            group.
        inplace : bool, default True
            Default of true writes filtered dataframe back to data_filtered
            attribute.

        Returns
        -------
        Dataframe when inplace is false.
        """
        if percent is not None:
            power = power * (1 - percent)

        multiple_columns = False

        if columns is None:
            power_data = self.get_reg_cols('power')
        elif isinstance(columns, str):
            if columns in self.column_groups.keys():
                power_data = self.view(columns, filtered_data=True)
                multiple_columns = True
            else:
                power_data = pd.DataFrame(self.data_filtered[columns])
                power_data.rename(columns={power_data.columns[0]: 'power'},
                                  inplace=True)
        else:
            return warnings.warn('columns must be None or a string.')

        if multiple_columns:
            filtered_power_bool = power_data.apply(lambda x: all(x < power),
                                                   axis=1)
        else:
            filtered_power_bool = power_data['power'] < power

        df_flt = self.data_filtered[filtered_power_bool]

        if inplace:
            self.data_filtered = df_flt
        else:
            return df_flt

    @update_summary
    def filter_custom(self, func, *args, **kwargs):
        """
        Apply `update_summary` decorator to passed function.

        Parameters
        ----------
        func : function
            Any function that takes a dataframe as the first argument and
            returns a dataframe.
            Many pandas dataframe methods meet this requirement, like
            pd.DataFrame.between_time.
        *args
            Additional positional arguments passed to func.
        **kwds
            Additional keyword arguments passed to func.

        Examples
        --------
        Example use of the pandas dropna method to remove rows with missing
        data.

        >>> das.custom_filter(pd.DataFrame.dropna, axis=0, how='any')
        >>> summary = das.get_summary()
        >>> summary['pts_before_filter'][0]
        1424
        >>> summary['pts_removed'][0]
        16

        Example use of the pandas between_time method to remove time periods.

        >>> das.reset_filter()
        >>> das.custom_filter(pd.DataFrame.between_time, '9:00', '13:00')
        >>> summary = das.get_summary()
        >>> summary['pts_before_filter'][0]
        245
        >>> summary['pts_removed'][0]
        1195
        >>> das.data_filtered.index[0].hour
        9
        >>> das.data_filtered.index[-1].hour
        13
        """
        self.data_filtered = func(self.data_filtered, *args, **kwargs)

    @update_summary
    def filter_sensors(self, perc_diff=None, inplace=True):
        """
        Drop suspicious measurments by comparing values from different sensors.

        This method ignores columns generated by the agg_sensors method.

        Parameters
        ----------
        perc_diff : dict
            Dictionary to specify a different threshold for
            each group of sensors.  Dictionary keys should be translation
            dictionary keys and values are floats, like {'irr-poa-': 0.05}.
            By default the poa sensors as set by the regression_cols dictionary
            are filtered with a 5% percent difference threshold.
        inplace : bool, default True
            If True, writes over current filtered dataframe. If False, returns
            CapData object.

        Returns
        -------
        DataFrame
            Returns filtered dataframe if inplace is False.
        """
        if self.pre_agg_cols is not None:
            df = self.data_filtered[self.pre_agg_cols]
            trans = self.pre_agg_trans
            regression_cols = self.pre_agg_reg_trans
        else:
            df = self.data_filtered
            trans = self.column_groups
            regression_cols = self.regression_cols

        if perc_diff is None:
            poa_trans_key = regression_cols['poa']
            perc_diff = {poa_trans_key: 0.05}

        for key, perc_diff_for_key in perc_diff.items():
            if 'index' in locals():
                # if index has been assigned then take intersection
                sensors_df = df[trans[key]]
                next_index = sensor_filter(sensors_df, perc_diff_for_key)
                index = index.intersection(next_index)  # noqa: F821
            else:
                # if index has not been assigned then assign it
                sensors_df = df[trans[key]]
                index = sensor_filter(sensors_df, perc_diff_for_key)

        df_out = self.data_filtered.loc[index, :]

        if inplace:
            self.data_filtered = df_out
        else:
            return df_out

    @update_summary
    def filter_clearsky(self, window_length=20, ghi_col=None, inplace=True,
                        keep_clear=True, **kwargs):
        """
        Use pvlib detect_clearsky to remove periods with unstable irradiance.

        The pvlib detect_clearsky function compares modeled clear sky ghi
        against measured clear sky ghi to detect periods of clear sky.  Refer
        to the pvlib documentation for additional information.

        By default uses data identified by the `column_groups` dictionary
        as ghi and modeled ghi.  Issues warning if there is no modeled ghi
        data, or the measured ghi data has not been aggregated.

        Parameters:
        window_length : int, default 20
            Length of sliding time window in minutes. Must be greater than 2
            periods. Default of 20 works well for 5 minute data intervals.
            pvlib default of 10 minutes works well for 1min data.
        ghi_col : str, default None
            The name of a column name of measured GHI data. Overrides default
            attempt to automatically identify a column of GHI data.
        inplace : bool, default True
            When true removes periods with unstable irradiance.  When false
            returns pvlib detect_clearsky results, which by default is a series
            of booleans.
        keep_clear : bool, default True
            Set to False to keep cloudy periods.
        **kwargs
            kwargs are passed to pvlib detect_clearsky.  See pvlib
            documentation for details.
        """
        if 'ghi_mod_csky' not in self.data_filtered.columns:
            return warnings.warn('Modeled clear sky data must be availabe to '
                                 'run this filter method. Use CapData '
                                 'load_data clear_sky option.')
        if ghi_col is None:
            ghi_keys = []
            for key in self.trans_keys:
                defs = key.split('-')
                if len(defs) == 1:
                    continue
                if 'ghi' == key.split('-')[1]:
                    ghi_keys.append(key)
            ghi_keys.remove('irr-ghi-clear_sky')

            if len(ghi_keys) > 1:
                return warnings.warn('Too many ghi categories. Pass column '
                                     'name to ghi_col to use a specific '
                                     'column.')
            else:
                meas_ghi = ghi_keys[0]

            meas_ghi = self.view(meas_ghi, filtered_data=True)
            if meas_ghi.shape[1] > 1:
                warnings.warn('Averaging measured GHI data.  Pass column name '
                              'to ghi_col to use a specific column.')
            meas_ghi = meas_ghi.mean(axis=1)
        else:
            meas_ghi = self.data_filtered[ghi_col]

        clear_per = detect_clearsky(
            meas_ghi,
            self.data_filtered['ghi_mod_csky'],
            meas_ghi.index,
            window_length,
            **kwargs,
        )
        if not any(clear_per):
            return warnings.warn('No clear periods detected. Try increasing '
                                 'the window length.')

        if keep_clear:
            df_out = self.data_filtered[clear_per]
        else:
            df_out = self.data_filtered[~clear_per]

        if inplace:
            self.data_filtered = df_out
        else:
            return df_out

    @update_summary
    def filter_missing(self, columns=None):
        """
        Drops time intervals with missing data for specified columns.

        By default drops intervals which have missing data in the columns defined
        by `regression_cols`.

        Parameters
        ----------
        columns : list, default None
            Subset of columns to check for missing data.
        """
        if columns is None:
            columns = list(self.regression_cols.values())
        df_reg_vars = self.data_filtered[columns]
        ix = df_reg_vars.dropna().index
        self.data_filtered = self.data_filtered.loc[ix, :]

    def filter_op_state(self, op_state, mult_inv=None, inplace=True):
        """
        NOT CURRENTLY IMPLEMENTED - Filter on inverter operation state.

        This filter is rarely useful in practice, but will be re-implemented
        if requested.

        Parameters
        ----------
        data : str
            'sim' or 'das' determines if filter is on sim or das data
        op_state : int
            integer inverter operating state to keep
        mult_inv : list of tuples, [(start, stop, op_state), ...]
            List of tuples where start is the first column of an type of
            inverter, stop is the last column and op_state is the operating
            state for the inverter type.
        inplace : bool, default True
            When True writes over current filtered dataframe.  When False
            returns CapData object.

        Returns
        -------
        CapData
            Returns filtered CapData object when inplace is False.

        Todo
        ----
        Complete move to capdata
            Needs to be updated to work as capdata rather than captest method.
            Remove call to __flt_setup and related subsequent use of flt_cd.
        """
        pass
        # if data == 'sim':
        #     print('Method not implemented for pvsyst data.')
        #     return None
        #
        # flt_cd = self.__flt_setup(data)
        #
        # for key in flt_cd.trans_keys:
        #     if key.find('op') == 0:
        #         selection = key
        #
        # df = flt_cd.df[flt_cd.trans[selection]]
        # # print('df shape: {}'.format(df.shape))
        #
        # if mult_inv is not None:
        #     return_index = flt_cd.df.index
        #     for pos_tup in mult_inv:
        #         # print('pos_tup: {}'.format(pos_tup))
        #         inverters = df.iloc[:, pos_tup[0]:pos_tup[1]]
        #         # print('inv shape: {}'.format(inverters.shape))
        #         df_temp = flt_cd.df[(inverters == pos_tup[2]).all(axis=1)]
        #         # print('df_temp shape: {}'.format(df_temp.shape))
        #         return_index = return_index.intersection(df_temp.index)
        #     flt_cd.df = flt_cd.df.loc[return_index, :]
        # else:
        #     flt_cd.df = flt_cd.df[(df == op_state).all(axis=1)]
        #
        # if inplace:
        #     if data == 'das':
        #         self.flt_das = flt_cd
        #     if data == 'sim':
        #         # should not run as 'sim' is not implemented
        #         self.flt_sim = flt_cd
        # else:
        #     return flt_cd

    def get_summary(self):
        """
        Print a summary of filtering applied to the data_filtered attribute.

        The summary dataframe shows the history of the filtering steps applied
        to the data including the timestamps remaining after each step, the
        timestamps removed by each step and the arguments used to call each
        filtering method.

        If the filter arguments are cutoff, the max column width can be
        increased by setting pd.options.display.max_colwidth.

        Parameters
        ----------
        None

        Returns
        -------
        Pandas DataFrame
        """
        try:
            df = pd.DataFrame(data=self.summary,
                              index=pd.MultiIndex.from_tuples(self.summary_ix),
                              columns=columns)
            return df
        except TypeError:
            print('No filters have been run.')

    @update_summary
    def rep_cond(
        self,
        irr_bal=False,
        percent_filter=20,
        w_vel=None,
        inplace=True,
        func={'poa': perc_wrap(60), 't_amb': 'mean', 'w_vel': 'mean'},
        freq=None,
        grouper_kwargs={},
        rc_kwargs={}):
        """
        Calculate reporting conditons.

        Parameters
        ----------
        irr_bal: boolean, default False
            If true, uses the irr_rc_balanced function to determine the
            reporting conditions. Replaces the calculations specified by func
            with or without freq.
        percent_filter : Int, default 20
            Percentage as integer used to filter around reporting
            irradiance in the irr_rc_balanced function.
        func: callable, string, dictionary, or list of string/callables
            Determines how the reporting condition is calculated.
            Default is a dictionary poa - 60th numpy_percentile, t_amb - mean
                                          w_vel - mean
            Can pass a string function ('mean') to calculate each reporting
            condition the same way.
        freq: str
            String pandas offset alias to specify aggregation frequency
            for reporting condition calculation. Ex '60D' for 60 Days or
            'MS' for months start.
        w_vel: int
            If w_vel is not none, then wind reporting condition will be set to
            value specified for predictions. Does not affect output unless pred
            is True and irr_bal is True.
        inplace: bool, True by default
            When true updates object rc parameter, when false returns
            dicitionary of reporting conditions.
        grouper_kwargs : dict
            Passed to pandas Grouper to control label and closed side of
            intervals. See pandas Grouper doucmentation for details. Default is
            left labeled and left closed.
        rc_kwargs : dict
            Passed to the irr_rc_balanced function if `irr_bal` is set to True.

        Returns
        -------
        dict
            Returns a dictionary of reporting conditions if inplace=False
            otherwise returns None.
        pandas DataFrame
            If pred=True, then returns a pandas dataframe of results.
        """
        df = self.rview(['poa', 't_amb', 'w_vel'],
                        filtered_data=True)
        df = df.rename(columns={df.columns[0]: 'poa',
                                df.columns[1]: 't_amb',
                                df.columns[2]: 'w_vel'})

        RCs_df = pd.DataFrame(df.agg(func)).T

        if irr_bal:
            self.rc_tool = ReportingIrradiance(
                df,
                'poa',
                percent_band=percent_filter,
                **rc_kwargs,
            )
            results = self.rc_tool.get_rep_irr()
            flt_df = results[1]
            temp_RC = flt_df['t_amb'].mean()
            wind_RC = flt_df['w_vel'].mean()
            RCs_df = pd.DataFrame({'poa': results[0],
                                   't_amb': temp_RC,
                                   'w_vel': wind_RC}, index=[0])

        if w_vel is not None:
            RCs_df['w_vel'][0] = w_vel

        if freq is not None:
            # wrap_seasons passes df through unchanged unless freq is one of
            # 'BQ-JAN', 'BQ-FEB', 'BQ-APR', 'BQ-MAY', 'BQ-JUL',
            # 'BQ-AUG', 'BQ-OCT', 'BQ-NOV'
            df = wrap_seasons(df, freq)
            df_grpd = df.groupby(pd.Grouper(freq=freq, **grouper_kwargs))

            if irr_bal:
                ix = pd.DatetimeIndex(list(df_grpd.groups.keys()), freq=freq)
                poa_RC = []
                temp_RC = []
                wind_RC = []
                for name, month in df_grpd:
                    self.rc_tool = ReportingIrradiance(
                        month,
                        'poa',
                        percent_band=percent_filter,
                        **rc_kwargs,
                    )
                    results = self.rc_tool.get_rep_irr()
                    poa_RC.append(results[0])
                    flt_df = results[1]
                    temp_RC.append(flt_df['t_amb'].mean())
                    wind_RC.append(flt_df['w_vel'].mean())
                RCs_df = pd.DataFrame({'poa': poa_RC,
                                       't_amb': temp_RC,
                                       'w_vel': wind_RC}, index=ix)
            else:
                RCs_df = df_grpd.agg(func)

            if w_vel is not None:
                RCs_df['w_vel'] = w_vel

        if inplace:
            print('Reporting conditions saved to rc attribute.')
            print(RCs_df)
            self.rc = RCs_df
        else:
            return RCs_df

    def predict_capacities(self, irr_filter=True, percent_filter=20, **kwargs):
        """
        Calculate expected capacities.

        Parameters
        ----------
        irr_filter : bool, default True
            When true will filter each group of data by a percentage around the
            reporting irradiance for that group.  The data groups are
            determined from the reporting irradiance attribute.
        percent_filter : float or int or tuple, default 20
            Percentage or tuple of percentages used to filter around reporting
            irradiance in the irr_rc_balanced function.  Required argument when
            irr_bal is True.
            Tuple option allows specifying different percentage for above and
            below reporting irradiance. (below, above)
        **kwargs
            NOTE: Should match kwargs used to calculate reporting conditions.
            Passed to filter_grps which passes on to pandas Grouper to control
            label and closed side of intervals.
            See pandas Grouper doucmentation for details. Default is left
            labeled and left closed.
        """
        df = self.rview(['poa', 't_amb', 'w_vel', 'power'],
                        filtered_data=True)
        df = df.rename(columns={df.columns[0]: 'poa',
                                df.columns[1]: 't_amb',
                                df.columns[2]: 'w_vel',
                                df.columns[3]: 'power'})

        if self.rc is None:
            return warnings.warn('Reporting condition attribute is None.\
                                 Use rep_cond to generate RCs.')

        low, high = perc_bounds(percent_filter)
        freq = self.rc.index.freq
        df = wrap_seasons(df, freq)
        grps = df.groupby(by=pd.Grouper(freq=freq, **kwargs))

        if irr_filter:
            grps = filter_grps(grps, self.rc, 'poa', low, high, freq)

        error = float(self.tolerance.split(sep=' ')[1]) / 100
        results = pred_summary(grps, self.rc, error, fml=self.regression_formula)

        return results

    @update_summary
    def fit_regression(self, filter=False, inplace=True, summary=True):
        """
        Perform a regression with statsmodels on filtered data.

        Parameters
        ----------
        filter: bool, default False
            When true removes timestamps where the residuals are greater than
            two standard deviations.  When false just calcualtes ordinary least
            squares regression.
        inplace: bool, default True
            If filter is true and inplace is true, then function overwrites the
            filtered data for sim or das.  If false returns a CapData object.
        summary: bool, default True
            Set to false to not print regression summary.

        Returns
        -------
        CapData
            Returns a filtered CapData object if filter is True and inplace is
            False.
        """
        df = self.get_reg_cols()

        reg = fit_model(df, fml=self.regression_formula)

        if filter:
            print('NOTE: Regression used to filter outlying points.\n\n')
            if summary:
                print(reg.summary())
            df = df[np.abs(reg.resid) < 2 * np.sqrt(reg.scale)]
            dframe_flt = self.data_filtered.loc[df.index, :]
            if inplace:
                self.data_filtered = dframe_flt
            else:
                return dframe_flt
        else:
            if summary:
                print(reg.summary())
            self.regression_results = reg

    def uncertainty():
        """Calculate random standard uncertainty of the regression.

        (SEE times the square root of the leverage of the reporting
        conditions).

        Not fully implemented yet.  Need to review and determine what actual
        variable should be.
        """
        pass
        # SEE = np.sqrt(self.regression_results.mse_resid)
        #
        # df = self.get_reg_cols()
        #
        # rc_pt = {key: val[0] for key, val in self.rc.items()}
        # rc_pt['power'] = actual
        # df.append([rc_pt])
        #
        # reg = fit_model(df, fml=self.regression_formula)
        #
        # infl = reg.get_influence()
        # leverage = infl.hat_matrix_diag[-1]
        # sy = SEE * np.sqrt(leverage)
        #
        # return(sy)

    def spatial_uncert(self, column_groups):
        """
        Spatial uncertainties of the independent regression variables.

        Parameters
        ----------
        column_groups : list
            Measurement groups to calculate spatial uncertainty.

        Returns
        -------
        None, stores dictionary of spatial uncertainties as an attribute.
        """
        spatial_uncerts = {}
        for group in column_groups:
            df = self.view(group, filtered_data=True)
            # prevent aggregation from updating column groups?
            # would not need the below line then
            df = df[[col for col in df.columns if 'agg' not in col]]
            qty_sensors = df.shape[1]
            s_spatial = df.std(axis=1)
            b_spatial_j = s_spatial / (qty_sensors ** (1 / 2))
            b_spatial = ((b_spatial_j ** 2).sum() / b_spatial_j.shape[0]) ** (1 / 2)
            spatial_uncerts[group] = b_spatial
        self.spatial_uncerts = spatial_uncerts

    def expanded_uncert(self, grp_to_term, k=1.96):
        """
        Calculate expanded uncertainty of the predicted power.

        Adds instrument uncertainty and spatial uncertainty in quadrature and
        passes the result through the regression to calculate the
        Systematic Standard Uncertainty, which is then added in quadrature with
        the Random Standard Uncertainty of the regression and multiplied by the
        k factor, `k`.

        1. Combine by adding in quadrature the spatial and instrument uncertainties
        for each measurand.
        2. Add the absolute uncertainties from step 1 to each of the respective
        reporting conditions to determine a value for the reporting condition
        plus the uncertainty.
        3. Calculate the predicted power using the RCs plus uncertainty three
        times i.e. calculate for each RC plus uncertainty. For example, to
        estimate the impact of the uncertainty of the reporting irradiance one
        would calculate expected power using the irradiance RC plus irradiance
        uncertainty at the reporting irradiance and the original temperature and
        wind reporting conditions that have not had any uncertainty added to them.
        6. Calculate the percent difference between the three new expected power
        values that include uncertainty of the RCs and the expected power with
        the unmodified RC.
        7. Take the square root of the sum of the squares of those three percent
        differences to obtain the Systematic Standard Uncertainty (bY).

        Expects CapData to have a instrument_uncert and spatial_uncerts
        attributes with matching keys.

        Parameters
        ----------
        grp_to_term : dict
            Map the groups of measurement types to the term in the
            regression formula that was regressed against an aggregated value
            (typically mean) from that group.
        k : numeric
            Coverage factor.

        Returns
        -------
            Expanded uncertainty as a decimal value.
        """
        pred = self.regression_results.get_prediction(self.rc)
        pred_cap = pred.predicted_mean[0]
        perc_diffs = {}
        for group, inst_uncert in self.instrument_uncert.items():
            by_group = (inst_uncert ** 2 + self.spatial_uncerts[group] ** 2) ** (1 / 2)
            rcs = self.rc.copy()
            rcs.loc[0, grp_to_term[group]] = rcs.loc[0, grp_to_term[group]] + by_group
            pred_cap_uncert = self.regression_results.get_prediction(rcs).predicted_mean[0]
            perc_diffs[group] = (pred_cap_uncert - pred_cap) / pred_cap
        df = pd.DataFrame(perc_diffs.values())
        by = (df ** 2).sum().values[0] ** (1 / 2)
        sy = pred.se_obs[0] / pred_cap
        return (by ** 2 + sy ** 2) ** (1 / 2) * k

    def get_filtering_table(self):
        """
        Returns DataFrame showing which filter removed each filtered time interval.

        Time intervals removed are marked with a "1".
        Time intervals kept are marked with a "0".
        Time intervals removed by a previous filter are np.NaN/blank.
        Columns/filters are in order they are run from left to right.
        The last column labeled "all_filters" shows is True for intervals that were
        not removed by any of the filters.
        """
        filtering_data = pd.DataFrame(index=self.data.index)
        for i, (flt_step_kept, flt_step_removed) in (
            enumerate(zip(self.kept, self.removed))
        ):
            if i == 0:
                filtering_data.loc[:, flt_step_removed['name']] = 0
            else:
                filtering_data.loc[self.kept[i - 1]['index'], flt_step_kept['name']] = 0
            filtering_data.loc[flt_step_removed['index'], flt_step_removed['name']] = 1

        filtering_data['all_filters'] = filtering_data.apply(
            lambda x: all(x == 0), axis=1
        )
        return filtering_data

    def print_points_summary(self, hrs_req=12.5):
        """
        print summary data on the number of points collected.
        """
        self.get_length_test_period()
        self.get_pts_required(hrs_req=hrs_req)
        self.set_test_complete(self.pts_required)
        pts_collected = self.data_filtered.shape[0]
        avg_pts_per_day = pts_collected / self.length_test_period
        print('length of test period to date: {} days'.format(self.length_test_period))
        if self.test_complete:
            print('sufficient points have been collected. {} points required; '
                  '{} points collected'.format(self.pts_required, pts_collected))
        else:
            print('{} points of {} points needed, {} remaining to collect.'.format(
                pts_collected,
                self.pts_required,
                self.pts_required - pts_collected)
            )
            print('{:0.2f} points / day on average.'.format(avg_pts_per_day))
            print('Approximate days remaining: {:0.0f}'.format(
                round(((self.pts_required - pts_collected) / avg_pts_per_day), 0) + 1)
            )

    def get_length_test_period(self):
        """
        Get length of test period.

        Uses length of `data` unless `filter_time` has been run, then uses length
        of the kept data after `filter_time` was run the first time. Subsequent
        uses of `filter_time` are ignored.

        Rounds up to a period of full days.

        Returns
        -------
        int
            Days in test period.
        """
        test_period = self.data.index[-1] - self.data.index[0]
        for filter in self.kept:
            if 'filter_time' == filter['name']:
                test_period = filter['index'][-1] - filter['index'][0]
        self.length_test_period = test_period.ceil('D').days

    def get_pts_required(self, hrs_req=12.5):
        """
        Set number of data points required for complete test attribute.

        Parameters
        ----------
        hrs_req : numeric, default 12.5
            Number of hours to be represented by final filtered test data set.
            Default of 12.5 hours is dictated by ASTM E2848 and corresponds to
            750 1-minute data points, 150 5-minute, or 50 15-minute points.
        """
        self.pts_required = (
            (hrs_req * 60) /
            util.get_common_timestep(self.data, units='m', string_output=False)
        )

    def set_test_complete(self, pts_required):
        """Sets `test_complete` attribute.

        Parameters
        ----------
        pts_required : int
            Number of points required to remain after filtering for a complete test.
        """
        self.test_complete = self.data_filtered.shape[0] >= pts_required

    def column_groups_to_excel(self, save_to='./column_groups.xlsx'):
        """Export the column groups attribute to an excel file.

        Parameters
        ----------
        save_to : str
            File path to save column groups to. Should include .xlsx.
        """
        pd.DataFrame.from_dict(
                self.column_groups.data, orient='index'
        ).stack().to_frame().droplevel(1).to_excel(save_to, header=False)


if __name__ == "__main__":
    import doctest
    import pandas as pd  # noqa F811

    das = CapData('das')
    das.load_data(path='../examples/data/', fname='example_meas_data.csv',
                  source='AlsoEnergy')
    das.set_regression_cols(power='-mtr-', poa='irr-poa-',
                            t_amb='temp-amb-', w_vel='wind--')

    doctest.testmod()
