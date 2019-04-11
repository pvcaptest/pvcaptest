# standard library imports
import os
import datetime
import re
import math
import copy
import collections
from functools import wraps
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
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.palettes import Category10, Category20c, Category20b
from bokeh.layouts import gridplot
from bokeh.models import Legend, HoverTool, tools, ColumnDataSource

# pvlib imports
pvlib_spec = importlib.util.find_spec('pvlib')
if pvlib_spec is not None:
    from pvlib.location import Location
    from pvlib.pvsystem import PVSystem
    from pvlib.tracking import SingleAxisTracker
    from pvlib.pvsystem import retrieve_sam
    from pvlib.modelchain import ModelChain
else:
    warnings.warn('Clear sky functions will not work without the '
                  'pvlib package.')


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
                        'amb', 'cell temperature', 'TArray'],
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
             ('pvsyt_losses', [['IL Pmax', 'IL Pmin', 'IL Vmax', 'IL Vmin'],
                               (-1000000000, 100000000)]),
             ('index', [['index'], ('', 'z')])])

sub_type_defs = collections.OrderedDict([
                 ('ghi', [['sun2', 'global horizontal', 'ghi', 'global',
                           'GlobHor']]),
                 ('poa', [['sun', 'plane of array', 'poa', 'GlobInc']]),
                 ('amb', [['TempF', 'ambient', 'amb']]),
                 ('mod', [['Temp1', 'module', 'mod', 'TArray']]),
                 ('mtr', [['revenue meter', 'rev meter', 'billing meter', 'meter']]),
                 ('inv', [['inverter', 'inv']])])

irr_sensors_defs = {'ref_cell': [['reference cell', 'reference', 'ref',
                                  'referance', 'pvel']],
                    'pyran': [['pyranometer', 'pyran']],
                    'clear_sky':[['csky']]}

columns = ['pts_before_filter', 'pts_removed', 'filter_arguments']


def update_summary(func):
    """
    Todo
    ----
    not in place
        Check if summary is updated when function is called with inplace=False.
        It should not be.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        pts_before = self.df_flt.shape[0]
        if pts_before == 0:
            pts_before = self.df.shape[0]
            self.summary_ix.append((self.name, 'count'))
            self.summary.append({columns[0]: pts_before,
                                 columns[1]: 0,
                                 columns[2]: 'no filters'})

        ret_val = func(self, *args, **kwargs)

        arg_str = args.__repr__()
        lst = arg_str.split(',')
        arg_lst = [item.strip("'() ") for item in lst]
        # arg_lst_one = arg_lst[0]
        # if arg_lst_one == 'das' or arg_lst_one == 'sim':
        #     arg_lst = arg_lst[1:]
        # arg_str = ', '.join(arg_lst)

        kwarg_str = kwargs.__repr__()
        kwarg_str = kwarg_str.strip('{}')

        if len(arg_str) == 0 and len(kwarg_str) == 0:
            arg_str = 'no arguments'
        elif len(arg_str) == 0:
            arg_str = kwarg_str
        else:
            arg_str = arg_str + ', ' + kwarg_str

        pts_after = self.df_flt.shape[0]
        pts_removed = pts_before - pts_after
        self.summary_ix.append((self.name, func.__name__))
        self.summary.append({columns[0]: pts_after,
                             columns[1]: pts_removed,
                             columns[2]: arg_str})

        return ret_val
    return wrapper


def cntg_eoy(df, start, end):
    """
    Shifts data before or after new year to form a contigous time period.

    This function shifts data from the end of the year a year back or data from
    the begining of the year a year forward, to create a contiguous time period.
    Intended to be used on historical typical year data.

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
        df_beg = df.loc[start:, :]

        df_end = df.copy()
        df_end.index = df_end.index + pd.DateOffset(days=365)
        df_end = df_end.loc[:end, :]

    elif df.index[0].year == end.year:
        df_end = df.loc[:end, :]

        df_beg = df.copy()
        df_beg.index = df_beg.index - pd.DateOffset(days=365)
        df_beg = df_beg.loc[start:, :]

    df_return = pd.concat([df_beg, df_end], axis=0)
    ix_ser = df_return.index.to_series()
    df_return['index'] = ix_ser.apply(lambda x: x.strftime('%m/%d/%Y %H %M'))
    return df_return


def spans_year(start_date, end_date):
    """
    Returns boolean indicating if dates passes are in the same year.

    Parameters
    ----------

    start_date: pandas Timestamp
    end_date: pandas Timestamp
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
    mnth_int = {'JAN': 1, 'FEB': 2, 'APR': 4, 'MAY': 5, 'JUL': 7,
                'AUG': 8, 'OCT': 10, 'NOV': 11}

    if freq in check_freqs:
        warnings.warn('DataFrame index adjusted to be continous through new'
                      'year, but not returned or set to attribute for user.'
                      'This is not an issue if using RCs with'
                      'predict_capacities.')
        if isinstance(freq, str):
            mnth = mnth_int[freq.split('-')[1]]
        else:
            mnth = freq.startingMonth
        year = df.index[0].year
        mnths_eoy = 12 - mnth
        mnths_boy = 3 - mnths_eoy
        if int(mnth) >= 10:
            str_date = str(mnths_boy) + '/' + str(year)
        else:
            str_date = str(mnth) + '/' + str(year)
        tdelta = df.index[1] - df.index[0]
        date_to_offset = df.loc[str_date].index[-1].to_pydatetime()
        start = date_to_offset + tdelta
        end = date_to_offset + pd.DateOffset(years=1)
        if mnth < 8 or mnth >= 10:
            df = cntg_eoy(df, start, end)
        else:
            df = cntg_eoy(df, end, start)
        return df
    else:
        return df


def perc_wrap(p):
    def numpy_percentile(x):
        return np.percentile(x.T, p, interpolation='nearest')
    return numpy_percentile


def perc_bounds(perc):
    """
    perc_flt : float or tuple, default None
        Percentage or tuple of percentages used to filter around reporting
        irradiance in the irrRC_balanced function.  Required argument when
            irr_bal is True.
    """
    if isinstance(perc, tuple):
        perc_low = perc[0] / 100
        perc_high = perc[1] / 100
    else:
        perc_low = perc / 100
        perc_high = perc / 100
    low = 1 - (perc_low)
    high = 1 + (perc_high)
    return (low, high)


def flt_irr(df, irr_col, low, high, ref_val=None):
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
        Must provide arg when min/max are fractions

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


def filter_grps(grps, rcs, irr_col, low, high, **kwargs):
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
    **kwargs
        Passed to pandas Grouper to control label and closed side of intervals.
        See pandas Grouper doucmentation for details. Default is left labeled
        and left closed.

    Returns
    -------
    pandas groupby
    """
    flt_dfs = []
    freq = list(grps.groups.keys())[0].freq
    for grp_name, grp_df in grps:
        ref_val = rcs.loc[grp_name, 'poa']
        grp_df_flt = flt_irr(grp_df, irr_col, low, high, ref_val=ref_val)
        flt_dfs.append(grp_df_flt)
    df_flt = pd.concat(flt_dfs)
    df_flt_grpby = df_flt.groupby(pd.Grouper(freq=freq, **kwargs))
    return df_flt_grpby


def irrRC_balanced(df, low, high, irr_col='GlobInc', plot=False):
    """
    Calculates max irradiance reporting condition that is below 60th percentile.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing irradiance data for calculating the irradiance
        reporting condition.
    low: float
        Bottom value for irradiance filter, usually between 0.5 and 0.8.
    high: float
        Top value for irradiance filter, usually between 1.2 and 1.5.
    irr_col: str
        String that is the name of the column with the irradiance data.
    plot: bool, default False
        Plots graphical view of algorithim searching for reporting irradiance.
        Useful for troubleshooting or understanding the method.

    Returns
    -------
    Tuple
        Float reporting irradiance and filtered dataframe.

    """
    if plot:
        irr = df[irr_col].values
        x = np.ones(irr.shape[0])
        plt.plot(x, irr, 'o', markerfacecolor=(0.5, 0.7, 0.5, 0.1))
        plt.ylabel('irr')
        x_inc = 1.01

    vals_above = 10
    perc = 100.
    pt_qty = 0
    loop_cnt = 0
    pt_qty_array = []
    # print('--------------- MONTH START --------------')
    while perc > 0.6 or pt_qty < 50:
        # print('####### LOOP START #######')
        df_count = df.shape[0]
        df_perc = 1 - (vals_above / df_count)
        # print('in percent: {}'.format(df_perc))
        irr_RC = (df[irr_col].agg(perc_wrap(df_perc * 100)))
        # print('ref irr: {}'.format(irr_RC))
        flt_df = flt_irr(df, irr_col, low, high, ref_val=irr_RC)
        # print('number of vals: {}'.format(df.shape))
        pt_qty = flt_df.shape[0]
        # print('flt pt qty: {}'.format(pt_qty))
        perc = stats.percentileofscore(flt_df[irr_col], irr_RC) / 100
        # print('out percent: {}'.format(perc))
        vals_above += 1
        pt_qty_array.append(pt_qty)
        if perc <= 0.6 and pt_qty <= pt_qty_array[loop_cnt - 1]:
            break
        loop_cnt += 1

        if plot:
            x_inc += 0.02
            y1 = irr_RC * low
            y2 = irr_RC * high
            plt.plot(x_inc, irr_RC, 'ro')
            plt.plot([x_inc, x_inc], [y1, y2])

    if plot:
        plt.show()
    return(irr_RC, flt_df)


def fit_model(df, fml='power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'):
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
    Calculates predicted values for given linear models and predictor values.

    Evaluates the first linear model in the iterable with the first row of the
    predictor values in the dataframe.  Passed arguments must be aligned.

    Parameters
    ----------
    regs : iterable of statsmodels regression results wrappers
    rcs : pandas dataframe
        Dataframe of predictor values used to evaluate each linear model.
        The column names must match the strings used in the regression formuala.

    Returns
    -------
    Pandas series of predicted values.
    """
    pred_cap = pd.Series()
    for i, mod in enumerate(regs):
        RC_df = pd.DataFrame(rcs.iloc[i, :]).T
        pred_cap = pred_cap.append(mod.predict(RC_df))
    return pred_cap


def pred_summary(grps, rcs, allowance, **kwargs):
    """
    Creates summary table of reporting conditions, pred cap, and gauranteed cap.

    This method does not calculate reporting conditions.

    Parameters
    ----------
    grps : pandas groupby object
        Solar data grouped by season or month used to calculate reporting
        conditions.  This argument is used to fit models for each group.
    rcs : pandas dataframe
        Dataframe of reporting conditions used to predict capacities.
    allowance : float
        Percent allowance to calculate gauranteed capacity from predicted capacity.

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
                params.rename(columns={param_col_name: param_col_name + '-param'},
                              inplace=True)

    results = pd.concat([rcs, predictions, params], axis=1)

    results['guaranteedCap'] = results['PredCap'] * (1 - allowance)
    results['pt_qty'] = pt_qty.values

    return results


def pvlib_location(loc):
    """
    Creates a pvlib location object.

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
    Creates a pvlib PVSystem or SingleAxisTracker object.

    A SingleAxisTracker object is created if any of the keyword arguments for
    initiating a SingleAxisTracker object are found in the keys of the passed
    dictionary.

    Parameters
    ----------
    sys : dict
        Dictionary of keywords required to create a pvlib SingleAxisTracker
        or PVSystem.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.
        https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.pvsystem.PVSystem.html
        https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.tracking.SingleAxisTracker.html

    Returns
    -------
    pvlib PVSystem or SingleAxisTracker object.
    """
    sandia_modules = retrieve_sam('SandiaMod')
    cec_inverters = retrieve_sam('cecinverter')
    sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    trck_kwords = ['axis_tilt', 'axis_azimuth', 'max_angle', 'backtrack', 'gcr']
    if any(kword in sys.keys() for kword in trck_kwords):
        system = SingleAxisTracker(**sys,
                                   module_parameters=sandia_module,
                                   inverter_parameters=cec_inverter)
    else:
        system = PVSystem(**sys,
                          module_parameters=sandia_module,
                          inverter_parameters=cec_inverter)

    return system


def get_tz_index(time_source, loc):
    """
    Creates DatetimeIndex with timezone aligned with location dictionary.

    Handles generating a DatetimeIndex with a timezone for use as an agrument
    to pvlib ModelChain prepare_inputs method or pvlib Location get_clearsky
    method.

    Parameters
    ----------
    time_source : dataframe or DatetimeIndex
        If passing a dataframe the index of the dataframe will be used.  If the
        index does not have a timezone the timezone will be set using the
        timezone in the passed loc dictionary.
        If passing a DatetimeIndex with a timezone it will be returned directly.
        If passing a DatetimeIndex without a timezone the timezone in the
        timezone dictionary will be used.

    Returns
    -------
    DatetimeIndex with timezone
    """

    if isinstance(time_source, pd.core.indexes.datetimes.DatetimeIndex):
        if time_source.tz is None:
            time_source = time_source.tz_localize(loc['tz'], ambiguous='infer',
                                                  errors='coerce')
            return time_source
        else:
            if pytz.timezone(loc['tz']) != time_source.tz:
                warnings.warn('Passed a DatetimeIndex with a timezone that '
                              'does not match the timezone in the loc dict. '
                              'Using the timezone of the DatetimeIndex.')
            return time_source
    elif isinstance(time_source, pd.core.frame.DataFrame):
        if time_source.index.tz is None:
            return time_source.index.tz_localize(loc['tz'], ambiguous='infer',
                                                 errors='coerce')
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
        timezone in the passed loc dictionary.
        If passing a DatetimeIndex with a timezone it will be returned directly.
        If passing a DatetimeIndex without a timezone the timezone in the
        timezone dictionary will be used.
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
        Dictionary of keywords required to create a pvlib SingleAxisTracker
        or PVSystem.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.
        https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.pvsystem.PVSystem.html
        https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.tracking.SingleAxisTracker.html
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

    if output == 'both':
        ghi = location.get_clearsky(times=times)
        mc.prepare_inputs(times=times)
        csky_df = pd.DataFrame({'poa_mod_csky': mc.total_irrad['poa_global'],
                                'ghi_mod_csky': ghi['ghi']})
    if output == 'poa_all':
        mc.prepare_inputs(times=times)
        csky_df = mc.total_irrad
    if output == 'ghi_all':
        csky_df = location.get_clearsky(times=times)
    if output == 'all':
        ghi = location.get_clearsky(times=times)
        mc.prepare_inputs(times=times)
        csky_df = pd.concat([mc.total_irrad, ghi], axis=1)

    ix_no_tz = csky_df.index.tz_localize(None, ambiguous='infer',
                                         errors='coerce')
    csky_df.index = ix_no_tz

    if concat:
        if isinstance(time_source, pd.core.frame.DataFrame):
            df_with_csky = pd.concat([time_source, csky_df], axis=1)
            return df_with_csky
        else:
            warnings.warn('time_source is not a dataframe; only clear sky data\
                           returned')
            return csky_df
    else:
        return csky_df


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
    name : str
        Name for the CapData object.
    df : pandas dataframe
        Used to store measured or simulated data imported from csv.
    df_flt : pandas dataframe
        Holds filtered data.  Filtering methods act on and write to this
        attribute.
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
    summary_ix : list of tuples
        Holds the row index data modified by the update_summary decorator
        function.
    summary : list of dicts
        Holds the data modifiedby the update_summary decorator function.
    rc : DataFrame
        Dataframe for the reporting conditions (poa, t_amb, and w_vel).
    ols_model : statsmodels linear regression model
        Holds the linear regression model object.
    reg_fml : str
        Regression formula to be fit to measured and simulated data.  Must
        follow the requirements of statsmodels use of patsy.
    tolerance : str
        String representing error band.  Ex. '+ 3', '+/- 3', '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    """

    def __init__(self, name):
        super(CapData, self).__init__()
        self.name = name
        self.df = pd.DataFrame()
        self.df_flt = None
        self.trans = {}
        self.trans_keys = []
        self.reg_trans = {}
        self.trans_abrev = {}
        self.col_colors = {}
        self.summary_ix = []
        self.summary = []
        self.rc = None
        self.ols_model = None
        self.reg_fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        self.tolerance = None

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
        cd_c = CapData('')
        cd_c.df = self.df.copy()
        cd_c.df_flt = self.df_flt.copy()
        cd_c.trans = copy.copy(self.trans)
        cd_c.trans_keys = copy.copy(self.trans_keys)
        cd_c.reg_trans = copy.copy(self.reg_trans)
        cd_c.trans_abrev = copy.copy(self.trans_abrev)
        cd_c.col_colors = copy.copy(self.col_colors)
        cd_c.name = copy.copy(self.name)
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

    def load_data(self, path='./data/', fname=None, set_trans=True,
                  trans_report=True, source=None, load_pvsyst=False,
                  clear_sky=False, loc=None, sys=None, **kwargs):
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
        trans_report : bool, default True
            If set_trans is true, then method prints summary of translation
            dictionary process including any possible data issues.  No effect
            on method when set to False.
        source : str, default None
            Default of None uses general approach that concatenates header data.
            Set to 'AlsoEnergy' to use column heading parsing specific to
            downloads from AlsoEnergy.
        load_pvsyst : bool, default False
            By default skips any csv file that has 'pvsyst' in the name.  Is
            not case sensitive.  Set to true to import a csv with 'pvsyst' in
            the name and skip all other files.
        clear_sky : bool, default False
            Set to true and provide loc and sys arguments to add columns of
            clear sky modeled poa and ghi to loaded data.
        loc : dict
            See the csky function for details on dictionary options.
        sys : dict
            See the csky function for details on dictionary options.
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

        if not load_pvsyst:
            if clear_sky:
                if loc is None:
                    warnings.warn('Must provide loc and sys dictionary\
                                  when clear_sky is True.  Loc dict missing.')
                if sys is None:
                    warnings.warn('Must provide loc and sys dictionary\
                                  when clear_sky is True.  Sys dict missing.')
                self.df = csky(self.df, loc=loc, sys=sys, concat=True,
                               output='both')

        if set_trans:
            self.__set_trans(trans_report=trans_report)

        self.df_flt = self.df.copy()

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
                        type_min = type_defs[key][1][0]
                        type_max = type_defs[key][1][1]
                        ser_min = series.min()
                        ser_max = series.max()
                        min_bool = ser_min >= type_min
                        max_bool = ser_max <= type_max
                        if min_bool and max_bool:
                            return key
                        else:
                            if warnings:
                                if not min_bool and not max_bool:
                                    print('{} in {} is below {} for '
                                    '{}'.format(ser_min, series.name,
                                    type_min, key))
                                    print('{} in {} is above {} for '
                                    '{}'.format(ser_max, series.name,
                                    type_max, key))
                                elif not min_bool:
                                    print('{} in {} is below {} for '
                                    '{}'.format(ser_min, series.name,
                                    type_min, key))
                                elif not max_bool:
                                    print('{} in {} is above {} for '
                                    '{}'.format(ser_max, series.name,
                                    type_max, key))
                            return key
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

    def __set_trans(self, trans_report=True):
        """
        Creates a dict of raw column names paired to categorical column names.

        Uses multiple type_def formatted dictionaries to determine the type,
        sub-type, and equipment type for data series of a dataframe.  The determined
        types are concatenated to a string used as a dictionary key with a list
        of one or more oringal column names as the paried value.

        Parameters
        ----------
        trans_report : bool, default True
            Sets the warnings option of __series_type when applied to determine
            the column types.

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
        col_types = self.df.apply(self.__series_type, args=(type_defs,),
                                  warnings=trans_report).tolist()
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
            keys = list(self.reg_trans.values())
        elif isinstance(ind_var, list) and len(ind_var) > 1:
            keys = [self.reg_trans[key] for key in ind_var]
        elif ind_var in met_keys:
            ind_var = [ind_var]
            keys = [self.reg_trans[key] for key in ind_var]

        lst = []
        for key in keys:
            lst.extend(self.trans[key])
        if filtered_data:
            return self.df_flt[lst]
        else:
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

    def plot(self, marker='line', ncols=2, width=400, height=350,
             legends=False, merge_grps=['irr', 'temp'], subset=None,
             filtered=False, **kwargs):
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
            Use 'irr-poa' and 'irr-ghi' to plot clear sky modeled with measured
            data.
        subset : list, default None
            List of the translation dictionary keys to use to control order of
            plots or to plot only a subset of the plots.
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
            dframe = self.df_flt
        else:
            dframe = self.df
        dframe.index.name = 'Timestamp'

        names_to_abrev = {val: key for key, val in self.trans_abrev.items()}

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
                if col.find('csky') == -1:
                    line_dash = 'solid'
                else:
                    line_dash = (5, 2)
                if marker == 'line':
                    series = p.line('Timestamp', col, source=source,
                                    line_color=self.col_colors[col],
                                    line_dash=line_dash,
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

    def reset_flt(self):
        """
        Copies over filtered dataframe with raw data and removes all summary
        history.

        Parameters
        ----------
        data : str
            'sim' or 'das' determines if filter is on sim or das data.
        """
        self.df_flt = self.df.copy()
        self.summary_ix = []
        self.summary = []

    def __get_poa_col(self):
        """
        Returns poa column name from translation dictionary.

        Also, issues warning if there are more than one poa columns in the
        translation dictionary.
        """
        poa_cols = self.trans[self.reg_trans['poa']]
        if len(poa_cols) > 1:
            return warnings.warn('{} columns of irradiance data. '
                                 'Use col_name to specify a single '
                                 'column.'.format(len(poa_cols)))
        else:
            return poa_cols[0]

    @update_summary
    def filter_irr(self, low, high, ref_val=None, col_name=None, inplace=True):
        """
        Filter on irradiance values.

        Parameters
        ----------
        low : float or int
            Minimum value as fraction (0.8) or absolute 200 (W/m^2)
        high : float or int
            Max value as fraction (1.2) or absolute 800 (W/m^2)
        ref_val : float or int
            Must provide arg when min/max are fractions
        col_name : str, default None
            Column name of irradiance data to filter.  By default uses the POA
            irradiance set in reg_trans attribute or average of the POA columns.
        inplace : bool, default True
            Default true write back to df_flt or return filtered dataframe.

        Returns
        -------
        DataFrame
            Filtered dataframe if inplace is False.
        """
        if col_name is None:
            irr_col = self.__get_poa_col()
        else:
            irr_col = col_name

        df_flt = flt_irr(self.df_flt, irr_col, low, high,
                         ref_val=ref_val)
        if inplace:
            self.df_flt = df_flt
        else:
            return df_flt

    @update_summary
    def filter_pvsyst(self, shade=1.0, inplace=True):
        """
        Filter pvsyst data for shading and off mppt operation.

        This function is only applicable to simulated data generated by PVsyst.
        Filters the 'IL Pmin', IL Vmin', 'IL Pmax', 'IL Vmax' values if they are
        greater than 0.

        Parameters
        ----------
        shade: float, default 1.0
            Filters on the PVsyst output variable FShdBm.  Default is to remove
            any averaging interval with FshdBm < 1.0.
        inplace: bool, default True
            If inplace is true, then function overwrites the filtered data for
            sim or das.  If false returns a CapData object.

        Returns
        -------
        CapData object if inplace is set to False.
        """
        df = self.df_flt

        index_shd = df.query('FShdBm>=@shade').index

        columns = ['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax']
        index_IL = df[df[columns].sum(axis=1) <= 0].index
        index = index_shd.intersection(index_IL)

        if inplace:
            self.df_flt = self.df_flt.loc[index, :]
        else:
            return self.df_flt.loc[index, :]

    @update_summary
    def filter_time(self, start=None, end=None, days=None, test_date=None,
                    inplace=True, wrap_year=False):
        """
        Function wrapping pandas dataframe selection methods.

        Parameters
        ----------
        start: str
            Start date for data to be returned.  Must be in format that can be
            converted by pandas.to_datetime.  Not required if test_date and days
            arguments are passed.
        end: str
            End date for data to be returned.  Must be in format that can be
            converted by pandas.to_datetime.  Not required if test_date and days
            arguments are passed.
        days: int
            Days in time period to be returned.  Not required if start and end
            are specified.
        test_date: str
            Must be format that can be converted by pandas.to_datetime.  Not
            required if start and end are specified.  Requires days argument.
            Time period returned will be centered on this date.
        inplace : bool
            Default true write back to CapTest.flt_sim or flt_das

        Todo
        ----
        Add inverse options to remove time between start end rather than return
        it
        """
        if start is not None and end is not None:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if wrap_year and spans_year(start, end):
                df_temp = cntg_eoy(self.df_flt, start, end)
            else:
                df_temp = self.df_flt.loc[start:end, :]

        if start is not None and end is None:
            if days is None:
                return warnings.warn("Must specify end date or days.")
            else:
                start = pd.to_datetime(start)
                end = start + pd.DateOffset(days=days)
                if wrap_year and spans_year(start, end):
                    df_temp = cntg_eoy(self.df_flt, start, end)
                else:
                    df_temp = self.df_flt.loc[start:end, :]

        if start is None and end is not None:
            if days is None:
                return warnings.warn("Must specify end date or days.")
            else:
                end = pd.to_datetime(end)
                start = end - pd.DateOffset(days=days)
                if wrap_year and spans_year(start, end):
                    df_temp = cntg_eoy(self.df_flt, start, end)
                else:
                    df_temp = self.df_flt.loc[start:end, :]

        if test_date is not None:
            test_date = pd.to_datetime(test_date)
            if days is None:
                return warnings.warn("Must specify days")
            else:
                offset = pd.DateOffset(days=days // 2)
                start = test_date - offset
                end = test_date + offset
                if wrap_year and spans_year(start, end):
                    df_temp = cntg_eoy(self.df_flt, start, end)
                else:
                    df_temp = self.df_flt.loc[start:end, :]

        if inplace:
            self.df_flt = df_temp
        else:
            return df_temp

    @update_summary
    def filter_outliers(self, inplace=True, **kwargs):
        """
        Apply eliptic envelope from scikit-learn to remove outliers.

        Parameters
        ----------
        inplace : bool
            Default of true writes filtered dataframe back to df_flt attribute.
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
        X1 = XandY.values

        if 'support_fraction' not in kwargs.keys():
            kwargs['support_fraction'] = 0.9
        if 'contamination' not in kwargs.keys():
            kwargs['contamination'] = 0.04

        clf_1 = sk_cv.EllipticEnvelope(**kwargs)
        clf_1.fit(X1)

        if inplace:
            self.df_flt = self.df_flt[clf_1.predict(X1) == 1]
        else:
            return self.df_flt[clf_1.predict(X1) == 1]

    @update_summary
    def filter_pf(self, pf, inplace=True):
        """
        Keep timestamps where all power factors are greater than or equal to
        pf.

        Parameters
        ----------
        pf: float
            0.999 or similar to remove timestamps with lower PF values
        inplace : bool
            Default of true writes filtered dataframe back to df_flt attribute.

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

        df = self.df_flt[self.trans[selection]]

        df_flt = self.df_flt[(np.abs(df) >= pf).all(axis=1)]

        if inplace:
            self.df_flt = df_flt
        else:
            return df_flt

    @update_summary
    def custom_filter(self, func, *args, **kwargs):
        """
        Applies update_summary to custom function.

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

        >>> das.reset_flt()
        >>> das.custom_filter(pd.DataFrame.between_time, '9:00', '13:00')
        >>> summary = das.get_summary()
        >>> summary['pts_before_filter'][0]
        245
        >>> summary['pts_removed'][0]
        1195
        >>> das.df_flt.index[0].hour
        9
        >>> das.df_flt.index[-1].hour
        13
        """
        self.df_flt = func(self.df_flt, *args, **kwargs)

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
        Prints summary dataframe of the filtering applied df_flt attribute.

        The summary dataframe shows the history of the filtering steps applied
        to the data including the timestamps remaining after each step, the
        timestamps removed by each step and the arguments used to call each
        filtering method.

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
    def rep_cond(self, irr_bal=False, perc_flt=None, w_vel=None, inplace=True,
                 func={'poa': perc_wrap(60), 't_amb': 'mean', 'w_vel': 'mean'},
                 freq=None, **kwargs):

        """
        Calculate reporting conditons.

        Parameters
        ----------
        irr_bal: boolean, default False
            If true, uses the irrRC_balanced function to determine the reporting
            conditions. Replaces the calculations specified by func with or
            without freq.
        perc_flt : float or tuple, default None
            Percentage or tuple of percentages used to filter around reporting
            irradiance in the irrRC_balanced function.  Required argument when
            irr_bal is True.
            Tuple option allows specifying different percentage for above and
            below reporting irradiance. (below, above)
        func: callable, string, dictionary, or list of string/callables
            Determines how the reporting condition is calculated.
            Default is a dictionary poa - 60th numpy_percentile, t_amb - mean
                                          w_vel - mean
            Can pass a string function ('mean') to calculate each reporting
            condition the same way.
        freq: str
            String pandas offset alias to specify aggregattion frequency
            for reporting condition calculation. Ex '60D' for 60 Days or
            'MS' for months start.
        w_vel: int
            If w_vel is not none, then wind reporting condition will be set to
            value specified for predictions. Does not affect output unless pred
            is True and irr_bal is True.
        inplace: bool, True by default
            When true updates object rc parameter, when false returns dicitionary
            of reporting conditions.
        **kwargs
            Passed to pandas Grouper to control label and closed side of
            intervals. See pandas Grouper doucmentation for details. Default is
            left labeled and left closed.


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
            if perc_flt is None:
                return warnings.warn('perc_flt required when irr_bal is True')
            else:
                low, high = perc_bounds(perc_flt)

                results = irrRC_balanced(df, low, high, irr_col='poa')
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
            df_grpd = df.groupby(pd.Grouper(freq=freq, **kwargs))

            if irr_bal:
                freq = list(df_grpd.groups.keys())[0].freq
                ix = pd.DatetimeIndex(list(df_grpd.groups.keys()), freq=freq)
                low, high = perc_bounds(perc_flt)
                poa_RC = []
                temp_RC = []
                wind_RC = []
                for name, mnth in df_grpd:
                    results = irrRC_balanced(mnth, low, high, irr_col='poa')
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

    def predict_capacities(self, irr_flt=True, perc_flt=20, **kwargs):
        """
        Calculate expected capacities.

        Parameters
        ----------
        irr_flt : bool, default True
            When true will filter each group of data by a percentage around the
            reporting irradiance for that group.  The data groups are determined
            from the reporting irradiance attribute.
        perc_flt : float or int or tuple, default 20
            Percentage or tuple of percentages used to filter around reporting
            irradiance in the irrRC_balanced function.  Required argument when
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

        low, high = perc_bounds(perc_flt)
        freq = self.rc.index.freq
        df = wrap_seasons(df, freq)
        grps = df.groupby(by=pd.Grouper(freq=freq, **kwargs))

        if irr_flt:
            grps = filter_grps(grps, self.rc, 'poa', low, high)

        error = float(self.tolerance.split(sep=' ')[1]) / 100
        results = pred_summary(grps, self.rc, error,
                               fml=self.reg_fml)

        return results

if __name__ == "__main__":
    import doctest
    import pandas as pd

    das = CapData('das')
    das.load_data(path='../examples/data/', fname='example_meas_data.csv',
                  source='AlsoEnergy')
    das.set_reg_trans(power='-mtr-', poa='irr-poa-', t_amb='temp-amb-',
                      w_vel='wind--')

    doctest.testmod()
