# standard library imports
import os
import datetime
import re
import math
import copy
import collections
from functools import wraps
import warnings
import importlib
import warnings

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

# visualization library imports
hv_spec = importlib.util.find_spec('holoviews')
if hv_spec is not None:
    import holoviews as hv
else:
    warnings.warn('Some plotting functions will not work without the '
                   'holoviews package.')

from captest.capdata import CapData, met_keys

# EllipticEnvelope gives warning about increasing determinate that prints
# out in a loop and does not seem to affect result of removing outliers.
warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                        module='sklearn')

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

        arg_str = args.__repr__()
        lst = arg_str.split(',')
        arg_lst = [item.strip("'() ") for item in lst]
        arg_lst_one = arg_lst[0]
        if arg_lst_one == 'das' or arg_lst_one == 'sim':
            arg_lst = arg_lst[1:]
        arg_str = ', '.join(arg_lst)

        kwarg_str = kwargs.__repr__()
        kwarg_str = kwarg_str.strip('{}')

        if len(arg_str) == 0 and len(kwarg_str) == 0:
            arg_str = 'no arguments'
        elif len(arg_str) == 0:
            arg_str = kwarg_str
        else:
            arg_str = arg_str + ', ' + kwarg_str

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


def highlight_pvals(s):
    """
    Highlight vals greater than or equal to 0.05 in a Series yellow.
    """
    is_greaterthan = s >= 0.05
    return ['background-color: yellow' if v else '' for v in is_greaterthan]


def perc_wrap(p):
    def numpy_percentile(x):
        return np.percentile(x.T, p, interpolation='nearest')
    return numpy_percentile


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


def flt_irr(df, irr_col, low, high, ref_val=None):
    """
    Top level filter on irradiance values.

    Parameters
    ----------
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


class CapTest(object):
    """
    CapTest provides methods to facilitate solar PV capacity testing.

    The CapTest class provides a framework to facilitate visualizing, filtering,
    and performing regressions on data typically collected from operating solar
    pv plants or solar energy production models.

    The class parameters include an unmodified CapData object and filtered
    CapData object for both measured and simulated data.

    Parameters
    ----------
    das : CapData, required
        The CapData object containing data from a data acquisition system (das).
        This is the measured data used to perform a capacity test.
    flt_das : CapData
        A CapData object containing a filtered version of the das data.  Filter
        methods always modify this attribute or flt_sim.
    das_mindex : list of tuples
        Holds the row index data modified by the update_summary decorator
        function.
    das_summ_data : list of dicts
        Holds the data modifiedby the update_summary decorator function.
    sim : CapData, required
        Identical to das for data from an energy production simulation.
    flt_sim : CapData
        Identical to flt_das for data from an energy production simulation.
    sim_mindex : list of tuples
        Identical to das_mindex for data from an energy production simulation.
    sim_summ_data : list of dicts
        Identical to das_summ_data for data from an energy production simulation.
    rc : dict of lists
        Dictionary of lists for the reporting conditions (poa, t_amb, and w_vel).
    ols_model_das : statsmodels linear regression model
        Holds the linear regression model object for the das data.
    ols_model_sim : statsmodels linear regression model
        Identical to ols_model_das for simulated data.
    reg_fml : str
        Regression formula to be fit to measured and simulated data.  Must
        follow the requirements of statsmodels use of patsy.
    tolerance : str
        String representing error band.  Ex. '+ 3', '+/- 3', '- 5'
        There must be space between the sign and number. Number is
        interpreted as a percent.  For example, 5 percent is 5 not 0.05.
    """

    def __init__(self, das, sim, tolerance):
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
        self.reg_fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'
        self.tolerance = tolerance

    def summary(self):
        """
        Prints summary dataframe of the filtering applied to flt_das and flt_sim.

        The summary dataframe shows the history of the filtering steps applied
        to the measured and simulated data including the timestamps remaining
        after each step, the timestamps removed by each step and the arguments
        used to call each filtering method.

        Parameters
        ----------
        None

        Returns
        -------
        Pandas DataFrame
        """
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

    def scatter(self, data):
        """
        Create scatter plot of irradiance vs power.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if plot is of sim or das data.
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

    def scatter_hv(self, data, timeseries=False):
        """
        Create holoview scatter plot of irradiance vs power.  Optional linked
        time series plot of the same data.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if plot is of sim or das data.
        timeseries : boolean, default False
            True adds timeseries plot of power data with linked brushing.
        """
        flt_cd = self.__flt_setup(data)
        new_names = ['power', 'poa', 't_amb', 'w_vel']
        df = flt_cd.rview(new_names).copy()
        rename = {old: new for old, new in zip(df.columns, new_names)}
        df.rename(columns=rename, inplace=True)
        df['index'] = flt_cd.df.loc[:,'index']
        df.index.name = 'date_index'
        df['date'] = df.index.values

        opt_dict = {'Scatter': {'style': dict(size=5),
                                'plot': dict(tools=['box_select', 'lasso_select',
                                                    'hover'],
                                             legend_position='right',
                                             height=400, width=500,
                                             shared_datasource=True,)},
                    'Curve': {'plot': dict(tools=['box_select', 'lasso_select',
                                                  'hover'],
                                           shared_datasource=True, height=400,
                                           width=800)},
                    'Layout': {'plot': dict(shared_datasource=True)},
                    'VLine': {'style': dict(color='gray', line_width=1)}}

        poa_vs_kw = hv.Scatter(df, 'poa', ['power', 'poa', 'w_vel', 'index'])
        poa_vs_time = hv.Curve(df, 'date', 'power')
        layout_scatter = (poa_vs_kw).opts(opt_dict)
        layout_timeseries = (poa_vs_kw + poa_vs_time).opts(opt_dict)
        if timeseries:
            return(layout_timeseries.cols(1))
        else:
            return(layout_scatter)

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

    @update_summary
    def rep_cond(self, data, *args, test_date=None, days=60, inplace=True,
                 freq=None, func={'poa': perc_wrap(60), 't_amb': 'mean',
                                  'w_vel': 'mean'},
                 pred=False, irr_bal=False, w_vel=None, **kwargs):

        """
        Calculate reporting conditons.

        NOTE: Can pass additional positional arguments for low/high irradiance
        filter.

        Parameters
        ----------
        data: str, 'sim' or 'das'
            'sim' or 'das' determines if filter is on sim or das data
        test_date: str, 'mm/dd/yyyy', optional
            Date to center reporting conditions aggregation functions around.
            When not used specified reporting conditions for all data passed
            are returned grouped by the freq provided.
        days: int, default 60
            Number of days to use when calculating reporting conditons.
            Typically no less than 30 and no more than 90.
        inplace: bool, True by default
            When true updates object rc parameter, when false returns dicitionary
            of reporting conditions.
        freq: str
            String pandas offset alias to specify aggregattion frequency
            for reporting condition calculation. Ex '60D' for 60 Days or
            'M' for months. Typical 'M', '2M', or 'BQ-NOV'.
            'BQ-NOV' is business quarterly year ending in Novemnber i.e. seasons.
        func: callable, string, dictionary, or list of string/callables
            Determines how the reporting condition is calculated.
            Default is a dictionary poa - 60th numpy_percentile, t_amb - mean
                                          w_vel - mean
            Can pass a string function ('mean') to calculate each reporting
            condition the same way.
        pred: boolean, default False
            If true and frequency is specified, then method returns a dataframe
            with reporting conditions, regression parameters, predicted
            capacites, and point quantities for each group.
        irr_bal: boolean, default False
            If true, pred is set to True, and frequency is specified then the
            predictions for each group of reporting conditions use the
            irrRC_balanced function to determine the reporting conditions.
        w_vel: int
            If w_vel is not none, then wind reporting condition will be set to
            value specified for predictions. Does not affect output unless pred
            is True and irr_bal is True.

        Returns
        -------
        dict
            Returns a dictionary of reporting conditions if inplace=False
            otherwise returns None.
        pandas DataFrame
            If pred=True, then returns a pandas dataframe of results.
        """
        flt_cd = self.__flt_setup(data)
        df = flt_cd.rview(['power', 'poa', 't_amb', 'w_vel'])
        df = df.rename(columns={df.columns[0]: 'power',
                                df.columns[1]: 'poa',
                                df.columns[2]: 't_amb',
                                df.columns[3]: 'w_vel'})

        if test_date is not None:
            date = pd.to_datetime(test_date)
            offset = pd.DateOffset(days=days / 2)
            start = date - offset
            end = date + offset

        if data == 'das' and test_date is not None:
            if start < df.index[0]:
                start = df.index[0]
            if end > df.index[-1]:
                end = df.index[-1]
            df = df.loc[start:end, :]

        elif data == 'sim' and test_date is not None:
            if spans_year(start, end):
                df = cntg_eoy(df, start, end)
            else:
                df = df.loc[start:end, :]

        RCs = pd.DataFrame(df.agg(func)).T

        check_freqs = ['BQ-JAN', 'BQ-FEB', 'BQ-APR', 'BQ-MAY', 'BQ-JUL',
                       'BQ-AUG', 'BQ-OCT', 'BQ-NOV']
        mnth_int = {'JAN': 1, 'FEB': 2, 'APR': 4, 'MAY': 5, 'JUL': 7,
                    'AUG': 8, 'OCT': 10, 'NOV': 11}

        if freq is not None and test_date is None:
            if freq in check_freqs:
                mnth = mnth_int[freq.split('-')[1]]
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

            df_grpd = df.groupby(pd.Grouper(freq=freq, label='right'))
            RCs_df = df_grpd.agg(func)
            RCs = RCs_df.to_dict('list')

            if predict:
                if irr_bal:
                    RCs_df = pd.DataFrame()
                    flt_dfs = pd.DataFrame()
                    for name, mnth in df_grpd:
                        results = irrRC_balanced(mnth, *args, irr_col='poa',
                                                 **kwargs)
                        flt_df = results[1]
                        flt_dfs = flt_dfs.append(results[1])
                        temp_RC = flt_df['t_amb'].mean()
                        wind_RC = flt_df['w_vel'].mean()
                        if w_vel is not None:
                            wind_RC = w_vel
                        RCs_df = RCs_df.append({'poa': results[0],
                                                't_amb': temp_RC,
                                                'w_vel': wind_RC}, ignore_index=True)
                    df_grpd = flt_dfs.groupby(by=pd.Grouper(freq='M'))

                error = float(self.tolerance.split(sep=' ')[1]) / 100
                results = pred_summary(df_grpd, RCs_df, error,
                                       fml=self.reg_fml)

        if inplace:
            if pred:
                print('Results dataframe saved to rc attribute.')
                print(results)
                self.rc = results
            else:
                print('Reporting conditions saved to rc attribute.')
                print(RCs)
                self.rc = RCs
        else:
            if pred:
                return results
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

        Returns
        -------
        CapData obj
            If inplace is False, then returns a modified CapData object.
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

    def reg_data(self, arg):
        """
        Todo
        ----
        See rview and renaming code in reg_cpt method.  Move this to this
        function or a top level function.
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
        """
        Returns the filtered sim or das CapData object or a copy of the raw data.
        """
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
        Copies over filtered dataframe with raw data and removes all summary
        history.

        Parameters
        ----------
        data : str
            'sim' or 'das' determines if filter is on sim or das data.
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
    def filter_time(self, data, start=None, end=None, days=None, test_date=None,
                    inplace=True):
        """
        Function wrapping pandas dataframe selection methods.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
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
        flt_cd = self.__flt_setup(data)

        if start != None and end != None:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if data == 'sim' and spans_year(start, end):
                flt_cd.df = cntg_eoy(flt_cd.df, start, end)
            else:
                flt_cd.df = flt_cd.df.loc[start:end, :]

        if start != None and end == None:
            if days == None:
                print("Must specify end date or days.")
            else:
                start = pd.to_datetime(start)
                end = start + pd.DateOffset(days=days)
                if data == 'sim' and spans_year(start, end):
                    flt_cd.df = cntg_eoy(flt_cd.df, start, end)
                else:
                    flt_cd.df = flt_cd.df.loc[start:end, :]

        if start == None and end != None:
            if days == None:
                print("Must specify end date or days.")
            else:
                end = pd.to_datetime(end)
                start = end - pd.DateOffset(days=days)
                if data == 'sim' and spans_year(start, end):
                    flt_cd.df = cntg_eoy(flt_cd.df, start, end)
                else:
                    flt_cd.df = flt_cd.df.loc[start:end, :]

        if test_date != None:
            test_date = pd.to_datetime(test_date)
            if days == None:
                print("Must specify days")
                return
            else:
                offset = pd.DateOffset(days=days/2)
                start = test_date - offset
                end = test_date + offset
                if data == 'sim' and spans_year(start, end):
                    flt_cd.df = cntg_eoy(flt_cd.df, start, end)
                else:
                    flt_cd.df = flt_cd.df.loc[start:end, :]

        if inplace:
            if data == 'das':
                self.flt_das = flt_cd
            if data == 'sim':
                self.flt_sim = flt_cd
        else:
            return flt_cd

    @update_summary
    def filter_outliers(self, data, inplace=True, **kwargs):
        """
        Apply eliptic envelope from scikit-learn to remove outliers.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
        inplace : bool
            Default true write back to CapTest.flt_sim or flt_das
        kwargs
            Passed to sklearn EllipticEnvelope.  Contamination keyword
            is useful to adjust proportion of outliers in dataset.
            Default is 0.04.
        """
        flt_cd = self.__flt_setup(data)

        XandY = flt_cd.rview(['poa', 'power'])
        X1 = XandY.values

        if 'support_fraction' not in kwargs.keys():
            kwargs['support_fraction'] = 0.9
        if 'contamination' not in kwargs.keys():
            kwargs['contamination'] = 0.04

        clf_1 = sk_cv.EllipticEnvelope(**kwargs)
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
        inplace : bool
            Default true write back to CapTest.flt_sim or flt_das
        """
        flt_cd = self.__flt_setup(data)

        for key in flt_cd.trans_keys:
            if key.find('pf') == 0:
                selection = key

        df = flt_cd.df[flt_cd.trans[selection]]
        flt_cd.df = flt_cd.df[(np.abs(df) >= pf).all(axis=1)]

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
        data : str
            'sim' or 'das' determines if filter is on sim or das data
        low : float or int
            Minimum value as fraction (0.8) or absolute 200 (W/m^2)
        high : float or int
            Max value as fraction (1.2) or absolute 800 (W/m^2)
        ref_val : float or int
            Must provide arg when min/max are fractions
        inplace : bool
            Default true write back to CapTest.flt_sim or flt_das

        Returns
        -------
        CapData object
            Filtered CapData object if inplace is False.
        """
        flt_cd = self.__flt_setup(data)

        df = flt_cd.rview('poa')
        irr_col = df.columns[0]

        df_flt = flt_irr(df, irr_col, low, high, ref_val=ref_val)
        flt_cd.df = flt_cd.df.loc[df_flt.index, :]

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

    @update_summary
    def filter_missing(self, data, **kwargs):
        """
        Remove timestamps with missing data.

        Parameters
        ----------
        data: str
            'sim' or 'das' determines if filter is on sim or das data
        """
        flt_cd = self.__flt_setup(data)
        flt_cd.df = flt_cd.df.dropna(axis=0, inplace=False, **kwargs)
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
        data : str
            'sim' or 'das' determines if filter is on sim or das data
        skip_strs : list like
            Strings to search for in column label. If found, skip column.
        perc_diff : float
            Percent difference cutoff for readings of the same measurement from
            different sensors.
        inplace : bool, default True
            If True, writes over current filtered dataframe. If False, returns
            CapData object.

        Returns
        -------
        CapData
            Returns filtered CapData if inplace is False.

        Todo
        ----
        perc_diff dict
            perc_diff can be dict of sensor type keys paired with per_diff
            values
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
    def filter_pvsyst(self, data, shade=1.0, inplace=True):
        """
        Filter pvsyst data for shading and off mppt operation.

        This function is only applicable to simulated data generated by PVsyst.
        Filters the 'IL Pmin', IL Vmin', 'IL Pmax', 'IL Vmax' values if they are
        greater than 0.


        Parameters
        ----------
        data: str, 'sim'
            This function is only intended to be run on simulated data.
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
        cd_obj = self.__flt_setup(data)

        df = cd_obj.df

        index_shd = df.query('FShdBm>=@shade').index

        columns = ['IL Pmin', 'IL Vmin', 'IL Pmax', 'IL Vmax']
        index_IL = df[df[columns].sum(axis=1) <= 0].index
        index = index_shd.intersection(index_IL)

        cd_obj.df = cd_obj.df.loc[index, :]

        if inplace:
            self.flt_sim = cd_obj
        else:
            return cd_obj

    @update_summary
    def reg_cpt(self, data, filter=False, inplace=True, summary=True):
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
        summary: bool, default True
            Set to false to not print regression summary.

        Returns
        -------
        CapData
            Returns a filtered CapData object if filter is True and inplace is
            False.
        """
        cd_obj = self.__flt_setup(data)

        df = cd_obj.rview(['power', 'poa', 't_amb', 'w_vel'])
        rename = {df.columns[0]: 'power',
                  df.columns[1]: 'poa',
                  df.columns[2]: 't_amb',
                  df.columns[3]: 'w_vel'}
        df = df.rename(columns=rename)

        reg = fit_model(df, fml=self.reg_fml)

        if filter:
            print('NOTE: Regression used to filter outlying points.\n\n')
            if summary:
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
            if summary:
                print(reg.summary())
            if data == 'das':
                self.ols_model_das = reg
            elif data == 'sim':
                self.ols_model_sim = reg

    def cp_results(self, nameplate, check_pvalues=False, pval=0.05,
                   print_res=True):
        """
        Prints a summary indicating if system passed or failed capacity test.

        NOTE: Method will try to adjust for 1000x differences in units.

        Parameters
        ----------
        nameplate : numeric
            AC nameplate rating of the PV plant.
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
        Capacity test ratio
        """
        if check_pvalues:
            for key, val in self.ols_model_das.pvalues.iteritems():
                if val > pval:
                    self.ols_model_das.params[key] = 0
            for key, val in self.ols_model_sim.pvalues.iteritems():
                if val > pval:
                    self.ols_model_sim.params[key] = 0

        actual = self.ols_model_das.predict(self.rc)[0]
        expected = self.ols_model_sim.predict(self.rc)[0]
        cap_ratio = actual / expected
        if cap_ratio < 0.01:
            cap_ratio *= 1000
            actual *= 1000
        capacity = nameplate * cap_ratio

        sign = self.tolerance.split(sep=' ')[0]
        error = int(self.tolerance.split(sep=' ')[1])

        nameplate_plus_error = nameplate * (1 + error / 100)
        nameplate_minus_error = nameplate * (1 - error / 100)

        if print_res:
            if sign == '+/-' or sign == '-/+':
                if nameplate_minus_error <= capacity <= nameplate_plus_error:
                    print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
                else:
                    print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))
                bounds = str(nameplate_minus_error) + ', ' + str(nameplate_plus_error)
            elif sign == '+':
                if nameplate <= capacity <= nameplate_plus_error:
                    print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
                else:
                    print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))
                bounds = str(nameplate) + ', ' + str(nameplate_plus_error)
            elif sign == '-':
                if nameplate_minus_error <= capacity <= nameplate:
                    print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
                else:
                    print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))
                bounds = str(nameplate_minus_error) + ', ' + str(nameplate)
            else:
                print("Sign must be '+', '-', '+/-', or '-/+'.")

            print("{:<30s}{:0.3f}".format("Modeled test output:",
                                          expected) + "\n" +
                  "{:<30s}{:0.3f}".format("Actual test output:",
                                          actual) + "\n" +
                  "{:<30s}{:0.3f}".format("Tested output ratio:",
                                          cap_ratio) + "\n" +
                  "{:<30s}{:0.3f}".format("Tested Capacity:",
                                          capacity)
                  )

            print("{:<30s}{}".format("Bounds:", bounds))

        return(cap_ratio)

    def res_summary(self, nameplate):
        """
        Prints a summary of the regression results.

        Parameters
        ----------
        nameplate : numeric
            AC nameplate rating of the PV plant.

        Prints:
        Capacity ratio without setting parameters with high p-values to zero.
        Capacity ratio after setting paramters with high p-values to zero.
        P-values for simulated and measured regression coefficients.
        Regression coefficients (parameters) for simulated and measured data.
        """

        das_pvals = self.ols_model_das.pvalues
        sim_pvals = self.ols_model_sim.pvalues
        das_params = self.ols_model_das.params
        sim_params = self.ols_model_sim.params

        df_pvals = pd.DataFrame([das_pvals, sim_pvals, das_params, sim_params])
        df_pvals = df_pvals.transpose()
        df_pvals.rename(columns={0: 'das_pvals', 1: 'sim_pvals',
                                 2: 'das_params', 3: 'sim_params'}, inplace=True)

        cprat = self.cp_results(2000, check_pvalues=False, print_res=False)
        cprat_cpval = self.cp_results(2000, check_pvalues=True, print_res=False)

        print('{} - Cap Ratio'.format(np.round(cprat, decimals=3)))
        print('{} - Cap Ratio after pval check'.format(np.round(cprat_cpval,
                                                       decimals=3)))
        return(df_pvals.style.format('{:20,.5f}').apply(highlight_pvals,
                                                 subset=['das_pvals',
                                                         'sim_pvals']))

    def uncertainty():
        """Calculates random standard uncertainty of the regression
        (SEE times the square root of the leverage of the reporting
        conditions).

        NO TESTS YET!
        """

        SEE = np.sqrt(self.ols_model_das.mse_resid)

        cd_obj = self.__flt_setup('das')
        df = cd_obj.rview(['power', 'poa', 't_amb', 'w_vel'])
        new_names = ['power', 'poa', 't_amb', 'w_vel']
        rename = {new: old for new, old in zip(df.columns, new_names)}
        df = df.rename(columns=rename)

        rc_pt = {key: val[0] for key, val in self.rc.items()}
        rc_pt['power'] = actual
        df.append([rc_pt])

        reg = fit_model(df, fml=self.reg_fml)

        infl = reg.get_influence()
        leverage = infl.hat_matrix_diag[-1]
        sy = SEE * np.sqrt(leverage)

        return(sy)

def equip_counts(df):
    """
    Returns list of integers that are a count of columns with the same name.

    Todo
    ----
    Recycle
        Determine if code might be useful somewhere.
    """
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
