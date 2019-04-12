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

pvlib_spec = importlib.util.find_spec('pvlib')
if pvlib_spec is not None:
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn('Clear sky functions will not work without the '
                  'pvlib package.')

from captest.capdata import CapData, met_keys

# EllipticEnvelope gives warning about increasing determinate that prints
# out in a loop and does not seem to affect result of removing outliers.
warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                        module='sklearn')


def highlight_pvals(s):
    """
    Highlight vals greater than or equal to 0.05 in a Series yellow.
    """
    is_greaterthan = s >= 0.05
    return ['background-color: yellow' if v else '' for v in is_greaterthan]


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
    rc : DataFrame
        Dataframe for the reporting conditions (poa, t_amb, and w_vel).
    ols_model_das : statsmodels linear regression model
        Holds the linear regression model object for the das data.
    ols_model_sim : statsmodels linear regression model
        Identical to ols_model_das for simulated data.
    reg_fml : str
        Regression formula to be fit to measured and simulated data.  Must
        follow the requirements of statsmodels use of patsy.
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
        self.reg_fml = 'power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1'

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

    def filter_clearsky(self, data, window_length=20, ghi_col=None, inplace=True,
                        **kwargs):
        """
        Uses pvlib detect_clearsky to remove periods with unstable irradiance.

        The pvlib detect_clearsky function compares modeled clear sky ghi
        against measured clear sky ghi to detect periods of clear sky.  Refer
        to the pvlib documentation for additional information.

        Parameters:
        data : str, 'sim' or 'das'
            'sim' or 'das' determines if filter is on sim or das data
        window_length : int, default 20
            Length of sliding time window in minutes. Must be greater than 2
            periods. Default of 20 works well for 5 minute data intervals.
            pvlib default of 10 minutes works well for 1min data.
        ghi_col : str, default None
            By default uses data identified by the translation dictionary as
            ghi and modeled ghi.  Issues warning if there is no modeled ghi
            data, or the measured ghi data has not been aggregated.
            Or, a column name for specific column of measured ghi data.
        inplace : bool, default True
            When true removes periods with unstable irradiance.  When false
            returns pvlib detect_clearsky results, which by default is a series
            of booleans.
        **kwargs
            kwargs are passed to pvlib detect_clearsky.  See pvlib documentation
            for details.
        """
        cd_obj = self.__flt_setup(data)

        if 'ghi_mod_csky' not in cd_obj.df.columns:
            return warnings.warn('Modeled clear sky data must be availabe to '
                                 'run this filter method. Use CapData '
                                 'load_data clear_sky option.')
        if ghi_col is None:
            ghi_keys = []
            for key in cd_obj.trans_keys:
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

            meas_ghi = cd_obj.view(meas_ghi)
            if meas_ghi.shape[1] > 1:
                warnings.warn('Averaging measured GHI data.  Pass column name to '
                              'ghi_col to use a specific column.')
            meas_ghi = meas_ghi.mean(axis=1)
        else:
            meas_ghi = cd_obj.df[ghi_col]

        clear_per = detect_clearsky(meas_ghi, cd_obj.df['ghi_mod_csky'],
                                    meas_ghi.index, window_length, **kwargs)
        if not any(clear_per):
            return warnings.warn('No clear periods detected. Try increasing the '
                                 'window length.')

        cd_obj.df = cd_obj.df[clear_per]

        if inplace:
            if data == 'das':
                self.flt_das = cd_obj
            elif data == 'sim':
                self.flt_sim = cd_obj
        else:
            return cd_obj

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
