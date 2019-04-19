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
