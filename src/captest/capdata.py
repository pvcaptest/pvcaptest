"""
Provides the CapData class and supporting functions.

The CapData class provides methods for loading, filtering, and regressing
solar data. A capacity test following the ASTM E2848 standard is orchestrated
by ``captest.CapTest``, which binds a measured and a modeled ``CapData``
instance together and exposes the cross-CapData comparison methods
(``captest_results``, ``get_summary``, ``overlay_scatters``,
``residual_plot``, ``determine_pass_or_fail``).
"""

# standard library imports
import difflib
import copy

import warnings
import importlib
import inspect

# anaconda distribution defaults
import numpy as np
import pandas as pd

# anaconda distribution defaults
# statistics and machine learning imports
from patsy import dmatrix


# anaconda distribution defaults
# visualization library imports
from bokeh.models import HoverTool, NumeralTickFormatter

import param

from captest import util
from captest import plotting
from captest.filters import (
    AbsDiffPrev,
    BaseSummaryStep,
    Backtracking,
    BooleanFlag,
    Clearsky,
    Custom,
    Days,
    Irradiance,
    Missing,
    Outliers,
    PowerFactor,
    Power,
    Pvsyst,
    Regression,
    RollingStd,
    Sensors,
    Shade,
    Time,
    RepCond,
    filter_grps,
    filter_irr,
    fit_model,
    step_from_config,
    wrap_year_end,
)

# visualization library imports
hv_spec = importlib.util.find_spec("holoviews")
if hv_spec is not None:
    import holoviews as hv
    from holoviews.plotting.links import DataLink
    from holoviews import opts

    hv.extension("bokeh")
else:
    warnings.warn(
        "Some plotting functions will not work without the holoviews package."
    )

pn_spec = importlib.util.find_spec("panel")
if pn_spec is not None:
    import panel as pn

    pn.extension()
else:
    warnings.warn(
        "The ReportingIrradiance.dashboard method will not work without "
        "the panel package."
    )

xlsx_spec = importlib.util.find_spec("openpyxl")
if xlsx_spec is None:
    warnings.warn(
        "Specifying a column grouping in an excel file will not work without "
        "the openpyxl package."
    )

plot_colors_brewer = {
    "real_pwr": ["#2b8cbe", "#7bccc4", "#bae4bc", "#f0f9e8"],
    "irr_poa": ["#e31a1c", "#fd8d3c", "#fecc5c", "#ffffb2"],
    "irr_ghi": ["#91003f", "#e7298a", "#c994c7", "#e7e1ef"],
    "temp_amb": ["#238443", "#78c679", "#c2e699", "#ffffcc"],
    "temp_mod": ["#88419d", "#8c96c6", "#b3cde3", "#edf8fb"],
    "wind": ["#238b45", "#66c2a4", "#b2e2e2", "#edf8fb"],
}

met_keys = ["poa", "t_amb", "w_vel", "power"]


columns = ["function_name", "pts_after_filter", "pts_removed", "filter_arguments"]


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
            output_vals.append(val.strftime("%Y-%m-%d %H:%M"))
        else:
            output_vals.append(val)
    return {key: val for key, val in zip(kwarg_dict.keys(), output_vals)}


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
    check_freqs = [
        "BQE-JAN",
        "BQE-FEB",
        "BQE-APR",
        "BQE-MAY",
        "BQE-JUL",
        "BQE-AUG",
        "BQE-OCT",
        "BQE-NOV",
    ]
    month_int = {
        "JAN": 1,
        "FEB": 2,
        "APR": 4,
        "MAY": 5,
        "JUL": 7,
        "AUG": 8,
        "OCT": 10,
        "NOV": 11,
    }

    if freq in check_freqs:
        warnings.warn(
            "DataFrame index adjusted to be continous through new"
            "year, but not returned or set to attribute for user."
            "This is not an issue if using RCs with"
            "predict_capacities."
        )
        if isinstance(freq, str):
            month = month_int[freq.split("-")[1]]
        else:
            month = freq.startingMonth
        year = df.index[0].year
        months_year_end = 12 - month
        months_year_start = 3 - months_year_end
        if int(month) >= 10:
            str_date = str(months_year_start) + "/" + str(year)
        else:
            str_date = str(month) + "/" + str(year)
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


def perc_bounds(percent_filter):
    """
    Convert +/- percentage to decimals to be used to determine bounds.

    Parameters
    ----------
    percent_filter : float or tuple, default None
        Percentage or tuple of percentages used to filter around the reporting
        irradiance. Required when ``irr_bal`` is True in ``rep_cond``.

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


class ReportingIrradiance(param.Parameterized):
    df = param.DataFrame(
        doc="Data to use to calculate reporting irradiance.", precedence=-1
    )
    irr_col = param.String(
        default="GlobInc",
        doc="Name of column in `df` containing irradiance data.",
        precedence=-1,
    )
    irr_rc = param.Number(precedence=-1)
    poa_flt = param.DataFrame(precedence=-1)
    total_pts = param.Number(precedence=-1)
    rc_irr_60th_perc = param.Number(precedence=-1)
    percent_band = param.Integer(20, softbounds=(2, 50), step=1)
    min_percent_below = param.Integer(
        default=40,
        doc="Minimum number of points as a percentage allowed below the \
        reporting irradiance.",
    )
    max_percent_above = param.Integer(
        default=60,
        doc="Maximum number of points as a percentage allowed above the \
        reporting irradiance.",
    )
    min_ref_irradiance = param.Integer(
        default=None, doc="Minimum value allowed for the reference irradiance."
    )
    max_ref_irradiance = param.Integer(
        None,
        doc="Maximum value allowed for the reference irradiance. By default this\
        maximum is calculated by dividing the highest irradiance value in `df`\
        by `high`.",
    )
    points_required = param.Integer(
        default=750,
        doc="This is value is only used in the plot to overlay a horizontal \
        line on the plot of the total points.",
    )

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

        poa_flt["plus_perc"] = poa_flt[self.irr_col] * high
        poa_flt["minus_perc"] = poa_flt[self.irr_col] * low

        poa_flt["below_count"] = [
            poa_flt[self.irr_col].between(low, ref).sum()
            for low, ref in zip(poa_flt["minus_perc"], poa_flt[self.irr_col])
        ]
        poa_flt["above_count"] = [
            poa_flt[self.irr_col].between(ref, high).sum()
            for ref, high in zip(poa_flt[self.irr_col], poa_flt["plus_perc"])
        ]

        poa_flt["total_pts"] = poa_flt["above_count"] + poa_flt["below_count"]
        poa_flt["perc_above"] = (poa_flt["above_count"] / poa_flt["total_pts"]) * 100
        poa_flt["perc_below"] = (poa_flt["below_count"] / poa_flt["total_pts"]) * 100

        # set index to the poa irradiance
        poa_flt.set_index(self.irr_col, inplace=True)

        if self.max_ref_irradiance is None:
            self.max_ref_irradiance = int(poa_flt.index[-1] / high)
        if self.min_ref_irradiance is None:
            self.min_ref_irradiance = int(poa_flt.index[0] / low)
        if self.min_ref_irradiance > self.max_ref_irradiance:
            warnings.warn(
                "The minimum reference irradiance ({:.2f}) is greater than the maximum "
                "reference irradiance ({:.2f}). Setting the minimum to 400 and the "
                "maximum to 1000.".format(
                    self.min_ref_irradiance, self.max_ref_irradiance
                )
            )
            self.min_ref_irradiance = 400
            self.max_ref_irradiance = 1000

        # determine ref irradiance by finding 50/50 irradiance in upper group of data
        poa_flt["valid"] = poa_flt["perc_below"].between(
            self.min_percent_below, self.max_percent_above
        ) & poa_flt.index.to_series().between(
            self.min_ref_irradiance, self.max_ref_irradiance
        )
        if poa_flt["valid"].sum() == 0:
            self.poa_flt = poa_flt
            self.irr_rc = np.nan
            warnings.warn(
                "No valid reference irradiance found. Try reviewing the min and max "
                "reference irradiance values and the min and max percent below and "
                "above values. The dashboard method will show these values with "
                "related plots and allow you to adjust them."
            )
            return None
        poa_flt["perc_below_minus_50_abs"] = (poa_flt["perc_below"] - 50).abs()
        valid_df = poa_flt[poa_flt["valid"]].copy()
        valid_df.sort_values("perc_below_minus_50_abs", inplace=True)
        # if there are more than one points that are exactly 50 points above and
        # 50 above then pick the one that results in the most points
        self.valid_df = valid_df
        fifty_fifty_points = valid_df["perc_below_minus_50_abs"] == 0
        if (fifty_fifty_points).sum() > 1:
            possible_points = poa_flt.loc[
                fifty_fifty_points[fifty_fifty_points].index, "total_pts"
            ]
            possible_points.sort_values(ascending=False, inplace=True)
            irr_RC = possible_points.index[0]
        else:
            irr_RC = valid_df.index[0]
        flt_df = filter_irr(self.df, self.irr_col, low, high, ref_val=irr_RC)
        self.irr_rc = irr_RC
        self.poa_flt = poa_flt
        total_pts_value = poa_flt.loc[self.irr_rc, "total_pts"]
        # Handle case where .loc returns a Series instead of scalar
        self.total_pts = (
            total_pts_value.iloc[0]
            if isinstance(total_pts_value, pd.Series)
            else total_pts_value
        )

        return (irr_RC, flt_df)

    def save_plot(self, output_plot_path=None):
        """
        Save a plot of the possible reporting irradiances and time intervals.

        Saves plot as an html file at path given.

        output_plot_path : str or Path
            Path to save plot to.
        """
        hv.save(self.plot(), output_plot_path, fmt="html", toolbar=True)

    def save_csv(self, output_csv_path):
        """
        Save possible reporting irradiance data to csv file at given path.
        """
        self.poa_flt.to_csv(output_csv_path)

    @param.depends(
        "percent_band",
        "min_percent_below",
        "max_percent_above",
        "min_ref_irradiance",
        "points_required",
        "max_ref_irradiance",
    )
    def plot(self):
        self.get_rep_irr()
        below_count_scatter = hv.Scatter(
            self.poa_flt["below_count"].reset_index(),
            ["poa"],
            ["below_count"],
            label="Count pts below",
        )
        above_count_scatter = hv.Scatter(
            self.poa_flt["above_count"].reset_index(),
            ["poa"],
            ["above_count"],
            label="Count pts above",
        )
        if self.irr_rc is not np.nan:
            count_ellipse = hv.Ellipse(
                self.irr_rc, self.poa_flt.loc[self.irr_rc, "below_count"], (20, 50)
            )
        perc_below_scatter = (
            hv.Scatter(
                self.poa_flt["perc_below"].reset_index(), ["poa"], ["perc_below"]
            )
            * hv.HLine(self.min_percent_below)
            * hv.HLine(self.max_percent_above)
            * hv.VLine(self.min_ref_irradiance)
            * hv.VLine(self.max_ref_irradiance)
        )
        if self.irr_rc is not np.nan:
            perc_ellipse = hv.Ellipse(
                self.irr_rc, self.poa_flt.loc[self.irr_rc, "perc_below"], (20, 10)
            )
        total_points_scatter = hv.Scatter(
            self.poa_flt["total_pts"].reset_index(), ["poa"], ["total_pts"]
        ) * hv.HLine(self.points_required)
        if self.irr_rc is not np.nan:
            total_points_ellipse = hv.Ellipse(
                self.irr_rc, self.poa_flt.loc[self.irr_rc, "total_pts"], (20, 50)
            )

        ylim_bottom = self.poa_flt["total_pts"].min() - 20
        if self.total_pts < self.points_required:
            ylim_top = self.points_required + 20
        else:
            ylim_top = self.total_pts + 50
        vl = hv.VLine(self.rc_irr_60th_perc).opts(line_color="gray")
        if self.irr_rc is not np.nan:
            rep_cond_plot = (
                (
                    (
                        below_count_scatter * above_count_scatter * count_ellipse * vl
                    ).opts(ylabel="count points")
                    + (perc_below_scatter * perc_ellipse).opts(ylim=(0, 100))
                    + (total_points_scatter * total_points_ellipse).opts(
                        ylim=(ylim_bottom, ylim_top)
                    )
                )
                .opts(
                    opts.HLine(line_width=1),
                    opts.VLine(line_width=1),
                    opts.Scatter(
                        size=4,
                        show_legend=True,
                        legend_position="right",
                        tools=["hover"],
                    ),
                    opts.Overlay(width=700),
                    opts.Layout(
                        title="Reporting Irradiance: {:0.2f}, Total Points {}".format(
                            self.irr_rc, self.total_pts
                        )
                    ),
                )
                .cols(1)
            )
        else:
            rep_cond_plot = (
                (
                    (below_count_scatter * above_count_scatter * vl).opts(
                        ylabel="count points"
                    )
                    + perc_below_scatter.opts(ylim=(0, 100))
                    + total_points_scatter.opts(ylim=(ylim_bottom, ylim_top))
                )
                .opts(
                    opts.HLine(line_width=1),
                    opts.VLine(line_width=1),
                    opts.Scatter(
                        size=4,
                        show_legend=True,
                        legend_position="right",
                        tools=["hover"],
                    ),
                    opts.Overlay(width=700),
                    opts.Layout(
                        title=(
                            "Reporting Irradiance: None identified, "
                            f"Total Points {self.total_pts}"
                        )
                    ),
                )
                .cols(1)
            )
        return rep_cond_plot

    def dashboard(self):
        return pn.Row(self.param, self.plot)


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
    pt_qty = grps.agg("count").iloc[:, 0]
    predictions.index = pt_qty.index

    params.index = pt_qty.index
    rcs.index = pt_qty.index
    predictions.name = "PredCap"

    for rc_col_name in rcs.columns:
        for param_col_name in params.columns:
            if rc_col_name == param_col_name:
                new_col_name = param_col_name + "-param"
                params.rename(columns={param_col_name: new_col_name}, inplace=True)

    results = pd.concat([rcs, predictions, params], axis=1)

    results["guaranteedCap"] = results["PredCap"] * (1 - allowance)
    results["pt_qty"] = pt_qty.values

    return results


def predict_with_pvalue_check(cd, rc=None, pval_threshold=0.05):
    """
    Make prediction with optional p-value filtering of coefficients.

    Uses model.predict() with custom params to ensure consistent behavior
    across pandas 2.x and 3.0+ (avoids Copy-on-Write issues).

    Parameters
    ----------
    cd : CapData
        Instance of CapData with:
        - regression_results attribute (fitted statsmodels results)
        - rc attribute (reporting conditions DataFrame), used if rc param is None
    rc : DataFrame, optional
        Reporting conditions DataFrame. If None, uses cd.rc.
    pval_threshold : float, default 0.05
        If provided, coefficients with p-value > threshold are set to zero
        before making the prediction. Set to None to skip pval check.

    Returns
    -------
    float
        Predicted value at reporting conditions.
    """
    results = cd.regression_results
    if rc is None:
        rc = cd.rc
    # Copy params to avoid modifying original
    modified_params = results.params.copy()
    # Zero out coefficients with p-values above threshold
    if pval_threshold is not None:
        for key, pval in results.pvalues.items():
            if pval > pval_threshold:
                modified_params[key] = 0
    # Create design matrix from reporting conditions
    design_info = results.model.data.design_info
    exog = dmatrix(design_info, rc)
    # Predict using model.predict with custom params
    return results.model.predict(modified_params, exog)[0]


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


def index_capdata(capdata, label, filtered=True):
    """
    Like Dataframe.loc but for CapData objects.

    Pass a single label or list of labels to select the columns from the `data` or
    `data_filtered` DataFrames. The label can be a column name, a column group key, or
    a regression column key.

    The special label `regcols` will return the columns identified in `regression_cols`.

    Parameters
    ----------
    capdata : CapData
        The CapData object to select from.
    label : str or list
        The label or list of labels to select from the `data` or `data_filtered`
        DataFrames. The label can be a column name, a column group key, or a
        regression column key. The special label `regcols` will return the columns
        identified in `regression_cols`.
    filtered : bool, default True
        By default the method will return columns from the `data_filtered` DataFrame.
        Set to False to return columns from the `data` DataFrame.

    Returns
    --------
    DataFrame
    """
    if filtered:
        data = capdata.data_filtered
    else:
        data = capdata.data
    if label == "regcols":
        label = list(capdata.regression_cols.values())
    if isinstance(label, str):
        if label in capdata.column_groups.keys():
            selected_data = data[capdata.column_groups[label]]
        elif label in capdata.regression_cols.keys():
            col_or_grp = capdata.regression_cols[label]
            if col_or_grp in capdata.column_groups.keys():
                selected_data = data[capdata.column_groups[col_or_grp]]
            elif col_or_grp in data.columns:
                selected_data = data[col_or_grp]
            else:
                warnings.warn(
                    'Group or column "{}" mapped to the "{}" key of regression_cols '
                    "not found in column_groups keys or columns of CapData.data".format(
                        col_or_grp, label
                    )
                )
                return None
        elif label in data.columns:
            selected_data = data.loc[:, label]
        else:
            warnings.warn(
                'Label "{}" not found in column_groups keys, regression_cols keys, '
                "or columns of CapData.data".format(label)
            )
            return None
        if isinstance(selected_data, pd.Series):
            return selected_data.to_frame()
        else:
            return selected_data
    elif isinstance(label, list):
        cols_to_return = []
        for label_item in label:
            if label_item in capdata.column_groups.keys():
                cols_to_return.extend(capdata.column_groups[label_item])
            elif label_item in capdata.regression_cols.keys():
                col_or_grp = capdata.regression_cols[label_item]
                if col_or_grp in capdata.column_groups.keys():
                    cols_to_return.extend(capdata.column_groups[col_or_grp])
                elif col_or_grp in data.columns:
                    cols_to_return.append(col_or_grp)
            elif label_item in data.columns:
                cols_to_return.append(label_item)
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


class CapData(param.Parameterized):
    """
    Class to store capacity test data and column grouping.

    CapData objects store a pandas dataframe of measured or simulated data
    and a dictionary grouping columns by type of measurement.

    The `column_groups` dictionary allows maintaining the original column names
    while also grouping measurements of the same type from different
    sensors.  Many of the methods for plotting and filtering data rely on the
    column groupings.

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
    regression_cols : dictionary
        Dictionary identifying which columns in `data` or groups of columns as
        identified by the keys of `column_groups` are the independent variables
        of the ASTM Capacity test regression equation. Set using
        `set_regression_cols` or by directly assigning a dictionary.
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

    filters = param.List(
        default=[],
        item_type=BaseSummaryStep,
        doc="Ordered pipeline of filter/summary steps applied to the data.",
    )

    @property
    def data_filtered(self):
        """Working data after the applied filter chain (derived, read-only).

        Returns a copy of ``self.data`` when no filters are set, otherwise a
        copy of ``self.data`` restricted to the rows kept by the last filter
        (``self.filters[-1].ix_after``). The result is **always** a defensive
        ``.copy()`` (both branches) so downstream mutation of the returned
        frame cannot corrupt ``self.data`` under pandas < 3.0 (no
        Copy-on-Write). There is no setter: filter results flow through
        ``filters``; to clear filtering set ``self.filters = []``.
        """
        if not self.filters:
            return self.data.copy()
        return self.data.loc[self.filters[-1].ix_after, :].copy()

    def __init__(self, name):  # noqa: D107
        super().__init__(name=name)
        self.data = pd.DataFrame()
        self.column_groups = {}
        self.regression_cols = {}
        self.rc = None
        # Back-reference to the owning CapTest (set by CapTest.setup), or None
        # for standalone use. Opaque runtime reference only — capdata.py never
        # imports captest. Intentionally NOT copied by CapData.copy().
        self._captest = None
        self.regression_results = None
        self.regression_formula = (
            "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        )
        self.tolerance = None
        self.pre_agg_cols = None
        self.pre_agg_trans = None
        self.pre_agg_reg_trans = None
        self.loc = LocIndexer(self)
        self.floc = FilteredLocIndexer(self)

    def create_column_group_attributes(self):
        """Create callable attributes for each column group that return data views.

        For each key in self.column_groups, creates an attribute on the instance
        that when called returns a view of the data for that column group using
        the loc indexer functionality.
        """
        for grp_id in self.column_groups.keys():

            def make_getter(key):
                def getter(self):
                    return self.loc[key]

                return getter

            # Create the property and set it on the instance
            setattr(self.__class__, grp_id, property(make_getter(grp_id)))

    def create_agg_attributes(self):
        """Create callable attributes for each aggregated column that return data views.

        For each column in self.column_groups['agg'], creates an attribute on the instance
        that when called returns a view of the data for that column group using
        the loc indexer functionality.
        """
        if "agg" in self.column_groups:
            for grp_id in self.column_groups["agg"]:

                def make_getter(key):
                    def getter(self):
                        return self.loc[key]

                    return getter

                # Create the property and set it on the instance
                setattr(self.__class__, "aggs_" + grp_id, property(make_getter(grp_id)))

    def set_regression_cols(self, power="", poa="", t_amb="", w_vel=""):
        """
        Create a dictionary linking the regression variables to data.

        As of v0.15.0 prefer using a predefined test setup that includes
        a regression column dictionary or assigning a dictionary to the
        `regression_cols` attribute directly.

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
        self.regression_cols = {
            "power": power,
            "poa": poa,
            "t_amb": t_amb,
            "w_vel": w_vel,
        }

    def copy(self):
        """Create and returns a copy of self."""
        cd_c = CapData(self.name)
        cd_c.data = self.data.copy()
        cd_c.column_groups = copy.deepcopy(self.column_groups)
        cd_c.regression_cols = copy.copy(self.regression_cols)
        cd_c.filters = copy.deepcopy(self.filters)
        cd_c.rc = copy.copy(self.rc)
        cd_c.regression_results = copy.deepcopy(self.regression_results)
        cd_c.regression_formula = copy.copy(self.regression_formula)
        cd_c.pre_agg_cols = copy.copy(self.pre_agg_cols)
        cd_c.pre_agg_trans = copy.deepcopy(self.pre_agg_trans)
        cd_c.pre_agg_reg_trans = copy.deepcopy(self.pre_agg_reg_trans)
        return cd_c

    def empty(self):
        """Return a boolean indicating if the CapData object contains data."""
        tests_indicating_empty = [self.data.empty, len(self.column_groups) == 0]
        return all(tests_indicating_empty)

    @property
    def rep_irr(self):
        """Reporting POA irradiance anchoring relative irradiance filters.

        Resolved when ``filter_irr`` is called with ``ref_val='rep_irr'`` (or the
        legacy ``'self_val'``). Inside a CapTest (``self._captest`` set by
        ``CapTest.setup``) it reads the single test RC ``self._captest.rc``;
        standalone it reads this instance's own ``self.rc``.

        Returns
        -------
        float
            The ``'poa'`` reporting condition of the resolved RC.

        Raises
        ------
        ValueError
            If no RC is available, or the resolved RC lacks a ``'poa'`` column.
        """
        in_test = self._captest is not None
        rc = self._captest.rc if in_test else self.rc
        if rc is None:
            if in_test:
                raise ValueError(
                    "ref_val='rep_irr' requires test reporting conditions. Call "
                    "ct.rep_cond(which) or assign ct.rc = df before filtering."
                )
            raise ValueError(
                "ref_val='rep_irr' requires reporting conditions. Call "
                "rep_cond() before filtering with ref_val='rep_irr'."
            )
        if "poa" not in rc.columns:
            raise ValueError(
                "ref_val='rep_irr' requires a 'poa' column in the reporting conditions."
            )
        return float(rc["poa"].iloc[0])

    def drop_cols(self, columns):
        """
        Drop columns from CapData `data` and `column_groups`.

        `data_filtered` reflects the change automatically since it is derived
        from `data`.

        Parameters
        ----------
        columns : str or list
            Column name or list of column names to drop.
        """
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            print(f"Removing following column: {col}")
            for key, value in self.column_groups.items():
                try:
                    value.remove(col)
                    self.column_groups[key] = value
                    print("    Dropped from column grouping")
                except ValueError:
                    continue
            self.data.drop(col, axis=1, inplace=True)
            print("    Dropped from data attribute")

    def rename_cols(self, column_map):
        """
        Rename columns in `data` and `column_groups`.

        `data_filtered` reflects the change automatically since it is derived
        from `data`.

        Parameters
        ----------
        column_map : dict
            Dictionary mapping old column names to new column names.
        """
        self.data.rename(columns=column_map, inplace=True)
        for key, value in self.column_groups.items():
            self.column_groups[key] = [column_map.get(col, col) for col in value]

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
        if filtered_data:
            df = self.floc[reg_vars].copy()
        else:
            df = self.loc[reg_vars].copy()
        rename = {df.columns[0]: reg_vars}

        if isinstance(reg_vars, list):
            for reg_var in reg_vars:
                if self.regression_cols[reg_var] in self.data_filtered.columns:
                    continue
                else:
                    columns = self.column_groups[self.regression_cols[reg_var]]
                    if len(columns) != 1:
                        return warnings.warn(
                            "Multiple columns per translation "
                            "dictionary group. Run agg_sensors "
                            "before this method."
                        )
            rename = {old: new for old, new in zip(df.columns, reg_vars)}

        df.rename(columns=rename, inplace=True)
        return df

    def review_column_groups(self):
        """Print `column_groups` with nice formatting."""
        if len(self.column_groups) == 0:
            return "column_groups attribute is empty."
        else:
            for trans_grp, col_list in self.column_groups.items():
                print(trans_grp)
                for col in col_list:
                    print("    " + col)

    # PLOTTING METHODS
    def reg_scatter_matrix(self):
        """Create pandas scatter matrix of regression variables."""
        df = self.get_reg_cols(reg_vars=["poa", "t_amb", "w_vel"])
        df["poa_poa"] = df["poa"] * df["poa"]
        df["poa_t_amb"] = df["poa"] * df["t_amb"]
        df["poa_w_vel"] = df["poa"] * df["w_vel"]
        df.drop(["t_amb", "w_vel"], axis=1, inplace=True)
        return pd.plotting.scatter_matrix(df)

    def scatter(self, filtered=True):
        """
        Create a matplotlib scatter plot of regression lhs vs. first rhs var.

        Formula-agnostic: resolves the x and y columns from
        ``self.regression_formula`` via ``util.parse_regression_formula``.

        Parameters
        ----------
        filtered : bool, default True
            Plots filtered data when True and all data when False.

        Notes
        -----
        Prefer ``CapTest.scatter_plots`` for non-default regression presets;
        it picks the right callable from ``TEST_SETUPS`` (single or multi-
        panel) automatically.
        """
        lhs, rhs = util.parse_regression_formula(self.regression_formula)
        y_col, x_col = lhs[0], rhs[0]
        if filtered:
            df = self.floc[[y_col, x_col]]
        else:
            df = self.loc[[y_col, x_col]]

        if df.shape[1] != 2:
            return warnings.warn("Aggregate sensors before using this method.")

        df = df.rename(columns={df.columns[0]: y_col, df.columns[1]: x_col})
        plt = df.plot(kind="scatter", x=x_col, y=y_col, title=self.name, alpha=0.2)
        return plt

    def scatter_hv(self, timeseries=False, all_reg_columns=False):
        """
        Create a holoviews scatter plot of regression lhs vs. first rhs var.

        Formula-agnostic thin wrapper around ``captest.captest.scatter_default``
        (with additional timeseries-overlay support, which scatter_default does
        not provide). For non-default regression presets prefer
        ``CapTest.scatter_plots`` which picks the right callable (single or
        multi-panel) from ``TEST_SETUPS``.

        Parameters
        ----------
        timeseries : bool, default False
            If True, returns a layout with the scatter plot and a linked
            timeseries plot of the lhs variable. Selecting points in the
            scatter highlights them in the timeseries.
        all_reg_columns : bool, default False
            If True, includes every regression column in the scatter plot's
            hover tooltip in addition to the x and y variables.
        """
        lhs, rhs = util.parse_regression_formula(self.regression_formula)
        y_col, x_col = lhs[0], rhs[0]
        df = self.get_reg_cols(filtered_data=True)
        df.index.name = "index"
        df = df.reset_index()
        vdims = [y_col, "index"]
        if all_reg_columns:
            vdims.extend(list(df.columns.difference(vdims + [x_col])))
        hover = HoverTool(
            tooltips=[
                ("datetime", "@index{%Y-%m-%d %H:%M}"),
                (x_col, "@{" + x_col + "}{0,0.0}"),
                (y_col, "@{" + y_col + "}{0,0.0}"),
            ],
            formatters={
                "@index": "datetime",
            },
        )
        scatter = hv.Scatter(df, x_col, vdims).opts(
            size=5,
            tools=[hover, "lasso_select", "box_select"],
            legend_position="right",
            height=400,
            width=400,
            selection_fill_color="red",
            selection_line_color="red",
            yformatter=NumeralTickFormatter(format="0,0"),
        )
        if timeseries:
            power_vs_time = hv.Scatter(df, "index", [y_col, x_col]).opts(
                tools=[hover, "lasso_select", "box_select"],
                height=400,
                width=800,
                selection_fill_color="red",
                selection_line_color="red",
            )
            y_raw_col, x_raw_col = self.loc[[y_col, x_col]].columns
            underlay = hv.Curve(
                self.data.rename_axis("index", axis="index"),
                "index",
                [y_raw_col, x_raw_col],
            ).opts(
                tools=["lasso_select", "box_select"],
                height=400,
                width=800,
                line_color="gray",
                line_width=1,
                line_alpha=0.4,
                yformatter=NumeralTickFormatter(format="0,0"),
            )
            layout_timeseries = scatter + power_vs_time * underlay
            DataLink(scatter, power_vs_time)
            return layout_timeseries.cols(1)
        return scatter

    def plot(
        self,
        combine=plotting.COMBINE,
        default_groups=plotting.DEFAULT_GROUPS,
        width=1500,
        height=250,
        plot_defaults_path=None,
        **kwargs,
    ):
        """
        Create a dashboard to explore timeseries plots of the data.

        The dashboard contains three tabs: Groups, Layout, and Overlay. The first tab,
        Groups, presents a column of plots with a separate plot overlaying the
        measurements for each group of the `column_groups`. The groups plotted are
        defined by the `default_groups` argument.

        The second tab, Layout, allows manually selecting groups to plot. The button
        on this tab can be used to replace the column of plots on the Groups tab with
        the current figure on the Layout tab. Rerun this method after clicking the
        button to see the new plots in the Groups tab.

        The third tab, Overlay, allows picking a group or any combination of individual
        tags to overlay on a single plot. The list of groups and tags can be filtered
        using regular expressions. Adding a text id in the box and clicking Update will
        add the current overlay to the list of groups on the Layout tab.

        NOTE: If a plot defaults JSON file exists in the current working directory, the
        default groups will be read from that file. The file is named
        ``plot_defaults_{self.name}.json`` to avoid conflicts when multiple CapData
        objects are used in the same session. Columns in the file that are no longer
        present in the data are ignored with a warning.

        Parameters
        ----------
        combine : dict, optional
            Dictionary of group names and regex strings to use to identify groups from
            column groups and individual tags (columns) to combine into new groups. See
            the `parse_combine` function for more details.
        default_groups : list of str, optional
            List of regex strings to use to identify default groups to plot. See the
            `plotting.find_default_groups` function for more details.
        width : int, optional
            The width of the plots on the Groups tab.
        height : int, optional
            The height of the plots on the Groups tab.
        plot_defaults_path : str or Path, optional
            Path to the plot defaults JSON file. Overrides the default naming scheme.
            When None, defaults to ``./plot_defaults_{self.name}.json``.
        **kwargs : optional
            Additional keyword arguments are passed to the options of the scatter plot.

        Returns
        -------
        Panel tabbed layout
        """
        return plotting.plot(
            self,
            combine=combine,
            default_groups=default_groups,
            group_width=width,
            group_height=height,
            plot_defaults_path=plot_defaults_path,
            **kwargs,
        )

    def scatter_filters(self):
        """Overlay of power-vs-irradiance scatters attributing removed intervals.

        One layer per filter step that removed intervals is added first, then
        the ``retained`` layer (rows surviving all filters) is appended last so
        it renders on top — together a clean partition of the data.
        Zero-removal steps (e.g. ``RepCond``) are skipped; see
        ``get_summary``/``describe_filters`` for the full step list.
        """
        data = self.get_reg_cols(reg_vars=["power", "poa"], filtered_data=False)
        data["index"] = self.data.index

        scatters = []
        retained_ix = self.filters[-1].ix_after if self.filters else self.data.index
        for _i, label, removed_ix in self._removed_by_step():
            scatters.append(
                hv.Scatter(data.loc[removed_ix, :], "poa", ["power", "index"]).relabel(
                    label
                )
            )
        scatters.append(
            hv.Scatter(data.loc[retained_ix, :], "poa", ["power", "index"]).relabel(
                "retained"
            )
        )

        scatter_overlay = hv.Overlay(scatters)
        hover = HoverTool(
            tooltips=[
                ("datetime", "@index{%Y-%m-%d %H:%M}"),
                ("poa", "@poa{0,0.0}"),
                ("power", "@power{0,0.0}"),
            ],
            formatters={
                "@index": "datetime",
            },
        )
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                width=650,
                height=500,
                muted_fill_alpha=0,
                fill_alpha=0.4,
                line_width=0,
                tools=[hover],
                yformatter=NumeralTickFormatter(format="0,0"),
            ),
            hv.opts.Overlay(legend_position="right", toolbar="above"),
        )
        return scatter_overlay

    def timeseries_filters(self):
        """Power-vs-time line with removed intervals highlighted per filter.

        A full-data power ``Curve`` backdrop plus one scatter layer per filter
        step that removed intervals, followed by a final scatter of the points
        retained after all filtering. Zero-removal steps (e.g. ``RepCond``) are
        skipped for the per-step scatter layers; see ``get_summary``/
        ``describe_filters`` for the full step list.
        """
        data = self.get_reg_cols(reg_vars="power", filtered_data=False)
        data["Timestamp"] = data.index

        plots = []
        plt_no_filtering = hv.Curve(data, ["Timestamp"], ["power"], label="all")
        plt_no_filtering.opts(
            line_color="grey",
            line_width=1,
            width=1500,
            height=450,
        )
        plots.append(plt_no_filtering)
        for _i, label, removed_ix in self._removed_by_step():
            d_flt = data.loc[removed_ix, ["power", "Timestamp"]]
            plots.append(hv.Scatter(d_flt, ["Timestamp"], ["power"], label=label))

        retained_ix = self.filters[-1].ix_after if self.filters else self.data.index
        d_retained = data.loc[retained_ix, ["power", "Timestamp"]]
        plots.append(hv.Scatter(d_retained, ["Timestamp"], ["power"], label="retained"))

        scatter_overlay = hv.Overlay(plots)
        hover = HoverTool(
            tooltips=[
                ("datetime", "@Timestamp{%Y-%m-%d %H:%M}"),
                ("power", "@power{0,0.0}"),
            ],
            formatters={
                "@Timestamp": "datetime",
            },
        )
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                muted_fill_alpha=0,
                fill_alpha=1,
                line_width=0,
                tools=[hover],
                yformatter=NumeralTickFormatter(format="0,0"),
            ),
            hv.opts.Overlay(
                legend_position="bottom",
                toolbar="right",
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
        self.filters = []

    def reset_agg(self):
        """
        Remove aggregation columns from data and data_filtered attributes.

        Does not reset filtering of data or data_filtered.
        """
        if self.pre_agg_cols is None:
            return warnings.warn("Nothing to reset; agg_sensors has not beenused.")
        else:
            self.data = self.data[self.pre_agg_cols].copy()

            self.column_groups = self.pre_agg_trans.copy()
            self.regression_cols = self.pre_agg_reg_trans.copy()

    def _get_poa_col(self):
        """
        Return poa column name from `column_groups`.

        Also, issues warning if there are more than one poa columns in
        `column_groups`.
        """
        poa_trans_key = self.regression_cols["poa"]
        if poa_trans_key in self.data.columns:
            return poa_trans_key
        else:
            poa_cols = self.column_groups[poa_trans_key]
        if len(poa_cols) > 1:
            return warnings.warn(
                "{} columns of irradiance data. "
                "Use col_name to specify a single "
                "column.".format(len(poa_cols))
            )
        else:
            return poa_cols[0]

    def _get_group(self, group_id):
        """Look up a column group by id and return the corresponding DataFrame.

        Parameters
        ----------
        group_id : str
            Key from `column_groups`, `regression_cols`, or a column name in
            `data`.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        KeyError
            If `group_id` is not found. Includes fuzzy close-match suggestions
            when available.
        """
        result = self.loc[group_id]
        if result is None:
            close_matches = difflib.get_close_matches(
                group_id, self.column_groups.keys(), n=3, cutoff=0.6
            )
            suggestion = (
                f" Did you mean one of: {', '.join(close_matches)}?"
                if close_matches
                else ""
            )
            raise KeyError(
                f"Group '{group_id}' was not found in column_groups, "
                f"regression_cols, or the columns of CapData.data.{suggestion}"
            )
        return result

    def agg_group(
        self,
        group_id,
        agg_func,
        verbose=True,
        rename_map=None,
        inplace=True,
        cutoff=10,
        columns=None,
    ):
        """
        Aggregate columns in a group.

        Parameters
        ----------
        group_id : str
            Key from `column_groups` attribute.
        agg_func : str or callable
            Aggregation function to apply.
        verbose : bool, default True
            Set to True to print the columns that have been aggregated, the
            aggregation function used, and the new column name.
        cutoff : int, default 10
            Maximum number of columns to list individually when ``verbose=True``.
            When the group contains more columns than this value, the first three
            and last three column names are printed with an ellipsis in between.
            Increase this value to see more columns listed individually.
        columns : pd.DataFrame or None, default None
            Pre-fetched DataFrame of columns to aggregate. When provided the
            lookup via ``self._get_group`` is skipped. Intended for internal use
            by ``agg_sensors`` to avoid a redundant lookup.

        Notes
        -----
        When ``agg_func`` is ``"sum"`` the aggregation is performed with
        ``min_count=1`` so that a row in which every column is ``NaN`` returns
        ``NaN`` rather than ``0.0`` (the pandas default). Rows with at least one
        value still skip ``NaN`` and sum the remaining values.
        """
        if columns is None:
            columns_to_aggregate = self._get_group(group_id)
        else:
            columns_to_aggregate = columns
        # pandas sum() defaults to min_count=0, so a row where every column is
        # NaN sums to 0.0 instead of NaN. Pass min_count=1 for sum aggregations
        # so all-NaN rows remain NaN (missing) rather than being treated as real
        # zeros. This matters for single-column power groups where summing is a
        # no-op but would otherwise convert NaN to 0.0.
        if isinstance(agg_func, str) and agg_func == "sum":
            agg_result = columns_to_aggregate.agg(agg_func, axis=1, min_count=1)
        else:
            agg_result = columns_to_aggregate.agg(agg_func, axis=1)
        col_name = util.get_agg_column_name(group_id, agg_func)
        self.column_groups.setdefault("agg", []).append(col_name)
        agg_result = agg_result.rename(col_name).to_frame()
        if verbose:
            col_name_to_print = copy.copy(col_name)
            if rename_map is not None and col_name in rename_map.keys():
                col_name_to_print = rename_map[col_name]
            cols = list(columns_to_aggregate.columns)
            col_qty = len(cols)
            if col_qty > cutoff:
                truncated_warning = "OUTPUT TRUNCATED - "
            else:
                truncated_warning = ""
            print(
                "{}Aggregating the below {} columns of the {} group using the {} function. New column name: {}:".format(
                    truncated_warning, col_qty, group_id, agg_func, col_name_to_print
                )
            )
            if col_qty <= cutoff:
                for col in cols:
                    print("    " + col)
                print("\n")
            else:
                for col in cols[:3]:
                    print("    " + col)
                print("    ...")
                for col in cols[-3:]:
                    print("    " + col)
                print("\n")
        if inplace:
            self.data[col_name] = agg_result
            return col_name
        else:
            return (agg_result, col_name)

    def expand_agg_map(self, agg_map):
        """
        Traverses, expands, and sorts the agg_map.

        If a value of `agg_map` is a dictionary, the items in that dictionary are
        added to the returned expanded agg_map at the top level. Also, the following
        steps are completed to aggregate the subgroups:
        - The `column_groups` attribute is updated to add a new group with the aggregated
        columns from the subgroups.
        - This new group is added to the expanded returned agg_map after the subgroup
        aggregations.
        - The resulting aggregation of the subgroups is renamed.

        For example, given the following `agg_map`:
        ```python
        agg_map = {
            'irr_ghi': 'mean',
            'irr_poa': {
                'irr_poa_met1': 'mean',
                'irr_poa_met2': 'mean'
            },
        }
        ```
        The returned expanded `agg_map` would be:
        ```python
        agg_map = {
            'irr_ghi': 'mean',
            'irr_poa_met1': 'mean',
            'irr_poa_met2': 'mean',
            'irr_poa_aggs': 'mean',
        }

        and the column_groups attribute would be updated to add the group:
        'irr_poa_aggs': ['irr_poa_met1_mean_agg', 'irr_poa_met2_mean_agg']

        The column resulting from aggregating the "irr_poa_aggs" group would be
        "irr_poa_aggs_mean_agg", which is renamed to "irr_poa_mean_agg".

        Parameters
        ----------
        agg_map : dict
            Dictionary specifying aggregations to be performed on
            the specified groups from the `column_groups` attribute.

        Returns
        -------
        agg_map : dict
        """
        expanded_map = {}
        rename_map = {}
        subgroup_rename_map = {}

        # First pass: expand nested dictionaries and collect subgroup info
        for key, value in agg_map.items():
            if isinstance(value, dict):
                # Add the subgroup entries to the expanded map
                for sub_key, sub_value in value.items():
                    expanded_map[sub_key] = sub_value

                # Create a new group for the aggregated subgroups
                aggs_key = f"{key}_aggs"
                agg_columns = [
                    f"{sub_key}_{sub_value}_agg" for sub_key, sub_value in value.items()
                ]
                self.column_groups[aggs_key] = agg_columns

                # Add the aggs key to the expanded map
                expanded_map[aggs_key] = value[
                    next(iter(value))
                ]  # Use same agg function as subgroups
                rename_map[f"{aggs_key}_mean_agg"] = (
                    f"{key}_{value[next(iter(value))]}_agg"
                )
                subgroup_rename_map[key] = f"{key}_{value[next(iter(value))]}_agg"
            else:
                expanded_map[key] = value

        return expanded_map, rename_map, subgroup_rename_map

    def agg_sensors(self, agg_map=None, verbose=False):
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
        verbose : bool, default False
            Set to True to print the columns that have been aggregated, the
            aggregation function used, and the new column name. If the group being
            aggregated has more than 10 columns, only the group name will be printed.

        Returns
        -------
        None
            Acts in place on the data, data_filtered, and regression_cols attributes.

        Notes
        -----
        This method is intended to be used before any filtering methods are applied.
        Filtering steps applied when this method is used will be lost.

        This method modifies the `data`, `data_filtered`, and `regression_cols` attributes.
        """
        if self.filters:
            warnings.warn(
                "The data_filtered attribute has been overwritten "
                "and previously applied filtering steps have been "
                "lost.  It is recommended to use agg_sensors "
                "before any filtering methods."
            )
        self.pre_agg_cols = self.data.columns.copy()
        self.pre_agg_trans = copy.deepcopy(self.column_groups)
        self.pre_agg_reg_trans = copy.deepcopy(self.regression_cols)

        if agg_map is None:
            agg_map = {
                self.regression_cols["power"]: "sum",
                self.regression_cols["poa"]: "mean",
                self.regression_cols["t_amb"]: "mean",
                self.regression_cols["w_vel"]: "mean",
            }

        agg_names = {}
        agg_map, rename_map, subgroup_rename_map = self.expand_agg_map(agg_map)
        for group_id, agg_func in agg_map.items():
            col_name = util.get_agg_column_name(group_id, agg_func)
            if col_name in self.data.columns:
                if verbose:
                    print(
                        "Skipping aggregation of {} as column {} already exists".format(
                            group_id, col_name
                        )
                    )
                continue
            columns = self._get_group(group_id)
            if columns.shape[1] == 1:
                continue
            agg_result, col_name = self.agg_group(
                group_id,
                agg_func,
                verbose=verbose,
                rename_map=rename_map,
                inplace=False,
                columns=columns,
            )
            self.data = pd.concat([agg_result, self.data], axis=1)
            agg_names[group_id] = col_name
        self.filters = []
        self.rename_cols(rename_map)
        self.agg_name_mapper = agg_names
        self.create_column_group_attributes()
        self.create_agg_attributes()

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
        df["a"] = ""
        df = df[["a", 0]]
        # print(df)
        df.sort_values(by=0, inplace=True, ascending=True)
        if sort_by_reversed_names:
            df["reversed"] = df[0].str[::-1]
            df.sort_values(by="reversed", inplace=True, ascending=True)
            df = df[["a", 0]]
        if self.data_loader.path.is_dir():
            df.to_excel(
                self.data_loader.path / "column_groups.xlsx", index=False, header=False
            )
        elif self.data_loader.path.is_file():
            print(self.data_loader.path.parent)
            df.to_excel(
                self.data_loader.path.parent / "column_groups.xlsx",
                index=False,
                header=False,
            )

    def filter_irr(self, low, high, ref_val=None, col_name=None, custom_name=None):
        """
        Filter on irradiance values.

        Parameters
        ----------
        low : float or int
            Minimum value as fraction (0.8) or absolute 200 (W/m^2).
        high : float or int
            Max value as fraction (1.2) or absolute 800 (W/m^2).
        ref_val : float or int or 'rep_irr'
            Must provide arg when `low` and `high` are fractions.
            Pass ``'rep_irr'`` to use the reporting irradiance from
            :attr:`rep_irr` (set by calling :meth:`rep_cond` first). Within a
            :class:`~captest.captest.CapTest`, ``'rep_irr'`` resolves against the
            single test reporting conditions ``ct.rc``, so a ``sim`` filter can
            anchor on the test's reporting irradiance without passing the value
            manually.
        col_name : str, default None
            Column name of irradiance data to filter.  By default uses the POA
            irradiance set in regression_cols attribute or average of the POA
            columns.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Irradiance(
            low=low,
            high=high,
            ref_val=ref_val,
            col_name=col_name,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_rolling_std(self, window, threshold, column=None, custom_name=None):
        """Remove intervals where a column's rolling std is at or above a threshold.

        Parameters
        ----------
        window : int or str
            Rolling window passed to ``DataFrame.rolling`` — an int row count or
            a pandas offset alias such as ``'10min'``.
        threshold : float
            Intervals whose rolling standard deviation is at or above this value
            are removed.
        column : str, default None
            Column to evaluate. Defaults to the POA column from
            ``regression_cols``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = RollingStd(
            window=window,
            threshold=threshold,
            column=column,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_abs_diff_prev(self, threshold=0.05, column=None, custom_name=None):
        """Remove intervals with a large fractional change from the prior interval.

        Parameters
        ----------
        threshold : float, default 0.05
            Maximum allowed absolute fractional change
            (``abs(col.diff() / col)``) from the previous interval. Intervals
            above this are removed.
        column : str, default None
            Column to evaluate. Defaults to the POA column from
            ``regression_cols``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = AbsDiffPrev(threshold=threshold, column=column, custom_name=custom_name)
        flt.run(self)

    def filter_flag(self, column, invert=False, custom_name=None):
        """Remove intervals where a boolean/flag column is truthy.

        Parameters
        ----------
        column : str
            Boolean/flag column. Rows where this is truthy are removed.
        invert : bool, default False
            If True, remove rows where the column is falsy instead — keeping
            only the truthy rows.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = BooleanFlag(column=column, invert=invert, custom_name=custom_name)
        flt.run(self)

    def filter_threshold(self, column, low=None, high=None, custom_name=None):
        """Keep intervals where ``column`` is within ``[low, high]``.

        Either bound may be None for a one-sided filter: pass only ``low`` to
        keep rows at or above it, or only ``high`` to keep rows at or below it.
        Bounds are inclusive (``>=`` / ``<=``) — unlike the former
        ``filter_avail`` helper this replaces, which used a strict ``>``.
        Backed by the ``Irradiance`` filter, so the recorded step serializes
        and replays as an ``Irradiance`` step.

        Parameters
        ----------
        column : str
            Column to threshold.
        low : float, default None
            Lower bound (inclusive). None means unbounded below.
        high : float, default None
            Upper bound (inclusive). None means unbounded above.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Irradiance(
            low=low,
            high=high,
            col_name=column,
            units=None,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_pvsyst(self, custom_name=None):
        """Remove PVsyst intervals operating off the maximum power point.

        Drops rows where any IL Pmin/Vmin/Pmax/Vmax column is > 0. Only
        applicable to simulated data generated by PVsyst.

        Parameters
        ----------
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Pvsyst(custom_name=custom_name)
        flt.run(self)

    def filter_shade(self, fshdbm=1.0, query_str=None, custom_name=None):
        """Remove intervals of array shading.

        By default removes rows where the PVsyst ``FShdBm`` column is below
        ``fshdbm``. Pass ``query_str`` to filter via ``DataFrame.query``
        instead (e.g. ``"ShdLoss<=50"`` when only a shading-loss column is
        available; the column name must not contain spaces).

        Parameters
        ----------
        fshdbm : float, default 1.0
            Shading-fraction threshold; rows with FShdBm below this are removed.
        query_str : str, default None
            Optional DataFrame.query expression overriding the FShdBm test.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Shade(fshdbm=fshdbm, query_str=query_str, custom_name=custom_name)
        flt.run(self)

    def filter_time(
        self,
        start=None,
        end=None,
        drop=False,
        days=None,
        test_date=None,
        custom_name=None,
    ):
        """
        Select data for a specified time period.

        Parameters
        ----------
        start : str or pd.Timestamp or None, default None
            Start date for data to be returned.
        end : str or pd.Timestamp or None, default None
            End date for data to be returned.
        drop : bool, default False
            With start+end, remove the window instead of keeping it.
        days : int or None, default None
            Days in the time window.
        test_date : str or pd.Timestamp or None, default None
            Center of a symmetric ``days``-wide window.
        custom_name : str, default None
            Optional display label for the recorded filter step.

        Notes
        -----
        The ``wrap_year`` parameter from the previous implementation has been
        removed. Year-end-spanning data is now handled by an auto-wrap step at
        CapTest load time (see CapTest.auto_wrap_sim).
        """
        flt = Time(
            start=start,
            end=end,
            drop=drop,
            days=days,
            test_date=test_date,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_days(self, days, drop=False, custom_name=None):
        """Keep or drop the timestamps belonging to a list of days.

        Parameters
        ----------
        days : list
            Days to select or drop (date strings or Timestamps).
        drop : bool, default False
            Drop the listed days instead of keeping only them.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Days(days=days, drop=drop, custom_name=custom_name)
        flt.run(self)

    def filter_outliers(self, custom_name=None, **kwargs):
        """
        Apply EllipticEnvelope from scikit-learn to remove outliers in (poa, power).

        Parameters
        ----------
        custom_name : str, default None
            Optional display label for the recorded filter step.
        **kwargs
            Forwarded to ``sklearn.covariance.EllipticEnvelope``. Defaults
            ``support_fraction=0.9`` and ``contamination=0.04`` are applied
            when not overridden.

        Notes
        -----
        When NaN values are present in poa/power, ``filter_missing`` is
        invoked first (and recorded as a separate filter step). This
        preserves the legacy auto-handling behavior.
        """
        flt = Outliers(envelope_kwargs=kwargs or None, custom_name=custom_name)
        flt.run(self)

    def filter_pf(self, pf, custom_name=None):
        """Remove intervals with a power factor below ``pf``.

        Keeps rows where every column in the power-factor group (the first
        ``column_groups`` key beginning with ``pf``) has an absolute value at
        or above ``pf``.

        Parameters
        ----------
        pf : float
            Power-factor threshold (e.g. 0.999). Rows with any |pf| below this
            are removed.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = PowerFactor(pf=pf, custom_name=custom_name)
        flt.run(self)

    def filter_power(self, power, percent=None, columns=None, custom_name=None):
        """Remove intervals at or above a power threshold.

        Parameters
        ----------
        power : numeric
            Threshold, or nameplate power when ``percent`` is given.
        percent : numeric, default None
            If set, threshold is ``power * (1 - percent)`` (decimal).
        columns : str, default None
            Column or column-group to filter on. None uses the regression
            power column.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Power(
            power=power, percent=percent, columns=columns, custom_name=custom_name
        )
        flt.run(self)

    def filter_custom(self, func, *args, custom_name=None, **kwargs):
        """Apply ``func`` to ``data_filtered`` as a row filter and record the step.

        ``func`` is called as ``func(self.data_filtered, *args, **kwargs)`` and
        must return a DataFrame whose index is the rows to keep. Many pandas
        DataFrame methods qualify, e.g. ``pd.DataFrame.between_time`` or
        ``pd.DataFrame.dropna``.

        Parameters
        ----------
        func : callable
            Takes a DataFrame as the first argument and returns a DataFrame.
        *args, **kwargs
            Forwarded to ``func``.
        custom_name : str, default None
            Optional display label for the recorded filter step. Keyword-only
            so it cannot collide with positional args destined for ``func``.

        Notes
        -----
        The class-based pipeline preserves the original column set: only the
        returned frame's *index* is consumed. A function that drops or
        transforms columns will see its column changes discarded — pass
        column-transforming logic outside the filter pipeline.
        """
        Custom(func, *args, custom_name=custom_name, **kwargs).run(self)

    def filter_sensors(self, thresholds=None, method="percent_diff", custom_name=None):
        """Drop suspicious measurements by comparing readings across sensors.

        For each sensor group in ``thresholds``, the ``method`` comparison is
        applied row-by-row across that group's columns; rows whose sensors
        disagree beyond the threshold are removed. Ignores columns generated
        by ``agg_sensors``.

        Parameters
        ----------
        thresholds : dict, default None
            Map of sensor-group key -> threshold, e.g.
            ``{'irr-poa-': 0.05}``. Values are decimal fractions for
            ``'percent_diff'`` or absolute units for ``'abs_diff'``. None
            defaults to the POA group at a 5% percent difference (only valid
            for ``method='percent_diff'``).
        method : str or callable, default 'percent_diff'
            ``'percent_diff'`` (pairwise percent difference) or ``'abs_diff'``
            (absolute difference from the group average). A callable with
            signature ``func(series, threshold) -> bool`` may be passed for a
            custom comparison.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Sensors(method=method, thresholds=thresholds, custom_name=custom_name)
        flt.run(self)

    def filter_sensors_abs_diff(self, thresholds, custom_name=None):
        """Drop rows where a sensor deviates from its group average by too much.

        Convenience wrapper for :meth:`filter_sensors` with
        ``method='abs_diff'``: each sensor must be within ``threshold``
        (absolute units, e.g. W/m^2) of the average of the other sensors in
        its group.

        Parameters
        ----------
        thresholds : dict
            Map of sensor-group key -> absolute threshold, e.g.
            ``{'irr-poa-': 25}``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Sensors(method="abs_diff", thresholds=thresholds, custom_name=custom_name)
        flt.run(self)

    def filter_clearsky(
        self, ghi_col=None, keep_clear=True, custom_name=None, **kwargs
    ):
        """Remove unstable-irradiance intervals using pvlib detect_clearsky.

        Parameters
        ----------
        ghi_col : str, default None
            Measured GHI column. Auto-detected from ``column_groups`` if None.
        keep_clear : bool, default True
            Keep clear intervals (True) or keep cloudy intervals (False).
        custom_name : str, default None
            Optional display label for the recorded filter step.
        **kwargs
            Forwarded to pvlib ``detect_clearsky``. Default
            ``infer_limits=True`` is applied when not overridden.
        """
        flt = Clearsky(
            ghi_col=ghi_col,
            keep_clear=keep_clear,
            detect_kwargs=kwargs or None,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_backtracking(
        self,
        axis_tilt=None,
        axis_azimuth=None,
        gcr=None,
        cross_axis_tilt=None,
        keep_backtracking=False,
        custom_name=None,
    ):
        """Remove intervals where single-axis-tracker backtracking is active.

        Backtracking activity is decided per interval from solar geometry using
        a transcription of pvlib's ``tracking.singleaxis`` test. Solar position
        is computed from the site location; tracker geometry defaults to the
        site system definition (``site['sys']``) and may be overridden per
        argument. Degrades to a warn-and-no-op when site/geometry/pvlib are
        unavailable, the resolved geometry is invalid, or solar position cannot
        be computed (e.g. malformed ``site['loc']``).

        Parameters
        ----------
        axis_tilt : float, default None
            Tracker axis tilt (deg). Uses site['sys']['axis_tilt'] when None.
        axis_azimuth : float, default None
            Tracker axis azimuth (deg). Uses site['sys']['axis_azimuth'] when
            None.
        gcr : float, default None
            Ground coverage ratio. Uses site['sys']['gcr'] when None.
        cross_axis_tilt : float, default None
            Cross-axis tilt (deg) for sloped terrain. Uses
            site['sys'].get('cross_axis_tilt', 0) when None. Must be in
            (-90, 90).
        keep_backtracking : bool, default False
            If True, keep only backtracking intervals and remove true-tracking
            ones.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Backtracking(
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            gcr=gcr,
            cross_axis_tilt=cross_axis_tilt,
            keep_backtracking=keep_backtracking,
            custom_name=custom_name,
        )
        flt.run(self)

    def filter_missing(self, columns=None, custom_name=None):
        """Remove rows with missing data (NaN) in the regression columns.

        Parameters
        ----------
        columns : list, default None
            Subset of columns to check for NaN. By default uses the regression
            columns identified in ``regression_cols``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        Missing(columns=columns, custom_name=custom_name).run(self)

    def filter_op_state(self, op_state, mult_inv=None):
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

    def _ix_before(self, i):
        """Index passed *into* ``self.filters[i]`` (chain state just before it).

        The prior step's ``ix_after``, or ``self.data.index`` for the first step.
        """
        return self.filters[i - 1].ix_after if i > 0 else self.data.index

    def _pts_before(self, i):
        """Row count passed into ``self.filters[i]`` (see ``_ix_before``)."""
        return len(self._ix_before(i))

    def _step_labels(self):
        """Per-step display labels for the summary and visualization methods.

        Each label is the step's ``custom_name`` if set, otherwise its class
        name, with a ``-N`` suffix disambiguating repeated steps (the first
        occurrence is unsuffixed). Single source of the enumerated labels shared
        by ``get_summary`` and (in chunk 6) ``scatter_filters``/
        ``timeseries_filters``.
        """
        labels, seen = [], {}
        for step in self.filters:
            base = step.custom_name or type(step).__name__
            n = seen.get(base, 0)
            seen[base] = n + 1
            labels.append(base if n == 0 else f"{base}-{n}")
        return labels

    def _removed_by_step(self):
        """Per-step removal attribution for the visualization methods.

        Returns a list of ``(i, label, removed_ix)`` for each filter step that
        removed at least one interval, where ``removed_ix`` is
        ``_ix_before(i)`` minus ``filters[i].ix_after`` and ``label`` is the
        step's ``_step_labels()`` entry. Zero-removal steps (always ``RepCond``;
        also any filter that matched everything) are skipped — they have nothing
        to attribute. ``i`` is the step's real index in ``self.filters`` so
        callers can recover its input set via ``_ix_before(i)`` and its
        survivors via ``self.filters[i].ix_after``.
        """
        out = []
        for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
            removed_ix = self._ix_before(i).difference(step.ix_after)
            if len(removed_ix) > 0:
                out.append((i, label, removed_ix))
        return out

    def get_summary(self):
        """Return a DataFrame summarizing the applied filter chain.

        Rebuilt from ``self.filters``: one row per step, with the step's class
        name (``function_name``), the rows remaining after it
        (``pts_after_filter``), the rows it removed (``pts_removed``, derived
        from the prior step's ``ix_after`` via ``_pts_before``), and its
        rendered arguments (``filter_arguments``). The row index is a MultiIndex
        of ``(self.name, label)`` where ``label`` comes from ``_step_labels``.

        Returns an empty DataFrame (standard columns, no rows) when no filters
        have been applied.

        Returns
        -------
        pandas.DataFrame
        """
        if not self.filters:
            return pd.DataFrame(columns=columns)
        rows = []
        index = []
        for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
            index.append((self.name, label))
            pts_before = self._pts_before(i)
            rows.append(
                {
                    "function_name": type(step).__name__,
                    "pts_after_filter": step.pts_after,
                    "pts_removed": pts_before - step.pts_after,
                    "filter_arguments": step.args_repr,
                }
            )
        return pd.DataFrame(
            rows, index=pd.MultiIndex.from_tuples(index), columns=columns
        )

    def describe_filters(self):
        """Return a written, human-readable summary of the filtering run.

        Joins the ``explanation`` of each filter step in ``self.filters``, one
        per line. Steps without an explanation template are skipped; use
        ``get_summary()`` for the complete tabular history.
        """
        lines = [
            step.explanation for step in self.filters if step.explanation is not None
        ]
        return "\n".join(lines)

    def filters_to_config(self):
        """Serialize the applied filter chain to a list of config dicts.

        Each entry is a step's ``to_config()`` (a yaml-safe ``{type, ...params}``
        dict). Inverse of :meth:`run_pipeline`. Used by ``CapTest.to_yaml`` to
        embed this CapData's pipeline in the single config file.
        """
        return [step.to_config() for step in self.filters]

    def run_pipeline(self, config):
        """Rebuild and run each filter step from a list of config dicts.

        ``config`` is a list of ``to_config()`` dicts (e.g. from
        :meth:`filters_to_config` or a loaded YAML). Each step is constructed
        via ``filters.step_from_config`` and run against this CapData in order.
        Requires ``data`` loaded and ``regression_cols`` resolved (run
        ``process_regression_columns`` first) for filters that need them.
        """
        for step_config in config:
            step_from_config(step_config).run(self)

    def _calc_rep_cond(
        self, func, w_vel, irr_bal, percent_filter, front_poa, rc_kwargs
    ):
        """Compute reporting conditions and store them on ``self.rc``.

        Extracted verbatim from the former ``rep_cond`` body so the
        reporting-conditions math lives in one place. Called by
        ``filters.RepCond._execute`` (through the runtime ``capdata`` argument,
        so ``filters.py`` needs no import of ``capdata`` or
        ``ReportingIrradiance``) and by the thin ``rep_cond`` wrapper. Sets
        ``self.rc`` (and ``self.rc_tool`` when ``irr_bal`` is True) as a side
        effect; returns None.

        Parameters
        ----------
        func : dict, str, callable, or None
            See ``rep_cond``. When None, defaults to the mean of each
            right-hand-side variable.
        w_vel : numeric or None
            See ``rep_cond``.
        irr_bal : bool
            See ``rep_cond``.
        percent_filter : int
            See ``rep_cond``.
        front_poa : str
            See ``rep_cond``.
        rc_kwargs : dict or None
            See ``rep_cond``. None is treated as an empty dict.
        """
        if rc_kwargs is None:
            rc_kwargs = {}
        lhs, rhs = util.parse_regression_formula(self.regression_formula)
        df = self.get_reg_cols(reg_vars=rhs, filtered_data=True)

        if func is None:
            func = {var: "mean" for var in rhs}

        RCs_df = pd.DataFrame(df.agg(func)).T

        if irr_bal:
            if front_poa not in df.columns:
                raise ValueError(
                    f"front_poa={front_poa!r} is not a right-hand-side variable "
                    f"of the regression formula."
                )
            self.rc_tool = ReportingIrradiance(
                df,
                front_poa,
                percent_band=percent_filter,
                **rc_kwargs,
            )
            results = self.rc_tool.get_rep_irr()
            flt_df = results[1]
            RCs_df = pd.DataFrame(flt_df.agg(func)).T
            RCs_df.loc[RCs_df.index[0], front_poa] = results[0]

        if w_vel is not None and "w_vel" in RCs_df.columns:
            RCs_df.loc[RCs_df.index[0], "w_vel"] = w_vel

        print("Reporting conditions saved to rc attribute.")
        print(RCs_df)
        self.rc = RCs_df
        # When this CapData belongs to a CapTest, propagate the freshly computed
        # rc to the single test rc (last-writer-wins). The CapTest decides warn
        # vs silent and config-seeded load behavior. Opaque call — capdata.py
        # never imports captest.
        if self._captest is not None:
            self._captest._on_capdata_rep_cond(self)

    def rep_cond(
        self,
        func=None,
        w_vel=None,
        irr_bal=False,
        percent_filter=20,
        front_poa="poa",
        rc_kwargs=None,
        custom_name=None,
    ):
        """
        Calculate reporting conditions for the current regression formula.

        The calculation is formula-agnostic: the right-hand-side variables of
        ``self.regression_formula`` drive which columns are aggregated.

        The test setups defined in captest.TEST_SETUPS define values for the arguments
        of this method. For example, the ``e2848_default`` setup defines reporting
        conditions per ASTM E2939 - the POA irradiance value that exceeds 60 % of the
        filtered irradiance data, the mean ambient temperature, and the mean wind speed.

        Use ``rep_cond_freq`` for seasonal/monthly outputs.

        Parameters
        ----------
        func : dict, str, callable, or None, default None
            When None, defaults to calculating the mean for each term on right hand
            side of the regression formula. Passed to ``df.agg(...)``. A dict maps rhs
            variable names to aggregation functions (e.g.
            ``{'poa': perc_wrap(60), 't_amb': 'mean'}``).
        w_vel : numeric or None, default None
            If not None, overrides the calculated wind speed reporting
            condition with this value.
        irr_bal : bool, default False
            If True, uses `ReportingIrradiance` to determine the reporting
            irradiance (``front_poa``). When True, the other reporting
            conditions are aggregated from the subset of data within the
            balanced irradiance band.
        percent_filter : int, default 20
            Percentage used to define the irradiance band around the reporting
            irradiance when ``irr_bal`` is True. Has no effect when
            ``irr_bal`` is False.
        front_poa : str, default 'poa'
            Key in ``self.regression_cols`` whose column is used as the
            irradiance driver when ``irr_bal`` is True.
        rc_kwargs : dict or None, default None
            Passed to ``ReportingIrradiance`` when ``irr_bal`` is True.
        custom_name : str, default None
            Optional display label for the recorded filter step.

        Returns
        -------
        None
            Reporting conditions are stored on ``self.rc`` as a one-row
            DataFrame.
        """
        RepCond(
            func=func,
            w_vel=w_vel,
            irr_bal=irr_bal,
            percent_filter=percent_filter,
            front_poa=front_poa,
            rc_kwargs=rc_kwargs,
            custom_name=custom_name,
        ).run(self)

    def rep_cond_freq(
        self,
        irr_bal=False,
        percent_filter=20,
        front_poa="poa",
        w_vel=None,
        inplace=True,
        func=None,
        freq=None,
        grouper_kwargs={},
        rc_kwargs={},
    ):
        """
        Calculate frequency-grouped reporting conditions.

        Like ``rep_cond`` but aggregates within groups defined by ``freq``
        (e.g. ``'MS'`` for month-start, ``'60D'`` for 60-day). Used for
        seasonal or monthly reporting tests.

        Parameters
        ----------
        irr_bal : bool, default False
            See ``rep_cond``.
        percent_filter : int, default 20
            See ``rep_cond``.
        front_poa : str, default 'poa'
            See ``rep_cond``.
        w_vel : numeric or None
            See ``rep_cond``.
        inplace : bool, default True
            When True writes the multi-row RC DataFrame to ``self.rc``; when
            False returns the DataFrame.
        func : dict, str, callable, or None, default None
            See ``rep_cond``.
        freq : str or None
            Pandas offset alias. ``None`` falls back to single-row ``rep_cond``
            behavior.
        grouper_kwargs : dict
            Passed to ``pandas.Grouper``.
        rc_kwargs : dict
            Passed to ``ReportingIrradiance`` when ``irr_bal`` is True.

        Returns
        -------
        DataFrame or None
            Multi-row DataFrame of per-group reporting conditions when
            ``inplace=False``. Otherwise stores on ``self.rc`` and returns
            ``None``.
        """
        lhs, rhs = util.parse_regression_formula(self.regression_formula)
        df = self.get_reg_cols(reg_vars=rhs)

        if func is None:
            func = {var: "mean" for var in rhs}

        if freq is None:
            # Degenerate case: act like rep_cond.
            RCs_df = pd.DataFrame(df.agg(func)).T
            if irr_bal:
                if front_poa not in df.columns:
                    raise ValueError(
                        f"front_poa={front_poa!r} is not a right-hand-side variable "
                        f"of the regression formula."
                    )
                self.rc_tool = ReportingIrradiance(
                    df, front_poa, percent_band=percent_filter, **rc_kwargs
                )
                results = self.rc_tool.get_rep_irr()
                flt_df = results[1]
                RCs_df = pd.DataFrame(flt_df.agg(func)).T
                RCs_df.loc[RCs_df.index[0], front_poa] = results[0]
            if w_vel is not None and "w_vel" in RCs_df.columns:
                RCs_df.loc[RCs_df.index[0], "w_vel"] = w_vel
        else:
            # wrap_seasons passes df through unchanged unless freq is one of
            # 'BQE-JAN', 'BQE-FEB', 'BQE-APR', 'BQE-MAY', 'BQE-JUL',
            # 'BQE-AUG', 'BQE-OCT', 'BQE-NOV'
            df = wrap_seasons(df, freq)
            df_grpd = df.groupby(pd.Grouper(freq=freq, **grouper_kwargs))

            if irr_bal:
                if front_poa not in df.columns:
                    raise ValueError(
                        f"front_poa={front_poa!r} is not a right-hand-side variable "
                        f"of the regression formula."
                    )
                ix = pd.DatetimeIndex(list(df_grpd.groups.keys()), freq=freq)
                monthly_rcs = []
                for grp_key, month in df_grpd:
                    self.rc_tool = ReportingIrradiance(
                        month,
                        front_poa,
                        percent_band=percent_filter,
                        **rc_kwargs,
                    )
                    results = self.rc_tool.get_rep_irr()
                    flt_df = results[1]
                    monthly_rc = pd.DataFrame(flt_df.agg(func)).T
                    monthly_rc.index = [grp_key]
                    monthly_rc.loc[grp_key, front_poa] = results[0]
                    monthly_rcs.append(monthly_rc)
                RCs_df = pd.concat(monthly_rcs)
                RCs_df.index = ix
            else:
                RCs_df = df_grpd.agg(func)

            if w_vel is not None and "w_vel" in RCs_df.columns:
                RCs_df["w_vel"] = w_vel

        if inplace:
            print("Reporting conditions saved to rc attribute.")
            print(RCs_df)
            self.rc = RCs_df
            return None
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
            Percentage or tuple of percentages used to filter each time-period
            group of data around the group's reporting irradiance.
            Tuple option allows specifying different percentage for below and
            above the reporting irradiance: (below, above).
        **kwargs
            NOTE: Should match kwargs used to calculate reporting conditions.
            Passed to filter_grps which passes on to pandas Grouper to control
            label and closed side of intervals.
            See pandas Grouper doucmentation for details. Default is left
            labeled and left closed.
        """
        df = self.floc[["poa", "t_amb", "w_vel", "power"]]
        df = df.rename(
            columns={
                df.columns[0]: "poa",
                df.columns[1]: "t_amb",
                df.columns[2]: "w_vel",
                df.columns[3]: "power",
            }
        )

        if self.rc is None:
            return warnings.warn(
                "Reporting condition attribute is None.\
                                 Use rep_cond to generate RCs."
            )

        low, high = perc_bounds(percent_filter)
        freq = self.rc.index.freq
        df = wrap_seasons(df, freq)
        grps = df.groupby(by=pd.Grouper(freq=freq, **kwargs))

        if irr_filter:
            grps = filter_grps(grps, self.rc, "poa", low, high, freq)

        error = float(self.tolerance.split(sep=" ")[1]) / 100
        results = pred_summary(grps, self.rc, error, fml=self.regression_formula)

        return results

    def fit_regression(self, filter=False, summary=True, custom_name=None):
        """
        Perform a regression with statsmodels on the filtered data.

        Parameters
        ----------
        filter : bool, default False
            When True, removes timestamps whose residuals exceed two standard
            deviations (recorded as a Regression step). ``regression_results``
            is not updated in this case; call ``fit_regression(filter=False)``
            afterwards to store the final fit. When False, just fits ordinary
            least squares and stores the result in ``regression_results``.
        summary : bool, default True
            Set False to suppress printing the regression summary.
        custom_name : str, default None
            Optional display label for the recorded filter step. Only has an
            effect when ``filter=True``; the ``filter=False`` path records no
            step, so the label is ignored.
        """
        if filter:
            print("NOTE: Regression used to filter outlying points.\n\n")
            flt = Regression(n_std=2, custom_name=custom_name)
            flt.run(self)
            if summary:
                print(flt.regression_model.summary())
        else:
            df = self.get_reg_cols()
            reg = fit_model(df, fml=self.regression_formula)
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
            df = self.floc[group]
            # prevent aggregation from updating column groups?
            # would not need the below line then
            df = df[[col for col in df.columns if "agg" not in col]]
            qty_sensors = df.shape[1]
            s_spatial = df.std(axis=1)
            b_spatial_j = s_spatial / (qty_sensors ** (1 / 2))
            b_spatial = ((b_spatial_j**2).sum() / b_spatial_j.shape[0]) ** (1 / 2)
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
            by_group = (inst_uncert**2 + self.spatial_uncerts[group] ** 2) ** (1 / 2)
            rcs = self.rc.copy()
            rcs.loc[0, grp_to_term[group]] = rcs.loc[0, grp_to_term[group]] + by_group
            pred_cap_uncert = self.regression_results.get_prediction(
                rcs
            ).predicted_mean[0]
            perc_diffs[group] = (pred_cap_uncert - pred_cap) / pred_cap
        df = pd.DataFrame(perc_diffs.values())
        by = (df**2).sum().values[0] ** (1 / 2)
        sy = pred.se_obs[0] / pred_cap
        return (by**2 + sy**2) ** (1 / 2) * k

    def get_filtering_table(self):
        """
        Returns DataFrame showing which filter removed each filtered time interval.

        One column per filter step that removed intervals, in run order. Within a
        column: ``1`` marks the intervals that step removed, ``0`` the intervals
        present going into that step and kept by it, and ``NaN`` intervals already
        removed by an earlier step. The final ``all_filters`` column is True for
        intervals not removed by any filter. Zero-removal steps (e.g. ``RepCond``)
        get no column, consistent with the scatter/timeseries views.
        """
        filtering_data = pd.DataFrame(index=self.data.index)
        for i, label, removed_ix in self._removed_by_step():
            filtering_data.loc[self.filters[i].ix_after, label] = 0
            filtering_data.loc[removed_ix, label] = 1
        filtering_data["all_filters"] = filtering_data.apply(
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
        print("length of test period to date: {} days".format(self.length_test_period))
        if self.test_complete:
            print(
                "sufficient points have been collected. {} points required; "
                "{} points collected".format(self.pts_required, pts_collected)
            )
        else:
            print(
                "{} points of {} points needed, {} remaining to collect.".format(
                    pts_collected, self.pts_required, self.pts_required - pts_collected
                )
            )
            print("{:0.2f} points / day on average.".format(avg_pts_per_day))
            print(
                "Approximate days remaining: {:0.0f}".format(
                    round(((self.pts_required - pts_collected) / avg_pts_per_day), 0)
                    + 1
                )
            )

    def get_length_test_period(self):
        """
        Get length of test period.

        Uses length of `data` unless a `Time` step has been run, then uses
        the length of the kept data after `Time` was run the first time.
        Subsequent uses of `Time` are ignored.

        Rounds up to a period of full days.

        Returns
        -------
        int
            Days in test period.
        """
        test_period = self.data.index[-1] - self.data.index[0]
        for step in self.filters:
            if isinstance(step, Time):
                test_period = step.ix_after[-1] - step.ix_after[0]
                break
        self.length_test_period = test_period.ceil("D").days

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
        self.pts_required = (hrs_req * 60) / util.get_common_timestep(
            self.data, units="m", string_output=False
        )

    def set_test_complete(self, pts_required):
        """Sets `test_complete` attribute.

        Parameters
        ----------
        pts_required : int
            Number of points required to remain after filtering for a complete test.
        """
        self.test_complete = self.data_filtered.shape[0] >= pts_required

    def column_groups_to_excel(self, save_to="./column_groups.xlsx"):
        """Export the column groups attribute to an excel file.

        Parameters
        ----------
        save_to : str
            File path to save column groups to. Should include .xlsx.
        """
        pd.DataFrame.from_dict(
            self.column_groups.data, orient="index"
        ).stack().to_frame().droplevel(1).to_excel(save_to, header=False)

    def process_regression_columns(self, verbose=True):
        """
        Walk the regression column dictionary and calculate parameters.

        See util.process_reg_cols for additional documentation.

        Parameters
        ----------
        verbose : bool, default True
            By default prints summary of aggregations and parameter calculations
            performed while traversing the `regression_cols` dictionary.
            Set to False to prevent all output.
        """
        if self.filters:
            warnings.warn(
                "The data_filtered attribute has been overwritten "
                "and previously applied filtering steps have been "
                "lost.  It is recommended to use agg_sensors "
                "before any filtering methods."
            )
        self.regression_cols_preprocess = copy.deepcopy(self.regression_cols)
        util.process_reg_cols(self.regression_cols, cd=self, verbose=verbose)
        self.filters = []
        self.create_column_group_attributes()
        if "agg" in self.column_groups:
            self.create_agg_attributes()

    def custom_param(self, func, *args, **kwargs):
        """Applies the function `func` with kwargs and adds result as new column to `data`.

        Calculates and adds a new column to `data` using the function `func` with the
        provided arguments and keyword arguments. See the functions in the calcparams
        module for examples.

        Called by `util.process_reg_cols` to add new columns to the `data` attribute
        while recursively processing and updating the `regression_cols` attribute.

        Parameters
        ----------
        func : function
            Function that takes a DataFrame as its first argument and returns a Series.

        Returns
        -------
        None
            Adds a new column to the `data` attribute.
        """
        result = func.__name__
        signature = inspect.signature(func)
        for key in signature.parameters:
            if key == "data":
                continue
            if key not in kwargs or kwargs[key] is None:
                if key in self.column_groups.data.keys():
                    raise ValueError(
                        f"The kwarg {key} of the function {func.__name__} is also a "
                        f"column groups id. "
                        f"Change the name of the column group id or include the kwarg "
                        f"in the CapData.regression_cols"
                    )
                if hasattr(self, key):
                    kwargs[key] = getattr(self, key)
        self.data[result] = func(self.data, *args, **kwargs)


if __name__ == "__main__":
    import doctest
    import pandas as pd  # noqa F811

    das = CapData("das")
    das.load_data(
        path="../examples/data/", fname="example_meas_data.csv", source="AlsoEnergy"
    )
    das.set_regression_cols(
        power="-mtr-", poa="irr-poa-", t_amb="temp-amb-", w_vel="wind--"
    )

    doctest.testmod()
