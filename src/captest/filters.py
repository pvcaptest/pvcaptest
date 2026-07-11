"""Filter step classes and row-filter helper functions.

This module is imported one-way by ``capdata.py``; it never imports
``capdata``. Filter steps touch a ``CapData`` instance only through the
runtime ``capdata`` argument to ``run``/``_execute``.
"""

import copy
import difflib
import importlib.util
from itertools import combinations
import warnings

import pandas as pd
import param
import sklearn.covariance as sk_cv
import statsmodels.formula.api as smf

from captest import util

pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn("Clear sky filtering will not work without the pvlib package.")


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
    df_return["index"] = ix_series.apply(lambda x: x.strftime("%m/%d/%Y %H %M"))  # noqa E501
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


def fit_model(
    df, fml="power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
):  # noqa E501
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


class BaseSummaryStep(param.Parameterized):
    """Common ancestor for steps that appear in the filtering summary.

    Holds the shared lifecycle (`run`), the optional `custom_name` display
    parameter, and the `args_repr` rendering used by the summary table.
    Subclasses implement `_execute`, returning the pandas ``Index`` of rows
    to keep after the step.

    Runtime state (`pts_after`, `ix_after`) is set by `run` as plain attributes
    and is never serialized. A step's "before" index/count and points-removed
    are not stored — they are chain-derived on demand via
    `CapData._ix_before`/`_pts_before`.
    """

    custom_name = param.String(
        default=None,
        allow_None=True,
        doc="Optional display name in the summary table.",
    )

    # Class-intrinsic human-readable template; set by concrete subclasses.
    # Plain class attribute (not a param) so it is never serialized.
    _explanation_template = None

    def run(self, capdata):
        """Execute the step, record runtime state, and append self to filters.

        Records only ``ix_after``/``pts_after`` (the rows this step kept and
        their count). A step's "before" index/count and points-removed are not
        stored — they are derived from the chain on demand by
        ``CapData._ix_before(i)``/``_pts_before(i)`` and surfaced through
        ``get_summary`` / ``_removed_by_step``. This keeps attribution correct
        for filters that make nested filter calls during ``_execute`` (e.g.
        ``Outliers`` invoking ``filter_missing`` on NaN): the nested step
        is appended to ``capdata.filters`` before this one, so this step's
        ``_ix_before`` resolves to the post-nested-call state and counts only
        what this step itself removed. If ``_execute`` raises, any steps
        appended by nested filter calls are rolled back (truncation by
        reassignment) before the exception propagates.
        """
        n_before = len(capdata.filters)
        try:
            self.ix_after = self._execute(capdata)
        except Exception:
            # Roll back any steps appended by nested filter calls (e.g.
            # Outliers -> filter_missing) so a failed step leaves the
            # pipeline byte-identical. Truncate via REASSIGNMENT so param
            # watchers fire.
            if len(capdata.filters) != n_before:
                capdata.filters = capdata.filters[:n_before]
            raise
        self.pts_after = len(self.ix_after)
        capdata.filters = capdata.filters + [self]
        if self.pts_after == 0:
            warnings.warn("The last filter removed all data!")

    def _execute(self, capdata):
        """Return a pandas Index of rows to keep. Implemented by subclasses."""
        raise NotImplementedError

    @property
    def args_repr(self):
        """Render the step's params for the summary.

        Includes every non-None value (defaults included, for full
        transparency); only ``None`` values and the internal
        ``custom_name``/``name`` keys are skipped. Subclasses override
        ``_args_for_repr`` to substitute runtime-resolved display values
        (e.g. a resolved ``ref_val``) without mutating the serialized params.
        """
        skip = {"custom_name", "name"}
        items = [
            f"{k}={v}"
            for k, v in self._args_for_repr().items()
            if k not in skip and v is not None
        ]
        return ", ".join(items) if items else "Default arguments"

    def _args_for_repr(self):
        """Mapping of param name -> display value for ``args_repr``.

        Defaults to the serialized param values; subclasses may override to
        swap in runtime-resolved values.
        """
        return dict(self.param.values())

    @property
    def explanation(self):
        """Human-readable description of the step's effect (read after run()).

        Renders ``_explanation_template`` with ``_explanation_values()``.
        Returns None when no template is defined or when the step has not yet
        been run (``_explanation_values`` typically depends on runtime-resolved
        state set during ``_execute``). Subclasses whose phrasing depends on
        which params are set override this property directly.
        """
        if self._explanation_template is None:
            return None
        if not hasattr(self, "ix_after"):
            return None
        return self._explanation_template.format(**self._explanation_values())

    def _explanation_values(self):
        """Substitution mapping for ``_explanation_template``.

        Defaults to ``_args_for_repr()``; subclasses override to supply
        run-time-resolved values (resolved column names, effective bounds).
        """
        return self._args_for_repr()

    def to_config(self):
        """Serialize this step to a config dict (every param, defaults
        included; the param-system ``name`` is omitted).

        Param values are deep-copied so the returned dict is an independent
        snapshot — mutating it never reaches back into the step's params.
        Numpy scalars (e.g. a ``ref_val`` taken from ``cd.rc['poa'].iloc[0]``)
        are coerced to native Python types via ``util.to_native`` so the dict
        survives ``yaml.safe_dump``; subclasses whose params hold callables
        override to encode them.
        """
        config = {"type": type(self).__name__}
        config.update(
            {
                k: util.to_native(copy.deepcopy(v))
                for k, v in self.param.values().items()
                if k != "name"
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance from a ``to_config()`` dict.

        The ``type`` key (emitted by ``to_config``) is dropped if present, so
        ``Cls.from_config(Cls(...).to_config())`` round-trips directly, not only
        through ``step_from_config``.
        """
        config = {k: v for k, v in config.items() if k != "type"}
        return cls(**config)


class BaseFilter(BaseSummaryStep):
    """A pure row-filtering step.

    Adds no interface beyond `BaseSummaryStep`; exists to distinguish row
    filters from non-filter summary steps (e.g. RepCond, FitRegression) for
    GUI styling and type checks.
    """

    pass


class Irradiance(BaseFilter):
    """Filter rows by an irradiance column to a low/high band.

    ``low``/``high`` are absolute values (W/m^2) unless ``ref_val`` is set,
    in which case they are treated as fractions of ``ref_val``.
    """

    _explanation_template = (
        "Intervals where {col_name} is below {low} or above {high}{units} were removed."
    )

    low = param.Number(
        default=None,
        allow_None=True,
        doc="Lower bound (W/m^2, or fraction of ref_val when ref_val is set).",
    )
    high = param.Number(
        default=None,
        allow_None=True,
        doc="Upper bound (W/m^2, or fraction of ref_val when ref_val is set).",
    )
    ref_val = param.Parameter(
        default=None,
        doc="Reference value; low/high are fractions of it. May be 'rep_irr'/"
        "'self_val' to resolve from capdata.rc at run time.",
    )
    col_name = param.String(
        default=None,
        allow_None=True,
        doc="Irradiance column to filter. Inferred from regression_cols if None.",
    )
    units = param.String(
        default="W/m^2",
        allow_None=True,
        doc="Unit label appended to the summary explanation. None/empty omits "
        "it — used by filter_threshold for non-irradiance columns.",
    )

    def _execute(self, capdata):
        irr_col = self.col_name if self.col_name is not None else capdata._get_poa_col()

        ref_val = self.ref_val
        if ref_val == "self_val":
            ref_val = "rep_irr"
        if ref_val == "rep_irr":
            # Resolve the reporting irradiance via the CapData, which honors a
            # CapTest-wired rc_source (so e.g. a sim filter can anchor on the
            # meas reporting irradiance) and falls back to its own rc.
            ref_val = capdata.rep_irr

        # Store effective/resolved values as runtime state (NOT params, so they
        # are never serialized). The ref_val param keeps the user's intent
        # ('rep_irr'/'self_val'/number) for YAML round-trip and re-resolution.
        self.ref_val_resolved = ref_val
        self.col_name_resolved = irr_col
        self.low_effective = (
            self.low * ref_val
            if (ref_val is not None and self.low is not None)
            else self.low
        )
        self.high_effective = (
            self.high * ref_val
            if (ref_val is not None and self.high is not None)
            else self.high
        )

        # Filter from the effective bounds so a None bound means "unbounded on
        # that side" rather than raising (low/high are allow_None params).
        df = capdata.data_filtered
        mask = pd.Series(True, index=df.index)
        if self.low_effective is not None:
            mask &= df[irr_col] >= self.low_effective
        if self.high_effective is not None:
            mask &= df[irr_col] <= self.high_effective
        return df.index[mask]

    def _args_for_repr(self):
        vals = dict(self.param.values())
        resolved = getattr(self, "ref_val_resolved", None)
        if resolved is not None:
            vals["ref_val"] = resolved
        # ``units`` only affects explanation wording; keep it out of the
        # summary's filter_arguments column.
        vals.pop("units", None)
        return vals

    def _explanation_values(self):
        # Effect-oriented: resolved column + effective absolute bounds.
        return {
            "col_name": self.col_name_resolved,
            "low": self.low_effective,
            "high": self.high_effective,
            "units": f" {self.units}" if self.units else "",
        }


class RollingStd(BaseFilter):
    """Remove intervals where a column's rolling-window standard deviation is
    at or above ``threshold`` (unstable / variable irradiance).

    ``column`` defaults to the regression POA column when None. ``window`` is
    passed to ``DataFrame.rolling`` and may be an int row count or a pandas
    offset alias (e.g. ``'10min'``). The leading rows of the window produce a
    NaN std and are removed, matching the original ``unstable_irr_filter``.
    """

    _explanation_template = (
        "Intervals where the rolling std (window={window}) of {column} was at "
        "or above {threshold} were removed."
    )

    column = param.String(
        default=None,
        allow_None=True,
        doc="Column to evaluate. Inferred from the regression POA column when None.",
    )
    window = param.Parameter(
        default=None,
        doc="Rolling window: int row count or pandas offset alias (e.g. "
        "'10min'). Passed to DataFrame.rolling.",
    )
    threshold = param.Number(
        default=None,
        allow_None=True,
        doc="Standard-deviation threshold; intervals whose rolling std is at "
        "or above this are removed.",
    )

    def _execute(self, capdata):
        if self.window is None or self.threshold is None:
            raise ValueError("RollingStd requires both window and threshold.")
        col = self.column if self.column is not None else capdata._get_poa_col()
        self.column_resolved = col
        df = capdata.data_filtered
        std = df[col].rolling(self.window).std()
        return df.index[std < self.threshold]

    def _explanation_values(self):
        return {
            "window": self.window,
            "column": getattr(self, "column_resolved", self.column),
            "threshold": self.threshold,
        }


class AbsDiffPrev(BaseFilter):
    """Remove intervals with a large fractional change from the previous
    interval (a step-change / stability filter).

    For column ``c`` the test is ``abs(c.diff() / c) <= threshold``; intervals
    above the threshold are removed. ``column`` defaults to the regression POA
    column when None. The first interval has an undefined difference (NaN) and
    is removed, matching the original ``filter_abs_perc_diff_prev_interval``.
    """

    _explanation_template = (
        "Intervals where {column} changed by more than {threshold} "
        "(fractional) from the previous interval were removed."
    )

    column = param.String(
        default=None,
        allow_None=True,
        doc="Column to evaluate. Inferred from the regression POA column when None.",
    )
    threshold = param.Number(
        default=0.05,
        doc="Maximum allowed absolute fractional change from the previous "
        "interval; intervals above this are removed.",
    )

    def _execute(self, capdata):
        col = self.column if self.column is not None else capdata._get_poa_col()
        self.column_resolved = col
        df = capdata.data_filtered
        s = df[col]
        abs_diff = (s.diff() / s).abs()
        return df.index[abs_diff <= self.threshold]

    def _explanation_values(self):
        return {
            "column": getattr(self, "column_resolved", self.column),
            "threshold": self.threshold,
        }


class BooleanFlag(BaseFilter):
    """Remove intervals where a boolean/flag column is truthy.

    ``column`` values are coerced with ``astype(bool)`` so 0/1, real booleans,
    and NaN (which is truthy) are handled uniformly. By default rows where the
    column is truthy are removed; set ``invert=True`` to instead remove rows
    where the column is falsy (keeping only the truthy rows).
    """

    column = param.String(
        default=None,
        allow_None=True,
        doc="Boolean/flag column. Rows where this is truthy are removed (or "
        "falsy, when invert=True).",
    )
    invert = param.Boolean(
        default=False,
        doc="If True, remove rows where the column is falsy instead of truthy.",
    )

    def _execute(self, capdata):
        if self.column is None:
            raise ValueError("BooleanFlag requires a column.")
        df = capdata.data_filtered
        mask = df[self.column].astype(bool)
        keep = mask if self.invert else ~mask
        return df.index[keep]

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None
        flagged = "False" if self.invert else "True"
        return f"Intervals flagged {flagged} in {self.column} were removed."


class Sensors(BaseFilter):
    """Drop rows where redundant sensors in a group disagree beyond a threshold.

    For each sensor group named in ``thresholds``, a row-wise comparison
    (selected by ``method``) is applied across that group's columns; rows
    flagged inconsistent are removed. Ignores columns generated by
    ``agg_sensors`` by operating on the pre-aggregation columns when present.
    """

    _explanation_template = (
        "Rows with inconsistent readings within sensor group(s) {groups} "
        "(compared using {method}) were removed."
    )

    # String name -> row-filter callable. A callable assigned to ``method``
    # directly bypasses this table (the custom third option).
    _BUILTIN_COMPARISONS = {
        "percent_diff": check_all_perc_diff_comb,
        "abs_diff": abs_diff_from_average,
    }

    method = param.Selector(
        default="percent_diff",
        objects=["percent_diff", "abs_diff"],
        check_on_set=False,
        doc="Sensor-comparison method: 'percent_diff' (pairwise percent "
        "difference) or 'abs_diff' (absolute difference from the group "
        "average). A callable with signature func(series, threshold) -> bool "
        "may also be assigned for a custom comparison.",
    )
    thresholds = param.Dict(
        default=None,
        allow_None=True,
        doc="Map of sensor-group key -> threshold (e.g. {'irr-poa-': 0.05}). "
        "Values are decimal fractions for 'percent_diff' or absolute units "
        "for 'abs_diff'. None defaults to {<poa group>: 0.05} for "
        "'percent_diff'; other methods require an explicit dict.",
    )

    def _resolve_comparison(self):
        if callable(self.method):
            return self.method
        return self._BUILTIN_COMPARISONS[self.method]

    def _method_label(self):
        if callable(self.method):
            return self.method.__name__
        return self.method

    def _execute(self, capdata):
        if capdata.pre_agg_cols is not None:
            df = capdata.data_filtered[capdata.pre_agg_cols]
            trans = capdata.pre_agg_trans
            regression_cols = capdata.pre_agg_reg_trans
        else:
            df = capdata.data_filtered
            trans = capdata.column_groups
            regression_cols = capdata.regression_cols

        thresholds = self.thresholds
        if thresholds is None:
            if self.method == "percent_diff":
                thresholds = {regression_cols["poa"]: 0.05}
            else:
                raise ValueError(
                    "thresholds is required when method is not 'percent_diff'."
                )
        if not thresholds:
            raise ValueError("thresholds must not be empty")
        self.thresholds_resolved = thresholds

        comparison = self._resolve_comparison()
        index = None
        for key, threshold in thresholds.items():
            sensors_df = df[trans[key]]
            next_index = sensor_filter(sensors_df, threshold, row_filter=comparison)
            index = next_index if index is None else index.intersection(next_index)
        return index

    def _args_for_repr(self):
        vals = dict(self.param.values())
        vals["method"] = self._method_label()
        resolved = getattr(self, "thresholds_resolved", None)
        if resolved is not None:
            vals["thresholds"] = resolved
        return vals

    def _explanation_values(self):
        return {
            "groups": ", ".join(self.thresholds_resolved),
            "method": self._method_label(),
        }

    def to_config(self):
        config = super().to_config()
        if callable(self.method):
            config["method"] = util.callable_to_qualname(self.method)
        else:
            config["method"] = self.method
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("type", None)
        method = config.get("method")
        # A qualname-encoded custom callable contains ':'; a built-in name
        # ('percent_diff'/'abs_diff') is left as the plain string.
        if isinstance(method, str) and ":" in method:
            config["method"] = util.callable_from_qualname(method)
        return cls(**config)


class Time(BaseFilter):
    """Filter rows to a time window described by start/end/days/test_date.

    Multiple parameter combinations are supported:

    - ``start`` + ``end`` — keep the window (or drop it if ``drop=True``).
    - ``start`` + ``days`` — keep a window of ``days`` starting at ``start``.
    - ``end`` + ``days`` — keep a window of ``days`` ending at ``end``.
    - ``test_date`` + ``days`` — keep a window of ``days`` centered on
      ``test_date``.
    - ``start`` only — keep rows from ``start`` to the last timestamp.
    - ``end`` only — keep rows from the first timestamp to ``end``.

    The legacy ``wrap_year`` flag is intentionally not supported here; the
    wrap-year functionality has moved to CapTest's auto-wrap step at setup
    time so sim data is already contiguous before any filtering runs.
    """

    start = param.Parameter(default=None, doc="Window start (str or Timestamp).")
    end = param.Parameter(default=None, doc="Window end (str or Timestamp).")
    test_date = param.Parameter(
        default=None,
        doc="Center of a symmetric ``days``-wide window (str or Timestamp).",
    )
    days = param.Integer(default=None, allow_None=True, doc="Window length in days.")
    drop = param.Boolean(
        default=False,
        doc="When True, remove the resolved window from the data instead of keeping it.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        start = pd.to_datetime(self.start) if self.start is not None else None
        end = pd.to_datetime(self.end) if self.end is not None else None
        test_date = (
            pd.to_datetime(self.test_date) if self.test_date is not None else None
        )

        if test_date is not None:
            if self.days is None:
                warnings.warn("Must specify days")
                return df.index
            offset = pd.DateOffset(days=self.days // 2)
            start = test_date - offset
            end = test_date + offset
        elif start is not None and end is not None:
            pass
        elif start is not None:
            if self.days is not None:
                end = start + pd.DateOffset(days=self.days)
            else:
                end = df.index[-1]
        elif end is not None:
            if self.days is not None:
                start = end - pd.DateOffset(days=self.days)
            else:
                start = df.index[0]
        else:
            raise ValueError(
                "filter_time requires at least one of start, end, or test_date"
            )

        # By the time we get here every dispatch branch has set both `start`
        # and `end` (or raised); `drop` applies to whichever bounded window
        # was resolved, not just to the user-gave-both case.
        self._effective_start = start
        self._effective_end = end

        if self.drop:
            selected = df.loc[start:end, :]
            df_temp = df.loc[df.index.difference(selected.index), :]
        else:
            df_temp = df.loc[start:end, :]
        return df_temp.index

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None

        def _fmt(v):
            return str(pd.Timestamp(v).date()) if v is not None else None

        # Effective window edges (set by _execute); fall back to the params
        # for the rare cases where explanation is read without ix_after (the
        # base-class guard above normally prevents this).
        es = _fmt(getattr(self, "_effective_start", None))
        ee = _fmt(getattr(self, "_effective_end", None))
        s, e, td = _fmt(self.start), _fmt(self.end), _fmt(self.test_date)
        d = self.days

        if td is not None:
            if d is None:
                return None
            if self.drop:
                return (
                    f"Data within a {d}-day window centered on {td} "
                    f"({es} to {ee}) was removed."
                )
            return (
                f"Data outside a {d}-day window centered on {td} "
                f"({es} to {ee}) was removed."
            )
        if s is not None and e is not None:
            if self.drop:
                return f"Data between {es} and {ee} was removed."
            return f"Data outside the period {es} to {ee} was removed."
        if s is not None:
            if d is not None:
                if self.drop:
                    return (
                        f"Data within the {d}-day period from {es} to {ee} was removed."
                    )
                return f"Data outside the {d}-day period from {es} to {ee} was removed."
            if self.drop:
                return f"Data from {es} onward was removed."
            return f"Data before {es} was removed."
        if e is not None:
            if d is not None:
                if self.drop:
                    return (
                        f"Data within the {d}-day period from {es} to {ee} was removed."
                    )
                return f"Data outside the {d}-day period from {es} to {ee} was removed."
            if self.drop:
                return f"Data up to {ee} was removed."
            return f"Data after {ee} was removed."
        return None


class Custom(BaseFilter):
    """Apply an arbitrary callable to ``capdata.data_filtered`` as a row filter.

    ``func`` is a callable that takes a DataFrame as its first argument and
    returns a DataFrame whose index is the rows to keep. Typical use is a
    pandas DataFrame method like ``pd.DataFrame.dropna`` or
    ``pd.DataFrame.between_time``.

    Unlike most filters, ``func``/``*args``/``**kwargs`` are stored as plain
    instance attributes (not ``param`` parameters). Callables and variadics
    don't fit ``param``'s declared-parameter model; YAML serialization for
    ``func`` (module-qualified-name string) is handled by the YAML plan.
    """

    _explanation_template = "Custom filter {call} was applied."

    def __init__(self, func, *args, custom_name=None, **kwargs):
        super().__init__(custom_name=custom_name)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _execute(self, capdata):
        result = self.func(capdata.data_filtered, *self.args, **self.kwargs)
        return result.index

    @property
    def args_repr(self):
        """Render ``func_name(arg, ..., k=v, ...)`` — matches the legacy regex.

        Uses ``getattr(self.func, '__name__', repr(self.func))`` because some
        callables (e.g. ``functools.partial`` instances, callable class
        instances) do not expose ``__name__``. Without the guard, accessing
        ``args_repr`` would raise ``AttributeError`` when the step is
        rendered in the filtering summary.
        """
        name = getattr(self.func, "__name__", repr(self.func))
        arg_parts = [repr(a) for a in self.args]
        kwarg_parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{name}({', '.join(arg_parts + kwarg_parts)})"

    def _explanation_values(self):
        return {"call": self.args_repr}

    def to_config(self):
        return {
            "type": "Custom",
            "func": util.callable_to_qualname(self.func),
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "custom_name": self.custom_name,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        func = util.callable_from_qualname(config["func"])
        args = config.get("args") or []
        kwargs = config.get("kwargs") or {}
        return cls(func, *args, custom_name=config.get("custom_name"), **kwargs)


class Outliers(BaseFilter):
    """Remove statistical outliers in the (poa, power) plane via sklearn EllipticEnvelope.

    Reads ``capdata.floc[["poa", "power"]]`` (the regression-mapped columns),
    drops any NaN rows by delegating to ``capdata.filter_missing`` (which
    records its own step), fits ``sklearn.covariance.EllipticEnvelope`` on
    the cleaned 2-D matrix, and keeps the rows whose ``predict`` is 1.

    ``envelope_kwargs`` carries user overrides for ``EllipticEnvelope``;
    defaults (``support_fraction=0.9``, ``contamination=0.04``) are merged in
    at run time. The merged dict is exposed on ``envelope_kwargs_resolved``
    for display.
    """

    _explanation_template = (
        "Statistical outliers in (poa, power), detected via "
        "EllipticEnvelope({kwargs}), were removed."
    )
    _default_envelope_kwargs = {"support_fraction": 0.9, "contamination": 0.04}

    envelope_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Override kwargs for sklearn EllipticEnvelope. Defaults "
        "(support_fraction=0.9, contamination=0.04) are merged in at run time.",
    )

    def _execute(self, capdata):
        XandY = capdata.floc[["poa", "power"]]
        if XandY.shape[1] > 2:
            warnings.warn(
                "Too many columns. Try running aggregate_sensors before using "
                "filter_outliers."
            )
            return capdata.data_filtered.index

        if XandY.isna().any().any():
            warnings.warn(
                "Poa and/or power columns contain missing values. Calling "
                "filter_missing on poa and power columns before continuing "
                "with filter_outliers."
            )
            capdata.filter_missing(columns=XandY.columns.tolist())
            XandY = capdata.floc[["poa", "power"]]
            # The nested filter_missing step is appended to capdata.filters
            # before this Outliers step, so this step's chain-derived
            # _ix_before resolves to the post-filter_missing state — its
            # removed count reflects only the outlier drop, not the NaN rows.

        resolved = dict(self._default_envelope_kwargs)
        if self.envelope_kwargs:
            resolved.update(self.envelope_kwargs)
        self.envelope_kwargs_resolved = resolved

        X = XandY.values
        clf = sk_cv.EllipticEnvelope(**resolved)
        clf.fit(X)
        mask = clf.predict(X) == 1
        return capdata.data_filtered.index[mask]

    @property
    def args_repr(self):
        """Render ``EllipticEnvelope(k=v, ...)`` using the resolved kwargs."""
        resolved = getattr(self, "envelope_kwargs_resolved", None)
        if resolved is None:
            return super().args_repr
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return f"EllipticEnvelope({kw})"

    def _explanation_values(self):
        resolved = getattr(self, "envelope_kwargs_resolved", {}) or {}
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return {"kwargs": kw}


class Clearsky(BaseFilter):
    """Remove intervals where measured GHI doesn't match modeled clear-sky GHI.

    Uses ``pvlib.clearsky.detect_clearsky`` to classify each timestamp as
    clear or cloudy. By default keeps the clear timestamps and removes
    cloudy ones; set ``keep_clear=False`` to invert.

    Requires the ``ghi_mod_csky`` column in ``capdata.data_filtered`` (added
    by ``io.load_data`` when the ``site`` argument is supplied). When
    ``ghi_col`` is None, the measured GHI column is auto-detected from
    ``column_groups`` (the single ``irr-ghi-*`` entry other than
    ``irr-ghi-clear_sky``); multi-column groups are averaged with a warning.
    """

    _explanation_template = (
        "{removed_kind} intervals (detected via pvlib "
        "detect_clearsky({kwargs})) were removed."
    )
    _default_detect_kwargs = {"infer_limits": True}

    ghi_col = param.String(
        default=None,
        allow_None=True,
        doc="Measured GHI column name. Auto-detected from column_groups if None.",
    )
    keep_clear = param.Boolean(
        default=True,
        doc="Keep clear intervals (True) or keep cloudy intervals (False).",
    )
    detect_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Override kwargs for pvlib detect_clearsky. Default "
        "infer_limits=True is merged in at run time.",
    )

    def _execute(self, capdata):
        if "ghi_mod_csky" not in capdata.data_filtered.columns:
            warnings.warn(
                "Modeled clear sky data must be available to run this filter. "
                "Use CapData load_data clear_sky option."
            )
            return capdata.data_filtered.index

        if self.ghi_col is None:
            ghi_keys = []
            for key in capdata.column_groups.keys():
                defs = key.split("-")
                if len(defs) == 1:
                    continue
                if defs[1] == "ghi":
                    ghi_keys.append(key)
            ghi_keys = [k for k in ghi_keys if k != "irr-ghi-clear_sky"]

            if not ghi_keys:
                warnings.warn(
                    "No measured GHI column group found in column_groups. "
                    "Pass column name to ghi_col."
                )
                return capdata.data_filtered.index
            if len(ghi_keys) > 1:
                warnings.warn(
                    "Too many ghi categories. Pass column name to ghi_col to "
                    "use a specific column."
                )
                return capdata.data_filtered.index

            meas_ghi = capdata.floc[ghi_keys[0]]
            if meas_ghi.shape[1] > 1:
                warnings.warn(
                    "Averaging measured GHI data. Pass column name to ghi_col "
                    "to use a specific column."
                )
            meas_ghi = meas_ghi.mean(axis=1)
        else:
            meas_ghi = capdata.data_filtered[self.ghi_col]

        resolved = dict(self._default_detect_kwargs)
        if self.detect_kwargs:
            resolved.update(self.detect_kwargs)
        self.detect_kwargs_resolved = resolved

        clear_per = detect_clearsky(
            measured=meas_ghi,
            clearsky=capdata.data_filtered["ghi_mod_csky"],
            times=meas_ghi.index,
            **resolved,
        )
        if not any(clear_per):
            warnings.warn(
                "No clear periods detected. Try adjusting detect_clearsky "
                "parameters via kwargs."
            )
            return capdata.data_filtered.index

        mask = clear_per if self.keep_clear else ~clear_per
        return capdata.data_filtered.index[mask]

    @property
    def args_repr(self):
        """Render ``detect_clearsky(k=v, ...)`` using the resolved kwargs."""
        resolved = getattr(self, "detect_kwargs_resolved", None)
        if resolved is None:
            return super().args_repr
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return f"detect_clearsky({kw})"

    def _explanation_values(self):
        resolved = getattr(self, "detect_kwargs_resolved", {}) or {}
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return {
            "removed_kind": "Cloudy" if self.keep_clear else "Clear",
            "kwargs": kw,
        }


class Pvsyst(BaseFilter):
    """Remove PVsyst intervals operating off the maximum power point.

    Drops rows where any of the PVsyst current-limit columns
    (``IL Pmin``/``IL Vmin``/``IL Pmax``/``IL Vmax``) is greater than 0.
    Column names with spaces or underscores are both recognized; a missing
    column warns and is skipped.
    """

    _explanation_template = (
        "PVsyst intervals operating off the maximum power point "
        "(IL Pmin/Vmin/Pmax/Vmax > 0) were removed."
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        columns = ["IL Pmin", "IL Vmin", "IL Pmax", "IL Vmax"]
        index = df.index
        for column in columns:
            if column not in df.columns:
                column = column.replace(" ", "_")
            if column in df.columns:
                indices_to_drop = df[df[column] > 0].index
                if not index.equals(indices_to_drop):
                    index = index.difference(indices_to_drop)
            else:
                warnings.warn(
                    "{} or {} is not a column in the data.".format(
                        column, column.replace("_", " ")
                    )
                )
        return index


class Shade(BaseFilter):
    """Remove intervals of array shading.

    By default removes rows where the PVsyst ``FShdBm`` shading-fraction
    column is below ``fshdbm`` (default 1.0 — i.e. any shading). Pass a
    ``query_str`` to instead filter via ``DataFrame.query`` (e.g. when only
    a shading-loss column is available): ``"ShdLoss<=50"``.
    """

    _explanation_template = (
        "Intervals of array shading (kept where {query}) were removed."
    )

    fshdbm = param.Number(
        default=1.0,
        doc="Shading-fraction threshold; rows with FShdBm below this are "
        "removed. Ignored when query_str is given.",
    )
    query_str = param.String(
        default=None,
        allow_None=True,
        doc="Optional DataFrame.query expression overriding the FShdBm test.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        fshdbm = self.fshdbm  # noqa: F841 - referenced via @fshdbm in query
        query_str = self.query_str
        if query_str is None:
            query_str = "FShdBm>=@fshdbm"
        return df.query(query_str).index

    def _explanation_values(self):
        # For the default query, render the resolved fshdbm value rather than
        # the literal "@fshdbm" placeholder (which is only needed by df.query).
        if self.query_str is None:
            return {"query": f"FShdBm>={self.fshdbm}"}
        return {"query": self.query_str}


class PowerFactor(BaseFilter):
    """Remove intervals with a power factor below a threshold.

    Keeps rows where every column in the power-factor group (the first
    ``column_groups`` key beginning with ``pf``) has an absolute value at or
    above ``pf``.
    """

    _explanation_template = "Intervals with a power factor below {pf} were removed."

    pf = param.Number(
        default=None,
        allow_None=True,
        doc="Power-factor threshold, e.g. 0.999. Rows with any |pf| below "
        "this are removed.",
    )

    def _execute(self, capdata):
        selection = None
        for key in capdata.column_groups.keys():
            if key.find("pf") == 0:
                selection = key
        if selection is None:
            warnings.warn(
                "No power factor column group found in column_groups; "
                "filter_pf made no changes."
            )
            return capdata.data_filtered.index
        df = capdata.data_filtered[capdata.column_groups[selection]]
        mask = (df.abs() >= self.pf).all(axis=1)
        return capdata.data_filtered.index[mask]


class Power(BaseFilter):
    """Remove intervals at or above a power threshold.

    With ``percent`` set, ``power`` is treated as nameplate and the effective
    threshold is ``power * (1 - percent)``. ``columns`` selects the power
    data: None uses the regression power column; a column-group key applies
    the threshold across the group; a bare column name uses that column.
    """

    _explanation_template = "Intervals at or above {threshold} power were removed."

    power = param.Number(
        default=None,
        allow_None=True,
        doc="Power threshold (or nameplate if percent set).",
    )
    percent = param.Number(
        default=None,
        allow_None=True,
        doc="If set, threshold is power*(1-percent). Decimal, e.g. 0.01 for 1%.",
    )
    columns = param.Parameter(
        default=None,
        doc="Column or column-group to filter on. None uses the regression "
        "power column. A non-string non-None value warns and is a no-op "
        "(preserving the legacy validation path).",
    )

    def _execute(self, capdata):
        power = self.power
        if self.percent is not None:
            power = power * (1 - self.percent)
        self.power_threshold = power

        multiple_columns = False
        if self.columns is None:
            power_data = capdata.get_reg_cols("power")
        elif isinstance(self.columns, str):
            if self.columns in capdata.column_groups.keys():
                power_data = capdata.floc[self.columns]
                multiple_columns = True
            else:
                power_data = pd.DataFrame(capdata.data_filtered[self.columns])
                power_data = power_data.rename(columns={power_data.columns[0]: "power"})
        else:
            warnings.warn("columns must be None or a string.")
            return capdata.data_filtered.index

        if multiple_columns:
            mask = power_data.apply(lambda x: all(x.le(power, fill_value=True)), axis=1)
        else:
            mask = power_data["power"] < power
        return capdata.data_filtered.index[mask]

    def _explanation_values(self):
        return {"threshold": getattr(self, "power_threshold", self.power)}


class Days(BaseFilter):
    """Keep (or drop) the timestamps belonging to a list of days.

    Each entry in ``days`` selects all timestamps on that calendar day
    (``DataFrame.loc[day]``). By default only those days are kept; set
    ``drop=True`` to remove them and keep everything else.
    """

    days = param.List(
        default=None,
        allow_None=True,
        doc="Days to select (or drop). Each is a date string or Timestamp.",
    )
    drop = param.Boolean(
        default=False,
        doc="Drop the listed days instead of keeping only them.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        ix_all_days = None
        for day in self.days:
            ix_day = df.loc[day].index
            ix_all_days = ix_day if ix_all_days is None else ix_all_days.union(ix_day)
        if self.drop:
            return df.index.difference(ix_all_days)
        return ix_all_days

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None
        days = ", ".join(str(d) for d in (self.days or []))
        if self.drop:
            return f"Timestamps on the days [{days}] were removed."
        return f"All timestamps except the days [{days}] were removed."


class Missing(BaseFilter):
    """Remove rows with missing data (NaN) in the regression columns.

    By default checks the columns identified by ``regression_cols`` (via the
    ``regcols`` floc key); pass ``columns`` to restrict the NaN check to a
    subset.
    """

    _explanation_template = (
        "Intervals with missing data in the regression columns were removed."
    )

    columns = param.List(
        default=None,
        allow_None=True,
        doc="Subset of columns to check for NaN. None uses the regression columns.",
    )

    def _execute(self, capdata):
        if self.columns is None:
            return capdata.floc["regcols"].dropna().index
        return capdata.floc[self.columns].dropna().index


class Regression(BaseFilter):
    """Remove intervals whose regression residuals are statistical outliers.

    Fits the CapData regression formula (``capdata.regression_formula``) to the
    regression columns and keeps only rows whose residual is within ``n_std``
    residual standard deviations. The fitted statsmodels result is exposed on
    ``self.regression_model`` after ``_execute`` so callers (e.g.
    ``CapData.fit_regression``) can print its summary.
    """

    _explanation_template = (
        "Intervals with regression residuals beyond {n_std} standard "
        "deviations were removed."
    )

    n_std = param.Number(
        default=2,
        doc="Residual cutoff in standard deviations; rows beyond this are removed.",
    )

    def _execute(self, capdata):
        df = capdata.get_reg_cols()
        if df.isna().any().any():
            warnings.warn(
                "Regression columns contain missing values. Calling "
                "filter_missing before fitting the regression."
            )
            capdata.filter_missing()
            df = capdata.get_reg_cols()
        reg = fit_model(df, fml=capdata.regression_formula)
        self.regression_model = reg
        threshold = self.n_std * reg.scale**0.5
        return reg.resid[reg.resid.abs() < threshold].index


def _encode_func_value(v):
    """Make one RepCond.func value yaml-safe.

    ``perc_wrap(N)`` callable -> ``"perc_N"``; any other named callable ->
    ``"module:qualname"`` (lambdas/closures raise via ``callable_to_qualname``);
    strings (e.g. ``"mean"``) and ``None`` pass through unchanged.
    """
    if callable(v):
        encoded = util._perc_wrap_to_string(v)
        if callable(encoded):  # not a perc_wrap callable -> a named callable
            return util.callable_to_qualname(v)
        return encoded
    return v


def _decode_func_value(v):
    """Inverse of ``_encode_func_value``.

    ``"module:qualname"`` -> imported callable; ``"perc_N"`` -> ``perc_wrap(N)``;
    other strings (``"mean"``) and ``None`` pass through.
    """
    if isinstance(v, str) and ":" in v:
        return util.callable_from_qualname(v)
    return util._resolve_perc_string(v)


class RepCond(BaseSummaryStep):
    """Reporting-conditions calculation as a zero-removal summary step.

    Computes ``capdata.rc`` from the filtered data at this point in the chain
    and returns the index **unchanged** (``pts_removed == 0``), so the step
    appears in the summary at its position relative to the filters that
    preceded it. The reporting-conditions math is not duplicated here: it lives
    in ``CapData._calc_rep_cond``, reached via the runtime ``capdata`` argument
    so ``filters.py`` needs no import of ``capdata`` or ``ReportingIrradiance``.
    Inherits ``BaseSummaryStep`` directly (not ``BaseFilter``) because it is not
    a row filter; it still belongs in ``capdata.filters`` because that list
    accepts any ``BaseSummaryStep``.
    """

    func = param.Parameter(
        default=None,
        doc="Aggregation(s) for each rhs variable: dict/str/callable/None.",
    )
    w_vel = param.Parameter(
        default=None,
        doc="Override for the wind-speed reporting condition.",
    )
    irr_bal = param.Boolean(
        default=False,
        doc="Use ReportingIrradiance to balance the irradiance band.",
    )
    percent_filter = param.Number(
        default=20,
        doc="Percent band around the reporting irradiance (irr_bal only).",
    )
    front_poa = param.String(
        default="poa",
        doc="regression_cols key used as the irradiance driver (irr_bal only).",
    )
    rc_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Extra kwargs forwarded to ReportingIrradiance (irr_bal only).",
    )

    _explanation_template = (
        "Reporting conditions were calculated (no intervals removed)."
    )

    def _execute(self, capdata):
        capdata._calc_rep_cond(
            self.func,
            self.w_vel,
            self.irr_bal,
            self.percent_filter,
            self.front_poa,
            self.rc_kwargs,
        )
        return capdata.data_filtered.index

    def to_config(self):
        config = super().to_config()
        func = self.func
        if isinstance(func, dict):
            config["func"] = {k: _encode_func_value(v) for k, v in func.items()}
        else:
            config["func"] = _encode_func_value(func)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("type", None)
        func = config.get("func")
        if isinstance(func, dict):
            config["func"] = {k: _decode_func_value(v) for k, v in func.items()}
        else:
            config["func"] = _decode_func_value(func)
        return cls(**config)


FILTER_REGISTRY = {
    "Irradiance": Irradiance,
    "RollingStd": RollingStd,
    "AbsDiffPrev": AbsDiffPrev,
    "BooleanFlag": BooleanFlag,
    "Pvsyst": Pvsyst,
    "Shade": Shade,
    "Time": Time,
    "Days": Days,
    "Outliers": Outliers,
    "PowerFactor": PowerFactor,
    "Power": Power,
    "Custom": Custom,
    "Sensors": Sensors,
    "Clearsky": Clearsky,
    "Missing": Missing,
    "Regression": Regression,
    "RepCond": RepCond,
}


def step_from_config(d):
    """Build a filter step from a ``to_config()`` dict via ``FILTER_REGISTRY``."""
    d = dict(d)
    cls_name = d.pop("type")
    if cls_name not in FILTER_REGISTRY:
        suggestion = difflib.get_close_matches(cls_name, FILTER_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown filter type {cls_name!r} in pipeline config.{hint}")
    return FILTER_REGISTRY[cls_name].from_config(d)
