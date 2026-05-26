"""Filter step classes and row-filter helper functions.

This module is imported one-way by ``capdata.py``; it never imports
``capdata``. Filter steps touch a ``CapData`` instance only through the
runtime ``capdata`` argument to ``run``/``_execute``.
"""

from itertools import combinations
import warnings

import pandas as pd
import param


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


class BaseSummaryStep(param.Parameterized):
    """Common ancestor for steps that appear in the filtering summary.

    Holds the shared lifecycle (`run`), the optional `custom_name` display
    parameter, and the `args_repr` rendering used by the summary table.
    Subclasses implement `_execute`, returning the pandas ``Index`` of rows
    to keep after the step.

    Runtime state (`pts_before`, `pts_after`, `pts_removed`, `ix_before`,
    `ix_after`) is set by `run` as plain attributes and is never serialized.
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
        """Execute the step, record runtime state, and append self to filters."""
        self.pts_before = capdata.data_filtered.shape[0]
        self.ix_before = capdata.data_filtered.index
        self.ix_after = self._execute(capdata)
        self.pts_after = len(self.ix_after)
        self.pts_removed = self.pts_before - self.pts_after
        capdata.filters = capdata.filters + [self]
        # Transitional: keep the legacy data_filtered attribute consistent
        # until data_filtered becomes a derived property (plan 4).
        capdata.data_filtered = capdata.data.loc[self.ix_after, :]
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
        Returns None when no template is defined. Subclasses whose phrasing
        depends on which params are set override this property directly.
        """
        if self._explanation_template is None:
            return None
        return self._explanation_template.format(**self._explanation_values())

    def _explanation_values(self):
        """Substitution mapping for ``_explanation_template``.

        Defaults to ``_args_for_repr()``; subclasses override to supply
        run-time-resolved values (resolved column names, effective bounds).
        """
        return self._args_for_repr()


class BaseFilter(BaseSummaryStep):
    """A pure row-filtering step.

    Adds no interface beyond `BaseSummaryStep`; exists to distinguish row
    filters from non-filter summary steps (e.g. RepCond, FitRegression) for
    GUI styling and type checks.
    """

    pass


class FilterIrr(BaseFilter):
    """Filter rows by an irradiance column to a low/high band.

    ``low``/``high`` are absolute values (W/m^2) unless ``ref_val`` is set,
    in which case they are treated as fractions of ``ref_val``.
    """

    _legacy_name = "filter_irr"
    _explanation_template = (
        "Intervals where {col_name} is below {low} or above {high} W/m^2 were removed."
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

    def _execute(self, capdata):
        irr_col = self.col_name if self.col_name is not None else capdata._get_poa_col()

        ref_val = self.ref_val
        if ref_val == "self_val":
            ref_val = "rep_irr"
        if ref_val == "rep_irr":
            if capdata.rc is None:
                raise ValueError(
                    "ref_val='rep_irr' requires reporting conditions to be set. "
                    "Call rep_cond() before filtering with ref_val='rep_irr'."
                )
            if "poa" not in capdata.rc.columns:
                raise ValueError(
                    "ref_val='rep_irr' requires a 'poa' column in capdata.rc. "
                    "The reporting conditions DataFrame does not have a 'poa' column."
                )
            ref_val = float(capdata.rc["poa"].iloc[0])

        # Store effective/resolved values as runtime state (NOT params, so they
        # are never serialized). The ref_val param keeps the user's intent
        # ('rep_irr'/'self_val'/number) for YAML round-trip and re-resolution.
        self.ref_val_resolved = ref_val
        self.col_name_resolved = irr_col
        self.low_effective = self.low * ref_val if ref_val is not None else self.low
        self.high_effective = self.high * ref_val if ref_val is not None else self.high

        return filter_irr(
            capdata.data_filtered, irr_col, self.low, self.high, ref_val=ref_val
        ).index

    def _args_for_repr(self):
        vals = dict(self.param.values())
        resolved = getattr(self, "ref_val_resolved", None)
        if resolved is not None:
            vals["ref_val"] = resolved
        return vals

    def _explanation_values(self):
        # Effect-oriented: resolved column + effective absolute bounds.
        return {
            "col_name": self.col_name_resolved,
            "low": self.low_effective,
            "high": self.high_effective,
        }
