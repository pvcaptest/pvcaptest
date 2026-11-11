"""Serializable validity checks for a performance-ratio (PR) test.

A check is a ``param``-based step, modeled on the filter steps in
:mod:`captest.filters`, except that it never removes rows: it reports whether
the data satisfies a requirement (for example "at least 30 hours with
irradiance at or above 400 W/m2") together with a small detail table.

This module never imports :mod:`captest.capdata`; a check touches a
``CapData`` only through the runtime argument (``data``, ``data_filtered`` and
the flat ``regression_cols`` mapping of role to column).
"""

import difflib
import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
import param

from captest import util

NO_DATA_SUMMARY = "no data in the selected rows"
DATA_MODES = ["pr_rows", "filtered", "unfiltered_period"]


@dataclass
class CheckContext:
    """Run-time facts a check needs beyond the ``CapData`` itself.

    Parameters
    ----------
    dt_hours : float
        Dominant timestep of the dataset in hours.
    pr_rows : pd.Index or None, default None
        Index of the rows kept by the PR calculation (complete rows).
    period : tuple of pd.Timestamp or None, default None
        First and last day of the test period. A test run passes the measured
        test period for measured-side checks and ``None`` for modeled-side
        checks, whose calendar differs.
    """

    dt_hours: float
    pr_rows: pd.Index | None = None
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None


@dataclass
class CheckResult:
    """Outcome of one check.

    Parameters
    ----------
    name : str
        Label of the check that produced the result.
    passed : bool
        Whether the requirement is met.
    detail : pd.DataFrame
        Small table of the numbers behind the verdict.
    summary : str
        Short human-readable statement of the verdict.
    """

    name: str
    passed: bool
    detail: pd.DataFrame
    summary: str


def resolve_irradiance(capdata, irr):
    """Resolve an irradiance reference to a column of ``capdata.data``.

    Parameters
    ----------
    capdata : CapData
        Dataset whose ``regression_cols`` map roles to columns.
    irr : str
        A role name in ``capdata.regression_cols`` or a column of
        ``capdata.data``. ``"front_poa"`` falls back to the ``"poa"`` role
        when the dataset has no ``front_poa`` role.

    Returns
    -------
    str
        Column name in ``capdata.data``.

    Raises
    ------
    ValueError
        If ``irr`` is neither a role nor a column.
    """
    roles = capdata.regression_cols or {}
    role = (
        irr
        if irr in roles
        else "poa"
        if irr == "front_poa" and "poa" in roles
        else None
    )
    if role is not None:
        column = roles[role]
        if column not in capdata.data.columns:
            raise ValueError(
                f"Irradiance role {role!r} maps to column {column!r}, which is "
                "not a column of the data."
            )
        return column
    if irr in capdata.data.columns:
        return irr
    raise ValueError(
        f"Cannot resolve irradiance {irr!r}: it is neither a regression role "
        f"({sorted(roles)}) nor a column of the data."
    )


def _to_timestamp(value, index):
    """Convert ``value`` to a Timestamp in the timezone of ``index``."""
    stamp = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None and stamp.tzinfo is None:
        return stamp.tz_localize(tz)
    if tz is None and stamp.tzinfo is not None:
        return stamp.tz_localize(None)
    return stamp


def select_rows(capdata, check, context):
    """Select the rows a check evaluates, according to ``check.data``.

    Parameters
    ----------
    capdata : CapData
        Dataset to select from.
    check : BaseCheck
        Check whose ``data`` mode (and ``start`` / ``end``) drive the selection.
    context : CheckContext
        Supplies ``pr_rows`` and ``period``.

    Returns
    -------
    pd.DataFrame
        Selected rows of ``capdata.data``; possibly empty.

    Notes
    -----
    ``pr_rows`` uses ``context.pr_rows`` and falls back to
    ``capdata.data_filtered`` when it is ``None``. ``filtered`` uses
    ``capdata.data_filtered``. ``unfiltered_period`` uses ``capdata.data``
    restricted to whole calendar days (in the index's own timezone) from the
    period's first to last day, where the period is, in order: the check's own
    ``start`` / ``end``; ``context.period``; the first through last day
    holding a row of ``capdata.data_filtered``.
    """
    mode = check.data
    if mode == "filtered":
        return capdata.data_filtered
    if mode == "pr_rows":
        if context.pr_rows is None:
            return capdata.data_filtered
        data = capdata.data
        return data.loc[data.index.isin(context.pr_rows)]
    data = capdata.data
    if check.start is not None and check.end is not None:
        first, last = check.start, check.end
    elif context.period is not None:
        first, last = context.period
    else:
        kept = capdata.data_filtered.index
        if len(kept) == 0:
            return data.iloc[0:0]
        first, last = kept.min(), kept.max()
    first = _to_timestamp(first, data.index).normalize()
    stop = _to_timestamp(last, data.index).normalize() + pd.DateOffset(days=1)
    return data.loc[(data.index >= first) & (data.index < stop)]


def qualifying_count(irr, threshold):
    """Count intervals with finite irradiance at or above ``threshold``.

    Non-finite values (NaN, +/-inf) never qualify.
    """
    return int((np.isfinite(irr) & (irr >= threshold)).sum())


def _check_finite(name, value):
    """Raise ``ValueError`` unless ``value`` is a finite real number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")


def _check_non_negative(name, value):
    """Raise ``ValueError`` unless ``value`` is finite and not negative."""
    _check_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")


class BaseCheck(util.StrictAttrs, param.Parameterized):
    """Common ancestor of validity checks.

    A check selects rows, resolves an irradiance column and reports a
    `CheckResult`; it never removes rows. Subclasses implement `_evaluate`.

    Parameters
    ----------
    custom_name : str or None, default None
        Display label; the class name when ``None``.
    irr : str, default "front_poa"
        Irradiance as a regression role name or a column of the data.
    data : {"pr_rows", "filtered", "unfiltered_period"}
        Which rows the check evaluates; subclasses may override the default.
    start, end : str or None, default None
        Explicit period read only by ``data="unfiltered_period"``.
    """

    custom_name = param.String(default=None, allow_None=True)
    irr = param.String(default="front_poa")
    data = param.Selector(default="pr_rows", objects=DATA_MODES)
    start = param.String(default=None, allow_None=True)
    end = param.String(default=None, allow_None=True)

    @property
    def label(self):
        """Display label: ``custom_name`` or the class name."""
        return self.custom_name or type(self).__name__

    def _validate(self):
        """Validate parameters; subclasses extend this via ``super()``."""
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be set together or not at all.")

    def __init__(self, **params):
        super().__init__(**params)
        self._validate()

    def _empty_result(self):
        """Return the result for a selection with no rows."""
        return CheckResult(
            name=self.label,
            passed=False,
            detail=pd.DataFrame(),
            summary=NO_DATA_SUMMARY,
        )

    def evaluate(self, capdata, context):
        """Run the check against ``capdata``.

        Selects rows, returns a failing result when there are none, resolves
        the irradiance column and calls `_evaluate`. Subclasses that need a
        different pipeline override this method and can reuse `_empty_result`.

        Parameters
        ----------
        capdata : CapData
            Dataset to check.
        context : CheckContext
            Run-time facts (timestep, PR rows, period).

        Returns
        -------
        CheckResult
        """
        self._validate()
        frame = select_rows(capdata, self, context)
        if len(frame) == 0:
            return self._empty_result()
        column = resolve_irradiance(capdata, self.irr)
        return self._evaluate(frame, frame[column], context)

    def _evaluate(self, frame, irr, context):
        """Evaluate the selected rows; implemented by subclasses.

        Parameters
        ----------
        frame : pd.DataFrame
            Selected, non-empty rows.
        irr : pd.Series
            Irradiance of ``frame``.
        context : CheckContext

        Returns
        -------
        CheckResult
        """
        raise NotImplementedError

    def to_config(self):
        """Serialize to a config dict.

        Emits ``type`` and the check-specific parameters; ``irr``, ``data``,
        ``start``, ``end`` and ``custom_name`` appear only when they differ
        from the class default.
        """
        base = set(BaseCheck.param) | {"name"}
        config = {"type": type(self).__name__}
        values = util.params_to_config(self)
        config.update({k: v for k, v in values.items() if k not in base})
        for key in ("irr", "data", "start", "end", "custom_name"):
            if values[key] != type(self).param[key].default:
                config[key] = values[key]
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance from a `to_config` dict (``type`` is ignored)."""
        return cls(**{k: v for k, v in config.items() if k != "type"})


class IntervalCount(BaseCheck):
    """Pass when enough intervals have irradiance at or above a threshold.

    Parameters
    ----------
    threshold : float
        Irradiance threshold; an interval qualifies at or above it.
    min_count : int
        Minimum number of qualifying intervals.
    """

    threshold = param.Number(default=None, allow_None=True)
    min_count = param.Integer(default=None, allow_None=True)

    def _validate(self):
        super()._validate()
        if self.threshold is None or self.min_count is None:
            raise ValueError("IntervalCount requires threshold and min_count.")
        _check_finite("threshold", self.threshold)
        _check_non_negative("min_count", self.min_count)

    def _evaluate(self, frame, irr, context):
        count = qualifying_count(irr, self.threshold)
        detail = pd.DataFrame(
            {
                "count": [count],
                "threshold": [self.threshold],
                "min_count": [self.min_count],
            }
        )
        return CheckResult(
            name=self.label,
            passed=count >= self.min_count,
            detail=detail,
            summary=(
                f"{count} intervals at or above {self.threshold:g} W/m² "
                f"(need {self.min_count})"
            ),
        )


class Duration(BaseCheck):
    """Pass when enough hours have irradiance at or above a threshold.

    Hours are the number of qualifying intervals times the timestep; the
    intervals need not be continuous.

    Parameters
    ----------
    threshold : float
        Irradiance threshold; an interval qualifies at or above it.
    min_hours : float
        Minimum total qualifying hours.
    """

    threshold = param.Number(default=None, allow_None=True)
    min_hours = param.Number(default=None, allow_None=True)

    def _validate(self):
        super()._validate()
        if self.threshold is None or self.min_hours is None:
            raise ValueError("Duration requires threshold and min_hours.")
        _check_finite("threshold", self.threshold)
        _check_non_negative("min_hours", self.min_hours)

    def _evaluate(self, frame, irr, context):
        count = qualifying_count(irr, self.threshold)
        hours = count * context.dt_hours
        detail = pd.DataFrame(
            {
                "hours": [hours],
                "threshold": [self.threshold],
                "min_hours": [self.min_hours],
            }
        )
        return CheckResult(
            name=self.label,
            passed=hours >= self.min_hours,
            detail=detail,
            summary=(
                f"{hours:g} h at or above {self.threshold:g} W/m² "
                f"(need {self.min_hours:g})"
            ),
        )


CHECK_REGISTRY = {
    "IntervalCount": IntervalCount,
    "Duration": Duration,
}


def check_from_config(d):
    """Build a check from a ``to_config()`` dict via ``CHECK_REGISTRY``.

    Raises
    ------
    ValueError
        If ``type`` is not registered; the message suggests the closest name.
    """
    d = dict(d)
    if "type" not in d:
        raise ValueError("check config is missing the 'type' key")
    cls_name = d.pop("type")
    if cls_name not in CHECK_REGISTRY:
        suggestion = difflib.get_close_matches(cls_name, CHECK_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown check type {cls_name!r} in config.{hint}")
    return CHECK_REGISTRY[cls_name].from_config(d)
