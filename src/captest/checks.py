"""Serializable validity checks for a performance-ratio (PR) test.

A check is a ``param``-based step, modeled on the filter steps in
:mod:`captest.filters`, except that it never removes rows: it reports whether
the data satisfies a requirement (for example "at least 30 hours with
irradiance at or above 400 W/m2") together with a small detail table.

This module never imports :mod:`captest.capdata`; a check touches a
``CapData`` only through the runtime argument (``data``, ``data_filtered`` and
the flat ``regression_cols`` mapping of role to column).
"""

import copy
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
RUN_HOURS_TOLERANCE = 1e-9


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


def qualifying_mask(irr, threshold):
    """Flag intervals with finite irradiance at or above ``threshold``.

    Non-finite values (NaN, +/-inf) never qualify.
    """
    return np.isfinite(irr) & (irr >= threshold)


def qualifying_count(irr, threshold):
    """Count intervals with finite irradiance at or above ``threshold``.

    Non-finite values (NaN, +/-inf) never qualify.
    """
    return int(qualifying_mask(irr, threshold).sum())


def daily_longest_runs(irr, threshold, dt_hours):
    """Longest continuous qualifying run per local calendar day.

    A run is consecutive rows of the same local day (``index.normalize()`` in
    the index's own timezone) that all qualify (finite and at or above
    ``threshold``) and whose timestamps are exactly ``dt_hours`` apart. A
    non-qualifying row, a missing timestamp or a duplicate timestamp (zero gap)
    ends the run.

    Parameters
    ----------
    irr : pd.Series
        Irradiance with a datetime index.
    threshold : float
        Irradiance threshold; a row qualifies at or above it.
    dt_hours : float
        Timestep in hours.

    Returns
    -------
    pd.Series
        Longest run in hours (run rows times ``dt_hours``) for each day present
        in ``irr``, indexed by the day's midnight timestamp.
    """
    irr = irr.sort_index()
    day = irr.index.normalize()
    qualifies = qualifying_mask(irr, threshold).to_numpy()
    gap_seconds = irr.index.to_series().diff().dt.total_seconds().to_numpy()
    contiguous = np.zeros(len(irr), dtype=bool)
    contiguous[1:] = (
        qualifies[:-1]
        & (day[1:] == day[:-1])
        & np.isclose(gap_seconds[1:], dt_hours * 3600.0, rtol=0, atol=1e-6)
    )
    start = qualifies & ~contiguous
    run_id = np.cumsum(start)
    rows = pd.Series(qualifies, index=day).groupby([day, run_id]).sum()
    longest = rows.groupby(level=0).max() * dt_hours
    return longest.rename("longest_run_hours")


def run_meets_hours(run_hours, hours):
    """Whether run lengths are at or above ``hours``, tolerant of float error.

    Run lengths are rows times a float timestep, so an exact-length run can fall
    a few ulps short of ``hours``; a tolerance of 1e-9 h absorbs that.
    """
    return run_hours >= hours - RUN_HOURS_TOLERANCE


def daily_insolation(irr, dt_hours):
    """Insolation in Wh/m2 per local calendar day, non-finite values as 0."""
    energy = irr.where(np.isfinite(irr), 0.0) * dt_hours
    return energy.groupby(irr.index.normalize()).sum().rename("insolation_wh_m2")


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


class ContinuousHoursDays(BaseCheck):
    """Pass when enough days hold a long continuous run at or above a threshold.

    Days are the local calendar dates (``index.normalize()``, in the index's
    own timezone) that hold at least one row of the selected data; a day with
    no rows at all does not appear in the detail. A run is consecutive rows,
    exactly one timestep apart, each with finite irradiance at or above
    ``threshold``; its length is rows times the timestep. A day qualifies when
    its longest run is at or above ``hours``. Reads the unfiltered data inside
    the test period by default.

    Parameters
    ----------
    threshold : float
        Irradiance threshold; a row qualifies at or above it.
    hours : float
        Minimum run length in hours; a run qualifies at or above it.
    min_days : int
        Minimum number of qualifying days.

    Notes
    -----
    The detail has one row per day with ``longest_run_hours`` and
    ``qualifies``.
    """

    data = param.Selector(default="unfiltered_period")
    threshold = param.Number(default=None, allow_None=True)
    hours = param.Number(default=None, allow_None=True)
    min_days = param.Integer(default=None, allow_None=True)

    def _validate(self):
        super()._validate()
        if None in (self.threshold, self.hours, self.min_days):
            raise ValueError("ContinuousHoursDays requires threshold, hours, min_days.")
        _check_finite("threshold", self.threshold)
        _check_non_negative("hours", self.hours)
        _check_non_negative("min_days", self.min_days)

    def _evaluate(self, frame, irr, context):
        longest = daily_longest_runs(irr, self.threshold, context.dt_hours)
        detail = longest.to_frame()
        detail["qualifies"] = run_meets_hours(detail["longest_run_hours"], self.hours)
        detail.index.name = "day"
        days = int(detail["qualifies"].sum())
        return CheckResult(
            name=self.label,
            passed=days >= self.min_days,
            detail=detail,
            summary=(
                f"{days} days with a run of at least {self.hours:g} h at or above "
                f"{self.threshold:g} W/m² (need {self.min_days})"
            ),
        )


class DailyInsolationDays(BaseCheck):
    """Pass when enough days reach a daily insolation.

    Daily insolation is the sum of irradiance times the timestep over the
    local calendar day (``index.normalize()``); non-finite irradiance (NaN,
    +/-inf) contributes 0. Days are those holding at least one row of the
    selected data. A day qualifies when its insolation is at or above
    ``threshold_wh_m2``. Reads the unfiltered data inside the test period by
    default.

    Parameters
    ----------
    threshold_wh_m2 : float
        Minimum daily insolation in Wh/m2; a day qualifies at or above it.
    min_days : int
        Minimum number of qualifying days.

    Notes
    -----
    The detail has one row per day with ``insolation_wh_m2`` and
    ``qualifies``.
    """

    data = param.Selector(default="unfiltered_period")
    threshold_wh_m2 = param.Number(default=None, allow_None=True)
    min_days = param.Integer(default=None, allow_None=True)

    def _validate(self):
        super()._validate()
        if self.threshold_wh_m2 is None or self.min_days is None:
            raise ValueError("DailyInsolationDays requires threshold_wh_m2, min_days.")
        _check_finite("threshold_wh_m2", self.threshold_wh_m2)
        _check_non_negative("min_days", self.min_days)

    def _evaluate(self, frame, irr, context):
        detail = daily_insolation(irr, context.dt_hours).to_frame()
        detail["qualifies"] = detail["insolation_wh_m2"] >= self.threshold_wh_m2
        detail.index.name = "day"
        days = int(detail["qualifies"].sum())
        return CheckResult(
            name=self.label,
            passed=days >= self.min_days,
            detail=detail,
            summary=(
                f"{days} days with insolation at or above "
                f"{self.threshold_wh_m2:g} Wh/m² (need {self.min_days})"
            ),
        )


class NrelMinimumIrradiance(BaseCheck):
    """NREL minimum-irradiance criteria for a PR test period.

    Passes when at least ``min_insolation_days`` days have insolation at or
    above ``insolation_wh_m2`` and, among those days, at least
    ``min_irr_days`` hold a continuous run of at least ``hours`` at or above
    ``irr_threshold``. Days are local calendar dates holding at least one row
    of the selected data; run and insolation rules are those of
    `ContinuousHoursDays` and `DailyInsolationDays`. Reads the unfiltered data
    inside the test period by default.

    This is pvcaptest's count reading of an internally inconsistent passage
    in the source paper; it equals the decided reading on a period of exactly
    ``min_insolation_days`` (5) days, where every day must reach the insolation
    and ``min_irr_days`` of them the run.

    Parameters
    ----------
    irr_threshold : float, default 600
        Irradiance threshold in W/m2 for the continuous run.
    hours : float, default 3
        Minimum run length in hours.
    insolation_wh_m2 : float, default 3000
        Minimum daily insolation in Wh/m2.
    min_insolation_days : int, default 5
        Minimum number of days meeting the insolation.
    min_irr_days : int, default 3
        Minimum number of those days that also hold the run.

    Notes
    -----
    The detail has one row per day with ``insolation_wh_m2``,
    ``longest_run_hours``, ``insolation_ok`` and ``irradiance_ok``;
    ``irradiance_ok`` is only true on days with ``insolation_ok``.
    """

    data = param.Selector(default="unfiltered_period")
    irr_threshold = param.Number(default=600)
    hours = param.Number(default=3)
    insolation_wh_m2 = param.Number(default=3000)
    min_insolation_days = param.Integer(default=5)
    min_irr_days = param.Integer(default=3)

    def _validate(self):
        super()._validate()
        _check_finite("irr_threshold", self.irr_threshold)
        _check_non_negative("hours", self.hours)
        _check_finite("insolation_wh_m2", self.insolation_wh_m2)
        _check_non_negative("min_insolation_days", self.min_insolation_days)
        _check_non_negative("min_irr_days", self.min_irr_days)
        if self.min_irr_days > self.min_insolation_days:
            raise ValueError(
                f"min_irr_days ({self.min_irr_days}) cannot exceed "
                f"min_insolation_days ({self.min_insolation_days}); the run days "
                "are counted among the insolation days."
            )

    def _evaluate(self, frame, irr, context):
        detail = daily_insolation(irr, context.dt_hours).to_frame()
        detail["longest_run_hours"] = daily_longest_runs(
            irr, self.irr_threshold, context.dt_hours
        )
        detail["insolation_ok"] = detail["insolation_wh_m2"] >= self.insolation_wh_m2
        detail["irradiance_ok"] = detail["insolation_ok"] & (
            run_meets_hours(detail["longest_run_hours"], self.hours)
        )
        detail.index.name = "day"
        insolation_days = int(detail["insolation_ok"].sum())
        irradiance_days = int(detail["irradiance_ok"].sum())
        return CheckResult(
            name=self.label,
            passed=(
                insolation_days >= self.min_insolation_days
                and irradiance_days >= self.min_irr_days
            ),
            detail=detail,
            summary=(
                f"{insolation_days} days with insolation at or above "
                f"{self.insolation_wh_m2:g} Wh/m² (need {self.min_insolation_days}); "
                f"{irradiance_days} of them with a {self.hours:g} h run at or above "
                f"{self.irr_threshold:g} W/m² (need {self.min_irr_days})"
            ),
        )


class AnyOf(BaseCheck):
    """Pass when any member check passes.

    Expresses tiered contract schemes such as "at least 60 intervals at
    400 W/m2 or at least 40 at 600 W/m2". Members are owned by the combinator
    and evaluated against the same dataset and context; each applies its own
    row selection. The ``irr``, ``data``, ``start`` and ``end`` parameters
    inherited from `BaseCheck` are not used and are never serialized.

    Parameters
    ----------
    *checks : BaseCheck
        One or more member checks; may include other `AnyOf` instances.
    custom_name : str or None, default None
        Display label; ``"AnyOf"`` when ``None``.

    Raises
    ------
    ValueError
        If no member is given.
    TypeError
        If a member is not a `BaseCheck`.

    Notes
    -----
    The detail has one row per member with ``member`` (its label), ``passed``
    and ``summary``.
    """

    checks = param.List(default=[], item_type=BaseCheck)

    def __init__(self, *checks, custom_name=None):
        for member in checks:
            if not isinstance(member, BaseCheck):
                raise TypeError(
                    f"AnyOf members must be BaseCheck instances, got {member!r}."
                )
        super().__init__(checks=list(checks), custom_name=custom_name)

    def _validate(self):
        super()._validate()
        if not self.checks:
            raise ValueError("AnyOf requires at least one member check.")

    def evaluate(self, capdata, context):
        """Evaluate every member and pass when any member passes.

        Parameters
        ----------
        capdata : CapData
            Dataset to check.
        context : CheckContext
            Run-time facts passed unchanged to each member.

        Returns
        -------
        CheckResult
        """
        self._validate()
        results = [member.evaluate(capdata, context) for member in self.checks]
        detail = pd.DataFrame(
            {
                "member": [result.name for result in results],
                "passed": [bool(result.passed) for result in results],
                "summary": [result.summary for result in results],
            }
        )
        passed = any(result.passed for result in results)
        satisfied = int(detail["passed"].sum())
        return CheckResult(
            name=self.label,
            passed=passed,
            detail=detail,
            summary=f"{satisfied} of {len(results)} member checks passed",
        )

    def to_config(self):
        """Serialize to ``{"type", "checks"}`` plus ``custom_name`` when set."""
        config = {
            "type": "AnyOf",
            "checks": [member.to_config() for member in self.checks],
        }
        if self.custom_name is not None:
            config["custom_name"] = self.custom_name
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance, rebuilding members with `check_from_config`."""
        members = [check_from_config(member) for member in config["checks"]]
        return cls(*members, custom_name=config.get("custom_name"))


class CustomCheck(BaseCheck):
    """Run a user callable on the selected rows and report its verdict.

    ``func(frame, *args, **kwargs)`` receives the rows chosen by ``data`` and
    returns either a bool (``bool`` or ``numpy.bool_``) or a
    ``(bool, DataFrame)`` tuple whose frame becomes the detail. With a bare
    bool the detail is an empty DataFrame. Any other return raises
    ``TypeError`` naming the check. An exception raised by ``func`` propagates
    unchanged; on Python 3.11+ a note naming the check is added to it. A
    selection with no rows fails without calling ``func``. Irradiance is not
    resolved, so ``irr`` is ignored and the data need not hold a POA column.

    Config files name ``func`` by ``"module:qualname"``; loading one imports
    that module and later calls the callable, so load configs only from
    trusted sources. Lambdas and locally defined functions cannot be
    serialized.

    Parameters
    ----------
    func : callable
        Check function taking a DataFrame first.
    *args
        Extra positional arguments for ``func``; must be yaml-native to
        serialize.
    custom_name : str or None, default None
        Display label; ``"CustomCheck"`` when ``None``.
    data : {"pr_rows", "filtered", "unfiltered_period"}, default "filtered"
        Which rows ``func`` receives.
    **kwargs
        Extra keyword arguments for ``func``; must be yaml-native to
        serialize.
    """

    func = param.Callable(default=None, allow_None=True)
    args = param.List(default=[])
    kwargs = param.Dict(default={})
    data = param.Selector(default="filtered", objects=DATA_MODES)

    def __init__(self, func, *args, custom_name=None, data="filtered", **kwargs):
        super().__init__(
            func=func,
            args=list(args),
            kwargs=dict(kwargs),
            custom_name=custom_name,
            data=data,
        )

    def _validate(self):
        super()._validate()
        if self.func is None:
            raise ValueError("CustomCheck requires a callable func.")

    def evaluate(self, capdata, context):
        """Select rows and call ``func`` on them.

        Parameters
        ----------
        capdata : CapData
            Dataset to check.
        context : CheckContext
            Supplies ``pr_rows`` and ``period`` for row selection.

        Returns
        -------
        CheckResult

        Raises
        ------
        TypeError
            If ``func`` returns anything but a bool or ``(bool, DataFrame)``.
        """
        self._validate()
        frame = select_rows(capdata, self, context)
        if len(frame) == 0:
            return self._empty_result()
        try:
            outcome = self.func(frame, *self.args, **self.kwargs)
        except Exception as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"raised by custom check {self.label!r}")
            raise
        detail = pd.DataFrame()
        if (
            isinstance(outcome, tuple)
            and len(outcome) == 2
            and isinstance(outcome[0], (bool, np.bool_))
            and isinstance(outcome[1], pd.DataFrame)
        ):
            outcome, detail = outcome
        if not isinstance(outcome, (bool, np.bool_)):
            raise TypeError(
                f"Custom check {self.label!r} must return a bool or a "
                f"(bool, DataFrame) tuple, got {type(outcome).__name__}."
            )
        passed = bool(outcome)
        return CheckResult(
            name=self.label,
            passed=passed,
            detail=detail,
            summary=f"custom check {'passed' if passed else 'failed'}",
        )

    def to_config(self):
        """Serialize with ``func`` as ``"module:qualname"``.

        Raises
        ------
        ValueError
            If ``func`` is a lambda or locally defined function.
        """
        config = {
            "type": "CustomCheck",
            "func": util.callable_to_qualname(self.func),
            "args": util.to_native(copy.deepcopy(self.args)),
            "kwargs": util.to_native(copy.deepcopy(self.kwargs)),
            "custom_name": self.custom_name,
        }
        if self.data != type(self).param["data"].default:
            config["data"] = self.data
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance, importing ``func`` from its qualified name."""
        func = util.callable_from_qualname(config["func"])
        return cls(
            func,
            *(config.get("args") or []),
            custom_name=config.get("custom_name"),
            data=config.get("data", "filtered"),
            **(config.get("kwargs") or {}),
        )


CHECK_REGISTRY = {
    "IntervalCount": IntervalCount,
    "Duration": Duration,
    "ContinuousHoursDays": ContinuousHoursDays,
    "DailyInsolationDays": DailyInsolationDays,
    "NrelMinimumIrradiance": NrelMinimumIrradiance,
    "AnyOf": AnyOf,
    "CustomCheck": CustomCheck,
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
