"""
Performance ratio (PR) test calculations.

Pure functions implementing the energy-based performance ratio of NREL/TP-5200-57991
(Dierauf et al., "Weather-Corrected Performance Ratio"): unit conversion, expected
energy from plane-of-array irradiance with an optional temperature correction, and
input validators. Measured and expected energy are always handled in Wh internally.
"""

import calendar
import math
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from captest import calcparams
from captest.util import common_timestep_delta

G_STC = 1000.0
"""Plane-of-array irradiance at standard test conditions, W/m²."""

POWER_UNITS = {"W": 1.0, "kW": 1e3, "MW": 1e6}
"""Power units accepted by the unit helpers, as multipliers to W."""

ENERGY_UNITS = {"Wh": 1.0, "kWh": 1e3, "MWh": 1e6}
"""Energy units accepted by the unit helpers, as multipliers to Wh."""

_COEFF_TYPICAL_RANGE = (0.05, 1.0)


def _require_positive_finite(name, value):
    """Raise ValueError unless `value` is a finite number greater than zero."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite number greater than 0; got {value}.")


def _require_same_index(reference_name, reference, **others):
    """Raise ValueError when a Series in `others` is not indexed like `reference`."""
    for name, other in others.items():
        if isinstance(other, pd.Series) and not other.index.equals(reference.index):
            raise ValueError(
                f"{name} must have the same index as {reference_name}; the indexes "
                "differ, so the values would not line up row by row."
            )


def _unit_factor(units, table, label):
    """Return the multiplier for `units` from `table` or raise listing the options."""
    if units not in table:
        raise ValueError(
            f"Unknown {label} unit {units!r}; allowed units are {', '.join(table)}."
        )
    return table[units]


def dt_hours(index):
    """
    Return the dominant timestep of an index in hours.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Unfiltered index of a measured or modeled dataset.

    Returns
    -------
    float
        Timestep in hours.
    """
    return common_timestep_delta(index) / pd.Timedelta(hours=1)


_index_dt_hours = dt_hours


def to_energy_wh(values, *, kind, units, dt_hours):
    """
    Convert interval power or interval energy to interval energy in Wh.

    Parameters
    ----------
    values : pd.Series
        Interval power (average over the interval) or interval energy.
    kind : {"power", "energy"}
        What `values` holds.
    units : str
        One of `POWER_UNITS` when `kind` is "power", else one of `ENERGY_UNITS`.
    dt_hours : float
        Length of each interval in hours; must be finite and greater than 0.

    Returns
    -------
    pd.Series
        Energy per interval in Wh.

    Raises
    ------
    ValueError
        If `kind` or `units` is not recognized, the units do not match `kind`,
        or `dt_hours` is not a positive finite number.
    """
    _require_positive_finite("dt_hours", dt_hours)
    if kind == "power":
        return values * _unit_factor(units, POWER_UNITS, "power") * dt_hours
    if kind == "energy":
        return values * _unit_factor(units, ENERGY_UNITS, "energy")
    raise ValueError(f"kind must be 'power' or 'energy'; got {kind!r}.")


def power_from_energy(energy, *, energy_units, power_units, dt_hours):
    """
    Convert interval energy to average interval power.

    Parameters
    ----------
    energy : pd.Series
        Energy per interval.
    energy_units : str
        One of `ENERGY_UNITS`.
    power_units : str
        One of `POWER_UNITS`; units of the result.
    dt_hours : float
        Length of each interval in hours; must be finite and greater than 0.

    Returns
    -------
    pd.Series
        Average power over each interval in `power_units`.

    Raises
    ------
    ValueError
        If a unit is not recognized or `dt_hours` is not a positive finite number.
    """
    _require_positive_finite("dt_hours", dt_hours)
    energy_wh = energy * _unit_factor(energy_units, ENERGY_UNITS, "energy")
    return energy_wh / dt_hours / _unit_factor(power_units, POWER_UNITS, "power")


def validate_power_temp_coeff(power_temp_coeff):
    """
    Check a power temperature coefficient given in %/°C.

    Parameters
    ----------
    power_temp_coeff : float
        Power temperature coefficient in %/°C; must be finite and negative.

    Raises
    ------
    ValueError
        If the value is not finite or is not negative.

    Warns
    -----
    UserWarning
        If the magnitude is outside the range typical of %/°C values, which
        usually means the coefficient was given as a fraction or in the wrong units.
    """
    if not math.isfinite(power_temp_coeff):
        raise ValueError(
            f"power_temp_coeff must be a finite number; got {power_temp_coeff}."
        )
    if power_temp_coeff >= 0:
        raise ValueError(
            "power_temp_coeff must be negative (%/°C). pvcaptest follows NREL "
            "eq. 8, 1 − δ·(Tcell_typ_avg − T_cell), which needs the negative sign; "
            "a contract that states β unsigned, e.g. 1 + β·(Tref − Tmeas) with "
            "β = 0.32, is the same equation with power_temp_coeff=-0.32."
        )
    low, high = _COEFF_TYPICAL_RANGE
    if not low <= abs(power_temp_coeff) <= high:
        warnings.warn(
            f"power_temp_coeff={power_temp_coeff} is outside the typical "
            f"{low}-{high} %/°C magnitude; it is interpreted in %/°C, not as a "
            "fraction per °C.",
            UserWarning,
            stacklevel=2,
        )


def validate_availability(availability):
    """
    Check an availability factor lies in [0, 1].

    Parameters
    ----------
    availability : float or pd.Series
        Scalar or per-interval availability. NaN entries of a Series are allowed
        because the row set drops them.

    Raises
    ------
    ValueError
        If a scalar is not finite or any value is outside 0 ≤ A ≤ 1.
    """
    if isinstance(availability, pd.Series):
        if ((availability < 0) | (availability > 1)).any():
            raise ValueError("availability must satisfy 0 ≤ A ≤ 1 in every row.")
    elif not (math.isfinite(availability) and 0 <= availability <= 1):
        raise ValueError(f"availability must satisfy 0 ≤ A ≤ 1; got {availability}.")


def expected_energy_wh(
    poa,
    *,
    dc_nameplate,
    dt_hours,
    cell_temp=None,
    t_ref=None,
    power_temp_coeff=None,
    availability=1.0,
):
    """
    Compute expected interval energy in Wh from plane-of-array irradiance.

    ``E = P_nameplate · (POA / G_STC) · Δt · A``, multiplied by
    ``1 + γ·(T_cell − T_ref)`` (γ = ``power_temp_coeff`` / 100) when a
    temperature correction is requested.

    Parameters
    ----------
    poa : pd.Series
        Plane-of-array irradiance in W/m².
    dc_nameplate : float
        DC nameplate capacity in W; must be finite and greater than 0.
    dt_hours : float
        Length of each interval in hours; must be finite and greater than 0.
    cell_temp : pd.Series, optional
        Cell temperature in °C. Supplying it selects the temperature-corrected
        form, which also requires `t_ref` and `power_temp_coeff`.
    t_ref : float or pd.Series, optional
        Reference cell temperature in °C, scalar or aligned to `poa`.
    power_temp_coeff : float, optional
        Power temperature coefficient in %/°C; negative.
    availability : float or pd.Series, default 1.0
        Availability factor, 0 ≤ A ≤ 1, scalar or aligned to `poa`.

    Returns
    -------
    pd.Series
        Expected energy per interval in Wh.

    Raises
    ------
    ValueError
        If `dc_nameplate` or `dt_hours` is not a positive finite number, the
        availability or coefficient is invalid, `t_ref` or `power_temp_coeff` is
        given without `cell_temp`, or `cell_temp` is given without both, or a
        Series `cell_temp`, `t_ref` or `availability` is not indexed like `poa`.
    """
    _require_positive_finite("dc_nameplate", dc_nameplate)
    _require_positive_finite("dt_hours", dt_hours)
    validate_availability(availability)
    _require_same_index(
        "poa", poa, cell_temp=cell_temp, t_ref=t_ref, availability=availability
    )
    energy = dc_nameplate * (poa / G_STC) * dt_hours * availability
    if cell_temp is None:
        if t_ref is not None or power_temp_coeff is not None:
            raise ValueError(
                "t_ref and power_temp_coeff were given without cell_temp; "
                "a plain test takes none of them."
            )
        return energy
    if t_ref is None:
        raise ValueError("t_ref is required when cell_temp is given.")
    if power_temp_coeff is None:
        raise ValueError("power_temp_coeff is required when cell_temp is given.")
    validate_power_temp_coeff(power_temp_coeff)
    return energy * (1 + power_temp_coeff / 100 * (cell_temp - t_ref))


@dataclass
class RowSet:
    """
    Rows usable by a PR calculation and a count of those dropped.

    Attributes
    ----------
    index : pd.Index
        Index of rows where every input column is finite.
    dropped_nan : int
        Rows dropped because at least one input was NaN or missing.
    dropped_nonfinite : int
        Rows dropped because at least one input was +/-inf and none was NaN.
    """

    index: pd.Index
    dropped_nan: int
    dropped_nonfinite: int


def complete_rows(frame, *, label=""):
    """
    Select the rows where every column of `frame` is finite.

    Parameters
    ----------
    frame : pd.DataFrame
        Numeric input columns of one calculation.
    label : str, default ""
        Prefix for the warning, e.g. ``"meas"``.

    Returns
    -------
    RowSet
        Kept index plus separate counts of NaN and non-finite drops. A row with
        both NaN and inf counts as NaN.

    Warns
    -----
    UserWarning
        If any row is dropped; names both counts.
    """
    values = frame.to_numpy(dtype=float)
    has_nan = np.isnan(values).any(axis=1)
    has_inf = np.isinf(values).any(axis=1) & ~has_nan
    dropped_nan = int(has_nan.sum())
    dropped_nonfinite = int(has_inf.sum())
    if dropped_nan or dropped_nonfinite:
        prefix = f"{label}: " if label else ""
        warnings.warn(
            f"{prefix}dropped {dropped_nan} row(s) with NaN or missing inputs and "
            f"{dropped_nonfinite} row(s) with non-finite (inf) inputs.",
            UserWarning,
            stacklevel=2,
        )
    return RowSet(frame.index[~(has_nan | has_inf)], dropped_nan, dropped_nonfinite)


def weighted_cell_temp(poa, cell_temp):
    """
    Return the irradiance-weighted average cell temperature (NREL eq. 5).

    The inputs are expected to be the caller's `complete_rows` set; rows where
    either value is NaN are otherwise skipped by the sums.

    Parameters
    ----------
    poa : pd.Series
        Plane-of-array irradiance, used as the weight.
    cell_temp : pd.Series
        Cell temperature in °C, with the same index as `poa`.

    Returns
    -------
    float
        ``sum(poa * cell_temp) / sum(poa)`` in °C.

    Raises
    ------
    ValueError
        If `cell_temp` is not indexed like `poa`, no row has both values finite,
        or the total irradiance is not greater than zero.
    """
    _require_same_index("poa", poa, cell_temp=cell_temp)
    if not (np.isfinite(poa) & np.isfinite(cell_temp)).any():
        raise ValueError("no row has both poa and cell_temp finite.")
    total = poa.sum()
    if not total > 0:
        raise ValueError("total irradiance must be greater than 0 to weight Tcell.")
    return float((poa * cell_temp).sum() / total)


def pr_ratio(measured_wh, expected_wh):
    """
    Compute the performance ratio as total measured over total expected energy.

    Parameters
    ----------
    measured_wh : pd.Series
        Measured interval energy in Wh.
    expected_wh : pd.Series
        Expected interval energy in Wh, with the same index as `measured_wh`.

    Returns
    -------
    tuple of (float, pd.Series)
        The ratio of the totals and the per-interval ratio, which is NaN where
        the expected energy is zero.

    Raises
    ------
    ValueError
        If `expected_wh` is not indexed like `measured_wh`, or the total expected
        energy is not finite and greater than zero.
    """
    _require_same_index("measured_wh", measured_wh, expected_wh=expected_wh)
    total_expected = expected_wh.sum()
    if not math.isfinite(total_expected):
        raise ValueError(
            f"total expected energy is not finite ({total_expected}); the PR is "
            "undefined."
        )
    if total_expected <= 0:
        problem = "zero" if total_expected == 0 else "not positive"
        raise ValueError(f"total expected energy is {problem}; the PR is undefined.")
    interval = measured_wh / expected_wh.where(expected_wh != 0)
    return float(measured_wh.sum() / total_expected), interval


def resample_interval_energy(energy, freq, *, label="left", closed="left"):
    """
    Sum interval energy into coarser bins, voiding any bin that is not complete.

    Bins are aligned to midnight of the first day, and the grid check assumes
    left-anchored bins counted from ``index[0].normalize()``. A bin with a missing
    or NaN interval (including a register reset) or with fewer native intervals
    than it spans (an incomplete first or last bin) is NaN rather than a partial
    sum.

    Parameters
    ----------
    energy : pd.Series
        Interval energy on a regular grid, e.g. from
        `captest.calcparams.register_to_interval_energy`.
    freq : str
        Bin width, a whole multiple of the native timestep, e.g. ``"15min"``.
    label : {"left", "right"}, default "left"
        Which bin edge labels the result.
    closed : {"left", "right"}, default "left"
        Which bin edge is inclusive.

    Returns
    -------
    pd.Series
        Energy per bin, NaN for incomplete bins.

    Raises
    ------
    ValueError
        If `energy` is empty, `label` or `closed` is not "left" or "right", the
        index has duplicate timestamps or timestamps off the grid counted from
        midnight of the first day, or `freq` is not a fixed frequency that is a
        whole multiple of the native timestep.
    """
    for name, value in (("label", label), ("closed", closed)):
        if value not in ("left", "right"):
            raise ValueError(f"{name} must be 'left' or 'right'; got {value!r}.")
    if energy.empty:
        raise ValueError("energy is empty; there is nothing to resample.")
    if energy.index.has_duplicates:
        raise ValueError("energy index has duplicate timestamps.")
    origin = energy.index[0].normalize()
    native = common_timestep_delta(energy.index)
    if ((energy.index - origin) % native != pd.Timedelta(0)).any():
        raise ValueError(
            f"energy timestamps are not aligned to a {native} grid counted from "
            f"{origin}; resampling would mislabel the bins."
        )
    try:
        freq_delta = pd.Timedelta(freq)
    except ValueError as err:
        raise ValueError(
            f"freq {freq!r} cannot be converted to a fixed duration; a fixed "
            "frequency such as '15min' or '1h' is required."
        ) from err
    n_float = freq_delta / native
    n = round(n_float)
    if n < 1 or not math.isclose(n_float, n):
        raise ValueError(
            f"freq {freq!r} is not a whole multiple of the native timestep {native}."
        )
    return energy.resample(freq, label=label, closed=closed, origin=origin).sum(
        min_count=n
    )


def degradation_factor(rate, years):
    """
    Return the linear degradation factor ``1 − years · rate / 100``.

    Parameters
    ----------
    rate : float
        Annual degradation rate in %/yr; finite and not negative.
    years : float
        Elapsed time in years; finite and not negative.

    Returns
    -------
    float
        Factor applied to the expected performance ratio.

    Raises
    ------
    ValueError
        If `rate` or `years` is negative or not finite, or the resulting factor
        is not greater than zero.
    """
    for name, value in (("rate", rate), ("years", years)):
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(
                f"{name} must be a finite number not less than 0; got {value}."
            )
    factor = 1 - years * rate / 100
    if factor <= 0:
        raise ValueError(
            f"Degradation factor must be greater than 0; rate={rate} %/yr over "
            f"years={years} gives {factor}."
        )
    return factor


def degradation_years(warranty_start, test_start, test_end):
    """
    Return the years of degradation at the midpoint of a test period.

    Computed as ``((test_start − warranty_start).days +
    (test_end − warranty_start).days) / 2 / 365``.

    Parameters
    ----------
    warranty_start, test_start, test_end : timestamp-like
        Anything `pd.Timestamp` accepts.

    Returns
    -------
    float
        Years from the start of the warranty to the middle of the test period.

    Raises
    ------
    ValueError
        If `test_end` is before `test_start` or `test_start` is before
        `warranty_start`.
    """
    warranty_start = pd.Timestamp(warranty_start)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    if test_end < test_start:
        raise ValueError(
            f"test_end ({test_end}) must not be before test_start ({test_start})."
        )
    if test_start < warranty_start:
        raise ValueError(
            f"test_start ({test_start}) must not be before warranty_start "
            f"({warranty_start})."
        )
    days = (test_start - warranty_start).days + (test_end - warranty_start).days
    return days / 2 / 365


_TOLERANCE_PATTERN = re.compile(r"^- (\d+(?:\.\d+)?)$")


def parse_pr_tolerance(tolerance):
    """
    Parse a one-sided PR test tolerance string into a fraction.

    Parameters
    ----------
    tolerance : str
        ``"- N"`` with ``0 <= N < 100``, e.g. ``"- 3"`` for a 3 % shortfall.

    Returns
    -------
    float
        The tolerance as a fraction, e.g. 0.03.

    Raises
    ------
    ValueError
        If the string is two-sided (``"+/- N"``), malformed, or N is outside
        ``[0, 100)``.
    """
    text = tolerance.strip() if isinstance(tolerance, str) else tolerance
    if isinstance(text, str) and text.startswith("+/-"):
        raise ValueError(
            f"test_tolerance {tolerance!r} is two-sided; a PR test is one-sided, "
            "use the form '- N', e.g. '- 3'."
        )
    match = _TOLERANCE_PATTERN.match(text) if isinstance(text, str) else None
    if match is None or not float(match.group(1)) < 100:
        raise ValueError(
            f"test_tolerance must have the form '- N' with 0 <= N < 100, "
            f"e.g. '- 3'; got {tolerance!r}."
        )
    return float(match.group(1)) / 100


@dataclass
class Judgement:
    """
    Outcome of comparing a measured PR to its tolerance-adjusted threshold.

    Attributes
    ----------
    threshold : float
        ``(1 − tolerance) · expected_ratio``.
    margin : float
        ``ratio − threshold``.
    ratio_to_threshold : float
        ``ratio / threshold``.
    passed : bool
        True when ``ratio >= threshold``.
    """

    threshold: float
    margin: float
    ratio_to_threshold: float
    passed: bool


def judge(ratio, expected_ratio, tolerance):
    """
    Judge a measured performance ratio against the expected ratio and tolerance.

    Parameters
    ----------
    ratio : float
        Measured performance ratio.
    expected_ratio : float
        Expected performance ratio, including any degradation factor.
    tolerance : float
        One-sided tolerance as a fraction, e.g. from `parse_pr_tolerance`.

    Returns
    -------
    Judgement
        Threshold, margin, ratio to threshold and the pass/fail result.

    Raises
    ------
    ValueError
        If `ratio` is not finite, `expected_ratio` is not finite and greater
        than zero, or `tolerance` is not finite or outside ``0 <= tolerance < 1``.
        A non-finite input never yields a pass.
    """
    if not math.isfinite(ratio):
        raise ValueError(f"ratio must be a finite number; got {ratio}.")
    _require_positive_finite("expected_ratio", expected_ratio)
    if not (math.isfinite(tolerance) and 0 <= tolerance < 1):
        raise ValueError(f"tolerance must satisfy 0 <= tolerance < 1; got {tolerance}.")
    threshold = (1 - tolerance) * expected_ratio
    return Judgement(
        threshold=threshold,
        margin=ratio - threshold,
        ratio_to_threshold=ratio / threshold,
        passed=bool(ratio >= threshold),
    )


_MONTHS = range(1, 13)
_PARTIAL_MONTH_RUN_DAYS = 7


def _month_name(month):
    return calendar.month_name[month]


def _check_pr_value(name, value):
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite number greater than 0; got {value}.")


def _check_finite(name, value):
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}.")


def _monthly_table(name, table):
    """Return {month: value} for the stated entries of a 12-entry table."""
    if len(table) != 12:
        raise ValueError(
            f"{name} must have 12 entries (January first); got {len(table)}."
        )
    return {
        month: float(value)
        for month, value in zip(_MONTHS, table, strict=True)
        if value is not None
    }


@dataclass(frozen=True)
class Reference:
    """
    Expected performance ratio and reference cell temperature of a PR test.

    Attributes
    ----------
    pr : float or dict of int to float or None
        Expected PR, a single value or ``{month: value}`` for monthly granularity.
    t_ref : float or dict of int to float or None
        Reference cell temperature in °C in the same form, None for a plain test.
    source : str
        ``"fixed"`` (stated by the contract) or ``"derived"`` (from the simulation).
    granularity : str
        ``"single"`` or ``"monthly"``.
    """

    pr: float | dict[int, float] | None
    t_ref: float | dict[int, float] | None
    source: str
    granularity: str

    def require_months(self, months: Iterable[int], *, temp_corrected: bool) -> None:
        """
        Check a monthly reference has a value for each month in use.

        Parameters
        ----------
        months : iterable of int
            Month numbers (1-12) the test needs.
        temp_corrected : bool
            Whether the reference temperature table must also cover the months.

        Raises
        ------
        ValueError
            If a month has no expected PR (or, when `temp_corrected`, no `t_ref`);
            the message names the month and the table. A single reference needs
            no months and never raises.
        """
        if self.granularity != "monthly":
            return
        tables = [("pr", self.pr, "expected_pr_monthly")]
        if temp_corrected:
            tables.append(("t_ref", self.t_ref, "t_ref_monthly"))
        for month in sorted({int(m) for m in months}):
            for _, values, table in tables:
                if values is None or month not in values:
                    where = (
                        f"the {table} table"
                        if self.source == "fixed"
                        else "the modeled rows (no kept rows in that month)"
                    )
                    raise ValueError(
                        f"The reference has no value for month {month} "
                        f"({_month_name(month)}) in {where}."
                    )

    def t_ref_for(self, index: pd.DatetimeIndex) -> float | pd.Series | None:
        """
        Return the reference temperature applying to each row of `index`.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Row timestamps; a monthly reference picks each row's own month.

        Returns
        -------
        float or pd.Series or None
            The single value, a Series aligned to `index` for a monthly
            reference, or None for a plain reference.

        Raises
        ------
        ValueError
            If a monthly reference lacks a month present in `index`.
        """
        if self.t_ref is None or self.granularity != "monthly":
            return self.t_ref
        self.require_months(index.month.unique(), temp_corrected=True)
        return pd.Series(index.month.map(self.t_ref).to_numpy(float), index=index)


def fixed_reference(
    *,
    temp_corrected: bool,
    granularity: str = "single",
    expected_pr: float | None = None,
    t_ref: float | None = None,
    expected_pr_monthly: list | None = None,
    t_ref_monthly: list | None = None,
) -> Reference:
    """
    Build a reference from values stated by the contract.

    Parameters
    ----------
    temp_corrected : bool
        Whether the test is temperature corrected, which needs a stated t_ref.
    granularity : {"single", "monthly"}, default "single"
        One value pair for the whole test or one per calendar month.
    expected_pr : float, optional
        Expected PR for single granularity; finite and greater than 0.
    t_ref : float, optional
        Reference temperature in °C for single granularity.
    expected_pr_monthly : list, optional
        12 expected PR values, January first, None for months not stated.
    t_ref_monthly : list, optional
        12 reference temperatures in °C, January first, None where not stated.

    Returns
    -------
    Reference
        With ``source="fixed"``.

    Raises
    ------
    ValueError
        If the granularity is unknown, a needed value is missing, a reference
        temperature is given for a plain test, a value is not finite (or a PR is
        not greater than 0), a monthly table does not have 12 entries, or a
        value of the other granularity is supplied.
    """
    if granularity not in ("single", "monthly"):
        raise ValueError(
            f"granularity must be 'single' or 'monthly'; got {granularity!r}."
        )
    if not temp_corrected and (t_ref is not None or t_ref_monthly is not None):
        raise ValueError(
            "t_ref / t_ref_monthly were given for a plain test; a reference "
            "temperature only applies to a temperature-corrected test."
        )
    if granularity == "single":
        if expected_pr_monthly is not None or t_ref_monthly is not None:
            raise ValueError("Monthly tables were given with granularity='single'.")
        if expected_pr is None:
            raise ValueError("expected_pr is required for single granularity.")
        _check_pr_value("expected_pr", expected_pr)
        if temp_corrected:
            if t_ref is None:
                raise ValueError(
                    "A temperature-corrected fixed reference needs an explicit t_ref."
                )
            _check_finite("t_ref", t_ref)
        return Reference(
            float(expected_pr),
            None if t_ref is None else float(t_ref),
            "fixed",
            "single",
        )
    if expected_pr is not None or t_ref is not None:
        raise ValueError("expected_pr / t_ref were given with granularity='monthly'.")
    if expected_pr_monthly is None:
        raise ValueError("expected_pr_monthly is required for monthly granularity.")
    pr_table = _monthly_table("expected_pr_monthly", expected_pr_monthly)
    for month, value in pr_table.items():
        _check_pr_value(f"expected_pr_monthly[{month - 1}]", value)
    t_table = None
    if temp_corrected:
        if t_ref_monthly is None:
            raise ValueError(
                "A temperature-corrected fixed reference needs an explicit "
                "t_ref_monthly."
            )
        t_table = _monthly_table("t_ref_monthly", t_ref_monthly)
        for month, value in t_table.items():
            _check_finite(f"t_ref_monthly[{month - 1}]", value)
    return Reference(pr_table, t_table, "fixed", "monthly")


def _warn_partial_months(kept_index, unfiltered_index):
    """Warn for months with a run of 7+ consecutive days without kept rows."""
    all_days = pd.Series(unfiltered_index.normalize().unique()).sort_values()
    kept_days = set(kept_index.normalize())
    kept_months = set(kept_index.month)
    for month, days in all_days.groupby(all_days.dt.month):
        if month not in kept_months:
            continue
        run = longest = 0
        previous = None
        for day in days:
            if day in kept_days:
                run = 0
            else:
                contiguous = previous is not None and day - previous == pd.Timedelta(
                    days=1
                )
                run = run + 1 if contiguous else 1
                longest = max(longest, run)
            previous = day if day not in kept_days else None
        if longest >= _PARTIAL_MONTH_RUN_DAYS:
            warnings.warn(
                f"{_month_name(month)} is only partly present in the modeled rows: "
                f"a run of {longest} consecutive days has no kept rows. A partial "
                "month is not the contract's monthly value; check for a Time or "
                "Days filter left on the sim pipeline.",
                UserWarning,
                stacklevel=3,
            )


def derive_reference(
    energy_wh: pd.Series,
    poa: pd.Series,
    *,
    dc_nameplate: float,
    dt_hours: float,
    cell_temp: pd.Series | None = None,
    monthly: bool = False,
    unfiltered_index: pd.DatetimeIndex | None = None,
) -> Reference:
    """
    Derive the expected PR and reference temperature from simulated rows.

    The expected PR is the plain PR of the rows (no temperature correction or
    availability); the reference temperature is their irradiance-weighted cell
    temperature (NREL eq. 5). With `monthly`, rows are grouped by calendar month
    number only, so rows from different years pool together.

    Parameters
    ----------
    energy_wh : pd.Series
        Modeled interval energy in Wh; the complete-row set of the sim side.
    poa : pd.Series
        Modeled plane-of-array irradiance in W/m², same index as `energy_wh`.
    dc_nameplate : float
        DC nameplate capacity in W.
    dt_hours : float
        Interval length in hours, from the unfiltered sim index.
    cell_temp : pd.Series, optional
        Modeled cell temperature in °C; when given a reference temperature is
        derived, otherwise `t_ref` is None.
    monthly : bool, default False
        Derive one value pair per calendar month.
    unfiltered_index : pd.DatetimeIndex, optional
        Unfiltered sim index, used with `monthly` to detect partly present months.

    Returns
    -------
    Reference
        With ``source="derived"``.

    Raises
    ------
    ValueError
        If the inputs are misaligned or empty, or a derived PR or reference
        temperature is not finite.

    Warns
    -----
    UserWarning
        With `monthly` and `unfiltered_index`, when a month with kept rows has a
        run of 7 or more consecutive days with no kept rows.
    """
    _require_same_index("energy_wh", energy_wh, poa=poa, cell_temp=cell_temp)

    def pair(energy, irradiance, temperature, label):
        expected = expected_energy_wh(
            irradiance, dc_nameplate=dc_nameplate, dt_hours=dt_hours
        )
        pr = pr_ratio(energy, expected)[0]
        _check_finite(f"derived pr{label}", pr)
        t_ref = None
        if temperature is not None:
            t_ref = weighted_cell_temp(irradiance, temperature)
            _check_finite(f"derived t_ref{label}", t_ref)
        return pr, t_ref

    if not monthly:
        pr, t_ref = pair(energy_wh, poa, cell_temp, "")
        return Reference(pr, t_ref, "derived", "single")
    if unfiltered_index is not None:
        _warn_partial_months(energy_wh.index, unfiltered_index)
    pr_by_month, t_ref_by_month = {}, {}
    for month, rows in energy_wh.groupby(energy_wh.index.month):
        temperature = None if cell_temp is None else cell_temp[rows.index]
        pr, t_ref = pair(rows, poa[rows.index], temperature, f" for month {month}")
        pr_by_month[int(month)] = pr
        t_ref_by_month[int(month)] = t_ref
    return Reference(
        pr_by_month,
        t_ref_by_month if cell_temp is not None else None,
        "derived",
        "monthly",
    )


@dataclass
class MeasuredPr:
    """
    Measured performance ratio with its per-interval and per-month detail.

    Attributes
    ----------
    ratio : float
        ``sum(E) / sum(EN_DC · A)`` over all rows.
    ratio_without_availability : float
        The same ratio with the availability factor taken as 1.
    intervals : pd.DataFrame
        Per row ``energy_wh, expected_wh, poa, cell_temp, t_ref, availability, pr``;
        quantities that do not apply are NaN.
    monthly : pd.DataFrame
        Indexed by month number: ``intervals``, ``insolation_wh_m2`` (``sum(poa·Δt)``)
        and ``measured_pr`` of that month.
    """

    ratio: float
    ratio_without_availability: float
    intervals: pd.DataFrame
    monthly: pd.DataFrame


def measured_pr(
    energy_wh: pd.Series,
    poa: pd.Series,
    *,
    dc_nameplate: float,
    dt_hours: float,
    reference: Reference,
    cell_temp: pd.Series | None = None,
    power_temp_coeff: float | None = None,
    availability: float | pd.Series = 1.0,
) -> MeasuredPr:
    """
    Compute the measured PR over a test period.

    Expected energy uses the reference temperature of each row's own month for a
    monthly reference; no blended temperature is ever formed.

    Parameters
    ----------
    energy_wh : pd.Series
        Measured interval energy in Wh; the complete-row set of the measured side.
    poa : pd.Series
        Measured plane-of-array irradiance in W/m², same index as `energy_wh`.
    dc_nameplate : float
        DC nameplate capacity in W.
    dt_hours : float
        Interval length in hours, from the unfiltered measured index.
    reference : Reference
        Supplies the reference temperature for a temperature-corrected test.
    cell_temp : pd.Series, optional
        Measured cell temperature in °C; selects the temperature-corrected form.
    power_temp_coeff : float, optional
        Power temperature coefficient in %/°C; required with `cell_temp`.
    availability : float or pd.Series, default 1.0
        Availability factor, 0 ≤ A ≤ 1.

    Returns
    -------
    MeasuredPr
        The ratio with and without availability, per-interval and per-month tables.

    Raises
    ------
    ValueError
        If inputs are misaligned or invalid, the reference lacks a month present
        in the rows, or the total expected energy is not positive and finite.
    """
    _require_same_index("energy_wh", energy_wh, poa=poa, cell_temp=cell_temp)
    t_ref = None
    if cell_temp is not None:
        t_ref = reference.t_ref_for(energy_wh.index)
    common = {
        "dc_nameplate": dc_nameplate,
        "dt_hours": dt_hours,
        "cell_temp": cell_temp,
        "t_ref": t_ref,
        "power_temp_coeff": power_temp_coeff,
    }
    expected_wh = expected_energy_wh(poa, availability=availability, **common)
    expected_full_wh = expected_energy_wh(poa, **common)
    ratio, interval_pr = pr_ratio(energy_wh, expected_wh)
    ratio_without_availability = pr_ratio(energy_wh, expected_full_wh)[0]

    def column(value):
        if isinstance(value, pd.Series):
            return value
        return pd.Series(np.nan if value is None else value, index=energy_wh.index)

    intervals = pd.DataFrame(
        {
            "energy_wh": energy_wh,
            "expected_wh": expected_wh,
            "poa": poa,
            "cell_temp": column(cell_temp),
            "t_ref": column(t_ref),
            "availability": column(availability),
            "pr": interval_pr,
        }
    )
    by_month = intervals.groupby(intervals.index.month)
    expected_total = by_month["expected_wh"].sum()
    monthly = pd.DataFrame(
        {
            "intervals": by_month.size(),
            "insolation_wh_m2": by_month["poa"].sum() * dt_hours,
            "measured_pr": by_month["energy_wh"].sum()
            / expected_total.where(expected_total > 0),
        }
    )
    monthly.index.name = "month"
    return MeasuredPr(ratio, ratio_without_availability, intervals, monthly)


def weighted_expected_pr(reference: Reference, insolation_wh_m2: pd.Series) -> float:
    """
    Combine monthly expected PRs by measured insolation.

    ``sum(H_m · EPR_m) / sum(H_m)`` over months with ``H_m > 0``; a single
    reference returns its `pr` unchanged.

    Parameters
    ----------
    reference : Reference
        The expected PR reference.
    insolation_wh_m2 : pd.Series
        Measured insolation per month number over the kept test rows.

    Returns
    -------
    float
        The insolation-weighted expected PR.

    Raises
    ------
    ValueError
        If a month with insolation has no expected PR, or no month has positive
        insolation.
    """
    if reference.granularity != "monthly":
        return reference.pr
    weights = insolation_wh_m2[insolation_wh_m2 > 0]
    if weights.empty:
        raise ValueError(
            "No month has positive insolation; the expected PR is undefined."
        )
    reference.require_months(weights.index, temp_corrected=False)
    values = pd.Series({int(m): reference.pr[int(m)] for m in weights.index})
    values.index = weights.index
    return float((weights * values).sum() / weights.sum())


@dataclass
class PrTestResults:
    """
    Result of a performance ratio test.

    Attributes
    ----------
    ratio : float
        Measured PR including the availability factor.
    ratio_without_availability : float
        The same ratio with availability taken as 1.
    temp_corrected : bool
        Whether the temperature-corrected form was used.
    dc_nameplate : float
        DC nameplate capacity in W.
    power_temp_coeff : float or None
        Power temperature coefficient in %/°C, None for a plain test.
    intervals : pd.DataFrame
        Per-row energy, expected energy, irradiance, temperatures and PR.
    points_used, dropped_nan, dropped_nonfinite, dt_hours : dict
        Per side (``"meas"``, ``"sim"``): rows used, rows dropped for NaN,
        rows dropped for inf, and the interval length in hours.
    numerator : dict
        Per side ``{"kind", "units"}`` of the energy or power numerator.
    t_ref : float or None
        Reference cell temperature in °C for a single-valued reference.
    expected_ratio, expected_ratio_undegraded : float or None
        Expected PR with and without the degradation factor.
    degradation : dict or None
        ``{"rate", "years", "factor"}`` when degradation was applied.
    tolerance : str or None
        Tolerance as given, e.g. ``"5%"``.
    threshold, margin, ratio_to_threshold : float or None
        Judgement values; see `Judgement`.
    passed : bool or None
        Pass/fail; None when no expected PR was supplied.
    valid : bool or None
        Whether the validity checks in `checks` all hold.
    checks : dict
        Named validity checks.
    reference_source : str or None
        ``"fixed"``, ``"derived"`` or ``"standalone"``.
    granularity : str
        ``"single"`` or ``"monthly"``.
    monthly : pd.DataFrame or None
        Per-month intervals, insolation and measured PR.
    """

    kind: ClassVar[str] = "pr"

    ratio: float
    ratio_without_availability: float
    temp_corrected: bool
    dc_nameplate: float
    power_temp_coeff: float | None
    intervals: pd.DataFrame
    points_used: dict
    dropped_nan: dict
    dropped_nonfinite: dict
    dt_hours: dict
    numerator: dict
    t_ref: float | None = None
    expected_ratio: float | None = None
    expected_ratio_undegraded: float | None = None
    degradation: dict | None = None
    tolerance: str | None = None
    threshold: float | None = None
    margin: float | None = None
    ratio_to_threshold: float | None = None
    passed: bool | None = None
    valid: bool | None = None
    checks: dict = field(default_factory=dict)
    reference_source: str | None = None
    granularity: str = "single"
    monthly: pd.DataFrame | None = None

    def summary(self) -> str:
        """
        Return a plain-text summary of the result.

        Returns
        -------
        str
            Measured PR as a percentage; when an expected PR was judged, also
            the PASS/FAIL outcome, expected PR, threshold and margin in PR
            points.
        """
        lines = []
        if self.passed is not None:
            lines.append(
                "Performance ratio test: " + ("PASS" if self.passed else "FAIL")
            )
        form = "temperature-corrected" if self.temp_corrected else "plain"
        lines.append(f"Measured PR ({form}): {self.ratio * 100:.2f} %")
        if self.expected_ratio is not None:
            lines.append(f"Expected PR: {self.expected_ratio * 100:.2f} %")
        if self.threshold is not None:
            lines.append(f"Threshold: {self.threshold * 100:.2f} %")
        if self.margin is not None:
            lines.append(f"Margin: {self.margin * 100:+.2f} PR points")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def performance_ratio(
    data: pd.DataFrame,
    *,
    numerator: str,
    poa: str,
    dc_nameplate: float,
    kind: str = "power",
    units: str = "W",
    dt_hours: float | None = None,
    cell_temp: str | None = None,
    bom: str | None = None,
    temp_amb: str | None = None,
    wind_speed: str | None = None,
    power_temp_coeff: float | None = None,
    base_temp: float = 25.0,
    module_type: str = "glass_cell_poly",
    racking: str = "open_rack",
    temp_delta: float | None = None,
) -> PrTestResults:
    """
    Compute the performance ratio of one dataset, with no pass/fail judgement.

    At most one temperature source may be given: a measured cell temperature
    column, a back-of-module column (cell temperature from the Sandia model), or
    ambient temperature plus wind speed (back-of-module then cell temperature).
    Any of them selects the temperature-corrected form with the reference
    temperature `base_temp`; none gives the plain form.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed data; rows with a missing or non-finite input are dropped.
    numerator : str
        Column of measured interval power or interval energy.
    poa : str
        Plane-of-array irradiance column in W/m².
    dc_nameplate : float
        DC nameplate capacity in W.
    kind : {"power", "energy"}, default "power"
        What `numerator` holds.
    units : str, default "W"
        Units of `numerator`; ``W``/``kW``/``MW`` or ``Wh``/``kWh``/``MWh``.
    dt_hours : float, optional
        Interval length in hours. Default is the dominant timestep of
        `data.index`; pass it explicitly when `data` is filtered.
    cell_temp : str, optional
        Measured cell temperature column in °C.
    bom : str, optional
        Back-of-module temperature column in °C.
    temp_amb, wind_speed : str, optional
        Ambient temperature (°C) and wind speed (m/s) columns; give both.
    power_temp_coeff : float, optional
        Power temperature coefficient in %/°C; required with a temperature
        source and rejected without one.
    base_temp : float, default 25.0
        Reference cell temperature in °C.
    module_type, racking : str
        Sandia model selectors for `bom` / `temp_amb`.
    temp_delta : float, optional
        Cell-to-back-of-module temperature difference at 1000 W/m².

    Returns
    -------
    PrTestResults
        Measured PR and its detail, keyed by side ``"meas"``; ``passed``,
        ``valid`` and ``expected_ratio`` are None.

    Raises
    ------
    ValueError
        If more than one temperature source is given, ``temp_amb`` and
        ``wind_speed`` are not given together, `power_temp_coeff` is missing
        with a temperature source or given without one, or `dc_nameplate`,
        `dt_hours` or the units are invalid.
    """
    _require_positive_finite("dc_nameplate", dc_nameplate)
    if dt_hours is None:
        dt_hours = _index_dt_hours(data.index)
    _require_positive_finite("dt_hours", dt_hours)
    if (temp_amb is None) != (wind_speed is None):
        raise ValueError("temp_amb and wind_speed must be given together.")
    sources = [
        name
        for name, given in (
            ("cell_temp", cell_temp),
            ("bom", bom),
            ("temp_amb and wind_speed", temp_amb),
        )
        if given is not None
    ]
    if len(sources) > 1:
        raise ValueError(
            f"Give at most one temperature source; got {' and '.join(sources)}."
        )
    corrected = bool(sources)
    if corrected and power_temp_coeff is None:
        raise ValueError(f"power_temp_coeff is required with {sources[0]}.")
    if not corrected and power_temp_coeff is not None:
        raise ValueError(
            "power_temp_coeff was given without a temperature source "
            "(cell_temp, bom, or temp_amb with wind_speed)."
        )
    if corrected:
        validate_power_temp_coeff(power_temp_coeff)

    inputs = {"energy": data[numerator], "poa": data[poa]}
    if temp_amb is not None:
        bom_series = calcparams.bom_temp(
            data,
            poa=poa,
            temp_amb=temp_amb,
            wind_speed=wind_speed,
            module_type=module_type,
            racking=racking,
            verbose=False,
        )
        data = data.assign(_bom_temp=bom_series)
        bom = "_bom_temp"
    if bom is not None:
        inputs["cell_temp"] = calcparams.cell_temp(
            data,
            bom,
            poa,
            module_type=module_type,
            racking=racking,
            temp_delta=temp_delta,
            verbose=False,
        )
    elif cell_temp is not None:
        inputs["cell_temp"] = data[cell_temp]
    frame = pd.DataFrame(inputs)
    rows = complete_rows(frame, label="meas")
    kept = frame.loc[rows.index]
    energy_wh = to_energy_wh(kept["energy"], kind=kind, units=units, dt_hours=dt_hours)
    measured = measured_pr(
        energy_wh,
        kept["poa"],
        dc_nameplate=dc_nameplate,
        dt_hours=dt_hours,
        reference=Reference(
            pr=None,
            t_ref=base_temp if corrected else None,
            source="standalone",
            granularity="single",
        ),
        cell_temp=kept["cell_temp"] if corrected else None,
        power_temp_coeff=power_temp_coeff,
    )
    return PrTestResults(
        ratio=measured.ratio,
        ratio_without_availability=measured.ratio_without_availability,
        temp_corrected=corrected,
        dc_nameplate=dc_nameplate,
        power_temp_coeff=power_temp_coeff,
        intervals=measured.intervals,
        points_used={"meas": len(rows.index)},
        dropped_nan={"meas": rows.dropped_nan},
        dropped_nonfinite={"meas": rows.dropped_nonfinite},
        dt_hours={"meas": dt_hours},
        numerator={"meas": {"kind": kind, "units": units}},
        t_ref=base_temp if corrected else None,
        reference_source="standalone",
        granularity="single",
        monthly=measured.monthly,
    )
