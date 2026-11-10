"""
Performance ratio (PR) test calculations.

Pure functions implementing the energy-based performance ratio of NREL/TP-5200-57991
(Dierauf et al., "Weather-Corrected Performance Ratio"): unit conversion, expected
energy from plane-of-array irradiance with an optional temperature correction, and
input validators. Measured and expected energy are always handled in Wh internally.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
        given without `cell_temp`, or `cell_temp` is given without both.
    """
    _require_positive_finite("dc_nameplate", dc_nameplate)
    _require_positive_finite("dt_hours", dt_hours)
    validate_availability(availability)
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

    Parameters
    ----------
    poa : pd.Series
        Plane-of-array irradiance, used as the weight.
    cell_temp : pd.Series
        Cell temperature in °C, aligned to `poa`.

    Returns
    -------
    float
        ``sum(poa * cell_temp) / sum(poa)`` in °C.

    Raises
    ------
    ValueError
        If the total irradiance is not greater than zero.
    """
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
        Expected interval energy in Wh, aligned to `measured_wh`.

    Returns
    -------
    tuple of (float, pd.Series)
        The ratio of the totals and the per-interval ratio, which is NaN where
        the expected energy is zero.

    Raises
    ------
    ValueError
        If the total expected energy is zero.
    """
    total_expected = expected_wh.sum()
    if total_expected == 0:
        raise ValueError("total expected energy is zero; the PR is undefined.")
    interval = measured_wh / expected_wh.where(expected_wh != 0)
    return float(measured_wh.sum() / total_expected), interval


def resample_interval_energy(energy, freq, *, label="left", closed="left"):
    """
    Sum interval energy into coarser bins, voiding any bin that is not complete.

    Bins are aligned to midnight of the first day. A bin with a missing or NaN
    interval (including a register reset) or with fewer native intervals than
    it spans (an incomplete first or last bin) is NaN rather than a partial sum.

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
        If the index has duplicate timestamps or timestamps off the grid counted
        from midnight of the first day, or `freq` is not a whole multiple of the
        native timestep.
    """
    if energy.index.has_duplicates:
        raise ValueError("energy index has duplicate timestamps.")
    origin = energy.index[0].normalize()
    native = common_timestep_delta(energy.index)
    if ((energy.index - origin) % native != pd.Timedelta(0)).any():
        raise ValueError(
            f"energy timestamps are not aligned to a {native} grid counted from "
            f"{origin}; resampling would mislabel the bins."
        )
    n_float = pd.Timedelta(freq) / native
    n = round(n_float)
    if n < 1 or not math.isclose(n_float, n):
        raise ValueError(
            f"freq {freq!r} is not a whole multiple of the native timestep {native}."
        )
    return energy.resample(freq, label=label, closed=closed, origin=origin).sum(
        min_count=n
    )
