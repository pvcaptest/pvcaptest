"""
Functions to calculate derived values from measured data.

For example, back-of-module temperature from poa, wind speed, and ambient temp with the
Sandia module temperature model.
"""

import copy
import warnings

import numpy as np
import pvlib
from pvlib.location import Location

# Global record surface-pressure extremes (mBar) used by
# :func:`absolute_airmass` to sanity-check pressure inputs. Low: ~870 mBar
# (Typhoon Tip, 1979). High: ~1080 mBar (Mongolia, 2001).
PRESSURE_MIN_MBAR = 870
PRESSURE_MAX_MBAR = 1080

EMP_HEAT_COEFF = {
    "open_rack": {
        "glass_cell_glass": {"a": -3.47, "b": -0.0594, "del_tcnd": 3},
        "glass_cell_poly": {"a": -3.56, "b": -0.0750, "del_tcnd": 3},
        "poly_tf_steel": {"a": -3.58, "b": -0.1130, "del_tcnd": 3},
    },
    "close_roof_mount": {"glass_cell_glass": {"a": -2.98, "b": -0.0471, "del_tcnd": 1}},
    "insulated_back": {"glass_cell_poly": {"a": -2.81, "b": -0.0455, "del_tcnd": 0}},
}


def power_temp_correct(
    data, power, cell_temp, power_temp_coeff=None, base_temp=25, verbose=True
):
    """Apply temperature correction to PV power.

    Divides `power` by the temperature correction, so low power values that
    are above `base_temp` will be increased and high power values that are
    below the `base_temp` will be decreased.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    power : str
        The column name of the data attribute with the power to correct.
    cell_temp : str
        Name of the column in `data` containing the cell temperature (in Celsius) used
        to calculate temperature differential from the `base_temp`.
    power_temp_coeff : numeric
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    base_temp : numeric, default 25
        Base temperature (in Celsius) to correct power to. Default is the
        STC of 25 degrees Celsius.

    Returns
    -------
    Series
        Power corrected for temperature.
    """
    if verbose:
        print(
            f'Calculating and adding "{power_temp_correct.__name__}" column as '
            f"({power}) / (1 + (({power_temp_coeff} / 100) * "
            f"({cell_temp} - {base_temp})))"
        )
    power = data[power]
    cell_temp = data[cell_temp]
    return power / (1 + ((power_temp_coeff / 100) * (cell_temp - base_temp)))


def bom_temp(
    data,
    poa=None,
    temp_amb=None,
    wind_speed=None,
    module_type="glass_cell_poly",
    racking="open_rack",
    verbose=True,
):
    """Calculate back of module temperature from measured weather data.

    Calculate back of module temperature from POA irradiance, ambient
    temperature, wind speed (at height of 10 meters), and empirically
    derived heat transfer coefficients.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance in W/m^2.
    temp_amb : str
        Column name for Ambient temperature in degrees C.
    wind_speed : str
        Column name for Measured wind speed (m/sec) corrected to measurement height of
        10 meters.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'

    Returns
    -------
    numeric or Series
        Back of module temperatures.
    """
    a = EMP_HEAT_COEFF[racking][module_type]["a"]
    b = EMP_HEAT_COEFF[racking][module_type]["b"]
    if verbose:
        print(
            f'Calculating and adding "{bom_temp.__name__}" column as '
            f"{poa} * e^({a} + {b} * {wind_speed}) + {temp_amb}. "
            f'Coefficients a and b assume "{module_type}" modules and "{racking}" racking.'
        )
    return data[poa] * np.exp(a + b * data[wind_speed]) + data[temp_amb]


def cell_temp(
    data, bom, poa, module_type="glass_cell_poly", racking="open_rack", verbose=True
):
    """Calculate cell temp from BOM temp, POA, and heat transfer coefficient.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    bom : str
        Column name for back of module temperature (degrees C). Strictly following the NREL
        procedure this value would be obtained from the `back_of_module_temp`
        function.

        Alternatively, a measured BOM temperature may be used.

        Refer to p.7 of NREL Weather Corrected Performance Ratio Report.
    poa : str
        Column name for POA irradiance in W/m^2.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'
    verbose : bool, default True
        By default prints explanation of calculation. Set to False for no output
        message.

    Returns
    -------
    Series
        Cell temperatures.
    """
    if verbose:
        print(
            f'Calculating and adding "{cell_temp.__name__}" column using the Sandia temperature '
            f'model assuming "{module_type}" module type and "{racking}" racking '
            f'from the "{bom}" and "{poa}" columns.'
        )
    bom_data = data[bom]
    poa_data = data[poa]
    return (
        bom_data + (poa_data / 1000) * EMP_HEAT_COEFF[racking][module_type]["del_tcnd"]
    )


def avg_typ_cell_temp(data, poa, cell_temp, verbose=True):
    """Calculate irradiance weighted cell temperature.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance (W/m^2).
    cell_temp : str
        Column name for Cell temperature for each interval (degrees C).

    Returns
    -------
    float
        Average irradiance-weighted cell temperature.
    """
    return (data[poa] * data[cell_temp]).sum() / data[poa].sum()


def rpoa_pvsyst(data, globbak="GlobBak", backshd="BackShd", verbose=True):
    """Calculate the sum of PVsyst's global rear irradiance and rear shading and IAM losses.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance containing PVsyst 8760 data.
    globbak : str, default 'GlobBak'
        Column name for global rear irradiance (W/m^2).
    backshd : str, default 'BackShd'
        Column name for rear shading and IAM losses (W/m^2).
    verbose : bool, default True
        Set to False to not print calculation explanation.

    Returns
    -------
    Series
        Sum of global rear irradiance and rear shading and IAM losses.
    """
    if verbose:
        print(
            f'Calculating and adding "{rpoa_pvsyst.__name__}" column as '
            f"{globbak} + {backshd}. "
        )
    return data[globbak] + data[backshd]


def e_total(
    data, poa, rpoa, bifaciality=0.7, bifacial_frac=1, rear_shade=0, verbose=True
):
    """
    Calculate total irradiance from POA and rear irradiance.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance (W/m^2).
    rpoa : str
        Column name for rear irradiance (W/m^2).
    bifaciality : numeric, default 0.7
        Bifaciality factor.
    bifacial_frac : numeric, default 1
        Fraction of total array nameplate power that is bifacial. Pass to calculate
        total plane of array irradiance for plants with a mix of monofacial and
        bifacial modules.
    rear_shade : numeric, default 0
        Fraction of rear irradiance that is lost due to shading. Set to decimal
        fraction, e.g. 0.12, to include in calculation of `e_total`.

    Returns
    -------
    numeric or Series
        Total plane of array irradiance.
    """
    if verbose:
        print(
            f'Calculating and adding "{e_total.__name__}" column as '
            f"{poa} + {rpoa} * "
            f"{bifaciality} * {bifacial_frac} * "
            f"(1 - {rear_shade})"
        )
    return data[poa] + data[rpoa] * bifaciality * bifacial_frac * (1 - rear_shade)


def apparent_zenith(data, site=None, altitude_override=0, verbose=True):
    """Compute apparent solar zenith angle at each timestamp in ``data``.

    Wraps :py:meth:`pvlib.location.Location.get_solarposition` and returns the
    ``apparent_zenith`` column aligned to ``data.index``. Designed for use
    inside a ``CapData.regression_cols`` calc tuple: ``site`` is auto-injected
    by ``CapData.custom_param`` from ``cd.site``.

    Per the pvlib First Solar spectral-correction reference, the absolute
    airmass is computed against zenith at sea level. ``altitude_override``
    defaults to 0 so a deep copy of ``site`` has its ``loc.altitude`` forced
    to 0 before the ``Location`` is instantiated. The caller's ``site`` dict
    is not mutated.

    Night-time rows (``apparent_zenith > 90``) are set to NaN so downstream
    airmass / spectral-factor calls do not emit pvlib warnings on invalid
    geometry.

    Parameters
    ----------
    data : DataFrame
        DataFrame with a DatetimeIndex. The index may be tz-naive or tz-aware.
    site : dict
        Nested ``{"loc": {...}, "sys": {...}}`` dict as produced by
        ``load_data(site=...)``. Only the ``loc`` sub-dict is consumed here.
        Auto-injected from ``cd.site`` by ``custom_param`` when used in a
        ``regression_cols`` calc tuple.
    altitude_override : numeric, default 0
        Altitude (in meters) to use when building the ``pvlib.Location``.
        Set to ``None`` to respect ``site['loc']['altitude']`` unchanged.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        Apparent zenith angle (degrees) indexed like ``data.index`` with a
        tz-naive index. NaN where the sun is below the horizon.
    """
    if verbose:
        print(
            f'Calculating and adding "{apparent_zenith.__name__}" column as '
            f"pvlib.Location(**site['loc']).get_solarposition(data.index) "
            f"with altitude_override={altitude_override}."
        )
    loc = copy.deepcopy(site["loc"])
    if altitude_override is not None:
        loc["altitude"] = altitude_override
    location = Location(**loc)

    times = data.index
    if times.tz is None:
        times = times.tz_localize(loc["tz"], ambiguous="infer", nonexistent="NaT")
    solar_positions = location.get_solarposition(times)
    zenith = solar_positions["apparent_zenith"]
    zenith.index = zenith.index.tz_localize(None)
    zenith = zenith.reindex(
        data.index.tz_localize(None) if data.index.tz is not None else data.index
    )
    zenith = zenith.where(zenith <= 90)
    return zenith


def apparent_zenith_pvsyst(
    data, site=None, altitude_override=0, shift_minutes=30, verbose=True
):
    """Apparent solar zenith at the mid-point of each PVsyst interval.

    PVsyst reports hourly values labelled at the start of each interval but
    computes sun positions at the interval mid-point. To match that
    convention we shift ``data.index`` forward by ``shift_minutes`` before
    calling :py:meth:`pvlib.location.Location.get_solarposition`, then shift
    the resulting Series index back by the same amount so the output aligns
    with the original ``data.index``.

    The site timezone should be a fixed-offset ``Etc/GMT±N`` string because
    PVsyst data is not DST-aware. ``CapTest.setup()`` auto-converts
    ``meas.site`` to an ``Etc/GMT±N`` variant when propagating it to
    ``sim.site``.

    Parameters
    ----------
    data : DataFrame
        DataFrame with a tz-naive DatetimeIndex at the PVsyst cadence.
    site : dict
        Same shape as :func:`apparent_zenith`. Auto-injected from ``cd.site``.
    altitude_override : numeric, default 0
        See :func:`apparent_zenith`.
    shift_minutes : int, default 30
        Interval mid-point offset applied to ``data.index`` before the pvlib
        solar-position call. Set to 0 to disable the shift.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        Apparent zenith angle (degrees) indexed like ``data.index``.
    """
    if verbose:
        print(
            f'Calculating and adding "{apparent_zenith_pvsyst.__name__}" column as '
            f"pvlib.Location(**site['loc']).get_solarposition(data.index + {shift_minutes} min) "
            f"with the result shifted back by {shift_minutes} min and altitude_override={altitude_override}."
        )
    loc = copy.deepcopy(site["loc"])
    if altitude_override is not None:
        loc["altitude"] = altitude_override
    location = Location(**loc)

    shifted_index = data.index.shift(shift_minutes, "min")
    if shifted_index.tz is None:
        shifted_index = shifted_index.tz_localize(loc["tz"])
    solar_positions = location.get_solarposition(shifted_index)
    zenith = solar_positions["apparent_zenith"]
    zenith.index = zenith.index.tz_localize(None).shift(-shift_minutes, "min")
    # Align to the caller's index in case of any residual tz/frequency quirks.
    target_index = (
        data.index.tz_localize(None) if data.index.tz is not None else data.index
    )
    zenith = zenith.reindex(target_index)
    zenith = zenith.where(zenith <= 90)
    return zenith


def _check_pressure_range(pressure_pa, pressure_col):
    """Warn if the central 90% of ``pressure_pa`` falls outside plausible bounds.

    Uses the 5th and 95th percentiles to ignore isolated outlier readings from
    sensor noise or missing-value placeholders. Compares against the global
    record surface-pressure extremes (``PRESSURE_MIN_MBAR`` /
    ``PRESSURE_MAX_MBAR``) converted to Pa. Any percentile outside that band
    indicates a unit mismatch (e.g. a column already in Pa with
    ``pressure_scale=100`` applied, or a column in kPa) or bad data.

    Parameters
    ----------
    pressure_pa : Series
        Pressure values in Pa after applying ``pressure_scale``.
    pressure_col : str
        Column name used for the warning message context.
    """
    clean = pressure_pa.dropna()
    if clean.empty:
        return
    p5, p95 = clean.quantile([0.05, 0.95])
    min_pa = PRESSURE_MIN_MBAR * 100
    max_pa = PRESSURE_MAX_MBAR * 100
    if p5 < min_pa or p95 > max_pa:
        warnings.warn(
            f"absolute_airmass: scaled pressure values from column "
            f"{pressure_col!r} appear out of range. 5th/95th percentile: "
            f"{p5:.0f}/{p95:.0f} Pa; expected the central 90% to fall within "
            f"[{min_pa}, {max_pa}] Pa (global records "
            f"{PRESSURE_MIN_MBAR}-{PRESSURE_MAX_MBAR} mBar). Check the "
            f"column units and the 'pressure_scale' kwarg.",
            stacklevel=3,
        )


def absolute_airmass(
    data,
    apparent_zenith=None,
    pressure=None,
    pressure_scale=100,
    airmass_model="kastenyoung1989",
    verbose=True,
):
    """Compute absolute (pressure-corrected) airmass from apparent zenith.

    Uses :py:func:`pvlib.atmosphere.get_relative_airmass` with the
    ``kastenyoung1989`` model by default, then passes the result to
    :py:func:`pvlib.atmosphere.get_absolute_airmass`. If ``pressure`` is
    ``None`` the pvlib default (101325 Pa) is used; otherwise the column
    ``data[pressure]`` is scaled by ``pressure_scale`` (default 100 to
    convert hPa/mbar to Pa) and passed through.

    When a ``pressure`` column is supplied, the scaled pressure values are
    sanity-checked against global surface-pressure records
    (:data:`PRESSURE_MIN_MBAR` – :data:`PRESSURE_MAX_MBAR`). The 5th and
    95th percentiles are used to ignore isolated outliers from bad data. A
    :class:`UserWarning` is emitted if the central 90% of values falls
    outside that band, which typically indicates a unit mismatch between
    ``data[pressure]`` and ``pressure_scale``.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the ``apparent_zenith`` (and optionally
        ``pressure``) columns.
    apparent_zenith : str
        Column name for apparent zenith angle (degrees).
    pressure : str or None, default None
        Column name for station pressure. ``None`` falls back to pvlib's
        default sea-level pressure.
    pressure_scale : numeric, default 100
        Multiplier applied to ``data[pressure]`` before passing to pvlib.
        Default converts hPa/mbar to Pa.
    airmass_model : str, default 'kastenyoung1989'
        Model passed to :py:func:`pvlib.atmosphere.get_relative_airmass`.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        Absolute airmass indexed like ``data.index``.
    """
    if verbose:
        pressure_desc = (
            f"{pressure} * {pressure_scale} Pa"
            if pressure is not None
            else "pvlib default (101325 Pa)"
        )
        print(
            f'Calculating and adding "{absolute_airmass.__name__}" column using '
            f'airmass_model="{airmass_model}" on "{apparent_zenith}" and pressure={pressure_desc}.'
        )
    rel_airmass = pvlib.atmosphere.get_relative_airmass(
        data[apparent_zenith], model=airmass_model
    )
    if pressure is None:
        return pvlib.atmosphere.get_absolute_airmass(rel_airmass)
    pressure_pa = data[pressure] * pressure_scale
    _check_pressure_range(pressure_pa, pressure)
    return pvlib.atmosphere.get_absolute_airmass(rel_airmass, pressure_pa)


def precipitable_water_gueymard(data, temp_amb=None, rel_humidity=None, verbose=True):
    """Precipitable water (cm) from ambient temperature and relative humidity.

    Wraps :py:func:`pvlib.atmosphere.gueymard94_pw`.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the ambient-temperature and relative-humidity
        columns.
    temp_amb : str
        Column name for ambient (dry-bulb) temperature in degrees Celsius.
    rel_humidity : str
        Column name for relative humidity as a percentage (0-100).
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        Precipitable water (cm) indexed like ``data.index``.
    """
    if verbose:
        print(
            f'Calculating and adding "{precipitable_water_gueymard.__name__}" column as '
            f"pvlib.atmosphere.gueymard94_pw({temp_amb}, {rel_humidity})."
        )
    return pvlib.atmosphere.gueymard94_pw(data[temp_amb], data[rel_humidity])


def scale(data, col=None, factor=1.0, verbose=True):
    """Multiply a single column by a scalar factor.

    Generic unit-conversion / rescaling helper usable in
    ``regression_cols`` calc trees. Primary use in this module is converting
    PVsyst ``PrecWat`` from meters to centimeters with ``factor=100``.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame.
    col : str
        Column name to scale.
    factor : numeric, default 1.0
        Scalar multiplier applied elementwise to ``data[col]``.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        ``data[col] * factor`` indexed like ``data.index``.
    """
    if verbose:
        print(f'Calculating and adding "{scale.__name__}" column as {col} * {factor}.')
    return data[col] * factor


def spectral_factor_firstsolar(
    data,
    precipitable_water=None,
    absolute_airmass=None,
    spectral_module_type="cdte",
    verbose=True,
):
    """First Solar spectral correction factor.

    Wraps :py:func:`pvlib.spectrum.spectral_factor_firstsolar`.
    ``spectral_module_type`` defaults to ``'cdte'`` but can be overridden via
    a ``cd.spectral_module_type`` attribute which ``custom_param``
    auto-injects when the kwarg is left unset. ``CapTest`` propagates its
    ``spectral_module_type`` param onto both CapData instances at
    ``setup()``.

    The kwarg is named ``spectral_module_type`` (not ``module_type``) to
    avoid collisions with the ``module_type`` kwarg used by :func:`bom_temp`
    and :func:`cell_temp`, which expects values like ``'glass_cell_poly'``
    rather than the pvlib First Solar module-type strings.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the precipitable-water and absolute-airmass
        columns.
    precipitable_water : str
        Column name for precipitable water in cm.
    absolute_airmass : str
        Column name for absolute airmass.
    spectral_module_type : str, default 'cdte'
        Passed through to :py:func:`pvlib.spectrum.spectral_factor_firstsolar`
        as its ``module_type`` argument.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        Spectral correction factor indexed like ``data.index``.
    """
    if verbose:
        print(
            f'Calculating and adding "{spectral_factor_firstsolar.__name__}" column as '
            f"pvlib.spectrum.spectral_factor_firstsolar({precipitable_water}, {absolute_airmass}, "
            f'module_type="{spectral_module_type}").'
        )
    return pvlib.spectrum.spectral_factor_firstsolar(
        data[precipitable_water],
        data[absolute_airmass],
        module_type=spectral_module_type,
    )


def multiply(data, a=None, b=None, verbose=True):
    """Elementwise multiplication of two columns.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame.
    a, b : str
        Column names to multiply. Both kwarg names must not collide with any
        ``column_groups`` id, per ``CapData.custom_param`` semantics.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        ``data[a] * data[b]`` indexed like ``data.index``.
    """
    if verbose:
        print(f'Calculating and adding "{multiply.__name__}" column as {a} * {b}.')
    return data[a] * data[b]


def poa_spec_corrected(data, poa=None, spectral_correction=None, verbose=True):
    """Spectrally corrected plane-of-array irradiance.

    Thin named alias that multiplies a POA column by a spectral-correction
    column. Primary use is the top-level node of a ``regression_cols`` calc
    tree whose ``spectral_correction`` kwarg is itself a calc subtree ending
    in :func:`spectral_factor_firstsolar`.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame.
    poa : str
        Column name for plane-of-array irradiance (W/m^2).
    spectral_correction : str
        Column name for the spectral correction factor.
    verbose : bool, default True
        Set to False to suppress the explanatory print message.

    Returns
    -------
    Series
        ``data[poa] * data[spectral_correction]`` indexed like ``data.index``.
    """
    if verbose:
        print(
            f'Calculating and adding "{poa_spec_corrected.__name__}" column as '
            f"{poa} * {spectral_correction}."
        )
    return data[poa] * data[spectral_correction]
