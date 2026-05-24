"""Clear-sky irradiance modeling built on pvlib.

These functions model clear-sky GHI/POA (used by ``io.load_data`` when site
metadata is supplied). Clear-sky *filtering* (``FilterClearsky``) is separate
and uses ``pvlib.clearsky.detect_clearsky`` directly.
"""

import importlib.util
import warnings

import pandas as pd

pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.location import Location
    from pvlib.pvsystem import PVSystem, Array, FixedMount, SingleAxisTrackerMount
    from pvlib.pvsystem import retrieve_sam
    from pvlib.modelchain import ModelChain
else:
    warnings.warn("Clear sky functions will not work without the pvlib package.")


def pvlib_location(loc):
    """
    Create a pvlib location object.

    Parameters
    ----------
    loc : dict
        Dictionary of values required to instantiate a pvlib Location object.

        loc = {'latitude': float,
               'longitude': float,
               'altitude': float/int,
               'tz': str, int, float, default 'UTC'}
        See
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        ints and floats must be in hours from UTC.

    Returns
    -------
    pvlib location object.
    """
    return Location(**loc)


def pvlib_system(sys):
    """
    Create a pvlib :py:class:`~pvlib.pvsystem.PVSystem` object.

    The :py:class:`~pvlib.pvsystem.PVSystem` will have either a
    :py:class:`~pvlib.pvsystem.FixedMount` or a
    :py:class:`~pvlib.pvsystem.SingleAxisTrackerMount` depending on
    the keys of the passed dictionary.

    Parameters
    ----------
    sys : dict
        Dictionary of keywords required to create a pvlib
        ``SingleAxisTrackerMount`` or ``FixedMount``, plus ``albedo``.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.

    Returns
    -------
    pvlib PVSystem object.
    """
    sandia_modules = retrieve_sam("SandiaMod")
    cec_inverters = retrieve_sam("cecinverter")
    sandia_module = sandia_modules.iloc[:, 0]
    cec_inverter = cec_inverters.iloc[:, 0]

    albedo = sys.pop("albedo", None)
    trck_kwords = ["axis_tilt", "axis_azimuth", "max_angle", "backtrack", "gcr"]  # noqa: E501
    if any(kword in sys.keys() for kword in trck_kwords):
        mount = SingleAxisTrackerMount(**sys)
    else:
        mount = FixedMount(**sys)
    array = Array(
        mount,
        albedo=albedo,
        module_parameters=sandia_module,
        temperature_model_parameters={"u_c": 29.0, "u_v": 0.0},
    )
    system = PVSystem(arrays=[array], inverter_parameters=cec_inverter)

    return system


def get_tz_index(time_source, loc):
    """
    Create DatetimeIndex with timezone aligned with location dictionary.

    Handles generating a DatetimeIndex with a timezone for use as an agrument
    to pvlib ModelChain prepare_inputs method or pvlib Location get_clearsky
    method.

    Parameters
    ----------
    time_source : Dataframe, Series, or DatetimeIndex
        If passing a Dataframe or Series, the index of the dataframe will be used.
        If the index does not have a timezone, the timezone will be set using the
        timezone in the passed loc dictionary. If passing a DatetimeIndex with
        a timezone, it will be returned directly. If passing a DatetimeIndex
        without a timezone, the timezone will be set using the timezone in the passed
        loc dictionary.

    Returns
    -------
    DatetimeIndex with timezone
    """
    if isinstance(time_source, pd.core.series.Series) or isinstance(
        time_source, pd.core.frame.DataFrame
    ):
        time_source = time_source.index
    if isinstance(time_source, pd.core.indexes.datetimes.DatetimeIndex):
        if time_source.tz is None:
            time_source = time_source.tz_localize(
                loc["tz"], ambiguous="infer", nonexistent="NaT"
            )
            return time_source
        else:
            if loc["tz"] != str(time_source.tz):
                warnings.warn(
                    "The DatetimeIndex of time_source has a timezone that "
                    "does not match the timezone in the loc dict. "
                    "Using the timezone of the time_source DatetimeIndex."
                )
            return time_source


def csky(time_source, loc=None, sys=None, concat=True, output="both"):
    """
    Calculate clear sky poa and ghi.

    Parameters
    ----------
    time_source : dataframe or DatetimeIndex
        If passing a dataframe the index of the dataframe will be used.  If the
        index does not have a timezone the timezone will be set using the
        timezone in the passed loc dictionary. If passing a DatetimeIndex with
        a timezone it will be returned directly. If passing a DatetimeIndex
        without a timezone the timezone in the timezone dictionary will
        be used.
    loc : dict
        Dictionary of values required to instantiate a pvlib Location object.

        loc = {'latitude': float,
               'longitude': float,
               'altitude': float/int,
               'tz': str, int, float, default 'UTC'}
        See
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        ints and floats must be in hours from UTC.
    sys : dict
        Dictionary of keywords required to create a pvlib
        :py:class:`~pvlib.pvsystem.SingleAxisTrackerMount` or
        :py:class:`~pvlib.pvsystem.FixedMount`.

        Example dictionaries:

        fixed_sys = {'surface_tilt': 20,
                     'surface_azimuth': 180,
                     'albedo': 0.2}

        tracker_sys1 = {'axis_tilt': 0, 'axis_azimuth': 0,
                       'max_angle': 90, 'backtrack': True,
                       'gcr': 0.2, 'albedo': 0.2}

        Refer to pvlib documentation for details.
    concat : bool, default True
        If concat is True then returns columns as defined by return argument
        added to passed dataframe, otherwise returns just clear sky data.
    output : str, default 'both'
        both - returns only total poa and ghi
        poa_all - returns all components of poa
        ghi_all - returns all components of ghi
        all - returns all components of poa and ghi
    """
    location = pvlib_location(loc)
    system = pvlib_system(sys)
    mc = ModelChain(system, location)
    times = get_tz_index(time_source, loc)
    ghi = location.get_clearsky(times=times)
    # pvlib get_Clearsky also returns 'wind_speed' and 'temp_air'
    mc.prepare_inputs(weather=ghi)
    cols = [
        "poa_global",
        "poa_direct",
        "poa_diffuse",
        "poa_sky_diffuse",
        "poa_ground_diffuse",
    ]

    if output == "both":
        csky_df = pd.DataFrame(
            {
                "poa_mod_csky": mc.results.total_irrad["poa_global"],
                "ghi_mod_csky": ghi["ghi"],
            }
        )
    elif output == "poa_all":
        csky_df = mc.results.total_irrad[cols]
    elif output == "ghi_all":
        csky_df = ghi[["ghi", "dni", "dhi"]]
    elif output == "all":
        csky_df = pd.concat(
            [mc.results.total_irrad[cols], ghi[["ghi", "dni", "dhi"]]], axis=1
        )
    else:
        raise ValueError(
            f"Unrecognized output {output!r}; expected one of "
            "'both', 'poa_all', 'ghi_all', 'all'."
        )

    ix_no_tz = csky_df.index.tz_localize(None, ambiguous="infer", nonexistent="NaT")
    csky_df.index = ix_no_tz

    if concat:
        if isinstance(time_source, pd.core.frame.DataFrame):
            try:
                df_with_csky = pd.concat([time_source, csky_df], axis=1)
            except pd.errors.InvalidIndexError:
                # Drop NaT that occur for March DST shift in US data
                df_with_csky = pd.concat(
                    [time_source, csky_df.loc[csky_df.index.dropna(), :]], axis=1
                )
            return df_with_csky
        else:
            warnings.warn(
                "time_source is not a dataframe; only clear sky data\
                           returned"
            )
            return csky_df
    else:
        return csky_df
