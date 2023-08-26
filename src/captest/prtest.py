import warnings

import numpy as np
import pandas as pd
import param
from scipy import stats

from captest import capdata


emp_heat_coeff = {
    "open_rack": {
        "glass_cell_glass": {"a": -3.47, "b": -0.0594, "del_tcnd": 3},
        "glass_cell_poly": {"a": -3.56, "b": -0.0750, "del_tcnd": 3},
        "poly_tf_steel": {"a": -3.58, "b": -0.1130, "del_tcnd": 3},
    },
    "close_roof_mount": {"glass_cell_glass": {"a": -2.98, "b": -0.0471, "del_tcnd": 1}},
    "insulated_back": {"glass_cell_poly": {"a": -2.81, "b": -0.0455, "del_tcnd": 0}},
}


def get_common_timestep(data, units="m", string_output=True):
    """
    Get the most commonly occuring timestep of data as frequency string.
    Parameters
    ----------
    data : Series or DataFrame
        Data with a DateTimeIndex.
    units : str, default 'm'
        String representing date/time unit, such as (D)ay, (M)onth, (Y)ear,
        (h)ours, (m)inutes, or (s)econds.
    string_output : bool, default True
        Set to False to return a numeric value.
    Returns
    -------
    str
        frequency string
    """
    units_abbrev = {
        "D": "Day",
        "M": "Months",
        "Y": "Year",
        "h": "hours",
        "m": "minutes",
        "s": "seconds",
    }
    common_timestep = data.index.to_series().diff().mode().values[0]
    common_timestep_tdelta = common_timestep.astype("timedelta64[m]")
    freq = common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        return str(freq) + " " + units_abbrev[units]
    else:
        return freq


def temp_correct_power(power, power_temp_coeff, cell_temp, base_temp=25):
    """Apply temperature correction to PV power.

    Divides `power` by the temperature correction, so low power values that
    are above `base_temp` will be increased and high power values that are
    below the `base_temp` will be decreased.

    Parameters
    ----------
    power : numeric or Series
        PV power (in watts) to correct to the `base_temp`.
    power_temp_coeff : numeric
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    cell_temp : numeric or Series
        Cell temperature (in Celsius) used to calculate temperature
        differential from the `base_temp`.
    base_temp : numeric, default 25
        Base temperature (in Celsius) to correct power to. Default is the
        STC of 25 degrees Celsius.

    Returns
    -------
    type matches `power`
        Power corrected for temperature.
    """
    corr_power = (
        power /
        (1 + ((power_temp_coeff / 100) * (cell_temp - base_temp)))
    )
    return corr_power


def back_of_module_temp(
    poa, temp_amb, wind_speed, module_type="glass_cell_poly", racking="open_rack"
):
    """Calculate back of module temperature from measured weather data.

    Calculate back of module temperature from POA irradiance, ambient
    temperature, wind speed (at height of 10 meters), and empirically
    derived heat transfer coefficients.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    poa : numeric or Series
        POA irradiance in W/m^2.
    temp_amb : numeric or Series
        Ambient temperature in degrees C.
    wind_speed : numeric or Series
        Measured wind speed (m/sec) corrected to measurement height of
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
    a = emp_heat_coeff[racking][module_type]["a"]
    b = emp_heat_coeff[racking][module_type]["b"]
    return poa * np.exp(a + b * wind_speed) + temp_amb


def cell_temp(bom, poa, module_type="glass_cell_poly", racking="open_rack"):
    """Calculate cell temp from BOM temp, POA, and heat transfer coefficient.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    bom : numeric or Series
        Back of module temperature (degrees C). Strictly followin the NREL
        procedure this value would be obtained from the `back_of_module_temp`
        function.

        Alternatively, a measured BOM temperature may be used.

        Refer to p.7 of NREL Weather Corrected Performance Ratio Report.
    poa : numeric or Series
        POA irradiance in W/m^2.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'

    Returns
    -------
    numeric or Series
        Cell temperature(s).
    """
    return bom + (poa / 1000) * emp_heat_coeff[racking][module_type]["del_tcnd"]


def avg_typ_cell_temp(poa, cell_temp):
    """Calculate irradiance weighted cell temperature.

    Parameters
    ----------
    poa : Series
        POA irradiance (W/m^2).
    cell_temp : Series
        Cell temperature for each interval (degrees C).

    Returns
    -------
    float
        Average irradiance-weighted cell temperature.
    """
    return (poa * cell_temp).sum() / poa.sum()


"""DIRECTLY BELOW DRAFT PR FUNCTION TO DO ALL VERSIONS OF CALC
DECIDED TO BREAK INTO SMALLER FUNCTIONS, LEAVE TEMPORARILY"""


def perf_ratio_inputs_ok(ac_energy, dc_nameplate, poa, availability=1):
    """Check types of perf_ratio arguments.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (Wh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    availability : numeric or Series, default 1
        Apply an adjustment for plant availability to the expected power
        (denominator).
    """
    if not isinstance(ac_energy, pd.Series):
        warnings.warn("ac_energy must be a Pandas Series.")
        return False
    elif not isinstance(poa, pd.Series):
        warnings.warn("poa must be a Pandas Series.")
        return False
    elif not ac_energy.index.equals(poa.index):
        warnings.warn("indices of poa and ac_energy must match.")
        return False
    elif isinstance(availability, pd.Series):
        if not availability.index.equals(poa.index):
            warnings.warn(
                "Index of availability must match the index of "
                "the poa and ac_energy."
            )
            return False
        else:
            return True
    else:
        return True


def perf_ratio(
    ac_energy, dc_nameplate, poa, unit_adj=1, degradation=0, year=1, availability=1,
):
    """Calculate performance ratio.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (Wh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    unit_adj : numeric, default 1
        Scale factor to adjust units of `ac_energy`. For exmaple pass 1000
        to convert measured energy from kWh to Wh within PR calculation.
    degradation : numeric, default None
        Apply a derate (percent, Ex: 0.5%) for degradation to the expected
        power (denominator). Must also pass specify a value for the `year`
        argument.
        NOTE: Percent is divided by 100 to convert to decimal within function.
    year : numeric
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default 1
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    PrResults
        Instance of class PrResults.
    """
    if not perf_ratio_inputs_ok(
        ac_energy, dc_nameplate, poa, availability=availability
    ):
        return

    timestep = get_common_timestep(poa, units="h", string_output=False)
    timestep_str = get_common_timestep(poa, units="h", string_output=True)

    expected_dc = (
        availability
        * dc_nameplate
        * poa
        / 1000
        * (1 - degradation / 100) ** year
        * timestep
    )
    pr = ac_energy.sum() * unit_adj / expected_dc.sum()

    input_cd = capdata.CapData("input_cd")
    input_cd.data = pd.concat([poa, ac_energy], axis=1)

    pr_per_timestep = ac_energy * unit_adj / expected_dc
    results_data = pd.concat([ac_energy, expected_dc, pr_per_timestep], axis=1)
    results_data.columns = ["ac_energy", "expected_dc", "pr_per_timestep"]

    results = PrResults(
        timestep=(timestep, timestep_str),
        pr=pr,
        dc_nameplate=dc_nameplate,
        input_data=input_cd,
        results_data=results_data,
    )
    return results


def perf_ratio_temp_corr_nrel(
    ac_energy,
    dc_nameplate,
    poa,
    power_temp_coeff=None,
    temp_amb=None,
    wind_speed=None,
    base_temp=25,
    module_type="glass_cell_poly",
    racking="open_rack",
    unit_adj=1,
    degradation=None,
    year=None,
    availability=1,
):
    """Calculate performance ratio.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (kWh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    power_temp_coeff : numeric, default None
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    temp_amb : Series
        Ambient temperature (degrees C) measurements.
    wind_speed : Series
        Measured wind speed (m/sec) corrected to measurement height of
        10 meters.
    base_temp : numeric, default 25
        Base temperature (in Celsius) to correct power to. Default is the
        STC of 25 degrees Celsius. The NREL Weather-Corrected Performance
        Ratio technical report uses the term 'Tcell_typ_avg' for this value.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'
    unit_adj : numeric, default 1
        Scale factor to adjust units of `ac_energy`. For exmaple pass 1000
        to convert measured energy from kWh to Wh within PR calculation.
    degradation : numeric, default None
        NOT IMPLEMENTED
        Apply a derate for degradation to the expected power (denominator).
        Must also pass specify a value for the `year` argument.
    year : numeric
        NOT IMPLEMENTED
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default 1
        NOT IMPLEMENTED
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    """
    timestep = get_common_timestep(poa, units="h", string_output=False)
    timestep_str = get_common_timestep(poa, units="h", string_output=True)

    temp_bom = back_of_module_temp(poa, temp_amb, wind_speed, module_type, racking)
    temp_cell = cell_temp(temp_bom, poa, module_type, racking)
    dc_nameplate_temp_corr = temp_correct_power(
        dc_nameplate, power_temp_coeff, temp_cell, base_temp=base_temp
    )
    # below is same as the perf_ratio function
    # move to a separate function?
    expected_dc = (
        availability
        * dc_nameplate_temp_corr
        * poa
        / 1000
        # * (1 - degradation / 100)**year
        * timestep
    )
    pr = ac_energy.sum() * unit_adj / expected_dc.sum()

    input_cd = capdata.CapData("input_cd")
    input_cd.data = pd.concat([poa, ac_energy], axis=1)

    pr_per_timestep = ac_energy * unit_adj / expected_dc
    results_data = pd.concat([ac_energy, expected_dc, pr_per_timestep], axis=1)
    results_data.columns = ["ac_energy", "expected_dc", "pr_per_timestep"]

    results = PrResults(
        timestep=(timestep, timestep_str),
        pr=pr,
        dc_nameplate=dc_nameplate,
        input_data=input_cd,
        results_data=results_data,
    )
    return results


class PrResults(param.Parameterized):
    """
    Results from a PR calculation.
    """

    dc_nameplate = param.Number(
        bounds=(0, None),
        doc=(
            "Summation of nameplate ratings (W) for all installed modules" " of system."
        ),
    )
    pr = param.Number(doc="Performance ratio result decimal fraction.")
    timestep = param.Tuple(doc="Timestep of series.")
    expected_pr = param.Number(
        bounds=(0, 1), doc="Expected Performance ratio result decimal fraction."
    )
    input_data = param.ClassSelector(capdata.CapData)
    results_data = param.ClassSelector(pd.DataFrame)

    def print_pr_result(self):
        """Print summary of PR result - passing / failing and by how much
        """
        if self.pr >= self.expected_pr:
            print(
                "The test is PASSING with a measured PR of {:.2f}, "
                "which is {:.2f} above the expected PR of {:.2f}".format(
                    self.pr * 100,
                    (self.pr - self.expected_pr) * 100,
                    self.expected_pr * 100,
                )
            )
        else:
            print(
                "The test is FAILING with a measured PR of {:.2f}, "
                "which is {:.2f} below the expected PR of {:.2f}".format(
                    self.pr * 100,
                    (self.expected_pr - self.pr) * 100,
                    self.expected_pr * 100,
                )
            )
