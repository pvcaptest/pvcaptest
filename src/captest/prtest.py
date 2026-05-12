import warnings

import numpy as np
import pandas as pd
import param

from captest import capdata
from captest import util
from captest import calcparams

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
                "Index of availability must match the index of the poa and ac_energy."
            )
            return False
        else:
            return True
    else:
        return True


def perf_ratio(
    ac_energy,
    dc_nameplate,
    poa,
    unit_adj=1,
    degradation=0,
    year=1,
    availability=1,
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

    timestep = util.get_common_timestep(poa, units="h", string_output=False)
    timestep_str = util.get_common_timestep(poa, units="h", string_output=True)

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
    temp_bom=None,
    temp_amb=None,
    single_irr_weighted_temp=False,
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
    temp_bom : Series
        Measured back of module temperature. The `temp_amb` and `wind_speed` arguments
        are not used if this argument is not None; skips calculating BOM temps from
        ambient temperature, wind speed, and POA irradiance.
    single_irr_weighted_temp : bool, default False
        Set to True to calculate a single irradiance weighted temperature to use
        when temperature correcting the power. Some contract language calls for this
        but it does not follow the calculation defined in the NREL paper.
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
    timestep = util.get_common_timestep(poa, units="h", string_output=False)
    timestep_str = util.get_common_timestep(poa, units="h", string_output=True)

    coeffs = calcparams.EMP_HEAT_COEFF[racking][module_type]
    if temp_bom is None:
        temp_bom = poa * np.exp(coeffs["a"] + coeffs["b"] * wind_speed) + temp_amb
    temp_cell = temp_bom + (poa / 1000) * coeffs["del_tcnd"]
    if single_irr_weighted_temp:
        temp_cell = (poa * temp_cell).sum() / poa.sum()
    dc_nameplate_temp_corr = dc_nameplate / (
        1 + ((power_temp_coeff / 100) * (temp_cell - base_temp))
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
        doc=("Summation of nameplate ratings (W) for all installed modules of system."),
    )
    pr = param.Number(doc="Performance ratio result decimal fraction.")
    timestep = param.Tuple(doc="Timestep of series.")
    expected_pr = param.Number(
        bounds=(0, 1), doc="Expected Performance ratio result decimal fraction."
    )
    input_data = param.ClassSelector(class_=capdata.CapData)
    results_data = param.ClassSelector(class_=pd.DataFrame)

    def print_pr_result(self):
        """Print summary of PR result - passing / failing and by how much"""
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
