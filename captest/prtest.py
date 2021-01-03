from captest import capdata
import numpy as np
import pandas as pd

emp_heat_coeff = {
    'open_rack': {
        'glass_cell_glass': {
            'a': -3.47,
            'b': -0.0594,
            'del_tcnd': 3
        },
        'glass_cell_poly': {
            'a': -3.56,
            'b': -0.0750,
            'del_tcnd': 3
        },
        'poly_tf_steel': {
            'a': -3.58,
            'b': -0.1130,
            'del_tcnd': 3
        },
    },
    'close_roof_mount': {
        'glass_cell_glass': {
            'a': -2.98,
            'b': -0.0471,
            'del_tcnd': 1
        }
    },
    'insulated_back': {
        'glass_cell_poly': {
            'a': -2.81,
            'b': -0.0455,
            'del_tcnd': 0
        }
    }
}


def temp_correct_power(power, power_temp_coeff, cell_temp, base_temp=25):
    """Apply temperature correction to PV power.

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
    corr_power = power * (1 - (power_temp_coeff / 100) * (base_temp - cell_temp))
    return corr_power


def back_of_module_temp(
    poa,
    temp_amb,
    wind_speed,
    module_type='glass_cell_poly',
    racking='open_rack'
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
    a = emp_heat_coeff[racking][module_type]['a']
    b = emp_heat_coeff[racking][module_type]['b']
    return poa * np.exp(a + b * wind_speed) + temp_amb


"""
************************************************************************
********** BELOW FUNCTIONS ARE NOT FULLY IMPLEMENTED / TESTED **********
************************************************************************
"""
def cell_temp(
    df,
    bom="Tm",
    poa_col=None,
    wspd_col=None,
    ambt_col=None,
    a=-3.56,
    b=-0.0 - 0.0750,
    del_tcnd=3,
    tcell_col="Tcell",
):
    """Calculate cell temperature using thermal model presented in NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    bom : string, default 'Tm'
        Column of back of module temperature.  Default Tm uses NREL thermal model to calculate BOM temperature.
        This option can be used to specify a column of measured BOM temperatures if desired.
    """
    df[bom] = (
        df.loc[:, poa_col] * np.exp(a + b * df.loc[:, wspd_col]) + df.loc[:, ambt_col]
    )
    df[tcell_col] = df.loc[:, bom] + df.loc[:, poa_col] / 1000 * del_tcnd
    return df


def cell_typ_avg(df, poa_col=None, cellt_col=None):
    df["poa_cellt"] = df.loc[:, poa_col] * df.loc[:, cellt_col]
    cell_typ_avg = df.loc[:, "poa_cellt"].sum() / df.loc[:, poa_col].sum()
    return (cell_typ_avg, df)


def pr_test_temp_corr(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    ac_energy_col=None,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
    avail=1,
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or intc
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df = df.copy()
    df[en_dc] = (
        avail
        * DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (Tcell_typ_avg - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col].sum() * unit_adj / df[en_dc].sum() * 100


def pr_test(
    df,
    DC_nameplate,
    poa_col=None,
    ac_energy_col=None,
    degradation=0.005,
    year=1,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or intc
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df = df.copy()
    df[en_dc] = (
        DC_nameplate * (df[poa_col] / 1000) * (1 - degradation) ** year * timestep
    )
    return ((df[ac_energy_col].sum() * unit_adj) / df[en_dc].sum()) * 100


def pr_test_pertstep(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    ac_energy_col=None,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or int
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df[en_dc] = (
        DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (Tcell_typ_avg - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col] * unit_adj / df[en_dc] * 100


def apply_pr_test(df):
    df = cell_temp(df, poa_col="GlobInc", wspd_col="WindVel", ambt_col="TAmb")
    cell_typavg, df = cell_typ_avg(df, poa_col="GlobInc", cellt_col="Tcell")
    pr_exp = pr_test(
        df,
        cell_typavg,
        2754000,
        -0.37,
        poa_col="GlobInc",
        ac_energy_col="E_Grid",
        timestep=1,
    )
    return pr_exp, cell_typavg


# Using the annual cell_typ_avg value
def apply_pr_test_meas_annual(df):
    df = cell_temp(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        wspd_col="Rainwise Weather Station - WIND SPEED (SENSOR 1)",
        ambt_col="Rainwise Weather Station - AMBIENT TEMPERATURE (SENSOR 1)",
    )
    cell_typavg, df = cell_typ_avg(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        cellt_col="Tcell",
    )
    # annual_cell_typavg is from tmy data above
    pr_exp = pr_test(
        df,
        annual_cell_typavg,
        2754000,
        -0.37,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        ac_energy_col="PV Meter - ACTIVE POWER",
        timestep=1,
        unit_adj=1000,
    )
    return pr_exp


## example calculating PRs by week
# weekly_prs_annual_tcell_typ_avg = pd.DataFrame(meas.groupby(pd.Grouper(freq='W')).apply(apply_pr_test_meas_annual))
## example calculating PRs be month
# monthly_prs_annual_tcell_typ_avg = pd.DataFrame(meas.groupby(pd.Grouper(freq='MS')).apply(apply_pr_test_meas_annual))


def pr_test_monthly(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    tcell_typ_avg_col="tcell_typ_avg_monthly",
    ac_energy_col=None,
    timestep=1,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or int
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df[en_dc] = (
        DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (df[tcell_typ_avg_col] - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col].sum() * unit_adj / df[en_dc].sum() * 100


# Using the annual cell_typ_avg value
def apply_pr_test_meas_monthly(df):
    df = cell_temp(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        wspd_col="Rainwise Weather Station - WIND SPEED (SENSOR 1)",
        ambt_col="Rainwise Weather Station - AMBIENT TEMPERATURE (SENSOR 1)",
    )
    cell_typavg, df = cell_typ_avg(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        cellt_col="Tcell",
    )
    # annual_cell_typavg is from tmy data above
    pr_exp = pr_test_monthly(
        df,
        annual_cell_typavg,
        2754000,
        -0.37,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        ac_energy_col="PV Meter - ACTIVE POWER",
        timestep=1,
        unit_adj=1000,
    )
    return pr_exp
