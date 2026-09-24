"""Builders behind the CapData fixtures in ``conftest.py``.

Plain functions so the oracle-capture script and the fixtures share one
definition of each synthetic dataset.
"""

import warnings

import numpy as np
import pandas as pd

from captest import columngroups as cg
from captest.capdata import CapData
from captest.captest import CapTest
from captest.io import load_pvsyst


def build_meas_default():
    """Measured CapData from the example csv with a synthetic rear-POA group."""
    cd = CapData("meas")
    df = pd.read_csv(
        "./tests/data/example_measured_data.csv", index_col=0, parse_dates=True
    )
    df["met1_rpoa"] = df["met1_poa_pyranometer"] * 0.15
    df["met2_rpoa"] = df["met2_poa_pyranometer"] * 0.15
    cd.data = df
    cd.column_groups = cg.ColumnGroups(
        {
            "real_pwr_mtr": ["meter_power"],
            "irr_poa": ["met1_poa_pyranometer", "met2_poa_pyranometer"],
            "irr_rpoa": ["met1_rpoa", "met2_rpoa"],
            "temp_amb": ["met1_amb_temp", "met2_amb_temp"],
            "wind_speed": ["met1_windspeed", "met2_windspeed"],
        }
    )
    return cd


def build_sim_default():
    """PVsyst CapData with synthetic ``GlobBak`` / ``BackShd`` columns."""
    cd = load_pvsyst(path="./tests/data/pvsyst_example_HourlyRes_2.CSV")
    cd.data["GlobBak"] = cd.data["GlobInc"] * 0.15
    cd.data["BackShd"] = 0.0
    return cd


def add_bom_temp(cd):
    """Add a synthetic ``temp_bom`` group (ambient + 0.025 * POA)."""
    df = cd.data
    df["met1_bom_temp"] = df["met1_amb_temp"] + df["met1_poa_pyranometer"] * 0.025
    df["met2_bom_temp"] = df["met2_amb_temp"] + df["met2_poa_pyranometer"] * 0.025
    cd.data = df
    groups = dict(cd.column_groups)
    groups["temp_bom"] = ["met1_bom_temp", "met2_bom_temp"]
    cd.column_groups = cg.ColumnGroups(groups)
    return cd


def add_spec_corrected(cd):
    """Add humidity and pressure groups plus a ``site`` dict."""
    rng = np.random.default_rng(seed=42)
    n = cd.data.shape[0]
    cd.data["met1_humidity"] = np.clip(rng.normal(60.0, 10.0, n), 5.0, 95.0)
    cd.data["met2_humidity"] = np.clip(rng.normal(60.0, 10.0, n), 5.0, 95.0)
    cd.data["met1_pressure"] = rng.normal(1013.0, 3.0, n)
    cd.data["met2_pressure"] = rng.normal(1013.0, 3.0, n)
    groups = dict(cd.column_groups)
    groups["humidity"] = ["met1_humidity", "met2_humidity"]
    groups["pressure"] = ["met1_pressure", "met2_pressure"]
    cd.column_groups = cg.ColumnGroups(groups)
    cd.site = {
        "loc": {
            "latitude": 33.0,
            "longitude": -99.5,
            "altitude": 500,
            "tz": "America/Chicago",
        },
        "sys": {"surface_tilt": 20, "surface_azimuth": 180, "albedo": 0.2},
    }
    return cd


def add_precwat(cd):
    """Add a synthetic ``PrecWat`` column (metres) to a PVsyst CapData."""
    rng = np.random.default_rng(seed=43)
    cd.data["PrecWat"] = rng.uniform(0.005, 0.03, cd.data.shape[0])
    return cd


def _meas_bom():
    return add_bom_temp(build_meas_default())


def _meas_spec():
    return add_spec_corrected(build_meas_default())


def _sim_spec():
    return add_precwat(build_sim_default())


_BASE = {"ac_nameplate": 6_000_000, "test_tolerance": "- 4"}
_BIFI = {**_BASE, "bifaciality": 0.15}
_TC = {**_BIFI, "power_temp_coeff": -0.32, "base_temp": 25}

#: preset -> (meas builder, sim builder, CapTest.from_params kwargs)
PRESET_FIXTURES = {
    "e2848_default": (build_meas_default, build_sim_default, _BASE),
    "bifi_e2848_etotal_rear_shade_sim": (build_meas_default, build_sim_default, _BIFI),
    "bifi_e2848_etotal_rear_shade_meas": (
        build_meas_default,
        build_sim_default,
        _BIFI,
    ),
    "bifi_power_tc_meas_tbom": (_meas_bom, build_sim_default, _TC),
    "bifi_power_tc_calc_tbom": (build_meas_default, build_sim_default, _TC),
    "bifi_power_tc_etotal_rear_shade_sim": (_meas_bom, build_sim_default, _TC),
    "bifi_power_tc_etotal_rear_shade_meas": (_meas_bom, build_sim_default, _TC),
    "e2848_spec_corrected_poa": (_meas_spec, _sim_spec, _BASE),
    "bifi_e2848_etotal_rear_shade_sim_spec_corrected": (_meas_spec, _sim_spec, _BIFI),
    "bifi_e2848_etotal_rear_shade_meas_spec_corrected": (
        _meas_spec,
        _sim_spec,
        _BIFI,
    ),
}


def build_captest(preset):
    """Build and set up the CapTest for ``preset`` on its fixture data."""
    meas_builder, sim_builder, kwargs = PRESET_FIXTURES[preset]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Propagating meas.site")
        return CapTest.from_params(
            test_setup=preset, meas=meas_builder(), sim=sim_builder(), **kwargs
        )


def snapshot(tst):
    """Numbers a setup computes, independent of the grammar that produced them.

    Per side: the column each regression variable resolved to, the sum and
    mean of that column, and the unfiltered regression coefficients and
    p-values.
    """
    out = {}
    for side in ("meas", "sim"):
        cd = getattr(tst, side)
        cols = {}
        for var, column in cd.regression_cols.items():
            series = cd.data[column]
            cols[var] = {
                "column": column,
                "sum": float(series.sum()),
                "mean": float(series.mean()),
            }
        cd.fit_regression(filter=False, summary=False)
        res = cd.regression_results
        out[side] = {
            "regression_cols": cols,
            "params": {k: float(v) for k, v in res.params.items()},
            "pvalues": {k: float(v) for k, v in res.pvalues.items()},
        }
    return out
