"""Unified test orchestrator and supporting utilities.

This module houses the ``CapTest`` class, the ``TEST_SETUPS`` registry of
named regression presets, the ``CapTestResults`` results container, and small
formatting helpers (``highlight_pvals``, ``perc_wrap``) consumed by
``CapTest`` methods that compare a measured + modeled pair of ``CapData``
instances.

Import direction
----------------
At module-import time the dependency is one-way only:
``captest.captest`` -> ``captest.capdata``. ``CapData`` is imported here at
module scope so ``CapTest`` can declare ``meas``/``sim`` as
``param.ClassSelector(class_=CapData)``. ``captest.capdata`` does NOT import
anything from this module at import time; the single-CapData helper
``predict_with_pvalue_check`` is imported lazily from within
``CapTest.captest_results``.
"""

import copy
import difflib
import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import param
import yaml

from captest import util
from captest.util import (
    _perc_wrap_to_string,
    _resolve_func_strings,
    perc_wrap,
    to_native,
)
from captest.capdata import CapData
from captest.filters import wrap_year_end
from captest.plotting import ScatterBifiPowerTc, ScatterPlot
from captest.calcparams import (
    absolute_airmass,
    apparent_zenith,
    apparent_zenith_pvsyst,
    bom_temp,
    cell_temp,
    e_total,
    poa_spec_corrected,
    power_temp_correct,
    precipitable_water_gueymard,
    rpoa_pvsyst,
    scale,
    spectral_factor_firstsolar,
)

_hv_spec = importlib.util.find_spec("holoviews")
if _hv_spec is not None:
    import holoviews as hv
else:  # pragma: no cover - optional dep
    hv = None


def highlight_pvals(s):
    """Highlight Series entries >= 0.05 with a yellow background.

    Intended for use with ``pandas.io.formats.style.Styler.apply``. Consumed
    by ``CapTestResults.styled_pvalues``.
    """
    is_greaterthan = s >= 0.05
    return ["background-color: yellow" if v else "" for v in is_greaterthan]


@dataclass
class CapTestResults:
    """Structured results of a measured-vs-modeled capacity test.

    Returned by :meth:`CapTest.captest_results`. ``str(results)`` (or
    :meth:`summary`) reproduces the legacy printed report;
    :meth:`styled_pvalues` reproduces the legacy p-value Styler.

    Attributes
    ----------
    cap_ratio : float
        Headline capacity test ratio ``actual / expected`` — the ratio the
        pass/fail decision was made on. P-value-checked when the test ran
        with ``check_pvalues=True`` (see ``pvalues_checked``), otherwise
        computed without p-value filtering.
    cap_ratio_pval_check : float
        Capacity ratio computed with above-threshold coefficients zeroed.
    passed : bool
        Pass/fail result for the headline ratio against ``tolerance``.
    tolerance : str
        The ``CapTest.test_tolerance`` string the test was judged against.
    bounds : str
        Human-readable capacity bounds string for the tolerance.
    expected_capacity : float
        Predicted modeled test output at reporting conditions (headline
        variant; see ``pvalues_checked``).
    actual_capacity : float
        Predicted measured test output at reporting conditions (headline
        variant; see ``pvalues_checked``).
    tested_capacity : float
        ``ac_nameplate`` times the headline capacity ratio.
    points_used : dict
        Points remaining after filtering, keyed by ``'meas'`` / ``'sim'``.
    regression_tables : dict
        Per-side DataFrames of regression terms with ``coef`` and ``pvalue``
        columns, keyed by ``'meas'`` / ``'sim'``.
    rc : pandas.DataFrame
        The reporting conditions both regressions were predicted at.
    rc_source : str
        Provenance of ``rc`` (``'meas'``, ``'sim'``, or ``'manual'``).
    pvalues_checked : bool
        Which variant is the headline: ``True`` when ``cap_ratio``,
        ``actual_capacity``, ``expected_capacity``, and the pass/fail
        decision used the p-value-checked predictions
        (``check_pvalues=True``), ``False`` for the plain predictions.
    """

    cap_ratio: float
    cap_ratio_pval_check: float
    passed: bool
    tolerance: str
    bounds: str
    expected_capacity: float
    actual_capacity: float
    tested_capacity: float
    points_used: dict
    regression_tables: dict
    rc: pd.DataFrame
    rc_source: str
    pvalues_checked: bool = False

    def summary(self):
        """Return the legacy printed report as a string."""
        result = "PASS" if self.passed else "FAIL"
        lines = [
            f"Using reporting conditions from {self.rc_source}. \n",
            "{:<30s}{}".format("Capacity Test Result:", result),
            "{:<30s}{:0.3f}".format("Modeled test output:", self.expected_capacity),
            "{:<30s}{:0.3f}".format("Actual test output:", self.actual_capacity),
            "{:<30s}{:0.3f}".format("Tested output ratio:", self.cap_ratio),
            "{:<30s}{:0.3f}".format("Tested Capacity:", self.tested_capacity),
            "{:<30s}{}\n".format("Bounds:", self.bounds),
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def styled_pvalues(self):
        """Return the legacy p-value/params Styler built from this object.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with p-values and coefficients for both sides;
            p-values >= 0.05 are highlighted.
        """
        df_pvals = pd.DataFrame(
            {
                "das_pvals": self.regression_tables["meas"]["pvalue"],
                "sim_pvals": self.regression_tables["sim"]["pvalue"],
                "das_params": self.regression_tables["meas"]["coef"],
                "sim_params": self.regression_tables["sim"]["coef"],
            }
        )
        return df_pvals.style.format("{:20,.5f}").apply(
            highlight_pvals, subset=["das_pvals", "sim_pvals"]
        )


# --- TEST_SETUPS registry -------------------------------------------------


def scatter_default(cd, **kwargs):
    """Formula-agnostic scatter of regression lhs vs. first rhs variable.

    Thin wrapper around
    :class:`captest.plotting.ScatterPlot`. Forwards every keyword argument
    through to the class constructor, so callers can opt into the
    AM/PM split, temperature-corrected power, and timeseries-pairing
    features without changing call sites.

    Parameters
    ----------
    cd : CapData
        Must have ``regression_formula`` set and ``regression_cols``
        resolved (e.g. via ``CapTest.setup()`` or
        ``cd.process_regression_columns()``).
    **kwargs
        Forwarded to :class:`ScatterPlot`. See its docstring for the full
        parameter surface.

    Returns
    -------
    hv.Layout
        A single-panel Layout wrapping the scatter plot.
    """
    return ScatterPlot(cd=cd, **kwargs).view()


def scatter_etotal(cd, **kwargs):
    """Single scatter of regression lhs vs. the ``e_total`` column.

    Intended for the ``bifi_e2848_etotal_rear_shade_sim`` /
    ``bifi_e2848_etotal_rear_shade_meas`` presets. Thin wrapper around
    :class:`captest.plotting.ScatterPlot`; resolves the x column from
    ``cd.regression_cols['poa']`` after ``process_regression_columns``
    has materialized the calculated e_total column.
    """
    return ScatterPlot(cd=cd, **kwargs).view()


def scatter_bifi_power_tc(cd, **kwargs):
    """Two-panel layout: lhs vs. ``poa`` and lhs vs. ``rpoa``.

    Intended for the ``bifi_power_tc`` preset whose regression formula is
    ``power ~ poa + rpoa`` (with ``power`` resolved to the
    temperature-corrected calculated column). Thin wrapper around
    :class:`captest.plotting.ScatterBifiPowerTc`; each rhs variable gets
    its own panel.
    """
    return ScatterBifiPowerTc(cd=cd, **kwargs).view()


TEST_SETUPS = {
    "e2848_default": {
        "description": (
            "Standard ASTM E2848 regression of AC power against front-side POA irradiance, "
            "ambient temperature, and wind speed using the full four-term formula. "
            "This is the default setup for monofacial capacity tests."
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": ("irr_poa", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": "GlobInc",
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_default,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal_rear_shade_sim": {
        "description": (
            "Standard ASTM E2848 regression form with total effective "
            "irradiance replacing front-side POA as the independent variable. "
            "Rear shading and IAM losses are handled in the modeled (PVsyst) "
            "data: the modeled rear irradiance is rpoa_pvsyst = GlobBak + "
            "BackShd, while the measured rear sensor (irr_rpoa) is used "
            "as-measured (no rear_shade factor, i.e. rear_shade = 0). For the "
            "variant that instead applies rear shading on the measured side, "
            "see 'bifi_e2848_etotal_rear_shade_meas'. Total irradiance is "
            "E_Total = E_POA + E_Rear * bifaciality, following the NREL "
            "modified bifacial approach."
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                e_total,
                {
                    "poa": ("irr_poa", "mean"),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                e_total,
                {
                    "poa": "GlobInc",
                    "rpoa": (
                        rpoa_pvsyst,
                        {"globbak": "GlobBak", "backshd": "BackShd"},
                    ),
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal_rear_shade_meas": {
        "description": (
            "Variant of 'bifi_e2848_etotal_rear_shade_sim' for applying "
            "rear-shading losses on the measured side. The modeled rear "
            "irradiance maps directly to PVsyst's unshaded global rear "
            "('GlobBak') instead of rpoa_pvsyst, and rear shading is applied "
            "to the measured side through the e_total 'rear_shade' factor "
            "(propagated by CapTest to the measured CapData only). Total "
            "irradiance is E_Total = E_POA + E_Rear * bifaciality * "
            "(1 - rear_shade)."
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                e_total,
                {
                    "poa": ("irr_poa", "mean"),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                e_total,
                {
                    "poa": "GlobInc",
                    "rpoa": "GlobBak",
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_power_tc_meas_tbom": {
        "description": (
            "The regression equation is temperature corrected power regressed against "
            "front POA and rear POA. The back of module temperature is from field "
            "measurements and the cell temperature is calculated using the Sandia PV Array "
            "Performance Model from the POA irradiance and measured BOM temperature. "
            "Note that the PVsyst temperature correction uses the 'TArray' output variable."
        ),
        "reg_cols_meas": {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": ("temp_bom", "mean"),
                        },
                    ),
                },
            ),
            "poa": ("irr_poa", "mean"),
            "rpoa": ("irr_rpoa", "mean"),
        },
        "reg_cols_sim": {
            "power": (
                power_temp_correct,
                {
                    "power": "E_Grid",
                    "cell_temp": "TArray",
                },
            ),
            "poa": "GlobInc",
            "rpoa": (rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"}),
        },
        "reg_fml": "power ~ poa + rpoa",
        "scatter_plots": scatter_bifi_power_tc,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "rpoa": "mean",
            },
        },
    },
    "bifi_power_tc_calc_tbom": {
        "description": (
            "The regression equation is temperature corrected power regressed against "
            "front POA and rear POA. The back of module and cell temperature are "
            "calculated using the Sandia PV Array Performance Model from the POA "
            "irradiance, ambient temperature, and wind speed. Note that the PVsyst "
            "temperature correction uses the 'TArray' output variable."
        ),
        "reg_cols_meas": {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": (
                                bom_temp,
                                {
                                    "poa": ("irr_poa", "mean"),
                                    "temp_amb": ("temp_amb", "mean"),
                                    "wind_speed": ("wind_speed", "mean"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "poa": ("irr_poa", "mean"),
            "rpoa": ("irr_rpoa", "mean"),
        },
        "reg_cols_sim": {
            "power": (
                power_temp_correct,
                {
                    "power": "E_Grid",
                    "cell_temp": "TArray",
                },
            ),
            "poa": "GlobInc",
            "rpoa": (rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"}),
        },
        "reg_fml": "power ~ poa + rpoa",
        "scatter_plots": scatter_bifi_power_tc,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "rpoa": "mean",
            },
        },
    },
    "bifi_power_tc_etotal_rear_shade_sim": {
        "description": (
            "The regression equation is temperature corrected power regressed against "
            "total POA irradiance. The back of module temperature is from field "
            "measurements and the cell temperature is calculated using the Sandia PV Array "
            "Performance Model from the POA irradiance and measured BOM temperature. "
            "Note that the PVsyst temperature correction uses the 'TArray' output variable. "
            "Rear shading and IAM losses are handled in the modeled (PVsyst) "
            "data: the modeled rear irradiance is rpoa_pvsyst = GlobBak + "
            "BackShd, while the measured rear sensor (irr_rpoa) is used "
            "as-measured (no rear_shade factor, i.e. rear_shade = 0). For the "
            "variant that instead applies rear shading on the measured side, "
            "see 'bifi_power_tc_etotal_rear_shade_meas'. Total irradiance is "
            "E_Total = E_POA + E_Rear * bifaciality, following the NREL "
            "modified bifacial approach."
        ),
        "reg_cols_meas": {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": ("temp_bom", "mean"),
                        },
                    ),
                },
            ),
            "poa": (
                e_total,
                {
                    "poa": ("irr_poa", "mean"),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
        },
        "reg_cols_sim": {
            "power": (
                power_temp_correct,
                {
                    "power": "E_Grid",
                    "cell_temp": "TArray",
                },
            ),
            "poa": (
                e_total,
                {
                    "poa": "GlobInc",
                    "rpoa": (
                        rpoa_pvsyst,
                        {"globbak": "GlobBak", "backshd": "BackShd"},
                    ),
                },
            ),
        },
        "reg_fml": "power ~ poa",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
            },
        },
    },
    "bifi_power_tc_etotal_rear_shade_meas": {
        "description": (
            "The regression equation is temperature corrected power regressed against "
            "total POA irradiance. The back of module temperature is from field "
            "measurements and the cell temperature is calculated using the Sandia PV Array "
            "Performance Model from the POA irradiance and measured BOM temperature. "
            "Note that the PVsyst temperature correction uses the 'TArray' output variable. "
            "Variant of 'bifi_power_tc_etotal_rear_shade_sim' for applying "
            "rear-shading losses on the measured side. The modeled rear "
            "irradiance maps directly to PVsyst's unshaded global rear "
            "('GlobBak') instead of rpoa_pvsyst, and rear shading is applied "
            "to the measured side through the e_total 'rear_shade' factor "
            "(propagated by CapTest to the measured CapData only). Total "
            "irradiance is E_Total = E_POA + E_Rear * bifaciality * "
            "(1 - rear_shade)."
        ),
        "reg_cols_meas": {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": ("temp_bom", "mean"),
                        },
                    ),
                },
            ),
            "poa": (
                e_total,
                {
                    "poa": ("irr_poa", "mean"),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
        },
        "reg_cols_sim": {
            "power": (
                power_temp_correct,
                {
                    "power": "E_Grid",
                    "cell_temp": "TArray",
                },
            ),
            "poa": (
                e_total,
                {
                    "poa": "GlobInc",
                    "rpoa": "GlobBak",
                },
            ),
        },
        "reg_fml": "power ~ poa",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
            },
        },
    },
    "e2848_spec_corrected_poa": {
        "description": (
            "Standard ASTM E2848 regression with a First Solar spectral correction applied to "
            "front-side POA irradiance before fitting. Requires relative humidity and atmospheric "
            "pressure on the measured side and precipitable water from the PVsyst output."
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                poa_spec_corrected,
                {
                    "poa": ("irr_poa", "mean"),
                    "spectral_correction": (
                        spectral_factor_firstsolar,
                        {
                            "precipitable_water": (
                                precipitable_water_gueymard,
                                {
                                    "temp_amb": ("temp_amb", "mean"),
                                    "rel_humidity": ("humidity", "mean"),
                                },
                            ),
                            "absolute_airmass": (
                                absolute_airmass,
                                {
                                    "apparent_zenith": (
                                        apparent_zenith,
                                        {},
                                    ),
                                    "pressure": ("pressure", "mean"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                poa_spec_corrected,
                {
                    "poa": "GlobInc",
                    "spectral_correction": (
                        spectral_factor_firstsolar,
                        {
                            "precipitable_water": (
                                scale,
                                {"col": "PrecWat", "factor": 100},
                            ),
                            "absolute_airmass": (
                                absolute_airmass,
                                {
                                    "apparent_zenith": (
                                        apparent_zenith_pvsyst,
                                        {},
                                    ),
                                },
                            ),
                        },
                    ),
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_default,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal_rear_shade_sim_spec_corrected": {
        "description": (
            "Standard ASTM E2848 regression with total effective irradiance replacing "
            "front POA and a First Solar spectral correction applied to "
            "front-side POA irradiance used to calculate the total POA irradiance. "
            "Requires relative humidity and atmospheric "
            "pressure on the measured side and precipitable water from the PVsyst output. "
            "Rear shading and IAM losses are handled in the modeled (PVsyst) "
            "data: the modeled rear irradiance is rpoa_pvsyst = GlobBak + "
            "BackShd, while the measured rear sensor (irr_rpoa) is used "
            "as-measured (no rear_shade factor, i.e. rear_shade = 0). For the "
            "variant that instead applies rear shading on the measured side, "
            "see 'bifi_e2848_etotal_rear_shade_meas_spec_corrected'. Total irradiance is "
            "E_Total = E_POA + E_Rear * bifaciality with the spectral correction "
            "applied to E_POA. "
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                e_total,
                {
                    "poa": (  # front POA: spectrally corrected
                        poa_spec_corrected,
                        {
                            "poa": ("irr_poa", "mean"),
                            "spectral_correction": (
                                spectral_factor_firstsolar,
                                {
                                    "precipitable_water": (
                                        precipitable_water_gueymard,
                                        {
                                            "temp_amb": ("temp_amb", "mean"),
                                            "rel_humidity": ("humidity", "mean"),
                                        },
                                    ),
                                    "absolute_airmass": (
                                        absolute_airmass,
                                        {
                                            "apparent_zenith": (apparent_zenith, {}),
                                            "pressure": ("pressure", "mean"),
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                    "rpoa": ("irr_rpoa", "mean"),  # rear POA: unchanged
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                e_total,
                {
                    "poa": (  # front POA: spectrally corrected
                        poa_spec_corrected,
                        {
                            "poa": "GlobInc",
                            "spectral_correction": (
                                spectral_factor_firstsolar,
                                {
                                    "precipitable_water": (
                                        scale,
                                        {"col": "PrecWat", "factor": 100},
                                    ),
                                    "absolute_airmass": (
                                        absolute_airmass,
                                        {
                                            "apparent_zenith": (
                                                apparent_zenith_pvsyst,
                                                {},
                                            ),
                                            # no pressure col: PVsyst uses sea-level default
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                    "rpoa": (  # rear POA: unchanged
                        rpoa_pvsyst,
                        {"globbak": "GlobBak", "backshd": "BackShd"},
                    ),
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal_rear_shade_meas_spec_corrected": {
        "description": (
            "Standard ASTM E2848 regression with total effective irradiance replacing "
            "front POA and a First Solar spectral correction applied to "
            "front-side POA irradiance used to calculate the total POA irradiance. "
            "Requires relative humidity and atmospheric "
            "pressure on the measured side and precipitable water from the PVsyst output. "
            "The modeled rear irradiance maps directly to PVsyst's unshaded global rear "
            "('GlobBak') instead of rpoa_pvsyst, and rear shading is applied "
            "to the measured side through the e_total 'rear_shade' factor "
            "(propagated by CapTest to the measured CapData only). Total "
            "irradiance is E_Total = E_POA + E_Rear * bifaciality * (1 - rear_shade)."
        ),
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                e_total,
                {
                    "poa": (
                        poa_spec_corrected,
                        {
                            "poa": ("irr_poa", "mean"),
                            "spectral_correction": (
                                spectral_factor_firstsolar,
                                {
                                    "precipitable_water": (
                                        precipitable_water_gueymard,
                                        {
                                            "temp_amb": ("temp_amb", "mean"),
                                            "rel_humidity": ("humidity", "mean"),
                                        },
                                    ),
                                    "absolute_airmass": (
                                        absolute_airmass,
                                        {
                                            "apparent_zenith": (apparent_zenith, {}),
                                            "pressure": ("pressure", "mean"),
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                e_total,
                {
                    "poa": (  # front POA: spectrally corrected
                        poa_spec_corrected,
                        {
                            "poa": "GlobInc",
                            "spectral_correction": (
                                spectral_factor_firstsolar,
                                {
                                    "precipitable_water": (
                                        scale,
                                        {"col": "PrecWat", "factor": 100},
                                    ),
                                    "absolute_airmass": (
                                        absolute_airmass,
                                        {
                                            "apparent_zenith": (
                                                apparent_zenith_pvsyst,
                                                {},
                                            ),
                                            # no pressure col: PVsyst uses sea-level default
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                    "rpoa": "GlobBak",
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
}

_TEST_SETUP_REQUIRED_KEYS = frozenset(
    {
        "description",
        "reg_cols_meas",
        "reg_cols_sim",
        "reg_fml",
        "scatter_plots",
        "rep_conditions",
    }
)


def test_setups(options=True, descriptions=False):
    """
    Display test setups available.

    Parameters
    ----------
    options: bool, default True
        List the names of the test setups.
    descriptions: bool, default False
        List the descriptions of the test setups.

    Returns
    -------
    None
    """
    if options:
        print("All options")
        print("=" * 60)
        for name in TEST_SETUPS.keys():
            print(name)

    if descriptions:
        if options:
            print("\n\n")
        print("Descriptions")
        print("=" * 60)
        for name, setup in TEST_SETUPS.items():
            print("\n")
            print(f"{name}")
            print("-" * 60)
            print(textwrap.fill(setup["description"], 60))


def validate_test_setup(entry):
    """Validate a single ``TEST_SETUPS`` entry dict.

    Raises
    ------
    KeyError
        If required keys are missing or unknown keys are present.
    ValueError
        If ``reg_fml`` does not parse, lhs+rhs are not subsets of both
        ``reg_cols_meas`` and ``reg_cols_sim``, ``scatter_plots`` is not
        callable, or ``rep_conditions`` / ``rep_conditions['func']`` have an
        unexpected shape.
    """
    keys = set(entry.keys())
    missing = _TEST_SETUP_REQUIRED_KEYS - keys
    if missing:
        raise KeyError(f"TEST_SETUPS entry missing required keys: {sorted(missing)}")
    extra = keys - _TEST_SETUP_REQUIRED_KEYS
    if extra:
        raise KeyError(f"TEST_SETUPS entry has unknown keys: {sorted(extra)}")

    lhs, rhs = util.parse_regression_formula(entry["reg_fml"])
    formula_vars = set(lhs) | set(rhs)
    for side in ("reg_cols_meas", "reg_cols_sim"):
        if not isinstance(entry[side], dict):
            raise ValueError(f"{side!r} must be a dict.")
        missing_vars = formula_vars - set(entry[side].keys())
        if missing_vars:
            raise ValueError(
                f"{side!r} is missing keys required by reg_fml: {sorted(missing_vars)}"
            )

    if not callable(entry["scatter_plots"]):
        raise ValueError("'scatter_plots' must be callable.")

    rc = entry["rep_conditions"]
    if not isinstance(rc, dict):
        raise ValueError("'rep_conditions' must be a dict.")
    func = rc.get("func")
    if func is not None and isinstance(func, dict):
        extra_func = set(func.keys()) - set(rhs)
        if extra_func:
            raise ValueError(
                "'rep_conditions[\"func\"]' has keys that are not rhs "
                f"variables of reg_fml: {sorted(extra_func)}"
            )


def _merge_rep_conditions(base, override):
    """Partial-merge ``override`` onto ``base`` rep_conditions dict.

    Top-level keys in ``override`` replace corresponding keys in ``base``.
    If both have ``func`` dicts, the ``override['func']`` is merged one level
    deep (per-variable) onto ``base['func']``.
    """
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, val in override.items():
        if (
            key == "func"
            and isinstance(val, dict)
            and isinstance(merged.get("func"), dict)
        ):
            merged_func = copy.deepcopy(merged["func"])
            merged_func.update(val)
            merged["func"] = merged_func
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def resolve_test_setup(name, overrides=None):
    """Resolve a preset by name plus optional overrides.

    Parameters
    ----------
    name : str
        Key into ``TEST_SETUPS`` or the literal ``"custom"``.
    overrides : dict or None
        Optional dict with any of ``reg_cols_meas``, ``reg_cols_sim``,
        ``reg_fml``, ``scatter_plots``, ``rep_conditions`` to override the
        preset. ``rep_conditions`` is partial-merged; other keys replace.
        When ``name == "custom"``, ``reg_cols_meas``, ``reg_cols_sim``, and
        ``reg_fml`` are required in ``overrides``.

    Returns
    -------
    dict
        A fully-validated entry dict suitable for ``CapTest._resolved_setup``.
    """
    overrides = overrides or {}
    if name == "custom":
        required = {"reg_cols_meas", "reg_cols_sim", "reg_fml"}
        missing = required - set(overrides.keys())
        if missing:
            raise ValueError(
                f"test_setup='custom' requires overrides with keys: {sorted(required)}; "
                f"missing: {sorted(missing)}"
            )
        base = {
            "description": overrides.get("description", ""),
            "reg_cols_meas": copy.deepcopy(overrides["reg_cols_meas"]),
            "reg_cols_sim": copy.deepcopy(overrides["reg_cols_sim"]),
            "reg_fml": overrides["reg_fml"],
            "scatter_plots": overrides.get("scatter_plots", scatter_default),
            "rep_conditions": copy.deepcopy(overrides.get("rep_conditions", {})),
        }
    else:
        if name not in TEST_SETUPS:
            available = sorted(TEST_SETUPS.keys()) + ["custom"]
            raise KeyError(f"Unknown test_setup={name!r}. Available: {available}")
        base = copy.deepcopy(TEST_SETUPS[name])
        for key in ("reg_cols_meas", "reg_cols_sim", "reg_fml", "scatter_plots"):
            if overrides.get(key) is not None:
                base[key] = copy.deepcopy(overrides[key])
        if overrides.get("rep_conditions"):
            base["rep_conditions"] = _merge_rep_conditions(
                base["rep_conditions"], overrides["rep_conditions"]
            )

    validate_test_setup(base)
    return base


# --- yaml loading ---------------------------------------------------------


def _serialize_rep_conditions(rc):
    """Return a yaml-safe copy of a ``rep_conditions`` dict.

    Recursively walks the dict; ``func`` sub-dict values that are
    ``perc_wrap(N)`` callables are converted to ``"perc_N"`` strings, and
    numpy scalars are coerced to native Python types via ``util.to_native``,
    so the dict survives a yaml.safe_dump round-trip.
    """
    if not isinstance(rc, dict):
        return rc
    serialized = {}
    for key, val in rc.items():
        if key == "func" and isinstance(val, dict):
            serialized[key] = {k: _perc_wrap_to_string(v) for k, v in val.items()}
        else:
            serialized[key] = to_native(copy.deepcopy(val))
    return serialized


_AUTO_WRAP_DAYS = 60


def load_config(path, key="captest"):
    """Load and lightly validate the captest sub-mapping from a yaml file.

    Parameters
    ----------
    path : str or Path
        Path to the yaml file. Relative paths in ``meas_path`` / ``sim_path``
        are resolved by callers using ``Path(path).parent`` as the base.
    key : str, default 'captest'
        Top-level key whose value is the CapTest configuration sub-mapping.

    Returns
    -------
    dict
        The sub-mapping at ``key`` with string shorthands resolved. Does NOT
        validate against ``CapTest`` param types; ``CapTest.from_yaml`` does
        that.

    Raises
    ------
    KeyError
        If ``key`` is not present at the top level of the yaml file.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Top level of yaml file {path!s} must be a mapping; got {type(raw).__name__}."
        )
    if key not in raw:
        available = sorted(raw.keys())
        suggestion = difflib.get_close_matches(key, available, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(
            f"Top-level key {key!r} not found in {path!s}. "
            f"Top-level keys present: {available}.{hint}"
        )
    sub = raw[key]
    if not isinstance(sub, dict):
        raise ValueError(
            f"Value at {key!r} must be a mapping; got {type(sub).__name__}."
        )
    # Resolve perc_N shorthand in overrides.rep_conditions.func.
    overrides = sub.get("overrides") or {}
    if isinstance(overrides, dict) and isinstance(
        overrides.get("rep_conditions"), dict
    ):
        func_dict = overrides["rep_conditions"].get("func")
        if isinstance(func_dict, dict):
            overrides["rep_conditions"]["func"] = _resolve_func_strings(func_dict)
    # Also resolve top-level rep_conditions.func if someone put it there.
    rc = sub.get("rep_conditions")
    if isinstance(rc, dict) and isinstance(rc.get("func"), dict):
        rc["func"] = _resolve_func_strings(rc["func"])
    return sub


def _suggest_unknown_key(unknown, known):
    """Return a 'did you mean X?' hint or empty string."""
    matches = difflib.get_close_matches(unknown, list(known), n=1)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _is_uri_or_absolute_path(val):
    """Return True if ``val`` should be treated as an absolute location.

    A string is "absolute" in this context if it either:

    * carries a URI scheme (e.g. ``s3://bucket/key``, ``gs://...``,
      ``file:///...``) -- ``"://"`` substring check, or
    * is an absolute filesystem path per :meth:`pathlib.Path.is_absolute`.

    The scheme check is required because on posix systems
    ``Path("s3://bucket/key").is_absolute()`` returns False (the colon
    becomes part of the first path component), so relying on Path alone
    would incorrectly treat S3 URIs as relative and mangle them during
    path joining.
    """
    s = str(val)
    if "://" in s:
        return True
    return Path(s).is_absolute()


def _join_base_and_relative(base_dir, relative):
    """Join a relative path to a base directory, preserving URI schemes.

    Local ``base_dir`` values are joined via :class:`pathlib.Path`.
    URI-scheme ``base_dir`` values (e.g. ``s3://bucket/prefix``) are
    joined by string concatenation because ``Path("s3://...")`` mangles
    the double slash after the scheme.
    """
    base_str = str(base_dir)
    if "://" in base_str:
        return base_str.rstrip("/") + "/" + str(relative).lstrip("/")
    return str(Path(base_str) / relative)


# --- CapTest class --------------------------------------------------------

# Keys of ``captest.captest.CapTest`` params that may appear directly under the
# yaml captest sub-mapping. Used by ``from_yaml`` for unknown-key detection.
_CAPTEST_YAML_KEYS = frozenset(
    {
        "test_setup",
        "reg_fml",
        "reg_cols_meas",
        "reg_cols_sim",
        "rep_conditions",
        "rc_source",
        "sim_days",
        "shade_filter_start",
        "shade_filter_end",
        "ac_nameplate",
        "inv_ac_nameplate",
        "test_tolerance",
        "min_irr",
        "max_irr",
        "clipping_irr",
        "rep_irr_filter",
        "fshdbm",
        "irrad_stability",
        "irrad_stability_threshold",
        "hrs_req",
        "bifaciality",
        "bifacial_frac",
        "rear_shade",
        "power_temp_coeff",
        "base_temp",
        "module_type",
        "racking",
        "spectral_module_type",
        "airmass_model",
        "altitude_override",
        "meas_load_kwargs",
        "sim_load_kwargs",
        "meas_path",
        "sim_path",
        "overrides",
        "meas_filters",
        "sim_filters",
        "reporting_conditions_values",
    }
)

# Keys that may appear under the ``overrides`` sub-mapping.
_CAPTEST_OVERRIDE_KEYS = frozenset(
    {"reg_cols_meas", "reg_cols_sim", "reg_fml", "rep_conditions"}
)

# Keys whose ``None`` (yaml ``null``) value is a distinct, meaningful value
# rather than a request to fall back to the param default. These are NOT
# stripped by ``from_mapping`` so the value round-trips through ``to_yaml`` /
# ``from_yaml``. ``altitude_override`` is the only such key: it has a non-None
# default (0, the sea-level convention) but allows ``None`` to mean "respect
# the site's own altitude".
_CAPTEST_NONE_MEANINGFUL_KEYS = frozenset({"altitude_override"})


def _default_meas_loader():
    """Return the default measured-data loader (``captest.io.load_data``).

    Imported lazily so that callers who construct ``CapTest`` without
    supplying a ``meas_path`` do not need the ``io`` submodule and its
    transitive dependencies loaded.
    """
    from captest.io import load_data

    return load_data


def _default_sim_loader():
    """Return the default modeled-data loader (``captest.io.load_pvsyst``).

    Lazy-imported for the same reason as ``_default_meas_loader``.
    """
    from captest.io import load_pvsyst

    return load_pvsyst


class CapTest(param.Parameterized):
    """Config + state container for an ASTM E2848 capacity test.

    ``CapTest`` binds a measured ``CapData`` and a modeled ``CapData`` to a
    named regression preset from ``TEST_SETUPS`` and holds all test-level
    configuration in one place. It is intentionally a config + state
    container rather than a runner: users still invoke
    ``ct.meas.filter_*(...)``, ``ct.meas.rep_cond(...)``, and
    ``ct.meas.fit_regression()`` by hand.

    Typical workflows
    -----------------
    1. Programmatic::

        ct = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd,
            sim=sim_cd,
            ac_nameplate=125_000,
            test_tolerance="- 4",
        )
        # ``from_params`` runs ``setup()`` automatically because both meas
        # and sim were supplied as pre-built CapData instances.

    2. From a yaml file::

        ct = CapTest.from_yaml("./config.yaml")

    3. Bare + manual::

        ct = CapTest(test_setup="bifi_e2848_etotal_rear_shade_sim", bifaciality=0.15)
        ct.meas = my_meas_cd
        ct.sim = my_sim_cd
        ct.setup()

    Parameters
    ----------
    meas : CapData or None
        Measured-data ``CapData`` instance. Assigned via ``from_params``,
        ``from_yaml``, or directly.
    sim : CapData or None
        Modeled-data ``CapData`` instance.
    test_setup : str
        Key into ``TEST_SETUPS`` or the literal ``"custom"``. Default
        ``"e2848_default"``.
    reg_fml : str or None
        If set, overrides the preset's regression formula at ``setup()``.
    reg_cols_meas : dict or None
        If set, overrides the preset's measured ``regression_cols`` dict.
    reg_cols_sim : dict or None
        If set, overrides the preset's modeled ``regression_cols`` dict.
    rep_conditions : dict or None
        If set, partial-merged onto the preset's ``rep_conditions`` at
        ``setup()``. Top-level keys replace; the nested ``func`` dict is
        merged one level deep so users can override only a single
        variable's aggregation.
    rc_source : {"meas", "sim"}
        Which ``CapData`` provides reporting conditions. Used by
        ``captest_results`` and wired onto both ``meas`` and ``sim`` at
        ``setup()`` so ``filter_irr(ref_val='rep_irr')`` resolves against the
        same instance regardless of which dataset is being filtered. Default
        ``"meas"``.
    sim_days : int
        Days of simulated data used for the test. Default 30.
    shade_filter_start, shade_filter_end : str or None
        ``"HH:MM"`` between-time strings for shade filtering.
    ac_nameplate : float or None
        Nameplate AC power in watts.
    test_tolerance : str
        Tolerance string forwarded to pass/fail logic. Default ``"- 4"``.
    min_irr, max_irr, clipping_irr : float
        Irradiance filter bounds (W/m^2).
    rep_irr_filter : float
        Fractional reporting-irradiance filter band in ``[0, 1]``.
    fshdbm : float
        Shade filter threshold in ``[0, 1]``.
    irrad_stability : {"std", "filter_clearsky", "contract"}
        Irradiance stability strategy.
    irrad_stability_threshold : float
        Threshold value for ``irrad_stability``.
    hrs_req : float
        Hours of data required for a complete test. Default 12.5.
    bifaciality, bifacial_frac, power_temp_coeff, base_temp, altitude_override : float
        Numeric calc-params scalars propagated to both CapData instances at
        setup(). See ``_downstream_attrs``.
    module_type, racking, airmass_model, spectral_module_type : str
        String calc-params options propagated to both CapData instances at
        setup(). ``module_type``/``racking`` feed the Sandia temperature
        model (``calcparams.bom_temp`` / ``calcparams.cell_temp``);
        ``airmass_model`` feeds ``calcparams.absolute_airmass``;
        ``spectral_module_type`` feeds
        ``calcparams.spectral_factor_firstsolar``.
    rear_shade : float
        Fraction of rear irradiance lost to shading, propagated to the
        measured CapData instance only (see ``_downstream_attrs_meas_only``).
        Applied by ``calcparams.e_total`` on the measured side; the modeled
        side handles rear shading through its own ``reg_cols_sim`` definition.
    meas_loader, sim_loader : callable or None
        Programmatic-only data-loader callables. Default resolution when
        ``None``: ``captest.io.load_data`` and ``captest.io.load_pvsyst``
        respectively. Not serialized to yaml.
    meas_load_kwargs, sim_load_kwargs : dict or None
        Plain-dict kwargs splatted into the loaders.

    Attributes
    ----------
    _resolved_setup : dict or None
        The fully-resolved ``TEST_SETUPS`` entry after ``setup()`` has run.
        Plain instance attribute (not a ``param.*``) so ``setup()`` can be
        called multiple times.
    rep_irr_filter_low : float
        Read-only. Lower irradiance fraction bound derived from
        ``rep_irr_filter``: ``1 - rep_irr_filter``. For example, when
        ``rep_irr_filter=0.2`` this is ``0.8``. Pass as ``low`` to
        ``CapData.filter_irr`` together with a ``ref_val``.
    rep_irr_filter_high : float
        Read-only. Upper irradiance fraction bound derived from
        ``rep_irr_filter``: ``1 + rep_irr_filter``. For example, when
        ``rep_irr_filter=0.2`` this is ``1.2``. Pass as ``high`` to
        ``CapData.filter_irr`` together with a ``ref_val``.

    Notes
    -----
    The lhs key of the regression formula is always ``"power"`` across
    shipped presets, even when the formula regresses a derived quantity
    (e.g. temperature-corrected power).
    """

    # --- parameter declarations ------------------------------------------

    # Bound CapData instances
    meas = param.ClassSelector(
        class_=CapData, default=None, doc="Measured CapData instance."
    )
    sim = param.ClassSelector(
        class_=CapData, default=None, doc="Modeled CapData instance."
    )

    # Regression setup
    test_setup = param.String(
        default="e2848_default",
        doc="Key into TEST_SETUPS or the literal 'custom'.",
    )
    reg_fml = param.String(
        default=None,
        allow_None=True,
        doc="If set, overrides the preset regression formula.",
    )
    reg_cols_meas = param.Dict(
        default=None,
        allow_None=True,
        doc="If set, overrides the preset measured regression_cols dict.",
    )
    reg_cols_sim = param.Dict(
        default=None,
        allow_None=True,
        doc="If set, overrides the preset modeled regression_cols dict.",
    )
    rep_conditions = param.Dict(
        default=None,
        allow_None=True,
        doc="If set, partial-merged onto the preset rep_conditions at setup().",
    )
    rc_source = param.Selector(
        objects=["meas", "sim", "manual"],
        default="meas",
        doc="Provenance of the single test RC (CapTest.rc): 'meas'/'sim' when "
        "computed from that dataset's rep_cond, or 'manual' when set directly. "
        "Seeds the default 'which' for rep_cond. This is a provenance label "
        "managed alongside CapTest.rc by the sanctioned mutation paths "
        "(rep_cond / the ct.rc setter, both routed through _set_rc) and is "
        "accepted as a construction-time config input; assigning it directly "
        "afterward relabels provenance without changing the stored rc and is "
        "not recommended.",
    )

    # Test scope / time
    sim_days = param.Integer(
        default=30,
        bounds=(1, 365),
        doc="Days of simulated data used for the test.",
    )
    shade_filter_start = param.String(
        default=None,
        allow_None=True,
        doc="HH:MM start time for between-time shade filtering.",
    )
    shade_filter_end = param.String(
        default=None,
        allow_None=True,
        doc="HH:MM end time for between-time shade filtering.",
    )

    # Measurement / nameplate
    ac_nameplate = param.Number(
        default=None,
        allow_None=True,
        doc="Nameplate AC power in W.",
    )
    inv_ac_nameplate = param.Number(
        default=None,
        allow_None=True,
        bounds=(0, None),
        doc="Per-inverter AC nameplate rating, kW. Plant metadata and a "
        "prefill source for per-inverter clipping filters; never a hidden "
        "input to results (serialized filter steps record resolved "
        "thresholds).",
    )
    test_tolerance = param.String(
        default="- 4",
        doc="Tolerance string forwarded to pass/fail logic.",
    )

    auto_wrap_sim = param.Boolean(
        default=True,
        doc="When True, automatically apply wrap_year_end to sim.data during "
        "setup() if measured data is within 60 days of a year boundary. "
        "Set False to opt out and restore any prior auto-wrap.",
    )

    # Filter parameters
    min_irr = param.Number(default=400, doc="Minimum POA irradiance (W/m^2).")
    max_irr = param.Number(default=1400, doc="Maximum POA irradiance (W/m^2).")
    clipping_irr = param.Number(
        default=1000, doc="POA irradiance threshold for clipping filter (W/m^2)."
    )
    rep_irr_filter = param.Number(
        default=0.2,
        bounds=(0.0, 1.0),
        doc="Fractional reporting-irradiance filter band.",
    )
    fshdbm = param.Number(
        default=1.0,
        bounds=(0.0, 1.0),
        doc="Shade filter threshold (fraction).",
    )
    irrad_stability = param.Selector(
        objects=["std", "filter_clearsky", "contract"],
        default="std",
        doc="Irradiance stability strategy.",
    )
    irrad_stability_threshold = param.Number(
        default=30,
        doc="Threshold value for irradiance stability.",
    )
    hrs_req = param.Number(
        default=12.5,
        doc="Hours of data required for a complete test.",
    )

    # Calc-params scalars propagated to the CapData instances at setup().
    # All are propagated onto both meas and sim except rear_shade, which is
    # meas-only (see _downstream_attrs_meas_only).
    bifaciality = param.Number(
        default=0.0,
        bounds=(0.0, 1.0),
        doc="Bifaciality factor propagated onto both CapData instances.",
    )
    bifacial_frac = param.Number(
        default=1.0,
        bounds=(0.0, 1.0),
        doc=(
            "Fraction of array nameplate power that is bifacial, passed to "
            "calcparams.e_total. Propagated onto both CapData instances."
        ),
    )
    rear_shade = param.Number(
        default=0.0,
        bounds=(0.0, 1.0),
        doc=(
            "Fraction of rear irradiance lost due to shading, passed to "
            "calcparams.e_total. Propagated onto the measured CapData "
            "instance only (see _downstream_attrs_meas_only)."
        ),
    )
    power_temp_coeff = param.Number(
        default=-0.32,
        doc="Power temperature coefficient (percent per degree C).",
    )
    base_temp = param.Number(
        default=25,
        doc="Base temperature for temperature correction (deg C).",
    )
    module_type = param.String(
        default="glass_cell_poly",
        doc=(
            "Module construction passed to the Sandia temperature model via "
            "calcparams.bom_temp and calcparams.cell_temp. One of "
            "'glass_cell_poly', 'glass_cell_glass', or 'poly_tf_steel'. "
            "Propagated onto both CapData instances at setup(). Distinct from "
            "spectral_module_type, which feeds "
            "calcparams.spectral_factor_firstsolar."
        ),
    )
    racking = param.String(
        default="open_rack",
        doc=(
            "Racking configuration passed to the Sandia temperature model via "
            "calcparams.bom_temp and calcparams.cell_temp. One of 'open_rack', "
            "'close_roof_mount', or 'insulated_back'. Propagated onto both "
            "CapData instances at setup()."
        ),
    )
    spectral_module_type = param.String(
        default="cdte",
        doc=(
            "Module type passed to pvlib.spectrum.spectral_factor_firstsolar "
            "via calcparams.spectral_factor_firstsolar. Propagated onto both "
            "CapData instances at setup() so it is auto-injected by "
            "CapData.custom_param. Named to avoid collision with the "
            "'module_type' kwarg of calcparams.bom_temp and "
            "calcparams.cell_temp."
        ),
    )
    airmass_model = param.String(
        default="kastenyoung1989",
        doc=(
            "Relative airmass model passed to calcparams.absolute_airmass "
            "(pvlib.atmosphere.get_relative_airmass). Propagated onto both "
            "CapData instances at setup()."
        ),
    )
    altitude_override = param.Number(
        default=0,
        allow_None=True,
        doc=(
            "Altitude (m) used when building the pvlib.Location in "
            "calcparams.apparent_zenith / apparent_zenith_pvsyst. Defaults to "
            "0 (sea level) per the First Solar spectral-correction reference; "
            "set to None to respect the site's own altitude. Propagated onto "
            "both CapData instances at setup()."
        ),
    )

    # Data-loader injection (programmatic-only; never serialized to yaml).
    meas_loader = param.Callable(
        default=None,
        allow_None=True,
        doc="Callable used to build meas from meas_path. Defaults to load_data.",
    )
    meas_load_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Extra kwargs splatted into meas_loader.",
    )
    sim_loader = param.Callable(
        default=None,
        allow_None=True,
        doc="Callable used to build sim from sim_path. Defaults to load_pvsyst.",
    )
    sim_load_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Extra kwargs splatted into sim_loader.",
    )

    # Class-level tuple of param names to copy onto the CapData instances
    # during setup(). Names also listed in _downstream_attrs_meas_only are
    # copied onto meas only; all others are copied onto both meas and sim.
    # Extending is a one-line edit. Invariant: every name in
    # _downstream_attrs_meas_only MUST also appear in _downstream_attrs, since
    # setup() iterates _downstream_attrs (guarded by a subset test).
    _downstream_attrs = (
        "bifaciality",
        "bifacial_frac",
        "rear_shade",
        "power_temp_coeff",
        "base_temp",
        "module_type",
        "racking",
        "spectral_module_type",
        "airmass_model",
        "altitude_override",
    )
    _downstream_attrs_meas_only = ("rear_shade",)

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        # Plain instance attr rather than a param.* so setup() can be re-run.
        self._resolved_setup = None
        # Construction-time paths. Not ``param.*`` because they are strings
        # that only matter for ``to_yaml`` round-trip; tracking them here
        # lets ``from_params``/``from_yaml`` remember what paths the class
        # was built from without cluttering the param surface.
        self._meas_path = None
        self._sim_path = None
        # The single test reporting-conditions DataFrame (or None). Plain attr,
        # not a param.*, so the `rc` property setter can validate and the
        # `_set_rc` write point can manage provenance. `_loading` is True only
        # during run_test pipeline replay with rc_source='manual', to keep the
        # manual RC authoritative.
        self._rc = None
        self._loading = False
        # Serialized filter pipelines stored at load (from_mapping) and not
        # yet applied; consumed by run_test (spec R2). Plain lists of
        # filter-config dicts, public so users can inspect or edit them.
        self.meas_filters_pending = []
        self.sim_filters_pending = []
        # Manual reporting-conditions values stashed at load when setup()
        # has not run yet; consumed by the next full setup() (spec R1).
        self._pending_manual_rc = None
        # Transient set of sides ('meas'/'sim') whose pipeline re-run is still
        # ahead of an RC write; those sides are excluded from the RC-staleness
        # warning in `_set_rc`. Registered by orchestrated replays (run_test)
        # and always cleared in a `finally`, so it never outlives the call.
        self._rc_pending_sides = set()

    @property
    def rc(self):
        """The single test reporting-conditions DataFrame, or ``None``.

        Sourced from ``meas``/``sim`` via :meth:`rep_cond` or set manually via
        the property setter; provenance is tracked by :attr:`rc_source`. See the
        RC-ownership design spec for the full lifecycle.
        """
        return self._rc

    @rc.setter
    def rc(self, value):
        """Set the test reporting conditions manually (``rc_source='manual'``).

        This is the only public way to supply reporting conditions directly —
        e.g. for sensitivity analysis or to check results against a reviewing
        party's values. Computed conditions go through :meth:`rep_cond` instead.

        Parameters
        ----------
        value : pandas.DataFrame or pandas.Series or dict
            One-row reporting conditions. A Series or dict maps each regression
            variable to its value; a DataFrame is used as given. Must provide a
            value for every right-hand-side variable of the (shared meas/sim)
            regression formula. Extra columns are preserved.

        Raises
        ------
        RuntimeError
            If ``meas`` or ``sim`` is missing, or lacks a regression formula.
        ValueError
            If ``meas`` and ``sim`` have different regression formulas, if
            ``value`` coerces to more than one row, or if ``value`` omits a
            required right-hand-side variable.
        TypeError
            If ``value`` is not a DataFrame, Series, or dict.
        """
        self._set_rc(self._coerce_and_validate_manual_rc(value), "manual")

    def _coerce_and_validate_manual_rc(self, value):
        """Validate and coerce a candidate manual reporting-conditions value.

        Shared by the public :attr:`rc` setter and the ``from_mapping`` load
        path so that a hand-edited YAML with a missing required regression
        variable fails fast (at load) rather than silently at predict/rep_irr.

        Parameters
        ----------
        value : pandas.DataFrame or pandas.Series or dict
            Candidate reporting conditions. A Series or dict maps each
            regression variable to its value; a DataFrame is used as given.
            Must supply a value for every right-hand-side variable of the
            (shared meas/sim) regression formula.

        Returns
        -------
        pandas.DataFrame
            A validated, one-row reporting-conditions DataFrame.

        Raises
        ------
        RuntimeError
            If ``meas`` or ``sim`` is missing, or lacks a regression formula.
        ValueError
            If ``meas`` and ``sim`` have different regression formulas, if
            ``value`` coerces to more than one row, or if ``value`` omits a
            required right-hand-side variable.
        TypeError
            If ``value`` is not a DataFrame, Series, or dict.
        """
        self._require_regression_formula()
        meas_fml = self.meas.regression_formula
        sim_fml = self.sim.regression_formula
        if meas_fml != sim_fml:
            raise ValueError(
                "Cannot set reporting conditions manually: meas and sim have "
                f"different regression formulas ({meas_fml!r} vs {sim_fml!r})."
            )
        _, rhs = util.parse_regression_formula(meas_fml)
        if isinstance(value, pd.DataFrame):
            df = value.copy()
        elif isinstance(value, pd.Series):
            df = value.to_frame().T
        elif isinstance(value, dict):
            df = pd.DataFrame([value])
        else:
            raise TypeError(
                "ct.rc must be a one-row DataFrame, a pandas Series, or a dict "
                f"mapping regression variable -> value; got "
                f"{type(value).__name__}."
            )
        if len(df) != 1:
            raise ValueError(
                f"Reporting conditions must be a single row; got {len(df)} rows."
            )
        missing = [var for var in rhs if var not in df.columns]
        if missing:
            raise ValueError(
                "Manual reporting conditions are missing required regression "
                f"variable(s): {missing}. Required: {rhs}."
            )
        return df

    def _set_rc(self, rc, source, warn=True):
        """Single internal write point for ``_rc`` and ``rc_source``.

        With ``warn`` True and an RC already set, emits at most ONE
        ``UserWarning`` per write, merging (a) a source-change notice when
        ``source`` differs from the current ``rc_source`` and (b) an
        RC-staleness notice naming applied RC-dependent steps
        (``ref_val`` of ``'rep_irr'``/``'self_val'``) that resolved against
        the previous RC and are not excluded — sides in
        ``self._rc_pending_sides`` (registered by ``run_test`` for chains it
        is about to re-run) are excluded. Silent on first establishment and
        on a same-source write of an unchanged RC. ``warn=False`` (config
        load) suppresses both.

        Parameters
        ----------
        rc : pandas.DataFrame
            One-row reporting-conditions DataFrame.
        source : {'meas', 'sim', 'manual'}
            Provenance to record in ``rc_source``.
        warn : bool, default True
            Suppress the merged warning when False (used during load).
        """
        if warn and self._rc is not None:
            parts = []
            if source != self.rc_source:
                parts.append(
                    f"Test reporting conditions rc_source changed from "
                    f"'{self.rc_source}' to '{source}'."
                )
            if not self._rc.equals(rc):
                stale = self._stale_rc_dependent_steps()
                if stale:
                    parts.append(
                        "The test reporting conditions changed; these applied "
                        "filter steps resolved against the previous reporting "
                        "conditions and must be re-run: " + ", ".join(stale) + "."
                    )
            if parts:
                warnings.warn(" ".join(parts))
        self._rc = rc
        self.rc_source = source

    def _stale_rc_dependent_steps(self):
        """Applied steps whose ``ref_val`` resolves against the test RC.

        Scans both sides' applied chains for steps configured with
        ``ref_val`` in ``{'rep_irr', 'self_val'}`` (the param preserves the
        user's original token), skipping sides registered in
        ``self._rc_pending_sides``. Returns display labels like
        ``"sim.filters[2] (Irradiance)"``.
        """
        stale = []
        for side in ("meas", "sim"):
            if side in self._rc_pending_sides:
                continue
            cd = getattr(self, side)
            if cd is None:
                continue
            for i, step in enumerate(cd.filters):
                if getattr(step, "ref_val", None) in ("rep_irr", "self_val"):
                    stale.append(f"{side}.filters[{i}] ({type(step).__name__})")
        return stale

    def _on_capdata_rep_cond(self, cd):
        """Update the test RC after a member CapData computed its own ``rc``.

        Called by :meth:`CapData._calc_rep_cond` when the CapData belongs to
        this test. Behavior is last-writer-wins: the calling side's ``rc``
        becomes ``ct.rc`` and ``rc_source`` (a source-change ``UserWarning``
        is emitted by :meth:`_set_rc`). ``_loading`` exists solely for
        ``run_test``'s manual-RC replay: with ``rc_source='manual'`` the
        manual reporting conditions stay authoritative, so propagation from
        replayed RepCond steps is suppressed entirely (the step still
        computes that side's local ``cd.rc``).

        Parameters
        ----------
        cd : CapData
            The member CapData that just (re)computed its ``rc``.
        """
        if self._loading:
            return
        side = "meas" if cd is self.meas else "sim"
        self._set_rc(cd.rc.copy(), side, warn=True)

    # --- constructors ----------------------------------------------------

    @classmethod
    def from_params(cls, run_setup=True, **kwargs):
        """Construct a CapTest from parameter kwargs.

        Recognizes the non-param kwargs ``meas``, ``sim``, ``meas_path``,
        ``sim_path`` in addition to every declared ``param.*``. If both
        ``meas`` and ``meas_path`` are supplied the pre-built instance
        wins and a warning is emitted (same for ``sim`` / ``sim_path``).

        When both ``meas`` and ``sim`` end up populated and ``run_setup``
        is True, ``setup()`` is called automatically. Otherwise the
        partially-initialized instance is returned and the caller finishes
        the workflow manually.

        Parameters
        ----------
        run_setup : bool, default True
            When False, skip the automatic ``setup()`` even when both
            ``meas`` and ``sim`` are populated (load-only construction).
            Nothing setup produces is present: no scalar propagation, no
            derived-parameter calculation, no regression-column
            processing, no ``_captest`` back-references. A later
            ``ct.setup()`` or ``ct.run_test()`` proceeds normally.
        **kwargs
            Any declared CapTest parameter, plus ``meas``, ``sim``,
            ``meas_path``, ``sim_path``.

        Returns
        -------
        CapTest
        """
        meas = kwargs.pop("meas", None)
        sim = kwargs.pop("sim", None)
        meas_path = kwargs.pop("meas_path", None)
        sim_path = kwargs.pop("sim_path", None)

        inst = cls(**kwargs)
        inst._meas_path = meas_path
        inst._sim_path = sim_path

        # Resolve loaders lazily so tests don't need the io module unless
        # they actually load data from paths.
        def _meas_loader():
            return inst.meas_loader or _default_meas_loader()

        def _sim_loader():
            return inst.sim_loader or _default_sim_loader()

        # Wire up meas.
        if meas is not None and meas_path is not None:
            warnings.warn(
                "Both 'meas' and 'meas_path' supplied; using the pre-built "
                "meas CapData and ignoring meas_path.",
                stacklevel=2,
            )
            inst.meas = meas
        elif meas is not None:
            inst.meas = meas
        elif meas_path is not None:
            load_kwargs = inst.meas_load_kwargs or {}
            inst.meas = _meas_loader()(meas_path, **load_kwargs)

        # Wire up sim.
        if sim is not None and sim_path is not None:
            warnings.warn(
                "Both 'sim' and 'sim_path' supplied; using the pre-built "
                "sim CapData and ignoring sim_path.",
                stacklevel=2,
            )
            inst.sim = sim
        elif sim is not None:
            inst.sim = sim
        elif sim_path is not None:
            load_kwargs = inst.sim_load_kwargs or {}
            inst.sim = _sim_loader()(sim_path, **load_kwargs)

        if run_setup and inst.meas is not None and inst.sim is not None:
            inst.setup()

        return inst

    @classmethod
    def from_yaml(
        cls, path, key="captest", meas_loader=None, sim_loader=None, run_setup=True
    ):
        """Construct a CapTest from a yaml config file.

        Reads the sub-mapping at the given top-level ``key`` of the yaml
        file and delegates to :meth:`from_mapping` with
        ``base_dir=path.parent`` so relative ``meas_path`` / ``sim_path``
        values resolve against the yaml's directory. Serialized filter
        pipelines are stored as :attr:`meas_filters_pending` /
        :attr:`sim_filters_pending`, not applied; run them with
        :meth:`run_test` (or per side via ``CapData.run_pipeline``).

        Parameters
        ----------
        path : str or Path
            Path to a yaml file.
        key : str, default 'captest'
            Top-level key whose value is the CapTest sub-mapping.
        meas_loader, sim_loader : callable or None, optional
            Programmatic-only loader callables that override the default
            resolution (``captest.io.load_data`` / ``captest.io.load_pvsyst``).
            Supplied here because loader callables cannot be represented in
            yaml. Useful for downstream wrappers that drive yaml-based
            construction but need a custom measured-data loader.
            When ``None`` the default resolution applies.
        run_setup : bool, default True
            Forwarded to :meth:`from_mapping`. When False, only the data
            is loaded (see :meth:`from_params`).

        Returns
        -------
        CapTest
        """
        path = Path(path)
        sub = load_config(path, key=key)
        return cls.from_mapping(
            sub,
            key=key,
            base_dir=path.parent,
            meas_loader=meas_loader,
            sim_loader=sim_loader,
            run_setup=run_setup,
        )

    @classmethod
    def from_mapping(
        cls,
        sub,
        *,
        key="captest",
        base_dir=None,
        meas_loader=None,
        sim_loader=None,
        run_setup=True,
    ):
        """Construct a CapTest from an already-parsed captest sub-mapping.

        Direct-handoff constructor used by downstream wrappers that mutate
        the captest sub-mapping in memory -- applying project-specific
        defaults, promoting fields, injecting paths -- before asking captest
        to validate and build the ``CapTest``. Exposes the same
        validate-and-construct pipeline that ``from_yaml`` runs after
        reading the file, without the file read.

        Serialized ``meas_filters`` / ``sim_filters`` pipelines are stored
        as :attr:`meas_filters_pending` / :attr:`sim_filters_pending` —
        nothing is replayed at load. Run them with :meth:`run_test` (or per
        side via ``CapData.run_pipeline``). Manual reporting-conditions
        values (``reporting_conditions_values`` with
        ``rc_source='manual'``) are validated and seeded during the
        construction-time ``setup()``; with ``run_setup=False`` they are
        stashed and consumed by the next full ``setup()``.

        Parameters
        ----------
        sub : dict
            Captest sub-mapping. Typically obtained from
            :func:`load_config` or assembled by a downstream wrapper. Must
            contain ``test_setup``. Supported keys are declared by
            ``_CAPTEST_YAML_KEYS`` / ``_CAPTEST_OVERRIDE_KEYS``. ``sub``
            is not mutated.
        key : str, default 'captest'
            Purely used in error messages (e.g. "Unknown key 'x' under the
            'captest' sub-mapping"). Match the top-level yaml key under
            which this sub-mapping would normally live so error messages
            point users at the right place in their config file.
        base_dir : str, Path, or None, default None
            Base directory used to resolve relative ``meas_path`` /
            ``sim_path`` values in ``sub``. If the sub-mapping contains
            any relative path and ``base_dir`` is ``None``, raises
            ``ValueError``. URI-scheme values in the sub-mapping (e.g.
            ``s3://bucket/path``) are treated as absolute and skip
            resolution even though ``pathlib.Path.is_absolute()`` returns
            False for them. URI-scheme ``base_dir`` values are joined to
            relative paths via string concatenation so the scheme is
            preserved; local ``base_dir`` values are joined via
            :class:`pathlib.Path`.
        meas_loader, sim_loader : callable or None, optional
            Programmatic-only loader callables that override the default
            resolution (``captest.io.load_data`` / ``captest.io.load_pvsyst``).
            Same semantics as :meth:`from_yaml`.
        run_setup : bool, default True
            Forwarded to :meth:`from_params`. When False, only the data
            is loaded — no ``setup()``, nothing seeded (load-only).

        Returns
        -------
        CapTest
        """
        if not isinstance(sub, dict):
            raise TypeError(f"'sub' must be a mapping; got {type(sub).__name__}.")

        # Unknown-key detection with Levenshtein suggestion.
        for k in sub:
            if k not in _CAPTEST_YAML_KEYS:
                suggestion = _suggest_unknown_key(k, _CAPTEST_YAML_KEYS)
                raise ValueError(
                    f"Unknown key {k!r} under the {key!r} sub-mapping.{suggestion}"
                )
        overrides = sub.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("'overrides' must be a mapping.")
        for k in overrides:
            if k not in _CAPTEST_OVERRIDE_KEYS:
                suggestion = _suggest_unknown_key(k, _CAPTEST_OVERRIDE_KEYS)
                raise ValueError(f"Unknown key {k!r} under 'overrides'.{suggestion}")

        if "test_setup" not in sub:
            raise ValueError(f"'test_setup' is required under the {key!r} sub-mapping.")

        # Conflicting reg_fml at the top-level and under overrides.
        if sub.get("reg_fml") is not None and overrides.get("reg_fml") is not None:
            raise ValueError(
                "'reg_fml' cannot be set both at the captest top-level and "
                "under 'overrides'; pick one."
            )

        kwargs = {
            k: v
            for k, v in sub.items()
            if k
            not in (
                "overrides",
                "meas_filters",
                "sim_filters",
                "reporting_conditions_values",
            )
        }

        # Lift override keys into direct kwargs.
        for k in _CAPTEST_OVERRIDE_KEYS:
            if overrides.get(k) is not None:
                kwargs[k] = overrides[k]

        # 'custom' setup requires the three regression overrides.
        if kwargs.get("test_setup") == "custom":
            for req in ("reg_cols_meas", "reg_cols_sim", "reg_fml"):
                if kwargs.get(req) is None:
                    raise ValueError(
                        f"test_setup='custom' requires overrides.{req} to be set."
                    )

        # Resolve relative paths. URI-scheme paths (e.g. s3://) are treated
        # as absolute; Path.is_absolute() alone is not enough because on
        # posix systems Path("s3://...").is_absolute() returns False.
        for path_key in ("meas_path", "sim_path"):
            val = kwargs.get(path_key)
            if val is None:
                continue
            val_str = str(val)
            if _is_uri_or_absolute_path(val_str):
                continue
            if base_dir is None:
                raise ValueError(
                    f"Relative {path_key}={val_str!r} in the {key!r} sub-mapping "
                    f"but no base_dir was supplied to from_mapping. Pass "
                    f"base_dir= explicitly, or use absolute paths / URIs in "
                    f"the mapping."
                )
            kwargs[path_key] = _join_base_and_relative(base_dir, val_str)

        # ``null`` (None) in yaml is equivalent to omitting the key, except
        # for keys where None is a distinct, meaningful value (see
        # _CAPTEST_NONE_MEANINGFUL_KEYS) and must survive the round trip.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None or k in _CAPTEST_NONE_MEANINGFUL_KEYS
        }

        # Inject programmatic-only loader callables. Explicit kwargs win
        # over any value that happened to slip through the sub-mapping
        # (loaders are ``param.Callable`` so yaml would coerce-fail before
        # reaching here, but be defensive).
        if meas_loader is not None:
            kwargs["meas_loader"] = meas_loader
        if sim_loader is not None:
            kwargs["sim_loader"] = sim_loader

        inst = cls.from_params(run_setup=run_setup, **kwargs)
        # Preserve the raw relative-or-absolute paths the user wrote in
        # the sub-mapping so a later ``to_yaml`` round-trips them.
        # ``from_params`` overwrites ``_meas_path`` / ``_sim_path`` with
        # the resolved absolute paths; restore the original literal values
        # here.
        raw_meas_path = sub.get("meas_path")
        raw_sim_path = sub.get("sim_path")
        if raw_meas_path is not None:
            inst._meas_path = raw_meas_path
        if raw_sim_path is not None:
            inst._sim_path = raw_sim_path
        # Serialized filter pipelines are stored pending, never replayed at
        # load; run_test consumes them (spec R2). Manual RC values are seeded
        # by the construction-time setup() when it ran, else stashed for the
        # next full setup().
        meas_filters = sub.get("meas_filters")
        sim_filters = sub.get("sim_filters")
        rc_values = sub.get("reporting_conditions_values")

        def _has_repcond(cfg):
            return any(d.get("type") == "RepCond" for d in (cfg or []))

        # The dual-RepCond ambiguity warning scans the serialized configs,
        # so it fires at load regardless of whether setup() ran.
        if (
            inst.rc_source in ("meas", "sim")
            and _has_repcond(meas_filters)
            and _has_repcond(sim_filters)
        ):
            warnings.warn(
                "Config defines a RepCond step in both meas_filters and "
                "sim_filters with a computed rc_source "
                f"('{inst.rc_source}'): this is ambiguous and unsupported "
                "— on a re-run the non-rc_source side's RepCond will "
                "overwrite the test reporting conditions and flip "
                "rc_source. Remove the RepCond step from the non-rc_source "
                "pipeline."
            )

        inst.meas_filters_pending = list(meas_filters or [])
        inst.sim_filters_pending = list(sim_filters or [])
        if inst.rc_source == "manual" and rc_values is not None:
            if inst._resolved_setup is not None:
                df = inst._coerce_and_validate_manual_rc(rc_values)
                inst._set_rc(df, "manual", warn=False)
            else:
                inst._pending_manual_rc = dict(rc_values)
        return inst

    def reload(self, side, path=None, verbose=True):
        """Re-load one side's data and re-run per-side setup.

        Re-invokes the stored loader (``meas_loader``/``sim_loader`` or the
        module defaults) on the side's data path with the stored
        ``*_load_kwargs``, replaces that ``CapData``, then runs per-side
        ``setup(side=side)``. Pass ``path`` to point the side at a new data
        file first — the new path replaces the stored one, so later
        ``reload`` calls and ``to_yaml``/``to_mapping`` use it. Relative
        paths resolve against the current working directory.

        The outgoing side's applied filter chain is preserved: its config is
        snapshot into ``<side>_filters_pending`` before the data is
        replaced, so a follow-up ``run_test(side=side)`` re-applies the same
        filters against the fresh data. When the outgoing chain is empty, an
        existing pending config is left untouched.

        Parameters
        ----------
        side : {'meas', 'sim'}
            Which side to re-load.
        path : str or Path, optional
            New data file for this side. Stored (replacing the remembered
            ``meas_path``/``sim_path``) before loading.
        verbose : bool, default True
            Forwarded to ``setup``.

        Returns
        -------
        CapTest
            ``self``, for fluent chaining
            (``ct.reload('sim', path='new.CSV').run_test(side='sim')``).

        Raises
        ------
        ValueError
            If ``side`` is invalid, or no path is stored for that side and
            none was passed (instance constructed from pre-built ``CapData``
            objects).
        """
        if side not in ("meas", "sim"):
            raise ValueError(f"side must be 'meas' or 'sim', got {side!r}.")
        if path is not None:
            if side == "meas":
                self._meas_path = str(path)
            else:
                self._sim_path = str(path)
        stored_path = self._meas_path if side == "meas" else self._sim_path
        if stored_path is None:
            raise ValueError(
                f"CapTest holds no stored data path for '{side}'. Pass "
                "path=... or construct from meas_path/sim_path (from_params, "
                "from_yaml, or from_mapping)."
            )
        outgoing = getattr(self, side)
        if outgoing is not None and outgoing.filters:
            setattr(self, f"{side}_filters_pending", outgoing.filters_to_config())
        if side == "meas":
            loader = self.meas_loader or _default_meas_loader()
            self.meas = loader(stored_path, **(self.meas_load_kwargs or {}))
        else:
            loader = self.sim_loader or _default_sim_loader()
            self.sim = loader(stored_path, **(self.sim_load_kwargs or {}))
        self.setup(verbose=verbose, side=side)
        return self

    def to_yaml(self, path, key="captest", merge_into_existing=True):
        """Serialize the curated CapTest configuration to a yaml file.

        The written sub-mapping is :meth:`to_mapping`'s return value. It
        lives under the top-level ``key`` (default
        ``"captest"``) and contains every scalar ``param.*`` plus
        ``test_setup``, any non-None override of ``reg_fml`` /
        ``reg_cols_meas`` / ``reg_cols_sim`` / ``rep_conditions``,
        ``meas_path`` / ``sim_path`` (when the instance was constructed from
        paths), and non-empty ``meas_load_kwargs`` / ``sim_load_kwargs``.

        The filter pipelines of ``meas`` and ``sim`` are written as
        ``meas_filters`` / ``sim_filters`` (lists of filter-step config dicts
        from :meth:`CapData.filters_to_config`, or the side's pending config
        when its chain is empty), each only when non-empty; ``from_yaml``
        stores them as pending pipelines that :meth:`run_test` replays. When a
        ``RepCond`` step is present in either pipeline, ``overrides.rep_conditions``
        is omitted — the step is then the authoritative reporting-conditions
        source (avoids representing it in two places).

        Percentile ``perc_wrap(N)`` callables inside
        ``rep_conditions['func']`` are written back as ``"perc_N"`` strings
        so that ``from_yaml`` round-trips them. ``meas``, ``sim``,
        ``regression_results``, ``_resolved_setup``, and the loader
        callables are never serialized.

        Parameters
        ----------
        path : str or Path
            Destination yaml file.
        key : str, default 'captest'
            Top-level key under which the captest sub-mapping is written.
            Parametrizing this lets a single yaml hold multiple captest
            flavors (e.g. ``captest_e2848`` and ``captest_bifi``).
        merge_into_existing : bool, default True
            When True and the destination file already exists and parses as
            a mapping, preserve the other top-level keys and overwrite only
            the sub-tree at ``key``. When False, the destination is
            unconditionally replaced with a fresh mapping containing only
            ``key``.

        Returns
        -------
        None
        """
        path = Path(path)

        sub = self.to_mapping()

        # Merge with an existing file on disk when requested.
        root_doc = {}
        if merge_into_existing and path.exists():
            try:
                with path.open("r") as fh:
                    existing = yaml.safe_load(fh)
                if isinstance(existing, dict):
                    root_doc = existing
            except (OSError, yaml.YAMLError):  # pragma: no cover - rare IO/parse
                root_doc = {}
        root_doc[key] = sub

        with path.open("w") as fh:
            yaml.safe_dump(root_doc, fh, sort_keys=False)

    def to_mapping(self):
        """Return the curated config mapping ``to_yaml`` writes under ``key``.

        The public dict counterpart of :meth:`to_yaml` and the symmetric
        inverse of :meth:`from_mapping`. Emits the same programmatic-only
        attribute warning as ``to_yaml`` (loaders, mutated scatter_plots).

        Returns
        -------
        dict
            The captest sub-mapping (scalars, overrides, paths, pipelines).
        """
        self._warn_unserializable()
        return self._build_yaml_sub_mapping()

    def _warn_unserializable(self):
        """Warn once for any non-yaml-serializable user overrides.

        Loader callables and a user-mutated ``scatter_plots`` entry cannot be
        represented in the yaml config; name them in a single ``UserWarning``
        so the omission is visible at export time.
        """
        unserializable = []
        if self.meas_loader is not None:
            unserializable.append("meas_loader")
        if self.sim_loader is not None:
            unserializable.append("sim_loader")
        if self._resolved_setup is not None and self.test_setup != "custom":
            preset_scatter = TEST_SETUPS.get(self.test_setup, {}).get("scatter_plots")
            current_scatter = self._resolved_setup.get("scatter_plots")
            if (
                preset_scatter is not None
                and current_scatter is not None
                and current_scatter is not preset_scatter
            ):
                unserializable.append("scatter_plots")
        if unserializable:
            warnings.warn(
                "The following CapTest attributes are programmatic-only and "
                "will be omitted from the yaml file: "
                f"{sorted(unserializable)}",
                stacklevel=2,
            )

    def _build_yaml_sub_mapping(self):
        """Build the dict written under ``key:`` by :meth:`to_yaml`.

        Kept separate from ``to_yaml`` so it is testable in isolation and
        so the merge/write step stays short. Embeds each side's pipeline as
        ``meas_filters``/``sim_filters``: the applied filter chain when
        non-empty, else the side's pending config (so a load → save without
        running is lossless), else the key is omitted. Omits
        ``overrides.rep_conditions`` when a ``RepCond`` step is present in
        either pipeline (the step is then the single source of reporting
        conditions).
        """
        sub = {"test_setup": self.test_setup}

        # Paths are written only when the instance was constructed from
        # paths; we remember the raw (possibly relative) string in
        # ``_meas_path``/``_sim_path``.
        if self._meas_path is not None:
            sub["meas_path"] = str(self._meas_path)
        if self._sim_path is not None:
            sub["sim_path"] = str(self._sim_path)

        # Overrides sub-mapping.
        preset = TEST_SETUPS.get(self.test_setup, {})
        overrides = {}
        if self.test_setup == "custom":
            # ``custom`` has no preset; always include whatever the user set.
            for name in ("reg_cols_meas", "reg_cols_sim", "reg_fml"):
                val = getattr(self, name)
                if val is not None:
                    overrides[name] = copy.deepcopy(val)
        else:
            for name in ("reg_cols_meas", "reg_cols_sim", "reg_fml"):
                val = getattr(self, name)
                if val is not None and val != preset.get(name):
                    overrides[name] = copy.deepcopy(val)
        meas_filters = (
            self.meas.filters_to_config()
            if self.meas is not None and self.meas.filters
            else list(self.meas_filters_pending)
        )
        sim_filters = (
            self.sim.filters_to_config()
            if self.sim is not None and self.sim.filters
            else list(self.sim_filters_pending)
        )
        has_rep_cond_step = any(
            d["type"] == "RepCond" for d in (meas_filters + sim_filters)
        )
        # Decision B: when a RepCond step is in either pipeline, it is the
        # unambiguous source of reporting conditions — drop the redundant
        # overrides.rep_conditions.
        # For a manual rc_source, reporting_conditions_values (written below) is
        # the authoritative RC; do not also serialize overrides.rep_conditions,
        # which is only aggregation config and would read as a second RC source.
        if (
            self.rep_conditions is not None
            and not has_rep_cond_step
            and self.rc_source != "manual"
        ):
            overrides["rep_conditions"] = _serialize_rep_conditions(self.rep_conditions)
        if overrides:
            sub["overrides"] = overrides

        # Remaining scalar params (always written).
        scalar_names = (
            "rc_source",
            "ac_nameplate",
            "inv_ac_nameplate",
            "test_tolerance",
            "sim_days",
            "shade_filter_start",
            "shade_filter_end",
            "min_irr",
            "max_irr",
            "clipping_irr",
            "rep_irr_filter",
            "fshdbm",
            "irrad_stability",
            "irrad_stability_threshold",
            "hrs_req",
            "bifaciality",
            "bifacial_frac",
            "rear_shade",
            "power_temp_coeff",
            "base_temp",
            "module_type",
            "racking",
            "spectral_module_type",
            "airmass_model",
            "altitude_override",
        )
        for name in scalar_names:
            sub[name] = getattr(self, name)

        # Loader kwargs are plain dicts; only write when non-empty so a
        # default-constructed CapTest produces a clean yaml.
        if self.meas_load_kwargs:
            sub["meas_load_kwargs"] = copy.deepcopy(self.meas_load_kwargs)
        if self.sim_load_kwargs:
            sub["sim_load_kwargs"] = copy.deepcopy(self.sim_load_kwargs)

        if meas_filters:
            sub["meas_filters"] = meas_filters
        if sim_filters:
            sub["sim_filters"] = sim_filters

        # Manual reporting conditions are data, not config: serialize their
        # values so from_yaml can restore them (computed RC is recomputed by
        # replaying the source pipeline's RepCond step). Numpy scalars are
        # coerced to native python types for yaml.safe_dump. Values stashed
        # by a load-only construction (run_setup=False) round-trip too.
        if self.rc_source == "manual":
            if self._rc is not None:
                row = self._rc.iloc[0]
                sub["reporting_conditions_values"] = {
                    str(col): to_native(val) for col, val in row.items()
                }
            elif self._pending_manual_rc is not None:
                sub["reporting_conditions_values"] = dict(self._pending_manual_rc)

        return sub

    # --- workflow methods ------------------------------------------------

    def _propagate_sim_site(self):
        """Deep-copy ``meas.site`` onto ``sim.site`` with a fixed-offset tz.

        PVsyst data is not DST-aware, so presets that call
        :func:`captest.calcparams.apparent_zenith_pvsyst` need
        ``sim.site['loc']['tz']`` to be an ``Etc/GMT±N`` fixed-offset string.
        When ``sim.site`` is unset and ``meas.site`` is available, this
        helper deep-copies the latter and converts the tz to the nearest
        fixed offset (using the January 1 offset so DST never biases the
        conversion). Emits a ``UserWarning`` describing what was done.

        If ``sim.site`` is already set by the user, leaves it untouched.
        """
        meas_site = getattr(self.meas, "site", None)
        sim_site = getattr(self.sim, "site", None)
        if meas_site is None or sim_site is not None:
            return

        new_site = copy.deepcopy(meas_site)
        tz = new_site.get("loc", {}).get("tz")
        if isinstance(tz, str):
            try:
                import zoneinfo
                from datetime import datetime

                zi = zoneinfo.ZoneInfo(tz)
                # Use Jan 1 to avoid DST; PVsyst timestamps are non-DST.
                offset = datetime(2000, 1, 1, tzinfo=zi).utcoffset()
                offset_hours = int(offset.total_seconds() // 3600)
                # Etc/GMT uses inverted signs: UTC-6 is 'Etc/GMT+6'.
                etc_tz = f"Etc/GMT{-offset_hours:+d}"
                new_site["loc"]["tz"] = etc_tz
                warnings.warn(
                    f"Propagating meas.site onto sim.site and converting tz "
                    f"from {tz!r} to {etc_tz!r} (PVsyst data is not DST-aware).",
                    stacklevel=2,
                )
            except Exception:  # pragma: no cover - tz lookup failure is rare
                warnings.warn(
                    f"Propagating meas.site onto sim.site but could not "
                    f"convert tz {tz!r} to an Etc/GMT±N fixed offset; "
                    f"leaving tz unchanged.",
                    stacklevel=2,
                )
        self.sim.site = new_site

    def _maybe_wrap_sim_year_end(self):
        """Auto-apply ``wrap_year_end`` to ``self.sim.data`` when warranted.

        Idempotent and reversible: a prior wrap is restored from
        ``self.sim._pre_wrap_data`` before each check, so re-running
        ``setup()`` — or toggling ``self.auto_wrap_sim`` to False and
        re-running — leaves ``sim.data`` in the correct state. The snapshot
        lives on the sim CapData itself so a future ``reload_sim`` that
        replaces ``self.sim`` automatically discards the stale snapshot.
        """
        if self.sim is None:
            return

        snapshot = getattr(self.sim, "_pre_wrap_data", None)
        if snapshot is not None:
            self.sim.data = snapshot.copy()
            self.sim.filters = []
            self.sim._pre_wrap_data = None

        if not self.auto_wrap_sim:
            return
        if self.meas is None:
            return
        meas_idx = self.meas.data.index
        sim_idx = self.sim.data.index
        if not isinstance(meas_idx, pd.DatetimeIndex):
            return
        if not isinstance(sim_idx, pd.DatetimeIndex):
            return
        if len(meas_idx) == 0 or len(sim_idx) == 0:
            return

        meas_start = meas_idx[0]
        meas_end = meas_idx[-1]
        days_from_year_start = (
            meas_start - pd.Timestamp(year=meas_start.year, month=1, day=1)
        ).days
        days_to_year_end = (
            pd.Timestamp(year=meas_end.year, month=12, day=31) - meas_end
        ).days
        if (
            days_from_year_start > _AUTO_WRAP_DAYS
            and days_to_year_end > _AUTO_WRAP_DAYS
        ):
            return

        # Use a fixed July 1 -> June 30 window so the wrapped sim is a
        # contiguous full year centered on the Jan 1 boundary, regardless of
        # where the measured test falls. Years are derived from sim_year
        # (1989/1990 for pvsyst data, which load_pvsyst normalizes to 1990).
        sim_year = sim_idx[0].year
        start = pd.Timestamp(year=sim_year - 1, month=7, day=1, hour=0, minute=0)
        end = pd.Timestamp(year=sim_year, month=6, day=30, hour=23, minute=59)

        self.sim._pre_wrap_data = self.sim.data.copy()
        wrapped = wrap_year_end(self.sim.data, start, end)
        if "index" in wrapped.columns:
            wrapped = wrapped.drop(columns="index")
        self.sim.data = wrapped
        self.sim.filters = []

    def setup(self, verbose=True, side="both"):
        """Resolve TEST_SETUPS, propagate scalars, process regression cols.

        Raises ``RuntimeError`` if any ``CapData`` targeted by ``side`` is
        unset. Assigns the resolved TEST_SETUPS entry to
        ``self._resolved_setup`` and returns ``self`` for fluent chaining.

        A full setup (``side='both'``) also consumes manual
        reporting-conditions values stashed by a load-only
        ``from_mapping(run_setup=False)``, validating and seeding them as
        ``rc_source='manual'``; per-side setup leaves the stash untouched
        (validation needs both sides' regression formulas).

        With ``side='meas'`` or ``side='sim'`` only the target ``CapData``
        is re-wired; the other side's data, filter chain, and regression
        state are left untouched. Sim-side setup may *read* meas (the
        year-wrap span check and site propagation) but mutates only sim;
        meas-side setup never touches sim — in particular the year-end
        auto-wrap (``_maybe_wrap_sim_year_end``), which mutates ``sim.data``
        while reading the meas span, is skipped for ``side='meas'``.

        Parameters
        ----------
        verbose : bool, default True
            Forwarded to ``CapData.process_regression_columns``.
        side : {'both', 'meas', 'sim'}, default 'both'
            Which CapData instance(s) to (re)wire.

        Returns
        -------
        CapTest
            ``self``, for fluent chaining.

        Raises
        ------
        ValueError
            If ``side`` is not ``'meas'``, ``'sim'``, or ``'both'``.
        RuntimeError
            If a ``CapData`` targeted by ``side`` is unset.
        """
        if side not in ("meas", "sim", "both"):
            raise ValueError(f"side must be 'meas', 'sim', or 'both', got {side!r}.")
        sides = ("meas", "sim") if side == "both" else (side,)
        for s in sides:
            if getattr(self, s) is None:
                raise RuntimeError(f"CapTest.{s} must be set before setup().")

        # Auto-wrap sim.data when measured spans (within 60 days of) a year
        # boundary. Idempotent and reversible — re-running setup() or toggling
        # auto_wrap_sim restores the appropriate state. The wrap mutates
        # sim.data while reading the meas span, so it is skipped for
        # side='meas' (meas-side setup must not touch sim).
        if "sim" in sides and self.meas is not None:
            self._maybe_wrap_sim_year_end()

        # Build the overrides dict for resolve_test_setup. Only non-None
        # values are passed through so named-preset resolution falls back to
        # the preset's defaults for keys the user hasn't overridden.
        overrides = {}
        for name in ("reg_cols_meas", "reg_cols_sim", "reg_fml", "rep_conditions"):
            val = getattr(self, name)
            if val is not None:
                overrides[name] = val

        resolved = resolve_test_setup(self.test_setup, overrides=overrides)
        self._resolved_setup = resolved

        # Propagate scalar calc-params onto the targeted CapData instances.
        # Names in _downstream_attrs_meas_only are copied onto meas only;
        # all others are copied onto both meas and sim.
        for name in self._downstream_attrs:
            if "meas" in sides:
                setattr(self.meas, name, getattr(self, name))
            if "sim" in sides and name not in self._downstream_attrs_meas_only:
                setattr(self.sim, name, getattr(self, name))

        # Propagate site from meas -> sim with Etc/GMT±N tz for PVsyst.
        # Reads meas.site but mutates only sim, so it runs for sim-side setup.
        if "sim" in sides and self.meas is not None:
            self._propagate_sim_site()

        # Wire per-CapData regression state on each targeted side. Deepcopy
        # the regression_cols dict because process_regression_columns mutates
        # it in place. process_regression_columns also resets data_filtered
        # to data.copy() so any prior filter state on that side is dropped
        # (intended behavior per the design spec).
        for s in sides:
            cd = getattr(self, s)
            cd.regression_cols = copy.deepcopy(resolved[f"reg_cols_{s}"])
            cd.regression_formula = resolved["reg_fml"]
            cd.tolerance = self.test_tolerance
            cd.process_regression_columns(verbose=verbose)
            # Wire the CapData back to this CapTest so
            # filter_irr(ref_val='rep_irr') resolves against the single test
            # RC (ct.rc) and cd.rep_cond can update it. Runtime reference
            # only; capdata.py never imports captest.
            cd._captest = self

        # Consume manual reporting-conditions values stashed by a load-only
        # from_mapping (run_setup=False). Only a full setup can seed them:
        # validation needs both sides' regression formulas, wired just above.
        if side == "both" and self._pending_manual_rc is not None:
            df = self._coerce_and_validate_manual_rc(self._pending_manual_rc)
            self._set_rc(df, "manual", warn=False)
            self._pending_manual_rc = None

        return self

    def scatter_plots(self, which="meas", **kwargs):
        """Create the scatter plot for the active capacity-test setup.

        This method is intended primarily to plot a power vs irradiance scatter
        plot that fits with a preset capacity test from the ``TEST_SETUPS``
        defined in the ``captest`` module.

        To create manual scatter plots and to see the complete list of
        accepted kwargs and their behavior, see the docstrings for
        :class:`captest.plotting.ScatterPlot` and
        :class:`captest.plotting.ScatterBifiPowerTc`. ``ScatterBifiPowerTc``
        inherits most options from ``ScatterPlot`` but ignores ``tc_power``
        because the ``bifi_power_tc`` regression power term is already
        temperature corrected.

        The selected ``test_setup`` controls which plotting function is used.
        During :meth:`setup`, the named setup is resolved from ``TEST_SETUPS``;
        that resolved setup includes a ``scatter_plots`` callable matched to
        the setup's regression formula. This method picks ``self.meas`` or
        ``self.sim`` and forwards it, plus any keyword arguments, to that
        callable.

        Built-in setup behavior:

        - ``e2848_default``, ``bifi_e2848_etotal_rear_shade_sim``,
          ``bifi_e2848_etotal_rear_shade_meas``, and
          ``e2848_spec_corrected_poa`` use ``ScatterPlot`` through the
          ``scatter_default`` / ``scatter_etotal`` wrappers. These create a
          formula-driven scatter of the regression left-hand-side variable
          against the first right-hand-side variable.
        - ``bifi_power_tc`` uses ``ScatterBifiPowerTc`` through the
          ``scatter_bifi_power_tc`` wrapper. This creates one panel for each
          right-hand-side variable in the bifacial temperature-corrected
          regression, typically ``power vs poa`` and ``power vs rpoa``.

        All keyword arguments are forwarded to the underlying plotting class.
        The most commonly used options are:

        - ``filtered``: use ``data_filtered`` when True, otherwise ``data``.
        - ``split_day`` and ``split_time``: split points into AM and PM groups.
        - ``am_color``, ``pm_color``, ``am_marker``, and ``pm_marker``:
          customize AM / PM glyph style.
        - ``tc_power``, ``tc_mode``, ``tc_power_calc``, and
          ``tc_force_recompute``: show temperature-corrected power for setups
          whose regression still uses raw power. ``tc_mode`` can be
          ``"replace"``, ``"add_panel"``, or ``"overlay"``.
        - ``timeseries``: add a linked timeseries panel below the scatter.
        - ``height`` and ``width``: set plot dimensions.

        Parameters
        ----------
        which : {'meas', 'sim'}
            Which :class:`captest.capdata.CapData` instance to plot.
        **kwargs
            Plotting options forwarded to the preset's scatter callable.

        Returns
        -------
        holoviews.Layout
            Scatter plot layout for the selected measured or modeled data.

        Examples
        --------
        Plot measured data with the default options::

            ct.scatter_plots()

        Plot modeled data, split points into AM and PM groups, and add a
        linked timeseries panel::

            ct.scatter_plots(which="sim", split_day=True, timeseries=True)

        Add a temperature-corrected power panel for a setup that uses raw
        power in the regression::

            ct.scatter_plots(tc_power=True, tc_mode="add_panel")
        """
        cd = self._pick_cd(which)
        self._require_setup()
        return self._resolved_setup["scatter_plots"](cd, **kwargs)

    def rep_cond(self, which=None, **overrides):
        """Call ``cd.rep_cond`` with the resolved preset's rep_conditions.

        The preset's ``rep_conditions`` dict (after any ``self.rep_conditions``
        overrides from ``setup()``) is used as the default kwargs. ``overrides``
        is partial-merged on top: top-level keys replace, the nested ``func``
        dict merges one level deep.

        See :meth:`~captest.capdata.CapData.rep_cond` for details on the reporting conditions calculation
        options.

        Parameters
        ----------
        which : {'meas', 'sim', None}, default None
            Which CapData to compute reporting conditions on. When None, defaults
            to the current ``rc_source`` if it is ``'meas'``/``'sim'``, otherwise
            ``'meas'``. The computed conditions become the test ``rc`` (and set
            ``rc_source`` to ``which``) via the last-writer-wins sync.
        **overrides
            Partial-merged onto the resolved ``rep_conditions`` dict.

        Returns
        -------
        None
            ``cd.rep_cond`` writes to ``cd.rc``.
        """
        if which is None:
            which = self.rc_source if self.rc_source in ("meas", "sim") else "meas"
        cd = self._pick_cd(which)
        self._require_setup()
        resolved_rc = _merge_rep_conditions(
            self._resolved_setup["rep_conditions"], overrides
        )
        return cd.rep_cond(**resolved_rc)

    # --- ported cross-CapData methods ------------------------------------

    def determine_pass_or_fail(self, cap_ratio):
        """Determine a pass/fail result from a capacity ratio.

        Uses ``self.test_tolerance`` and ``self.ac_nameplate``. Replaces the
        pre-CapTest module-level ``capdata.determine_pass_or_fail``.

        Parameters
        ----------
        cap_ratio : float
            Ratio of the measured-data regression result to the modeled-data
            regression result.

        Returns
        -------
        tuple of (bool, str)
            Pass/fail flag and the tolerance bounds string.
        """
        sign = self.test_tolerance.split(sep=" ")[0]
        error = float(self.test_tolerance.split(sep=" ")[1]) / 100

        nameplate_plus_error = self.ac_nameplate * (1 + error)
        nameplate_minus_error = self.ac_nameplate * (1 - error)

        if sign in ("+/-", "-/+"):
            return (
                round(np.abs(1 - cap_ratio), ndigits=6) <= error,
                f"{nameplate_minus_error}, {nameplate_plus_error}",
            )
        if sign == "-":
            return (cap_ratio >= 1 - error, f"{nameplate_minus_error}, None")
        warnings.warn("Sign must be '-', '+/-', or '-/+'.")
        return None

    def captest_results(self, check_pvalues=False, pval=0.05, print_res=True):
        """Compute the capacity test results for ``self.meas`` vs ``self.sim``.

        Predicts both regressions at the single test reporting conditions
        ``self.rc`` (set via :meth:`rep_cond` or the ``rc`` setter);
        ``self.rc_source`` is reported for provenance. Raises ``ValueError``
        if ``self.rc`` is ``None``. Uses ``self.ac_nameplate`` for the
        tested capacity and ``self.test_tolerance`` (via
        ``self.determine_pass_or_fail``) for the pass/fail result. Both the
        plain and the p-value-checked predictions are always computed;
        ``check_pvalues`` selects which pair is the headline reported as
        ``cap_ratio`` / ``actual_capacity`` / ``expected_capacity`` and used
        for pass/fail and tested capacity (``cap_ratio_pval_check`` always
        carries the checked ratio; ``pvalues_checked`` records the choice).

        Parameters
        ----------
        check_pvalues : bool, default False
            When True, the headline predictions and ratio are the ones
            computed with above-``pval`` coefficients zeroed before
            prediction.
        pval : float, default 0.05
            P-value cutoff used for the p-value-checked ratio.
        print_res : bool, default True
            When True, prints the formatted results (``str(results)``).

        Returns
        -------
        CapTestResults or None
            Structured results object. Returns ``None`` (after a
            ``UserWarning``) when the two regression formulas differ.
        """
        self._require_meas_and_sim()
        if self.meas.regression_formula != self.sim.regression_formula:
            warnings.warn("CapData objects do not have the same regression formula.")
            return None

        rc = self.rc
        if rc is None:
            raise ValueError(
                "captest_results requires test reporting conditions. Call "
                "ct.rep_cond(which) or assign ct.rc = df first."
            )

        # predict_with_pvalue_check is a single-CapData helper that stays in
        # capdata.py. Imported lazily to avoid importing holoviews-heavy
        # capdata internals at module-load time for callers that never
        # compute cap ratios (e.g. notebooks that only use setup + plots).
        from captest.capdata import predict_with_pvalue_check

        checked_actual = predict_with_pvalue_check(
            self.meas, rc=rc, pval_threshold=pval
        )
        checked_expected = predict_with_pvalue_check(
            self.sim, rc=rc, pval_threshold=pval
        )
        plain_actual = predict_with_pvalue_check(self.meas, rc=rc, pval_threshold=None)
        plain_expected = predict_with_pvalue_check(self.sim, rc=rc, pval_threshold=None)
        cap_ratio_pval_check = checked_actual / checked_expected
        # The headline pair drives the report and the pass/fail decision.
        if check_pvalues:
            actual, expected = checked_actual, checked_expected
            cap_ratio = cap_ratio_pval_check
        else:
            actual, expected = plain_actual, plain_expected
            cap_ratio = plain_actual / plain_expected
        if cap_ratio < 0.01:
            cap_ratio *= 1000
            cap_ratio_pval_check *= 1000
            actual *= 1000
            warnings.warn(
                "Capacity ratio and actual capacity multiplied by 1000"
                " because the capacity ratio was less than 0.01."
            )
        test_passed = self.determine_pass_or_fail(cap_ratio)
        if test_passed is None:
            test_passed = (False, "")
        capacity = self.ac_nameplate * cap_ratio

        def _points_used(cd):
            if cd.filters:
                return cd.filters[-1].pts_after
            return len(cd.data)

        def _reg_table(cd):
            r = cd.regression_results
            return pd.DataFrame({"coef": r.params, "pvalue": r.pvalues})

        results = CapTestResults(
            cap_ratio=cap_ratio,
            cap_ratio_pval_check=cap_ratio_pval_check,
            passed=bool(test_passed[0]),
            tolerance=self.test_tolerance,
            bounds=test_passed[1],
            expected_capacity=expected,
            actual_capacity=actual,
            tested_capacity=capacity,
            points_used={
                "meas": _points_used(self.meas),
                "sim": _points_used(self.sim),
            },
            regression_tables={
                "meas": _reg_table(self.meas),
                "sim": _reg_table(self.sim),
            },
            rc=rc.copy(),
            rc_source=self.rc_source,
            pvalues_checked=check_pvalues,
        )
        if print_res:
            print(results)
        return results

    def run_test(self, side="both", check_pvalues=False, pval=0.05, print_res=False):
        """Run the full capacity test (or one side of it) end to end.

        Canonical sequence: (1) ``setup(side=side)``; (2) replay each side's
        filter pipeline, the ``rc_source`` side first so its RepCond step
        populates the test RC before the other side's RC-dependent filters
        resolve; (3) ``fit_regression`` per side; then, for ``side='both'``
        only, (4) verify the rc_source pipeline computed the RC this run and
        (5) return :class:`CapTestResults`.

        Each side's replay source is chosen before setup clears the chains:
        the live chain when non-empty (snapshotted via ``filters_to_config``
        — interactive edits win), else the side's pending config
        (``meas_filters_pending`` / ``sim_filters_pending``, stored by
        ``from_yaml`` / ``from_mapping``). The call is re-entrant. A side's
        pending list is consumed after its replay succeeds — a later
        ``reset_filter()`` + ``run_test`` means "no filters", not
        "resurrect the config's filters" — and retained (holding the full
        failed pipeline) when the replay fails. During a
        full run the not-yet-replayed side is registered in
        ``_rc_pending_sides`` so the RC write does not warn about steps this
        call is about to re-run; per-side runs register nothing, so an
        RC-changing recompute warns about the other side's applied
        RC-dependent steps. The ``process_regression_columns`` lost-filters
        warning is suppressed for this intentional orchestrated clearing.
        When ``rc_source='manual'`` the manual reporting conditions remain
        authoritative during replay: pipeline RepCond steps compute
        side-local RCs only, matching ``from_mapping``.

        Parameters
        ----------
        side : {'both', 'meas', 'sim'}, default 'both'
        check_pvalues, pval, print_res
            Forwarded to :meth:`captest_results` (``side='both'`` only).

        Returns
        -------
        CapTestResults or CapTest
            Results for ``side='both'``; ``self`` for per-side runs.

        Raises
        ------
        ValueError
            If ``side`` is not ``'meas'``, ``'sim'``, or ``'both'``.
        RuntimeError
            For ``side='both'``: a computed ``rc_source`` whose replayed
            pipeline contains no RepCond step, or ``rc_source='manual'``
            with no reporting conditions set. Exceptions raised in any
            stage carry a ``[CapTest.run_test stage: ...]`` note on
            Python 3.11+.
        """
        if side not in ("meas", "sim", "both"):
            raise ValueError(f"side must be 'meas', 'sim', or 'both', got {side!r}.")
        run_sides = ["meas", "sim"] if side == "both" else [side]
        if side == "both" and self.rc_source == "sim":
            run_sides = ["sim", "meas"]

        # Select each side's replay source BEFORE setup:
        # process_regression_columns clears each targeted side's applied
        # chain. The live chain wins when non-empty (interactive edits are
        # authoritative); otherwise the side's pending config is replayed.
        configs = {}
        for s in run_sides:
            cd = getattr(self, s)
            if cd is not None and cd.filters:
                configs[s] = cd.filters_to_config()
            else:
                configs[s] = list(getattr(self, f"{s}_filters_pending"))
        if (
            side == "both"
            and self.rc_source in ("meas", "sim")
            and all(
                any(d.get("type") == "RepCond" for d in configs[s])
                for s in ("meas", "sim")
            )
        ):
            warnings.warn(
                "Both pipelines contain a RepCond step with a computed "
                f"rc_source ('{self.rc_source}'): this is ambiguous and "
                "unsupported — the non-rc_source side's RepCond will "
                "overwrite the test reporting conditions and flip "
                "rc_source. Remove the RepCond step from the non-rc_source "
                "pipeline."
            )

        stage = "setup"
        try:
            with warnings.catch_warnings():
                # setup()'s chain-clearing is intentional here (the chains
                # were snapshotted above); keep the lost-filters warning for
                # direct interactive process_regression_columns calls.
                warnings.filterwarnings(
                    "ignore",
                    message="The data_filtered attribute has been overwritten",
                )
                self.setup(verbose=False, side=side)

            stage = "filter pipelines"
            if side == "both":
                self._rc_pending_sides = set(run_sides)
            # A manual RC is authoritative: suppress live RepCond propagation
            # during the replay (mirroring from_mapping's manual-RC replay
            # semantics) so a replayed RepCond step still computes that side's
            # local cd.rc but never overwrites ct.rc or flips rc_source.
            # Computed sources keep live propagation — the staleness
            # machinery depends on it.
            manual_rc = self.rc_source == "manual"
            if manual_rc:
                self._loading = True
            try:
                for s in run_sides:
                    # Consume the pending registration as this side's replay
                    # begins: the currently-replaying side is never in its
                    # own exclusion set.
                    self._rc_pending_sides.discard(s)
                    if configs[s]:
                        try:
                            getattr(self, s).run_pipeline(configs[s])
                        except Exception as e:
                            # Keep the failed pipeline's definition editable:
                            # setup() already cleared the live chain, so after
                            # the rollback the pending list is the only copy
                            # of a live-chain snapshot (spec R2 rule 3).
                            setattr(self, f"{s}_filters_pending", configs[s])
                            if hasattr(e, "add_note"):
                                e.add_note(
                                    f"The {s} filter pipeline failed and was "
                                    "rolled back. Its definition is retained "
                                    f"in ct.{s}_filters_pending — edit the "
                                    "failing step's dict and re-run "
                                    "ct.run_test() (or "
                                    f"ct.{s}.run_pipeline(ct.{s}_filters_pending"
                                    ")), or edit the yaml config and reload."
                                )
                            raise
                    # A completed pass makes the live chain the single source
                    # of truth for this side; the pending config is consumed
                    # regardless of which source was replayed (spec R2).
                    setattr(self, f"{s}_filters_pending", [])
            finally:
                self._rc_pending_sides = set()
                if manual_rc:
                    self._loading = False

            stage = "fit_regression"
            for s in run_sides:
                getattr(self, s).fit_regression(summary=False)

            if side != "both":
                return self

            stage = "reporting conditions"
            if self.rc_source in ("meas", "sim"):
                # Verification, not computation: run_pipeline truncated the
                # chain first, so a RepCond in the applied chain proves the
                # step executed and wrote ct.rc THIS run (a bare rc-is-set
                # check would accept a stale RC from a prior run).
                src_cd = getattr(self, self.rc_source)
                if not any(type(st).__name__ == "RepCond" for st in src_cd.filters):
                    raise RuntimeError(
                        f"rc_source='{self.rc_source}' but the "
                        f"{self.rc_source} pipeline contains no RepCond "
                        "step; the test reporting conditions were not "
                        "computed this run."
                    )
            elif self._rc is None:
                raise RuntimeError(
                    "rc_source='manual' but no reporting conditions are "
                    "set; assign ct.rc = df before run_test()."
                )

            stage = "results"
            return self.captest_results(
                check_pvalues=check_pvalues, pval=pval, print_res=print_res
            )
        except Exception as e:
            # add_note exists on 3.11+; the project floor is 3.10.
            if hasattr(e, "add_note"):
                e.add_note(f"[CapTest.run_test stage: {stage}]")
            raise

    def captest_results_check_pvalues(self, print_res=False, **kwargs):
        """Compute cap ratio with and without p-value filtering.

        Thin display wrapper around :meth:`captest_results` (called once):
        prints both capacity ratios and returns the p-value Styler view of
        the results (``CapTestResults.styled_pvalues``).

        Parameters
        ----------
        print_res : bool, default False
            Forwarded to the internal ``captest_results`` call.
        **kwargs
            Forwarded to ``captest_results`` (e.g. ``check_pvalues`` to pick
            the headline ratio, ``pval`` for the cutoff).

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with p-values and parameter values for both
            ``self.meas`` and ``self.sim``. P-values >= 0.05 are highlighted.
        """
        res = self.captest_results(print_res=print_res, **kwargs)

        cap_ratio_rounded = np.round(res.cap_ratio, decimals=4) * 100
        cap_ratio_check_pvalues_rounded = (
            np.round(res.cap_ratio_pval_check, decimals=4) * 100
        )

        print("{:.3f}% - Cap Ratio".format(cap_ratio_rounded))
        print(
            "{:.3f}% - Cap Ratio after pval check".format(
                cap_ratio_check_pvalues_rounded
            )
        )

        return res.styled_pvalues()

    def get_summary(self):
        """Concatenate ``self.meas.get_summary()`` and ``self.sim.get_summary()``.

        Returns
        -------
        pandas.DataFrame
            Filter history for both CapData instances, stacked.
        """
        self._require_meas_and_sim()
        return pd.concat([self.meas.get_summary(), self.sim.get_summary()])

    def overlay_scatters(self, expected_label="PVsyst"):
        """Overlay the final scatter plot from ``self.meas`` and ``self.sim``.

        Builds the scatter plot for each CapData instance via the resolved
        preset's ``scatter_plots`` callable, then overlays the two first-panel
        scatters with labels.

        Parameters
        ----------
        expected_label : str, default "PVsyst"
            Label used for the modeled-data scatter.

        Returns
        -------
        hv.Overlay
        """
        if hv is None:
            raise ImportError(
                "holoviews is required for overlay_scatters. Install with "
                "`uv add holoviews` or equivalent."
            )
        self._require_setup()
        scatter_fn = self._resolved_setup["scatter_plots"]
        meas_layout = scatter_fn(self.meas)
        sim_layout = scatter_fn(self.sim)
        # scatter_fn returns an hv.Layout whose first element is an hv.Scatter.
        meas_scatter = list(meas_layout)[0].relabel("Measured")
        sim_scatter = list(sim_layout)[0].relabel(expected_label)
        overlay = (meas_scatter * sim_scatter).opts(
            hv.opts.Overlay(legend_position="right")
        )
        return overlay

    def residual_plot(self):
        """Overlayed residual plots for ``self.meas`` and ``self.sim``.

        Each regression exogenous variable gets its own panel showing the
        residuals of both CapData instances overlaid. The single-CapData
        helper ``plotting.get_resid_exog_frame`` stays where it is.

        Returns
        -------
        hv.Layout
        """
        if hv is None:
            raise ImportError(
                "holoviews is required for residual_plot. Install with "
                "`uv add holoviews` or equivalent."
            )
        self._require_meas_and_sim()
        from captest.plotting import get_resid_exog_frame

        meas_exog_names, meas_resid_exog = get_resid_exog_frame(self.meas)
        _sim_exog_names, sim_resid_exog = get_resid_exog_frame(self.sim)

        resid_plots = []
        for exog_id in meas_exog_names:
            meas_plot = (
                hv.Scatter(meas_resid_exog, [exog_id], ["resid", "Timestamp", "source"])
                .redim(x=exog_id)
                .relabel(meas_resid_exog["source"][0])
            )
            sim_plot = (
                hv.Scatter(sim_resid_exog, [exog_id], ["resid", "Timestamp", "source"])
                .redim(x=exog_id)
                .relabel(sim_resid_exog["source"][0])
            )
            resid_plots.append(meas_plot * sim_plot)

        return hv.Layout(resid_plots).opts(
            hv.opts.Overlay(width=500, height=500),
            hv.opts.Scatter(tools=["hover"]),
        )

    # --- derived properties ----------------------------------------------

    @property
    def rep_irr_filter_low(self):
        """Lower irradiance fraction bound derived from ``rep_irr_filter``.

        Equal to ``1 - rep_irr_filter``. Updates automatically whenever
        ``rep_irr_filter`` is reassigned. Pass as the ``low`` argument to
        ``CapData.filter_irr`` with a ``ref_val`` to filter within the
        reporting-irradiance band.
        """
        return 1 - self.rep_irr_filter

    @property
    def rep_irr_filter_high(self):
        """Upper irradiance fraction bound derived from ``rep_irr_filter``.

        Equal to ``1 + rep_irr_filter``. Updates automatically whenever
        ``rep_irr_filter`` is reassigned. Pass as the ``high`` argument to
        ``CapData.filter_irr`` with a ``ref_val`` to filter within the
        reporting-irradiance band.
        """
        return 1 + self.rep_irr_filter

    # --- internal helpers ------------------------------------------------

    def _require_setup(self):
        if self._resolved_setup is None:
            raise RuntimeError("CapTest.setup() must be called first.")

    def _require_meas_and_sim(self):
        if self.meas is None:
            raise RuntimeError("CapTest.meas must be set.")
        if self.sim is None:
            raise RuntimeError("CapTest.sim must be set.")

    def _require_regression_formula(self):
        """Require meas/sim present and each carrying a regression formula.

        Looser than :meth:`_require_setup`: the manual ``rc`` setter only needs
        the regression formula (to validate RHS coverage and the meas/sim
        match), not a fully resolved ``test_setup``. This lets the "prepare each
        CapData, then wrap them in a CapTest" workflow set reporting conditions
        without calling :meth:`setup`.
        """
        self._require_meas_and_sim()
        if self.meas.regression_formula is None or self.sim.regression_formula is None:
            raise RuntimeError(
                "Setting reporting conditions requires a regression formula on "
                "meas and sim. Call CapTest.setup(), or set regression columns "
                "on each CapData, first."
            )

    def _pick_cd(self, which):
        if which == "meas":
            return self.meas
        if which == "sim":
            return self.sim
        raise ValueError(f"which must be 'meas' or 'sim'; got {which!r}.")

    @property
    def resolved_setup(self):
        """Return the resolved TEST_SETUPS entry or raise if setup() not run."""
        self._require_setup()
        return self._resolved_setup


# Silence ruff F401: these are public API; re-imported by `capdata.py`.
__all__ = [
    "CapTest",
    "CapTestResults",
    "TEST_SETUPS",
    "highlight_pvals",
    "load_config",
    "perc_wrap",
    "resolve_test_setup",
    "scatter_bifi_power_tc",
    "scatter_default",
    "scatter_etotal",
    "test_setups",
    "validate_test_setup",
]
