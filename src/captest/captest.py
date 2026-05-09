"""Unified test orchestrator and supporting utilities.

This module houses the ``CapTest`` class, the ``TEST_SETUPS`` registry of
named regression presets, and small formatting helpers (``print_results``,
``highlight_pvals``, ``perc_wrap``) consumed by ``CapTest`` methods that
compare a measured + modeled pair of ``CapData`` instances.

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
from pathlib import Path

import numpy as np
import pandas as pd
import param
import yaml

from captest import util
from captest.capdata import CapData
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


def print_results(test_passed, expected, actual, cap_ratio, capacity, bounds):
    """Print formatted results of a capacity test.

    Parameters
    ----------
    test_passed : tuple of (bool, str)
        Pass/fail flag and bounds string produced by
        ``CapTest.determine_pass_or_fail`` (or the legacy module-level
        ``determine_pass_or_fail`` in ``capdata.py`` until Unit 7 removes it).
    expected : float
        Predicted modeled test output at reporting conditions.
    actual : float
        Predicted measured test output at reporting conditions.
    cap_ratio : float
        Capacity test ratio (``actual / expected``).
    capacity : float
        Tested capacity (``nameplate * cap_ratio``).
    bounds : str
        Human-readable bounds string for the test tolerance.
    """
    if test_passed[0]:
        print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
    else:
        print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))

    print(
        "{:<30s}{:0.3f}".format("Modeled test output:", expected)
        + "\n"
        + "{:<30s}{:0.3f}".format("Actual test output:", actual)
        + "\n"
        + "{:<30s}{:0.3f}".format("Tested output ratio:", cap_ratio)
        + "\n"
        + "{:<30s}{:0.3f}".format("Tested Capacity:", capacity)
    )

    print("{:<30s}{}\n\n".format("Bounds:", bounds))


def highlight_pvals(s):
    """Highlight Series entries >= 0.05 with a yellow background.

    Intended for use with ``pandas.io.formats.style.Styler.apply``. Consumed by
    ``CapTest.captest_results_check_pvalues`` (ported in Unit 7).
    """
    is_greaterthan = s >= 0.05
    return ["background-color: yellow" if v else "" for v in is_greaterthan]


def perc_wrap(p):
    """Return a callable that computes the ``p``-th percentile of a Series.

    Used to build ``TEST_SETUPS[...]['rep_conditions']['func']`` dicts for
    percentile-based reporting irradiance (e.g. 60th percentile POA).

    Parameters
    ----------
    p : numeric
        Percentile in [0, 100].

    Returns
    -------
    callable
        Function that takes a pandas Series or array-like and returns the
        p-th percentile using ``method='nearest'``.
    """

    def numpy_percentile(x):
        return np.percentile(x.T, p, method="nearest")

    numpy_percentile.__name__ = f"perc_wrap({p})"
    return numpy_percentile


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

    Intended for the ``bifi_e2848_etotal`` preset. Thin wrapper around
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
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal": {
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
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_power_tc": {
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
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "rpoa": "mean",
            },
        },
    },
    "e2848_spec_corrected_poa": {
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
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
}

_TEST_SETUP_REQUIRED_KEYS = frozenset(
    {"reg_cols_meas", "reg_cols_sim", "reg_fml", "scatter_plots", "rep_conditions"}
)


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

_PERC_N_PREFIX = "perc_"


def _resolve_perc_string(val):
    """Resolve a "perc_N" string to ``perc_wrap(N)``.

    Non-matching strings pass through unchanged. Malformed ``perc_*`` strings
    raise ``ValueError``.
    """
    if not isinstance(val, str) or not val.startswith(_PERC_N_PREFIX):
        return val
    suffix = val[len(_PERC_N_PREFIX) :]
    if not suffix:
        raise ValueError(f"Malformed percentile string {val!r}: expected 'perc_<int>'.")
    try:
        n = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Malformed percentile string {val!r}: expected 'perc_<int>', "
            f"got suffix {suffix!r}."
        ) from exc
    return perc_wrap(n)


def _resolve_func_strings(func_dict):
    """Resolve ``perc_N`` strings inside a rep_conditions.func dict."""
    if not isinstance(func_dict, dict):
        return func_dict
    return {key: _resolve_perc_string(val) for key, val in func_dict.items()}


def _perc_wrap_to_string(val):
    """Inverse of :func:`_resolve_perc_string`.

    Converts a callable produced by :func:`perc_wrap` back into its
    round-trippable ``"perc_N"`` string form. Non-perc_wrap values pass
    through unchanged.
    """
    if not callable(val):
        return val
    name = getattr(val, "__name__", "")
    prefix = "perc_wrap("
    if name.startswith(prefix) and name.endswith(")"):
        inner = name[len(prefix) : -1]
        try:
            int(inner)
        except ValueError:
            return val
        return f"perc_{inner}"
    return val


def _serialize_rep_conditions(rc):
    """Return a yaml-safe copy of a ``rep_conditions`` dict.

    Recursively walks the dict; ``func`` sub-dict values that are
    ``perc_wrap(N)`` callables are converted to ``"perc_N"`` strings so the
    dict survives a yaml.safe_dump round-trip.
    """
    if not isinstance(rc, dict):
        return rc
    serialized = {}
    for key, val in rc.items():
        if key == "func" and isinstance(val, dict):
            serialized[key] = {k: _perc_wrap_to_string(v) for k, v in val.items()}
        else:
            serialized[key] = copy.deepcopy(val)
    return serialized


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
        "rep_cond_source",
        "sim_days",
        "shade_filter_start",
        "shade_filter_end",
        "ac_nameplate",
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
        "power_temp_coeff",
        "base_temp",
        "spectral_module_type",
        "meas_load_kwargs",
        "sim_load_kwargs",
        "meas_path",
        "sim_path",
        "overrides",
    }
)

# Keys that may appear under the ``overrides`` sub-mapping.
_CAPTEST_OVERRIDE_KEYS = frozenset(
    {"reg_cols_meas", "reg_cols_sim", "reg_fml", "rep_conditions"}
)


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

        ct = CapTest(test_setup="bifi_e2848_etotal", bifaciality=0.15)
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
    rep_cond_source : {"meas", "sim"}
        Which ``CapData.rc`` is used by ``captest_results``. Default
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
    bifaciality, power_temp_coeff, base_temp : float
        Calc-params scalars propagated onto both ``CapData`` instances at
        ``setup()``. See ``_downstream_attrs``.
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
    rep_cond_source = param.Selector(
        objects=["meas", "sim"],
        default="meas",
        doc="Which CapData.rc is used by captest_results.",
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
    test_tolerance = param.String(
        default="- 4",
        doc="Tolerance string forwarded to pass/fail logic.",
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

    # Calc-params scalars propagated to both CapData instances at setup().
    bifaciality = param.Number(
        default=0.0,
        bounds=(0.0, 1.0),
        doc="Bifaciality factor propagated onto both CapData instances.",
    )
    power_temp_coeff = param.Number(
        default=-0.32,
        doc="Power temperature coefficient (percent per degree C).",
    )
    base_temp = param.Number(
        default=25,
        doc="Base temperature for temperature correction (deg C).",
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

    # Class-level tuple of param names to copy onto both CapData instances
    # during setup(). Extending is a one-line edit.
    _downstream_attrs = (
        "bifaciality",
        "power_temp_coeff",
        "base_temp",
        "spectral_module_type",
    )

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

    # --- constructors ----------------------------------------------------

    @classmethod
    def from_params(cls, **kwargs):
        """Construct a CapTest from parameter kwargs.

        Recognizes the non-param kwargs ``meas``, ``sim``, ``meas_path``,
        ``sim_path`` in addition to every declared ``param.*``. If both
        ``meas`` and ``meas_path`` are supplied the pre-built instance
        wins and a warning is emitted (same for ``sim`` / ``sim_path``).

        When both ``meas`` and ``sim`` end up populated, ``setup()`` is
        called automatically. Otherwise the partially-initialized instance
        is returned and the caller finishes the workflow manually.

        Parameters
        ----------
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

        if inst.meas is not None and inst.sim is not None:
            inst.setup()

        return inst

    @classmethod
    def from_yaml(cls, path, key="captest", meas_loader=None, sim_loader=None):
        """Construct a CapTest from a yaml config file.

        Reads the sub-mapping at the given top-level ``key`` of the yaml
        file and delegates to :meth:`from_mapping` with
        ``base_dir=path.parent`` so relative ``meas_path`` / ``sim_path``
        values resolve against the yaml's directory.

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
        )

    @classmethod
    def from_mapping(
        cls, sub, *, key="captest", base_dir=None, meas_loader=None, sim_loader=None
    ):
        """Construct a CapTest from an already-parsed captest sub-mapping.

        Direct-handoff constructor used by downstream wrappers that mutate
        the captest sub-mapping in memory -- applying project-specific
        defaults, promoting fields, injecting paths -- before asking captest
        to validate and build the ``CapTest``. Exposes the same
        validate-and-construct pipeline that ``from_yaml`` runs after
        reading the file, without the file read.

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

        kwargs = {k: v for k, v in sub.items() if k != "overrides"}

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

        # ``null`` (None) in yaml is equivalent to omitting the key.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Inject programmatic-only loader callables. Explicit kwargs win
        # over any value that happened to slip through the sub-mapping
        # (loaders are ``param.Callable`` so yaml would coerce-fail before
        # reaching here, but be defensive).
        if meas_loader is not None:
            kwargs["meas_loader"] = meas_loader
        if sim_loader is not None:
            kwargs["sim_loader"] = sim_loader

        inst = cls.from_params(**kwargs)
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
        return inst

    def to_yaml(self, path, key="captest", merge_into_existing=True):
        """Serialize the curated CapTest configuration to a yaml file.

        The written sub-mapping lives under the top-level ``key`` (default
        ``"captest"``) and contains every scalar ``param.*`` plus
        ``test_setup``, any non-None override of ``reg_fml`` /
        ``reg_cols_meas`` / ``reg_cols_sim`` / ``rep_conditions``,
        ``meas_path`` / ``sim_path`` (when the instance was constructed from
        paths), and non-empty ``meas_load_kwargs`` / ``sim_load_kwargs``.

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

        # Warn once for any non-yaml-serializable user overrides.
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

        sub = self._build_yaml_sub_mapping()

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

    def _build_yaml_sub_mapping(self):
        """Build the dict written under ``key:`` by :meth:`to_yaml`.

        Kept separate from ``to_yaml`` so it is testable in isolation and
        so the merge/write step stays short.
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
        if self.rep_conditions is not None:
            overrides["rep_conditions"] = _serialize_rep_conditions(self.rep_conditions)
        if overrides:
            sub["overrides"] = overrides

        # Remaining scalar params (always written).
        scalar_names = (
            "rep_cond_source",
            "ac_nameplate",
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
            "power_temp_coeff",
            "base_temp",
            "spectral_module_type",
        )
        for name in scalar_names:
            sub[name] = getattr(self, name)

        # Loader kwargs are plain dicts; only write when non-empty so a
        # default-constructed CapTest produces a clean yaml.
        if self.meas_load_kwargs:
            sub["meas_load_kwargs"] = copy.deepcopy(self.meas_load_kwargs)
        if self.sim_load_kwargs:
            sub["sim_load_kwargs"] = copy.deepcopy(self.sim_load_kwargs)

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

    def setup(self, verbose=True):
        """Resolve TEST_SETUPS, propagate scalars, process regression cols.

        Raises ``RuntimeError`` if ``meas`` or ``sim`` is unset. Assigns the
        resolved TEST_SETUPS entry to ``self._resolved_setup`` and returns
        ``self`` for fluent chaining.

        Parameters
        ----------
        verbose : bool, default True
            Forwarded to ``CapData.process_regression_columns``.

        Returns
        -------
        CapTest
            ``self``, for fluent chaining.
        """
        if self.meas is None:
            raise RuntimeError("CapTest.meas must be set before setup().")
        if self.sim is None:
            raise RuntimeError("CapTest.sim must be set before setup().")

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

        # Propagate scalar calc-params onto both CapData instances.
        for name in self._downstream_attrs:
            setattr(self.meas, name, getattr(self, name))
            setattr(self.sim, name, getattr(self, name))

        # Propagate site from meas -> sim with Etc/GMT±N tz for PVsyst.
        self._propagate_sim_site()

        # Wire per-CapData regression state. Deepcopy the regression_cols
        # dict because process_regression_columns mutates it in place.
        self.meas.regression_cols = copy.deepcopy(resolved["reg_cols_meas"])
        self.sim.regression_cols = copy.deepcopy(resolved["reg_cols_sim"])
        self.meas.regression_formula = resolved["reg_fml"]
        self.sim.regression_formula = resolved["reg_fml"]
        self.meas.tolerance = self.test_tolerance
        self.sim.tolerance = self.test_tolerance

        # Run process_regression_columns on both. This also resets
        # data_filtered to data.copy() so any prior filter state is
        # dropped (intended behavior per the design spec).
        self.meas.process_regression_columns(verbose=verbose)
        self.sim.process_regression_columns(verbose=verbose)

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

        - ``e2848_default``, ``bifi_e2848_etotal``, and
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

    def rep_cond(self, which="meas", **overrides):
        """Call ``cd.rep_cond`` with the resolved preset's rep_conditions.

        The preset's ``rep_conditions`` dict (after any ``self.rep_conditions``
        overrides from ``setup()``) is used as the default kwargs. ``overrides``
        is partial-merged on top: top-level keys replace, the nested ``func``
        dict merges one level deep.

        Parameters
        ----------
        which : {'meas', 'sim'}
            Which CapData instance's ``rep_cond`` to call.
        **overrides
            Partial-merged onto the resolved ``rep_conditions`` dict.

        Returns
        -------
        None
            ``cd.rep_cond`` writes to ``cd.rc``.
        """
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
        """Compute the capacity test ratio for ``self.meas`` vs ``self.sim``.

        Picks reporting conditions from ``self.meas.rc`` or ``self.sim.rc``
        based on ``self.rep_cond_source``. Uses ``self.ac_nameplate`` for the
        tested-capacity printout and ``self.test_tolerance`` (via
        ``self.determine_pass_or_fail``) for the pass/fail result.

        Parameters
        ----------
        check_pvalues : bool, default False
            When True, coefficients with a p-value above ``pval`` are zeroed
            before prediction.
        pval : float, default 0.05
            P-value cutoff used when ``check_pvalues`` is True.
        print_res : bool, default True
            When True, prints the formatted results.

        Returns
        -------
        float
            Capacity test ratio ``actual / expected``.
        """
        self._require_meas_and_sim()
        if self.meas.regression_formula != self.sim.regression_formula:
            return warnings.warn(
                "CapData objects do not have the same regression formula."
            )

        if self.rep_cond_source == "meas":
            rc = self.meas.rc
        else:
            rc = self.sim.rc

        if print_res:
            print(f"Using reporting conditions from {self.rep_cond_source}. \n")

        # predict_with_pvalue_check is a single-CapData helper that stays in
        # capdata.py. Imported lazily to avoid importing holoviews-heavy
        # capdata internals at module-load time for callers that never
        # compute cap ratios (e.g. notebooks that only use setup + plots).
        from captest.capdata import predict_with_pvalue_check

        pval_threshold = pval if check_pvalues else None
        actual = predict_with_pvalue_check(
            self.meas, rc=rc, pval_threshold=pval_threshold
        )
        expected = predict_with_pvalue_check(
            self.sim, rc=rc, pval_threshold=pval_threshold
        )
        cap_ratio = actual / expected
        if cap_ratio < 0.01:
            cap_ratio *= 1000
            actual *= 1000
            warnings.warn(
                "Capacity ratio and actual capacity multiplied by 1000"
                " because the capacity ratio was less than 0.01."
            )
        capacity = self.ac_nameplate * cap_ratio

        if print_res:
            test_passed = self.determine_pass_or_fail(cap_ratio)
            print_results(
                test_passed, expected, actual, cap_ratio, capacity, test_passed[1]
            )

        return cap_ratio

    def captest_results_check_pvalues(self, print_res=False, **kwargs):
        """Compute cap ratio with and without p-value filtering.

        Parameters
        ----------
        print_res : bool, default False
            Forwarded to both internal ``captest_results`` calls.
        **kwargs
            Forwarded to ``captest_results``. Do not pass ``check_pvalues``;
            this method sets it explicitly for each internal call.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with p-values and parameter values for both
            ``self.meas`` and ``self.sim``. P-values >= 0.05 are highlighted.
        """
        self._require_meas_and_sim()
        das_pvals = self.meas.regression_results.pvalues
        sim_pvals = self.sim.regression_results.pvalues
        das_params = self.meas.regression_results.params
        sim_params = self.sim.regression_results.params

        df_pvals = pd.DataFrame([das_pvals, sim_pvals, das_params, sim_params])
        df_pvals = df_pvals.transpose()
        df_pvals.rename(
            columns={
                0: "das_pvals",
                1: "sim_pvals",
                2: "das_params",
                3: "sim_params",
            },
            inplace=True,
        )

        cap_ratio = self.captest_results(
            print_res=print_res, check_pvalues=False, **kwargs
        )
        cap_ratio_check_pvalues = self.captest_results(
            print_res=print_res, check_pvalues=True, **kwargs
        )

        cap_ratio_rounded = np.round(cap_ratio, decimals=4) * 100
        cap_ratio_check_pvalues_rounded = (
            np.round(cap_ratio_check_pvalues, decimals=4) * 100
        )

        print("{:.3f}% - Cap Ratio".format(cap_ratio_rounded))
        print(
            "{:.3f}% - Cap Ratio after pval check".format(
                cap_ratio_check_pvalues_rounded
            )
        )

        return df_pvals.style.format("{:20,.5f}").apply(
            highlight_pvals, subset=["das_pvals", "sim_pvals"]
        )

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
    "TEST_SETUPS",
    "highlight_pvals",
    "load_config",
    "perc_wrap",
    "print_results",
    "resolve_test_setup",
    "scatter_bifi_power_tc",
    "scatter_default",
    "scatter_etotal",
    "validate_test_setup",
]
