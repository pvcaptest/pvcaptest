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
import textwrap
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import param
import yaml

from captest import util
from captest.calcparams import DOWNSTREAM_PARAMS
from captest.capdata import CapData
from captest.filters import wrap_year_end
from captest.plotting import ScatterBifiPowerTc, ScatterPlot
from captest.setup import (
    DerivationError,
    RepConditions,
    SetupFitError,
    TestSetup,
    check_project_fit,
    derive,
)
from captest.util import perc_wrap, to_native

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


#: Scatter-plot callables a setup may name under ``scatter_plots``.
SCATTER_REGISTRY = {
    "default": scatter_default,
    "etotal": scatter_etotal,
    "bifi_power_tc": scatter_bifi_power_tc,
}

#: Directory of shipped preset documents.
SETUPS_DIR = Path(str(resources.files("captest").joinpath("setups")))


def _suggest_unknown_key(unknown, known):
    """Return a 'did you mean X?' hint or empty string."""
    matches = difflib.get_close_matches(unknown, list(known), n=1)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _check_scatter_name(name):
    """Raise ``ValueError`` (with a hint) if ``name`` is not a scatter callable."""
    if name not in SCATTER_REGISTRY:
        raise ValueError(
            f"Unknown scatter_plots {name!r}."
            f"{_suggest_unknown_key(name, SCATTER_REGISTRY)}"
        )


def load_presets(directory=None):
    """Load every ``*.yaml`` preset in ``directory`` (default ``SETUPS_DIR``).

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Directory of preset documents. Defaults to ``SETUPS_DIR``.

    Returns
    -------
    dict
        Preset name -> :class:`captest.setup.TestSetup`, sorted by name.

    Raises
    ------
    pydantic.ValidationError, ValueError
        A preset that fails to validate raises at import; a broken shipped
        preset must never be silently absent.
    """
    directory = Path(directory) if directory is not None else SETUPS_DIR
    presets = {}
    for path in sorted(directory.glob("*.yaml")):
        tsd = TestSetup.load(path)
        if tsd.name != path.stem:
            raise ValueError(f"{path.name}: name {tsd.name!r} must equal the file stem")
        _check_scatter_name(tsd.scatter_plots)
        presets[tsd.name] = tsd
    return presets


#: Registry of shipped capacity-test presets.
TEST_SETUPS = load_presets()


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
        for name in TEST_SETUPS:
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
            print(textwrap.fill(setup.description, 60))


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


#: Overrides merged onto the preset rather than replacing it; empty is no change.
_MERGED_OVERRIDES = ("reg_cols_meas", "reg_cols_sim", "rep_conditions")

_RESOLVE_KEYS = (
    "reg_cols_meas",
    "reg_cols_sim",
    "reg_fml",
    "rep_conditions",
    "params",
    "scatter_plots",
)


def resolve_test_setup(name, overrides=None):
    """Resolve a preset by name plus optional overrides into a ``TestSetup``.

    Parameters
    ----------
    name : str
        Key into ``TEST_SETUPS`` or the literal ``"custom"``.
    overrides : dict or None
        Any of ``reg_cols_meas`` / ``reg_cols_sim`` (merged key by key onto
        the preset's side; ``None`` removes a term), ``rep_conditions``
        (partial-merged: top-level keys replace, ``func`` merges one level
        deep), and ``reg_fml`` / ``params`` / ``scatter_plots`` (replace).
        ``"custom"`` has no base and requires complete ``reg_cols_meas``,
        ``reg_cols_sim`` and ``reg_fml``; it also accepts a ``description``.

    Returns
    -------
    captest.setup.TestSetup
        For a named preset whose overrides change nothing but provenance,
        the ``TEST_SETUPS`` entry itself (``derived_from`` unset, same
        ``content_digest()``).

    Raises
    ------
    KeyError
        Unknown preset name.
    ValueError
        Unknown override key, unknown ``scatter_plots`` name, missing
        ``custom`` requirements, or a ``reg_cols`` override that cannot be
        applied (see :class:`captest.setup.DerivationError`).
    """
    overrides = dict(overrides or {})
    allowed = set(_RESOLVE_KEYS) | ({"description"} if name == "custom" else set())
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown override key(s) {sorted(unknown)}")
    if name == "custom":
        missing = {"reg_cols_meas", "reg_cols_sim", "reg_fml"} - set(overrides)
        if missing:
            raise ValueError(
                "test_setup='custom' requires overrides with keys: "
                f"['reg_cols_meas', 'reg_cols_sim', 'reg_fml']; missing: {sorted(missing)}"
            )
        doc = {
            "name": "custom",
            "description": overrides.get("description", ""),
            "reg_fml": overrides["reg_fml"],
            "meas": {"reg_cols": overrides["reg_cols_meas"]},
            "sim": {"reg_cols": overrides["reg_cols_sim"]},
            "params": overrides.get("params") or {},
            "rep_conditions": overrides.get("rep_conditions") or {},
            "scatter_plots": overrides.get("scatter_plots") or "default",
        }
        _check_scatter_name(doc["scatter_plots"])
        return TestSetup.model_validate(doc)
    if name not in TEST_SETUPS:
        available = sorted(TEST_SETUPS) + ["custom"]
        raise KeyError(f"Unknown test_setup={name!r}. Available: {available}")
    base = TEST_SETUPS[name]
    if not any(v is not None for v in overrides.values()):
        return base  # the preset itself, provenance and digest untouched
    if overrides.get("scatter_plots") is not None:
        _check_scatter_name(overrides["scatter_plots"])
    rep_conditions = None
    if overrides.get("rep_conditions"):
        rep_conditions = _merge_rep_conditions(
            base.rep_conditions.model_dump(mode="json"), overrides["rep_conditions"]
        )
    try:
        derived = derive(
            base,
            reg_fml=overrides.get("reg_fml"),
            reg_cols_meas=overrides.get("reg_cols_meas"),
            reg_cols_sim=overrides.get("reg_cols_sim"),
            params=overrides.get("params"),
            rep_conditions=rep_conditions,
            scatter_plots=overrides.get("scatter_plots"),
        )
    except DerivationError as exc:
        raise ValueError(str(exc)) from exc
    # Overrides that change nothing keep the preset's identity: to_yaml writes
    # only the difference, so a reload must resolve to the same setup.
    if derived.model_copy(update={"derived_from": base.derived_from}) == base:
        return base
    return derived


# --- yaml loading ---------------------------------------------------------


def _serialize_rep_conditions(rc):
    """Return a yaml-safe copy of a ``rep_conditions`` dict.

    ``func`` values are already ``"mean"`` / ``"median"`` / ``"perc_N"``
    strings (the document form); numpy scalars are coerced to native Python
    types via ``util.to_native`` so the dict survives ``yaml.safe_dump``.
    """
    if not isinstance(rc, dict):
        return rc
    return {key: to_native(copy.deepcopy(val)) for key, val in rc.items()}


def _reg_cols_diff(base, resolved):
    """Overrides that turn ``base`` into ``resolved`` (document form).

    Changed or added terms are written as document nodes; terms ``base``
    has and ``resolved`` lacks are written as ``None``.

    Parameters
    ----------
    base, resolved : dict
        Formula variable -> node (``Side.reg_cols``).

    Returns
    -------
    dict
    """
    diff = {}
    for var, node in resolved.items():
        if base.get(var) != node:
            diff[var] = node.model_dump(mode="json")
    for var in base:
        if var not in resolved:
            diff[var] = None
    return diff


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
        The sub-mapping at ``key``, as written. ``rep_conditions.func``
        values stay ``"perc_N"`` strings (the document form). Does NOT
        validate against ``CapTest`` param types; ``CapTest.from_yaml`` does
        that.

    Raises
    ------
    KeyError
        If ``key`` is not present at the top level of the yaml file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
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
    return sub


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
        "meas_prep",
        "sim_prep",
        "reporting_conditions_values",
    }
)

# Keys that may appear under the ``overrides`` sub-mapping.
_CAPTEST_OVERRIDE_KEYS = frozenset(
    {
        "reg_cols_meas",
        "reg_cols_sim",
        "reg_fml",
        "rep_conditions",
        "params",
        "scatter_plots",
    }
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
    ``tst.meas.filter_*(...)``, ``tst.meas.rep_cond(...)``, and
    ``tst.meas.fit_regression()`` by hand.

    Typical workflows
    -----------------
    1. Programmatic::

        tst = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd,
            sim=sim_cd,
            ac_nameplate=125_000,
            test_tolerance="- 4",
        )
        # ``from_params`` runs ``setup()`` automatically because both meas
        # and sim were supplied as pre-built CapData instances.

    2. From a yaml file::

        tst = CapTest.from_yaml("./config.yaml")

    3. Bare + manual::

        tst = CapTest(test_setup="bifi_e2848_etotal_rear_shade_sim", bifaciality=0.15)
        tst.meas = my_meas_cd
        tst.sim = my_sim_cd
        tst.setup()

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
    reg_cols_meas, reg_cols_sim : dict or None
        Formula variable -> node in document form (see
        :mod:`captest.setup`), merged key by key onto the preset's measured /
        modeled regression columns at ``setup()``: a present key replaces
        that variable's whole node, an absent key keeps the preset's node,
        and a value of ``None`` removes the variable. Under
        ``test_setup="custom"`` both sides must be complete.
    rep_conditions : dict or None
        If set, partial-merged onto the preset's ``rep_conditions`` at
        ``setup()``. Top-level keys replace; the nested ``func`` dict is
        merged one level deep so users can override only a single
        variable's aggregation.
    params : dict or None
        Required test-level parameter values (see
        :attr:`captest.setup.TestSetup.params`); replaces the preset's
        ``params`` wholesale. Checked against the effective value each
        calculation would receive at ``setup()``.
    scatter_plots_name : str or None
        Name in ``SCATTER_REGISTRY`` overriding the preset's scatter plot.
        Written to yaml as ``overrides.scatter_plots``.
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
        Belongs with the ``*_rear_shade_meas`` presets. The
        ``*_rear_shade_sim`` presets already carry rear shading in the modeled
        rear irradiance (``rpoa_pvsyst``) and declare ``params: {rear_shade:
        0}``, so ``setup()`` raises :class:`~captest.setup.SetupFitError` for
        a non-zero ``rear_shade`` rather than double-counting the loss.
    meas_loader, sim_loader : callable or None
        Programmatic-only data-loader callables. Default resolution when
        ``None``: ``captest.io.load_data`` and ``captest.io.load_pvsyst``
        respectively. Not serialized to yaml.
    meas_load_kwargs, sim_load_kwargs : dict or None
        Plain-dict kwargs splatted into the loaders.
    meas_prep, sim_prep : list of dict
        Serialized data-preparation pipelines (``CapData.prep_to_config``
        dicts) replayed onto the corresponding ``CapData`` immediately after
        every load — ``from_params``/``from_mapping``/``from_yaml`` when the
        side is built from a path, and ``reload``. Prep mutates ``data`` and
        is not idempotent, so ``setup()`` and ``run_test()`` never replay it,
        and a side supplied as a pre-built ``CapData`` keeps its config but
        does not apply it (a ``UserWarning`` names the skipped steps).

    Attributes
    ----------
    resolved_setup : captest.setup.TestSetup or None
        The complete setup resolved from ``test_setup`` plus the overrides
        by the last ``setup()``; ``None`` before ``setup()`` has run.

    See Also
    --------
    rep_irr_filter_low : Lower reporting-irradiance fraction bound.
    rep_irr_filter_high : Upper reporting-irradiance fraction bound.

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
        doc="Merged key by key onto the preset's measured regression columns; "
        "a value of None removes that term.",
    )
    reg_cols_sim = param.Dict(
        default=None,
        allow_None=True,
        doc="Merged key by key onto the preset's modeled regression columns; "
        "a value of None removes that term.",
    )
    params = param.Dict(
        default=None,
        allow_None=True,
        doc="Required test-level parameter values for this setup (see "
        "captest.setup.TestSetup.params). Replaces the preset's wholesale.",
    )
    scatter_plots_name = param.String(
        default=None,
        allow_None=True,
        doc="Name in SCATTER_REGISTRY overriding the preset's scatter plot. "
        "Written to yaml as overrides.scatter_plots.",
    )
    resolved_setup = param.ClassSelector(
        class_=TestSetup,
        default=None,
        allow_None=True,
        doc="The complete TestSetup resolved by setup(); None before setup().",
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
        "(rep_cond / the tst.rc setter, both routed through _set_rc) and is "
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
            "instance only (see _downstream_attrs_meas_only). Set a non-zero "
            "value only with the '*_rear_shade_meas' presets; the "
            "'*_rear_shade_sim' presets carry rear shading in the modeled rear "
            "irradiance and declare params: {rear_shade: 0}, so setup() raises "
            "SetupFitError for a non-zero value with them."
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

    # Declared data-preparation pipelines. Serialized to yaml, replayed once
    # per load (never by setup() or run_test(): prep is not idempotent).
    meas_prep = param.List(
        default=[],
        doc=(
            "Serialized prep-step configs replayed onto `meas` immediately "
            "after every load, before setup()."
        ),
    )
    sim_prep = param.List(
        default=[],
        doc=(
            "Serialized prep-step configs replayed onto `sim` immediately "
            "after every load, before setup()."
        ),
    )

    # Param names copied onto the CapData instances during setup(); the same
    # list the TestSetup ``params`` tier-1 check uses. Names also listed in
    # _downstream_attrs_meas_only are copied onto meas only; all others onto
    # both. Every name in _downstream_attrs_meas_only MUST also appear in
    # _downstream_attrs (guarded by a subset test).
    _downstream_attrs = DOWNSTREAM_PARAMS
    _downstream_attrs_meas_only = ("rear_shade",)

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
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
                "tst.rc must be a one-row DataFrame, a pandas Series, or a dict "
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
        becomes ``tst.rc`` and ``rc_source`` (a source-change ``UserWarning``
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

    # --- data preparation -------------------------------------------------

    def _replay_prep(self, side):
        """Replay ``<side>_prep`` onto the freshly-loaded CapData for ``side``.

        Called immediately after a load and before ``setup()``. Prep mutates
        ``data`` and is not idempotent, so this is the only place a stored
        prep config is applied — ``setup()`` and ``run_test()`` never run it.

        Parameters
        ----------
        side : {'meas', 'sim'}
            Which side's stored prep config to replay. A no-op when that
            config is empty or the side's ``CapData`` is unset.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Propagated from ``CapData.run_prep`` if the target ``CapData``
            already has an applied prep chain.
        """
        config = self.meas_prep if side == "meas" else self.sim_prep
        capdata = getattr(self, side)
        if not config or capdata is None:
            return
        capdata.run_prep(config)

    def _warn_prep_not_applied(self, side):
        """Warn that a stored prep config was skipped for a pre-built CapData.

        A pre-built ``CapData`` may already have been prepped by the caller,
        and prep is not idempotent, so the config is kept (it still
        round-trips through ``to_yaml``) but never applied.

        Parameters
        ----------
        side : {'meas', 'sim'}
            Which side was supplied pre-built. A no-op when that side's
            stored prep config is empty.

        Returns
        -------
        None
        """
        config = self.meas_prep if side == "meas" else self.sim_prep
        if not config:
            return
        warnings.warn(
            f"{len(config)} stored '{side}_prep' steps were not applied — "
            f"'{side}' was supplied as a pre-built CapData, not loaded from a "
            "path. Apply them explicitly with: "
            f"tst.{side}.run_prep(tst.{side}_prep)",
            # warn -> this helper -> from_params -> the user's call site.
            stacklevel=3,
        )

    # --- constructors ----------------------------------------------------

    @classmethod
    def from_params(cls, run_setup=True, verbose=True, **kwargs):
        """Construct a CapTest from parameter kwargs.

        Recognizes the non-param kwargs ``meas``, ``sim``, ``meas_path``,
        ``sim_path`` in addition to every declared ``param.*``. If both
        ``meas`` and ``meas_path`` are supplied the pre-built instance
        wins and a warning is emitted (same for ``sim`` / ``sim_path``).

        A side built from a path has its declared prep pipeline
        (``meas_prep`` / ``sim_prep``) replayed onto the loaded ``CapData``
        immediately, before any ``setup()`` — including when
        ``run_setup=False``. A side supplied as a pre-built ``CapData``
        keeps its prep config but does not apply it (prep is not idempotent
        and the object may already be prepped); a ``UserWarning`` names the
        skipped steps.

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
            ``tst.setup()`` or ``tst.run_test()`` proceeds normally.
        verbose : bool, default True
            Forwarded to the automatic ``setup()``.
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
            inst._warn_prep_not_applied("meas")
        elif meas is not None:
            inst.meas = meas
            inst._warn_prep_not_applied("meas")
        elif meas_path is not None:
            load_kwargs = inst.meas_load_kwargs or {}
            inst.meas = _meas_loader()(meas_path, **load_kwargs)
            inst._replay_prep("meas")

        # Wire up sim.
        if sim is not None and sim_path is not None:
            warnings.warn(
                "Both 'sim' and 'sim_path' supplied; using the pre-built "
                "sim CapData and ignoring sim_path.",
                stacklevel=2,
            )
            inst.sim = sim
            inst._warn_prep_not_applied("sim")
        elif sim is not None:
            inst.sim = sim
            inst._warn_prep_not_applied("sim")
        elif sim_path is not None:
            load_kwargs = inst.sim_load_kwargs or {}
            inst.sim = _sim_loader()(sim_path, **load_kwargs)
            inst._replay_prep("sim")

        if run_setup and inst.meas is not None and inst.sim is not None:
            inst.setup(verbose=verbose)

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
        Serialized ``meas_prep`` / ``sim_prep`` pipelines, by contrast, are
        applied at load — see :meth:`from_params`.

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
        side via ``CapData.run_pipeline``). Serialized ``meas_prep`` /
        ``sim_prep`` pipelines are the exception: they reach
        :meth:`from_params` as parameters and are applied to each
        path-loaded side at load, before any ``setup()``. Manual
        reporting-conditions values (``reporting_conditions_values`` with
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

        # Lift override keys into direct kwargs; values pass straight to the
        # params and the TestSetup model validates them at setup().
        # ``overrides.scatter_plots`` is the ``scatter_plots_name`` param.
        for k in _CAPTEST_OVERRIDE_KEYS:
            if overrides.get(k) is not None:
                kwargs["scatter_plots_name" if k == "scatter_plots" else k] = overrides[
                    k
                ]

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
            if inst.resolved_setup is not None:
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

        The outgoing side's applied prep chain is preserved the same way,
        into ``<side>_prep``, and — unlike the filters — is replayed
        immediately onto the freshly loaded data, before the per-side
        ``setup()``. A reload is the supported way to re-prep, since the
        loader supplies a fresh un-prepped frame. Note that the stash
        rewrites ``<side>_prep`` in place: a hand-written config of partial
        step dicts is replaced by the fully-expanded ``to_config()`` form
        (every step param, defaults included). The two are behaviorally
        equivalent, but the expanded form is what a later ``to_yaml`` and
        any read of ``tst.<side>_prep`` will show.

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
            (``tst.reload('sim', path='new.CSV').run_test(side='sim')``).

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
        if outgoing is not None and outgoing.prep:
            setattr(self, f"{side}_prep", outgoing.prep_to_config())
        if side == "meas":
            loader = self.meas_loader or _default_meas_loader()
            self.meas = loader(stored_path, **(self.meas_load_kwargs or {}))
        else:
            loader = self.sim_loader or _default_sim_loader()
            self.sim = loader(stored_path, **(self.sim_load_kwargs or {}))
        self._replay_prep(side)
        self.setup(verbose=verbose, side=side)
        return self

    def to_yaml(self, path, key="captest", merge_into_existing=True):
        """Serialize the curated CapTest configuration to a yaml file.

        The written sub-mapping is :meth:`to_mapping`'s return value. It
        lives under the top-level ``key`` (default
        ``"captest"``) and contains every scalar ``param.*`` plus
        ``test_setup``, an ``overrides`` sub-mapping (see below),
        ``meas_path`` / ``sim_path`` (when the instance was constructed from
        paths), and non-empty ``meas_load_kwargs`` / ``sim_load_kwargs``.

        The filter pipelines of ``meas`` and ``sim`` are written as
        ``meas_filters`` / ``sim_filters`` (lists of filter-step config dicts
        from :meth:`CapData.filters_to_config`, or the side's pending config
        when its chain is empty), each only when non-empty; ``from_yaml``
        stores them as pending pipelines that :meth:`run_test` replays. The
        prep pipelines are written the same way as ``meas_prep`` /
        ``sim_prep``; ``from_yaml`` applies those at load.
        ``overrides.rep_conditions`` is written whenever
        :attr:`rep_conditions` is set, also beside a ``RepCond`` step or a
        manual ``rc_source``: it is part of the resolved setup's identity,
        and loading never applies it (a replayed ``RepCond`` step uses its
        own arguments, and a manual rc is restored from
        ``reporting_conditions_values``).

        ``overrides`` is written in document form. For a named preset,
        ``reg_cols_meas`` / ``reg_cols_sim`` hold only the difference from
        the preset (changed formula variables as nodes, ``null`` for a
        variable the preset has and the resolved setup lacks) and
        ``reg_fml`` only when it differs; under ``test_setup: custom`` both
        sides and the formula are written in full. ``params`` and
        ``scatter_plots`` (from :attr:`scatter_plots_name`) are written when
        set. ``from_yaml`` merging the file back onto the same preset
        reproduces the resolved setup. ``meas``, ``sim``,
        ``regression_results``, :attr:`resolved_setup`, and the loader
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
                with path.open("r", encoding="utf-8") as fh:
                    existing = yaml.safe_load(fh)
                if isinstance(existing, dict):
                    root_doc = existing
            except (OSError, yaml.YAMLError):  # pragma: no cover - rare IO/parse
                root_doc = {}
        root_doc[key] = sub

        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(root_doc, fh, sort_keys=False)

    def to_mapping(self):
        """Return the curated config mapping ``to_yaml`` writes under ``key``.

        The public dict counterpart of :meth:`to_yaml` and the symmetric
        inverse of :meth:`from_mapping`. Emits the same programmatic-only
        attribute warning as ``to_yaml`` (loader callables).

        Returns
        -------
        dict
            The captest sub-mapping (scalars, overrides, paths, pipelines).
        """
        self._warn_unserializable()
        return self._build_yaml_sub_mapping()

    def _warn_unserializable(self):
        """Warn once for any non-yaml-serializable user overrides.

        Loader callables cannot be represented in the yaml config; name them
        in a single ``UserWarning`` so the omission is visible at export time.
        """
        unserializable = []
        if self.meas_loader is not None:
            unserializable.append("meas_loader")
        if self.sim_loader is not None:
            unserializable.append("sim_loader")
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
        running is lossless), else the key is omitted. Embeds each side's
        prep pipeline as ``meas_prep``/``sim_prep`` under the same three-way
        rule (applied prep chain, else the stored ``meas_prep``/``sim_prep``
        config, else the key is omitted). Writes
        ``overrides.rep_conditions`` whenever :attr:`rep_conditions` is set,
        including beside a ``RepCond`` step or a manual ``rc_source``, so the
        reloaded ``resolved_setup`` keeps the original's ``content_digest``.
        """
        sub = {"test_setup": self.test_setup}

        # Paths are written only when the instance was constructed from
        # paths; we remember the raw (possibly relative) string in
        # ``_meas_path``/``_sim_path``.
        if self._meas_path is not None:
            sub["meas_path"] = str(self._meas_path)
        if self._sim_path is not None:
            sub["sim_path"] = str(self._sim_path)

        overrides = {}
        # Always resolve from the *current* params, never from the cached
        # ``resolved_setup``: an override edited after setup() must be what
        # gets written, and a changed ``test_setup`` must not be paired with
        # the previous preset's document.
        resolved = resolve_test_setup(self.test_setup, self._collect_overrides())
        if self.test_setup == "custom":
            overrides["reg_cols_meas"] = resolved.meas.model_dump(mode="json")[
                "reg_cols"
            ]
            overrides["reg_cols_sim"] = resolved.sim.model_dump(mode="json")["reg_cols"]
            overrides["reg_fml"] = resolved.reg_fml
        else:
            preset = TEST_SETUPS[self.test_setup]
            for side in ("meas", "sim"):
                diff = _reg_cols_diff(
                    getattr(preset, side).reg_cols, getattr(resolved, side).reg_cols
                )
                if diff:
                    overrides[f"reg_cols_{side}"] = diff
            if resolved.reg_fml != preset.reg_fml:
                overrides["reg_fml"] = resolved.reg_fml
        if self.params is not None:
            overrides["params"] = {k: to_native(v) for k, v in self.params.items()}
        if self.scatter_plots_name is not None:
            overrides["scatter_plots"] = self.scatter_plots_name
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
        meas_prep = (
            self.meas.prep_to_config()
            if self.meas is not None and self.meas.prep
            else copy.deepcopy(self.meas_prep)
        )
        sim_prep = (
            self.sim.prep_to_config()
            if self.sim is not None and self.sim.prep
            else copy.deepcopy(self.sim_prep)
        )
        # Written even beside a RepCond step or a manual rc: it is part of
        # the resolved setup's identity (content_digest), and nothing applies
        # it on load. Replayed RepCond steps carry their own kwargs and a
        # manual rc is restored from reporting_conditions_values, so it only
        # feeds resolved_setup and the defaults of a later tst.rep_cond().
        if self.rep_conditions is not None:
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

        if meas_prep:
            sub["meas_prep"] = meas_prep
        if sim_prep:
            sub["sim_prep"] = sim_prep

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

    def _collect_overrides(self):
        """The setup overrides currently set on this instance.

        ``None`` values are skipped, and so are empty ``reg_cols_meas`` /
        ``reg_cols_sim`` / ``rep_conditions`` mappings (merging nothing is
        no change), so an untouched test resolves to the preset itself (same
        provenance and digest). An empty ``params`` is kept: it replaces the
        preset's constraints wholesale, removing them.

        Returns
        -------
        dict
            Keyword overrides for :func:`resolve_test_setup`.
        """
        overrides = {}
        for name in (
            "reg_cols_meas",
            "reg_cols_sim",
            "reg_fml",
            "rep_conditions",
            "params",
        ):
            val = getattr(self, name)
            if val is None or (name in _MERGED_OVERRIDES and not val):
                continue
            overrides[name] = val
        if self.scatter_plots_name is not None:
            overrides["scatter_plots"] = self.scatter_plots_name
        return overrides

    def _prepare_sides(self, sides):
        """Propagate downstream params and ``meas.site`` onto ``sides``.

        The preparation :meth:`setup` performs before tier 2 and evaluation:
        names in ``_downstream_attrs`` are copied onto each targeted
        ``CapData`` (``_downstream_attrs_meas_only`` onto meas only), and for
        sim-side setup ``meas.site`` is copied onto ``sim`` with a fixed-offset
        tz for PVsyst. Neither touches ``data``.
        """
        for name in self._downstream_attrs:
            if "meas" in sides:
                setattr(self.meas, name, getattr(self, name))
            if "sim" in sides and name not in self._downstream_attrs_meas_only:
                setattr(self.sim, name, getattr(self, name))
        # Reads meas.site but mutates only sim, so it runs for sim-side setup.
        if "sim" in sides and self.meas is not None:
            self._propagate_sim_site()

    def check_fit(self, side="both"):
        """Tier-2 project-fit findings for the resolved setup, without running setup.

        Resolves the preset and overrides, propagates the downstream
        parameters onto the targeted ``CapData`` instances exactly as
        :meth:`setup` would, and returns
        :func:`captest.setup.check_project_fit`'s findings for each side.
        Nothing is written to ``data`` and ``resolved_setup`` is untouched.

        Parameters
        ----------
        side : {"both", "meas", "sim"}
            Which side(s) to check.

        Returns
        -------
        list of captest.setup.FitError
            Empty when every checked side fits.

        Raises
        ------
        ValueError
            If ``side`` is invalid, or the setup cannot be resolved.
        RuntimeError
            If a ``CapData`` targeted by ``side`` is unset.
        """
        if side not in ("meas", "sim", "both"):
            raise ValueError(f"side must be 'meas', 'sim', or 'both', got {side!r}.")
        sides = ("meas", "sim") if side == "both" else (side,)
        resolved = resolve_test_setup(self.test_setup, self._collect_overrides())
        for s in sides:
            if getattr(self, s) is None:
                raise RuntimeError(f"CapTest.{s} must be set before check_fit().")
        self._prepare_sides(sides)
        errors = []
        for s in sides:
            errors.extend(check_project_fit(resolved, s, getattr(self, s)))
        return errors

    def setup(self, verbose=True, side="both"):
        """Resolve the setup, propagate scalars, check fit, process regression cols.

        Resolves ``test_setup`` plus the overrides (``reg_cols_meas`` /
        ``reg_cols_sim`` merged key by key, ``reg_fml``, ``rep_conditions``,
        ``params``, ``scatter_plots_name``) to a
        :class:`captest.setup.TestSetup`, propagates the downstream params,
        runs the tier-2 project-fit check on each targeted side, then
        evaluates the regression columns. Raises ``RuntimeError`` if any
        ``CapData`` targeted by ``side`` is unset. Assigns the resolved setup
        to :attr:`resolved_setup` and returns ``self`` for fluent chaining.

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
        :attr:`resolved_setup` is still replaced by the setup resolved from
        the current overrides, which describes both sides: if the overrides
        changed since the other side was wired, that side's
        ``regression_cols`` / ``regression_formula`` no longer match
        ``resolved_setup`` until it is set up again (``setup()`` or
        ``setup(side=<other side>)``).

        When the tier-2 check fails, nothing is evaluated: ``resolved_setup``
        and each side's regression state stay those of the last successful
        ``setup()``, so the pair remains consistent (the downstream params
        and the year-end wrap have already been applied).

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
            If ``side`` is not ``'meas'``, ``'sim'``, or ``'both'``, or the
            setup cannot be resolved.
        captest.setup.SetupFitError
            If the resolved setup does not fit a targeted side's project
            (tier 2); raised before any column is written.
        RuntimeError
            If a ``CapData`` targeted by ``side`` is unset.
        """
        if side not in ("meas", "sim", "both"):
            raise ValueError(f"side must be 'meas', 'sim', or 'both', got {side!r}.")
        sides = ("meas", "sim") if side == "both" else (side,)
        for s in sides:
            if getattr(self, s) is None:
                raise RuntimeError(f"CapTest.{s} must be set before setup().")

        # Resolve first: a resolution error must leave sim.data untouched.
        resolved = resolve_test_setup(self.test_setup, self._collect_overrides())

        # Auto-wrap sim.data when measured spans (within 60 days of) a year
        # boundary. Idempotent and reversible — re-running setup() or toggling
        # auto_wrap_sim restores the appropriate state. The wrap mutates
        # sim.data while reading the meas span, so it is skipped for
        # side='meas' (meas-side setup must not touch sim).
        if "sim" in sides and self.meas is not None:
            self._maybe_wrap_sim_year_end()

        self._prepare_sides(sides)

        # Tier 2 runs against the prepared CapData and before any column is
        # written, so a setup that does not fit leaves ``data`` untouched.
        fit_errors = []
        for s in sides:
            fit_errors.extend(check_project_fit(resolved, s, getattr(self, s)))
        if fit_errors:
            raise SetupFitError(fit_errors)
        self.resolved_setup = resolved

        # Wire per-CapData regression state on each targeted side. A shallow
        # copy suffices: the node tree is immutable, and
        # process_regression_columns replaces (never mutates) the values.
        for s in sides:
            cd = getattr(self, s)
            cd.regression_cols = dict(getattr(resolved, s).reg_cols)
            cd.regression_formula = resolved.reg_fml
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
        that resolved setup's ``scatter_plots`` field names a function in
        :data:`SCATTER_REGISTRY` matched to the setup's regression formula.
        This method picks ``self.meas`` or ``self.sim`` and forwards it, plus
        any keyword arguments, to that registered function.

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
            Plotting options forwarded to the registered scatter function.

        Returns
        -------
        holoviews.Layout
            Scatter plot layout for the selected measured or modeled data.

        Examples
        --------
        Plot measured data with the default options::

            tst.scatter_plots()

        Plot modeled data, split points into AM and PM groups, and add a
        linked timeseries panel::

            tst.scatter_plots(which="sim", split_day=True, timeseries=True)

        Add a temperature-corrected power panel for a setup that uses raw
        power in the regression::

            tst.scatter_plots(tc_power=True, tc_mode="add_panel")
        """
        cd = self._pick_cd(which)
        self._require_setup()
        return SCATTER_REGISTRY[self.resolved_setup.scatter_plots](cd, **kwargs)

    def rep_cond(self, which=None, **overrides):
        """Call ``cd.rep_cond`` with the resolved preset's rep_conditions.

        The resolved setup's ``rep_conditions`` (after any
        ``self.rep_conditions`` overrides from ``setup()``) is used as the
        default kwargs. ``overrides`` is partial-merged on top: top-level keys
        replace, the nested ``func`` dict merges one level deep. ``func``
        values may be ``"mean"``, ``"median"`` or ``"perc_N"`` strings;
        ``"perc_N"`` is resolved to ``perc_wrap(N)`` before the call.

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
            Partial-merged onto the resolved ``rep_conditions`` dict. Keys are
            the :class:`captest.setup.RepConditions` fields plus
            ``custom_name``.

        Returns
        -------
        None
            ``cd.rep_cond`` writes to ``cd.rc``.

        Raises
        ------
        ValueError
            If an override key is not a ``CapData.rep_cond`` option.
        """
        if which is None:
            which = self.rc_source if self.rc_source in ("meas", "sim") else "meas"
        cd = self._pick_cd(which)
        self._require_setup()
        allowed = set(RepConditions.model_fields) | {"custom_name"}
        for key in overrides:
            if key not in allowed:
                raise ValueError(
                    f"Unknown rep_cond override {key!r}."
                    f"{_suggest_unknown_key(key, allowed)}"
                )
        resolved_rc = _merge_rep_conditions(
            self.resolved_setup.rep_conditions.model_dump(mode="json"), overrides
        )
        # An empty func mapping means "mean of every rhs variable", which
        # CapData.rep_cond spells func=None.
        resolved_rc["func"] = util._resolve_func_strings(resolved_rc["func"]) or None
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
                "tst.rep_cond(which) or assign tst.rc = df first."
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
                                    f"in tst.{s}_filters_pending — edit the "
                                    "failing step's dict and re-run "
                                    "tst.run_test() (or "
                                    f"tst.{s}.run_pipeline(tst.{s}_filters_pending"
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
                    "set; assign tst.rc = df before run_test()."
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

        print(f"{cap_ratio_rounded:.3f}% - Cap Ratio")
        print(f"{cap_ratio_check_pvalues_rounded:.3f}% - Cap Ratio after pval check")

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

        Builds the scatter plot for each CapData instance via the
        :data:`SCATTER_REGISTRY` function the resolved setup's
        ``scatter_plots`` names, then overlays the two first-panel
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
        scatter_fn = SCATTER_REGISTRY[self.resolved_setup.scatter_plots]
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

        Read-only. Equal to ``1 - rep_irr_filter``; for example, when
        ``rep_irr_filter=0.2`` this is ``0.8``. Updates automatically whenever
        ``rep_irr_filter`` is reassigned. Pass as the ``low`` argument to
        ``CapData.filter_irr`` with a ``ref_val`` to filter within the
        reporting-irradiance band.
        """
        return 1 - self.rep_irr_filter

    @property
    def rep_irr_filter_high(self):
        """Upper irradiance fraction bound derived from ``rep_irr_filter``.

        Read-only. Equal to ``1 + rep_irr_filter``; for example, when
        ``rep_irr_filter=0.2`` this is ``1.2``. Updates automatically whenever
        ``rep_irr_filter`` is reassigned. Pass as the ``high`` argument to
        ``CapData.filter_irr`` with a ``ref_val`` to filter within the
        reporting-irradiance band.
        """
        return 1 + self.rep_irr_filter

    # --- internal helpers ------------------------------------------------

    def _require_setup(self):
        if self.resolved_setup is None:
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


# Silence ruff F401: these are public API; re-imported by `capdata.py`.
__all__ = [
    "CapTest",
    "CapTestResults",
    "SCATTER_REGISTRY",
    "SETUPS_DIR",
    "TEST_SETUPS",
    "highlight_pvals",
    "load_config",
    "load_presets",
    "perc_wrap",
    "resolve_test_setup",
    "scatter_bifi_power_tc",
    "scatter_default",
    "scatter_etotal",
    "test_setups",
]
