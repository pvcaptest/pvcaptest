# Test Setup Documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tuple-tree `regression_cols` grammar with validated pure-data setup documents (pydantic models, a calculation registry, yaml presets, tiered validation) while reproducing every existing preset's numbers.

**Architecture:** A new `captest/setup.py` holds the `Group` / `Column` / `Calc` node models, `Side`, `RepConditions`, `TestSetup`, `derive`, loading/normalisation/digest, and tier-2 `check_project_fit`. `calcparams.py` gains `CALC_REGISTRY` + `@register_calc`. `util.transform_calc_params` dispatches on node types only. Presets ship as `src/captest/setups/*.yaml` and load into `TEST_SETUPS` at import. `CapTest` holds the resolved `TestSetup`, merges `reg_cols_*` overrides key by key, and writes only differing terms to yaml.

**Tech Stack:** Python ≥3.10, pydantic v2 (`>=2.5,<3`), PyYAML, pytest, `uv`, `just`, ruff.

**Spec:** `docs/superpowers/specs/2026-09-22-test-setup-documents-design.md` (HEAD `88b8197`, roborev-clean). Read it first; this plan argues from it.

## Global Constraints

- `pydantic>=2.5,<3` and `pyyaml` are core dependencies in `[project] dependencies` (PyYAML was previously only transitive via bokeh; `captest.py` and `util.py` already `import yaml`).
- Presets live in **`src/captest/setups/`** (not `test_setups/` as the spec says): `captest.test_setups` is an existing exported *function*. Task 2 amends the spec line.
- Import direction, one way: `calcparams.py` ← `setup.py` ← `util.py` / `capdata.py` / `plotting.py` ← `captest.py`. `setup.py` never imports `util`, `capdata`, `plotting` or `captest`. Consequently `parse_regression_formula` and `canonical_json` are *defined* in `setup.py` and re-exported by `util.py` (public names `util.parse_regression_formula` / `util.canonical_json` keep working).
- `DOWNSTREAM_PARAMS` is defined in `calcparams.py` (the registry test needs it) and re-exported by `setup.py`; `CapTest._downstream_attrs` becomes a reference to it.
- The tuple grammar is removed outright: no `_is_aggregation_tuple`, `_is_calculation_tuple`, `encode_reg_cols`, `decode_reg_cols`, `update_by_path`, `validate_test_setup`, `_TEST_SETUP_REQUIRED_KEYS`, `_encode_override`, or callable-in-tree handling survives.
- `rep_conditions.func` values are **strings** (`mean`, `median`, `perc_N`) everywhere until `CapTest.rep_cond` resolves them; `perc_wrap` remains but is no longer accepted inside a setup or an override.
- Every public function/class/method gets a NumPy-style docstring. Line length 88. `just lint` and `just fmt` before each commit. `tst` is the variable name for a `CapTest` in docs and messages.
- Tests are pytest, Arrange-Act-Assert, in the existing files unless a task says otherwise. New test module: `tests/test_setup.py`.
- **Red window:** the full suite is green through Task 4 and again from the end of Task 8. Tasks 5–7 change evaluation and presets in steps that cannot each be green on their own; those tasks run only the modules named in their steps. Do not "fix" unrelated red tests in the window.
- **Review gate — run after every commit, no exceptions:**
  1. `sleep 8; roborev list | sed -n 2p` — note the job id queued for the new SHA by the post-commit hook.
  2. `roborev wait <id>` (up to 10 minutes).
  3. For each finding: verify it against the code. Fix valid findings, commit as `fix: address roborev review <id>` (with the same trailer lines), and repeat from step 1. For an invalid finding, say why in the next commit body.
  4. Proceed to the next task only on "No issues found".
  5. If the wait fails with a rate-limit or `402 Payment Required`, stop and report; do not retry in a loop.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AkqRVnUricrbyNJvB8VDk1
  ```

## File Structure

| File | Responsibility |
|---|---|
| `src/captest/calcparams.py` (modify) | `DOWNSTREAM_PARAMS`, `INJECTED_PARAMS`, `NULLABLE_INJECTED`, `CalcEntry`, `CALC_REGISTRY`, `register_calc`; every public calculation decorated |
| `src/captest/setup.py` (create) | node models, `Side`, `RepConditions`, `TestSetup`, `parse_regression_formula`, `canonical_json`, `derive`, `FitError`, `SetupFitError`, `check_project_fit` |
| `src/captest/setups/*.yaml` (create) | one preset per file |
| `src/captest/util.py` (modify) | `transform_calc_params` over nodes; re-exports; deletions |
| `src/captest/capdata.py` (modify) | `custom_param(output=)`, `process_regression_columns` normalisation, `set_regression_cols`, `agg_sensors` default map |
| `src/captest/plotting.py` (modify) | `DEFAULT_TC_POWER_CALC`, `_missing_column_groups`, `calc_tc_power_column` over nodes |
| `src/captest/captest.py` (modify) | `TEST_SETUPS` loader, `SCATTER_REGISTRY`, `resolve_test_setup`, `CapTest` params/setup/rep_cond/scatter/to_mapping/from_mapping/`check_fit` |
| `tests/setup_fixtures.py` (create) | reusable builders behind the conftest CapData fixtures + `PRESET_FIXTURES` |
| `tests/tools/capture_setup_oracles.py` (create) | writes `tests/data/setup_oracles/<preset>.json` |
| `tests/test_setup.py` (create) | model, derive, load/digest, tier-2 tests |
| `tests/test_setup_oracles.py` (create) | equivalence against the oracles |
| `tests/data/setup_digests.json` (create) | stored content digest per preset |
| docs, skill, CHANGELOG, `CLAUDE.md`, `pyproject.toml`, `tests/smoke_test.py` | Task 10 |

---

### Task 1: Oracle capture on the current code

Run this task **before any other change**. It records what every preset computes today so later tasks can prove equivalence.

**Files:**
- Create: `tests/setup_fixtures.py`
- Modify: `tests/conftest.py:185-420` (fixtures delegate to the builders)
- Create: `tests/tools/__init__.py` (empty), `tests/tools/capture_setup_oracles.py`
- Create: `tests/data/setup_oracles/<preset>.json` (generated, committed)
- Create: `tests/test_setup_oracles.py`

**Interfaces:**
- Produces: `tests.setup_fixtures.build_meas_default() -> CapData`, `build_sim_default() -> CapData`, `add_bom_temp(cd) -> CapData`, `add_spec_corrected(cd) -> CapData`, `add_precwat(cd) -> CapData`, `PRESET_FIXTURES: dict[str, tuple[callable, callable, dict]]` mapping preset → (meas builder, sim builder, `CapTest.from_params` kwargs); `snapshot(tst) -> dict`.

- [x] **Step 1: Create the builders module**

```python
# tests/setup_fixtures.py
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
```

- [x] **Step 2: Point the conftest fixtures at the builders**

In `tests/conftest.py`, replace the bodies of `meas_cd_default`, `sim_cd_default`, `meas_cd_bom_temp`, `meas_cd_spec_corrected`, `sim_cd_spec_corrected` so each is a one-liner over `tests.setup_fixtures` (keep the docstrings):

```python
from tests.setup_fixtures import (
    add_bom_temp,
    add_precwat,
    add_spec_corrected,
    build_meas_default,
    build_sim_default,
)


@pytest.fixture
def meas_cd_default():
    """(docstring unchanged)"""
    return build_meas_default()


@pytest.fixture
def sim_cd_default():
    """(docstring unchanged)"""
    return build_sim_default()


@pytest.fixture
def meas_cd_bom_temp(meas_cd_default):
    """(docstring unchanged)"""
    return add_bom_temp(meas_cd_default)


@pytest.fixture
def meas_cd_spec_corrected(meas_cd_default):
    """(docstring unchanged)"""
    return add_spec_corrected(meas_cd_default)


@pytest.fixture
def sim_cd_spec_corrected(sim_cd_default):
    """(docstring unchanged)"""
    return add_precwat(sim_cd_default)
```

Leave the `ct_*` fixtures as they are.

- [x] **Step 3: Run the existing suite to confirm the refactor is neutral**

Run: `just test-module test_captest.py`
Expected: same pass count as before the change (no failures).

- [x] **Step 4: Write the capture script**

```python
# tests/tools/capture_setup_oracles.py
"""Write ``tests/data/setup_oracles/<preset>.json`` for every preset.

Run once on the code that is being replaced:

    uv run python -m tests.tools.capture_setup_oracles

Re-run only when a preset's numbers are meant to change.
"""

import json
from pathlib import Path

from captest.captest import TEST_SETUPS

from tests.setup_fixtures import PRESET_FIXTURES, build_captest, snapshot

OUT_DIR = Path("tests/data/setup_oracles")


def main():
    missing = set(TEST_SETUPS) - set(PRESET_FIXTURES)
    if missing:
        raise SystemExit(f"no fixture mapping for presets: {sorted(missing)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for preset in sorted(TEST_SETUPS):
        tst = build_captest(preset)
        path = OUT_DIR / f"{preset}.json"
        path.write_text(json.dumps(snapshot(tst), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run it and inspect**

Run: `touch tests/tools/__init__.py && uv run python -m tests.tools.capture_setup_oracles && ls tests/data/setup_oracles`
Expected: ten json files, one per `TEST_SETUPS` key; each has `meas` and `sim` blocks with non-empty `params`.

- [x] **Step 6: Write the equivalence test**

```python
# tests/test_setup_oracles.py
"""Every shipped preset reproduces the numbers captured before the
document migration (``tests/tools/capture_setup_oracles.py``)."""

import json
from pathlib import Path

import numpy as np
import pytest

from captest.captest import TEST_SETUPS

from tests.setup_fixtures import build_captest, snapshot

ORACLES = Path("tests/data/setup_oracles")


@pytest.mark.parametrize("preset", sorted(TEST_SETUPS))
def test_preset_reproduces_oracle(preset):
    expected = json.loads((ORACLES / f"{preset}.json").read_text())
    actual = snapshot(build_captest(preset))
    for side in ("meas", "sim"):
        exp, act = expected[side], actual[side]
        assert set(act["regression_cols"]) == set(exp["regression_cols"])
        for var, e in exp["regression_cols"].items():
            a = act["regression_cols"][var]
            np.testing.assert_allclose(a["sum"], e["sum"], rtol=1e-9)
            np.testing.assert_allclose(a["mean"], e["mean"], rtol=1e-9)
        for key in ("params", "pvalues"):
            assert set(act[key]) == set(exp[key])
            for term, value in exp[key].items():
                np.testing.assert_allclose(act[key][term], value, rtol=1e-9)
```

- [x] **Step 7: Run it**

Run: `uv run pytest tests/test_setup_oracles.py -v`
Expected: 10 passed.

- [x] **Step 8: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add tests/setup_fixtures.py tests/conftest.py tests/tools tests/data/setup_oracles tests/test_setup_oracles.py
git commit -m "test: capture per-preset oracles before the setup document migration"
```
Then run the review gate.

---

### Task 2: Dependencies, spec amendment, calculation registry

**Files:**
- Modify: `pyproject.toml:25-36` (dependencies), `pyproject.toml:78-82` (package data)
- Modify: `docs/superpowers/specs/2026-09-22-test-setup-documents-design.md` (directory name)
- Modify: `src/captest/calcparams.py` (registry + decorators)
- Test: `tests/test_calc_params.py`

**Interfaces:**
- Produces: `calcparams.DOWNSTREAM_PARAMS: tuple[str, ...]`, `calcparams.INJECTED_PARAMS`, `calcparams.NULLABLE_INJECTED`, `calcparams.CalcEntry(func, requires_params, requires_import)`, `calcparams.CALC_REGISTRY: dict[str, CalcEntry]`, `calcparams.register_calc(name=None, *, requires_params=(), requires_import=())`.

- [ ] **Step 1: Declare the dependencies and package data**

In `pyproject.toml` add to `dependencies`:
```toml
    "pydantic>=2.5,<3",
    "pyyaml>=6",
```
and after `[tool.setuptools.packages.find]`:
```toml
[tool.setuptools.package-data]
captest = ["setups/*.yaml"]
```
Run: `uv sync`
Expected: pydantic resolves; `uv run python -c "import pydantic; print(pydantic.VERSION)"` prints a 2.x version.

- [ ] **Step 2: Amend the spec's directory name**

In the spec, replace every `test_setups/` and `src/captest/test_setups/` with `setups/` / `src/captest/setups/`, and add one sentence under "`captest.py` — presets and `CapTest`": "The directory is `setups/`, not `test_setups/`, because `captest.test_setups` is an existing exported function."

- [ ] **Step 3: Write the registry declaration tests**

Append to `tests/test_calc_params.py`:

```python
import inspect
import re

from captest import calcparams
from captest.calcparams import (
    CALC_REGISTRY,
    INJECTED_PARAMS,
    NULLABLE_INJECTED,
    register_calc,
)

#: optional package -> regex that proves the function body uses it
_IMPORT_TOKENS = {"pvlib": re.compile(r"\bpvlib\.|\bLocation\(")}

PUBLIC_CALCS = [
    "power_temp_correct",
    "bom_temp",
    "cell_temp",
    "avg_typ_cell_temp",
    "rpoa_pvsyst",
    "e_total",
    "apparent_zenith",
    "apparent_zenith_pvsyst",
    "absolute_airmass",
    "precipitable_water_gueymard",
    "scale",
    "spectral_factor_firstsolar",
    "multiply",
    "poa_spec_corrected",
]


class TestCalcRegistry:
    def test_every_public_calculation_is_registered(self):
        assert set(PUBLIC_CALCS) <= set(CALC_REGISTRY)

    @pytest.mark.parametrize("name", PUBLIC_CALCS)
    def test_entry_points_at_the_module_function(self, name):
        assert CALC_REGISTRY[name].func is getattr(calcparams, name)

    @pytest.mark.parametrize("name", PUBLIC_CALCS)
    def test_requires_params_is_the_injected_subset_of_the_signature(self, name):
        entry = CALC_REGISTRY[name]
        params = set(inspect.signature(entry.func).parameters)
        expected = (params & set(INJECTED_PARAMS)) - set(NULLABLE_INJECTED)
        assert set(entry.requires_params) == expected

    @pytest.mark.parametrize("name", PUBLIC_CALCS)
    def test_requires_import_matches_the_source(self, name):
        entry = CALC_REGISTRY[name]
        source = inspect.getsource(entry.func)
        for package, token in _IMPORT_TOKENS.items():
            assert (package in entry.requires_import) == bool(token.search(source))

    def test_register_calc_defaults_name_to_function_name(self, monkeypatch):
        monkeypatch.delitem(CALC_REGISTRY, "tmp_calc", raising=False)

        @register_calc()
        def tmp_calc(data, x=None):
            return data[x]

        assert CALC_REGISTRY["tmp_calc"].func is tmp_calc
        monkeypatch.delitem(CALC_REGISTRY, "tmp_calc")

    def test_register_calc_rejects_a_different_function_under_a_taken_name(self):
        def power_temp_correct(data):
            return data

        with pytest.raises(ValueError, match="already registered"):
            register_calc()(power_temp_correct)

    def test_register_calc_is_idempotent_for_the_same_function(self):
        entry = CALC_REGISTRY["e_total"]
        register_calc(requires_params=entry.requires_params)(entry.func)
        assert CALC_REGISTRY["e_total"] is not None
```

- [ ] **Step 4: Run to verify they fail**

Run: `uv run pytest tests/test_calc_params.py::TestCalcRegistry -v`
Expected: ImportError on `CALC_REGISTRY`.

- [ ] **Step 5: Implement the registry**

At the top of `src/captest/calcparams.py` (after the existing imports; add `import inspect` only if used, and `from dataclasses import dataclass`):

```python
#: ``CapTest`` parameters propagated onto ``CapData`` and injected by name
#: into calculations that declare them. A setup may constrain any of these.
DOWNSTREAM_PARAMS = (
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

#: Every ``CapData`` attribute ``custom_param`` injects by name.
INJECTED_PARAMS = DOWNSTREAM_PARAMS + ("site",)

#: Injected parameters for which ``None`` is a meaningful value, so they are
#: never listed in ``requires_params``.
NULLABLE_INJECTED = ("altitude_override",)


@dataclass(frozen=True)
class CalcEntry:
    """A registered calculation.

    Parameters
    ----------
    func : callable
        The calculation; takes ``data`` first and keyword arguments after.
    requires_params : tuple of str
        Parameters that must resolve to a non-``None`` value at setup time,
        from the document, a ``CapData`` attribute, or the function default.
    requires_import : tuple of str
        Optional packages the function imports.
    """

    func: object
    requires_params: tuple = ()
    requires_import: tuple = ()


#: Registry of calculations a setup document may name under ``calc``.
CALC_REGISTRY = {}


def register_calc(name=None, *, requires_params=(), requires_import=()):
    """Register a calculation under ``name`` (default: the function name).

    Parameters
    ----------
    name : str or None
        Registry key. Defaults to ``func.__name__``.
    requires_params : iterable of str
        See :class:`CalcEntry`.
    requires_import : iterable of str
        See :class:`CalcEntry`.

    Returns
    -------
    callable
        Decorator that records the function and returns it unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered to a different function.
    """

    def decorator(func):
        key = name or func.__name__
        existing = CALC_REGISTRY.get(key)
        if existing is not None and existing.func is not func:
            raise ValueError(f"calculation {key!r} is already registered")
        CALC_REGISTRY[key] = CalcEntry(
            func, tuple(requires_params), tuple(requires_import)
        )
        return func

    return decorator
```

Then decorate each public calculation. Apply the rule mechanically: `requires_params` = the function's parameter names that appear in `INJECTED_PARAMS`, minus `NULLABLE_INJECTED`; `requires_import=("pvlib",)` exactly when the body uses `pvlib.` or `Location(`. From the current signatures that gives:

```python
@register_calc(requires_params=("power_temp_coeff", "base_temp"))
def power_temp_correct(...): ...

@register_calc(requires_params=("module_type", "racking"))
def bom_temp(...): ...          # confirm against its signature

@register_calc(requires_params=("module_type", "racking"))
def cell_temp(...): ...

@register_calc()
def avg_typ_cell_temp(...): ...

@register_calc()
def rpoa_pvsyst(...): ...

@register_calc(requires_params=("bifaciality", "bifacial_frac", "rear_shade"))
def e_total(...): ...

@register_calc(requires_params=("site",), requires_import=("pvlib",))
def apparent_zenith(...): ...   # altitude_override is nullable, so excluded

@register_calc(requires_params=("site",), requires_import=("pvlib",))
def apparent_zenith_pvsyst(...): ...  # confirm parameters

@register_calc(requires_params=("airmass_model",), requires_import=("pvlib",))
def absolute_airmass(...): ...

@register_calc()
def precipitable_water_gueymard(...): ...

@register_calc()
def scale(...): ...

@register_calc(requires_params=("spectral_module_type",), requires_import=("pvlib",))
def spectral_factor_firstsolar(...): ...

@register_calc()
def multiply(...): ...

@register_calc()
def poa_spec_corrected(...): ...
```

Where a comment says "confirm", read the signature and body; the tests in Step 3 are the arbiter — adjust the declaration, never the test.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_calc_params.py -v`
Expected: all pass (existing calculation tests untouched).

- [ ] **Step 7: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add pyproject.toml uv.lock docs/superpowers/specs/2026-09-22-test-setup-documents-design.md src/captest/calcparams.py tests/test_calc_params.py
git commit -m "feat: calculation registry with declared injected params and imports"
```
Then run the review gate.

---
### Task 3: `setup.py` — node models, `Side`, `RepConditions`, `TestSetup` (tier 1)

**Files:**
- Create: `src/captest/setup.py`
- Modify: `src/captest/util.py` (move `parse_regression_formula`; re-export it and `canonical_json`)
- Create: `tests/test_setup.py`

**Interfaces:**
- Produces: `setup.Group(group, agg="mean")`, `setup.Column(column)`, `setup.Calc(calc, args={})`, `setup.Node`, `setup.NODE_TAGS`, `setup.Side(reg_cols)`, `setup.RepConditions(func={}, w_vel=None, irr_bal=False, percent_filter=20, front_poa="poa", rc_kwargs=None)`, `setup.TestSetup(name, description="", derived_from=None, reg_fml, meas, sim, params={}, rep_conditions=RepConditions(), scatter_plots="default")`, `setup.agg_column_name(group, agg) -> str`, `setup.calc_output_name(node: Calc) -> str`, `setup.parse_regression_formula(formula) -> (lhs, rhs)`, `setup.canonical_json(obj) -> str`, `setup.DOWNSTREAM_PARAMS` (re-export), `TestSetup.load(source)`, `TestSetup.to_dict()`, `TestSetup.to_yaml(path)`, `TestSetup.to_json(path)`, `TestSetup.content_digest()`, `TestSetup.json_schema()`.
- Consumes: `calcparams.CALC_REGISTRY`, `calcparams.DOWNSTREAM_PARAMS`.

- [ ] **Step 1: Move `parse_regression_formula` out of `util.py`**

Cut the `parse_regression_formula` function (util.py ≈ line 742, through the end of its body) and paste it unchanged into the new `src/captest/setup.py` (Step 3 shows where). It uses `ModelDesc`: move `from patsy import ModelDesc` (util.py line 11) with it, unless `grep -n ModelDesc src/captest/util.py` shows another user, in which case add the import to `setup.py` and leave util's. In `util.py`, at the import block, add:

```python
from captest.setup import canonical_json, parse_regression_formula  # noqa: F401
```

(`util` imports `setup`; `setup` must never import `util`.) Run `uv run pytest tests/test_util.py::TestParseRegressionFormula -q` after Step 3 to confirm the re-export.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_setup.py
"""Tests for ``captest.setup``: node models, documents, derive, tiers."""

import json

import pytest
import yaml
from pydantic import ValidationError

from captest import setup
from captest.setup import Calc, Column, Group, RepConditions, Side, TestSetup

E2848_FML = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"


def e2848_doc(**changes):
    """A complete e2848_default-shaped document as plain data."""
    doc = {
        "name": "e2848_default",
        "reg_fml": E2848_FML,
        "meas": {
            "reg_cols": {
                "power": {"group": "real_pwr_mtr", "agg": "sum"},
                "poa": {"group": "irr_poa"},
                "t_amb": {"group": "temp_amb"},
                "w_vel": {"group": "wind_speed"},
            }
        },
        "sim": {
            "reg_cols": {
                "power": {"column": "E_Grid"},
                "poa": {"column": "GlobInc"},
                "t_amb": {"column": "T_Amb"},
                "w_vel": {"column": "WindVel"},
            }
        },
        "rep_conditions": {"func": {"poa": "perc_60", "t_amb": "mean", "w_vel": "mean"}},
    }
    doc.update(changes)
    return doc


def _paths(exc):
    return [".".join(str(p) for p in err["loc"]) for err in exc.value.errors()]


@pytest.fixture
def probe_calc(monkeypatch):
    """Register a calculation whose signature accepts the literal kinds under test."""
    from captest.calcparams import CALC_REGISTRY, CalcEntry

    def probe(data, col=None, factor=1.0, flag=None, xs=None):
        return data[col] * factor

    monkeypatch.setitem(CALC_REGISTRY, "probe", CalcEntry(probe, (), ()))
    return probe


class TestNodes:
    def test_group_defaults_agg_to_mean(self):
        assert Group(group="irr_poa").agg == "mean"

    def test_group_is_hashable_and_equal_by_value(self):
        assert hash(Group(group="a")) == hash(Group(group="a", agg="mean"))
        assert Group(group="a") == Group(group="a", agg="mean")

    def test_group_rejects_unknown_agg(self):
        with pytest.raises(ValidationError):
            Group(group="a", agg="product")

    def test_node_kind_is_decided_by_the_tag(self):
        side = Side.model_validate(
            {"reg_cols": {"x": {"group": "g"}, "y": {"column": "c"}}}
        )
        assert isinstance(side.reg_cols["x"], Group)
        assert isinstance(side.reg_cols["y"], Column)

    def test_mapping_without_a_tag_is_an_error_not_a_literal(self):
        with pytest.raises(ValidationError) as exc:
            Calc(calc="e_total", args={"poa": {"grp": "irr_poa"}})
        assert any("args.poa" in p for p in _paths(exc))

    def test_mapping_with_two_tags_is_an_error(self):
        with pytest.raises(ValidationError):
            Calc(calc="e_total", args={"poa": {"group": "a", "column": "b"}})

    def test_literals_pass_through_inside_args(self, probe_calc):
        node = Calc(
            calc="probe",
            args={"col": {"column": "PrecWat"}, "factor": 100, "flag": None, "xs": [1, 2]},
        )
        assert node.args["factor"] == 100
        assert node.args["flag"] is None
        assert node.args["xs"] == [1, 2]

    def test_non_finite_float_literal_is_rejected(self, probe_calc):
        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": float("nan")})

    def test_dict_literal_is_rejected(self, probe_calc):
        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": {"a": 1}})

    def test_callable_argument_is_rejected(self, probe_calc):
        import numpy as np

        with pytest.raises(ValidationError):
            Calc(calc="probe", args={"col": {"column": "x"}, "factor": np.mean})

    def test_unknown_calc_names_the_path_and_suggests(self):
        with pytest.raises(ValidationError) as exc:
            Side.model_validate({"reg_cols": {"poa": {"calc": "e_totl", "args": {}}}})
        assert any(p.endswith("reg_cols.poa") for p in _paths(exc))
        assert "e_total" in str(exc.value)

    def test_unknown_arg_key_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown argument"):
            Calc(calc="e_total", args={"poa": {"group": "a"}, "rpoa": {"group": "b"}, "x": 1})

    def test_missing_required_arg_is_rejected(self):
        with pytest.raises(ValidationError, match="missing"):
            Calc(calc="e_total", args={"poa": {"group": "a"}})

    def test_injected_param_may_be_given_explicitly(self):
        node = Calc(
            calc="power_temp_correct",
            args={"power": {"group": "p", "agg": "sum"}, "cell_temp": {"column": "T"},
                  "base_temp": 20},
        )
        assert node.args["base_temp"] == 20

    def test_extra_keys_are_forbidden(self):
        with pytest.raises(ValidationError):
            Group(group="a", aggr="mean")


class TestOutputNames:
    def test_group_writes_agg_column(self):
        assert setup.agg_column_name("irr_poa", "mean") == "irr_poa_mean_agg"

    def test_calc_writes_registry_name(self):
        assert setup.calc_output_name(Calc(calc="e_total", args={
            "poa": {"group": "a"}, "rpoa": {"group": "b"}})) == "e_total"

    def test_same_calc_different_args_on_one_side_is_rejected(self):
        with pytest.raises(ValidationError, match="e_total"):
            Side.model_validate({"reg_cols": {
                "poa": {"calc": "e_total", "args": {"poa": {"group": "a"}, "rpoa": {"group": "b"}}},
                "poa2": {"calc": "e_total", "args": {"poa": {"group": "a"}, "rpoa": {"group": "c"}}},
            }})

    def test_identical_calc_nodes_are_permitted(self):
        node = {"calc": "e_total", "args": {"poa": {"group": "a"}, "rpoa": {"group": "b"}}}
        Side.model_validate({"reg_cols": {"poa": node, "poa2": node}})

    def test_top_level_literal_is_rejected(self):
        with pytest.raises(ValidationError, match="literal"):
            Side.model_validate({"reg_cols": {"poa": "irr_poa"}})


class TestTestSetupDocument:
    def test_loads_a_complete_document(self):
        tsd = TestSetup.model_validate(e2848_doc())
        assert isinstance(tsd.meas.reg_cols["power"], Group)
        assert tsd.rep_conditions.percent_filter == 20

    def test_formula_variable_missing_from_a_side_is_rejected(self):
        doc = e2848_doc()
        del doc["sim"]["reg_cols"]["w_vel"]
        with pytest.raises(ValidationError) as exc:
            TestSetup.model_validate(doc)
        assert any(p.startswith("sim") for p in _paths(exc))
        assert "w_vel" in str(exc.value)

    def test_rep_conditions_func_key_not_in_rhs_is_rejected(self):
        doc = e2848_doc(rep_conditions={"func": {"ghi": "mean"}})
        with pytest.raises(ValidationError, match="ghi"):
            TestSetup.model_validate(doc)

    def test_rep_conditions_func_value_must_be_mean_median_or_perc(self):
        doc = e2848_doc(rep_conditions={"func": {"poa": "p60"}})
        with pytest.raises(ValidationError, match="perc_N"):
            TestSetup.model_validate(doc)

    def test_percent_filter_is_numeric_only(self):
        doc = e2848_doc(rep_conditions={"percent_filter": [10, 20]})
        with pytest.raises(ValidationError):
            TestSetup.model_validate(doc)

    def test_params_keys_must_be_downstream_params(self):
        with pytest.raises(ValidationError, match="params"):
            TestSetup.model_validate(e2848_doc(params={"ac_nameplate": 1}))

    def test_params_accepts_downstream_param(self):
        assert TestSetup.model_validate(e2848_doc(params={"rear_shade": 0})).params == {
            "rear_shade": 0
        }

    def test_setup_is_equal_by_value_and_not_hashable(self):
        a, b = TestSetup.model_validate(e2848_doc()), TestSetup.model_validate(e2848_doc())
        assert a == b
        with pytest.raises(TypeError):
            hash(a)


class TestNormalisationAndIdentity:
    def test_to_dict_materialises_defaults_and_keeps_nulls(self):
        d = TestSetup.model_validate(e2848_doc()).to_dict()
        assert d["meas"]["reg_cols"]["poa"] == {"group": "irr_poa", "agg": "mean"}
        assert d["rep_conditions"]["w_vel"] is None
        assert d["derived_from"] is None

    def test_to_dict_is_plain_json_types(self):
        d = TestSetup.model_validate(e2848_doc()).to_dict()
        json.dumps(d, allow_nan=False)

    def test_normalisation_is_idempotent(self):
        first = TestSetup.model_validate(e2848_doc()).to_dict()
        second = TestSetup.model_validate(first).to_dict()
        assert first == second

    def test_explicit_null_and_omitted_field_digest_equal(self):
        with_null = e2848_doc(rep_conditions={"func": {"poa": "perc_60", "t_amb": "mean",
                                                        "w_vel": "mean"}, "w_vel": None})
        assert (TestSetup.model_validate(with_null).content_digest()
                == TestSetup.model_validate(e2848_doc()).content_digest())

    def test_content_digest_changes_with_content(self):
        base = TestSetup.model_validate(e2848_doc()).content_digest()
        doc = e2848_doc()
        doc["meas"]["reg_cols"]["power"] = {"group": "real_pwr_inv", "agg": "sum"}
        assert TestSetup.model_validate(doc).content_digest() != base

    def test_canonical_json_rejects_nan_and_non_string_keys(self):
        with pytest.raises(ValueError):
            setup.canonical_json({"x": float("nan")})
        with pytest.raises(TypeError):
            setup.canonical_json({1: "x"})

    def test_load_from_yaml_path_json_path_text_and_mapping(self, tmp_path):
        doc = e2848_doc()
        y = tmp_path / "s.yaml"
        y.write_text(yaml.safe_dump(doc))
        j = tmp_path / "s.json"
        j.write_text(json.dumps(doc))
        from_yaml = TestSetup.load(y)
        assert TestSetup.load(j) == from_yaml
        assert TestSetup.load(str(j)) == from_yaml
        assert TestSetup.loads(y.read_text()) == from_yaml
        assert TestSetup.loads(json.dumps(doc)) == from_yaml   # long text, never a path
        assert TestSetup.load(doc) == from_yaml

    def test_to_yaml_and_to_json_round_trip(self, tmp_path):
        tsd = TestSetup.model_validate(e2848_doc())
        tsd.to_yaml(tmp_path / "out.yaml")
        tsd.to_json(tmp_path / "out.json")
        assert TestSetup.load(tmp_path / "out.yaml") == tsd
        assert TestSetup.load(tmp_path / "out.json") == tsd

    def test_json_schema_marks_extra_keys_forbidden(self):
        schema = TestSetup.json_schema()
        assert schema["additionalProperties"] is False
        assert "Group" in schema["$defs"]


class TestParseRegressionFormula:
    def test_reexported_from_util(self):
        from captest import util

        assert util.parse_regression_formula is setup.parse_regression_formula
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_setup.py -q`
Expected: ImportError (`captest.setup` does not exist).

- [ ] **Step 4: Write `setup.py`**

```python
# src/captest/setup.py
"""Capacity-test setup documents.

A setup is pure data: yaml or json made of strings, numbers, booleans,
lists and tagged mappings. This module holds the typed view of that
document (``TestSetup``), the node grammar (``Group`` / ``Column`` /
``Calc`` / literal), normalisation and identity (``to_dict`` /
``content_digest``), and the validation that needs no data (tier 1 here,
tier 2 in :func:`check_project_fit`). It never imports ``capdata`` or
``captest``; a ``CapData`` reaches it only as a runtime argument.
"""

import difflib
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from patsy import ModelDesc
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from captest.calcparams import CALC_REGISTRY, DOWNSTREAM_PARAMS  # noqa: F401

# --- formula ---------------------------------------------------------------

# (paste parse_regression_formula here, unchanged, from util.py)


# --- canonical json --------------------------------------------------------


def _validate_canonical(obj, path):
    if obj is None or isinstance(obj, (bool, int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite float cannot be canonical JSON")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: canonical JSON requires string keys")
            _validate_canonical(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            _validate_canonical(value, f"{path}[{index}]")
        return
    raise TypeError(f"{type(obj).__name__} at {path} is not JSON-serializable")


def canonical_json(obj):
    """Serialize ``obj`` to the one canonical JSON form used for identity.

    Sorted keys, no insignificant whitespace, non-ASCII kept as-is, NaN and
    Infinity rejected — byte-identical to the form the pft-mono ``ctsweep``
    store hashes.

    Parameters
    ----------
    obj : dict, list or scalar
        Plain JSON-representable data.

    Returns
    -------
    str

    Raises
    ------
    TypeError
        On a value or mapping key that cannot be represented.
    ValueError
        On a non-finite float.
    """
    _validate_canonical(obj, "$")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


# --- nodes -----------------------------------------------------------------

NODE_TAGS = ("group", "column", "calc")
AGG_FUNCS = ("mean", "sum", "median", "min", "max")
_REP_FUNC_RE = re.compile(r"^(mean|median|perc_\d+)$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Group(_Frozen):
    """Aggregate a column group; writes ``<group>_<agg>_agg``."""

    group: str
    agg: Literal["mean", "sum", "median", "min", "max"] = "mean"


class Column(_Frozen):
    """A raw column of ``data`` by name."""

    column: str


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Scalar = Union[None, bool, int, FiniteFloat, str]
Literal_ = Union[Scalar, list[Scalar]]


def _node_kind(node):
    """Name the union member ``node`` belongs to, for the discriminator.

    A mapping is identified by which one of ``NODE_TAGS`` it carries; a
    mapping with none or several is unclassifiable and returns ``None`` so
    pydantic reports ``union_tag_not_found`` at that path. An already-built
    model reports its own class; anything else is a literal.
    """
    if isinstance(node, dict):
        present_tags = [tag for tag in NODE_TAGS if tag in node]
        return present_tags[0] if len(present_tags) == 1 else None
    if isinstance(node, BaseModel):
        return type(node).__name__.lower()
    return "literal"


class Calc(_Frozen):
    """A registered calculation; writes a column named for the registry key."""

    calc: str
    args: dict[str, "Node"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_against_registry(self):
        entry = CALC_REGISTRY.get(self.calc)
        if entry is None:
            hint = difflib.get_close_matches(self.calc, CALC_REGISTRY, n=1)
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(f"unknown calculation {self.calc!r}.{suffix}")
        params = inspect.signature(entry.func).parameters
        allowed = set(params) - {"data", "verbose"}
        unknown = sorted(set(self.args) - allowed)
        if unknown:
            raise ValueError(
                f"unknown argument(s) {unknown} for {self.calc}; "
                f"expected {sorted(allowed)}"
            )
        required = {
            name
            for name, p in params.items()
            if p.default is p.empty
            and name not in ("data", "verbose")
            and name not in entry.requires_params
        }
        missing = sorted(required - set(self.args))
        if missing:
            raise ValueError(f"missing required argument(s) {missing} for {self.calc}")
        return self


Node = Annotated[
    Union[
        Annotated[Group, Tag("group")],
        Annotated[Column, Tag("column")],
        Annotated[Calc, Tag("calc")],
        Annotated[Literal_, Tag("literal")],
    ],
    Discriminator(_node_kind),
]
Calc.model_rebuild()


def agg_column_name(group, agg):
    """Column a ``Group`` node writes (the ``agg_group`` naming)."""
    return f"{group}_{agg}_agg"


def calc_output_name(node):
    """Column a ``Calc`` node writes: its registry name."""
    return node.calc


def _producers(node, path, out):
    """Collect (output column, path, node) for every producing node."""
    if isinstance(node, dict):
        for key, value in node.items():
            _producers(value, f"{path}.{key}", out)
    elif isinstance(node, Group):
        out.append((agg_column_name(node.group, node.agg), path, node))
    elif isinstance(node, Calc):
        out.append((calc_output_name(node), path, node))
        _producers(node.args, f"{path}.args", out)


def walk_nodes(node, path=""):
    """Yield ``(path, node)`` for every model node in a tree, depth first."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk_nodes(value, f"{path}.{key}" if path else key)
    elif isinstance(node, (Group, Column)):
        yield path, node
    elif isinstance(node, Calc):
        yield path, node
        yield from walk_nodes(node.args, f"{path}.args")


class Side(_Frozen):
    """One side's regression columns: formula variable -> node."""

    reg_cols: dict[str, Node]

    @field_validator("reg_cols", mode="after")
    @classmethod
    def _top_level_nodes_and_unique_outputs(cls, reg_cols):
        for var, node in reg_cols.items():
            if not isinstance(node, (Group, Column, Calc)):
                raise ValueError(
                    f"{var}: a top-level regression column must be a group, "
                    f"column or calc node, not a literal ({node!r})"
                )
        producers = []
        _producers(reg_cols, "reg_cols", producers)
        seen = {}
        for output, path, node in producers:
            if output in seen and seen[output][1] != node:
                raise ValueError(
                    f"two different nodes write column {output!r}: "
                    f"{seen[output][0]} and {path}"
                )
            seen.setdefault(output, (path, node))
        return reg_cols


class RepConditions(_Frozen):
    """Keyword arguments to ``CapData.rep_cond`` in document form."""

    func: dict[str, str] = Field(default_factory=dict)
    w_vel: FiniteFloat | None = None
    irr_bal: bool = False
    percent_filter: FiniteFloat = 20
    front_poa: str = "poa"
    rc_kwargs: dict[str, Scalar] | None = None

    @field_validator("func", mode="after")
    @classmethod
    def _func_values(cls, func):
        bad = {k: v for k, v in func.items() if not _REP_FUNC_RE.match(v)}
        if bad:
            raise ValueError(f"func values must be mean, median or perc_N; got {bad}")
        return func


class TestSetup(_Frozen):
    """A complete capacity-test setup document.

    Parameters
    ----------
    name : str
        Preset or variant name.
    description : str
        Prose description shown by ``captest.test_setups()``.
    derived_from : str or None
        Provenance only (a preset name or content digest); never used to
        fill omitted fields.
    reg_fml : str
        patsy regression formula.
    meas, sim : Side
        Regression columns for the measured and modeled data.
    params : dict
        Test-level parameters this setup requires, checked at tier 2 against
        the effective value each calculation would receive.
    rep_conditions : RepConditions
        Reporting-condition options.
    scatter_plots : str
        Name in ``captest.SCATTER_REGISTRY``; membership is checked by
        ``captest.py``.
    """

    name: str
    description: str = ""
    derived_from: str | None = None
    reg_fml: str
    meas: Side
    sim: Side
    params: dict[str, Scalar] = Field(default_factory=dict)
    rep_conditions: RepConditions = Field(default_factory=RepConditions)
    scatter_plots: str = Field(default="default", min_length=1)

    @field_validator("reg_fml", mode="after")
    @classmethod
    def _formula_parses(cls, reg_fml):
        parse_regression_formula(reg_fml)
        return reg_fml

    @field_validator("meas", "sim", mode="after")
    @classmethod
    def _formula_variables_present(cls, side, info):
        reg_fml = info.data.get("reg_fml")
        if reg_fml is None:
            return side
        lhs, rhs = parse_regression_formula(reg_fml)
        missing = sorted((set(lhs) | set(rhs)) - set(side.reg_cols))
        if missing:
            raise ValueError(f"reg_cols is missing formula variable(s) {missing}")
        return side

    @field_validator("params", mode="after")
    @classmethod
    def _params_are_downstream(cls, params):
        bad = sorted(set(params) - set(DOWNSTREAM_PARAMS))
        if bad:
            raise ValueError(f"params keys {bad} are not CapTest downstream params")
        return params

    @field_validator("rep_conditions", mode="after")
    @classmethod
    def _func_keys_are_rhs(cls, rc, info):
        reg_fml = info.data.get("reg_fml")
        if reg_fml is None:
            return rc
        _, rhs = parse_regression_formula(reg_fml)
        bad = sorted(set(rc.func) - set(rhs))
        if bad:
            raise ValueError(f"func keys {bad} are not rhs variables of reg_fml")
        return rc

    # --- serialization -------------------------------------------------

    def to_dict(self):
        """The normalised document: every field, defaults materialised."""
        return self.model_dump(mode="json")

    def content_digest(self):
        """sha256 of ``canonical_json(self.to_dict())``; the setup's identity."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, source):
        """Build a setup from a yaml/json file path or a mapping.

        Always validates through the model; yaml is read with ``safe_load``
        (json is valid yaml). Use :meth:`loads` for document text.
        """
        if isinstance(source, dict):
            return cls.model_validate(source)
        return cls.loads(Path(source).read_text())

    @classmethod
    def loads(cls, text):
        """Build a setup from yaml or json document text."""
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self, path):
        """Write ``to_dict()`` as yaml."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_json(self, path):
        """Write ``to_dict()`` as json."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def json_schema(cls):
        """JSON Schema of the document shape (an authoring contract)."""
        return cls.model_json_schema()
```

Note on `_paths`: pydantic reports a field-validator error at the field's loc (`meas` / `sim` / `rep_conditions`); the variable or key names are in the message. The tests above assert on both.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_setup.py tests/test_util.py::TestParseRegressionFormula -v`
Expected: all pass. If `test_json_schema_marks_extra_keys_forbidden` fails on `$defs` naming, print `schema["$defs"].keys()` and adjust the assertion to the generated name (it must reference the `Group` model).

- [ ] **Step 6: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/setup.py src/captest/util.py tests/test_setup.py
git commit -m "feat: setup document model with tagged nodes and tier-1 validation"
```
Then run the review gate.

---
### Task 4: `derive` — key-level merge, `null` removal, `func` pruning

**Files:**
- Modify: `src/captest/setup.py`
- Test: `tests/test_setup.py`

**Interfaces:**
- Produces: `setup.derive(base: TestSetup, *, name=None, description=None, reg_fml=None, reg_cols_meas=None, reg_cols_sim=None, params=None, rep_conditions=None, scatter_plots=None) -> TestSetup`; `setup.merge_reg_cols(base_side: Side, override: dict | None, formula_vars: set[str]) -> dict` (the normalised merged mapping); `setup.DerivationError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_setup.py`:

```python
class TestDerive:
    def _base(self):
        return TestSetup.model_validate(e2848_doc())

    def test_replaces_one_term_and_keeps_the_others(self):
        out = setup.derive(
            self._base(), reg_cols_meas={"power": {"group": "real_pwr_inv", "agg": "sum"}}
        )
        assert out.meas.reg_cols["power"] == Group(group="real_pwr_inv", agg="sum")
        assert out.meas.reg_cols["poa"] == Group(group="irr_poa")
        assert out.sim == self._base().sim

    def test_accepts_model_nodes_as_well_as_mappings(self):
        out = setup.derive(self._base(), reg_cols_meas={"poa": Group(group="irr_ghi")})
        assert out.meas.reg_cols["poa"].group == "irr_ghi"

    def test_records_provenance_and_keeps_name(self):
        out = setup.derive(self._base(), reg_fml="power ~ poa")
        assert out.derived_from == "e2848_default"
        assert out.name == "e2848_default"

    def test_null_removes_a_term_and_prunes_rep_conditions_func(self):
        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) - 1"
        out = setup.derive(
            self._base(),
            reg_fml=fml,
            reg_cols_meas={"w_vel": None},
            reg_cols_sim={"w_vel": None},
        )
        assert "w_vel" not in out.meas.reg_cols
        assert "w_vel" not in out.sim.reg_cols
        assert set(out.rep_conditions.func) == {"poa", "t_amb"}
        assert out.rep_conditions.func["poa"] == "perc_60"

    def test_pruning_only_removes(self):
        out = setup.derive(self._base(), rep_conditions={"func": {"poa": "perc_55"}})
        assert out.rep_conditions.func == {"poa": "perc_55"}

    def test_null_for_a_term_the_base_lacks_is_rejected(self):
        with pytest.raises(setup.DerivationError, match="ghi"):
            setup.derive(self._base(), reg_cols_meas={"ghi": None})

    def test_override_key_that_is_not_a_formula_variable_is_rejected(self):
        with pytest.raises(setup.DerivationError, match="ghi"):
            setup.derive(self._base(), reg_cols_meas={"ghi": {"group": "irr_ghi"}})

    def test_removing_a_term_the_formula_still_uses_is_rejected(self):
        with pytest.raises(ValidationError, match="w_vel"):
            setup.derive(self._base(), reg_cols_meas={"w_vel": None})

    def test_other_fields_replace_wholesale(self):
        out = setup.derive(self._base(), params={"rear_shade": 0}, scatter_plots="etotal")
        assert out.params == {"rear_shade": 0}
        assert out.scatter_plots == "etotal"

    def test_result_is_complete_and_digests_like_a_fresh_document(self):
        out = setup.derive(
            self._base(), reg_cols_meas={"power": {"group": "real_pwr_inv", "agg": "sum"}}
        )
        doc = out.to_dict()
        doc.pop("derived_from")
        fresh = e2848_doc()
        fresh["meas"]["reg_cols"]["power"] = {"group": "real_pwr_inv", "agg": "sum"}
        expected = TestSetup.model_validate(fresh).to_dict()
        expected.pop("derived_from")
        assert doc == expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_setup.py::TestDerive -q`
Expected: AttributeError: module has no attribute `derive`.

- [ ] **Step 3: Implement**

Append to `src/captest/setup.py`:

```python
# --- derive ----------------------------------------------------------------


class DerivationError(ValueError):
    """A ``reg_cols`` override names a term the derivation cannot apply."""


def _node_to_data(node):
    return node.model_dump(mode="json") if isinstance(node, BaseModel) else node


def merge_reg_cols(base_side, override, formula_vars):
    """Merge a ``reg_cols`` override onto one side, key by key.

    Parameters
    ----------
    base_side : Side
    override : dict or None
        Formula variable -> node (mapping or model), or ``None`` to remove.
    formula_vars : set of str
        Variables of the resulting formula; every non-``None`` override key
        must be one of them.

    Returns
    -------
    dict
        Plain-data ``reg_cols`` mapping ready for ``Side.model_validate``.

    Raises
    ------
    DerivationError
        For ``None`` on a variable the base lacks, or a key that is not a
        formula variable.
    """
    merged = {var: _node_to_data(node) for var, node in base_side.reg_cols.items()}
    for var, node in (override or {}).items():
        if node is None:
            if var not in merged:
                raise DerivationError(
                    f"cannot remove {var!r}: the base setup has no such term"
                )
            del merged[var]
            continue
        if var not in formula_vars:
            raise DerivationError(
                f"override key {var!r} is not a variable of the regression formula "
                f"{sorted(formula_vars)}"
            )
        merged[var] = _node_to_data(node)
    return merged


def derive(
    base,
    *,
    name=None,
    description=None,
    reg_fml=None,
    reg_cols_meas=None,
    reg_cols_sim=None,
    params=None,
    rep_conditions=None,
    scatter_plots=None,
):
    """Return a new complete setup derived from ``base``.

    ``reg_cols_meas`` / ``reg_cols_sim`` merge key by key (a present key
    replaces that variable's whole node, ``None`` removes the variable);
    every other argument replaces its field wholesale. After the formula and
    sides are resolved, ``rep_conditions.func`` entries for variables no
    longer on the right-hand side are pruned. ``derived_from`` is set to
    ``base.name``; ``name`` defaults to ``base.name``.

    Parameters
    ----------
    base : TestSetup
    name, description, reg_fml, scatter_plots : str or None
    reg_cols_meas, reg_cols_sim : dict or None
    params : dict or None
    rep_conditions : dict or RepConditions or None
        Replaces the base's reporting conditions wholesale; callers that
        want a partial merge do it before calling.

    Returns
    -------
    TestSetup

    Raises
    ------
    DerivationError
        See :func:`merge_reg_cols`.
    pydantic.ValidationError
        If the result is not a valid setup.
    """
    data = base.to_dict()
    data["derived_from"] = base.name
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if reg_fml is not None:
        data["reg_fml"] = reg_fml
    if params is not None:
        data["params"] = dict(params)
    if scatter_plots is not None:
        data["scatter_plots"] = scatter_plots
    if rep_conditions is not None:
        data["rep_conditions"] = _node_to_data(rep_conditions)

    lhs, rhs = parse_regression_formula(data["reg_fml"])
    formula_vars = set(lhs) | set(rhs)
    data["meas"] = {"reg_cols": merge_reg_cols(base.meas, reg_cols_meas, formula_vars)}
    data["sim"] = {"reg_cols": merge_reg_cols(base.sim, reg_cols_sim, formula_vars)}

    func = dict(data["rep_conditions"].get("func") or {})
    data["rep_conditions"]["func"] = {k: v for k, v in func.items() if k in rhs}
    return TestSetup.model_validate(data)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_setup.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/setup.py tests/test_setup.py
git commit -m "feat: TestSetup.derive with key-level reg_cols merge and func pruning"
```
Then run the review gate.

---

### Task 5: Evaluation over nodes (`util`) and `CapData` integration

**Red window starts here.** After this task `tests/test_util.py`, `tests/test_CapData.py` and `tests/test_setup.py` are green; `tests/test_captest.py`, `tests/test_plotting.py` and `tests/test_setup_oracles.py` are red until Tasks 7–9.

**Files:**
- Modify: `src/captest/util.py:305-560` (`_is_*_tuple`, `_resolve_column_group`, `_get_or_create_aggregation`, `transform_calc_params`, `process_reg_cols`), `util.py:690-741` (delete `encode_reg_cols` / `decode_reg_cols`), `util.py:263-302` (delete `update_by_path`)
- Modify: `src/captest/capdata.py:900-925` (`set_regression_cols`), `:1673-1690` (`agg_sensors` default map), `:3286-3350` (`process_regression_columns`, `custom_param`)
- Test: `tests/test_util.py`, `tests/test_CapData.py`

**Interfaces:**
- Produces: `util.transform_calc_params(node, cd, agg_cache=None, verbose=True)` over `Group`/`Column`/`Calc`/literal; `util._get_or_create_aggregation(node: Group, cd, agg_cache, verbose) -> str`; `CapData.custom_param(func, *, output=None, verbose=True, **kwargs)`; `CapData.process_regression_columns` accepting nodes or plain mappings; `CapData.regression_cols_preprocess: Side`; `CapData.set_regression_cols` building nodes.
- Consumes: `setup.Group/Column/Calc/Side`, `calcparams.CALC_REGISTRY`.

- [ ] **Step 1: Rewrite the util tests**

Replace the `nested_calc_dict` fixture and the `TestUpdateByPath`, `TestProcessRegCols`, `TestGetOrCreateAggregationReuse` and `TestRegColsEncodeDecode` classes in `tests/test_util.py` with:

```python
@pytest.fixture
def dummy_cd(monkeypatch):
    """A CapData stand-in plus four registered dummy calculations."""
    from captest.calcparams import CALC_REGISTRY, CalcEntry

    class DummyCapData:
        def __init__(self):
            self.data = pd.DataFrame()
            self.column_groups = {
                "real_pwr_mtr": ["metered_power_kw"],
                "irr_poa": ["pyran1", "pyran2"],
                "temp_amb": ["temp_amb1", "temp_amb2"],
                "wind_speed": ["wind_speed1", "wind_speed2"],
                "irr_rpoa": ["irr_rpoa1", "irr_rpoa2"],
            }
            self.calls = {}

        def agg_group(self, group_id, agg_func, **kwargs):
            self.agg_group_kwargs = kwargs
            col_name = f"{group_id}_{agg_func}_agg"
            self.data[col_name] = np.full(10, 5)
            return col_name

        def custom_param(self, func, *, output=None, verbose=True, **kwargs):
            self.calls[func.__name__] = kwargs
            self.data[output or func.__name__] = func(self.data, **kwargs)

    # Explicit signatures: the Calc validator checks args against them.
    def test_func1(data, power=None, cell_temp=None, factor=None, enabled=None,
                   offset=None, cols=None, nothing=None):
        return np.full(10, 1)

    def test_func2(data, poa=None, bom=None):
        return np.full(10, 2)

    def test_func3(data, poa=None, temp_amb=None, wind_speed=None):
        return np.full(10, 3)

    def test_func4(data, poa=None, rpoa=None):
        return np.full(10, 4)

    for f in (test_func1, test_func2, test_func3, test_func4):
        monkeypatch.setitem(CALC_REGISTRY, f.__name__, CalcEntry(f, (), ()))
    return DummyCapData()


NESTED_TREE = {
    "power_tc": {
        "calc": "test_func1",
        "args": {
            "power": {"column": "metered_power_kw"},
            "cell_temp": {
                "calc": "test_func2",
                "args": {
                    "poa": {"group": "irr_poa"},
                    "bom": {
                        "calc": "test_func3",
                        "args": {
                            "poa": {"group": "irr_poa"},
                            "temp_amb": {"group": "temp_amb"},
                            "wind_speed": {"group": "wind_speed"},
                        },
                    },
                },
            },
        },
    },
    "irr_total": {
        "calc": "test_func4",
        "args": {"poa": {"group": "irr_poa"}, "rpoa": {"group": "irr_rpoa"}},
    },
}


def _side(tree):
    from captest.setup import Side

    return dict(Side.model_validate({"reg_cols": tree}).reg_cols)


class TestProcessRegCols:
    def test_literals_reach_the_calculation_unchanged(self, dummy_cd):
        dummy_cd.data["metered_power_kw"] = np.full(10, 1.0)
        reg_cols = _side(
            {
                "scaled": {
                    "calc": "test_func1",
                    "args": {
                        "power": {"column": "metered_power_kw"},
                        "factor": 100,
                        "enabled": True,
                        "offset": 1.5,
                        "cols": ["a", "b"],
                        "nothing": None,
                    },
                }
            }
        )
        util.process_reg_cols(reg_cols, cd=dummy_cd)
        kwargs = dummy_cd.calls["test_func1"]
        assert kwargs["factor"] == 100 and kwargs["enabled"] is True
        assert kwargs["offset"] == 1.5 and kwargs["cols"] == ["a", "b"]
        assert kwargs["nothing"] is None
        assert reg_cols["scaled"] == "test_func1"

    def test_nested_tree_evaluates_bottom_up_and_aggregates_once(self, dummy_cd):
        dummy_cd.data["metered_power_kw"] = np.full(10, 1.0)
        reg_cols = _side(NESTED_TREE)
        util.process_reg_cols(reg_cols, cd=dummy_cd)
        assert reg_cols == {"power_tc": "test_func1", "irr_total": "test_func4"}
        assert list(dummy_cd.data.columns) == [
            "metered_power_kw",
            "irr_poa_mean_agg",
            "temp_amb_mean_agg",
            "wind_speed_mean_agg",
            "test_func3",
            "test_func2",
            "test_func1",
            "irr_rpoa_mean_agg",
            "test_func4",
        ]
        assert dummy_cd.calls["test_func3"] == {
            "poa": "irr_poa_mean_agg",
            "temp_amb": "temp_amb_mean_agg",
            "wind_speed": "wind_speed_mean_agg",
        }

    def test_missing_column_raises_key_error(self, dummy_cd):
        reg_cols = _side({"power": {"column": "absent"}})
        with pytest.raises(KeyError, match="absent"):
            util.process_reg_cols(reg_cols, cd=dummy_cd)


class TestGetOrCreateAggregationReuse:
    def test_reuses_existing_column_and_prints_message(self, dummy_cd, capsys):
        dummy_cd.data["irr_poa_mean_agg"] = np.full(10, 7)
        reg_cols = _side({"poa": {"group": "irr_poa"}})
        util.process_reg_cols(reg_cols, cd=dummy_cd)
        assert "Reusing existing column 'irr_poa_mean_agg'" in capsys.readouterr().out
        assert reg_cols["poa"] == "irr_poa_mean_agg"
        assert not hasattr(dummy_cd, "agg_group_kwargs")

    def test_reuse_is_silent_when_verbose_false(self, dummy_cd, capsys):
        dummy_cd.data["irr_poa_mean_agg"] = np.full(10, 7)
        reg_cols = _side({"poa": {"group": "irr_poa"}})
        util.process_reg_cols(reg_cols, cd=dummy_cd, verbose=False)
        assert capsys.readouterr().out == ""
```

Delete every remaining reference in `tests/test_util.py` to `update_by_path`, `encode_reg_cols`, `decode_reg_cols`, `nested_calc_dict`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_util.py -q`
Expected: failures in `TestProcessRegCols` / `TestGetOrCreateAggregationReuse` (tuple dispatch does not recognise nodes; `custom_param` signature).

- [ ] **Step 3: Rewrite the evaluation in `util.py`**

Delete `_is_aggregation_tuple`, `_is_calculation_tuple`, `_resolve_column_group`, `update_by_path`, `encode_reg_cols`, `decode_reg_cols`. Add `from captest.calcparams import CALC_REGISTRY` and `from captest.setup import Calc, Column, Group` to the imports. Replace `_get_or_create_aggregation` and `transform_calc_params`:

```python
def _get_or_create_aggregation(node, cd, agg_cache, verbose):
    """Return the column a ``Group`` node resolves to, aggregating if needed.

    An existing ``<group>_<agg>_agg`` column (e.g. measured data loaded from
    an exported test-data file) is reused rather than re-aggregated, with a
    message when ``verbose``.

    Parameters
    ----------
    node : captest.setup.Group
    cd : CapData
    agg_cache : dict
        ``Group`` node -> aggregated column name, shared across one walk.
    verbose : bool

    Returns
    -------
    str
    """
    if node in agg_cache:
        return agg_cache[node]
    expected = get_agg_column_name(node.group, node.agg)
    if expected in cd.data.columns:
        if verbose:
            print(
                f"Reusing existing column '{expected}'; skipping "
                f"aggregation of the {node.group} group.\n"
            )
        agg_name = expected
    else:
        agg_name = cd.agg_group(group_id=node.group, agg_func=node.agg, verbose=verbose)
    agg_cache[node] = agg_name
    return agg_name


def transform_calc_params(node, cd, agg_cache=None, verbose=True):
    """Evaluate a regression-columns tree bottom-up, returning column names.

    Node types (see :mod:`captest.setup`):

    - dict: transform each value (``Side.reg_cols`` or ``Calc.args``);
    - ``Group``: aggregate the column group, return the new column name;
    - ``Column``: return the column name after checking it exists;
    - ``Calc``: transform ``args``, run the registered function through
      ``cd.custom_param`` and return the column it wrote (its registry name);
    - anything else is a literal argument, passed through unchanged.

    Parameters
    ----------
    node : dict, Group, Column, Calc or literal
    cd : CapData
    agg_cache : dict or None
    verbose : bool

    Returns
    -------
    object
        The transformed node.

    Raises
    ------
    KeyError
        If a ``Column`` names a column absent from ``cd.data``.
    """
    if agg_cache is None:
        agg_cache = {}
    if isinstance(node, dict):
        return {
            key: transform_calc_params(value, cd, agg_cache, verbose)
            for key, value in node.items()
        }
    if isinstance(node, Group):
        return _get_or_create_aggregation(node, cd, agg_cache, verbose)
    if isinstance(node, Column):
        if node.column not in cd.data.columns:
            raise KeyError(f"column {node.column!r} not in data")
        return node.column
    if isinstance(node, Calc):
        entry = CALC_REGISTRY[node.calc]
        resolved = transform_calc_params(node.args, cd, agg_cache, verbose)
        cd.custom_param(entry.func, output=node.calc, verbose=verbose, **resolved)
        return node.calc
    return node
```

Update the `process_reg_cols` docstring: replace the tuple example with the document example from the spec's "Setup" section and drop the sentence about `CapData` methods as tuple heads. Update the `get_agg_column_name` docstring's `agg_func` type to `str`.

- [ ] **Step 4: Run the util tests**

Run: `uv run pytest tests/test_util.py -q`
Expected: all pass.

- [ ] **Step 5: Write the CapData tests**

Append to `tests/test_CapData.py` (find the class that tests `process_regression_columns`, near line 640, and add after it):

```python
class TestRegressionColumnsDocumentForm:
    def test_process_accepts_plain_mappings(self, meas):
        meas.regression_cols = {
            "power": {"column": "meter_power"},
            "poa": {"group": "irr_poa_pyran"},
        }
        meas.process_regression_columns(verbose=False)
        assert meas.regression_cols == {
            "power": "meter_power",
            "poa": "irr_poa_pyran_mean_agg",
        }

    def test_preprocess_keeps_the_normalised_side(self, meas):
        from captest.setup import Group, Side

        meas.regression_cols = {"poa": {"group": "irr_poa_pyran"}}
        meas.process_regression_columns(verbose=False)
        assert isinstance(meas.regression_cols_preprocess, Side)
        assert meas.regression_cols_preprocess.reg_cols["poa"] == Group(
            group="irr_poa_pyran"
        )

    def test_set_regression_cols_builds_group_or_column_nodes(self, meas):
        from captest.setup import Column, Group

        meas.set_regression_cols(
            power="meter_power", poa="irr_poa_pyran", t_amb="temp_amb", w_vel="wind"
        )
        assert meas.regression_cols["power"] == Column(column="meter_power")
        assert meas.regression_cols["poa"] == Group(group="irr_poa_pyran")

    def test_custom_param_output_and_absent_only_injection(self, meas):
        from captest.calcparams import scale

        meas.power_temp_coeff = -0.4
        seen = {}

        def probe(data, col=None, power_temp_coeff=None, pressure=None):
            seen.update(col=col, power_temp_coeff=power_temp_coeff, pressure=pressure)
            return data[col]

        meas.custom_param(probe, output="probed", col="meter_power", pressure=None)
        assert "probed" in meas.data.columns
        assert seen["power_temp_coeff"] == -0.4      # absent -> injected
        assert seen["pressure"] is None              # explicit None passes through
        meas.power_temp_coeff = None
        meas.custom_param(probe, output="probed2", col="meter_power")
        assert seen["power_temp_coeff"] is None      # None attribute -> function default
        meas.custom_param(scale, col="meter_power", factor=2.0, verbose=False)
        assert "scale" in meas.data.columns          # output defaults to __name__

    def test_agg_sensors_default_map_reads_group_nodes(self, meas):
        meas.regression_cols = {
            "power": {"column": "meter_power"},
            "poa": {"group": "irr_poa_pyran"},
            "t_amb": {"group": "temp_amb"},
            "w_vel": {"group": "wind"},
        }
        meas.agg_sensors(verbose=False)
        assert meas.regression_cols["poa"] == "irr_poa_pyran_mean_agg"
```

Then convert every existing `regression_cols = {...}` literal in `tests/test_CapData.py` (≈15 sites, lines 650–3600): `("g", "mean")` → `{"group": "g"}`, `("g", "sum")` → `{"group": "g", "agg": "sum"}`, a bare column-name string → `{"column": "name"}`. Post-processing flat assignments such as `{"poa": "irr_poa_pyran_mean_agg", "t_amb": "temp_amb"}` at line 793 stay as they are when no `process_regression_columns()` follows them.

- [ ] **Step 6: Run to verify the new tests fail**

Run: `uv run pytest tests/test_CapData.py::TestRegressionColumnsDocumentForm -q`
Expected: failures (`custom_param` has no `output`; mappings not normalised).

- [ ] **Step 7: Implement the CapData changes**

In `capdata.py`, add `from captest.setup import Column, Group, Side` to the imports. Replace `custom_param`:

```python
    def custom_param(self, func, *, output=None, verbose=True, **kwargs):
        """Run ``func`` on ``data`` and store the result as a new column.

        Called by ``util.transform_calc_params`` for every ``Calc`` node.
        Parameters of ``func`` that are **absent** from ``kwargs`` and match
        an attribute of this ``CapData`` (``power_temp_coeff``, ``site``,
        ...) are injected from that attribute; a value passed explicitly —
        including ``None`` — reaches ``func`` unchanged.

        Parameters
        ----------
        func : callable
            Takes ``data`` first and keyword arguments after; returns a Series.
        output : str or None
            Column to write. Defaults to ``func.__name__``.
        verbose : bool, default True
            Forwarded to ``func`` when it accepts ``verbose``.
        **kwargs
            Keyword arguments for ``func``.

        Raises
        ------
        ValueError
            If an absent parameter's name is also a column group id (the
            call is ambiguous; name it in ``regression_cols``).
        """
        signature = inspect.signature(func)
        for key in signature.parameters:
            if key in ("data", "verbose") or key in kwargs:
                continue
            if key in self.column_groups.data:
                raise ValueError(
                    f"The kwarg {key} of the function {func.__name__} is also a "
                    f"column groups id. Change the name of the column group id or "
                    f"include the kwarg in the CapData.regression_cols"
                )
            # Same rule as setup.effective_value: an attribute that is None
            # supplies nothing, so the function's own default applies.
            value = getattr(self, key, None)
            if value is not None:
                kwargs[key] = value
        if "verbose" in signature.parameters:
            kwargs["verbose"] = verbose
        self.data[output or func.__name__] = func(self.data, **kwargs)
```

Replace the body of `process_regression_columns` after the filters warning:

```python
        side = Side.model_validate({"reg_cols": dict(self.regression_cols)})
        self.regression_cols_preprocess = side
        self.regression_cols = dict(side.reg_cols)
        util.process_reg_cols(self.regression_cols, cd=self, verbose=verbose)
        self.filters = []
        self.create_column_group_attributes()
        if "agg" in self.column_groups:
            self.create_agg_attributes()
```

Replace the body of `set_regression_cols`:

```python
        def node(value):
            if value in self.column_groups:
                return Group(group=value)
            return Column(column=value)

        self.regression_cols = {
            "power": node(power),
            "poa": node(poa),
            "t_amb": node(t_amb),
            "w_vel": node(w_vel),
        }
```
and update its docstring: "Each value is a column group id (becomes a mean aggregation node) or a column name (becomes a column node)."

In `agg_sensors`, replace the default `agg_map` construction with:

```python
        # ``regression_cols`` may hold document mappings, node models, or (after
        # processing) flat column names. Normalise once so the rest of this
        # method sees nodes or strings only.
        if any(isinstance(v, dict) for v in self.regression_cols.values()):
            self.regression_cols = dict(
                Side.model_validate({"reg_cols": self.regression_cols}).reg_cols
            )

        def node_group_id(node):
            """Group id a node aggregates, or None for a raw-column reference.

            Named to stay clear of the existing ``for group_id, agg_func in
            agg_map.items()`` loop variable below, which would shadow it.
            """
            if isinstance(node, Group):
                return node.group
            if isinstance(node, Column):
                return None
            if isinstance(node, str):
                return node
            raise ValueError(
                "agg_sensors needs group or column nodes (or group ids) in "
                f"regression_cols; got {node!r}. Use process_regression_columns "
                "for calc nodes."
            )

        if agg_map is None:
            defaults = {"power": "sum", "poa": "mean", "t_amb": "mean", "w_vel": "mean"}
            agg_map = {}
            for var, func in defaults.items():
                gid = node_group_id(self.regression_cols[var])
                if gid is not None:          # Column nodes are already resolved
                    agg_map[gid] = func
```

In the existing aggregation loop, the branch that skips a group because its output column already exists must record that column before continuing, so the resolver below can use it:

```python
            if col_name in self.data.columns:
                if verbose:
                    print(f"Skipping aggregation of {group_id} as column "
                          f"{col_name} already exists")
                agg_names[group_id] = col_name
                continue
```

and, where the existing body updates `regression_cols` to point at the aggregated columns (the block that today rewrites string values), replace it with a flattening pass that runs after the aggregation loop. `agg_names` maps each group id that was actually aggregated to the column written; a group the loop skipped (single-column groups are not aggregated today; an explicit `agg_map` may omit a group) has no entry:

```python
        def resolve(var, node):
            if isinstance(node, Column):
                return node.column
            gid = node_group_id(node)
            if gid in agg_names:                      # aggregated or reused in this call
                return agg_names[gid]
            columns = self.column_groups.get(gid, [])
            if len(columns) == 1:                     # single-column group: the column
                return columns[0]
            return node                               # untouched; still a node / group id

        self.regression_cols = {
            var: resolve(var, node) for var, node in self.regression_cols.items()
        }
```

Add two tests beside `test_agg_sensors_default_map_reads_group_nodes`:

```python
    def test_agg_sensors_single_column_group_resolves_to_its_column(self, meas):
        meas.regression_cols = {
            "power": {"group": "real_pwr_mtr"},   # one column in this fixture
            "poa": {"group": "irr_poa_pyran"},
            "t_amb": {"group": "temp_amb"},
            "w_vel": {"group": "wind"},
        }
        meas.agg_sensors(verbose=False)
        assert meas.regression_cols["power"] == meas.column_groups["real_pwr_mtr"][0]

    def test_agg_sensors_explicit_map_leaves_unselected_groups_as_nodes(self, meas):
        from captest.setup import Group

        meas.regression_cols = {
            "power": {"column": "meter_power"},
            "poa": {"group": "irr_poa_pyran"},
            "t_amb": {"group": "temp_amb"},
            "w_vel": {"group": "wind"},
        }
        meas.agg_sensors(agg_map={"irr_poa_pyran": "mean"}, verbose=False)
        assert meas.regression_cols["poa"] == "irr_poa_pyran_mean_agg"
        assert meas.regression_cols["power"] == "meter_power"
        assert meas.regression_cols["t_amb"] == Group(group="temp_amb")

    def test_agg_sensors_reuses_an_existing_aggregate_column(self, meas):
        meas.data["irr_poa_pyran_mean_agg"] = 1.0
        meas.regression_cols = {
            "power": {"column": "meter_power"},
            "poa": {"group": "irr_poa_pyran"},
            "t_amb": {"group": "temp_amb"},
            "w_vel": {"group": "wind"},
        }
        meas.agg_sensors(verbose=False)
        assert meas.regression_cols["poa"] == "irr_poa_pyran_mean_agg"
        assert (meas.data["irr_poa_pyran_mean_agg"] == 1.0).all()
```

(If `real_pwr_mtr` is not a single-column group in the `meas` fixture, pick the fixture's single-column group for the first test; `grep -n "column_groups" tests/conftest.py` shows them.)

Run `grep -n "regression_cols_preprocess" src tests` and update any reader that expected a dict to read `.reg_cols` (there should be none outside `capdata.py`).

- [ ] **Step 8: Run the CapData and util tests**

Run: `uv run pytest tests/test_CapData.py tests/test_util.py tests/test_setup.py -q`
Expected: all pass. (`tests/test_captest.py` is expected red now.)

- [ ] **Step 9: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/util.py src/captest/capdata.py tests/test_util.py tests/test_CapData.py
git commit -m "feat!: evaluate regression columns from node models; drop the tuple grammar"
```
Then run the review gate. Tell the reviewer in the commit body that `test_captest.py` is red until the presets migrate (Task 7).

---
### Task 6: Tier 2 — `check_project_fit`

**Files:**
- Modify: `src/captest/setup.py`
- Test: `tests/test_setup.py`

**Interfaces:**
- Produces: `setup.FitError(path: str, message: str)` (frozen dataclass), `setup.SetupFitError(ValueError)` with `.errors: list[FitError]`, `setup.check_project_fit(setup: TestSetup, side: str, cd) -> list[FitError]`, `setup.effective_value(name, node: Calc, cd) -> object` (resolves one parameter under the precedence rules; returns `inspect.Parameter.empty` when nothing supplies it).
- Consumes: `calcparams.CALC_REGISTRY`, a duck-typed `cd` with `column_groups`, `data.columns` and attributes.

Note on the spec's "existing raw column" shadow rule: a `Calc` output legitimately exists in `data` on a second `setup()` of the same instance, so the raw-column check compares against the columns *listed in `column_groups`* (the sensor columns), not all of `data.columns`. Record this as a one-line amendment in the spec's tier-2 bullet.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_setup.py`:

```python
class _Cd:
    """Duck-typed CapData for tier-2 tests."""

    def __init__(self, groups, columns=(), **attrs):
        import pandas as pd

        self.column_groups = groups
        all_cols = [c for cols in groups.values() for c in cols] + list(columns)
        self.data = pd.DataFrame(columns=all_cols)
        for key, value in attrs.items():
            setattr(self, key, value)


def _fit(doc, side="meas", **cd_kwargs):
    return setup.check_project_fit(TestSetup.model_validate(doc), side, _Cd(**cd_kwargs))


MEAS_GROUPS = {
    "real_pwr_mtr": ["meter_power"],
    "irr_poa": ["poa1", "poa2"],
    "temp_amb": ["ta1"],
    "wind_speed": ["ws1"],
}


class TestCheckProjectFit:
    def test_clean_project_has_no_errors(self):
        assert _fit(e2848_doc(), groups=MEAS_GROUPS) == []

    def test_missing_group_is_reported_with_path(self):
        groups = {k: v for k, v in MEAS_GROUPS.items() if k != "wind_speed"}
        errors = _fit(e2848_doc(), groups=groups)
        assert [e.path for e in errors] == ["meas.reg_cols.w_vel.group"]
        assert "wind_speed" in errors[0].message

    def test_missing_column_on_sim_side(self):
        errors = _fit(e2848_doc(), side="sim", groups={}, columns=["E_Grid", "GlobInc"])
        assert {e.path for e in errors} == {
            "sim.reg_cols.t_amb.column",
            "sim.reg_cols.w_vel.column",
        }

    def _tc_doc(self, **args):
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {
                "calc": "power_temp_correct",
                "args": {
                    "power": {"group": "real_pwr_mtr", "agg": "sum"},
                    "cell_temp": {"column": "bom"},
                    **args,
                },
            },
            "poa": {"group": "irr_poa"},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        return doc

    def test_requires_param_none_by_every_route_is_reported(self):
        # 1. CapData attribute None (power_temp_correct's own default is None)
        errors = _fit(self._tc_doc(), groups=MEAS_GROUPS, columns=["bom"],
                      power_temp_coeff=None, base_temp=25)
        assert [e.path for e in errors] == ["meas.reg_cols.power"]
        assert "power_temp_coeff" in errors[0].message
        # 2. explicit null in args
        errors = _fit(self._tc_doc(power_temp_coeff=None), groups=MEAS_GROUPS,
                      columns=["bom"], power_temp_coeff=-0.3, base_temp=25)
        assert "power_temp_coeff" in errors[0].message
        # 3. no attribute at all
        errors = _fit(self._tc_doc(), groups=MEAS_GROUPS, columns=["bom"])
        assert any("power_temp_coeff" in e.message for e in errors)

    def test_explicit_arg_satisfies_requires_param(self):
        errors = _fit(self._tc_doc(power_temp_coeff=-0.3), groups=MEAS_GROUPS,
                      columns=["bom"], base_temp=25)
        assert errors == []

    def test_function_default_satisfies_requires_param(self):
        # base_temp has a real default (25); no attribute needed.
        errors = _fit(self._tc_doc(power_temp_coeff=-0.3), groups=MEAS_GROUPS,
                      columns=["bom"])
        assert errors == []

    def test_requires_import_missing_is_reported(self, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda n: None if n == "pvlib" else real(n))
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {"calc": "absolute_airmass",
                    "args": {"apparent_zenith": {"column": "z"}, "pressure": None}},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        errors = _fit(doc, groups=MEAS_GROUPS, columns=["z"], airmass_model="kastenyoung1989")
        assert any("pvlib" in e.message for e in errors)

    def test_output_shadowing_a_group_id_or_sensor_column(self):
        groups = {**MEAS_GROUPS, "e_total": ["e_total"]}
        doc = e2848_doc(reg_fml="power ~ poa")
        doc["meas"]["reg_cols"] = {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "poa": {"calc": "e_total", "args": {"poa": {"group": "irr_poa"},
                                                "rpoa": {"column": "poa2"}}},
        }
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        errors = _fit(doc, groups=groups, bifaciality=0.7, bifacial_frac=1, rear_shade=0)
        assert [e.path for e in errors] == ["meas.reg_cols.poa"]
        assert "e_total" in errors[0].message

    def test_params_constraint_checks_effective_value_per_side(self):
        doc = e2848_doc(reg_fml="power ~ poa", params={"rear_shade": 0})
        etotal = {"calc": "e_total", "args": {"poa": {"group": "irr_poa"},
                                              "rpoa": {"column": "poa2"}}}
        doc["meas"]["reg_cols"] = {"power": {"group": "real_pwr_mtr", "agg": "sum"},
                                   "poa": etotal}
        doc["rep_conditions"] = {"func": {"poa": "perc_60"}}
        # meas with rear_shade 0.2 -> violation
        errors = _fit(doc, groups=MEAS_GROUPS, bifaciality=0.7, bifacial_frac=1,
                      rear_shade=0.2)
        assert any(e.path == "params.rear_shade" for e in errors)
        # meas with rear_shade 0 -> fine
        assert _fit(doc, groups=MEAS_GROUPS, bifaciality=0.7, bifacial_frac=1,
                    rear_shade=0) == []
        # sim side has no rear_shade attribute; e_total's default 0 satisfies it
        doc["sim"]["reg_cols"] = {"power": {"column": "E_Grid"},
                                  "poa": {"calc": "e_total",
                                          "args": {"poa": {"column": "GlobInc"},
                                                   "rpoa": {"column": "GlobBak"}}}}
        assert _fit(doc, side="sim", groups={}, columns=["E_Grid", "GlobInc", "GlobBak"],
                    bifaciality=0.7, bifacial_frac=1) == []

    def test_side_without_the_param_is_not_checked(self):
        doc = e2848_doc(params={"rear_shade": 0})
        assert _fit(doc, groups=MEAS_GROUPS, rear_shade=0.5) == []

    def test_setup_fit_error_lists_every_path(self):
        errors = [setup.FitError("a", "x"), setup.FitError("b", "y")]
        exc = setup.SetupFitError(errors)
        assert exc.errors == errors
        assert "a: x" in str(exc) and "b: y" in str(exc)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_setup.py::TestCheckProjectFit -q`
Expected: AttributeError (`check_project_fit`).

- [ ] **Step 3: Implement**

Append to `src/captest/setup.py`:

```python
# --- tier 2: project fit ---------------------------------------------------

import importlib.util  # noqa: E402  (keep with the other imports when formatting)
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class FitError:
    """One tier-2 finding: a document path and what is wrong there."""

    path: str
    message: str


class SetupFitError(ValueError):
    """A setup does not fit the project; ``errors`` lists every finding."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__(
            "setup does not fit this project:\n"
            + "\n".join(f"  {e.path}: {e.message}" for e in self.errors)
        )


def effective_value(name, node, cd):
    """Value a calculation would receive for ``name`` under the precedence rules.

    1. the node's ``args`` entry, if present (an explicit ``None`` counts);
    2. else the ``cd`` attribute of that name when present and not ``None``;
    3. else the function's own default;
    4. else ``inspect.Parameter.empty``.
    """
    if name in node.args:
        return node.args[name]
    attr = getattr(cd, name, None)
    if attr is not None:
        return attr
    param = inspect.signature(CALC_REGISTRY[node.calc].func).parameters.get(name)
    if param is None:
        return inspect.Parameter.empty
    return param.default


def check_project_fit(setup, side, cd):
    """Tier-2 validation of one side of ``setup`` against a project.

    Needs only ``cd.column_groups``, ``cd.data.columns`` and the attributes
    named by each calculation's ``requires_params``; no data is read.

    Parameters
    ----------
    setup : TestSetup
    side : {"meas", "sim"}
    cd : CapData-like

    Returns
    -------
    list of FitError
        Empty when the side fits. Every finding is collected so a document
        can be fixed in one pass.
    """
    errors = []
    reg_cols = getattr(setup, side).reg_cols
    groups = dict(cd.column_groups)
    sensor_columns = {c for cols in groups.values() for c in cols}
    data_columns = set(cd.data.columns)

    for path, node in walk_nodes(reg_cols, f"{side}.reg_cols"):
        if isinstance(node, Group):
            if node.group not in groups:
                errors.append(
                    FitError(f"{path}.group", f"{node.group!r} is not a column group")
                )
            continue
        if isinstance(node, Column):
            if node.column not in data_columns:
                errors.append(
                    FitError(f"{path}.column", f"{node.column!r} is not a column")
                )
            continue
        entry = CALC_REGISTRY[node.calc]
        for package in entry.requires_import:
            if importlib.util.find_spec(package) is None:
                errors.append(
                    FitError(path, f"{node.calc} requires the {package!r} package")
                )
        for name in entry.requires_params:
            value = effective_value(name, node, cd)
            if value is None or value is inspect.Parameter.empty:
                errors.append(
                    FitError(
                        path,
                        f"{node.calc} requires {name}, which resolves to None; set "
                        f"it on the test or pass it in args",
                    )
                )
        output = calc_output_name(node)
        if output in groups or output in sensor_columns:
            errors.append(
                FitError(path, f"output column {output!r} shadows a column group "
                               f"id or sensor column")
            )
        for key, required in setup.params.items():
            if key in entry.requires_params:
                value = effective_value(key, node, cd)
                if value != required:
                    errors.append(
                        FitError(
                            f"params.{key}",
                            f"setup requires {key}={required!r} but {node.calc} at "
                            f"{path} would receive {value!r}",
                        )
                    )
    return errors
```

Move the two `import` lines to the module's import block when formatting (they are shown inline only to keep this step self-contained). Amend the spec's tier-2 shadow bullet: "…is a column group id or a column listed in `column_groups` on that side (not any `data` column: a calculation's own output from an earlier `setup()` is expected to be present)."

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_setup.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/setup.py tests/test_setup.py docs/superpowers/specs/2026-09-22-test-setup-documents-design.md
git commit -m "feat: tier-2 project-fit validation for setup documents"
```
Then run the review gate.

---
### Task 7: Presets as yaml, `TEST_SETUPS` loader, `SCATTER_REGISTRY`, `resolve_test_setup`

**Files:**
- Create: `src/captest/setups/<name>.yaml` × 10 (generated by a throwaway converter, then reviewed)
- Create: `tests/data/setup_digests.json`
- Modify: `src/captest/captest.py:178-230` (scatter functions → registry), `:230-895` (delete the `TEST_SETUPS` dicts, `_TEST_SETUP_REQUIRED_KEYS`, `validate_test_setup`), `:897-930` (`test_setups()`), `:1000-1062` (`resolve_test_setup`, delete `_encode_override`)
- Test: `tests/test_captest.py` (`TestTestSetupsRegistry`, `TestResolveTestSetup`), `tests/test_setup_oracles.py` (goes green again), new `tests/test_presets.py`

**Interfaces:**
- Produces: `captest.SCATTER_REGISTRY: dict[str, callable]`, `captest.SETUPS_DIR: Path`, `captest.load_presets() -> dict[str, TestSetup]`, `captest.TEST_SETUPS: dict[str, TestSetup]`, `captest.resolve_test_setup(name, overrides=None) -> TestSetup` (overrides keys: `reg_cols_meas`, `reg_cols_sim`, `reg_fml`, `rep_conditions` (partial-merged), `params`, `scatter_plots`), `captest._check_scatter_name(name)`.
- Consumes: `setup.TestSetup`, `setup.derive`, `setup.DerivationError`.

- [ ] **Step 1: Generate the yaml files with a throwaway converter**

The old tuple dicts are still in `captest.py` at this point; convert them mechanically rather than by hand. Save this in the scratchpad (not the repo) and run it once:

```python
# scratch: convert_presets.py
from pathlib import Path

import yaml

from captest import captest as ctm
from captest.calcparams import CALC_REGISTRY
from captest.util import _perc_wrap_to_string

OUT = Path("src/captest/setups")
SCATTER = {ctm.scatter_default: "default", ctm.scatter_etotal: "etotal",
           ctm.scatter_bifi_power_tc: "bifi_power_tc"}


def name_of(func):
    return next(k for k, e in CALC_REGISTRY.items() if e.func is func)


def convert(node, side):
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str):
        return {"group": node[0], "agg": node[1]}
    if isinstance(node, tuple) and len(node) == 2 and callable(node[0]):
        return {"calc": name_of(node[0]),
                "args": {k: convert(v, side) for k, v in node[1].items()}}
    if isinstance(node, str):
        return {"group": node} if side == "meas" else {"column": node}
    return node


def uses(tree, func_name):
    if isinstance(tree, dict):
        return any(uses(v, func_name) for v in tree.values())
    if isinstance(tree, tuple) and callable(tree[0]):
        return name_of(tree[0]) == func_name or uses(tree[1], func_name)
    return False


OUT.mkdir(exist_ok=True)
for name, entry in ctm.TEST_SETUPS.items():
    rc = dict(entry["rep_conditions"])
    rc["func"] = {k: _perc_wrap_to_string(v) for k, v in rc.get("func", {}).items()}
    doc = {
        "name": name,
        "description": " ".join(entry["description"].split()),
        "reg_fml": entry["reg_fml"],
        "meas": {"reg_cols": {k: convert(v, "meas") for k, v in entry["reg_cols_meas"].items()}},
        "sim": {"reg_cols": {k: convert(v, "sim") for k, v in entry["reg_cols_sim"].items()}},
        "rep_conditions": rc,
        "scatter_plots": SCATTER[entry["scatter_plots"]],
    }
    if uses(entry["reg_cols_sim"], "rpoa_pvsyst") and uses(entry["reg_cols_meas"], "e_total"):
        doc["params"] = {"rear_shade": 0}
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, width=88))
    print("wrote", name)
```

Run: `uv run python /path/to/scratch/convert_presets.py && ls src/captest/setups`
Expected: ten files. Open each and check: every meas leaf that was a bare string became `{group: …}` — if any of those is really a raw column name in the fixture data, change it to `{column: …}` (the oracle test in Step 8 will also catch it). Reflow descriptions if the dumper folded them badly; content, not layout, is what matters.

- [ ] **Step 2: Write the failing registry tests**

Replace `TestTestSetupsRegistry` in `tests/test_captest.py`:

```python
class TestTestSetupsRegistry:
    @pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
    def test_each_preset_is_a_test_setup(self, preset):
        from captest.setup import TestSetup

        assert isinstance(ct.TEST_SETUPS[preset], TestSetup)
        assert ct.TEST_SETUPS[preset].name == preset

    @pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
    def test_each_preset_lhs_is_power(self, preset):
        assert ct.TEST_SETUPS[preset].reg_fml.split("~")[0].strip() == "power"

    @pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
    def test_each_preset_scatter_name_is_registered(self, preset):
        assert ct.TEST_SETUPS[preset].scatter_plots in ct.SCATTER_REGISTRY

    def test_registry_matches_the_files_on_disk(self):
        names = {p.stem for p in ct.SETUPS_DIR.glob("*.yaml")}
        assert names == set(ct.TEST_SETUPS)

    def test_e2848_default_shape(self):
        entry = ct.TEST_SETUPS["e2848_default"]
        assert set(entry.meas.reg_cols) == {"power", "poa", "t_amb", "w_vel"}
        assert set(entry.sim.reg_cols) == {"power", "poa", "t_amb", "w_vel"}

    def test_bifi_e2848_etotal_rear_shade_sim_uses_e_total(self):
        from captest.setup import Calc

        node = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim"].meas.reg_cols["poa"]
        assert isinstance(node, Calc) and node.calc == "e_total"

    def test_rear_shade_sim_presets_constrain_rear_shade(self):
        for name in (
            "bifi_e2848_etotal_rear_shade_sim",
            "bifi_power_tc_etotal_rear_shade_sim",
            "bifi_e2848_etotal_rear_shade_sim_spec_corrected",
        ):
            assert ct.TEST_SETUPS[name].params == {"rear_shade": 0}, name

    def test_rear_shade_meas_presets_are_unconstrained(self):
        assert ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_meas"].params == {}

    def test_e2848_spec_corrected_poa_meas_tree_uses_spectral_factor(self):
        poa = ct.TEST_SETUPS["e2848_spec_corrected_poa"].meas.reg_cols["poa"]
        assert poa.calc == "poa_spec_corrected"
        assert poa.args["spectral_correction"].calc == "spectral_factor_firstsolar"

    def test_e2848_spec_corrected_poa_sim_uses_apparent_zenith_pvsyst(self):
        poa = ct.TEST_SETUPS["e2848_spec_corrected_poa"].sim.reg_cols["poa"]
        zenith = poa.args["spectral_correction"].args["absolute_airmass"].args[
            "apparent_zenith"
        ]
        assert zenith.calc == "apparent_zenith_pvsyst"
```

Create `tests/test_presets.py`:

```python
"""Shipped presets: normalisation, identity, and the stored digests."""

import json
from pathlib import Path

import pytest

from captest import captest as ct
from captest.setup import TestSetup

DIGESTS = Path("tests/data/setup_digests.json")


@pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
def test_normalisation_is_idempotent(preset):
    tsd = ct.TEST_SETUPS[preset]
    assert TestSetup.model_validate(tsd.to_dict()).to_dict() == tsd.to_dict()


@pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
def test_digest_matches_the_stored_value(preset):
    stored = json.loads(DIGESTS.read_text())
    assert ct.TEST_SETUPS[preset].content_digest() == stored[preset], (
        f"{preset} changed; if intended, regenerate tests/data/setup_digests.json"
    )


def test_every_preset_has_a_stored_digest():
    assert set(json.loads(DIGESTS.read_text())) == set(ct.TEST_SETUPS)
```

Replace `TestResolveTestSetup`:

```python
class TestResolveTestSetup:
    def test_named_preset_no_overrides(self):
        assert ct.resolve_test_setup("e2848_default") == ct.TEST_SETUPS["e2848_default"]

    def test_reg_fml_override(self):
        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel)"
        assert ct.resolve_test_setup("e2848_default", {"reg_fml": fml}).reg_fml == fml

    def test_rep_conditions_partial_merge(self):
        out = ct.resolve_test_setup(
            "e2848_default", {"rep_conditions": {"percent_filter": 10}}
        )
        assert out.rep_conditions.percent_filter == 10
        assert out.rep_conditions.irr_bal is False
        assert set(out.rep_conditions.func) == {"poa", "t_amb", "w_vel"}

    def test_rep_conditions_func_partial_merge_takes_strings(self):
        out = ct.resolve_test_setup(
            "e2848_default", {"rep_conditions": {"func": {"poa": "perc_55"}}}
        )
        assert out.rep_conditions.func == {"poa": "perc_55", "t_amb": "mean", "w_vel": "mean"}

    def test_reg_cols_override_merges_one_term(self):
        out = ct.resolve_test_setup(
            "e2848_default", {"reg_cols_meas": {"poa": {"group": "irr_ghi"}}}
        )
        assert out.meas.reg_cols["poa"].group == "irr_ghi"
        assert out.meas.reg_cols["power"] == ct.TEST_SETUPS["e2848_default"].meas.reg_cols["power"]
        assert out.derived_from == "e2848_default"

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown test_setup"):
            ct.resolve_test_setup("nonexistent")

    def test_unknown_scatter_name_raises_with_hint(self):
        with pytest.raises(ValueError, match="etotal"):
            ct.resolve_test_setup("e2848_default", {"scatter_plots": "etotl"})

    def test_custom_requires_all_three_overrides(self):
        with pytest.raises(ValueError, match="test_setup='custom'"):
            ct.resolve_test_setup("custom", overrides={"reg_fml": "y ~ x"})

    def test_custom_with_minimal_overrides(self):
        out = ct.resolve_test_setup(
            "custom",
            overrides={
                "reg_cols_meas": {"power": {"column": "p"}, "poa": {"column": "i"}},
                "reg_cols_sim": {"power": {"column": "p"}, "poa": {"column": "i"}},
                "reg_fml": "power ~ poa",
            },
        )
        assert out.name == "custom"
        assert out.scatter_plots == "default"
        assert out.rep_conditions.func == {}
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_captest.py::TestTestSetupsRegistry tests/test_presets.py -q`
Expected: failures / errors (no `SETUPS_DIR`, `TEST_SETUPS` entries are dicts).

- [ ] **Step 4: Rewrite the preset machinery in `captest.py`**

Delete the `TEST_SETUPS = {...}` literal, `_TEST_SETUP_REQUIRED_KEYS`, `validate_test_setup`, `_encode_override`, and the `from captest.calcparams import (...)` block that only fed the dicts (keep any name still used elsewhere; `grep` before deleting). Add imports:

```python
from importlib import resources

from captest.setup import DerivationError, TestSetup, derive
```

After the three `scatter_*` functions:

```python
#: Scatter-plot callables a setup may name under ``scatter_plots``.
SCATTER_REGISTRY = {
    "default": scatter_default,
    "etotal": scatter_etotal,
    "bifi_power_tc": scatter_bifi_power_tc,
}

#: Directory of shipped preset documents.
SETUPS_DIR = Path(str(resources.files("captest").joinpath("setups")))


def _check_scatter_name(name):
    """Raise ``ValueError`` (with a hint) if ``name`` is not a scatter callable."""
    if name not in SCATTER_REGISTRY:
        raise ValueError(
            f"Unknown scatter_plots {name!r}."
            f"{_suggest_unknown_key(name, SCATTER_REGISTRY)}"
        )


def load_presets(directory=None):
    """Load every ``*.yaml`` preset in ``directory`` (default ``SETUPS_DIR``).

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
```

(`_suggest_unknown_key` is defined further down the module today; move it above this block.) Update `test_setups()` to read `setup.description` instead of `setup["description"]`.

Replace `resolve_test_setup`:

```python
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
        ``reg_cols_sim`` and ``reg_fml``.

    Returns
    -------
    captest.setup.TestSetup

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
    unknown = set(overrides) - set(_RESOLVE_KEYS)
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
        return derive(
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
```

`_merge_rep_conditions` is unchanged (it merges plain dicts). Keep `perc_wrap` exported; delete `_perc_wrap_to_string` from the `captest.py` imports if nothing else uses it there.

- [ ] **Step 5: Store the digests**

```bash
uv run python -c "
import json
from captest.captest import TEST_SETUPS
json.dump({k: v.content_digest() for k, v in TEST_SETUPS.items()},
          open('tests/data/setup_digests.json', 'w'), indent=2, sort_keys=True)
print(open('tests/data/setup_digests.json').read())
"
```

- [ ] **Step 6: Run the registry, preset and resolve tests**

Run: `uv run pytest tests/test_captest.py::TestTestSetupsRegistry tests/test_captest.py::TestResolveTestSetup tests/test_presets.py -q`
Expected: all pass.

- [ ] **Step 7: Leave `util._perc_wrap_to_string` in place.** `filters._encode_func_value` still serializes the `perc_wrap` callables a `RepCond` step holds; only the `captest.py` preset code stops using it.

- [ ] **Step 8: Run the oracle test (the equivalence proof)**

Run: `uv run pytest tests/test_setup_oracles.py -q`
Expected: fails on `CapTest.setup()` because `CapTest` still reads `resolved["reg_cols_meas"]` — that is Task 8. Skip to Step 9; the oracle test is the first thing Task 8 makes green.

- [ ] **Step 9: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/setups src/captest/captest.py tests/data/setup_digests.json tests/test_captest.py tests/test_presets.py
git commit -m "feat!: ship presets as yaml documents loaded into TEST_SETUPS"
```
Then run the review gate. State in the commit body that `CapTest.setup()` is wired in the next commit.

---
### Task 8: `CapTest` integration — params, `setup()`, `rep_cond`, scatter, yaml round trip, `check_fit`

**Red window ends here:** the full suite must be green at the end of this task except `tests/test_plotting.py` (Task 9).

**Files:**
- Modify: `src/captest/captest.py` — params (≈1429-1445), `_downstream_attrs` (1665), `from_mapping` (2180-2262), `_build_yaml_sub_mapping` (2511-2600), `setup()` (≈2790-2860), `scatter_plots` (2946), `rep_cond` (2950-2985), `overlay_scatters` (3380), `resolved_setup` (3495), `load_config` (1128-1145), `_CAPTEST_OVERRIDE_KEYS` (1239), `_serialize_rep_conditions` (1066)
- Modify: `src/captest/util.py` (delete `_perc_wrap_to_string`; keep `_resolve_perc_string`, `_resolve_func_strings`)
- Test: `tests/test_captest.py`

**Interfaces:**
- Produces: `CapTest.params` (param.Dict), `CapTest.scatter_plots_name` (param.String, yaml key `overrides.scatter_plots`), `CapTest.resolved_setup` (param.ClassSelector of `TestSetup`, `None` before `setup()`), `CapTest.check_fit(side="both") -> list[FitError]`, `to_mapping()` writing `overrides.reg_cols_*` as the diff against the preset.
- Consumes: `resolve_test_setup`, `SCATTER_REGISTRY`, `setup.check_project_fit`, `setup.SetupFitError`, `util._resolve_func_strings`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captest.py` (new classes; place after `TestResolveTestSetup`):

```python
class TestSetupWiring:
    def test_resolved_setup_is_a_test_setup(self, ct_default):
        from captest.setup import TestSetup

        assert isinstance(ct_default.resolved_setup, TestSetup)

    def test_resolved_setup_is_none_before_setup(self):
        tst = CapTest(test_setup="e2848_default")
        assert tst.resolved_setup is None
        with pytest.raises(RuntimeError):
            tst.scatter_plots()

    def test_reg_cols_override_merges_one_term(self, meas_cd_default, sim_cd_default):
        tst = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            reg_cols_meas={"poa": {"group": "irr_rpoa"}},
            ac_nameplate=6_000_000,
            verbose=False,
        )
        assert tst.meas.regression_cols["poa"] == "irr_rpoa_mean_agg"
        assert tst.meas.regression_cols["power"] == "real_pwr_mtr_sum_agg"

    def test_tier_two_runs_before_evaluation(self, meas_cd_default, sim_cd_default):
        from captest.setup import SetupFitError

        with pytest.raises(SetupFitError, match="meas.reg_cols.poa.group"):
            CapTest.from_params(
                test_setup="e2848_default",
                meas=meas_cd_default,
                sim=sim_cd_default,
                reg_cols_meas={"poa": {"group": "irr_ghi"}},
                verbose=False,
            )
        assert "irr_ghi_mean_agg" not in meas_cd_default.data.columns

    def test_params_constraint_is_enforced(self, meas_cd_default, sim_cd_default):
        from captest.setup import SetupFitError

        with pytest.raises(SetupFitError, match="rear_shade"):
            CapTest.from_params(
                test_setup="bifi_e2848_etotal_rear_shade_sim",
                meas=meas_cd_default,
                sim=sim_cd_default,
                bifaciality=0.15,
                rear_shade=0.2,
                verbose=False,
            )

    def test_check_fit_lists_errors_without_running_setup(
        self, meas_cd_default, sim_cd_default
    ):
        tst = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            reg_cols_meas={"poa": {"group": "irr_ghi"}},
            run_setup=False,
        )
        errors = tst.check_fit()
        assert [e.path for e in errors] == ["meas.reg_cols.poa.group"]
        assert tst.resolved_setup is None

    def test_rep_cond_resolves_func_strings(self, ct_default):
        ct_default.rep_cond()
        assert ct_default.rc is not None

    def test_rep_cond_override_takes_perc_string(self, ct_default):
        ct_default.rep_cond(func={"poa": "perc_55"})
        assert ct_default.rc is not None

    def test_scatter_plots_uses_the_registry(self, ct_default, monkeypatch):
        called = {}
        monkeypatch.setitem(ct.SCATTER_REGISTRY, "default", lambda cd, **kw: called.update(cd=cd))
        ct_default.scatter_plots()
        assert called["cd"] is ct_default.meas

    def test_rep_cond_then_yaml_round_trip(self, ct_default, tmp_path):
        """The RepCond step's perc_wrap callable still serializes and reloads."""
        ct_default.meas.filter_irr(200, 2000)
        ct_default.rep_cond()
        ct_default.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        with open(tmp_path / "c.yaml") as fh:
            sub = yaml.safe_load(fh)["captest"]
        rep_steps = [d for d in sub["meas_filters"] if d["type"] == "RepCond"]
        assert rep_steps and rep_steps[0]["func"]["poa"] == "perc_60"
        CapTest.from_yaml(tmp_path / "c.yaml", run_setup=False)

    def test_edit_after_setup_is_what_gets_written(self, ct_default, tmp_path):
        ct_default.reg_cols_meas = {"poa": {"group": "irr_rpoa", "agg": "mean"}}
        ct_default.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        with open(tmp_path / "c.yaml") as fh:
            sub = yaml.safe_load(fh)["captest"]
        assert sub["overrides"]["reg_cols_meas"] == {"poa": {"group": "irr_rpoa", "agg": "mean"}}


class TestRegColsYamlRoundTrip:
    def _load(self, path):
        with open(path) as fh:
            return yaml.safe_load(fh)["captest"]

    def test_one_overridden_term_writes_only_that_term(self, tmp_path):
        tst = CapTest(test_setup="e2848_default",
                      reg_cols_meas={"poa": {"group": "irr_ghi", "agg": "mean"}})
        tst.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        sub = self._load(tmp_path / "c.yaml")
        assert sub["overrides"]["reg_cols_meas"] == {"poa": {"group": "irr_ghi", "agg": "mean"}}
        assert "reg_cols_sim" not in sub["overrides"]

    def test_removed_term_writes_null_and_reloads_pruned(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) - 1"
        tst = CapTest.from_params(
            test_setup="e2848_default", meas=meas_cd_default, sim=sim_cd_default,
            reg_fml=fml, reg_cols_meas={"w_vel": None}, reg_cols_sim={"w_vel": None},
            verbose=False,
        )
        assert "w_vel" not in tst.resolved_setup.rep_conditions.func
        tst.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        sub = self._load(tmp_path / "c.yaml")
        assert sub["overrides"]["reg_cols_meas"] == {"w_vel": None}
        again = CapTest.from_yaml(tmp_path / "c.yaml", run_setup=False)
        again.meas, again.sim = meas_cd_default, sim_cd_default
        again.setup(verbose=False)
        assert again.resolved_setup == tst.resolved_setup
        assert again.resolved_setup.content_digest() == tst.resolved_setup.content_digest()

    def test_custom_writes_both_sides_in_full(self, tmp_path):
        tst = CapTest(
            test_setup="custom",
            reg_fml="power ~ poa",
            reg_cols_meas={"power": {"column": "p"}, "poa": {"column": "i"}},
            reg_cols_sim={"power": {"column": "p"}, "poa": {"column": "i"}},
        )
        tst.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        sub = self._load(tmp_path / "c.yaml")
        assert set(sub["overrides"]["reg_cols_meas"]) == {"power", "poa"}

    def test_custom_registered_calculation_round_trips(self, tmp_path, monkeypatch):
        from captest.calcparams import CALC_REGISTRY, CalcEntry

        def double(data, col=None):
            return data[col] * 2

        monkeypatch.setitem(CALC_REGISTRY, "double", CalcEntry(double, (), ()))
        tst = CapTest(
            test_setup="e2848_default",
            reg_cols_meas={"power": {"calc": "double",
                                     "args": {"col": {"column": "meter_power"}}}},
        )
        tst.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        again = CapTest.from_yaml(tmp_path / "c.yaml", run_setup=False)
        assert again.reg_cols_meas["power"]["calc"] == "double"

    def test_params_and_scatter_overrides_round_trip(self, tmp_path):
        tst = CapTest(test_setup="e2848_default", params={"rear_shade": 0},
                      scatter_plots_name="etotal")
        tst.to_yaml(tmp_path / "c.yaml", merge_into_existing=False)
        sub = self._load(tmp_path / "c.yaml")
        assert sub["overrides"]["params"] == {"rear_shade": 0}
        assert sub["overrides"]["scatter_plots"] == "etotal"
        again = CapTest.from_yaml(tmp_path / "c.yaml", run_setup=False)
        assert again.params == {"rear_shade": 0}
        assert again.scatter_plots_name == "etotal"
```

Then update the existing tests that the grammar change invalidates (grep the file for each):
- any `rep_conditions` override passing `ct.perc_wrap(N)` → `"perc_N"`;
- `TestSetup` / `TestReload` / `TestDownstreamPropagation` assertions that read `capt._resolved_setup[...]` → `capt.resolved_setup.<field>`;
- `TestResolvedSetupProperty` → the property is now a param that is `None` before setup;
- `TestToMapping` / `TestToYamlAndRoundTrip` expectations that `overrides.rep_conditions.func` holds callables → strings;
- the three tuple-form sites (`grep -n '("irr_poa", "mean")' tests/test_captest.py`) → document form.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_captest.py -q -x`
Expected: first failure in `setup()` reading `resolved["reg_cols_meas"]`.

- [ ] **Step 3: Params and downstream attrs**

In `CapTest`:
```python
    reg_cols_meas = param.Dict(
        default=None, allow_None=True,
        doc="Merged key by key onto the preset's measured regression columns; "
            "a value of None removes that term.",
    )
    reg_cols_sim = param.Dict(
        default=None, allow_None=True,
        doc="Merged key by key onto the preset's modeled regression columns; "
            "a value of None removes that term.",
    )
    params = param.Dict(
        default=None, allow_None=True,
        doc="Required test-level parameter values for this setup (see "
            "captest.setup.TestSetup.params). Replaces the preset's wholesale.",
    )
    scatter_plots_name = param.String(
        default=None, allow_None=True,
        doc="Name in SCATTER_REGISTRY overriding the preset's scatter plot. "
            "Written to yaml as overrides.scatter_plots.",
    )
    resolved_setup = param.ClassSelector(
        class_=TestSetup, default=None, allow_None=True,
        doc="The complete TestSetup resolved by setup(); None before setup().",
    )
```
Replace the `_downstream_attrs` tuple with `_downstream_attrs = DOWNSTREAM_PARAMS` (import from `captest.calcparams`). Delete the `resolved_setup` *property* at the end of the class and `_resolved_setup` wherever it appears (`grep -n _resolved_setup src/captest/captest.py`); `_require_setup` checks `self.resolved_setup is None`.

- [ ] **Step 4: `setup()`**

Replace the override collection and the per-side wiring in `setup()`:

```python
        overrides = {}
        for name in ("reg_cols_meas", "reg_cols_sim", "reg_fml", "rep_conditions", "params"):
            val = getattr(self, name)
            if val is not None:
                overrides[name] = val
        if self.scatter_plots_name is not None:
            overrides["scatter_plots"] = self.scatter_plots_name
        resolved = resolve_test_setup(self.test_setup, overrides=overrides)

        # (existing propagation of _downstream_attrs and _propagate_sim_site stay here)

        fit_errors = []
        for s in sides:
            fit_errors.extend(check_project_fit(resolved, s, getattr(self, s)))
        if fit_errors:
            raise SetupFitError(fit_errors)
        self.resolved_setup = resolved

        for s in sides:
            cd = getattr(self, s)
            cd.regression_cols = dict(getattr(resolved, s).reg_cols)
            cd.regression_formula = resolved.reg_fml
            cd.tolerance = self.test_tolerance
            cd.process_regression_columns(verbose=verbose)
            cd._captest = self
```

Tier 2 runs after the downstream attributes are propagated (so `power_temp_coeff` etc. are on the `CapData`) and before any column is written. Add `check_fit`:

```python
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

        Returns
        -------
        list of captest.setup.FitError
        """
        sides = ("meas", "sim") if side == "both" else (side,)
        resolved = resolve_test_setup(self.test_setup, self._collect_overrides())
        for s in sides:
            if getattr(self, s) is None:
                raise RuntimeError(f"CapTest.{s} must be set before check_fit().")
        # The same preparation setup() performs before it evaluates anything:
        # downstream params onto each side, and meas.site onto sim (the
        # spectral presets need it there). Neither touches ``data``.
        for s in sides:
            cd = getattr(self, s)
            for name in self._downstream_attrs:
                if s == "sim" and name in self._downstream_attrs_meas_only:
                    continue
                setattr(cd, name, getattr(self, name))
        if "sim" in sides and self.meas is not None:
            self._propagate_sim_site()
        errors = []
        for s in sides:
            errors.extend(check_project_fit(resolved, s, getattr(self, s)))
        return errors
```

Factor the override-collection block into `_collect_overrides(self) -> dict` used by both methods (write it out; the `{...}` above is the block shown in the `setup()` snippet).

- [ ] **Step 5: `rep_cond`, scatter, `overlay_scatters`**

```python
        resolved_rc = _merge_rep_conditions(
            self.resolved_setup.rep_conditions.model_dump(mode="json"), overrides
        )
        resolved_rc["func"] = util._resolve_func_strings(resolved_rc.get("func") or {})
        return cd.rep_cond(**resolved_rc)
```
`scatter_plots()` → `return SCATTER_REGISTRY[self.resolved_setup.scatter_plots](cd, **kwargs)`; `overlay_scatters` (≈3380) → `scatter_fn = SCATTER_REGISTRY[self.resolved_setup.scatter_plots]`. `captest_results` and anything else reading `_resolved_setup` follow the same substitution.

- [ ] **Step 6: `load_config`, `from_mapping`, `_serialize_rep_conditions`**

- `load_config`: delete the two `_resolve_func_strings` blocks (strings stay strings). Drop `_perc_wrap_to_string` from the `captest.py` import list only if `captest.py` no longer references it; **keep it in `util.py`** — `filters._encode_func_value` (filters.py ≈ 1776) uses it to serialize the `perc_wrap` callables that `rep_cond()` hands to the `RepCond` step, so the pipeline yaml round trip after `rep_cond()` depends on it (tested below). Keep `_resolve_perc_string` / `_resolve_func_strings`.
- `_CAPTEST_OVERRIDE_KEYS = frozenset({"reg_cols_meas", "reg_cols_sim", "reg_fml", "rep_conditions", "params", "scatter_plots"})`.
- `from_mapping`: delete the `decode_reg_cols` block; when lifting overrides, map `overrides["scatter_plots"]` to `kwargs["scatter_plots_name"]`; the `custom` requirement check stays.
- `_serialize_rep_conditions`: drop the `func` special case (values are strings already); keep `to_native`.

- [ ] **Step 7: `_build_yaml_sub_mapping`**

Replace the overrides block:

```python
        overrides = {}
        # Always resolve from the *current* params, never from the cached
        # ``resolved_setup``: an override edited after setup() must be what
        # gets written, and a changed ``test_setup`` must not be paired with
        # the previous preset's document.
        resolved = resolve_test_setup(self.test_setup, self._collect_overrides())
        if self.test_setup == "custom":
            overrides["reg_cols_meas"] = resolved.meas.model_dump(mode="json")["reg_cols"]
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
            overrides["params"] = dict(self.params)
        if self.scatter_plots_name is not None:
            overrides["scatter_plots"] = self.scatter_plots_name
```
with the module-level helper:

```python
def _reg_cols_diff(base, resolved):
    """Overrides that turn ``base`` into ``resolved``: changed terms as
    document nodes, removed terms as ``None``."""
    diff = {}
    for var, node in resolved.items():
        if base.get(var) != node:
            diff[var] = node.model_dump(mode="json")
    for var in base:
        if var not in resolved:
            diff[var] = None
    return diff
```
The `rep_conditions` block that follows is unchanged.

- [ ] **Step 8: Run the whole suite except plotting**

Run: `uv run pytest tests --ignore=tests/test_plotting.py -q`
Expected: all pass, including `tests/test_setup_oracles.py` (every preset reproduces its pre-migration numbers) and `tests/test_captest.py`. Work through remaining failures in `test_captest.py` one at a time; each is either a test still written against the tuple/callable form (update it) or a real regression (fix the code).

- [ ] **Step 9: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/captest.py src/captest/util.py tests/test_captest.py
git commit -m "feat!: CapTest resolves TestSetup documents, merges reg_cols overrides, checks fit"
```
Then run the review gate.

---

### Task 9: `plotting.py` over nodes

**Files:**
- Modify: `src/captest/plotting.py:61-75` (`DEFAULT_TC_POWER_CALC`), `:623-651` (`_missing_column_groups`), `:695-810` (`calc_tc_power_column`)
- Test: `tests/test_plotting.py:149-240`

**Interfaces:**
- Produces: `plotting.DEFAULT_TC_POWER_CALC: dict[str, Calc]`, `plotting._missing_column_groups(node, available_groups) -> set[str]`, `plotting.calc_tc_power_column(cd, tc_power_calc: dict, ...)` accepting document mappings or nodes.

- [ ] **Step 1: Update the tests**

In `TestCalcTcPowerColumn._calc_spec` return document form:

```python
        return {
            "power": {
                "calc": "power_temp_correct",
                "args": {
                    "power": {"group": "real_pwr_mtr", "agg": "sum"},
                    "cell_temp": {
                        "calc": "cell_temp",
                        "args": {"poa": {"group": "irr_poa"}, "bom": {"group": "temp_bom"}},
                    },
                },
            },
        }
```
In `test_rejects_spec_without_top_level_power_calculation` use `{"power": {"group": "real_pwr_mtr", "agg": "sum"}, ...}` and `match="top-level 'power' calc"`. Add:

```python
    def test_accepts_node_models(self, synth_cd):
        from captest.setup import Side

        spec = dict(Side.model_validate({"reg_cols": self._calc_spec()}).reg_cols)
        assert plotting.calc_tc_power_column(synth_cd, spec) == plotting.TC_POWER_PLOT_COL

    def test_default_spec_is_valid(self):
        from captest.setup import Calc, Side

        side = Side.model_validate({"reg_cols": plotting.DEFAULT_TC_POWER_CALC})
        assert isinstance(side.reg_cols["power"], Calc)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_plotting.py::TestCalcTcPowerColumn -q`
Expected: failures (tuple checks reject mappings).

- [ ] **Step 3: Implement**

```python
from captest.setup import Calc, Group, Side

DEFAULT_TC_POWER_CALC = {
    "power": {
        "calc": "power_temp_correct",
        "args": {
            "power": {"group": "real_pwr_mtr", "agg": "sum"},
            "cell_temp": {
                "calc": "cell_temp",
                "args": {"poa": {"group": "irr_poa"}, "bom": {"group": "temp_bom"}},
            },
        },
    },
}


def _missing_column_groups(node, available_groups):
    """Column-group ids referenced by ``Group`` nodes in ``node`` but absent
    from ``available_groups``. ``Column`` nodes and literals are not checked."""
    missing = set()
    if isinstance(node, dict):
        for value in node.values():
            missing |= _missing_column_groups(value, available_groups)
    elif isinstance(node, Group):
        if node.group not in available_groups:
            missing.add(node.group)
    elif isinstance(node, Calc):
        missing |= _missing_column_groups(node.args, available_groups)
    return missing
```
In `calc_tc_power_column`, replace the tuple validation with:

```python
    side = Side.model_validate({"reg_cols": dict(tc_power_calc)})
    spec = dict(side.reg_cols)
    if not isinstance(spec.get("power"), Calc):
        raise ValueError(
            "calc_tc_power_column requires tc_power_calc to include a top-level "
            "'power' calc node, such as 'power': {'calc': 'power_temp_correct', ...}."
        )
    missing_groups = sorted(_missing_column_groups(spec, set(cd.column_groups.keys())))
    if missing_groups:
        raise KeyError(...)  # existing message
    result = util.transform_calc_params(spec, cd, verbose=verbose)
```
(the `copy.deepcopy` is no longer needed — nodes are immutable). Update the function's docstring to describe document nodes and drop `import copy` if unused.

- [ ] **Step 4: Run the full suite**

Run: `just test`
Expected: all pass.

- [ ] **Step 5: Lint, format, commit, review gate**

```bash
just lint && just fmt
git add src/captest/plotting.py tests/test_plotting.py
git commit -m "feat: plotting temperature-corrected power spec in document form"
```
Then run the review gate.

---
### Task 10: Packaging check, changelog, docs, skill, `CLAUDE.md`

**Files:**
- Modify: `tests/smoke_test.py`
- Modify: `CHANGELOG.md` (`## [Unreleased]`)
- Modify: `docs/user_guide/custom_test_setups.rst`, `docs/user_guide/captest.rst`, `docs/user_guide/dataload.rst`, `docs/user_guide/bifacial.rst`
- Modify: `docs/source/api_reference/captest.rst`, `docs/source/api_reference/util.rst`, `docs/source/api_reference/index.rst`; create `docs/source/api_reference/setup.rst`
- Modify: `.agents/skills/add-test-setup/SKILL.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Smoke test asserts the presets ship in the wheel**

Append to `test_smoke()` in `tests/smoke_test.py`:

```python
    # Preset documents are package data; a missing package-data entry only
    # shows up in a built artifact, which is what this test runs against.
    from captest.captest import SETUPS_DIR, TEST_SETUPS

    assert SETUPS_DIR.is_dir(), SETUPS_DIR
    assert len(TEST_SETUPS) >= 10
    assert "e2848_default" in TEST_SETUPS
```
Run: `just build && uv run --isolated --with dist/*.whl python tests/smoke_test.py` (or the `test-install` recipe in `.justfile`).
Expected: "Smoke test succeeded".

- [ ] **Step 2: Changelog**

Under `## [Unreleased]` add a `### Changed` (breaking) entry and extend `### Added` / `### Removed`:

```markdown
### Changed
- **Breaking:** test setups are now pure-data documents. `regression_cols`
trees use tagged nodes — `{group: irr_poa, agg: mean}`, `{column: E_Grid}`,
`{calc: e_total, args: {...}}` — instead of `(group, agg)` / `(callable,
kwargs)` tuples, and every calculation is referenced by its registry name.
The tuple grammar is not accepted anywhere. Presets ship as
`src/captest/setups/<name>.yaml` and `TEST_SETUPS` holds
`captest.setup.TestSetup` models. `reg_cols_meas` / `reg_cols_sim` overrides
merge key by key onto the preset (`null` removes a term) and `to_yaml` writes
only the changed terms. `rep_conditions.func` values are the strings `mean`,
`median` or `perc_N`; `perc_wrap(...)` callables are no longer accepted in a
setup or override. `CapData.custom_param` gains `output=` and injects
`CapData` attributes only for absent keyword arguments (an explicit `None`
now reaches the function). New dependency: `pydantic>=2.5,<3`.

### Added
- `captest.setup`: `TestSetup` documents with tier-1 validation, `derive`,
`content_digest`, `TestSetup.load` / `to_yaml` / `to_json` / `json_schema`,
and tier-2 `check_project_fit`; `CapTest.check_fit()`; `CapTest.params` and
`CapTest.scatter_plots_name` overrides; `calcparams.register_calc` /
`CALC_REGISTRY`; `captest.SCATTER_REGISTRY`.

### Removed
- `util.encode_reg_cols`, `util.decode_reg_cols`, `util.update_by_path`,
`captest.validate_test_setup`.
```

- [ ] **Step 3: User guide**

Invoke the `docs-update` skill with this brief, then verify each item landed:

- `custom_test_setups.rst` — rewrite "The regression column dictionary grammar" around the three node kinds (table from the spec), keyword-only `args`, literals, the no-shorthand rule; rewrite "Using calcparams functions" / "Using custom functions" around `@register_calc` (a custom function must be registered in your code before the setup that names it is loaded; a document never imports code); "Creating a Custom Regression Columns Dictionary" and "Wiring a custom dict into CapTest" show yaml documents and the key-level `overrides` (the `irr_ghi` example), `tst.check_fit()`, and `test_setup: custom`. Every example is yaml or a plain-dict document; no tuples, no callables.
- `captest.rst` — lines 71-79 and 1337-1360 (the overrides examples): node form; state that an override replaces only the named terms.
- `dataload.rst` lines 11, 198-206, 230-250 — `set_regression_cols` builds nodes; the `regression_cols` examples in document form; `agg_sensors` paragraph unchanged in substance.
- `bifacial.rst` lines 49-75 — document form.

- [ ] **Step 4: API reference**

- Create `docs/source/api_reference/setup.rst` with `.. currentmodule:: captest` and an autosummary of `setup.TestSetup`, `setup.Side`, `setup.RepConditions`, `setup.Group`, `setup.Column`, `setup.Calc`, `setup.derive`, `setup.check_project_fit`, `setup.FitError`, `setup.SetupFitError`, `setup.canonical_json`, `setup.parse_regression_formula`; add `setup` to `index.rst` after `captest`.
- `captest.rst`: in "Module-level Functions" remove `validate_test_setup`, add `captest.captest.load_presets`, `captest.captest.SETUPS_DIR`, `captest.captest.SCATTER_REGISTRY`; in "Setup" add `CapTest.check_fit`, `CapTest.params`, `CapTest.scatter_plots_name`; rewrite the `TEST_SETUPS` data description ("values are `captest.setup.TestSetup` documents loaded from `setups/*.yaml`") and the `rear_shade` warning (the `_sim` presets now *refuse* a non-zero `rear_shade` via `params`).
- `util.rst`: remove `update_by_path` and the "Configuration" section; add `util.canonical_json` under Regression or a new "Documents" section.
- `calcparams.rst`: add `calcparams.register_calc`, `calcparams.CalcEntry`, `calcparams.CALC_REGISTRY`.
Run: `just docs` — no new warnings from these pages.

- [ ] **Step 5: Rewrite the `add-test-setup` skill**

Replace the workflow in `.agents/skills/add-test-setup/SKILL.md` so a new preset is: (1) expand the description; (2) regression equation → approval gate; (3) write `src/captest/setups/<name>.yaml` in the node grammar, copying the nearest preset; (4) `params` if the sim side carries rear shading and the meas side calls `e_total`; (5) `uv run python -c "from captest.captest import TEST_SETUPS; print(TEST_SETUPS['<name>'])"` to prove it loads; (6) add the digest line to `tests/data/setup_digests.json`, a `PRESET_FIXTURES` entry in `tests/setup_fixtures.py`, and capture its oracle; (7) approval gate on the review summary; (8) docs entry in `api_reference/captest.rst`. Update the frontmatter description to say "yaml preset document in `src/captest/setups/`".

- [ ] **Step 6: `CLAUDE.md`**

In "Key Modules" add a `src/captest/setup.py` entry (document model, node grammar, `derive`, tiers, import rule: never imports `capdata`/`captest`), note the registry in `calcparams.py`, and change the `captest.py` entry: `TEST_SETUPS` is loaded from `setups/*.yaml`, `resolve_test_setup` returns a `TestSetup`, overrides merge key by key. Update the "Standard Workflow → CapData step 2" sentence to mention nodes.

- [ ] **Step 7: Full verification, commit, review gate**

Run: `just lint && just fmt && just test && just docs`
Expected: clean lint, all tests pass, docs build.

```bash
git add tests/smoke_test.py CHANGELOG.md docs CLAUDE.md .agents/skills/add-test-setup/SKILL.md
git commit -m "docs: document setup documents, registry and overrides; update skill and changelog"
```
Then run the review gate.

---

### Task 11: Branch wrap-up

- [ ] **Step 1: Confirm the branch state**

Run: `git status --short && git log --oneline master..HEAD`
Expected: clean tree apart from the pre-existing `pyproject.toml` change the owner left uncommitted (leave it), and one commit per task above plus any `fix: address roborev review` commits.

- [ ] **Step 2: Hand off**

Invoke `superpowers:finishing-a-development-branch`. The PR description should name the breaking change, the migration steps for pft-mono (`perfactory/captest.py`, `ctsweep/adhoc.py`, `captest-gui`), and point at the spec and this plan.

---

## Self-review notes

- **Spec coverage:** grammar and node rules → Task 3; document shape, `params` semantics, output names → Tasks 3, 6; architecture/import direction → Global Constraints + Tasks 2-3 (`parse_regression_formula` relocation); registry → Task 2; precedence/injection → Tasks 5-6; evaluation and ownership → Task 5; presets, `resolve_test_setup`, `SCATTER_REGISTRY` → Task 7; `CapTest` params, setup ordering, `rep_cond`, `to_mapping` diff, `from_mapping`, `check_fit` → Task 8; plotting → Task 9; tiers → Tasks 3, 6, 8; trust boundary → no code change (docs mention in Task 10 via `custom_test_setups.rst`); error handling → Tasks 3, 6, 7, 8; testing section → Tasks 1-9 (oracles Task 1, equivalence Task 8, rejection cases Tasks 3/4/6/8, registry declarations Task 2, evaluation Task 5, round trip Task 8, plotting Task 9); packaging/docs → Tasks 2, 10; Extending → Task 10 skill + docs.
- **Deviations from the spec, each amended in the spec text by the task that introduces it:** `setups/` directory name (Task 2); tier-2 raw-column shadow check against sensor columns, not all `data` columns (Task 6); `DOWNSTREAM_PARAMS` defined in `calcparams.py` and re-exported (Task 2/3); `INJECTED_PARAMS` / `NULLABLE_INJECTED` making `site` a requireable injection and `altitude_override` never required (Task 2); `scatter_plots` override exposed as the `CapTest.scatter_plots_name` param because `CapTest.scatter_plots` is a method (Task 8).
- **Type consistency:** `check_project_fit(setup, side, cd)`, `effective_value(name, node, cd)`, `derive(base, *, ...)`, `merge_reg_cols(base_side, override, formula_vars)`, `custom_param(func, *, output=None, verbose=True, **kwargs)`, `_get_or_create_aggregation(node, cd, agg_cache, verbose)`, `resolve_test_setup(name, overrides=None)`, `_reg_cols_diff(base, resolved)`, `_collect_overrides(self)` are used with the same signatures everywhere above.
