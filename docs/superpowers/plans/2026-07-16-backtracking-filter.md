# Tracker Backtracking Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `Backtracking` filter that removes (or isolates) single-axis-tracker backtracking intervals from a `CapData` dataset, decided per interval by a transcription of pvlib's own `tracking.singleaxis` backtracking predicate.

**Architecture:** Three additions to `filters.py` — a geometry-validation function `_backtracking_geometry_error(...)`, a pure predicate helper `backtracking_active(...)`, and a `Backtracking(BaseFilter)` step class — plus a thin in-place `CapData.filter_backtracking(...)` wrapper and a `FILTER_REGISTRY` entry. The filter reads tracker geometry from `capdata.site['sys']` (with per-param overrides), computes solar position on the fly from `capdata.site['loc']` via pvlib, and degrades to warn-and-no-op when site/geometry/pvlib are unavailable or geometry is invalid.

**Tech Stack:** Python, `param` (parameterized filter classes), `pandas`, `pvlib` (`shading.projected_solar_zenith_angle`, `tools.cosd`, `location.Location.get_solarposition`, `tracking.singleaxis` in tests), `pytest`, `uv`, `ruff`.

## Global Constraints

- Line length: 88 characters (ruff default).
- Docstrings: NumPy-style for all public functions/classes/methods.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants.
- `filters.py` is imported one-way by `capdata.py`; it **never** imports `capdata`. Steps touch a `CapData` only through the runtime `capdata` argument.
- Filter wrappers act **in place** (no `inplace` kwarg) and accept an optional `custom_name`.
- `data_filtered` is a derived read-only property — never assign to it; return the kept `Index` from `_execute`.
- pvlib is guarded at module import in `filters.py` (pattern already present for `detect_clearsky`).
- Run tooling via `uv` directly, not `just` (the local `just` dotenv-load is broken by a malformed `~/.env`): lint `uv run ruff check --fix <path>`, format `uv run ruff format <path>`, tests `uv run --python 3.12 pytest <path>`.
- Commit messages end with the co-author trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- **`src/captest/filters.py`** (modify) — add `import math` and `import numbers` to the stdlib import block; add `_backtracking_geometry_error(...)`, `backtracking_active(...)`, and `class Backtracking(BaseFilter)`; add `"Backtracking": Backtracking` to `FILTER_REGISTRY`. Insert the class immediately after `Clearsky` (ends ~line 1126) and before `Pvsyst`.
- **`src/captest/capdata.py`** (modify) — add `Backtracking` to the `from captest.filters import (...)` block (line 37); add the `filter_backtracking(...)` wrapper immediately after `filter_clearsky` (ends ~line 2107) and before `filter_missing`.
- **`tests/test_filter_classes.py`** (modify) — add `Backtracking, backtracking_active` (and `_backtracking_geometry_error`) to the `from captest.filters import (...)` block; add a `cd_backtrack` fixture; add `TestBacktrackingGeometryError`, `TestBacktrackingActiveHelper`, `TestFilterBacktracking`, and `TestBacktrackingWrapper` classes. Config round-trip assertions extend the existing base-config test area.

---

### Task 1: Geometry validation function `_backtracking_geometry_error`

Pure, pvlib-free validation of the resolved geometry. Single source of truth used by both the helper (which raises) and the filter (which warns-and-no-ops).

**Files:**
- Modify: `src/captest/filters.py` (stdlib imports near lines 8-12; add function after `fit_model`, before `class BaseSummaryStep` ~line 267)
- Test: `tests/test_filter_classes.py` (new class `TestBacktrackingGeometryError`)

**Interfaces:**
- Consumes: nothing (stdlib `math`, `numbers` only).
- Produces: `_backtracking_geometry_error(axis_tilt, axis_azimuth, gcr, cross_axis_tilt) -> str | None` — returns a human-readable reason string when geometry is invalid, else `None`. Invalid = any value `None`; any value not a real number (bools rejected explicitly); any value non-finite; `gcr <= 0`; `cross_axis_tilt` not in the open interval `(-90, 90)`. Check order: `None` → non-real → non-finite → `gcr` sign → `cross_axis_tilt` range.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_filter_classes.py`. First extend the filters import to include the new name:

```python
from captest.filters import (
    # ... existing names ...
    _backtracking_geometry_error,
)
```

Then add the test class:

```python
import math

import pandas as pd  # already imported at top; do not duplicate


class TestBacktrackingGeometryError:
    def test_valid_geometry_returns_none(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 0) is None

    def test_none_value_reports_param_name(self):
        assert "gcr" in _backtracking_geometry_error(0, 180, None, 0)
        assert "axis_tilt" in _backtracking_geometry_error(None, 180, 0.3, 0)

    def test_gcr_zero_is_invalid(self):
        reason = _backtracking_geometry_error(0, 180, 0, 0)
        assert reason is not None
        assert "gcr" in reason

    def test_gcr_negative_is_invalid(self):
        assert "gcr" in _backtracking_geometry_error(0, 180, -0.3, 0)

    def test_non_finite_values_are_invalid(self):
        assert _backtracking_geometry_error(0, 180, math.nan, 0) is not None
        assert _backtracking_geometry_error(0, 180, math.inf, 0) is not None
        assert _backtracking_geometry_error(math.nan, 180, 0.3, 0) is not None

    def test_string_value_is_invalid_without_typeerror(self):
        # Must not raise TypeError from math.isfinite on a str.
        reason = _backtracking_geometry_error(0, 180, "0.3", 0)
        assert reason is not None
        assert "gcr" in reason

    def test_pd_na_is_invalid_without_typeerror(self):
        reason = _backtracking_geometry_error(0, 180, pd.NA, 0)
        assert reason is not None

    def test_bool_is_invalid(self):
        # bool is a numbers.Real subtype; must be rejected, not coerced to 1/0.
        assert _backtracking_geometry_error(0, 180, True, 0) is not None

    def test_cross_axis_tilt_at_90_is_invalid(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 90) is not None
        assert _backtracking_geometry_error(0, 180, 0.3, -90) is not None

    def test_cross_axis_tilt_within_range_is_valid(self):
        assert _backtracking_geometry_error(0, 180, 0.3, 45) is None
        assert _backtracking_geometry_error(0, 180, 0.3, -45) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingGeometryError -v`
Expected: FAIL — `ImportError: cannot import name '_backtracking_geometry_error'`.

- [ ] **Step 3: Add stdlib imports**

In `src/captest/filters.py`, extend the stdlib import block (currently `import copy`, `import difflib`, `import importlib.util`, `from itertools import combinations`, `import warnings`) to add:

```python
import copy
import difflib
import importlib.util
from itertools import combinations
import math
import numbers
import warnings
```

- [ ] **Step 4: Implement the function**

In `src/captest/filters.py`, add immediately before `class BaseSummaryStep` (after the `fit_model` function, ~line 265):

```python
def _backtracking_geometry_error(axis_tilt, axis_azimuth, gcr, cross_axis_tilt):
    """Return a reason string if the resolved backtracking geometry is invalid.

    Returns ``None`` when every value is usable. Invalid conditions, checked in
    order: any value is ``None`` (unresolved); any value is not a real number
    (``bool`` is rejected explicitly, since it is a ``numbers.Real`` subtype and
    would otherwise be silently coerced to 1/0); any value is non-finite
    (``NaN``/``inf``); ``gcr <= 0`` (the axes-distance denominator must be
    positive and nonzero); or ``cross_axis_tilt`` outside the open interval
    ``(-90, 90)`` (so ``cosd(cross_axis_tilt)`` is finite and strictly positive
    — a robust replacement for a ``cosd(...) == 0`` check that pvlib's non-exact
    ``cosd(90) ≈ 6e-17`` would defeat).

    The type check precedes ``math.isfinite`` so non-numeric site metadata (a
    string, ``pd.NA``, an arbitrary object) is reported as a reason rather than
    raising ``TypeError``.

    Parameters
    ----------
    axis_tilt, axis_azimuth, gcr, cross_axis_tilt
        Resolved tracker-geometry values to validate.

    Returns
    -------
    str or None
        A human-readable reason when the geometry is invalid, otherwise None.
    """
    checks = {
        "axis_tilt": axis_tilt,
        "axis_azimuth": axis_azimuth,
        "gcr": gcr,
        "cross_axis_tilt": cross_axis_tilt,
    }
    for name, value in checks.items():
        if value is None:
            return (
                f"{name} could not be resolved (pass it explicitly or set it "
                "in site['sys'])"
            )
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{name}={value!r} is not a real number"
        if not math.isfinite(value):
            return f"{name}={value!r} is not a finite number"
    if gcr <= 0:
        return f"gcr={gcr!r} must be greater than 0"
    if not (-90 < cross_axis_tilt < 90):
        return f"cross_axis_tilt={cross_axis_tilt!r} must be between -90 and 90"
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingGeometryError -v`
Expected: PASS (all 10 tests).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/captest/filters.py tests/test_filter_classes.py
uv run ruff format src/captest/filters.py tests/test_filter_classes.py
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add backtracking geometry validation helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Predicate helper `backtracking_active`

The pure geometry test transcribing pvlib's `tracking.singleaxis` backtracking decision. Validated numerically against `pvlib.tracking.singleaxis`.

**Files:**
- Modify: `src/captest/filters.py` (add function after `_backtracking_geometry_error`)
- Test: `tests/test_filter_classes.py` (new class `TestBacktrackingActiveHelper`)

**Interfaces:**
- Consumes: `_backtracking_geometry_error` (Task 1).
- Produces: `backtracking_active(apparent_zenith, solar_azimuth, axis_tilt, axis_azimuth, gcr, cross_axis_tilt=0) -> pd.Series` (boolean, indexed like the inputs) — `True` where backtracking is geometrically active (sun up AND true-tracking would shade). Raises `ValueError` when `_backtracking_geometry_error` returns a reason.

- [ ] **Step 1: Write the failing tests**

Extend the filters import in `tests/test_filter_classes.py` to add `backtracking_active`. Then add:

```python
class TestBacktrackingActiveHelper:
    @pytest.fixture
    def clear_day_solpos(self):
        """Solar position over a clear June day at a mid-latitude site."""
        from pvlib.location import Location

        loc = Location(35.0, -100.0, altitude=300, tz="Etc/GMT+7")
        times = pd.date_range(
            "2023-06-15 04:00", "2023-06-15 20:00", freq="5min", tz="Etc/GMT+7"
        )
        sp = loc.get_solarposition(times)
        return sp["apparent_zenith"], sp["azimuth"]

    def test_matches_pvlib_singleaxis_at_sun_up(self, clear_day_solpos):
        from pvlib import tracking

        zen, azi = clear_day_solpos
        axis_tilt, axis_azimuth, gcr = 0, 180, 0.4
        # Oracle: singleaxis with max_angle high enough to avoid clipping so the
        # backtrack on/off difference isolates the backtracking decision.
        tracked = tracking.singleaxis(
            apparent_zenith=zen, solar_azimuth=azi,
            axis_tilt=axis_tilt, axis_azimuth=axis_azimuth,
            max_angle=90, backtrack=True, gcr=gcr, cross_axis_tilt=0,
        )
        true_track = tracking.singleaxis(
            apparent_zenith=zen, solar_azimuth=azi,
            axis_tilt=axis_tilt, axis_azimuth=axis_azimuth,
            max_angle=90, backtrack=False, gcr=gcr, cross_axis_tilt=0,
        )
        # pvlib backtracks exactly where the tracked angle differs from the
        # true-tracking angle (both non-NaN, i.e. sun up).
        sun_up = tracked["tracker_theta"].notna() & true_track["tracker_theta"].notna()
        pvlib_backtracking = (
            (tracked["tracker_theta"] - true_track["tracker_theta"]).abs() > 1e-6
        ) & sun_up

        mask = backtracking_active(zen, azi, axis_tilt, axis_azimuth, gcr)
        # Compare only where the sun is up (the helper's <=90 term and pvlib's
        # NaN handling agree there).
        assert mask[sun_up].equals(pvlib_backtracking[sun_up])

    def test_sun_down_intervals_are_false(self, clear_day_solpos):
        zen, azi = clear_day_solpos
        mask = backtracking_active(zen, azi, 0, 180, 0.4)
        assert not mask[zen > 90].any()

    def test_cross_axis_tilt_changes_result(self, clear_day_solpos):
        zen, azi = clear_day_solpos
        flat = backtracking_active(zen, azi, 0, 180, 0.4, cross_axis_tilt=0)
        sloped = backtracking_active(zen, azi, 0, 180, 0.4, cross_axis_tilt=20)
        assert not flat.equals(sloped)

    def test_invalid_gcr_raises(self):
        zen = pd.Series([30.0, 45.0])
        azi = pd.Series([90.0, 100.0])
        with pytest.raises(ValueError, match="gcr"):
            backtracking_active(zen, azi, 0, 180, 0)

    def test_non_numeric_geometry_raises_valueerror_not_typeerror(self):
        zen = pd.Series([30.0, 45.0])
        azi = pd.Series([90.0, 100.0])
        with pytest.raises(ValueError):
            backtracking_active(zen, azi, 0, 180, "0.4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingActiveHelper -v`
Expected: FAIL — `ImportError: cannot import name 'backtracking_active'`.

- [ ] **Step 3: Implement the helper**

In `src/captest/filters.py`, add immediately after `_backtracking_geometry_error`:

```python
def backtracking_active(
    apparent_zenith, solar_azimuth, axis_tilt, axis_azimuth, gcr,
    cross_axis_tilt=0,
):
    """Return a boolean Series marking single-axis-tracker backtracking.

    Direct transcription of pvlib's ``tracking.singleaxis`` backtracking test
    (Anderson & Mikofski 2020): an interval is backtracking-active when the sun
    is above the horizon (``apparent_zenith <= 90``) and row-to-row shade would
    occur under true-tracking.

    Parameters
    ----------
    apparent_zenith : Series
        Apparent solar zenith angle (degrees).
    solar_azimuth : Series
        Solar azimuth angle (degrees).
    axis_tilt : float
        Tracker axis tilt (degrees).
    axis_azimuth : float
        Tracker axis azimuth (degrees).
    gcr : float
        Ground coverage ratio; must be greater than 0.
    cross_axis_tilt : float, default 0
        Cross-axis tilt (degrees); must be in the open interval (-90, 90).

    Returns
    -------
    Series
        Boolean Series indexed like ``apparent_zenith``; True where backtracking
        is geometrically active.

    Raises
    ------
    ValueError
        If the geometry is invalid (see ``_backtracking_geometry_error``).
    """
    from pvlib import shading
    from pvlib.tools import cosd

    reason = _backtracking_geometry_error(
        axis_tilt, axis_azimuth, gcr, cross_axis_tilt
    )
    if reason is not None:
        raise ValueError(f"Invalid backtracking geometry: {reason}.")

    omega_ideal = shading.projected_solar_zenith_angle(
        solar_zenith=apparent_zenith,
        solar_azimuth=solar_azimuth,
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
    )
    axes_distance = 1 / (gcr * cosd(cross_axis_tilt))
    return (apparent_zenith <= 90) & (
        (axes_distance * cosd(omega_ideal - cross_axis_tilt)).abs() < 1
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingActiveHelper -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/captest/filters.py tests/test_filter_classes.py
uv run ruff format src/captest/filters.py tests/test_filter_classes.py
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add backtracking_active predicate helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `Backtracking` filter class + registry

The `BaseFilter` step: resolve geometry from params/`site['sys']`, guard, compute solar position, apply the predicate.

**Files:**
- Modify: `src/captest/filters.py` (add class after `Clearsky` ~line 1126, before `Pvsyst`; add registry entry ~line 1509)
- Test: `tests/test_filter_classes.py` (new fixture `cd_backtrack`, new class `TestFilterBacktracking`)

**Interfaces:**
- Consumes: `backtracking_active` (Task 2), `_backtracking_geometry_error` (Task 1), `BaseFilter` (existing).
- Produces: `class Backtracking(BaseFilter)` with params `axis_tilt` (Number, None), `axis_azimuth` (Number, None), `gcr` (Number, None), `cross_axis_tilt` (Number, default 0), `keep_backtracking` (Boolean, default False). `_execute(capdata) -> pd.Index`. Registry key `"Backtracking"`. On no-op paths returns `capdata.data_filtered.index` unchanged after `warnings.warn`.

- [ ] **Step 1: Write the failing tests**

Extend the filters import in `tests/test_filter_classes.py` to add `Backtracking`. Add a shared fixture and the test class:

```python
@pytest.fixture
def cd_backtrack():
    """A tracker CapData with a clear-day index and a tracker site dict.

    Solar position is computed by the filter from ``site['loc']``; geometry
    defaults come from ``site['sys']``.
    """
    from pvlib.location import Location

    idx = pd.date_range(
        "2023-06-15 04:00", "2023-06-15 20:00", freq="5min", tz="Etc/GMT+7"
    )
    cd = CapData("backtrack")
    # A single poa column is enough; the filter reads geometry/solpos, not poa.
    loc = Location(35.0, -100.0, altitude=300, tz="Etc/GMT+7")
    poa = loc.get_solarposition(idx)["apparent_zenith"].to_numpy()
    cd.data = pd.DataFrame({"poa": poa}, index=idx.tz_localize(None))
    cd.regression_cols = {"poa": "poa"}
    cd.site = {
        "loc": {
            "latitude": 35.0,
            "longitude": -100.0,
            "altitude": 300,
            "tz": "Etc/GMT+7",
        },
        "sys": {
            "axis_tilt": 0,
            "axis_azimuth": 180,
            "gcr": 0.4,
            "max_angle": 60,
            "backtrack": True,
            "albedo": 0.2,
        },
    }
    return cd


class TestFilterBacktracking:
    def test_removes_backtracking_keeps_true_tracking(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) < n_before
        # data_filtered is not mutated by _execute alone.
        assert cd_backtrack.data_filtered.shape[0] == n_before

    def test_keep_backtracking_inverts_mask(self, cd_backtrack):
        removed_default = Backtracking()._execute(cd_backtrack)
        kept_backtracking = Backtracking(keep_backtracking=True)._execute(cd_backtrack)
        full = cd_backtrack.data_filtered.index
        assert removed_default.union(kept_backtracking).equals(full)
        assert removed_default.intersection(kept_backtracking).empty

    def test_resolves_geometry_from_site(self, cd_backtrack):
        f = Backtracking()
        f._execute(cd_backtrack)
        assert f.gcr_resolved == 0.4
        assert f.axis_tilt_resolved == 0
        assert f.axis_azimuth_resolved == 180

    def test_explicit_params_override_site(self, cd_backtrack):
        f = Backtracking(gcr=0.25)
        f._execute(cd_backtrack)
        assert f.gcr_resolved == 0.25

    def test_no_site_warns_and_keeps_all(self, cd_backtrack):
        cd_backtrack.site = None
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="site"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_gcr_zero_in_site_warns_and_keeps_all(self, cd_backtrack):
        cd_backtrack.site["sys"]["gcr"] = 0
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="gcr"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_missing_gcr_key_warns_and_keeps_all(self, cd_backtrack):
        del cd_backtrack.site["sys"]["gcr"]
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="gcr"):
            kept = Backtracking()._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_invalid_cross_axis_tilt_warns_and_keeps_all(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="cross_axis_tilt"):
            kept = Backtracking(cross_axis_tilt=90)._execute(cd_backtrack)
        assert len(kept) == n_before

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["Backtracking"] is Backtracking

    def test_config_round_trips(self):
        f = Backtracking(gcr=0.3, axis_tilt=5, cross_axis_tilt=10,
                         keep_backtracking=True)
        cfg = f.to_config()
        assert cfg["type"] == "Backtracking"
        f2 = step_from_config(cfg)
        assert isinstance(f2, Backtracking)
        assert f2.gcr == 0.3
        assert f2.axis_tilt == 5
        assert f2.cross_axis_tilt == 10
        assert f2.keep_backtracking is True

    def test_config_preserves_none_geometry(self):
        # None geometry (resolve-from-site intent) must survive serialization.
        cfg = Backtracking().to_config()
        assert cfg["gcr"] is None
        assert cfg["axis_tilt"] is None
        assert step_from_config(cfg).gcr is None

    def test_explanation_reports_resolved_geometry(self, cd_backtrack):
        f = Backtracking()
        f.run(cd_backtrack)
        assert "0.4" in f.explanation
        assert "removed" in f.explanation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestFilterBacktracking -v`
Expected: FAIL — `ImportError: cannot import name 'Backtracking'`.

- [ ] **Step 3: Implement the class**

In `src/captest/filters.py`, insert immediately after the end of `class Clearsky` (before `class Pvsyst`):

```python
class Backtracking(BaseFilter):
    """Remove intervals where single-axis-tracker backtracking is active.

    Transcribes pvlib's own ``tracking.singleaxis`` backtracking test (Anderson
    & Mikofski 2020): an interval is backtracking-active when the sun is above
    the horizon and row-to-row shade would occur under true-tracking. Solar
    position is computed from ``capdata.site['loc']`` via pvlib; tracker
    geometry defaults to ``capdata.site['sys']`` and is overridable per
    parameter.

    By default keeps true-tracking intervals and removes backtracking-active
    ones; set ``keep_backtracking=True`` to invert (keep only backtracking).

    Degrades to a warn-and-no-op (returns the index unchanged) when pvlib is
    unavailable, ``capdata.site`` (or its ``loc``/``sys``) is missing, or the
    resolved geometry is invalid.
    """

    _explanation_template = (
        "{kind} intervals (gcr={gcr}, axis_tilt={axis_tilt}, "
        "axis_azimuth={axis_azimuth}, cross_axis_tilt={cross_axis_tilt}) "
        "were removed."
    )

    axis_tilt = param.Number(
        default=None, allow_None=True,
        doc="Tracker axis tilt (deg). Resolved from site['sys']['axis_tilt'] "
        "when None.",
    )
    axis_azimuth = param.Number(
        default=None, allow_None=True,
        doc="Tracker axis azimuth (deg). Resolved from "
        "site['sys']['axis_azimuth'] when None.",
    )
    gcr = param.Number(
        default=None, allow_None=True,
        doc="Ground coverage ratio. Resolved from site['sys']['gcr'] when None.",
    )
    cross_axis_tilt = param.Number(
        default=0,
        doc="Cross-axis tilt (deg) for sloped terrain. Defaults to 0 (flat), "
        "matching pvlib's default. Must be in (-90, 90).",
    )
    keep_backtracking = param.Boolean(
        default=False,
        doc="Keep true-tracking intervals (False) or keep backtracking "
        "intervals (True).",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered

        if pvlib_spec is None:
            warnings.warn(
                "Backtracking filtering requires the pvlib package; "
                "no intervals removed."
            )
            return df.index

        site = getattr(capdata, "site", None)
        if not site or "loc" not in site or "sys" not in site:
            warnings.warn(
                "Backtracking filter requires capdata.site with 'loc' and "
                "'sys'; no intervals removed."
            )
            return df.index

        sys = site["sys"]
        self.axis_tilt_resolved = (
            self.axis_tilt if self.axis_tilt is not None else sys.get("axis_tilt")
        )
        self.axis_azimuth_resolved = (
            self.axis_azimuth
            if self.axis_azimuth is not None
            else sys.get("axis_azimuth")
        )
        self.gcr_resolved = self.gcr if self.gcr is not None else sys.get("gcr")
        self.cross_axis_tilt_resolved = self.cross_axis_tilt

        reason = _backtracking_geometry_error(
            self.axis_tilt_resolved,
            self.axis_azimuth_resolved,
            self.gcr_resolved,
            self.cross_axis_tilt_resolved,
        )
        if reason is not None:
            warnings.warn(
                f"Backtracking filter geometry invalid: {reason}; "
                "no intervals removed."
            )
            return df.index

        apparent_zenith, solar_azimuth = self._solar_position(capdata, df)

        mask = backtracking_active(
            apparent_zenith,
            solar_azimuth,
            self.axis_tilt_resolved,
            self.axis_azimuth_resolved,
            self.gcr_resolved,
            cross_axis_tilt=self.cross_axis_tilt_resolved,
        )
        keep = mask if self.keep_backtracking else ~mask
        return df.index[keep.to_numpy()]

    def _solar_position(self, capdata, df):
        """Compute (apparent_zenith, solar_azimuth) aligned to ``df.index``.

        Builds a pvlib Location from ``capdata.site['loc']`` at the site's true
        altitude and calls ``get_solarposition``. Timezone handling mirrors
        ``calcparams.apparent_zenith``: a tz-naive index is localized with the
        site tz; results are returned tz-naive and reindexed to ``df.index``.
        """
        from pvlib.location import Location

        loc = capdata.site["loc"]
        location = Location(**loc)
        times = df.index
        if times.tz is None:
            times = times.tz_localize(
                loc["tz"], ambiguous="infer", nonexistent="NaT"
            )
        solpos = location.get_solarposition(times)
        solpos.index = solpos.index.tz_localize(None)
        target = df.index.tz_localize(None) if df.index.tz is not None else df.index
        apparent_zenith = solpos["apparent_zenith"].reindex(target)
        solar_azimuth = solpos["azimuth"].reindex(target)
        return apparent_zenith, solar_azimuth

    def _args_for_repr(self):
        vals = dict(self.param.values())
        for key in ("axis_tilt", "axis_azimuth", "gcr"):
            resolved = getattr(self, f"{key}_resolved", None)
            if resolved is not None:
                vals[key] = resolved
        return vals

    def _explanation_values(self):
        kind = "Non-backtracking" if self.keep_backtracking else "Backtracking-active"
        return {
            "kind": kind,
            "gcr": getattr(self, "gcr_resolved", self.gcr),
            "axis_tilt": getattr(self, "axis_tilt_resolved", self.axis_tilt),
            "axis_azimuth": getattr(self, "axis_azimuth_resolved", self.axis_azimuth),
            "cross_axis_tilt": self.cross_axis_tilt,
        }
```

- [ ] **Step 4: Register the filter**

In `src/captest/filters.py`, add to `FILTER_REGISTRY` (after `"Clearsky": Clearsky,`):

```python
    "Backtracking": Backtracking,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestFilterBacktracking -v`
Expected: PASS (all 12 tests).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/captest/filters.py tests/test_filter_classes.py
uv run ruff format src/captest/filters.py tests/test_filter_classes.py
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add Backtracking filter step class

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `CapData.filter_backtracking` wrapper + pipeline replay

The thin in-place wrapper, plus a full serialize/replay integration test.

**Files:**
- Modify: `src/captest/capdata.py` (import block line 37; wrapper after `filter_clearsky` ~line 2107)
- Test: `tests/test_filter_classes.py` (new class `TestBacktrackingWrapper`)

**Interfaces:**
- Consumes: `Backtracking` (Task 3).
- Produces: `CapData.filter_backtracking(self, axis_tilt=None, axis_azimuth=None, gcr=None, cross_axis_tilt=0, keep_backtracking=False, custom_name=None) -> None` — builds a `Backtracking` step and `run()`s it in place (appends to `self.filters`, updates `data_filtered`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_filter_classes.py` (reuses the `cd_backtrack` fixture from Task 3):

```python
class TestBacktrackingWrapper:
    def test_wrapper_records_step(self, cd_backtrack):
        cd_backtrack.filter_backtracking()
        assert len(cd_backtrack.filters) == 1
        assert isinstance(cd_backtrack.filters[0], Backtracking)

    def test_wrapper_filters_data(self, cd_backtrack):
        n_before = cd_backtrack.data_filtered.shape[0]
        cd_backtrack.filter_backtracking()
        assert cd_backtrack.data_filtered.shape[0] < n_before

    def test_wrapper_custom_name_sets_step_label(self, cd_backtrack):
        cd_backtrack.filter_backtracking(custom_name="no backtrack")
        assert cd_backtrack.filters[-1].custom_name == "no backtrack"

    def test_wrapper_keep_backtracking(self, cd_backtrack):
        cd_default = cd_backtrack
        n_full = cd_default.data.shape[0]
        cd_default.filter_backtracking(keep_backtracking=True)
        # Keeping only backtracking removes the true-tracking midday rows.
        assert cd_default.data_filtered.shape[0] < n_full

    def test_serializes_and_replays(self, cd_backtrack):
        cd_backtrack.filter_backtracking()
        expected_index = list(cd_backtrack.data_filtered.index)
        config = cd_backtrack.filters_to_config()
        assert config[0]["type"] == "Backtracking"

        fresh = CapData("fresh")
        fresh.data = cd_backtrack.data.copy()
        fresh.site = cd_backtrack.site
        fresh.regression_cols = dict(cd_backtrack.regression_cols)
        fresh.run_pipeline(config)
        # None geometry re-resolves from site on replay -> identical result.
        assert list(fresh.data_filtered.index) == expected_index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingWrapper -v`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute 'filter_backtracking'`.

- [ ] **Step 3: Add the import**

In `src/captest/capdata.py`, add `Backtracking` to the `from captest.filters import (` block (keep alphabetical-ish grouping; place after `BaseSummaryStep` / before `BooleanFlag`):

```python
from captest.filters import (
    AbsDiffPrev,
    BaseSummaryStep,
    Backtracking,
    BooleanFlag,
    Clearsky,
    # ... rest unchanged ...
)
```

- [ ] **Step 4: Implement the wrapper**

In `src/captest/capdata.py`, insert immediately after the end of `filter_clearsky` (before `def filter_missing`):

```python
    def filter_backtracking(
        self,
        axis_tilt=None,
        axis_azimuth=None,
        gcr=None,
        cross_axis_tilt=0,
        keep_backtracking=False,
        custom_name=None,
    ):
        """Remove intervals where single-axis-tracker backtracking is active.

        Backtracking activity is decided per interval from solar geometry using
        a transcription of pvlib's ``tracking.singleaxis`` test. Solar position
        is computed from the site location; tracker geometry defaults to the
        site system definition (``site['sys']``) and may be overridden per
        argument. Degrades to a warn-and-no-op when site/geometry/pvlib are
        unavailable or the resolved geometry is invalid.

        Parameters
        ----------
        axis_tilt : float, default None
            Tracker axis tilt (deg). Uses site['sys']['axis_tilt'] when None.
        axis_azimuth : float, default None
            Tracker axis azimuth (deg). Uses site['sys']['axis_azimuth'] when
            None.
        gcr : float, default None
            Ground coverage ratio. Uses site['sys']['gcr'] when None.
        cross_axis_tilt : float, default 0
            Cross-axis tilt (deg) for sloped terrain. Must be in (-90, 90).
        keep_backtracking : bool, default False
            If True, keep only backtracking intervals and remove true-tracking
            ones.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Backtracking(
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            gcr=gcr,
            cross_axis_tilt=cross_axis_tilt,
            keep_backtracking=keep_backtracking,
            custom_name=custom_name,
        )
        flt.run(self)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --python 3.12 pytest tests/test_filter_classes.py::TestBacktrackingWrapper -v`
Expected: PASS (all 5 tests).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/captest/capdata.py tests/test_filter_classes.py
uv run ruff format src/captest/capdata.py tests/test_filter_classes.py
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData.filter_backtracking wrapper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full-suite verification

Confirm nothing regressed and the new code integrates cleanly.

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run --python 3.12 pytest tests/ -q`
Expected: PASS — the prior baseline of `953 passed` plus the new tests (32 added across Tasks 1–4), no failures. Warnings unrelated to backtracking are pre-existing.

- [ ] **Step 2: Lint and format the whole tree**

```bash
uv run ruff check src/captest/ tests/
uv run ruff format --check src/captest/ tests/
```
Expected: "All checks passed!" and no files needing reformat.

- [ ] **Step 3: Confirm clean tree (all work committed)**

Run: `git status --porcelain`
Expected: empty output.

---

## Notes for the implementer

- **Docs are out of scope for this plan.** A separate follow-up (the `docs-update` skill) will add the changelog entry, the `filter_backtracking` API-doc reference, and any user-guide mention. Do not block on docs here.
- **`cd.site` shape** matches `tests/conftest.py:363` and `clearsky.py` — `{"loc": {latitude, longitude, altitude, tz}, "sys": {...}}`. Tracker `sys` uses `axis_tilt`/`axis_azimuth`/`gcr` (plus `max_angle`/`backtrack`/`albedo`); a fixed-tilt `sys` (`surface_tilt`/`surface_azimuth`) has no `axis_*`/`gcr`, so the filter warn-and-no-ops on it — expected.
- **Why `max_angle=90` in the oracle test:** it prevents pvlib from clipping the true-tracking angle, so the on/off `tracker_theta` difference isolates the backtracking decision the geometry predicate computes.
- **Index alignment:** `_execute` returns `df.index[keep.to_numpy()]`; `keep` is a boolean Series aligned to `df.index` (reindexed inside `_solar_position`), and `.to_numpy()` avoids any label-vs-position ambiguity when indexing `df.index`.
