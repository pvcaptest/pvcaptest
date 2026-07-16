# Design: Tracker Backtracking Filter

**Date:** 2026-07-16
**Branch:** `backtracking-filter`
**Package:** `captest`

## Problem

Single-axis trackers (SATs) *backtrack* near sunrise and sunset: they tilt back
from true-tracking to avoid row-to-row shading. During these intervals a
tracker's measured position legitimately diverges from a curve that assumes
true-tracking, and the array's response differs from mid-day behavior. For
capacity testing and tracker-availability analysis, users need a way to
**exclude (or isolate) backtracking intervals** from a `CapData` dataset.

`captest` has no filter for this today. Rather than a bespoke callable, this belongs as a
first-class `captest` filter so it serializes, replays, and appears in the
filtering summary like every other filter.

## Goal

Add a new first-class filter step, `Backtracking`, in `filters.py`, plus a thin
`CapData.filter_backtracking(...)` wrapper in `capdata.py`, following the exact
pattern used by every existing filter (`Clearsky`, `Sensors`, `Regression`).
The filter decides backtracking activity per interval using a **direct
transcription of pvlib's own `tracking.singleaxis` backtracking test**.

## Backtracking predicate

pvlib's `tracking.singleaxis` decides backtracking per interval from solar
geometry (Anderson & Mikofski 2020, *Slope-Aware Backtracking for Single-Axis
Trackers*, Eqs. 14–16). The filter computes the same condition:

```python
from pvlib import shading
from pvlib.tools import cosd

omega_ideal = shading.projected_solar_zenith_angle(
    solar_zenith=apparent_zenith,
    solar_azimuth=solar_azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
)
axes_distance = 1 / (gcr * cosd(cross_axis_tilt))
backtracking_active = (
    (apparent_zenith <= 90)
    & (abs(axes_distance * cosd(omega_ideal - cross_axis_tilt)) < 1)
)
```

`backtracking_active` is `True` exactly where the sun is above the horizon
**and** row-to-row shade would occur under true-tracking (the same `temp < 1`
test `singleaxis` uses to switch from true-tracking to backtracking).

## Architecture

Three pieces, each mirroring an established `captest` pattern:

1. **Module-level helper** `backtracking_active(...)` in `filters.py` — the pure
   geometry test. Mirrors how `Regression._execute` calls module-level
   `fit_model(...)` and `Sensors._execute` calls `sensor_filter(...)`.
2. **`Backtracking` filter class** in `filters.py` — a `BaseFilter` subclass
   that resolves geometry, computes solar position, calls the helper, and
   returns the kept index.
3. **`CapData.filter_backtracking(...)` wrapper** in `capdata.py` — thin,
   in-place, instantiates the step and `run()`s it.

Plus registration in `FILTER_REGISTRY`. Config serialization is inherited (all
params are plain scalars/booleans) — no `to_config`/`from_config` override.

### 1. Geometry validation + predicate helper (`filters.py`)

The `axes_distance = 1 / (gcr * cosd(cross_axis_tilt))` term is undefined or
meaningless for several geometry values, none of which the raw predicate
handles: `gcr == 0` raises `ZeroDivisionError` (Python float) or yields a silent
`inf` (numpy); `gcr < 0`, `NaN`, or `inf` produce invalid classifications that
pass through as "removed nothing" rather than the documented graceful behavior;
and a `cross_axis_tilt` at/beyond ±90° drives the denominator to (near-)zero.
Note that a `cosd(cross_axis_tilt) == 0` guard is **insufficient**: pvlib's
`cosd(90)` returns `≈6.1e-17`, not exactly `0`, so ±90° slips past an equality
check and yields an absurd ~1.6e16 `axes_distance`. Non-numeric site metadata (a
stray string, `pd.NA`, etc.) is a further hazard: passing it to `math.isfinite`
raises `TypeError`, which would escape the warn-and-no-op contract — so values
are checked by *type* before any numeric test. Geometry is therefore validated
**before** any calculation via a shared reason function.

```python
import math     # added to the stdlib import block
import numbers  # added to the stdlib import block

def _backtracking_geometry_error(axis_tilt, axis_azimuth, gcr, cross_axis_tilt):
    """Return a human-readable reason string if the resolved geometry is invalid
    for the backtracking predicate, else None.

    Requires every value to be a finite real number, ``gcr > 0`` (so the
    axes-distance denominator is nonzero and positive), and
    ``-90 < cross_axis_tilt < 90`` (so ``cosd(cross_axis_tilt)`` is finite and
    strictly positive — a physical cross-axis slope, and a robust replacement
    for a ``cosd(...) == 0`` check that pvlib's non-exact ``cosd(90) ≈ 6e-17``
    would defeat).

    The ``isinstance(value, numbers.Real)`` guard runs **before** any numeric
    test so non-numeric site metadata (a string, ``pd.NA``, an arbitrary object)
    is rejected with a reason rather than allowed to raise ``TypeError`` from
    ``math.isfinite`` — keeping the caller's warn-and-no-op contract intact.
    ``bool`` is excluded explicitly (it is a ``numbers.Real`` subtype) so a stray
    ``True``/``False`` is treated as invalid instead of silently coerced to 1/0.
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
                f"{name} could not be resolved (pass it explicitly or set it in "
                "site['sys'])"
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


def backtracking_active(apparent_zenith, solar_azimuth, axis_tilt,
                        axis_azimuth, gcr, cross_axis_tilt=0):
    """Return a boolean Series: True where single-axis-tracker backtracking is
    geometrically active (sun above the horizon AND true-tracking would cause
    row-to-row shade).

    Direct transcription of pvlib's tracking.singleaxis backtracking test
    (Anderson & Mikofski 2020). Raises ``ValueError`` on invalid geometry (see
    ``_backtracking_geometry_error``) so direct callers get a clear error rather
    than a division-by-zero or a silently wrong mask.
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

`shading`/`cosd` are imported inside the helper. The module already guards the
top-level `pvlib` import (it does this for `detect_clearsky`); the filter's
`_execute` warns-and-no-ops when pvlib is unavailable so the helper is only
reached when pvlib is present. `_backtracking_geometry_error` uses only
`math`/comparisons (no pvlib), so it is safe to call during the filter's no-op
guard regardless of pvlib availability.

The reason function is the **single source of truth** for geometry validity,
called from two places with different failure modes: the helper *raises* (safe
for direct callers and reuse), while the filter's `_execute`
*warns and no-ops* (see below) — so the filter never actually triggers the
helper's raise, but the helper remains independently correct.

### 2. `Backtracking` filter class (`filters.py`)

A `BaseFilter` subclass declaring its config as `param` parameters, following
`Clearsky`/`RollingStd` conventions:

```python
class Backtracking(BaseFilter):
    """Remove intervals where single-axis-tracker backtracking is active.

    Transcribes pvlib's own singleaxis backtracking test (Anderson & Mikofski
    2020): an interval is backtracking-active when the sun is above the horizon
    and row-to-row shade would occur under true-tracking. Solar position is
    computed from ``capdata.site['loc']`` via pvlib; tracker geometry defaults
    to ``capdata.site['sys']`` and is overridable per parameter.

    By default keeps true-tracking intervals and removes backtracking-active
    ones; set ``keep_backtracking=True`` to invert (keep only backtracking).
    """

    axis_tilt = param.Number(
        default=None, allow_None=True,
        doc="Tracker axis tilt (deg). Resolved from site['sys']['axis_tilt'] "
            "when None.")
    axis_azimuth = param.Number(
        default=None, allow_None=True,
        doc="Tracker axis azimuth (deg). Resolved from "
            "site['sys']['axis_azimuth'] when None.")
    gcr = param.Number(
        default=None, allow_None=True,
        doc="Ground coverage ratio. Resolved from site['sys']['gcr'] when None.")
    cross_axis_tilt = param.Number(
        default=0,
        doc="Cross-axis tilt (deg) for sloped terrain. Defaults to 0 (flat), "
            "matching pvlib's default.")
    keep_backtracking = param.Boolean(
        default=False,
        doc="Keep true-tracking intervals (False) or keep backtracking "
            "intervals (True).")
```

**`_execute` logic:**

1. **pvlib / site guard.** If pvlib is unavailable, or `capdata.site` is
   absent/None (or has no `sys`/`loc` sub-dict) → `warnings.warn(...)` and return
   `capdata.data_filtered.index` unchanged. This matches `Clearsky`'s behavior
   when `ghi_mod_csky` is missing: the filter still appears in the summary but
   removes nothing.
2. **Resolve geometry.** For each of `axis_tilt`/`axis_azimuth`/`gcr`, use the
   param value if not None, else `capdata.site['sys'].get(<key>)` (missing key →
   `None`). `cross_axis_tilt` uses the param directly (default 0). Store resolved
   values as **runtime attributes** (`self.axis_tilt_resolved`, etc.) for
   `args_repr`/`explanation` — never as params, so the serialized config
   preserves the user's intent (`None` = "resolve from site"). This mirrors
   `Irradiance` keeping `ref_val='rep_irr'` in config while storing
   `ref_val_resolved` at runtime.
3. **Validate geometry (no-op guard).** Call
   `_backtracking_geometry_error(axis_tilt_resolved, axis_azimuth_resolved,
   gcr_resolved, cross_axis_tilt)`. If it returns a reason (unresolved `None`,
   non-finite value, `gcr <= 0`, or out-of-range `cross_axis_tilt`), include it
   in `warnings.warn(...)` and return `capdata.data_filtered.index` unchanged.
   This is the single guard covering both "geometry could not be resolved" and
   "geometry is numerically invalid" — the filter never reaches the helper's
   `ValueError`, so `filter_backtracking` degrades gracefully in every case
   while the standalone helper still raises for direct callers.
4. **Compute solar position.** Build a `pvlib.location.Location` from
   `capdata.site['loc']` at the site's **true altitude** (no altitude override —
   altitude=0 in `calcparams.apparent_zenith` is an airmass concern, irrelevant
   to backtracking geometry). Call `location.get_solarposition(times)` and pull
   both `apparent_zenith` and `azimuth`. Timezone handling mirrors
   `calcparams.apparent_zenith`: tz-localize a tz-naive index using
   `site['loc']['tz']`, then align results back to `data_filtered.index`.
5. **Apply predicate.** Call the module-level `backtracking_active(...)` helper,
   then return `data_filtered.index[~mask]` (default) or `data_filtered.index[mask]`
   when `keep_backtracking=True`. (Geometry is already validated in step 3, so
   this call will not hit the helper's `ValueError`.)

**Display.** `args_repr` renders the resolved geometry values (via
`_args_for_repr` override, like `Irradiance`/`Sensors`); `explanation`
(`_explanation_template` + `_explanation_values`) states the resolved geometry
and which side was removed:

> "Backtracking-active intervals (gcr={gcr}, axis_tilt={axis_tilt},
> axis_azimuth={axis_azimuth}, cross_axis_tilt={cross_axis_tilt}) were removed."

(inverted wording when `keep_backtracking=True`).

### 3. `CapData.filter_backtracking(...)` wrapper (`capdata.py`)

Thin, in-place, following every other `filter_*`:

```python
def filter_backtracking(self, axis_tilt=None, axis_azimuth=None, gcr=None,
                        cross_axis_tilt=0, keep_backtracking=False,
                        custom_name=None):
    """Remove intervals where single-axis-tracker backtracking is active.

    Backtracking activity is decided per interval from solar geometry using a
    transcription of pvlib's tracking.singleaxis test. Solar position is
    computed from the site location; tracker geometry defaults to the site
    system definition and may be overridden per argument.

    Parameters
    ----------
    axis_tilt : float, default None
        Tracker axis tilt (deg). Uses site['sys']['axis_tilt'] when None.
    axis_azimuth : float, default None
        Tracker axis azimuth (deg). Uses site['sys']['axis_azimuth'] when None.
    gcr : float, default None
        Ground coverage ratio. Uses site['sys']['gcr'] when None.
    cross_axis_tilt : float, default 0
        Cross-axis tilt (deg) for sloped terrain.
    keep_backtracking : bool, default False
        If True, keep only backtracking intervals and remove true-tracking ones.
    custom_name : str, default None
        Optional display label for the recorded filter step.
    """
    flt = Backtracking(
        axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, gcr=gcr,
        cross_axis_tilt=cross_axis_tilt, keep_backtracking=keep_backtracking,
        custom_name=custom_name,
    )
    flt.run(self)
```

### 4. Registration & config IO

- Add `"Backtracking": Backtracking` to `FILTER_REGISTRY`.
- All five params are plain numbers/booleans, so the inherited
  `BaseSummaryStep.to_config`/`from_config` serialize and round-trip them with
  no override. The step serializes to
  `{type: Backtracking, axis_tilt: ..., gcr: ..., cross_axis_tilt: ...,
  keep_backtracking: ...}` and replays through `run_pipeline` /
  `step_from_config` for free.
- Because geometry-from-site is resolved at run time (runtime attrs, not
  params), a serialized `None` geometry re-resolves from `site` on replay.

## Data source: `cd.site`

The tracker geometry and location the predicate needs already live on
`cd.site`, set by `io.load_data(site=...)`:

- `site['sys']` — tracker keys `axis_tilt`, `axis_azimuth`, `gcr` (also
  `max_angle`, `backtrack`, `albedo`).
- `site['loc']` — `latitude`, `longitude`, `altitude`, `tz`.

Solar position is not currently stored as columns on `cd.data`; the filter
computes it on the fly via pvlib, the same "call pvlib directly" approach
`Clearsky` takes with `detect_clearsky` and `calcparams.apparent_zenith` takes
with `Location.get_solarposition`. `calcparams.apparent_zenith` is **not** a
drop-in helper here: it returns only the zenith column, NaN-masks night, and
forces altitude 0 — whereas this filter also needs `solar_azimuth`, needs
unmasked zenith for the `<= 90` test, and wants true altitude.

## Error handling & edge cases

- **Missing `site` / fixed-tilt system / unresolvable geometry** → warn and
  no-op (return index unchanged). Backtracking is meaningless without tracker
  geometry, but consistent with `Clearsky`, we degrade gracefully rather than
  raise, so a pipeline replayed against data lacking site metadata does not
  crash mid-chain. "Unresolvable" includes a required geometry key absent from
  both the filter params and `site['sys']`.
- **Invalid geometry values** — a non-real value (a string, `pd.NA`, a boolean,
  or any non-numeric object, e.g. from malformed site metadata), any non-finite
  value (`NaN`/`inf`), `gcr <= 0` (including the division-by-zero case
  `gcr == 0`), or `cross_axis_tilt` outside `(-90, 90)` → warn (with the
  specific reason) and no-op. Validated by `_backtracking_geometry_error`
  **before** any division, `math.isfinite`, or pvlib call — the type check runs
  first so non-numeric input never raises `TypeError` — so the filter never
  raises and never produces a silently-wrong mask. The standalone
  `backtracking_active` helper *raises* `ValueError` on the same conditions for
  direct callers.
- **pvlib unavailable** → warn and no-op.
- **`cross_axis_tilt`** defaults to 0 (flat terrain, pvlib's own default) and is
  overridable via kwarg, constrained to `(-90, 90)` by validation. Deriving it
  from site slope (via `pvlib.tracking.calc_cross_axis_tilt`) is out of scope —
  `load_data` does not currently capture the slope keys that would require.

## Testing

pytest, Arrange-Act-Assert, per repo convention (`just test`). Tests in
`tests/test_filter_classes.py` plus a `CapData` wrapper test. pvlib is already a
test dependency.

**Predicate helper (`backtracking_active`) — the correctness core:**

- **Numerical agreement with pvlib.** Over a synthetic clear day, assert the
  helper's mask matches `pvlib.tracking.singleaxis(backtrack=True vs False)` at
  every sun-up interval. This is the key test proving the transcription is faithful.
- Sun-down intervals (`apparent_zenith > 90`) → `False`.
- A `cross_axis_tilt` override changes the result as expected on a sloped case.

**Geometry validation (`_backtracking_geometry_error` / helper raise):**

- `gcr == 0` (division by zero), `gcr < 0`, and non-finite `gcr` (`NaN`/`inf`)
  each produce a reason string; `backtracking_active` **raises `ValueError`**
  rather than dividing by zero or returning a wrong mask.
- Non-finite `axis_tilt`/`axis_azimuth`/`cross_axis_tilt` each produce a reason.
- `cross_axis_tilt` at ±90 and beyond is rejected (guards the near-zero
  denominator that a `cosd == 0` check would miss, since `cosd(90) ≈ 6e-17`);
  values inside `(-90, 90)` pass.
- `None` (unresolved) values produce a reason naming the parameter.
- Non-real values — a string (e.g. `gcr="0.3"`), `pd.NA`, and a boolean
  (`gcr=True`) — each produce a reason **without raising `TypeError`** from
  `math.isfinite`; the helper then raises `ValueError`, confirming the type
  guard precedes the numeric test.
- A valid geometry set returns `None` (no error).

**`Backtracking` filter `_execute`:**

- Default removes backtracking-active intervals and keeps true-tracking;
  `keep_backtracking=True` inverts.
- Geometry resolves from `capdata.site['sys']` when params are None; explicit
  params override site.
- Warn-and-no-op when `site` is absent, when a required geometry value cannot be
  resolved, and when pvlib is unavailable — asserting the index is returned
  unchanged and a warning is emitted.
- Warn-and-no-op (not raise) when resolved geometry is invalid — e.g. a `site`
  with `gcr=0`, `gcr` absent, a non-numeric `gcr` (a string or `pd.NA`), or an
  explicit `cross_axis_tilt=90` — asserting the index is unchanged, a warning
  naming the reason is emitted, and the step still appears in the summary. This
  confirms the filter degrades gracefully (including on `TypeError`-prone
  non-numeric metadata) where the standalone helper would raise.

**Wrapper + integration:**

- `cd.filter_backtracking(...)` appends exactly one `Backtracking` step and
  updates `data_filtered`.
- Config round-trip: `to_config()` → `step_from_config()` reproduces params; a
  full `filters_to_config()` / `run_pipeline(config)` replay reproduces the
  filtered result, with `None` geometry re-resolving from site.
- `args_repr` / `explanation` render resolved values and the correct removed
  side.

**Fixtures:** a small tracker `CapData` with a `site` dict (`loc` + tracker
`sys`) and a clear-day POA/index.

## Out of scope

- Deriving `cross_axis_tilt` from site terrain slope.
- Persisting solar-position columns onto `cd.data` for reuse by other steps.
