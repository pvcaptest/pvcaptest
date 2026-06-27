# CapTest as the single owner of test reporting conditions

- **Status:** Draft for review
- **Date:** 2026-06-27
- **Branch:** filters-refactor
- **Supersedes:** the `CapData.rc_source_resolved` reference mechanism added earlier
  on this branch (commit `71c462e`).

## 1. Background and motivation

A capacity test conceptually has **exactly one set of reporting conditions
(RCs)**. Today the codebase does not model that directly:

- `CapData` owns its own `rc` DataFrame, computed by `CapData.rep_cond()` (which
  appends a `RepCond` step to the filter pipeline).
- `CapTest.rc_source` (a `param.Selector` of `{"meas", "sim"}`) tells
  `captest_results` *which* CapData's `rc` to predict at.
- `CapTest.rc_source_resolved` (a `CapData` → `CapData` reference wired in
  `setup()`) tells the `Irradiance` filter which CapData's `rc` to resolve
  `ref_val="rep_irr"` against.

So "the test's reporting conditions" is spread across three related concepts
(`cd.rc`, `rc_source`, `rc_source_resolved`), both `meas.rc` and `sim.rc` can
exist simultaneously, and it is non-obvious that one of them is ignored for
filtering. There is also no way to **manually supply** an RC set that does not
come from either dataset — needed for sensitivity analysis and for checking
results against RCs a reviewing party calculated.

## 2. Goals

1. `CapTest.rc` is **the** test reporting conditions: one DataFrame, accessible
   in one place.
2. The RC is sourced from `meas`, from `sim`, **or** set manually — tracked by
   `CapTest.rc_source ∈ {"meas", "sim", "manual"}`.
3. Warn when RC is replaced **from a different source** (e.g. recomputing over a
   manual override, or switching meas→sim).
4. `filter_irr(ref_val="rep_irr")` inside a test resolves from `CapTest.rc`.
5. Net **simplification**: remove `rc_source_resolved`; `captest_results` and
   irradiance filtering read the same `CapTest.rc`.

## 3. Non-goals

- Standalone `CapData` behavior is **unchanged**: `cd.rep_cond()` still populates
  `cd.rc`, and `cd.filter_irr(ref_val="rep_irr")` outside a test still resolves
  from `cd.rc`. (Confirmed hard constraint.)
- No changes to `rep_cond_freq` / grouped `predict_capacities` RC handling.
- No deprecation shims (pre-1.0 refactor branch; clean break is acceptable).

## 4. Design

### 4.1 Data model

On `CapTest`:

- `rc` — `param.Parameter(default=None)`, the one-row test RC DataFrame (or None).
- `rc_source` — extend the existing `param.Selector` objects to
  `["meas", "sim", "manual"]`, default `"meas"`. It now records the **provenance
  of the current `rc`** (and serves as the default `which` for `rep_cond`).

On `CapData`:

- Replace `rc_source_resolved` (CapData→CapData reference) with `_captest`
  (CapData→CapTest reference, default `None`), set in `CapTest.setup()`. This is a
  runtime object reference only; `capdata.py` still never imports `captest`, so
  the module import layering is preserved.
- `cd.rc` is retained for standalone use, unchanged.

### 4.2 Reporting-conditions resolution (`CapData.rep_irr`)

```python
@property
def rep_irr(self):
    rc = self._captest.rc if self._captest is not None else self.rc
    if rc is None:
        raise ValueError(
            # in-test message points to ct.rep_cond(which); standalone to rep_cond()
            ...
        )
    if "poa" not in rc.columns:
        raise ValueError(...)
    return float(rc["poa"].iloc[0])
```

In a test, resolution reads **only** `CapTest.rc` (no silent fallback to
`cd.rc`). `Irradiance._execute` is unchanged — it already delegates to
`capdata.rep_irr`.

### 4.3 Computing RC inside a test (`CapTest.rep_cond`)

`ct.rep_cond(which="meas", **overrides)` is the single in-test entry point
(point 3, confirmed). It keeps its current behavior — delegating to
`cd.rep_cond(...)`, which appends the `RepCond` step to that CapData's pipeline
and populates `cd.rc` — and then:

1. Sets `self.rc = cd.rc` (the just-computed value).
2. Sets `self.rc_source = which`.
3. Emits the overwrite warning per §4.5 **before** overwriting.

`which` defaults to the current `self.rc_source` when it is `"meas"`/`"sim"`,
else `"meas"`.

**Drift decision (point 3):** calling `cd.rep_cond()` directly on a test's
CapData updates `cd.rc` but **not** `CapTest.rc`. This is documented and
discouraged in favor of `ct.rep_cond(which)`. Because `rep_irr` reads
`CapTest.rc`, a stray `cd.rep_cond()` cannot silently change filtering behavior.

### 4.4 Manual override

`ct.set_reporting_conditions(rc)`:

- Accepts a one-row DataFrame (or a mapping coerced to one) with at least a
  `"poa"` column.
- Sets `self.rc = rc`, `self.rc_source = "manual"`, after the §4.5 warning check.

A `CapTest.rc` property setter routes assignment (`ct.rc = df`) through the same
path so both spellings behave identically.

### 4.5 Overwrite warning

When setting RC (via `rep_cond` or `set_reporting_conditions`): if `self.rc is
not None` **and** the new source differs from the current `self.rc_source`, warn:

> "Overwriting test reporting conditions sourced from '{old}' with conditions
> from '{new}'."

Re-running the **same** source (e.g. recomputing meas RC after changing filters)
does not warn.

### 4.6 `captest_results`

Replace the `rc_source`-based pick of `meas.rc`/`sim.rc` with `self.rc`. If
`self.rc is None`, raise a clear error directing the user to call
`ct.rep_cond(which)` or `ct.set_reporting_conditions(...)`. The printed
provenance line uses `self.rc_source`.

### 4.7 Serialization

Two cases, distinguished by `rc_source`:

- **Computed (`"meas"`/`"sim"`)** — RC *values* are not serialized; they are
  recomputed by replaying the source pipeline's `RepCond` step on `from_yaml`.
  The existing "drop `overrides.rep_conditions` when a `RepCond` step is present"
  rule is unchanged. After pipeline replay, `from_yaml` sets `ct.rc` from the
  `rc_source` CapData's recomputed `rc`.
- **Manual (`"manual"`)** — RC values *are* data, not config, so they cannot be
  recomputed. Serialize the one-row RC as a yaml-safe mapping under a new
  optional key `reporting_conditions_values` in the captest sub-mapping (numpy
  scalars coerced via `util.to_native`). On load, reconstruct the DataFrame, set
  `ct.rc`, and set `rc_source="manual"` **after** pipeline replay so the manual
  override wins over anything a `RepCond` step produced.

`rc_source` continues to serialize via the existing `scalar_names` list.

### 4.8 `setup()` changes

- Remove the `rc_source_resolved` wiring.
- Set `self.meas._captest = self` and `self.sim._captest = self`.
- `setup()` does **not** compute or clear `ct.rc` (RC lifecycle is driven by
  `rep_cond`/`set_reporting_conditions`). Re-running `setup()` leaves an existing
  `ct.rc` intact unless the user recomputes.

## 5. Components and interfaces (summary)

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `CapTest.rc` / `rc_source` | Hold the one test RC + provenance | — |
| `CapTest.rep_cond(which)` | Compute via a CapData, store onto `ct.rc` | `CapData.rep_cond` |
| `CapTest.set_reporting_conditions(rc)` | Manual override | — |
| `CapData.rep_irr` | Resolve reporting POA (test → `ct.rc`, else `self.rc`) | `_captest` |
| `Irradiance._execute` | `ref_val="rep_irr"` → `capdata.rep_irr` | unchanged |
| `CapTest.captest_results` | Predict at `ct.rc` | `ct.rc` |
| `to_yaml` / `from_yaml` | Persist/restore `rc_source` (+ manual values) | `util.to_native` |

## 6. Error handling and warnings

- `rep_irr` with no resolvable RC → `ValueError` naming the in-test vs standalone
  remedy.
- `captest_results` with `ct.rc is None` → `ValueError` directing to `rep_cond`.
- Source-change overwrite → `UserWarning` (§4.5).
- `set_reporting_conditions` with a missing `"poa"` column → `ValueError`.

## 7. Migration impact

- Internal-only removals (added this session, unreleased): `rc_source_resolved`
  attribute, its `setup()` wiring, and the `rep_irr` branch that read it.
- Behavior change: `captest_results` now reads `ct.rc`. Existing flows that call
  `cd.rep_cond()` (e.g. via `run_test`) and then `captest_results` must switch to
  `ct.rep_cond(which)`. The example notebook is being reworked by the user
  separately.
- Tests added this session (`TestRepIrrCrossInstance`, the `rc_source_resolved`
  wiring asserts in `TestSetup`) are updated to the `_captest` / `ct.rc` model.

## 8. Testing plan

1. `CapTest.rc`/`rc_source` defaults; `_captest` wired on both CapData by
   `setup()`.
2. `ct.rep_cond("meas")` sets `ct.rc` (== meas computation) and
   `rc_source="meas"`; same for `"sim"`.
3. `set_reporting_conditions(df)` sets `ct.rc` and `rc_source="manual"`.
4. Overwrite warning fires on source change; no warning on same-source recompute.
5. `rep_irr`: in-test reads `ct.rc`; standalone reads `cd.rc`; `ValueError` when
   `ct.rc` is None in a test.
6. `filter_irr(ref_val="rep_irr")` on sim uses `ct.rc` set from meas, without a
   manual value pass (the original bug's real-world scenario).
7. `captest_results` predicts at `ct.rc`; `ValueError` when None.
8. Serialization round-trip: computed source recomputes `ct.rc` on load; manual
   override serializes values and restores `ct.rc` + `rc_source="manual"`.
9. Standalone `CapData` regression tests remain green (no behavior change).

## 9. Open questions

None outstanding. Resolved during brainstorming:

- Standalone CapData keeps owning `cd.rc` (yes).
- `ct.rep_cond(which)` is the single in-test entry point; `cd.rep_cond()` does
  not sync to `ct.rc` (point 3).
- Manual override is a first-class source (`rc_source="manual"`), serialized by
  value.
