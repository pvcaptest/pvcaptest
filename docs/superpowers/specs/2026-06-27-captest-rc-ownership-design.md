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

- `rc` — a **Python property** backed by a private `self._rc` (one-row test RC
  DataFrame, or None). It is intentionally *not* a `param`, so the setter can run
  validation and provenance logic (§4.4). The getter returns `self._rc`.
- `rc_source` — extend the existing `param.Selector` objects to
  `["meas", "sim", "manual"]`, default `"meas"`. It records the **provenance
  of the current `rc`** (and serves as the default `which` for `rep_cond`). It is
  set by internal code (`_set_rc`), not assigned directly by users in normal use.

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

1. Emits the overwrite warning per §4.5 **before** overwriting.
2. Records the result via `self._set_rc(cd.rc, which)` (a **copy**, per §4.3),
   which sets both `self._rc` and `self.rc_source = which`.

`which` defaults to the current `self.rc_source` when it is `"meas"`/`"sim"`,
else `"meas"`.

**Drift decision (point 3).** Inside a test there are two RC stores: each
CapData's own `cd.rc` and the test-level `CapTest.rc`. `ct.rep_cond(which)` is the
**only** supported way to populate the test RC from data. `cd.rep_cond()` is not
removed — it still works on a CapData that happens to belong to a test (it
updates that CapData's own `cd.rc` and appends the `RepCond` step to its
pipeline) — but it deliberately does **not** propagate to `CapTest.rc`.

Rationale:

1. **Unambiguous ownership.** The test RC changes only when a `CapTest` method is
   called, so "what are the test's reporting conditions?" has one answer with one
   place to look.
2. **No hidden cross-dataset coupling.** Filtering on either dataset resolves
   `ref_val="rep_irr"` from `CapTest.rc` (§4.2). If a bare `cd.rep_cond()` on
   `meas` silently became the test RC, it would change `sim`'s filtering behavior
   from across the pair — exactly the kind of action-at-a-distance this design
   removes.
3. **Drift fails loudly, not silently.** Because in-test resolution reads only
   `CapTest.rc` (never `cd.rc`), a user who calls `cd.rep_cond()` but never
   `ct.rep_cond(...)` will hit the "`CapTest.rc` is None" error from `rep_irr` /
   `captest_results` (§6), which points them at `ct.rep_cond(which)`. They cannot
   accidentally filter or compute results against a stale or mismatched per-
   dataset value.

Consistency note: `ct.rep_cond(which)` calls `cd.rep_cond` internally, so
immediately afterward `CapTest.rc` equals a **copy** of that CapData's `cd.rc`
(copied to avoid aliasing). A *subsequent* direct `cd.rep_cond()` may then
diverge `cd.rc` from `CapTest.rc`; that divergence is intentional and affects
only standalone-style reads of `cd.rc`, never in-test behavior.

### 4.4 Manual override

The **only** public way to set RCs manually is assigning the `CapTest.rc`
property (one spelling, by design):

```python
ct.rc = df   # df: one-row DataFrame, or a mapping coerced to one row
```

The setter:

1. **Requires `setup()`** (`_require_setup()`) — needed to know the regression
   formula.
2. **Validates RHS coverage.** The required variables are
   `util.parse_regression_formula(self.meas.regression_formula)[1]` — the same
   rhs variable set the computed path aggregates in `_calc_rep_cond`
   (e.g. `poa, t_amb, w_vel`). `meas` and `sim` share the formula after
   `setup()`; if they differ, raise pointing to the mismatch. If `df` is missing
   any required variable, raise `ValueError` listing the missing names. (Extra
   columns are allowed and preserved.)
3. Coerces to a one-row DataFrame, applies the §4.5 overwrite warning, then
   records the override via `self._set_rc(df, "manual")`.

**Internal vs public path.** Because the public setter always means "manual", the
computed-RC assignments from `rep_cond` (§4.3) and `from_yaml` (§4.7) must not go
through it. They call the private helper `self._set_rc(df, source)`, which sets
`self._rc` and `self.rc_source` and records the true `source`. The public
property setter is exactly that helper with `source="manual"`, preceded by the
validation above. This preserves a single public spelling (`ct.rc = df`) while
keeping provenance correct for internally-computed RCs.

### 4.5 Overwrite warning

When setting RC (via `rep_cond` or the `rc` setter): if `self.rc is
not None` **and** the new source differs from the current `self.rc_source`, warn:

> "Overwriting test reporting conditions sourced from '{old}' with conditions
> from '{new}'."

Re-running the **same** source (e.g. recomputing meas RC after changing filters)
does not warn.

### 4.6 `captest_results`

Replace the `rc_source`-based pick of `meas.rc`/`sim.rc` with `self.rc`. If
`self.rc is None`, raise a clear error directing the user to call
`ct.rep_cond(which)` or assign `ct.rc = df`. The printed provenance line uses
`self.rc_source`.

### 4.7 Serialization

Two cases, distinguished by `rc_source`:

- **Computed (`"meas"`/`"sim"`)** — RC *values* are not serialized; they are
  recomputed by replaying the source pipeline's `RepCond` step on `from_yaml`.
  The existing "drop `overrides.rep_conditions` when a `RepCond` step is present"
  rule is unchanged. After pipeline replay, `from_yaml` calls
  `_set_rc(<rc_source cd>.rc, rc_source)` to populate `ct.rc` from the recomputed
  per-dataset `rc`.
- **Manual (`"manual"`)** — RC values *are* data, not config, so they cannot be
  recomputed. Serialize the one-row RC as a yaml-safe mapping under a new
  optional key `reporting_conditions_values` in the captest sub-mapping (numpy
  scalars coerced via `util.to_native`). On load, reconstruct the DataFrame and
  call `_set_rc(df, "manual")` **after** pipeline replay so the manual override
  wins over anything a `RepCond` step produced.

`rc_source` continues to serialize via the existing `scalar_names` list.

### 4.8 `setup()` changes

- Remove the `rc_source_resolved` wiring.
- Set `self.meas._captest = self` and `self.sim._captest = self`.
- `setup()` does **not** compute or clear `ct.rc` (RC lifecycle is driven by
  `rep_cond` / the `rc` setter). Re-running `setup()` leaves an existing `ct.rc`
  intact unless the user recomputes.

## 5. Components and interfaces (summary)

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `CapTest.rc` (property) / `rc_source` | Hold the one test RC + provenance; getter over `_rc` | `_set_rc` |
| `CapTest._set_rc(df, source)` (private) | Single internal write point for `_rc` + `rc_source` | — |
| `CapTest.rc` setter | Manual override: validate RHS coverage → `_set_rc(df, "manual")` | `util.parse_regression_formula` |
| `CapTest.rep_cond(which)` | Compute via a CapData, store onto `ct.rc` | `CapData.rep_cond` |
| `CapData.rep_irr` | Resolve reporting POA (test → `ct.rc`, else `self.rc`) | `_captest` |
| `Irradiance._execute` | `ref_val="rep_irr"` → `capdata.rep_irr` | unchanged |
| `CapTest.captest_results` | Predict at `ct.rc` | `ct.rc` |
| `to_yaml` / `from_yaml` | Persist/restore `rc_source` (+ manual values) | `util.to_native` |

## 6. Error handling and warnings

- `rep_irr` with no resolvable RC → `ValueError` naming the in-test vs standalone
  remedy.
- `captest_results` with `ct.rc is None` → `ValueError` directing to `rep_cond`.
- Source-change overwrite → `UserWarning` (§4.5).
- `rc` setter when `setup()` has not run → `ValueError` (`_require_setup`).
- `rc` setter when `df` omits a required RHS regression variable → `ValueError`
  listing the missing names (§4.4).
- `rc` setter when `meas`/`sim` regression formulas differ → `ValueError`.

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
3. `ct.rc = df` sets `ct.rc` and `rc_source="manual"`; raises when `setup()` has
   not run, when a required RHS variable is missing (message lists them), and
   when `meas`/`sim` formulas differ; extra columns are preserved.
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
