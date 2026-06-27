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

### 4.3 Updating the test RC from a CapData (`rep_cond`)

**Runtime rule (after construction): last-writer-wins.** Any `cd.rep_cond()` on a
CapData that belongs to a test (`cd._captest is not None`) does what it does
standalone — appends the `RepCond` step and sets `cd.rc` — and then updates the
test RC by calling `ct._set_rc(cd.rc.copy(), <this cd's side>)`, which sets
`ct.rc` and sets `ct.rc_source` to that side. When this **changes the source**
(§4.5) it emits a `UserWarning` so the flip is visible; recomputing the current
source is silent.

`ct.rep_cond(which="meas", **overrides)` remains the convenience entry point: it
picks the CapData, merges the preset / `self.rep_conditions` config as today, and
calls `cd.rep_cond(...)`, so the test RC is updated through the same path. The
source-change warning rule (§4.5) applies uniformly — there is no path-specific
suppression — so a flip warns whether triggered via `cd.rep_cond()` or
`ct.rep_cond(other_side)`. `which` defaults to the current `self.rc_source` when
it is `"meas"`/`"sim"`, else `"meas"`.

`ct.rc` stores a **copy** of `cd.rc` (avoids aliasing); a later direct
`cd.rep_cond()` re-runs the whole update, so `ct.rc` simply tracks the most recent
`rep_cond` across either dataset.

**Standalone unaffected:** when `cd._captest is None`, `cd.rep_cond()` only sets
`cd.rc` — no test-level update, no warning.

**Why last-writer-wins (vs pinning to config).** Chosen for the interactive
post-load workflow: a user adjusting filters and recomputing `rep_cond` wants the
latest computation to become the test RC, flagged by a warning rather than
blocked by an error. Config/constructor `rc_source` *seeds* the value (and is
restored deterministically on load, §4.7) but does not lock it afterward.

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
3. Coerces to a one-row DataFrame, applies the §4.5 source-change warning, then
   records the override via `self._set_rc(df, "manual")`.

**Internal vs public path.** Because the public setter always means "manual", the
computed-RC assignments from `rep_cond` (§4.3) and `from_yaml` (§4.7) must not go
through it. They call the private helper `self._set_rc(df, source)`, which sets
`self._rc` and `self.rc_source` and records the true `source`. The public
property setter is exactly that helper with `source="manual"`, preceded by the
validation above. This preserves a single public spelling (`ct.rc = df`) while
keeping provenance correct for internally-computed RCs.

### 4.5 Source-change warning

A single rule governs **every** runtime path that sets the test RC
(`cd.rep_cond()` in a test, `ct.rep_cond(which)`, and the `ct.rc = df` manual
setter): warn **only when the source changes** — i.e. `self.rc is not None` *and*
the new source differs from the current `self.rc_source`. This covers a meas↔sim
flip and any change to or from `"manual"`. Recomputing the current source (same
`rc_source`) and the first time an RC is established (`self.rc is None`) are
**silent**.

The warning names the old and new source, e.g. *"Test reporting conditions
rc_source changed from 'meas' to 'sim'."* It is suppressed during `from_yaml`
load (§4.7).

### 4.6 `captest_results`

Replace the `rc_source`-based pick of `meas.rc`/`sim.rc` with `self.rc`. If
`self.rc is None`, raise a clear error directing the user to call
`ct.rep_cond(which)` or assign `ct.rc = df`. The printed provenance line uses
`self.rc_source`.

### 4.7 Serialization and load

`rc_source` serializes via the existing `scalar_names` list. RC *values* are
serialized only for the manual case:

- **Computed (`"meas"`/`"sim"`)** — values are not serialized; they are recomputed
  by replaying the source pipeline's `RepCond` step. The existing "drop
  `overrides.rep_conditions` when a `RepCond` step is present" rule is unchanged.
- **Manual (`"manual"`)** — values *are* data; serialize the one-row RC as a
  yaml-safe mapping under a new optional key `reporting_conditions_values`
  (numpy scalars coerced via `util.to_native`).

**Load differs from interactive runtime** so round-trips are deterministic and
self-filtering pipelines stay correct. During `from_yaml`, a `_loading` flag on
the CapTest modifies the §4.3 auto-update for the duration of replay:

1. Warnings are suppressed.
2. **The configured `rc_source` side's pipeline is replayed first** (not a fixed
   meas-before-sim order). This is required: the *other* side's pipeline may
   contain `filter_irr(ref_val="rep_irr")` (the standard "filter one dataset
   around the other's rep irr" pattern), which resolves against `ct.rc`. `ct.rc`
   is populated only by the configured source's `RepCond` (point 3), so that side
   must run first or the other side's filter raises on a `None` `ct.rc`. For
   `rc_source="meas"` this is meas→sim; for `rc_source="sim"` it is **sim→meas**.
3. The auto-update applies **only when the CapData is the configured `rc_source`
   side** (config-seeded, not last-writer-wins). So the configured source's
   `RepCond` step populates `ct.rc` *mid-replay*, making `ref_val="rep_irr"`
   resolvable for both sides' filters during replay, while the other side's
   `RepCond` (if any) updates only its `cd.rc`.
4. For `rc_source="manual"`, `ct.rc` is set from `reporting_conditions_values`
   **before** replay; no `RepCond` step overwrites it (no side matches a
   `"manual"` source), so the manual values are in effect for any self-filtering
   pipeline during replay regardless of order (meas→sim is fine).

After replay, `_loading` is cleared; `ct.rc` / `rc_source` reflect the configured
state and interactive last-writer-wins (§4.3) takes over.

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
| `CapTest.rep_cond(which)` | Convenience: pick cd, merge config, call `cd.rep_cond` | `CapData.rep_cond` |
| `CapData.rep_cond` / `_calc_rep_cond` | Set `cd.rc`; if in a test, update `ct.rc` (last-writer) + warn on source change | `_captest`, `_set_rc` |
| `CapData.rep_irr` | Resolve reporting POA (test → `ct.rc`, else `self.rc`) | `_captest` |
| `Irradiance._execute` | `ref_val="rep_irr"` → `capdata.rep_irr` | unchanged |
| `CapTest.captest_results` | Predict at `ct.rc` | `ct.rc` |
| `to_yaml` / `from_yaml` | Persist/restore `rc_source` (+ manual values) | `util.to_native` |

## 6. Error handling and warnings

- `rep_irr` with no resolvable RC → `ValueError` naming the in-test vs standalone
  remedy.
- `captest_results` with `ct.rc is None` → `ValueError` directing to `rep_cond`.
- Test RC **source change** (meas↔sim, or to/from `"manual"`) via any runtime
  path → `UserWarning` (§4.5); silent on same-source recompute and first
  establishment; suppressed during load.
- `rc` setter when `setup()` has not run → `ValueError` (`_require_setup`).
- `rc` setter when `df` omits a required RHS regression variable → `ValueError`
  listing the missing names (§4.4).
- `rc` setter when `meas`/`sim` regression formulas differ → `ValueError`.

## 7. Migration impact

- Internal-only removals (added this session, unreleased): `rc_source_resolved`
  attribute, its `setup()` wiring, and the `rep_irr` branch that read it.
- Implementation change: `from_yaml` currently replays meas then sim
  (captest.py:1818-1821, the unconditional order). It must instead replay the
  configured `rc_source` side first (§4.7), falling back to meas→sim for
  `rc_source="manual"`.
- Behavior change: `captest_results` now reads `ct.rc`. Existing flows that call
  `cd.rep_cond()` (e.g. via `run_test`) and then `captest_results` keep working —
  last-writer-wins means the bare `cd.rep_cond()` now populates `ct.rc`. With the
  default `rc_source` matching the computed side (the common case) this is silent
  (first establishment / same source); the §4.5 warning fires only if a flow
  computes RC on the other side and flips the source. The example notebook is
  being reworked by the user separately.
- Tests added this session (`TestRepIrrCrossInstance`, the `rc_source_resolved`
  wiring asserts in `TestSetup`) are updated to the `_captest` / `ct.rc` model.

## 8. Testing plan

1. `CapTest.rc`/`rc_source` defaults; `_captest` wired on both CapData by
   `setup()`.
2. `ct.rep_cond("meas")` sets `ct.rc` (== meas computation) and `rc_source="meas"`;
   same for `"sim"`.
3. **Last-writer-wins + source-change warning:** with `rc_source="meas"`,
   `meas.rep_cond()` again is **silent** (same source, recomputed values); a
   following `sim.rep_cond()` flips to `sim` and **warns**; the first
   establishment (when `ct.rc` was None) is silent. Standalone `cd.rep_cond()`
   (no `_captest`) updates only `cd.rc` and never warns.
4. `ct.rc = df` sets `ct.rc` and `rc_source="manual"`; raises when `setup()` has
   not run, when a required RHS variable is missing (message lists them), and when
   `meas`/`sim` formulas differ; extra columns are preserved; **warns** when it
   changes the source (e.g. meas→manual), silent if the prior source was already
   `"manual"`.
5. `rep_irr`: in-test reads `ct.rc`; standalone reads `cd.rc`; `ValueError` when
   `ct.rc` is None in a test.
6. `filter_irr(ref_val="rep_irr")` on sim uses `ct.rc` set from meas, without a
   manual value pass (the original bug's real-world scenario).
7. `captest_results` predicts at `ct.rc`; `ValueError` when None.
8. Serialization round-trip — computed source recomputes `ct.rc` on load; manual
   override serializes values and restores `ct.rc` + `rc_source="manual"`.
9. **Load determinism:** a config where both pipelines contain a `RepCond` step
   restores `ct.rc`/`rc_source` to the configured source (config-seeded, not the
   last replayed step); no warnings emit during load.
10. **Mid-replay self-filtering, both source sides:**
    - `rc_source="meas"`: a meas pipeline of
      `RepCond → filter_irr(ref_val="rep_irr")` (plus a sim pipeline whose
      `filter_irr(ref_val="rep_irr")` references `ct.rc`) round-trips through
      `to_yaml`/`from_yaml` without error.
    - `rc_source="sim"`: the **other** side carries the `rep_irr` filter — a meas
      pipeline with `filter_irr(ref_val="rep_irr")` and a sim pipeline with the
      `RepCond` — round-trips without error, proving the sim→meas replay order
      (the configured-source-first rule); this case fails under a fixed
      meas-before-sim order, so it is the explicit regression guard.
11. **Replay order:** `from_yaml` replays the configured `rc_source` side first
    (sim→meas when `rc_source="sim"`).
12. Standalone `CapData` regression tests remain green (no behavior change).

## 9. Open questions

None outstanding. Resolved during brainstorming:

- Standalone CapData keeps owning `cd.rc` (yes).
- **Runtime: last-writer-wins.** Any `cd.rep_cond()` in a test updates `ct.rc` +
  `rc_source`; a `UserWarning` fires **only on a source change** (meas↔sim or to/
  from `"manual"`), uniformly across paths (no per-path suppression).
  Config/constructor `rc_source` seeds the value but does not pin it after load.
- Load is config-seeded and warning-suppressed for deterministic round-trips
  (§4.7).
- Manual override is a first-class source (`rc_source="manual"`), serialized by
  value, set only via the `ct.rc = df` setter.

## 10. Forward compatibility: separate meas/sim setup + filter re-runs

Known future work (not started; deferred until after this branch merges and
releases): support re-running **all of one side's** setup + filtering when its
source data changes — e.g. new `sim` data, re-run sim only. This design is built
to accommodate it; notes for the future implementer:

- **The common case is already decoupled.** With `rc_source="meas"`, re-running
  sim's setup + filter pipeline needs nothing from meas: sim's
  `filter_irr(ref_val="rep_irr")` reads the test-level `ct.rc` (meas-derived,
  unchanged), sim refits, `captest_results` recomputes. Moving RC to the test
  level is what removes the old per-CapData dependency.
- **`_set_rc` is the single mutation point** for `ct.rc` — the natural hook for a
  future "RC changed → invalidate dependent filters" mechanism.
- **`_captest` wiring and any per-side RC (re)population must stay decomposable**
  so a future `setup(which="sim")` / per-side re-run can touch one side without
  disturbing the other. Keep the two `_captest` assignments and the
  `process_regression_columns` calls independently invokable per side.
- **Directional staleness to handle later.** When `rc_source` points at the side
  *not* being re-run, dependents can go stale: e.g. `rc_source="sim"` + re-run-sim
  changes `ct.rc`, leaving meas's `rep_irr`-based filters stale. This is inherent
  to "filter one dataset around the other's rep irr," not a flaw here; the future
  work should add staleness detection at the `_set_rc` hook.
- **Replay the configured `rc_source` side first** in `from_yaml` (§4.7) so the
  source's `ct.rc` exists before the other side's `rep_irr` filters at initial
  load (sim→meas when `rc_source="sim"`; meas→sim when `"meas"`). At re-run time
  `ct.rc` persists, so ordering no longer matters then — but a future
  `setup(which=...)` re-run must still ensure `ct.rc` is current for the source
  side before re-filtering the dependent side.
