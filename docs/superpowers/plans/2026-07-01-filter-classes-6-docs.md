# Docs Update Implementation Plan (chunk 6 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the new filter functionality from chunks 1–5 in the API reference and the changelog: the three new step classes (`RollingStd`, `AbsDiffPrev`, `BooleanFlag`), the new `CapData` wrappers (`filter_rolling_std`, `filter_abs_diff_prev`, `filter_flag`, `filter_threshold`, `filter_sensors_abs_diff`), and the breaking `Sensors` API change.

**Architecture:** Docstrings already ship inline with each class/wrapper (chunks 1–5). This chunk only updates (a) the Sphinx `autosummary` listings that enumerate the public filter surface, and (b) the changelog. No source code changes.

**Tech Stack:** Sphinx autosummary (`.rst`), Keep-a-Changelog markdown, `uv`, `just`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "Documentation").

## Global Constraints

- The `.rst` autosummary entries must reference the exact public names added in
  chunks 1–5: classes `RollingStd`, `AbsDiffPrev`, `BooleanFlag`; methods
  `filter_rolling_std`, `filter_abs_diff_prev`, `filter_flag`,
  `filter_threshold`, `filter_sensors_abs_diff`.
- Match the surrounding markdown/rst style (manual line wrapping in
  `changelog.md`; two-space-indented autosummary entries).
- No source `.py` changes in this chunk — docs only. (Ruff does not lint `.md`
  or `.rst`.)

---

### Task 1 (only task): API-reference + changelog updates

**Files:**
- Modify: `docs/source/api_reference/filters.rst` (Filter Steps autosummary)
- Modify: `docs/source/api_reference/capdata.rst` (filter-methods autosummary)
- Modify: `docs/changelog.md` (`[Unreleased]` → `### Added` and `### Changed`)

**Interfaces:**
- Consumes: the public names shipped by chunks 1–5. No code produced.

- [ ] **Step 1: Add the new step classes to `docs/source/api_reference/filters.rst`**

In the "Filter Steps" `autosummary` block, the current last three entries are:

```rst
   filters.Sensors
   filters.Clearsky
   filters.Missing
   filters.Regression
```

Insert the three new step classes immediately after `filters.Missing` (keeping
`filters.Regression` last):

```rst
   filters.Sensors
   filters.Clearsky
   filters.Missing
   filters.RollingStd
   filters.AbsDiffPrev
   filters.BooleanFlag
   filters.Regression
```

- [ ] **Step 2: Add the new wrappers to `docs/source/api_reference/capdata.rst`**

In the filter-methods `autosummary` block, add the four new column/flag wrappers
after `capdata.CapData.filter_power` and the sensor variant right after
`capdata.CapData.filter_sensors`. The block becomes:

```rst
   capdata.CapData.filter_irr
   capdata.CapData.filter_pvsyst
   capdata.CapData.filter_shade
   capdata.CapData.filter_time
   capdata.CapData.filter_days
   capdata.CapData.filter_outliers
   capdata.CapData.filter_pf
   capdata.CapData.filter_power
   capdata.CapData.filter_rolling_std
   capdata.CapData.filter_abs_diff_prev
   capdata.CapData.filter_flag
   capdata.CapData.filter_threshold
   capdata.CapData.filter_custom
   capdata.CapData.filter_sensors
   capdata.CapData.filter_sensors_abs_diff
   capdata.CapData.filter_clearsky
   capdata.CapData.filter_missing
   capdata.CapData.filter_op_state
   capdata.CapData.reset_filter
   capdata.CapData.describe_filters
   capdata.CapData.filters_to_config
   capdata.CapData.run_pipeline
```

(Only additions — do not remove or reorder the existing entries beyond inserting
the five new lines in the positions shown.)

- [ ] **Step 3: Add an `### Added` changelog entry**

In `docs/changelog.md`, under `## [Unreleased]` → `### Added`, append this bullet
as the last item of the `### Added` list (immediately before the `### Changed`
heading):

```markdown
- New filter classes and `CapData.filter_*` wrappers promoting filters that were
previously passed to `filter_custom` into first-class steps: `RollingStd` /
`filter_rolling_std` (removes intervals whose rolling-window standard deviation is
at or above a threshold — unstable/variable irradiance), `AbsDiffPrev` /
`filter_abs_diff_prev` (removes intervals whose absolute fractional change from the
previous interval exceeds a threshold), and `BooleanFlag` / `filter_flag` (removes
intervals where a boolean/flag column is truthy, with an `invert` toggle to keep
only the truthy rows). Added `CapData.filter_threshold(column, low, high)` — a
one-sided or two-sided inclusive threshold on any column, backed by the
`Irradiance` step. Added `CapData.filter_sensors_abs_diff(thresholds)` for the
absolute-difference sensor comparison.
```

- [ ] **Step 4: Add a `### Changed` (breaking) changelog entry**

In `docs/changelog.md`, under `## [Unreleased]` → `### Changed`, append this bullet
as the last item of the `### Changed` list (immediately before the `### Fixed`
heading):

```markdown
- **Breaking:** Unified the `Sensors` filter's sensor-comparison selection. The
`Sensors` step now takes `method` (`'percent_diff'` or `'abs_diff'`, or a custom
`func(series, threshold) -> bool` callable) and `thresholds` (renamed from
`perc_diff`); the separate `row_filter` parameter is removed.
`CapData.filter_sensors(perc_diff=..., row_filter=...)` becomes
`filter_sensors(thresholds=..., method=...)`. Callers that passed
`row_filter=abs_diff_from_average` should use the new
`filter_sensors_abs_diff(...)` wrapper or `method='abs_diff'`. The
`check_all_perc_diff_comb` helper is no longer re-exported from `captest.capdata`;
import it from `captest.filters`.
```

- [ ] **Step 5: Verify the referenced names exist and the changelog is well-formed**

Confirm every autosummary name added resolves to a real public object:

```bash
uv run python -c "
from captest import filters, capdata
for n in ['RollingStd', 'AbsDiffPrev', 'BooleanFlag']:
    assert hasattr(filters, n), n
for m in ['filter_rolling_std', 'filter_abs_diff_prev', 'filter_flag',
          'filter_threshold', 'filter_sensors_abs_diff']:
    assert hasattr(capdata.CapData, m), m
print('all doc-referenced names resolve')
"
```
Expected: `all doc-referenced names resolve`.

Then build the docs and confirm the new entries do not introduce errors:

```bash
just docs
```
Expected: the build completes. Pre-existing Sphinx warnings unrelated to this
change are acceptable, but there must be no new error naming `RollingStd`,
`AbsDiffPrev`, `BooleanFlag`, `filter_rolling_std`, `filter_abs_diff_prev`,
`filter_flag`, `filter_threshold`, or `filter_sensors_abs_diff` (which would mean
an autosummary entry points at a missing object). If `just docs` cannot run in
this environment (missing Sphinx deps), report that and rely on the import check
above.

- [ ] **Step 6: Commit**

```bash
git add docs/source/api_reference/filters.rst docs/source/api_reference/capdata.rst docs/changelog.md
git commit -m "docs: document new filter classes, wrappers, and Sensors API change"
```

---

## Self-Review

- **Spec coverage:** The spec's Documentation section (NumPy docstrings — already
  inline from chunks 1–5; user-facing filter list; changelog incl. the breaking
  `Sensors` note) is covered: the API-reference autosummary lists gain the three
  new classes and five new wrappers, and the changelog gains an `### Added` entry
  plus a `### Changed` **Breaking** entry naming the `perc_diff`→`thresholds` /
  `row_filter`→`method` change and the dropped `check_all_perc_diff_comb`
  re-export.
- **Placeholder scan:** none — every edit is concrete text with an exact location.
- **Type consistency:** All eight names in the autosummary additions match the
  public API delivered in chunks 1–5 (verified names: `RollingStd`, `AbsDiffPrev`,
  `BooleanFlag`, `filter_rolling_std`, `filter_abs_diff_prev`, `filter_flag`,
  `filter_threshold`, `filter_sensors_abs_diff`), and Step 5's import check gates
  exactly those names.
```
