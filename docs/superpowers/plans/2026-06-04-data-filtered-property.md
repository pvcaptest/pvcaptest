# data_filtered Property Flip Implementation Plan (chunk 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CapData.data_filtered` a read-only property derived from the `filters` chain, eliminating it as a second mutable source of truth.

**Architecture:** `data_filtered` becomes `@property` returning `self.data` when `filters` is empty, else `self.data.loc[self.filters[-1].ix_after, :].copy()`. No setter. Every site that assigned or mutated `data_filtered` is redirected: writes to the *source* go to `self.data`; "reset to unfiltered" becomes `self.filters = []`; filter results already flow through `_execute`→`run()`. This is a **big-bang migration** — once the property lands, the shared conftest fixtures and ~100 test sites must all be migrated before the suite goes green again, so the work lands in one commit.

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "`data_filtered` as a Property", "Impact on Other Methods", "`reset_filter()` Simplification", "`__copy__` Simplification".

**Sequencing:** All 12 filters are class-based (done). Only `rep_cond`/`fit_regression` still carry `@update_summary`. This plan unblocks chunk 5 (summary rebuild).

## Migration shape (big-bang + workflow)

This plan is executed in three phases, landing as **one commit** (the property flip is atomic — partial states are red):

1. **Src + conftest (sequential, by the executor):** Tasks 1-6. Make the property and redirect every src write; migrate the shared `tests/conftest.py` fixtures.
2. **Test files (parallel workflow):** Task 7. Fan out the 4 non-conftest test files to one agent each (independent files → no edit conflict). Each agent applies the transformation rules and greens its own file.
3. **Verify + commit (sequential):** Task 8. Full suite, lint, format, single commit.

**The transformation rules** (used in conftest by hand and by the Task 7 agents) — applied to any `X.data_filtered` site:
- **(R1)** `X.data_filtered = X.data.copy()` / `.copy(deep=True)` → **delete the line.** The property returns `data` when no filters are set.
- **(R2)** `X.data_filtered = X.data.iloc[A:B, :].copy()` (or any subset of `X.data`) → `X.data = X.data.iloc[A:B, :].copy()`. **Judgment required:** only safe if the test does not separately assert on the full `X.data`. If it does, instead shrink via a kept-rows filter or restructure — see R5.
- **(R3)** In-place mutation `X.data_filtered.iloc[i, j] = v` / `X.data_filtered.loc[k, c] = v` → `X.data.iloc[i, j] = v` / `X.data.loc[k, c] = v`. A property returns a copy, so mutating the returned frame is a silent no-op; the mutation must target the source `data`.
- **(R4)** `X.data_filtered = <some other frame>` (e.g. `df.copy()`, `X.data.iloc[::2].copy()`) → set `X.data = <that frame>` if the test's intent is "this is the working data," **or** leave the row-selection to a filter if the intent is "this is a filtered subset of unchanged data." Judgment required per site.
- **(R5)** A test that needs a *filtered* working set distinct from `data` (rare) → apply an actual filter (e.g. `X.filter_time(...)`) or construct the state via `filters`, since the property can no longer be set directly.

---

### Task 1: Add the `data_filtered` property; delete the transitional `run()` line

**Files:**
- Modify: `src/captest/filters.py` (delete the transitional assignment in `BaseSummaryStep.run`)
- Modify: `src/captest/capdata.py` (add the property; remove `self.data_filtered = None` from `__init__`)

- [ ] **Step 1: Delete the transitional assignment in `run()`**

In `src/captest/filters.py`, in `BaseSummaryStep.run`, remove the two-line transitional block (currently after the `capdata.filters = capdata.filters + [self]` line):

```python
        # Transitional: keep the legacy data_filtered attribute consistent
        # until data_filtered becomes a derived property (plan 4).
        capdata.data_filtered = capdata.data.loc[self.ix_after, :]
```

So `run()` becomes:

```python
    def run(self, capdata):
        """..."""  # docstring unchanged
        self.ix_after = self._execute(capdata)
        self.pts_after = len(self.ix_after)
        self.ix_before = capdata.data_filtered.index
        self.pts_before = len(self.ix_before)
        self.pts_removed = self.pts_before - self.pts_after
        capdata.filters = capdata.filters + [self]
        self._record_legacy_summary(capdata)
        if self.pts_after == 0:
            warnings.warn("The last filter removed all data!")
```

> Note the ordering: `ix_before`/`pts_before` are read from `capdata.data_filtered` (the property) *before* `self` is appended to `capdata.filters`, so they reflect the prior chain state. Then appending `self` makes the property reflect `self.ix_after`. This is correct because `_execute` already ran against the pre-append `data_filtered`.

- [ ] **Step 2: Add the property to `CapData`**

In `src/captest/capdata.py`, immediately after the `filters = param.List(...)` declaration (before `def __init__`), add:

```python
    @property
    def data_filtered(self):
        """Working data after the applied filter chain (derived, read-only).

        Returns ``self.data`` when no filters are set, otherwise
        ``self.data`` restricted to the rows kept by the last filter
        (``self.filters[-1].ix_after``). A defensive ``.copy()`` is returned so
        downstream mutation of the result cannot corrupt ``self.data`` under
        pandas < 3.0 (no Copy-on-Write). There is no setter: filter results
        flow through ``filters``; to clear filtering set ``self.filters = []``.
        """
        if not self.filters:
            return self.data
        return self.data.loc[self.filters[-1].ix_after, :].copy()
```

- [ ] **Step 3: Remove `self.data_filtered = None` from `__init__`**

In `CapData.__init__`, delete the line:

```python
        self.data_filtered = None
```

(The property now derives it; with empty `self.data` and no filters it returns an empty DataFrame.)

- [ ] **Step 4: Confirm the module imports**

Run: `uv run python -c "import captest.capdata; cd = captest.capdata.CapData('x'); print(type(cd).data_filtered)"`
Expected: prints `<property object ...>` and no error. (The suite is now red — that is expected until Task 8.)

---

### Task 2: Redirect `copy()`, `reset_filter()`, `reset_agg()`, `agg_sensors`

**Files:** Modify `src/captest/capdata.py`.

- [ ] **Step 1: `copy()`** — remove the `data_filtered` copy line (it's derived from the copied `filters`). Delete:

```python
        cd_c.data_filtered = self.data_filtered.copy()
```

- [ ] **Step 2: `reset_filter()`** — replace the body's data line. Change:

```python
        self.data_filtered = self.data.copy()
```

to:

```python
        self.filters = []
```

(The other lines in `reset_filter` — clearing `summary_ix`/`summary`/`filter_counts`/`removed`/`kept` and the existing `self.filters = []` if present — stay; if `self.filters = []` is already there, this collapses to just removing the `data_filtered` line.)

- [ ] **Step 3: `reset_agg()`** — the agg columns are removed from `self.data`; the derived `data_filtered` follows. Replace:

```python
            self.data = self.data[self.pre_agg_cols].copy()
            self.data_filtered = self.data_filtered[self.pre_agg_cols].copy()
```

with:

```python
            self.data = self.data[self.pre_agg_cols].copy()
```

> `reset_agg`'s docstring says "Does not reset filtering" — preserve that. Dropping *columns* from `self.data` does not invalidate the row-based `ix_after`, so filters stay valid and `data_filtered` simply has fewer columns. (This intentionally differs from the spec's Impact table, which over-broadly says `self.filters = []`; clearing filters here would break the documented contract and is unnecessary.)

- [ ] **Step 4: `agg_sensors`** — agg columns are added to `self.data`; clear filters so the derived `data_filtered` reflects the new columns from a clean slate (matches the legacy `self.data_filtered = self.data.copy()` reset). Replace:

```python
        self.data_filtered = self.data.copy()
```

with:

```python
        self.filters = []
```

> The pre-existing guard `if not len(self.summary) == 0: warnings.warn(...)` still fires (legacy `summary` is still populated by `_record_legacy_summary` until chunk 5), so the "filtering steps have been lost" warning behavior is unchanged.

- [ ] **Step 5: Confirm import still works**

Run: `uv run python -c "import captest.capdata"`
Expected: no error.

---

### Task 3: Redirect column mutations (`drop_cols`, `rename_cols`)

**Files:** Modify `src/captest/capdata.py`.

- [ ] **Step 1: `drop_cols`** — drop from the source only. Delete the two `data_filtered` lines:

```python
            self.data_filtered.drop(col, axis=1, inplace=True)
            print("    Dropped from data filtered attribute")
```

(`self.data.drop(col, axis=1, inplace=True)` already runs above it; the derived `data_filtered` no longer has the column automatically.)

- [ ] **Step 2: `rename_cols`** — rename the source only. Delete:

```python
        self.data_filtered.rename(columns=column_map, inplace=True)
```

(`self.data.rename(columns=column_map, inplace=True)` above it suffices.)

- [ ] **Step 3: Confirm import**

Run: `uv run python -c "import captest.capdata"`
Expected: no error.

---

### Task 4: Fix `fit_regression(filter=True)` to record through `filters`

**Files:** Modify `src/captest/capdata.py`.

The residual filter computes `df.index` (kept rows). It must update the derived `data_filtered` by recording a step, not by assigning. Use a module-level helper passed to `FilterCustom`.

- [ ] **Step 1: Check whether any test asserts the summary label `"fit_regression"`**

Run: `grep -rnE "fit_regression" tests/ | grep -iE "summary|filter_names|summary_ix"`
Expected: note any test that asserts the recorded label. If none assert `"fit_regression"` specifically, the `FilterCustom` label (`filter_custom`) is acceptable. If one does, set `custom_name` accordingly and confirm the assertion reads `custom_name` (it does not in the legacy-mirroring path until chunk 5 — flag for that test's update if so).

- [ ] **Step 2: Add a module-level keep-rows helper near the other helpers in `capdata.py`** (e.g. just below `predict_with_pvalue_check`):

```python
def keep_rows(df, index):
    """Return ``df`` restricted to ``index`` (helper for FilterCustom steps)."""
    return df.loc[index, :]
```

- [ ] **Step 3: Replace the `filter=True` assignment block in `fit_regression`**

Replace:

```python
            df = df[np.abs(reg.resid) < 2 * np.sqrt(reg.scale)]
            dframe_flt = self.data_filtered.loc[df.index, :]
            if inplace:
                self.data_filtered = dframe_flt
            else:
                return dframe_flt
```

with:

```python
            df = df[np.abs(reg.resid) < 2 * np.sqrt(reg.scale)]
            if inplace:
                FilterCustom(keep_rows, df.index).run(self)
            else:
                return self.data_filtered.loc[df.index, :]
```

> `FilterCustom(keep_rows, df.index)` stores `func=keep_rows`, `args=(df.index,)`; its `_execute` calls `keep_rows(capdata.data_filtered, df.index)` → returns rows `df.index`, and `run()` records the step and updates the derived `data_filtered`. `FilterCustom` is already imported in `capdata.py`.

- [ ] **Step 4: Decide `@update_summary` on `fit_regression`**

The `@update_summary` decorator only *reads* `data_filtered` (it never assigns it), so it remains compatible with the property. But with `filter=True` now recording via `FilterCustom.run()`, leaving `@update_summary` would **double-record** the step. Remove `@update_summary` from `fit_regression` so recording happens once (via `run()` on the filter path; `filter=False` records nothing, which is correct — it changes no rows).

Confirm by running: `grep -nB1 "def fit_regression" src/captest/capdata.py` shows no `@update_summary` above it after the edit.

- [ ] **Step 5: Confirm import**

Run: `uv run python -c "import captest.capdata"`
Expected: no error.

---

### Task 5: Fix `io.py`, `captest.py`, and `plotting.py` src sites

**Files:** Modify `src/captest/io.py`, `src/captest/captest.py`, `src/captest/plotting.py`.

- [ ] **Step 1: `io.py`** — three sites (`cd.data_filtered = cd.data.copy()` at ~167, ~591, ~622) build `data` then set `data_filtered`. Apply **R1**: delete each `cd.data_filtered = cd.data.copy()` line. The property returns `cd.data` (no filters at load time).

- [ ] **Step 2: `captest.py`** — two sites (~1560 in `_maybe_wrap_sim_year_end`, ~1622 in `process_regression_columns` call path) set `self.sim.data_filtered = self.sim.data.copy()` after changing `self.sim.data`. Apply: replace each with `self.sim.filters = []` (data changed → derived `data_filtered` reflects new data from a clean filter slate).

> Verify the exact lines first: `grep -n "data_filtered" src/captest/captest.py`. Each should be a post-data-change reset → `filters = []`.

- [ ] **Step 3: `plotting.py`** — the calc-params helper (~813-815) adds a derived column to both `cd.data` and `cd.data_filtered`. Since the column is added to `cd.data` first, the derived `data_filtered` picks it up automatically and the `data_filtered` write is redundant (and a silent no-op under the property). Delete the block:

```python
    if cd.data_filtered is not None:
        cd.data_filtered[col_name] = cd.data.loc[cd.data_filtered.index, produced_col]
```

leaving just `cd.data[col_name] = cd.data[produced_col]` above it. (The `is not None` guard is also obsolete — the property never returns None.)

- [ ] **Step 4: Confirm package imports**

Run: `uv run python -c "import captest"`
Expected: no error.

---

### Task 6: Migrate `tests/conftest.py` (shared fixtures, by the executor)

**Files:** Modify `tests/conftest.py`.

conftest fixtures are shared by every test file; they must be migrated first or all dependent tests error at fixture setup. There are ~10 sites, mostly `cd.data_filtered = cd.data.copy(deep=True)`.

- [ ] **Step 1: Apply the transformation rules** to every `data_filtered` site in `tests/conftest.py`:
  - `X.data_filtered = X.data.copy()` / `.copy(deep=True)` → **delete** (R1).
  - `cd.data_filtered = df.copy()` (conftest:~183, where `df` is the freshly built data) → set `cd.data = df` if not already, else delete (R4 — confirm `cd.data` is `df`).

- [ ] **Step 2: Sanity-check a representative fixture loads**

Run: `uv run python -c "import pandas as pd; import tests.conftest as c"` is not how pytest fixtures work; instead run one fixture-dependent test to confirm conftest no longer errors at setup once the source is migrated. Defer the actual pass/fail to Task 8 (the test bodies aren't migrated yet). For now, confirm no syntax error: `uv run python -m py_compile tests/conftest.py`.

---

### Task 7: Parallel-migrate the 4 non-conftest test files (dynamic workflow)

**Files:** `tests/test_CapData.py`, `tests/test_captest.py`, `tests/test_plotting.py`, `tests/test_filter_classes.py`.

This task is run as a **dynamic workflow** — one agent per file (independent files, no edit conflict). Each agent:
1. Reads its assigned file.
2. Finds every `*.data_filtered` assignment and in-place mutation.
3. Applies transformation rules R1-R5, using per-site judgment (especially R2: do not convert `data_filtered = data.iloc[...]` to `data = ...` if the test also asserts on full `data`).
4. Runs *its own file's* tests (`uv run pytest tests/<file> -q`) and iterates until green. (The src property flip and conftest migration are already done, so a correctly-migrated file should pass.)
5. Returns a structured report: sites changed, rule applied per site, final pass/fail, any site needing human judgment.

- [ ] **Step 1: Launch the workflow** (see the executor's workflow invocation; one `agent()` per file with Edit access, no worktree isolation so edits land in the shared tree).

- [ ] **Step 2: Read each agent's report.** For any file the agent left red or flagged, resolve manually.

---

### Task 8: Full-suite verification and commit

**Files:** all of the above.

- [ ] **Step 1:** `just test-wo-warnings` → all pass. Investigate any failure (likely an R2/R4 judgment site or a test that genuinely depended on `data_filtered` diverging from `data`).
- [ ] **Step 2:** `just lint && just fmt` → clean.
- [ ] **Step 3:** Confirm no `data_filtered` *assignment* or *mutation* remains anywhere (ignore `.ipynb_checkpoints/`):
  - `grep -rnE "\.data_filtered\s*=[^=]" src/captest/ tests/ | grep -v ".ipynb_checkpoints"` → empty (no `self.data_filtered = ...` statements; the `@property def data_filtered` line has no `=`).
  - `grep -rnE "data_filtered\[[^]]*\]\s*=[^=]" src/captest/ tests/ | grep -v ".ipynb_checkpoints"` → empty (no column/value bracket-assignments).
  - `grep -rnE "data_filtered\.(iloc|loc)\[[^]]*\]\s*=|data_filtered\.(drop|rename|fillna)\(" src/captest/ tests/ | grep -v ".ipynb_checkpoints"` → empty (no in-place mutations).
- [ ] **Step 4:** Commit:

```bash
git add -A
git commit -m "refactor: make CapData.data_filtered a derived read-only property"
```

---

## Self-Review

**1. Spec coverage:**
- "`data_filtered` as a Property" (getter, no setter, `.copy()` for pre-CoW safety) → Task 1. ✓
- "Impact on Other Methods" table: `data_filtered = df_flt` already returns index from `_execute` (done in prior chunks); reset → `filters = []` (Task 2); `drop`/`rename` → `self.data` (Task 3); `reset_agg` (Task 2 — with the documented "does not reset filtering" deviation noted); `agg_sensors` write-data-only (Task 2). ✓
- "`reset_filter()` Simplification" → Task 2. ✓
- "`__copy__` Simplification" (no explicit `data_filtered` copy) → Task 2 Step 1. ✓
- `fit_regression(filter=True)` (not in the spec's table but forced by the flip) → Task 4. ✓

**2. Placeholder scan:** No TBDs. Every src change shows complete before/after code. The test migration is rule-driven (R1-R5 with concrete examples) and verified per-file by the Task 7 agents + the Task 8 grep gate. Task 4 Step 1 and Task 5 Step 2 include explicit "verify exact lines first" checks because line numbers drift.

**3. Type/name consistency:** `data_filtered` (property), `filters`, `ix_after`, `keep_rows`, `FilterCustom` are used consistently. The property reads `self.filters[-1].ix_after` — matches `BaseSummaryStep`'s `ix_after` attribute set in `run()`.

**Known deviations from the spec (deliberate):**
- `reset_agg` keeps filters (preserving its "does not reset filtering" docstring) rather than clearing them as the spec's Impact table says — clearing is unnecessary for a columns-only change and would break the contract.
- `fit_regression(filter=True)` records via `FilterCustom(keep_rows, df.index)` rather than a dedicated `FitRegression` step (the spec's eventual design) — minimal change appropriate for the property flip; the recorded legacy label becomes `filter_custom` unless Task 4 Step 1 finds a test requiring otherwise.

**Big-bang commit note:** Tasks 1-7 leave the suite red until all test sites are migrated; the single commit in Task 8 lands the flip atomically. This is inherent to flipping a read-only-property invariant whose old attribute was assigned across the codebase.
