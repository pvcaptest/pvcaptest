# data_filtered Property Flip Implementation Plan (chunk 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CapData.data_filtered` a read-only property derived from the `filters` chain, eliminating it as a second mutable source of truth.

**Architecture:** `data_filtered` becomes `@property` returning `self.data` when `filters` is empty, else `self.data.loc[self.filters[-1].ix_after, :].copy()`. No setter. Every site that assigned or mutated `data_filtered` is redirected: writes to the *source* go to `self.data`; "reset to unfiltered" becomes `self.filters = []`; filter results already flow through `_execute`→`run()`. This is a **big-bang migration** — once the property lands, the shared conftest fixtures and ~100 test sites must all be migrated before the suite goes green again, so the work lands in one commit.

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "`data_filtered` as a Property", "Impact on Other Methods", "`reset_filter()` Simplification", "`__copy__` Simplification".

**Sequencing:** Execute **after** the FilterRegression plan (`2026-06-04-filter-regression.md`). All 12 row filters are class-based; once FilterRegression lands, `fit_regression` routes through `run()` too, so only `rep_cond` still carries `@update_summary` (its decorator only *reads* `data_filtered`, so it stays compatible with the property). That plan also already removed this plan's old "fit_regression via FilterCustom" task, and it introduces two new `data_filtered` assignments in `tests/test_filter_classes.py` that this plan's Task 6 must sweep (see the dependency note there). This plan unblocks chunk 5 (summary rebuild).

## Migration shape (big-bang + workflow)

This plan is executed in three phases, landing as **one commit** (the property flip is atomic — partial states are red):

1. **Src + conftest (sequential, by the executor):** Tasks 1-6. Make the property and redirect every src write; migrate the shared `tests/conftest.py` fixtures.
2. **Test files (parallel workflow):** Task 6. Fan out the 4 non-conftest test files to one agent each (independent files → no edit conflict). Each agent applies the transformation rules and greens its own file.
3. **Verify + commit (sequential):** Task 7. Full suite, lint, format, single commit.

**The transformation rules** (used in conftest by hand and by the Task 6 agents) — applied to any `X.data_filtered` site:
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
Expected: prints `<property object ...>` and no error. (The suite is now red — that is expected until Task 7.)

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

> **`fit_regression(filter=True)` is intentionally NOT handled here** — the FilterRegression plan (`2026-06-04-filter-regression.md`, executed before this one) already converted it to delegate to `FilterRegression().run(self)`, so it no longer assigns `data_filtered` and needs no change in this plan.

### Task 4: Fix `io.py`, `captest.py`, and `plotting.py` src sites

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

### Task 5: Migrate `tests/conftest.py` (shared fixtures, by the executor)

**Files:** Modify `tests/conftest.py`.

conftest fixtures are shared by every test file; they must be migrated first or all dependent tests error at fixture setup. There are ~10 sites, mostly `cd.data_filtered = cd.data.copy(deep=True)`.

- [ ] **Step 1: Apply the transformation rules** to every `data_filtered` site in `tests/conftest.py`:
  - `X.data_filtered = X.data.copy()` / `.copy(deep=True)` → **delete** (R1).
  - `cd.data_filtered = df.copy()` (conftest:~183, where `df` is the freshly built data) → set `cd.data = df` if not already, else delete (R4 — confirm `cd.data` is `df`).

- [ ] **Step 2: Sanity-check a representative fixture loads**

Run: `uv run python -c "import pandas as pd; import tests.conftest as c"` is not how pytest fixtures work; instead run one fixture-dependent test to confirm conftest no longer errors at setup once the source is migrated. Defer the actual pass/fail to Task 7 (the test bodies aren't migrated yet). For now, confirm no syntax error: `uv run python -m py_compile tests/conftest.py`.

---

### Task 6: Parallel-migrate the 4 non-conftest test files (dynamic workflow)

**Files:** `tests/test_CapData.py`, `tests/test_captest.py`, `tests/test_plotting.py`, `tests/test_filter_classes.py`.

> **Dependency on the FilterRegression plan (`2026-06-04-filter-regression.md`, runs before this one):** that plan adds two new `data_filtered` *assignments* in `tests/test_filter_classes.py` — the `cd_reg` fixture (`cd.data_filtered = cd.data.copy()`) and `test_execute_nan_calls_filter_missing` (`cd_reg.data_filtered = cd_reg.data.copy()` after injecting a NaN into `cd_reg.data`). Both are plain **R1** deletions (the property returns `data` when no filters are set; the NaN injection into `cd_reg.data` is reflected automatically). The `test_filter_classes.py` agent below must sweep them along with the rest; the Task 7 grep gate (which scans all of `tests/`) will catch any miss.

This task is run as a **dynamic workflow** — one agent per file (independent files, no edit conflict). Each agent:
1. Reads its assigned file.
2. Finds every `*.data_filtered` assignment and in-place mutation.
3. Applies transformation rules R1-R5, using per-site judgment (especially R2: do not convert `data_filtered = data.iloc[...]` to `data = ...` if the test also asserts on full `data`).
4. Runs *its own file's* tests (`uv run pytest tests/<file> -q`) and iterates until green. (The src property flip and conftest migration are already done, so a correctly-migrated file should pass.)
5. Returns a structured report: sites changed, rule applied per site, final pass/fail, any site needing human judgment.

- [ ] **Step 1: Launch the workflow** (see the executor's workflow invocation; one `agent()` per file with Edit access, no worktree isolation so edits land in the shared tree).

- [ ] **Step 2: Read each agent's report.** For any file the agent left red or flagged, resolve manually.

---

### Task 7: Full-suite verification and commit

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
- `fit_regression(filter=True)` (forced by the flip) → already handled by the FilterRegression plan (runs first); no task here. ✓

**2. Placeholder scan:** No TBDs. Every src change shows complete before/after code. The test migration is rule-driven (R1-R5 with concrete examples) and verified per-file by the Task 6 agents + the Task 7 grep gate. Task 4 Step 2 (`captest.py`) includes an explicit "verify exact lines first" check because line numbers drift.

**3. Type/name consistency:** `data_filtered` (property), `filters`, `ix_after` are used consistently. The property reads `self.filters[-1].ix_after` — matches `BaseSummaryStep`'s `ix_after` attribute set in `run()`.

**Known deviations from the spec (deliberate):**
- `reset_agg` keeps filters (preserving its "does not reset filtering" docstring) rather than clearing them as the spec's Impact table says — clearing is unnecessary for a columns-only change and would break the contract.

**Big-bang commit note:** Tasks 1-6 leave the suite red until all test sites are migrated; the single commit in Task 7 lands the flip atomically. This is inherent to flipping a read-only-property invariant whose old attribute was assigned across the codebase.
