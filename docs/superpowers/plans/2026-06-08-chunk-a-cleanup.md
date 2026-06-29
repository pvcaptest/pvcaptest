# Chunk A — Filters Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the filter step classes (drop the `Filter` prefix), remove the `inplace` kwarg from the `filter_*` wrappers and `fit_regression`, and expose `custom_name` on the wrapper methods.

**Architecture:** Three behaviour-preserving-or-narrowing edits to `src/captest/filters.py` and `src/captest/capdata.py`, plus matching test updates. The class rename is a pure mechanical rename done atomically across all `.py` files so the suite stays green. `inplace` removal deletes the dual preview/record code path; the preview-only tests are deleted (the behaviour is gone) while in-place tests just drop the kwarg. `custom_name` is forwarded into each step constructor.

**Tech Stack:** Python 3.12, `param`, pandas, pytest, `uv`, `ruff`, `just`.

Reference spec: `docs/superpowers/specs/2026-06-08-filters-refactor-cleanup-design.md` (Group A).

**Run all commands from the repo root** `/home/ben/python/pvcaptest_bt-`. Tests run via `uv run pytest ...`; full suite/lint via `just test` / `just lint` / `just fmt`.

---

## Task 1: Rename filter step classes (atomic)

Pure rename — no behaviour change. Done across `src/` and `tests/` in one pass so the suite passes immediately afterward. Word boundaries (`\b`) ensure `class TestFilterIrr` (test classes) and unrelated identifiers are **not** matched — only standalone `FilterIrr`, `filters.FilterIrr`, and the quoted registry/type strings `"FilterIrr"`.

Rename map:

| Old | New |
| --- | --- |
| `FilterIrr` | `Irradiance` |
| `FilterPvsyst` | `Pvsyst` |
| `FilterShade` | `Shade` |
| `FilterTime` | `Time` |
| `FilterDays` | `Days` |
| `FilterOutliers` | `Outliers` |
| `FilterPf` | `PowerFactor` |
| `FilterPower` | `Power` |
| `FilterCustom` | `Custom` |
| `FilterSensors` | `Sensors` |
| `FilterClearsky` | `Clearsky` |
| `FilterMissing` | `Missing` |
| `FilterRegression` | `Regression` |

**Files:**
- Modify: `src/captest/filters.py` (class defs, `FILTER_REGISTRY` keys, the hardcoded `"type": "FilterCustom"` in `Custom.to_config`)
- Modify: `src/captest/capdata.py` (imports block at lines 37–52, every wrapper body, `isinstance(step, FilterTime)` at ~2672, docstrings)
- Modify: `src/captest/clearsky.py` (module docstring mention at line 4)
- Modify: `tests/test_filter_classes.py`, `tests/test_CapData.py`, `tests/test_captest.py`

- [ ] **Step 1: Confirm the pre-rename suite is green**

Run: `just test`
Expected: PASS (establish baseline before renaming).

- [ ] **Step 2: Apply the rename across all `.py` files**

Run:

```bash
cd /home/ben/python/pvcaptest_bt-
files=$(grep -rlE "\bFilter(Irr|Pvsyst|Shade|Time|Days|Outliers|Pf|Power|Custom|Sensors|Clearsky|Missing|Regression)\b" src tests)
sed -i \
 -e 's/\bFilterIrr\b/Irradiance/g' \
 -e 's/\bFilterPvsyst\b/Pvsyst/g' \
 -e 's/\bFilterShade\b/Shade/g' \
 -e 's/\bFilterTime\b/Time/g' \
 -e 's/\bFilterDays\b/Days/g' \
 -e 's/\bFilterOutliers\b/Outliers/g' \
 -e 's/\bFilterPf\b/PowerFactor/g' \
 -e 's/\bFilterPower\b/Power/g' \
 -e 's/\bFilterCustom\b/Custom/g' \
 -e 's/\bFilterSensors\b/Sensors/g' \
 -e 's/\bFilterClearsky\b/Clearsky/g' \
 -e 's/\bFilterMissing\b/Missing/g' \
 -e 's/\bFilterRegression\b/Regression/g' \
 $files
```

Note: `FilterPf` is processed by its own expression before any `Power` substitution exists in the text, and `\bFilterPf\b` cannot match inside `FilterPower`, so ordering is safe.

- [ ] **Step 3: Verify no old names remain and registry/imports are consistent**

Run:

```bash
grep -rnE "\bFilter(Irr|Pvsyst|Shade|Time|Days|Outliers|Pf|Power|Custom|Sensors|Clearsky|Missing|Regression)\b" src tests
```
Expected: no output (exit 1). `class TestFilterIrr` etc. must still be present (verify with `grep -rn "class TestFilter" tests | head` — these are intentionally untouched).

- [ ] **Step 4: Lint/format (fix import sorting in capdata.py)**

Run: `just lint && just fmt`
Expected: ruff re-sorts the `from captest.filters import (...)` block (now `Clearsky, Custom, Days, Irradiance, Missing, Outliers, Power, PowerFactor, Pvsyst, Regression, Sensors, Shade, Time, RepCond`) and reports success.

- [ ] **Step 5: Run the full suite (must be green — pure rename)**

Run: `just test`
Expected: PASS, same test count as Step 1. A failure here means a reference was missed or a new name collides; resolve before committing.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: drop Filter prefix from filter step class names"
```

---

## Task 2: Remove `inplace` from the `filter_*` wrappers

The `inplace=False` branch returned a preview DataFrame without recording a step. That dual behaviour is removed: wrappers always build the step and `run()` it. Affected wrappers: `filter_irr`, `filter_pvsyst`, `filter_shade`, `filter_time`, `filter_days`, `filter_outliers`, `filter_pf`, `filter_power`, `filter_sensors`, `filter_clearsky`, and the unimplemented stub `filter_op_state` (signature only). `filter_custom` and `filter_missing` already lack `inplace`.

**Files:**
- Modify: `src/captest/capdata.py` (the wrapper signatures/bodies/docstrings)
- Modify: `tests/test_CapData.py`, `tests/test_filter_classes.py`

- [ ] **Step 1: Write a failing test that the wrapper rejects `inplace`**

Add to `tests/test_filter_classes.py` inside `class TestFilterIrr`:

```python
    def test_wrapper_rejects_inplace_kwarg(self, cd_irr):
        with pytest.raises(TypeError):
            cd_irr.filter_irr(200, 800, inplace=False)
```

(`pytest` is already imported in this module.)

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr::test_wrapper_rejects_inplace_kwarg -v`
Expected: FAIL (currently `filter_irr` accepts `inplace`, so no TypeError is raised).

- [ ] **Step 3: Collapse the run/preview branch in all ten wrappers**

Every one of the ten wrappers ends with this identical block. Replace **all occurrences** of:

```python
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

with:

```python
        flt.run(self)
```

(An Edit with `replace_all=True` on that exact block handles all ten at once.)

- [ ] **Step 4: Remove `inplace` from each wrapper signature**

Apply these exact signature edits in `src/captest/capdata.py`:

- `def filter_irr(self, low, high, ref_val=None, col_name=None, inplace=True):` → `def filter_irr(self, low, high, ref_val=None, col_name=None):`
- `def filter_pvsyst(self, inplace=True):` → `def filter_pvsyst(self):`
- `def filter_shade(self, fshdbm=1.0, query_str=None, inplace=True):` → `def filter_shade(self, fshdbm=1.0, query_str=None):`
- `def filter_days(self, days, drop=False, inplace=True):` → `def filter_days(self, days, drop=False):`
- `def filter_outliers(self, inplace=True, **kwargs):` → `def filter_outliers(self, **kwargs):`
- `def filter_pf(self, pf, inplace=True):` → `def filter_pf(self, pf):`
- `def filter_power(self, power, percent=None, columns=None, inplace=True):` → `def filter_power(self, power, percent=None, columns=None):`
- `def filter_op_state(self, op_state, mult_inv=None, inplace=True):` → `def filter_op_state(self, op_state, mult_inv=None):`

For `filter_time`, delete the `        inplace=True,` line from the multi-line signature:

```python
    def filter_time(
        self,
        start=None,
        end=None,
        drop=False,
        days=None,
        test_date=None,
        inplace=True,
    ):
```
→ remove the `        inplace=True,` line.

For `filter_sensors`, delete `inplace=True, `:

```python
    def filter_sensors(
        self, perc_diff=None, inplace=True, row_filter=check_all_perc_diff_comb
    ):
```
→
```python
    def filter_sensors(
        self, perc_diff=None, row_filter=check_all_perc_diff_comb
    ):
```

For `filter_clearsky`, delete `inplace=True, `:

```python
    def filter_clearsky(self, ghi_col=None, inplace=True, keep_clear=True, **kwargs):
```
→
```python
    def filter_clearsky(self, ghi_col=None, keep_clear=True, **kwargs):
```

- [ ] **Step 5: Remove the `inplace` docstring entries**

The following block appears verbatim in `filter_irr`, `filter_pvsyst`, `filter_shade`, `filter_time`, `filter_days`, `filter_outliers`, `filter_pf`, `filter_power` — delete every occurrence (Edit with `replace_all=True`):

```
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
```

`filter_sensors` has a near-identical block plus a `Returns` section — delete both:

```
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        row_filter : callable, default check_all_perc_diff_comb
            Row-wise consistency check applied across a group's columns.

        Returns
        -------
        DataFrame
            Returns filtered dataframe if inplace is False.
```
→
```
        row_filter : callable, default check_all_perc_diff_comb
            Row-wise consistency check applied across a group's columns.
```

`filter_clearsky` has the block between `ghi_col` and `keep_clear` — delete just the three `inplace` lines there.

`filter_op_state` (the unimplemented stub) — delete its `inplace` parameter doc lines and the `Returns ... CapData ... when inplace is False` section from its docstring.

- [ ] **Step 6: Run the new TypeError test to verify it passes**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr::test_wrapper_rejects_inplace_kwarg -v`
Expected: PASS.

- [ ] **Step 7: Delete the preview-only tests (behaviour removed)**

Delete these test methods entirely:

In `tests/test_filter_classes.py`:
- `TestFilterIrr::test_wrapper_inplace_false_records_no_step`
- `TestFilterSensors::test_wrapper_inplace_false_records_no_step`
- `TestFilterTime::test_wrapper_inplace_false_records_no_step`
- `TestFilterOutliers::test_wrapper_inplace_false_records_no_step`
- `TestFilterClearsky::test_wrapper_inplace_false_records_no_step`

In `tests/test_CapData.py`:
- `test_filter_pvsyst_not_inplace`
- `test_filter_shade_default_not_inplace`
- `test_refval_withcol_notinplace`
- `test_start_end_not_inplace`
- `test_not_inplace` (the `filter_days` one, in the filter-days test class)

- [ ] **Step 8: Run the full suite; strip remaining `inplace=True` from filter calls**

Run: `just test`
Expected initially: failures (`TypeError: ... got an unexpected keyword argument 'inplace'`) on tests that still pass `inplace=True` to a `filter_*` call.

For each such failure, remove the `inplace=True` argument from the **filter call only** (do not touch pandas `inplace=True` on `.rename`/`.drop`, nor `agg_group(..., inplace=...)`). Known locations in `tests/test_CapData.py` (verify by re-running): the `filter_sensors`, `filter_irr`, `filter_days`, and `filter_power` calls around lines 1978–2006, 2216–2252, 2405–2431, and 2486–2513.

Re-run `just test` until green.

- [ ] **Step 9: Lint/format**

Run: `just lint && just fmt`
Expected: success.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: remove inplace kwarg from filter_* wrappers"
```

---

## Task 3: Remove `inplace` from `fit_regression`

`fit_regression`'s `inplace` only mattered when `filter=True`, governing whether the `Regression` step was recorded. It now always records.

**Files:**
- Modify: `src/captest/capdata.py` (`fit_regression`, ~lines 2453–2493)
- Modify: `tests/test_filter_classes.py`

- [ ] **Step 1: Write a failing test that `fit_regression` rejects `inplace`**

Add to `tests/test_filter_classes.py` inside `class TestFilterRegression`:

```python
    def test_fit_regression_rejects_inplace_kwarg(self, cd_reg):
        with pytest.raises(TypeError):
            cd_reg.fit_regression(filter=True, inplace=False, summary=False)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterRegression::test_fit_regression_rejects_inplace_kwarg -v`
Expected: FAIL (no TypeError yet).

- [ ] **Step 3: Rewrite `fit_regression`**

Replace:

```python
    def fit_regression(self, filter=False, inplace=True, summary=True):
```
with:
```python
    def fit_regression(self, filter=False, summary=True):
```

Replace the body's filter branch:

```python
        if filter:
            print("NOTE: Regression used to filter outlying points.\n\n")
            flt = Regression(n_std=2)
            if inplace:
                flt.run(self)
                if summary:
                    print(flt.regression_model.summary())
            else:
                kept = flt._execute(self)
                if summary:
                    print(flt.regression_model.summary())
                return self.data_filtered.loc[kept, :]
```
with:
```python
        if filter:
            print("NOTE: Regression used to filter outlying points.\n\n")
            flt = Regression(n_std=2)
            flt.run(self)
            if summary:
                print(flt.regression_model.summary())
```

Update the docstring: remove the `inplace : bool, default True ...` parameter block and the `Returns ... Filtered DataFrame when filter=True and inplace=False.` section.

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterRegression::test_fit_regression_rejects_inplace_kwarg -v`
Expected: PASS.

- [ ] **Step 5: Delete the preview-path test**

Delete `TestFilterRegression::test_filter_true_not_inplace_records_no_step` from `tests/test_filter_classes.py`.

- [ ] **Step 6: Run the full suite**

Run: `just test`
Expected: PASS (resolve any remaining `fit_regression(..., inplace=...)` callers by dropping the kwarg).

- [ ] **Step 7: Lint/format and commit**

```bash
just lint && just fmt
git add -A
git commit -m "refactor: remove inplace kwarg from fit_regression"
```

---

## Task 4: Add `custom_name` to the wrapper methods

Forward an optional `custom_name` display label into each step constructor, mirroring `filter_custom` (which already has it). Targets: `filter_irr`, `filter_pvsyst`, `filter_shade`, `filter_time`, `filter_days`, `filter_outliers`, `filter_pf`, `filter_power`, `filter_sensors`, `filter_clearsky`, `filter_missing`, `rep_cond`, and `fit_regression` (the `Regression` step). `filter_op_state` is skipped (builds no step).

**Files:**
- Modify: `src/captest/capdata.py`
- Modify: `tests/test_filter_classes.py`

- [ ] **Step 1: Write a failing test that `custom_name` is recorded as the step label**

Add to `tests/test_filter_classes.py` inside `class TestFilterIrr`:

```python
    def test_wrapper_custom_name_sets_step_label(self, cd_irr):
        cd_irr.filter_irr(200, 800, custom_name="my irr step")
        assert cd_irr.filters[-1].custom_name == "my irr step"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr::test_wrapper_custom_name_sets_step_label -v`
Expected: FAIL (`filter_irr` has no `custom_name` parameter → TypeError).

- [ ] **Step 3: Add `custom_name=None` to each wrapper signature and forward it**

For each wrapper, add `custom_name=None` as the last parameter and pass `custom_name=custom_name` into the step constructor. Exact edits:

`filter_irr`:
```python
    def filter_irr(self, low, high, ref_val=None, col_name=None, custom_name=None):
        ...
        flt = Irradiance(
            low=low, high=high, ref_val=ref_val, col_name=col_name,
            custom_name=custom_name,
        )
        flt.run(self)
```

`filter_pvsyst`:
```python
    def filter_pvsyst(self, custom_name=None):
        ...
        flt = Pvsyst(custom_name=custom_name)
        flt.run(self)
```

`filter_shade`:
```python
    def filter_shade(self, fshdbm=1.0, query_str=None, custom_name=None):
        ...
        flt = Shade(fshdbm=fshdbm, query_str=query_str, custom_name=custom_name)
        flt.run(self)
```

`filter_time` (add `custom_name=None` after `test_date=None,` in the signature):
```python
        flt = Time(
            start=start,
            end=end,
            drop=drop,
            days=days,
            test_date=test_date,
            custom_name=custom_name,
        )
        flt.run(self)
```

`filter_days`:
```python
    def filter_days(self, days, drop=False, custom_name=None):
        ...
        flt = Days(days=days, drop=drop, custom_name=custom_name)
        flt.run(self)
```

`filter_outliers` (keyword-only, before `**kwargs`):
```python
    def filter_outliers(self, custom_name=None, **kwargs):
        ...
        flt = Outliers(envelope_kwargs=kwargs or None, custom_name=custom_name)
        flt.run(self)
```

`filter_pf`:
```python
    def filter_pf(self, pf, custom_name=None):
        ...
        flt = PowerFactor(pf=pf, custom_name=custom_name)
        flt.run(self)
```

`filter_power`:
```python
    def filter_power(self, power, percent=None, columns=None, custom_name=None):
        ...
        flt = Power(power=power, percent=percent, columns=columns, custom_name=custom_name)
        flt.run(self)
```

`filter_sensors`:
```python
    def filter_sensors(
        self, perc_diff=None, row_filter=check_all_perc_diff_comb, custom_name=None
    ):
        ...
        flt = Sensors(perc_diff=perc_diff, row_filter=row_filter, custom_name=custom_name)
        flt.run(self)
```

`filter_clearsky` (keyword-only, before `**kwargs`):
```python
    def filter_clearsky(self, ghi_col=None, keep_clear=True, custom_name=None, **kwargs):
        ...
        flt = Clearsky(
            ghi_col=ghi_col,
            keep_clear=keep_clear,
            detect_kwargs=kwargs or None,
            custom_name=custom_name,
        )
        flt.run(self)
```

`filter_missing`:
```python
    def filter_missing(self, columns=None, custom_name=None):
        ...
        Missing(columns=columns, custom_name=custom_name).run(self)
```

`rep_cond` (add `custom_name=None` as the last parameter after `rc_kwargs=None,`):
```python
        RepCond(
            func=func,
            w_vel=w_vel,
            irr_bal=irr_bal,
            percent_filter=percent_filter,
            front_poa=front_poa,
            rc_kwargs=rc_kwargs,
            custom_name=custom_name,
        ).run(self)
```

`fit_regression` (add `custom_name=None` after `summary=True`):
```python
    def fit_regression(self, filter=False, summary=True, custom_name=None):
        ...
            flt = Regression(n_std=2, custom_name=custom_name)
            flt.run(self)
            if summary:
                print(flt.regression_model.summary())
```

- [ ] **Step 4: Add the `custom_name` docstring entry to each wrapper**

For each of the wrappers above, add this NumPy parameter entry to the `Parameters` section (matching the wording already in `filter_custom`):

```
        custom_name : str, default None
            Optional display label for the recorded filter step.
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr::test_wrapper_custom_name_sets_step_label -v`
Expected: PASS.

- [ ] **Step 6: Add one cross-wrapper test for a non-Irr wrapper**

Add to `tests/test_filter_classes.py` inside `class TestFilterTime`:

```python
    def test_wrapper_custom_name_sets_step_label(self, cd_time):
        cd_time.filter_time(start="2023-02-01", end="2023-02-15", custom_name="window")
        assert cd_time.filters[-1].custom_name == "window"
```

Run: `uv run pytest tests/test_filter_classes.py::TestFilterTime::test_wrapper_custom_name_sets_step_label -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite, lint/format**

Run: `just test && just lint && just fmt`
Expected: PASS / success.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: expose custom_name on CapData filter wrapper methods"
```

---

## Self-review notes

- **Spec coverage:** A1 (custom_name) → Task 4; A2 (remove inplace) → Tasks 2 & 3; A3 (rename) → Task 1. Clean-break YAML: Task 1 Step 2 renames `FILTER_REGISTRY` keys and the hardcoded `Custom.to_config` type string; no aliases added. No YAML/JSON fixtures embed old type strings (verified).
- **`rep_cond_freq`** keeps its `inplace` — intentionally untouched (compute-and-return, not filtering).
- **Type consistency:** new class names used in Tasks 2–4 (`Irradiance`, `Pvsyst`, `Shade`, `Time`, `Days`, `Outliers`, `PowerFactor`, `Power`, `Sensors`, `Clearsky`, `Missing`, `Regression`, `RepCond`) match the Task 1 rename map.
- **Ordering:** Task 1 (rename) must precede 2–4 so the new constructor names exist. Tasks are independent commits, reviewed sequentially.
