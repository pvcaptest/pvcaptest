# FilterIrr Example Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the first filter — `filter_irr` — to the class-based architecture: add a `FilterIrr` class, add legacy-summary mirroring to `BaseSummaryStep.run()`, and turn `CapData.filter_irr` into a thin wrapper that delegates to `FilterIrr().run(self)`.

**Architecture:** `FilterIrr` declares `low`/`high`/`ref_val`/`col_name` as `param`s; its `_execute()` resolves `col_name` (→ POA column) and `ref_val` (`'rep_irr'`/`'self_val'` → numeric from `capdata.rc`), then returns the kept index via the `filter_irr` helper. Because the other 11 filters still use `@update_summary`, `run()` must populate the **same** legacy tracking lists (`summary`, `summary_ix`, `removed`, `kept`, `filter_counts`) so `get_summary()` and the existing tests keep working during the transition. This mirroring is throwaway scaffolding removed in the summary-rebuild plan (plan 6).

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes (FilterIrr)", "Thin Wrapper Methods", "Summary Table".

**Sequencing:** Execute *after* `2026-05-24-filter-classes-base.md`.

## Key design decisions (flag if you disagree before implementing)

1. **`__get_poa_col` → `_get_poa_col` rename.** `FilterIrr._execute` lives in `filters.py` and must call the POA-column resolver on the `capdata` instance. Name-mangled `__get_poa_col` is awkward cross-module, so rename to a single-underscore internal method. Updates its one internal caller and two test references (`nrel._CapData__get_poa_col()` → `nrel._get_poa_col()`).
2. **`ref_val` resolution mutates the stored param.** `_execute` resolves `'rep_irr'`/`'self_val'` to the numeric `capdata.rc["poa"]` value and stores `float(ref_val)` back onto `self.ref_val`. This makes `args_repr` (and therefore the summary) show the resolved number — satisfying `test_refval_rep_irr_shows_in_summary` — and keeps re-runs reproducible. Trade-off: a YAML round-trip after a run serializes the resolved number, not the `'rep_irr'` sentinel. Acceptable (more reproducible); revisit if the GUI needs to preserve the sentinel.
3. **`inplace` is kept for now (deferral of the spec's removal).** The spec removes `inplace` entirely, but doing so here would touch every `filter_irr(..., inplace=...)` test call. To keep the suite green with minimal churn, the thin wrapper keeps `inplace`: `inplace=True` (default) runs the step and records it; `inplace=False` returns the filtered DataFrame **without** recording a step. Full `inplace` removal is deferred to a later cleanup plan.
4. **Legacy label via `_legacy_name`.** `BaseSummaryStep` gets a class attribute `_legacy_name = None`; `run()`'s mirroring uses `self._legacy_name or type(self).__name__`. `FilterIrr._legacy_name = "filter_irr"` so the summary label stays `filter_irr` during the transition. Removed in plan 6 when the summary switches to class names.

---

### Task 1: Add `FilterIrr` and rename `_get_poa_col`

**Files:**
- Modify: `src/captest/filters.py` (add `FilterIrr`)
- Modify: `src/captest/capdata.py` (rename `__get_poa_col` → `_get_poa_col` at def ~1541 and caller ~1900)
- Modify: `tests/test_CapData.py` (update `_CapData__get_poa_col` refs at lines 2203, 2215)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Rename `__get_poa_col` → `_get_poa_col`**

In `src/captest/capdata.py`, rename the method definition:

```python
    def _get_poa_col(self):
```

Update its caller inside the current `filter_irr` method body (the `irr_col = self.__get_poa_col()` line) to:

```python
            irr_col = self._get_poa_col()
```

In `tests/test_CapData.py`, change both `nrel._CapData__get_poa_col()` references (lines 2203 and 2215) to `nrel._get_poa_col()`.

- [ ] **Step 2: Run the rename-affected tests to confirm green**

Run: `uv run pytest tests/test_CapData.py -k "get_poa_col" -v`
Expected: PASS (2 tests) — pure rename.

- [ ] **Step 3: Write the failing `FilterIrr` tests**

Append to `tests/test_filter_classes.py`:

```python
from captest.filters import FilterIrr


class TestFilterIrr:
    def _cd(self):
        cd = CapData("irr")
        cd.data = pd.DataFrame(
            {"poa": [100.0, 300.0, 500.0, 700.0, 900.0]},
            index=pd.RangeIndex(5),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"poa": "poa"}
        return cd

    def test_execute_absolute_bounds(self):
        cd = self._cd()
        f = FilterIrr(low=200, high=800)
        kept = f._execute(cd)
        assert list(kept) == [1, 2, 3]

    def test_execute_uses_explicit_col_name(self):
        cd = self._cd()
        cd.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 1000.0]
        cd.data_filtered = cd.data.copy()
        f = FilterIrr(low=500, high=2000, col_name="ghi")
        assert list(f._execute(cd)) == [4]

    def test_execute_fraction_with_ref_val(self):
        cd = self._cd()
        # low/high are fractions of ref_val 500 -> [400, 600]
        f = FilterIrr(low=0.8, high=1.2, ref_val=500)
        assert list(f._execute(cd)) == [2]

    def test_execute_resolves_rep_irr_and_stores_float(self):
        cd = self._cd()
        cd.rc = pd.DataFrame({"poa": [500.0]})
        f = FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")
        f._execute(cd)
        assert f.ref_val == 500.0
        assert isinstance(f.ref_val, float)

    def test_execute_rep_irr_without_rc_raises(self):
        cd = self._cd()
        cd.rc = None
        with pytest.raises(ValueError, match="Call rep_cond"):
            FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd)

    def test_execute_rep_irr_without_poa_col_raises(self):
        cd = self._cd()
        cd.rc = pd.DataFrame({"irr": [500.0]})
        with pytest.raises(ValueError, match="does not have a 'poa' column"):
            FilterIrr(low=0.8, high=1.2, ref_val="rep_irr")._execute(cd)
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr -v`
Expected: FAIL — `ImportError: cannot import name 'FilterIrr'`.

- [ ] **Step 5: Implement `FilterIrr` in `filters.py`**

Append to `src/captest/filters.py`:

```python
class FilterIrr(BaseFilter):
    """Filter rows by an irradiance column to a low/high band.

    ``low``/``high`` are absolute values (W/m^2) unless ``ref_val`` is set,
    in which case they are treated as fractions of ``ref_val``.
    """

    _legacy_name = "filter_irr"

    low = param.Number(
        default=None, allow_None=True,
        doc="Lower bound (W/m^2, or fraction of ref_val when ref_val is set).",
    )
    high = param.Number(
        default=None, allow_None=True,
        doc="Upper bound (W/m^2, or fraction of ref_val when ref_val is set).",
    )
    ref_val = param.Parameter(
        default=None,
        doc="Reference value; low/high are fractions of it. May be 'rep_irr'/"
        "'self_val' to resolve from capdata.rc at run time.",
    )
    col_name = param.String(
        default=None, allow_None=True,
        doc="Irradiance column to filter. Inferred from regression_cols if None.",
    )

    def _execute(self, capdata):
        irr_col = self.col_name if self.col_name is not None else capdata._get_poa_col()

        ref_val = self.ref_val
        if ref_val == "self_val":
            ref_val = "rep_irr"
        if ref_val == "rep_irr":
            if capdata.rc is None:
                raise ValueError(
                    "ref_val='rep_irr' requires reporting conditions to be set. "
                    "Call rep_cond() before filtering with ref_val='rep_irr'."
                )
            if "poa" not in capdata.rc.columns:
                raise ValueError(
                    "ref_val='rep_irr' requires a 'poa' column in capdata.rc. "
                    "The reporting conditions DataFrame does not have a 'poa' column."
                )
            ref_val = float(capdata.rc["poa"].iloc[0])
            self.ref_val = ref_val  # store resolved value for summary + reproducibility

        return filter_irr(
            capdata.data_filtered, irr_col, self.low, self.high, ref_val=ref_val
        ).index
```

> Note: `param.Parameter` (not `param.Number`) is used for `ref_val` because it accepts both numbers and the `'rep_irr'`/`'self_val'` sentinel strings.

- [ ] **Step 6: Run the `FilterIrr` tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrr -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add src/captest/filters.py src/captest/capdata.py tests/test_CapData.py tests/test_filter_classes.py
git commit -m "feat: add FilterIrr class and rename __get_poa_col to _get_poa_col"
```

---

### Task 2: Legacy-summary mirroring in `BaseSummaryStep.run()`

**Files:**
- Modify: `src/captest/filters.py` (`BaseSummaryStep`: add `_legacy_name`, extend `run()`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing mirroring tests**

Append to `tests/test_filter_classes.py`:

```python
class TestRunLegacyMirroring:
    def _cd(self):
        cd = CapData("irr")
        cd.data = pd.DataFrame(
            {"poa": [100.0, 300.0, 500.0, 700.0, 900.0]},
            index=pd.RangeIndex(5),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"poa": "poa"}
        return cd

    def test_run_populates_legacy_summary(self):
        cd = self._cd()
        FilterIrr(low=200, high=800).run(cd)
        assert cd.summary_ix == [("irr", "filter_irr")]
        assert cd.summary[0]["pts_after_filter"] == 3
        assert cd.summary[0]["pts_removed"] == 2
        assert "low=200" in cd.summary[0]["filter_arguments"]

    def test_run_populates_removed_and_kept(self):
        cd = self._cd()
        FilterIrr(low=200, high=800).run(cd)
        assert list(cd.removed[0]["index"]) == [0, 4]
        assert list(cd.kept[0]["index"]) == [1, 2, 3]
        assert cd.removed[0]["name"] == "filter_irr"

    def test_run_enumerates_repeated_filters(self):
        cd = self._cd()
        FilterIrr(low=200, high=800).run(cd)
        FilterIrr(low=400, high=800).run(cd)
        assert [ix[1] for ix in cd.summary_ix] == ["filter_irr", "filter_irr-1"]

    def test_run_summary_shows_resolved_ref_val(self):
        cd = self._cd()
        cd.rc = pd.DataFrame({"poa": [500.0]})
        FilterIrr(low=0.8, high=1.2, ref_val="rep_irr").run(cd)
        args = cd.summary[0]["filter_arguments"]
        assert "rep_irr" not in args
        assert "np." not in args
        assert "500" in args
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestRunLegacyMirroring -v`
Expected: FAIL — `assert cd.summary_ix == [...]` fails because `run()` does not yet populate legacy lists (lists stay empty).

- [ ] **Step 3: Add `_legacy_name` and extend `run()`**

In `src/captest/filters.py`, add the class attribute to `BaseSummaryStep` (just below the `custom_name` param):

```python
    # Transitional: legacy summary label, used by run()'s mirroring until the
    # summary-rebuild plan switches the summary table to class names. Plain
    # class attribute (not a param) so it is never serialized.
    _legacy_name = None
```

Replace the end of `run()` (the `capdata.filters = ...` reassignment through the warning) with:

```python
        capdata.filters = capdata.filters + [self]
        # Transitional: keep the legacy data_filtered attribute consistent
        # until data_filtered becomes a derived property (plan 4).
        capdata.data_filtered = capdata.data.loc[self.ix_after, :]
        self._record_legacy_summary(capdata)
        if self.pts_after == 0:
            warnings.warn("The last filter removed all data!")

    def _record_legacy_summary(self, capdata):
        """Populate the legacy summary/removed/kept lists.

        Transitional scaffolding so get_summary() and the visualization
        methods keep working while filters that still use @update_summary
        coexist with class-based filters. Removed in the summary-rebuild plan.
        """
        label = self._legacy_name or type(self).__name__
        if label in capdata.filter_counts:
            capdata.filter_counts[label] += 1
            label_enum = f"{label}-{capdata.filter_counts[label] - 1}"
        else:
            capdata.filter_counts[label] = 1
            label_enum = label

        capdata.summary_ix.append((capdata.name, label_enum))
        capdata.summary.append(
            {
                "pts_after_filter": self.pts_after,
                "pts_removed": self.pts_removed,
                "filter_arguments": self.args_repr,
            }
        )
        capdata.removed.append(
            {"name": label_enum, "index": self.ix_before.difference(self.ix_after)}
        )
        capdata.kept.append({"name": label_enum, "index": self.ix_after})
```

> The enumeration mirrors `@update_summary` exactly (first occurrence has no suffix; the Nth repeat is `label-(N-1)`), sharing `capdata.filter_counts` so class-based and decorator-based filters enumerate consistently.

- [ ] **Step 4: Run the mirroring tests**

Run: `uv run pytest tests/test_filter_classes.py::TestRunLegacyMirroring -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full `test_filter_classes.py`**

Run: `uv run pytest tests/test_filter_classes.py -v`
Expected: PASS — including the `TestBaseSummaryStep` lifecycle tests (the dummy `_DropFirstRow`/`_ConfiguredFilter` use the default `_legacy_name=None` → class-name label, which is fine for those tests).

- [ ] **Step 6: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: mirror legacy summary state from BaseSummaryStep.run()"
```

---

### Task 3: Convert `CapData.filter_irr` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_irr` method ~1871-1924: drop `@update_summary`, delegate to `FilterIrr`)
- Modify: `src/captest/capdata.py` import block (add `FilterIrr`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterIrrWrapper:
    def _cd(self):
        cd = CapData("irr")
        cd.data = pd.DataFrame(
            {"poa": [100.0, 300.0, 500.0, 700.0, 900.0]},
            index=pd.RangeIndex(5),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"poa": "poa"}
        return cd

    def test_wrapper_records_filter_step(self):
        cd = self._cd()
        cd.filter_irr(200, 800)
        assert len(cd.filters) == 1
        assert isinstance(cd.filters[0], FilterIrr)
        assert list(cd.data_filtered.index) == [1, 2, 3]

    def test_wrapper_inplace_false_returns_df_without_recording(self):
        cd = self._cd()
        before = cd.data_filtered.shape[0]
        out = cd.filter_irr(200, 800, inplace=False)
        assert cd.data_filtered.shape[0] == before  # unchanged
        assert cd.filters == []  # no step recorded
        assert list(out.index) == [1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrrWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterIrr` (still the decorated method).

- [ ] **Step 3: Add `FilterIrr` to the capdata import**

In `src/captest/capdata.py`, extend the `from captest.filters import (...)` block:

```python
from captest.filters import (
    BaseSummaryStep,
    FilterIrr,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    sensor_filter,
)
```

- [ ] **Step 4: Replace the `filter_irr` method body with a thin wrapper**

Replace the entire current method (the `@update_summary` decorator line through `return df_flt`) with:

```python
    def filter_irr(self, low, high, ref_val=None, col_name=None, inplace=True):
        """
        Filter on irradiance values.

        Parameters
        ----------
        low : float or int
            Minimum value as fraction (0.8) or absolute 200 (W/m^2).
        high : float or int
            Max value as fraction (1.2) or absolute 800 (W/m^2).
        ref_val : float or int or 'rep_irr'
            Must provide arg when `low` and `high` are fractions.
            Pass ``'rep_irr'`` to use the reporting irradiance from ``self.rc``
            (set by calling :meth:`rep_cond` first).
        col_name : str, default None
            Column name of irradiance data to filter.  By default uses the POA
            irradiance set in regression_cols attribute or average of the POA
            columns.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.

        Returns
        -------
        DataFrame
            Filtered dataframe if inplace is False.
        """
        flt = FilterIrr(low=low, high=high, ref_val=ref_val, col_name=col_name)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterIrrWrapper -v`
Expected: PASS.

- [ ] **Step 6: Run the pre-existing `filter_irr` tests**

Run: `uv run pytest tests/test_CapData.py -k "filter_irr or refval or GetSummary" -v`
Expected: PASS — including `test_refval_rep_irr_shows_in_summary`, `test_refval_withcol_notinplace`, and the `self_val`/`rep_irr` cases.

- [ ] **Step 7: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS — `filter_irr` now routes through `FilterIrr.run()`; all other filters still use `@update_summary`; `get_summary()` reads the mirrored legacy lists.

- [ ] **Step 8: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: make CapData.filter_irr a thin wrapper over FilterIrr"
```

---

## Self-Review

**1. Spec coverage (this plan's slice):**
- "Concrete Filter Classes → FilterIrr" (low/high/ref_val/col_name params, `_execute` returns kept index) → Task 1. ✓
- "Thin Wrapper Methods" (`filter_irr` instantiates the class and calls `run()`) → Task 3. ✓
- "Summary Table" continuity during transition (legacy lists kept populated) → Task 2. ✓
- Deferred: `custom_name` kwarg on the wrapper (the branch has no `filter_name` feature today — added with the broader wrapper work), `inplace` removal (decision 3), the other 11 filter classes, the `function_name` summary column and class-name labels (plan 6).

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result. The two ValueError messages match the originals so `test_refval_rep_irr_rc_none_raises` / `test_refval_rep_irr_no_poa_col_raises` (which match `"Call rep_cond"` and `"does not have a 'poa' column"`) still pass.

**3. Type/name consistency:** `FilterIrr` params (`low`, `high`, `ref_val`, `col_name`), `_legacy_name`, `_record_legacy_summary`, `_get_poa_col`, and the summary keys (`pts_after_filter`, `pts_removed`, `filter_arguments`) match between `filters.py`, `capdata.py`, and the tests. `ref_val` is `param.Parameter` (accepts numbers and the sentinel strings); resolution stores a plain `float`.

**Risk note:** the same class of gap hit in plans 1-2 (source-caller analysis missing `pvc.`-style test references) is pre-checked here: the only external references to the renamed `_get_poa_col` are the two test lines (2203, 2215), updated in Task 1. After implementing, grep `_CapData__get_poa_col` to confirm none remain.
