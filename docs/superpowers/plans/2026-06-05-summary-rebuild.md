# Summary Rebuild Implementation Plan (chunk 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the filtering summary table so it is derived from the `filters` chain rather than from per-step mirroring, convert `rep_cond` into a zero-removal `RepCond` summary step, and delete the legacy `update_summary` decorator and `summary`/`summary_ix`/`filter_counts` attributes.

**Architecture:** `CapData.get_summary()` is rewritten to iterate `self.filters`, computing each step's "before" count from its predecessor's `ix_after` (`_ix_before(i)`/`_pts_before(i)`) and its display label from a shared `_step_labels()` helper. Step labels become **class names** (`FilterIrr`, `RepCond`), not snake-case method names, so the per-filter `_legacy_name` attribute is deleted. `rep_cond` becomes a thin wrapper over a new `RepCond(BaseSummaryStep)` step whose `_execute` delegates the reporting-conditions math to a new private `CapData._calc_rep_cond(...)` helper and returns the index **unchanged** (`pts_removed == 0`). The `removed`/`kept` lists stay populated (the visualization methods still read them; chunk 6 removes them), but their per-step mirroring is slimmed to drop the summary append and relabel via `_step_labels()`.

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Summary Table" (Columns, `get_summary()`, Chain-derived per-step counts, `_step_labels()`), and the `RepCond` paragraphs under "Filter Class Catalog".

**Sequencing:** Execute **after** the `data_filtered` property plan (`2026-06-04-data-filtered-property.md`, already landed at `9db72fc`). All 12 row filters are class-based; `rep_cond` is the *only* remaining `@update_summary` user, and `FilterIrr` already resolves `ref_val='rep_irr'/'self_val'` in its own `_execute`, so deleting the decorator (including its dead `ref_val` branch) loses nothing. This plan unblocks chunk 6 (visualization rewrite), which deletes `removed`/`kept`.

## Commit shape (three commits)

The work lands as **three commits**, each leaving the suite green:

1. **Task 1 — `RepCond` conversion.** Self-contained: `RepCond` records through the *existing* legacy machinery (it inherits `run()` → `_record_legacy_summary`, falling back to the class name since it has no `_legacy_name`). Green on its own.
2. **Task 2 — chain-derived helpers.** Additive `_step_labels()`/`_ix_before()`/`_pts_before()` + unit tests. Green on its own.
3. **Tasks 3–5 — summary rebuild (big-bang).** Rewriting `get_summary()` and deleting `summary`/`summary_ix`/`filter_counts` is atomic: the moment the attributes are deleted, every test reading them is red until migrated. Src deletion (Task 3) + test migration (Task 4) land in **one commit** verified by Task 5.

---

## File Structure

- `src/captest/capdata.py` — add `_calc_rep_cond`; add `_step_labels`/`_ix_before`/`_pts_before`; rewrite `get_summary`; convert `rep_cond` to a wrapper; delete the `update_summary` decorator; redefine the module `columns` constant; remove `summary`/`summary_ix`/`filter_counts` from `__init__`/`copy`/`reset_filter`/`agg_sensors`/`process_regression_columns`; add `RepCond` to the `filters` import.
- `src/captest/filters.py` — add the `RepCond` class; slim and rename `_record_legacy_summary` → `_record_removed_kept`; delete `_legacy_name` from `BaseSummaryStep` and all concrete filter classes.
- `tests/test_CapData.py` — add the `RepCond` conversion test; migrate `summary`/`summary_ix` reads to `get_summary()`; fix `test_col_names` for the new `function_name` column; fix `test_reset_summary`.
- `tests/test_filter_classes.py` — add `_step_labels`/`_ix_before`/`_pts_before` unit tests; migrate the `TestRunLegacyMirroring` summary reads and the `FilterOutliers`/`FilterRegression` nan-records tests; delete `test_legacy_name_is_filter_missing`.

---

### Task 1: Convert `rep_cond` to a `RepCond` zero-removal step

**Files:**
- Modify: `src/captest/capdata.py` (add `_calc_rep_cond`; convert `rep_cond`; import `RepCond`)
- Modify: `src/captest/filters.py` (add `RepCond`)
- Test: `tests/test_CapData.py` (new test in `TestRepCondNoFreq`)

- [ ] **Step 1: Write the failing test**

In `tests/test_CapData.py`, add to class `TestRepCondNoFreq` (the file already imports the `filters` module — see its use of `filters.abs_diff_from_average`):

```python
    def test_appends_zero_removal_step(self, nrel):
        """rep_cond records a RepCond step in the chain that removes nothing."""
        pts_before = nrel.data_filtered.shape[0]
        nrel.rep_cond()
        assert isinstance(nrel.filters[-1], filters.RepCond)
        assert nrel.filters[-1].pts_removed == 0
        assert nrel.data_filtered.shape[0] == pts_before
        assert isinstance(nrel.rc, pd.core.frame.DataFrame)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest "tests/test_CapData.py::TestRepCondNoFreq::test_appends_zero_removal_step" -v`
Expected: FAIL — `rep_cond` is still decorator-based and does not append to `filters` (so `filters[-1]` is the prior step or raises `IndexError`), and `filters.RepCond` does not exist (AttributeError).

- [ ] **Step 3: Add `_calc_rep_cond` to `CapData`**

In `src/captest/capdata.py`, add this method immediately **above** the current `def rep_cond(` (the body is the current `rep_cond` body verbatim, with `rc_kwargs=None` coerced to `{}`):

```python
    def _calc_rep_cond(self, func, w_vel, irr_bal, percent_filter, front_poa, rc_kwargs):
        """Compute reporting conditions and store them on ``self.rc``.

        Extracted verbatim from the former ``rep_cond`` body so the
        reporting-conditions math lives in one place. Called by
        ``filters.RepCond._execute`` (through the runtime ``capdata`` argument,
        so ``filters.py`` needs no import of ``capdata`` or
        ``ReportingIrradiance``) and by the thin ``rep_cond`` wrapper. Sets
        ``self.rc`` (and ``self.rc_tool`` when ``irr_bal`` is True) as a side
        effect; returns None.

        Parameters
        ----------
        func : dict, str, callable, or None
            See ``rep_cond``. When None, defaults to the mean of each
            right-hand-side variable.
        w_vel : numeric or None
            See ``rep_cond``.
        irr_bal : bool
            See ``rep_cond``.
        percent_filter : int
            See ``rep_cond``.
        front_poa : str
            See ``rep_cond``.
        rc_kwargs : dict or None
            See ``rep_cond``. None is treated as an empty dict.
        """
        if rc_kwargs is None:
            rc_kwargs = {}
        lhs, rhs = util.parse_regression_formula(self.regression_formula)
        df = self.get_reg_cols(reg_vars=rhs, filtered_data=True)

        if func is None:
            func = {var: "mean" for var in rhs}

        RCs_df = pd.DataFrame(df.agg(func)).T

        if irr_bal:
            if front_poa not in df.columns:
                raise ValueError(
                    f"front_poa={front_poa!r} is not a right-hand-side variable "
                    f"of the regression formula."
                )
            self.rc_tool = ReportingIrradiance(
                df,
                front_poa,
                percent_band=percent_filter,
                **rc_kwargs,
            )
            results = self.rc_tool.get_rep_irr()
            flt_df = results[1]
            RCs_df = pd.DataFrame(flt_df.agg(func)).T
            RCs_df.loc[RCs_df.index[0], front_poa] = results[0]

        if w_vel is not None and "w_vel" in RCs_df.columns:
            RCs_df.loc[RCs_df.index[0], "w_vel"] = w_vel

        print("Reporting conditions saved to rc attribute.")
        print(RCs_df)
        self.rc = RCs_df
```

- [ ] **Step 4: Add the `RepCond` class to `filters.py`**

In `src/captest/filters.py`, add (placement is not load-bearing; put it after `BaseFilter`/the concrete filters, e.g. directly before the module-level `FILTER_REGISTRY` if present, otherwise at the end of the class definitions):

```python
class RepCond(BaseSummaryStep):
    """Reporting-conditions calculation as a zero-removal summary step.

    Computes ``capdata.rc`` from the filtered data at this point in the chain
    and returns the index **unchanged** (``pts_removed == 0``), so the step
    appears in the summary at its position relative to the filters that
    preceded it. The reporting-conditions math is not duplicated here: it lives
    in ``CapData._calc_rep_cond``, reached via the runtime ``capdata`` argument
    so ``filters.py`` needs no import of ``capdata`` or ``ReportingIrradiance``.
    Inherits ``BaseSummaryStep`` directly (not ``BaseFilter``) because it is not
    a row filter; it still belongs in ``capdata.filters`` because that list
    accepts any ``BaseSummaryStep``.
    """

    func = param.Parameter(
        default=None,
        doc="Aggregation(s) for each rhs variable: dict/str/callable/None.",
    )
    w_vel = param.Parameter(
        default=None,
        doc="Override for the wind-speed reporting condition.",
    )
    irr_bal = param.Boolean(
        default=False,
        doc="Use ReportingIrradiance to balance the irradiance band.",
    )
    percent_filter = param.Number(
        default=20,
        doc="Percent band around the reporting irradiance (irr_bal only).",
    )
    front_poa = param.String(
        default="poa",
        doc="regression_cols key used as the irradiance driver (irr_bal only).",
    )
    rc_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Extra kwargs forwarded to ReportingIrradiance (irr_bal only).",
    )

    _explanation_template = (
        "Reporting conditions were calculated (no intervals removed)."
    )

    def _execute(self, capdata):
        capdata._calc_rep_cond(
            self.func,
            self.w_vel,
            self.irr_bal,
            self.percent_filter,
            self.front_poa,
            self.rc_kwargs,
        )
        return capdata.data_filtered.index
```

- [ ] **Step 5: Import `RepCond` in `capdata.py`**

In `src/captest/capdata.py`, add `RepCond` to the existing `from captest.filters import (...)` block (keep the block alphabetized; insert after `FilterTime` and before `check_all_perc_diff_comb`):

```python
    FilterTime,
    RepCond,
    check_all_perc_diff_comb,
```

- [ ] **Step 6: Convert `rep_cond` to a thin wrapper**

In `src/captest/capdata.py`, replace the `@update_summary`-decorated `def rep_cond(...)` (the whole method, decorator line included) with the wrapper below. **Preserve the existing NumPy docstring** between the triple-quotes exactly as it is today, except change the `rc_kwargs` parameter entry to read `rc_kwargs : dict or None, default None` (its default is now `None`, coerced to `{}` by `_calc_rep_cond`):

```python
    def rep_cond(
        self,
        func=None,
        w_vel=None,
        irr_bal=False,
        percent_filter=20,
        front_poa="poa",
        rc_kwargs=None,
    ):
        """<keep the current rep_cond docstring verbatim, with the rc_kwargs
        entry changed to 'dict or None, default None'>"""
        RepCond(
            func=func,
            w_vel=w_vel,
            irr_bal=irr_bal,
            percent_filter=percent_filter,
            front_poa=front_poa,
            rc_kwargs=rc_kwargs,
        ).run(self)
```

> The decorator function `update_summary` itself is **not** deleted here — `rep_cond` was its only remaining user, but the helpers it relies on (`round_kwarg_floats`, `tstamp_kwarg_to_strings`) and the function are removed in Task 3 to keep this commit minimal and green.

- [ ] **Step 7: Run the new test and the existing rep_cond suite**

Run: `uv run pytest "tests/test_CapData.py::TestRepCondNoFreq" -v`
Expected: PASS (all of `test_defaults`, `test_defaults_wvel`, `test_irr_bal`, `test_irr_bal_wvel`, `test_custom_func_dict`, and the new `test_appends_zero_removal_step`).

- [ ] **Step 8: Full suite, lint, commit**

```bash
just test-wo-warnings
just lint && just fmt
git add -A
git commit -m "refactor: convert rep_cond to a RepCond zero-removal step"
```

Expected: suite green. If `test_custom_func_dict` regresses, confirm `RepCond.func` (a `param.Parameter`) accepts a dict containing a callable — it does, because `param.Parameter` performs no type coercion.

---

### Task 2: Add chain-derived label/count helpers

**Files:**
- Modify: `src/captest/capdata.py` (add `_step_labels`, `_ix_before`, `_pts_before`)
- Test: `tests/test_filter_classes.py` (new test class)

- [ ] **Step 1: Write the failing tests**

In `tests/test_filter_classes.py`, add a new test class. Use the `cd_irr` fixture (5-row data, name `"irr"`, already used by `TestRunLegacyMirroring`):

```python
class TestChainDerivedHelpers:
    def test_step_labels_enumerate_repeats(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert cd_irr._step_labels() == ["FilterIrr", "FilterIrr-1"]

    def test_step_labels_use_custom_name(self, cd_irr):
        FilterIrr(low=200, high=800, custom_name="Irradiance bounds").run(cd_irr)
        assert cd_irr._step_labels() == ["Irradiance bounds"]

    def test_pts_before_first_step_is_full_data(self, cd_irr):
        full = cd_irr.data.shape[0]
        FilterIrr(low=200, high=800).run(cd_irr)
        assert cd_irr._pts_before(0) == full

    def test_ix_before_second_step_is_prior_ix_after(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert list(cd_irr._ix_before(1)) == list(cd_irr.filters[0].ix_after)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest "tests/test_filter_classes.py::TestChainDerivedHelpers" -v`
Expected: FAIL — `_step_labels`/`_ix_before`/`_pts_before` are not defined (AttributeError).

- [ ] **Step 3: Add the helpers to `CapData`**

In `src/captest/capdata.py`, add these three methods (place them immediately above `def get_summary`):

```python
    def _ix_before(self, i):
        """Index passed *into* ``self.filters[i]`` (chain state just before it).

        The prior step's ``ix_after``, or ``self.data.index`` for the first step.
        """
        return self.filters[i - 1].ix_after if i > 0 else self.data.index

    def _pts_before(self, i):
        """Row count passed into ``self.filters[i]`` (see ``_ix_before``)."""
        return len(self._ix_before(i))

    def _step_labels(self):
        """Per-step display labels for the summary and visualization methods.

        Each label is the step's ``custom_name`` if set, otherwise its class
        name, with a ``-N`` suffix disambiguating repeated steps (the first
        occurrence is unsuffixed). Single source of the enumerated labels shared
        by ``get_summary`` and (in chunk 6) ``scatter_filters``/
        ``timeseries_filters``.
        """
        labels, seen = [], {}
        for step in self.filters:
            base = step.custom_name or type(step).__name__
            n = seen.get(base, 0)
            seen[base] = n + 1
            labels.append(base if n == 0 else f"{base}-{n}")
        return labels
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest "tests/test_filter_classes.py::TestChainDerivedHelpers" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Full suite, lint, commit**

```bash
just test-wo-warnings
just lint && just fmt
git add -A
git commit -m "feat: add chain-derived step label/count helpers to CapData"
```

Expected: suite green (the helpers are additive; nothing else changed).

---

### Task 3: Rewrite `get_summary` and delete the legacy summary machinery (src)

**Files:**
- Modify: `src/captest/capdata.py`
- Modify: `src/captest/filters.py`

> **Big-bang note:** This task deletes `summary`/`summary_ix`/`filter_counts` and the `update_summary` decorator. The suite goes red until Task 4 migrates the tests; both tasks land in the single commit made in Task 5.

- [ ] **Step 1: Redefine the module `columns` constant**

In `src/captest/capdata.py`, change the constant (currently `columns = ["pts_after_filter", "pts_removed", "filter_arguments"]`) to prepend `function_name`:

```python
columns = ["function_name", "pts_after_filter", "pts_removed", "filter_arguments"]
```

- [ ] **Step 2: Delete the `update_summary` decorator**

In `src/captest/capdata.py`, delete the entire `def update_summary(func):` function (the decorator and its `wrapper`, roughly the block from `def update_summary(func):` through its `return wrapper`). Leave `round_kwarg_floats` and `tstamp_kwarg_to_strings` in place — they are still exercised by `tests/test_CapData.py::TestUpdateSummary` and are otherwise harmless utilities (now unused by src). Remove the now-unused `from functools import wraps` import **only if** nothing else in the file uses `wraps` (grep first: `grep -n "wraps" src/captest/capdata.py`).

- [ ] **Step 3: Rewrite `get_summary`**

In `src/captest/capdata.py`, replace the body of `get_summary` (keep/extend the docstring) with the chain-derived version. Return an empty DataFrame with the standard columns when no filters have run (so `CapTest.get_summary`'s `pd.concat` of two CapData summaries still works):

```python
    def get_summary(self):
        """Return a DataFrame summarizing the applied filter chain.

        Rebuilt from ``self.filters``: one row per step, with the step's class
        name (``function_name``), the rows remaining after it
        (``pts_after_filter``), the rows it removed (``pts_removed``, derived
        from the prior step's ``ix_after`` via ``_pts_before``), and its
        rendered arguments (``filter_arguments``). The row index is a MultiIndex
        of ``(self.name, label)`` where ``label`` comes from ``_step_labels``.

        Returns an empty DataFrame (standard columns, no rows) when no filters
        have been applied.

        Returns
        -------
        pandas.DataFrame
        """
        if not self.filters:
            return pd.DataFrame(columns=columns)
        rows = []
        index = []
        for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
            index.append((self.name, label))
            pts_before = self._pts_before(i)
            rows.append(
                {
                    "function_name": type(step).__name__,
                    "pts_after_filter": step.pts_after,
                    "pts_removed": pts_before - step.pts_after,
                    "filter_arguments": step.args_repr,
                }
            )
        return pd.DataFrame(
            rows, index=pd.MultiIndex.from_tuples(index), columns=columns
        )
```

- [ ] **Step 4: Remove legacy summary fields from `__init__`**

In `CapData.__init__`, delete these three lines (keep `self.removed = []` and `self.kept = []`):

```python
        self.summary_ix = []
        self.summary = []
        self.filter_counts = {}
```

- [ ] **Step 5: Remove legacy summary copies from `copy()`**

In `CapData.copy()`, delete:

```python
        cd_c.summary_ix = copy.copy(self.summary_ix)
        cd_c.summary = copy.copy(self.summary)
```

(`cd_c.filters = copy.deepcopy(self.filters)` below them remains — the summary derives from it.)

- [ ] **Step 6: Simplify `reset_filter()`**

In `CapData.reset_filter()`, delete the three legacy lines, keeping the `removed`/`kept`/`filters` resets:

```python
        self.summary_ix = []
        self.summary = []
        self.filter_counts = {}
```

The method body becomes:

```python
        self.removed = []
        self.kept = []
        self.filters = []
```

- [ ] **Step 7: Update the `agg_sensors` guard and reset**

In `CapData.agg_sensors`, change the guard from `if not len(self.summary) == 0:` to test the chain, and replace the two-line legacy reset with `removed`/`kept` resets (the method already sets `self.filters = []` for the data change):

Change:
```python
        if not len(self.summary) == 0:
            warnings.warn(
```
to:
```python
        if self.filters:
            warnings.warn(
```

Then replace:
```python
        # reset summary data
        self.summary_ix = []
        self.summary = []
```
with:
```python
        # reset filter-history mirrors (filters itself is cleared below)
        self.removed = []
        self.kept = []
```

- [ ] **Step 8: Update the `process_regression_columns` guard and reset**

In `CapData.process_regression_columns`, apply the identical change as Step 7:

Change `if not len(self.summary) == 0:` → `if self.filters:`, and replace:
```python
        # reset summary data
        self.summary_ix = []
        self.summary = []
```
with:
```python
        # reset filter-history mirrors (filters itself is cleared below)
        self.removed = []
        self.kept = []
```

(`self.filters = []` already runs later in this method — verify with `grep -n "self.filters = \[\]" src/captest/capdata.py`.)

- [ ] **Step 9: Slim and rename `_record_legacy_summary` → `_record_removed_kept`**

In `src/captest/filters.py`, replace the `_record_legacy_summary` method on `BaseSummaryStep` with the slimmed version below (drops the `summary`/`summary_ix`/`filter_counts` appends; relabels via `_step_labels()`; keeps `removed`/`kept`):

```python
    def _record_removed_kept(self, capdata):
        """Append this step's removed/kept index entries for the viz methods.

        Transitional: ``scatter_filters``/``timeseries_filters`` still read
        ``capdata.removed``/``capdata.kept``. The summary table is rebuilt from
        the filter chain by ``CapData.get_summary`` and is no longer mirrored
        here. Both lists are removed when the visualization methods are
        rewritten to derive removed-by-filter from the chain (chunk 6).

        ``self`` has already been appended to ``capdata.filters`` by ``run`` at
        this point, so ``capdata._step_labels()[-1]`` is this step's label.
        """
        label = capdata._step_labels()[-1]
        capdata.removed.append(
            {"name": label, "index": self.ix_before.difference(self.ix_after)}
        )
        capdata.kept.append({"name": label, "index": self.ix_after})
```

- [ ] **Step 10: Update the `run()` call site**

In `src/captest/filters.py`, in `BaseSummaryStep.run`, change:

```python
        self._record_legacy_summary(capdata)
```

to:

```python
        self._record_removed_kept(capdata)
```

Also update the `run` docstring paragraph that references the "summary-rebuild plan" deriving counts — it is now done; reword to: *"``ix_before``/``pts_before`` reflect the prior chain state; the summary itself is derived from the chain by ``CapData.get_summary``."* (Keep the nested-filter-call explanation about `FilterOutliers`.)

- [ ] **Step 11: Delete `_legacy_name` everywhere**

In `src/captest/filters.py`, delete every `_legacy_name` declaration — the base `_legacy_name = None` on `BaseSummaryStep` and the `_legacy_name = "filter_x"` line in each concrete filter class (`FilterIrr`, `FilterSensors`, `FilterTime`, `FilterCustom`, `FilterOutliers`, `FilterClearsky`, `FilterPvsyst`, `FilterShade`, `FilterPf`, `FilterPower`, `FilterDays`, `FilterMissing`, `FilterRegression`). Verify none remain:

Run: `grep -rn "_legacy_name" src/captest/`
Expected: empty.

- [ ] **Step 12: Confirm the package imports**

Run: `uv run python -c "import captest; from captest.filters import RepCond; print('ok')"`
Expected: prints `ok`, no error. (The suite is now red until Task 4 — expected.)

---

### Task 4: Migrate the tests reading legacy summary fields

**Files:**
- Modify: `tests/test_filter_classes.py`
- Modify: `tests/test_CapData.py`

All sites that read `summary`/`summary_ix`/`_legacy_name` are migrated to `get_summary()`/`filters`, and snake-case label assertions become class names.

- [ ] **Step 1: `tests/test_filter_classes.py` — `TestRunLegacyMirroring`**

Rename `test_run_populates_legacy_summary` to `test_run_populates_summary` and rewrite its body; fix the label in `test_run_populates_removed_and_kept`; rewrite `test_run_enumerates_repeated_filters` and `test_run_summary_shows_resolved_ref_val`:

```python
    def test_run_populates_summary(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        gs = cd_irr.get_summary()
        assert list(gs.index) == [("irr", "FilterIrr")]
        assert gs["pts_after_filter"].iloc[0] == 3
        assert gs["pts_removed"].iloc[0] == 2
        assert "low=200" in gs["filter_arguments"].iloc[0]

    def test_run_populates_removed_and_kept(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        assert list(cd_irr.removed[0]["index"]) == [0, 4]
        assert list(cd_irr.kept[0]["index"]) == [1, 2, 3]
        assert cd_irr.removed[0]["name"] == "FilterIrr"

    def test_run_enumerates_repeated_filters(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        FilterIrr(low=400, high=800).run(cd_irr)
        assert [ix[1] for ix in cd_irr.get_summary().index] == [
            "FilterIrr",
            "FilterIrr-1",
        ]

    def test_run_summary_shows_resolved_ref_val(self, cd_irr):
        cd_irr.rc = pd.DataFrame({"poa": [500.0]})
        FilterIrr(low=0.8, high=1.2, ref_val="rep_irr").run(cd_irr)
        args = cd_irr.get_summary()["filter_arguments"].iloc[0]
        assert "rep_irr" not in args
        assert "np." not in args
        assert "500" in args
```

- [ ] **Step 2: `tests/test_filter_classes.py` — `FilterOutliers` nan-records test**

Rewrite `test_execute_nan_calls_filter_missing` (the `cd_pp` one) to read `get_summary()` and the class name, and update the stale comment (FilterMissing is now class-based):

```python
    def test_execute_nan_calls_filter_missing(self, cd_pp):
        cd_pp.data.iloc[0, cd_pp.data.columns.get_loc("poa")] = np.nan
        with pytest.warns(UserWarning, match="missing values"):
            kept = FilterOutliers()._execute(cd_pp)
        assert 0 not in kept
        # The nested filter_missing is recorded as its own step in the chain.
        gs = cd_pp.get_summary()
        assert len(gs) == 1
        assert gs.index[0][1] == "FilterMissing"
```

- [ ] **Step 3: `tests/test_filter_classes.py` — `test_pts_removed_excludes_nan_drop`**

Rewrite to read a local `gs = cd_pp.get_summary()`:

```python
    def test_pts_removed_excludes_nan_drop(self, cd_pp):
        cd_pp.data.iloc[1, cd_pp.data.columns.get_loc("poa")] = np.nan
        pre_run_pts = len(cd_pp.data_filtered)  # includes the NaN row
        f = FilterOutliers()
        with pytest.warns(UserWarning):
            f.run(cd_pp)
        gs = cd_pp.get_summary()
        assert gs.index[-2][1] == "FilterMissing"
        assert gs.index[-1][1] == "FilterOutliers"
        assert f.pts_before == gs["pts_after_filter"].iloc[-2]
        assert (
            gs["pts_removed"].iloc[-1]
            == gs["pts_after_filter"].iloc[-2] - gs["pts_after_filter"].iloc[-1]
        )
        assert gs["pts_removed"].iloc[-1] < (
            pre_run_pts - gs["pts_after_filter"].iloc[-1]
        )
```

- [ ] **Step 4: `tests/test_filter_classes.py` — delete the `_legacy_name` test**

Delete the whole method:

```python
    def test_legacy_name_is_filter_missing(self):
        assert FilterMissing._legacy_name == "filter_missing"
```

- [ ] **Step 5: `tests/test_filter_classes.py` — `FilterRegression` nan-records test**

In `TestFilterRegression::test_execute_nan_calls_filter_missing`, change the final assertion:

```python
        assert cd_reg.get_summary().index[-1][1] == "FilterMissing"
```

(replacing `assert cd_reg.summary_ix[-1][1] == "filter_missing"`).

- [ ] **Step 6: `tests/test_CapData.py` — `test_reset_summary`**

Rewrite to assert the chain is empty after `agg_sensors` (the summary derives from it):

```python
    def test_reset_summary(self, meas):
        meas.agg_sensors()
        # Aggregation clears the filter chain (and thus the summary).
        assert len(meas.filters) == 0
```

(replacing the two `len(meas.summary*)` assertions).

- [ ] **Step 7: `tests/test_CapData.py` — `TestGetSummary::test_col_names`**

Update for the new leading `function_name` column:

```python
    def test_col_names(self, nrel):
        nrel.filter_irr(200, 500)
        smry = nrel.get_summary()
        assert smry.columns[0] == "function_name"
        assert smry.columns[1] == "pts_after_filter"
        assert smry.columns[2] == "pts_removed"
        assert smry.columns[3] == "filter_arguments"
```

- [ ] **Step 8: `tests/test_CapData.py` — `test_filter_outliers_nan_records_filter_missing_in_summary`**

Rewrite to read `get_summary().index` and class-name labels:

```python
    def test_filter_outliers_nan_records_filter_missing_in_summary(self, pvsyst):
        """When filter_outliers auto-calls filter_missing, both are recorded in summary."""
        pvsyst.data.iloc[0, pvsyst.data.columns.get_loc("GlobInc")] = np.nan

        with pytest.warns(UserWarning):
            pvsyst.filter_outliers()

        filter_names = [ix[1] for ix in pvsyst.get_summary().index]
        assert filter_names.index("FilterMissing") < filter_names.index(
            "FilterOutliers"
        )
```

---

### Task 5: Full-suite verification and commit (Tasks 3–4)

**Files:** all of the above.

- [ ] **Step 1:** `just test-wo-warnings` → all pass. Investigate any failure (most likely a missed `summary`/`summary_ix` read or a stale snake-case label).
- [ ] **Step 2:** `just lint && just fmt` → clean.
- [ ] **Step 3:** Confirm no legacy summary references remain in src or tests:
  - `grep -rnE "\.summary_ix|\.filter_counts|\bupdate_summary\b|_legacy_name" src/captest/ tests/ | grep -v ".ipynb_checkpoints"` → empty.
  - `grep -rnE "\.summary\b" src/captest/ tests/ | grep -v "get_summary|summary_check|\.ipynb_checkpoints"` → empty (no remaining reads of the deleted `summary` list; `get_summary` is fine).
- [ ] **Step 4:** Commit Tasks 3–4 together:

```bash
git add -A
git commit -m "refactor: rebuild summary table from the filter chain; drop update_summary"
```

---

## Self-Review

**1. Spec coverage** (`docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md`):
- "Summary Table → Columns" (`function_name` first) → Task 3 Step 1 (`columns` const) + Step 3 (`get_summary` writes `function_name`). ✓
- "Summary Table → `get_summary()`" (iterate `filters`, lazy enumeration, chain-derived `pts_removed`) → Task 3 Step 3. ✓
- "Chain-derived per-step counts" (`_ix_before`/`_pts_before`) → Task 2 Step 3. ✓ The spec's note that nested-filter steps attribute correctly is exercised by Task 4 Steps 2–3 & 5 (FilterOutliers/FilterRegression → FilterMissing). ✓
- "`_step_labels()`" (shared label enumeration, `custom_name` or class, `-N` repeats) → Task 2 Step 3; consumed by `get_summary` (Task 3) and `_record_removed_kept` (Task 3 Step 9). ✓
- `RepCond` (BaseSummaryStep, zero-removal, delegates to `CapData._calc_rep_cond`, returns unchanged index, `_explanation_template`) → Task 1 Steps 3–4. ✓
- `RepCond` params (func/w_vel `param.Parameter`, irr_bal `param.Boolean`, percent_filter `param.Number`, front_poa `param.String`, rc_kwargs `param.Dict(default=None)`→`{}` in helper) → Task 1 Step 4 (class) + Step 3 (helper coercion). ✓
- `rep_cond(...)` becomes `RepCond(...).run(self)` → Task 1 Step 6. ✓
- "`filter_counts` removed from `__init__` and `reset_filter`" → Task 3 Steps 4 & 6. ✓
- Visualization `removed`/`kept` retained through this chunk (deleted in chunk 6) → Task 3 Steps 9–10 keep them; guards in Steps 7–8 reset them with `filters`. ✓

**2. Placeholder scan:** No TBDs. Every src change shows complete before/after code. The one "keep the docstring verbatim" instruction (Task 1 Step 6) is deliberate — the existing `rep_cond` docstring is long and unchanged except the one `rc_kwargs` line called out explicitly. Every test edit shows the full replacement method.

**3. Type/name consistency:** `_step_labels`/`_ix_before`/`_pts_before` (Task 2) are used by `get_summary` and `_record_removed_kept` (Task 3) with matching names. `RepCond` (filters.py, Task 1 Step 4) matches the import (Task 1 Step 5) and the wrapper (Task 1 Step 6) and the test (Task 1 Step 1). `_calc_rep_cond`'s signature `(func, w_vel, irr_bal, percent_filter, front_poa, rc_kwargs)` matches `RepCond._execute`'s call. `columns` (4 elements) matches `get_summary`'s row dict keys and `test_col_names`. Summary labels are class names everywhere (`FilterIrr`, `FilterMissing`, `FilterOutliers`, `RepCond`) — `_record_removed_kept` and `get_summary` both source them from `_step_labels()`.

**Deliberate decisions / deviations:**
- `update_summary`, `round_kwarg_floats`, `tstamp_kwarg_to_strings` — the decorator is deleted (Task 3 Step 2) but the two helpers stay, since `TestUpdateSummary` still tests them directly and they are harmless. (They become src-unused; chunk 7 or a later cleanup can revisit.)
- `rep_cond`'s `rc_kwargs` default changes `{}` → `None` (mutable-default avoidance; `_calc_rep_cond` coerces). Behaviorally identical for callers passing nothing.
- `RepCond` now appears in `filters`/`removed`/`kept`/`get_summary` after `rep_cond()` — previously the decorator recorded it in `summary`/`removed`/`kept` but **not** in `filters`. No test asserts `rep_cond`'s presence/absence in the summary or a post-`rep_cond` `filters` length, so this is safe. The viz methods (chunk 6) will skip zero-removal steps.
- `agg_sensors`/`process_regression_columns` now also reset `removed`/`kept` (not just `summary`), keeping the viz mirrors consistent with the `filters = []` they already perform.
