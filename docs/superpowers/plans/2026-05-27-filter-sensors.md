# FilterSensors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `filter_sensors` to the class-based architecture as `FilterSensors`, handling its `row_filter` **callable** and `perc_diff` **dict** parameters.

**Architecture:** `FilterSensors` declares `perc_diff` (a `param.Dict`) and `row_filter` (a `param.Callable`). `_execute()` resolves a `None` `perc_diff` to `{<poa group>: 0.05}` (stored on a runtime attr for display), then applies `sensor_filter` per sensor group, intersecting the kept indices. `CapData.filter_sensors` becomes a thin wrapper. The explanation/summary machinery (`_record_legacy_summary`, `explanation`, `_args_for_repr`) already exists from the FilterIrr plan; this plan adds per-class overrides so the callable renders by name and the resolved `perc_diff` shows through.

**Tech Stack:** Python, `param`, pandas, scikit-learn-independent (pure pandas helper), pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Special Cases (FilterSensors)", "Callable Parameters", "Thin Wrapper Methods".

**Sequencing:** Execute *after* `2026-05-24-filter-irr-example.md` (needs `BaseSummaryStep`, `_record_legacy_summary`, the explanation hooks, and the `sensor_filter`/`check_all_perc_diff_comb`/`abs_diff_from_average` helpers already in `filters.py`).

## Key design decisions (flag if you disagree before implementing)

1. **`row_filter` is a `param.Callable`, not a plain attribute.** The spec's "Special Cases" describes storing `row_filter` as a plain attribute serialized as a module-qualified name. But a plain attribute can't be passed through `param.Parameterized.__init__` without overriding `__init__`. Using `param.Callable(default=check_all_perc_diff_comb)` keeps the constructor uniform (`FilterSensors(row_filter=abs_diff_from_average)` just works) and is GUI-friendly. **This does not change the serialization strategy** — the YAML plan still serializes callables as module-qualified name strings (per the spec's "Callable Parameters" table); it just special-cases `param.Callable` values in `to_yaml`/`load_pipeline` instead of plain attributes. Flagged because it diverges from the spec's literal wording.
2. **`row_filter` renders by name in summaries.** A raw callable renders as `<function check_all_perc_diff_comb at 0x...>`. `FilterSensors` overrides `_args_for_repr` to substitute `row_filter.__name__`, and supplies the same to the explanation — reproducing what the legacy `@update_summary` regex did.
3. **`perc_diff=None` resolves at run time and is stored for display.** Like `FilterIrr.ref_val`, the `perc_diff` param keeps the user's intent (`None` = "use the default poa threshold"); `_execute` resolves it to `{<poa group key>: 0.05}` and stores the resolved dict on `self.perc_diff_resolved` (runtime attr, not a param) for the summary and explanation.
4. **`inplace` kept for now.** Same transitional decision as FilterIrr: `inplace=True` records the step; `inplace=False` returns the filtered frame without recording. Full removal deferred.

---

### Task 1: Add `FilterSensors` to `filters.py`

**Files:**
- Modify: `src/captest/filters.py` (add `FilterSensors`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

Extend the top-of-file import (do **not** add a mid-file import — no `E402` exemption):

```python
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterIrr,
    FilterSensors,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)
```

Then append a `TestFilterSensors` class. These tests reuse the existing `capdata_irr` conftest fixture (4 POA sensor columns `poa1..poa4`, `column_groups={"poa": [...]}`):

```python
class TestFilterSensors:
    def test_execute_default_perc_diff_resolves(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = FilterSensors()
        kept = f._execute(capdata_irr)
        # tightly-clustered random data (876-900) is within the 5% default,
        # so no rows are removed
        assert list(kept) == list(capdata_irr.data_filtered.index)
        assert f.perc_diff_resolved == {"poa": 0.05}

    def test_execute_explicit_row_filter_drops_outliers(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        capdata_irr.data_filtered = capdata_irr.data.copy()
        f = FilterSensors(perc_diff={"poa": 25}, row_filter=abs_diff_from_average)
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_row_filter_defaults_to_check_all_perc_diff_comb(self):
        assert FilterSensors().row_filter is check_all_perc_diff_comb

    def test_args_repr_renders_row_filter_by_name(self):
        f = FilterSensors(perc_diff={"poa": 0.05})
        args = f.args_repr
        assert "row_filter=check_all_perc_diff_comb" in args
        assert "<function" not in args

    def test_explanation_names_group_and_row_filter(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = FilterSensors()
        f.run(capdata_irr)
        exp = f.explanation
        assert "poa" in exp
        assert "check_all_perc_diff_comb" in exp
        assert exp.endswith("were removed.")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterSensors -v`
Expected: FAIL — `ImportError: cannot import name 'FilterSensors'`.

- [ ] **Step 3: Implement `FilterSensors`**

Append to `src/captest/filters.py`:

```python
class FilterSensors(BaseFilter):
    """Drop rows where redundant sensors in a group disagree beyond a threshold.

    For each sensor group named in ``perc_diff``, ``row_filter`` is applied
    across that group's columns row-by-row; rows flagged inconsistent are
    removed. Ignores columns generated by ``agg_sensors`` by operating on the
    pre-aggregation columns when present.
    """

    _legacy_name = "filter_sensors"
    _explanation_template = (
        "Rows with inconsistent readings within sensor group(s) {groups} "
        "(compared using {row_filter}) were removed."
    )

    perc_diff = param.Dict(
        default=None,
        allow_None=True,
        doc="Map of sensor-group key -> percent-difference threshold "
        "(e.g. {'irr-poa-': 0.05}). None uses {<poa group>: 0.05}.",
    )
    row_filter = param.Callable(
        default=check_all_perc_diff_comb,
        doc="Row-wise consistency check applied across a group's columns.",
    )

    def _execute(self, capdata):
        if capdata.pre_agg_cols is not None:
            df = capdata.data_filtered[capdata.pre_agg_cols]
            trans = capdata.pre_agg_trans
            regression_cols = capdata.pre_agg_reg_trans
        else:
            df = capdata.data_filtered
            trans = capdata.column_groups
            regression_cols = capdata.regression_cols

        perc_diff = self.perc_diff
        if perc_diff is None:
            perc_diff = {regression_cols["poa"]: 0.05}
        self.perc_diff_resolved = perc_diff

        index = None
        for key, threshold in perc_diff.items():
            sensors_df = df[trans[key]]
            next_index = sensor_filter(sensors_df, threshold, row_filter=self.row_filter)
            index = next_index if index is None else index.intersection(next_index)
        return index

    def _args_for_repr(self):
        vals = dict(self.param.values())
        vals["row_filter"] = self.row_filter.__name__
        resolved = getattr(self, "perc_diff_resolved", None)
        if resolved is not None:
            vals["perc_diff"] = resolved
        return vals

    def _explanation_values(self):
        return {
            "groups": ", ".join(self.perc_diff_resolved),
            "row_filter": self.row_filter.__name__,
        }
```

> Notes:
> - `param.Callable` accepts a function as default and via the constructor, so `FilterSensors(row_filter=abs_diff_from_average)` works without a custom `__init__`.
> - `_execute` rewrites the legacy `"index" in locals()` pattern as an explicit `index = None` accumulator (same intersection semantics).
> - The returned index is a subset of `capdata.data_filtered.index`; `run()` reindexes `data_filtered` from `capdata.data` transitionally, which preserves any `agg_sensors` columns (they live on `capdata.data`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterSensors -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add FilterSensors class with param.Callable row_filter"
```

---

### Task 2: Convert `CapData.filter_sensors` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_sensors` method ~2327-2381: drop `@update_summary`, delegate to `FilterSensors`; in the `from captest.filters import (...)` block, add `FilterSensors` and drop `sensor_filter` — the only `sensor_filter` callers were inside the removed method body)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterSensorsWrapper:
    def test_wrapper_records_filtersensors_step(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        capdata_irr.filter_sensors()
        assert len(capdata_irr.filters) == 1
        assert isinstance(capdata_irr.filters[0], FilterSensors)

    def test_wrapper_inplace_false_records_no_step(self, capdata_irr):
        result = capdata_irr.filter_sensors(
            perc_diff={"poa": 0.05}, inplace=False
        )
        assert capdata_irr.filters == []
        assert result.shape[0] == capdata_irr.data_filtered.shape[0]
        assert capdata_irr.data_filtered.shape[0] == capdata_irr.data.shape[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterSensorsWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterSensors` (still the decorated method).

- [ ] **Step 3: Add `FilterSensors` to the capdata import**

In `src/captest/capdata.py`, edit the import block: **add `FilterSensors`** and **drop `sensor_filter`** (after Task 2 Step 4 removes the method body, `sensor_filter` is no longer called anywhere in `capdata.py` — leaving it imported would fail `ruff` `F401`). `check_all_perc_diff_comb` stays because it remains the wrapper's default arg.

```python
from captest.filters import (
    BaseSummaryStep,
    FilterIrr,
    FilterSensors,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
)
```

- [ ] **Step 4: Replace the `filter_sensors` method body with a thin wrapper**

Replace the entire current method (the `@update_summary` line through its final `return df_out`) with:

```python
    def filter_sensors(
        self, perc_diff=None, inplace=True, row_filter=check_all_perc_diff_comb
    ):
        """
        Drop suspicious measurments by comparing values from different sensors.

        This method ignores columns generated by the agg_sensors method.

        Parameters
        ----------
        perc_diff : dict
            Dictionary to specify a different threshold for
            each group of sensors.  Dictionary keys should be translation
            dictionary keys and values are floats, like {'irr-poa-': 0.05}.
            By default the poa sensors as set by the regression_cols dictionary
            are filtered with a 5% percent difference threshold.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        row_filter : callable, default check_all_perc_diff_comb
            Row-wise consistency check applied across a group's columns.

        Returns
        -------
        DataFrame
            Returns filtered dataframe if inplace is False.
        """
        flt = FilterSensors(perc_diff=perc_diff, row_filter=row_filter)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterSensorsWrapper -v`
Expected: PASS.

- [ ] **Step 6: Run the pre-existing filter_sensors suite**

Run: `uv run pytest tests/test_CapData.py -k "FilterSensors or filter_sensors or sensor_filter or AbsDiff" -v`
Expected: PASS — including `TestFilterSensors::test_perc_diff_none`, `test_perc_diff`, `test_after_agg_sensors` (verifies the `power_inv_sum_agg` column survives), and `TestFilterSensorsWithAbsDiffFromAverage`.

- [ ] **Step 7: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS — `filter_sensors` now routes through `FilterSensors.run()`; other filters unchanged.

- [ ] **Step 8: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: make CapData.filter_sensors a thin wrapper over FilterSensors"
```

---

## Self-Review

**1. Spec coverage (this filter):**
- "Special Cases → FilterSensors" (`row_filter` callable default `check_all_perc_diff_comb`) → Task 1, as a `param.Callable` (decision 1). ✓
- "Thin Wrapper Methods" → Task 2. ✓
- "Callable Parameters" serialization is deferred to the YAML plan; this plan only ensures the in-memory model and name-based display. Noted.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result.

**3. Type/name consistency:** `FilterSensors` params (`perc_diff`, `row_filter`), the runtime attr `perc_diff_resolved`, the overrides (`_execute`, `_args_for_repr`, `_explanation_values`), `_legacy_name`, and `_explanation_template` placeholders (`{groups}`, `{row_filter}`) are consistent between `filters.py`, the wrapper in `capdata.py`, and the tests. `sensor_filter`/`check_all_perc_diff_comb`/`abs_diff_from_average` already live in `filters.py` (same module — no import needed).

**Risk note:** `test_after_agg_sensors` asserts the aggregated `power_inv_sum_agg` column is retained. The new path reindexes `data_filtered` from `capdata.data` in `run()`; this preserves the column only if `agg_sensors` wrote it to `capdata.data` (it does in the current implementation). If that test fails, check whether `agg_sensors` is writing agg columns to `data` vs only `data_filtered`.
