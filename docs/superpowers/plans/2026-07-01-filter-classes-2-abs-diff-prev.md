# AbsDiffPrev Filter Implementation Plan (chunk 2 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `AbsDiffPrev` filter that removes intervals whose absolute fractional change from the previous interval exceeds a threshold (a step-change / stability filter), replacing the notebook `filter_abs_perc_diff_prev_interval` custom function.

**Architecture:** Follow the established `filters.py` step-class pattern: a `BaseFilter` subclass with `param` attributes and `_execute(capdata)` returning the kept `pandas.Index`; register in `FILTER_REGISTRY`; add a thin `CapData.filter_abs_diff_prev` wrapper. Serialization/summary/explanation/YAML round-trip are inherited (all params are scalars/strings). This mirrors the `RollingStd` filter added in chunk 1.

**Tech Stack:** Python 3.12, `param`, `pandas`, `pytest`, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "New filter classes → `AbsDiffPrev`").

## Global Constraints

- Line length: 88 characters (ruff default). `src/captest/capdata.py` and
  `src/captest/filters.py` are NOT in the E501 ignore list — every line,
  including docstring lines, must be ≤ 88 chars.
- Docstrings: NumPy-style for all public classes/methods.
- Naming: `snake_case` functions/vars, `PascalCase` classes.
- No backward-compatibility shims required (pre-1.0 branch).
- Run lint/format with `just lint <files>` and `just fmt`; run tests with
  `uv run pytest`.
- POA-column inference uses the existing `CapData._get_poa_col()`.
- The original oracle function (for parity), from
  `untracked_bin/filters_convert_custom_to_filter_classes.ipynb`:
  ```python
  def filter_abs_perc_diff_prev_interval(data, column, threshold=0.05):
      flt = (data.assign(diff=lambda x: x[column].diff())
                 .assign(abs_diff=lambda x: abs(x['diff'] / x[column]))
                 .loc[lambda x: x['abs_diff'] <= threshold])
      return flt
  ```
  i.e. keep rows where `abs(col.diff() / col) <= threshold`; the first row's
  diff is NaN and is dropped.

---

### Task 1: `AbsDiffPrev` filter class + registry entry

**Files:**
- Modify: `src/captest/filters.py` (add class after the `RollingStd` class; add registry entry in `FILTER_REGISTRY` after `"RollingStd": RollingStd,`)
- Test: `tests/test_filter_classes.py` (add `AbsDiffPrev` to the filters import block, add a `cd_step` fixture near the other `cd_*` fixtures, add a `TestAbsDiffPrev` class)

**Interfaces:**
- Consumes: `BaseFilter`, `param`, `pandas`, `CapData._get_poa_col()`, `CapData.data_filtered`, `step_from_config`, `FILTER_REGISTRY`.
- Produces: `filters.AbsDiffPrev(column=None, threshold=0.05, custom_name=None)` — a `BaseFilter` subclass whose `_execute(capdata)` returns the kept `pandas.Index`; registered under key `"AbsDiffPrev"`.

- [ ] **Step 1: Write the failing tests**

Add `AbsDiffPrev` to the `from captest.filters import (...)` block in `tests/test_filter_classes.py` (with the other class imports):

```python
    AbsDiffPrev,
```

Add a fixture near the other `cd_*` fixtures:

```python
@pytest.fixture
def cd_step():
    """A CapData with a poa column that has one large step change."""
    cd = CapData("step")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 102.0, 300.0, 305.0, 310.0]},
        index=pd.RangeIndex(5),
    )
    cd.regression_cols = {"poa": "poa"}
    return cd
```

Add a new test class. To avoid the class-boundary hazard, place `class TestAbsDiffPrev:` on its own, with a blank line before and after, immediately after the end of the existing `class TestRollingStdWrapper:` block and before `class TestFilterOutliers:`. Do not insert it inside any existing class.

```python
class TestAbsDiffPrev:
    def test_execute_removes_step_and_leading_nan(self, cd_step):
        # abs(diff/col): row0 NaN (dropped), row2 ~0.66 (dropped), the rest
        # are well under 0.05.
        f = AbsDiffPrev(threshold=0.05, column="poa")
        assert list(f._execute(cd_step)) == [1, 3, 4]

    def test_execute_matches_oracle(self, cd_step):
        def filter_abs_perc_diff_prev_interval(data, column, threshold=0.05):
            return (
                data.assign(diff=lambda x: x[column].diff())
                .assign(abs_diff=lambda x: abs(x["diff"] / x[column]))
                .loc[lambda x: x["abs_diff"] <= threshold]
            )

        f = AbsDiffPrev(threshold=0.05, column="poa")
        oracle = filter_abs_perc_diff_prev_interval(cd_step.data, "poa", 0.05)
        assert list(f._execute(cd_step)) == list(oracle.index)

    def test_execute_defaults_column_to_poa(self, cd_step):
        f = AbsDiffPrev(threshold=0.05)  # column=None -> poa
        assert list(f._execute(cd_step)) == [1, 3, 4]

    def test_threshold_defaults_to_005(self):
        assert AbsDiffPrev().threshold == 0.05

    def test_execute_larger_threshold_keeps_more(self, cd_step):
        # threshold 0.7 keeps the ~0.66 step row too; only the NaN row drops.
        f = AbsDiffPrev(threshold=0.7, column="poa")
        assert list(f._execute(cd_step)) == [1, 2, 3, 4]

    def test_config_round_trips(self):
        f = AbsDiffPrev(threshold=0.1, column="poa")
        cfg = f.to_config()
        assert cfg["type"] == "AbsDiffPrev"
        f2 = step_from_config(cfg)
        assert isinstance(f2, AbsDiffPrev)
        assert f2.threshold == 0.1
        assert f2.column == "poa"

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["AbsDiffPrev"] is AbsDiffPrev

    def test_explanation_reports_resolved_column(self, cd_step):
        f = AbsDiffPrev(threshold=0.05)
        f.run(cd_step)
        assert f.explanation == (
            "Intervals where poa changed by more than 0.05 (fractional) from "
            "the previous interval were removed."
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k AbsDiffPrev -v`
Expected: FAIL — `ImportError: cannot import name 'AbsDiffPrev'` (collection error).

- [ ] **Step 3: Implement the `AbsDiffPrev` class**

Add to `src/captest/filters.py` immediately after the `RollingStd` class:

```python
class AbsDiffPrev(BaseFilter):
    """Remove intervals with a large fractional change from the previous
    interval (a step-change / stability filter).

    For column ``c`` the test is ``abs(c.diff() / c) <= threshold``; intervals
    above the threshold are removed. ``column`` defaults to the regression POA
    column when None. The first interval has an undefined difference (NaN) and
    is removed, matching the original ``filter_abs_perc_diff_prev_interval``.
    """

    _explanation_template = (
        "Intervals where {column} changed by more than {threshold} "
        "(fractional) from the previous interval were removed."
    )

    column = param.String(
        default=None,
        allow_None=True,
        doc="Column to evaluate. Inferred from the regression POA column when "
        "None.",
    )
    threshold = param.Number(
        default=0.05,
        doc="Maximum allowed absolute fractional change from the previous "
        "interval; intervals above this are removed.",
    )

    def _execute(self, capdata):
        col = self.column if self.column is not None else capdata._get_poa_col()
        self.column_resolved = col
        df = capdata.data_filtered
        s = df[col]
        abs_diff = (s.diff() / s).abs()
        return df.index[abs_diff <= self.threshold]

    def _explanation_values(self):
        return {
            "column": getattr(self, "column_resolved", self.column),
            "threshold": self.threshold,
        }
```

Add the registry entry to `FILTER_REGISTRY` after `"RollingStd": RollingStd,`:

```python
    "AbsDiffPrev": AbsDiffPrev,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k AbsDiffPrev -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
just lint src/captest/filters.py tests/test_filter_classes.py
just fmt
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add AbsDiffPrev filter class for step-change filtering"
```

---

### Task 2: `CapData.filter_abs_diff_prev` wrapper

**Files:**
- Modify: `src/captest/capdata.py` (add `AbsDiffPrev` to the `from captest.filters import (...)` block; add wrapper method after `filter_rolling_std`)
- Test: `tests/test_filter_classes.py` (add a `TestAbsDiffPrevWrapper` class)

**Interfaces:**
- Consumes: `filters.AbsDiffPrev` (from Task 1).
- Produces: `CapData.filter_abs_diff_prev(threshold=0.05, column=None, custom_name=None)` — builds an `AbsDiffPrev` step and runs it in place (returns `None`, appends one step to `self.filters`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_filter_classes.py`. Place `class TestAbsDiffPrevWrapper:` on its own, with a blank line before and after, immediately after the end of `class TestAbsDiffPrev:` (from Task 1). Do not insert it inside any existing class.

```python
class TestAbsDiffPrevWrapper:
    def test_wrapper_records_step(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05)
        assert len(cd_step.filters) == 1
        assert isinstance(cd_step.filters[0], AbsDiffPrev)

    def test_wrapper_filters_data(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05)
        assert list(cd_step.data_filtered.index) == [1, 3, 4]

    def test_wrapper_default_threshold(self, cd_step):
        cd_step.filter_abs_diff_prev()  # default 0.05
        assert list(cd_step.data_filtered.index) == [1, 3, 4]

    def test_wrapper_custom_name_sets_step_label(self, cd_step):
        cd_step.filter_abs_diff_prev(0.05, custom_name="stability")
        assert cd_step.filters[-1].custom_name == "stability"

    def test_wrapper_explicit_column(self, cd_step):
        cd_step.data["ghi"] = [500.0, 500.0, 500.0, 500.0, 500.0]
        cd_step.filter_abs_diff_prev(0.05, column="ghi")
        # ghi is constant -> only the leading-NaN row drops.
        assert list(cd_step.data_filtered.index) == [1, 2, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k AbsDiffPrevWrapper -v`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute 'filter_abs_diff_prev'`.

- [ ] **Step 3: Implement the wrapper and import**

Add `AbsDiffPrev` to the `from captest.filters import (...)` block in `src/captest/capdata.py` (with the other class imports, e.g. after `RollingStd,`):

```python
    AbsDiffPrev,
```

Add the wrapper method after `filter_rolling_std` in `src/captest/capdata.py`:

```python
    def filter_abs_diff_prev(self, threshold=0.05, column=None, custom_name=None):
        """Remove intervals with a large fractional change from the prior interval.

        Parameters
        ----------
        threshold : float, default 0.05
            Maximum allowed absolute fractional change
            (``abs(col.diff() / col)``) from the previous interval. Intervals
            above this are removed.
        column : str, default None
            Column to evaluate. Defaults to the POA column from
            ``regression_cols``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = AbsDiffPrev(
            threshold=threshold, column=column, custom_name=custom_name
        )
        flt.run(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k AbsDiffPrev -v`
Expected: PASS (13 tests total — Task 1's 8 plus these 5).

- [ ] **Step 5: Run the full filter-class suite to check for regressions**

Run: `uv run pytest tests/test_filter_classes.py -q`
Expected: PASS (all existing plus the new tests).

- [ ] **Step 6: Lint, format, commit**

```bash
just lint src/captest/capdata.py tests/test_filter_classes.py
just fmt
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData.filter_abs_diff_prev wrapper"
```

---

## Self-Review

- **Spec coverage:** The spec's `AbsDiffPrev` section (params `column`/`threshold=0.05`, POA default, first-row-NaN parity, explanation template, registry entry, `filter_abs_diff_prev` wrapper) is implemented across Tasks 1–2. Parity with `filter_abs_perc_diff_prev_interval` is covered by `test_execute_matches_oracle` and `test_execute_removes_step_and_leading_nan`. Serialization covered by `test_config_round_trips`/`test_registered_in_registry`. Out of scope for this chunk: user-guide/changelog docs (chunk 6).
- **Placeholder scan:** none — every step has concrete code and exact commands.
- **Type consistency:** `AbsDiffPrev(column, threshold, custom_name)` and the wrapper `filter_abs_diff_prev(threshold, column, custom_name)` names/order match between class, wrapper, and tests. `_get_poa_col()` and `data_filtered` match the current `CapData` API. The explanation-string assertion matches the `_explanation_template` exactly (the split template collapses to a single space between "than {threshold}" and "(fractional)").
```
