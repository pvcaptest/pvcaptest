# RollingStd Filter Implementation Plan (chunk 1 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `RollingStd` filter that removes intervals where a column's rolling-window standard deviation is at or above a threshold (unstable / variable irradiance), replacing the notebook `unstable_irr_filter` custom function.

**Architecture:** Follow the established `filters.py` step-class pattern: a `BaseFilter` subclass declaring config as `param` attributes and implementing `_execute(capdata)` to return the kept `pandas.Index`; register it in `FILTER_REGISTRY`; add a thin `CapData.filter_rolling_std` wrapper. Serialization (`to_config`/`from_config`), summary, `explanation`, and YAML round-trip are inherited from `BaseSummaryStep` since all params are scalars/strings.

**Tech Stack:** Python 3.12, `param`, `pandas`, `pytest`, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "New filter classes → `RollingStd`").

## Global Constraints

- Line length: 88 characters (ruff default).
- Docstrings: NumPy-style for all public classes/methods.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants.
- No backward-compatibility shims required (pre-1.0 branch).
- Run lint/format with `just lint` and `just fmt`; run tests with `uv run pytest`.
- POA-column inference uses the existing `CapData._get_poa_col()`.
- The original oracle function (for parity), from
  `untracked_bin/filters_convert_custom_to_filter_classes.ipynb`:
  ```python
  def unstable_irr_filter(df, irr_col, window, threshold):
      std = df[irr_col].rolling(window).std()
      return df[std < threshold]
  ```

---

### Task 1: `RollingStd` filter class + registry entry

**Files:**
- Modify: `src/captest/filters.py` (add class after the `Irradiance` class, ~line 492; add registry entry in `FILTER_REGISTRY`, ~line 1322)
- Test: `tests/test_filter_classes.py` (add imports, a `cd_roll` fixture, and a `TestRollingStd` class)

**Interfaces:**
- Consumes: `BaseFilter`, `param`, `pandas`, `CapData._get_poa_col()`, `CapData.data_filtered`, `step_from_config`, `FILTER_REGISTRY`.
- Produces: `filters.RollingStd(column=None, window=None, threshold=None, custom_name=None)` — a `BaseFilter` subclass whose `_execute(capdata)` returns the kept `pandas.Index`; registered under key `"RollingStd"`.

- [ ] **Step 1: Write the failing tests**

Add these imports to the existing `from captest.filters import (...)` block in `tests/test_filter_classes.py` (keep alphabetical-ish grouping with the other class imports):

```python
    RollingStd,
```

Add a fixture near the other `cd_*` fixtures:

```python
@pytest.fixture
def cd_roll():
    """A CapData with a poa column that has a stable stretch and a spike."""
    cd = CapData("roll")
    cd.data = pd.DataFrame(
        {"poa": [100.0, 100.0, 100.0, 500.0, 100.0, 100.0]},
        index=pd.RangeIndex(6),
    )
    cd.regression_cols = {"poa": "poa"}
    return cd
```

Add a new test class:

```python
class TestRollingStd:
    def test_execute_removes_unstable_and_leading_nan(self, cd_roll):
        # window=2: rolling std is NaN at row 0 (dropped), 0 on the stable
        # rows, and large where the spike enters/leaves (rows 3, 4 dropped).
        f = RollingStd(window=2, threshold=50, column="poa")
        assert list(f._execute(cd_roll)) == [1, 2, 5]

    def test_execute_matches_oracle(self, cd_roll):
        def unstable_irr_filter(df, irr_col, window, threshold):
            std = df[irr_col].rolling(window).std()
            return df[std < threshold]

        f = RollingStd(window=2, threshold=50, column="poa")
        oracle = unstable_irr_filter(cd_roll.data, "poa", 2, 50)
        assert list(f._execute(cd_roll)) == list(oracle.index)

    def test_execute_defaults_column_to_poa(self, cd_roll):
        f = RollingStd(window=2, threshold=50)  # column=None -> poa
        assert list(f._execute(cd_roll)) == [1, 2, 5]

    def test_execute_requires_window_and_threshold(self, cd_roll):
        with pytest.raises(ValueError, match="window and threshold"):
            RollingStd(threshold=50, column="poa")._execute(cd_roll)
        with pytest.raises(ValueError, match="window and threshold"):
            RollingStd(window=2, column="poa")._execute(cd_roll)

    def test_config_round_trips(self):
        f = RollingStd(window="10min", threshold=20, column="poa")
        cfg = f.to_config()
        assert cfg["type"] == "RollingStd"
        f2 = step_from_config(cfg)
        assert isinstance(f2, RollingStd)
        assert f2.window == "10min"
        assert f2.threshold == 20
        assert f2.column == "poa"

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["RollingStd"] is RollingStd

    def test_explanation_reports_resolved_column(self, cd_roll):
        f = RollingStd(window=2, threshold=50)
        f.run(cd_roll)
        assert f.explanation == (
            "Intervals where the rolling std (window=2) of poa was at or "
            "above 50 were removed."
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k RollingStd -v`
Expected: FAIL — `ImportError: cannot import name 'RollingStd'` (collection error).

- [ ] **Step 3: Implement the `RollingStd` class**

Add to `src/captest/filters.py` immediately after the `Irradiance` class (before `Sensors`):

```python
class RollingStd(BaseFilter):
    """Remove intervals where a column's rolling-window standard deviation is
    at or above ``threshold`` (unstable / variable irradiance).

    ``column`` defaults to the regression POA column when None. ``window`` is
    passed to ``DataFrame.rolling`` and may be an int row count or a pandas
    offset alias (e.g. ``'10min'``). The leading rows of the window produce a
    NaN std and are removed, matching the original ``unstable_irr_filter``.
    """

    _explanation_template = (
        "Intervals where the rolling std (window={window}) of {column} was at "
        "or above {threshold} were removed."
    )

    column = param.String(
        default=None,
        allow_None=True,
        doc="Column to evaluate. Inferred from the regression POA column when "
        "None.",
    )
    window = param.Parameter(
        default=None,
        doc="Rolling window: int row count or pandas offset alias (e.g. "
        "'10min'). Passed to DataFrame.rolling.",
    )
    threshold = param.Number(
        default=None,
        allow_None=True,
        doc="Standard-deviation threshold; intervals whose rolling std is at "
        "or above this are removed.",
    )

    def _execute(self, capdata):
        if self.window is None or self.threshold is None:
            raise ValueError("RollingStd requires both window and threshold.")
        col = self.column if self.column is not None else capdata._get_poa_col()
        self.column_resolved = col
        df = capdata.data_filtered
        std = df[col].rolling(self.window).std()
        return df.index[std < self.threshold]

    def _explanation_values(self):
        return {
            "window": self.window,
            "column": getattr(self, "column_resolved", self.column),
            "threshold": self.threshold,
        }
```

Add the registry entry to `FILTER_REGISTRY` (place after `"Irradiance": Irradiance,`):

```python
    "RollingStd": RollingStd,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k RollingStd -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
just lint src/captest/filters.py tests/test_filter_classes.py
just fmt
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add RollingStd filter class for unstable-irradiance filtering"
```

---

### Task 2: `CapData.filter_rolling_std` wrapper

**Files:**
- Modify: `src/captest/capdata.py` (add `RollingStd` to the `from captest.filters import (...)` block ~line 37; add wrapper method in the filter-wrapper section, after `filter_irr`, ~line 1765)
- Test: `tests/test_filter_classes.py` (add a `TestRollingStdWrapper` class)

**Interfaces:**
- Consumes: `filters.RollingStd` (from Task 1).
- Produces: `CapData.filter_rolling_std(window, threshold, column=None, custom_name=None)` — builds a `RollingStd` step and runs it in place (returns `None`, appends one step to `self.filters`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_filter_classes.py`:

```python
class TestRollingStdWrapper:
    def test_wrapper_records_step(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50)
        assert len(cd_roll.filters) == 1
        assert isinstance(cd_roll.filters[0], RollingStd)

    def test_wrapper_filters_data(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50)
        assert list(cd_roll.data_filtered.index) == [1, 2, 5]

    def test_wrapper_custom_name_sets_step_label(self, cd_roll):
        cd_roll.filter_rolling_std(2, 50, custom_name="stability")
        assert cd_roll.filters[-1].custom_name == "stability"

    def test_wrapper_explicit_column(self, cd_roll):
        cd_roll.data["ghi"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cd_roll.filter_rolling_std(2, 50, column="ghi")
        assert list(cd_roll.data_filtered.index) == [1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k RollingStdWrapper -v`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute 'filter_rolling_std'`.

- [ ] **Step 3: Implement the wrapper and import**

Add `RollingStd` to the `from captest.filters import (...)` block in `src/captest/capdata.py` (with the other class imports, after `Regression,`):

```python
    RollingStd,
```

Add the wrapper method after `filter_irr` in `src/captest/capdata.py`:

```python
    def filter_rolling_std(self, window, threshold, column=None, custom_name=None):
        """Remove intervals where a column's rolling std is at or above a threshold.

        Parameters
        ----------
        window : int or str
            Rolling window passed to ``DataFrame.rolling`` — an int row count or
            a pandas offset alias such as ``'10min'``.
        threshold : float
            Intervals whose rolling standard deviation is at or above this value
            are removed.
        column : str, default None
            Column to evaluate. Defaults to the POA column from
            ``regression_cols``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = RollingStd(
            window=window,
            threshold=threshold,
            column=column,
            custom_name=custom_name,
        )
        flt.run(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k RollingStd -v`
Expected: PASS (11 tests total — Task 1's 7 plus these 4).

- [ ] **Step 5: Run the full filter-class suite to check for regressions**

Run: `uv run pytest tests/test_filter_classes.py -q`
Expected: PASS (all existing tests plus the new ones).

- [ ] **Step 6: Lint, format, commit**

```bash
just lint src/captest/capdata.py tests/test_filter_classes.py
just fmt
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData.filter_rolling_std wrapper"
```

---

## Self-Review

- **Spec coverage:** The spec's `RollingStd` section (params `column`/`window`/`threshold`, POA default, leading-NaN parity, explanation template, registry entry, `filter_rolling_std` wrapper) is implemented across Tasks 1–2. Serialization is covered by `test_config_round_trips` and `test_registered_in_registry`. Parity with `unstable_irr_filter` is covered by `test_execute_matches_oracle` and `test_execute_removes_unstable_and_leading_nan`. Docstrings are inline. Out of scope for this chunk: user-guide/changelog docs (chunk 6).
- **Placeholder scan:** none — every step has concrete code and exact commands.
- **Type consistency:** `RollingStd(column, window, threshold, custom_name)` names match between the class (Task 1), the wrapper (Task 2), and all tests. `_get_poa_col()` and `data_filtered` match the current `CapData` API.
