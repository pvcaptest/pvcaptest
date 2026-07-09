# BooleanFlag Filter Implementation Plan (chunk 3 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `BooleanFlag` filter that removes intervals where a boolean/flag column is truthy (with an `invert` toggle to instead keep only the truthy rows), replacing the notebook `remove_inter_row_shading` custom function.

**Architecture:** Follow the established `filters.py` step-class pattern: a `BaseFilter` subclass with `param` attributes and `_execute(capdata)` returning the kept `pandas.Index`; register in `FILTER_REGISTRY`; add a thin `CapData.filter_flag` wrapper. Because the summary phrasing depends on `invert`, this class overrides the `explanation` property directly (like `Time`/`Days`) instead of using `_explanation_template`. Serialization is inherited (params are a string + a bool).

**Tech Stack:** Python 3.12, `param`, `pandas`, `pytest`, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "New filter classes → `BooleanFlag`").

## Global Constraints

- Line length: 88 characters (ruff default). `src/captest/capdata.py` and
  `src/captest/filters.py` are NOT E501-exempt — every line ≤ 88 chars.
- Docstrings: NumPy-style for all public classes/methods.
- Naming: `snake_case` functions/vars, `PascalCase` classes.
- No backward-compatibility shims required (pre-1.0 branch).
- Run lint/format with `just lint <files>` and `just fmt`; run tests with
  `uv run pytest`.
- The original oracle function, from
  `untracked_bin/filters_convert_custom_to_filter_classes.ipynb`:
  ```python
  def remove_inter_row_shading(data, boolean_column='backtrack_on'):
      return data[~data[boolean_column].astype(bool)]
  ```
  Generalized here: `column` is required (no hard-coded default), and an
  `invert` flag flips which side is removed. Coercion uses `.astype(bool)`
  (0/1, real booleans, and NaN—which is truthy—handled uniformly).

---

### Task 1: `BooleanFlag` filter class + registry entry

**Files:**
- Modify: `src/captest/filters.py` (add class after the `AbsDiffPrev` class; add registry entry in `FILTER_REGISTRY` after `"AbsDiffPrev": AbsDiffPrev,`)
- Test: `tests/test_filter_classes.py` (add `BooleanFlag` to the filters import block, add a `cd_flag` fixture near the other `cd_*` fixtures, add a `TestBooleanFlag` class)

**Interfaces:**
- Consumes: `BaseFilter`, `param`, `pandas`, `CapData.data_filtered`, `step_from_config`, `FILTER_REGISTRY`.
- Produces: `filters.BooleanFlag(column=None, invert=False, custom_name=None)` — a `BaseFilter` subclass whose `_execute(capdata)` returns the kept `pandas.Index`; registered under key `"BooleanFlag"`.

- [ ] **Step 1: Write the failing tests**

Add `BooleanFlag` to the `from captest.filters import (...)` block in `tests/test_filter_classes.py` (with the other class imports):

```python
    BooleanFlag,
```

Add a fixture near the other `cd_*` fixtures:

```python
@pytest.fixture
def cd_flag():
    """A CapData with a boolean flag column (e.g. tracker backtracking)."""
    cd = CapData("flag")
    cd.data = pd.DataFrame(
        {
            "power": [1.0, 2.0, 3.0, 4.0, 5.0],
            "backtrack_on": [False, True, False, True, False],
        },
        index=pd.RangeIndex(5),
    )
    return cd
```

Add a new test class. Place `class TestBooleanFlag:` on its own, with a blank line before and after, immediately after the end of the existing `class TestAbsDiffPrevWrapper:` block and before `class TestFilterOutliers:`. Do not insert it inside any existing class.

```python
class TestBooleanFlag:
    def test_execute_drops_truthy_rows(self, cd_flag):
        f = BooleanFlag(column="backtrack_on")
        assert list(f._execute(cd_flag)) == [0, 2, 4]

    def test_execute_matches_oracle(self, cd_flag):
        def remove_inter_row_shading(data, boolean_column="backtrack_on"):
            return data[~data[boolean_column].astype(bool)]

        f = BooleanFlag(column="backtrack_on")
        oracle = remove_inter_row_shading(cd_flag.data, "backtrack_on")
        assert list(f._execute(cd_flag)) == list(oracle.index)

    def test_execute_invert_keeps_truthy(self, cd_flag):
        f = BooleanFlag(column="backtrack_on", invert=True)
        assert list(f._execute(cd_flag)) == [1, 3]

    def test_execute_coerces_int_and_nan(self, cd_flag):
        # astype(bool): 0->False, nonzero->True, NaN->True.
        cd_flag.data["mixed"] = [0, 1, 0, np.nan, 2]
        f = BooleanFlag(column="mixed")
        assert list(f._execute(cd_flag)) == [0, 2]

    def test_execute_requires_column(self, cd_flag):
        with pytest.raises(ValueError, match="requires a column"):
            BooleanFlag()._execute(cd_flag)

    def test_config_round_trips(self):
        f = BooleanFlag(column="backtrack_on", invert=True)
        cfg = f.to_config()
        assert cfg["type"] == "BooleanFlag"
        f2 = step_from_config(cfg)
        assert isinstance(f2, BooleanFlag)
        assert f2.column == "backtrack_on"
        assert f2.invert is True

    def test_registered_in_registry(self):
        assert FILTER_REGISTRY["BooleanFlag"] is BooleanFlag

    def test_explanation_default(self, cd_flag):
        f = BooleanFlag(column="backtrack_on")
        f.run(cd_flag)
        assert f.explanation == (
            "Intervals flagged True in backtrack_on were removed."
        )

    def test_explanation_invert(self, cd_flag):
        f = BooleanFlag(column="backtrack_on", invert=True)
        f.run(cd_flag)
        assert f.explanation == (
            "Intervals flagged False in backtrack_on were removed."
        )

    def test_explanation_none_before_run(self):
        assert BooleanFlag(column="backtrack_on").explanation is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k BooleanFlag -v`
Expected: FAIL — `ImportError: cannot import name 'BooleanFlag'` (collection error).

- [ ] **Step 3: Implement the `BooleanFlag` class**

Add to `src/captest/filters.py` immediately after the `AbsDiffPrev` class:

```python
class BooleanFlag(BaseFilter):
    """Remove intervals where a boolean/flag column is truthy.

    ``column`` values are coerced with ``astype(bool)`` so 0/1, real booleans,
    and NaN (which is truthy) are handled uniformly. By default rows where the
    column is truthy are removed; set ``invert=True`` to instead remove rows
    where the column is falsy (keeping only the truthy rows).
    """

    column = param.String(
        default=None,
        allow_None=True,
        doc="Boolean/flag column. Rows where this is truthy are removed (or "
        "falsy, when invert=True).",
    )
    invert = param.Boolean(
        default=False,
        doc="If True, remove rows where the column is falsy instead of truthy.",
    )

    def _execute(self, capdata):
        if self.column is None:
            raise ValueError("BooleanFlag requires a column.")
        df = capdata.data_filtered
        mask = df[self.column].astype(bool)
        keep = mask if self.invert else ~mask
        return df.index[keep]

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None
        flagged = "False" if self.invert else "True"
        return f"Intervals flagged {flagged} in {self.column} were removed."
```

Add the registry entry to `FILTER_REGISTRY` after `"AbsDiffPrev": AbsDiffPrev,`:

```python
    "BooleanFlag": BooleanFlag,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k BooleanFlag -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
just lint src/captest/filters.py tests/test_filter_classes.py
just fmt
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add BooleanFlag filter class for flag-column filtering"
```

---

### Task 2: `CapData.filter_flag` wrapper

**Files:**
- Modify: `src/captest/capdata.py` (add `BooleanFlag` to the `from captest.filters import (...)` block; add wrapper method after `filter_abs_diff_prev`)
- Test: `tests/test_filter_classes.py` (add a `TestBooleanFlagWrapper` class)

**Interfaces:**
- Consumes: `filters.BooleanFlag` (from Task 1).
- Produces: `CapData.filter_flag(column, invert=False, custom_name=None)` — builds a `BooleanFlag` step and runs it in place (returns `None`, appends one step to `self.filters`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_filter_classes.py`. Place `class TestBooleanFlagWrapper:` on its own, with a blank line before and after, immediately after the end of `class TestBooleanFlag:` (from Task 1). Do not insert it inside any existing class.

```python
class TestBooleanFlagWrapper:
    def test_wrapper_records_step(self, cd_flag):
        cd_flag.filter_flag("backtrack_on")
        assert len(cd_flag.filters) == 1
        assert isinstance(cd_flag.filters[0], BooleanFlag)

    def test_wrapper_filters_data(self, cd_flag):
        cd_flag.filter_flag("backtrack_on")
        assert list(cd_flag.data_filtered.index) == [0, 2, 4]

    def test_wrapper_invert(self, cd_flag):
        cd_flag.filter_flag("backtrack_on", invert=True)
        assert list(cd_flag.data_filtered.index) == [1, 3]

    def test_wrapper_custom_name_sets_step_label(self, cd_flag):
        cd_flag.filter_flag("backtrack_on", custom_name="no backtracking")
        assert cd_flag.filters[-1].custom_name == "no backtracking"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k BooleanFlagWrapper -v`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute 'filter_flag'`.

- [ ] **Step 3: Implement the wrapper and import**

Add `BooleanFlag` to the `from captest.filters import (...)` block in `src/captest/capdata.py` (with the other class imports, e.g. after `BaseSummaryStep,` — keep alphabetical grouping):

```python
    BooleanFlag,
```

Add the wrapper method after `filter_abs_diff_prev` in `src/captest/capdata.py`:

```python
    def filter_flag(self, column, invert=False, custom_name=None):
        """Remove intervals where a boolean/flag column is truthy.

        Parameters
        ----------
        column : str
            Boolean/flag column. Rows where this is truthy are removed.
        invert : bool, default False
            If True, remove rows where the column is falsy instead — keeping
            only the truthy rows.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = BooleanFlag(column=column, invert=invert, custom_name=custom_name)
        flt.run(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k BooleanFlag -v`
Expected: PASS (14 tests total — Task 1's 10 plus these 4).

- [ ] **Step 5: Run the full filter-class suite to check for regressions**

Run: `uv run pytest tests/test_filter_classes.py -q`
Expected: PASS (all existing plus the new tests).

- [ ] **Step 6: Lint, format, commit**

```bash
just lint src/captest/capdata.py tests/test_filter_classes.py
just fmt
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData.filter_flag wrapper"
```

---

## Self-Review

- **Spec coverage:** The spec's `BooleanFlag` section (required `column`, `invert` toggle, `.astype(bool)` coercion incl. NaN, `ValueError` when `column` is None, `invert`-dependent explanation override, registry entry, `filter_flag` wrapper) is implemented across Tasks 1–2. Parity with `remove_inter_row_shading` is covered by `test_execute_matches_oracle` and `test_execute_drops_truthy_rows`. Serialization covered by `test_config_round_trips`/`test_registered_in_registry`. Out of scope: user-guide/changelog docs (chunk 6).
- **Placeholder scan:** none — every step has concrete code and exact commands.
- **Type consistency:** `BooleanFlag(column, invert, custom_name)` and the wrapper `filter_flag(column, invert, custom_name)` match between class, wrapper, and tests. The `explanation` override returns `None` before `run()` (guarded by `hasattr(self, "ix_after")`), matching the base-class contract that `test_explanation_none_before_run` asserts. `invert is True`/`is False` assertions rely on `param.Boolean` storing real bools, which it does.
```
