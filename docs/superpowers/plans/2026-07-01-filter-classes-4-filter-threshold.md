# filter_threshold Wrapper Implementation Plan (chunk 4 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `CapData.filter_threshold(column, low=None, high=None)` wrapper that exposes one-sided (or two-sided) thresholding on any column, replacing the notebook `filter_avail` custom function. It reuses the existing `Irradiance` filter class — no new class.

**Architecture:** The `Irradiance` filter class already supports one-sided bounds (`low`/`high` default `None`, `allow_None`) on any column via its `col_name` param; only the `filter_irr` wrapper (with required positional `low, high`) fails to expose this. `filter_threshold` is a thin wrapper that builds an `Irradiance` step with `col_name=column` and both bounds optional. The recorded step is a normal `Irradiance` step, so it serializes and replays with no registry change.

**Tech Stack:** Python 3.12, `param`, `pandas`, `pytest`, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "`CapData` wrappers → `filter_threshold`").

## Global Constraints

- Line length: 88 characters (ruff default). `src/captest/capdata.py` is NOT
  E501-exempt — every line ≤ 88 chars.
- Docstrings: NumPy-style for all public methods.
- No backward-compatibility shims required (pre-1.0 branch).
- Run lint/format with `just lint <files>` and `just fmt`; run tests with
  `uv run pytest`.
- `Irradiance` is ALREADY imported in `capdata.py` — do not add an import.
- Bounds are inclusive (`>=` / `<=`), matching `Irradiance` semantics. This is
  a deliberate change from the original `filter_avail`'s strict `>`; document
  it in the wrapper docstring.
- The original oracle function, from
  `untracked_bin/filters_convert_custom_to_filter_classes.ipynb`:
  ```python
  def filter_avail(data, avail_col, threshold):
      return data.loc[data[avail_col] > threshold, :]
  ```
  (`filter_threshold(col, low=threshold)` is the inclusive-boundary equivalent.)

---

### Task 1: `CapData.filter_threshold` wrapper (backed by `Irradiance`)

**Files:**
- Modify: `src/captest/capdata.py` (add wrapper method after `filter_flag`)
- Test: `tests/test_filter_classes.py` (add a `cd_thresh` fixture near the other `cd_*` fixtures, add a `TestFilterThreshold` class)

**Interfaces:**
- Consumes: `filters.Irradiance` (already imported in `capdata.py`), `CapData.filters_to_config()`, `CapData.run_pipeline(config)`.
- Produces: `CapData.filter_threshold(column, low=None, high=None, custom_name=None)` — builds an `Irradiance(low=low, high=high, col_name=column)` step and runs it in place (returns `None`, appends one `Irradiance` step to `self.filters`).

- [ ] **Step 1: Write the failing tests**

Add a fixture near the other `cd_*` fixtures in `tests/test_filter_classes.py`:

```python
@pytest.fixture
def cd_thresh():
    """A CapData with an availability column and a temperature column."""
    cd = CapData("thresh")
    cd.data = pd.DataFrame(
        {
            "avail": [95.0, 97.4, 98.0, 99.0, 100.0],
            "temp": [30.0, 40.0, 45.0, 50.0, 35.0],
        },
        index=pd.RangeIndex(5),
    )
    return cd
```

Add a new test class. Place `class TestFilterThreshold:` on its own, with a blank line before and after, immediately after the end of the existing `class TestBooleanFlagWrapper:` block and before `class TestFilterOutliers:`. Do not insert it inside any existing class.

```python
class TestFilterThreshold:
    def test_wrapper_records_irradiance_step(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        assert len(cd_thresh.filters) == 1
        step = cd_thresh.filters[0]
        assert isinstance(step, Irradiance)
        assert step.col_name == "avail"

    def test_low_only_keeps_at_or_above(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        assert list(cd_thresh.data_filtered.index) == [1, 2, 3, 4]

    def test_high_only_keeps_at_or_below(self, cd_thresh):
        cd_thresh.filter_threshold("temp", high=40)
        assert list(cd_thresh.data_filtered.index) == [0, 1, 4]

    def test_both_bounds_keep_band(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4, high=99.0)
        assert list(cd_thresh.data_filtered.index) == [1, 2, 3]

    def test_custom_name_sets_step_label(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4, custom_name="availability")
        assert cd_thresh.filters[-1].custom_name == "availability"

    def test_serializes_and_replays_as_irradiance(self, cd_thresh):
        cd_thresh.filter_threshold("avail", low=97.4)
        config = cd_thresh.filters_to_config()
        assert config[0]["type"] == "Irradiance"
        assert config[0]["col_name"] == "avail"

        fresh = CapData("fresh")
        fresh.data = cd_thresh.data.copy()
        fresh.run_pipeline(config)
        assert list(fresh.data_filtered.index) == [1, 2, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k FilterThreshold -v`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute 'filter_threshold'`.

- [ ] **Step 3: Implement the wrapper**

Add the wrapper method after `filter_flag` in `src/captest/capdata.py` (`Irradiance` is already imported — do not add an import):

```python
    def filter_threshold(self, column, low=None, high=None, custom_name=None):
        """Keep intervals where ``column`` is within ``[low, high]``.

        Either bound may be None for a one-sided filter: pass only ``low`` to
        keep rows at or above it, or only ``high`` to keep rows at or below it.
        Bounds are inclusive (``>=`` / ``<=``). Backed by the ``Irradiance``
        filter, so the recorded step serializes and replays as an ``Irradiance``
        step.

        Parameters
        ----------
        column : str
            Column to threshold.
        low : float, default None
            Lower bound (inclusive). None means unbounded below.
        high : float, default None
            Upper bound (inclusive). None means unbounded above.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Irradiance(
            low=low, high=high, col_name=column, custom_name=custom_name
        )
        flt.run(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k FilterThreshold -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full filter-class suite to check for regressions**

Run: `uv run pytest tests/test_filter_classes.py -q`
Expected: PASS (all existing plus the 6 new tests).

- [ ] **Step 6: Lint, format, commit**

```bash
just lint src/captest/capdata.py tests/test_filter_classes.py
just fmt
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData.filter_threshold wrapper (one-sided Irradiance)"
```

---

## Self-Review

- **Spec coverage:** The spec's `filter_threshold` requirement (one-sided/two-sided thresholding on any column, backed by `Irradiance`, inclusive bounds, serializes as an `Irradiance` step, no new registry entry) is implemented in Task 1. Inclusive-vs-strict difference from `filter_avail` is documented in the docstring. One-sided low-only and high-only cases are both covered, as is the serialize/replay round-trip. Out of scope: user-guide/changelog docs (chunk 6).
- **Placeholder scan:** none — every step has concrete code and exact commands.
- **Type consistency:** `filter_threshold(column, low, high, custom_name)` maps `column` → `Irradiance.col_name` and `low`/`high` → `Irradiance.low`/`high`. `Irradiance` is confirmed already imported in `capdata.py` (used by `filter_irr`). `filters_to_config()` returns a list of `to_config()` dicts and `run_pipeline(config)` replays them — both confirmed present on `CapData`. `Irradiance` with an explicit `col_name` needs no `regression_cols`, so `run_pipeline` works on the bare `fresh` CapData in the round-trip test.
```
