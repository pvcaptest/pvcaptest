# CapTest Auto-Wrap on Sim Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the wrap-year functionality (currently in `filter_time(wrap_year=True)`) up to CapTest's `setup()`. When the measured data sits within 60 days of a year boundary, automatically wrap the **sim** data so `CapTest.sim.data` is already contiguous through year-end by the time any filtering runs. A toggle (`auto_wrap_sim`, default `True`) opts callers out.

**Architecture:** Add a `param.Boolean` `auto_wrap_sim` (default `True`) and a private `_sim_data_pre_wrap` instance attribute on `CapTest`. A new `_maybe_wrap_sim_year_end()` method runs early in `setup()`: it restores any prior wrap from `_sim_data_pre_wrap` (so toggling off, or repeated `setup()` calls, behave correctly), checks whether the measured-data range is within 60 days of a year boundary, and — if it is — saves a pre-wrap snapshot and applies `wrap_year_end` to `self.sim.data` with start/end month-day-anchored to the sim year. `self.sim.data_filtered` is refreshed and `self.sim.filters` is cleared (consistent with the existing "data changed → drop filter state" pattern).

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** No direct entry; this plan restores functionality removed in `2026-06-01-filter-time.md` (decision 7) by relocating it from the filter layer to the load-time/setup layer.

**Sequencing:** Execute **before** `2026-06-01-filter-time.md` so the wrap-year functionality is testable and maintained end-to-end before `FilterTime` drops the `wrap_year` flag. Both plans depend on `wrap_year_end` living in `filters.py`, which the FilterTime plan moves there — so the right order is:
1. **This plan** (which imports `wrap_year_end` *from `captest.capdata`* — its current home — as Step 1).
2. **The FilterTime plan**, which moves `wrap_year_end` to `filters.py`; this plan's import switches to `captest.filters` as part of that plan's "grep for stale references" step.

(Alternatively, this plan can be authored to import from `captest.filters` from day 1 if executed after the FilterTime plan's Task 1 Step 3 helper move. Pick whichever ordering matches the execution sequence.)

## Key design decisions (flag if you disagree before implementing)

1. **Toggle: `auto_wrap_sim = param.Boolean(default=True)`** — on by default, callers opt out via `CapTest(auto_wrap_sim=False)` or `cd.auto_wrap_sim = False`. Declared on `CapTest`, exposed via the same `param` mechanism as the rest of CapTest's config.
2. **Trigger condition: measured start *or* end within 60 days of a year boundary.** Specifically:
   - `(meas_start - Jan-1 of meas_start.year).days <= 60`, OR
   - `(Dec-31 of meas_end.year - meas_end).days <= 60`.
   The constant 60 lives as a `_AUTO_WRAP_DAYS = 60` module-level constant in `captest.py` so it can be tuned without changing CapTest's public API.
3. **Action: `wrap_year_end(self.sim.data, start, end)` with sim-year anchoring.** The wrap helper requires `start`/`end` to land on the sim's typical year. Map measured's first/last month-day onto `(sim_year - 1, sim_year)` where `sim_year = self.sim.data.index[0].year`. Verified empirically: typical-year 1990 hourly sim, `start = 1989-12-01`, `end = 1990-02-15` → contiguous 1825-row window straddling year-end. The plain `start`/`end` for `wrap_year_end` are constructed from `pd.Timestamp(year=..., month=meas.month, day=meas.day, hour=..., minute=...)`.
4. **`wrap_year_end` adds an `"index"` column to its output (pre-existing quirk).** This plan drops that column after the wrap so `sim.data` retains only its real measurement columns.
5. **Restore-on-disable / idempotency via `_sim_data_pre_wrap` snapshot.** A plain instance attribute (`None` initially, holds the pre-wrap `sim.data` after a wrap). Every `_maybe_wrap_sim_year_end` call:
   1. If `_sim_data_pre_wrap is not None`, restores `sim.data = _sim_data_pre_wrap.copy()` and clears the snapshot.
   2. Returns early if `auto_wrap_sim` is `False`, or `meas`/`sim` are unset, or either index is not a `DatetimeIndex`.
   3. Returns early if the trigger condition fails.
   4. Otherwise snapshots the current `sim.data` and applies the wrap.
   This makes `setup()` safely re-runnable, makes `auto_wrap_sim=False` reversibly disable a prior wrap, and avoids re-wrapping already-wrapped data.
6. **Side effects on the sim CapData.** After wrapping, the plan sets:
   - `self.sim.data` to the wrap output (with the helper's added `"index"` column dropped).
   - `self.sim.data_filtered = self.sim.data.copy()` (matches the spec's invariant that `data_filtered` is rooted in `data`).
   - `self.sim.filters = []` (any prior filter state is invalidated by the data change; same pattern as `reset_filter`).
7. **Location in `setup()`: just after the `meas`/`sim` not-None validation, before the overrides block.** Earlier than `process_regression_columns`, which copies `data` into `data_filtered` — so we need `data` finalized first.
8. **Does not touch `self.meas.data`.** Auto-wrap only modifies sim. Measured data stays exactly as the user loaded it.

---

### Task 1: Add `auto_wrap_sim` param and `_maybe_wrap_sim_year_end` method

**Files:**
- Modify: `src/captest/captest.py` (add the constant, the param, the snapshot attribute, the method; import `wrap_year_end`)
- Test: `tests/test_captest.py` (new test class `TestAutoWrapSim`)

- [ ] **Step 1: Add the import and module-level constant**

Near the top of `src/captest/captest.py` (alongside the other captest-internal imports), add:

```python
from captest.capdata import wrap_year_end
```

> If executed **after** the FilterTime plan's Task 1 Step 3 (which moves `wrap_year_end` to `filters.py`), change this to `from captest.filters import wrap_year_end`. The FilterTime plan's Task 2 Step 7 grep will flag this site automatically.

Just below the existing module constants (e.g. above `def load_config`), add:

```python
_AUTO_WRAP_DAYS = 60
```

- [ ] **Step 2: Declare the param and the snapshot attribute**

In `class CapTest(param.Parameterized):`, alongside the other top-level params, add:

```python
    auto_wrap_sim = param.Boolean(
        default=True,
        doc="When True, automatically apply wrap_year_end to sim.data during "
        "setup() if measured data is within 60 days of a year boundary. "
        "Set False to opt out and restore any prior auto-wrap.",
    )
```

In `CapTest.__init__` (around line 1083 — after `self._sim_path = None`), add:

```python
        # Pre-wrap snapshot of sim.data (set by _maybe_wrap_sim_year_end so a
        # subsequent setup() with auto_wrap_sim=False or a non-triggering
        # measured range can restore the original sim data).
        self._sim_data_pre_wrap = None
```

- [ ] **Step 3: Write the failing unit tests**

Append a new `TestAutoWrapSim` class to `tests/test_captest.py`. These tests call `_maybe_wrap_sim_year_end` directly (without going through `setup()`), exercising every decision-5 branch:

```python
import numpy as np
import pandas as pd
import pytest

from captest.capdata import CapData
from captest.captest import CapTest


def _hourly_typical_year(year=1990):
    """1-year hourly DataFrame indexed by ``year``."""
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    return pd.DataFrame({"poa": np.arange(len(idx), dtype=float)}, index=idx)


def _ct_with(meas_idx, sim_idx, **kwargs):
    """A CapTest with minimal meas/sim CapData attached for wrap testing."""
    ct = CapTest(**kwargs)
    meas = CapData("meas")
    meas.data = pd.DataFrame({"poa": np.zeros(len(meas_idx))}, index=meas_idx)
    meas.data_filtered = meas.data.copy()
    sim = CapData("sim")
    sim.data = pd.DataFrame({"poa": np.arange(len(sim_idx), dtype=float)}, index=sim_idx)
    sim.data_filtered = sim.data.copy()
    ct.meas = meas
    ct.sim = sim
    return ct


class TestAutoWrapSim:
    def test_default_is_true(self):
        assert CapTest().auto_wrap_sim is True

    def test_no_wrap_when_meas_centered_in_year(self):
        # meas spans May-Aug; nowhere near a year boundary
        meas_idx = pd.date_range("2023-05-01", "2023-08-31", freq="h")
        sim_idx = _hourly_typical_year(1990).index
        ct = _ct_with(meas_idx, sim_idx)
        before = ct.sim.data.copy()
        ct._maybe_wrap_sim_year_end()
        pd.testing.assert_frame_equal(ct.sim.data, before)
        assert ct._sim_data_pre_wrap is None

    def test_wraps_when_meas_starts_near_year_start(self):
        # meas Jan 15 -> Feb 15 -> start is within 60 days of Jan 1
        meas_idx = pd.date_range("2023-01-15", "2023-02-15", freq="h")
        sim = _hourly_typical_year(1990)
        ct = _ct_with(meas_idx, sim.index)
        ct._maybe_wrap_sim_year_end()
        # sim.data now spans 1989-12 through 1990-02 (the wrapped window)
        assert ct.sim.data.index.min() < pd.Timestamp("1990-01-01")
        assert ct.sim.data.index.max() <= pd.Timestamp("1990-02-15")
        assert "index" not in ct.sim.data.columns  # helper's quirk column dropped
        assert ct._sim_data_pre_wrap is not None
        assert ct.sim.filters == []

    def test_wraps_when_meas_ends_near_year_end(self):
        # meas Nov 1 -> Dec 15 -> end is within 60 days of Dec 31.
        # The wrap prepends sim's Nov-Dec data shifted into 1989; the post-
        # 1989 portion is df.loc[:1990-12-15] so the max is 1990-12-15, not
        # anything past 1990-12-31. The observable effect is the prepended
        # 1989 data — assert min < 1990-01-01 instead.
        meas_idx = pd.date_range("2023-11-01", "2023-12-15", freq="h")
        sim = _hourly_typical_year(1990)
        ct = _ct_with(meas_idx, sim.index)
        ct._maybe_wrap_sim_year_end()
        assert ct.sim.data.index.min() < pd.Timestamp("1990-01-01")
        assert ct._sim_data_pre_wrap is not None

    def test_disabled_does_not_wrap(self):
        meas_idx = pd.date_range("2023-01-15", "2023-02-15", freq="h")
        sim = _hourly_typical_year(1990)
        ct = _ct_with(meas_idx, sim.index, auto_wrap_sim=False)
        before = ct.sim.data.copy()
        ct._maybe_wrap_sim_year_end()
        pd.testing.assert_frame_equal(ct.sim.data, before)
        assert ct._sim_data_pre_wrap is None

    def test_disabling_after_wrap_restores(self):
        meas_idx = pd.date_range("2023-01-15", "2023-02-15", freq="h")
        sim = _hourly_typical_year(1990)
        ct = _ct_with(meas_idx, sim.index)
        ct._maybe_wrap_sim_year_end()  # wrap
        wrapped_len = len(ct.sim.data)
        ct.auto_wrap_sim = False
        ct._maybe_wrap_sim_year_end()  # should restore
        assert len(ct.sim.data) == len(sim)
        assert wrapped_len != len(sim)  # sanity: wrap did change the length
        assert ct._sim_data_pre_wrap is None

    def test_idempotent_repeated_runs(self):
        meas_idx = pd.date_range("2023-01-15", "2023-02-15", freq="h")
        sim = _hourly_typical_year(1990)
        ct = _ct_with(meas_idx, sim.index)
        ct._maybe_wrap_sim_year_end()
        first = ct.sim.data.copy()
        ct._maybe_wrap_sim_year_end()
        pd.testing.assert_frame_equal(ct.sim.data, first)

    def test_returns_early_when_meas_or_sim_missing(self):
        ct = CapTest()
        ct.meas = None
        ct.sim = None
        ct._maybe_wrap_sim_year_end()  # must not raise

    def test_returns_early_for_non_datetime_index(self):
        # Non-DatetimeIndex on either side -> skip the wrap check entirely.
        ct = CapTest()
        ct.meas = CapData("meas")
        ct.meas.data = pd.DataFrame({"poa": [0, 1, 2]})  # RangeIndex
        ct.meas.data_filtered = ct.meas.data.copy()
        ct.sim = CapData("sim")
        sim = _hourly_typical_year(1990)
        ct.sim.data = sim
        ct.sim.data_filtered = sim.copy()
        ct._maybe_wrap_sim_year_end()  # must not raise
        assert ct._sim_data_pre_wrap is None
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_captest.py::TestAutoWrapSim -v`
Expected: FAIL — `AttributeError: 'CapTest' object has no attribute '_maybe_wrap_sim_year_end'` (also some assertions on `auto_wrap_sim` will fail until Step 2 lands).

- [ ] **Step 5: Implement `_maybe_wrap_sim_year_end`**

Add the method to `CapTest` (a reasonable spot is just before `def setup` so it sits with the other setup-related helpers):

```python
    def _maybe_wrap_sim_year_end(self):
        """Auto-apply ``wrap_year_end`` to ``self.sim.data`` when warranted.

        Idempotent and reversible: a prior wrap is restored from
        ``self._sim_data_pre_wrap`` before each check, so re-running ``setup()``
        — or toggling ``self.auto_wrap_sim`` to False and re-running — leaves
        ``sim.data`` in the correct state.
        """
        # Restore any prior wrap first so each call starts from the original.
        if self._sim_data_pre_wrap is not None:
            self.sim.data = self._sim_data_pre_wrap.copy()
            self.sim.data_filtered = self.sim.data.copy()
            self.sim.filters = []
            self._sim_data_pre_wrap = None

        if not self.auto_wrap_sim:
            return
        if self.meas is None or self.sim is None:
            return
        meas_idx = self.meas.data.index
        sim_idx = self.sim.data.index
        if not isinstance(meas_idx, pd.DatetimeIndex):
            return
        if not isinstance(sim_idx, pd.DatetimeIndex):
            return
        if len(meas_idx) == 0 or len(sim_idx) == 0:
            return

        meas_start = meas_idx[0]
        meas_end = meas_idx[-1]
        days_from_year_start = (
            meas_start - pd.Timestamp(year=meas_start.year, month=1, day=1)
        ).days
        days_to_year_end = (
            pd.Timestamp(year=meas_end.year, month=12, day=31) - meas_end
        ).days
        if (
            days_from_year_start > _AUTO_WRAP_DAYS
            and days_to_year_end > _AUTO_WRAP_DAYS
        ):
            return

        # Anchor the wrap window onto sim's typical year. sim_year is taken
        # from sim.data.index[0]; the wrap uses (sim_year - 1, sim_year) so
        # wrap_year_end's "df.index[0].year == end.year" branch fires.
        sim_year = sim_idx[0].year
        start = pd.Timestamp(
            year=sim_year - 1,
            month=meas_start.month,
            day=meas_start.day,
            hour=meas_start.hour,
            minute=meas_start.minute,
        )
        end = pd.Timestamp(
            year=sim_year,
            month=meas_end.month,
            day=meas_end.day,
            hour=meas_end.hour,
            minute=meas_end.minute,
        )

        self._sim_data_pre_wrap = self.sim.data.copy()
        wrapped = wrap_year_end(self.sim.data, start, end)
        # wrap_year_end adds an "index" column (pre-existing helper quirk).
        if "index" in wrapped.columns:
            wrapped = wrapped.drop(columns="index")
        self.sim.data = wrapped
        self.sim.data_filtered = self.sim.data.copy()
        self.sim.filters = []
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_captest.py::TestAutoWrapSim -v`
Expected: PASS (9 tests).

- [ ] **Step 7: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: add auto_wrap_sim param and _maybe_wrap_sim_year_end on CapTest"
```

---

### Task 2: Wire `_maybe_wrap_sim_year_end` into `setup()`

**Files:**
- Modify: `src/captest/captest.py` (`setup` method ~line 1539)
- Test: `tests/test_captest.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `TestAutoWrapSim` (or a sibling class) in `tests/test_captest.py`:

```python
class TestSetupAutoWrap:
    def _make_ct(self, meas_idx, sim_idx, **kwargs):
        ct = _ct_with(meas_idx, sim_idx, **kwargs)
        # Minimal regression_cols so setup() does not blow up. Use a single
        # 'poa' column on both CapData instances — that's the only column the
        # _ct_with helper installs.
        ct.meas.regression_cols = {"poa": "poa"}
        ct.sim.regression_cols = {"poa": "poa"}
        ct.test_setup = "custom"
        ct.reg_cols_meas = {"poa": "poa"}
        ct.reg_cols_sim = {"poa": "poa"}
        ct.reg_fml = "poa ~ poa - 1"
        return ct

    def test_setup_calls_wrap_when_meas_near_year_end(self):
        meas_idx = pd.date_range("2023-11-01", "2023-12-15", freq="h")
        sim_idx = _hourly_typical_year(1990).index
        ct = self._make_ct(meas_idx, sim_idx)
        ct.setup(verbose=False)
        assert ct._sim_data_pre_wrap is not None
        # Wrap prepends 1989-shifted Nov-Dec; min < 1990-01-01 confirms it.
        assert ct.sim.data.index.min() < pd.Timestamp("1990-01-01")

    def test_setup_skips_wrap_when_disabled(self):
        meas_idx = pd.date_range("2023-11-01", "2023-12-15", freq="h")
        sim_idx = _hourly_typical_year(1990).index
        ct = self._make_ct(meas_idx, sim_idx, auto_wrap_sim=False)
        ct.setup(verbose=False)
        assert ct._sim_data_pre_wrap is None
        # sim.data unchanged: still spans only 1990
        assert ct.sim.data.index.min() >= pd.Timestamp("1990-01-01")
        assert ct.sim.data.index.max() <= pd.Timestamp("1990-12-31 23:00")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_captest.py::TestSetupAutoWrap -v`
Expected: FAIL — `setup()` does not call the wrap method yet (`_sim_data_pre_wrap` stays `None` even when wrap is warranted).

- [ ] **Step 3: Wire into `setup()`**

In `CapTest.setup()` (around line 1549), immediately after the `meas`/`sim` not-`None` validation and **before** the `overrides = {}` block, insert:

```python
        # Auto-wrap sim.data when measured spans (within 60 days of) a year
        # boundary. Idempotent and reversible — re-running setup() or toggling
        # auto_wrap_sim restores the appropriate state.
        self._maybe_wrap_sim_year_end()
```

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest tests/test_captest.py::TestSetupAutoWrap -v`
Expected: PASS.

- [ ] **Step 5: Run the full captest suite**

Run: `uv run pytest tests/test_captest.py -q`
Expected: PASS — all pre-existing CapTest tests still pass (auto_wrap defaults to True, but the existing fixtures use measured ranges that do not trigger the wrap, so behavior is unchanged for them).

> If a pre-existing test does trigger the wrap inadvertently, it will fail because `sim.data` no longer matches what the fixture set up. The fix is to set `auto_wrap_sim=False` on that test's CapTest (or update the test to reflect the new behavior). Treat any such failure as an opportunity to document which existing tests effectively exercised the trigger condition.

- [ ] **Step 6: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS.

- [ ] **Step 7: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "refactor: invoke auto_wrap_sim at the start of CapTest.setup()"
```

---

## Self-Review

**1. Coverage of the user's stated requirement:**
- "automatic check when loading data with CapTest" — runs in `setup()` (called from `from_params` once both meas+sim are set). ✓
- "if the measured data is within 60 days of the end / beginning of the year" — trigger checks `days_from_year_start <= 60` OR `days_to_year_end <= 60`. ✓
- "the wrap year should be run so `CapTest.sim.data` is the result of running wrap_year on the loaded data" — wraps sim.data and refreshes sim.data_filtered/sim.filters. ✓
- "should be a kwarg that can be toggled, default True" — `auto_wrap_sim = param.Boolean(default=True)`. ✓ Toggling off after a wrap reverses it.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result. The single deferred decision (which module to import `wrap_year_end` from) is explicitly tied to the execution-order question at the top.

**3. Type/name consistency:** `auto_wrap_sim`, `_sim_data_pre_wrap`, `_AUTO_WRAP_DAYS`, `_maybe_wrap_sim_year_end`, and the wrap helper's signature (`wrap_year_end(df, start, end)`) match across `captest.py` and the tests.

**Risk note:** the existing CapTest test suite may contain a fixture whose measured range happens to fall within 60 days of a year boundary. That test would change behavior under `auto_wrap_sim=True` and either need `auto_wrap_sim=False` set, or its assertions updated. Task 2 Step 5 surfaces this; treat any failure as a documented behavior change, not a regression.
