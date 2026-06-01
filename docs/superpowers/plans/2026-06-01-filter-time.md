# FilterTime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `filter_time` — the most branch-heavy filter — to `FilterTime`, with a conditional `explanation` property override (the escape hatch we designed for filters whose phrasing depends on which params are set).

**Architecture:** `FilterTime` declares `start`, `end`, `test_date`, `days`, `drop`, `wrap_year` as params. `_execute` consolidates the original's four overlapping `if`-blocks into one resolution step (compute effective `start`/`end`, then choose wrap-year / drop-complement / slice). Because phrasing varies with which params are set, `FilterTime` **overrides `explanation` directly** instead of relying on a flat `_explanation_template`. Two `filter_time` helpers (`spans_year`, `wrap_year_end`) live in `capdata.py` today; they move to `filters.py` so `FilterTime._execute` can use them without a back-import cycle. `capdata.py` re-imports `wrap_year_end` (still used by `wrap_seasons`); `spans_year` has no remaining `capdata.py` caller.

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes", "Thin Wrapper Methods" — `FilterTime` is the spec's prototypical conditional-explanation case (handled by overriding `explanation`, per "Filter Explanations").

**Sequencing:** Execute *after* `2026-05-24-filter-irr-example.md` (needs `BaseSummaryStep`/explanation hooks/legacy mirroring).

## Key design decisions (flag if you disagree before implementing)

1. **`spans_year`/`wrap_year_end` move to `filters.py`.** `FilterTime._execute` needs them. They are pure pandas helpers (no `CapData` dependency); `capdata.py` re-imports `wrap_year_end` because `wrap_seasons` still calls it. `spans_year` has no remaining `capdata.py` caller. Same one-way import pattern as `filter_irr`/`filter_grps`/`sensor_filter`.
2. **`start`/`end`/`test_date` as `param.Parameter`, not `param.Date`.** The original accepts strings (`"2/1/90"`) and `pd.Timestamp` values interchangeably, and converts via `pd.to_datetime` inside the method. `param.Date` rejects non-`datetime.date` types. `param.Parameter` (no type check) preserves the legacy flexibility; `_execute` does the conversion.
3. **No-args call raises `ValueError`.** The original `filter_time()` with all params `None` leaves `df_temp` unset and crashes with `NameError`. `_execute` raises `ValueError("filter_time requires at least one of start, end, or test_date")` — surfaces the misuse cleanly.
4. **`test_date` without `days` warns and is a no-op.** The original returns `warnings.warn(...)` (i.e. `None`) and leaves `data_filtered` unchanged. The existing `test_test_date_no_days` asserts a `UserWarning` is emitted; it does not check `data_filtered` shape. New behavior: emit the same warning and return `data_filtered.index` (no filtering), so `run()`'s `len(ix_after)` doesn't blow up on `None`. The warning text matches the original (`"Must specify days"`).
5. **`explanation` is overridden (escape hatch).** Phrasing varies by which params are set; a flat template can't express the matrix cleanly. `_explanation_template` is left unset and `FilterTime.explanation` returns a different sentence per branch (drop vs keep, start-only, end-only, days, test_date). This is exactly the case the plan-3 spec called out as needing the override hatch.
6. **`inplace` kept for now.** Same transitional decision as `FilterIrr`/`FilterSensors`.

---

### Task 1: Move helpers to `filters.py` and add `FilterTime`

**Files:**
- Modify: `src/captest/filters.py` (move `spans_year` and `wrap_year_end` verbatim from `capdata.py:239-302`; add `FilterTime`)
- Modify: `src/captest/capdata.py` (remove the moved helper defs; in the existing `from captest.filters import (...)` block add `wrap_year_end` — still used by `wrap_seasons` at ~lines 371/373 — and `FilterTime` will be added in Task 2)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

Extend the top-of-file import (no mid-file imports — `E402`):

```python
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterIrr,
    FilterSensors,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)
```

Add a module-level `cd_time` fixture near the existing `make_capdata`/`cd_irr` fixtures:

```python
@pytest.fixture
def cd_time():
    """A CapData with a 90-day daily DatetimeIndex for time-window tests."""
    cd = CapData("time")
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    cd.data = pd.DataFrame({"power": range(90)}, index=idx)
    cd.data_filtered = cd.data.copy()
    return cd
```

Then append `TestFilterTime`:

```python
class TestFilterTime:
    def test_execute_start_end(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_start_end_drop(self, cd_time):
        # drop=True removes rows BETWEEN start and end (keeps the complement)
        n_before = len(cd_time.data_filtered)
        f = FilterTime(start="2023-02-01", end="2023-02-15", drop=True)
        kept = f._execute(cd_time)
        # the 15-day window inclusive is 15 rows; complement is n_before - 15
        assert len(kept) == n_before - 15

    def test_execute_start_days(self, cd_time):
        f = FilterTime(start="2023-02-01", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == pd.Timestamp("2023-02-11")  # start + 10 days

    def test_execute_end_days(self, cd_time):
        f = FilterTime(end="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-05")  # end - 10 days
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_test_date(self, cd_time):
        # 10-day window centered on 2023-02-15 -> 2023-02-10 .. 2023-02-20
        f = FilterTime(test_date="2023-02-15", days=10)
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-10")
        assert kept[-1] == pd.Timestamp("2023-02-20")

    def test_execute_start_only_defaults_to_last(self, cd_time):
        f = FilterTime(start="2023-02-01")
        kept = f._execute(cd_time)
        assert kept[0] == pd.Timestamp("2023-02-01")
        assert kept[-1] == cd_time.data_filtered.index[-1]

    def test_execute_end_only_defaults_to_first(self, cd_time):
        f = FilterTime(end="2023-02-15")
        kept = f._execute(cd_time)
        assert kept[0] == cd_time.data_filtered.index[0]
        assert kept[-1] == pd.Timestamp("2023-02-15")

    def test_execute_no_args_raises(self, cd_time):
        with pytest.raises(ValueError, match="at least one of"):
            FilterTime()._execute(cd_time)

    def test_execute_test_date_no_days_warns_and_keeps_all(self, cd_time):
        n_before = len(cd_time.data_filtered)
        f = FilterTime(test_date="2023-02-15")
        with pytest.warns(UserWarning, match="Must specify days"):
            kept = f._execute(cd_time)
        assert len(kept) == n_before  # no filtering applied

    def test_explanation_start_end(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15")
        f.run(cd_time)
        assert "outside" in f.explanation
        assert "2023-02-01" in f.explanation
        assert "2023-02-15" in f.explanation

    def test_explanation_drop(self, cd_time):
        f = FilterTime(start="2023-02-01", end="2023-02-15", drop=True)
        f.run(cd_time)
        assert f.explanation.startswith("Data between")
        assert f.explanation.endswith("was removed.")

    def test_explanation_test_date(self, cd_time):
        f = FilterTime(test_date="2023-02-15", days=10)
        f.run(cd_time)
        assert "centered" in f.explanation
        assert "10-day" in f.explanation

    def test_explanation_start_only(self, cd_time):
        f = FilterTime(start="2023-02-01")
        f.run(cd_time)
        assert f.explanation == "Data before 2023-02-01 was removed."

    def test_explanation_end_only(self, cd_time):
        f = FilterTime(end="2023-02-15")
        f.run(cd_time)
        assert f.explanation == "Data after 2023-02-15 was removed."

    def test_helpers_importable_from_filters(self):
        from captest.filters import spans_year, wrap_year_end
        assert callable(spans_year)
        assert callable(wrap_year_end)

    def test_capdata_still_exposes_wrap_year_end(self):
        # wrap_seasons in capdata.py still calls it
        from captest import capdata
        assert callable(capdata.wrap_year_end)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterTime -v`
Expected: FAIL — `ImportError: cannot import name 'FilterTime'` (also `spans_year`/`wrap_year_end` not yet in `filters.py`).

- [ ] **Step 3: Move the helpers to `filters.py`**

Move `spans_year` (capdata.py:286-302) and `wrap_year_end` (capdata.py:239-283) **verbatim** to `src/captest/filters.py` — place them near the other top-level helpers (after `filter_grps`).

Then **delete** those two function defs from `src/captest/capdata.py`.

- [ ] **Step 4: Re-import `wrap_year_end` into `capdata.py`**

In the existing `from captest.filters import (...)` block in `capdata.py`, add `wrap_year_end` (kept alphabetical):

```python
from captest.filters import (
    BaseSummaryStep,
    FilterIrr,
    FilterSensors,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    wrap_year_end,
)
```

`spans_year` is **not** re-imported (no remaining `capdata.py` caller — `filter_time` will go through `FilterTime`).

> Quick check before proceeding: `grep -n "spans_year\|wrap_year_end" src/captest/capdata.py` should now show only the `wrap_year_end` import and its two `wrap_seasons` call sites. If `spans_year` shows anywhere else, re-import it too.

- [ ] **Step 5: Implement `FilterTime`**

Append to `src/captest/filters.py`:

```python
class FilterTime(BaseFilter):
    """Filter rows to a time window described by start/end/days/test_date.

    Multiple parameter combinations are supported:

    - ``start`` + ``end`` — keep the window (or drop it if ``drop=True``).
    - ``start`` + ``days`` — keep a window of ``days`` starting at ``start``.
    - ``end`` + ``days`` — keep a window of ``days`` ending at ``end``.
    - ``test_date`` + ``days`` — keep a window of ``days`` centered on
      ``test_date``.
    - ``start`` only — keep rows from ``start`` to the last timestamp.
    - ``end`` only — keep rows from the first timestamp to ``end``.

    With ``wrap_year=True`` and a window that spans year-end, the data are
    rotated so the window is contiguous (see ``wrap_year_end``).
    """

    _legacy_name = "filter_time"
    # explanation is overridden below; no flat template fits the matrix.

    start = param.Parameter(default=None, doc="Window start (str or Timestamp).")
    end = param.Parameter(default=None, doc="Window end (str or Timestamp).")
    test_date = param.Parameter(
        default=None,
        doc="Center of a symmetric ``days``-wide window (str or Timestamp).",
    )
    days = param.Integer(
        default=None, allow_None=True, doc="Window length in days."
    )
    drop = param.Boolean(
        default=False,
        doc="When True with start+end, remove the window instead of keeping it.",
    )
    wrap_year = param.Boolean(
        default=False,
        doc="Rotate year-end-spanning windows into a contiguous period.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        start = pd.to_datetime(self.start) if self.start is not None else None
        end = pd.to_datetime(self.end) if self.end is not None else None
        test_date = (
            pd.to_datetime(self.test_date) if self.test_date is not None else None
        )

        # Resolve the effective window from whichever combination was given.
        if test_date is not None:
            if self.days is None:
                warnings.warn("Must specify days")
                return df.index
            offset = pd.DateOffset(days=self.days // 2)
            start = test_date - offset
            end = test_date + offset
        elif start is not None and end is not None:
            pass
        elif start is not None:
            if self.days is not None:
                end = start + pd.DateOffset(days=self.days)
            else:
                end = df.index[-1]
        elif end is not None:
            if self.days is not None:
                start = end - pd.DateOffset(days=self.days)
            else:
                start = df.index[0]
        else:
            raise ValueError(
                "filter_time requires at least one of start, end, or test_date"
            )

        # wrap_year only meaningful when a bounded window is in play (user gave
        # both start+end, or any combination involving days/test_date).
        bounded = self.days is not None or (
            self.start is not None and self.end is not None
        )
        should_wrap = self.wrap_year and bounded and spans_year(start, end)
        should_drop = (
            self.drop
            and self.start is not None
            and self.end is not None
            and not should_wrap
        )

        if should_wrap:
            df_temp = wrap_year_end(df, start, end)
        elif should_drop:
            selected = df.loc[start:end, :]
            df_temp = df.loc[df.index.difference(selected.index), :]
        else:
            df_temp = df.loc[start:end, :]

        self._effective_start = start
        self._effective_end = end
        return df_temp.index

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None

        s, e, td, d = self.start, self.end, self.test_date, self.days
        if td is not None:
            if d is None:
                return None  # test_date without days is a no-op
            return f"Data outside a {d}-day window centered on {td} was removed."
        if s is not None and e is not None:
            if self.drop:
                return f"Data between {s} and {e} was removed."
            return f"Data outside the period {s} to {e} was removed."
        if s is not None:
            if d is not None:
                return (
                    f"Data outside the {d}-day period starting at {s} was removed."
                )
            return f"Data before {s} was removed."
        if e is not None:
            if d is not None:
                return f"Data outside the {d}-day period ending at {e} was removed."
            return f"Data after {e} was removed."
        return None
```

> Notes:
> - The `_effective_start`/`_effective_end` runtime attrs aren't surfaced in `args_repr` here; if a future override is wanted (e.g. to show the resolved end after `start+days`), add a `_args_for_repr` override.
> - The override of `explanation` keeps `BaseSummaryStep`'s pre-run-returns-None guard (`hasattr(self, "ix_after")`); replicating it here keeps the contract consistent.

- [ ] **Step 6: Run the unit tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterTime -v`
Expected: PASS (15 tests).

- [ ] **Step 7: Confirm `capdata` still imports and `wrap_seasons` still works**

Run: `uv run pytest tests/ -k "wrap_seasons or rep_cond_freq" -q`
Expected: PASS — `wrap_seasons` indirectly exercises the re-imported `wrap_year_end`. If no tests match, at minimum confirm import: `uv run python -c "import captest.capdata"`.

- [ ] **Step 8: Commit**

```bash
git add src/captest/filters.py src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add FilterTime with conditional explanation override"
```

---

### Task 2: Convert `CapData.filter_time` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_time` method ~1994-2097: drop `@update_summary`, delegate to `FilterTime`; in the `from captest.filters import (...)` block add `FilterTime`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterTimeWrapper:
    def test_wrapper_records_filtertime_step(self, cd_time):
        cd_time.filter_time(start="2023-02-01", end="2023-02-15")
        assert len(cd_time.filters) == 1
        assert isinstance(cd_time.filters[0], FilterTime)

    def test_wrapper_inplace_false_records_no_step(self, cd_time):
        n_before = cd_time.data_filtered.shape[0]
        result = cd_time.filter_time(
            start="2023-02-01", end="2023-02-15", inplace=False
        )
        assert cd_time.filters == []
        assert cd_time.data_filtered.shape[0] == n_before
        assert result.shape[0] == 15
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterTimeWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterTime` (still the decorated method).

- [ ] **Step 3: Add `FilterTime` to the capdata import**

In `src/captest/capdata.py`:

```python
from captest.filters import (
    BaseSummaryStep,
    FilterIrr,
    FilterSensors,
    FilterTime,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    wrap_year_end,
)
```

- [ ] **Step 4: Replace the `filter_time` method body with a thin wrapper**

Replace the entire current method (the `@update_summary` line through its final `return df_temp`) with:

```python
    def filter_time(
        self,
        start=None,
        end=None,
        drop=False,
        days=None,
        test_date=None,
        inplace=True,
        wrap_year=False,
    ):
        """
        Select data for a specified time period.

        Parameters
        ----------
        start : str or pd.Timestamp or None, default None
            Start date for data to be returned.
        end : str or pd.Timestamp or None, default None
            End date for data to be returned.
        drop : bool, default False
            With start+end, remove the window instead of keeping it.
        days : int or None, default None
            Days in the time window.
        test_date : str or pd.Timestamp or None, default None
            Center of a symmetric ``days``-wide window.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        wrap_year : bool, default False
            If True, rotate year-end-spanning windows into a contiguous period
            (see ``wrap_year_end``).
        """
        flt = FilterTime(
            start=start,
            end=end,
            drop=drop,
            days=days,
            test_date=test_date,
            wrap_year=wrap_year,
        )
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterTimeWrapper -v`
Expected: PASS.

- [ ] **Step 6: Run the pre-existing `filter_time` suite**

Run: `uv run pytest tests/test_CapData.py -k "FilterTime or filter_time" -v`
Expected: PASS — all 11 tests in `TestFilterTime` plus the `test_length_test_period_after_*_filter_time` cases in the regression-result suite.

- [ ] **Step 7: Grep for any `pvc.filter_time` / `capdata.filter_time` references**

Run: `grep -rnE "pvc\.(filter_time|spans_year|wrap_year_end)|capdata\.(filter_time|spans_year|wrap_year_end)" tests/ src/captest/`
Expected: only `from captest import capdata` style imports or the `capdata.wrap_year_end` reference in `test_capdata_still_exposes_wrap_year_end`. Any other hits need the no-shim repoint to `captest.filters`.

- [ ] **Step 8: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS.

- [ ] **Step 9: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: make CapData.filter_time a thin wrapper over FilterTime"
```

---

## Self-Review

**1. Spec coverage (this filter):**
- "Concrete Filter Classes" (FilterTime with start/end/drop/days/test_date/wrap_year) → Task 1. ✓
- "Filter Explanations → override hatch for conditional filters" → Task 1's `explanation` override. ✓
- "Thin Wrapper Methods" → Task 2. ✓
- Defensive improvements (no-args `ValueError`, test_date-no-days warn + keep-all) are honest reads of the original's known bugs and don't break any existing test.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result. The "Quick check" after Step 4 of Task 1 catches stray `spans_year` callers before they bite.

**3. Type/name consistency:** `FilterTime` params (`start`, `end`, `test_date`, `days`, `drop`, `wrap_year`), the runtime attrs `_effective_start`/`_effective_end`, `_legacy_name`, the `explanation` override, and the wrapper signature match across `filters.py`, `capdata.py`, and the tests. `spans_year`/`wrap_year_end` move with their existing signatures; `wrap_year_end` is re-imported alphabetically.

**Risk note (repeat of the FilterSensors-class gap):** before deleting `spans_year`/`wrap_year_end` defs from `capdata.py`, the existing `grep -rnE` in Task 2 Step 7 will catch any `pvc.spans_year` / `pvc.wrap_year_end` test references. Treat any hit as a test-update step (consistent with the no-shim policy used for `csky`, `perc_difference`, `sensor_filter`).
