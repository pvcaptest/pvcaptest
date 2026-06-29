# Filter Classes — Module Setup & Base Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the new `src/captest/filters.py` module — moving the row-filter helper functions into it and adding the `BaseSummaryStep`/`BaseFilter` class hierarchy — and make `CapData` a `param.Parameterized` with a `filters` `param.List`, laying the groundwork for the reactive Panel GUI.

**Architecture:** `filters.py` becomes the one-way-imported home for filter code (`capdata.py → filters.py`, never the reverse). This plan moves the DataFrame-level helpers (`perc_difference`, `check_all_perc_diff_comb`, `abs_diff_from_average`, `sensor_filter`, `filter_irr`, `filter_grps`) out of `capdata.py` into `filters.py`, then adds the two `param.Parameterized` base classes that define the filter-step lifecycle. `capdata.py` imports the moved helpers back where its own methods still use them. **`CapData` itself becomes a `param.Parameterized`** so `filters` can be a real `param.List(item_type=BaseSummaryStep)` — this is what lets the future GUI do `cd.param.watch(callback, 'filters')` and have panes react to pipeline edits. `run()` reassigns the list (`capdata.filters = capdata.filters + [self]`) so watchers fire (in-place `.append` does not notify — verified). This plan is **additive in behavior** — no existing filter method routes through the new classes yet, so the full suite stays green. `run()` keeps the legacy `data_filtered` attribute consistent transitionally; legacy summary/removed/kept mirroring is added in the next plan (example filter) where it is actually exercised.

**param migration note:** `param.Parameterized` reserves a **constant** `name` parameter. `CapData("system_a")` passes the name through `super().__init__(name=name)`; `name` cannot be reassigned afterward (verified). The only code that reassigned it was `copy()` (`cd_c.name = ...`), which this plan changes to construct with the name directly. No tests or other modules reassign a `CapData`'s `.name` (verified by grep). `param` tolerates `CapData`'s plain instance attributes (`data`, `column_groups`, …) and the dynamic class-level `property` setattr in `create_column_group_attributes` (both verified).

**Tech Stack:** Python, `param` (already a dependency), pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Module Organization", "BaseSummaryStep", "BaseFilter", "CapData Changes".

**Sequencing:** Execute *after* `2026-05-24-clearsky-module-extraction.md`.

**Scope boundary (explicitly NOT in this plan):**
- No concrete filter classes (`FilterIrr`, etc.) — next plan.
- No thin-wrapper delegation — next plan.
- `data_filtered` remains a normal attribute (not yet a property) — plan 4.
- Legacy `summary`/`summary_ix`/`removed`/`kept`/`filter_counts` untouched, still populated by `@update_summary`.
- `ReportingIrradiance`, `perc_bounds`, `csky`/`detect_clearsky` stay in `capdata.py` (not row filters).
- GUI groundwork in scope is limited to the `param.Parameterized` base + `param.List` so watchers are available; no widgets, panes, or `rerun_from` are built here.

---

### Task 1: Create `filters.py` and move the row-filter helpers

**Files:**
- Create: `src/captest/filters.py`
- Modify: `src/captest/capdata.py` (remove lines 399-544; add import of moved helpers)
- Test: `tests/test_filter_helpers_move.py` (create — guards the move)

- [ ] **Step 1: Write a failing test that the helpers live in `filters.py`**

Create `tests/test_filter_helpers_move.py`:

```python
"""Guards that row-filter helpers are importable from captest.filters."""

import numpy as np
import pandas as pd


def test_helpers_importable_from_filters():
    from captest.filters import (
        perc_difference,
        check_all_perc_diff_comb,
        abs_diff_from_average,
        sensor_filter,
        filter_irr,
        filter_grps,
    )

    assert callable(filter_irr)


def test_filter_irr_behavior_unchanged():
    from captest.filters import filter_irr

    df = pd.DataFrame({"poa": [100, 500, 900]})
    out = filter_irr(df, "poa", 200, 800)
    assert list(out["poa"]) == [500]


def test_capdata_still_exposes_helpers():
    """capdata re-imports the helpers it still uses internally."""
    from captest import capdata

    assert callable(capdata.filter_irr)
    assert callable(capdata.filter_grps)
    assert callable(capdata.sensor_filter)
    assert callable(capdata.check_all_perc_diff_comb)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_helpers_move.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'captest.filters'`

- [ ] **Step 3: Create `filters.py` and move the helper cluster**

Create `src/captest/filters.py` with this header, then move the six functions verbatim from `capdata.py:399-544` (`perc_difference`, `check_all_perc_diff_comb`, `abs_diff_from_average`, `sensor_filter`, `filter_irr`, `filter_grps`):

```python
"""Filter step classes and row-filter helper functions.

This module is imported one-way by ``capdata.py``; it never imports
``capdata``. Filter steps touch a ``CapData`` instance only through the
runtime ``capdata`` argument to ``run``/``_execute``.
"""

from itertools import combinations

import pandas as pd


# <-- the six moved functions go here, unchanged -->
```

- [ ] **Step 4: Remove the moved code from `capdata.py` and import the helpers back**

Delete lines 399-544 (the six helper functions — stop before `ReportingIrradiance` at line 547) from `src/captest/capdata.py`.

Add to the local-import area near the top of `capdata.py` (after `from captest import util`, line 40):

```python
from captest.filters import (
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    sensor_filter,
)
```

> `perc_difference` and `abs_diff_from_average` are NOT re-imported — `perc_difference` is only used by `check_all_perc_diff_comb` (now in `filters.py`), and `abs_diff_from_average` has no `capdata.py` caller. The remaining four are still referenced by `capdata.py` (`filter_irr` in `ReportingIrradiance` and the `filter_irr` method; `filter_grps` in `predict_capacities`; `sensor_filter`/`check_all_perc_diff_comb` in the `filter_sensors` method).

- [ ] **Step 5: Run the move tests**

Run: `uv run pytest tests/test_filter_helpers_move.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS — pure move + re-import; no behavior change.

- [ ] **Step 7: Commit**

```bash
git add src/captest/filters.py src/captest/capdata.py tests/test_filter_helpers_move.py
git commit -m "refactor: move row-filter helpers into captest.filters module"
```

---

### Task 2: Add `BaseSummaryStep` and `BaseFilter` to `filters.py`

**Files:**
- Modify: `src/captest/filters.py` (add `param`/`warnings` imports + classes)
- Test: `tests/test_filter_classes.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_filter_classes.py`:

```python
"""Tests for the filter-step class hierarchy (BaseSummaryStep / BaseFilter)."""

import numpy as np
import pandas as pd
import pytest

from captest.capdata import CapData
from captest.filters import BaseSummaryStep, BaseFilter


def _make_capdata(n=5):
    """CapData with a tiny integer-indexed DataFrame and data_filtered set."""
    cd = CapData("test")
    cd.data = pd.DataFrame({"power": np.arange(n), "poa": np.arange(n) * 10.0})
    cd.data_filtered = cd.data.copy()
    return cd


class _DropFirstRow(BaseFilter):
    """Test-only filter: drops the first remaining row."""

    def _execute(self, capdata):
        return capdata.data_filtered.index[1:]


class TestBaseSummaryStep:
    def test_base_filter_is_summary_step(self):
        assert issubclass(BaseFilter, BaseSummaryStep)

    def test_custom_name_defaults_to_none(self):
        assert _DropFirstRow().custom_name is None

    def test_execute_not_implemented_on_base(self):
        cd = _make_capdata()
        with pytest.raises(NotImplementedError):
            BaseSummaryStep().run(cd)

    def test_run_records_runtime_state(self):
        cd = _make_capdata(n=5)
        step = _DropFirstRow()
        step.run(cd)
        assert step.pts_before == 5
        assert step.pts_after == 4
        assert step.pts_removed == 1
        assert list(step.ix_before) == [0, 1, 2, 3, 4]
        assert list(step.ix_after) == [1, 2, 3, 4]

    def test_run_appends_to_filters(self):
        cd = _make_capdata()
        step = _DropFirstRow()
        step.run(cd)
        assert cd.filters == [step]

    def test_run_reassigns_filters_list(self):
        """run() must reassign (not in-place append) so param watchers fire."""
        cd = _make_capdata()
        original = cd.filters
        _DropFirstRow().run(cd)
        assert cd.filters is not original

    def test_run_updates_data_filtered_transitionally(self):
        cd = _make_capdata(n=5)
        _DropFirstRow().run(cd)
        assert list(cd.data_filtered.index) == [1, 2, 3, 4]

    def test_run_warns_when_all_data_removed(self):
        cd = _make_capdata(n=1)
        with pytest.warns(UserWarning, match="removed all data"):
            _DropFirstRow().run(cd)

    def test_args_repr_default(self):
        assert _DropFirstRow().args_repr == "Default arguments"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py -v`
Expected: FAIL — `ImportError: cannot import name 'BaseSummaryStep' from 'captest.filters'`

- [ ] **Step 3: Add the base classes to `filters.py`**

Add `import param` and `import warnings` to the imports at the top of `src/captest/filters.py`, then append:

```python
class BaseSummaryStep(param.Parameterized):
    """Common ancestor for steps that appear in the filtering summary.

    Holds the shared lifecycle (`run`), the optional `custom_name` display
    parameter, and the `args_repr` rendering used by the summary table.
    Subclasses implement `_execute`, returning the pandas ``Index`` of rows
    to keep after the step.

    Runtime state (`pts_before`, `pts_after`, `pts_removed`, `ix_before`,
    `ix_after`) is set by `run` as plain attributes and is never serialized.
    """

    custom_name = param.String(
        default=None,
        allow_None=True,
        doc="Optional display name in the summary table.",
    )

    def run(self, capdata):
        """Execute the step, record runtime state, and append self to filters."""
        self.pts_before = capdata.data_filtered.shape[0]
        self.ix_before = capdata.data_filtered.index
        self.ix_after = self._execute(capdata)
        self.pts_after = len(self.ix_after)
        self.pts_removed = self.pts_before - self.pts_after
        capdata.filters = capdata.filters + [self]
        # Transitional: keep the legacy data_filtered attribute consistent
        # until data_filtered becomes a derived property (plan 4).
        capdata.data_filtered = capdata.data.loc[self.ix_after, :]
        if self.pts_after == 0:
            warnings.warn("The last filter removed all data!")

    def _execute(self, capdata):
        """Return a pandas Index of rows to keep. Implemented by subclasses."""
        raise NotImplementedError

    @property
    def args_repr(self):
        """Render configured (non-default, non-None) params for the summary."""
        skip = {"custom_name", "name"}
        items = [
            f"{k}={v}"
            for k, v in self.param.values().items()
            if k not in skip and v is not None
        ]
        return ", ".join(items) if items else "Default arguments"


class BaseFilter(BaseSummaryStep):
    """A pure row-filtering step.

    Adds no interface beyond `BaseSummaryStep`; exists to distinguish row
    filters from non-filter summary steps (e.g. RepCond, FitRegression) for
    GUI styling and type checks.
    """

    pass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add BaseSummaryStep and BaseFilter step classes"
```

---

### Task 3: Make `CapData` a `param.Parameterized` with a `filters` `param.List`

This is the GUI-groundwork task: a typed, watchable `filters` parameter on a `Parameterized` `CapData`.

**Files:**
- Modify: `src/captest/capdata.py` — class declaration (`class CapData(object):`), the `from captest.filters import (...)` block (add `BaseSummaryStep`), `__init__`, `copy()`, `reset_filter()`
- Test: `tests/test_filter_classes.py`

> Line numbers shifted after Task 1 deleted lines 399-544 (~146 lines) and added a 6-line import. **Locate by anchor string, not absolute line number.**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_filter_classes.py`:

```python
class TestCapDataFiltersParam:
    def test_capdata_is_parameterized(self):
        import param

        assert isinstance(CapData("x"), param.Parameterized)

    def test_filters_is_a_param_list(self):
        cd = CapData("x")
        assert "filters" in cd.param
        assert cd.filters == []

    def test_default_filters_distinct_per_instance(self):
        a, b = CapData("a"), CapData("b")
        assert a.filters is not b.filters

    def test_name_preserved_and_constant(self):
        cd = CapData("system_a")
        assert cd.name == "system_a"
        with pytest.raises(TypeError):
            cd.name = "other"

    def test_filters_item_type_enforced(self):
        cd = CapData("x")
        with pytest.raises(TypeError):
            cd.filters = ["not a step"]

    def test_run_triggers_filters_watcher(self):
        cd = _make_capdata()
        events = []
        cd.param.watch(lambda e: events.append(e), "filters")
        _DropFirstRow().run(cd)
        assert len(events) == 1
        assert isinstance(events[0].new[-1], _DropFirstRow)

    def test_copy_copies_filters_and_name(self):
        cd = _make_capdata()
        _DropFirstRow().run(cd)
        cd_c = cd.copy()
        assert len(cd_c.filters) == 1
        assert cd_c.filters is not cd.filters
        assert cd_c.name == cd.name

    def test_reset_filter_clears_filters(self):
        cd = _make_capdata()
        _DropFirstRow().run(cd)
        assert len(cd.filters) == 1
        cd.reset_filter()
        assert cd.filters == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestCapDataFiltersParam -v`
Expected: FAIL — `test_capdata_is_parameterized` fails (`CapData` is not `param.Parameterized`); `filters`-related tests fail with `AttributeError`.

- [ ] **Step 3: Make `CapData` `param.Parameterized` and declare the `filters` param**

In `src/captest/capdata.py`, add `BaseSummaryStep` to the import block introduced in Task 1:

```python
from captest.filters import (
    BaseSummaryStep,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    sensor_filter,
)
```

Change the class declaration:

```python
class CapData(param.Parameterized):
```

Immediately after the class docstring (before `def __init__`), add the class-level parameter:

```python
    filters = param.List(
        default=[],
        item_type=BaseSummaryStep,
        doc="Ordered pipeline of filter/summary steps applied to the data.",
    )
```

- [ ] **Step 4: Update `__init__`**

In `CapData.__init__`, change the super call and drop the manual `name` assignment (param now owns `name`). Replace:

```python
        super(CapData, self).__init__()
        self.name = name
```

with:

```python
        super().__init__(name=name)
```

Do **not** add `self.filters = []` — the class-level `param.List` default supplies a fresh per-instance list automatically (verified distinct per instance).

- [ ] **Step 5: Fix `copy()` for the constant `name` + copy `filters`**

In `CapData.copy()`, replace the construction + name line:

```python
        cd_c = CapData("")
        cd_c.name = copy.copy(self.name)
```

with:

```python
        cd_c = CapData(self.name)
```

Then, after the line `cd_c.summary = copy.copy(self.summary)`, add:

```python
        cd_c.filters = copy.deepcopy(self.filters)
```

> `deepcopy` is a deliberate stop-gap so copied steps are independent. Full copy semantics (and the `__copy__` simplification in the spec) are revisited in plan 4 when `data_filtered` becomes a property.
>
> **Known inconsistency (deferred, not fixed here):** `copy()` already does not copy the legacy `removed`/`kept`/`filter_counts` lists (a pre-existing gap). Adding the `filters` copy makes the asymmetry more visible — on a copy, `filters` is mirrored but `removed`/`kept` are empty. We intentionally do **not** patch `copy()` to mirror the legacy lists, because those lists are removed entirely in the summary-rebuild plan (plan 6); mirroring them now would be throwaway code. After plan 6, `filters` is the single source of truth and the inconsistency disappears. Do not rely on `removed`/`kept` being populated on a copied CapData in the interim.

- [ ] **Step 6: Clear `filters` in `reset_filter()`**

In `CapData.reset_filter()`, after the line `self.kept = []` add (reassignment, so watchers fire):

```python
        self.filters = []
```

- [ ] **Step 7: Run the new tests**

Run: `uv run pytest tests/test_filter_classes.py::TestCapDataFiltersParam -v`
Expected: PASS

- [ ] **Step 8: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS — behavior-additive; all pre-existing tests still pass. If any test fails on `CapData` construction or `name`, re-check Steps 3-5.

- [ ] **Step 9: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: make CapData param.Parameterized with watchable filters param.List"
```

---

## Self-Review

**1. Spec coverage (this plan's slice):**
- "Module Organization → `filters.py`": module created (Task 1), holds the moved row-filter helpers (Task 1) and the base classes (Task 2); one-way import direction enforced (`capdata.py` imports back, `filters.py` imports nothing from `capdata`). ✓
- "BaseSummaryStep" (run lifecycle, custom_name, args_repr, runtime state as plain attrs) → Task 2. ✓
- "BaseFilter" (empty distinguishing subclass) → Task 2. ✓
- "CapData Changes → Added: `filters = param.List(default=[], item_type=BaseSummaryStep)`" → Task 3, implemented exactly as the spec declares (not deferred). ✓
- "Panel GUI Enablement Notes → `param.List` mutation note" (reassignment notifies, in-place append does not) → satisfied by `run()` reassigning + verified by `test_run_triggers_filters_watcher`. ✓
- Deferred to later plans: concrete filters + wrappers, `data_filtered` property, summary rebuild, `rerun_from`, widgets/panes.

**2. Placeholder scan:** No TBDs. The one non-pasted block is the verbatim helper move (Task 1 Step 3), guarded by behavior tests in Step 1. Every other code step shows complete code; every run step shows command + expected result.

**3. Type/name consistency:** `run`, `_execute`, `custom_name`, `args_repr`, `pts_before/after/removed`, `ix_before/after`, `filters`, and the six helper names match the spec and are used consistently across tasks and tests. Test imports correctly split: `CapData` from `captest.capdata`, base classes from `captest.filters`. `filters` is declared once as a class-level `param.List`; `__init__` no longer assigns it.

**GUI groundwork delivered:** `CapData` is now `param.Parameterized` with a typed, watchable `filters` `param.List`. `cd.param.watch(cb, 'filters')` works, `item_type` is enforced, and `run()`'s reassignment fires watchers — the reactive data model the spec's "Panel GUI Enablement Notes" depend on. Widgets, `pn.Param`, and `rerun_from()` remain future work.
