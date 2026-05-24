# Clear-Sky Module Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the clear-sky *modeling* functions out of `capdata.py` into a new `src/captest/clearsky.py` module, updating all consumers.

**Architecture:** Move `pvlib_location`, `pvlib_system`, `get_tz_index`, and `csky` (capdata.py:918-1140) plus the clear-sky-modeling pvlib import guard into `clearsky.py`. `capdata.py` retains only the `detect_clearsky` pvlib import (still used by the `filter_clearsky` method until the later `FilterClearsky` work). Consumers — `io.py`, `captest/__init__.py`, and the test suite — are updated to import `csky` from the new module. No backward-compat shim is left behind (pre-1.0 package; internal references updated in the same change).

**Tech Stack:** Python, pvlib (optional dependency), pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Module Organization → `src/captest/clearsky.py`".

**Why this is a separate plan:** Clear-sky *modeling* (`csky` for `load_data` site metadata) is independent of the filter-class refactor. `FilterClearsky` uses `detect_clearsky` directly from pvlib and does **not** depend on this module. Extracting it first leaves `capdata.py` smaller before the filter work begins, and is independently testable.

**Sequencing:** Execute this plan first (before the filter-class plans). It does not depend on them.

---

### Task 1: Create `clearsky.py` with the modeling functions

**Files:**
- Create: `src/captest/clearsky.py`
- Modify: `src/captest/capdata.py` (remove lines 918-1140 and the modeling pvlib imports at 77-80; keep `detect_clearsky`)

- [ ] **Step 1: Create the new module**

Create `src/captest/clearsky.py`. Move the four functions verbatim from `capdata.py:918-1140` (`pvlib_location`, `pvlib_system`, `get_tz_index`, `csky`) and add this header with the modeling pvlib import guard:

```python
"""Clear-sky irradiance modeling built on pvlib.

These functions model clear-sky GHI/POA (used by ``io.load_data`` when site
metadata is supplied). Clear-sky *filtering* (``FilterClearsky``) is separate
and uses ``pvlib.clearsky.detect_clearsky`` directly.
"""

import importlib.util
import warnings

import pandas as pd

pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.location import Location
    from pvlib.pvsystem import PVSystem, Array, FixedMount, SingleAxisTrackerMount
    from pvlib.pvsystem import retrieve_sam
    from pvlib.modelchain import ModelChain
else:
    warnings.warn("Clear sky functions will not work without the pvlib package.")


# <-- the four moved functions go here, unchanged -->
```

> Verify the moved function bodies reference only names now present in `clearsky.py` (`Location`, `PVSystem`, `Array`, `FixedMount`, `SingleAxisTrackerMount`, `retrieve_sam`, `ModelChain`, `pd`). They do not reference `CapData` or any other `capdata.py` symbol.

- [ ] **Step 2: Remove the moved code from `capdata.py`**

Delete lines 918-1140 (`pvlib_location` through `csky`) from `src/captest/capdata.py`.

Then edit the pvlib import guard (currently capdata.py:76-83) to keep **only** `detect_clearsky`:

```python
pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn("Clear sky functions will not work without the pvlib package.")
```

- [ ] **Step 3: Verify capdata still imports**

Run: `uv run python -c "import captest.capdata"`
Expected: no error (no `NameError` for removed pvlib symbols). If a `NameError` appears, a `capdata.py` method still references a moved modeling symbol — re-check.

- [ ] **Step 4: Commit**

```bash
git add src/captest/clearsky.py src/captest/capdata.py
git commit -m "refactor: extract clear-sky modeling into captest.clearsky module"
```

---

### Task 2: Update consumers (io, __init__, tests)

**Files:**
- Modify: `src/captest/io.py:15`
- Modify: `src/captest/__init__.py`
- Modify: `tests/test_CapData.py` (import + `pvc.csky` references at lines 1344, 1367, 1409, 1427, 1449, 1465)

- [ ] **Step 1: Update `io.py` import**

In `src/captest/io.py`, change line 15:

```python
from captest.clearsky import csky
```

(The `csky(...)` call at io.py:621 is unchanged.)

- [ ] **Step 2: Re-export the new submodule in `__init__.py`**

In `src/captest/__init__.py`, add `clearsky` to the `from captest import (...)` block:

```python
from captest import (
    capdata as capdata,
    util as util,
    prtest as prtest,
    columngroups as columngroups,
    io as io,
    plotting as plotting,
    calcparams as calcparams,
    captest as captest,
    clearsky as clearsky,
)
```

- [ ] **Step 3: Update test references**

In `tests/test_CapData.py`, add near the existing imports (after line 17 `from captest import capdata as pvc`):

```python
from captest import clearsky
```

Then replace every `pvc.csky(` with `clearsky.csky(` (lines 1344, 1367, 1409, 1427, 1449, 1465; leave the commented line 1391 as-is or update its comment text to match).

- [ ] **Step 4: Run the clear-sky tests**

Run: `uv run pytest tests/test_CapData.py -k "csky or clear or clearsky" -v`
Expected: PASS (same tests that passed before the move).

- [ ] **Step 5: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS — pure move + import updates, no behavior change.

- [ ] **Step 6: Lint and format**

Run: `just lint && just fmt`
Expected: clean. (Note: `io.py`'s `E501` ruff exception is unaffected.)

- [ ] **Step 7: Commit**

```bash
git add src/captest/io.py src/captest/__init__.py tests/test_CapData.py
git commit -m "refactor: point csky consumers at captest.clearsky"
```

---

## Self-Review

**1. Spec coverage:** "Module Organization → `clearsky.py`" lists `pvlib_location`, `pvlib_system`, `get_tz_index`, `csky` + the modeling pvlib guard (Task 1); io/__init__/test consumer updates and "no shim" (Task 2). `FilterClearsky` independence from this module is noted and respected (`detect_clearsky` stays in `capdata.py`). ✓

**2. Placeholder scan:** The one non-pasted block is the verbatim move of four existing functions (Task 1 Step 1), with an explicit verification step that they reference no `capdata.py` symbols — appropriate for a pure move rather than re-pasting ~220 lines.

**3. Type/name consistency:** `csky`, `pvlib_location`, `pvlib_system`, `get_tz_index` names are preserved exactly; only their module home changes. `detect_clearsky` deliberately remains in `capdata.py`.
