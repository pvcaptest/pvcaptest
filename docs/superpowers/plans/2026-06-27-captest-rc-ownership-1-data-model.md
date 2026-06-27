# CapTest RC Ownership — Plan 1: Data Model + Resolution Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `CapTest.rc` as the single stored test reporting-conditions value (with `rc_source` provenance incl. `"manual"`), replace the `CapData.rc_source_resolved` CapData→CapData reference with a `CapData._captest` back-reference, and make `CapData.rep_irr` resolve from `ct.rc` inside a test (else `self.rc`).

**Architecture:** `CapTest` stores `_rc` (a one-row DataFrame or None) behind a read-only `rc` property and a private `_set_rc(df, source, warn)` write point that also sets the `rc_source` Selector. `CapTest.setup()` wires `cd._captest = self` on both CapData instances. `CapData.rep_irr` reads `self._captest.rc` when in a test, otherwise `self.rc`. This plan does **not** yet auto-populate `ct.rc` from `rep_cond` (Plan 3) or add the manual setter (Plan 2); tests here populate `ct.rc` directly via `_set_rc`.

**Tech Stack:** Python, `param` (Parameterized/Selector), pandas, pytest, `uv`, `just`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-captest-rc-ownership-design.md`.
- Standalone `CapData` behavior must not change: `cd.rep_cond()` still sets `cd.rc`; `cd.filter_irr(ref_val="rep_irr")` outside a test still resolves from `cd.rc`.
- `capdata.py` must NOT import `captest` — `_captest` is an opaque runtime reference only.
- Line length 88 (ruff). NumPy-style docstrings on public API.
- Execution is deferred until after the `filters-refactor` branch merges and releases.
- Run tests with `just -f ~/python/pvcaptest_bt-/.justfile test-module <file>`; lint with `uv run ruff check` and `uv run ruff format`.

---

### Task 1: `CapTest` RC storage (`_rc`, `_loading`, `rc` getter, `_set_rc`, `rc_source` += `"manual"`)

**Files:**
- Modify: `src/captest/captest.py` — `__init__` (around line 1546-1555), `rc_source` Selector (line 1347), add `rc` property + `_set_rc` after `__init__`.
- Test: `tests/test_captest.py` — new class `TestTestRc`.

**Interfaces:**
- Produces:
  - `CapTest.rc` → property returning the stored one-row `pandas.DataFrame` or `None`.
  - `CapTest._set_rc(rc: pandas.DataFrame, source: str, warn: bool = True) -> None` — sets `self._rc = rc` and `self.rc_source = source`; emits a `UserWarning` when `warn and self._rc is not None and source != self.rc_source`.
  - `CapTest.rc_source` Selector objects become `["meas", "sim", "manual"]`.
  - `CapTest._rc` (private, default `None`), `CapTest._loading` (private bool, default `False`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captest.py` (top of file already has `import pytest`, `import pandas as pd`, `from captest import CapTest`):

```python
class TestTestRc:
    """CapTest.rc storage, _set_rc write point, and source-change warning."""

    def _rc_df(self, poa=800.0):
        return pd.DataFrame({"poa": [poa], "t_amb": [25.0], "w_vel": [2.0]})

    def test_rc_defaults_none_and_source_meas(self):
        ct = CapTest()
        assert ct.rc is None
        assert ct.rc_source == "meas"
        assert ct._loading is False

    def test_set_rc_first_set_is_silent_and_records_source(self, recwarn):
        ct = CapTest()
        df = self._rc_df()
        ct._set_rc(df, "sim")
        assert ct.rc is df
        assert ct.rc_source == "sim"
        assert len(recwarn) == 0

    def test_set_rc_same_source_is_silent(self, recwarn):
        ct = CapTest()
        ct._set_rc(self._rc_df(), "meas")
        ct._set_rc(self._rc_df(810.0), "meas")
        assert ct.rc_source == "meas"
        assert len(recwarn) == 0

    def test_set_rc_source_change_warns(self):
        ct = CapTest()
        ct._set_rc(self._rc_df(), "meas")
        with pytest.warns(UserWarning, match="rc_source changed from 'meas' to 'sim'"):
            ct._set_rc(self._rc_df(), "sim")
        assert ct.rc_source == "sim"

    def test_set_rc_warn_false_suppresses(self, recwarn):
        ct = CapTest()
        ct._set_rc(self._rc_df(), "meas")
        ct._set_rc(self._rc_df(), "manual", warn=False)
        assert ct.rc_source == "manual"
        assert len(recwarn) == 0

    def test_rc_source_accepts_manual(self):
        ct = CapTest()
        ct.rc_source = "manual"  # must not raise (Selector now allows it)
        assert ct.rc_source == "manual"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `TestTestRc` errors (`AttributeError: 'CapTest' object has no attribute '_set_rc'` / `_loading`; `test_rc_source_accepts_manual` raises `ValueError` from the Selector).

- [ ] **Step 3: Extend the `rc_source` Selector**

In `src/captest/captest.py`, change the Selector at line ~1347 from:

```python
    rc_source = param.Selector(
        objects=["meas", "sim"],
        default="meas",
        doc="Which CapData provides reporting conditions: used by "
        "captest_results and as the source that filter_irr(ref_val='rep_irr') "
        "resolves against on both meas and sim.",
    )
```

to:

```python
    rc_source = param.Selector(
        objects=["meas", "sim", "manual"],
        default="meas",
        doc="Provenance of the single test RC (CapTest.rc): 'meas'/'sim' when "
        "computed from that dataset's rep_cond, or 'manual' when set directly. "
        "Seeds the default 'which' for rep_cond.",
    )
```

- [ ] **Step 4: Initialize `_rc` and `_loading` in `__init__`**

In `src/captest/captest.py`, in `__init__` (after `self._sim_path = None`, line ~1555), add:

```python
        # The single test reporting-conditions DataFrame (or None). Plain attr,
        # not a param.*, so the `rc` property setter can validate and the
        # `_set_rc` write point can manage provenance. `_loading` is True only
        # during from_yaml replay (see Plan 5) to seed RC from config.
        self._rc = None
        self._loading = False
```

- [ ] **Step 5: Add the `rc` getter and `_set_rc` write point**

In `src/captest/captest.py`, immediately after `__init__` (before the `# --- constructors ---` comment at line ~1557), add:

```python
    @property
    def rc(self):
        """The single test reporting-conditions DataFrame, or ``None``.

        Sourced from ``meas``/``sim`` via :meth:`rep_cond` or set manually via
        the property setter; provenance is tracked by :attr:`rc_source`. See the
        RC-ownership design spec for the full lifecycle.
        """
        return self._rc

    def _set_rc(self, rc, source, warn=True):
        """Single internal write point for ``_rc`` and ``rc_source``.

        Emits a source-change ``UserWarning`` when ``warn`` is True, an RC is
        already set, and ``source`` differs from the current ``rc_source``
        (silent on first establishment and same-source recompute).

        Parameters
        ----------
        rc : pandas.DataFrame
            One-row reporting-conditions DataFrame.
        source : {'meas', 'sim', 'manual'}
            Provenance to record in ``rc_source``.
        warn : bool, default True
            Suppress the source-change warning when False (used during load).
        """
        if warn and self._rc is not None and source != self.rc_source:
            warnings.warn(
                f"Test reporting conditions rc_source changed from "
                f"'{self.rc_source}' to '{source}'."
            )
        self._rc = rc
        self.rc_source = source
```

(`warnings` is already imported in `captest.py`.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for `TestTestRc`; rest of module still passes.

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass; files formatted.

- [ ] **Step 8: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: add CapTest.rc storage and _set_rc write point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `CapData._captest` back-reference, `rep_irr` rewrite, `setup()` wiring

**Files:**
- Modify: `src/captest/capdata.py` — `__init__` (line 842), `rep_irr` property (lines 943-980).
- Modify: `src/captest/captest.py` — `setup()` wiring (lines 2185-2193).
- Test: `tests/test_CapData.py` (standalone `rep_irr`), `tests/test_captest.py` (`setup()` wiring + in-test `rep_irr`).

**Interfaces:**
- Consumes: `CapTest.rc` / `CapTest._set_rc` (Task 1).
- Produces:
  - `CapData._captest` (private, default `None`; a `CapTest` reference or `None`).
  - `CapData.rep_irr` → `float`; reads `self._captest.rc` when `_captest is not None`, else `self.rc`; raises `ValueError` (distinct in-test vs standalone messages) when the resolved RC is `None`, and when it lacks a `"poa"` column.
  - `CapTest.setup()` sets `self.meas._captest = self` and `self.sim._captest = self`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_CapData.py` — replace the existing `TestRepIrr` class (lines ~2320-2353, the one using `rc_source_resolved`) with:

```python
class TestRepIrr:
    """The standalone CapData.rep_irr property (no CapTest attached)."""

    def test_rep_irr_reads_self_rc_when_standalone(self, nrel):
        """rep_irr reads self.rc when this CapData has no _captest."""
        assert nrel._captest is None
        nrel.rc = pd.DataFrame({"poa": [222.0], "t_amb": [20], "w_vel": [2]})
        assert nrel.rep_irr == pytest.approx(222.0)

    def test_rep_irr_standalone_none_rc_raises(self, nrel):
        """Standalone with no rc raises directing to rep_cond()."""
        assert nrel.rc is None
        with pytest.raises(ValueError, match="requires reporting conditions"):
            nrel.rep_irr

    def test_rep_irr_missing_poa_column_raises(self, nrel):
        """rep_irr raises when the resolved rc lacks a 'poa' column."""
        nrel.rc = pd.DataFrame({"irr": [500.0], "t_amb": [20], "w_vel": [2]})
        with pytest.raises(ValueError, match="requires a 'poa' column"):
            nrel.rep_irr
```

Add to `tests/test_captest.py` — replace the two `test_setup_wires_rc_source_resolved_*` methods in `TestSetup` (lines ~979-996) with:

```python
    def test_setup_wires_captest_backref_on_both(self, ct_default):
        """setup() wires _captest back to the CapTest on both CapData."""
        assert ct_default.meas._captest is ct_default
        assert ct_default.sim._captest is ct_default
```

And in `TestRepIrrCrossInstance` (class at line ~1061), replace
`test_sim_rep_irr_filter_uses_meas_reporting_irradiance` and
`test_sim_rep_irr_filter_roundtrips_through_yaml` with these (they set `ct.rc`
via `_set_rc` because `rep_cond` auto-sync lands in Plan 3):

```python
    def test_sim_rep_irr_resolves_from_ct_rc(self, ct_default):
        """sim.rep_irr reads the test-level ct.rc (set here via _set_rc)."""
        rc = pd.DataFrame({"poa": [777.0], "t_amb": [25.0], "w_vel": [2.0]})
        ct_default._set_rc(rc, "meas")
        assert ct_default.sim.rep_irr == pytest.approx(777.0)
        assert ct_default.meas.rep_irr == pytest.approx(777.0)

    def test_rep_irr_in_test_without_rc_raises(self, ct_default):
        """In a test, rep_irr with ct.rc unset raises directing to rep_cond."""
        assert ct_default.rc is None
        with pytest.raises(ValueError, match="test reporting conditions"):
            ct_default.sim.rep_irr

    def test_sim_filter_irr_rep_irr_uses_ct_rc(self, ct_default):
        """filter_irr(ref_val='rep_irr') on sim filters around ct.rc's poa."""
        rc = pd.DataFrame({"poa": [500.0], "t_amb": [25.0], "w_vel": [2.0]})
        ct_default._set_rc(rc, "meas")
        ct_default.sim.filter_irr(0.8, 1.2, ref_val="rep_irr")
        step = ct_default.sim.filters[-1]
        assert step.ref_val_resolved == pytest.approx(500.0)
        assert step.low_effective == pytest.approx(0.8 * 500.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_CapData.py` then `... test-module test_captest.py`
Expected: FAIL — `AttributeError: 'CapData' object has no attribute '_captest'` (and the in-test `rep_irr` tests fail because `rep_irr` still reads `rc_source_resolved`).

- [ ] **Step 3: Rename the CapData attribute in `__init__`**

In `src/captest/capdata.py` line 842, change:

```python
        self.rc_source_resolved = None
```

to:

```python
        # Back-reference to the owning CapTest (set by CapTest.setup), or None
        # for standalone use. Opaque runtime reference only — capdata.py never
        # imports captest. Intentionally NOT copied by CapData.copy().
        self._captest = None
```

- [ ] **Step 4: Rewrite the `rep_irr` property**

In `src/captest/capdata.py`, replace the whole `rep_irr` property (lines ~943-980, ending at the `return float(...)`):

```python
    @property
    def rep_irr(self):
        """Reporting POA irradiance anchoring relative irradiance filters.

        Resolved when ``filter_irr`` is called with ``ref_val='rep_irr'`` (or the
        legacy ``'self_val'``). Inside a CapTest (``self._captest`` set by
        ``CapTest.setup``) it reads the single test RC ``self._captest.rc``;
        standalone it reads this instance's own ``self.rc``.

        Returns
        -------
        float
            The ``'poa'`` reporting condition of the resolved RC.

        Raises
        ------
        ValueError
            If no RC is available, or the resolved RC lacks a ``'poa'`` column.
        """
        in_test = self._captest is not None
        rc = self._captest.rc if in_test else self.rc
        if rc is None:
            if in_test:
                raise ValueError(
                    "ref_val='rep_irr' requires test reporting conditions. Call "
                    "ct.rep_cond(which) or assign ct.rc = df before filtering."
                )
            raise ValueError(
                "ref_val='rep_irr' requires reporting conditions. Call "
                "rep_cond() before filtering with ref_val='rep_irr'."
            )
        if "poa" not in rc.columns:
            raise ValueError(
                "ref_val='rep_irr' requires a 'poa' column in the reporting "
                "conditions."
            )
        return float(rc["poa"].iloc[0])
```

- [ ] **Step 5: Rewrite the `setup()` wiring**

In `src/captest/captest.py`, replace lines ~2185-2193 (the `rc_source_resolved` block):

```python
        # Wire each CapData back to this CapTest so filter_irr(ref_val='rep_irr')
        # resolves against the single test RC (ct.rc), and (Plan 3) so cd.rep_cond
        # can update it. Runtime reference only; capdata.py never imports captest.
        # Assigned per side so a future per-side setup can re-wire one side alone.
        self.meas._captest = self
        self.sim._captest = self
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_CapData.py` then `... test-module test_captest.py`
Expected: PASS for the new/updated tests.

- [ ] **Step 7: Verify no stale `rc_source_resolved` references remain**

Run: `grep -rn "rc_source_resolved" src/ tests/`
Expected: no output.

- [ ] **Step 8: Lint**

Run: `uv run ruff check src/captest/capdata.py src/captest/captest.py tests/test_CapData.py tests/test_captest.py && uv run ruff format src/captest/capdata.py src/captest/captest.py tests/test_CapData.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 9: Commit**

```bash
git add src/captest/capdata.py src/captest/captest.py tests/test_CapData.py tests/test_captest.py
git commit -m "refactor: resolve rep_irr from CapTest.rc via _captest back-reference

Replace CapData.rc_source_resolved (CapData->CapData) with a _captest
back-reference; rep_irr reads the single test rc (ct.rc) in a test, else
self.rc. setup() wires _captest on both sides.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Full-suite green + standalone regression guard

**Files:**
- Test: full suite (no new source changes expected; this task is the integration gate).

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: nothing new — verifies the plan's deliverable holds across the whole suite.

- [ ] **Step 1: Run the full test suite**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test`
Expected: all tests pass. If any test outside the updated classes references `rc_source_resolved` or relied on the old per-CapData `rep_irr` resolution, fix it to the `_captest`/`ct.rc` model (set `ct.rc` via `_set_rc` in test setup) and re-run.

- [ ] **Step 2: Confirm standalone CapData behavior is unchanged**

Run: `uv run pytest tests/test_CapData.py -q -k "rep_cond or filter_irr or RepIrr"`
Expected: PASS — standalone `rep_cond`/`filter_irr` tests are green, demonstrating no behavior change for CapData used without a CapTest.

- [ ] **Step 3: Lint + format the whole change**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/captest/capdata.py src/captest/captest.py`
Expected: All checks pass.

- [ ] **Step 4: Commit (only if Step 1 required fixes; otherwise skip)**

```bash
git add -A
git commit -m "test: migrate remaining suites to _captest/ct.rc resolution model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Plan 1 scope = §4.1 data model, §4.2 resolution, §4.8 setup wiring, partial §4.5 via `_set_rc`):**
- §4.1 `CapTest.rc` property + `_rc` + `rc_source` incl. `"manual"` → Task 1. ✓
- §4.1 `CapData._captest` replacing `rc_source_resolved` → Task 2. ✓
- §4.2 `rep_irr` reads `ct.rc` in test else `self.rc`, with messages → Task 2. ✓
- §4.5 source-change warning logic lives in `_set_rc` → Task 1 (the rep_cond/manual *callers* are Plans 2-3). ✓
- §4.8 `setup()` wires `_captest`; does not touch `ct.rc` → Task 2. ✓
- Out of scope (later plans): manual setter (Plan 2), `rep_cond` auto-sync (Plan 3), `captest_results` (Plan 4), serialization/load incl. `_loading` use (Plan 5). `_loading` is *declared* here (Task 1) but only read in Plan 5.

**Placeholder scan:** none — every step shows exact code/commands.

**Type consistency:** `_set_rc(rc, source, warn=True)` signature is consistent across Task 1 definition and Task 2/3 usage; `rc` getter returns DataFrame|None; `_captest` is CapTest|None; `rep_irr` returns float. The `TestRepIrrCrossInstance` tests set `ct.rc` via `_set_rc` (the only write path available until Plan 2's setter / Plan 3's sync).
