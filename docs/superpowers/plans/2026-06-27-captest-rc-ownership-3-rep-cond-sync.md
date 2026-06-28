# CapTest RC Ownership — Plan 3: Last-Writer-Wins `rep_cond → ct.rc` Sync

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make any `cd.rep_cond()` on a CapData that belongs to a test update the single test RC (`ct.rc` + `ct.rc_source`), with a warning only when the source changes; and make `ct.rep_cond()`'s `which` default to the current `rc_source`.

**Architecture:** `CapData._calc_rep_cond` calls `self._captest._on_capdata_rep_cond(self)` after it sets `self.rc` (only when `_captest` is set — standalone is untouched, and the call is duck-typed so `capdata.py` still never imports `captest`). `CapTest._on_capdata_rep_cond(cd)` resolves which side `cd` is and records the value via `_set_rc` (Plan 1). It has two modes: runtime last-writer-wins (warn on source change) and a config-seeded `_loading` branch (only the configured `rc_source` side updates, silently) — the `_loading` branch is implemented here but stays dormant until Plan 5 sets `_loading`.

**Tech Stack:** Python, `param`, pandas, pytest, `uv`, `just`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-captest-rc-ownership-design.md` §4.3, §4.5.
- Builds on Plans 1-2 (`CapTest.rc`/`_set_rc`/`_loading`, `CapData._captest`, manual setter). Apply them first.
- **Plans 1-5 must merge and release as a unit.** This plan makes the *runtime* cross-instance workflow (`ct.meas.rep_cond(); ct.sim.filter_irr(ref_val="rep_irr")`) functional; the `from_yaml` load path for the `rc_source="sim"` and manual cases still needs Plan 5.
- `capdata.py` must NOT import `captest` — `self._captest._on_capdata_rep_cond(...)` is an opaque runtime call.
- Standalone `CapData` (no `_captest`) behavior must not change: `cd.rep_cond()` sets only `cd.rc`, no sync, no warning.
- Line length 88 (ruff). NumPy-style docstrings. Run tests with `just -f ~/python/pvcaptest_bt-/.justfile test-module <file>`; lint with `uv run ruff check` / `uv run ruff format`.

---

### Task 1: `_on_capdata_rep_cond` + sync call from `_calc_rep_cond`

**Files:**
- Modify: `src/captest/captest.py` — add `_on_capdata_rep_cond` immediately after `_set_rc`.
- Modify: `src/captest/capdata.py` — `_calc_rep_cond`, after `self.rc = RCs_df` (line ~2245).
- Test: `tests/test_captest.py` — new class `TestRepCondSync`.

**Interfaces:**
- Consumes: `CapTest._set_rc(rc, source, warn=True)` (Plan 1), `CapTest._loading` (Plan 1), `CapData._captest` (Plan 1), `CapData.rc`.
- Produces:
  - `CapTest._on_capdata_rep_cond(cd: CapData) -> None` — called by `CapData._calc_rep_cond` when `cd._captest is self`. Runtime: `_set_rc(cd.rc.copy(), side, warn=True)` where `side` is `"meas"`/`"sim"`. Load (`_loading`): updates only when `side == self.rc_source`, `warn=False`.
  - `CapData._calc_rep_cond` now ends by calling `self._captest._on_capdata_rep_cond(self)` when `self._captest is not None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captest.py` (module already imports `pytest`, `pandas as pd`; `ct_default` and `pvsyst` are fixtures — `pvsyst` is a standalone CapData with `regression_cols` set, defined in `conftest.py`):

```python
class TestRepCondSync:
    """cd.rep_cond updates ct.rc (last-writer-wins); standalone is unaffected."""

    def test_meas_rep_cond_sets_ct_rc_from_meas(self, ct_default):
        ct_default.meas.rep_cond()
        assert ct_default.rc_source == "meas"
        assert ct_default.rc is not None
        assert ct_default.rc["poa"].iloc[0] == pytest.approx(
            ct_default.meas.rc["poa"].iloc[0]
        )

    def test_meas_rep_cond_first_set_is_silent(self, ct_default, recwarn):
        ct_default.meas.rep_cond()
        assert not any(
            "rc_source changed" in str(w.message) for w in recwarn
        )

    def test_sim_rep_cond_after_meas_flips_source_and_warns(self, ct_default):
        ct_default.meas.rep_cond()
        with pytest.warns(UserWarning, match="changed from 'meas' to 'sim'"):
            ct_default.sim.rep_cond()
        assert ct_default.rc_source == "sim"
        assert ct_default.rc["poa"].iloc[0] == pytest.approx(
            ct_default.sim.rc["poa"].iloc[0]
        )

    def test_same_source_recompute_is_silent(self, ct_default, recwarn):
        ct_default.meas.rep_cond()
        ct_default.meas.filter_irr(200, 800)
        ct_default.meas.rep_cond()  # still 'meas' -> no source-change warning
        assert ct_default.rc_source == "meas"
        assert not any(
            "rc_source changed" in str(w.message) for w in recwarn
        )

    def test_ct_rc_is_a_copy_not_aliased(self, ct_default):
        ct_default.meas.rep_cond()
        # Mutating meas.rc must not change ct.rc.
        ct_default.meas.rc.loc[ct_default.meas.rc.index[0], "poa"] = -999.0
        assert ct_default.rc["poa"].iloc[0] != -999.0

    def test_standalone_rep_cond_does_not_sync_or_warn(self, pvsyst, recwarn):
        assert pvsyst._captest is None
        pvsyst.filter_irr(200, 800)
        pvsyst.rep_cond()  # must not raise (the _captest guard) and not warn
        assert pvsyst.rc is not None
        assert not any(
            "rc_source changed" in str(w.message) for w in recwarn
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `ct.rc` stays `None` after `meas.rep_cond()` (no sync yet), so `test_meas_rep_cond_sets_ct_rc_from_meas` fails on `ct.rc is not None`, and the flip/warn test does not warn.

- [ ] **Step 3: Add `_on_capdata_rep_cond` to `CapTest`**

In `src/captest/captest.py`, immediately after the `_set_rc` method (added in Plan 1), add:

```python
    def _on_capdata_rep_cond(self, cd):
        """Update the test RC after a member CapData computed its own ``rc``.

        Called by :meth:`CapData._calc_rep_cond` when the CapData belongs to this
        test. Runtime behavior is last-writer-wins: the calling side's ``rc``
        becomes ``ct.rc`` and ``rc_source`` (a source-change ``UserWarning`` is
        emitted by :meth:`_set_rc`). During ``from_yaml`` load (``_loading``
        True) the update is config-seeded: only the configured ``rc_source``
        side updates ``ct.rc``, silently (see Plan 5 / spec §4.7).

        Parameters
        ----------
        cd : CapData
            The member CapData that just (re)computed its ``rc``.
        """
        side = "meas" if cd is self.meas else "sim"
        if self._loading:
            if side == self.rc_source:
                self._set_rc(cd.rc.copy(), side, warn=False)
            return
        self._set_rc(cd.rc.copy(), side, warn=True)
```

- [ ] **Step 4: Call the sync from `_calc_rep_cond`**

In `src/captest/capdata.py`, at the end of `_calc_rep_cond` (after `self.rc = RCs_df`, line ~2245), add:

```python
        # When this CapData belongs to a CapTest, propagate the freshly computed
        # rc to the single test rc (last-writer-wins). The CapTest decides warn
        # vs silent and config-seeded load behavior. Opaque call — capdata.py
        # never imports captest.
        if self._captest is not None:
            self._captest._on_capdata_rep_cond(self)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for `TestRepCondSync`.

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/captest/capdata.py src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/capdata.py src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 7: Commit**

```bash
git add src/captest/capdata.py src/captest/captest.py tests/test_captest.py
git commit -m "feat: sync cd.rep_cond to CapTest.rc (last-writer-wins)

A cd.rep_cond on a test member updates ct.rc/rc_source via
_on_capdata_rep_cond, warning only on a source change; standalone CapData
is unaffected. Includes the dormant _loading config-seeded branch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ct.rep_cond()` `which` defaults to current `rc_source`

**Files:**
- Modify: `src/captest/captest.py` — `rep_cond` (signature + body, lines ~2278-2306).
- Test: `tests/test_captest.py` — add to `TestRepCondSync`.

**Interfaces:**
- Consumes: `CapTest.rc_source`, `CapTest._pick_cd`, `_merge_rep_conditions`, `CapData.rep_cond`.
- Produces: `CapTest.rep_cond(which=None, **overrides)` — when `which is None`, resolves to `self.rc_source` if it is `"meas"`/`"sim"`, else `"meas"`.

- [ ] **Step 1: Write the failing tests**

Add to `TestRepCondSync` in `tests/test_captest.py`:

```python
    def test_ct_rep_cond_default_which_follows_rc_source(self, ct_default, recwarn):
        ct_default.sim.rep_cond()        # rc_source -> 'sim'
        ct_default.rep_cond()            # which=None -> should pick 'sim'
        assert ct_default.rc_source == "sim"
        # If the default had picked 'meas', this would flip+warn.
        assert not any(
            "rc_source changed" in str(w.message) for w in recwarn
        )

    def test_ct_rep_cond_explicit_which_sim(self, ct_default):
        ct_default.rep_cond("sim")
        assert ct_default.rc_source == "sim"
        assert ct_default.rc is not None

    def test_ct_rep_cond_default_which_meas_when_rc_source_manual(self, ct_default):
        # Seed a manual source, then default rep_cond should fall back to 'meas'.
        ct_default.rc = pd.DataFrame(
            {"poa": [800.0], "t_amb": [25.0], "w_vel": [2.0]}
        )
        assert ct_default.rc_source == "manual"
        ct_default.rep_cond()  # which=None, rc_source='manual' -> 'meas'
        assert ct_default.rc_source == "meas"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — with `which="meas"` still the default, `test_ct_rep_cond_default_which_follows_rc_source` flips to `"meas"` and warns.

- [ ] **Step 3: Change the `rep_cond` signature and resolve `which`**

In `src/captest/captest.py`, change the signature (line ~2278) from `def rep_cond(self, which="meas", **overrides):` to `def rep_cond(self, which=None, **overrides):` and update the body so it starts:

```python
        if which is None:
            which = self.rc_source if self.rc_source in ("meas", "sim") else "meas"
        cd = self._pick_cd(which)
        self._require_setup()
        resolved_rc = _merge_rep_conditions(
            self._resolved_setup["rep_conditions"], overrides
        )
        return cd.rep_cond(**resolved_rc)
```

Also update the `rep_cond` docstring's `which` parameter entry to:

```python
        which : {'meas', 'sim', None}, default None
            Which CapData to compute reporting conditions on. When None, defaults
            to the current ``rc_source`` if it is ``'meas'``/``'sim'``, otherwise
            ``'meas'``. The computed conditions become the test ``rc`` (and set
            ``rc_source`` to ``which``) via the last-writer-wins sync.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for the three new tests.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: ct.rep_cond which defaults to current rc_source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Un-mark the Plan 1 `xfail` placeholder + full-suite green

**Files:**
- Modify: `tests/test_captest.py` — `TestRepIrrCrossInstance.test_meas_rep_cond_then_sim_rep_irr_roundtrips` (drop the `xfail` marker).

**Interfaces:**
- Consumes: the runtime sync from Tasks 1-2.
- Produces: nothing new — converts the deferred placeholder into a live passing test now that the runtime path works.

- [ ] **Step 1: Confirm the placeholder now XPASSes**

Run: `uv run pytest "tests/test_captest.py::TestRepIrrCrossInstance::test_meas_rep_cond_then_sim_rep_irr_roundtrips" -rXp`
Expected: XPASS — `meas.rep_cond()` now seeds `ct.rc`, so `sim.filter_irr(ref_val="rep_irr")` resolves and `to_yaml` succeeds. (The test only exercises `to_yaml`; the `from_yaml` round-trip for the `rc_source="sim"`/manual cases is added in Plan 5.)

- [ ] **Step 2: Remove the `xfail` marker**

In `tests/test_captest.py`, delete the decorator above `test_meas_rep_cond_then_sim_rep_irr_roundtrips`:

```python
    @pytest.mark.xfail(
        reason="rep_cond->ct.rc auto-sync lands in Plan 3 and the rep_irr YAML "
        "round-trip in Plan 5; placeholder keeps the deferred end-to-end "
        "coverage gap visible. Plan 5 removes this marker.",
        strict=False,
    )
```

Leave the test body unchanged; it now passes as a live test.

- [ ] **Step 3: Run the full suite**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test`
Expected: all tests pass; no XFAIL/XPASS remaining for this test.

- [ ] **Step 4: Lint**

Run: `uv run ruff check tests/test_captest.py && uv run ruff format tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_captest.py
git commit -m "test: un-xfail cross-instance rep_cond->rep_irr now that sync lands

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Plan 3 scope = §4.3 last-writer-wins + §4.5 callers):**
- §4.3 any `cd.rep_cond()` in a test updates `ct.rc` + `rc_source` → Task 1 (`_on_capdata_rep_cond` called from `_calc_rep_cond`). ✓
- §4.3 `ct.rc` stores a *copy* → `cd.rc.copy()`; `test_ct_rc_is_a_copy_not_aliased`. ✓
- §4.3 standalone unaffected (no `_captest` → no sync/warn) → `test_standalone_rep_cond_does_not_sync_or_warn`. ✓
- §4.3 `ct.rep_cond()` `which` default to current `rc_source` (else meas) → Task 2. ✓
- §4.5 warn only on source change, uniformly (incl. via `ct.rep_cond`) → handled by `_set_rc` (Plan 1); `test_sim_rep_cond_after_meas_flips_source_and_warns`, `test_same_source_recompute_is_silent`, `test_meas_rep_cond_first_set_is_silent`. ✓
- `_loading` config-seeded branch implemented but dormant (set only in Plan 5). ✓

**Placeholder scan:** none — every step has exact code/commands. The Plan 1 `xfail` is intentionally *removed* in Task 3 (the deferred path it guarded is now live for the runtime case).

**Cross-plan note:** the Plan 1 placeholder exercises `to_yaml` only, which passes after Task 1-2; the genuine `from_yaml` round-trips for `rc_source="sim"` and the manual case (spec testing-plan items 8-11) are added in Plan 5. No coverage is silently dropped — Plan 5 introduces those tests alongside the load behavior they exercise.

**Type consistency:** `_on_capdata_rep_cond(cd)` calls `_set_rc(cd.rc.copy(), side, warn=...)` with `side ∈ {"meas","sim"}`, matching the Plan 1 `_set_rc` signature and the `rc_source` Selector objects. `rep_cond(which=None, ...)` resolves `which` to a valid `_pick_cd` argument.
