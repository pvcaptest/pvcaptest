# CapTest RC Ownership — Plan 5: Serialization + Deterministic Load

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist/restore the test RC across `to_yaml`/`from_yaml`: serialize *values* only for the manual case, and on load (a) seed a manual RC before replay, (b) replay the configured `rc_source` side first so cross-side `rep_irr` filters resolve mid-replay, and (c) make the `rep_cond` sync config-seeded and warning-suppressed via `_loading`.

**Architecture:** `_build_yaml_sub_mapping` adds `reporting_conditions_values` (a yaml-safe one-row dict) only when `rc_source == "manual"`; computed RC is recomputed by replaying the source pipeline's `RepCond` step. `from_mapping` seeds a manual RC via `_set_rc(..., warn=False)` before replay, orders the two pipeline replays so the configured `rc_source` side runs first, and wraps replay in `_loading = True` (consumed by Plan 3's `_on_capdata_rep_cond` loading branch).

**Tech Stack:** Python, `param`, pandas, pytest, `unittest.mock.MagicMock`, `uv`, `just`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-captest-rc-ownership-design.md` §4.7.
- Builds on Plans 1-4 (`CapTest.rc`/`_set_rc`/`_loading`, manual setter, `rep_cond` sync + `_on_capdata_rep_cond` loading branch, `captest_results` reads `ct.rc`). Apply them first.
- **Plans 1-5 must merge and release as a unit.** This is the final plan; after it the full design is live.
- `to_native` and `pd` are already imported in `captest.py`.
- `from_yaml` delegates to `from_mapping`; the replay block lives in `from_mapping`.
- Computed-RC serialization assumes the `rc_source` side's pipeline contains the `RepCond` step that produced `ct.rc` (true whenever `ct.rc` came from `rep_cond`). Manual RC carries its values; no `RepCond` is relied upon.
- Line length 88 (ruff). NumPy-style docstrings. Run tests with `just -f ~/python/pvcaptest_bt-/.justfile test-module <file>`; lint with `uv run ruff check` / `uv run ruff format`.

---

### Task 1: Serialize manual RC values in `_build_yaml_sub_mapping`

**Files:**
- Modify: `src/captest/captest.py` — `_build_yaml_sub_mapping`, just before `return sub`.
- Test: `tests/test_captest.py` — new class `TestManualRcSerialization`.

**Interfaces:**
- Consumes: `CapTest._rc`, `CapTest.rc_source`, `util.to_native` (imported as `to_native`).
- Produces: the captest sub-mapping gains an optional `reporting_conditions_values` key — a `{column: native_scalar}` one-row dict — present iff `rc_source == "manual"` and `_rc is not None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captest.py` (module imports `pytest`, `pandas as pd`, `CapTest`; `meas_cd_default`/`sim_cd_default` are `conftest.py` fixtures):

```python
class TestManualRcSerialization:
    """to_yaml serializes manual RC values; computed RC is not value-serialized."""

    def _capt(self, meas_cd_default, sim_cd_default):
        return CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
        )

    def test_manual_rc_serializes_native_values(self, meas_cd_default, sim_cd_default):
        capt = self._capt(meas_cd_default, sim_cd_default)
        capt.rc = pd.DataFrame({"poa": [805.0], "t_amb": [25.0], "w_vel": [2.0]})
        sub = capt._build_yaml_sub_mapping()
        assert sub["rc_source"] == "manual"
        vals = sub["reporting_conditions_values"]
        assert vals["poa"] == pytest.approx(805.0)
        assert {"poa", "t_amb", "w_vel"} <= set(vals)
        # Values must be native python types so yaml.safe_dump can represent them.
        assert type(vals["poa"]) is float

    def test_computed_rc_omits_values(self, meas_cd_default, sim_cd_default):
        capt = self._capt(meas_cd_default, sim_cd_default)
        capt.rep_cond(which="meas")  # rc_source='meas' (computed)
        sub = capt._build_yaml_sub_mapping()
        assert sub["rc_source"] == "meas"
        assert "reporting_conditions_values" not in sub

    def test_no_rc_omits_values(self, meas_cd_default, sim_cd_default):
        capt = self._capt(meas_cd_default, sim_cd_default)
        sub = capt._build_yaml_sub_mapping()
        assert "reporting_conditions_values" not in sub
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `KeyError: 'reporting_conditions_values'` for the manual case (key not written yet).

- [ ] **Step 3: Serialize manual RC values**

In `src/captest/captest.py`, in `_build_yaml_sub_mapping`, replace:

```python
        if meas_filters:
            sub["meas_filters"] = meas_filters
        if sim_filters:
            sub["sim_filters"] = sim_filters

        return sub
```

with:

```python
        if meas_filters:
            sub["meas_filters"] = meas_filters
        if sim_filters:
            sub["sim_filters"] = sim_filters

        # Manual reporting conditions are data, not config: serialize their
        # values so from_yaml can restore them (computed RC is recomputed by
        # replaying the source pipeline's RepCond step). Numpy scalars are
        # coerced to native python types for yaml.safe_dump.
        if self.rc_source == "manual" and self._rc is not None:
            row = self._rc.iloc[0]
            sub["reporting_conditions_values"] = {
                str(col): to_native(val) for col, val in row.items()
            }

        return sub
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for `TestManualRcSerialization`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: serialize manual reporting-conditions values to yaml

When rc_source='manual', write the one-row RC under reporting_conditions_values
(numpy scalars coerced via to_native). Computed RC is omitted; it is recomputed
on load by replaying the source pipeline's RepCond step.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Deterministic load — manual seed, `_loading`, source-first replay

**Files:**
- Modify: `src/captest/captest.py` — `from_mapping` replay block.
- Test: `tests/test_captest.py` — new class `TestRcOwnershipRoundTrip`.

**Interfaces:**
- Consumes: `CapTest._set_rc` (Plan 1), `CapTest._loading` (Plan 1), `CapTest.rc_source`, `CapData.run_pipeline`, `_on_capdata_rep_cond` loading branch (Plan 3).
- Produces: `from_mapping` now (1) seeds `ct.rc` from `reporting_conditions_values` via `_set_rc(df, "manual", warn=False)` before replay when `rc_source == "manual"`, (2) replays the configured `rc_source` side's pipeline first (`sim`→`meas` when `rc_source == "sim"`, else `meas`→`sim`), and (3) sets `inst._loading = True` for the duration of replay (cleared in a `finally`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captest.py` (module already imports `from unittest.mock import MagicMock`):

```python
class TestRcOwnershipRoundTrip:
    """to_yaml/from_yaml round-trips of the single test RC across sources."""

    def _capt(self, meas_cd_default, sim_cd_default, **kw):
        return CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
            **kw,
        )

    def _roundtrip(self, capt, tmp_path, clean_meas, clean_sim):
        capt._meas_path = str(tmp_path / "meas.csv")
        capt._sim_path = str(tmp_path / "sim.csv")
        path = tmp_path / "config.yaml"
        capt.to_yaml(path, merge_into_existing=False)
        return CapTest.from_yaml(
            path,
            meas_loader=MagicMock(return_value=clean_meas),
            sim_loader=MagicMock(return_value=clean_sim),
        )

    def test_roundtrip_computed_meas_recomputes_ct_rc(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        capt = self._capt(meas_cd_default, sim_cd_default)
        clean_meas, clean_sim = capt.meas.copy(), capt.sim.copy()
        capt.meas.filter_irr(200, 800)
        capt.rep_cond(which="meas")
        expected_poa = capt.rc["poa"].iloc[0]

        reloaded = self._roundtrip(capt, tmp_path, clean_meas, clean_sim)

        assert reloaded.rc_source == "meas"
        assert reloaded.rc is not None
        assert reloaded.rc["poa"].iloc[0] == pytest.approx(expected_poa)

    def test_roundtrip_manual_rc_restores_values(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        capt = self._capt(meas_cd_default, sim_cd_default)
        clean_meas, clean_sim = capt.meas.copy(), capt.sim.copy()
        capt.rc = pd.DataFrame({"poa": [805.0], "t_amb": [25.0], "w_vel": [2.0]})

        reloaded = self._roundtrip(capt, tmp_path, clean_meas, clean_sim)

        assert reloaded.rc_source == "manual"
        assert reloaded.rc["poa"].iloc[0] == pytest.approx(805.0)

    def test_roundtrip_rc_source_sim_replays_sim_first(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        # The OTHER side (meas) carries the rep_irr filter; sim is the source.
        # This fails under a fixed meas-before-sim load order.
        capt = self._capt(meas_cd_default, sim_cd_default, rc_source="sim")
        clean_meas, clean_sim = capt.meas.copy(), capt.sim.copy()
        capt.sim.filter_irr(200, 800)
        capt.rep_cond(which="sim")  # ct.rc from sim, rc_source='sim'
        capt.meas.filter_irr(0.8, 1.2, ref_val="rep_irr")  # anchors on sim rep irr

        reloaded = self._roundtrip(capt, tmp_path, clean_meas, clean_sim)

        assert reloaded.rc_source == "sim"
        meas_irr = [
            s for s in reloaded.meas.filters if type(s).__name__ == "Irradiance"
        ][-1]
        assert meas_irr.ref_val_resolved == pytest.approx(reloaded.rc["poa"].iloc[0])

    def test_load_is_config_seeded_and_silent(
        self, tmp_path, meas_cd_default, sim_cd_default, recwarn
    ):
        capt = self._capt(meas_cd_default, sim_cd_default)
        clean_meas, clean_sim = capt.meas.copy(), capt.sim.copy()
        capt.rep_cond(which="sim")  # sim RepCond step (rc_source -> 'sim')
        capt.rep_cond(which="meas")  # meas RepCond step (rc_source -> 'meas')
        assert capt.rc_source == "meas"
        recwarn.clear()

        reloaded = self._roundtrip(capt, tmp_path, clean_meas, clean_sim)

        # Config-seeded: meas drives despite sim also carrying a RepCond step.
        assert reloaded.rc_source == "meas"
        # No source-change warning during load.
        assert not any("rc_source changed" in str(w.message) for w in recwarn)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `test_roundtrip_rc_source_sim_replays_sim_first` raises `ValueError` ("requires test reporting conditions") during load because the fixed meas-first order runs meas's `rep_irr` filter before sim establishes `ct.rc`; `test_roundtrip_manual_rc_restores_values` fails because manual values are never re-seeded; `test_load_is_config_seeded_and_silent` ends with `rc_source == "sim"` (last replayed) and emits a source-change warning.

- [ ] **Step 3: Rewrite the `from_mapping` replay block**

In `src/captest/captest.py`, in `from_mapping`, replace:

```python
        meas_filters = sub.get("meas_filters")
        sim_filters = sub.get("sim_filters")
        if inst._resolved_setup is not None:
            if meas_filters and inst.meas is not None:
                inst.meas.run_pipeline(meas_filters)
            if sim_filters and inst.sim is not None:
                inst.sim.run_pipeline(sim_filters)
        elif meas_filters or sim_filters:
```

with:

```python
        meas_filters = sub.get("meas_filters")
        sim_filters = sub.get("sim_filters")
        rc_values = sub.get("reporting_conditions_values")
        if inst._resolved_setup is not None:
            # Seed a manual RC before replay so self-filtering pipelines resolve
            # ref_val='rep_irr' against it; no RepCond step will set it. Computed
            # RC is (re)established during replay by the configured rc_source
            # side's RepCond step (see _on_capdata_rep_cond's _loading branch).
            if inst.rc_source == "manual" and rc_values is not None:
                inst._set_rc(pd.DataFrame([rc_values]), "manual", warn=False)
            # Replay the configured rc_source side FIRST so its RepCond populates
            # ct.rc before the other side's rep_irr filters run. _loading makes
            # the rep_cond sync config-seeded and warning-suppressed.
            pipelines = [(inst.meas, meas_filters), (inst.sim, sim_filters)]
            if inst.rc_source == "sim":
                pipelines.reverse()
            inst._loading = True
            try:
                for cd, filters in pipelines:
                    if filters and cd is not None:
                        cd.run_pipeline(filters)
            finally:
                inst._loading = False
        elif meas_filters or sim_filters:
```

(Leave the `elif meas_filters or sim_filters:` warning block and `return inst` unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for all four `TestRcOwnershipRoundTrip` tests.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: deterministic RC load - source-first replay, manual seed, _loading

from_mapping seeds a manual RC before replay, replays the configured rc_source
side first (sim->meas when rc_source='sim') so cross-side rep_irr filters
resolve mid-replay, and wraps replay in _loading so the rep_cond sync is
config-seeded and warning-suppressed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Extend the existing reconstitute test + full-suite green

**Files:**
- Modify: `tests/test_captest.py` — `TestPipelineYaml.test_file_roundtrip_with_rep_cond_step_reconstitutes_rc`.

**Interfaces:**
- Consumes: the load path from Task 2.
- Produces: nothing new — strengthens an existing test to assert the test-level `ct.rc` is reconstituted, and gates the whole stack.

- [ ] **Step 1: Strengthen the existing reconstitute test**

In `tests/test_captest.py`, in `TestPipelineYaml.test_file_roundtrip_with_rep_cond_step_reconstitutes_rc`, after the existing assertions:

```python
        assert any(type(s).__name__ == "RepCond" for s in reloaded.meas.filters)
        assert reloaded.meas.rc is not None
```

add:

```python
        # The test-level ct.rc is reconstituted from the replayed RepCond step.
        assert reloaded.rc is not None
        assert reloaded.rc_source == "meas"
```

- [ ] **Step 2: Run the full suite**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test`
Expected: all tests pass (Plans 1-5 complete).

- [ ] **Step 3: Lint + format the whole change**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_captest.py
git commit -m "test: assert ct.rc is reconstituted on RepCond-step round-trip

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Plan 5 scope = §4.7 serialization + load):**
- §4.7 computed RC not value-serialized; recomputed by replaying the source's `RepCond` → `test_computed_rc_omits_values`, `test_roundtrip_computed_meas_recomputes_ct_rc`. ✓
- §4.7 manual RC serialized as `reporting_conditions_values` (numpy → native via `to_native`) → Task 1; `test_manual_rc_serializes_native_values`, `test_roundtrip_manual_rc_restores_values`. ✓
- §4.7 point 1 warnings suppressed during load → `_loading` + `_set_rc(warn=False)` in the loading branch; `test_load_is_config_seeded_and_silent`. ✓
- §4.7 point 2 configured `rc_source` side replayed first → `pipelines.reverse()` when `rc_source == "sim"`; `test_roundtrip_rc_source_sim_replays_sim_first` (the regression guard that fails under fixed meas-first). ✓ (testing-plan items 10 sim-case + 11.)
- §4.7 point 3 config-seeded auto-update (only configured side updates `ct.rc`) → Plan 3's `_on_capdata_rep_cond` `_loading` branch, exercised here; `test_load_is_config_seeded_and_silent` (item 9). ✓
- §4.7 point 4 manual RC seeded before replay, not clobbered by any `RepCond` (no side matches `"manual"`) → seed step + Plan 3 loading guard. ✓

**Placeholder scan:** none — every step has exact code/commands. No `xfail` remains (Plan 3 removed the Plan 1 placeholder; the genuine `from_yaml` round-trips land here as live tests).

**Type consistency:** `reporting_conditions_values` is a `{str: native}` dict; load reconstructs via `pd.DataFrame([rc_values])` (one row) and `_set_rc(df, "manual", warn=False)` (Plan 1 signature). `pipelines` is a list of `(CapData, filters)` tuples; `_loading` is the bool from Plan 1; the `_on_capdata_rep_cond` loading branch (Plan 3) keys off `side == self.rc_source`. Round-trip tests use the established `clean_* = capt.<cd>.copy()` + `MagicMock(return_value=clean_*)` loader pattern from `TestPipelineYaml`.

**Whole-design check (Plans 1-5):** data model + resolution (1), manual override (2), last-writer-wins sync (3), `captest_results` (4), serialization/load (5). Spec §§4.1-4.8 and testing-plan items 1-12 are each covered by a task across the five plans.
