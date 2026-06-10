# Spectral-corrected e_total preset test coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the two new `TEST_SETUPS` presets (`bifi_e2848_etotal_rear_shade_sim_spec_corrected` / `_meas`) to full test-coverage parity with the existing presets, and fix the test breakage their addition caused.

**Architecture:** Four sequential tasks, each TDD red→green→commit. Task 1 fixes the existing parametrized-test breakage. Tasks 2–4 add the four coverage layers (structural calc-tree, downstream propagation, column existence, end-to-end integration), reusing the existing `*_spec_corrected` data fixtures.

**Tech Stack:** Python, pytest, `uv run pytest`, ruff (`just lint` / `just fmt`).

**Spec:** `docs/superpowers/specs/2026-06-09-spec-corrected-etotal-test-coverage-design.md`

---

### Task 1: Fix the `_DEFAULT_FIXTURE_PRESETS` breakage

The two new presets need humidity/pressure/site (meas) and `PrecWat` (sim), which
`meas_cd_default` / `sim_cd_default` lack. They are currently included in
`_DEFAULT_FIXTURE_PRESETS`, so two parametrized tests fail. Exclude them (like
`e2848_spec_corrected_poa`).

**Files:**
- Modify: `tests/test_captest.py:32-36`

- [ ] **Step 1: Confirm the tests are currently red**

Run: `uv run pytest "tests/test_captest.py::TestSetup::test_setup_wires_regression_formula" "tests/test_captest.py::TestPresetRepConditionsRoundTrip" -q`

(The second node id may differ; the failing test is `test_each_preset_rep_conditions_round_trips_through_rep_cond`.) Run instead:

Run: `uv run pytest tests/test_captest.py -k "wires_regression_formula or round_trips_through_rep_cond" -q`
Expected: FAIL — 2 failures for the `bifi_e2848_spec_corrected_etotal_*` params with `KeyError: "Group 'humidity' was not found..."`

- [ ] **Step 2: Add the new presets to the exclusion set**

In `tests/test_captest.py`, change:

```python
_DEFAULT_FIXTURE_PRESETS = [
    p
    for p in ct.TEST_SETUPS.keys()
    if p not in {"e2848_spec_corrected_poa", "bifi_power_tc_meas_tbom"}
]
```

to:

```python
_DEFAULT_FIXTURE_PRESETS = [
    p
    for p in ct.TEST_SETUPS.keys()
    if p
    not in {
        "e2848_spec_corrected_poa",
        "bifi_power_tc_meas_tbom",
        "bifi_e2848_etotal_rear_shade_sim_spec_corrected",
        "bifi_e2848_etotal_rear_shade_meas_spec_corrected",
    }
]
```

- [ ] **Step 3: Run to verify green**

Run: `uv run pytest tests/test_captest.py -k "wires_regression_formula or round_trips_through_rep_cond" -q`
Expected: PASS (no `bifi_e2848_spec_corrected_etotal_*` params remain; all pass)

- [ ] **Step 4: Commit**

```bash
git add tests/test_captest.py
git commit -m "test: exclude spec-corrected etotal presets from default-fixture tests"
```

---

### Task 2: Layer 1 — structural calc-tree tests + scatter_etotal switch

Add per-preset structural assertions mirroring the existing
`test_bifi_e2848_etotal_rear_shade_sim_uses_e_total` /
`test_e2848_spec_corrected_poa_*` tests. The `scatter_etotal` assertion drives the
source switch from `scatter_default`.

**Files:**
- Modify: `tests/test_captest.py` (imports near line 19; new tests in `class TestTestSetupsRegistry`, after the existing `test_e2848_spec_corrected_poa_sim_uses_apparent_zenith_pvsyst` at ~line 108)
- Modify: `src/captest/captest.py:582` and `:676` (the `"scatter_plots": scatter_default,` lines inside the two new presets)

- [ ] **Step 1: Extend the calcparams import in the test module**

In `tests/test_captest.py`, change the import block (currently lines 19–26):

```python
from captest.calcparams import (
    apparent_zenith_pvsyst,
    cell_temp,
    e_total,
    poa_spec_corrected,
    power_temp_correct,
    spectral_factor_firstsolar,
)
```

to add `rpoa_pvsyst` and `scale`:

```python
from captest.calcparams import (
    apparent_zenith_pvsyst,
    cell_temp,
    e_total,
    poa_spec_corrected,
    power_temp_correct,
    rpoa_pvsyst,
    scale,
    spectral_factor_firstsolar,
)
```

- [ ] **Step 2: Write the failing structural tests**

Add these methods to `class TestTestSetupsRegistry` (after
`test_e2848_spec_corrected_poa_sim_uses_apparent_zenith_pvsyst`):

```python
    def test_spec_corrected_etotal_sim_front_spec_corrected_rear_raw(self):
        """_sim meas poa = e_total(front=poa_spec_corrected, rear=raw irr_rpoa)."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim_spec_corrected"]
        meas_poa = entry["reg_cols_meas"]["poa"]
        assert isinstance(meas_poa, tuple)
        assert meas_poa[0] is e_total
        assert meas_poa[1]["poa"][0] is poa_spec_corrected
        assert meas_poa[1]["rpoa"] == ("irr_rpoa", "mean")

    def test_spec_corrected_etotal_sim_rear_uses_rpoa_pvsyst(self):
        """_sim sim-side rear routes through rpoa_pvsyst (shading in model)."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim_spec_corrected"]
        sim_rear = entry["reg_cols_sim"]["poa"][1]["rpoa"]
        assert isinstance(sim_rear, tuple)
        assert sim_rear[0] is rpoa_pvsyst

    def test_spec_corrected_etotal_meas_rear_maps_to_globbak(self):
        """_meas sim-side rear maps directly to GlobBak; meas side matches _sim."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_meas_spec_corrected"]
        assert entry["reg_cols_sim"]["poa"][1]["rpoa"] == "GlobBak"
        meas_poa = entry["reg_cols_meas"]["poa"]
        assert meas_poa[0] is e_total
        assert meas_poa[1]["poa"][0] is poa_spec_corrected
        assert meas_poa[1]["rpoa"] == ("irr_rpoa", "mean")

    def test_spec_corrected_etotal_sim_routes_through_pvsyst_zenith_and_scale(self):
        """_sim sim-side spectral tree uses apparent_zenith_pvsyst + scale(PrecWat)."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim_spec_corrected"]
        front = entry["reg_cols_sim"]["poa"][1]["poa"]  # poa_spec_corrected tuple
        assert front[0] is poa_spec_corrected
        spec_node = front[1]["spectral_correction"]
        assert spec_node[0] is spectral_factor_firstsolar
        zenith = spec_node[1]["absolute_airmass"][1]["apparent_zenith"]
        assert zenith[0] is apparent_zenith_pvsyst
        assert spec_node[1]["precipitable_water"][0] is scale

    def test_spec_corrected_etotal_presets_use_scatter_etotal(self):
        """Both presets use scatter_etotal, matching the other e_total presets."""
        for name in (
            "bifi_e2848_etotal_rear_shade_sim_spec_corrected",
            "bifi_e2848_etotal_rear_shade_meas_spec_corrected",
        ):
            assert ct.TEST_SETUPS[name]["scatter_plots"] is ct.scatter_etotal
```

- [ ] **Step 3: Run to verify the scatter test fails, the rest pass**

Run: `uv run pytest tests/test_captest.py::TestTestSetupsRegistry -k "spec_corrected_etotal" -q`
Expected: FAIL — only `test_spec_corrected_etotal_presets_use_scatter_etotal` fails (asserts `scatter_etotal`, source still `scatter_default`); the other four pass.

- [ ] **Step 4: Switch the source scatter callable on both new presets**

In `src/captest/captest.py`, inside `bifi_e2848_etotal_rear_shade_sim_spec_corrected` and `bifi_e2848_etotal_rear_shade_meas_spec_corrected`, change each `"scatter_plots": scatter_default,` to:

```python
            "scatter_plots": scatter_etotal,
```

(Note the indentation differs between the two entries until `just fmt` runs in Step 6 — match the surrounding lines of each entry.)

- [ ] **Step 5: Run to verify green**

Run: `uv run pytest tests/test_captest.py::TestTestSetupsRegistry -k "spec_corrected_etotal" -q`
Expected: PASS (all 5)

- [ ] **Step 6: Normalize formatting and lint**

Run: `just fmt && just lint`
Expected: the `_sim` entry indentation is normalized; no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "test: structural coverage for spec-corrected etotal presets; use scatter_etotal"
```

---

### Task 3: Fixtures + Layer 2 downstream-propagation tests

Add two `CapTest` fixtures and the propagation test that asserts the *novel*
behavior: the front term inside `e_total` is the spectrally-corrected column, and
`rear_shade` is meas-only.

**Files:**
- Modify: `tests/conftest.py` (add `import warnings` near line 1; add two fixtures after `sim_cd_spec_corrected` at ~line 372)
- Modify: `tests/test_captest.py` (new test in `class TestDownstreamPropagation`, after `test_meas_shade_setup_applies_shade_to_meas_only` at ~line 1027)

- [ ] **Step 1: Add `import warnings` to conftest**

In `tests/conftest.py`, add at the top with the other stdlib import (after line 1 `import pytest` is fine; keep stdlib first):

```python
import warnings

import pytest
import numpy as np
import pandas as pd
```

- [ ] **Step 2: Add the two CapTest fixtures**

After the `sim_cd_spec_corrected` fixture in `tests/conftest.py`, add:

```python
@pytest.fixture
def ct_spec_corrected_etotal_sim(meas_cd_spec_corrected, sim_cd_spec_corrected):
    """CapTest for the bifi_e2848_etotal_rear_shade_sim_spec_corrected preset.

    The "Propagating meas.site" UserWarning (sim has no site) is suppressed at
    setup; that auto-propagation behavior is covered separately by the
    e2848_spec_corrected_poa tests.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Propagating meas.site")
        return CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_sim_spec_corrected",
            meas=meas_cd_spec_corrected,
            sim=sim_cd_spec_corrected,
            ac_nameplate=6_000_000,
            bifaciality=0.15,
            test_tolerance="- 4",
        )


@pytest.fixture
def ct_spec_corrected_etotal_meas(meas_cd_spec_corrected, sim_cd_spec_corrected):
    """CapTest for the bifi_e2848_etotal_rear_shade_meas_spec_corrected preset.

    The "Propagating meas.site" UserWarning (sim has no site) is suppressed at
    setup; that auto-propagation behavior is covered separately by the
    e2848_spec_corrected_poa tests.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Propagating meas.site")
        return CapTest.from_params(
            test_setup="bifi_e2848_etotal_rear_shade_meas_spec_corrected",
            meas=meas_cd_spec_corrected,
            sim=sim_cd_spec_corrected,
            ac_nameplate=6_000_000,
            bifaciality=0.15,
            test_tolerance="- 4",
        )
```

- [ ] **Step 3: Write the failing propagation test**

Add to `class TestDownstreamPropagation` in `tests/test_captest.py`:

```python
    def test_spec_correction_applied_to_front_inside_e_total(
        self, ct_spec_corrected_etotal_sim
    ):
        """Meas e_total front term is poa_spec_corrected (not raw irr_poa),
        and rear is discounted by bifaciality (rear_shade=0 by default)."""
        capt = ct_spec_corrected_etotal_sim
        meas_df = capt.meas.data
        assert "poa_spec_corrected" in meas_df.columns
        mask = meas_df["irr_poa_mean_agg"] > 0
        first = meas_df.loc[mask].iloc[0]
        expected = (
            first["poa_spec_corrected"] + first["irr_rpoa_mean_agg"] * 0.15
        )
        assert first["e_total"] == pytest.approx(expected)
        # The corrected front differs from the raw front (spectral factor != 1).
        assert first["poa_spec_corrected"] != pytest.approx(first["irr_poa_mean_agg"])

    def test_rear_shade_meas_only_for_spec_corrected_etotal(
        self, meas_cd_spec_corrected, sim_cd_spec_corrected
    ):
        """rear_shade discounts the measured rear but is absent on sim."""
        with pytest.warns(UserWarning, match="Propagating meas.site"):
            capt = CapTest.from_params(
                test_setup="bifi_e2848_etotal_rear_shade_meas_spec_corrected",
                meas=meas_cd_spec_corrected,
                sim=sim_cd_spec_corrected,
                bifaciality=0.5,
                rear_shade=0.12,
            )
        assert capt.meas.rear_shade == 0.12
        assert not hasattr(capt.sim, "rear_shade")
        meas_df = capt.meas.data
        m = meas_df.loc[meas_df["irr_poa_mean_agg"] > 0].iloc[0]
        expected = m["poa_spec_corrected"] + m["irr_rpoa_mean_agg"] * 0.5 * (1 - 0.12)
        assert m["e_total"] == pytest.approx(expected)
```

- [ ] **Step 4: Run to verify green**

Run: `uv run pytest tests/test_captest.py::TestDownstreamPropagation -k "spec_correction_applied or rear_shade_meas_only_for_spec" -q`
Expected: PASS (2)

- [ ] **Step 5: Lint and commit**

```bash
just fmt && just lint
git add tests/conftest.py tests/test_captest.py
git commit -m "test: fixtures + propagation coverage for spec-corrected etotal presets"
```

---

### Task 4: Layer 3 column-existence + Layer 4 integration tests

**Files:**
- Modify: `tests/test_captest.py` (new test class after `TestCapTestSpectralCorrection` at ~line 1259; new integration tests in `class TestIntegration` after `test_end_to_end_bifi_power_tc_meas_tbom` at ~line 2043)

- [ ] **Step 1: Write the failing column-existence tests**

Add a new class after `TestCapTestSpectralCorrection`:

```python
class TestSpecCorrectedEtotalColumns:
    """Both poa_spec_corrected and e_total materialize on meas and sim."""

    def test_sim_variant_materializes_both_columns(
        self, ct_spec_corrected_etotal_sim
    ):
        capt = ct_spec_corrected_etotal_sim
        for cd in (capt.meas, capt.sim):
            assert "poa_spec_corrected" in cd.data.columns
            assert "e_total" in cd.data.columns

    def test_meas_variant_materializes_both_columns(
        self, ct_spec_corrected_etotal_meas
    ):
        capt = ct_spec_corrected_etotal_meas
        for cd in (capt.meas, capt.sim):
            assert "poa_spec_corrected" in cd.data.columns
            assert "e_total" in cd.data.columns
```

- [ ] **Step 2: Run to verify green**

Run: `uv run pytest tests/test_captest.py::TestSpecCorrectedEtotalColumns -q`
Expected: PASS (2)

- [ ] **Step 3: Write the failing integration tests**

Add to `class TestIntegration` (after `test_end_to_end_bifi_power_tc_meas_tbom`):

```python
    def test_end_to_end_spec_corrected_etotal_sim(
        self, ct_spec_corrected_etotal_sim
    ):
        """Spectral-corrected e_total (_sim) runs end-to-end to a plausible ratio."""
        capt = ct_spec_corrected_etotal_sim
        assert "e_total" in capt.meas.data.columns
        assert "poa_spec_corrected" in capt.meas.data.columns
        self._run_canonical_sequence(capt)
        cap_ratio = capt.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2
        assert capt.meas.regression_cols["poa"] == "e_total"
        assert capt.sim.regression_cols["poa"] == "e_total"

    def test_end_to_end_spec_corrected_etotal_meas(
        self, ct_spec_corrected_etotal_meas
    ):
        """Spectral-corrected e_total (_meas) runs end-to-end to a plausible ratio."""
        capt = ct_spec_corrected_etotal_meas
        assert "e_total" in capt.meas.data.columns
        assert "poa_spec_corrected" in capt.meas.data.columns
        self._run_canonical_sequence(capt)
        cap_ratio = capt.captest_results(print_res=False)
        assert 0.8 < cap_ratio < 1.2
        assert capt.meas.regression_cols["poa"] == "e_total"
        assert capt.sim.regression_cols["poa"] == "e_total"
```

- [ ] **Step 4: Run to verify green**

Run: `uv run pytest tests/test_captest.py::TestIntegration -k "spec_corrected_etotal" -q`
Expected: PASS (2)

- [ ] **Step 5: Run the full test module + lint**

Run: `uv run pytest tests/test_captest.py -q`
Expected: PASS (all)

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/test_captest.py
git commit -m "test: column-existence + integration coverage for spec-corrected etotal presets"
```

---

## Self-review notes

- **Spec coverage:** Task 1 = exclusion fix; Task 2 = Layer 1 + scatter switch + fmt; Task 3 = fixtures + Layer 2; Task 4 = Layer 3 + Layer 4. All spec sections mapped.
- **Fixture-warning approach:** the spec said "pre-assign sim.site"; this plan instead suppresses the `Propagating meas.site` warning in the fixture (simpler, and exercises the real auto-propagation path to a fixed-offset tz). Same intent — integration tests stay quiet and the warning is not re-tested here.
- **Imports:** `rpoa_pvsyst`, `scale` added to the test module; `warnings` added to conftest.
