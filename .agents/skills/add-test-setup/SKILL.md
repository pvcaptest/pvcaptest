---
name: add-test-setup
description: Use when adding a new preset/entry to the captest TEST_SETUPS registry in src/captest/captest.py — turning a brief description of a capacity-test regression (monofacial, bifacial e_total, spectral-corrected, temperature-corrected) into a complete, validated test-setup dict plus its test coverage. Triggers include "add a test setup", "new TEST_SETUPS preset", "add an e2848/bifacial/spectral preset", or a one-line description of a regression form to register.
---

# Adding a captest TEST_SETUPS preset

`TEST_SETUPS` (in `src/captest/captest.py`) maps a preset name to a dict that
fully specifies a capacity-test regression: how measured and simulated columns
map to regression variables, the regression formula, the scatter plot, and the
reporting conditions. This skill turns a **brief description** into a complete,
validated entry and a review summary, then (after approval) full test coverage.

**Do not skip the approval gates.** Two things get explicit user sign-off before
you build downstream: the **regression equation** and the **assembled dict
summary**. Build nothing past a gate until the user approves it.

## Inputs you expect

- **A brief description** of the test type (required). You will expand it.
- **A regression equation** (optional). If absent, you propose one and get
  approval before building the dict.
- Convention: `reg_cols_meas` leaves are measured **column-group** refs
  `(group, agg)`; `reg_cols_sim` leaves are PVsyst **column-name** strings.

## Workflow

```
brief description
  → 1. expand description to match existing detail level
  → 2. regression equation: use given, OR propose → APPROVAL GATE
  → 3. map each variable to reg_cols_meas / reg_cols_sim (building-block catalog)
  → 4. choose scatter callable
  → 5. rep_conditions
  → 6. assemble dict → validate + resolve + end-to-end probe → just fmt
  → 7. review summary → APPROVAL GATE
  → 8. (after approval) fixtures + exclusion + 4-layer test coverage
```

### Step 1 — Expand the description

Match the detail level of the existing `description` fields. A good description
states, in prose: the regression form (which ASTM E2848 variant), what each
correction does and **which side it lives on** (modeled vs measured), the
measured→simulated variable mapping in words, the governing equation, and a
cross-reference to any sibling variant. Read 2–3 existing entries first and mirror
their voice and length (~4–10 lines).

### Step 2 — Regression equation (APPROVAL GATE)

If the user gave an equation, use it verbatim. Otherwise propose one and **ask for
approval before continuing.** The standard four-term E2848 form is:

```
power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1
```

- `lhs` must be `power` (a project naming convention enforced by tests).
- For bifacial e_total presets, `poa` *is* the total-irradiance column — the
  formula text is unchanged; the meaning of `poa` changes via the calc-tree.
- `power ~ poa + rpoa` is the two-term bifacial temp-corrected form.

**Formula syntax (Patsy).** Regression formulas use the Patsy formula
mini-language (the same one statsmodels consumes via `smf.ols`):
https://patsy.readthedocs.io/en/latest/formulas.html. The string must be a valid
Patsy expression — e.g. `I(...)` wraps arithmetic so `*` means multiply rather
than Patsy's interaction operator, and `- 1` drops the intercept. A
user-supplied formula must parse as valid Patsy.

**Terms must match the reg_cols keys.** Every variable (term) in the formula —
lhs and rhs — must match a **top-level key** of both `reg_cols_meas` and
`reg_cols_sim`. `CapData.process_regression_columns` builds, from each reg_cols
dict, a mapping from regression term (key) → the column name that term resolves
to in `CapData.data`; a term with no matching key has no column to fit against.
`validate_test_setup` enforces this: the formula's lhs+rhs must be a subset of
both reg_cols dicts' keys.

### Step 3 — Map variables to columns (building-block catalog)

Each `reg_cols_*` value is **either** a direct ref **or** a *calc-tuple*
`(func, {arg: <nested spec>})`, where each nested spec is itself a direct ref or
another calc-tuple. Direct refs: meas `("group", "agg")`; sim `"ColName"`.

Sim leaves are exact PVsyst output-variable names (e.g. `GlobInc`, `GlobBak`,
`E_Grid`, `TArray`, `PrecWat`). To confirm a variable exists or find its correct
abbreviation, check the PVsyst docs — meteo & irradiance variables:
https://www.pvsyst.com/help/project-design/results/simulation-variables-meteo-and-irradiations.html
and grid-system variables:
https://www.pvsyst.com/help/project-design/results/simulation-variables-grid-system.html.
A variable absent from PVsyst output cannot be a `reg_cols_sim` leaf (the
sim-side `validate_test_setup` / `process_regression_columns` checks will fail).

Calc functions live in `captest.calcparams`. Catalog (arg → wire as nested spec):

| Function | Computes | Wire as |
|---|---|---|
| `e_total` | total effective irr = front + rear·bifaciality·… | `(e_total, {"poa": <front>, "rpoa": <rear>})` |
| `rpoa_pvsyst` | modeled rear with shading baked in | `(rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"})` |
| `poa_spec_corrected` | spectrally corrected front POA | `(poa_spec_corrected, {"poa": <poa>, "spectral_correction": <spec>})` |
| `spectral_factor_firstsolar` | First Solar spectral factor | `(spectral_factor_firstsolar, {"precipitable_water": <pw>, "absolute_airmass": <am>})` |
| `precipitable_water_gueymard` | pw from temp + RH (meas) | `(precipitable_water_gueymard, {"temp_amb": ("temp_amb","mean"), "rel_humidity": ("humidity","mean")})` |
| `scale` | scale a column (sim pw: PrecWat·100) | `(scale, {"col": "PrecWat", "factor": 100})` |
| `absolute_airmass` | airmass from zenith (+pressure on meas) | `(absolute_airmass, {"apparent_zenith": <z>, "pressure": ("pressure","mean")})` |
| `apparent_zenith` / `apparent_zenith_pvsyst` | solar zenith (meas / PVsyst ½-hr shift) | `(apparent_zenith, {})` / `(apparent_zenith_pvsyst, {})` |
| `power_temp_correct` | temperature-corrected power | `(power_temp_correct, {"power": <p>, "cell_temp": <ct>})` |
| `cell_temp` | Sandia cell temp from POA + BOM | `(cell_temp, {"poa": ("irr_poa","mean"), "bom": <bom>})` |
| `bom_temp` | modeled BOM temp | `(bom_temp, {"poa": (...), "temp_amb": (...), "wind_speed": (...)})` |

Scalars like `bifaciality`, `bifacial_frac`, `rear_shade` are **not** wired in the
dict — they flow in from `CapTest` attributes. `rear_shade` is **meas-only**:
the `_sim` variant bakes shading into the modeled rear via `rpoa_pvsyst`; the
`_meas` variant maps sim rear directly to `"GlobBak"` and applies `rear_shade` on
the measured side. Name variants `..._sim` / `..._meas` accordingly.

Sim spectral note: the PVsyst side uses `apparent_zenith_pvsyst` and **omits
`pressure`** (pvlib sea-level default); meas uses `apparent_zenith` + measured
`pressure`.

### Step 4 — Scatter callable

| Regression form | `scatter_plots` |
|---|---|
| POA-based / generic | `scatter_default` |
| e_total-based (incl. spectral e_total) | `scatter_etotal` |
| `power ~ poa + rpoa` (two-panel) | `scatter_bifi_power_tc` |

### Step 5 — rep_conditions

Default shape (override only with reason):

```python
"rep_conditions": {
    "irr_bal": False,
    "percent_filter": 20,
    "func": {"poa": perc_wrap(60), "t_amb": "mean", "w_vel": "mean"},
},
```

**`func` keys must be exactly the rhs variables of `reg_fml`** — no more, no less.
`validate_test_setup` raises if `func` has a non-rhs key. Drop a variable from the
formula → drop it from `func`.

### Step 6 — Assemble, validate, probe, format

Add the entry to the dict, then verify before showing the user:

```python
import warnings, captest.captest as ct
from captest import CapTest
name = "your_new_preset"
ct.validate_test_setup(ct.TEST_SETUPS[name])   # raises on bad shape
ct.resolve_test_setup(name)                      # raises on bad formula/cols
# end-to-end probe with fixtures that satisfy the preset's required inputs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    capt = CapTest.from_params(test_setup=name, meas=meas_cd, sim=sim_cd,
                               ac_nameplate=6_000_000, bifaciality=0.15)
```

Then **always run `just fmt`** — nested dict literals over-indent silently; this
passes Python and tests but fails formatting. Run `just lint` too.

### Step 7 — Review summary (APPROVAL GATE)

Present a scannable summary and wait for approval before touching tests:

```
Preset: <name>
Description: <2-line gist of the expanded description>
Regression: <reg_fml>
Variable → meas → sim
  power : (real_pwr_mtr, sum)        → E_Grid
  poa   : e_total(spec-corrected …)  → e_total(…, rpoa_pvsyst)
  t_amb : (temp_amb, mean)           → T_Amb
  w_vel : (wind_speed, mean)         → WindVel
Scatter: <scatter_callable>
Rep conditions: percent_filter=20, func keys = {<rhs vars>}
Validates: yes | Resolves: yes | End-to-end probe: cap ratio <x>
Required inputs (for fixtures): <column groups / site / sim cols>
```

### Step 8 — Test coverage (after the dict is approved)

Use the **unit-tests** skill for conventions. (Optional: delegate this whole step
to a subagent — see *Optional: using subagents* below.) Mirror the existing
presets' four layers in `tests/test_captest.py` (see the spec-corrected etotal
presets as the worked example):

1. **Structural** (`TestTestSetupsRegistry`): assert the calc-tree shape — the
   identity of each calc function in the meas/sim trees and the `scatter_plots`
   callable.
2. **Downstream propagation** (`TestDownstreamPropagation`): assert the *novel*
   numeric behavior of this preset (e.g. the corrected/total column equals the
   expected combination of its inputs and the `CapTest` scalars).
3. **Column existence** (a class like `TestCapTestSpectralCorrection`): the
   calculated columns materialize on both meas and sim.
4. **Integration** (`TestIntegration`): run the canonical filter→rep_cond→fit
   sequence to `0.8 < cap_ratio < 1.2` and assert `regression_cols["poa"]`.

Fixtures (`tests/conftest.py`): reuse before adding. The existing fixtures and
what they supply:

| Fixture | Supplies |
|---|---|
| `meas_cd_default` / `sim_cd_default` | power, POA, amb, wind; `irr_rpoa`; sim `GlobBak`/`BackShd` |
| `meas_cd_spec_corrected` / `sim_cd_spec_corrected` | adds `humidity`, `pressure`, `cd.site` (meas); `PrecWat` (sim) |
| `meas_cd_bom_temp` | adds `temp_bom` group |

If the default fixtures don't satisfy the preset's required inputs, **add the
preset to the `_DEFAULT_FIXTURE_PRESETS` exclusion set** in `tests/test_captest.py`
(otherwise the parametrized setup / rep_cond tests run it against the default
fixtures and fail). Add a `ct_*` fixture; for spectral presets, suppress the
`Propagating meas.site` `UserWarning` in the fixture.

Run `just test-module test_captest.py`, then the full suite, then `just lint` /
`just fmt`. Commit.

## Optional: using subagents

Default to doing the work in the main conversation — the dict is small, iterated
with the user, and gated by the two user approvals (a subagent cannot run those
gates, and an LLM "reviewing" another LLM's dict is a weaker check than the
objective `validate`/`resolve`/probe steps). **Keep dict generation in the main
agent.** Reach for subagents only where they earn their keep:

- **Delegate Step 8 test coverage.** Once the dict is approved, the four test
  layers + fixtures are a bulky, well-specified, independent chunk. Dispatch a
  subagent to implement them (it can use the **unit-tests** skill); the main agent
  reviews the diff and runs the suite + `just lint`. Worth it for complex presets;
  skip it for a trivial monofacial one.
- **Semantic verification of a complex dict.** `validate_test_setup` catches
  shape errors but not a valid-but-wrong wiring — e.g. rear mapped to the wrong
  column, or `apparent_zenith` used where `apparent_zenith_pvsyst` belongs on the
  sim side. For deeply nested presets, dispatch one subagent prompted to *refute*
  the mapping against the expanded description and a sibling preset. Optional;
  overkill for shallow trees.
- **Batch-adding presets.** Adding several at once → one subagent per preset (each
  is independent), each running this skill end-to-end.

## Gotchas

| Gotcha | Reality |
|---|---|
| Dict over-indentation | Passes Python & tests, fails `just fmt`. Always run `just fmt`. |
| `func` has a non-rhs key | `validate_test_setup` raises. `func` keys = rhs vars exactly. |
| Spectral factor NaN at sunrise/sunset | Select test rows on `poa_spec_corrected.notna() & > 0`, not `irr_poa > 0`. |
| `Propagating meas.site` warning | Fires when sim has no `site`; suppress in spectral `ct_*` fixtures. |
| New preset breaks parametrized tests | If it needs special fixtures, exclude it from `_DEFAULT_FIXTURE_PRESETS`. |
| Wiring scalars into the dict | `bifaciality`/`bifacial_frac`/`rear_shade` come from `CapTest`, not the dict. |

## Red flags — you skipped a gate

- You wrote the dict before the user approved the regression equation.
- You started writing tests before the user approved the dict summary.
- You reported "done" without running `validate_test_setup`, `resolve_test_setup`,
  an end-to-end probe, and `just fmt` / `just lint`.

All of these mean: stop, back up to the gate you skipped.
