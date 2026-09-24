---
name: add-test-setup
description: Use when adding a new preset to the captest TEST_SETUPS registry — turning a brief description of a capacity-test regression (monofacial, bifacial e_total, spectral-corrected, temperature-corrected) into a complete, validated yaml preset document in `src/captest/setups/` plus its digest, fixture, oracle and docs entry. Triggers include "add a test setup", "new TEST_SETUPS preset", "add an e2848/bifacial/spectral preset", or a one-line description of a regression form to register.
---

# Adding a captest TEST_SETUPS preset

A preset is a **yaml document** in `src/captest/setups/<name>.yaml`. At import,
`captest.captest.load_presets()` reads every file in that directory into
`TEST_SETUPS: dict[str, captest.setup.TestSetup]`, validating each one (tier 1),
so a broken preset fails `import captest`. The document says how measured and
simulated columns map to regression variables, the regression formula, the
scatter plot, the reporting conditions and any test-level parameters the setup
requires. It is pure data: every calculation is referenced by its name in
`captest.calcparams.CALC_REGISTRY`, never as a Python object.

This skill turns a **brief description** into a complete, validated preset and a
review summary, then (after approval) the fixture lines and docs entry.

**Do not skip the approval gates.** Two things get explicit user sign-off before
you build downstream: the **regression equation** and the **review summary** of
the assembled document. Build nothing past a gate until the user approves it.

## Inputs you expect

- **A brief description** of the test type (required). You will expand it.
- **A regression equation** (optional). If absent, you propose one and get
  approval before writing the document.
- Convention: `meas` leaves are **column-group** nodes `{group: <id>, agg: <fn>}`;
  `sim` leaves are PVsyst **column** nodes `{column: <PVsyst name>}`.

## Workflow

```
brief description
  → 1. expand description to match existing detail level
  → 2. regression equation: use given, OR propose → APPROVAL GATE
  → 3. write src/captest/setups/<name>.yaml in the node grammar (copy nearest preset)
  → 4. params (sim side carries rear shading and meas side calls e_total)
  → 5. prove it loads
  → 6. digest line, PRESET_FIXTURES entry, oracle capture
  → 7. review summary → APPROVAL GATE
  → 8. docs entry in docs/source/api_reference/captest.rst
```

### Step 1 — Expand the description

Match the detail level of the existing `description` fields. A good description
states, in prose: the regression form (which ASTM E2848 variant), what each
correction does and **which side it lives on** (modeled vs measured), the
measured→simulated variable mapping in words, the governing equation, and a
cross-reference to any sibling variant. Read 2–3 existing files in
`src/captest/setups/` first and mirror their voice and length (~4–10 lines). Write
it as a folded block scalar (`description: >-`).

### Step 2 — Regression equation (APPROVAL GATE)

If the user gave an equation, use it verbatim. Otherwise propose one and **ask for
approval before continuing.** The standard four-term E2848 form is:

```
power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1
```

- `lhs` must be `power` (a project naming convention enforced by tests).
- For bifacial e_total presets, `poa` *is* the total-irradiance column — the
  formula text is unchanged; the meaning of `poa` changes via the `calc` node.
- `power ~ poa + rpoa` is the two-term bifacial temp-corrected form.

**Formula syntax (Patsy).** Regression formulas use the Patsy formula
mini-language (the same one statsmodels consumes via `smf.ols`):
https://patsy.readthedocs.io/en/latest/formulas.html. `I(...)` wraps arithmetic so
`*` means multiply rather than Patsy's interaction operator, and `- 1` drops the
intercept. A formula that does not parse is a tier-1 error at `reg_fml`.

**Terms must match the reg_cols keys.** Every variable in the formula — lhs and
rhs — must be a top-level key of both `meas.reg_cols` and `sim.reg_cols`
(tier 1 rejects a missing one). Keep `reg_fml` on one line in the file.

### Step 3 — Write the document (node grammar)

Copy the nearest existing preset to `src/captest/setups/<name>.yaml` and edit it.
The file name must equal `name:`. Top-level fields: `name`, `description`,
`reg_fml`, `meas`, `sim`, optional `params`, `rep_conditions`, `scatter_plots`
(`derived_from` is optional provenance only).

Each `reg_cols` value is a **node**, a mapping with exactly one of these keys:

| Node | Meaning |
|---|---|
| `{group: irr_poa, agg: mean}` | aggregate a column group (`agg`: mean/sum/median/min/max; default mean); writes `irr_poa_mean_agg` |
| `{column: GlobInc}` | one column of `data` by name |
| `{calc: e_total, args: {...}}` | a registered calculation; `args` are **keyword-only**, each value a node or a literal; writes a column named `e_total` |

Literals (`"cdte"`, `100`, `true`, `null`, `[1, 2]`) are allowed only as `args`
values. There is **no shorthand**: a bare string is always a literal, so a PVsyst
column is `{column: E_Grid}`, never `E_Grid`. A mapping without one of the three
keys is an error, and so is a top-level `reg_cols` value that is a literal.

Sim leaves are exact PVsyst output-variable names (e.g. `GlobInc`, `GlobBak`,
`E_Grid`, `TArray`, `PrecWat`). To confirm a variable exists, check the PVsyst
docs — meteo & irradiance variables:
https://www.pvsyst.com/help/project-design/results/simulation-variables-meteo-and-irradiations.html
and grid-system variables:
https://www.pvsyst.com/help/project-design/results/simulation-variables-grid-system.html.

Calculation catalog (`captest.calcparams`, registered under these names; list
them with `sorted(CALC_REGISTRY)`):

| `calc` | Computes | `args` |
|---|---|---|
| `e_total` | total effective irr = front + rear·bifaciality·… | `{poa: <front>, rpoa: <rear>}` |
| `rpoa_pvsyst` | modeled rear with shading baked in | `{globbak: {column: GlobBak}, backshd: {column: BackShd}}` |
| `poa_spec_corrected` | spectrally corrected front POA | `{poa: <poa>, spectral_correction: <spec>}` |
| `spectral_factor_firstsolar` | First Solar spectral factor | `{precipitable_water: <pw>, absolute_airmass: <am>}` |
| `precipitable_water_gueymard` | pw from temp + RH (meas) | `{temp_amb: {group: temp_amb, agg: mean}, rel_humidity: {group: humidity, agg: mean}}` |
| `scale` | scale a column (sim pw: PrecWat·100) | `{col: {column: PrecWat}, factor: 100}` |
| `absolute_airmass` | airmass from zenith (+pressure on meas) | `{apparent_zenith: <z>, pressure: {group: pressure, agg: mean}}` |
| `apparent_zenith` / `apparent_zenith_pvsyst` | solar zenith (meas / PVsyst ½-hr shift) | `{}` |
| `power_temp_correct` | temperature-corrected power | `{power: <p>, cell_temp: <ct>}` |
| `cell_temp` | Sandia cell temp from POA + BOM | `{poa: {group: irr_poa, agg: mean}, bom: <bom>}` |
| `bom_temp` | modeled BOM temp | `{poa: ..., temp_amb: ..., wind_speed: ...}` |

A calculation the catalog lacks must first be added to `calcparams.py` with
`@register_calc(requires_params=..., requires_import=...)` (the
registry-declaration tests in `tests/test_calc_params.py` check the declaration
against the source). A document never imports code.

Scalars like `bifaciality`, `bifacial_frac`, `rear_shade`, `power_temp_coeff` are
**not** written in `args` — `CapData.custom_param` injects them from the `CapData`
attributes `CapTest.setup()` propagates. Give one in `args` only to override the
test-wide value for that one calculation. `rear_shade` is **meas-only**: the `_sim`
variant bakes shading into the modeled rear via `rpoa_pvsyst`; the `_meas` variant
maps the sim rear to `{column: GlobBak}` and applies `rear_shade` on the measured
side. Name variants `..._sim` / `..._meas` accordingly.

Sim spectral note: the PVsyst side uses `apparent_zenith_pvsyst` and **omits
`pressure`** (pvlib sea-level default); meas uses `apparent_zenith` + measured
`pressure`.

One column per producer: two nodes on one side may write the same column only if
they are identical, so do not use the same `calc` twice on a side with different
`args`.

`scatter_plots` is a name in `captest.SCATTER_REGISTRY`:

| Regression form | `scatter_plots` |
|---|---|
| POA-based / generic | `default` |
| e_total-based (incl. spectral e_total) | `etotal` |
| `power ~ poa + rpoa` (two-panel) | `bifi_power_tc` |

`rep_conditions` default shape (override only with reason):

```yaml
rep_conditions:
  irr_bal: false
  percent_filter: 20
  func: {poa: perc_60, t_amb: mean, w_vel: mean}
```

`func` values are the strings `mean`, `median` or `perc_N` (never a `perc_wrap`
callable), and **`func` keys must be rhs variables of `reg_fml`** — tier 1 rejects
any other key. Drop a variable from the formula → drop it from `func`.

### Step 4 — `params`

`params` maps a `CapTest` downstream parameter to the value the setup
**requires**; tier 2 (`CapTest.setup()` / `check_fit()`) raises
`SetupFitError` when the value a calculation would receive differs. When the sim
side carries rear shading (`rpoa_pvsyst`) and the meas side calls `e_total`, add

```yaml
params: {rear_shade: 0}
```

so a non-zero measured `rear_shade` is refused instead of double-counting the
loss. Omit `params` otherwise.

### Step 5 — Prove it loads

```bash
uv run python -c "from captest.captest import TEST_SETUPS; print(TEST_SETUPS['<name>'])"
```

A tier-1 error names the document path (e.g. `meas.reg_cols.poa.args.rpoa`).
Then run an end-to-end probe with fixtures that satisfy the preset's inputs
(`tests/setup_fixtures.py` builders), e.g.
`CapTest.from_params(test_setup=name, meas=..., sim=..., ac_nameplate=6_000_000,
bifaciality=0.15)`, and check `tst.check_fit()` is empty.

### Step 6 — Digest, fixture, oracle

The parametrised preset tests pick the new file up by directory listing and fail
until these exist:

1. **Digest** — add `"<name>": "<digest>"` to `tests/data/setup_digests.json`
   (keys sorted), where the digest is
   `uv run python -c "from captest.captest import TEST_SETUPS; print(TEST_SETUPS['<name>'].content_digest())"`.
2. **Fixture** — add a `PRESET_FIXTURES` entry in `tests/setup_fixtures.py`:
   `(meas builder, sim builder, CapTest.from_params kwargs)`. Reuse the existing
   builders (`build_meas_default`, `_meas_bom`, `_meas_spec`, `build_sim_default`,
   `build_sim_rear_shade`, `_sim_spec`, `_sim_rear_shade_spec`) and kwargs sets
   (`_BASE`, `_BIFI`, `_BIFI_SHADE`, `_TC`, `_TC_SHADE`). A `_meas` rear-shade
   variant should use a non-zero `rear_shade`, and its sim builder a non-zero
   `BackShd`, so the oracle is not identical to its `_sim` sibling.
3. **Oracle** — capture **only the new preset**; never regenerate an existing
   oracle file:

   ```bash
   uv run python -c "
   import json; from pathlib import Path
   from tests.setup_fixtures import build_captest, snapshot
   name = '<name>'
   Path(f'tests/data/setup_oracles/{name}.json').write_text(
       json.dumps(snapshot(build_captest(name)), indent=2, sort_keys=True) + '\n')"
   ```

If the default `meas_cd_default` / `sim_cd_default` fixtures do not satisfy the
preset's inputs, add it to the exclusion set in `_DEFAULT_FIXTURE_PRESETS` in
`tests/test_captest.py`. Add targeted tests for any *novel* behaviour (e.g. a new
calculated column equals the expected combination of its inputs) in the existing
classes (`TestDownstreamPropagation`, `TestCapTestSpectralCorrection`,
`TestIntegration`); use the **unit-tests** skill for conventions.

Run `uv run pytest tests/test_presets.py tests/test_setup_oracles.py
tests/test_captest.py -q`, then the full suite, then `just lint` / `just fmt`.

### Step 7 — Review summary (APPROVAL GATE)

Present a scannable summary and wait for approval before the docs entry and
commit:

```
Preset: <name>            File: src/captest/setups/<name>.yaml
Description: <2-line gist of the expanded description>
Regression: <reg_fml>
Variable → meas → sim
  power : {group: real_pwr_mtr, agg: sum}  → {column: E_Grid}
  poa   : e_total(spec-corrected …)        → e_total(…, rpoa_pvsyst)
  t_amb : {group: temp_amb, agg: mean}     → {column: T_Amb}
  w_vel : {group: wind_speed, agg: mean}   → {column: WindVel}
Scatter: <name>   Params: <params or none>
Rep conditions: percent_filter=20, func = {<rhs vars>}
Loads: yes | check_fit: [] | Oracle captured | Digest: <first 12 chars>
Required inputs (for fixtures): <column groups / site / sim cols>
```

### Step 8 — Docs entry

Add the preset to "Predefined Test Setups" in
`docs/source/api_reference/captest.rst`, mirroring its `description`, and add a
CHANGELOG `### Added` line. Run `just docs`. Commit.

## Gotchas

| Gotcha | Reality |
|---|---|
| Bare string as a sim leaf | A literal, not a column; tier 1 rejects it at top level. Write `{column: ...}`. |
| `func` has a non-rhs key | Tier 1 rejects it. `func` keys ⊆ rhs vars. |
| `perc_wrap(60)` in the file | Not data. Write `perc_60`. |
| Changing a shipped preset's content | Changes its digest and may change its oracle; both are deliberate-change-only. Layout-only edits (line folding) do not change the digest. |
| Spectral factor NaN at sunrise/sunset | Select test rows on `poa_spec_corrected.notna() & > 0`, not `irr_poa > 0`. |
| `Propagating meas.site` warning | Fires when sim has no `site`; suppress in spectral `ct_*` fixtures. |
| Wiring scalars into `args` | `bifaciality`/`bifacial_frac`/`rear_shade` come from `CapTest`; an `args` entry overrides the test-wide value. |

## Red flags — you skipped a gate

- You wrote the document before the user approved the regression equation.
- You wrote the docs entry or committed before the user approved the summary.
- You reported "done" without the load check, the digest / fixture / oracle
  lines, the preset tests, and `just fmt` / `just lint`.
- You regenerated an existing oracle file.

All of these mean: stop, back up to the gate you skipped.
