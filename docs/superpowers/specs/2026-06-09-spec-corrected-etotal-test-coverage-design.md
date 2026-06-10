# Test coverage for the spectral-corrected e_total presets

**Date:** 2026-06-09
**Branch:** etotal-fixes

## Summary

Two new entries were added to `TEST_SETUPS` in `src/captest/captest.py`:

- `bifi_e2848_etotal_rear_shade_sim_spec_corrected`
- `bifi_e2848_etotal_rear_shade_meas_spec_corrected`

They combine the First Solar spectral correction (front POA) with the bifacial
total-effective-irradiance (`e_total`) regression, splitting rear-shade handling
between the modeled (`_sim`) and measured (`_meas`) sides. This spec defines the
test coverage to bring them to parity with the existing presets, plus the source
touch-ups and breakage fix the additions require.

## Review findings (pre-existing state)

- **No syntax errors or functional bugs.** Both entries import, pass
  `validate_test_setup`, resolve via `resolve_test_setup`, and run end-to-end
  through `CapTest.setup()` -> filter -> `rep_cond` -> fit. Verified with a probe
  script. The novel interaction is correct: the spectral correction is applied to
  the **front** POA *inside* the `e_total` calc-tree, rear POA left uncorrected:

  ```
  e_total = poa_spec_corrected(front) + rpoa * bifaciality * bifacial_frac * (1 - rear_shade)
  ```

  The `_sim`/`_meas` split matches the existing non-spectral pair: sim rear =
  `rpoa_pvsyst(GlobBak, BackShd)` (`_sim`) vs direct `"GlobBak"` (`_meas`);
  `rear_shade` is meas-only.

- **Style only:** the `_sim` entry (captest.py:493-592) is indented one level
  deeper than every other entry. Not a syntax error (ignored inside a dict
  literal); `ruff format` / `just fmt` normalizes it. The `_meas` entry is already
  correct.

- **Existing tests broken by the additions:** `_DEFAULT_FIXTURE_PRESETS`
  (tests/test_captest.py:32) excludes only `e2848_spec_corrected_poa` and
  `bifi_power_tc_meas_tbom`. The two new presets need humidity/pressure/site
  (meas) and `PrecWat` (sim), which the default fixtures lack, so they currently
  fail two parametrized tests (`test_setup_wires_regression_formula`,
  `test_each_preset_rep_conditions_round_trips_through_rep_cond`). They must be
  added to that exclusion set.

- **Decision (resolved):** switch `scatter_plots` from `scatter_default` to
  `scatter_etotal` on both new presets, for consistency with the non-spectral
  e_total presets. (The callables are functionally identical today.)

- **Decision (resolved):** full 4-layer coverage parity.

## Source changes (`src/captest/captest.py`)

- Switch `scatter_plots` -> `scatter_etotal` on both new presets.
- Normalize the `_sim` entry indentation via `just fmt`.

## Fixtures (`tests/conftest.py`)

No new *data* fixtures. `meas_cd_spec_corrected` and `sim_cd_spec_corrected`
already carry everything required (they extend the default fixtures, which have
`irr_rpoa` and `GlobBak`/`BackShd`). Add two `CapTest` fixtures:

- `ct_spec_corrected_etotal_sim` (test_setup `..._rear_shade_sim`)
- `ct_spec_corrected_etotal_meas` (test_setup `..._rear_shade_meas`)

Both: `meas=meas_cd_spec_corrected`, `sim=sim_cd_spec_corrected`,
`ac_nameplate=6_000_000`, `bifaciality=0.15`, `test_tolerance="- 4"`. Each
pre-assigns `sim.site` so the "Propagating meas.site" `UserWarning` does not fire
at fixture setup (that warning's behavior is already covered by the existing
`e2848_spec_corrected_poa` tests).

## Test changes (`tests/test_captest.py`)

1. **Exclusion fix** — add both new presets to the `_DEFAULT_FIXTURE_PRESETS`
   exclusion set. Turns the two currently-failing parametrized tests green.

2. **Layer 1 — structural** (`TestTestSetupsRegistry`): per preset, assert meas
   `poa` is an `e_total` tuple whose front sub-node is `poa_spec_corrected` and
   rear is `("irr_rpoa", "mean")`; sim rear is `rpoa_pvsyst` (`_sim`) vs literal
   `"GlobBak"` (`_meas`); sim front routes through `apparent_zenith_pvsyst` +
   `scale(PrecWat)`; `scatter_plots is scatter_etotal`.

3. **Layer 2 — downstream propagation** (`TestDownstreamPropagation`): the novel
   behavior — meas `e_total` equals `poa_spec_corrected + irr_rpoa * bifaciality *
   (1 - rear_shade)` (front term is the spectrally-corrected column, not raw
   `irr_poa`); `rear_shade` is meas-only.

4. **Layer 3 — column existence** (new class mirroring
   `TestCapTestSpectralCorrection`): both `poa_spec_corrected` and `e_total`
   columns exist on meas & sim for both presets.

5. **Layer 4 — integration** (`TestIntegration`): `test_end_to_end_*_sim` /
   `_meas` run the canonical sequence to `0.8 < cap_ratio < 1.2` and assert
   `regression_cols["poa"] == "e_total"`.

## TDD flow

The dict already exists; the code-under-test being driven red->green is the
scatter switch, the new fixtures, and the exclusion. For each test: write it,
watch it fail for the right reason, make the minimal source/fixture/exclusion
change, confirm green, then run the full suite + `just lint` / `just fmt`.

## Plan chunking (for writing-plans)

Per the user's preference for fine-grained, sequentially-reviewed plans:

1. Source touch-ups (`scatter_etotal`, `just fmt`) + `_DEFAULT_FIXTURE_PRESETS`
   exclusion fix.
2. Layer 1 structural tests.
3. Fixtures + Layer 2 propagation tests.
4. Layer 3 column-existence + Layer 4 integration tests.

## Skill notes — repeatable checklist for "adding a test setup"

Captured while doing this work, for the future skill. Two more presets remain to
be added after these.

1. Add the dict entry to `TEST_SETUPS`.
2. Validate it: `validate_test_setup(entry)` and `resolve_test_setup(name)`; run a
   quick end-to-end probe (`CapTest.from_params(...)`) to confirm columns
   materialize and regression cols resolve.
3. Run `just fmt` — dict-literal indentation drift is silent and easy to
   introduce.
4. Choose the scatter callable (`scatter_default` / `scatter_etotal` /
   `scatter_bifi_power_tc`) to match the regression form, consistent with sibling
   presets.
5. Map the preset's required `column_groups`, `cd.site`, and sim columns. Reuse an
   existing fixture if one already supplies them; only add a fixture when none
   does.
6. If the default fixtures don't satisfy the preset, add the preset to the
   `_DEFAULT_FIXTURE_PRESETS` exclusion set (otherwise the parametrized setup /
   rep_cond round-trip tests break).
7. Add the 4 test layers: structural calc-tree, downstream-scalar propagation,
   end-to-end column existence, end-to-end integration to a cap ratio.
8. Add a `ct_*` CapTest fixture; for spectral presets, pre-set `sim.site` to avoid
   the site-propagation `UserWarning` at fixture setup.
9. Run the full suite + `just lint` / `just fmt`.

### Known gotchas
- Dict-literal over-indentation passes Python and tests but fails formatting.
- The parametrized registry tests (`test_each_shipped_preset_validates`, etc.)
  auto-cover new presets — no work — but the `_DEFAULT_FIXTURE_PRESETS`-driven
  tests will break unless the preset is excluded.
- `scatter_etotal` and `scatter_default` are currently identical bodies; the
  choice is about intent/consistency, not runtime behavior.
- Spectral presets emit a "Propagating meas.site" `UserWarning` when `sim` has no
  `site`; pre-set `sim.site` in fixtures to keep integration tests quiet.
