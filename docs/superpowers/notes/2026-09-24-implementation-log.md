# Implementation log — test setup documents

Plan: `docs/superpowers/plans/2026-09-23-test-setup-documents.md`
Spec: `docs/superpowers/specs/2026-09-22-test-setup-documents-design.md`
Branch: `reg-cols-serialization`

## Before Task 1

- `roborev status`: daemon running. `uv run pytest tests -q`: 1256 passed on 5846ed3.
- Decisions made up front (record for review):
  - **Review routing.** The tiered post-commit hook sends any commit that touches
    `docs/superpowers/plans/**` or `specs/**` to the gpt-6-astra design review. Every task
    commit includes the plan (checkboxes) and this log, so each commit carries a
    `Review: code` trailer, the tier script's own escape hatch, which queues glm-5.3
    (`--reasoning medium`) as you asked.
  - **Trailers.** Otherwise exactly as the plan states (`Co-Authored-By: Claude Fable 5.1`
    plus the plan's `Claude-Session` URL), because you asked for the plan's trailers verbatim.
    Note that these name the plan-writing model and session, not this one.
  - **`util._perc_wrap_to_string`.** Task 8's file list says to delete it, but Task 7 step 7
    and Task 8 step 6 both say keep it (`filters._encode_func_value` uses it). Kept.
  - **402 / rate limit.** Handled with your override (wait 15 min, re-queue, up to 4 tries),
    not the plan's "stop and report".

## Tasks

### Task 1: Oracle capture on the current code

- Added `tests/setup_fixtures.py` (builders + `PRESET_FIXTURES` + `build_captest` +
  `snapshot`, verbatim from the brief) and pointed the `meas_cd_default` /
  `sim_cd_default` / `meas_cd_bom_temp` / `meas_cd_spec_corrected` /
  `sim_cd_spec_corrected` fixtures in `tests/conftest.py` at those builders
  (docstrings kept, bodies now one-liners). Added `tests/tools/__init__.py` +
  `tests/tools/capture_setup_oracles.py`, ran it once against the unmodified
  `src/` to write the 10 `tests/data/setup_oracles/<preset>.json` files, and added
  `tests/test_setup_oracles.py` asserting each preset reproduces its oracle.
- Test result: `tests/test_captest.py` 350 passed both before and after the
  conftest refactor (neutral, per Step 3). `tests/test_setup_oracles.py` 10
  passed. Full suite before any edit: 1256 passed (5846ed3); full suite after
  Task 1: 1266 passed (1256 + 10 new oracle tests), no regressions.
- Deviations: none — the brief's code was used verbatim; `ruff --fix` merged the
  `tests.setup_fixtures` import into the existing `from captest import ...`
  import block's neighboring group and dropped the now-unused `load_pvsyst`
  import from `conftest.py` (mechanical, no behavior change). Also added a short
  docstring to `capture_setup_oracles.main()` to satisfy the project's
  NumPy-style-docstring-on-every-public-function rule (not shown in the brief's
  snippet).
- Owner should look at: the "Label 'pressure' not found in column_groups keys,
  regression_cols keys, or columns of CapData.data" `UserWarning` emitted while
  building the 3 spec-corrected presets (`e2848_spec_corrected_poa` and its two
  bifi variants) — pre-existing behavior of unmodified `src/`, not something
  Task 1 introduced or should fix, but worth a look in a later task.
- Commits / roborev: (filled in by controller)

#### Fix round 1: rear-shade `_sim` / `_meas` oracles were byte-identical

- Controller review flagged that `bifi_e2848_etotal_rear_shade_{meas,sim}.json`,
  their `_spec_corrected` variants, and `bifi_power_tc_etotal_rear_shade_{meas,sim}.json`
  were byte-identical pairs: `build_sim_default` sets `BackShd = 0.0`, so
  `rpoa_pvsyst(GlobBak, BackShd) == GlobBak`, and `PRESET_FIXTURES` never set a
  non-zero `rear_shade`, so the measured-side rear-shade factor was a no-op
  too. A later bug that swapped the two presets' trees would have passed the
  equivalence test undetected.
- Fix: added `build_sim_rear_shade()` (and its spec-corrected wrapper
  `_sim_rear_shade_spec()`) to `tests/setup_fixtures.py`, setting
  `BackShd = GlobBak * -0.1` (PVsyst reports rear shading/IAM loss as a
  negative term, per `calcparams.rpoa_pvsyst`'s docstring) — plausible -10%
  rear-shading loss. Used it as the sim builder for all six affected presets
  (both `_sim` and `_meas` variants); left `sim_cd_default` /
  `build_sim_default` (used by `conftest.py` and by the non-rear-shade
  presets) untouched. Added `_BIFI_SHADE` / `_TC_SHADE` (`rear_shade=0.1`)
  and used them only for the three `_meas` variants' `CapTest.from_params`
  kwargs — the `_sim` variants keep `rear_shade` at its default `0`, per the
  preset docstrings' own warning against double-counting the loss (and the
  plan's note that a later task makes the `_sim` presets refuse non-zero
  `rear_shade`).
- Deviation from the plan/brief's fixture code: `PRESET_FIXTURES` in
  `task-1-brief.md` used `build_sim_default` and no `rear_shade` override for
  all ten presets; this fix diverges from that snippet for the six
  rear-shade-pair entries only, for the reason above (a controller-mandated,
  plan-consistent correctness fix, not a style choice).
- Regenerated all ten oracle files from the unmodified `src/` (only the six
  affected files' bytes changed; the other four are untouched — confirmed by
  `git diff --stat`). Added `test_oracle_files_are_pairwise_distinct` to
  `tests/test_setup_oracles.py` so a future collapse is caught automatically.
- Test result: `uv run pytest tests -q` → 1267 passed (1266 + 1 new pairwise
  test), 0 regressions. `tests/test_setup_oracles.py -v` → 11 passed. `just
  lint` / `just fmt` clean.
- Nothing else the owner needs to look at for this fix round.
- Commits / roborev: (filled in by controller)
