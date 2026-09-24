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
- Commits / roborev: `ce059d9` — roborev job 460: No issues found. Controller task review: oracles for the rear-shade `_sim`/`_meas` pairs were byte-identical (plan-mandated fixture flaw) → fix round 1.

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
- Commits / roborev: `362a4ee` — roborev job 461: No issues found. Scoped re-review: addressed. **Task 1 complete.**

### Task 2: Dependencies, spec amendment, calculation registry

- Added `pydantic>=2.5,<3` and `pyyaml>=6` to `pyproject.toml` dependencies and
  a `[tool.setuptools.package-data]` entry for `captest = ["setups/*.yaml"]`;
  `uv sync` resolved pydantic 2.13.5 / pyyaml 6.0.3. Amended the spec
  (`docs/superpowers/specs/2026-09-22-test-setup-documents-design.md`) to say
  `setups/` in place of `test_setups/` at its four occurrences and added the
  explanatory sentence under "`captest.py` — presets and `CapTest`" (confirmed
  `captest.test_setups` exists at `src/captest/captest.py:897`). Implemented
  `DOWNSTREAM_PARAMS`, `INJECTED_PARAMS`, `NULLABLE_INJECTED`, `CalcEntry`,
  `CALC_REGISTRY`, and `register_calc` in `src/captest/calcparams.py` verbatim
  from the brief, and decorated all 14 public calculations with
  `@register_calc(...)`, function bodies unchanged.
- Test result: RED confirmed first (`ImportError: cannot import name
  'CALC_REGISTRY'`) before implementing the registry. After implementing:
  `uv run pytest tests/test_calc_params.py -v` → 99 passed. Full suite
  `uv run pytest tests -q` → 1313 passed (1267 + 46 new registry tests), 0
  regressions. `just lint` / `just fmt` clean.
- Deviations:
  - `precipitable_water_gueymard` needed `requires_import=("pvlib",)` even
    though the brief's shown decorator for it was bare `@register_calc()`
    (not flagged "confirm" in the brief). Its body calls
    `pvlib.atmosphere.gueymard94_pw(...)`, which matches the mechanical rule
    ("`requires_import=("pvlib",)` exactly when the body uses `pvlib.` or
    `Location(`") and is required by
    `test_requires_import_matches_the_source`; the brief's other
    "confirm"-tagged entries (`bom_temp`, `cell_temp`, `apparent_zenith_pvsyst`)
    matched the brief's shown values exactly on inspection, no change needed
    there.
  - The brief's Step 3 snippet places `import inspect`, `import re`, and
    `from captest.calcparams import (...)` after the existing test classes
    (mid-file). `ruff` flags that as E402 (module-level import not at top of
    file) and `just lint` is required to pass before committing, so those
    imports were merged into the file's existing top-of-file import block
    instead (mechanical relocation only; no test content changed).
  - `uv.lock` is listed in the brief's `git add` for the commit, but the repo's
    `.gitignore` intentionally excludes it ("uv lock file - intentionally
    untracked so CI resolves latest deps and fails early"). Left it out of the
    commit; `uv sync` was still run per Step 1 and pydantic/pyyaml resolve
    correctly in the active environment.
- Owner should look at: nothing.
- Commits / roborev: (filled in by controller)
