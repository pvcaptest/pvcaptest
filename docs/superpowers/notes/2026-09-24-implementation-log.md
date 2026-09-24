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
- Commits / roborev: `fd7ddaa` — roborev job 462: No issues found. Controller task review: approved, no findings. **Task 2 complete.**

### Task 3: `setup.py` — node models, `Side`, `RepConditions`, `TestSetup` (tier 1)

- Moved `parse_regression_formula` (and its sole `ModelDesc` import) from
  `util.py` into the new `src/captest/setup.py`, verbatim; `util.py` now does
  `from captest.setup import canonical_json, parse_regression_formula  #
  noqa: F401` so `util.parse_regression_formula` / `util.canonical_json` keep
  working. Wrote `tests/test_setup.py` (brief's Step 2 code, one deviation
  below) and confirmed RED (`ImportError: cannot import name 'setup'`) before
  writing `src/captest/setup.py` with the node models (`Group`, `Column`,
  `Calc`, the tagged `Node` union), `Side`, `RepConditions`, `TestSetup`,
  `canonical_json`, `agg_column_name`, `calc_output_name`, `walk_nodes`, and
  the tier-1 validators, verbatim from the brief's Step 4 snippet.
- Test result: RED confirmed first (`ImportError`). GREEN after implementing:
  `uv run pytest tests/test_setup.py tests/test_util.py::TestParseRegressionFormula
  -v` → 44 passed. Full suite `uv run pytest tests -q` → 1351 passed (1313 +
  38 new tests in `tests/test_setup.py`; the reexport test alongside
  `util.py`'s pre-existing 6 `TestParseRegressionFormula` tests accounts for
  the rest of the 44), 0 regressions. `just lint` / `just fmt` clean.
- Deviations:
  - **R3 (controller ruling, applied):** in
    `test_unknown_calc_names_the_path_and_suggests`, changed
    `p.endswith("reg_cols.poa")` to `"reg_cols.poa" in p`, because pydantic's
    `Discriminator`-driven `union_tag_not_found` error for an unknown `calc`
    is reported at `reg_cols.poa.calc` (the loc includes the tag), not at
    `reg_cols.poa`. Intent (the path names the node) is preserved.
  - **R4 (controller ruling, applied):** added `__test__ = False` on
    `TestSetup` — its name starts with `Test` and it is imported into
    `tests/test_setup.py`, so pytest would otherwise try to collect it as a
    test class.
  - **Ruff `UP007` (mechanical, not in the brief's snippet):** the brief's
    `Scalar = Union[None, bool, int, FiniteFloat, str]` /
    `Literal_ = Union[Scalar, list[Scalar]]` and the `Node` union's inner
    `Union[...]` fail this repo's `ruff check` (`UP007: use X | Y`). `ruff
    --fix` auto-converted the `Node` union; `Scalar`/`Literal_` needed a
    manual edit to `None | bool | int | FiniteFloat | str` /
    `Scalar | list[Scalar]`, after which the now-unused `Union` import was
    dropped. No behavior change — same types, PEP 604 syntax (Python ≥3.10 is
    the project floor). Recorded because `just lint` is a hard gate this repo
    enforces that the brief's snippet as written does not pass.
- Owner should look at: nothing.
- Commits / roborev: `32499b9` — roborev job 463: No issues found.

#### Fix round 1: an unparseable `reg_fml` escaped as a raw `patsy.PatsyError`

- Controller review (Important, F1) found that `_formula_parses` called
  `parse_regression_formula(reg_fml)` unguarded, so a malformed formula
  (`"power ~ poa +"`, `"power ~ (poa"`) raised `patsy.PatsyError` — not a
  `ValueError` — which pydantic does not catch inside a field validator, so
  it escaped `TestSetup.model_validate` as a raw `PatsyError` instead of a
  located `pydantic.ValidationError`. This broke tier 1's contract that
  every rejection is a located `ValidationError`.
- Fix: `_formula_parses` now catches `patsy.PatsyError` and re-raises
  `ValueError(f"reg_fml does not parse: {exc}") from exc`, which pydantic
  wraps into a `ValidationError` located at `reg_fml`. Confirmed
  `_formula_variables_present` and `_func_keys_are_rhs` do not re-raise a
  raw `PatsyError` in this case: both already guard on
  `info.data.get("reg_fml") is None` and return early, and pydantic omits a
  field from `info.data` once that field's own validation has failed, so
  neither validator calls `parse_regression_formula` again when `reg_fml` is
  invalid (verified: the full suite, including both of those validators'
  existing tests, stays green with no early-return path changed).
- Test added: `test_unparseable_formula_is_a_located_validation_error` in
  `tests/test_setup.py` — `TestSetup.model_validate(e2848_doc(reg_fml="power
  ~ poa +"))` raises `ValidationError` with `"reg_fml"` in an error path.
  RED confirmed first (raw `patsy.PatsyError` escaped, not caught by
  `pytest.raises(ValidationError)`); GREEN after the fix.
- Test result: `tests/test_setup.py` → 39 passed (was 38 substantive +
  reexport; now +1). Full suite `uv run pytest tests -q` → 1352 passed (1351
  + 1), 0 regressions. `just lint` / `just fmt` clean.
- Deviations: none — fix matches the finding exactly.
- Commits / roborev: `7dd0362` — roborev job 464: No issues found. Controller re-review of the fix: addressed.
  Six Minor review notes deferred to the final review (regex `$` vs fullmatch,
  unused `AGG_FUNCS`, `load` Mapping/utf-8 encoding, extra tier-1 tests,
  explicit `__hash__ = None`, spec wording for `load` of a string). **Task 3 complete.**

### Task 4: `derive` — key-level merge, `null` removal, `func` pruning
- Added `setup.derive`, `setup.merge_reg_cols`, `setup.DerivationError` to
  `src/captest/setup.py`, verbatim per the brief: `derive` copies
  `base.to_dict()`, sets `derived_from = base.name`, replaces
  `name`/`description`/`reg_fml`/`params`/`scatter_plots`/`rep_conditions`
  wholesale when given, merges `reg_cols_meas`/`reg_cols_sim` key by key via
  `merge_reg_cols` against the resolved formula's variables, then prunes
  `rep_conditions.func` entries whose key is no longer in the formula's rhs,
  and revalidates through `TestSetup.model_validate`. Added `TestDerive` (10
  tests) to `tests/test_setup.py` covering term replacement, model-node
  overrides, provenance/name defaulting, `null` removal + func pruning,
  prune-only-removes, rejecting `null` for a term the base lacks, rejecting
  an override key that isn't a formula variable, rejecting removal of a term
  the formula still uses (tier-1 `ValidationError`), wholesale replacement
  of other fields, and digest/document equivalence with a freshly-built
  document.
- TDD: RED confirmed first —
  `uv run pytest tests/test_setup.py::TestDerive -q` → 10 failed,
  `AttributeError: module 'captest.setup' has no attribute 'derive'`.
  GREEN after implementing: `uv run pytest tests/test_setup.py -q` → 49
  passed. Full suite: `uv run pytest tests -q` → 1362 passed, 0 regressions
  (outside the red window, per the plan's Task 4 note). `just lint` /
  `just fmt` clean (fmt only re-wrapped two long call sites in the new test
  class to the 88-column limit).
- Deviations: none — implementation and tests match the brief verbatim.
- Anything the owner should look at: nothing.
- Commits / roborev: `57f1026` — roborev job 465: No issues found (a verification note spotted a wrong test count in this log); `e5e803b` corrects that count — job 466: No issues found. Controller task review: approved. The spec writes `TestSetup.derive(base, ...)` but the plan built a module function `setup.derive`; a `TestSetup.derive` alias is added in Task 6 so the spec's spelling works too. **Task 4 complete.**

### Task 5: Evaluation over nodes (`util`) and `CapData` integration
- `util.transform_calc_params` / `_get_or_create_aggregation` now evaluate
  `Group`/`Column`/`Calc`/literal nodes (Calc via `CALC_REGISTRY`, output =
  registry name); deleted `update_by_path`, `_is_aggregation_tuple`,
  `_is_calculation_tuple`, `_resolve_column_group`, `encode_reg_cols`,
  `decode_reg_cols`. `CapData.custom_param(func, *, output=None, verbose=True,
  **kwargs)` injects attributes only for absent, non-None-attribute params;
  `process_regression_columns` validates through `Side` and keeps it in
  `regression_cols_preprocess`; `set_regression_cols` builds nodes;
  `agg_sensors` reads node defaults and flattens `regression_cols` afterwards.
- Tests: RED `test_util.py` 4 failed; RED `TestRegressionColumnsDocumentForm`
  8 failed. GREEN: test_util 52, test_CapData 279, test_setup 49, test_io 63
  passed. Red window (expected): test_captest 75 failed + 86 errors,
  test_plotting 8 failed, test_setup_oracles 10 failed, and 1 error in
  test_filter_classes (see below). Full run: 1183 passed, 93 failed, 87 errors.
- Deviations:
  - `set_regression_cols`: the brief's `node()` returns `Group` for any group
    id, but the brief's own test expects `Column(column="meter_power")` for
    `meter_power`, which in the `meas` fixture is both a single-column group
    and a column. Rule used: not a group -> `Column`; single-column group ->
    `Column` of its one column (the old `_resolve_column_group` semantics, so
    such a group is used as is, not aggregated); else `Group`. Docstring says so.
  - Added `util.reg_col_label(value)` (Group -> group id, Column -> column
    name, else unchanged) and used it in the readers that look
    `regression_cols` values up in `column_groups`/`data`:
    `capdata.index_capdata`, `CapData.get_reg_cols`, `CapData._get_poa_col`,
    `filters.Sensors` default thresholds. Without it the `meas` fixture (which
    calls `set_regression_cols`) broke 14 existing CapData tests.
  - `agg_sensors`: the mapping->Side normalisation runs before the
    `pre_agg_reg_trans` snapshot (not after), so `reset_agg` and the
    `Sensors` filter never see raw dict mappings. The brief's "block that
    today rewrites string values" does not exist (old `agg_sensors` never
    rewrote `regression_cols`); the flattening pass was added after the loop.
  - `get_agg_column_name` docstring kept `agg_func : str or callable` (brief
    said `str`): `agg_group`/`agg_sensors` still accept callables
    (`test_agg_map_non_str_func`), so `str` would be wrong.
  - Fixture-name caveat: `real_pwr_mtr` is not in the `meas` fixture; the
    single-column group test uses `meter_power`.
  - Test-side reads of `regression_cols["poa"]` as a group key updated to
    `.group` (`test_warn_if_filters_already_run`) / `.column`
    (`test_filter_grps`, pvsyst groups are single-column).
  - `tests/test_io.py`: 5 `load_pvsyst` assertions now expect `Column` nodes
    (load_pvsyst calls `set_regression_cols`); not in the plan's file list.
  - Docs: removed `util.update_by_path` from `docs/source/api_reference/util.rst`
    and deleted its generated stub so the docs build does not reference a
    deleted function.
  - captest.py / plotting.py untouched: captest.py already references
    `util.encode_reg_cols` / `util.decode_reg_cols` by attribute, so import
    and lint pass; those calls fail at runtime until Tasks 7-9.
- Owner should look at: `tests/test_filter_classes.py::TestReplayRollback::
  test_run_pipeline_failure_restores_captest_rc` errors in setup (the
  `ct_default` fixture builds a CapTest from a tuple-grammar preset); it is
  outside the three named red modules but cannot go green until the presets
  migrate (Task 7).
- Review 467 fix: `setup.Calc` now rejects a registered function with a
  parameter named `output` (reserved by `custom_param`'s keyword; it would
  otherwise raise `TypeError: multiple values`). Findings judged invalid:
  captest.py `to_yaml`/`from_yaml` runtime breakage (the plan's red window,
  fixed in Tasks 7-9); CHANGELOG entries (Task 10 owns CHANGELOG); coercing
  bare strings in `process_regression_columns` to `Column` (the grammar has no
  untagged top-level form; `Side` rejects literals by design, Task 3).
- Controller fix round 1: (F1) `agg_sensors` honours an explicitly given
  `Group.agg` (`"agg" in node.model_fields_set`) and falls back to the
  per-variable default (sum power, mean otherwise) only when unset; `resolve`
  leaves a node with an explicit `agg` unresolved when its group was
  aggregated with a different function. (F2) `process_regression_columns`
  on already-flattened `regression_cols` (all strings, same variables as the
  stored `regression_cols_preprocess`) re-runs from that `Side`; other string
  values raise a `ValueError` naming them. `regression_cols_preprocess` is now
  initialised to `None` in `__init__` and carried by `CapData.copy()` (the
  copy-completeness guard test required a sentinel). (F3) docstrings for both
  methods updated. 6 new tests in `TestRegressionColumnsDocumentForm`;
  test_CapData 285 passed; full-suite red set unchanged (93 failed / 87 errors,
  all in test_captest / test_plotting / test_setup_oracles plus the one
  test_filter_classes CapTest-fixture error).
- Review 469 fix (`901bfcb`): the re-run now also requires each flattened
  string to equal the column its stored node produces, so a hand-edited value
  raises instead of being silently reverted; `agg_sensors` docstring scoped.
- Controller fix round 2: `_regression_side` no longer reuses the stored
  `Side` wholesale; it replaces each string value that equals its stored
  node's output column by that node, keeps mappings/nodes as given (so a
  variable added between calls is evaluated, not dropped), and raises a
  `ValueError` naming any other string. Removed the redundant "Column nodes
  are already resolved" line from `agg_sensors`' `agg_map` doc. New test
  `test_process_regression_columns_keeps_a_variable_added_between_calls`.
- Commits / roborev: `5f13d51` (job 467: 4 findings — 1 fixed in `ebaadf4`, 3 judged
  invalid: captest.py breakage is the planned red window, CHANGELOG is Task 10, bare
  strings are not grammar), `ebaadf4` (job 468 clean); controller review fix round 1
  `b6def33` (job 469: 2 low findings fixed in `901bfcb`, job 470 clean); fix round 2
  `af08d43` (job 471 clean). Scoped re-reviews: all addressed. **Task 5 complete.**
- **For the owner:** commits `5f13d51` and `ebaadf4` carry this session's attribution
  trailers (`Claude Opus 5.5 (1M context)` / session `01VrnFv4…`), not the plan's lines.
  The implementer followed a harness reminder. They were already reviewed, so they were
  not amended; every later commit uses the plan's trailers.

### Task 6: Tier 2 — `check_project_fit`

- Added `setup.FitError` (frozen dataclass), `setup.SetupFitError`, `setup.effective_value`
  and `setup.check_project_fit(setup, side, cd)` verbatim from the brief, moving the two
  `import importlib.util` / `from dataclasses import dataclass` lines into the module's
  top import block. Amended the spec's tier-2 shadow bullet to compare a `Group`/`Calc`
  output against columns listed in `column_groups`, not all of `cd.data.columns` (a
  calculation's own output legitimately reappears in `data` on a second `setup()`).
  Also added `TestSetup.derive` as a `@staticmethod` delegating to the module-level
  `setup.derive` (same signature, NumPy docstring pointing at `setup.derive`), per the
  controller ruling that the spec spells derivation as `TestSetup.derive(base,
  **changes)` while Task 4 only built the module function; one new test
  (`TestDerive::test_staticmethod_delegates_to_module_function`) asserts the two calls
  return equal `TestSetup`s.
- Test result (initial pass, before task-review fix round below): `tests/test_setup.py`
  → 63 passed (not 64 as an earlier draft of this entry / the report said — corrected
  here). Scoped-module run (`tests/test_setup.py tests/test_util.py tests/test_CapData.py
  tests/test_calc_params.py -q`) → 501 passed. Full suite (`uv run pytest tests -q`) →
  93 failed, 1205 passed, 87 errors — identical red count to the pre-task baseline (all
  in `test_captest.py` / `test_plotting.py` / `test_setup_oracles.py` plus the one
  pre-existing `test_filter_classes` CapTest-fixture error); no new red.
- Deviations from plan/spec and why: none beyond the two amendments the brief and
  controller ruling explicitly called for (both recorded above and in the spec diff).
  Review round 1 (job 472) found the brief's verbatim `check_project_fit` only
  shadow-checked `Calc` outputs, not `Group` outputs, though the (pre-existing,
  unmodified-by-this-task) spec bullet said "Group or Calc"; fixed by adding the same
  shadow check to the `Group` branch, with a new covering test
  (`test_group_output_shadowing_a_sensor_column_is_reported`).
- **Task-review fix round 1** (coordinator-requested, after the review gate above had
  already gone clean): the coordinator's own review found that the Group/Calc shadow
  check added in round 1 above still had a real bug — `agg_group` appends every column
  it writes to `column_groups["agg"]`, and `expand_agg_map` adds `<key>_aggs` groups of
  pre-rename subgroup columns, so on a second `setup()` of the same instance a `Group`
  node's own prior output was wrongly flagged as shadowing itself (empirically: `[]`
  before `transform_calc_params`, several `FitError`s after). Fixed by excluding the
  `"agg"` key and any key ending `"_aggs"` from both the group-id-existence check and
  `sensor_columns` in `check_project_fit`. Also, in the same pass: `effective_value`'s
  step 2 now mirrors `CapData.custom_param`'s guard — a `cd` attribute is not used when
  its name is also a column-group id, since evaluation would never inject it either;
  the `requires_params` message now says "has no value" for `inspect.Parameter.empty`
  vs. "resolves to None" for a literal `None`; and
  `test_requires_param_none_by_every_route_is_reported`'s route 2 now asserts the full
  error-path list like route 1. New tests:
  `test_generated_aggregate_bookkeeping_does_not_shadow_a_group_node` (synthetic
  `column_groups["agg"]` / `"..._aggs"` bookkeeping), the real end-to-end
  `TestRegressionColumnsDocumentForm::test_check_project_fit_after_evaluation_does_not_shadow_itself`
  in `tests/test_CapData.py` (a real `CapData`, `util.transform_calc_params` run once,
  `check_project_fit` before and after both `[]`), `test_effective_value_skips_a_cd_attribute_shadowed_by_a_column_group_id`,
  and `test_requires_param_message_distinguishes_no_value_from_none`.
- Test result (after fix round 1): `tests/test_setup.py` → 66 passed. Scoped-module run
  → 505 passed. Full suite → 93 failed, 1209 passed, 87 errors — red count still
  unchanged, 4 more tests passing.
- **Task-review fix round 2**: the coordinator's own re-review caught that its fix-round-1
  instruction (`effective_value` skipping a `cd` attribute shadowed by a column-group id)
  was itself wrong: `CapData.custom_param` does not skip such an attribute and fall back
  to the default — for *any* parameter of the function (not just `requires_params`
  names) that is absent from `kwargs` and whose name is a column-group id, it **raises**
  `ValueError`. So round 1's `effective_value` change made tier 2 report `[]` in exactly
  the case where evaluation would raise. Reverted `effective_value` to the plain
  precedence (args, then a non-`None` `cd` attribute, then the function default, then
  `Parameter.empty`) with a docstring note that the collision is a separate check, not a
  value substitution. Added that check to `check_project_fit`: for every `Calc` node,
  every parameter of its registered function (excluding `data`/`verbose`) absent from
  `node.args` and present in the column-group ids is now a `FitError` at the node path,
  mirroring `custom_param`'s message and telling the author to pass the argument
  explicitly in `args` or rename the group. The initial commit checked the
  `agg`/`_aggs`-excluded `groups` dict (like the shadow checks above); roborev (job 479)
  caught that this was itself an asymmetry with `custom_param`, which raises against the
  **raw** `cd.column_groups` with no such exclusion — a parameter literally named `agg`
  or ending `_aggs` would have raised at evaluation while tier 2 reported clean. Fixed
  by checking `name in cd.column_groups` directly, with a docstring note that this one
  check is deliberately the exception to the `agg`/`_aggs` reservation. A further
  roborev pass (job 480) asked for a test that actually exercises the raw-vs-filtered
  distinction; added `test_collision_check_uses_the_raw_column_groups_including_bookkeeping`,
  confirmed discriminating against the old (filtered) logic by hand before committing.
  Replaced `test_effective_value_skips_a_cd_attribute_shadowed_by_a_column_group_id` with
  `test_calc_argument_colliding_with_a_column_group_id_is_reported` (the collision is
  reported), `test_explicit_arg_clears_a_column_group_id_collision` (passing the
  argument explicitly in `args` clears it), and the raw-vs-filtered test above.
- Test result (after fix round 2, final): `tests/test_setup.py` → 68 passed.
  Scoped-module run → 507 passed. Full suite → 93 failed, 1211 passed, 87 errors — red
  count still unchanged.
- Anything the owner should look at: the `"agg"` / `"_aggs"` column-group-id reservation
  (fix round 1, for the `Group`/`Calc` output-shadow checks only — the `Calc`-argument
  collision check deliberately does not share it) remains convention/documentation-only,
  not structurally enforced on `CapData`; flagged for a possible follow-up. Otherwise
  nothing.
- Commits / roborev: `b96f71f`..`6d972ab` (10 commits). roborev jobs 472-481; the final
  job, 481, found no issues. One finding (job 473) was judged invalid: a missing group
  produces two findings, which is by design since tier 2 collects every finding. Controller
  task review: in fix round 1 the Group-output shadow check flagged each group's own
  generated aggregate (`column_groups["agg"]`), so a second `setup()` would fail. In fix
  round 2 the controller's own instruction was wrong: tier 2 skipped the group-id collision
  that `custom_param` raises on. It now reports it. Both scoped re-reviews addressed.
  **Task 6 complete.**
- **For the owner:** tier 2 treats `column_groups` two ways by design. The output-shadow
  checks ignore the generated `agg` / `*_aggs` groups; the argument-collision check uses the
  raw groups, as `custom_param` does. The `agg`/`_aggs` reservation is by naming convention
  only, so a project with a real group named `agg` would be misjudged.

### Task 7: Presets as yaml, `TEST_SETUPS` loader, `SCATTER_REGISTRY`, `resolve_test_setup`
- Generated the ten `src/captest/setups/*.yaml` with the brief's throwaway converter (run
  before deleting the tuple dicts), plus a scratch check that each yaml matches its old
  tuple entry leaf by leaf. `TEST_SETUPS` is now loaded from those files at import;
  `SCATTER_REGISTRY`, `SETUPS_DIR`, `load_presets`, `_check_scatter_name` added;
  `resolve_test_setup` returns a `TestSetup` (named preset via `derive`, `custom` built
  from overrides); `validate_test_setup`, `_TEST_SETUP_REQUIRED_KEYS`, `_encode_override`
  and the calcparams imports deleted. Digests stored in `tests/data/setup_digests.json`.
- Test result: registry + resolve + `tests/test_presets.py` → 67 passed. Full suite →
  95 failed, 1217 passed, 87 errors: the +2 are `TestToYamlAndRoundTrip` tests in
  `test_captest.py` (red window) that index `TEST_SETUPS[...]` as a dict or reach
  `to_mapping`'s `preset.get(...)`; both are Task 8's rewrite. Other modules unchanged
  (test_plotting 8, test_setup_oracles 10, test_filter_classes 1 error). The oracle test
  still fails at `CapTest.setup()` (`'TestSetup' object is not subscriptable`), as
  expected. A scratch probe that adapts the resolved `TestSetup` back to the old dict
  shape `CapTest.setup()` reads reproduces all ten oracles at rtol 1e-9.
- Deviations: (1) the converter's dump styling was changed (flow-style leaf mappings,
  folded `description`, `params` placed before `rep_conditions`, width 70) for readable
  files; content is identical to the brief's converter. (2) `_encode_override` is
  deleted as the brief says, but its two call sites in `_build_yaml_sub_mapping` would
  then be undefined names (ruff F821), so they now `copy.deepcopy(val)` — overrides are
  plain document-form data now. Task 8 rewrites that code. (3) `SCATTER_REGISTRY`,
  `SETUPS_DIR` and `load_presets` added to `__all__`; `validate_test_setup` removed.
  (4) `_perc_wrap_to_string` stays imported in `captest.py`: `_serialize_rep_conditions`
  still uses it until Task 8 deletes that branch.
- Anything the owner should look at: no preset yaml needed hand edits. There were no
  bare-string meas leaves (every meas leaf was a `(group, agg)` tuple). Every sim bare
  string is a PVsyst column, including `scale`'s `col: PrecWat`; `factor: 100` stays an
  int literal. `docs/source/api_reference/captest.rst` still lists `validate_test_setup`;
  it is left for the docs task.
- Review 482: `resolve_test_setup("custom", ...)` read `overrides["description"]` but the
  unknown-key check rejected it (dead code). Fixed by accepting `description` for
  `custom` only, as the old code did. Invalid findings: the CHANGELOG entry and the
  `captest.rst` `validate_test_setup` line are Task 10's scope. Keeping
  `util.encode_reg_cols` in `to_yaml` is not possible: Task 5 already deleted it, and
  Task 8 rewrites that code.
- Commits / roborev: `c8cbcd2` (job 482: 1 valid finding fixed in `174b5c2`; 2 judged invalid — CHANGELOG/api docs are Task 10, `encode_reg_cols` no longer exists), `174b5c2` (job 483 clean). Controller task review: approved. The reviewer checked all 10 yaml presets node by node against the deleted tuple dicts, independently, and they match exactly. **Task 7 complete.**

### Task 8: `CapTest` integration — params, `setup()`, `rep_cond`, scatter, yaml round trip, `check_fit`
- What was done: `CapTest` gains `params`, `scatter_plots_name`, `resolved_setup`
  (a `TestSetup` param, `None` before `setup()`) and `check_fit()`; `setup()` resolves the
  setup, propagates `DOWNSTREAM_PARAMS`, runs tier 2 (`SetupFitError`) before writing any
  column, then evaluates the node trees. `rep_cond` resolves `perc_N` strings,
  `scatter_plots`/`overlay_scatters` read `SCATTER_REGISTRY`, `to_mapping` writes
  `overrides.reg_cols_*` as the diff from the preset (full sides under `custom`), plus
  `params` / `scatter_plots`. `load_config` no longer turns `perc_N` into callables;
  `from_mapping` drops `decode_reg_cols` and lifts `overrides.scatter_plots` to
  `scatter_plots_name`.
- Test result: `uv run pytest tests --ignore=tests/test_plotting.py -q` → 1391 passed.
  `tests/test_setup_oracles.py` → 11 passed (all ten presets reproduce their oracles,
  unchanged). `tests/test_plotting.py` → 8 failed (Task 9).
- Deviations: (1) `from_params` gains `verbose=True`, forwarded to the automatic
  `setup()`: the brief's tests call `from_params(..., verbose=False)`, which raised
  `TypeError` before. (2) `_collect_overrides` skips `None` and empty `reg_cols_meas` /
  `reg_cols_sim` / `rep_conditions` (controller ruling: an untouched test resolves to the
  preset itself), but keeps an empty `params`: `params` replaces wholesale, so `{}`
  means "no constraints" and is a real override (used by two
  `TestDownstreamPropagation` tests that set `rear_shade=0.12` on the `_rear_shade_sim`
  preset, now rejected by its `params: {rear_shade: 0}` without that lift). (3)
  `rep_cond` passes `func=None` when the resolved `func` is empty (a `custom` setup with
  no `rep_conditions`), which is what the pre-migration code passed (it only forwarded
  keys the dict had); the brief's snippet would pass `{}`. (4) `util._perc_wrap_to_string`
  kept (ruling R2); only the `captest.py` import is gone. (5) The downstream-attr
  propagation plus `meas.site` copy is factored into `_prepare_sides()`, shared by
  `setup()` and `check_fit()`, instead of duplicated. (6) `_warn_unserializable` drops the
  "user-mutated scatter_plots" branch (a scatter is now a name, always serializable).
- Tests changed to the spec: tuple / callable `reg_cols` round-trip tests replaced by
  document-form equivalents (nested calc override writes only that term; the `[group,
  agg]` list form and a Python callable are rejected at resolution); the mutated-scatter
  warning test deleted; `perc_N` strings stay strings through `load_config` /
  `from_yaml` / `to_yaml` and a malformed one is rejected at resolution (`perc_N` pattern)
  instead of at load; `TestResolvedSetupProperty` asserts `None` before setup;
  `TestSetupAutoWrap` uses `{"column": ...}` nodes; the path round-trip test gives `sim`
  its own fixture (tier 2 now rejects meas data on the sim side). Added: spec Testing
  item 6 second half (re-running `setup()` with a derived setup whose `e_total` args
  differ overwrites the `e_total` column), `args: {pressure: null}` reaching
  `absolute_airmass` as `None` beside a `pressure` column group (`test_CapData.py`), two
  `load_presets` error paths plus the valid case, a second `setup()` on the same
  instance, and the untouched-test digest equality.
- Anything the owner should look at: `to_yaml` / `to_mapping` now resolve the setup, so
  a `custom` test missing a side or the formula (or an invalid override) raises at export
  instead of writing a file that could not be loaded. `check_fit()` propagates the
  downstream params onto the `CapData` (as `setup()` would) — it does not write `data`.
- Fix round 1 (task review): (F1) `resolve_test_setup` returns the `TEST_SETUPS` entry
  itself when the derived setup equals the preset apart from `derived_from`, so a
  redundant override (preset `reg_fml`, an unchanged `reg_cols` term, `percent_filter: 20`)
  keeps the preset's identity and digest across `to_yaml` / `from_yaml`. (F2) the
  one-term, `custom` and registered-calculation round-trip tests now reload and compare
  the resolved setup by `==` and `content_digest()`. `to_yaml` writes `params` values
  through `to_native`. `setup()` resolves before `_maybe_wrap_sim_year_end`, so a
  resolution error leaves `sim.data` untouched; tier 2 stays after the wrap and the prep
  stage (it reads only `column_groups` and `data.columns`, which the wrap does not change,
  but it must see the propagated downstream params). Suite: 1398 passed excluding plotting;
  oracles 11 passed.
- Review 486 (on b3d8ef8), one low finding judged not to act on: `overrides.params` is
  written whenever `params` is set, even when it equals the preset's. The plan's Step 7
  code writes it that way (like `scatter_plots`), and the round trip already keeps the
  preset's identity because the resolve-time equality check treats an equal `params` as
  redundant. So no behaviour is wrong; only the file is a little more verbose.
- Commits / roborev: `c53ba6b` (job 484: 2 low; 1 fixed in `96f1d89`, 1 kept+documented+pinned), `96f1d89` (job 485 clean), controller fix round 1 `b3d8ef8` (job 486: 1 low judged invalid — `overrides.params` written whenever set is the plan's Step 7 rule and identity still holds; explanation carried in the next commit body). **Oracle equivalence test: 11 passed with the oracle files untouched — the migration preserves every preset's numbers.** Controller review: redundant overrides changed the digest across a round trip → fixed; scoped re-review addressed. **Task 8 complete; red window closed (only test_plotting red, Task 9).**
- **For the owner:** already the case before this work: when a `RepCond` step exists, `to_yaml` drops `overrides.rep_conditions`, so a reloaded test's `resolved_setup.rep_conditions` (and digest) can differ from the original. `to_yaml`/`to_mapping` now resolve the setup and raise for an incomplete/invalid config instead of writing an unloadable file.

### Task 9: `plotting.py` over nodes
- Ported `DEFAULT_TC_POWER_CALC`, `_missing_column_groups`, and `calc_tc_power_column`
  from the tuple/callable calc-params grammar to document nodes: `_missing_column_groups`
  now walks `dict` / `Group` / `Calc` (checking only `Group.group`), and
  `calc_tc_power_column` validates via `Side.model_validate({"reg_cols": dict(tc_power_calc)})`,
  requires `spec["power"]` to be a `Calc`, and drops the now-unneeded `copy.deepcopy`
  (nodes are frozen). Also fixed the two carried-over review items: `_ensure_tc_power`'s
  already-tc check and the timeseries background-curve lookup both compared
  `regression_cols` values directly to strings / `data.columns`, which silently mis-behaves
  once those values are `Group`/`Column` nodes (pre-`process_regression_columns`, e.g.
  right after `load_pvsyst`); both now go through `util.reg_col_label`, matching the
  pattern already used in `capdata.py`'s `index_capdata`.
- Tests: `TestCalcTcPowerColumn._calc_spec` and the sibling reject-spec test moved to
  document form (`match="top-level 'power' calc"`); added `test_accepts_node_models` and
  `test_default_spec_is_valid` verbatim from the brief. Added two regression tests for the
  carried review items — `test_timeseries_curve_resolves_semantic_y_via_regression_col_nodes`
  and `test_tc_power_check_resolves_unprocessed_column_node` — both confirmed to fail
  against the pre-fix code (temporarily reverted the two `reg_col_label` call sites) and
  pass with the fix.
- Test result: `tests/test_plotting.py` 37 passed (was 8 failed / 25 passed). Full suite
  `uv run pytest tests -q`: 1435 passed. Red window closed — no module is red.
- Deviations from plan/spec and why: none. The brief's code snippets were used verbatim;
  the two carried-over fixes were additive (not in the brief's snippet) and scoped exactly
  to the two call sites named in the carried review item.
- Invalid finding carried from the previous task (recorded here per instruction, not a
  finding against this task): roborev 486 (b3d8ef8) argued `overrides.params` should not
  be written when equal to the preset's. Judged invalid in Task 8's log/commit: the plan's
  Step 7 writes `params` whenever set (the same rule used for `scatter_plots`), and the
  round trip still keeps the preset's identity because `resolve_test_setup` collapses a
  redundant derivation (equal `reg_cols`/`reg_fml`/`params`/etc.) onto the preset itself —
  so the extra verbosity in the yaml does not change behavior or digest identity.
- Anything the owner should look at: nothing.
- Review 487 (on cc7305a), one low finding fixed: the timeseries background-curve lookup's
  `reg_col_label` resolution returns the bare group id for a `Group` node, which is a
  `column_groups` key, not a column of `cd.data` — so if that group's aggregated column
  already exists (e.g. a direct `agg_group` call made ahead of `agg_sensors`), the curve
  still silently dropped to `None`. Added `_resolve_reg_col_column`, which tries the
  node's `<group>_<agg>_agg` name first, and a regression test that fails against cc7305a
  and passes with the fix.
- Commits / roborev: `cc7305a` (job 487: 1 low, fixed in `ee9ef74`), `ee9ef74` (job 488 clean). Controller task review: approved, no findings. Full suite green (1436 passed). **Task 9 complete.**

### Task 10: Packaging check, changelog, docs, skill, `CLAUDE.md`
- What was done: smoke test asserts `SETUPS_DIR` / `TEST_SETUPS` ship in the wheel (passes
  against the built wheel in an isolated env; the wheel and sdist each carry the 10
  `setups/*.yaml`). CHANGELOG `[Unreleased]` gains the breaking `### Changed` entry plus the
  carried items (to_yaml/to_mapping resolve and raise, `set_regression_cols` → `Column` for a
  single-column group, `agg_sensors` honours `Group.agg`, `process_regression_columns` re-run
  rule, pydantic + pyyaml core deps, `_sim` presets refuse `rear_shade`), `### Added`
  (`captest.setup`, `TestSetup.derive` alias, `load`/`loads`, `resolved_setup`, `load_presets`,
  `SETUPS_DIR`, `util.canonical_json`, `util.reg_col_label`) and `### Removed`. User guide
  (`custom_test_setups.rst` rewritten around nodes / `@register_calc` / key-level overrides /
  `check_fit` / `derive`; `captest.rst`, `dataload.rst`, `bifacial.rst` in document form),
  API reference (new `setup.rst`; `captest.rst` drops `validate_test_setup`, adds
  `load_presets`, `SETUPS_DIR`, `SCATTER_REGISTRY`, `check_fit`, `params`,
  `scatter_plots_name`, rewrites `TEST_SETUPS` and the `rear_shade` warning; `util.rst` adds
  `reg_col_label` and a Documents section with `canonical_json`; `calcparams.rst` adds the
  registry), `add-test-setup` skill rewritten for yaml presets, `CLAUDE.md` (`AGENTS.md`)
  updated. Carried code items: `perc_wrap` docstring, `CapTest.rear_shade` param doc, the six
  presets' `reg_fml` re-emitted on one line (parsed values asserted equal; digests and oracle
  files unchanged). Spec amended: `tests/test_calc_params.py` (3 places) and `TestSetup.load`
  (path or mapping) / `TestSetup.loads` (text).
- Test result: `just lint` / `just fmt` clean; `just test` → 1436 passed; `just docs` builds.
  Clean (`-E`) docs build: 85 warnings vs 95 on a baseline worktree at `ee9ef74` (with only
  the notebook fix applied so it could finish); no new warning (diff of the sorted lists).
  Wheel smoke test: "Smoke test succeeded".
- Deviations:
  - `docs/examples/complete_capacity_test.ipynb` and `concise_capacity_test.ipynb` (not in
    the brief's file list) set `regression_cols` in the removed tuple/bare-string grammar, so
    `just docs` aborted with a `CellExecutionError` before any edit of mine. Converted those
    two cells to `{group, agg}` nodes (plus one sentence in a markdown cell); all 7 example
    notebooks now execute (nbclient). The docs-update skill says not to modify notebooks;
    overridden because the brief requires `just docs` to pass.
  - CHANGELOG: removed the two unreleased `### Fixed` entries about `to_yaml` encoding
    calculation callables as `module:qualname` and `from_yaml` turning two-element lists back
    into tuples; that code (`encode_reg_cols` / `decode_reg_cols`) was never released and is
    deleted by this plan, so the entries described behaviour that no longer exists.
  - Also added a "pyyaml is now a declared core dependency" line beside the brief's
    `pydantic` sentence, per the carried item.
  - `api_reference/captest.rst` "Module-level Functions": the new `load_presets` entry, spelled
    `captest.captest.load_presets` under `currentmodule:: captest`, raised a new
    `autosummary.import_cycle` warning (as the existing `captest.captest.*` entries already
    did). Split the table: `load_config` stays under `captest`; `test_setups`,
    `resolve_test_setup`, `load_presets`, `perc_wrap` are listed under
    `currentmodule:: captest.captest` (stub names unchanged), which also clears three
    pre-existing warnings.
  - Tracked autosummary stubs under `docs/source/api_reference/generated/` are committed as
    regenerated (new setup/registry stubs; stale `validate_test_setup` stub deleted;
    `resolved_setup` now `autoattribute`, since it is a param).
  - `CapTest.rear_shade`'s param doc still said "no preset overrides this value"; updated to
    the `params: {rear_shade: 0}` refusal (docstring only).
- Anything the owner should look at: the `*_rear_shade_sim` description wording (fixed in
  fix round 1 below). `just docs` on an existing `_build` does not
  generate autosummary stubs for a newly added page until the second run (Sphinx reads the
  pickled `found_docs`); a clean build is fine.
- Commits / roborev: (filled in by controller)

#### Fix round 1 (controller task review)
- F1: CHANGELOG `### Changed` gains two breaking entries: a bare string in
  `regression_cols` is a literal, not a column/group reference (rewrite `cd.regression_cols`
  assignments and yaml `reg_cols_*` overrides as `{column: ...}` / `{group: ..., agg: ...}`;
  `set_regression_cols` still takes names), and `TEST_SETUPS` values /
  `resolve_test_setup` return `TestSetup` objects, not dicts, while a `scatter_plots`
  override is a `SCATTER_REGISTRY` name, not a callable.
- F2 (controller ruling: digests never released): reworded the `description` of
  `bifi_e2848_etotal_rear_shade_sim`, `bifi_e2848_etotal_rear_shade_sim_spec_corrected` and
  `bifi_power_tc_etotal_rear_shade_sim` — a non-zero `rear_shade` is refused (`params:
  {rear_shade: 0}`, `SetupFitError`) rather than "still applied ... double-counting". Only
  the description changed (every other parsed field asserted equal), so only those three
  entries of `tests/data/setup_digests.json` were regenerated (the digest hashes the
  normalised document, which includes `description`). Oracle files untouched;
  `tests/test_presets.py` + `tests/test_setup_oracles.py` 32 passed.
- Minors: CHANGELOG `captest.SCATTER_REGISTRY` / `captest.validate_test_setup` →
  `captest.captest.*`; `AGENTS.md` uses `tst` for a `CapTest` instance (4 places);
  `CapData` class docstring's `regression_cols` entry describes document nodes.
- Test result: `just lint` / `just fmt` clean; `just test` 1436 passed; `just docs` exit 0,
  clean build 85 warnings, none new vs. baseline.
- Deviations: none.
- Commits / roborev: (filled in by controller)
