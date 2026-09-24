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
- Commits / roborev: (filled in by controller)
