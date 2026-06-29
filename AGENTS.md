# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or other coding agents when working with code in this repository.

## Overview

`pvcaptest` is a Python package (pip name: `captest`) for photovoltaic capacity testing following the ASTM E2848 standard. The core public API is `CapData` in `src/captest/capdata.py` — one measured- or modeled-data dataset, its filter pipeline, and its regression. As of v0.15, `CapTest` in `src/captest/captest.py` is a higher-level orchestrator that binds a measured + modeled `CapData` pair to a named regression preset (`TEST_SETUPS`, or a `"custom"` setup) and round-trips the whole test — test-level parameters plus both filter pipelines — through a single YAML config file.

## Development Setup

Uses `uv` for dependency management and `just` (stored in `.justfile`) as a task runner.

- Install just: `uv tool install rust-just`
- Sync dependencies: `uv sync`
- Install pre-commit hooks: `pre-commit install`
- List all recipes: `just --list`

## Common Commands

```bash
# Lint and format
just lint                     # ruff check --fix on all files
just lint src/captest/io.py   # lint specific file
just fmt                      # ruff format all files

# Tests
just test                     # full suite (Python 3.12)
just test-wo-warnings         # full suite without warnings
just test-wo-warnings 3.13    # specify python version for tests
just test-cov                 # full suite with HTML coverage report
just test-module test_io.py   # single test module

# Single test with pytest node syntax
uv run pytest tests/test_CapData.py::TestUpdateSummary::test_round_kwarg_floats

# Build
just build
just ver                      # print installed version

# Docs
just docs                     # build HTML docs with sphinx-build
```

## Tools
- uv is available system wide
- ruff is available system wide
- just is available system wide
- The Github CLI is available to use. When directed use it to create and monitor PRs
  or other Github functionality as requeted..

## Code quality guidelines
### Code Style

- Line length: 88 characters (ruff default)
- Docstrings: NumPy-style for all public functions/classes/methods
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- Imports: standard library → third-party → local `captest`; no wildcard imports. Ruff
  handles import sorting; run `just lint` to auto-fix.
- Use meaningful, descriptive variable and function names.
- Avoid redundant or tautological comments; only comment where the intent is not obvious from the code.
- Never commit commented-out code or debug print statements.

### Documentation
- Include docstrings for all public functions, classes, and methods.
- Document parameters, return values, and exceptions raised.
- You MUST keep comments and docstrings up-to-date when modifying code.

### Error handling
- Never use bare `except:` clauses; catch specific exceptions.
- Never silently swallow exceptions without logging or warning.

### Security
- Never store secrets, API keys, or passwords in code; use `.env` (already in `.gitignore`).
- Never print or log sensitive information.

### Before committing
- All tests pass (`just test`).
- Linter and formatter pass (`just lint` and `just fmt`).
- No commented-out code, debug statements, or hardcoded credentials.

## Architecture

### Standard Workflow

There are two levels of API.

**CapData (single dataset):**
1. Load data with `load_data(...)` (measured) or `load_pvsyst(...)` (simulated).
2. Set `regression_cols` / `regression_formula` (or run `process_regression_columns()`); optionally aggregate sensors.
3. Apply filters. Each filter is a `param`-based step class in `filters.py` (`Irradiance`, `Time`, ...); the `CapData.filter_*(...)` methods are thin wrappers that build the step and `run()` it, appending it to the `filters` list. The wrappers always act in place (there is no `inplace` kwarg) and accept an optional `custom_name` label. `data_filtered` is a **derived read-only property** over that list (no setter; clear with `reset_filter()`).
4. Compute reporting conditions with `rep_cond(...)` — a thin wrapper that appends a zero-removal `RepCond` step.
5. Fit regression with `fit_regression(...)`.
6. Inspect via `get_summary()` / `describe_filters()`; serialize/replay the pipeline with `filters_to_config()` / `run_pipeline(config)`.

**CapTest (measured + modeled pair):**
1. Build with `CapTest.from_params(...)`, `CapTest.from_yaml(path, ...)`, or bare + `setup()`. `test_setup` selects a `TEST_SETUPS` preset, or `"custom"` (which requires `reg_cols_meas` / `reg_cols_sim` / `reg_fml` overrides).
2. `setup()` propagates config to `ct.meas` / `ct.sim` and resolves regression columns; the user then applies filters / `rep_cond` on each `CapData` directly (CapTest is a config + state container, not a runner).
3. Compare measured vs. modeled with `ct.captest_results(...)`.
4. `ct.to_yaml(path)` writes the full test config — parameters plus both `meas`/`sim` filter pipelines — to one file; `from_yaml(...)` reloads and re-applies it.

### Key Modules

**`src/captest/io.py`** — Data ingestion
- `DataLoader`: loads one or many files, handles reindexing/frequency normalization, joins into a time-indexed DataFrame
- `load_data(...)`: high-level measured-data entrypoint; builds `CapData`, groups columns, optionally appends modeled clear-sky POA/GHI when `site` metadata is provided
- `load_pvsyst(...)`: PVsyst-specific loader with date/encoding normalization and default regression-column mapping

**`src/captest/capdata.py`** — Core single-dataset engine
- `CapData` (a `param.Parameterized`) holds: raw `data`, the `filters` list (the applied filter steps), semantic mappings (`column_groups`, `regression_cols`, `regression_formula`), reporting conditions (`rc`), and regression outputs (`regression_results`).
- `data_filtered` is a **derived read-only property** — `data` restricted to the rows kept by the last filter (`filters[-1].ix_after`), or `data` when no filters; it returns a defensive copy and has no setter (clear filtering with `reset_filter()`).
- `filter_*(...)` and `rep_cond(...)` are thin wrappers that instantiate a step class from `filters.py` and `run()` it. `get_summary()`, `describe_filters()`, and the visualization methods (`scatter_filters` / `timeseries_filters` / `get_filtering_table`) derive everything from the `filters` chain (helpers `_ix_before` / `_pts_before` / `_step_labels` / `_removed_by_step`); there is no longer a `summary`/`removed`/`kept` mirror or an `update_summary` decorator.
- `filters_to_config()` / `run_pipeline(config)` serialize and replay the pipeline. Column groups are also accessible as attributes (e.g. `cd.poa`). Grouped/time-period predictions via `predict_capacities(...)` depend on `rc`, `tolerance`, and grouped irradiance filtering.
- Optional features (clear-sky, interactive plotting) are conditionally enabled if `pvlib`, `holoviews`, `panel`, `openpyxl` are installed.

**`src/captest/captest.py`** — Test orchestrator (v0.15)
- `CapTest` (a `param.Parameterized`): a config + state container binding a measured and a modeled `CapData` to a named preset, holding all test-level parameters. It is intentionally not a runner — users still call `ct.meas.filter_*(...)` / `rep_cond(...)` / `fit_regression()` themselves.
- `TEST_SETUPS`: registry of named regression presets (`e2848_default`, bifacial and spectral-corrected variants). `test_setup="custom"` requires explicit `reg_cols_meas` / `reg_cols_sim` / `reg_fml` overrides.
- Constructors: `from_params(...)` (auto-runs `setup()` when both `meas` and `sim` are supplied), `from_yaml(path, key="captest", meas_loader=, sim_loader=)`, and `from_mapping(...)`. `setup()` resolves the preset and wires regression state onto both `CapData` instances; `captest_results(...)` runs the measured-vs-modeled comparison.
- `to_yaml(path)` writes the single config file (scalar params + `meas_filters` / `sim_filters` pipelines); `_serialize_rep_conditions` / `perc_wrap` handle the `perc_N` percentile encoding (now shared via `util`).

**`src/captest/filters.py`** — Filter step classes
- `BaseSummaryStep` / `BaseFilter` base classes plus concrete steps (`Irradiance`, `Time`, `Pvsyst`, `Shade`, `Days`, `PowerFactor`, `Power`, `Missing`, `Sensors`, `Outliers`, `Clearsky`, `Custom`, `Regression`, `RepCond`). Each declares its config as `param` parameters and implements `_execute(capdata)` returning the kept index; `run()` records `ix_after` / `pts_after` and appends itself to `capdata.filters`.
- `FILTER_REGISTRY` + `step_from_config()` (de)serialize steps via each class's `to_config` / `from_config`; callable params encode through `util` helpers. Also holds the moved row-filter helpers (`filter_irr`, `sensor_filter`, `check_all_perc_diff_comb`, ...). **Imported one-way by `capdata.py`; it never imports `capdata`** — steps touch a `CapData` only through the runtime `capdata` argument.

**`src/captest/clearsky.py`** — Clear-sky modeling
- pvlib-based clear-sky GHI/POA modeling used by `io.load_data` when site metadata is supplied. Clear-sky *filtering* (`Clearsky`) is separate and uses `pvlib.clearsky.detect_clearsky` directly.

**`src/captest/calcparams.py`** — Derived measured values
- Functions to calculate derived values from measured data (e.g. back-of-module temperature from POA, wind speed, and ambient temperature via the Sandia model).

**`src/captest/columngroups.py`** — Column grouping
- `group_columns(...)`: infers semantic groups from raw column names using type/subtype/sensor keyword dictionaries
- `ColumnGroups`: dict-like container used by filtering and plotting APIs

**`src/captest/plotting.py`** — Interactive visualization
- `plot(...)`: builds Panel/HoloViews dashboard with Groups, Layout, Overlay, Scatter tabs
- `plot_defaults.json` in the working directory overrides default displayed groups
- `residual_plot(...)`: compares residual-vs-exogenous behavior between two fitted `CapData` objects

**`src/captest/prtest.py`** — Performance ratio
- `perf_ratio(...)` and `perf_ratio_temp_corr_nrel(...)` are separate from ASTM capacity-test calculation
- Results wrapped in `PrResults` with aggregate PR outputs and per-timestep data

### Public API Surface
`src/captest/__init__.py` re-exports `load_data`, `load_pvsyst`, `DataLoader` (from `io`) and `CapTest`, `TEST_SETUPS`, `load_config` (from `captest`), plus submodules `capdata`, `captest`, `io`, `columngroups`, `calcparams`, `clearsky`, `plotting`, `prtest`, `util`. (`filters.py` is internal — imported by `capdata`, not re-exported.)

## Test Layout
- Use pytest as the testing framework.
- Mock external dependencies when needed (pytest-mock is available).
- `tests/conftest.py`: shared fixtures
- `tests/data/`: fixture datasets and column-group YAML templates
- `tests/smoke_test.py`: used in publish workflows to validate built wheel/sdist artifacts
- Test classes follow Arrange-Act-Assert pattern with pytest

## Ruff Exceptions

`E501` (line length) is ignored in `docs/conf.py`, `src/captest/io.py`, `src/captest/plotting.py`, `tests/test_io.py`, `tests/test_CapData.py`, and `tests/test_prtest.py` due to unavoidably long docstrings or test data strings.
