# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pvcaptest` is a Python package (pip name: `captest`) for photovoltaic capacity testing following the ASTM E2848 standard. The primary public API is `CapData` in `src/captest/capdata.py`.

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
- Keep comments and docstrings up-to-date when modifying code.

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
1. Load data with `load_data(...)` (measured) or `load_pvsyst(...)` (simulated)
2. Adjust `regression_cols` and optionally aggregate sensors
3. Apply filter methods on `CapData` (mutate `data_filtered`)
4. Compute reporting conditions with `rep_cond(...)`
5. Fit regression with `fit_regression(...)`
6. Compare results with module-level `captest_results(...)`

### Key Modules

**`src/captest/io.py`** — Data ingestion
- `DataLoader`: loads one or many files, handles reindexing/frequency normalization, joins into a time-indexed DataFrame
- `load_data(...)`: high-level measured-data entrypoint; builds `CapData`, groups columns, optionally appends modeled clear-sky POA/GHI when `site` metadata is provided
- `load_pvsyst(...)`: PVsyst-specific loader with date/encoding normalization and default regression-column mapping

**`src/captest/capdata.py`** — Core engine
- `CapData` holds: raw `data`, mutable working set `data_filtered`, semantic mappings (`column_groups`, `regression_cols`), filter history (`summary`, `removed`, `kept`), and regression outputs (`rc`, `regression_results`)
- Filtering methods are wrapped by `update_summary`, which records rows kept/removed and filter arguments
- Column groups are also accessible as `CapData` attributes (e.g., `cd.poa`)
- Grouped/time-period predictions via `predict_capacities(...)`, which depends on `rc`, `tolerance`, and grouped irradiance filtering
- Optional features (clear-sky, interactive plotting) are conditionally enabled if `pvlib`, `holoviews`, `panel`, `openpyxl` are installed

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
`src/captest/__init__.py` re-exports: `load_data`, `load_pvsyst`, `DataLoader`, plus submodules `capdata`, `util`, `prtest`, `columngroups`, `io`, `plotting`.

## Test Layout
- Use pytest as the testing framework.
- Mock external dependencies when needed (pytest-mock is available).
- `tests/conftest.py`: shared fixtures
- `tests/data/`: fixture datasets and column-group YAML templates
- `tests/smoke_test.py`: used in publish workflows to validate built wheel/sdist artifacts
- Test classes follow Arrange-Act-Assert pattern with pytest

## Ruff Exceptions

`E501` (line length) is ignored in `docs/conf.py`, `src/captest/io.py`, `src/captest/plotting.py`, `tests/test_io.py`, `tests/test_CapData.py`, and `tests/test_prtest.py` due to unavoidably long docstrings or test data strings.
