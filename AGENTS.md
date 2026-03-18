# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Code quality guidelines

### Code style
- Follow PEP 8 conventions: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants.
- Line length limit is 88 characters (ruff default, configured in `pyproject.toml`).
- Use meaningful, descriptive variable and function names.
- Avoid redundant or tautological comments; only comment where the intent is not obvious from the code.
- Never commit commented-out code or debug print statements.

### Documentation
- Include docstrings for all public functions, classes, and methods.
- Document parameters, return values, and exceptions raised.
- Keep comments and docstrings up-to-date when modifying code.
- Follow the existing NumPy-style docstring convention used throughout `src/captest/`.

### Error handling
- Never use bare `except:` clauses; catch specific exceptions.
- Never silently swallow exceptions without logging or warning.
- Use context managers (`with` statements) for resource cleanup.

### Imports
- Never use wildcard imports (`from module import *`).
- Organize imports: standard library, third-party, then local (`captest`) imports.
- Ruff handles import sorting; run `just lint` to auto-fix.

### Testing
- Use pytest as the testing framework.
- Follow the Arrange-Act-Assert pattern.
- Mock external dependencies when needed (pytest-mock is available).

### Security
- Never store secrets, API keys, or passwords in code; use `.env` (already in `.gitignore`).
- Never print or log sensitive information.

### Before committing
- All tests pass (`just test`).
- Linter and formatter pass (`just lint` and `just fmt`).
- No commented-out code, debug statements, or hardcoded credentials.

## Repository overview
- Python package name is `captest`; source code is under `src/captest`.
- The primary API surface is `CapData` in `src/captest/capdata.py`.
- Loader helpers are re-exported from `src/captest/__init__.py` (`load_data`, `load_pvsyst`, `DataLoader`).
- `src/captest/captest.py` currently contains only a module docstring (no implemented runtime code).

## Common commands
Run commands from the repository root.

### Environment and task runner
- Sync dependencies: `uv sync`
- List available task recipes: `just --list`

### Lint and format (use just recipes)
- Lint and auto-fix: `just lint`
- Lint a specific file/path: `just lint src/captest/io.py`
- Format: `just fmt`
- Format a specific file/path: `just fmt src/captest/io.py`

### Tests (use just recipes)
- Full suite (default Python 3.12): `just test`
- Full suite without warnings: `just test-wo-warnings`
- Coverage report (HTML): `just test-cov`
- Run one test module file: `just test-module test_io.py`
- Run one test class/case with pytest node syntax: `uv run pytest tests/test_CapData.py::TestUpdateSummary::test_round_kwarg_floats`

### Build and package checks (use just recipes)
- Build artifacts: `just build`
- Verify install from built wheel in a fresh venv: `just test-install`
- Print installed package version in project env: `just ver`

### Documentation
- Build docs: `just docs`

## Architecture map

### Ingestion and preprocessing (`src/captest/io.py`)
- `DataLoader` loads one file or many files, handles reindexing and frequency normalization, then joins data into one time-indexed frame.
- `load_data(...)` is the high-level measured-data entrypoint: it builds a `CapData`, groups columns, and can append modeled clear-sky POA/GHI when `site` metadata is provided.
- `load_pvsyst(...)` is the PVsyst-specific loader with date/encoding normalization and default regression-column mapping.

### Column grouping (`src/captest/columngroups.py`)
- `group_columns(...)` infers semantic groups from raw column names using type/subtype/sensor keyword dictionaries.
- `ColumnGroups` is a dict-like container used by filtering and plotting APIs.

### Core capacity-test engine (`src/captest/capdata.py`)
- `CapData` holds:
  - raw data (`data`)
  - mutable working set (`data_filtered`)
  - semantic mappings (`column_groups`, `regression_cols`)
  - filter history (`summary`, `removed`, `kept`)
  - reporting/regression outputs (`rc`, `regression_results`)
- Most filtering methods are wrapped with `update_summary`, which records rows kept/removed and filter arguments.
- Standard workflow:
  1. Load into `CapData` (`load_data` or `load_pvsyst`).
  2. Set or adjust `regression_cols` and (optionally) aggregate sensors.
  3. Apply filter methods (mutate `data_filtered`).
  4. Compute reporting conditions with `rep_cond(...)`.
  5. Fit regression with `fit_regression(...)`.
  6. Compare measured vs simulated results with module-level `captest_results(...)`.
- Grouped/time-period predictions are handled by `predict_capacities(...)`, which depends on `rc`, `tolerance`, and grouped irradiance filtering.
- Optional plotting/clear-sky paths are conditionally enabled if optional dependencies are installed (`pvlib`, `holoviews`, `panel`, `openpyxl`).

### Plotting layer (`src/captest/plotting.py`)
- `plot(...)` builds the interactive Panel/HoloViews dashboard (Groups, Layout, Overlay, Scatter tabs).
- `parse_combine(...)` and regex-based selection define combined/default trace groups.
- `plot_defaults.json` in the working directory overrides default displayed groups.
- `residual_plot(...)` compares residual-vs-exogenous behavior between two fitted `CapData` objects.

### Performance-ratio utilities (`src/captest/prtest.py`)
- PR functions (`perf_ratio`, `perf_ratio_temp_corr_nrel`) are separate from ASTM pass/fail capacity-test calculation.
- Results are wrapped in `PrResults` with both aggregate PR outputs and per-timestep result data.

## Test layout
- Test modules are in `tests/`.
- Shared fixtures are in `tests/conftest.py`.
- Fixture datasets and column-group templates are in `tests/data/`.
- `tests/smoke_test.py` is used in publish workflows to validate built wheel/sdist artifacts.
