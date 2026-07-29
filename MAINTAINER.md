# Maintainer Notes

This document contains notes and procedures for project maintainers.

## Just Task Runner

This project is using [just](https://just.systems/man/en/introduction.html) to provide short nnames for recipes of common tasks, like building, running tests, and publishing.

Install just with `uv tool install rust-just` or see the [packages](https://just.systems/man/en/packages.html) for other options.

### Just notes
- Run `just` commands from the project root directory
- `just --list` shows available recipes
- recipes are stored in `.justfile` in the project root directory

## Pre-commit Hooks

This project uses pre-commit to automatically run code quality checks before commits. The hooks include ruff for linting and formatting.

### Setup

After cloning the repository, install the pre-commit hooks:

```bash
pre-commit install
```

This will configure git to run the hooks automatically before each commit.

### Running Manually

To run pre-commit on all files without committing:

```bash
pre-commit run --all-files
```

Or, run ruff commands. Run `just --list` to see options.


## Publishing to TestPyPI

TestPyPI (test.pypi.org) is a separate instance of the Python Package Index for testing distribution tools and processes without affecting the real index.

### One-time Setup

1. **Create a TestPyPI account** at https://test.pypi.org

2. **Generate an API token**:
   - Log in to TestPyPI
   - Go to Account Settings → API tokens
   - Create a new token with appropriate scope
   - Copy the token (it starts with `pypi-`)

3. **Store the token securely**:
   - save 'UV_PUBLISH_TESTPYPI_TOKEN="pypi-your-token-here" to .env

4. **Ensure `.env` is in `.gitignore`**:
   ```bash
   echo '.env' >> .gitignore
   ```

### Publishing a Test Release

1. **Build and publish**:
    - Use the just command `publish-to-testpypi` to build and publish to test pypi server.

### Notes

- TestPyPI is completely separate from the production PyPI index
- Packages and accounts on TestPyPI do not affect the real index
- Use TestPyPI to verify the publishing process before releasing to production PyPI

## Smoke Testing a Release in an Isolated Environment

After a release candidate is published, install it from PyPI into a throwaway
environment and check that it imports and runs. The point is to test the package
*as a user receives it* — only the dependencies declared in `pyproject.toml`,
none of the development environment.

Run these from the project root:

```bash
# Minimal install: only the required dependencies, no optional extras.
uv run --isolated --no-project --refresh --with 'captest==0.17.0rc1' \
    python tests/smoke_test.py

# With optional extras (holoviews, panel, pvlib, openpyxl, fsspec[s3]).
uv run --isolated --no-project --with 'captest[optional]==0.17.0rc1' \
    python path/to/functional_check.py
```

### What each flag does

| Flag | Effect |
| --- | --- |
| `--isolated` | Runs the command in an isolated virtual environment rather than the project's `.venv`, so nothing already installed for development is visible. |
| `--no-project` | Stops uv from discovering the project or workspace, so the local source tree is *not* installed and the dev dependency groups are not applied. Without it, `uv run` installs the working copy and the smoke test would exercise local source instead of the published artifact. |
| `--with 'captest==0.17.0rc1'` | Installs the named package into the ephemeral environment. Pin the exact version — an unpinned `captest` resolves to the latest *stable* release, because pip and uv exclude pre-releases unless a pre-release version is requested explicitly. |
| `--refresh` | Refreshes all cached data. Worth including on the first run after publishing, so uv fetches current index metadata instead of a cached listing that predates the upload. |

The trailing `python <script>` is the command uv runs inside that environment.
The script is read from disk by the isolated interpreter and is not installed
into it, so the environment contains only what `--with` put there — which is why
the script may live outside the repository.

### Why this catches things the test suite does not

`just test` runs against the development environment, where every optional
dependency is installed and a large set of packages has already been imported.
An isolated install has neither, so it surfaces failures the suite cannot see:

- Missing package data — files present in the source tree but not included in
  the built wheel or sdist.
- Optional-dependency guards that are wrong. Every `importlib.util.find_spec`
  branch takes its unavailable path here, which is the only place that code runs.
- Imports that only resolve by accident. Before v0.17.0, several modules called
  `importlib.util.find_spec` after a plain `import importlib`, which does not
  bind the `util` submodule. It worked in development only because another
  installed package had already imported `importlib.util`; a minimal install
  raised `AttributeError` on `import captest`.

The publish workflow (`.github/workflows/publish.yml`) runs the same check
against the freshly built artifacts before uploading, and fails the release if
either fails:

```bash
uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
```

Both the wheel and the source distribution are tested because they are built
separately and can differ in the files they include.
