# Maintainer Notes

This document contains notes and procedures for project maintainers.

## Just Task Runner

This project is using [just](https://just.systems/man/en/introduction.html) to provide short nnames for recipes of common tasks, like building, running tests, and publishing.

Install just with `uv tool install rust-just` or see the [packages](https://just.systems/man/en/packages.html) for other options.

### Just notes
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
