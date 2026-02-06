set dotenv-load # loads env variables from .env
set shell := ["zsh", "-c"]

# Delete dist folder and uv build
build:
	rm -r dist/
	uv build

# Run build and then publish to testpypi
publish-to-testpypi: build
	uv publish --index testpypi --token $UV_PUBLISH_TESTPYPI_TOKEN # token from .env

# Lint using ruff
lint file=".":
	uv run ruff check --fix {{file}}

# Format using ruff, applies black formatting
fmt file=".":
	uv run ruff format {{file}}

# Run all tests
test python-ver="3.12":
	uv run --python {{python-ver}} pytest tests/

# Run all tests without warnings
test-wo-warnings python-ver="3.12":
	uv run --python {{python-ver}} pytest tests/ --disable-warnings

# Run all tests with coverage report
test-cov:
	uv run pytest --cov-report html --cov=src/captest tests/

# Reminder on how to run a specific test module
test-module-example:
	@echo "To run a class of tests: "
	@echo "uv run pytest tests/test_CapData.py::TestCapDataEmpty\n"
	@echo "To run a specific test:"
	@echo "uv run pytest tests/test_CapData.py::TestCapDataEmpty::test_capdata_empty\n"

# Run a specific test 
test-module module_name python-ver="3.12":
	uv run --python {{python-ver}} pytest --disable-warnings tests/{{module_name}}

# Test install package in new venv
test-install python-ver="3.12":
	uv venv ../_pvc_test_dir/.venv --python {{python-ver}}
	uv pip install --python ../_pvc_test_dir/.venv ./dist/*.whl
	cd ../_pvc_test_dir && .venv/bin/python -c "import captest; print(captest.__version__)"
	rm -rf ../_pvc_test_dir

# Check current version of package
ver:
	uv run python -c "import captest; print(captest.__version__)"

# Build docs with sphinx-build
docs:
	uv run sphinx-build -M html ./docs/ ./docs/_build/
