---
name: unit-tests
description: Write unit tests for pvcaptest (captest) features and bug fixes. Use this skill whenever the user asks to write tests, add test coverage, do TDD, test a new feature, or test a bug fix — even if they just say "write tests for this" or "make sure this is covered". The skill detects the situation automatically, if no uncommitted changes exist, it follows TDD by writing failing tests first; if changes are already written, it reviews the implementation, checks testability, writes covering tests, and verifies coverage on the new code only.
---

# Unit Tests (pvcaptest)

This skill covers two workflows depending on where you are in the development cycle:

- **TDD mode** — no uncommitted changes yet. Write failing tests that define expected behavior, then hand off to implementation.
- **Post-implementation mode** — uncommitted changes exist. Review the code, assess testability, write covering tests, and verify coverage on new code only.

Both workflows produce `pytest`-style tests following the project's conventions.

---

## Step 1: Detect which mode to use

```bash
g -C ~/python/pvcaptest_bt- status --porcelain
```

- **Empty output** → TDD mode. Follow the [TDD Workflow](#tdd-workflow).
- **Output present** → Post-implementation mode. Follow the [Post-Implementation Workflow](#post-implementation-workflow).

---

## TDD Workflow

### TDD Step 1: Understand what needs to be built

If the feature or fix isn't described clearly, ask:
- Which function, method, or behavior should be tested?
- What inputs should it handle? What should it return or do?
- Are there error conditions or edge cases to cover?

Read the relevant source module(s) to understand existing patterns, signatures, and related logic. Source files live under `src/captest/`.

### TDD Step 2: Identify the right test module

| Source file | Test module |
|-------------|-------------|
| `src/captest/capdata.py` | `tests/test_CapData.py` |
| `src/captest/io.py` | `tests/test_io.py` |
| `src/captest/columngroups.py` | `tests/test_columngroups.py` |
| `src/captest/prtest.py` | `tests/test_prtest.py` |
| `src/captest/util.py` | `tests/test_util.py` |
| New module | New `tests/test_<module>.py` |

Read the existing test module to match its style and decide whether to add to an existing class or create a new one.

### TDD Step 3: Write failing tests

Write tests that will fail *because the feature doesn't exist yet*, not because of import errors or fixture problems. The failures should be the right kind (e.g., `AttributeError: has no attribute`, `AssertionError`) — that's what tells you the test harness is wired up correctly and will actually validate the implementation once it exists.

Cover at minimum:
- The primary expected behavior
- At least one edge case or boundary condition

Follow the [Test Writing Conventions](#test-writing-conventions) section below.

### TDD Step 4: Confirm tests fail for the right reason

```bash
just -f ~/python/pvcaptest_bt-/.justfile test-module <test_module.py>
```

If failures are `ImportError` or `SyntaxError`, fix the tests first — those indicate setup problems, not missing implementation. Only hand off when the failures are substantive.

### TDD Step 5: Report and hand off

Summarize:
- Which file the tests are in and which test class/functions were added
- What behavior each test asserts
- How to run them: `just test-module <test_file.py>`

The user implements the feature next. When done, `just test` should go green.

---

## Post-Implementation Workflow

### Post-Impl Step 1: Understand what changed

```bash
g -C ~/python/pvcaptest_bt- diff
g -C ~/python/pvcaptest_bt- status --porcelain
```

Read the changed source files. Focus on:
- What new functions or methods were added?
- What existing behavior changed?
- What are the inputs, outputs, and side effects of the new code?

### Post-Impl Step 2: Assess testability

Before writing tests, consider whether the new code is structured in a way that makes tests straightforward. Look for warning signs:

- **Functions that do too many things at once**: A function that reads a file, transforms data, and emits output in one body is hard to test in isolation. Each concern tested separately is more reliable.
- **Hard-coded external dependencies**: New code that calls `pvlib`, reads from fixed file paths, or makes I/O calls without accepting those as parameters will require either real data or complex mocking.
- **Unobservable outputs**: If the only way to verify correctness is through side effects buried several calls deep, it's worth reconsidering the interface.

If you identify issues that would make tests significantly more fragile or complex than the code itself, describe the specific problem clearly and ask: *"This would be easier to test if [specific refactor]. Would you like to make that change first?"*

If the code looks clean and testable — or if the user says to proceed as-is — continue to Step 3.

### Post-Impl Step 3: Determine test targets

For each new or changed function or method, decide what to test:

1. **Happy path** — typical valid inputs produce the expected output
2. **Boundary/edge cases** — empty DataFrames, zero values, single-row inputs, missing optional parameters
3. **Error/warning paths** — expected exceptions (`pytest.raises`) or warnings (`pytest.warns`)
4. **Regression guard** — if fixing a bug, include a test using the exact inputs that triggered the bug

You don't need to test private helpers unless they contain complex logic that's meaningfully isolated. Focus on the public API surface.

### Post-Impl Step 4: Write the tests

Add new tests to the appropriate test module (see the mapping in TDD Step 2). Add to an existing class if the new tests fit logically there; create a new class if testing a genuinely new capability.

Follow the [Test Writing Conventions](#test-writing-conventions) section below.

### Post-Impl Step 5: Run and iterate on coverage

Run the new tests first to confirm they pass:

```bash
just -f ~/python/pvcaptest_bt-/.justfile test-module <test_module.py>
```

Once they pass, run the full suite with coverage:

```bash
just -f ~/python/pvcaptest_bt-/.justfile test-cov
```

Open `htmlcov/index.html` and navigate to the changed source files. The new lines should be highlighted green. For any uncovered lines:
- Identify what scenario would exercise them
- Add a targeted test
- Re-run `just test-cov` until the new code is fully covered

**Scope boundary**: stop at the new code. Don't attempt to improve coverage for pre-existing uncovered lines — that's a separate task.

---

## Test Writing Conventions

**Framework**: Write new tests in `pytest` style — plain classes with `def test_...` methods, or standalone test functions. The `unittest.TestCase`-based tests in `test_CapData.py` are legacy; don't add new ones in that style.

**Class naming**: Group related tests under a class named `TestSomethingDescriptive`, where the name makes the scope obvious — e.g., `TestFilterIrr`, `TestRepCond`, `TestLoadDataColumnGrouping`.

**Fixtures**: Reuse shared fixtures from `conftest.py` whenever they fit: `meas`, `pvsyst`, `nrel`, `nrel_clear_sky`, `pvsyst_irr_filter`, `capdata_irr`, `cd_nested_col_groups`, `location_and_system`. For file I/O tests, use pytest's built-in `tmp_path` fixture.

**Working directory**: All tests run from the project root (`~/python/pvcaptest_bt-`). Fixtures use relative paths like `"./tests/data/..."` that depend on this. Never hard-code absolute paths in test files.

**Assertions**: Use specific assertions rather than generic ones. For floating-point comparisons, use `pytest.approx` or `np.testing.assert_allclose`. Avoid testing more than one thing per test method — if a test is doing a lot, split it.

**Test docstrings**: Write a one-sentence docstring per test method that describes *what it verifies*, not what it does mechanically.

```python
def test_filter_irr_excludes_out_of_range_rows(self):
    """Verify rows outside [low, high] irradiance bounds are removed."""
```

**What to avoid**:
- Don't test pandas/numpy internals — only test captest behavior
- Don't use `print()` for debugging — assert against intermediate values directly
- Don't leave commented-out test code

---

## Running tests

```bash
# Run the full suite
just -f ~/python/pvcaptest_bt-/.justfile test

# Run a single test module
just -f ~/python/pvcaptest_bt-/.justfile test-module test_io.py

# Run a single test by node ID
uv run pytest tests/test_CapData.py::TestUpdateSummary::test_round_kwarg_floats

# Full suite with HTML coverage report (output in htmlcov/)
just -f ~/python/pvcaptest_bt-/.justfile test-cov
```

All tests must pass before committing or opening a PR.
