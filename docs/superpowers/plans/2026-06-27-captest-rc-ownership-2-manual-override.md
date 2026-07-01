# CapTest RC Ownership — Plan 2: Manual Override (`ct.rc = df`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CapTest.rc` settable — the single public way to supply reporting conditions manually (`rc_source="manual"`) — with validation that the value covers every right-hand-side regression variable.

**Architecture:** Add a setter to the `rc` property introduced in Plan 1. The setter requires `setup()`, checks that `meas`/`sim` share a regression formula, coerces a DataFrame / Series / dict to a one-row DataFrame, validates RHS coverage via `util.parse_regression_formula`, then records the value through the existing `_set_rc(df, "manual")` write point (which applies the Plan 1 source-change warning). No new write path — the setter is the public face of `_set_rc(..., "manual")`.

**Tech Stack:** Python, `param`, pandas, pytest, `uv`, `just`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-captest-rc-ownership-design.md` §4.4, §6.
- Builds on Plan 1 (`CapTest.rc` getter, `_set_rc`, `rc_source` incl. `"manual"`, `CapData._captest`). Plan 1 must be applied first.
- **Plans 1-5 must merge and release as a unit** — the cross-instance `rep_cond → rep_irr` workflow remains intentionally non-functional until Plans 3/5 (see Plan 1).
- `from captest import util` and `import pandas as pd` are already present in `captest.py` — no new imports required.
- `_require_setup()` raises `RuntimeError` (existing helper). The spec §6 phrasing says "ValueError"; this plan reuses `_require_setup()` for DRY, so the not-setup error is `RuntimeError`. (Noted in Self-Review.)
- Line length 88 (ruff). NumPy-style docstrings. Run tests with `just -f ~/python/pvcaptest_bt-/.justfile test-module <file>`; lint with `uv run ruff check` / `uv run ruff format`.

---

### Task 1: `CapTest.rc` setter with RHS-coverage validation

**Files:**
- Modify: `src/captest/captest.py` — add an `@rc.setter` immediately after the `rc` getter (added in Plan 1, just after `__init__`).
- Test: `tests/test_captest.py` — add `from captest import util` to imports; new class `TestManualRc`.

**Interfaces:**
- Consumes: `CapTest._set_rc(rc, source, warn=True)`, `CapTest._require_setup()`, `util.parse_regression_formula(formula) -> (lhs_list, rhs_list)`, `CapData.regression_formula`.
- Produces:
  - `CapTest.rc` setter: `ct.rc = value` where `value` is a one-row `pandas.DataFrame`, a `pandas.Series`, or a `dict` of `regression_variable -> value`. Sets `rc_source="manual"`. Raises `RuntimeError` (not setup), `ValueError` (meas/sim formula mismatch, or missing RHS variables), `TypeError` (unsupported `value` type). Warns (via `_set_rc`) only when it changes the source.

- [ ] **Step 1: Write the failing tests**

First add `from captest import util` to the imports of `tests/test_captest.py` (the module imports `pytest`, `pandas as pd`, `CapTest` but not `util`; one new test calls `util.parse_regression_formula`). Then add the test class below.

`ct_default` is a fixture in `conftest.py` providing a `setup()`-run CapTest with the `e2848_default` formula `power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1`. Its RHS variables appear *only* inside `I(...)` interaction blocks; `util.parse_regression_formula` unwraps them to the component set `poa, t_amb, w_vel`, which is what coverage is validated against:

```python
class TestManualRc:
    """The public ct.rc = df manual-override setter (rc_source='manual')."""

    def _full_rc(self, poa=805.0):
        return pd.DataFrame({"poa": [poa], "t_amb": [25.0], "w_vel": [2.0]})

    def test_set_rc_dataframe_sets_manual_source(self, ct_default):
        df = self._full_rc()
        ct_default.rc = df
        assert ct_default.rc_source == "manual"
        assert ct_default.rc["poa"].iloc[0] == pytest.approx(805.0)

    def test_set_rc_accepts_dict(self, ct_default):
        ct_default.rc = {"poa": 700.0, "t_amb": 20.0, "w_vel": 1.5}
        assert ct_default.rc_source == "manual"
        assert ct_default.rc["poa"].iloc[0] == pytest.approx(700.0)

    def test_set_rc_accepts_series(self, ct_default):
        ct_default.rc = pd.Series({"poa": 650.0, "t_amb": 18.0, "w_vel": 1.0})
        assert ct_default.rc_source == "manual"
        assert ct_default.rc["poa"].iloc[0] == pytest.approx(650.0)

    def test_set_rc_preserves_extra_columns(self, ct_default):
        ct_default.rc = {"poa": 805.0, "t_amb": 25.0, "w_vel": 2.0, "note": 1.0}
        assert "note" in ct_default.rc.columns

    def test_set_rc_series_preserves_extra_columns(self, ct_default):
        """Extra fields survive the Series -> one-row DataFrame coercion."""
        s = pd.Series({"poa": 805.0, "t_amb": 25.0, "w_vel": 2.0, "note": 9.0})
        ct_default.rc = s
        assert "note" in ct_default.rc.columns
        assert ct_default.rc["note"].iloc[0] == pytest.approx(9.0)

    def test_required_vars_are_unwrapped_interaction_components(self, ct_default):
        """Coverage keys off RHS *component* variables (poa, t_amb, w_vel), not
        the I(...) interaction terms, for the default formula. Guards against a
        parse_regression_formula change silently weakening validation."""
        _, rhs = util.parse_regression_formula(ct_default.meas.regression_formula)
        assert sorted(rhs) == ["poa", "t_amb", "w_vel"]
        # A df with only the component vars (no I(poa*poa) columns) is accepted.
        ct_default.rc = {"poa": 805.0, "t_amb": 25.0, "w_vel": 2.0}
        assert ct_default.rc_source == "manual"

    def test_set_rc_multirow_raises(self, ct_default):
        """A multi-row DataFrame is rejected with a clear error at set time."""
        df = pd.DataFrame(
            {"poa": [805.0, 810.0], "t_amb": [25.0, 26.0], "w_vel": [2.0, 2.1]}
        )
        with pytest.raises(ValueError, match="single row"):
            ct_default.rc = df

    def test_set_rc_missing_rhs_var_raises_listing_names(self, ct_default):
        with pytest.raises(ValueError, match=r"missing required regression"):
            ct_default.rc = pd.DataFrame({"poa": [805.0]})

    def test_set_rc_missing_var_message_names_the_missing(self, ct_default):
        with pytest.raises(ValueError) as exc:
            ct_default.rc = {"poa": 805.0, "t_amb": 25.0}  # w_vel missing
        assert "w_vel" in str(exc.value)

    def test_set_rc_requires_setup(self):
        ct = CapTest()  # bare, setup() not run
        with pytest.raises(RuntimeError, match="setup"):
            ct.rc = pd.DataFrame({"poa": [805.0], "t_amb": [25.0], "w_vel": [2.0]})

    def test_set_rc_formula_mismatch_raises(self, ct_default):
        ct_default.sim.regression_formula = "power ~ poa"
        with pytest.raises(ValueError, match="different regression formulas"):
            ct_default.rc = self._full_rc()

    def test_set_rc_bad_type_raises(self, ct_default):
        with pytest.raises(TypeError, match="DataFrame"):
            ct_default.rc = [805.0, 25.0, 2.0]

    def test_set_rc_first_set_is_silent(self, ct_default, recwarn):
        ct_default.rc = self._full_rc()
        assert len(recwarn) == 0

    def test_set_rc_over_computed_source_warns(self, ct_default):
        ct_default._set_rc(self._full_rc(), "meas")  # seed a computed source
        with pytest.warns(UserWarning, match="changed from 'meas' to 'manual'"):
            ct_default.rc = self._full_rc(810.0)

    def test_set_rc_over_manual_is_silent(self, ct_default, recwarn):
        ct_default.rc = self._full_rc()        # first -> manual (silent)
        ct_default.rc = self._full_rc(810.0)   # manual -> manual (silent)
        assert len(recwarn) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `AttributeError: can't set attribute 'rc'` (the property has no setter yet).

- [ ] **Step 3: Add the `rc` setter**

In `src/captest/captest.py`, immediately after the `rc` getter (added in Plan 1), add:

```python
    @rc.setter
    def rc(self, value):
        """Set the test reporting conditions manually (``rc_source='manual'``).

        This is the only public way to supply reporting conditions directly —
        e.g. for sensitivity analysis or to check results against a reviewing
        party's values. Computed conditions go through :meth:`rep_cond` instead.

        Parameters
        ----------
        value : pandas.DataFrame or pandas.Series or dict
            One-row reporting conditions. A Series or dict maps each regression
            variable to its value; a DataFrame is used as given. Must provide a
            value for every right-hand-side variable of the (shared meas/sim)
            regression formula. Extra columns are preserved.

        Raises
        ------
        RuntimeError
            If :meth:`setup` has not run (the regression formula is unknown).
        ValueError
            If ``meas`` and ``sim`` have different regression formulas, if
            ``value`` coerces to more than one row, or if ``value`` omits a
            required right-hand-side variable.
        TypeError
            If ``value`` is not a DataFrame, Series, or dict.
        """
        self._require_setup()
        meas_fml = self.meas.regression_formula
        sim_fml = self.sim.regression_formula
        if meas_fml != sim_fml:
            raise ValueError(
                "Cannot set reporting conditions manually: meas and sim have "
                f"different regression formulas ({meas_fml!r} vs {sim_fml!r})."
            )
        _, rhs = util.parse_regression_formula(meas_fml)
        if isinstance(value, pd.DataFrame):
            df = value.copy()
        elif isinstance(value, pd.Series):
            df = value.to_frame().T
        elif isinstance(value, dict):
            df = pd.DataFrame([value])
        else:
            raise TypeError(
                "ct.rc must be a one-row DataFrame, a pandas Series, or a dict "
                f"mapping regression variable -> value; got "
                f"{type(value).__name__}."
            )
        if len(df) != 1:
            raise ValueError(
                f"Reporting conditions must be a single row; got {len(df)} rows."
            )
        missing = [var for var in rhs if var not in df.columns]
        if missing:
            raise ValueError(
                "Manual reporting conditions are missing required regression "
                f"variable(s): {missing}. Required: {rhs}."
            )
        self._set_rc(df, "manual")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for `TestManualRc`; rest of module still passes.

- [ ] **Step 5: Run the full suite**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test`
Expected: all tests pass (Plan 1 tests included; the `xfail` placeholder still reports XFAIL).

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 7: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: add manual CapTest.rc setter with RHS-coverage validation

ct.rc = df is the single public way to set reporting conditions manually
(rc_source='manual'); validates the value covers every RHS regression
variable of the shared meas/sim formula and records it via _set_rc.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Plan 2 scope = §4.4 manual override):**
- §4.4 "only public way is `ct.rc = df`" → setter is the sole public write path; `_set_rc` stays internal. ✓
- §4.4 step 1 requires `setup()` → `self._require_setup()`. ✓
- §4.4 step 2 RHS validation via `util.parse_regression_formula(self.meas.regression_formula)[1]`; formula-mismatch raises; missing vars listed; extra columns preserved (DataFrame *and* Series paths tested) → covered. A dedicated test asserts the resolved RHS for the default formula is the *unwrapped* `["poa", "t_amb", "w_vel"]`, guarding that coverage keys off `I(...)` component symbols, not the interaction terms. ✓
- §4.4 "one-row" contract enforced: after coercion the setter rejects `len(df) != 1` with a `ValueError` ("single row"); `test_set_rc_multirow_raises` covers it. ✓
- §4.4 step 3 coerce to one-row DataFrame; apply §4.5 warning; record via `_set_rc(df, "manual")` → the warning lives in `_set_rc` (Plan 1), exercised by `test_set_rc_over_computed_source_warns`. ✓
- §4.4 "internal vs public path" — computed/load assignments use `_set_rc` directly (Plans 3/5); the public setter is `_set_rc(..., "manual")` plus validation. ✓

**Placeholder scan:** none — every step has exact code/commands.

**Spec deviation (noted):** §6 lists the not-setup error as `ValueError`, but this plan reuses `_require_setup()` which raises `RuntimeError` (DRY with the rest of `CapTest`). `test_set_rc_requires_setup` asserts `RuntimeError`. Flag for reviewer; trivially switchable to a `ValueError` raise if the spec's letter is preferred over reuse.

**Type consistency:** setter accepts `DataFrame | Series | dict`; coerces to a one-row DataFrame; delegates to `_set_rc(df, "manual")` (signature from Plan 1). `util.parse_regression_formula` returns `(lhs, rhs)` and only `rhs` is used. Required-vars list and the error message both reference `rhs`.
