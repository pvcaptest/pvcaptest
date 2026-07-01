# CapTest RC Ownership — Plan 4: `captest_results` Reads `ct.rc`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `captest_results` predict at the single test RC `ct.rc` instead of re-picking `meas.rc`/`sim.rc` by `rc_source`, and raise a clear error when no test RC has been established.

**Architecture:** Replace the `if self.rc_source == "meas": rc = self.meas.rc else: rc = self.sim.rc` block in `captest_results` with `rc = self.rc` plus a `None` guard. The provenance print line keeps using `self.rc_source`. The formula-mismatch check and `_require_meas_and_sim` ordering are unchanged. `captest_results_check_pvalues` calls `captest_results` internally, so it inherits the change with no edits.

**Tech Stack:** Python, `param`, pandas, statsmodels, pytest, `uv`, `just`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-27-captest-rc-ownership-design.md` §4.6.
- Builds on Plans 1-3 (`CapTest.rc`/`rc_source`, `_set_rc`, manual setter, `rep_cond` sync). Apply them first; the integration harness fix lands in Plan 3.
- **Plans 1-5 must merge and release as a unit.**
- Order of checks in `captest_results` is preserved: `_require_meas_and_sim()` → formula-match → RC resolution. The `None`-RC `ValueError` therefore only fires after meas/sim exist and formulas match.
- Line length 88 (ruff). NumPy-style docstrings. Run tests with `just -f ~/python/pvcaptest_bt-/.justfile test-module <file>`; lint with `uv run ruff check` / `uv run ruff format`.

---

### Task 1: `captest_results` predicts at `ct.rc`

**Files:**
- Modify: `src/captest/captest.py` — `captest_results` (the `rc_source`-based pick).
- Test: `tests/test_captest.py` — update `TestPortedMethods` RC tests; add a `None`-RC test.

**Interfaces:**
- Consumes: `CapTest.rc` (Plan 1), `CapTest.rc_source`, `CapTest._set_rc` (Plan 1, used by tests to seed `ct.rc`).
- Produces: `captest_results(check_pvalues=False, pval=0.05, print_res=True)` now reads `self.rc`; raises `ValueError` when `self.rc is None` (after the meas/sim and formula checks). Return type unchanged (float cap ratio). `captest_results_check_pvalues` unchanged but now requires `ct.rc` (it calls `captest_results`).

- [ ] **Step 1: Update the existing RC-dependent tests and add the None-guard test**

In `tests/test_captest.py`, replace the three tests `test_captest_results_matches_direct_prediction`, `test_captest_results_uses_rc_source_meas_by_default`, and `test_captest_results_uses_rc_source_sim` (the ones that set `meas.rc`/`sim.rc` and rely on the `rc_source` pick) with these, which seed the single `ct.rc` via `_set_rc`:

```python
    def test_captest_results_predicts_at_ct_rc(self):
        capt = self._build_ct()
        rc = pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]})
        capt._set_rc(rc, "meas")
        expected_actual = capt.meas.regression_results.predict(rc)[0]
        expected_expected = capt.sim.regression_results.predict(rc)[0]
        expected_ratio = expected_actual / expected_expected

        cp_rat = capt.captest_results(print_res=False)

        assert cp_rat == pytest.approx(expected_ratio, rel=1e-10)

    def test_captest_results_uses_ct_rc_values_regardless_of_source(self):
        capt = self._build_ct()
        # Distinct RC values; both meas and sim predict at the SAME ct.rc.
        rc = pd.DataFrame({"poa": [8], "t_amb": [4], "w_vel": [2]})
        capt._set_rc(rc, "sim")
        expected_ratio = (
            capt.meas.regression_results.predict(rc)[0]
            / capt.sim.regression_results.predict(rc)[0]
        )

        cp_rat = capt.captest_results(print_res=False)

        assert cp_rat == pytest.approx(expected_ratio, rel=1e-10)

    def test_captest_results_raises_when_ct_rc_none(self):
        capt = self._build_ct()
        assert capt.rc is None
        with pytest.raises(ValueError, match="requires test reporting conditions"):
            capt.captest_results(print_res=False)
```

Then update `test_captest_results_check_pvalues_returns_styled_df` to seed `ct.rc` before calling (it routes through `captest_results`):

```python
    def test_captest_results_check_pvalues_returns_styled_df(self):
        capt = self._build_ct()
        capt._set_rc(pd.DataFrame({"poa": [6], "t_amb": [5], "w_vel": [3]}), "meas")
        styled = capt.captest_results_check_pvalues(print_res=False)
```

(Leave the remaining body of `test_captest_results_check_pvalues_returns_styled_df` unchanged. `test_captest_results_requires_meas_and_sim` and `test_captest_results_warns_on_mismatched_formulas` need no change — the meas/sim and formula checks run before the RC resolution.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: FAIL — `captest_results` still reads `meas.rc`/`sim.rc` (which `_build_ct` sets to `{poa:6,...}`), so `test_captest_results_predicts_at_ct_rc` may pass by luck but `test_captest_results_uses_ct_rc_values_regardless_of_source` fails (it reads `meas.rc`=`{poa:6}`, not the seeded `ct.rc`=`{poa:8}`) and `test_captest_results_raises_when_ct_rc_none` fails (no error raised; it reads `meas.rc`).

- [ ] **Step 3: Rewrite the RC resolution in `captest_results`**

In `src/captest/captest.py`, in `captest_results`, replace:

```python
        if self.rc_source == "meas":
            rc = self.meas.rc
        else:
            rc = self.sim.rc

        if print_res:
            print(f"Using reporting conditions from {self.rc_source}. \n")
```

with:

```python
        rc = self.rc
        if rc is None:
            raise ValueError(
                "captest_results requires test reporting conditions. Call "
                "ct.rep_cond(which) or assign ct.rc = df first."
            )

        if print_res:
            print(f"Using reporting conditions from {self.rc_source}. \n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test-module test_captest.py`
Expected: PASS for the updated/added tests.

- [ ] **Step 5: Update the `captest_results` docstring**

In `src/captest/captest.py`, the `captest_results` docstring currently says it "Picks reporting conditions from `self.meas.rc` or `self.sim.rc` based on `self.rc_source`." Replace that sentence with:

```
        Predicts both regressions at the single test reporting conditions
        ``self.rc`` (set via :meth:`rep_cond` or the ``rc`` setter);
        ``self.rc_source`` is reported for provenance. Raises ``ValueError``
        if ``self.rc`` is ``None``.
```

(Find the existing wording in the method's docstring and swap it; keep the rest.)

- [ ] **Step 6: Run the full suite**

Run: `just -f ~/python/pvcaptest_bt-/.justfile test`
Expected: all tests pass. `TestIntegration` already seeds `ct.rc` via `capt.rep_cond()` in `_run_canonical_sequence` (Plan 3), so its `captest_results` calls resolve normally.

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/captest/captest.py tests/test_captest.py && uv run ruff format src/captest/captest.py tests/test_captest.py`
Expected: All checks pass.

- [ ] **Step 8: Commit**

```bash
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: captest_results predicts at the single test rc (ct.rc)

Replace the rc_source-based pick of meas.rc/sim.rc with ct.rc; raise a clear
ValueError when no test reporting conditions are set. captest_results_check_pvalues
inherits the change via its internal captest_results calls.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Plan 4 scope = §4.6 `captest_results`):**
- §4.6 replace `rc_source` pick with `self.rc` → Task 1, Step 3. ✓
- §4.6 raise a clear error directing to `ct.rep_cond(which)` / `ct.rc = df` when `self.rc is None` → `ValueError`; `test_captest_results_raises_when_ct_rc_none`. ✓
- §4.6 provenance print uses `self.rc_source` → unchanged print line. ✓
- `captest_results_check_pvalues` inherits the change (calls `captest_results`) → test seeds `ct.rc`. ✓

**Placeholder scan:** none — every step shows exact code/commands.

**Check ordering:** `_require_meas_and_sim()` and the formula-mismatch check run *before* the `None`-RC guard, so `test_captest_results_requires_meas_and_sim` (RuntimeError) and `test_captest_results_warns_on_mismatched_formulas` (UserWarning + early return) keep passing without seeding `ct.rc`. ✓

**Type consistency:** `captest_results` still returns the float cap ratio; `rc = self.rc` is a one-row DataFrame (or None → ValueError). Tests seed `ct.rc` via `_set_rc(df, source)` (Plan 1 signature). The integration harness change that makes `TestIntegration` compatible with the single-`ct.rc` model lives in Plan 3 (`_run_canonical_sequence`), not here.
