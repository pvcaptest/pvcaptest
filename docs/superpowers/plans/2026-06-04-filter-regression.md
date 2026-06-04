# FilterRegression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the residual-outlier filtering done by `fit_regression(filter=True)` into a first-class `FilterRegression` class, so it records a proper summary step and explanation like every other filter — instead of an ad-hoc `data_filtered` assignment.

**Architecture:** `FilterRegression(BaseFilter)` fits the CapData regression formula (via the shared `fit_model` helper) and returns the index of rows whose residual is within `n_std` residual standard deviations; it exposes the fitted model on `self.regression_model` so callers can print its summary. `fit_regression(filter=True)` becomes a thin delegation to `FilterRegression().run(self)`; `fit_regression(filter=False)` keeps its own plain fit. To make `fit_model` reachable from `filters.py` without an import cycle, `fit_model` (and its statsmodels import) moves into `filters.py`; `capdata.py` re-imports it (Strategy A1). The residual cutoff is a configurable `n_std` param (default 2).

**Tech Stack:** Python, `param`, pandas, statsmodels, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes" (`FitRegression` is envisioned as a summary step; this realizes the filtering half as a `BaseFilter`).

**Sequencing:** Execute **before** the `data_filtered` property plan (`2026-06-04-data-filtered-property.md`). Doing so **removes Task 4 from that plan** — `fit_regression` will already route through `run()`, so the property flip needs no `FilterCustom(keep_rows)` workaround.

## Design notes (decided with the user)

- **Minimal duplication (Strategy A):** the residual-filter rule lives once, in `FilterRegression._execute`. `fit_model` is the single shared fit helper used by both the filter and the plain-fit path. Each call path fits exactly once — no double-fit. (Considered and rejected: a separate `residual_filter()` helper only `FilterRegression` would use; injecting a pre-fit model, which standalone pipeline use can't provide.)
- **`fit_model` location = A1:** moved to `filters.py`. `capdata.py` re-imports it (needed anyway for `predict_capacities`' `grps.apply(fit_model)` at ~line 684), so `capdata.fit_model` / `pvc.fit_model` still resolve — the three existing `pvc.fit_model` test refs and `test_fit_model` keep working with no repoint.
- **`n_std` is a param** (default 2). `fit_regression(filter=True)` passes `n_std=2` to preserve current behavior.
- **`_legacy_name = "fit_regression"`** preserves the transitional summary label. No test asserts this label (verified), and the existing `fit_regression` calls in `test_captest.py` use `filter=False`, so the conversion is safe.
- **`filter=True` never stored `regression_results`** (only `filter=False` does) — preserved.
- **NaN handling (mirrors `FilterOutliers`):** if the regression columns contain NaN, statsmodels' formula fit drops those rows, leaving `reg.resid` shorter than `df`, and the residual mask raises `IndexingError: Unalignable boolean Series` (verified empirically; the legacy `fit_regression` had the same latent bug). `FilterRegression._execute` handles it **proactively** — like `FilterOutliers` does for `(poa, power)` — by warning and calling `capdata.filter_missing()` (which records its own `filter_missing` step) before fitting. The kept index is taken from `reg.resid[...]` (not `df[...]`), which is self-aligned and avoids both the misalignment error and pandas' "Boolean Series key will be reindexed" warning.
  - *Deviation from review #82's Low finding:* the reviewer suggested `smf.ols(..., missing="raise")` + catch-and-retry. I use the proactive approach instead because (a) it matches the established `FilterOutliers` precedent in this codebase, and (b) it keeps the **shared** `fit_model` (also used by `fit_regression(filter=False)` and `predict_capacities`' `grps.apply(fit_model)`) on its current `missing="drop"` default — changing that globally to `"raise"` is a broad behavior change out of this plan's scope. The net effect (run `filter_missing` when NaN is present) is the same.

---

### Task 1: Move `fit_model` from `capdata.py` to `filters.py`

**Files:** Modify `src/captest/filters.py`, `src/captest/capdata.py`.

- [ ] **Step 1: Add the statsmodels import to `filters.py`**

In `src/captest/filters.py`, add to the third-party imports (after `import param`):

```python
import statsmodels.formula.api as smf
```

- [ ] **Step 2: Move `fit_model` into `filters.py`**

Move the `fit_model` function **verbatim** from `capdata.py` (currently ~lines 611-633) into `filters.py`, placing it after `spans_year` and before `class BaseSummaryStep`:

```python
def fit_model(
    df, fml="power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
):  # noqa E501
    """
    Fits linear regression using statsmodels to dataframe passed.

    Dataframe must be first argument for use with pandas groupby object
    apply method.

    Parameters
    ----------
    df : pandas dataframe
    fml : str
        Formula to fit refer to statsmodels and patsy documentation for format.
        Default is the formula in ASTM E2848.

    Returns
    -------
    Statsmodels linear model regression results wrapper object.
    """
    mod = smf.ols(formula=fml, data=df)
    reg = mod.fit()
    return reg
```

- [ ] **Step 3: Delete `fit_model` from `capdata.py`** (the function body just moved).

- [ ] **Step 4: Re-import `fit_model` into `capdata.py`**

Add `fit_model` to the `from captest.filters import (...)` block in `capdata.py` (alphabetical among the lowercase helpers):

```python
from captest.filters import (
    BaseSummaryStep,
    FilterClearsky,
    FilterCustom,
    FilterDays,
    FilterIrr,
    FilterMissing,
    FilterOutliers,
    FilterPf,
    FilterPower,
    FilterPvsyst,
    FilterSensors,
    FilterShade,
    FilterTime,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    fit_model,
    wrap_year_end,
)
```

- [ ] **Step 5: Remove the now-unused statsmodels import from `capdata.py`**

`smf` is used only by `fit_model` (verified). Delete from `capdata.py`:

```python
import statsmodels.formula.api as smf
```

- [ ] **Step 6: Verify the move**

Run: `uv run python -c "from captest import capdata; print(capdata.fit_model)"`
Expected: prints the function (re-exported via the import), no error.
Run: `uv run pytest tests/test_CapData.py -k "fit_model" -q`
Expected: PASS (`test_fit_model` and the other `pvc.fit_model` references resolve via the re-import).
Run: `just lint`
Expected: clean (no `F401` for `smf` in capdata.py).

- [ ] **Step 7: Commit**

```bash
git add src/captest/filters.py src/captest/capdata.py
git commit -m "refactor: move fit_model into captest.filters (capdata re-imports)"
```

---

### Task 2: Add the `FilterRegression` class

**Files:** Modify `src/captest/filters.py`, `tests/test_filter_classes.py`.

- [ ] **Step 1: Write the failing tests**

Add `FilterRegression` to the top-of-file `filters` import in `tests/test_filter_classes.py`. Add a self-contained `cd_reg` fixture near the other fixtures (the conftest `nrel` fixture has **no power column** and cannot fit the ASTM formula, so it can't be used here — verified). This fixture uses a simple `power ~ poa` formula with one gross outlier at index 10 (deterministic — verified empirically that `n_std=2` removes exactly that point):

```python
@pytest.fixture
def cd_reg():
    """A CapData with a clean linear power~poa relationship + one outlier."""
    n = 40
    poa = np.linspace(100, 1000, n)
    rng = np.random.default_rng(0)
    power = poa * 0.5 + rng.normal(0, 3, n)
    power[10] += 400  # gross residual outlier at index 10
    cd = CapData("reg")
    cd.data = pd.DataFrame({"poa": poa, "power": power}, index=pd.RangeIndex(n))
    cd.data_filtered = cd.data.copy()
    cd.column_groups = {"irr-poa-": ["poa"], "real_pwr--": ["power"]}
    cd.regression_cols = {"power": "power", "poa": "poa"}
    cd.regression_formula = "power ~ poa"
    return cd
```

Append `TestFilterRegression`:

```python
class TestFilterRegression:
    def test_n_std_default_is_2(self):
        assert FilterRegression().n_std == 2

    def test_execute_exposes_fitted_model(self, cd_reg):
        f = FilterRegression()
        f._execute(cd_reg)
        assert hasattr(f, "regression_model")
        assert hasattr(f.regression_model, "resid")

    def test_execute_removes_the_outlier(self, cd_reg):
        kept = FilterRegression(n_std=2)._execute(cd_reg)
        assert 10 not in kept  # the injected outlier is dropped
        assert len(kept) == cd_reg.data_filtered.shape[0] - 1

    def test_execute_kept_rows_within_threshold(self, cd_reg):
        f = FilterRegression(n_std=2)
        kept = f._execute(cd_reg)
        reg = f.regression_model
        threshold = 2 * reg.scale**0.5
        assert (reg.resid.loc[kept].abs() < threshold).all()

    def test_larger_n_std_keeps_more(self, cd_reg):
        kept_2 = FilterRegression(n_std=2)._execute(cd_reg)
        kept_4 = FilterRegression(n_std=4)._execute(cd_reg)
        assert len(kept_4) >= len(kept_2)

    def test_execute_nan_calls_filter_missing(self, cd_reg):
        # NaN in a regression column: must run filter_missing (recording its
        # own step) rather than raise an unalignable-boolean error.
        cd_reg.data.iloc[5, cd_reg.data.columns.get_loc("power")] = np.nan
        cd_reg.data_filtered = cd_reg.data.copy()
        with pytest.warns(UserWarning, match="missing values"):
            kept = FilterRegression(n_std=2)._execute(cd_reg)
        assert 5 not in kept  # NaN row removed
        assert cd_reg.summary_ix[-1][1] == "filter_missing"

    def test_explanation(self, cd_reg):
        f = FilterRegression(n_std=2)
        f.run(cd_reg)
        assert "residual" in f.explanation.lower()
        assert "2" in f.explanation
        assert f.explanation.endswith("were removed.")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterRegression -v`
Expected: FAIL — `ImportError: cannot import name 'FilterRegression'`.

- [ ] **Step 3: Implement `FilterRegression`** (append to `filters.py`, after the last filter class):

```python
class FilterRegression(BaseFilter):
    """Remove intervals whose regression residuals are statistical outliers.

    Fits the CapData regression formula (``capdata.regression_formula``) to the
    regression columns and keeps only rows whose residual is within ``n_std``
    residual standard deviations. The fitted statsmodels result is exposed on
    ``self.regression_model`` after ``_execute`` so callers (e.g.
    ``CapData.fit_regression``) can print its summary.
    """

    _legacy_name = "fit_regression"
    _explanation_template = (
        "Intervals with regression residuals beyond {n_std} standard "
        "deviations were removed."
    )

    n_std = param.Number(
        default=2,
        doc="Residual cutoff in standard deviations; rows beyond this are removed.",
    )

    def _execute(self, capdata):
        df = capdata.get_reg_cols()
        if df.isna().any().any():
            warnings.warn(
                "Regression columns contain missing values. Calling "
                "filter_missing before fitting the regression."
            )
            capdata.filter_missing()
            df = capdata.get_reg_cols()
        reg = fit_model(df, fml=capdata.regression_formula)
        self.regression_model = reg
        threshold = self.n_std * reg.scale**0.5
        return reg.resid[reg.resid.abs() < threshold].index
```

> Notes:
> - NaN handling mirrors `FilterOutliers`: warn + `capdata.filter_missing()` (records its own step) before fitting. `run()` snapshots `ix_before`/`pts_before` *after* `_execute`, so the NaN drop is attributed to the nested `filter_missing` step, not to `FilterRegression` (same as `FilterOutliers`).
> - The kept index comes from `reg.resid[...]` (self-aligned), not `df[...]` — this avoids both the unalignable-boolean error and pandas' "Boolean Series key will be reindexed" warning (verified empirically).
> - `reg.resid.abs()` / `reg.scale ** 0.5` avoid a numpy import in `filters.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterRegression -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add FilterRegression class (residual-outlier filter)"
```

---

### Task 3: Convert `fit_regression(filter=True)` to delegate to `FilterRegression`

**Files:** Modify `src/captest/capdata.py`, `tests/test_filter_classes.py`.

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFitRegressionWrapper:
    def test_filter_true_records_filterregression_step(self, cd_reg):
        cd_reg.fit_regression(filter=True, summary=False)
        assert len(cd_reg.filters) == 1
        assert isinstance(cd_reg.filters[0], FilterRegression)

    def test_filter_true_not_inplace_records_no_step(self, cd_reg):
        n_before = cd_reg.data_filtered.shape[0]
        out = cd_reg.fit_regression(filter=True, inplace=False, summary=False)
        assert cd_reg.filters == []
        assert cd_reg.data_filtered.shape[0] == n_before
        assert out.shape[0] < n_before  # the outlier is removed

    def test_filter_false_stores_regression_results(self, cd_reg):
        cd_reg.fit_regression(filter=False, summary=False)
        assert cd_reg.regression_results is not None
        assert cd_reg.filters == []  # plain fit records no filter step
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFitRegressionWrapper -v`
Expected: FAIL — `filter=True` does not yet record a `FilterRegression` step (legacy assigns `data_filtered`).

- [ ] **Step 3: Add `FilterRegression` to the capdata import**

Extend the `from captest.filters import (...)` block in `capdata.py` to include `FilterRegression` (after `FilterPvsyst`, before `FilterSensors`).

- [ ] **Step 4: Replace the `fit_regression` body and drop `@update_summary`**

Replace the whole `@update_summary`-decorated `fit_regression` method with (note: no decorator):

```python
    def fit_regression(self, filter=False, inplace=True, summary=True):
        """
        Perform a regression with statsmodels on the filtered data.

        Parameters
        ----------
        filter : bool, default False
            When True, removes timestamps whose residuals exceed two standard
            deviations (recorded as a FilterRegression step). When False, just
            fits ordinary least squares and stores the result in
            ``regression_results``.
        inplace : bool, default True
            With filter=True: if True, record the filter step and update
            data_filtered; if False, return the filtered DataFrame without
            recording a step.
        summary : bool, default True
            Set False to suppress printing the regression summary.

        Returns
        -------
        DataFrame
            Filtered DataFrame when filter=True and inplace=False.
        """
        if filter:
            print("NOTE: Regression used to filter outlying points.\n\n")
            flt = FilterRegression(n_std=2)
            if inplace:
                flt.run(self)
                if summary:
                    print(flt.regression_model.summary())
            else:
                kept = flt._execute(self)
                if summary:
                    print(flt.regression_model.summary())
                return self.data_filtered.loc[kept, :]
        else:
            df = self.get_reg_cols()
            reg = fit_model(df, fml=self.regression_formula)
            if summary:
                print(reg.summary())
            self.regression_results = reg
```

- [ ] **Step 5: Run the wrapper tests + existing fit_regression usage**

Run: `uv run pytest tests/test_filter_classes.py::TestFitRegressionWrapper -v`
Expected: PASS.
Run: `uv run pytest tests/test_captest.py -q`
Expected: PASS (the `fit_regression(summary=False)` calls in `test_captest.py` use `filter=False` — the plain-fit path is unchanged and still stores `regression_results`).

- [ ] **Step 6: Full suite + lint**

Run: `just test-wo-warnings`
Expected: PASS.
Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 7: Grep gate — confirm no stray fit_model/fit_regression repoint needed**

Run: `grep -rnE "pvc\.fit_model|capdata\.fit_model" tests/`
Expected: the three existing refs (test_CapData.py:134/159/2176) — all still resolve via the capdata re-import, so they PASS; no edits needed. Confirm by the green suite in Step 6.

- [ ] **Step 8: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: route fit_regression(filter=True) through FilterRegression"
```

---

### Task 4: Update the chunk-4 plan (remove its now-obsolete Task 4)

**Files:** Modify `docs/superpowers/plans/2026-06-04-data-filtered-property.md`.

- [ ] **Step 1:** Delete "Task 4: Fix `fit_regression(filter=True)` to record through `filters`" from the chunk-4 plan (the `keep_rows`/`FilterCustom` approach), since `fit_regression` now already routes through `run()`. Renumber the subsequent tasks (old Task 5 → 4, etc.). Update the Self-Review's `fit_regression` bullet to note it's handled by the FilterRegression plan instead.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-06-04-data-filtered-property.md
git commit -m "docs: drop chunk-4 fit_regression task (handled by FilterRegression plan)"
```

---

## Self-Review

**1. Spec coverage:** `FitRegression`'s filtering half realized as a `BaseFilter` (`FilterRegression`) that records a summary step + explanation → Tasks 2-3. The `fit_model` relocation follows the established one-way-import helper-move pattern (`filter_irr`, `wrap_year_end`) → Task 1. ✓

**2. Placeholder scan:** No TBDs. Every step shows complete code; the tests use residual-band correctness assertions (kept rows within threshold, removed rows beyond) that pin the exact semantics without fragile row counts.

**3. Type/name consistency:** `FilterRegression`, `n_std`, `regression_model` (runtime attr), `fit_model`, `_legacy_name="fit_regression"`, and `_explanation_template` are consistent across `filters.py`, `capdata.py`, and the tests. `get_reg_cols()` (default `filtered_data=True`) yields a frame whose index is a subset of `data_filtered.index`, so `_execute`'s returned index is a valid kept set.

**No repoint needed (verified):** `fit_model` moves to `filters.py` but `capdata.py` re-imports it (required by `predict_capacities`), so `pvc.fit_model` test references keep resolving — same pattern as `wrap_year_end`.

**Behavioral preservation:** `filter=False` path unchanged (stores `regression_results`, no step); `filter=True` preserves "no `regression_results` stored" and the two `print` statements; default `n_std=2` matches the legacy hardcoded cutoff.

**Cross-plan effect:** Task 4 removes the chunk-4 plan's `fit_regression`/`FilterCustom(keep_rows)` task, simplifying the property flip.
