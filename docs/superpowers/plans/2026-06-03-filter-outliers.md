# FilterOutliers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `filter_outliers` to `FilterOutliers` — the spec's `**kwargs`-passthrough case for an external library (sklearn's `EllipticEnvelope`). Preserves the auto-NaN-handling side effect (calling `filter_missing` first) that the existing tests pin.

**Architecture:** `FilterOutliers` declares `envelope_kwargs` as a `param.Dict` (resolved at run time by merging defaults `{support_fraction: 0.9, contamination: 0.04}` with user overrides; resolved dict stored on `self.envelope_kwargs_resolved` for display). `_execute` reads `capdata.floc[["poa", "power"]]`, warns and no-ops on more than 2 columns (the "aggregate first" path), warns and calls `capdata.filter_missing(...)` when NaN is present (which records a separate `FilterMissing` step — matches the existing-test invariant), fits an `EllipticEnvelope` on the cleaned 2-D matrix, and returns the index of rows where `predict == 1`. `args_repr` renders `EllipticEnvelope(k=v, ...)` using the resolved dict; the explanation reuses that string.

**Tech Stack:** Python, `param`, pandas, scikit-learn (`sk_cv.EllipticEnvelope`), pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes", "Thin Wrapper Methods". `FilterOutliers` is one of the spec's "kwargs/callable filters" that needs explicit modeling beyond the FilterIrr template.

**Sequencing:** Execute *after* `2026-05-24-filter-irr-example.md`. Independent of FilterTime / FilterClearsky / FilterCustom / the straightforward batch. Order vs. FilterMissing (in the straightforward batch): doesn't matter — `FilterOutliers._execute` calls `capdata.filter_missing(...)` which routes through either the legacy decorator or the future `FilterMissing` thin wrapper identically.

## Key design decisions (flag if you disagree before implementing)

1. **`envelope_kwargs` as `param.Dict(default=None, allow_None=True)`.** The user's overrides are stored as the param value; defaults are *not* baked in so `args_repr` can distinguish "user said `contamination=0.10`" from "default `0.04` applied at run time." `_execute` merges `{"support_fraction": 0.9, "contamination": 0.04}` with `self.envelope_kwargs or {}` and stores the resolved dict on `self.envelope_kwargs_resolved` (runtime attr, like `ref_val_resolved` on FilterIrr). Same Option-B pattern: param keeps the user intent for YAML; the resolved view is for display.
2. **NaN handling preserves the auto-`filter_missing` side effect.** `tests/test_CapData.py::TestFilterOutliersAndPower::test_filter_outliers_nan_records_filter_missing_in_summary` pins the recorded ordering (`filter_missing` then `filter_outliers` in `summary_ix`). `_execute` calls `capdata.filter_missing(columns=XandY.columns.tolist())` when NaN is detected, which records a separate step via `FilterMissing` (or the legacy `@update_summary` decorator until `FilterMissing` is converted — the recorded shape is identical either way). This carries the pre-existing wart that `FilterOutliers`' own `pts_removed` over-counts (it includes the NaN drop the nested call already attributed to `filter_missing`); the wart exists in the legacy method too and no test pins it.
3. **`>2` columns is a warn-and-no-op, not an error.** The existing `test_not_aggregated` pins the `UserWarning`. `_execute` returns `capdata.data_filtered.index` in this branch (no rows removed) so `run()` doesn't blow up on `len(None)`. The recorded step has `pts_removed=0` — same as the legacy method, which silently returned from `warnings.warn(...)` (which returns `None`) and left `data_filtered` untouched.
4. **`args_repr` over-rides to render `EllipticEnvelope(k=v, ...)`** using the resolved kwargs (so defaults show up in the summary, matching the "full transparency" decision from FilterIrr's `args_repr`). The `_explanation_template` reuses this via `_explanation_values()`.
5. **`inplace` kept for now.** Same transitional choice as the other complex filters.
6. **No `param.Callable` for `EllipticEnvelope`.** The estimator class is hard-coded into `_execute`. Future flexibility (any sklearn outlier detector) is out of scope; the legacy method also hard-codes the class.

---

### Task 1: Add `FilterOutliers` to `filters.py`

**Files:**
- Modify: `src/captest/filters.py` (add `import` of `sk_cv`; add `FilterOutliers`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

Extend the top-of-file `filters` import:

```python
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterCustom,
    FilterIrr,
    FilterOutliers,
    FilterSensors,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)
```

Add a `cd_pp` fixture near the other fixtures — a small CapData with a roughly-linear poa→power relationship plus three obvious outliers:

```python
@pytest.fixture
def cd_pp():
    """A CapData with poa+power columns and three injected outliers."""
    np.random.seed(0)
    n = 50
    poa = np.linspace(100.0, 1000.0, n)
    power = poa * 0.2 + np.random.normal(0, 5, n)
    # Inject obvious outliers
    power[5] = 500.0   # high
    power[20] = 0.0    # low
    power[40] = -100.0 # extreme
    cd = CapData("pp")
    cd.data = pd.DataFrame({"poa": poa, "power": power}, index=pd.RangeIndex(n))
    cd.data_filtered = cd.data.copy()
    cd.regression_cols = {"poa": "poa", "power": "power"}
    return cd
```

Append `TestFilterOutliers`:

```python
class TestFilterOutliers:
    def test_execute_removes_outliers(self, cd_pp):
        # Default contamination=0.04 on n=50 removes 2 points; verified
        # empirically that indices 5 and 40 (the two most extreme injected
        # outliers) go and index 20 stays at the default contamination.
        n_before = len(cd_pp.data_filtered)
        kept = FilterOutliers()._execute(cd_pp)
        assert len(kept) < n_before
        assert 5 not in kept
        assert 40 not in kept

    def test_execute_higher_contamination_removes_more(self, cd_pp):
        # Bumping contamination to 0.10 catches the third injected outlier too.
        f = FilterOutliers(envelope_kwargs={"contamination": 0.10})
        kept = f._execute(cd_pp)
        for ix in (5, 20, 40):
            assert ix not in kept

    def test_execute_resolves_default_kwargs(self, cd_pp):
        f = FilterOutliers()
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved == {
            "support_fraction": 0.9,
            "contamination": 0.04,
        }

    def test_execute_user_kwargs_override(self, cd_pp):
        f = FilterOutliers(envelope_kwargs={"contamination": 0.10})
        f._execute(cd_pp)
        assert f.envelope_kwargs_resolved["contamination"] == 0.10
        # default support_fraction still merged in
        assert f.envelope_kwargs_resolved["support_fraction"] == 0.9

    def test_execute_too_many_columns_warns_and_keeps_all(self, cd_pp):
        # Add a third 'poa'-mapped column so floc[["poa","power"]] is wide
        cd_pp.data["poa2"] = cd_pp.data["poa"]
        cd_pp.data_filtered = cd_pp.data.copy()
        cd_pp.column_groups = {"poa": ["poa", "poa2"], "power": ["power"]}
        n_before = len(cd_pp.data_filtered)
        f = FilterOutliers()
        with pytest.warns(UserWarning, match="aggregate_sensors"):
            kept = f._execute(cd_pp)
        assert len(kept) == n_before  # no-op

    def test_execute_nan_calls_filter_missing(self, cd_pp):
        cd_pp.data.iloc[0, cd_pp.data.columns.get_loc("poa")] = np.nan
        cd_pp.data_filtered = cd_pp.data.copy()
        with pytest.warns(UserWarning, match="missing values"):
            kept = FilterOutliers()._execute(cd_pp)
        # NaN row should not appear in kept
        assert 0 not in kept

    def test_args_repr_renders_envelope_call(self):
        args = FilterOutliers().args_repr
        # Before _execute, only the user's param (None) and custom_name appear.
        # After _execute the resolved dict is shown via the override.
        assert "EllipticEnvelope" not in args  # pre-run shows the param value
        cd = CapData("x")
        cd.data = pd.DataFrame(
            {"poa": np.linspace(100, 1000, 30), "power": np.linspace(20, 200, 30)},
            index=pd.RangeIndex(30),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"poa": "poa", "power": "power"}
        f = FilterOutliers()
        f._execute(cd)
        post = f.args_repr
        assert "EllipticEnvelope(" in post
        assert "contamination=0.04" in post
        assert "support_fraction=0.9" in post

    def test_explanation_describes_envelope(self, cd_pp):
        f = FilterOutliers()
        f.run(cd_pp)
        exp = f.explanation
        assert "EllipticEnvelope" in exp
        assert exp.endswith("were removed.")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterOutliers -v`
Expected: FAIL — `ImportError: cannot import name 'FilterOutliers'`.

- [ ] **Step 3: Implement `FilterOutliers`**

Add the sklearn import near the top of `src/captest/filters.py` (matching the existing imports):

```python
import sklearn.covariance as sk_cv
```

Append to `src/captest/filters.py`:

```python
class FilterOutliers(BaseFilter):
    """Remove statistical outliers in the (poa, power) plane via sklearn EllipticEnvelope.

    Reads ``capdata.floc[["poa", "power"]]`` (the regression-mapped columns),
    drops any NaN rows by delegating to ``capdata.filter_missing`` (which
    records its own step), fits ``sklearn.covariance.EllipticEnvelope`` on
    the cleaned 2-D matrix, and keeps the rows whose ``predict`` is 1.

    ``envelope_kwargs`` carries user overrides for ``EllipticEnvelope``;
    defaults (``support_fraction=0.9``, ``contamination=0.04``) are merged in
    at run time. The merged dict is exposed on ``envelope_kwargs_resolved``
    for display.
    """

    _legacy_name = "filter_outliers"
    _explanation_template = (
        "Statistical outliers in (poa, power) were detected via "
        "EllipticEnvelope({kwargs}) and removed."
    )
    _default_envelope_kwargs = {"support_fraction": 0.9, "contamination": 0.04}

    envelope_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Override kwargs for sklearn EllipticEnvelope. Defaults "
        "(support_fraction=0.9, contamination=0.04) are merged in at run time.",
    )

    def _execute(self, capdata):
        XandY = capdata.floc[["poa", "power"]]
        if XandY.shape[1] > 2:
            warnings.warn(
                "Too many columns. Try running aggregate_sensors before using "
                "filter_outliers."
            )
            return capdata.data_filtered.index

        if XandY.isna().any().any():
            warnings.warn(
                "Poa and/or power columns contain missing values. Calling "
                "filter_missing on poa and power columns before continuing "
                "with filter_outliers."
            )
            capdata.filter_missing(columns=XandY.columns.tolist())
            XandY = capdata.floc[["poa", "power"]]

        resolved = dict(self._default_envelope_kwargs)
        if self.envelope_kwargs:
            resolved.update(self.envelope_kwargs)
        self.envelope_kwargs_resolved = resolved

        X = XandY.values
        clf = sk_cv.EllipticEnvelope(**resolved)
        clf.fit(X)
        mask = clf.predict(X) == 1
        return capdata.data_filtered.index[mask]

    @property
    def args_repr(self):
        """Render ``EllipticEnvelope(k=v, ...)`` using the resolved kwargs."""
        resolved = getattr(self, "envelope_kwargs_resolved", None)
        if resolved is None:
            # Fall back to the base rendering before _execute resolved the dict.
            return super().args_repr
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return f"EllipticEnvelope({kw})"

    def _explanation_values(self):
        resolved = getattr(self, "envelope_kwargs_resolved", {}) or {}
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return {"kwargs": kw}
```

> Notes:
> - `args_repr` is fully overridden (the base hook expects a `key=value` mapping that would join as `envelope_kwargs={...}`; rendering as a constructor call matches the spec's "args_repr uses the function's `__name__`" treatment for FilterCustom).
> - The `>2`-columns and NaN branches both warn *before* the EllipticEnvelope call, matching the legacy method's warning order and message text.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterOutliers -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add FilterOutliers class wrapping sklearn EllipticEnvelope"
```

---

### Task 2: Convert `CapData.filter_outliers` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_outliers` method ~line 2135 pre-FilterTime; locate by anchor `def filter_outliers(self, inplace=True, **kwargs):`. Add `FilterOutliers` to the `from captest.filters import (...)` block. Drop the no-longer-used `import sklearn.covariance as sk_cv` if no other capdata method still uses it — verify by grep.)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterOutliersWrapper:
    def test_wrapper_records_filteroutliers_step(self, cd_pp):
        cd_pp.filter_outliers()
        assert len(cd_pp.filters) == 1
        assert isinstance(cd_pp.filters[0], FilterOutliers)

    def test_wrapper_passes_kwargs(self, cd_pp):
        cd_pp.filter_outliers(contamination=0.10)
        assert cd_pp.filters[0].envelope_kwargs_resolved["contamination"] == 0.10

    def test_wrapper_inplace_false_records_no_step(self, cd_pp):
        result = cd_pp.filter_outliers(inplace=False)
        assert cd_pp.filters == []
        assert result is not None
        assert result.shape[0] < cd_pp.data.shape[0]  # outliers removed
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterOutliersWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterOutliers`.

- [ ] **Step 3: Add `FilterOutliers` to the capdata import**

In `src/captest/capdata.py`, extend the import block:

```python
from captest.filters import (
    BaseSummaryStep,
    FilterCustom,
    FilterIrr,
    FilterOutliers,
    FilterSensors,
    FilterTime,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    wrap_year_end,
)
```

> After Step 4 removes the only `sk_cv` usage from `capdata.py`'s `filter_outliers` body, run:
> ```bash
> grep -n "sk_cv\|sklearn.covariance" src/captest/capdata.py
> ```
> If no hits remain, also delete the `import sklearn.covariance as sk_cv` line near the top of `capdata.py` (this is the file's only sklearn import — let `filters.py` own it). Ruff will flag it via `F401` otherwise.

- [ ] **Step 4: Replace the `filter_outliers` method body with a thin wrapper**

Locate by anchor `def filter_outliers(self, inplace=True, **kwargs):` and replace the entire decorated method with:

```python
    def filter_outliers(self, inplace=True, **kwargs):
        """
        Apply EllipticEnvelope from scikit-learn to remove outliers in (poa, power).

        Parameters
        ----------
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        **kwargs
            Forwarded to ``sklearn.covariance.EllipticEnvelope``. Defaults
            ``support_fraction=0.9`` and ``contamination=0.04`` are applied
            when not overridden.

        Notes
        -----
        When NaN values are present in poa/power, ``filter_missing`` is
        invoked first (and recorded as a separate filter step). This
        preserves the legacy auto-handling behavior.
        """
        flt = FilterOutliers(envelope_kwargs=kwargs or None)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterOutliersWrapper -v`
Expected: PASS.

- [ ] **Step 6: Run the pre-existing `filter_outliers` suite**

Run: `uv run pytest tests/test_CapData.py -k "FilterOutliersAndPower or filter_outliers" -v`
Expected: PASS — `test_not_aggregated` (warns on >2 cols), `test_filter_outliers_warns_and_succeeds_when_nans_present`, `test_filter_outliers_nan_records_filter_missing_in_summary` (the recorded-ordering invariant), plus the `filter_power`/`filter_power_percent` cases in the same class.

- [ ] **Step 7: Grep for `pvc.filter_outliers` / `capdata.filter_outliers` references**

Run:
```bash
grep -rnE "pvc\.filter_outliers|capdata\.filter_outliers" tests/ src/captest/
```
Expected: no hits.

- [ ] **Step 8: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS.

- [ ] **Step 9: Lint and format**

Run: `just lint && just fmt`
Expected: clean. Watch for `F401` on the removed `sk_cv` import.

- [ ] **Step 10: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: make CapData.filter_outliers a thin wrapper over FilterOutliers"
```

---

## Self-Review

**1. Spec coverage (this filter):**
- "Concrete Filter Classes" (sklearn-backed `EllipticEnvelope` filter) → Task 1. ✓
- "Thin Wrapper Methods" → Task 2. ✓
- The `**kwargs` passthrough pattern fits the spec's "kwargs/callable filters" tier — modeled as a single `param.Dict` rather than per-kwarg params, since user overrides are arbitrary sklearn-side. YAML serialization for `param.Dict` is straightforward; the YAML plan handles it.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result.

**3. Type/name consistency:** `FilterOutliers.envelope_kwargs` (param), `envelope_kwargs_resolved` (runtime), `_default_envelope_kwargs` (class attr), `_legacy_name`, `_explanation_template`, the `args_repr`/`_explanation_values` overrides, and the wrapper signature (`inplace`, `**kwargs`) match between `filters.py`, `capdata.py`, and the tests.

**Behavioral invariants preserved:**
- NaN auto-call to `filter_missing` (test_filter_outliers_nan_records_filter_missing_in_summary). ✓
- `>2`-cols warning + no-op (test_not_aggregated). ✓
- Default `support_fraction=0.9`, `contamination=0.04` (legacy `if "support_fraction" not in kwargs` etc.). ✓
- Existing summary recording shape (filter_missing then filter_outliers) — `run()` mirrors via `_record_legacy_summary` so both routes (legacy decorator on filter_missing, new run on FilterOutliers) end up in the same `summary_ix` list. ✓

**Pre-existing wart (not fixed):** `FilterOutliers`' own `pts_removed` over-counts when the NaN auto-call fires (it includes the NaN drop already attributed to `filter_missing`). The legacy method has the same wart; no test pins it.
