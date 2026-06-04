# FilterClearsky Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `filter_clearsky` to `FilterClearsky`. The last complex filter — `**kwargs` passthrough into pvlib's `detect_clearsky`, conditional behavior on `keep_clear`, an auto-detect path for the measured GHI column, and four distinct warn-and-no-op branches.

**Architecture:** `FilterClearsky` declares `ghi_col` (`param.String`), `keep_clear` (`param.Boolean`), and `detect_kwargs` (`param.Dict`). `_execute` runs the spec sequence: check `ghi_mod_csky` is present → resolve the measured GHI column (auto-detect from `column_groups`, average multi-column groups with a warning, or use the user's explicit `ghi_col`) → call `detect_clearsky` with merged kwargs → invert the mask if `keep_clear=False` → return the kept index. Three guard branches (`no ghi_mod_csky`, `multiple ghi categories`, `no clear periods detected`) warn and return `data_filtered.index` (recorded with `pts_removed=0`, matching the legacy method's no-op shape). `args_repr` renders `detect_clearsky(k=v, ...)`; the explanation template substitutes `{removed_kind}` ("Cloudy" or "Clear") based on `keep_clear` and embeds the resolved kwargs.

**Tech Stack:** Python, `param`, pandas, pvlib (`pvlib.clearsky.detect_clearsky`), pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes", "Thin Wrapper Methods". `FilterClearsky` is the kwargs-passthrough sibling of `FilterOutliers`, but with conditional explanation (like `FilterTime` got the override hatch for) — solved here without an override by parameterising the template via `{removed_kind}`.

**Sequencing:** Execute *after* `2026-05-24-filter-irr-example.md`. Independent of FilterTime / FilterCustom / FilterOutliers / the straightforward batch. Closes out the complex-filter tier.

## Key design decisions (flag if you disagree before implementing)

1. **`detect_clearsky` import moves to `filters.py`.** It currently lives in `capdata.py` inside the pvlib guard (kept there by the `clearsky` extraction plan because `filter_clearsky` still needed it). After this conversion, `capdata.py` no longer uses `detect_clearsky` — the entire pvlib guard becomes dead and can be removed from `capdata.py`. `filters.py` gets its own pvlib guard for `detect_clearsky`. The clearsky-modeling functions (`csky`, `pvlib_location`, etc.) stay in `captest.clearsky` as they were.

2. **`detect_kwargs` as `param.Dict(default=None, allow_None=True)`** with `_default_detect_kwargs = {"infer_limits": True}` merged in at run time and stored on `self.detect_kwargs_resolved`. Same Option-B pattern as `FilterOutliers.envelope_kwargs` — param keeps user intent for YAML, the resolved dict is for display.

3. **Conditional explanation via a `{removed_kind}` template substitution, not an `explanation` override.** `FilterTime` overrides `explanation` because its phrasing matrix is large (drop, days, test_date, start-only, end-only, …); FilterClearsky has just one boolean flip, so a single template with `_explanation_values` swapping in `"Cloudy"`/`"Clear"` is cleaner than overriding the property. Template: `"{removed_kind} intervals (detected via pvlib detect_clearsky({kwargs})) were removed."`

4. **Three warn-and-no-op branches return `capdata.data_filtered.index`.** Matches the legacy method's "warning then `return None`" behaviour (which left `data_filtered` unchanged but still recorded a step with `pts_removed=0` via `@update_summary`). The new design records a step with `pts_removed=0` too — same observable summary shape.

5. **Multi-column-ghi auto-detect averages, with a warning.** Carries the legacy method's behaviour for `column_groups["irr-ghi-…"]` having more than one entry. Pinned by `test_two_ghi_cols` in the existing suite.

6. **`inplace` kept for now.** Same transitional choice as the other complex filters.

7. **Mock-patch targets in the existing suite must update.** `tests/test_CapData.py::TestCskyFilter::test_infer_limits_default` and `test_kwargs_passed_to_detect_clearsky` use `unittest.mock.patch("captest.capdata.detect_clearsky", wraps=pvc.detect_clearsky)`. After the import moves, both references (`captest.capdata.detect_clearsky` and `pvc.detect_clearsky`) need to point at `captest.filters.detect_clearsky`. The grep step in Task 2 surfaces these explicitly.

---

### Task 1: Add `FilterClearsky` to `filters.py`

**Files:**
- Modify: `src/captest/filters.py` (add the pvlib guard + `detect_clearsky` import; add `FilterClearsky`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

Extend the top-of-file `filters` import:

```python
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterClearsky,
    FilterCustom,
    FilterIrr,
    FilterOutliers,
    FilterSensors,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)
```

Append `TestFilterClearsky` (reuses the conftest `nrel_clear_sky` fixture):

```python
class TestFilterClearsky:
    def test_execute_keeps_clear_periods(self, nrel_clear_sky):
        n_before = nrel_clear_sky.data_filtered.shape[0]
        kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) < n_before  # some intervals removed
        # data isn't mutated here — _execute returns an Index only
        assert nrel_clear_sky.data_filtered.shape[0] == n_before

    def test_execute_keep_clear_false_inverts_mask(self, nrel_clear_sky):
        # The two masks must partition the full index.
        clear_kept = FilterClearsky()._execute(nrel_clear_sky)
        cloudy_kept = FilterClearsky(keep_clear=False)._execute(nrel_clear_sky)
        full = nrel_clear_sky.data_filtered.index
        assert clear_kept.union(cloudy_kept).equals(full)
        assert clear_kept.intersection(cloudy_kept).empty

    def test_execute_resolves_default_detect_kwargs(self, nrel_clear_sky):
        f = FilterClearsky()
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved == {"infer_limits": True}

    def test_execute_user_kwargs_override(self, nrel_clear_sky):
        f = FilterClearsky(detect_kwargs={"infer_limits": False, "window_length": 30})
        f._execute(nrel_clear_sky)
        assert f.detect_kwargs_resolved["infer_limits"] is False
        assert f.detect_kwargs_resolved["window_length"] == 30

    def test_execute_no_ghi_mod_csky_warns_and_keeps_all(self, nrel_clear_sky):
        nrel_clear_sky.drop_cols("ghi_mod_csky")
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="Modeled clear sky"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_no_measured_ghi_group_warns_and_keeps_all(self, nrel_clear_sky):
        # ghi_mod_csky present (so the first guard passes) but no measured
        # GHI column group at all — must warn and no-op rather than IndexError.
        del nrel_clear_sky.column_groups["irr-ghi-"]
        n_before = nrel_clear_sky.data_filtered.shape[0]
        with pytest.warns(UserWarning, match="No measured GHI"):
            kept = FilterClearsky()._execute(nrel_clear_sky)
        assert len(kept) == n_before

    def test_execute_specify_ghi_col(self, nrel_clear_sky):
        nrel_clear_sky.data["ws 2 ghi W/m^2"] = (
            nrel_clear_sky.loc["irr-ghi-"] * 1.05
        )
        nrel_clear_sky.data_filtered = nrel_clear_sky.data.copy()
        nrel_clear_sky.column_groups["irr-ghi-"].append("ws 2 ghi W/m^2")
        f = FilterClearsky(ghi_col="ws 2 ghi W/m^2")
        kept = f._execute(nrel_clear_sky)
        assert len(kept) < nrel_clear_sky.data_filtered.shape[0]

    def test_args_repr_renders_detect_call(self, nrel_clear_sky):
        f = FilterClearsky()
        # Pre-_execute: detect_kwargs is None; args_repr falls back.
        assert "detect_clearsky" not in f.args_repr
        f._execute(nrel_clear_sky)
        assert "detect_clearsky(" in f.args_repr
        assert "infer_limits=True" in f.args_repr

    def test_explanation_default_says_cloudy(self, nrel_clear_sky):
        f = FilterClearsky()
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Cloudy intervals")
        assert "detect_clearsky" in f.explanation
        assert f.explanation.endswith("were removed.")

    def test_explanation_keep_clear_false_says_clear(self, nrel_clear_sky):
        f = FilterClearsky(keep_clear=False)
        f.run(nrel_clear_sky)
        assert f.explanation.startswith("Clear intervals")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterClearsky -v`
Expected: FAIL — `ImportError: cannot import name 'FilterClearsky'`.

- [ ] **Step 3: Add the pvlib guard and `FilterClearsky` to `filters.py`**

Near the top of `src/captest/filters.py` (after the existing imports, before the first helper function), add the pvlib guard:

```python
import importlib.util

pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn(
        "Clear sky filtering will not work without the pvlib package."
    )
```

Append `FilterClearsky` after `FilterOutliers`:

```python
class FilterClearsky(BaseFilter):
    """Remove intervals where measured GHI doesn't match modeled clear-sky GHI.

    Uses ``pvlib.clearsky.detect_clearsky`` to classify each timestamp as
    clear or cloudy. By default keeps the clear timestamps and removes
    cloudy ones; set ``keep_clear=False`` to invert.

    Requires the ``ghi_mod_csky`` column in ``capdata.data_filtered`` (added
    by ``io.load_data`` when the ``site`` argument is supplied). When
    ``ghi_col`` is None, the measured GHI column is auto-detected from
    ``column_groups`` (the single ``irr-ghi-*`` entry other than
    ``irr-ghi-clear_sky``); multi-column groups are averaged with a warning.
    """

    _legacy_name = "filter_clearsky"
    _explanation_template = (
        "{removed_kind} intervals (detected via pvlib "
        "detect_clearsky({kwargs})) were removed."
    )
    _default_detect_kwargs = {"infer_limits": True}

    ghi_col = param.String(
        default=None,
        allow_None=True,
        doc="Measured GHI column name. Auto-detected from column_groups if None.",
    )
    keep_clear = param.Boolean(
        default=True,
        doc="Keep clear intervals (True) or keep cloudy intervals (False).",
    )
    detect_kwargs = param.Dict(
        default=None,
        allow_None=True,
        doc="Override kwargs for pvlib detect_clearsky. Default "
        "infer_limits=True is merged in at run time.",
    )

    def _execute(self, capdata):
        if "ghi_mod_csky" not in capdata.data_filtered.columns:
            warnings.warn(
                "Modeled clear sky data must be available to run this filter. "
                "Use CapData load_data clear_sky option."
            )
            return capdata.data_filtered.index

        if self.ghi_col is None:
            ghi_keys = []
            for key in capdata.column_groups.keys():
                defs = key.split("-")
                if len(defs) == 1:
                    continue
                if defs[1] == "ghi":
                    ghi_keys.append(key)
            ghi_keys = [k for k in ghi_keys if k != "irr-ghi-clear_sky"]

            if not ghi_keys:
                warnings.warn(
                    "No measured GHI column group found in column_groups. "
                    "Pass column name to ghi_col."
                )
                return capdata.data_filtered.index
            if len(ghi_keys) > 1:
                warnings.warn(
                    "Too many ghi categories. Pass column name to ghi_col to "
                    "use a specific column."
                )
                return capdata.data_filtered.index

            meas_ghi = capdata.floc[ghi_keys[0]]
            if meas_ghi.shape[1] > 1:
                warnings.warn(
                    "Averaging measured GHI data. Pass column name to ghi_col "
                    "to use a specific column."
                )
            meas_ghi = meas_ghi.mean(axis=1)
        else:
            meas_ghi = capdata.data_filtered[self.ghi_col]

        resolved = dict(self._default_detect_kwargs)
        if self.detect_kwargs:
            resolved.update(self.detect_kwargs)
        self.detect_kwargs_resolved = resolved

        clear_per = detect_clearsky(
            measured=meas_ghi,
            clearsky=capdata.data_filtered["ghi_mod_csky"],
            times=meas_ghi.index,
            **resolved,
        )
        if not any(clear_per):
            warnings.warn(
                "No clear periods detected. Try adjusting detect_clearsky "
                "parameters via kwargs."
            )
            return capdata.data_filtered.index

        mask = clear_per if self.keep_clear else ~clear_per
        return capdata.data_filtered.index[mask]

    @property
    def args_repr(self):
        """Render ``detect_clearsky(k=v, ...)`` using the resolved kwargs."""
        resolved = getattr(self, "detect_kwargs_resolved", None)
        if resolved is None:
            return super().args_repr
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return f"detect_clearsky({kw})"

    def _explanation_values(self):
        resolved = getattr(self, "detect_kwargs_resolved", {}) or {}
        kw = ", ".join(f"{k}={v}" for k, v in resolved.items())
        return {
            "removed_kind": "Cloudy" if self.keep_clear else "Clear",
            "kwargs": kw,
        }
```

> Notes:
> - `args_repr` is fully overridden (same shape-mismatch reasoning as `FilterOutliers` and `FilterCustom` — a call expression doesn't fit the generic `key=value` joining).
> - The mask invert (`~clear_per` when `keep_clear=False`) preserves the legacy `keep_clear=False` semantics; the existing `test_default_drop_clear_sky` invariant (`data.index.difference(clear_ix).equals(cloudy_ix)`) flows through.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterClearsky -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add FilterClearsky class wrapping pvlib detect_clearsky"
```

---

### Task 2: Convert `CapData.filter_clearsky` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_clearsky` method ~line 2238 pre-conversion; locate by anchor `def filter_clearsky(self, ghi_col=None`). Add `FilterClearsky` to the `from captest.filters import (...)` block. Drop the pvlib guard (the only remaining `detect_clearsky` usage was inside this method's body — verify by grep — and the guard imports nothing else.)
- Modify: `tests/test_CapData.py` — two mock-patch targets in `TestCskyFilter` (`test_infer_limits_default`, `test_kwargs_passed_to_detect_clearsky`) point at `captest.capdata.detect_clearsky` and use `pvc.detect_clearsky`. Both must repoint to `captest.filters.detect_clearsky` / `filters.detect_clearsky`.
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterClearskyWrapper:
    def test_wrapper_records_filterclearsky_step(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky()
        assert len(nrel_clear_sky.filters) == 1
        assert isinstance(nrel_clear_sky.filters[0], FilterClearsky)

    def test_wrapper_passes_kwargs(self, nrel_clear_sky):
        nrel_clear_sky.filter_clearsky(infer_limits=False, window_length=30)
        resolved = nrel_clear_sky.filters[0].detect_kwargs_resolved
        assert resolved["infer_limits"] is False
        assert resolved["window_length"] == 30

    def test_wrapper_inplace_false_records_no_step(self, nrel_clear_sky):
        # Pin full immutability of data_filtered, not just row count — a
        # mutation that changes values without changing length would otherwise
        # slip through.
        original = nrel_clear_sky.data_filtered.copy()
        result = nrel_clear_sky.filter_clearsky(inplace=False)
        assert nrel_clear_sky.filters == []
        pd.testing.assert_frame_equal(nrel_clear_sky.data_filtered, original)
        assert result.shape[0] < original.shape[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterClearskyWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterClearsky`.

- [ ] **Step 3: Add `FilterClearsky` to the capdata import**

In `src/captest/capdata.py`, extend the import block:

```python
from captest.filters import (
    BaseSummaryStep,
    FilterClearsky,
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

- [ ] **Step 4: Replace the `filter_clearsky` method body with a thin wrapper**

Locate the current method by anchor `def filter_clearsky(self, ghi_col=None, inplace=True, keep_clear=True, **kwargs):` (with `@update_summary` above) and replace the whole decorated method with:

```python
    def filter_clearsky(self, ghi_col=None, inplace=True, keep_clear=True, **kwargs):
        """Remove unstable-irradiance intervals using pvlib detect_clearsky.

        Parameters
        ----------
        ghi_col : str, default None
            Measured GHI column. Auto-detected from ``column_groups`` if None.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        keep_clear : bool, default True
            Keep clear intervals (True) or keep cloudy intervals (False).
        **kwargs
            Forwarded to pvlib ``detect_clearsky``. Default
            ``infer_limits=True`` is applied when not overridden.
        """
        flt = FilterClearsky(
            ghi_col=ghi_col,
            keep_clear=keep_clear,
            detect_kwargs=kwargs or None,
        )
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5: Drop the pvlib guard from `capdata.py`**

Locate the existing pvlib guard near the top of `src/captest/capdata.py`:

```python
pvlib_spec = importlib.util.find_spec("pvlib")
if pvlib_spec is not None:
    from pvlib.clearsky import detect_clearsky
else:
    warnings.warn("Clear sky functions will not work without the pvlib package.")
```

Confirm it's now dead by grepping for `detect_clearsky` in `capdata.py`:

```bash
grep -n "detect_clearsky" src/captest/capdata.py
```

If only the import inside the guard appears (no callers), delete the entire guard block (4 lines + the `pvlib_spec = ...` line). If `find_spec` is no longer used after the deletion, drop the `import importlib.util` import too. Ruff `F401` will flag any leftovers.

- [ ] **Step 6: Repoint the two mock-patch targets in `tests/test_CapData.py`**

In `TestCskyFilter::test_infer_limits_default` and `test_kwargs_passed_to_detect_clearsky`, change the mock target and the `wraps=` reference from `captest.capdata.detect_clearsky`/`pvc.detect_clearsky` to `captest.filters.detect_clearsky`/`filters.detect_clearsky`. Add `from captest import filters` near the top of `test_CapData.py` if it isn't already there (it isn't as of writing).

- [ ] **Step 7: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterClearskyWrapper -v`
Expected: PASS.

- [ ] **Step 8: Run the pre-existing `filter_clearsky` suite**

Run: `uv run pytest tests/test_CapData.py::TestCskyFilter -v`
Expected: PASS (8 tests) — including the two re-pointed mock tests.

- [ ] **Step 9: Grep for any other `pvc.filter_clearsky` / `capdata.filter_clearsky` / `detect_clearsky` references**

Run:
```bash
grep -rnE "pvc\.(filter_clearsky|detect_clearsky)|capdata\.(filter_clearsky|detect_clearsky)" tests/ src/captest/ docs/examples/
```
Expected: only the two `TestCskyFilter` tests touched in Step 6 (now pointing at `filters`), plus any legitimate `capdata.filter_clearsky` re-export of the **method** (not the import). Treat any stale `capdata.detect_clearsky` hit as a repoint to `captest.filters.detect_clearsky` per the no-shim policy.

- [ ] **Step 10: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS.

- [ ] **Step 11: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py tests/test_CapData.py
git commit -m "refactor: make CapData.filter_clearsky a thin wrapper over FilterClearsky"
```

---

## Self-Review

**1. Spec coverage (this filter):**
- "Concrete Filter Classes" (pvlib-backed `detect_clearsky` filter with auto-detected measured GHI column) → Task 1. ✓
- "Thin Wrapper Methods" → Task 2. ✓
- Conditional explanation handled by `{removed_kind}` template substitution rather than an `explanation` override — extends the override-hatch design with a lighter-weight alternative for single-flip filters.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result.

**3. Type/name consistency:** `FilterClearsky` params (`ghi_col`, `keep_clear`, `detect_kwargs`), the runtime attr `detect_kwargs_resolved`, the class attrs (`_legacy_name`, `_explanation_template`, `_default_detect_kwargs`), the overrides (`_execute`, `args_repr`, `_explanation_values`), and the wrapper signature (`ghi_col`, `inplace`, `keep_clear`, `**kwargs`) match across `filters.py`, `capdata.py`, and the tests.

**Behavioral invariants preserved (from existing TestCskyFilter):**
- Default `keep_clear=True` removes intervals (`test_default`). ✓
- `keep_clear=False` returns the complementary index (`test_default_drop_clear_sky`). ✓ — covered explicitly by `test_execute_keep_clear_false_inverts_mask` at the class level.
- Multi-column GHI warns (`test_two_ghi_cols`). ✓
- Multiple GHI categories warns (`test_mult_ghi_categories`). ✓
- Missing `ghi_mod_csky` warns (`test_no_clear_ghi`). ✓
- `ghi_col` override works (`test_specify_ghi_col`). ✓
- `infer_limits=True` default (`test_infer_limits_default`) — preserved via `_default_detect_kwargs`. ✓ Mock-patch target updates to `captest.filters.detect_clearsky`.
- `infer_limits=False` + `window_length` kwargs pass through (`test_kwargs_passed_to_detect_clearsky`) — preserved via `**kwargs` → `detect_kwargs` merge. ✓ Same mock-patch update.

**Closes the complex-filter tier.** After this plan executes, all five complex filters (`FilterIrr`, `FilterSensors`, `FilterTime`, `FilterCustom`, `FilterOutliers`, `FilterClearsky`) are class-based; only the 6-straightforward batch remains before chunk 4 (the `data_filtered` property flip) is unblocked.
