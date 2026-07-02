# Sensors `method` Selector Implementation Plan (chunk 5 of 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the sensor-disagreement comparison choice into a single GUI-renderable `method` parameter on the `Sensors` filter (built-ins `'percent_diff'` / `'abs_diff'`, plus an assignable custom callable), rename its threshold dict `perc_diff` → `thresholds`, and add a `CapData.filter_sensors_abs_diff` convenience wrapper. This replaces the awkward `filter_sensors(..., row_filter=abs_diff_from_average)` invocation.

**Architecture:** `Sensors.method` becomes a `param.Selector(objects=['percent_diff', 'abs_diff'], check_on_set=False)` that also accepts a callable (the "custom" third option, replacing the separate `row_filter` kwarg). A `_BUILTIN_COMPARISONS` dict maps the string names to `check_all_perc_diff_comb` / `abs_diff_from_average`; `_resolve_comparison()` returns the callable to hand to `sensor_filter`. Serialization encodes `method` as its string, or as a `module:qualname` for a custom callable. `CapData.filter_sensors` and the new `filter_sensors_abs_diff` are thin wrappers.

**This is an intentional BREAKING change.** Back-compat for the old `Sensors(perc_diff=, row_filter=)` / `filter_sensors(perc_diff=, row_filter=)` API is explicitly not required (pre-1.0 branch, per the spec). Because leaving the suite green requires the class change, both wrappers, and all call-site/test migrations to land together, this chunk is a **single atomic task/commit**.

**Tech Stack:** Python 3.12, `param`, `pandas`, `pytest`, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-28-filter-classes-from-custom-functions-design.md` (section "`Sensors` enhancement").

## Global Constraints

- Line length: 88 characters. `src/captest/filters.py` and `src/captest/capdata.py`
  are NOT E501-exempt; `tests/test_CapData.py` and `tests/test_filter_classes.py`
  ARE E501-exempt (long lines allowed there).
- Docstrings: NumPy-style for public classes/methods.
- Naming: `snake_case`, `PascalCase` classes, `UPPER_CASE`/leading-underscore
  constants.
- No backward-compatibility shims.
- Run lint/format with `just lint <files>` and `just fmt`; tests with
  `uv run pytest`. `just fmt` / pre-commit `ruff --fix` will auto-remove imports
  that become unused — that is expected here (see Step 4's import note).
- Row-filter callable signature is `func(series, threshold) -> bool`
  (`check_all_perc_diff_comb`, `abs_diff_from_average` both match).

---

### Task 1 (only task): `Sensors` method selector + `thresholds` + `filter_sensors_abs_diff`, with full migration

**Files:**
- Modify: `src/captest/filters.py` — rewrite the `Sensors` class body.
- Modify: `src/captest/capdata.py` — rewrite `filter_sensors`, add `filter_sensors_abs_diff`, drop the now-unused `check_all_perc_diff_comb` import.
- Modify: `tests/test_filter_classes.py` — migrate `TestFilterSensors` + the two serialization tests; add method/callable coverage.
- Modify: `tests/test_CapData.py` — migrate `TestFilterSensors`, `TestFilterSensorsWithAbsDiffFromAverage`, and `test_check_all_perc_diff_comb`.
- Modify: `tests/test_filter_helpers_move.py` — drop the `capdata.check_all_perc_diff_comb` re-export assertion.

**Interfaces:**
- Consumes: `check_all_perc_diff_comb`, `abs_diff_from_average`, `sensor_filter` (all already in `filters.py`); `util.callable_to_qualname` / `util.callable_from_qualname`.
- Produces:
  - `filters.Sensors(method='percent_diff', thresholds=None, custom_name=None)` — `method` is a Selector over `['percent_diff', 'abs_diff']` (also accepts a callable); `thresholds` is the per-group dict. `_execute` returns the kept `pandas.Index`. Registered under `"Sensors"` (unchanged).
  - `CapData.filter_sensors(thresholds=None, method='percent_diff', custom_name=None)`.
  - `CapData.filter_sensors_abs_diff(thresholds, custom_name=None)` → `Sensors(method='abs_diff', thresholds=thresholds)`.

- [ ] **Step 1: Migrate + add the failing tests**

**1a. `tests/test_filter_classes.py` — replace the whole `class TestFilterSensors:` body** (the class currently spanning from `class TestFilterSensors:` through `test_explanation_before_run_returns_none`, i.e. the methods `test_execute_default_perc_diff_resolves`, `test_execute_explicit_row_filter_drops_outliers`, `test_row_filter_defaults_to_check_all_perc_diff_comb`, `test_args_repr_renders_row_filter_by_name`, `test_explanation_names_group_and_row_filter`, `test_execute_empty_perc_diff_raises`, `test_explanation_before_run_returns_none`) with:

```python
class TestFilterSensors:
    def test_execute_default_thresholds_resolves(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors()
        kept = f._execute(capdata_irr)
        # tightly-clustered random data (876-900) is within the 5% default,
        # so no rows are removed
        assert list(kept) == list(capdata_irr.data_filtered.index)
        assert f.thresholds_resolved == {"poa": 0.05}

    def test_method_defaults_to_percent_diff(self):
        assert Sensors().method == "percent_diff"

    def test_execute_abs_diff_method_drops_outliers(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        f = Sensors(method="abs_diff", thresholds={"poa": 25})
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_execute_custom_callable_method(self, capdata_irr):
        capdata_irr.data.iloc[0, 2] = 926
        capdata_irr.data.iloc[3, 0] = 850
        f = Sensors(method=abs_diff_from_average, thresholds={"poa": 25})
        assert f._resolve_comparison() is abs_diff_from_average
        kept = f._execute(capdata_irr)
        assert len(kept) == capdata_irr.data.shape[0] - 2

    def test_execute_abs_diff_without_thresholds_raises(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors(method="abs_diff")
        with pytest.raises(ValueError, match="thresholds is required"):
            f._execute(capdata_irr)

    def test_execute_empty_thresholds_raises(self, capdata_irr):
        f = Sensors(thresholds={})
        with pytest.raises(ValueError, match="must not be empty"):
            f._execute(capdata_irr)

    def test_args_repr_renders_method_name(self):
        f = Sensors(thresholds={"poa": 0.05})
        args = f.args_repr
        assert "method=percent_diff" in args
        assert "<function" not in args

    def test_explanation_names_group_and_method(self, capdata_irr):
        capdata_irr.regression_cols = {"poa": "poa"}
        f = Sensors()
        f.run(capdata_irr)
        exp = f.explanation
        assert "poa" in exp
        assert "percent_diff" in exp
        assert exp.endswith("were removed.")

    def test_explanation_before_run_returns_none(self):
        # explanation is post-run; reading it before run() must not raise
        assert Sensors().explanation is None
        # Irradiance has the same property by virtue of the base-class guard
        assert Irradiance(low=0, high=1).explanation is None
```

**1b. `tests/test_filter_classes.py` — replace `test_filter_sensors_row_filter_roundtrips`** with a `method`-based version (add a custom-callable round-trip):

```python
    def test_filter_sensors_method_roundtrips(self):
        cfg = Sensors(thresholds={"irr-poa-": 0.05}).to_config()
        assert cfg["method"] == "percent_diff"
        step = step_from_config(cfg)
        assert step.method == "percent_diff"
        assert step.thresholds == {"irr-poa-": 0.05}

    def test_filter_sensors_custom_callable_roundtrips(self):
        cfg = Sensors(
            method=abs_diff_from_average, thresholds={"poa": 25}
        ).to_config()
        assert cfg["method"] == "captest.filters:abs_diff_from_average"
        step = step_from_config(cfg)
        assert step.method is abs_diff_from_average
        assert step.thresholds == {"poa": 25}
```

**1c. `tests/test_filter_classes.py` — in `test_from_config_direct_tolerates_type_key`**, replace the `sensors = ...` block:

```python
        sensors = Sensors.from_config(Sensors(perc_diff={"irr-poa-": 0.05}).to_config())
        assert sensors.perc_diff == {"irr-poa-": 0.05}
        assert sensors.row_filter is check_all_perc_diff_comb
```

with:

```python
        sensors = Sensors.from_config(
            Sensors(thresholds={"irr-poa-": 0.05}).to_config()
        )
        assert sensors.thresholds == {"irr-poa-": 0.05}
        assert sensors.method == "percent_diff"
```

**1d. `tests/test_CapData.py` — `class TestFilterSensors`**: change the two `perc_diff=` kwargs to `thresholds=`:
- `meas.filter_sensors(perc_diff=None)` → `meas.filter_sensors(thresholds=None)`
- both `perc_diff={"irr_poa_ref_cell": 0.05, "temp_amb": 0.1},` → `thresholds={"irr_poa_ref_cell": 0.05, "temp_amb": 0.1},`

**1e. `tests/test_CapData.py` — `class TestFilterSensorsWithAbsDiffFromAverage`**: replace both call sites:
- `capdata_irr.filter_sensors(perc_diff={"poa": 25}, row_filter=filters.abs_diff_from_average)` → `capdata_irr.filter_sensors_abs_diff({"poa": 25})` (both `test_does_not_drop_rows_when_no_outliers` and `test_drops_rows_with_outliers`).

**1f. `tests/test_CapData.py` — `test_check_all_perc_diff_comb`**: change the three `pvc.check_all_perc_diff_comb(...)` calls to `filters.check_all_perc_diff_comb(...)` (the module is already imported as `from captest import filters`).

**1g. `tests/test_filter_helpers_move.py` — `test_capdata_still_exposes_helpers`**: remove the line `assert callable(capdata.check_all_perc_diff_comb)` (capdata no longer uses that helper internally; it still uses `filter_irr`/`filter_grps`).

- [ ] **Step 2: Run the migrated tests to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py -k "FilterSensors or roundtrip or tolerates_type_key" tests/test_CapData.py -k "FilterSensors or check_all_perc_diff_comb" -v`
Expected: FAIL — e.g. `AttributeError` on `thresholds_resolved`/`method`, `TypeError` for unexpected `thresholds`/`method` kwargs, and `AttributeError` for missing `filter_sensors_abs_diff`. (Collection may also error until the source is updated — that counts as the expected RED.)

- [ ] **Step 3: Rewrite the `Sensors` class in `src/captest/filters.py`**

Replace the entire existing `class Sensors(BaseFilter):` (from its `class` line through the end of its `from_config` classmethod) with:

```python
class Sensors(BaseFilter):
    """Drop rows where redundant sensors in a group disagree beyond a threshold.

    For each sensor group named in ``thresholds``, a row-wise comparison
    (selected by ``method``) is applied across that group's columns; rows
    flagged inconsistent are removed. Ignores columns generated by
    ``agg_sensors`` by operating on the pre-aggregation columns when present.
    """

    _explanation_template = (
        "Rows with inconsistent readings within sensor group(s) {groups} "
        "(compared using {method}) were removed."
    )

    # String name -> row-filter callable. A callable assigned to ``method``
    # directly bypasses this table (the custom third option).
    _BUILTIN_COMPARISONS = {
        "percent_diff": check_all_perc_diff_comb,
        "abs_diff": abs_diff_from_average,
    }

    method = param.Selector(
        default="percent_diff",
        objects=["percent_diff", "abs_diff"],
        check_on_set=False,
        doc="Sensor-comparison method: 'percent_diff' (pairwise percent "
        "difference) or 'abs_diff' (absolute difference from the group "
        "average). A callable with signature func(series, threshold) -> bool "
        "may also be assigned for a custom comparison.",
    )
    thresholds = param.Dict(
        default=None,
        allow_None=True,
        doc="Map of sensor-group key -> threshold (e.g. {'irr-poa-': 0.05}). "
        "Values are decimal fractions for 'percent_diff' or absolute units "
        "for 'abs_diff'. None defaults to {<poa group>: 0.05} for "
        "'percent_diff'; other methods require an explicit dict.",
    )

    def _resolve_comparison(self):
        if callable(self.method):
            return self.method
        return self._BUILTIN_COMPARISONS[self.method]

    def _method_label(self):
        if callable(self.method):
            return self.method.__name__
        return self.method

    def _execute(self, capdata):
        if capdata.pre_agg_cols is not None:
            df = capdata.data_filtered[capdata.pre_agg_cols]
            trans = capdata.pre_agg_trans
            regression_cols = capdata.pre_agg_reg_trans
        else:
            df = capdata.data_filtered
            trans = capdata.column_groups
            regression_cols = capdata.regression_cols

        thresholds = self.thresholds
        if thresholds is None:
            if self.method == "percent_diff":
                thresholds = {regression_cols["poa"]: 0.05}
            else:
                raise ValueError(
                    "thresholds is required when method is not 'percent_diff'."
                )
        if not thresholds:
            raise ValueError("thresholds must not be empty")
        self.thresholds_resolved = thresholds

        comparison = self._resolve_comparison()
        index = None
        for key, threshold in thresholds.items():
            sensors_df = df[trans[key]]
            next_index = sensor_filter(sensors_df, threshold, row_filter=comparison)
            index = next_index if index is None else index.intersection(next_index)
        return index

    def _args_for_repr(self):
        vals = dict(self.param.values())
        vals["method"] = self._method_label()
        resolved = getattr(self, "thresholds_resolved", None)
        if resolved is not None:
            vals["thresholds"] = resolved
        return vals

    def _explanation_values(self):
        return {
            "groups": ", ".join(self.thresholds_resolved),
            "method": self._method_label(),
        }

    def to_config(self):
        config = super().to_config()
        if callable(self.method):
            config["method"] = util.callable_to_qualname(self.method)
        else:
            config["method"] = self.method
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("type", None)
        method = config.get("method")
        # A qualname-encoded custom callable contains ':'; a built-in name
        # ('percent_diff'/'abs_diff') is left as the plain string.
        if isinstance(method, str) and ":" in method:
            config["method"] = util.callable_from_qualname(method)
        return cls(**config)
```

- [ ] **Step 4: Update `src/captest/capdata.py` — wrappers and import**

Replace the existing `filter_sensors` method (signature `def filter_sensors(self, perc_diff=None, row_filter=check_all_perc_diff_comb, custom_name=None):` through its `flt.run(self)`) with the two wrappers below:

```python
    def filter_sensors(self, thresholds=None, method="percent_diff", custom_name=None):
        """Drop suspicious measurements by comparing readings across sensors.

        For each sensor group in ``thresholds``, the ``method`` comparison is
        applied row-by-row across that group's columns; rows whose sensors
        disagree beyond the threshold are removed. Ignores columns generated
        by ``agg_sensors``.

        Parameters
        ----------
        thresholds : dict, default None
            Map of sensor-group key -> threshold, e.g.
            ``{'irr-poa-': 0.05}``. Values are decimal fractions for
            ``'percent_diff'`` or absolute units for ``'abs_diff'``. None
            defaults to the POA group at a 5% percent difference (only valid
            for ``method='percent_diff'``).
        method : str or callable, default 'percent_diff'
            ``'percent_diff'`` (pairwise percent difference) or ``'abs_diff'``
            (absolute difference from the group average). A callable with
            signature ``func(series, threshold) -> bool`` may be passed for a
            custom comparison.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Sensors(
            method=method, thresholds=thresholds, custom_name=custom_name
        )
        flt.run(self)

    def filter_sensors_abs_diff(self, thresholds, custom_name=None):
        """Drop rows where a sensor deviates from its group average by too much.

        Convenience wrapper for :meth:`filter_sensors` with
        ``method='abs_diff'``: each sensor must be within ``threshold``
        (absolute units, e.g. W/m^2) of the average of the other sensors in
        its group.

        Parameters
        ----------
        thresholds : dict
            Map of sensor-group key -> absolute threshold, e.g.
            ``{'irr-poa-': 25}``.
        custom_name : str, default None
            Optional display label for the recorded filter step.
        """
        flt = Sensors(
            method="abs_diff", thresholds=thresholds, custom_name=custom_name
        )
        flt.run(self)
```

Then remove the now-unused `check_all_perc_diff_comb,` line from the `from captest.filters import (...)` block in `capdata.py` (it was only referenced by the old `filter_sensors` default). If `just fmt`/pre-commit's `ruff --fix` removes it automatically, that is fine — just confirm it is gone.

- [ ] **Step 5: Run the migrated tests to verify they pass**

Run: `uv run pytest tests/test_filter_classes.py -k "FilterSensors or roundtrip or tolerates_type_key" tests/test_CapData.py -k "FilterSensors or check_all_perc_diff_comb" -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `uv run pytest tests/test_filter_classes.py tests/test_CapData.py tests/test_filter_helpers_move.py tests/test_util.py -q`
Expected: PASS. (`test_util.py` exercises `callable_to_qualname` on `check_all_perc_diff_comb` from `filters` — unaffected. `test_filter_helpers_move.py` must pass with the dropped assertion.)

Then the complete suite:

Run: `uv run pytest -q`
Expected: PASS (no regressions anywhere, including `test_captest.py`).

- [ ] **Step 7: Lint, format, commit**

```bash
just lint src/captest/filters.py src/captest/capdata.py tests/test_filter_classes.py tests/test_CapData.py tests/test_filter_helpers_move.py
just fmt
git add src/captest/filters.py src/captest/capdata.py tests/test_filter_classes.py tests/test_CapData.py tests/test_filter_helpers_move.py
git commit -m "feat!: unify Sensors comparison into a method selector, add filter_sensors_abs_diff

BREAKING: Sensors(perc_diff=, row_filter=) -> Sensors(thresholds=, method=);
filter_sensors(perc_diff=, row_filter=) -> filter_sensors(thresholds=, method=)."
```

---

## Self-Review

- **Spec coverage:** The spec's `Sensors` enhancement is fully implemented — `method` Selector (`percent_diff`/`abs_diff` + assignable callable via `check_on_set=False`), `perc_diff`→`thresholds` rename, `_BUILTIN_COMPARISONS`/`_resolve_comparison`, POA default only for `percent_diff` with a clear `ValueError` otherwise, single `to_config`/`from_config` encode path (string for built-ins, `module:qualname` for callables), and the `filter_sensors` (updated) + `filter_sensors_abs_diff` wrappers. All old-API call sites/tests are migrated so the suite stays green. Out of scope: user-guide/changelog docs (chunk 6).
- **Placeholder scan:** none — every step has concrete code and exact old→new edits.
- **Type consistency:** `Sensors(method, thresholds, custom_name)`, `filter_sensors(thresholds, method, custom_name)`, and `filter_sensors_abs_diff(thresholds, custom_name)` names match across class, wrappers, and tests. `thresholds_resolved`/`_resolve_comparison`/`_method_label` are used consistently by `_execute`/`_args_for_repr`/`_explanation_values`. The `from_config` `":" in method` discriminator matches the `callable_to_qualname` colon format asserted by `test_filter_sensors_custom_callable_roundtrips` (`"captest.filters:abs_diff_from_average"`). Removing the `check_all_perc_diff_comb` import from `capdata.py` is paired with migrating its only external test consumers (`test_CapData.py::test_check_all_perc_diff_comb`, `test_filter_helpers_move.py`) to `filters.check_all_perc_diff_comb`.
```
