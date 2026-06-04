# Six Straightforward Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the remaining six filters — `FilterPvsyst`, `FilterShade`, `FilterPf`, `FilterPower`, `FilterDays`, `FilterMissing` — to the class-based architecture, completing the filter migration. After this, no `@update_summary`-decorated filter remains, unblocking the `data_filtered` property flip (chunk 4).

**Architecture:** Each filter follows the now-six-times-proven recipe: a `FilterX(BaseFilter)` class in `filters.py` with `param` declarations, `_legacy_name`, `_execute` returning the kept `Index`, and an `_explanation_template` (or override); a thin `CapData.filter_x` wrapper delegating to `FilterX(...).run(self)` (or returning `data_filtered.loc[flt._execute(self), :]` for `inplace=False`). These six are genuinely straightforward — none moves a module-level helper, so there are **no `pvc.X`/`capdata.X` test references to repoint** (verified by grep).

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Concrete Filter Classes", "Thin Wrapper Methods".

**Sequencing:** Execute *after* all complex-filter plans (done). One task per filter; one commit per filter, each leaving the suite green.

## Cross-cutting decisions

1. **No `inplace` for `FilterMissing`** — the legacy `filter_missing` has no `inplace` param (always mutates), like `filter_custom`. Its wrapper keeps that: `filter_missing(self, columns=None)` always records the step.
2. **`FilterMissing._legacy_name = "filter_missing"` is load-bearing** — `tests/test_CapData.py::TestFilterOutliersAndPower::test_filter_outliers_nan_records_filter_missing_in_summary` asserts the recorded label `"filter_missing"` appears (FilterOutliers auto-calls `capdata.filter_missing` on NaN). The wrapper must still record under that label.
3. **`filter_pf` uses `df.abs()`, not `np.abs`** — `filters.py` has no numpy import; `df.abs() >= pf` is equivalent on a DataFrame. Avoids adding an import.
4. **`filter_shade` binds `fshdbm` as a local in `_execute`** — `df.query("FShdBm>=@fshdbm")` resolves `@fshdbm` from local scope, so `_execute` must do `fshdbm = self.fshdbm` before the query.
5. **Warn-and-no-op branches return `capdata.data_filtered.index`** — preserves the legacy "warn then leave data unchanged, record pts_removed=0" shape (FilterPvsyst missing-column, FilterPower bad-`columns`).
6. **Two latent NameError guards added** — `filter_pf` raised `NameError` if no `pf` column group existed (`selection` unbound); the new `FilterPf._execute` warns and no-ops instead. No test covered this; documented as a defensive improvement.
7. **`inplace` kept** on the other five wrappers, same transitional choice as the converted filters.

---

### Task 1: FilterPvsyst

**Files:** `src/captest/filters.py` (add class), `src/captest/capdata.py` (import + wrapper), `tests/test_filter_classes.py` (tests).

- [ ] **Step 1: Tests** — add to the `filters` import: `FilterPvsyst`. Append:

```python
class TestFilterPvsyst:
    def _cd(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame(
            {
                "IL Pmin": [0.0, 1.0, 0.0, 2.0],
                "power": [10.0, 20.0, 30.0, 40.0],
            },
            index=pd.RangeIndex(4),
        )
        cd.data_filtered = cd.data.copy()
        return cd

    def test_execute_drops_positive_rows(self):
        cd = self._cd()
        kept = FilterPvsyst()._execute(cd)
        assert list(kept) == [0, 2]  # rows where IL Pmin > 0 removed

    def test_execute_underscored_column_names(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame(
            {"IL_Pmax": [0.0, 5.0, 0.0]}, index=pd.RangeIndex(3)
        )
        cd.data_filtered = cd.data.copy()
        assert list(FilterPvsyst()._execute(cd)) == [0, 2]

    def test_execute_missing_column_warns(self):
        cd = CapData("pv")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        cd.data_filtered = cd.data.copy()
        with pytest.warns(UserWarning, match="not a column"):
            kept = FilterPvsyst()._execute(cd)
        assert list(kept) == [0, 1]  # nothing dropped

    def test_explanation(self):
        cd = self._cd()
        f = FilterPvsyst()
        f.run(cd)
        assert "off-MPPT" in f.explanation or "off max power" in f.explanation
        assert f.explanation.endswith("were removed.")
```

- [ ] **Step 2: Run → FAIL** (`ImportError: FilterPvsyst`).

- [ ] **Step 3: Implement** — append to `filters.py`:

```python
class FilterPvsyst(BaseFilter):
    """Remove PVsyst intervals operating off the maximum power point.

    Drops rows where any of the PVsyst current-limit columns
    (``IL Pmin``/``IL Vmin``/``IL Pmax``/``IL Vmax``) is greater than 0.
    Column names with spaces or underscores are both recognized; a missing
    column warns and is skipped.
    """

    _legacy_name = "filter_pvsyst"
    _explanation_template = (
        "PVsyst intervals operating off the maximum power point "
        "(IL Pmin/Vmin/Pmax/Vmax > 0) were removed."
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        columns = ["IL Pmin", "IL Vmin", "IL Pmax", "IL Vmax"]
        index = df.index
        for column in columns:
            if column not in df.columns:
                column = column.replace(" ", "_")
            if column in df.columns:
                indices_to_drop = df[df[column] > 0].index
                if not index.equals(indices_to_drop):
                    index = index.difference(indices_to_drop)
            else:
                warnings.warn(
                    "{} or {} is not a column in the data.".format(
                        column, column.replace("_", " ")
                    )
                )
        return index
```

- [ ] **Step 4: Wrapper** — add `FilterPvsyst` to the `from captest.filters import (...)` block in `capdata.py`. Replace the `@update_summary`-decorated `filter_pvsyst` method with:

```python
    def filter_pvsyst(self, inplace=True):
        """Remove PVsyst intervals operating off the maximum power point.

        Drops rows where any IL Pmin/Vmin/Pmax/Vmax column is > 0.

        Parameters
        ----------
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        """
        flt = FilterPvsyst()
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5:** Run `tests/test_filter_classes.py::TestFilterPvsyst` + `tests/test_CapData.py -k filter_pvsyst` → PASS.
- [ ] **Step 6:** `just lint && just fmt`; commit `feat: add FilterPvsyst class and thin wrapper`.

---

### Task 2: FilterShade

- [ ] **Step 1: Tests** — add `FilterShade` to import. Append:

```python
class TestFilterShade:
    def _cd(self):
        cd = CapData("sh")
        cd.data = pd.DataFrame(
            {"FShdBm": [1.0, 0.5, 1.0, 0.8], "ShdLoss": [0.0, 50.0, 0.0, 130.0]},
            index=pd.RangeIndex(4),
        )
        cd.data_filtered = cd.data.copy()
        return cd

    def test_execute_default_fshdbm(self):
        cd = self._cd()
        # keep rows where FShdBm >= 1.0
        assert list(FilterShade()._execute(cd)) == [0, 2]

    def test_execute_custom_fshdbm(self):
        cd = self._cd()
        assert list(FilterShade(fshdbm=0.6)._execute(cd)) == [0, 2, 3]

    def test_execute_query_str(self):
        cd = self._cd()
        # keep rows where ShdLoss <= 125
        assert list(FilterShade(query_str="ShdLoss<=125")._execute(cd)) == [0, 1, 2]

    def test_explanation(self):
        cd = self._cd()
        f = FilterShade()
        f.run(cd)
        assert "shad" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement:**

```python
class FilterShade(BaseFilter):
    """Remove intervals of array shading.

    By default removes rows where the PVsyst ``FShdBm`` shading-fraction
    column is below ``fshdbm`` (default 1.0 — i.e. any shading). Pass a
    ``query_str`` to instead filter via ``DataFrame.query`` (e.g. when only
    a shading-loss column is available): ``"ShdLoss<=50"``.
    """

    _legacy_name = "filter_shade"
    _explanation_template = (
        "Intervals of array shading (kept where {query}) were removed."
    )

    fshdbm = param.Number(
        default=1.0,
        doc="Shading-fraction threshold; rows with FShdBm below this are "
        "removed. Ignored when query_str is given.",
    )
    query_str = param.String(
        default=None,
        allow_None=True,
        doc="Optional DataFrame.query expression overriding the FShdBm test.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        fshdbm = self.fshdbm  # noqa: F841 — referenced via @fshdbm in query
        query_str = self.query_str
        if query_str is None:
            query_str = "FShdBm>=@fshdbm"
        self._query_resolved = query_str
        return df.query(query_str).index

    def _explanation_values(self):
        return {"query": getattr(self, "_query_resolved", "FShdBm>=@fshdbm")}
```

- [ ] **Step 4: Wrapper** — add `FilterShade` to import; replace the method:

```python
    def filter_shade(self, fshdbm=1.0, query_str=None, inplace=True):
        """Remove intervals of array shading.

        Parameters
        ----------
        fshdbm : float, default 1.0
            Shading-fraction threshold; rows with FShdBm below this are removed.
        query_str : str, default None
            Optional DataFrame.query expression overriding the FShdBm test.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        """
        flt = FilterShade(fshdbm=fshdbm, query_str=query_str)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5:** Run `TestFilterShade` + `tests/test_CapData.py -k filter_shade` → PASS.
- [ ] **Step 6:** lint/fmt; commit `feat: add FilterShade class and thin wrapper`.

---

### Task 3: FilterDays

- [ ] **Step 1: Tests** — add `FilterDays` to import. Append (daily index so `.loc["day"]` selects):

```python
class TestFilterDays:
    def _cd(self):
        cd = CapData("d")
        idx = pd.date_range("1990-10-01", periods=10, freq="D")
        cd.data = pd.DataFrame({"power": range(10)}, index=idx)
        cd.data_filtered = cd.data.copy()
        return cd

    def test_execute_keep_days(self):
        cd = self._cd()
        kept = FilterDays(days=["10/5/1990", "10/6/1990"])._execute(cd)
        assert list(kept) == [pd.Timestamp("1990-10-05"), pd.Timestamp("1990-10-06")]

    def test_execute_drop_days(self):
        cd = self._cd()
        kept = FilterDays(days=["10/1/1990"], drop=True)._execute(cd)
        assert pd.Timestamp("1990-10-01") not in kept
        assert len(kept) == 9

    def test_explanation_keep(self):
        cd = self._cd()
        f = FilterDays(days=["10/5/1990"])
        f.run(cd)
        assert "kept" in f.explanation.lower() or "only" in f.explanation.lower()

    def test_explanation_drop(self):
        cd = self._cd()
        f = FilterDays(days=["10/5/1990"], drop=True)
        f.run(cd)
        assert "removed" in f.explanation.lower()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement:**

```python
class FilterDays(BaseFilter):
    """Keep (or drop) the timestamps belonging to a list of days.

    Each entry in ``days`` selects all timestamps on that calendar day
    (``DataFrame.loc[day]``). By default only those days are kept; set
    ``drop=True`` to remove them and keep everything else.
    """

    _legacy_name = "filter_days"

    days = param.List(
        default=None,
        allow_None=True,
        doc="Days to select (or drop). Each is a date string or Timestamp.",
    )
    drop = param.Boolean(
        default=False,
        doc="Drop the listed days instead of keeping only them.",
    )

    def _execute(self, capdata):
        df = capdata.data_filtered
        ix_all_days = None
        for day in self.days:
            ix_day = df.loc[day].index
            ix_all_days = ix_day if ix_all_days is None else ix_all_days.union(ix_day)
        if self.drop:
            return df.index.difference(ix_all_days)
        return ix_all_days

    @property
    def explanation(self):
        if not hasattr(self, "ix_after"):
            return None
        days = ", ".join(str(d) for d in (self.days or []))
        if self.drop:
            return f"Timestamps on the days [{days}] were removed."
        return f"All timestamps except the days [{days}] were removed."
```

- [ ] **Step 4: Wrapper** — add `FilterDays` to import; replace the method:

```python
    def filter_days(self, days, drop=False, inplace=True):
        """Keep or drop the timestamps belonging to a list of days.

        Parameters
        ----------
        days : list
            Days to select or drop (date strings or Timestamps).
        drop : bool, default False
            Drop the listed days instead of keeping only them.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        """
        flt = FilterDays(days=days, drop=drop)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5:** Run `TestFilterDays` (new) + `tests/test_CapData.py::TestFilterDays` → PASS.
- [ ] **Step 6:** lint/fmt; commit `feat: add FilterDays class and thin wrapper`.

> Note: there are now two `TestFilterDays` classes (one in `test_filter_classes.py`, one in `test_CapData.py`). That's fine — different modules. The existing `test_CapData.py::TestFilterDays` exercises the wrapper end-to-end and must stay green.

---

### Task 4: FilterPf

- [ ] **Step 1: Tests** — add `FilterPf` to import. Append:

```python
class TestFilterPf:
    def _cd(self):
        cd = CapData("pf")
        cd.data = pd.DataFrame(
            {"inv1 pf": [1.0, 0.5, 0.99], "inv2 pf": [0.999, 0.9, 1.0]},
            index=pd.RangeIndex(3),
        )
        cd.data_filtered = cd.data.copy()
        cd.column_groups = {"pf--": ["inv1 pf", "inv2 pf"]}
        return cd

    def test_execute_keeps_high_pf(self):
        cd = self._cd()
        # keep rows where all |pf| >= 0.95 -> row 0 only (row1 has 0.5, row2 has 0.99 & 1.0 -> kept)
        kept = FilterPf(pf=0.95)._execute(cd)
        assert list(kept) == [0, 2]

    def test_execute_no_pf_group_warns(self):
        cd = CapData("pf")
        cd.data = pd.DataFrame({"power": [1.0, 2.0]}, index=pd.RangeIndex(2))
        cd.data_filtered = cd.data.copy()
        cd.column_groups = {"real_pwr--": ["power"]}
        with pytest.warns(UserWarning, match="power factor"):
            kept = FilterPf(pf=0.99)._execute(cd)
        assert list(kept) == [0, 1]

    def test_explanation(self):
        cd = self._cd()
        f = FilterPf(pf=0.95)
        f.run(cd)
        assert "0.95" in f.explanation
        assert "power factor" in f.explanation.lower()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** (note `df.abs()` not `np.abs`; guard the missing-group NameError):

```python
class FilterPf(BaseFilter):
    """Remove intervals with a power factor below a threshold.

    Keeps rows where every column in the power-factor group (the first
    ``column_groups`` key beginning with ``pf``) has an absolute value at or
    above ``pf``.
    """

    _legacy_name = "filter_pf"
    _explanation_template = (
        "Intervals with a power factor below {pf} were removed."
    )

    pf = param.Number(
        default=None,
        allow_None=True,
        doc="Power-factor threshold, e.g. 0.999. Rows with any |pf| below "
        "this are removed.",
    )

    def _execute(self, capdata):
        selection = None
        for key in capdata.column_groups.keys():
            if key.find("pf") == 0:
                selection = key
        if selection is None:
            warnings.warn(
                "No power factor column group found in column_groups; "
                "filter_pf made no changes."
            )
            return capdata.data_filtered.index
        df = capdata.data_filtered[capdata.column_groups[selection]]
        mask = (df.abs() >= self.pf).all(axis=1)
        return capdata.data_filtered.index[mask]
```

- [ ] **Step 4: Wrapper** — add `FilterPf` to import; replace the method:

```python
    def filter_pf(self, pf, inplace=True):
        """Remove intervals with a power factor below ``pf``.

        Parameters
        ----------
        pf : float
            Power-factor threshold (e.g. 0.999). Rows with any |pf| below this
            are removed.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        """
        flt = FilterPf(pf=pf)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5:** Run `TestFilterPf` + `tests/test_CapData.py -k filter_pf` → PASS.
- [ ] **Step 6:** lint/fmt; commit `feat: add FilterPf class and thin wrapper`.

---

### Task 5: FilterPower

- [ ] **Step 1: Tests** — add `FilterPower` to import. Append:

```python
class TestFilterPower:
    def _cd(self):
        cd = CapData("pw")
        cd.data = pd.DataFrame(
            {"meter_power": [100.0, 600.0, 300.0, 900.0]},
            index=pd.RangeIndex(4),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"power": "meter_power"}
        return cd

    def test_execute_threshold(self):
        cd = self._cd()
        # keep rows with power < 500
        assert list(FilterPower(power=500)._execute(cd)) == [0, 2]

    def test_execute_percent(self):
        cd = self._cd()
        # power*(1-percent) = 1000*(1-0.5) = 500 threshold
        assert list(FilterPower(power=1000, percent=0.5)._execute(cd)) == [0, 2]

    def test_execute_named_column(self):
        cd = self._cd()
        assert list(FilterPower(power=500, columns="meter_power")._execute(cd)) == [0, 2]

    def test_execute_bad_columns_warns(self):
        cd = self._cd()
        f = FilterPower(power=500, columns=1)
        with pytest.warns(UserWarning, match="None or a string"):
            kept = f._execute(cd)
        assert len(kept) == cd.data_filtered.shape[0]

    def test_explanation(self):
        cd = self._cd()
        f = FilterPower(power=500)
        f.run(cd)
        assert "power" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement:**

```python
class FilterPower(BaseFilter):
    """Remove intervals at or above a power threshold.

    With ``percent`` set, ``power`` is treated as nameplate and the effective
    threshold is ``power * (1 - percent)``. ``columns`` selects the power
    data: None uses the regression power column; a column-group key applies
    the threshold across the group; a bare column name uses that column.
    """

    _legacy_name = "filter_power"
    _explanation_template = (
        "Intervals at or above {threshold} power were removed."
    )

    power = param.Number(
        default=None, allow_None=True, doc="Power threshold (or nameplate if percent set)."
    )
    percent = param.Number(
        default=None, allow_None=True,
        doc="If set, threshold is power*(1-percent). Decimal, e.g. 0.01 for 1%.",
    )
    columns = param.String(
        default=None, allow_None=True,
        doc="Column or column-group to filter on. None uses the regression power column.",
    )

    def _execute(self, capdata):
        power = self.power
        if self.percent is not None:
            power = power * (1 - self.percent)
        self.power_threshold = power

        multiple_columns = False
        if self.columns is None:
            power_data = capdata.get_reg_cols("power")
        elif isinstance(self.columns, str):
            if self.columns in capdata.column_groups.keys():
                power_data = capdata.floc[self.columns]
                multiple_columns = True
            else:
                power_data = pd.DataFrame(capdata.data_filtered[self.columns])
                power_data = power_data.rename(
                    columns={power_data.columns[0]: "power"}
                )
        else:
            warnings.warn("columns must be None or a string.")
            return capdata.data_filtered.index

        if multiple_columns:
            mask = power_data.apply(
                lambda x: all(x.le(power, fill_value=True)), axis=1
            )
        else:
            mask = power_data["power"] < power
        return capdata.data_filtered.index[mask]

    def _explanation_values(self):
        return {"threshold": getattr(self, "power_threshold", self.power)}
```

- [ ] **Step 4: Wrapper** — add `FilterPower` to import; replace the method:

```python
    def filter_power(self, power, percent=None, columns=None, inplace=True):
        """Remove intervals at or above a power threshold.

        Parameters
        ----------
        power : numeric
            Threshold, or nameplate power when ``percent`` is given.
        percent : numeric, default None
            If set, threshold is ``power * (1 - percent)`` (decimal).
        columns : str, default None
            Column or column-group to filter on. None uses the regression
            power column.
        inplace : bool, default True
            If True, record the filter step and update data_filtered. If False,
            return the filtered DataFrame without recording a step.
        """
        flt = FilterPower(power=power, percent=percent, columns=columns)
        if inplace:
            flt.run(self)
        else:
            return self.data_filtered.loc[flt._execute(self), :]
```

- [ ] **Step 5:** Run `TestFilterPower` + `tests/test_CapData.py::TestFilterOutliersAndPower -k power` → PASS (covers defaults, percent, a_column, column_group, column_group_with_nan, columns_not_str).
- [ ] **Step 6:** lint/fmt; commit `feat: add FilterPower class and thin wrapper`.

> Note: the `fill_value=True` in the column-group path preserves the legacy NaN behavior (`test_filter_power_column_group_with_nan`): NaN compares as "below threshold" → kept.

---

### Task 6: FilterMissing

- [ ] **Step 1: Tests** — add `FilterMissing` to import. Append:

```python
class TestFilterMissing:
    def _cd(self):
        cd = CapData("m")
        cd.data = pd.DataFrame(
            {"poa": [1.0, np.nan, 3.0], "power": [10.0, 20.0, np.nan]},
            index=pd.RangeIndex(3),
        )
        cd.data_filtered = cd.data.copy()
        cd.regression_cols = {"poa": "poa", "power": "power"}
        return cd

    def test_execute_default_regcols(self):
        cd = self._cd()
        # rows with NaN in any regression col dropped -> only row 0
        assert list(FilterMissing()._execute(cd)) == [0]

    def test_execute_subset_columns(self):
        cd = self._cd()
        # only consider 'poa' -> drop row 1 -> keep 0, 2
        assert list(FilterMissing(columns=["poa"])._execute(cd)) == [0, 2]

    def test_legacy_name_is_filter_missing(self):
        assert FilterMissing._legacy_name == "filter_missing"

    def test_explanation(self):
        cd = self._cd()
        f = FilterMissing()
        f.run(cd)
        assert "missing" in f.explanation.lower()
        assert f.explanation.endswith("were removed.")
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement:**

```python
class FilterMissing(BaseFilter):
    """Remove rows with missing data (NaN) in the regression columns.

    By default checks the columns identified by ``regression_cols`` (via the
    ``regcols`` floc key); pass ``columns`` to restrict the NaN check to a
    subset.
    """

    _legacy_name = "filter_missing"
    _explanation_template = (
        "Intervals with missing data in the regression columns were removed."
    )

    columns = param.List(
        default=None,
        allow_None=True,
        doc="Subset of columns to check for NaN. None uses the regression columns.",
    )

    def _execute(self, capdata):
        if self.columns is None:
            return capdata.floc["regcols"].dropna().index
        return capdata.floc[self.columns].dropna().index
```

- [ ] **Step 4: Wrapper** — add `FilterMissing` to import; replace the method (no `inplace`, like `filter_custom`):

```python
    def filter_missing(self, columns=None):
        """Remove rows with missing data (NaN) in the regression columns.

        Parameters
        ----------
        columns : list, default None
            Subset of columns to check for NaN. By default uses the regression
            columns identified in ``regression_cols``.
        """
        FilterMissing(columns=columns).run(self)
```

- [ ] **Step 5:** Run `TestFilterMissing` (new) + `tests/test_CapData.py::TestFilterMissing` + `tests/test_CapData.py::TestFilterOutliersAndPower::test_filter_outliers_nan_records_filter_missing_in_summary` → PASS (the last confirms the `_legacy_name` is still recorded when FilterOutliers auto-calls it).
- [ ] **Step 6:** lint/fmt; commit `feat: add FilterMissing class and thin wrapper`.

---

### Task 7: Full-suite verification

- [ ] **Step 1:** `just test-wo-warnings` → all pass.
- [ ] **Step 2:** `just lint && just fmt` → clean.
- [ ] **Step 3:** Confirm no `@update_summary`-decorated filter methods remain (grep) — only non-filter steps (`rep_cond`, `fit_regression`, `rep_cond_freq`, `predict_capacities`) should still carry it, if any. This documents that chunk 4 (the `data_filtered` property flip) is now unblocked.

---

## Self-Review

**1. Spec coverage:** All six "Concrete Filter Classes" + their "Thin Wrapper Methods" → Tasks 1-6. ✓

**2. Placeholder scan:** No TBDs. Every class and wrapper shows complete code; every test block is concrete.

**3. Type/name consistency:** Each `FilterX` param set matches its wrapper signature and tests. `_legacy_name` strings exactly match the original method names (critical for `FilterMissing`). `df.abs()` replaces `np.abs` (no numpy import added). `fshdbm` bound locally in `FilterShade._execute` for the `@fshdbm` query.

**Behavioral invariants preserved (from existing test_CapData.py):**
- `filter_pvsyst`: default drop, underscored names, missing-column warn, not-inplace. ✓
- `filter_shade`: default, not-inplace, query_str. ✓
- `filter_days`: keep one/two/multiple, drop, not-inplace. ✓
- `filter_pf`: `filter_pf(1)`. ✓ (plus new no-group guard)
- `filter_power`: defaults, percent, a column, column group, column-group-with-nan (`fill_value=True`), columns-not-str warn. ✓
- `filter_missing`: default, missing-not-in-columns, passed columns; **and** the cross-filter `test_filter_outliers_nan_records_filter_missing_in_summary`. ✓

**No test references to repoint:** verified — these six use only CapData methods, not module-level functions, so nothing moves out of `capdata.py` and no `pvc.X`/`capdata.X` references exist for them.
