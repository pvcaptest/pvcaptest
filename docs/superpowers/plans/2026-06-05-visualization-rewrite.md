# Visualization Rewrite Implementation Plan (chunk 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the three filter-attribution views (`scatter_filters`, `timeseries_filters`, `get_filtering_table`) to derive removed-by-filter sets from the `filters` chain via a shared `_removed_by_step()` helper, retarget `get_length_test_period`, and delete the now-unused `removed`/`kept` mirror lists.

**Architecture:** A new `CapData._removed_by_step()` computes, for each filter step that removed ≥1 interval, the tuple `(i, label, removed_ix)` where `removed_ix = _ix_before(i) \ filters[i].ix_after`. Zero-removal steps (always `RepCond`; also any filter that matched everything) are skipped. The two plots and the filtering table all consume this one helper, so they agree on which steps appear. With every consumer moved off `self.removed`/`self.kept`, those lists — and `BaseSummaryStep._record_removed_kept` plus its `run()` call — are deleted, leaving `run()` to simply append the step to `filters`.

**Tech Stack:** Python, `param`, pandas, HoloViews, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Visualization Methods (chunk 6)".

**Sequencing:** Execute **after** the chunk-5 summary rebuild (already landed; tip `04c6647`). The chain helpers `_ix_before(i)`/`_step_labels()` and the `function_name`/chain-derived `get_summary` exist. `RepCond` is already a zero-removal step in `filters`.

## Commit shape (four commits, each green)

The deletion of `removed`/`kept` is only safe once every consumer is migrated, so the work is staged so each commit leaves the suite green:

1. **Task 1 — `_removed_by_step()` helper.** Additive + unit tests. Green.
2. **Task 2 — rewrite the three attribution views** to consume the helper. They stop reading `removed`/`kept` (which are still populated, now unused by them). Green.
3. **Task 3 — retarget `get_length_test_period`** to `isinstance(step, FilterTime)`. The last remaining src reader of `kept` moves off it. Green.
4. **Task 4 — delete `removed`/`kept`** + `_record_removed_kept` + its `run()` call + stale docstrings + the one test that asserted the lists. Green.

---

## File Structure

- `src/captest/capdata.py` — add `_removed_by_step()`; rewrite `scatter_filters`, `timeseries_filters`, `get_filtering_table`, `get_length_test_period`; delete `self.removed`/`self.kept` from `__init__`/`reset_filter`/`agg_sensors`/`process_regression_columns`.
- `src/captest/filters.py` — delete `_record_removed_kept` and its call in `run()`; fix the stale `_record_removed_kept` reference in `FilterCustom.args_repr`'s docstring.
- `src/captest/plotting.py` — fix the stale `cd.kept`/`cd.removed` docstring reference.
- `tests/test_filter_classes.py` — add `TestRemovedByStep`; delete `test_run_populates_removed_and_kept`.
- `tests/test_CapData.py` — add scatter/timeseries layer-count + skip tests, a `get_filtering_table` zero-removal-skip test, and a `get_length_test_period` `custom_name` robustness test.

---

### Task 1: Add the `_removed_by_step()` helper

**Files:**
- Modify: `src/captest/capdata.py` (add `_removed_by_step`, near `_step_labels`)
- Test: `tests/test_filter_classes.py` (new `TestRemovedByStep`)

- [ ] **Step 1: Write the failing tests**

In `tests/test_filter_classes.py`, add a new class. The `cd_irr` fixture has a 5-row poa frame `[100, 300, 500, 700, 900]` at `RangeIndex(5)`, name `"irr"`, `regression_cols={"poa": "poa"}`. `FilterIrr(200, 800)` keeps indices `[1, 2, 3]` and removes `[0, 4]`.

```python
class TestRemovedByStep:
    def test_single_filter(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        result = cd_irr._removed_by_step()
        assert len(result) == 1
        i, label, removed_ix = result[0]
        assert i == 0
        assert label == "FilterIrr"
        assert list(removed_ix) == [0, 4]

    def test_two_filters_indices_and_labels(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)  # keeps [1,2,3]
        FilterIrr(low=400, high=800).run(cd_irr)  # keeps [2,3], removes [1]
        result = cd_irr._removed_by_step()
        assert [(i, label) for i, label, _ in result] == [
            (0, "FilterIrr"),
            (1, "FilterIrr-1"),
        ]
        assert list(result[1][2]) == [1]

    def test_skips_zero_removal_step(self, cd_irr):
        FilterIrr(low=0, high=10000).run(cd_irr)  # keeps all 5 -> removes nothing
        assert cd_irr._removed_by_step() == []

    def test_zero_removal_step_between_real_filters(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)  # removes [0,4]
        FilterIrr(low=0, high=10000).run(cd_irr)  # removes nothing -> skipped
        FilterIrr(low=400, high=800).run(cd_irr)  # removes [1]
        result = cd_irr._removed_by_step()
        # The no-op middle filter is skipped; the third filter keeps its real
        # index i=2 (skipping affects only which entries are returned, not the
        # chain math).
        assert [(i, label) for i, label, _ in result] == [
            (0, "FilterIrr"),
            (2, "FilterIrr-2"),
        ]
        assert list(result[1][2]) == [1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest "tests/test_filter_classes.py::TestRemovedByStep" -v`
Expected: FAIL — `_removed_by_step` is undefined (AttributeError). (If a `uv` read-only-cache sandbox error appears on the first run, re-run.)

- [ ] **Step 3: Add the helper**

In `src/captest/capdata.py`, add this method immediately after `_step_labels` (which is just above `get_summary`):

```python
    def _removed_by_step(self):
        """Per-step removal attribution for the visualization methods.

        Returns a list of ``(i, label, removed_ix)`` for each filter step that
        removed at least one interval, where ``removed_ix`` is
        ``_ix_before(i)`` minus ``filters[i].ix_after`` and ``label`` is the
        step's ``_step_labels()`` entry. Zero-removal steps (always ``RepCond``;
        also any filter that matched everything) are skipped — they have nothing
        to attribute. ``i`` is the step's real index in ``self.filters`` so
        callers can recover its input set via ``_ix_before(i)`` and its
        survivors via ``self.filters[i].ix_after``.
        """
        out = []
        for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
            removed_ix = self._ix_before(i).difference(step.ix_after)
            if len(removed_ix) > 0:
                out.append((i, label, removed_ix))
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest "tests/test_filter_classes.py::TestRemovedByStep" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
just lint && just fmt
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "feat: add CapData._removed_by_step for filter-removal attribution"
```

---

### Task 2: Rewrite the three attribution views to consume `_removed_by_step()`

**Files:**
- Modify: `src/captest/capdata.py` (`scatter_filters`, `timeseries_filters`, `get_filtering_table`)
- Test: `tests/test_CapData.py` (extend `TestScatterFilters`, `TestTimeseriesFilters`, `TestGetFilteringTable`)

> These rewrites stop reading `self.removed`/`self.kept`; those lists are still populated by `run()` (deleted in Task 4) but no longer consumed here. The existing `test_returns_overlay` and `test_get_filtering_table` tests must stay green.

- [ ] **Step 1: Write the failing tests**

In `tests/test_CapData.py`, add to the existing `TestScatterFilters` class (it already has a `test_returns_overlay` using `meas` with `regression_cols` set + `process_regression_columns()` + two `filter_irr` calls). Add:

```python
    def test_layer_count_is_retained_plus_removing_filters(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.scatter_filters()
        # 1 retained baseline + 2 removing filters
        assert len(overlay) == 3

    def test_zero_removal_step_adds_no_layer(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.rep_cond()  # RepCond: zero-removal -> no layer
        overlay = meas.scatter_filters()
        # 1 retained baseline + 1 removing filter; RepCond contributes nothing
        assert len(overlay) == 2

    def test_layers_carry_the_right_rows(self, meas):
        """Pin the row-selection glue: the retained baseline holds the survivors
        and each removed layer holds exactly that filter's removed rows (a
        retained/removed swap would still pass the count assertions above)."""
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.scatter_filters()
        # Ordered leaf elements: retained baseline first, then one per removing
        # filter. The Scatter's backing frame carries an "index" column set to
        # the original data index, so we can check which rows landed in a layer.
        layers = list(overlay)
        assert list(layers[0].data["index"]) == list(meas.filters[-1].ix_after)
        _i, _label, removed_ix = meas._removed_by_step()[0]
        assert list(layers[1].data["index"]) == list(removed_ix)
```

> Note on overlay introspection: `list(overlay)` yields the leaf elements in
> insertion order, and `element.data` is the pandas frame the layer was built
> from. If a HoloViews version returns elements via `overlay.values()` instead,
> use that — both give the ordered element list. The `"index"`/`"Timestamp"`
> *column* (not the frame's row index) is what `scatter_filters`/
> `timeseries_filters` populate with the original data index, so assert on it.

Add to `TestTimeseriesFilters` (same fixture/setup pattern):

```python
    def test_layer_count_is_curve_plus_removing_filters(self, meas):
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.timeseries_filters()
        # 1 full-data curve + 2 removing-filter scatters
        assert len(overlay) == 3

    def test_removed_layer_carries_the_right_rows(self, meas):
        """Pin that a removed-filter scatter layer holds exactly that filter's
        removed rows (the full-data Curve baseline is layer 0)."""
        meas.regression_cols = {
            "power": "meter_power",
            "poa": ("irr_poa_pyran", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind", "mean"),
        }
        meas.process_regression_columns()
        meas.filter_irr(200, 900)
        meas.filter_irr(400, 800)
        overlay = meas.timeseries_filters()
        layers = list(overlay)  # curve baseline first, then one scatter per remover
        _i, _label, removed_ix = meas._removed_by_step()[0]
        assert list(layers[1].data["Timestamp"]) == list(removed_ix)
```

Add to `TestGetFilteringTable` (the existing `test_get_filtering_table` uses `nrel` with three `filter_irr` calls):

```python
    def test_zero_removal_step_gets_no_column(self, nrel):
        nrel.filter_irr(200, 900)
        nrel.filter_irr(400, 800)
        nrel.rep_cond()  # RepCond: zero-removal -> no column
        flt_table = nrel.get_filtering_table()
        # Pin the column-per-removing-step contract by label and order: one
        # column per removing filter (named via _step_labels) then all_filters;
        # the zero-removal RepCond step gets no column.
        assert list(flt_table.columns) == ["FilterIrr", "FilterIrr-1", "all_filters"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest "tests/test_CapData.py::TestScatterFilters" "tests/test_CapData.py::TestTimeseriesFilters" "tests/test_CapData.py::TestGetFilteringTable" -v`
Expected: the three new tests FAIL on layer/column counts (current methods build cumulative-kept layers and include a `RepCond` column), while the pre-existing `test_returns_overlay`/`test_get_filtering_table` still pass.

- [ ] **Step 3: Rewrite `scatter_filters`**

In `src/captest/capdata.py`, replace the entire body of `scatter_filters` with the version below (the `HoverTool`/`.opts(...)` blocks are unchanged from the current method; only the layer construction changes):

```python
    def scatter_filters(self):
        """Overlay of power-vs-irradiance scatters attributing removed intervals.

        A baseline ``retained`` layer (rows surviving all filters) plus one
        layer per filter step that removed intervals — together a clean
        partition of the data. Zero-removal steps (e.g. ``RepCond``) are
        skipped; see ``get_summary``/``describe_filters`` for the full step list.
        """
        data = self.get_reg_cols(reg_vars=["power", "poa"], filtered_data=False)
        data["index"] = self.data.index

        scatters = []
        retained_ix = self.filters[-1].ix_after if self.filters else self.data.index
        scatters.append(
            hv.Scatter(data.loc[retained_ix, :], "poa", ["power", "index"]).relabel(
                "retained"
            )
        )
        for _i, label, removed_ix in self._removed_by_step():
            scatters.append(
                hv.Scatter(data.loc[removed_ix, :], "poa", ["power", "index"]).relabel(
                    label
                )
            )

        scatter_overlay = hv.Overlay(scatters)
        hover = HoverTool(
            tooltips=[
                ("datetime", "@index{%Y-%m-%d %H:%M}"),
                ("poa", "@poa{0,0.0}"),
                ("power", "@power{0,0.0}"),
            ],
            formatters={
                "@index": "datetime",
            },
        )
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                width=650,
                height=500,
                muted_fill_alpha=0,
                fill_alpha=0.4,
                line_width=0,
                tools=[hover],
                yformatter=NumeralTickFormatter(format="0,0"),
            ),
            hv.opts.Overlay(legend_position="right", toolbar="above"),
        )
        return scatter_overlay
```

- [ ] **Step 4: Rewrite `timeseries_filters`**

Replace the entire body of `timeseries_filters` with the version below. Note the deliberate per-method difference: the baseline here is the **full-data power Curve** (a continuous line of all data), labeled `"all"`, with removed points overlaid as scatters — a line of only-retained points would draw misleading segments across gaps, so unlike `scatter_filters` it keeps the full curve as backdrop. The `HoverTool`/`.opts(...)` blocks are unchanged from the current method.

```python
    def timeseries_filters(self):
        """Power-vs-time line with removed intervals highlighted per filter.

        A full-data power ``Curve`` backdrop plus one scatter layer per filter
        step that removed intervals. Zero-removal steps (e.g. ``RepCond``) are
        skipped; see ``get_summary``/``describe_filters`` for the full step list.
        """
        data = self.get_reg_cols(reg_vars="power", filtered_data=False)
        data["Timestamp"] = data.index

        plots = []
        plt_no_filtering = hv.Curve(data, ["Timestamp"], ["power"], label="all")
        plt_no_filtering.opts(
            line_color="grey",
            line_width=1,
            width=1500,
            height=450,
        )
        plots.append(plt_no_filtering)
        for _i, label, removed_ix in self._removed_by_step():
            d_flt = data.loc[removed_ix, ["power", "Timestamp"]]
            plots.append(hv.Scatter(d_flt, ["Timestamp"], ["power"], label=label))

        scatter_overlay = hv.Overlay(plots)
        hover = HoverTool(
            tooltips=[
                ("datetime", "@Timestamp{%Y-%m-%d %H:%M}"),
                ("power", "@power{0,0.0}"),
            ],
            formatters={
                "@Timestamp": "datetime",
            },
        )
        scatter_overlay.opts(
            hv.opts.Scatter(
                size=5,
                muted_fill_alpha=0,
                fill_alpha=1,
                line_width=0,
                tools=[hover],
                yformatter=NumeralTickFormatter(format="0,0"),
            ),
            hv.opts.Overlay(
                legend_position="bottom",
                toolbar="right",
            ),
        )
        return scatter_overlay
```

- [ ] **Step 5: Rewrite `get_filtering_table`**

Replace the entire body of `get_filtering_table` with:

```python
    def get_filtering_table(self):
        """
        Returns DataFrame showing which filter removed each filtered time interval.

        One column per filter step that removed intervals, in run order. Within a
        column: ``1`` marks the intervals that step removed, ``0`` the intervals
        present going into that step and kept by it, and ``NaN`` intervals already
        removed by an earlier step. The final ``all_filters`` column is True for
        intervals not removed by any filter. Zero-removal steps (e.g. ``RepCond``)
        get no column, consistent with the scatter/timeseries views.
        """
        filtering_data = pd.DataFrame(index=self.data.index)
        for i, label, removed_ix in self._removed_by_step():
            filtering_data.loc[self.filters[i].ix_after, label] = 0
            filtering_data.loc[removed_ix, label] = 1
        filtering_data["all_filters"] = filtering_data.apply(
            lambda x: all(x == 0), axis=1
        )
        return filtering_data
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest "tests/test_CapData.py::TestScatterFilters" "tests/test_CapData.py::TestTimeseriesFilters" "tests/test_CapData.py::TestGetFilteringTable" -v`
Expected: PASS — the new count/skip tests and the pre-existing `test_returns_overlay`/`test_get_filtering_table` (the three real filters in that test each remove rows, so it still gets 3 columns + `all_filters`).

- [ ] **Step 7: Commit**

```bash
just lint && just fmt
git add src/captest/capdata.py tests/test_CapData.py
git commit -m "refactor: rebuild scatter/timeseries/filtering-table from _removed_by_step"
```

---

### Task 3: Retarget `get_length_test_period` off `self.kept`

**Files:**
- Modify: `src/captest/capdata.py` (`get_length_test_period`)
- Test: `tests/test_CapData.py` (`TestPointsSummary`)

- [ ] **Step 1: Write the failing test**

In `tests/test_CapData.py`, add to `TestPointsSummary` (the file imports `from captest import filters`). The existing tests cover no-filter (→5), one `filter_time` (→4), and two `filter_time` (→4, first wins). Add a `custom_name` robustness case that the old label-matching implementation would miss:

```python
    def test_length_test_period_custom_name_filter_time(self, meas):
        # A custom_name'd FilterTime must still be found: the period comes from
        # isinstance(step, FilterTime), not a label-string match.
        filters.FilterTime(
            start="10/9/1990", end="10/12/1990 23:00", custom_name="window"
        ).run(meas)
        meas.get_length_test_period()
        assert meas.length_test_period == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest "tests/test_CapData.py::TestPointsSummary::test_length_test_period_custom_name_filter_time" -v`
Expected: FAIL — the current loop matches the label string `"FilterTime"`, but this step's `kept` label is `"window"`, so no match → falls back to the full-data length (5), not 4.

- [ ] **Step 3: Rewrite the method**

In `src/captest/capdata.py`, replace the body of `get_length_test_period` (keep the docstring, which already describes "first `FilterTime` wins, subsequent ignored"). `FilterTime` is already imported at the top of `capdata.py`.

Change:

```python
        test_period = self.data.index[-1] - self.data.index[0]
        for filter in self.kept:
            if "FilterTime" == filter["name"]:
                test_period = filter["index"][-1] - filter["index"][0]
        self.length_test_period = test_period.ceil("D").days
```

to:

```python
        test_period = self.data.index[-1] - self.data.index[0]
        for step in self.filters:
            if isinstance(step, FilterTime):
                test_period = step.ix_after[-1] - step.ix_after[0]
                break
        self.length_test_period = test_period.ceil("D").days
```

> The `break` makes the **first** `FilterTime` win, preserving the documented "subsequent uses are ignored" behavior and keeping `test_length_test_period_after_two_filter_time` (which expects 4, the first window's span) green.

- [ ] **Step 4: Run the `TestPointsSummary` length tests**

Run: `uv run pytest "tests/test_CapData.py::TestPointsSummary" -k "length_test_period" -v`
Expected: PASS — all four (`no_filter`→5, `after_one_filter_time`→4, `after_two_filter_time`→4, `custom_name`→4).

- [ ] **Step 5: Commit**

```bash
just lint && just fmt
git add src/captest/capdata.py tests/test_CapData.py
git commit -m "refactor: derive get_length_test_period from filters (isinstance FilterTime)"
```

---

### Task 4: Delete `removed`/`kept` and the `_record_removed_kept` mirror

**Files:**
- Modify: `src/captest/capdata.py` (4 deletion sites)
- Modify: `src/captest/filters.py` (`run()` call + `_record_removed_kept` + a docstring ref)
- Modify: `src/captest/plotting.py` (docstring ref)
- Test: `tests/test_filter_classes.py` (delete `test_run_populates_removed_and_kept`)

> By now no src code reads `self.removed`/`self.kept` (Tasks 2–3 moved every consumer to the chain). This task deletes the lists and the machinery that populated them.

- [ ] **Step 1: Remove the `run()` call and `_record_removed_kept`**

In `src/captest/filters.py`, in `BaseSummaryStep.run`, delete the line:

```python
        self._record_removed_kept(capdata)
```

so the tail of `run` becomes:

```python
        capdata.filters = capdata.filters + [self]
        if self.pts_after == 0:
            warnings.warn("The last filter removed all data!")
```

Then delete the entire `_record_removed_kept` method (the `def _record_removed_kept(self, capdata):` block through its final `capdata.kept.append(...)` line).

- [ ] **Step 2: Fix the stale docstring reference in `FilterCustom.args_repr`**

In `src/captest/filters.py`, in `FilterCustom.args_repr`'s docstring, the phrase references the now-deleted method. Change:

```python
        instances) do not expose ``__name__``. Without the guard, accessing
        ``args_repr`` from inside ``run()``'s ``_record_removed_kept`` would
        raise ``AttributeError`` *after* ``_execute`` had already mutated
```

to:

```python
        instances) do not expose ``__name__``. Without the guard, accessing
        ``args_repr`` from inside ``run()`` would raise ``AttributeError``
        *after* ``_execute`` had already mutated
```

- [ ] **Step 3: Delete `self.removed`/`self.kept` from `__init__`**

In `src/captest/capdata.py`, in `__init__`, delete:

```python
        self.removed = []
        self.kept = []
```

- [ ] **Step 4: Delete them from `reset_filter`**

In `reset_filter`, delete the two lines, leaving `self.filters = []`:

```python
        self.removed = []
        self.kept = []
```

- [ ] **Step 5: Delete them (and the now-empty comment) from `agg_sensors`**

In `agg_sensors`, delete:

```python
        # reset filter-history mirrors (filters itself is cleared below)
        self.removed = []
        self.kept = []
```

(The blank line and `self.pre_agg_cols = ...` that follow remain; `self.filters = []` later in the method still clears the chain.)

- [ ] **Step 6: Delete them (and the comment) from `process_regression_columns`**

In `process_regression_columns`, delete:

```python
        # reset filter-history mirrors (filters itself is cleared below)
        self.removed = []
        self.kept = []
```

- [ ] **Step 7: Fix the `plotting.py` docstring reference**

In `src/captest/plotting.py`, in the `calc_tc_power_column` docstring, drop the now-deleted attributes from the "does NOT touch" list. Change:

```python
    This helper is intentionally isolated from
    ``CapData.process_regression_columns``: it does NOT touch
    ``cd.regression_cols``, ``cd.regression_formula``, ``cd.kept``,
    or ``cd.removed``.
```

to:

```python
    This helper is intentionally isolated from
    ``CapData.process_regression_columns``: it does NOT touch
    ``cd.regression_cols`` or ``cd.regression_formula``.
```

- [ ] **Step 8: Delete the obsolete `removed`/`kept` test**

In `tests/test_filter_classes.py`, delete the whole method (its behavior — the mirror lists — no longer exists; `TestRemovedByStep` from Task 1 covers the replacement):

```python
    def test_run_populates_removed_and_kept(self, cd_irr):
        FilterIrr(low=200, high=800).run(cd_irr)
        assert list(cd_irr.removed[0]["index"]) == [0, 4]
        assert list(cd_irr.kept[0]["index"]) == [1, 2, 3]
        assert cd_irr.removed[0]["name"] == "FilterIrr"
```

- [ ] **Step 9: Full-suite verification**

Run: `just test-wo-warnings`
Expected: all pass. Then confirm no `removed`/`kept` references remain in src or tests (ignore `.ipynb_checkpoints/`):

Run: `grep -rnE "\.removed\b|\.kept\b|_record_removed_kept" src/captest/ tests/ | grep -v ".ipynb_checkpoints"`
Expected: empty.

- [ ] **Step 10: Commit**

```bash
just lint && just fmt
git add -A
git commit -m "refactor: delete removed/kept mirror lists and _record_removed_kept"
```

---

## Self-Review

**1. Spec coverage** (`docs/superpowers/specs/.../2026-04-03-...md` → "Visualization Methods (chunk 6)"):
- Shared `_removed_by_step()` (skip zero-removal, return `(i, label, removed_ix)`) → Task 1. ✓
- `scatter_filters` / `timeseries_filters` rewrite (retained/all baseline + per-removing-filter layers) → Task 2 Steps 3–4. ✓
- `get_filtering_table` chain rewrite (0 on survivors, 1 on removed, NaN earlier; `all_filters` unchanged; zero-removal steps get no column) → Task 2 Step 5. ✓
- "Zero-removal steps skipped in attribution views" → enforced in `_removed_by_step` (Task 1), tested in Task 2 (scatter + table skip tests). ✓
- `get_length_test_period` → `isinstance(step, FilterTime)`, **first** match + break → Task 3. ✓
- Delete `removed`/`kept` from `__init__`/`reset_filter`/`agg_sensors`/`process_regression_columns`; delete `_record_removed_kept` + its `run()` call → Task 4 Steps 1, 3–6. ✓
- Stale docstrings (`FilterCustom.args_repr`, `plotting.calc_tc_power_column`) → Task 4 Steps 2, 7. ✓

**2. Placeholder scan:** No TBDs. Every rewritten method body is shown in full, including the unchanged `HoverTool`/`.opts(...)` blocks (reproduced verbatim so the engineer replaces the whole method, not a fragment). Test bodies are complete.

**3. Type/name consistency:** `_removed_by_step()` returns `(i, label, removed_ix)` and every consumer unpacks that shape: `scatter_filters`/`timeseries_filters` use `for _i, label, removed_ix in ...`; `get_filtering_table` uses `for i, label, removed_ix in ...` then `self.filters[i].ix_after`. `_ix_before`/`_step_labels`/`filters`/`ix_after` all exist from chunk 5. `FilterTime` is imported in `capdata.py`. The baseline label differs intentionally by method (`"retained"` scatter, `"all"` timeseries) — documented in Task 2 and the spec.

**Deliberate decisions / deviations:**
- `timeseries_filters` keeps a **full-data** `Curve` baseline (label `"all"`) rather than a "retained" layer — a line of only-retained points would draw misleading segments across removed gaps; the removed scatters highlight drops over the continuous line. `scatter_filters` uses a `"retained"` points baseline so its layers are a clean non-overlapping partition.
- `_removed_by_step` returns the step's **real** index `i` (not a re-enumerated position), so skipping a zero-removal step does not shift later steps' `i` — `get_filtering_table` relies on `self.filters[i].ix_after` resolving to the correct step.
- Plot tests assert layer **counts** (`len(overlay)`) plus, for one case each, the **rows in a layer** by reading the element's backing frame `"index"`/`"Timestamp"` column (which the methods set to the original data index) — catching a retained/removed slice swap that counts alone would miss. They avoid asserting HoloViews display *labels*, which the library sanitizes/transforms. `get_filtering_table` is pinned by **column labels/order** (`list(flt_table.columns)`), making the column-per-removing-step skip contract explicit rather than positional.
