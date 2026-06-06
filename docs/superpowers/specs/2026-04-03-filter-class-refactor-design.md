# Filter Class Refactor Design

**Date:** 2026-04-03
**Status:** Approved for implementation — prerequisites satisfied as of v0.15.1 (see Implementation Sequencing)
**Scope:** Filter class architecture, `data_filtered` property, YAML round-trip serialization

---

## Background and Motivation

Filters on the `CapData` class are currently plain methods decorated with `@update_summary`. The decorator maintains four parallel lists on `CapData` (`summary`, `summary_ix`, `removed`, `kept`) and a `filter_counts` dict that must all stay in sync. `data_filtered` is a mutable copy of `data` that each filter reads from and writes back to.

This design has several limitations:

- No persistent filter objects — filter steps cannot be inspected, edited, reordered, or replayed after the fact
- The custom name feature (PR #119) requires adding `filter_name=None` to every filter method signature, which is boilerplate with no functional purpose (the wrapper already pops it from kwargs before calling the function)
- The actual function name called is not retained in the summary table when a custom name is provided
- Four parallel lists must be managed in sync, and `reset_filter()` must clear all of them
- `data_filtered` is a full copy of `data`, doubling memory usage for large datasets
- No path to YAML serialization or a reactive GUI without duplicating the parameter model

The class-based architecture resolves all of these by making each filter a first-class object.

---

## Module Organization

The refactor introduces a dedicated `src/captest/filters.py` module rather than growing `capdata.py` (already ~3,600 lines). This matches the codebase's existing modular split (`io.py`, `columngroups.py`, `plotting.py`, `prtest.py`).

### `src/captest/filters.py` (new)

Holds the entire filter-step surface:

- `BaseSummaryStep`, `BaseFilter` base classes
- All concrete filter classes (`FilterIrr` … `FilterMissing`, `FilterRegression`), plus `RepCond`
- `FILTER_REGISTRY` and the YAML serialization/deserialization helpers
- The **row-filter helper functions** moved out of `capdata.py`: `filter_irr`, `filter_grps`, `sensor_filter`, `check_all_perc_diff_comb`

**Import direction is one-way: `capdata.py` imports from `filters.py`, never the reverse.** This is possible because filter steps only touch a `CapData` instance through the runtime `capdata` argument passed to `run()`/`_execute()` — there is no module-load-time dependency on the `CapData` class. `filters.py` imports only `param`, `pandas`, `numpy`, `sklearn` (for outliers), `captest.util`, and `pvlib.clearsky.detect_clearsky` (for `FilterClearsky`, which uses pvlib directly, not the clearsky-modeling functions below).

`capdata.py` imports the moved helpers back where its own non-filter methods still need them — `filter_grps` (in `predict_capacities`), `filter_irr` (in `ReportingIrradiance`). `ReportingIrradiance` stays in `capdata.py` (it is reporting-conditions tooling, not a row filter) and imports `filter_irr` from `filters.py`.

### `src/captest/clearsky.py` (new)

Clear-sky **modeling** (distinct from clear-sky *filtering*) is extracted from `capdata.py` into its own module:

- `pvlib_location`, `pvlib_system`, `get_tz_index`, `csky`
- The pvlib import guard for `Location`, `PVSystem`, `Array`, `FixedMount`, `SingleAxisTrackerMount`, `retrieve_sam`, `ModelChain`

Consumers are updated: `io.py` imports `csky` from `captest.clearsky`; `captest/__init__.py` re-exports `clearsky` as a submodule; test references `pvc.csky` become `clearsky.csky`. No backward-compat shim is left in `capdata.py` (pre-1.0 package; internal references are updated in the same change). Note `FilterClearsky` does **not** depend on this module — it imports `detect_clearsky` straight from pvlib.

---

## Architecture Overview

### Class Hierarchy

```
BaseSummaryStep  (param.Parameterized)
├── BaseFilter
│   ├── FilterIrr
│   ├── FilterPvsyst
│   ├── FilterShade
│   ├── FilterTime
│   ├── FilterDays
│   ├── FilterOutliers
│   ├── FilterPf
│   ├── FilterPower
│   ├── FilterCustom
│   ├── FilterSensors
│   ├── FilterClearsky
│   ├── FilterMissing
│   └── FilterRegression   # residual-outlier filter; fit_regression(filter=True) delegates to it
└── RepCond                # zero-removal summary step; computes capdata.rc
```

`BaseSummaryStep` is the common ancestor for everything that appears in the summary table. It holds the shared lifecycle (`run()`), summary attributes, and the `custom_name` parameter. `BaseFilter` extends it with the contract that `_execute()` returns a pandas `Index` of the rows to keep. `RepCond` inherits directly from `BaseSummaryStep` because it is not a row filter — its `_execute` computes `capdata.rc` and returns the unchanged index (zero rows removed). There is no separate `FitRegression` class: the residual-outlier filtering of `fit_regression(filter=True)` is realized as `FilterRegression(BaseFilter)`, and the `filter=False` path stays a plain method.

`param.Parameterized` is used as the base for `BaseSummaryStep`. This provides typed, named parameters with default values, which serves as the serialization boundary for YAML and the widget model for a future Panel GUI.

### `BaseSummaryStep`

`_execute()` returns a pandas `Index` of the rows to keep after the step. `run()` captures this as `self.ix_after` before appending `self` to `capdata.filters`. Because `data_filtered` is a property derived from `filters[-1].ix_after`, it reflects the new state as soon as `self` is appended — no direct assignment to `data_filtered` is ever needed.

Runtime state stored per step is **just `ix_after` and `pts_after`** (a pandas `Index` and its length) — these are the authoritative "rows this step kept" set by `run()` after `_execute()` returns. They are plain Python attributes, not `param` parameters, and are not serialized to YAML.

`pts_before`, `ix_before`, and `pts_removed` are **derived from the chain**, not stored on each step: a step's "before" state is the prior step's `ix_after` (or `capdata.data` for the first step in `filters`). This keeps `ix_after` as the single source of truth and avoids the trap of a step needing to snapshot its own input — `data_filtered` (a property derived from `filters[-1].ix_after`) already represents that state.

```python
class BaseSummaryStep(param.Parameterized):
    custom_name = param.String(default=None, allow_None=True,
                               doc="Optional display name in summary table.")

    def run(self, capdata):
        """Execute the step, record state, and append self to capdata.filters."""
        self.ix_after = self._execute(capdata)   # _execute returns a pandas Index
        self.pts_after = len(self.ix_after)
        capdata.filters = capdata.filters + [self]  # reassignment triggers param watchers
        if self.pts_after == 0:
            warnings.warn('The last filter removed all data!')

    def _execute(self, capdata):
        raise NotImplementedError
```

### `BaseFilter`

`BaseFilter` extends `BaseSummaryStep` with no additional interface — it exists to distinguish pure row-filtering steps from non-filter summary steps (used for card styling in the GUI and for type-checking in `rerun_from()`).

```python
class BaseFilter(BaseSummaryStep):
    pass
```

### Concrete Filter Classes

Each filter class declares its configuration as `param` parameters and implements `_execute()`. Arguments that were previously function parameters become instance attributes. `_execute()` reads `capdata.data_filtered` (the property), applies filter logic, and returns the resulting index.

**Example — `FilterIrr`:**

```python
class FilterIrr(BaseFilter):
    low     = param.Number(default=None, allow_None=True,
                           doc="Lower irradiance bound (W/m^2 or fraction of ref_val).")
    high    = param.Number(default=None, allow_None=True,
                           doc="Upper irradiance bound (W/m^2 or fraction of ref_val).")
    ref_val = param.Number(default=None, allow_None=True,
                           doc="Reference value; low/high treated as fractions if set.")
    col_name = param.String(default=None, allow_None=True,
                            doc="Column name to filter on. Inferred if None.")

    def _execute(self, capdata):
        df = capdata.data_filtered
        return filter_irr(df, ..., self.low, self.high, self.ref_val).index
```

The `args_repr` property (used to build the `filter_arguments` column in the summary) reads from `self.param.values()`, excluding runtime-only and `None`-valued fields:

```python
@property
def args_repr(self):
    skip = {'custom_name', 'name'}
    return ', '.join(
        f'{k}={v}' for k, v in self.param.values().items()
        if k not in skip and v is not None
    )
```

### Special Cases

**`FilterCustom`**: Accepts a callable `func` plus `*args` and `**kwargs` to pass through. The callable cannot be a `param` parameter (not serializable to YAML in general). `func` is stored as a plain instance attribute. `args_repr` uses the function's `__name__`, handling the `<function foo at 0x...>` case that the current wrapper handles via regex.

**`FilterSensors`**: `row_filter` callable parameter (default `check_all_perc_diff_comb`) is stored as a plain attribute, serialized as a module-qualified name string for YAML.

**`RepCond`** (implemented in chunk 5 — summary rebuild): `rep_cond` is converted to a `RepCond(BaseSummaryStep)` step so the reporting-conditions calculation appears in the summary at its position in the filter chain — letting the user see which filters preceded it. It is a **zero-removal step**: `_execute` computes `capdata.rc` as a side effect and returns the *unchanged* index (`capdata.data_filtered.index`), so it records with `pts_removed=0`.

*Implementation strategy (minimal duplication, user-approved):* the substantial reporting-conditions math is **not** rewritten into the class. The current `rep_cond` body is extracted verbatim into a private `CapData._calc_rep_cond(func, w_vel, irr_bal, percent_filter, front_poa, rc_kwargs)` helper (which sets `self.rc`/`self.rc_tool`). `RepCond._execute` delegates to it via the runtime `capdata` argument — so `filters.py` needs no import of `capdata` or `ReportingIrradiance`. `rep_cond(...)` becomes a thin wrapper: `RepCond(...).run(self)`.

`RepCond` params: `func` (`param.Parameter`, accepts dict/str/callable/None), `w_vel` (`param.Parameter`), `irr_bal` (`param.Boolean`), `percent_filter` (`param.Number`), `front_poa` (`param.String`), `rc_kwargs` (`param.Dict(default=None)` — coerced to `{}` in the helper to avoid param's shared-mutable-default pitfall). `RepCond` inherits `BaseSummaryStep` directly (not `BaseFilter`) since it is not a row filter; it still belongs in `filters` because the list accepts any `BaseSummaryStep`. `_explanation_template = "Reporting conditions were calculated (no intervals removed)."`

*YAML note (chunk 7):* the `func` dict may contain callables (`{'poa': perc_wrap(60), 't_amb': 'mean', ...}`). `perc_wrap(N)` entries serialize as `perc_N` strings (e.g. `perc_60`, the existing convention shared with `CapTest`'s `rep_conditions`); plain string values like `'mean'` serialize directly. See [[Config Round-Trip (chunk 7)]].

*Chunk-6 implication:* `RepCond` is a zero-removal step, so the visualization methods (which plot the points removed by each filter) must skip zero-removal steps.

**`FitRegression`**: The `filter=True` residual-outlier filtering has already been extracted (FilterRegression plan) into a `FilterRegression(BaseFilter)` step that `fit_regression(filter=True)` delegates to via `run()`; it records in the summary like any filter. The `filter=False` path remains a plain method that fits the model and stores `regression_results` (it changes no rows and records no step). A unified `FitRegression(BaseSummaryStep)` class is not required.

---

## `data_filtered` as a Property

### Current Behavior

`data_filtered` is initialized to `None`, set to `self.data.copy()` on load or `reset_filter()`, and mutated by each filter method. It is a full independent copy of `data` with rows progressively removed.

### New Behavior

`data_filtered` becomes a computed property:

```python
@property
def data_filtered(self):
    if not self.filters:
        return self.data
    return self.data.loc[self.filters[-1].ix_after, :]
```

This is correct because each filter's `ix_after` is derived from `capdata.data_filtered` at call time — which is itself the accumulated result of all prior filters. So `filters[-1].ix_after` always contains the current post-all-filters index.

There is no setter. Filter `_execute()` methods return an index; `run()` stores it as `self.ix_after`. No filter code assigns to `data_filtered` directly.

### pandas Copy Semantics

`data.loc[ix, :]` may return a view in some cases. Under pandas Copy-on-Write (default in pandas 3.0, opt-in in 2.0+), mutations to the result are safe — a copy is made on write. For pandas < 3.0 without CoW, a downstream consumer that mutates the returned frame (e.g. `rep_cond`/`fit_regression` adding a column) would raise `SettingWithCopyWarning`.

Note that within this architecture the property itself is never written back to: `_execute()` returns an index and column-dropping operations target `self.data` (see Impact on Other Methods). So the `.copy()` is purely defensive against downstream mutation of the returned frame, not a correctness requirement of the filter pipeline.

**Should we require pandas >= 3.0 to drop the `.copy()`?** No — not worth it under the current support matrix:

- `pyproject.toml` currently pins `pandas>=1` and `requires-python = ">=3.10"`.
- The resolved environment already splits on Python version: the lockfile pulls **pandas 3.0.2 on Python ≥ 3.11** (CoW default, `.copy()` unnecessary) and **pandas 2.3.3 on Python 3.10** (CoW opt-in, `.copy()` needed).
- **pandas 3.0 requires Python ≥ 3.11.** Requiring `pandas>=3.0` therefore forces dropping Python 3.10 support. Python 3.10 is supported until Oct 2026, so dropping it solely to avoid one defensive `.copy()` is a poor trade.

**Recommendation:** keep the unconditional `.copy()` in the property — it is correct on every supported pandas version and the cost is negligible for the access pattern (filters applied once; `data_filtered` read a handful of times in `rep_cond`/`fit_regression`):

```python
@property
def data_filtered(self):
    if not self.filters:
        return self.data
    return self.data.loc[self.filters[-1].ix_after, :].copy()
```

Separately, bumping the misleadingly-loose `pandas>=1` floor to `pandas>=2.0` (the oldest version this code is actually exercised against) is reasonable hygiene but is independent of this refactor. When Python 3.10 is eventually dropped, revisit: require `pandas>=3.0` and remove the `.copy()` to return a CoW view.

### Impact on Other Methods

All filter methods that currently do `self.data_filtered = result` are replaced by `_execute()` returning an index. Methods that modify `data_filtered` columns must instead modify `data`:

| Current operation | New operation |
|---|---|
| `self.data_filtered = df_flt` | return `df_flt.index` from `_execute()` |
| `self.data_filtered = self.data.copy()` (reset) | `self.filters = []` |
| `self.data_filtered.drop(col, inplace=True)` | `self.data.drop(col, inplace=True)` |
| `self.data_filtered = self.data_filtered[cols].copy()` (reset_agg) | `self.data = self.data[cols].copy(); self.filters = []` |
| `agg_sensors` writes to both `data` and `data_filtered` | write to `data` only; `self.filters = []` |

### `reset_filter()` Simplification

```python
def reset_filter(self):
    self.filters = []   # triggers param watcher if CapData uses param
```

No copy of `data` is needed; `data_filtered` derives from `data` automatically.

### `__copy__` Simplification

The copy method no longer needs to explicitly copy `data_filtered`. Copying `data` and `filters` is sufficient.

---

## Summary Table

### Columns

The summary table gains a `function_name` column retaining the actual class name regardless of `custom_name`:

```python
columns = ['function_name', 'pts_after_filter', 'pts_removed', 'filter_arguments']
```

`function_name` is always `type(step).__name__`. The row index uses `custom_name` if set, otherwise `function_name`, with `-N` enumeration suffix for repeated steps.

### `get_summary()`

Rebuilds the DataFrame by iterating `self.filters`. Enumeration is computed lazily; `pts_removed` is derived from the chain rather than read from a per-step attribute (see [[Chain-derived per-step counts]] below):

```python
def get_summary(self):
    rows = []
    index = []
    for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
        index.append((self.name, label))
        pts_before = self._pts_before(i)
        rows.append({
            'function_name': type(step).__name__,
            'pts_after_filter': step.pts_after,
            'pts_removed': pts_before - step.pts_after,
            'filter_arguments': step.args_repr,
        })
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=columns)
```

`_step_labels()` (defined in the Visualization Methods section) is the single source of the enumerated `custom_name`/class-name labels, shared with `scatter_filters()` and `timeseries_filters()`.

`filter_counts` is removed from `CapData.__init__` and `reset_filter()`.

#### Chain-derived per-step counts

Each step stores only `ix_after`/`pts_after`. A step's "before" count and index are derived from its predecessor in the chain — the prior step's `ix_after`, or `self.data` for the first step. `CapData` exposes two small helpers used by `get_summary()` and the visualization methods:

```python
def _ix_before(self, i):
    """Index passed *into* ``self.filters[i]`` (the chain's state just before)."""
    return self.filters[i - 1].ix_after if i > 0 else self.data.index

def _pts_before(self, i):
    return len(self._ix_before(i))
```

This is the single source of "what came into this step." Filters whose `_execute` makes nested filter calls (e.g. `FilterOutliers` calling `filter_missing` when NaN is present) get correct attribution automatically: the nested call appends its own step to `self.filters`, so the calling step's `_ix_before(i)` resolves to that nested step's `ix_after`. No per-subclass re-snapshotting is needed.

### Visualization Methods (chunk 6)

`scatter_filters()`, `timeseries_filters()`, and `get_filtering_table()` are **attribution views** — each answers "which intervals did each filter remove." They currently read `self.removed[0]['index']` and iterate `self.kept`. In chunk 6 all three are rewritten to derive everything from `self.filters`, and the `self.removed`/`self.kept` lists are **deleted entirely** — from `CapData.__init__`, `reset_filter()`, `agg_sensors()`, and `process_regression_columns()` — along with `BaseSummaryStep._record_removed_kept` and its call site in `run()`. After this, `run()` simply appends the step to `self.filters` (it no longer mirrors removed/kept).

**Removed-by-filter, not cumulative-kept.** The current methods layer the cumulative *kept* set after each step (each scatter shows the points that survived up to step `i`, labeled with step `i+1`'s name), relying on lower layers showing through to imply what was removed. The new behavior plots, for each step, exactly the rows that step removed — derived from the chain via `_ix_before`:

```
removed_ix(step_i) = self._ix_before(i).difference(self.filters[i].ix_after)
```

Because filters run sequentially and removed rows never reappear, each step's removed set is disjoint from every other step's. Together with the rows retained after all filters, the steps form a complete partition of the original index:

```
data.index = retained ⊎ removed(f0) ⊎ removed(f1) ⊎ … ⊎ removed(fN)
```

This makes each plotted layer directly attributable to a single filter, rather than inferred from overlapping cumulative layers.

**Shared helper: `_removed_by_step()`.** The per-step removal computation is factored into one method on `CapData` so the two plots and the filtering table agree on a single source of truth:

```python
def _removed_by_step(self):
    """Per-step removal attribution for the visualization methods.

    Returns ``(i, label, removed_ix)`` for each filter step that removed at
    least one interval, where ``removed_ix = _ix_before(i) \\ filters[i].ix_after``
    and ``label`` is the step's ``_step_labels()`` entry. Zero-removal steps
    (always ``RepCond``; also any filter that matched everything) are skipped —
    they have nothing to attribute. ``i`` is returned so callers can recover the
    step's input set via ``_ix_before(i)`` and its survivors via
    ``self.filters[i].ix_after``.
    """
    out = []
    for i, (step, label) in enumerate(zip(self.filters, self._step_labels())):
        removed_ix = self._ix_before(i).difference(step.ix_after)
        if len(removed_ix) > 0:
            out.append((i, label, removed_ix))
    return out
```

**Zero-removal steps are skipped in attribution views (decision).** Comprehensive views — `get_summary()` and `describe_filters()` — list *every* step, including `RepCond` (shown with `pts_removed=0`) and any no-op filter; they reconcile the difference for the reader. The attribution views show only steps that removed intervals. Nothing is plot-only: a `rep_cond()` call appears as a `pts_removed=0` row in `get_summary` and a line in `describe_filters` (which is where its chain position is communicated), but contributes no plot layer or table column. An empty plotted layer would be invisible on the canvas anyway, leaving only a dead legend entry; skipping keeps the views focused on filters that actually removed data.

**`scatter_filters()` / `timeseries_filters()`:** a baseline `retained` layer plus one layer per `_removed_by_step()` entry:

```python
def scatter_filters(self):
    data = self.get_reg_cols(reg_vars=["power", "poa"], filtered_data=False)
    data["index"] = self.data.index

    scatters = []
    # baseline: rows retained after all filters (or all rows if no filters)
    retained_ix = self.filters[-1].ix_after if self.filters else self.data.index
    scatters.append(
        hv.Scatter(data.loc[retained_ix, :], "poa", ["power", "index"]).relabel("retained")
    )
    # one layer per filter that removed something
    for _i, label, removed_ix in self._removed_by_step():
        scatters.append(
            hv.Scatter(data.loc[removed_ix, :], "poa", ["power", "index"]).relabel(label)
        )
    return hv.Overlay(scatters).opts(...)
```

`timeseries_filters()` follows the same structure with `hv.Curve`/`hv.Scatter` on the `Timestamp` axis.

**`get_filtering_table()`:** the per-interval attribution matrix is rebuilt from `_removed_by_step()`. For each surviving step, mark its column `0` on the step's survivors (`self.filters[i].ix_after`) and `1` on its removed rows; rows absent from the step's input (`_ix_before(i)`) stay `NaN` (removed by an earlier filter). The old `i == 0` special case disappears because `_ix_before(0)` is `self.data.index`. Zero-removal steps get no column (consistent with the plots). The trailing `all_filters` boolean column is unchanged.

```python
def get_filtering_table(self):
    filtering_data = pd.DataFrame(index=self.data.index)
    for i, label, removed_ix in self._removed_by_step():
        filtering_data.loc[self.filters[i].ix_after, label] = 0
        filtering_data.loc[removed_ix, label] = 1
    filtering_data["all_filters"] = filtering_data.apply(lambda x: all(x == 0), axis=1)
    return filtering_data
```

**`get_length_test_period()` retargeting.** This method currently finds the test period by matching the `"FilterTime"` label string against `self.kept`. With `removed`/`kept` deleted — and because step labels honor `custom_name`/`-N` enumeration, so a `custom_name`'d `FilterTime` would no longer match the bare label — it is retargeted to iterate `self.filters` and test `isinstance(step, FilterTime)`, using the **first** matching step's `ix_after` for the period and then breaking — preserving the documented "subsequent uses of `FilterTime` are ignored" behavior. This is robust to `custom_name` and independent of display labels.

**Shared label enumeration.** Both the visualization methods and `get_summary()` use the same per-step display labels (`custom_name` or class name, with `-N` suffix for repeats), already factored into the `_step_labels()` helper during the summary rebuild:

```python
def _step_labels(self):
    labels, seen = [], {}
    for step in self.filters:
        base = step.custom_name or type(step).__name__
        n = seen.get(base, 0)
        seen[base] = n + 1
        labels.append(base if n == 0 else f"{base}-{n}")
    return labels
```

---

## Filter Explanations

Each step can produce a human-readable sentence describing its effect, with the *effective* (resolved) values substituted in. These are aggregated by a new `CapData.describe_filters()` method into a written summary of a filtering run — complementary to the tabular `get_summary()`.

### `explanation` property and template

Explanation text is **class-intrinsic boilerplate, not user configuration**, so it is a class-level template attribute plus a property — *not* a `param` (a `param` would be serialized to YAML; see [[Serialization Boundary]]). This mirrors the treatment of `_legacy_name` and runtime resolved values.

```python
class BaseSummaryStep(param.Parameterized):
    _explanation_template = None  # class-level; set by concrete subclasses

    @property
    def explanation(self):
        """Human-readable description of the step's effect (read after run()).

        Renders `_explanation_template` with `_explanation_values()`. Returns
        None when no template is defined. Subclasses whose phrasing depends on
        which params are set (e.g. FilterTime, FilterClearsky, FilterCustom)
        override this property directly instead of relying on a flat template.
        """
        if self._explanation_template is None:
            return None
        return self._explanation_template.format(**self._explanation_values())

    def _explanation_values(self):
        """Substitution mapping for `_explanation_template`.

        Defaults to `_args_for_repr()`; subclasses override to supply
        run-time-resolved values (resolved column names, effective bounds).
        """
        return self._args_for_repr()
```

**Rendered with resolved/effective values, post-run.** Like `args_repr`, the explanation is meaningful only after `run()` has resolved runtime values. Unlike `args_repr` (which reproduces the *call arguments* for the summary's `filter_arguments` column), the explanation describes the *effect* using resolved column names and effective numeric bounds. For example, `FilterIrr` stores the resolved POA column and the effective absolute bounds (`low * ref_val` / `high * ref_val` when `ref_val` is set) during `_execute`, and overrides `_explanation_values()` to supply them:

```python
class FilterIrr(BaseFilter):
    _explanation_template = (
        "Intervals where {col_name} is below {low} or above {high} W/m^2 "
        "were removed."
    )

    def _explanation_values(self):
        return {
            "col_name": self.col_name_resolved,
            "low": self.low_effective,
            "high": self.high_effective,
        }
```

### `CapData.describe_filters()`

Iterates `self.filters` and joins each step's non-None `explanation` into a written summary, one sentence per line:

```python
def describe_filters(self):
    """Return a written, human-readable summary of the filtering run."""
    lines = [
        step.explanation for step in self.filters if step.explanation is not None
    ]
    return "\n".join(lines)
```

**Transition note:** until every filter is class-based (end of the rest-of-filters plan), `describe_filters()` only reports steps that route through `filters` (class-based ones). Filters still using `@update_summary` do not appear. This is an interim limitation of the incremental rollout, resolved once all filters are converted; it does not affect the tabular `get_summary()`, which keeps reading the mirrored legacy lists during the transition.

---

## CapData Changes

### Attribute Changes

| Removed | Replacement |
|---|---|
| `summary` (list) | derived from `self.filters` in `get_summary()` |
| `summary_ix` (list) | derived from `self.filters` in `get_summary()` |
| `removed` (list) | derived from `self.filters` in visualization methods |
| `kept` (list) | derived from `self.filters` in visualization methods |
| `filter_counts` (dict) | computed lazily in `get_summary()` |

Added:
- `filters = param.List(default=[], item_type=BaseSummaryStep)` — the single source of truth for the filter pipeline. This requires `CapData` to inherit `param.Parameterized` (so the list is watchable for the GUI). Consequence: `param` reserves a **constant** `name` parameter, so `name` is passed via `super().__init__(name=name)` and cannot be reassigned afterward — `copy()` constructs the copy with the name directly rather than assigning `cd_c.name`.
- `describe_filters()` — written, human-readable summary of the filtering run, built from each step's `explanation` (see [[Filter Explanations]]).

### Thin Wrapper Methods

Existing `CapData` filter methods are retained as thin wrappers for backwards compatibility. Each instantiates the corresponding filter class and calls `run()`:

```python
def filter_irr(self, low, high, ref_val=None, col_name=None, custom_name=None):
    FilterIrr(low=low, high=high, ref_val=ref_val, col_name=col_name,
              custom_name=custom_name).run(self)
```

The `filter_name` kwarg from PR #119 is renamed to `custom_name` in the new API. The thin wrappers accept both names during a transition period.

The `inplace` parameter is dropped from all filter methods — the refactor removes `inplace=False` support entirely.

### New API

Filter classes can be used directly, enabling pipeline construction and reuse:

```python
# Direct use
FilterIrr(low=200, high=800, custom_name="Irradiance bounds").run(cd)

# Pipeline
pipeline = [
    FilterIrr(low=200, high=800),
    FilterPvsyst(),
    RepCond(percent_filter=20),
    FilterRegression(n_std=2),
]
for step in pipeline:
    step.run(cd)
```

### `rerun_from(index)`

Supports re-running the pipeline from a given position (used by GUI edit interactions and by pipeline reload from YAML):

```python
def rerun_from(self, index):
    """Re-execute all steps from filters[index] onwards using stored parameters."""
    steps = self.filters[index:]
    self.filters = self.filters[:index]   # trim to before the changed step
    for step in steps:
        step.run(self)
```

---

## Config Round-Trip (chunk 7)

### Goal and architecture

The driving goal is a **single config file** that captures both the test-level parameters *and* the applied filtering steps, so a future GUI (or an ipynb user) can export it and a different user can reload it — in the GUI or a notebook — to re-run the test identically.

`CapData` owns its filter-pipeline serialization as reusable, I/O-free building blocks; `CapTest` integrates both the `meas` and `sim` pipelines into the single `captest:` config file it already writes (`CapTest.to_yaml`/`from_yaml`). There is no separate pipeline file and no new top-level key — the pipelines live inside the existing `captest:` sub-mapping, so one file round-trips everything.

### Serialization Boundary

`param` parameters on each filter class are the serialization boundary. Runtime state (`pts_after`, `ix_after`, and the chain-derived `pts_before`/`ix_before`/`pts_removed`) is never serialized. **Every param is exported explicitly — defaults and `None` values included** — so a reviewer can read the config and understand exactly what each step does without having memorized the class defaults. Only the param-system `name` identity is omitted.

### Per-step encode/decode

Each filter class owns its (de)serialization so callable-handling lives next to the class that needs it:

```python
class BaseSummaryStep:
    def to_config(self):
        """Return a yaml-safe dict: {'type': ClassName, **all params except name}."""
        d = {"type": type(self).__name__}
        d.update({k: v for k, v in self.param.values().items() if k != "name"})
        return d

    @classmethod
    def from_config(cls, d):
        """Build an instance from a to_config() dict (already has 'type' popped)."""
        return cls(**d)
```

`FilterCustom`, `FilterSensors`, and `RepCond` override `to_config`/`from_config` to encode/decode their callable params (see Callable Parameters below). All params are emitted; e.g. a `FilterIrr` exports `low`, `high`, `ref_val`, `col_name`, and `custom_name` even when some are `None`.

### Filter Registry

A module-level registry in `filters.py` maps class-name strings to classes, with a `step_from_config` helper used by deserialization:

```python
FILTER_REGISTRY = {
    'FilterIrr': FilterIrr, 'FilterPvsyst': FilterPvsyst, 'FilterShade': FilterShade,
    'FilterTime': FilterTime, 'FilterDays': FilterDays, 'FilterOutliers': FilterOutliers,
    'FilterPf': FilterPf, 'FilterPower': FilterPower, 'FilterCustom': FilterCustom,
    'FilterSensors': FilterSensors, 'FilterClearsky': FilterClearsky,
    'FilterMissing': FilterMissing, 'FilterRegression': FilterRegression, 'RepCond': RepCond,
}

def step_from_config(d):
    d = dict(d)
    cls_name = d.pop("type")
    if cls_name not in FILTER_REGISTRY:
        # Fuzzy-match the bad name against the registry (stdlib difflib) to
        # suggest the likely intended type — helps GUI/hand-edited configs.
        suggestion = difflib.get_close_matches(cls_name, FILTER_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(
            f"Unknown filter type {cls_name!r} in pipeline config.{hint}"
        )
    return FILTER_REGISTRY[cls_name].from_config(d)
```

### CapData building blocks

```python
def filters_to_config(self):
    """Serialize this CapData's filter chain to a list of dicts."""
    return [f.to_config() for f in self.filters]

def run_pipeline(self, config):
    """Build each step from its config dict and run it on this CapData."""
    for d in config:
        step_from_config(d).run(self)
```

These are pure data ⇄ filters (no file I/O), so they compose into `CapTest`'s single-file YAML and are independently testable. Standalone notebook use:

```python
cd = CapData('system_a'); cd.load_data(...)
cd.process_regression_columns()
cd.run_pipeline(config['captest']['meas_filters'])
```

### CapTest integration: one file

`CapTest._build_yaml_sub_mapping` adds `meas_filters` and `sim_filters` (from `self.meas.filters_to_config()` / `self.sim.filters_to_config()`) to the `captest:` sub-mapping it already builds; `CapTest.from_mapping` (reached by `from_yaml`) rebuilds and re-applies them **after** data load and `setup()` (filters need `regression_cols`/data in place), via `run_pipeline`. The result: a single `from_yaml(..., meas_loader=, sim_loader=)` call reconstructs the whole test.

```yaml
captest:
  test_setup: e2848_default
  meas_path: ./meas.csv
  sim_path: ./sim.csv
  overrides: {}                 # rep_conditions omitted — see below
  ac_nameplate: 6000000
  # ... remaining scalar params ...
  meas_filters:
    - {type: FilterIrr, low: 200, high: 800, ref_val: null, col_name: null, custom_name: null}
    - {type: RepCond, func: {poa: perc_60, t_amb: mean, w_vel: mean},
       w_vel: null, irr_bal: false, percent_filter: 20, front_poa: poa,
       rc_kwargs: null, custom_name: null}
  sim_filters:
    - {type: FilterPvsyst, custom_name: null}
```

**`rep_conditions` vs `RepCond` step (decision B).** `rep_conditions` is a CapTest-level param (the default feeding `CapTest.rep_cond`); a `RepCond` pipeline step is the *applied* reporting-conditions calculation at its chain position. To keep the file unambiguous about when and with what arguments reporting conditions run, `_build_yaml_sub_mapping` **omits `overrides.rep_conditions` whenever a `RepCond` step is present in `meas.filters` or `sim.filters`** — the step(s) are then the single source of truth. When no `RepCond` step exists, `overrides.rep_conditions` is written as today. Because `setup()` never auto-runs `rep_cond`, replaying a pipeline that contains a `RepCond` step applies it exactly once (no double-apply).

### Callable Parameters

Three filter types accept callables that cannot be directly serialized. The `perc_wrap` factory and the `perc_N` string helpers (`_perc_wrap_to_string`, `_resolve_func_strings`) move from `captest.py` to `util.py` (re-exported from `captest.captest` for backward compatibility) so `RepCond` (in `filters.py`) and the existing `CapTest` `rep_conditions` serialization share one implementation. `util.py` is import-safe for this (it imports no captest/capdata modules).

| Class | Parameter | Serialization approach |
|---|---|---|
| `FilterCustom` | `func` (required) | Module-qualified name string (`pkg.module.func_name`); deserialize via import. Lambdas/closures are not importable — `to_config` **raises** a clear `ValueError` for them (a GUI/ipynb pipeline is limited to importable named functions). |
| `FilterSensors` | `row_filter` (default `check_all_perc_diff_comb`) | Module-qualified name; deserialize via import. The default round-trips losslessly (`captest.filters.check_all_perc_diff_comb`). |
| `RepCond` | `func` dict (`{'poa': perc_wrap(60), ...}`) | `perc_wrap(N)` entries → `"perc_N"` strings (the existing convention); plain string values (`'mean'`) serialize directly; `None` (the default) → `null`. |

### Errors

Deserialization raises explicit errors: an unknown `type` not in `FILTER_REGISTRY`, a non-importable module-qualified callable name, or (on export) an unserializable lambda/closure each raise with a message naming the offending value. For an unknown `type`, the message includes a `difflib.get_close_matches` suggestion of the nearest valid type name when one is close enough (e.g. `FilterIrradiance` → "Did you mean 'FilterIrr'?").

---

## Implementation Sequencing

Both prerequisites for this refactor are now **satisfied** as of the v0.15.0 / v0.15.1 release (2026-05-12). The refactor is clear to begin from the current branch (descended from `v0.15.1`).

1. **Release current master as a stable version** — ✅ Done. `v0.15.0` and `v0.15.1` are tagged and released. The refactor branch starts from released, stable master.
2. **Merge `agg-in-process-reg-cols` onto master** — ✅ Done. The final `agg_sensors` is in place, decomposed into `agg_group` + `expand_agg_map`, and `process_regression_columns` was added (both shipped in `v0.15.0`). Confirmed at the `v0.15.0` tag: `agg_group`, `expand_agg_map`, `agg_sensors`, and `process_regression_columns` are all present. `agg_sensors` still writes to `data`, `data_filtered`, and `regression_cols` — i.e. the final implementation, so the `data_filtered` interaction only needs reworking once (per the "Impact on Other Methods" table: write to `data` only, then `self.filters = []`).

The earlier cherry-pick-vs-rebase concern (`agg-in-process-reg-cols` diverged before the uv migration) is now moot — that work is already on released master.

---

## Panel GUI Enablement Notes

This refactor is a prerequisite for a reactive Panel GUI. It is not the GUI design itself — that is a separate effort — but the following properties of this architecture directly enable it:

**Filter objects as the widget model.** Each filter class declares its configuration as typed `param` parameters. `pn.Param(filter_instance)` auto-generates an appropriate widget panel (number sliders for `low`/`high`, text input for `custom_name`, etc.) with no per-filter UI code. The filter parameters *are* the widget model — there is no separate representation to keep in sync.

**`cd.filters` as the UI data model.** A Panel sortable list widget renders each filter in `cd.filters` as an expandable card backed by `pn.Param(filter_instance)`. Add/remove buttons append to or remove from `cd.filters`. Reordering calls `rerun_from()`.

**Reactive data updates via the `data_filtered` property.** When a user edits a filter attribute via a slider, a `param.watch` callback triggers `capdata.rerun_from(index)`. Because `data_filtered` is a property derived from `filters[-1].ix_after`, it reflects the new state automatically. Any Panel pane bound to `cd.data_filtered` or `cd.get_summary()` updates without manual sync.

**`param.List` mutation note.** `param` observes list reassignment, not in-place mutation. `cd.filters.append(f)` does not trigger watchers; `cd.filters = cd.filters + [f]` does. `BaseSummaryStep.run()` uses reassignment for this reason. The GUI layer should do the same.

**YAML as the shared config format.** The GUI exports `cd.to_yaml()`. A notebook or script loads the same file with `CapData.load_pipeline()`. The user works in either environment on the same configuration, with no translation layer between them.
