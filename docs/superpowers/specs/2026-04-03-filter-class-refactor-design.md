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
- All concrete filter classes (`FilterIrr` … `FilterMissing`), plus `RepCond` and `FitRegression`
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
│   └── FilterMissing
├── RepCond
└── FitRegression
```

`BaseSummaryStep` is the common ancestor for everything that appears in the summary table. It holds the shared lifecycle (`run()`), summary attributes, and the `custom_name` parameter. `BaseFilter` extends it with the contract that `_execute()` returns a pandas `Index` of the rows to keep. `RepCond` and `FitRegression` inherit directly from `BaseSummaryStep` because they are not pure row filters.

`param.Parameterized` is used as the base for `BaseSummaryStep`. This provides typed, named parameters with default values, which serves as the serialization boundary for YAML and the widget model for a future Panel GUI.

### `BaseSummaryStep`

`_execute()` returns a pandas `Index` of the rows to keep after the step. `run()` captures this as `self.ix_after` before appending `self` to `capdata.filters`. Because `data_filtered` is a property derived from `filters[-1].ix_after`, it reflects the new state as soon as `self` is appended — no direct assignment to `data_filtered` is ever needed.

Runtime state (`pts_before`, `pts_after`, `pts_removed`, `ix_before`, `ix_after`) is set by `run()` as plain Python attributes, not `param` parameters. They are not serialized to YAML.

```python
class BaseSummaryStep(param.Parameterized):
    custom_name = param.String(default=None, allow_None=True,
                               doc="Optional display name in summary table.")

    def run(self, capdata):
        """Execute the step, record state, and append self to capdata.filters."""
        self.pts_before = capdata.data_filtered.shape[0]
        self.ix_before = capdata.data_filtered.index
        self.ix_after = self._execute(capdata)   # _execute returns a pandas Index
        self.pts_after = len(self.ix_after)
        self.pts_removed = self.pts_before - self.pts_after
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

**`RepCond`**: The `func` dict parameter (`{'poa': perc_wrap(60), 't_amb': 'mean', 'w_vel': 'mean'}`) contains callables. The `perc_wrap` entries are serialized as `perc_wrap(N)` strings; string values like `'mean'` serialize directly.

**`FitRegression`**: The `filter=True` path currently filters data internally. Under this design, when `filter=True`, `FitRegression._execute()` applies the internal filter to `capdata.data_filtered` and the resulting index is stored in `self.ix_after`. `FitRegression` inherits from `BaseSummaryStep` (not `BaseFilter`), but the `filter=True` path makes it behave like one for summary purposes.

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

Rebuilds the DataFrame by iterating `self.filters`. Enumeration is computed lazily:

```python
def get_summary(self):
    rows = []
    index = []
    for step, label in zip(self.filters, self._step_labels()):
        index.append((self.name, label))
        rows.append({
            'function_name': type(step).__name__,
            'pts_after_filter': step.pts_after,
            'pts_removed': step.pts_removed,
            'filter_arguments': step.args_repr,
        })
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=columns)
```

`_step_labels()` (defined in the Visualization Methods section) is the single source of the enumerated `custom_name`/class-name labels, shared with `scatter_filters()` and `timeseries_filters()`.

`filter_counts` is removed from `CapData.__init__` and `reset_filter()`.

### Visualization Methods

`scatter_filters()` and `timeseries_filters()` currently read from `self.removed[0]['index']` and iterate `self.kept`. Two changes apply:

1. **Read from `self.filters` directly.** `self.removed` and `self.kept` are removed from `CapData.__init__` and `reset_filter()`; both methods derive everything from `self.filters`.
2. **Plot the points *removed by* each filter.** The current methods layer the cumulative *kept* set after each step (each scatter shows the points that survived up to step `i`, labeled with step `i+1`'s name), relying on lower layers showing through to imply what was removed. The new behavior plots, for each step, exactly the rows that step removed:

   ```
   removed_ix(step) = step.ix_before.difference(step.ix_after)
   ```

   Because filters run sequentially and removed rows never reappear, each step's removed set is disjoint from every other step's. Together with the rows retained after all filters, the steps form a complete partition of the original index:

   ```
   data.index = retained ⊎ removed(f0) ⊎ removed(f1) ⊎ … ⊎ removed(fN)
   ```

   This makes each plotted layer directly attributable to a single filter, rather than inferred from overlapping cumulative layers.

**New plotting structure (applies to both methods):**

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

    # one layer per filter: the rows that filter removed
    for step, label in zip(self.filters, self._step_labels()):
        removed_ix = step.ix_before.difference(step.ix_after)
        scatters.append(
            hv.Scatter(data.loc[removed_ix, :], "poa", ["power", "index"]).relabel(label)
        )

    return hv.Overlay(scatters).opts(...)
```

`timeseries_filters()` follows the same structure with `hv.Curve`/`hv.Scatter` on the `Timestamp` axis.

**Shared label enumeration.** Both visualization methods and `get_summary()` need the same per-step display labels (`custom_name` or class name, with `-N` suffix for repeats). This is factored into a `_step_labels()` helper so the enumeration logic lives in one place:

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

`get_summary()` is updated to consume `_step_labels()` rather than recomputing enumeration inline.

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
    FitRegression(),
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

## YAML Round-Trip

### Serialization Boundary

`param` parameters on each filter class are the serialization boundary. Runtime state (`pts_before`, `pts_after`, `ix_before`, `ix_after`) is stored as plain Python attributes and is never serialized.

### YAML Format

```yaml
captest:
  name: system_a_2023
  filters:
    - type: FilterIrr
      custom_name: Irradiance bounds
      low: 200
      high: 800
    - type: FilterPvsyst
    - type: FilterTime
      start: "2023-03-01"
      end: "2023-09-30"
    - type: RepCond
      percent_filter: 20
      irr_bal: false
    - type: FitRegression
      filter: false
```

Omitted `param` values use class defaults on load — no need to serialize defaults explicitly.

### Filter Registry

A module-level registry maps class name strings to classes, used for deserialization:

```python
FILTER_REGISTRY = {
    'FilterIrr': FilterIrr,
    'FilterPvsyst': FilterPvsyst,
    'FilterShade': FilterShade,
    'FilterTime': FilterTime,
    'FilterDays': FilterDays,
    'FilterOutliers': FilterOutliers,
    'FilterPf': FilterPf,
    'FilterPower': FilterPower,
    'FilterCustom': FilterCustom,
    'FilterSensors': FilterSensors,
    'FilterClearsky': FilterClearsky,
    'FilterMissing': FilterMissing,
    'RepCond': RepCond,
    'FitRegression': FitRegression,
}
```

### Serialization

```python
def to_yaml(self, path):
    steps = []
    for f in self.filters:
        d = {'type': type(f).__name__}
        d.update({
            k: v for k, v in f.param.values().items()
            if k not in ('name',) and v is not None
        })
        steps.append(d)
    config = {'captest': {'name': self.name, 'filters': steps}}
    with open(path, 'w') as fh:
        yaml.dump(config, fh, default_flow_style=False)
```

### Deserialization

```python
@classmethod
def load_pipeline(cls, path):
    """Return a list of filter instances from a YAML pipeline definition."""
    with open(path) as fh:
        config = yaml.safe_load(fh)
    steps = []
    for d in config['captest']['filters']:
        d = dict(d)
        cls_name = d.pop('type')
        steps.append(FILTER_REGISTRY[cls_name](**d))
    return steps

def run_pipeline(self, pipeline):
    for step in pipeline:
        step.run(self)
```

Typical notebook usage:

```python
cd = CapData('system_a')
cd.load_data(...)
pipeline = CapData.load_pipeline('captest_config.yaml')
cd.run_pipeline(pipeline)
cd.get_summary()
```

### Callable Parameters

Three filter types accept callables that cannot be directly serialized:

| Class | Parameter | Serialization approach |
|---|---|---|
| `FilterCustom` | `func` (required, arbitrary) | Module-qualified name string (`mypackage.module.func_name`). Lambdas and closures are not supported for YAML round-trip; `FilterCustom` in a YAML pipeline is limited to importable named functions. |
| `FilterSensors` | `row_filter` (default: `check_all_perc_diff_comb`) | Serialize as module-qualified name; deserialize via `importlib.import_module`. Default round-trips losslessly. |
| `RepCond` | `func` dict (`{'poa': perc_wrap(60), ...}`) | `perc_wrap(N)` entries serialized as `perc_wrap:60` or similar tagged string; plain string values (`'mean'`) serialize directly. |

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
