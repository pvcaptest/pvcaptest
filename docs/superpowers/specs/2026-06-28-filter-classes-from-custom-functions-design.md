# Promote custom filter functions to first-class filter classes

**Date:** 2026-06-28
**Branch:** `filters-new-classes` (off `filters-refactor`)
**Status:** Approved design — pending implementation plan

## Background

Several row filters were written and used in test notebooks prior to the
`filters-refactor` work and are currently applied by passing plain functions to
`CapData.filter_custom`. The source of truth for these is
`untracked_bin/filters_convert_custom_to_filter_classes.ipynb`. They are:

- `unstable_irr_filter(df, irr_col, window, threshold)` — rolling-window std of
  an irradiance column; removes intervals where the std is at or above a
  threshold (variable/unstable irradiance).
- `filter_abs_perc_diff_prev_interval(data, column, threshold=0.05)` — removes
  intervals where the absolute fractional change from the previous interval
  exceeds a threshold (a step-change / stability filter).
- `remove_inter_row_shading(data, boolean_column='backtrack_on')` — removes
  intervals where a boolean column (e.g. tracker backtracking / inter-row
  shading flag) is `True`.
- `filter_avail(data, avail_col, threshold)` — keeps intervals where an
  availability column exceeds a threshold.

The notebook also flags that the absolute-W/m² sensor-disagreement variant —
`filter_sensors(..., row_filter=abs_diff_from_average)` — is awkward to invoke
(the `perc_diff` kwarg name is misleading because the values are absolute
W/m², not percentages) and asks for a class-backed, GUI-friendly way to choose
it.

The goal is to make all of this available as first-class filter functionality —
real filter step classes and thin `CapData.filter_*` wrappers — so it is
available to all `pvcaptest` users and to a future GUI, with no need to define
functions and pass them to `filter_custom`.

## Goals

- Promote the four custom functions to first-class filter functionality
  following the established `filters.py` step-class pattern.
- Add a class-backed, GUI-renderable way to select the sensor-disagreement
  comparison method (percent-difference vs absolute-difference vs a custom
  callable).
- Full serialization parity with existing filters (`to_config`/`from_config`,
  `step_from_config`, `filters_to_config`/`run_pipeline`, YAML round-trip).

## Non-goals

- The visualization helper sketched in the notebook (a plot of what the
  step-change / unstable-irradiance filters do) is out of scope.
- Backward compatibility for the existing `Sensors`/`filter_sensors` API is
  **not** required on this branch; the `Sensors` change is intentionally
  breaking in service of consistent internals.

## Design overview

The established pattern (see the `filters-refactor` work) is: each filter is a
`BaseFilter` subclass declaring its config as `param` parameters, implementing
`_execute(capdata)` to return the kept `pandas.Index`; `run()` records
`ix_after`/`pts_after` and appends the step to `capdata.filters`. Classes are
registered in `FILTER_REGISTRY` for (de)serialization, and a thin
`CapData.filter_*` wrapper instantiates the class and calls `run()`. All new
work follows this pattern so summary rendering, `explanation`, and YAML
round-trip come for free.

### Summary of changes

| New class | `CapData` wrapper | Backed by |
| --- | --- | --- |
| `RollingStd` | `filter_rolling_std` | new class |
| `AbsDiffPrev` | `filter_abs_diff_prev` | new class |
| `BooleanFlag` | `filter_flag` | new class |
| — | `filter_threshold` | existing `Irradiance` |
| — | `filter_sensors_abs_diff` | existing `Sensors` (enhanced) |

Three new classes are added to `FILTER_REGISTRY`. `Irradiance` and `Sensors` are
already registered, so the two reuse wrappers need no registry change.

## New filter classes (`src/captest/filters.py`)

### `RollingStd(BaseFilter)`

Removes intervals where the rolling-window standard deviation of a column is at
or above `threshold` (unstable / variable irradiance). Replaces
`unstable_irr_filter`.

Parameters:
- `column` — `param.String(default=None, allow_None=True)`. `None` infers the
  POA column via `capdata._get_poa_col()` (same inference `Irradiance.col_name`
  uses).
- `window` — `param.Parameter(default=None)`. An `int` row count or a pandas
  offset alias string (e.g. `'10min'`), passed straight to
  `DataFrame.rolling`. Required.
- `threshold` — `param.Number(default=None, allow_None=True)`. Required.

`_execute`:
```python
col = self.column if self.column is not None else capdata._get_poa_col()
if self.window is None or self.threshold is None:
    raise ValueError("RollingStd requires both window and threshold.")
df = capdata.data_filtered
std = df[col].rolling(self.window).std()
self.column_resolved = col
return df.index[std < self.threshold]
```

Parity note: the leading rows of the window produce `NaN` std, and
`NaN < threshold` is `False`, so those rows are dropped — matching
`unstable_irr_filter`. Pinned by a parity test.

`_explanation_template`: `"Intervals where the rolling std (window={window}) of
{column} was at or above {threshold} were removed."` `_explanation_values`
substitutes `column_resolved`.

### `AbsDiffPrev(BaseFilter)`

Removes intervals where the absolute fractional change of a column from the
previous interval exceeds `threshold`. Replaces
`filter_abs_perc_diff_prev_interval`.

Parameters:
- `column` — `param.String(default=None, allow_None=True)`. `None` infers POA.
- `threshold` — `param.Number(default=0.05)`.

`_execute`:
```python
col = self.column if self.column is not None else capdata._get_poa_col()
df = capdata.data_filtered
s = df[col]
abs_diff = (s.diff() / s).abs()
self.column_resolved = col
return df.index[abs_diff <= self.threshold]
```

Parity note: division by the current value matches the original
(`abs(diff / column)`); the first row's `diff` is `NaN` and is dropped — matches
`filter_abs_perc_diff_prev_interval`. The original returned a frame with added
`*_diff`/`*_abs_diff` columns; only the index is needed here (`filter_custom`
already discards non-index column changes), so those helper columns are not
produced.

`_explanation_template`: `"Intervals where {column} changed by more than
{threshold} (fractional) from the previous interval were removed."`

### `BooleanFlag(BaseFilter)`

Drops intervals where a boolean/flag column is truthy. Replaces
`remove_inter_row_shading` (generalized: any boolean column, no hard-coded
`backtrack_on` default). Distinct from the existing `Shade` filter, which is
PVsyst-specific (`FShdBm` shading fraction).

Parameters:
- `column` — `param.String(default=None, allow_None=True)`. Required at run
  time (a `None` column raises a clear `ValueError`).
- `invert` — `param.Boolean(default=False)`. Flips the filter: with the default
  `False`, rows where `column` is truthy are removed (keep the `False` rows);
  with `True`, rows where `column` is falsy are removed (keep the `True` rows).

`_execute`:
```python
if self.column is None:
    raise ValueError("BooleanFlag requires a column.")
df = capdata.data_filtered
mask = df[self.column].astype(bool)
keep = mask if self.invert else ~mask
return df.index[keep]
```

`.astype(bool)` coerces 0/1, real booleans, and `NaN` (truthy) consistently;
covered by a test. `_explanation_template` (phrasing depends on `invert`, so the
`explanation` property is overridden rather than templated): removed intervals
are those flagged `True` in `{column}` when `invert=False`, or those flagged
`False` when `invert=True`.

### Serialization

`RollingStd`, `AbsDiffPrev`, and `BooleanFlag` declare only scalar/string
parameters (`window` may be an `int` or an offset string; `util.to_native`
already coerces numpy scalars), so they inherit `BaseSummaryStep.to_config` /
`from_config` unchanged. Each is added to `FILTER_REGISTRY` under its class
name.

## `Sensors` enhancement (`src/captest/filters.py`)

Unify the comparison choice into a single GUI-renderable parameter, with the
custom callable as a third option rather than a separate kwarg. This is a
**breaking change** (acceptable on this branch).

Parameters (replacing today's `perc_diff` dict + `row_filter` callable):
- `method` — `param.Selector(default='percent_diff',
  objects=['percent_diff', 'abs_diff'], check_on_set=False)`. The two built-ins
  render as a dropdown for a future GUI; `check_on_set=False` also allows a
  power user to assign a **callable** into the same slot (the "custom" third
  option), with the row-filter signature `func(series, threshold) -> bool`.
- `thresholds` — `param.Dict(default=None, allow_None=True)`. Maps a sensor-group
  key to its threshold (renamed from `perc_diff`, since values are not
  percentages under `abs_diff`/custom). For `percent_diff` the values are
  decimal fractions (e.g. `0.05`); for `abs_diff` they are absolute W/m².

Internals:
```python
_BUILTIN_COMPARISONS = {
    "percent_diff": check_all_perc_diff_comb,
    "abs_diff": abs_diff_from_average,
}

def _resolve_comparison(self):
    if callable(self.method):
        return self.method
    return self._BUILTIN_COMPARISONS[self.method]
```

`_execute` resolves the comparison via `_resolve_comparison()` and applies the
existing `sensor_filter` machinery across each group in `thresholds`. Default
resolution when `thresholds is None`: `{<poa group>: 0.05}` only for
`percent_diff`; `abs_diff` (and a custom callable) raise a clear `ValueError`
when `thresholds` is `None`, since there is no universal absolute default.

Serialization (`to_config`/`from_config` override): `method` is emitted as the
plain string for a built-in, or via `util.callable_to_qualname` when it is a
callable; `from_config` decodes a qualname back to a callable and leaves known
strings as-is. This is a single encode/decode path — simpler than today's
separate callable-handling for `row_filter`.

## `CapData` wrappers (`src/captest/capdata.py`)

Import `RollingStd`, `AbsDiffPrev`, `BooleanFlag` alongside the existing filter
imports. Add the wrappers (each thin: build the step, `run(self)`, no return;
`custom_name` keyword-only label, per existing convention):

```python
def filter_rolling_std(self, window, threshold, column=None, custom_name=None): ...
    # -> RollingStd(window=window, threshold=threshold, column=column, ...)

def filter_abs_diff_prev(self, threshold=0.05, column=None, custom_name=None): ...
    # -> AbsDiffPrev(threshold=threshold, column=column, ...)

def filter_flag(self, column, invert=False, custom_name=None): ...
    # -> BooleanFlag(column=column, invert=invert, ...)

def filter_threshold(self, column, low=None, high=None, custom_name=None): ...
    # -> Irradiance(low=low, high=high, col_name=column, ...)

def filter_sensors(self, thresholds=None, method="percent_diff", custom_name=None): ...
    # -> Sensors(method=method, thresholds=thresholds, ...)

def filter_sensors_abs_diff(self, thresholds, custom_name=None): ...
    # -> Sensors(method="abs_diff", thresholds=thresholds, ...)
```

`filter_threshold` exposes the one-sided filtering the `Irradiance` class
already supports (`low`/`high` default `None`, `allow_None`) but which
`filter_irr` (required positional `low, high`) does not. `filter_threshold(
'avail_inverters', low=97.4)` keeps rows ≥ 97.4; `filter_threshold('temp',
high=40)` keeps rows ≤ 40. Boundaries are inclusive (`>=`/`<=`), per
`Irradiance` semantics — a minor change from the original `filter_avail`'s
strict `>`, documented in the wrapper docstring. The step serializes as an
`Irradiance` step (no new registry entry).

`filter_sensors` and `filter_sensors_abs_diff` are both class-backed
(`Sensors`), so the comparison method is exposed to a future GUI through the
`method` selector.

## Testing

Following existing patterns in `tests/test_filters.py` / `tests/test_CapData.py`:

- **Parity tests:** each new filter vs. the original notebook function as an
  oracle on a fixture frame, including the leading-`NaN` drop behavior for
  `RollingStd` and `AbsDiffPrev`.
- **POA default inference:** `RollingStd` / `AbsDiffPrev` with `column=None`
  resolve to the regression POA column.
- **`filter_threshold`:** one-sided low-only and high-only cases; inclusive
  boundary semantics; serializes/replays as an `Irradiance` step.
- **`BooleanFlag`:** truthy coercion across `0`/`1`, real booleans, and `NaN`;
  `invert=True` keeps the truthy rows (and `invert=False` removes them).
- **`Sensors`:** `method` selector resolves both built-ins; a custom callable
  assigned to `method` is used; `abs_diff`/custom with `thresholds=None` raises;
  `percent_diff` with `thresholds=None` defaults to `{poa: 0.05}`.
- **Serialization round-trips:** `to_config`/`from_config` and
  `step_from_config` for each new class; full pipeline `filters_to_config()` →
  `run_pipeline()`; YAML round-trip including `Sensors(method=...)` for both a
  built-in string and a custom callable.
- **Wrappers:** each appends exactly one step to `filters`, `data_filtered`
  reflects the removal, `custom_name` is honored, and the summary /
  `explanation` render.

The `unit-tests` skill governs test authoring; coverage is checked on the new
code.

## Documentation

Handled at implementation time via the `docs-update` skill:

- NumPy-style docstrings for all new classes and wrappers.
- Note `filter_threshold`'s inclusive boundary vs the original strict `>`.
- User-guide filter list and changelog entry; note the breaking `Sensors` API
  change (`perc_diff` → `thresholds`, `row_filter` kwarg removed, comparison
  unified into `method`).

## Files touched

- `src/captest/filters.py` — three new classes, `Sensors` enhancement,
  `FILTER_REGISTRY` additions.
- `src/captest/capdata.py` — imports + six wrappers
  (`filter_rolling_std`, `filter_abs_diff_prev`, `filter_flag`,
  `filter_threshold`, `filter_sensors` updated, `filter_sensors_abs_diff`).
- `tests/test_filters.py`, `tests/test_CapData.py` — new tests.
- Docs (user guide + changelog) at implementation time.
