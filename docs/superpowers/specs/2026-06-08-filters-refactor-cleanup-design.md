# Filters-refactor cleanup, plotting, and bug fixes

**Date:** 2026-06-08
**Branch:** `filters-refactor`
**Status:** Approved (pending spec review)

## Context

Integration testing of the `filters-refactor` branch surfaced six issues across
three groups: API cleanup, filter-attribution plotting, and a sim year-end
wrapping bug. Each is small and independent; they are documented together here
but will be implemented as **six separate, sequentially reviewed plans** (one
per change) per the maintainer's fine-grained-plan preference.

## Decisions (settled during brainstorming)

- **Class naming:** drop the `Filter` prefix; disambiguate the terse names.
- **`inplace` scope:** remove from all `filter_*` methods and `fit_regression`;
  keep `rep_cond_freq`'s `inplace` (compute-and-return, not filtering).
- **YAML compatibility:** clean break — no legacy `type`-string aliases.
- **Wrap-window years:** derive from `sim_year` (not literal constants).

---

## Group A — Cleanup

### A1. Add `custom_name` to wrapper methods

Every `BaseSummaryStep` subclass already accepts a `custom_name` display label
(declared on the base class). The `CapData.filter_*` / `rep_cond` /
`fit_regression` wrappers should expose it so callers can label steps without
constructing the step class directly.

Add a keyword-only `custom_name=None` parameter to each wrapper that builds a
step but does not yet forward one, passing it into the step constructor:

- `filter_irr`, `filter_pvsyst`, `filter_shade`, `filter_time`, `filter_days`,
  `filter_outliers`, `filter_pf`, `filter_power`, `filter_sensors`,
  `filter_clearsky`, `filter_missing`
- `rep_cond` (the `RepCond` step)
- `fit_regression` (the `Regression` step, built only when `filter=True`)

Already done / skipped:

- `filter_custom` — already has `custom_name`.
- `filter_op_state` — unimplemented `pass` stub, builds no step; skip.

Each wrapper's NumPy docstring gains a `custom_name : str, default None` entry
mirroring the wording already used in `filter_custom`.

### A2. Remove the `inplace` kwarg

The `inplace=False` branch returns a *preview* DataFrame
(`self.data_filtered.loc[flt._execute(self), :]`) without recording a step.
This dual behavior is being removed — filters always act in place.

Remove `inplace` from the signatures, bodies, and docstrings of:

- All `filter_*` methods: `filter_irr`, `filter_pvsyst`, `filter_shade`,
  `filter_time`, `filter_days`, `filter_outliers`, `filter_pf`, `filter_power`,
  `filter_sensors`, `filter_clearsky`, `filter_op_state` (signature only — stub).
- `fit_regression` — its `inplace` governs whether the `Regression` step is
  recorded when `filter=True`; it always records now.

For each, delete the `if inplace: ... else: return ...` branching and keep only
the `flt.run(self)` path (plus any `summary` printing in `fit_regression`).
Remove the `inplace` parameter docstring lines and the now-obsolete
"Returns DataFrame / CapData when inplace is False" return sections.

**Keep** `rep_cond_freq`'s `inplace` — there `inplace=False` returns the computed
multi-row reporting-conditions DataFrame instead of storing it on `self.rc`,
which is a legitimate compute-and-return toggle, not filtering.

### A3. Rename the filter classes

Drop the redundant `Filter` prefix (the classes already live in `filters.py`);
expand the two terse names (`Irr`, `Pf`) for readability.

| Old | New |
| --- | --- |
| `FilterIrr` | `Irradiance` |
| `FilterSensors` | `Sensors` |
| `FilterTime` | `Time` |
| `FilterCustom` | `Custom` |
| `FilterOutliers` | `Outliers` |
| `FilterClearsky` | `Clearsky` |
| `FilterPvsyst` | `Pvsyst` |
| `FilterShade` | `Shade` |
| `FilterPf` | `PowerFactor` |
| `FilterPower` | `Power` |
| `FilterDays` | `Days` |
| `FilterMissing` | `Missing` |
| `FilterRegression` | `Regression` |
| `RepCond` | `RepCond` (unchanged) |

Touch points:

- `filters.py`: class definitions; `FILTER_REGISTRY` keys; the hardcoded
  `"type": "FilterCustom"` in `Custom.to_config` → `"Custom"`. `to_config`
  otherwise derives `type` from `type(self).__name__`, so the serialized
  `type` strings follow the new names automatically.
- `capdata.py`: the imports and every wrapper-method instantiation.
- `clearsky.py`: the single matching reference (confirm during implementation).
- Tests: `test_filter_classes.py` (~190 refs), `test_CapData.py`,
  `test_captest.py`.

**Clean break:** `FILTER_REGISTRY` holds only the new keys. Any pre-existing
YAML configs using old `type` strings must be regenerated; `step_from_config`
already raises a helpful close-match error for unknown types.

---

## Group B — Plotting

### B1. `scatter_filters` — retained layer rendered last

Currently the `retained` scatter is appended first, then one scatter per
removing filter step. Reorder so the per-step removed scatters are appended
first and the `retained` scatter is appended **last**, so retained points draw
on top of the removed-point layers in the Overlay. No data/labeling changes;
only append order.

### B2. `timeseries_filters` — add a final retained-points scatter

Currently the overlay is a full-data power `Curve` ("all") plus one scatter per
removing filter step. Append **one additional scatter, last**, plotting the
points retained after all filtering (`self.filters[-1].ix_after` if filters
exist, else `self.data.index`), labeled `"retained"`, using the same
`["Timestamp"], ["power"]` mapping and existing scatter opts.

---

## Group C — Bug

### C1. Fixed July 1 → June 30 wrap window in `_maybe_wrap_sim_year_end`

**Problem.** The wrap window is currently derived from the measured test dates
(`meas_start`/`meas_end` month/day/hour/minute, with leap-day clipping). When
the measured test date is within 60 days of the year boundary, the window is
centered on the measured dates, which does not produce the contiguous wrapped
year the wrap is meant to achieve.

**Fix.** Leave the 60-day-from-boundary *trigger* logic unchanged (it still
decides whether to wrap at all). Replace the measured-derived window with a
fixed July 1 → June 30 window, keeping the years `sim_year`-derived:

```python
sim_year = sim_idx[0].year  # load_pvsyst normalizes pvsyst data to 1990
start = pd.Timestamp(year=sim_year - 1, month=7, day=1, hour=0, minute=0)
end = pd.Timestamp(year=sim_year, month=6, day=30, hour=23, minute=59)
```

For the standard pvsyst case this is `1989-07-01 00:00` → `1990-06-30 23:59`.
`wrap_year_end` takes the `elif df.index[0].year == end.year` branch (sim data
is year 1990): Jan 1–Jun 30 1990 stays, Jul 1–Dec 31 1990 shifts back to 1989,
concatenating to a contiguous `1989-07-01 … 1990-06-30` series. The leap-day
clipping and `meas_*` hour/minute logic are removed.

The `23:59` end label ensures the final hourly row (`23:00`) is included under
label-based `.loc` slicing regardless of sub-hourly frequency.

**Verification.** A test asserts the wrapped sim DataFrame has exactly **8760**
rows (365 days × 24 h; neither 1989 nor 1990-H1 includes Feb 29).

---

## Testing

- A1: assert a `custom_name` passed to a wrapper appears as the recorded step's
  label (e.g. via `describe_filters` / step `custom_name`).
- A2: assert wrappers no longer accept `inplace` (TypeError on
  `inplace=` kwarg); existing in-place behavior unchanged. Update/remove tests
  that exercised the `inplace=False` preview path.
- A3: update all class-name references; assert `FILTER_REGISTRY` round-trips the
  new `type` strings via `filters_to_config` / `run_pipeline`.
- B1/B2: assert overlay layer count and that the last layer is labeled
  `retained`.
- C1: assert wrapped row count == 8760 and contiguous index for a measured test
  near year-end.

Run `just test`, `just lint`, `just fmt` before each chunk's commit.

## Out of scope

- No changes to `rep_cond_freq` behavior beyond keeping its `inplace`.
- No legacy YAML `type`-string compatibility shims.
- No refactor of the unimplemented `filter_op_state` body.
