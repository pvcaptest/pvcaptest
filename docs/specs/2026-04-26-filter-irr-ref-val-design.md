# Design: Improve `filter_irr` `ref_val` Sentinel Clarity

## Problem
`CapData.filter_irr` accepts the magic string `"self_val"` for the `ref_val` parameter to
signal that the reporting irradiance from `self.rc` should be used as the reference value.
This string is opaque — it is not self-documenting at the call site — and it also appears
literally in the output of `CapData.get_summary()`, where a numeric value would be far more
useful to the analyst reviewing the filter history.

Additionally, when `"self_val"` is passed but `self.rc` has not been set, the resulting
`AttributeError` or `TypeError` gives no guidance on what the user should do to fix the
problem.

## Current State
- `filter_irr` (capdata.py:2160) accepts `ref_val="self_val"` and resolves it to
  `self.rc["poa"][0]` inside the method body.
- The `update_summary` decorator (capdata.py:145) captures the original `kwargs` dict from
  the wrapper — because Python unpacks keyword arguments into named parameters, the local
  reassignment inside `filter_irr` never writes back to the wrapper's `kwargs`. The summary
  therefore always shows the literal string `"self_val"`, not the resolved numeric value.
- No validation is performed before the `self.rc` access, so a missing or malformed `rc`
  produces an unhelpful traceback.

## Proposed Changes

### 1. Rename sentinel: `"self_val"` → `"rep_irr"`
The new sentinel name makes the intent explicit at the call site: the reference value is the
**rep**orting **irr**adiance stored in `self.rc`.

Backward compatibility: `"self_val"` is silently translated to `"rep_irr"` inside
`filter_irr`. No deprecation warning is emitted (the package is pre-1.0). This translation
will be removed in a future release.

### 2. Validation guard and resolution in `filter_irr`
After any `"self_val"` → `"rep_irr"` translation, raise a `ValueError` with a specific
message before attempting to access `self.rc`:

- If `self.rc is None`:
  `"ref_val='rep_irr' requires reporting conditions to be set. Call rep_cond() before
  calling filter_irr() with ref_val='rep_irr'."`
- If `self.rc` exists but lacks a `"poa"` column:
  `"ref_val='rep_irr' requires a 'poa' column in self.rc. The reporting conditions
  DataFrame does not have a 'poa' column."`

After validation, resolve the sentinel using `self.rc["poa"].iloc[0]` (positional access)
rather than the current `self.rc["poa"][0]` (index-label access), for the same robustness
reason noted in Section 3.

### 3. Sentinel resolution in `update_summary` (Approach A)
Immediately after `ret_val = func(self, *args, **kwargs)` and before `round_kwarg_floats`,
add:

```python
if kwargs.get("ref_val") in ("rep_irr", "self_val") and self.rc is not None:
    kwargs = {**kwargs, "ref_val": self.rc["poa"].iloc[0]}
```

Both sentinels are checked so that legacy `"self_val"` callers also see a resolved numeric
value in the summary. The resolution runs before `round_kwarg_floats` so the value is
rounded to 3 decimal places. The `self.rc is not None` guard is redundant (a successful
`func()` call guarantees `rc` is set) but makes intent explicit.

Switch from `self.rc["poa"][0]` (index-label access) to `self.rc["poa"].iloc[0]`
(positional access) for robustness with multi-row `rc` DataFrames produced by
`rep_cond_freq`.

### 4. Docstring update in `filter_irr`
Replace `self_val` with `rep_irr` in the `ref_val` parameter description.

## Test Changes (`tests/test_CapData.py`, `TestFilterIrr`)
- **Update** `test_refval_use_attribute`: change `ref_val="self_val"` → `ref_val="rep_irr"`.
- **Add** `test_refval_self_val_translation`: confirms `"self_val"` still filters correctly
  (silent translation).
- **Add** `test_refval_rep_irr_shows_in_summary`: calls with `ref_val="rep_irr"`, checks
  that `get_summary()["filter_arguments"]` contains the resolved numeric value, not the
  string `"rep_irr"`.
- **Add** `test_refval_rep_irr_rc_none_raises`: asserts `ValueError` when `self.rc is None`.
- **Add** `test_refval_rep_irr_no_poa_col_raises`: sets `self.rc` to a DataFrame without a
  `"poa"` column and asserts the second `ValueError` message.

## Files Affected
- `src/captest/capdata.py` — `update_summary` decorator and `filter_irr` method
- `tests/test_CapData.py` — `TestFilterIrr` class
