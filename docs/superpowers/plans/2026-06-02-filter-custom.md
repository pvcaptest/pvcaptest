# FilterCustom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `filter_custom` to the class-based architecture as `FilterCustom` — the spec's prototypical case for **non-`param` configuration** (an arbitrary callable plus `*args`/`**kwargs`). Establishes the pattern for storing callables and variadic args as plain instance attributes via a custom `__init__`, since `param` can't model them generically.

**Architecture:** `FilterCustom` inherits `BaseFilter` but overrides `__init__` to accept `(func, *args, custom_name=None, **kwargs)`. `func`/`args`/`kwargs` are stored as plain instance attributes (not `param` parameters — per the spec's "Callable Parameters" table, callables and variadics aren't directly serializable and are serialized as module-qualified-name strings by the YAML plan). `_execute` calls `self.func(capdata.data_filtered, *self.args, **self.kwargs)` and returns the resulting frame's index. `args_repr` is overridden to render `func.__name__(arg, ..., kw=val, ...)` — matching what the legacy `@update_summary` decorator's regex was extracting. `explanation` reuses that string in a single template sentence.

**Tech Stack:** Python, `param`, pandas, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Special Cases (FilterCustom)", "Callable Parameters", "Thin Wrapper Methods".

**Sequencing:** Execute *after* `2026-05-24-filter-irr-example.md` (needs `BaseSummaryStep`, `_record_legacy_summary`, the explanation hooks). Independent of the other complex-filter plans.

**No existing tests:** `tests/test_CapData.py` has no `TestFilterCustom` class — the method is currently uncovered. This plan adds the tests that were missing, exercising the docstring's pandas-method use cases (`between_time`, `dropna`) plus arg/kwarg passthrough.

## Key design decisions (flag if you disagree before implementing)

1. **`func`/`args`/`kwargs` are plain instance attributes, not `param` parameters.** Per the spec's "Callable Parameters" guidance — callables aren't directly serializable, and variadics don't fit `param`'s declared-parameter model. `FilterCustom.__init__` accepts them and stores them on `self`. YAML serialization (later plan) will handle the module-qualified-name string for `func`; this plan does not address that.
2. **Custom `__init__` with keyword-only `custom_name`.** Signature: `def __init__(self, func, *args, custom_name=None, **kwargs)`. `custom_name` is passed through to `super().__init__(custom_name=custom_name)`. Making it keyword-only avoids collision with any positional arg the user wants to pass to `func`. A user who genuinely needs to forward `custom_name=` to `func` can pass it inside a partial or a wrapper; that's an acceptable edge.
3. **`_execute` returns `result.index`, not `result` itself — deliberate row-filter semantics.** The legacy method assigned `func`'s entire return (including any column changes) to `data_filtered`. The new lifecycle (`run()` reindexes from `capdata.data` via `data.loc[ix_after]`) keeps **the original columns regardless of what `func` did to columns**. This is a deliberate tightening: `filter_custom` becomes strictly a row-filter (the docstring's example use cases — `pd.DataFrame.between_time`, `pd.DataFrame.dropna` — are row filters, so this matches intent). If a caller passed a column-transforming function, the new behavior diverges; that's a deliberate scope cut, documented in the wrapper docstring.
4. **`args_repr` override renders `func.__name__(...)`.** The legacy `@update_summary` decorator regex-replaced `<function foo at 0x...>` with `foo` to render arguments cleanly. `FilterCustom` overrides `args_repr` directly to produce the same shape: `func_name(repr(a1), ..., k1=repr(v1), ...)`. The base `_args_for_repr` hook isn't used here because the shape (a *call* rather than a parameter list) doesn't fit the generic `key=value` joining.
5. **No `inplace`.** The legacy method has none (it always mutates `data_filtered`). Matching that: the wrapper has no `inplace` param. Callers who want a non-recording dry run can invoke `FilterCustom(func, ...)._execute(cd)` directly.
6. **`_explanation_template` reuses `args_repr` in a single sentence.** Format: `"Custom filter {call} was applied."` with `_explanation_values` returning `{"call": self.args_repr}`. Doesn't need an `explanation` property override.

---

### Task 1: Add `FilterCustom` to `filters.py`

**Files:**
- Modify: `src/captest/filters.py` (add `FilterCustom`)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

Extend the top-of-file `filters` import:

```python
from captest.filters import (
    BaseSummaryStep,
    BaseFilter,
    FilterCustom,
    FilterIrr,
    FilterSensors,
    FilterTime,
    abs_diff_from_average,
    check_all_perc_diff_comb,
)
```

Add a module-level helper (above the test classes, near the fixtures) for tests that need a stable named function — lambdas would render as `<lambda>` which isn't a useful args_repr check:

```python
def _drop_first(df):
    return df.iloc[1:]


def _gt_threshold(df, threshold=0, col="poa"):
    return df[df[col] > threshold]
```

Append `TestFilterCustom` to `tests/test_filter_classes.py`:

```python
class TestFilterCustom:
    def test_execute_applies_func(self, cd_irr):
        kept = FilterCustom(_drop_first)._execute(cd_irr)
        assert list(kept) == [1, 2, 3, 4]

    def test_execute_passes_args_and_kwargs(self, cd_irr):
        # poa values [100, 300, 500, 700, 900] -> > 400 -> indices [2, 3, 4]
        f = FilterCustom(_gt_threshold, threshold=400)
        assert list(f._execute(cd_irr)) == [2, 3, 4]

    def test_execute_with_pandas_method_dropna(self):
        cd = CapData("c")
        cd.data = pd.DataFrame(
            {"power": [1.0, np.nan, 3.0, np.nan, 5.0]},
            index=pd.RangeIndex(5),
        )
        cd.data_filtered = cd.data.copy()
        kept = FilterCustom(pd.DataFrame.dropna)._execute(cd)
        assert list(kept) == [0, 2, 4]

    def test_args_repr_renders_func_name(self):
        f = FilterCustom(_gt_threshold, threshold=400)
        args = f.args_repr
        assert "_gt_threshold" in args
        assert "threshold=400" in args
        assert "<function" not in args

    def test_args_repr_with_positional_args(self):
        # Use between_time as the docstring example
        f = FilterCustom(pd.DataFrame.between_time, "9:00", "17:00")
        args = f.args_repr
        assert "between_time" in args
        assert "'9:00'" in args
        assert "'17:00'" in args

    def test_explanation_reuses_call(self, cd_irr):
        f = FilterCustom(_drop_first)
        f.run(cd_irr)
        exp = f.explanation
        assert "_drop_first" in exp
        assert exp.endswith("was applied.")

    def test_custom_name_passes_through(self):
        f = FilterCustom(_drop_first, custom_name="prune")
        assert f.custom_name == "prune"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterCustom -v`
Expected: FAIL — `ImportError: cannot import name 'FilterCustom'`.

- [ ] **Step 3: Implement `FilterCustom`**

Append to `src/captest/filters.py`:

```python
class FilterCustom(BaseFilter):
    """Apply an arbitrary callable to ``capdata.data_filtered`` as a row filter.

    ``func`` is a callable that takes a DataFrame as its first argument and
    returns a DataFrame whose index is the rows to keep. Typical use is a
    pandas DataFrame method like ``pd.DataFrame.dropna`` or
    ``pd.DataFrame.between_time``.

    Unlike most filters, ``func``/``*args``/``**kwargs`` are stored as plain
    instance attributes (not ``param`` parameters). Callables and variadics
    don't fit ``param``'s declared-parameter model; YAML serialization for
    ``func`` (module-qualified-name string) is handled by the YAML plan.
    """

    _legacy_name = "filter_custom"
    _explanation_template = "Custom filter {call} was applied."

    def __init__(self, func, *args, custom_name=None, **kwargs):
        super().__init__(custom_name=custom_name)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _execute(self, capdata):
        result = self.func(capdata.data_filtered, *self.args, **self.kwargs)
        return result.index

    @property
    def args_repr(self):
        """Render ``func_name(arg, ..., k=v, ...)`` — matches the legacy regex."""
        arg_parts = [repr(a) for a in self.args]
        kwarg_parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.func.__name__}({', '.join(arg_parts + kwarg_parts)})"

    def _explanation_values(self):
        return {"call": self.args_repr}
```

> Notes:
> - `args_repr` is fully overridden (the base hook expects a `key=value` mapping, which doesn't fit a call expression). The `_explanation_template` then plugs `args_repr` in via `_explanation_values`.
> - `func.__name__` works for ordinary `def` functions, unbound methods (`pd.DataFrame.dropna.__name__ == "dropna"`), and lambdas (`<lambda>`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterCustom -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add FilterCustom class with callable + variadic args"
```

---

### Task 2: Convert `CapData.filter_custom` to a thin wrapper

**Files:**
- Modify: `src/captest/capdata.py` (`filter_custom` method ~line 2282 in the original; line numbers shift after earlier plans — locate by anchor `def filter_custom`. Add `FilterCustom` to the `from captest.filters import (...)` block.)
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_filter_classes.py`:

```python
class TestFilterCustomWrapper:
    def test_wrapper_records_filtercustom_step(self, cd_irr):
        cd_irr.filter_custom(_drop_first)
        assert len(cd_irr.filters) == 1
        assert isinstance(cd_irr.filters[0], FilterCustom)

    def test_wrapper_passes_args_kwargs_to_func(self, cd_irr):
        cd_irr.filter_custom(_gt_threshold, threshold=400)
        assert list(cd_irr.data_filtered.index) == [2, 3, 4]

    def test_wrapper_custom_name_kwarg_is_kwonly(self, cd_irr):
        cd_irr.filter_custom(_drop_first, custom_name="prune")
        assert cd_irr.filters[0].custom_name == "prune"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterCustomWrapper -v`
Expected: FAIL — `cd.filters` empty / not a `FilterCustom` (still the decorated method).

- [ ] **Step 3: Add `FilterCustom` to the capdata import**

In `src/captest/capdata.py`:

```python
from captest.filters import (
    BaseSummaryStep,
    FilterCustom,
    FilterIrr,
    FilterSensors,
    FilterTime,
    check_all_perc_diff_comb,
    filter_grps,
    filter_irr,
    wrap_year_end,
)
```

- [ ] **Step 4: Replace the `filter_custom` method body with a thin wrapper**

Locate the current method by anchor `def filter_custom(self, func, *args, **kwargs):` (and the `@update_summary` decorator immediately above it). Replace the entire decorated method with:

```python
    def filter_custom(self, func, *args, custom_name=None, **kwargs):
        """Apply ``func`` to ``data_filtered`` as a row filter and record the step.

        ``func`` is called as ``func(self.data_filtered, *args, **kwargs)`` and
        must return a DataFrame whose index is the rows to keep. Many pandas
        DataFrame methods qualify, e.g. ``pd.DataFrame.between_time`` or
        ``pd.DataFrame.dropna``.

        Parameters
        ----------
        func : callable
            Takes a DataFrame as the first argument and returns a DataFrame.
        *args, **kwargs
            Forwarded to ``func``.
        custom_name : str, default None
            Optional display label for the recorded filter step. Keyword-only
            so it cannot collide with positional args destined for ``func``.

        Notes
        -----
        The class-based pipeline preserves the original column set: only the
        returned frame's *index* is consumed. A function that drops or
        transforms columns will see its column changes discarded — pass
        column-transforming logic outside the filter pipeline.
        """
        FilterCustom(func, *args, custom_name=custom_name, **kwargs).run(self)
```

> Note: no `inplace` param, matching the legacy method (which had none). Callers who want a non-recording dry run can use `FilterCustom(func, ...)._execute(cd)` directly.

- [ ] **Step 5: Run the wrapper tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterCustomWrapper -v`
Expected: PASS.

- [ ] **Step 6: Grep for any `pvc.filter_custom` / `capdata.filter_custom` test references**

Run:
```bash
grep -rnE "pvc\.filter_custom|capdata\.filter_custom" tests/ src/captest/
```
Expected: no hits (the method has no existing test references — see the plan intro). Any hit would need the no-shim repoint.

- [ ] **Step 7: Run the full suite**

Run: `just test-wo-warnings`
Expected: PASS.

- [ ] **Step 8: Lint and format**

Run: `just lint && just fmt`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/captest/capdata.py tests/test_filter_classes.py
git commit -m "refactor: make CapData.filter_custom a thin wrapper over FilterCustom"
```

---

## Self-Review

**1. Spec coverage (this filter):**
- "Special Cases → FilterCustom" (callable + variadic args as plain attributes) → Task 1 with custom `__init__`. ✓
- "Callable Parameters" — `func.__name__` rendering matches the legacy regex behavior; YAML serialization deferred to the YAML plan. ✓
- "Thin Wrapper Methods" → Task 2. ✓

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step has a command + expected result.

**3. Type/name consistency:** `FilterCustom` instance attrs (`func`, `args`, `kwargs`), `_legacy_name`, `_explanation_template`, the `args_repr`/`_explanation_values` overrides, and the wrapper signature (`func, *args, custom_name=None, **kwargs`) match between `filters.py`, `capdata.py`, and the tests.

**Deliberate behavior change documented:** the legacy method assigned `func`'s whole return frame to `data_filtered` (including column changes); the new pipeline preserves original columns via `data.loc[ix_after]`. This is a tightening to row-filter semantics, called out in both the `FilterCustom` class docstring and the wrapper docstring. The docstring's pandas-method examples (`between_time`, `dropna`) are unaffected.
