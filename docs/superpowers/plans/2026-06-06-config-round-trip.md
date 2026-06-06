# Config Round-Trip Implementation Plan (chunk 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serialize a `CapData`'s applied filter pipeline to/from config and integrate both `meas`/`sim` pipelines into the single `captest:` YAML file `CapTest` already produces, so a test (test-level params + filtering steps) round-trips through one file.

**Architecture:** Each filter class owns `to_config()`/`from_config()` (callable params encoded next to the class that uses them). `CapData` exposes I/O-free `filters_to_config()` and `run_pipeline(config)`; `filters.py` holds `FILTER_REGISTRY` + `step_from_config` (with a `difflib` suggestion for unknown types). `CapTest._build_yaml_sub_mapping`/`from_mapping` embed and re-apply the pipelines. Shared `perc_wrap` / `perc_N` helpers and new callable-qualname helpers move to `util.py` so `filters.py` can reach them without an import cycle.

**Tech Stack:** Python, `param`, pandas, PyYAML, `importlib`, `difflib`, pytest, `just`.

**Spec:** `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md` → "Config Round-Trip (chunk 7)".

**Sequencing:** Execute **after** chunk 6 (already landed; tip `6898c2f`). All filter classes are `param`-based; `RepCond`/`FilterCustom`/`FilterSensors` carry the callable params.

## Commit shape (four commits, each green)

1. **Task 1** — move `perc_wrap`/`perc_N` helpers to `util.py` + add callable-qualname helpers. Pure move + additions; green.
2. **Task 2** — per-class `to_config`/`from_config` + `FILTER_REGISTRY` + `step_from_config` in `filters.py`. Additive; green.
3. **Task 3** — `CapData.filters_to_config()` + `run_pipeline()`. Additive; green.
4. **Task 4** — `CapTest` single-file integration (embed + re-apply pipelines; omit redundant `rep_conditions`). Green.

---

## File Structure

- `src/captest/util.py` — gains `perc_wrap`, `_PERC_N_PREFIX`, `_resolve_perc_string`, `_resolve_func_strings`, `_perc_wrap_to_string` (moved from `captest.py`), and new `callable_to_qualname`/`callable_from_qualname`.
- `src/captest/captest.py` — imports the moved helpers from `util` (re-exporting `perc_wrap` for back-compat); integrates pipelines in `_build_yaml_sub_mapping` and `from_mapping`; adds the two keys to `_CAPTEST_YAML_KEYS`.
- `src/captest/filters.py` — `to_config`/`from_config` on `BaseSummaryStep`, `FilterCustom`, `FilterSensors`, `RepCond`; `FILTER_REGISTRY` + `step_from_config`.
- `src/captest/capdata.py` — `filters_to_config()`, `run_pipeline()`; import `step_from_config`.
- `tests/` — `test_util.py` (helpers), `test_filter_classes.py` (per-class round-trip), `test_CapData.py` (pipeline round-trip), `test_captest.py` (single-file integration).

---

### Task 1: Move serialization helpers to `util.py`; add callable-qualname helpers

**Files:**
- Modify: `src/captest/util.py`, `src/captest/captest.py`
- Test: `tests/test_util.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_util.py`, add (the file already imports the module under test as `from captest import util` or `import captest.util`; if not, add `from captest import util`):

```python
import pandas as pd
import pytest

from captest import util
from captest.filters import check_all_perc_diff_comb


class TestCallableQualname:
    def test_roundtrip_module_function(self):
        s = util.callable_to_qualname(check_all_perc_diff_comb)
        assert s == "captest.filters:check_all_perc_diff_comb"
        assert util.callable_from_qualname(s) is check_all_perc_diff_comb

    def test_roundtrip_method(self):
        s = util.callable_to_qualname(pd.DataFrame.head)
        assert util.callable_from_qualname(s) is pd.DataFrame.head

    def test_lambda_raises(self):
        with pytest.raises(ValueError, match="lambdas and closures"):
            util.callable_to_qualname(lambda df: df)

    def test_malformed_reference_raises(self):
        with pytest.raises(ValueError, match="module:qualname"):
            util.callable_from_qualname("no_colon_here")


class TestPercHelpersInUtil:
    def test_perc_wrap_roundtrips_through_string(self):
        f = util.perc_wrap(60)
        assert util._perc_wrap_to_string(f) == "perc_60"
        restored = util._resolve_perc_string("perc_60")
        assert restored.__name__ == "perc_wrap(60)"

    def test_plain_string_passes_through(self):
        assert util._perc_wrap_to_string("mean") == "mean"
        assert util._resolve_perc_string("mean") == "mean"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_util.py::TestCallableQualname tests/test_util.py::TestPercHelpersInUtil -v`
Expected: FAIL — `util.callable_to_qualname` / `util.perc_wrap` don't exist yet. (If a `uv` read-only-cache sandbox error appears, re-run.)

- [ ] **Step 3: Move the perc helpers from `captest.py` into `util.py`**

In `src/captest/util.py`, add `import importlib` to the imports, and add these (moved verbatim from `captest.py` — `perc_wrap`, `_PERC_N_PREFIX`, `_resolve_perc_string`, `_resolve_func_strings`, `_perc_wrap_to_string`; `perc_wrap` needs `numpy` which `util` already imports as `np`):

```python
_PERC_N_PREFIX = "perc_"


def perc_wrap(p):
    """Return a callable that computes the ``p``-th percentile of a Series.

    Used to build ``TEST_SETUPS[...]['rep_conditions']['func']`` dicts for
    percentile-based reporting irradiance (e.g. 60th percentile POA).

    Parameters
    ----------
    p : numeric
        Percentile in [0, 100].

    Returns
    -------
    callable
        Function that takes a pandas Series or array-like and returns the
        p-th percentile using ``method='nearest'``.
    """

    def numpy_percentile(x):
        return np.percentile(x.T, p, method="nearest")

    numpy_percentile.__name__ = f"perc_wrap({p})"
    return numpy_percentile


def _resolve_perc_string(val):
    """Resolve a "perc_N" string to ``perc_wrap(N)``.

    Non-matching strings pass through unchanged. Malformed ``perc_*`` strings
    raise ``ValueError``.
    """
    if not isinstance(val, str) or not val.startswith(_PERC_N_PREFIX):
        return val
    suffix = val[len(_PERC_N_PREFIX) :]
    if not suffix:
        raise ValueError(f"Malformed percentile string {val!r}: expected 'perc_<int>'.")
    try:
        n = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Malformed percentile string {val!r}: expected 'perc_<int>', "
            f"got suffix {suffix!r}."
        ) from exc
    return perc_wrap(n)


def _resolve_func_strings(func_dict):
    """Resolve ``perc_N`` strings inside a rep_conditions.func dict."""
    if not isinstance(func_dict, dict):
        return func_dict
    return {key: _resolve_perc_string(val) for key, val in func_dict.items()}


def _perc_wrap_to_string(val):
    """Inverse of :func:`_resolve_perc_string`.

    Converts a callable produced by :func:`perc_wrap` back into its
    round-trippable ``"perc_N"`` string form. Non-perc_wrap values pass
    through unchanged.
    """
    if not callable(val):
        return val
    name = getattr(val, "__name__", "")
    prefix = "perc_wrap("
    if name.startswith(prefix) and name.endswith(")"):
        inner = name[len(prefix) : -1]
        try:
            int(inner)
        except ValueError:
            return val
        return f"perc_{inner}"
    return val
```

- [ ] **Step 4: Add the callable-qualname helpers to `util.py`**

```python
def callable_to_qualname(func):
    """Return a ``'module:qualname'`` import string for a named callable.

    Raises ``ValueError`` for lambdas and closures (``<lambda>`` / ``<locals>``
    in the qualname) — they are not importable and cannot round-trip.
    """
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(
            f"Cannot serialize callable {func!r}: missing __module__/__qualname__."
        )
    if "<lambda>" in qualname or "<locals>" in qualname:
        raise ValueError(
            f"Cannot serialize callable {func!r}: lambdas and closures are not "
            f"importable. Use a module-level named function."
        )
    return f"{module}:{qualname}"


def callable_from_qualname(ref):
    """Import a callable from a ``'module:qualname'`` string (inverse of above)."""
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(
            f"Malformed callable reference {ref!r}: expected 'module:qualname'."
        )
    module_name, _, qualname = ref.partition(":")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
```

- [ ] **Step 5: Delete the moved definitions from `captest.py` and import them from `util`**

In `src/captest/captest.py`, delete the now-moved `perc_wrap`, `_PERC_N_PREFIX`, `_resolve_perc_string`, `_resolve_func_strings`, `_perc_wrap_to_string` definitions. Add to the existing `from captest import util` usage an explicit import near the top (the file already has `from captest import util` at line 31 — add a dedicated line so the bare names keep working and `captest.captest.perc_wrap` stays importable):

```python
from captest.util import (
    _PERC_N_PREFIX,
    _perc_wrap_to_string,
    _resolve_func_strings,
    _resolve_perc_string,
    perc_wrap,
)
```

Place this import **above** the `TEST_SETUPS` definition (which calls `perc_wrap(...)` at module load). Then confirm every remaining reference resolves:

Run: `grep -nE "perc_wrap|_perc_wrap_to_string|_resolve_func_strings|_resolve_perc_string|_PERC_N_PREFIX" src/captest/captest.py`
Expected: only the new import line plus call-sites (no leftover `def perc_wrap`/`def _resolve_*`/`def _perc_wrap_to_string`).

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_util.py -v` then `just test-wo-warnings`
Expected: the new tests pass and the full suite stays green (the helpers moved, not changed; `captest.captest.perc_wrap` still resolves via the import).

- [ ] **Step 7: Commit**

```bash
just lint && just fmt
git add src/captest/util.py src/captest/captest.py tests/test_util.py
git commit -m "refactor: move perc_wrap/perc_N helpers to util; add callable qualname helpers"
```

---

### Task 2: Per-step `to_config`/`from_config`, `FILTER_REGISTRY`, `step_from_config`

**Files:**
- Modify: `src/captest/filters.py`
- Test: `tests/test_filter_classes.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_filter_classes.py`, add (`FilterIrr`, `FilterCustom`, `FilterSensors`, `RepCond`, `check_all_perc_diff_comb` are already imported; add `from captest.filters import FILTER_REGISTRY, step_from_config` and `from captest import util`, `import pandas as pd`, `import pytest` if not present):

```python
class TestFilterConfigRoundTrip:
    def test_base_to_config_includes_all_params(self):
        cfg = FilterIrr(low=200, high=800).to_config()
        assert cfg == {
            "type": "FilterIrr",
            "low": 200,
            "high": 800,
            "ref_val": None,
            "col_name": None,
            "custom_name": None,
        }

    def test_base_roundtrip(self):
        cfg = FilterIrr(low=200, high=800, custom_name="bounds").to_config()
        step = step_from_config(cfg)
        assert isinstance(step, FilterIrr)
        assert step.to_config() == cfg

    def test_rep_cond_func_dict_roundtrips_perc(self):
        rc = RepCond(func={"poa": util.perc_wrap(60), "t_amb": "mean"})
        cfg = rc.to_config()
        assert cfg["func"] == {"poa": "perc_60", "t_amb": "mean"}
        rebuilt = step_from_config(cfg)
        assert rebuilt.func["poa"].__name__ == "perc_wrap(60)"
        assert rebuilt.func["t_amb"] == "mean"

    def test_rep_cond_none_func_roundtrips(self):
        cfg = RepCond().to_config()
        assert cfg["func"] is None
        assert step_from_config(cfg).func is None

    def test_rep_cond_str_func_roundtrips(self):
        cfg = RepCond(func="mean").to_config()
        assert cfg["func"] == "mean"
        assert step_from_config(cfg).func == "mean"

    def test_rep_cond_bare_perc_wrap_func_roundtrips(self):
        cfg = RepCond(func=util.perc_wrap(60)).to_config()
        assert cfg["func"] == "perc_60"
        assert step_from_config(cfg).func.__name__ == "perc_wrap(60)"

    def test_rep_cond_bare_named_callable_func_roundtrips(self):
        import numpy as np

        cfg = RepCond(func=np.mean).to_config()
        assert isinstance(cfg["func"], str) and ":" in cfg["func"]
        assert step_from_config(cfg).func is np.mean

    def test_rep_cond_dict_with_named_callable_roundtrips(self):
        import numpy as np

        cfg = RepCond(func={"poa": np.mean, "t_amb": "mean"}).to_config()
        assert cfg["func"]["t_amb"] == "mean"
        assert ":" in cfg["func"]["poa"]
        rebuilt = step_from_config(cfg).func
        assert rebuilt["poa"] is np.mean
        assert rebuilt["t_amb"] == "mean"

    def test_filter_custom_named_func_roundtrips(self):
        cfg = FilterCustom(pd.DataFrame.head, 3).to_config()
        assert cfg["func"] == "pandas.core.generic:NDFrame.head"
        assert cfg["args"] == [3]
        step = step_from_config(cfg)
        assert step.func is pd.DataFrame.head
        assert step.args == (3,)

    def test_filter_custom_lambda_raises(self):
        with pytest.raises(ValueError, match="lambdas and closures"):
            FilterCustom(lambda df: df).to_config()

    def test_filter_sensors_row_filter_roundtrips(self):
        cfg = FilterSensors(perc_diff={"irr-poa-": 0.05}).to_config()
        assert cfg["row_filter"] == "captest.filters:check_all_perc_diff_comb"
        step = step_from_config(cfg)
        assert step.row_filter is check_all_perc_diff_comb
        assert step.perc_diff == {"irr-poa-": 0.05}

    def test_unknown_type_suggests_closest(self):
        with pytest.raises(ValueError, match="Did you mean 'FilterIrr'"):
            step_from_config({"type": "FilterIrradiance"})

    def test_registry_covers_all_step_classes(self):
        assert FILTER_REGISTRY["RepCond"] is RepCond
        assert FILTER_REGISTRY["FilterCustom"] is FilterCustom
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterConfigRoundTrip -v`
Expected: FAIL — `to_config`/`step_from_config`/`FILTER_REGISTRY` not defined.

- [ ] **Step 3: Add `to_config`/`from_config` to `BaseSummaryStep`**

In `src/captest/filters.py`, add `import difflib` to the imports and `from captest import util` (placed with the other imports — `capdata.py` already imports `util` this way during the same import chain, so it is cycle-safe). Add to `BaseSummaryStep`:

```python
    def to_config(self):
        """Serialize this step to a yaml-safe dict (every param, defaults
        included; the param-system ``name`` is omitted)."""
        config = {"type": type(self).__name__}
        config.update({k: v for k, v in self.param.values().items() if k != "name"})
        return config

    @classmethod
    def from_config(cls, config):
        """Build an instance from a ``to_config()`` dict (``type`` removed)."""
        return cls(**config)
```

- [ ] **Step 4: Override on `FilterCustom`** (its `func`/`args`/`kwargs` are plain attributes, not params)

Add to `FilterCustom`:

```python
    def to_config(self):
        return {
            "type": "FilterCustom",
            "func": util.callable_to_qualname(self.func),
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "custom_name": self.custom_name,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        func = util.callable_from_qualname(config["func"])
        args = config.get("args") or []
        kwargs = config.get("kwargs") or {}
        return cls(func, *args, custom_name=config.get("custom_name"), **kwargs)
```

- [ ] **Step 5: Override on `FilterSensors`** (encode the `row_filter` callable)

Add to `FilterSensors`:

```python
    def to_config(self):
        config = super().to_config()
        config["row_filter"] = util.callable_to_qualname(self.row_filter)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if isinstance(config.get("row_filter"), str):
            config["row_filter"] = util.callable_from_qualname(config["row_filter"])
        return cls(**config)
```

- [ ] **Step 6: Override on `RepCond`** (encode `func` in all its documented forms)

`RepCond.func` accepts `dict / str / callable / None` (see its docstring). Every callable — whether `func` is a bare callable or a value inside the `func` dict — must be made yaml-safe, or `CapTest.to_yaml`'s `yaml.safe_dump` will raise. Add these two module-level helpers in `filters.py` (near `RepCond`, using the `util` primitives), which encode/decode a single value:

```python
def _encode_func_value(v):
    """Make one RepCond.func value yaml-safe.

    ``perc_wrap(N)`` callable -> ``"perc_N"``; any other named callable ->
    ``"module:qualname"`` (lambdas/closures raise via ``callable_to_qualname``);
    strings (e.g. ``"mean"``) and ``None`` pass through unchanged.
    """
    if callable(v):
        encoded = util._perc_wrap_to_string(v)
        if callable(encoded):  # not a perc_wrap callable -> a named callable
            return util.callable_to_qualname(v)
        return encoded
    return v


def _decode_func_value(v):
    """Inverse of ``_encode_func_value``.

    ``"module:qualname"`` -> imported callable; ``"perc_N"`` -> ``perc_wrap(N)``;
    other strings (``"mean"``) and ``None`` pass through.
    """
    if isinstance(v, str) and ":" in v:
        return util.callable_from_qualname(v)
    return util._resolve_perc_string(v)
```

Add to `RepCond` (dispatch dict vs. bare value through the helpers):

```python
    def to_config(self):
        config = super().to_config()
        func = self.func
        if isinstance(func, dict):
            config["func"] = {k: _encode_func_value(v) for k, v in func.items()}
        else:
            config["func"] = _encode_func_value(func)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        func = config.get("func")
        if isinstance(func, dict):
            config["func"] = {k: _decode_func_value(v) for k, v in func.items()}
        else:
            config["func"] = _decode_func_value(func)
        return cls(**config)
```

This covers all four forms: `None` → `null`; `"mean"` → `"mean"`; a bare `perc_wrap(60)` → `"perc_60"`; a bare named callable (`np.mean`) → `"numpy:mean"`; and dict values of any of those (including non-`perc_wrap` callables like `{'poa': np.mean}`).

- [ ] **Step 7: Add `FILTER_REGISTRY` and `step_from_config`** (near the end of `filters.py`, after all the classes are defined)

```python
FILTER_REGISTRY = {
    "FilterIrr": FilterIrr,
    "FilterPvsyst": FilterPvsyst,
    "FilterShade": FilterShade,
    "FilterTime": FilterTime,
    "FilterDays": FilterDays,
    "FilterOutliers": FilterOutliers,
    "FilterPf": FilterPf,
    "FilterPower": FilterPower,
    "FilterCustom": FilterCustom,
    "FilterSensors": FilterSensors,
    "FilterClearsky": FilterClearsky,
    "FilterMissing": FilterMissing,
    "FilterRegression": FilterRegression,
    "RepCond": RepCond,
}


def step_from_config(d):
    """Build a filter step from a ``to_config()`` dict via ``FILTER_REGISTRY``."""
    d = dict(d)
    cls_name = d.pop("type")
    if cls_name not in FILTER_REGISTRY:
        suggestion = difflib.get_close_matches(cls_name, FILTER_REGISTRY, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown filter type {cls_name!r} in pipeline config.{hint}")
    return FILTER_REGISTRY[cls_name].from_config(d)
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_filter_classes.py::TestFilterConfigRoundTrip -v` then `just test-wo-warnings`
Expected: PASS (9 new tests) and full suite green.

- [ ] **Step 9: Commit**

```bash
just lint && just fmt
git add src/captest/filters.py tests/test_filter_classes.py
git commit -m "feat: add to_config/from_config + FILTER_REGISTRY/step_from_config to filters"
```

---

### Task 3: `CapData.filters_to_config()` and `run_pipeline()`

**Files:**
- Modify: `src/captest/capdata.py`
- Test: `tests/test_CapData.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_CapData.py`, add a class (the file imports `from captest import capdata as pvc` and `from captest import filters`):

```python
class TestPipelineConfig:
    def test_filters_to_config_lists_steps(self, nrel):
        nrel.filter_irr(200, 800)
        nrel.filter_irr(400, 700)
        config = nrel.filters_to_config()
        assert [d["type"] for d in config] == ["FilterIrr", "FilterIrr"]
        assert config[0]["low"] == 200 and config[1]["low"] == 400

    def test_run_pipeline_replays_filters(self, nrel):
        nrel.filter_irr(200, 800)
        nrel.filter_irr(400, 700)
        config = nrel.filters_to_config()
        expected_ix = list(nrel.data_filtered.index)

        fresh = nrel.copy()
        fresh.reset_filter()
        fresh.run_pipeline(config)
        assert [type(s).__name__ for s in fresh.filters] == ["FilterIrr", "FilterIrr"]
        assert list(fresh.data_filtered.index) == expected_ix

    def test_run_pipeline_empty_is_noop(self, nrel):
        nrel.run_pipeline([])
        assert nrel.filters == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_CapData.py::TestPipelineConfig -v`
Expected: FAIL — `filters_to_config`/`run_pipeline` not defined.

- [ ] **Step 3: Import `step_from_config` and add the two methods**

In `src/captest/capdata.py`, add `step_from_config` to the existing `from captest.filters import (...)` block. Add these methods to `CapData` (place them just after `_removed_by_step`/`get_summary`, with the other chain helpers):

```python
    def filters_to_config(self):
        """Serialize the applied filter chain to a list of config dicts.

        Each entry is a step's ``to_config()`` (a yaml-safe ``{type, ...params}``
        dict). Inverse of :meth:`run_pipeline`. Used by ``CapTest.to_yaml`` to
        embed this CapData's pipeline in the single config file.
        """
        return [step.to_config() for step in self.filters]

    def run_pipeline(self, config):
        """Rebuild and run each filter step from a list of config dicts.

        ``config`` is a list of ``to_config()`` dicts (e.g. from
        :meth:`filters_to_config` or a loaded YAML). Each step is constructed
        via ``filters.step_from_config`` and run against this CapData in order.
        Requires ``data`` loaded and ``regression_cols`` resolved (run
        ``process_regression_columns`` first) for filters that need them.
        """
        for step_config in config:
            step_from_config(step_config).run(self)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_CapData.py::TestPipelineConfig -v` then `just test-wo-warnings`
Expected: PASS (3 new) and full suite green.

- [ ] **Step 5: Commit**

```bash
just lint && just fmt
git add src/captest/capdata.py tests/test_CapData.py
git commit -m "feat: add CapData.filters_to_config and run_pipeline"
```

---

### Task 4: `CapTest` single-file integration

**Files:**
- Modify: `src/captest/captest.py`
- Test: `tests/test_captest.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_captest.py` (imports include `from captest import CapTest`, `from unittest.mock import MagicMock`; the `meas_cd_default`/`sim_cd_default` fixtures exist), add:

```python
class TestPipelineYaml:
    def _capt(self, meas_cd_default, sim_cd_default):
        return CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
        )

    def test_sub_mapping_embeds_filter_pipelines(self, meas_cd_default, sim_cd_default):
        capt = self._capt(meas_cd_default, sim_cd_default)
        capt.meas.filter_irr(200, 800)
        capt.sim.filter_irr(200, 800)
        sub = capt._build_yaml_sub_mapping()
        assert sub["meas_filters"][0]["type"] == "FilterIrr"
        assert sub["meas_filters"][0]["low"] == 200
        assert sub["sim_filters"][0]["type"] == "FilterIrr"

    def test_no_filters_omits_pipeline_keys(self, meas_cd_default, sim_cd_default):
        capt = self._capt(meas_cd_default, sim_cd_default)
        sub = capt._build_yaml_sub_mapping()
        assert "meas_filters" not in sub
        assert "sim_filters" not in sub

    def test_rep_cond_step_omits_overrides_rep_conditions(
        self, meas_cd_default, sim_cd_default
    ):
        capt = self._capt(meas_cd_default, sim_cd_default)
        capt.rep_conditions = {"func": {"poa": "mean"}}  # non-None -> would serialize
        capt.rep_cond(which="meas")  # creates a RepCond step in meas.filters
        sub = capt._build_yaml_sub_mapping()
        assert any(d["type"] == "RepCond" for d in sub["meas_filters"])
        assert "rep_conditions" not in sub.get("overrides", {})

    def test_rep_conditions_kept_without_rep_cond_step(
        self, meas_cd_default, sim_cd_default
    ):
        capt = self._capt(meas_cd_default, sim_cd_default)
        capt.rep_conditions = {"func": {"poa": "mean"}}
        sub = capt._build_yaml_sub_mapping()
        assert sub["overrides"]["rep_conditions"]["func"]["poa"] == "mean"

    def test_from_mapping_reapplies_pipelines(
        self, tmp_path, meas_cd_default, sim_cd_default
    ):
        meas_file = tmp_path / "meas.csv"
        meas_file.write_text("x")
        sim_file = tmp_path / "sim.csv"
        sim_file.write_text("x")
        sub = {
            "test_setup": "e2848_default",
            "meas_path": str(meas_file),
            "sim_path": str(sim_file),
            "meas_filters": [
                {"type": "FilterIrr", "low": 200, "high": 800,
                 "ref_val": None, "col_name": None, "custom_name": None}
            ],
            "sim_filters": [
                {"type": "FilterIrr", "low": 200, "high": 800,
                 "ref_val": None, "col_name": None, "custom_name": None}
            ],
        }
        capt = CapTest.from_mapping(
            sub,
            meas_loader=MagicMock(return_value=meas_cd_default),
            sim_loader=MagicMock(return_value=sim_cd_default),
        )
        assert [type(s).__name__ for s in capt.meas.filters] == ["FilterIrr"]
        assert [type(s).__name__ for s in capt.sim.filters] == ["FilterIrr"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_captest.py::TestPipelineYaml -v`
Expected: FAIL — pipelines not embedded; `meas_filters` rejected as an unknown key by `from_mapping`.

- [ ] **Step 3: Embed pipelines in `_build_yaml_sub_mapping` (with decision B)**

In `src/captest/captest.py`, in `_build_yaml_sub_mapping`, compute the pipelines **early** (before the overrides block) and guard the `rep_conditions` line. Replace the existing overrides `rep_conditions` line:

```python
        if self.rep_conditions is not None:
            overrides["rep_conditions"] = _serialize_rep_conditions(self.rep_conditions)
```

with:

```python
        meas_filters = self.meas.filters_to_config() if self.meas is not None else []
        sim_filters = self.sim.filters_to_config() if self.sim is not None else []
        has_rep_cond_step = any(
            d["type"] == "RepCond" for d in (meas_filters + sim_filters)
        )
        # Decision B: when a RepCond step is in either pipeline, it is the
        # unambiguous source of reporting conditions — drop the redundant
        # overrides.rep_conditions.
        if self.rep_conditions is not None and not has_rep_cond_step:
            overrides["rep_conditions"] = _serialize_rep_conditions(self.rep_conditions)
```

Then, just before `return sub` at the end of the method, add (write only when non-empty, matching the load-kwargs pattern):

```python
        if meas_filters:
            sub["meas_filters"] = meas_filters
        if sim_filters:
            sub["sim_filters"] = sim_filters
```

- [ ] **Step 4: Register the new keys and re-apply on load in `from_mapping`**

In `src/captest/captest.py`, add `"meas_filters"` and `"sim_filters"` to the `_CAPTEST_YAML_KEYS` frozenset.

In `from_mapping`, exclude the pipeline keys from the constructor kwargs and apply them after construction. Change:

```python
        kwargs = {k: v for k, v in sub.items() if k != "overrides"}
```

to:

```python
        kwargs = {
            k: v
            for k, v in sub.items()
            if k not in ("overrides", "meas_filters", "sim_filters")
        }
```

Then, replace the final `return inst` with (apply pipelines after `from_params`, which has loaded data and auto-run `setup()` when both `meas`/`sim` are present, so `regression_cols` are in place):

```python
        # Re-apply serialized filter pipelines. from_params auto-runs setup()
        # when both meas and sim are populated, so regression_cols/data are
        # ready; guard on _resolved_setup so a partially-built CapTest doesn't
        # try to filter un-setup data.
        if inst._resolved_setup is not None:
            meas_filters = sub.get("meas_filters")
            sim_filters = sub.get("sim_filters")
            if meas_filters and inst.meas is not None:
                inst.meas.run_pipeline(meas_filters)
            if sim_filters and inst.sim is not None:
                inst.sim.run_pipeline(sim_filters)
        return inst
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_captest.py::TestPipelineYaml -v` then `just test-wo-warnings`
Expected: PASS (5 new) and full suite green.

- [ ] **Step 6: Commit**

```bash
just lint && just fmt
git add src/captest/captest.py tests/test_captest.py
git commit -m "feat: round-trip meas/sim filter pipelines in the single CapTest yaml"
```

---

## Self-Review

**1. Spec coverage** ("Config Round-Trip (chunk 7)"):
- One file, CapTest-orchestrated, CapData building blocks → Tasks 3 (`filters_to_config`/`run_pipeline`) + 4 (CapTest embed/re-apply). ✓
- Per-class `to_config`/`from_config` (base + `FilterCustom`/`FilterSensors`/`RepCond`) → Task 2. ✓
- `FILTER_REGISTRY` + `step_from_config` with `difflib` suggestion → Task 2 Step 7 + the `test_unknown_type_suggests_closest` test. ✓
- Export **all params explicitly** (decision A) → base `to_config` dumps every param incl. `None`; pinned by `test_base_to_config_includes_all_params`. ✓
- Omit `overrides.rep_conditions` when a `RepCond` step exists (decision B) → Task 4 Step 3; pinned by `test_rep_cond_step_omits_overrides_rep_conditions` / `test_rep_conditions_kept_without_rep_cond_step`. ✓
- Callable serialization: `FilterCustom.func`/`FilterSensors.row_filter` module-qualified (`callable_to/from_qualname`), lambdas raise; `RepCond.func` handles **all documented forms** — `None`, `str` (`"mean"`), bare `perc_wrap` (→`perc_N`), bare named callable (→`module:qualname`), and dict values of any of those — via `_encode_func_value`/`_decode_func_value` → Task 1 (helpers) + Task 2 Step 6. Covered by `test_rep_cond_*` (dict-perc, none, str, bare-perc, bare-callable, dict-with-named-callable). ✓
- `perc_wrap` + perc helpers moved to `util.py`, re-exported from `captest.captest` → Task 1. ✓
- Load/replay: `from_mapping`/`from_yaml` re-applies after data load + `setup()` → Task 4 Step 4; pinned by `test_from_mapping_reapplies_pipelines`. ✓
- Errors: unknown type (difflib), malformed callable ref, lambda export → covered by Task 1/2 tests. ✓

**2. Placeholder scan:** No TBDs. Every code change shows the full block; moved helpers are reproduced verbatim. The `pandas.core.generic:NDFrame.head` qualname in `test_filter_custom_named_func_roundtrips` is the concrete expected value (if a future pandas relocates `head`, the test's `func is pd.DataFrame.head` assertion still holds and the executor updates the string literal — noted here so it isn't mistaken for a placeholder).

**3. Type/name consistency:** `to_config()`/`from_config()`/`step_from_config`/`FILTER_REGISTRY`/`filters_to_config()`/`run_pipeline()`/`callable_to_qualname`/`callable_from_qualname` are used identically across tasks. `step_from_config` pops `type` then calls `cls.from_config(d)`; each override accepts the remaining dict. `_perc_wrap_to_string`/`_resolve_func_strings`/`perc_wrap` are referenced from `util` in both `captest.py` (re-import) and `filters.py` (RepCond).

**Deliberate decisions / notes:**
- Callable encoding uses `module:qualname` (colon separator) so methods like `pd.DataFrame.head` (qualname `NDFrame.head`) round-trip unambiguously; the spec's "module-qualified name" is satisfied. `FilterSensors`'s default `check_all_perc_diff_comb` → `captest.filters:check_all_perc_diff_comb`.
- `FilterTime.start`/`end`/`test_date` are expected to be YAML-safe strings (the `filter_time` wrapper passes strings). A `pd.Timestamp` param would need string coercion; out of scope here (no current path produces it), noted for a future refinement if a Timestamp ever reaches a stored param.
- Pipelines are written only when non-empty (matches the existing "write load_kwargs only when non-empty" convention); empty pipelines round-trip as "no filters."
- `from_mapping` re-applies pipelines only when `setup()` ran (`_resolved_setup is not None`) — the standard both-`meas`-and-`sim` path; a partial CapTest (rare) leaves them unapplied rather than erroring on un-setup data.
