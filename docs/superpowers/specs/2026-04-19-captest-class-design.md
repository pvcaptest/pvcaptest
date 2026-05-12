# CapTest class: unified test orchestrator with config-driven setup

Status: Approved for implementation. Depends on `rep-cond-calcparams` being merged into `captest-class` first. The filter refactor (`filters-refactor`) remains future work; `CapTest` lives with the current filter API for now.

## 1. Problem

Running an ASTM E2848 capacity test with `pvcaptest` today requires orchestrating two `CapData` instances (measured + modeled), hand-wiring identical filter/config parameters on each, and manually calling `process_regression_columns`, `rep_cond`, `fit_regression`, and `captest_results`. The orchestration lives in notebooks parameterized with papermill. There is no single API entrance that:

- Binds the measured and modeled `CapData` instances together.
- Holds test-level configuration (test period, irradiance bounds, nameplate, tolerance, etc.) as a single source of truth that flows down to both `CapData` instances and, in the future, their filters.
- Is driven by a yaml config so tests are reproducible and shareable across different entry points.
- Abstracts common regression-equation "setups" (default E2848, bifacial e-total, bifacial temperature-corrected power) so alternate regression formulas and their required calc params + matching scatter plots are selectable by name.

This spec adds a new `CapTest` class in a new module, defines a `TEST_SETUPS` registry of regression presets, moves the functions that compare two `CapData` instances onto the new class, refactors `CapData.scatter`/`CapData.scatter_hv` to be formula-agnostic, and surfaces the remaining places in `CapData` that still assume the default ASTM E2848 regression equation so they can be addressed as follow-up work.

## 2. Scope

In scope:

- New module `src/captest/captest.py` containing the `CapTest` class, the `TEST_SETUPS` registry, three shipped presets (`e2848_default`, `bifi_e2848_etotal`, `bifi_power_tc`) with their scatter-plot callables, and two formatting helpers (`print_results`, `highlight_pvals`).
- Removal of the module-level cross-`CapData` functions (`captest_results`, `captest_results_check_pvalues`, `pick_attr`, `get_summary(*args)`, `overlay_scatters`, `determine_pass_or_fail`, `plotting.residual_plot`) and their re-implementation as `CapTest` methods. The project is below v1; no backwards-compat shims are provided.
- Reimplementing `CapData.scatter` and `CapData.scatter_hv` as formula-agnostic thin wrappers around the `scatter_default` callable.
- A yaml schema for `CapTest.from_yaml`, plus `CapTest.to_yaml` for a curated round-trip.
- An issue-tracker appendix listing the remaining `CapData` methods that still assume the ASTM E2848 regression equation; these are deferred to follow-up PRs.

Out of scope (explicit dependencies):

- Merging `rep-cond-calcparams`. `rep_cond` / `rep_cond_freq` arbitrary-formula support is a prerequisite that lands first.
- The filter-class refactor documented in `docs/superpowers/specs/2026-04-03-filter-class-refactor-design.md`. Filter-level parameter propagation is handled ad-hoc via `CapTest` attributes today; after the filter refactor, CapTest params will flow directly onto filter objects.
- Any automated pipeline / `run()` method. `CapTest` is a config + state container; users still call `cd.filter_*()`, `cd.rep_cond()`, `cd.fit_regression()` by hand until the filter refactor lands.

## 3. Current state (summary)

- `CapData` (`src/captest/capdata.py:1597`) is the primary user-facing class. Its `regression_cols` attribute supports a nested dict of plain strings, aggregation tuples, and calc tuples; `process_regression_columns` (`capdata.py:3602`) walks it via `util.process_reg_cols` to materialize aggregations and calculated columns using functions from `src/captest/calcparams.py` (`e_total`, `bom_temp`, `cell_temp`, `power_temp_correct`, `rpoa_pvsyst`, `avg_typ_cell_temp`).
- `CapData.custom_param` (`capdata.py:3633`) auto-pulls missing kwargs from matching attribute names on the `CapData` instance. This is the mechanism by which scalar params set on `CapData` (e.g. `cd.bifaciality`) reach calc-param functions without having to be written into every `reg_cols_*` tuple.
- Seven module-level functions currently coordinate two `CapData` instances (see section 6).
- Commit `837853b` on branch `rep-cond-calcparams` extends `rep_cond` with `front_poa`, `func`, `rc_kwargs`, and splits the frequency path into a separate `rep_cond_freq`. That branch is the first merge in the implementation sequence. Immediately after the merge, this spec also removes the `func='E2939'` string-trigger branch so `func` always expects a plain dict (or `None`, which falls back to `{var: 'mean' for var in rhs}`). Responsibility for supplying the right `df.agg()` dict moves into each `TEST_SETUPS` preset's `rep_conditions` entry (section 6).

## 4. Module layout and architecture

New module: `src/captest/captest.py`. Exported via `src/captest/__init__.py` so users can `from captest import CapTest, TEST_SETUPS`.

Contents of `captest.py`:

1. `TEST_SETUPS: dict[str, dict]` — module-level registry of named regression-equation presets. Each value has `reg_cols_meas`, `reg_cols_sim`, `reg_fml`, `scatter_plots`, `rep_conditions`.
2. Three scatter-plot callables: `scatter_default`, `scatter_etotal`, `scatter_bifi_power_tc`. All accept `(cd: CapData, **kwargs) -> hv.Layout`.
3. `print_results(...)` and `highlight_pvals(...)` — pure formatters relocated from `capdata.py`, consumed only by `CapTest` methods.
4. `validate_test_setup(entry: dict)`, `resolve_test_setup(name: str, overrides: dict) -> dict`, `load_config(path) -> dict` — internal helpers used by `CapTest`.
5. `CapTest(param.Parameterized)` — the orchestrator class (see section 5).

`src/captest/__init__.py` gains:

```python
from captest.captest import CapTest, TEST_SETUPS
```

## 5. `CapTest` class surface

`CapTest` is a `param.Parameterized` subclass. All parameters are declared with `param.<Type>(default=..., doc=...)` and also described in the class's NumPy-style docstring (per the package convention). The `doc=` kwarg exists only for `param`-native introspection and for future Panel auto-widget generation; user-facing documentation lives in the NumPy-style docstring.

### 5.1 Parameters

Grouped for readability. All scalar params have sane defaults; `meas` / `sim` default to `None`.

Bound `CapData` instances:

- `meas: param.ClassSelector(class_=CapData, default=None)`
- `sim: param.ClassSelector(class_=CapData, default=None)`

Regression setup:

- `test_setup: param.String(default="e2848_default")` — key into `TEST_SETUPS` or the literal string `"custom"`.
- `reg_fml: param.String(default=None)` — if set, overrides the preset formula.
- `reg_cols_meas: param.Dict(default=None)` — if set, overrides the preset measured regression columns.
- `reg_cols_sim: param.Dict(default=None)` — if set, overrides the preset modeled regression columns.
- `rep_conditions: param.Dict(default=None)` — if set, partial-merged onto the preset `rep_conditions` at `setup()` time. Top-level keys replace; the nested `func` dict is merged one level deep so users can override only a single variable's aggregation (e.g. `{"func": {"poa": perc_wrap(55)}}` changes only the POA aggregation and leaves `t_amb`, `w_vel` unchanged). Projects/contracts that dictate different percentile values plug in here.
- `rep_cond_source: param.Selector(objects=["meas", "sim"], default="meas")` — which CapData's `rc` is used by `captest_results`.

Test scope / time:

- `sim_days: param.Integer(default=30, bounds=(1, 365))` — days of simulated data used for the test; values above ~90 are rarely advisable.
- `shade_filter_start: param.String(default=None)` and `shade_filter_end: param.String(default=None)` — `"HH:MM"` strings for between-time shade filtering.

Measurement / nameplate:

- `ac_nameplate: param.Number(default=None)` — W.
- `test_tolerance: param.String(default="- 4")` — forwarded to the pass/fail logic.

Filter parameters (held centrally today; the filter refactor will consume them natively later):

- `min_irr: param.Number(default=400)`, `max_irr: param.Number(default=1400)`
- `clipping_irr: param.Number(default=1000)`
- `rep_irr_filter: param.Number(default=0.2, bounds=(0.0, 1.0))`
- `fshdbm: param.Number(default=1.0, bounds=(0.0, 1.0))`
- `irrad_stability: param.Selector(objects=["std", "filter_clearsky", "contract"], default="std")`
- `irrad_stability_threshold: param.Number(default=30)`
- `hrs_req: param.Number(default=12.5)`

Calc-params scalars (propagated to both `CapData` instances at `setup()`):

- `bifaciality: param.Number(default=0.0, bounds=(0.0, 1.0))`
- `power_temp_coeff: param.Number(default=-0.32)` — percent per degree Celsius.
- `base_temp: param.Number(default=25)`

Data-loader injection (used by `from_params` / `from_yaml` when a path is supplied; callables are programmatic-only and never serialized to yaml):

- `meas_loader: param.Callable(default=None)` — called as `meas_loader(meas_path, **(meas_load_kwargs or {}))` to build `self.meas`. Default resolution when `None` is `captest.io.load_data`. Projects with bespoke loaders set this to their own callable at construction time.
- `meas_load_kwargs: param.Dict(default=None)` — extra kwargs splatted into `meas_loader`. Plain dicts so they CAN be written to yaml.
- `sim_loader: param.Callable(default=None)` — same as `meas_loader` but default resolution is `captest.io.load_pvsyst`.
- `sim_load_kwargs: param.Dict(default=None)` — same shape as `meas_load_kwargs`.

Internal (not `param.*`, not in yaml):

- `self._resolved_setup: dict | None` — plain instance attribute, initialized to `None` in `__init__` and assigned each time `setup()` runs. Not a `param.*` because `setup()` is re-runnable; `constant=True` would block re-assignment on the second call.

Class-level (not a `param.*`):

- `_downstream_attrs: tuple[str, ...] = ("bifaciality", "power_temp_coeff", "base_temp")` — names of params copied onto both `CapData` instances during `setup()`. Extending is a one-line edit.

### 5.2 Methods

```python
class CapTest(param.Parameterized):
    """NumPy-style class docstring with Parameters, Attributes, and Notes sections."""

    # param.* declarations above

    @classmethod
    def from_yaml(cls, path: str | Path, key: str = "captest") -> "CapTest":
        """Construct from a yaml file. Reads the sub-mapping at the given top-level
        `key`, defaulting to "captest". Supports multiple captest sections in one
        file (e.g. key="captest_bifi") so users can run different flavors of the
        capacity test against the same project."""

    @classmethod
    def from_params(cls, **kwargs) -> "CapTest": ...

    def setup(self, verbose: bool = True) -> "CapTest":
        """Resolve TEST_SETUPS, propagate scalars to meas/sim, run
        process_regression_columns on both. Returns self for fluent chaining."""

    def to_yaml(self, path: str | Path, key: str = "captest",
                merge_into_existing: bool = True) -> None:
        """Write a curated yaml under the top-level `key`. When merge_into_existing
        is True and the file already exists and parses as a mapping, preserves the
        other top-level keys (only the `key` subtree is overwritten). Data, CapData
        instances, regression_results, _resolved_setup, and loader callables are
        never written. Callable scatter_plots is never written; a warning is emitted
        if any callable was user-overridden."""

    def scatter_plots(self, which: str = "meas", **kwargs):
        """Call the scatter_plots callable from the resolved TEST_SETUPS on
        self.meas (default) or self.sim."""

    def rep_cond(self, which: str = "meas", **overrides) -> None:
        """Call cd.rep_cond using the resolved TEST_SETUPS rep_conditions as
        defaults. `**overrides` partial-merges over the resolved dict (top-level
        keys replace; nested `func` merges one level deep). Mirrors the
        scatter_plots(which=...) shape."""

    def captest_results(self, check_pvalues: bool = False,
                        pval: float = 0.05, print_res: bool = True) -> float: ...

    def captest_results_check_pvalues(self, print_res: bool = False, **kwargs): ...

    def determine_pass_or_fail(self, cap_ratio: float) -> tuple[bool, str]: ...

    def get_summary(self) -> pd.DataFrame:
        """Concatenate self.meas.get_summary() and self.sim.get_summary()."""

    def overlay_scatters(self, expected_label: str = "PVsyst"): ...

    def residual_plot(self): ...

    @property
    def resolved_setup(self) -> dict: ...
```

No `run()`, `apply_filters()`, or `fit()` methods. The caller invokes `ct.meas.filter_*(...)`, `ct.meas.rep_cond(...)`, and so on directly.

### 5.3 Constructor behavior

`CapTest(**kwargs)`:

1. `param.Parameterized.__init__(**kwargs)` assigns param.* attributes; type and bounds validation fires immediately.
2. `self.meas = None`, `self.sim = None` (defaults).
3. Returns. No I/O, no `setup()` call. Mutating attributes afterward and finally calling `ct.setup()` is the manual workflow.

`CapTest.from_params(**kwargs)`:

1. Pop the non-param keys that control construction: `meas`, `sim`, `meas_path`, `sim_path`. Keys assigned directly as `param.*` values (including the loader callables and kwargs) flow through normal `CapTest(**kwargs)` assignment.
2. Resolve loaders: `meas_loader = self.meas_loader or captest.io.load_data` and `sim_loader = self.sim_loader or captest.io.load_pvsyst`.
3. If `meas` (a `CapData`) is supplied, assign it; else if `meas_path`, call `meas_loader(meas_path, **(self.meas_load_kwargs or {}))` and assign; else leave `None`. Same pattern for `sim` with `sim_loader`.
4. If both `meas` and `meas_path` are supplied for the same side, pre-built wins and a warning is emitted.
5. If `self.meas` and `self.sim` are both set, call `self.setup()`. Otherwise return the partially-initialized instance; the user finishes manually.

`CapTest.from_yaml(path, key="captest")`:

1. Resolve `path`, remember `path.parent` as the base for relative path resolution.
2. Parse the entire file with `yaml.safe_load`. Extract `config[key]` as the captest-specific sub-mapping. Raise with a clear message if `key` is absent, listing the top-level keys present in the file.
3. Validate the sub-mapping: unknown keys inside it raise with a Levenshtein suggestion; `test_setup` is required; `test_setup: "custom"` requires `overrides.reg_cols_meas`, `overrides.reg_cols_sim`, `overrides.reg_fml`; a top-level `reg_fml` and `overrides.reg_fml` cannot both be set; scalars coerced to their param types. `"perc_N"` string shorthand in `rep_conditions.func` is resolved to `perc_wrap(N)` at this stage (see section 7.1). Loader callables are NOT expected in yaml.
4. Resolve `meas_path` / `sim_path` relative to `path.parent`.
5. Call `CapTest.from_params(**flattened)`.

This supports multiple captest sections in one yaml, e.g. `CapTest.from_yaml("config.yaml", key="captest_e2848")` and `CapTest.from_yaml("config.yaml", key="captest_bifi")` against the same project file.

### 5.4 `setup()` in detail

Preconditions:

- `self.meas` is a `CapData` — else `RuntimeError`.
- `self.sim` is a `CapData` — else `RuntimeError`.

Steps:

1. Resolve the active `TEST_SETUPS` entry:
   - If `self.test_setup == "custom"`: require the three override params to be populated (`reg_cols_meas`, `reg_cols_sim`, `reg_fml`); build the resolved dict from them. `scatter_plots` defaults to `scatter_default` if not provided. `rep_conditions` defaults to an empty dict (letting `rep_cond`'s own `func=None` fallback apply); any value in `self.rep_conditions` is partial-merged in.
   - Else: `base = copy.deepcopy(TEST_SETUPS[self.test_setup])`; for each of `reg_cols_meas`, `reg_cols_sim`, `reg_fml` set on `self`, overwrite the corresponding key in `base`. If `self.rep_conditions` is set, partial-merge it over `base["rep_conditions"]` (top-level keys replace; nested `func` dict is merged one level deep so a user can override only a single variable's aggregation).
   - `validate_test_setup(base)`: parse `reg_fml` via `util.parse_regression_formula`; check that lhs + rhs variables are subsets of the keys of both `reg_cols_meas` and `reg_cols_sim`; `scatter_plots` must be callable; `rep_conditions` must be a dict; if `rep_conditions["func"]` is present it must be a dict whose keys are a subset of the formula's rhs variables.
   - Assign to `self._resolved_setup` (plain instance attribute; setup is re-runnable so the value is not constant).
2. Propagate scalars: for each name in `CapTest._downstream_attrs`, `setattr(cd, name, getattr(self, name))` on both `self.meas` and `self.sim`.
3. Wire per-CapData state:
   - `self.meas.regression_cols = copy.deepcopy(self._resolved_setup["reg_cols_meas"])`
   - `self.sim.regression_cols = copy.deepcopy(self._resolved_setup["reg_cols_sim"])`
   - `self.meas.regression_formula = self._resolved_setup["reg_fml"]`
   - `self.sim.regression_formula  = self._resolved_setup["reg_fml"]`
   - `self.meas.tolerance = self.test_tolerance`
   - `self.sim.tolerance  = self.test_tolerance`
4. Run `self.meas.process_regression_columns(verbose=verbose)` and `self.sim.process_regression_columns(verbose=verbose)`. Default `verbose` is `True`.
5. `process_regression_columns` resets `data_filtered` to `data.copy()` on each CapData, so any filter state from a prior `setup()` call is dropped. This is the intended behavior.
6. Return `self`.

Methods that depend on `setup()` having run (anything touching `self.meas` / `self.sim` or `self._resolved_setup`) raise `RuntimeError("CapTest.setup() must be called first")` if it has not.

## 6. `TEST_SETUPS` registry

### 6.1 Entry shape

```python
TEST_SETUPS: dict[str, dict] = {
    "<preset_name>": {
        "reg_cols_meas":  {...},   # dict; values are column-group ids, (group, agg_func)
                                   # tuples, or (callable, kwargs_dict) tuples -- matches
                                   # the existing util.process_reg_cols structure.
        "reg_cols_sim":   {...},   # same structure, oriented at modeled data.
        "reg_fml":        str,     # patsy-compatible regression formula.
        "scatter_plots":  callable,# (cd: CapData, **kwargs) -> hv.Layout
        "rep_conditions": {...},   # kwargs dict forwarded to cd.rep_cond(...). Keys:
                                   # func (dict passed to df.agg()), irr_bal,
                                   # percent_filter, front_poa, rc_kwargs, etc.
    },
    ...
}
```

Rules enforced by `validate_test_setup`:

- All five keys are required; unknown keys raise.
- `reg_fml` must parse via `util.parse_regression_formula`.
- The lhs and rhs variables returned by that parser must be subsets of the keys of both `reg_cols_meas` and `reg_cols_sim`.
- `scatter_plots` must be callable.
- `rep_conditions` must be a dict. Its `func` entry, if present, must be a dict whose keys are a subset of the formula's rhs variable names.

### 6.2 Naming convention

The regression-formula **lhs key is always `"power"`** across shipped presets, even when the formula regresses a derived quantity like temperature-corrected power. The value of `reg_cols_meas["power"]` holds the calc-tuple that defines the actual computation (e.g. `(power_temp_correct, {...})`). This convention keeps code that still hard-codes `"power"` as the lhs key (`timeseries_filters`, `filter_power` defaults) working with all shipped presets.

### 6.3 Shipped presets

**`e2848_default`** — the current ASTM E2848 regression.

```python
"e2848_default": {
    "reg_cols_meas": {
        "power": ("real_pwr_mtr", "sum"),
        "poa":   ("irr_poa",     "mean"),
        "t_amb": ("temp_amb",    "mean"),
        "w_vel": ("wind_speed",  "mean"),
    },
    "reg_cols_sim": {
        "power": "E_Grid",
        "poa":   "GlobInc",
        "t_amb": "T_Amb",
        "w_vel": "WindVel",
    },
    "reg_fml": "power ~ poa + I(poa*poa) + I(poa*t_amb) + I(poa*w_vel) - 1",
    "scatter_plots": scatter_default,
    "rep_conditions": {
        "irr_bal": False,
        "percent_filter": 20,
        "front_poa": "poa",
        "func": {
            "poa":   perc_wrap(60),
            "t_amb": "mean",
            "w_vel": "mean",
        },
    },
}
```

**`bifi_e2848_etotal`** — bifacial E2848 with `e_total` as the regression irradiance.

```python
"bifi_e2848_etotal": {
    "reg_cols_meas": {
        "power": ("real_pwr_mtr", "sum"),
        "poa": (e_total, {
            "poa":  ("irr_poa",  "mean"),
            "rpoa": ("irr_rpoa", "mean"),
        }),
        "t_amb": ("temp_amb",   "mean"),
        "w_vel": ("wind_speed", "mean"),
    },
    "reg_cols_sim": {
        "power": "E_Grid",
        "poa": (e_total, {
            "poa":  "GlobInc",
            "rpoa": (rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"}),
        }),
        "t_amb": "T_Amb",
        "w_vel": "WindVel",
    },
    "reg_fml": "power ~ poa + I(poa*poa) + I(poa*t_amb) + I(poa*w_vel) - 1",
    "scatter_plots": scatter_etotal,
    "rep_conditions": {
        "irr_bal": False,
        "percent_filter": 20,
        "front_poa": "poa",
        "func": {
            "poa":   perc_wrap(60),
            "t_amb": "mean",
            "w_vel": "mean",
        },
    },
}
```

`bifaciality` (and any future `bifacial_frac` / `rear_shade` params) are NOT hardcoded in the preset — they come from `CapTest` attributes that `setup()` assigns onto the `CapData` instances, so `e_total`'s `custom_param` auto-binding picks them up.

**`bifi_power_tc`** — bifacial temperature-corrected-power regression.

```python
"bifi_power_tc": {
    "reg_cols_meas": {
        "power": (power_temp_correct, {
            "power":     ("real_pwr_mtr", "sum"),
            "cell_temp": (cell_temp, {
                "poa": ("irr_poa", "mean"),
                "bom": (bom_temp, {
                    "poa":        ("irr_poa",    "mean"),
                    "temp_amb":   ("temp_amb",   "mean"),
                    "wind_speed": ("wind_speed", "mean"),
                }),
            }),
        }),
        "poa":  ("irr_poa",  "mean"),
        "rpoa": ("irr_rpoa", "mean"),
    },
    "reg_cols_sim": {
        "power": (power_temp_correct, {
            "power":     "E_Grid",
            "cell_temp": "TArray",
        }),
        "poa":  "GlobInc",
        "rpoa": (rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"}),
    },
    "reg_fml": "power ~ poa + rpoa",
    "scatter_plots": scatter_bifi_power_tc,
    "rep_conditions": {
        "irr_bal": False,
        "percent_filter": 20,
        "front_poa": "poa",
        "func": {
            "poa":  perc_wrap(60),
            "rpoa": "mean",
        },
    },
}
```

`power_temp_coeff` (default -0.32) and `base_temp` (default 25) come from `CapTest` attributes via the same `custom_param` auto-binding.

### 6.4 Scatter-plot callables

All three live next to `TEST_SETUPS` in `captest.py`. Signature `(cd: CapData, **kwargs) -> hv.Layout`. All of them are formula-agnostic — they resolve lhs and rhs variable names via `util.parse_regression_formula(cd.regression_formula)` rather than hard-coding `"power"` or `"poa"`.

- `scatter_default(cd)` — single `hv.Scatter` of lhs vs. first rhs term. Today's `CapData.scatter_hv` logic relocated here and generalized.
- `scatter_etotal(cd)` — single `hv.Scatter` where the x variable is the calculated `e_total` column (the value of `cd.regression_cols["poa"]` after `process_regression_columns`).
- `scatter_bifi_power_tc(cd)` — `hv.Layout` of two panels: `power ~ poa` and `power ~ rpoa`. The only preset whose layout exceeds a single panel.

### 6.5 Extensibility

Users can register their own preset by assigning into `captest.TEST_SETUPS` at import time. `validate_test_setup` runs the same checks when `CapTest.setup()` resolves the preset. Alternatively, `test_setup: "custom"` at construction time accepts an inline `reg_cols_meas` + `reg_cols_sim` + `reg_fml`; `scatter_plots` then defaults to `scatter_default` unless supplied explicitly, and `rep_conditions` is built from `self.rep_conditions` if set, or defaults to an empty dict (letting `rep_cond`'s own `func=None` fallback apply).

Project- or contract-specific variations of a preset are typically handled via the `CapTest.rep_conditions` override param at construction time (in yaml: under the `overrides.rep_conditions` sub-key). Contracts that require e.g. a 55th percentile POA reporting irradiance can set `rep_conditions={"func": {"poa": perc_wrap(55)}}` without redefining the rest of the preset.

## 7. YAML schema

All `CapTest` configuration lives under a single top-level key (default `"captest"`), so the same yaml file can hold other project-level data without collision. The sub-key name is parametrizable so one file can hold multiple captest sections (e.g. `captest_e2848`, `captest_bifi`) — enabling different flavors of the capacity test to be run against the same project file. Unknown keys inside the captest sub-mapping raise on load; unknown keys at the file top level are ignored.

```yaml
# Project-level keys (ignored by CapTest; may be consumed by other tooling).
client: barnhart
loc:    {latitude: 42.28, longitude: -84.65, altitude: 294, tz: America/Detroit}
system: {albedo: 0.2, axis_azimuth: 180, axis_tilt: 0, max_angle: 60, gcr: 0.315, backtrack: true}

# CapTest section. Name ("captest") is parametrizable via from_yaml(path, key=...)
captest:
  # Name of the TEST_SETUPS preset. Required.
  test_setup: bifi_e2848_etotal

  # Optional: paths. If present, from_yaml loads the data automatically.
  # Resolved relative to the yaml file's directory.
  meas_path: ./data/meas/
  sim_path:  ./data/pvsyst.csv

  # Optional: per-key overrides of the TEST_SETUPS preset.
  overrides:
    reg_fml: "power ~ poa + I(poa*poa) + I(poa*t_amb) - 1"
    # reg_cols_meas: {...}
    # reg_cols_sim:  {...}
    rep_conditions:
      percent_filter: 10            # top-level override (replaces just this key)
      func:
        poa: "perc_55"              # resolved to perc_wrap(55) at load time
                                    # (see section 7.1); t_amb/w_vel preserved
                                    # from the preset via nested merge.

  # Regression / reporting
  rep_cond_source: meas
  reg_fml: null
  ac_nameplate: 125000
  test_tolerance: "- 4"

  # Test scope
  sim_days: 30
  shade_filter_start: null
  shade_filter_end:   null

  # Filter parameters
  min_irr: 400
  max_irr: 1400
  clipping_irr: 1000
  rep_irr_filter: 0.2
  fshdbm: 1.0
  irrad_stability: std
  irrad_stability_threshold: 30
  hrs_req: 12.5

  # Calc-params scalars
  bifaciality: 0.15
  power_temp_coeff: -0.32
  base_temp: 25

  # Loader kwargs (used when meas_path / sim_path are set). Plain dicts are fine
  # in yaml. Loader callables (meas_loader, sim_loader) are programmatic-only.
  meas_load_kwargs:
    # Example custom loader kwargs:
    period: {start_day: "2026-03-26", end_day: "2026-04-12"}
    groups_to_load: [irr_poa, irr_rpoa, temp_amb, wind_speed, real_pwr_mtr]
  sim_load_kwargs: {}
```

Validation rules:

- `test_setup` is required under the captest key; must be a key in `TEST_SETUPS` or the literal `"custom"`.
- `"custom"` requires `overrides.reg_cols_meas`, `overrides.reg_cols_sim`, `overrides.reg_fml` all present.
- `reg_fml:` at the captest top level and `overrides.reg_fml:` cannot both be set.
- `null` for an optional scalar is equivalent to omitting the key.
- Unknown keys inside the captest sub-mapping raise with a Levenshtein suggestion.
- `scatter_plots`, `meas_loader`, `sim_loader` are never serialized.

### 7.1 `perc_wrap` and other non-serializable function values

The `func` dict in `rep_conditions` typically uses `perc_wrap(N)` closures for percentile aggregations. Plain yaml cannot represent a Python callable. Two supported options:

1. **Preset defaults only** — yaml omits `overrides.rep_conditions.func` and relies on the preset's built-in dict. Most users never have to write a callable in yaml.
2. **Named-percentile shorthand** — users write `func: {poa: "perc_60"}` in yaml; the loader converts the `"perc_N"` string into `perc_wrap(N)` at parse time. Supported pattern: `"perc_<int>"`. Strings that don't match the pattern (e.g. `"mean"`, `"median"`, `"sum"`) pass through unchanged as pandas agg names. Malformed `"perc_..."` strings (`"perc_"`, `"perc_x"`) raise a clear error at load time.

Arbitrary Python callables cannot be loaded from yaml.

### 7.2 `to_yaml` output

- Writes every scalar `param.*` under the given top-level `key` (default `"captest"`), excluding `meas`, `sim`, `_resolved_setup`, and loader callables.
- Writes `test_setup:` and, if any of `reg_cols_meas`, `reg_cols_sim`, `reg_fml`, `rep_conditions` differ from the resolved preset, those keys under `overrides:`. Equivalent values are omitted.
- Writes `meas_path` / `sim_path` only when the class was constructed with those.
- Writes `meas_load_kwargs` / `sim_load_kwargs` when non-empty.
- Never writes data, CapData instances, `regression_results`, filtered data, loader callables, or `scatter_plots` callables.
- Percentile `perc_wrap(N)` values in `rep_conditions.func` are written back as `"perc_N"` strings (round-trippable).
- Emits a single warning if the user overrode `scatter_plots` or a loader callable (these cannot round-trip).
- When `merge_into_existing=True` and the target file already exists and parses as a mapping, preserves untouched top-level keys; only the sub-tree at `key` is overwritten.
- Output order is deterministic; rendered with `yaml.safe_dump(sort_keys=False)`.

## 8. Ported functions: `CapData` -> `CapTest`

The following cross-`CapData` functions are REMOVED from the module level and ported as `CapTest` methods. No backwards-compat shims — the project is below v1.

1. `capdata.captest_results(sim, das, nameplate, tolerance, ...)` -> `CapTest.captest_results(check_pvalues=False, pval=0.05, print_res=True) -> float`. Uses `self.meas`, `self.sim`, `self.ac_nameplate`, `self.test_tolerance`, `self.rep_cond_source`. Internally picks `rc = self.meas.rc if self.rep_cond_source == "meas" else self.sim.rc`, calls `predict_with_pvalue_check` for both predictions, computes cap ratio with the existing logic, uses `self.determine_pass_or_fail(cap_ratio)` and the `print_results` helper.
2. `capdata.captest_results_check_pvalues` -> `CapTest.captest_results_check_pvalues(print_res=False, **kwargs)`. Pulls `regression_results.pvalues`/`params` off `self.meas` and `self.sim`, calls `self.captest_results` twice (with and without the pval check), returns the same styled DataFrame shape as today.
3. `capdata.pick_attr` -> deleted. `rep_cond_source` replaces it as the authoritative decision.
4. `capdata.get_summary(*args)` -> `CapTest.get_summary() -> pd.DataFrame`. Concatenates `self.meas.get_summary()` and `self.sim.get_summary()`. Single-CapData callers continue to use `cd.get_summary()` on the `CapData` method, unchanged.
5. `capdata.overlay_scatters(measured, expected, expected_label="PVsyst")` -> `CapTest.overlay_scatters(expected_label="PVsyst")`. Builds the two scatters internally via `self._resolved_setup["scatter_plots"]` applied to `self.meas` and `self.sim`, then overlays them. Caller no longer has to pre-build the scatters.
6. `plotting.residual_plot(cd1, cd2)` -> `CapTest.residual_plot()`. Uses `self.meas` and `self.sim` directly. `plotting.get_resid_exog_frame(cd)` stays where it is (single-CapData).
7. `capdata.determine_pass_or_fail(cap_ratio, tolerance, nameplate)` -> `CapTest.determine_pass_or_fail(cap_ratio: float) -> tuple[bool, str]`. Uses `self.test_tolerance` and `self.ac_nameplate`.

Kept at module level (pure helpers or single-CapData):

- `predict_with_pvalue_check(cd, rc=None, pval_threshold=0.05)` — single-CapData.
- `run_test(cd, steps)` — single-CapData.
- `plotting.get_resid_exog_frame(cd)` — single-CapData.

Relocated to `src/captest/captest.py` (only callers live there now):

- `print_results(test_passed, expected, actual, cap_ratio, capacity, bounds)`
- `highlight_pvals(s)`

Every test and example that called the module-level forms is updated to use the `CapTest` method. Tests of `pick_attr` are removed. The CHANGELOG entry flags this as a breaking API change.

## 9. `CapData` refactor landscape

Methods in `CapData` (and `plotting.py`) that still assume the ASTM E2848 regression or the canonical four-key `regression_cols`. These are surfaced here as a **dependency inventory**; only section 9.2 is implemented in this PR. Everything else is deferred and tracked in the issue-tracker appendix (section 12).

### 9.1 Addressed by dependency (with additional simplification)

- `CapData.rep_cond` and `CapData.rep_cond_freq` — handled by `rep-cond-calcparams`, merged before this work starts. Immediately after the merge, this spec removes the `func='E2939'` string-trigger branch: `func` now accepts a plain dict or `None` (fallback: `{var: 'mean' for var in rhs}`). Responsibility for supplying the right `df.agg()` dict moves into each `TEST_SETUPS` preset's `rep_conditions` entry (section 6). This is a small net-negative diff in `capdata.py` and removes a string sentinel that's otherwise hard to discover.

### 9.2 In scope for this spec

- `CapData.scatter` and `CapData.scatter_hv` — reimplemented as formula-agnostic thin wrappers that forward to `captest.scatter_default(self, ...)`. `scatter_default` resolves lhs and first rhs variable via `util.parse_regression_formula` rather than hard-coding `"power"` and `"poa"`. The wrapper docstrings call out `CapTest.scatter_plots` as the preferred API for non-default presets.

### 9.3 Deferred blockers (raise or misbehave under `bifi_*` presets)

- `CapData.predict_capacities` — hard-codes `["poa","t_amb","w_vel","power"]`.
- `CapData.reg_scatter_matrix` — hard-codes `["poa","t_amb","w_vel"]` and specific interaction terms.
- `CapData.filter_outliers` — hard-codes `self.floc[["poa","power"]]`.
- `CapData.scatter_filters` — hard-codes `["power","poa"]` axes.

### 9.4 Deferred warnings (only problematic for custom presets)

- `CapData.timeseries_filters` — hard-codes `"power"` as the lhs. Fine for shipped presets because of the `power` lhs naming convention.
- `CapData.filter_power` default `columns` — same applicability.
- `CapData.filter_irr` default column and `CapData.__get_poa_col` — depends on `regression_cols["poa"]` existing.
- `CapData.filter_sensors` default `perc_diff` — same shape as above.
- `CapData.agg_sensors` default `agg_map`.
- `CapData.set_regression_cols` — hardcoded kwargs.

### 9.5 Unaffected

- `CapData.fit_regression`, `captest_results` / `predict_with_pvalue_check`, `plotting.residual_plot` / `get_resid_exog_frame`, `filter_time`, `filter_shade`, `filter_pvsyst`, `filter_missing`, `filter_clearsky`, `filter_pf`, `update_summary`, `get_summary` (single-CapData), `reset_filter`, `reset_agg` — all agnostic of the regression formula.

## 10. Testing plan

All tests use pytest per the project rule. Arrange-Act-Assert. No `unittest.TestCase`.

### 10.1 Test file layout

- New: `tests/test_captest.py` — all tests for `CapTest`, the `TEST_SETUPS` registry, the three scatter callables, `print_results`, `highlight_pvals`, and the `load_config`/`resolve_test_setup`/`validate_test_setup` helpers.
- Updated: `tests/test_CapData.py` — migrates every existing test that called the module-level `captest_results` / `captest_results_check_pvalues` / `overlay_scatters` / `get_summary(meas, sim)` to the new `CapTest` methods. Removes tests of `pick_attr` and the module-level `determine_pass_or_fail`.
- Updated: `tests/test_plotting.py` (existing or new) — migrates `residual_plot` tests to `CapTest.residual_plot`.
- Updated: `tests/conftest.py` — adds the shared fixtures below.

### 10.2 Shared fixtures

```python
@pytest.fixture
def meas_cd_default():
    """Minimal CapData loaded from the existing tests/data/example_meas_data.csv."""

@pytest.fixture
def sim_cd_default():
    """Minimal PVsyst CapData loaded from tests/data/pvsyst_example_HourlyRes_2.CSV."""

@pytest.fixture
def ct_default(meas_cd_default, sim_cd_default):
    """CapTest(test_setup='e2848_default', ac_nameplate=..., test_tolerance='- 4')
    with meas/sim assigned and setup() run."""

@pytest.fixture
def ct_etotal(meas_cd_default, sim_cd_default):
    """test_setup='bifi_e2848_etotal', bifaciality=0.15, setup() run."""

@pytest.fixture
def ct_bifi_power_tc(meas_cd_default, sim_cd_default):
    """test_setup='bifi_power_tc', bifaciality/power_temp_coeff/base_temp set,
    setup() run."""

@pytest.fixture
def captest_yaml(tmp_path, meas_cd_default, sim_cd_default):
    """Writes a minimal yaml file to tmp_path for from_yaml tests; returns the
    path."""
```

All fixtures use `scope="function"` — setup is re-run per test so state is hermetic. On the example data `process_regression_columns` runs in <100 ms, so the overhead is acceptable.

### 10.3 Test classes

`TestConstruction`:

- `test_bare_init_has_defaults`, `test_bare_init_accepts_kwargs`, `test_bare_init_rejects_unknown_kwarg`.
- `test_from_params_with_capdata_instances_triggers_setup`.
- `test_from_params_with_paths_loads_data`.
- `test_from_params_pre_built_wins_over_path`.
- `test_from_params_partial_leaves_unset_defers_setup`.

`TestFromYaml`:

- `test_from_yaml_happy_path`.
- `test_from_yaml_relative_paths_resolve_to_yaml_dir`.
- `test_from_yaml_unknown_key_raises_with_suggestion`.
- `test_from_yaml_missing_test_setup_raises`.
- `test_from_yaml_custom_setup_requires_overrides`.
- `test_from_yaml_conflicting_reg_fml_raises`.
- `test_from_yaml_null_values_equivalent_to_absence`.

`TestSetup`:

- `test_setup_requires_meas_and_sim`.
- `test_setup_propagates_downstream_attrs_to_both_cd`.
- `test_setup_wires_regression_cols_and_formula` — parametrized over shipped presets.
- `test_setup_wires_tolerance`.
- `test_setup_runs_process_regression_columns_on_both`.
- `test_setup_rerun_resets_data_filtered`.
- `test_setup_verbose_default_true`.
- `test_setup_returns_self`.

`TestTestSetupsRegistry`:

- `test_each_shipped_preset_validates` — parametrized over `TEST_SETUPS`.
- `test_each_preset_lhs_is_power`.
- `test_validate_rejects_unknown_keys_in_entry`.
- `test_validate_rejects_missing_required_keys`.
- `test_validate_rejects_non_callable_scatter_plots`.
- `test_validate_rejects_formula_vars_missing_from_reg_cols`.
- `test_custom_setup_requires_all_three_overrides`.

`TestScatterPlotsCallables`:

- `test_scatter_default_on_ct_default_returns_layout`.
- `test_scatter_etotal_uses_e_total_column`.
- `test_scatter_bifi_power_tc_layout_has_two_panels`.
- `test_ct_scatter_plots_dispatches_to_resolved_setup`.
- `test_capdata_scatter_hv_wrapper_delegates_to_scatter_default`.

`TestPortedMethods`:

- `test_captest_results_matches_legacy_for_default_preset` — reference expected value computed from direct `predict_with_pvalue_check` calls; equality to ~1e-10.
- `test_captest_results_uses_rep_cond_source_meas` / `..._sim`.
- `test_captest_results_rejects_mismatched_formulas`.
- `test_captest_results_check_pvalues_returns_styled_df`.
- `test_determine_pass_or_fail_uses_ct_attrs`.
- `test_get_summary_concatenates_meas_and_sim`.
- `test_overlay_scatters_returns_overlay`.
- `test_residual_plot_returns_layout`.

`TestToYamlAndRoundTrip`:

- `test_to_yaml_writes_curated_set`.
- `test_to_yaml_writes_paths_only_when_loaded_from_paths`.
- `test_to_yaml_omits_override_keys_equal_to_preset`.
- `test_to_yaml_round_trip` — `CapTest.from_yaml(path)` -> `ct.to_yaml(path2)` -> both files parse to equivalent dicts (ignoring non-serializable `scatter_plots`).
- `test_to_yaml_warns_when_custom_scatter_plots_callable`.

`TestDownstreamPropagation`:

- `test_bifaciality_flows_into_e_total`.
- `test_power_temp_coeff_flows_into_power_temp_correct`.
- `test_base_temp_flows_into_power_temp_correct`.

`TestLoaderInjection`:

- `test_default_meas_loader_is_load_data` — without an explicit `meas_loader`, `from_params(meas_path=...)` calls `captest.io.load_data`.
- `test_default_sim_loader_is_load_pvsyst`.
- `test_custom_meas_loader_called` — supply a mock loader; verify it was called with `(path, **meas_load_kwargs)`.
- `test_custom_meas_loader_kwargs_splatted` — verify every key in `meas_load_kwargs` flows through as a kwarg to the loader.
- `test_loader_callables_not_serialized_in_to_yaml` — setting `meas_loader=custom` and calling `to_yaml` writes nothing for it (warning emitted once).

`TestRepCondConvenience`:

- `test_rep_cond_calls_cd_rep_cond_with_resolved_defaults` — `ct.rep_cond()` forwards the preset's `rep_conditions` unchanged to `cd.rep_cond`.
- `test_rep_cond_partial_merge_overrides` — `ct.rep_cond(percent_filter=10)` replaces only `percent_filter`; `func` preserved.
- `test_rep_cond_func_partial_merge` — `ct.rep_cond(func={"poa": perc_wrap(55)})` replaces only the POA entry in `func`; `t_amb`/`w_vel` preserved.
- `test_rep_cond_which_sim` — `ct.rep_cond(which="sim")` calls `self.sim.rep_cond(...)` not `self.meas`.
- `test_rep_conditions_override_from_init_partial_merges_onto_preset` — setting `rep_conditions={"percent_filter": 10}` in `CapTest(...)` leaves everything else from the preset intact after `setup()`.
- `test_each_preset_rep_conditions_round_trips_through_rep_cond` — parametrized over presets; `ct.rep_cond()` runs without errors on each.

`TestYamlPercShorthand`:

- `test_perc_N_string_converts_to_perc_wrap` — yaml with `overrides.rep_conditions.func: {poa: "perc_55"}` loads to a dict whose `poa` value, when applied to a sample Series, matches `perc_wrap(55)` applied to the same series.
- `test_mean_string_passes_through` — `"mean"` in yaml remains a plain `"mean"` string.
- `test_invalid_perc_string_raises` — `"perc_x"`, `"perc_"` raise at load time with a clear message.
- `test_to_yaml_emits_perc_N_for_perc_wrap` — round-trip: a `CapTest` with `rep_conditions.func["poa"] == perc_wrap(55)` serializes that entry as the string `"perc_55"`.

`TestYamlKeyParametrization`:

- `test_from_yaml_nested_under_captest_key_by_default`.
- `test_from_yaml_custom_key_e_g_captest_bifi`.
- `test_from_yaml_missing_key_raises_with_available_keys_listed`.
- `test_to_yaml_writes_under_custom_key`.
- `test_to_yaml_merge_into_existing_preserves_other_top_level_keys`.

`TestIntegration` (end-to-end per preset):

- `test_end_to_end_e2848_default` — full canonical sequence: load data -> `CapTest.setup()` -> `ct.meas.filter_irr(ct.min_irr, ct.max_irr)` -> `ct.meas.filter_shade(fshdbm=ct.fshdbm)` (if applicable) -> `ct.meas.filter_time(start=..., end=...)` -> `ct.meas.rep_cond(...)` -> `ct.meas.fit_regression()`; same on sim; `ct.captest_results()` returns a cap ratio in a realistic range (e.g. `0.8 < cap_ratio < 1.2`) without errors or warnings.
- `test_end_to_end_bifi_e2848_etotal` — same shape; cap ratio plausible; `e_total` column materialized and present in `ct.meas.data`.
- `test_end_to_end_bifi_power_tc` — same shape; cap ratio plausible; temperature-corrected `power` column materialized; scatter_plots layout has two panels.

### 10.4 Acceptance criteria

- `rep-cond-calcparams` is merged into the working branch first; `rep_cond`/`rep_cond_freq` support arbitrary formulas.
- `src/captest/captest.py` exists with `CapTest`, `TEST_SETUPS`, `scatter_default`, `scatter_etotal`, `scatter_bifi_power_tc`, `print_results`, `highlight_pvals`, `load_config`, `resolve_test_setup`, `validate_test_setup`.
- `src/captest/__init__.py` exports `CapTest` and `TEST_SETUPS`.
- All seven cross-`CapData` functions listed in section 8 are removed from `capdata.py`/`plotting.py` and present as `CapTest` methods.
- `CapData.scatter` and `CapData.scatter_hv` are formula-agnostic thin wrappers around `scatter_default`.
- All three `TestIntegration` tests pass against the existing example data.
- `CapTest.from_yaml`/`CapTest.from_params` round-trip through `CapTest.to_yaml` for the curated set of params.
- All new lines in `src/captest/captest.py` are covered by the `tests/test_captest.py` suite in the `just test-cov` HTML report.
- `just test` (Python 3.12) passes; `just test-wo-warnings 3.13` also passes.
- `just lint` and `just fmt` are clean.
- CHANGELOG updated: breaking API entry for the removed module-level comparison functions; new-feature entry for `CapTest`; naming-convention note (lhs key is always `power`).

## 11. Implementation sequence

All test writing follows `.agents/skills/unit-tests/SKILL.md` (TDD-first for new code; coverage checks on net-new lines). User-facing doc updates follow `.agents/skills/docs-update/SKILL.md`. PR creation at the end follows `.agents/skills/create-pr/SKILL.md`.

1. Merge `rep-cond-calcparams` into `captest-class`. Resolve any conflicts; run `just test` before proceeding.
2. Simplify `rep_cond` immediately after the merge: remove the `func='E2939'` string-trigger so `func` accepts a plain dict or `None` with a `{var: 'mean' for var in rhs}` fallback. Update existing `rep_cond` tests (parametrize over the three `TEST_SETUPS` `rep_conditions["func"]` dicts).
3. Relocate `print_results` and `highlight_pvals` to a new `src/captest/captest.py` stub. Tests for those helpers continue to pass via the new import.
4. Add `TEST_SETUPS` (with `rep_conditions` on each preset), `validate_test_setup`, `resolve_test_setup`, `load_config`, and the three scatter callables. Unit-test the registry and callables standalone.
5. Reimplement `CapData.scatter` / `CapData.scatter_hv` as wrappers calling `scatter_default`. Update unit tests.
6. Add the `CapTest` class with `param.*` attributes (including the four loader params and `rep_conditions`), constructors, `setup()`, and the `rep_cond` / `scatter_plots` convenience methods. Wire `_downstream_attrs`. Add `TestConstruction`, `TestFromYaml`, `TestSetup`, `TestDownstreamPropagation`, `TestLoaderInjection`, `TestRepCondConvenience` tests.
7. Port the seven cross-`CapData` functions onto `CapTest`. Remove the module-level versions. Update all callsite tests. Add `TestPortedMethods`.
8. Implement `to_yaml` with `key=` and `merge_into_existing=` kwargs, plus the `"perc_N"` string shorthand for `rep_conditions.func` values. Add `TestToYamlAndRoundTrip`, `TestYamlPercShorthand`, `TestYamlKeyParametrization`.
9. Add the three `TestIntegration` tests.
10. Update CHANGELOG and user-facing docs (quickstart / API reference), following the docs-update skill. Add the issue-tracker entries from section 12 as GitHub issues.
11. Open the PR following the create-pr skill.

## 12. Issue tracker appendix

Proposed follow-up issues for the deferred refactors in section 9. Each is independently testable against the three shipped `TEST_SETUPS` presets using the `CapTest` fixtures introduced in this spec.

- **[CapData] Generalize `predict_capacities` to arbitrary regression formulas** — Blocker under non-default presets. Replace hardcoded `["poa","t_amb","w_vel","power"]` with the lhs+rhs returned by `util.parse_regression_formula(self.regression_formula)`.
- **[CapData] Rework or deprecate `reg_scatter_matrix` for non-E2848 formulas** — Blocker under non-default presets. Either derive columns and interactions from the formula, or raise a helpful error when the formula is not the canonical E2848.
- **[CapData] Generalize `filter_outliers` x/y columns** — Blocker under non-default presets. Accept `x_key`/`y_key` kwargs defaulting to `"poa"`/`"power"`; pick from the formula when absent.
- **[CapData] Generalize `scatter_filters` axes** — Blocker under non-default presets. Drive from the formula (lhs as y, first rhs as x), or convert to a per-preset `scatter_filters` callable analogous to `scatter_plots`.
- **[CapData] Drive `timeseries_filters` lhs from regression formula** — Warning. Fine for shipped presets under the `power` lhs convention; breaks for custom presets with other lhs. Replace `"power"` with `parse_regression_formula(...)[0][0]`.
- **[CapData] Default `filter_power` column from regression formula** — Same severity and fix shape as above.
- **[CapData] Configurable primary irradiance key for `filter_irr`/`filter_sensors`** — Warning. Add `cd.primary_irr_key` (set from `CapTest.setup()`), consumed by `__get_poa_col` and `filter_sensors` defaults.
- **[CapData] Generalize `agg_sensors` default `agg_map`** — Low priority. Derive default from the regression formula, or document as E2848-only.
- **[CapData] `set_regression_cols` accepts arbitrary keys** — Low priority. Add `**kwargs` support for non-canonical regression variables.
