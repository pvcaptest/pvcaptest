# Design: Test Setups as Pure-Data Documents

**Date:** 2026-09-22
**Branch:** `reg-cols-serialization`
**Package:** `captest`
**Background:** `docs/superpowers/notes/2026-09-22-test-setup-representation-options.md`
(the options survey, three roborev design reviews, and the library comparison).

## Problem

A capacity-test setup — the `TEST_SETUPS` entry that says which columns feed
the regression, how they are aggregated and derived, the formula, and the
reporting-condition rules — is today a Python dict whose `reg_cols_meas` /
`reg_cols_sim` trees identify node kinds by *shape* (`(str, str)` is an
aggregation, `(callable, dict)` is a calculation, a bare string is a group id
or a column name) and hold live function objects.

That representation cannot survive a file. Commit `7d0c2f4` patched the
yaml round trip with `encode_reg_cols` / `decode_reg_cols`, and review found
the patch itself ambiguous: any two-element list becomes a tuple (so a
literal list argument is mistaken for an aggregation), and a callable in an
argument position comes back as a string. The pft-mono `ctsweep` store,
which keeps every run's config as canonical JSON and content-hashes it, is
where the defect surfaced, and it is where the intended use lives: an agent
defining and running many test types on one project, each setup stored,
deduplicated, validated before it runs, and reproducible afterwards.

## Goal

A test setup is a **document**: yaml or json made only of strings, numbers,
booleans, lists and tagged mappings. Every function is referenced by name
through a registry, never held as an object. A pydantic model is the typed,
validated view of the document; the tree keeps its nested, recursive shape
and bottom-up evaluation, and a calculation's output column — named for the
calculation — is what the parent consumes. Only the *encoding* of a node
changes.

Concretely:

1. Node kinds cannot be confused: reference vs literal is decided by a tag,
   never by shape or position, and nothing in a setup can fail to serialize.
2. A setup validates in three tiers — structural, project fit, runtime — and
   the first two run without loading data, with path-bearing errors an
   authoring agent can act on.
3. Presets, user setups and agent-authored setups are one kind of thing,
   loaded through one path.
4. `canonical_json(setup.to_dict())` is the setup's identity.

Backward compatibility with the tuple grammar is **not** a goal. The tuple
form is removed outright; downstream consumers in pft-mono migrate when
they take this captest version.

## Document grammar

### Nodes

Every node is a mapping with exactly one discriminating key, or a literal.
There is no shorthand: a bare string is always a literal.

| Node | Meaning |
|---|---|
| `{group: irr_poa, agg: mean}` | aggregate a column group with `CapData.agg_group`; `agg` defaults to `mean` |
| `{column: E_Grid}` | a raw column of `data` by name |
| `{calc: cell_temp, args: {...}}` | a registered calculation; `args` are keyword-only, each value a node; its output column is named `cell_temp` |
| `"cdte"`, `3.5`, `true`, `null`, `[1, 2]` | a literal argument, passed through unchanged |

Rules:

- A mapping with none of `group` / `column` / `calc` is an error, not a
  literal. Dict literals are not supported; no calculation takes one.
- Literal floats must be finite (`canonical_json` rejects NaN and Infinity).
  `null` is allowed (`altitude_override=None` is a real argument).
- Lists are lists of scalars only.
- `calc` always means *execute and supply the output column*. It is never a
  way to pass a function object. A calculation that must choose a function
  takes a plain string it resolves itself, the way `agg` does.
- `args` bind by keyword only. The registered function's positional order
  is never used; the first review found `cell_temp(mean(irr_poa),
  mean(temp_bom))` would silently swap irradiance and module temperature
  against the real signature `cell_temp(data, bom, poa, ...)`.
- The allowed `agg` values are those `agg_group` accepts today.
- A top-level `reg_cols` value (a formula variable's node) must be a
  `Group`, `Column` or `Calc`; a literal there is a tier-1 error. Literals
  belong only inside `args`.

### Setup

```yaml
name: bifi_power_tc_etotal_rear_shade_sim
description: Bifacial, temperature-corrected power, E_total, rear shade in the model.
derived_from: e2848_default            # provenance only; never used to fill fields
reg_fml: power ~ poa
meas:
  reg_cols:
    power:
      calc: power_temp_correct
      args:
        power: {group: real_pwr_mtr, agg: sum}
        cell_temp:
          calc: cell_temp
          args:
            poa: {group: irr_poa, agg: mean}
            bom: {group: temp_bom, agg: mean}
    poa:
      calc: e_total
      args:
        poa:  {group: irr_poa,  agg: mean}
        rpoa: {group: irr_rpoa, agg: mean}
sim:
  reg_cols:
    power:
      calc: power_temp_correct
      args: {power: {column: E_Grid}, cell_temp: {column: TArray}}
    poa:
      calc: e_total
      args:
        poa:  {column: GlobInc}
        rpoa: {calc: rpoa_pvsyst, args: {globbak: {column: GlobBak}, backshd: {column: BackShd}}}
params:
  rear_shade: 0
rep_conditions:
  irr_bal: false
  percent_filter: 20
  func: {poa: perc_60}
scatter_plots: etotal
```

Each side has its own tree because the sides need different *derivations*,
not merely different source columns (measured cell temperature is computed;
PVsyst's `TArray` is read; measured precipitable water is computed by
`precipitable_water_gueymard`; PVsyst's `PrecWat` is scaled by 100).

A setup is a complete value. `derived_from` (a preset name or a content
digest) is provenance; loading never fills omitted fields from it. A
variant is made by copying and editing, or by `TestSetup.derive(base,
**changes)` in Python, which returns a new complete setup with
`derived_from = base.name`. `derive` merges `reg_cols_meas` /
`reg_cols_sim` **key by key** onto the base side (each formula variable's
node is replaced whole; `null` removes the variable) and replaces every
other field wholesale, so pointing one term at a different sensor group is
a one-line change:

```python
TestSetup.derive(TEST_SETUPS["e2848_default"],
                 reg_cols_meas={"power": {"group": "real_pwr_inv", "agg": "sum"}})
```

Removing a term has one knock-on effect `derive` handles itself: after the
formula and both sides are resolved, any `rep_conditions.func` entry whose
variable is no longer on the formula's right-hand side is **pruned**, so
dropping `w_vel` from `e2848_default` does not leave the inherited
`func.w_vel` behind for tier 1 to reject. Pruning happens after
`_merge_rep_conditions` has applied any `rep_conditions` override, and only
removes entries; it never adds or changes one.

The stored value is always the complete result. Diffs against a base are
computed, never stored.

`params` maps a test-level parameter name to the value this setup
*requires*. It is neither a default nor an override: at tier 2, on each side
and for each `Calc` on that side whose registry entry lists the parameter
in `requires_params`, the **effective value that calculation would
receive** must equal the declared value. The effective value is resolved
with the same precedence evaluation uses (below): the document's explicit
`args` entry, else the `CapData` attribute when present, else the
function's default. A side with no calculation that uses the parameter is
not checked — `CapTest.setup()` propagates `rear_shade` onto `meas` only,
so on `sim` the `e_total` node resolves `rear_shade` to its default `0` and
passes. The rear-shade `_sim` presets declare `params: {rear_shade: 0}` so
they refuse to double-count a non-zero measured `rear_shade`. Absent keys
are unconstrained.

`rep_conditions.func` values are the strings `mean`, `median` or `perc_N`,
resolved at setup time by `util._resolve_perc_string` (the current
`perc_wrap` encoding, now the only form). `scatter_plots` is a name in
`captest.SCATTER_REGISTRY` (`default`, `etotal`, `bifi_power_tc`).

### Output column names

Every `Group` and `Calc` node *produces* a column of `data`:

| Node | Column it writes |
|---|---|
| `{group: g, agg: a}` | `<g>_<a>_agg` (the existing `agg_group` naming) |
| `{calc: c}` | `c` (the registry name; there is no alias) |

Two producers on one side may write the same column only if they are the
**same node** — equal kind, and for `Calc` equal registry name and equal
`args`. Anything else that would write a column another node writes is a
tier-1 error naming both paths: the same calculation twice with different
`args`, or a registry name that happens to equal an aggregation's column
(`<g>_<a>_agg`). The check needs no project: the names are determined by
the document alone. An alias key (`as:`) was considered and dropped — no
existing or planned setup needs the same calculation twice with different
inputs on one side, and the collision error is the safer default until one
does.

## Architecture

### Modules and import direction

```
calcparams.py   CALC_REGISTRY, @register_calc, the calculation functions
      ▲
setup.py        Group / Column / Calc / Side / TestSetup, load, to_dict,
                content_digest, tier 1 (validators), tier 2 (check_project_fit)
      ▲
util.py         transform_calc_params over nodes, process_reg_cols
capdata.py      CapData.regression_cols holds dict[str, Node]; custom_param(output=)
plotting.py     SCATTER_REGISTRY; calc_tc_power_column over nodes
      ▲
captest.py      TEST_SETUPS: dict[str, TestSetup] loaded from setups/*.yaml;
                resolve_test_setup returns a TestSetup; CapTest.resolved_setup
```

`setup.py` never imports `capdata` or `captest`. Tier 2 takes the `CapData`
as a duck-typed runtime argument (`column_groups`, `data.columns`, and the
attributes named by `requires_params`), the same rule `filters.py` follows.

### `setup.py` — the model

```python
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class Group(_Frozen):
    group: str
    agg: Literal["mean", "sum", "median", "min", "max"] = "mean"

class Column(_Frozen):
    column: str

class Calc(_Frozen):
    calc: str
    args: dict[str, "Node"] = Field(default_factory=dict)

Scalar = None | bool | int | float | str
Literal_ = Scalar | list[Scalar]

NODE_TAGS = ("group", "column", "calc")

def _node_kind(node):
    """Name the union member ``node`` belongs to, for the discriminator.

    A mapping is identified by which one of ``NODE_TAGS`` it carries; a
    mapping with none or several is unclassifiable and returns ``None`` so
    pydantic reports ``union_tag_not_found`` at that path. An already-built
    model reports its own class; anything else is a literal.
    """
    if isinstance(node, dict):
        present_tags = [tag for tag in NODE_TAGS if tag in node]
        return present_tags[0] if len(present_tags) == 1 else None
    if isinstance(node, BaseModel):
        return type(node).__name__.lower()
    return "literal"

Node = Annotated[
    Union[Annotated[Group, Tag("group")], Annotated[Column, Tag("column")],
          Annotated[Calc, Tag("calc")], Annotated[Literal_, Tag("literal")]],
    Discriminator(_node_kind),
]

class Side(_Frozen):
    reg_cols: dict[str, Node]

class RepConditions(_Frozen):
    """Keyword arguments to ``CapData.rep_cond`` in document form."""
    func: dict[str, str] = Field(default_factory=dict)   # "mean" | "median" | "perc_N"
    w_vel: float | None = None
    irr_bal: bool = False
    percent_filter: float = 20        # numeric only: filters.RepCond.percent_filter is param.Number
    front_poa: str = "poa"
    rc_kwargs: dict[str, Scalar] | None = None

#: CapTest params a setup may constrain; CapTest._downstream_attrs is this tuple.
DOWNSTREAM_PARAMS = ("bifaciality", "bifacial_frac", "rear_shade", "power_temp_coeff",
                     "base_temp", "module_type", "racking", "spectral_module_type",
                     "airmass_model", "altitude_override")

class TestSetup(_Frozen):
    name: str
    description: str = ""
    derived_from: str | None = None
    reg_fml: str
    meas: Side
    sim: Side
    params: dict[str, Scalar] = Field(default_factory=dict)
    rep_conditions: RepConditions = Field(default_factory=RepConditions)
    scatter_plots: str = "default"
```

`Group` is hashable (frozen, scalar fields) and is the `agg_cache` key.
`TestSetup` has value equality and **no `__hash__`**: nested containers stay
mutable under `frozen=True` and `1 == 1.0` would dump differently.

Identity and serialization:

- `to_dict()` = `model_dump(mode="json")`. Every field is
  written, including those whose value is `null`, and defaults are
  materialised (`{group: temp_amb}` dumps as `{group: temp_amb, agg:
  mean}`), so identity belongs to the *normalised* document and a `null`
  never changes meaning by being dropped. Normalisation is idempotent: `load(to_dict(load(d))).to_dict() ==
  load(d).to_dict()` — that, not byte equality with the author's input, is
  the round-trip property. Preset yaml files may omit defaults; the
  normalised form is what is compared and digested.
- `content_digest()` = sha256 of `canonical_json(to_dict())`, using the same
  canonical form ctsweep's `identity.canonical_json` defines (sorted keys, no
  whitespace, `ensure_ascii=False`, `allow_nan=False`). `util` gains
  `canonical_json` so captest does not import ctsweep.
- `TestSetup.load(source)` accepts a path (`.yaml` / `.json`), a string, or a
  mapping, and always validates through the model. yaml is read with
  `yaml.safe_load`; documents may not rely on yaml's own type guessing.
- `TestSetup.to_yaml(path)` / `to_json(path)` write `to_dict()`.
- `TestSetup.json_schema()` returns `model_json_schema()`. The schema is a
  **shape** contract for authoring agents (`calc` is an unrestricted string
  in it); the registry, formula and collision checks are model validators.
  "Valid" means "loads through the model".

### `calcparams.py` — the registry

```python
@dataclass(frozen=True)
class CalcEntry:
    func: Callable
    requires_params: tuple[str, ...]   # CapData attributes custom_param injects by name
    requires_import: tuple[str, ...]   # optional packages the function needs

CALC_REGISTRY: dict[str, CalcEntry] = {}

def register_calc(name=None, *, requires_params=(), requires_import=()):
    """Decorator registering a calculation under ``name`` (default ``__name__``)."""
```

Every public calculation is decorated; the function bodies do not change:

```python
@register_calc(requires_params=("power_temp_coeff", "base_temp"))
def power_temp_correct(data, power, cell_temp, power_temp_coeff=None, base_temp=25, verbose=True):
    ...

@register_calc(requires_params=("module_type", "racking"))
def cell_temp(data, bom, poa, module_type="glass_cell_poly", racking="open_rack", verbose=True):
    ...

@register_calc(requires_params=("spectral_module_type",), requires_import=("pvlib",))
def spectral_factor_firstsolar(...):
    ...
```

`requires_params` names the parameters `CapTest.setup()` propagates through
`DOWNSTREAM_PARAMS` and `custom_param` fills by `getattr(cd, key)` when the
document does not supply them. A document **may** supply one explicitly
(`args: {base_temp: 20}`), which is how a single calculation overrides the
test-wide value today. Precedence, for every parameter of a calculation:

1. the document's `args` entry, if the key is present — including an
   explicit `null`, which reaches the function as `None`
   (`absolute_airmass(pressure=None)` means "use the default pressure");
2. else the `CapData` attribute of that name, if the attribute exists and
   is not `None`;
3. else the function's own default.

`verbose` is supplied by the evaluator and is neither an argument nor a
requirement. `data` is injected.

A test in `tests/test_calcparams.py` checks every entry against its
function: `requires_params ⊆ signature.parameters`; each name in
`requires_import` appears in the function's source (`inspect.getsource`);
and no function whose source references `pvlib` lacks the declaration. The
`cell_temp`/pvlib mistake in the survey note is the reason this test exists.

User code registers custom calculations with the same decorator. A document
stores only the name. **Loading a document never imports code the document
names**; a name missing from the registry is a tier-1 error with a
`difflib` did-you-mean hint, as `step_from_config` does for filters.

`captest.SCATTER_REGISTRY = {"default": scatter_default, "etotal":
scatter_etotal, "bifi_power_tc": scatter_bifi_power_tc}` lives beside the
three functions, which are defined in `captest.py`. Because `setup.py`
cannot import `captest.py`, the model only requires `scatter_plots` to be a
non-empty string; membership is checked by `captest.py` when `TEST_SETUPS`
loads and in `resolve_test_setup`, with the same did-you-mean hint. It is
still a load-time (tier-1) failure for every path a setup enters `CapTest`
through.

### Evaluation — `util.transform_calc_params`

Same recursive function, four branches, tuple branches removed:

```python
def transform_calc_params(node, cd, agg_cache=None, verbose=True):
    if isinstance(node, dict):                      # Side.reg_cols or Calc.args
        return {k: transform_calc_params(v, cd, agg_cache, verbose) for k, v in node.items()}
    if isinstance(node, Group):
        return _get_or_create_aggregation(node, cd, agg_cache, verbose)   # cache keyed on node
    if isinstance(node, Column):
        if node.column not in cd.data.columns:
            raise KeyError(f"column {node.column!r} not in data")
        return node.column
    if isinstance(node, Calc):
        entry = CALC_REGISTRY[node.calc]
        resolved = transform_calc_params(node.args, cd, agg_cache, verbose)
        cd.custom_param(entry.func, output=node.calc, verbose=verbose, **resolved)
        return node.calc
    return node                                     # literal
```

`CapData.custom_param(func, *, output=None, verbose=True, **kwargs)` gains
the `output` keyword (the column it writes; `func.__name__` when omitted,
so direct callers are unaffected), drops its unused `*args`, and injects a
`CapData` attribute **only for parameters absent from `kwargs`** — an
explicit `None` is passed through, implementing precedence rule 1 above.
(Today it also injects on `None`; the change is deliberate and tested.)
The existing guard that refuses to inject a parameter whose name is also a
column group id applies to absent parameters only. `process_reg_cols`
still flattens `regression_cols` to `{variable: column}` in place;
`regression_cols_preprocess` keeps the `Side` for `to_mapping`.

Generated-column ownership: a `Calc` overwrites its output column on every
`process_regression_columns`, so a re-run with a different setup never
reuses a stale column of the same name. Aggregations keep today's reuse of
an existing `<group>_<agg>_agg` column — the same group and function give
the same numbers regardless of setup. The output-name rule under "Output
column names" guarantees, before anything runs, that no two different
nodes on a side write the same column; two *identical* nodes may, and the
second evaluation overwrites with identical values (caching is a possible
later optimisation, not part of this design).

### `captest.py` — presets and `CapTest`

The directory is `setups/`, not `test_setups/`, because `captest.test_setups`
is an existing exported function.

- `src/captest/setups/<name>.yaml`, one per current `TEST_SETUPS` key,
  shipped as package data. `TEST_SETUPS: dict[str, TestSetup]` is built at
  import by loading the directory, so the loader runs on every import and
  every preset is validated at tier 1 then.
- `resolve_test_setup(name, overrides)` returns a `TestSetup`. For a named
  preset it first partial-merges `overrides["rep_conditions"]` onto the
  preset's with `_merge_rep_conditions` (today's behaviour, unchanged), then
  calls `TestSetup.derive(TEST_SETUPS[name], **overrides)`, which merges
  `reg_cols_*` key by key and replaces every other field wholesale. For
  `"custom"` there is no base: it builds a `TestSetup` from the overrides
  and requires complete `reg_cols_meas`, `reg_cols_sim` and `reg_fml` as
  today. `validate_test_setup` is deleted; the model validates, and
  `captest.py` adds the `scatter_plots` membership check.
- `CapTest._downstream_attrs` becomes `setup.DOWNSTREAM_PARAMS` so the
  `params` tier-1 check and the propagation loop share one list.
- The `CapTest` params `reg_cols_meas` / `reg_cols_sim` take a mapping of
  formula variable to node in document form and are **merged key by key**
  onto the named preset's side: a key present in the override replaces that
  variable's whole node; a key absent from the override keeps the preset's
  node; a key whose value is `null` removes the variable (needed when an
  overridden `reg_fml` drops a term). So

  ```yaml
  overrides:
    reg_cols_meas:
      poa: {group: irr_ghi, agg: mean}
  ```

  swaps the irradiance term to the GHI sensors for a stowed tracker and
  leaves `power`, `t_amb` and `w_vel` as the preset defines them. `reg_fml`,
  `rep_conditions` (partial-merged, as today), `scatter_plots` (a name) and,
  new, `params` are the other overrides; `reg_fml`, `scatter_plots` and
  `params` replace wholesale. Under `test_setup: custom` there is no base
  and both sides must be complete. Test-level parameters (`bifaciality`,
  `power_temp_coeff`, ...) are not part of the setup document; they remain
  `CapTest` params set from yaml or kwargs exactly as today, and interact
  with a setup only through its `params` constraints.
- `CapTest.resolved_setup` is a `param.ClassSelector(class_=TestSetup)`
  replacing `_resolved_setup`; `setup()` assigns
  `cd.regression_cols = dict(resolved.meas.reg_cols)` (a shallow copy of an
  immutable tree suffices), runs tier 2 on each side *after* the prep stage
  has been applied and *before* `process_regression_columns`, then proceeds
  as now. `scatter_plots()` and `captest_results()` read `resolved_setup`.
- `to_mapping()` keeps the current yaml layout — `test_setup: <name>` plus
  an `overrides` sub-mapping — with each override written in document form
  via `to_dict()` of the relevant piece. For `reg_cols_*` it writes the
  **difference** from the named preset: only the formula variables whose
  node differs, plus `null` for variables the preset has and the resolved
  setup lacks. A file therefore says exactly what was changed, and
  `from_mapping` merging it back onto the same preset reproduces the
  resolved setup (a tested round trip). Under `custom` both sides are
  written in full. `_encode_override`,
  `encode_reg_cols`, `decode_reg_cols` and `_serialize_rep_conditions`'s
  `perc_wrap` branch are deleted. A consumer that needs the complete
  normalised setup (ctsweep) reads `ct.resolved_setup.to_dict()` and
  `content_digest()`; storing it is ctsweep's concern (out of scope).
- `from_mapping()` passes `overrides` values straight to the params; the
  model validates.

### `plotting.py`

`_missing_column_groups(node, available_groups)` walks nodes: `Group` leaves
are checked, `Column` and literals are not. `calc_tc_power_column(cd,
tc_power_calc)` takes a `dict[str, Node]` whose `"power"` entry is a `Calc`.

## Extending

What a contributor does after this lands, and nothing more:

**A new calculation** — in `calcparams.py`, write the function with `data`
as its first parameter and keyword parameters for everything else, give it
a NumPy-style docstring, and decorate it:

```python
@register_calc(requires_params=("power_temp_coeff",), requires_import=())
def power_temp_correct_clipped(data, power, cell_temp, power_temp_coeff=None, cap=0.98):
    ...
```

`requires_params` lists the parameters that come from `CapData` attributes
(the `DOWNSTREAM_PARAMS` names); `requires_import` lists optional packages
the body imports. The registry-declaration test in `tests/test_calcparams.py`
runs over every entry automatically and fails if the declaration disagrees
with the source. A project-specific calculation is registered the same way
in the project's own code before the setup that names it is loaded.

**A new test setup** — write `src/captest/setups/<name>.yaml` in the
grammar above (copy the nearest preset and edit). It is validated at tier 1
the next time `captest` is imported, and the parametrised preset tests
(load, normalise, digest, oracle) pick it up by directory listing. Two
fixture lines accompany it: its digest in `tests/data/setup_digests.json`
and, if its numbers should be regression-locked, an oracle captured with
the existing capture script into `tests/data/setup_oracles/<name>.json`.
The `add-test-setup` skill is rewritten for this workflow as part of the
implementation.

**A one-off variant for a project** — no code at all: an `overrides`
block in the project's yaml, merged key by key as described under
`captest.py`.

## Validation

All errors carry a path in document coordinates
(`meas.reg_cols.power.args.cell_temp.calc`), produced by pydantic's `loc`
for tier 1 and by an explicit path argument for tier 2.

### Tier 1 — structural (no project, no data); model validators

- pydantic: unknown keys (`extra="forbid"`), wrong types, node with zero or
  several discriminating keys, non-finite float, `agg` outside the allowed
  set.
- `Calc`: `calc` in `CALC_REGISTRY` (did-you-mean hint); `args` keys ⊆ the
  function's parameters minus `data` and `verbose` (a `requires_params`
  name may appear explicitly); every parameter that has no default and is
  not in `requires_params` is present in `args`.
- `TestSetup`: `reg_fml` parses (`util.parse_regression_formula`); lhs ∪
  rhs ⊆ keys of `meas.reg_cols` and of `sim.reg_cols`; `rep_conditions.func`
  keys ⊆ rhs and values match `mean|median|perc_\d+`; every top-level
  `reg_cols` value is a `Group`, `Column` or `Calc` (not a literal); per
  side, the output-column rule — every column written by a `Group` or
  `Calc` node is written only by nodes equal to it; `params` keys are names
  in `setup.DOWNSTREAM_PARAMS`.
- `TestSetup.derive` / `resolve_test_setup`: a `reg_cols_*` override key
  that is neither a formula variable of the resulting `reg_fml` nor `null`
  is an error (it would aggregate a column nothing uses); `null` for a key
  the base lacks is an error naming the key.
- `captest.py`, on loading `TEST_SETUPS` and in `resolve_test_setup`:
  `scatter_plots` in `SCATTER_REGISTRY`.

Whole-document checks are raised from a `field_validator` (or with
`PydanticCustomError`) so they carry a location; a bare `model_validator`
loses it.

### Tier 2 — project fit (`column_groups` and the sim header, no data);
`setup.check_project_fit(setup, side, cd) -> list[FitError]`

Runs in `CapTest.setup()` after prep and before evaluation, against the
`CapData` as it is at that moment (so the names checked are post-prep):

- every `Group.group` is a key of `cd.column_groups`;
- every `Column.column` is in `cd.data.columns`;
- for every `Calc` reached, every name in `requires_params` resolves under
  the precedence rules to a value that is **not `None`**. `requires_params`
  is the declaration that the calculation cannot run without a real value,
  so a `None` reached by any route fails: an explicit `args: {power_temp_coeff:
  null}`, a `None` `CapData` attribute, or a function default of `None`
  (`power_temp_correct(power_temp_coeff=None)` is a placeholder, not a
  default). Parameters *not* in `requires_params` may legitimately be `None`
  (`absolute_airmass(pressure=None)`), which is why the two are distinct.
  Every package in `requires_import` is importable
  (`importlib.util.find_spec`);
- no column written by a `Group` or `Calc` node is a column group id or an
  existing raw column on that side;
- for every key in `setup.params` and every `Calc` on this side whose
  `requires_params` contains it, the effective value under the precedence
  rules equals `setup.params[key]`. A side where no calculation uses the
  key is not checked.

The checks are collected and raised together as one `SetupFitError`
listing every path, so an agent fixes a document in one pass. `CapTest`
exposes `check_fit()` for the same list without running setup.

### Tier 3 — runtime

What only running establishes: enough points survive, the regression fits,
a calculation raises on real data. Unchanged.

## Trust boundary

Loading a setup document runs no code the document names: `calc` and
`scatter_plots` are looked up in registries populated by Python code the
process already imported. Two things a full `CapTest` config still
executes are outside this design and unchanged: the patsy formula, which
statsmodels evaluates (patsy can call Python functions named in the
formula), and the filter pipeline codecs, which import
`module:qualname` callables for `Custom` / `RepCond` steps. A config file
from an untrusted author is therefore still untrusted; this design narrows
the surface to those two, and the survey note records them for a later
pass.

## Error handling

- Unknown `calc`: `ValueError` from tier 1 with path and hint.
- Tier-2 failures: `SetupFitError(ValueError)` with `.errors: list[FitError]`
  (`path`, `message`); the message string lists them all.
- `custom_param` with `output` colliding with a formula variable's raw
  column: tier 2 catches it; at runtime the write proceeds (documented).
- Loading a preset file that fails validation raises at `import captest`
  — deliberate; a broken shipped preset must not be silently absent.

## Testing

Tests follow the existing layout (`tests/test_setup.py` new;
`tests/test_calcparams.py`, `tests/test_captest.py`, `tests/test_util.py`,
`tests/test_plotting.py` updated). Oracles are captured **before** the
presets are migrated.

1. **Oracle capture (first task, on the old code):** for every
   `TEST_SETUPS` key, run the current preset on the fixture measured and
   PVsyst data through `CapTest.setup()` + `run_test()` and store the
   flattened `regression_cols` of each side, the regression parameters and
   p-values, and `rc` to `tests/data/setup_oracles/<name>.json`.
2. **Equivalence:** each migrated yaml preset reproduces its oracle:
   identical column names, parameters equal within `rtol=1e-9`.
3. **Model / identity:** every preset loads; `to_dict()` is idempotent
   under reload; `content_digest()` is stable across processes (compare to
   a stored digest per preset); `TestSetup == TestSetup` by value; `Group`
   hashable, `TestSetup` not.
4. **Rejection cases** (each asserts the error path):
   - node mapping with no discriminating key, with two;
   - a Python callable in an argument (`{"reducer": np.mean}`) — the
     original defect;
   - a two-element list literal argument stays a list and reaches the
     function unchanged (the other original defect);
   - unknown `calc`, unknown `args` key, missing required arg;
   - `reg_fml` variable absent from one side; `rep_conditions.func` key not
     in rhs;
   - output-column collisions: same `calc`, different `args`, on one side;
     a registered name equal to an aggregation column beside that `Group`;
     and the accepted case — two identical `Calc` nodes — evaluated to one
     column;
   - a literal as a top-level `reg_cols` value;
   - override merge: `reg_cols_meas: {poa: {group: irr_ghi}}` keeps the
     other three e2848 terms; `null` removes a term; removing `w_vel` from
     `e2848_default` (formula override plus `null` on both sides) yields a
     valid setup whose `rep_conditions.func` no longer has `w_vel`, while a
     `func` entry for a surviving variable is untouched; `null` for a term the
     preset lacks is rejected; an override key that is not a formula
     variable is rejected; `custom` with a partial side is rejected;
   - non-finite float; dict literal;
   - `params: {rear_shade: 0}` against a `CapData` with `rear_shade=0.2`
     (tier 2), and the accepted case — the `_rear_shade_sim` preset on a
     `sim` that has no `rear_shade` attribute, where `e_total` resolves it
     to its default `0`;
   - precedence: `args: {base_temp: 20}` reaches `power_temp_correct` while
     the `CapData` has `base_temp=25`; `args: {pressure: null}` reaches
     `absolute_airmass` as `None` even when a `pressure` column group
     exists; a `requires_params` name absent from `args` is injected;
   - normalisation with explicit `null` fields (`w_vel: null`) is
     idempotent and digests equal the field-omitted form;
   - `Group` naming a group the project lacks; `Column` absent from the
     sim header; `requires_params` resolving to `None` by each route —
     `CapData.power_temp_coeff` unset, explicit `args: {power_temp_coeff:
     null}`, and the function's own `None` default;
     `requires_import` package missing (monkeypatched `find_spec`).
5. **Registry declarations:** the source-inspection test described above,
   over every entry.
6. **Evaluation:** nested three-deep `Calc` with a repeated `Group` leaf
   aggregates once; re-running with a different setup overwrites a
   same-named output.
7. **CapTest round trip:** `to_yaml` → `from_yaml` for a preset with one
   overridden term (the file contains only that term), for a removed term
   (the file contains the formula override and `null`, and the reloaded
   setup's `rep_conditions.func` is pruned identically), for `custom`, and
   for a document using a
   custom registered calculation; the reloaded `resolved_setup` equals the
   original by value and by `content_digest()`; `check_fit()` lists errors
   without running setup.
8. **Plotting:** `_missing_column_groups` and `calc_tc_power_column` over
   node trees.

## Documentation and packaging

- `pyproject.toml`: add `pydantic>=2.5,<3` to `[project] dependencies`;
  add `[tool.setuptools.package-data] captest = ["setups/*.yaml"]`
  (the build is setuptools with `packages.find`, which does not pick up
  non-Python files on its own). `tests/smoke_test.py` gains an assertion
  that a built wheel contains the preset files and that `TEST_SETUPS` is
  non-empty, since a missing package-data entry would only show up in the
  published artifact.
- conda-forge feedstock: `pydantic >=2.5,<3` in `run:` at the next release
  (the `conda-release` skill gates on this).
- `CHANGELOG.md`: breaking change — tuple-form `regression_cols` removed;
  new document form; `encode_reg_cols` / `decode_reg_cols` /
  `validate_test_setup` removed; `custom_param(output=)`.
- User guide (`docs/user_guide/captest.rst`) and the `CapData.regression_cols`
  section: rewrite the grammar description around nodes; show authoring a
  setup file and `check_fit()`; API reference entries for `captest.setup`.
  Via the `docs-update` skill after implementation.

## Out of scope

- ctsweep's `test_setups` table, `derived_from_hash`, and storing the
  resolved document in the run envelope (pft-mono spec).
- Migrating pft-mono consumers (`perfactory/captest.py`, `ctsweep/adhoc.py`,
  `captest-gui`).
- Orthogonal aspects / generated preset library (survey option C), the
  expression DSL (A), a project-level `variables` map (E).
- Callable handling in the filter pipeline codecs and the patsy trust
  surface.
- Any change to the filter step classes; `param` remains their base.
