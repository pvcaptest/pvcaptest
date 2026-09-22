# Test setup representation: design options

**Date:** 2026-09-22 (revised after two roborev design reviews)
**Purpose:** Survey alternative representations for a capacity-test setup (the
`TEST_SETUPS` entry: regression columns trees, formula, reporting conditions)
ahead of a redesign, and record the chosen direction. Backward compatibility
is deliberately out of scope for this note.

## Background / problem being solved

Commit `7d0c2f4` fixed two defects in the yaml round trip of a
`reg_cols_meas` / `reg_cols_sim` override:

1. An override replaces the preset dict wholesale (`resolve_test_setup`), so
   changing one key carries the preset's calculation callables into
   `to_mapping()`, and `to_yaml` failed with a yaml `RepresenterError`.
2. Tuples read back from yaml are lists, `process_reg_cols` dispatches on
   tuples, and `run_test` failed with `TypeError: unhashable type: 'list'`.

The fix added `util.encode_reg_cols` / `util.decode_reg_cols` (callables to
`module:qualname` strings, lists to tuples). A roborev review of the commit
found two follow-on defects in that fix, both reproduced:

- `decode_reg_cols` turns *every* two-element list into a tuple, so a literal
  list argument to a custom calculation (`{"columns": ["poa1", "poa2"]}`)
  becomes `("poa1", "poa2")` and is then mistaken for an aggregation pair.
  Because `from_mapping` always decodes, this also affects trees passed in
  native Python form.
- `encode_reg_cols` encodes every callable, but `decode_reg_cols` restores one
  only in the head position of a calculation pair, so a callable passed as a
  kwarg (`{"reducer": np.mean}`) comes back as the string `"numpy:mean"`.

The root cause is the representation itself, not the fix. The current tree
grammar

```python
{
    "power": (power_temp_correct, {
        "power": ("real_pwr_mtr", "sum"),
        "cell_temp": (cell_temp, {"poa": ("irr_poa", "mean"), "bom": ("temp_bom", "mean")}),
    }),
    "poa": (e_total, {"poa": ("irr_poa", "mean"), "rpoa": ("irr_rpoa", "mean")}),
}
```

identifies node kinds by *shape* (tuple length and element types) and holds
live function objects. Shape-based dispatch is ambiguous once the tree passes
through yaml or json, and function objects cannot be serialized, compared, or
inspected without importing them.

The defect was found storing a hand-run test into the pft-mono `ctsweep`
store, which holds each run's `CapTest.to_mapping()` as a canonical-JSON
document in SQLite and content-hashes it for identity. That store already
demands what this redesign provides: a setup made only of JSON-representable
data. See "Storage model" below.

## Goals for a redesign

Ranked by the intended use — an agent defining and running many test types
on one project:

1. Resistant to bugs by construction: node kinds cannot be confused, nothing
   in a setup can fail to serialize.
2. Conceptually clear and easy to maintain.
3. Readable and writable by both humans and agents, with new setups easy to
   author.
4. Validity checks that can run without loading data: is the setup
   well-formed, internally consistent, and runnable on a given project?

## Chosen direction: pure data with a calculation registry

The yaml or json document *is* the setup; a typed Python model is a validated
view of it. Every function is referenced by name through a registry (the
`FILTER_REGISTRY` / `step_from_config` pattern in `filters.py`), never held
as a Python object. The tree then contains only strings, numbers, and tagged
nodes: nothing in it can fail to serialize, and nothing has to be imported to
read it.

### Node grammar

Every node is a mapping with exactly one discriminating key, or a literal.
There is no shorthand: a bare string is always a literal.

| Node | Meaning |
|---|---|
| `{group: irr_poa, agg: mean}` | aggregate a column group; `agg` defaults to `mean` |
| `{column: E_Grid}` | a raw column by name (the sim side, or a single-sensor meas column) |
| `{calc: cell_temp, args: {...}}` | a registered calculation; `args` are keyword-only, each value a node |
| `"cdte"`, `3.5`, `true`, `[1, 2]` | a literal argument, passed through unchanged |

Callable-valued arguments are not supported in the document form. `calc`
always means *execute and supply the output column*; it is never a way to
pass a function object. A calculation that needs to choose a function
takes a plain string it resolves itself (`reducer: "mean"`), the way
`agg` already does. This is the rule that makes the motivating defects
impossible: reference vs literal is decided by the tag, never by shape or
position.

`args` are keyword-only and bind by name to the registered function's
parameters after `data` is injected. Positional binding is not offered: the
first review of a DSL sketch showed `cell_temp(mean(irr_poa), mean(temp_bom))`
silently swapping irradiance for module temperature because the real
signature is `cell_temp(data, bom, poa, ...)`.

### Document shape

The tree keeps its current nested, recursive form. Each formula variable
maps to one node; a `calc` node's `args` are themselves nodes, evaluated
bottom-up exactly as `transform_calc_params` does today, and a calculation's
output column — named for the calculation — is what the parent consumes.
Only the *encoding* of a node changes (tagged mapping instead of a
shape-detected tuple), not the architecture.

Each side has its own tree, because the sides need *different derivations*,
not merely different source columns: temperature-corrected presets compute
measured cell temperature but read PVsyst's `TArray` directly; spectral
presets compute measured precipitable water (`precipitable_water_gueymard`)
but scale PVsyst's `PrecWat` by 100.

```yaml
name: bifi_power_tc_etotal_rear_shade_sim
derived_from: e2848_default            # provenance only; never used to fill fields
description: Bifacial, temperature-corrected power, E_total, rear shade in the model.
reg_fml: power ~ poa                    # the preset's own formula, intercept included
reg_cols:
  meas:
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
    t_amb: {group: temp_amb}
    w_vel: {group: wind}
  sim:
    power:
      calc: power_temp_correct
      args: {power: {column: E_Grid}, cell_temp: {column: TArray}}
    poa:
      calc: e_total
      args:
        poa:  {column: GlobInc}
        rpoa: {calc: rpoa_pvsyst, args: {globbak: {column: GlobBak}, backshd: {column: BackShd}}}
params:
  rear_shade: 0            # required value; see "Parameter constraints" below
rep_conditions:
  irr_bal: false
  percent_filter: 20
  func: {poa: perc_60}
scatter_plots: default
```

The `t_amb` / `w_vel` entries of the E2848 presets are absent here because
this preset's formula does not use them; the example matches
`TEST_SETUPS["bifi_power_tc_etotal_rear_shade_sim"]` key for key so it can
serve as the equivalence oracle in the spike.

Repeated sub-expressions (`{group: irr_poa, agg: mean}` appears under both
`power` and `poa`) are evaluated once: `Group` nodes are frozen and
hashable, so the existing `agg_cache` keys on the node itself rather than on
a `(group_id, agg_func)` tuple. `reg_cols` after processing is the flat
`{formula variable: column}` map it is today.

A setup is a complete value. `derived_from` is provenance; loading never
fills omitted fields from it. To make a variant, copy and edit (or
`TestSetup.derive(base, ...)` in Python, which returns a new complete
setup). A diff against the base is computed, never stored. There is no
override merge, so the `null`-to-delete and partial-vs-whole questions from
the current `resolve_test_setup` do not arise.

### Output naming

`CapData.custom_param` names its output column after `func.__name__`; with
the registry the output column is the registry *name* (`calc: cell_temp`
writes `cell_temp`), which keeps today's convention and keeps the column
readable in `data`. Two `calc` nodes with the same name but different
`args` on one side would collide, so that is a structural check (tier 1
below). An optional `as: <column>` key on a `calc` node is the escape
hatch when the same calculation is genuinely needed twice with different
inputs. A calculation's output name may not shadow a column group id or a
raw column on that side (tier 2).

### Validation, in three tiers

1. **Structural** — no data, no project. Schema validation of the document:
   unknown keys, wrong types, `agg` not in the allowed set, `calc` not in
   the registry, the same `calc` name used twice on one side with different
   `args` and no `as`. Then: formula variables ⊆ keys of both sides of
   `reg_cols`; `rep_conditions.func` keys ⊆ rhs variables; each `calc`'s
   `args` match the registered function's signature (required kwargs
   present, no unknown ones). Errors carry a path
   (`reg_cols.meas.power.args.cell_temp.calc`) so an authoring agent can
   self-correct.
2. **Project fit** — needs `column_groups` and the sim header, not data. Every
   `group` exists in the project's groups; every `column` exists on that
   side; every test-level parameter the setup's calculations need
   (`power_temp_coeff`, `bifaciality`, `spectral_module_type`, site
   metadata for solar position) is present and not `None`; optional
   dependencies the calculations import (`pvlib`) are installed. The
   registry entry for each calculation declares these requirements so the
   check is a lookup, not an execution. This tier is per setup, not once per
   project.
3. **Runtime** — what only running can establish: enough points survive,
   the regression fits, a calculation raises on the actual data.

The registry entry is therefore more than a name-to-function map:

```python
register_calc(
    "cell_temp", cell_temp,
    requires_params=("module_type", "racking"),
)
register_calc(
    "spectral_factor_firstsolar", spectral_factor_firstsolar,
    requires_params=("spectral_module_type",),
    requires_import=("pvlib",),
)
```

Signature matching alone is not enough: `power_temp_correct` accepts
`power_temp_coeff=None` and then fails, so the requirement has to be
declared, not inferred. The declarations must come from the implementation,
not from memory: `cell_temp` uses only numpy and pandas (`calcparams.py`
imports pvlib only for the solar-position and spectral functions), and an
over-declared `requires_import` would wrongly reject temperature-corrected
setups on installations without pvlib. Each registry entry's declaration is
therefore tested against its function: an import guard is required exactly
when the function fails without the package.

### Parameter constraints

`params` in a document is a mapping of test-level parameter name to the
value this setup *requires*. It is neither a default nor an override: at
tier 2 the effective value — after `CapTest` defaults, the project's
config, and any per-side injection — must equal the declared value, or
validation fails with the path (`params.rear_shade: setup requires 0, test
has 0.2`). This is how the rear-shade `_sim` variant refuses to double-count
a non-zero measured `rear_shade`. Absent keys are unconstrained.

### Custom calculations

A calculation not in the registry must be registered before a setup can name
it. There is no `module:qualname` form in the document: loading a setup
never imports code named by the document. A project that needs a one-off
calculation registers it in its own code (`register_calc("my_calc", fn)`)
before loading; the stored document then records the *name*, and
reproducing a run requires the same registration — which is the honest
statement of the dependency.

### Storage model (ctsweep)

The pft-mono `ctsweep` store is stdlib SQLite. `runs.config_json` and
`test_configs.config_json` hold the unwrapped `CapTest.to_mapping()` as
canonical JSON (`identity.canonical_json`: sorted keys, no whitespace, only
str/int/float/bool/list/dict, NaN rejected), and `run_id` / `config_hash`
are sha256 of that text. `json_extract` builds generated columns from it.

A setup document as specified here is exactly what that store accepts: the
model's `dump()` yields JSON types only, `canonical_json` of it is the
setup's identity, and yaml versus json is only a choice of encoding for the
same validated document (always load through the model; never rely on a yaml
parser's own type guessing for dates or `yes`/`no`).

Identity is assigned to the *normalised* document — the model's `to_dict()`
of the loaded document, with defaults materialised (`{group: temp_amb}`
becomes `{group: temp_amb, agg: mean}`) — never to the bytes the author
wrote. Normalisation must be idempotent (`to_dict(load(to_dict(load(d))))
== to_dict(load(d))`), and that, not byte equality with the input, is the
round-trip criterion.

Proposed store change: a setup is today nested inside a test config (setup +
paths + filters + window). Give setups their own table keyed by content
digest — `test_setups(setup_hash, name, derived_from_hash, setup_json)` —
with `runs` referencing `setup_hash`. Setups then dedupe automatically and
"every run of this recipe across projects" is an index lookup. `name` is a
display label; ancestry is `derived_from_hash`, a nullable reference to
another row, so revisions that share a name stay distinguishable. In a
document, `derived_from` may be a name (for a preset) or a digest; the
store resolves a name to the digest of the preset shipped with the captest
version recorded on the run, and records `NULL` when the ancestor is not
present.

Insert-time validation runs the model and registry checks, not just the
exported JSON Schema: the schema is a *shape* contract (`calc` is an
unrestricted string in it; the registry, formula ⊆ keys and collision
checks are validators) and is what an authoring agent is handed, while the
store's guarantee is "loads through the model". The schema is versioned
alongside `id_algo`.

An exported run must stay reproducible after the project or registry
changes. The run envelope should therefore carry the setup document itself
(not a name), the captest version, and the registry version, and the project
side of tier-2 (the group and column bindings actually used) should be
recorded in `metrics_json` at run time.

### Library choice: pydantic v2

Pydantic v2 (`pydantic>=2.5,<3`, a core dependency, not an extra) was
compared against stdlib dataclasses, `TypedDict` + `Annotated`,
attrs/cattrs, msgspec, and the already-present `param`. Every API claim
below was run in isolated `uv` environments (pydantic 2.13.5, msgspec
0.21.1, attrs/cattrs 26.1.0, param 2.4.1 from the lockfile).

| Requirement | pydantic v2 | msgspec | attrs + cattrs | dataclasses | TypedDict | param |
|---|---|---|---|---|---|---|
| Recursive discriminated union, path-bearing errors | ✓ callable `Discriminator` dispatches on which key is present, so the grammar above needs no `type` field; error is `reg_cols.meas.power.args.cell_temp.calc: unknown calculation 'foo'` | partial: tagged unions need a `type:` key in every node; error paths elide dict keys (`$.reg_cols[...][...].args[...]`) | partial: hand-written structure hooks; three attempts did not get the recursive scalar-mixing union working | ✗ no validation | ✗ static only; runtime validation is pydantic `TypeAdapter` anyway | ✗ per-field only, no unions, no paths |
| JSON Schema export | ✓ `model_json_schema()` with `$defs` per node, `oneOf`, `additionalProperties: false` | ✓ `msgspec.json.schema()` (tag field appears in schema) | ✗ | ✗ | ✗ | partial: flat per-class list |
| JSON-type round trip, `==`, hashability | ✓ `model_dump(mode="json")` yields only JSON types (tuples → lists, `np.float64` → `float`); frozen models compare by value; **not hashable with `dict` fields** — define `__hash__` over the canonical dump | ✓ `to_builtins()`; same hash caveat | ✓ `unstructure()`; same caveat | ✓ `asdict` | ✓ it is a dict | ✗ identity equality; dump includes auto `name` |
| Custom validators | ✓ `model_validator` / `field_validator` with path | partial: `__post_init__`, path elided | ✓ attrs validators | manual | manual | per-parameter only |
| Dependency cost | compiled `pydantic-core` + 3 small pure deps (~2 MB); conda-forge current (2.13.5, 2026-08-28); v1/v2 split settled, pin `<3`; py ≥ 3.9 (captest is ≥ 3.10) | compiled C; conda-forge current; 0.x, no stability promise | attrs already in lock (dev only); cattrs pure Python | zero | zero | zero |
| Maintainability, NumPy docstrings | ✓ one class per node; docstring on the class; `Field(description=)` feeds the schema | ✓ but `tag=True` boilerplate and 0.x churn | partial: converter wiring lives apart from the classes | ✗ walker grows with every node kind | ✗ | ✗ wrong tool: identity semantics, mutable |

Pydantic is the only option that meets the first requirement as stated: the
grammar above, with no tag field, parses directly through a callable
`Discriminator`, and the error names the full dict-key path an agent needs
to self-correct. msgspec, the runner-up, forces a `type:` key into every
node and hides dict keys in error paths.

Sketch of the node union and the setup model:

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class Group(_Frozen):  group: str; agg: str = "mean"
class Column(_Frozen): column: str
class Calc(_Frozen):
    calc: str
    args: dict[str, "Node"] = Field(default_factory=dict)
    @model_validator(mode="after")
    def _known(self):
        if self.calc not in CALC_REGISTRY:
            raise ValueError(f"unknown calculation {self.calc!r}")
        return self

Scalar = str | int | float | bool | list[str | int | float | bool]

def _node_kind(v):            # which key is present decides the member
    if isinstance(v, dict):
        return next((k for k in ("group", "column", "calc") if k in v), None)
    return "literal" if not isinstance(v, BaseModel) else type(v).__name__.lower()

Node = Annotated[
    Union[Annotated[Group, Tag("group")], Annotated[Column, Tag("column")],
          Annotated[Calc, Tag("calc")], Annotated[Scalar, Tag("literal")]],
    Discriminator(_node_kind),
]
Calc.model_rebuild()

class TestSetup(_Frozen):
    name: str
    reg_fml: str
    reg_cols: dict[Literal["meas", "sim"], dict[str, Node]]
    def to_dict(self):  return self.model_dump(mode="json", exclude_none=True)
    def content_digest(self): return sha256(canonical_json(self.to_dict()))
    # deliberately NOT __hash__: nested dicts/lists stay mutable under
    # frozen=True and 1 == 1.0 would hash differently. Only Group is hashable.

TestSetup.model_json_schema()               # agent contract / DB insert check
TestSetup.model_validate(d).to_dict() == d  # verified True
```

`CapTest` stays a `param.Parameterized` and holds the setup in a
`param.ClassSelector(class_=TestSetup)`. The two systems meet at one
boundary: pydantic for pure-data documents, `param` for live objects with
runtime state. Filter steps are not migrated.

Risks and mitigations:

- **Hashability**: frozen models with `dict` fields raise on `hash()`
  (verified), and defining `__hash__` over the dump would break the hash
  contract (nested containers stay mutable; `1 == 1.0` but they dump
  differently). Leave `TestSetup` unhashable; expose `content_digest()`
  (sha256 of the canonical dump) as the identity ctsweep uses. Only the
  leaf `Group` node is hashable, for the aggregation cache.
- **Hash stability across dump options**: `exclude_none` / `exclude_defaults`
  change the bytes. Fix one `to_dict()` policy (`exclude_none=True`, keep
  defaults) and test that a reloaded document dumps byte-identically.
- **Model-level validators lose the path** (a whole-document check such as
  formula variables ⊆ keys printed no location). Raise from a
  `field_validator("reg_cols")` reading `info.data["reg_fml"]`, or use
  `PydanticCustomError`, so every error carries a location.
- **Compiled `pydantic-core`**: wheels cover every captest platform and the
  conda-forge feedstock tracks releases within days; the `conda-release`
  skill gates on the dependency change.
- **Union with scalar literals**: without the callable discriminator pydantic
  reports eleven errors for one bad node (verified). Keep `_node_kind` as the
  single dispatch point and unit-test the "no recognised key" case.

## Options considered and not chosen

### 0. Current form plus key-level override merge

Keep the tuple tree; make `reg_cols_*` overrides merge per top-level key
(as `_merge_rep_conditions` does), with `null` deleting a key; decode only
file-loaded trees. Smallest change and removes the original trigger, but the
literal-vs-pair ambiguity remains, there would be two merge regimes to
document, and a full dict that deliberately omits a preset key gets it
merged back.

### 1. Typed nodes holding callables

Frozen dataclasses `Agg` / `Calc(func: Callable, kwargs)` with tuple
conversion at the boundary. Node kind becomes the type, which fixes the
shape ambiguity at the outer level, but callable-valued *arguments* are still
untyped, function identity still decides equality, `__main__` callables
still cannot round-trip, and a setup is still a Python object that happens
to serialize.

### A. Expression strings (a small DSL)

`power: power_temp_correct(power=sum(real_pwr_mtr), cell_temp=cell_temp(bom=mean(temp_bom), poa=mean(irr_poa)))`,
parsed with `ast` restricted to calls, names, keywords and constants. The
most readable form, consistent with `reg_fml` being a patsy string, and the
AST is the same node model. Costs: owning a parser and its error messages,
yaml quoting, literal-list syntax, and positional arguments must be
forbidden (see the `cell_temp` swap above). Deferred: it is a surface syntax
over the chosen model and can be added later without changing the model.

### B. Flat named derivations instead of a nested tree

An ordered `derive` list of named columns per side, each `{name, group|calc,
args}` with `{ref: name}` nodes pointing at earlier entries, and a flat
`vars` map from formula variable to column. Every intermediate is named and
shared sub-expressions are written once, and validation is a DAG check. Not
chosen: the nested tree with recursive bottom-up evaluation, where a
calculation's output name is consumed by the parent, already works and is
the architecture `transform_calc_params`, `plotting.py` and the presets are
built on. Its two costs — repeated sub-expressions and unnamed intermediates
— are handled by hashable `Group` nodes as the `agg_cache` key and by
naming outputs for the calculation (with `as` for collisions). The flat
form would have added a `ref` node kind, a name-resolution pass, and a
second grammar for the plotting consumer, for no gain in the properties
that matter (tagged nodes, no callables in the document).

### C. Orthogonal aspects instead of a flat preset list

`TEST_SETUPS` is roughly a cross product of power (raw / temp-corrected),
irradiance (POA / E_total / spectral / both), rep-condition rule, and
formula. Making the axes explicit makes "run every valid test type" an
enumeration. Not chosen now: the eligibility rule is more than "required
groups present" — the rear-shade `_sim` and `_meas` variants need identical
groups but combining sim-side rear shading with a non-zero measured
`rear_shade` double-counts the loss — so aspects need parameter constraints,
and the composition rules are a project of their own. Revisit once complete
setups exist as documents; aspects are then reusable sub-trees, and the
preset library can be generated.

### D. Setups as Python classes, config as parameters only

Full expressiveness, IDE support, ordinary unit tests. The wrong trade for
the goal: agent-authored setups would have to be importable Python, nothing
is statically inspectable, and "which setups can run here" needs a
`requires` declaration that will drift. Dropped.

### E. Split project variables from the test transform

Project config declares `variables: {poa: {group: irr_poa, agg: mean}, ...}`
and setups reference only those. Attractive for portability, but review
showed two problems: the sim side needs its own derivations (above), and
some aggregation choices (`sum` versus `mean` of inverter power) are test
decisions, not site facts. The chosen document keeps the aggregation in the
setup. A project-level `variables` map can be added later as a *source* for
`{group, agg}` nodes without changing the setup grammar.

## Open questions before a spec

- Whether `as` on a `calc` node is needed in the first cut, or whether a
  same-name-different-args collision can simply be a structural error until
  a real preset needs it.
- Where the prep stage (column renames and unit conversions before setup)
  sits relative to tier-2 validation: the bindings checked must be the
  post-prep names.
- Migration of `plotting.py`'s calc specs and the pft-mono consumers
  (`perfactory/captest.py`, `ctsweep/adhoc.py`, `captest-gui` views) —
  out of scope for this note but must be sized.
- Whether `scatter_plots` stays a named entry in a small registry (as
  sketched) or becomes a document of its own.

## Review history

- roborev job 420 (design, astra, on the first version of this note):
  argument ambiguity unresolved; incomplete example contradicting the
  no-inheritance rule; project/test split ignoring per-side derivations;
  static validation overstated; spike would not exercise the motivating
  defects; missing: output naming, import trust, reproducibility.
- roborev job 425 (design, astra, on a brief covering A–E): DSL example
  bound `cell_temp` args positionally and swapped them; E's example
  referenced a local derivation while claiming "project variables only";
  C's eligibility rule missed the rear-shade double-count constraint;
  spike named a non-existent preset (`bifi_power_tc_etotal`) and had no
  numerical acceptance criteria; missing: migration, prep-stage
  interaction, callable trust, serialization scope beyond derivations.
- roborev job 427 (design, astra, on the chosen-direction revision): the
  example used the E2848 formula where the named preset uses `power ~ poa`;
  `params` constraints had no enforcement rule; `calc` doubled as a
  function reference; `__hash__` over the dump broke the hash contract;
  JSON Schema overstated as the insert-time contract; `derived_from` a
  name where the store needs a digest; round-trip criterion contradicted
  default materialisation; spike had no rejection cases or oracles;
  `cell_temp` wrongly shown requiring pvlib; literal domain (non-finite
  floats, `null`) undefined; generated-column ownership across reruns and
  the patsy/filter-codec trust boundary unaddressed.

All findings are folded into the chosen direction above except the last
two, which the spec must define.

## Next step: the spec

Three review rounds have moved this from "direction unclear" to "direction
agreed, rules missing". The remaining items are spec content, not survey
content: the literal domain, generated-column ownership, the trust boundary
for patsy formulas and filter codecs, and a spike with rejection cases
(callable object, mixed tags, output collision, violated `params`
constraint) and oracles for every document, comparing the three real
presets to their existing regression outputs at a stated tolerance.
