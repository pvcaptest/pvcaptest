# Test setup representation: design options

**Date:** 2026-09-22
**Purpose:** Survey alternative representations for a capacity-test setup (the
`TEST_SETUPS` entry: regression columns trees, formula, reporting conditions)
ahead of a redesign. For external review before choosing a direction.
Backward compatibility is deliberately out of scope for this note.

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

## Options

### 0. Current form plus key-level override merge

Keep the tuple tree. Change `resolve_test_setup` so a `reg_cols_*` override
merges per top-level key onto the preset (as `rep_conditions` already does via
`_merge_rep_conditions`), with `null` deleting a key. Keep the encoder and
decoder, but only decode data read from a file.

- For: smallest change; removes the original cause (preset callables leaking
  into an override that changed one key).
- Against: the literal-list ambiguity remains for file-loaded trees; two
  merge regimes (`rep_conditions` merges `func` one level deep, `reg_cols`
  merges at the top level) to document; a full dict that deliberately omits a
  preset key now gets that key merged back.

### 1. Typed nodes holding callables

Frozen dataclasses `Agg(group, func)` and `Calc(func: Callable, kwargs)`,
with tuple-to-node conversion at the API boundary and a tagged yaml form
(`{agg: ..., func: ...}`, `{calc: module:qualname, kwargs: ...}`).

- For: node kind is the type, not the shape, so both roborev findings are
  impossible; construction-time checks; hashable `Agg` replaces the
  `agg_cache` key hack; same `to_config` / `from_config` pattern as filter
  steps.
- Against: still holds function objects, so equality is by identity, `__main__`
  callables cannot round-trip, and a setup is still a Python object that
  happens to serialize rather than data.

### 2. Pure data with a calculation registry

The yaml *is* the setup; a pydantic (or dataclass) model is a typed view of
it. Functions are referenced by name through `CALC_REGISTRY` (the
`FILTER_REGISTRY` pattern). Node grammar: `{group, agg?}`, `{column}`,
`{calc, args}`, or a literal.

```yaml
name: bifi_power_tc_etotal
derived_from: e2848_default
reg_fml: power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1
reg_cols:
  meas:
    power:
      calc: power_temp_correct
      args:
        power: {group: real_pwr_mtr, agg: sum}
        cell_temp:
          calc: cell_temp
          args: {poa: {group: irr_poa}, bom: {group: temp_bom}}
    poa: {calc: e_total, args: {poa: {group: irr_poa}, rpoa: {group: irr_rpoa}}}
    t_amb: {group: temp_amb}
    w_vel: {group: wind}
  sim:
    power: {calc: power_temp_correct, args: {power: {column: E_Grid}, cell_temp: {column: TArray}}}
    t_amb: {column: T_Amb}
rep_conditions:
  func: {poa: perc_60, t_amb: mean, w_vel: mean}
scatter_plots: default
```

- For: every node says what it is; nothing can fail to serialize; equality
  and diffing are plain data comparison; presets become yaml files loaded
  through the same path users and agents use; `group` vs `column` makes the
  current implicit "group id if it resolves, else column name" rule explicit.
- For (validation): three layers, each a pure function — schema (pydantic,
  path-bearing errors, JSON Schema export for agents); internal consistency
  (formula variables are keys on both sides, `rep_conditions.func` keys are
  rhs variables, `calc` args match the registered function's signature);
  project fit (`required_groups(setup, side)` compared against
  `column_groups`, no data load needed).
- For (overrides): a setup is a complete value. `derive(base, ...)` returns a
  new complete setup; `derived_from` records provenance; diffs are computed,
  never stored. No merge semantics at all.
- Against: pydantic is a new dependency alongside `param` (dataclasses plus a
  hand-written validator work but lose the path-bearing errors and schema
  export); the registry is a boundary — custom calculations must be
  registered, with a `module:qualname` escape hatch marked non-portable;
  every preset is rewritten and `transform_calc_params` and its `plotting.py`
  twin are re-pointed.

### 3. Expression strings (a small DSL)

One line per regression variable, parsed with `ast` restricted to calls,
names, keywords, and constants:

```yaml
reg_cols:
  meas:
    power: power_temp_correct(sum(real_pwr_mtr), cell_temp=cell_temp(mean(irr_poa), mean(temp_bom)))
    poa:   e_total(mean(irr_poa), mean(irr_rpoa))
    t_amb: mean(temp_amb)
  sim:
    power: power_temp_correct(col("E_Grid"), cell_temp=col("TArray"))
```

- For: the most readable form for humans and agents; line diffs are
  meaningful; consistent with `reg_fml`, which is already a string DSL
  (patsy); the AST is the typed tree, so validation is the same as option 2.
- Against: owning a parser and its error messages; quoting inside yaml;
  literal list arguments need syntax. This is a surface syntax over a node
  model, not a replacement for one, and can be layered on later.

### 4. Flat named derivations instead of a nested tree

An ordered list of named columns, like the filter pipeline:

```yaml
derive:
  - {name: poa_mean,  group: irr_poa,      agg: mean}
  - {name: bom_mean,  group: temp_bom,     agg: mean}
  - {name: pwr_sum,   group: real_pwr_mtr, agg: sum}
  - {name: cell_temp, calc: cell_temp,           args: {poa: poa_mean, bom: bom_mean}}
  - {name: power_tc,  calc: power_temp_correct,  args: {power: pwr_sum, cell_temp: cell_temp}}
reg_cols: {power: power_tc, poa: poa_mean, t_amb: t_amb_mean, w_vel: ws_mean}
```

- For: matches what `CapData` actually does (adds columns to `data`); every
  intermediate is named, inspectable, and plottable; sharing is natural
  (`poa_mean` feeds both `power` and `poa` without `agg_cache`); `reg_cols`
  is a flat `{var: column}` map, the same shape before and after processing;
  validation is a DAG check (references resolve, no cycles, args match
  signatures), simpler than a recursive tree walker.
- Against: more verbose for simple setups; every intermediate needs a name;
  nesting is closer to how a single formula term is thought about.

### 5. Orthogonal aspects instead of a flat preset list

`TEST_SETUPS` is roughly the cross product of a few choices. Make the axes
explicit:

```yaml
power:      temp_corrected      # or: raw
irradiance: e_total_spectral    # or: poa, e_total, poa_spectral
rep_cond:   perc_60
formula:    e2848
```

Each aspect declares the variables it provides, the groups it requires, and
the parameters it needs (`bifaciality`, `spectral_module_type`, ...). The
resolver composes the derivation list.

- For: "run every valid test type on this project" is `product(axes)`
  filtered by `required_groups ⊆ project.groups`; new combinations are free;
  validation is per aspect; provenance is built in (the setup *is* its
  choices).
- Against: not all combinations are meaningful, so compatibility rules are
  needed; fully custom tests still need an escape hatch (a raw derivation
  list); preset descriptions would be generated; the largest conceptual
  change.

### 6. Setups as Python classes, config as parameters only

Each test type is a class with a `derive(cd)` method in plain pandas; the
yaml holds only the class name and numeric parameters.

- For: full expressiveness, IDE support, ordinary unit tests, no DSL or
  registry.
- Against: the wrong trade for the goal. Agent-authored setups must be
  importable Python; nothing is statically inspectable; "which setups can run
  here" needs a `requires` declaration that will drift. Ruled out.

### 7. Split project variables from the test transform

The current tree mixes *what the site has* (sensor aggregation:
`("irr_poa", "mean")`, repeated in every preset) with *what the test
computes* (`e_total`, `cell_temp`). Separate them:

```yaml
# project (lives with column_groups)
variables:
  poa:   {group: irr_poa,      agg: mean}
  rpoa:  {group: irr_rpoa,     agg: mean}
  power: {group: real_pwr_mtr, agg: sum}
  t_bom: {group: temp_bom,     agg: mean}

# test setup — references project variables only
derive:
  - {name: cell_temp, calc: cell_temp,          args: {poa: poa, bom: t_bom}}
  - {name: power_tc,  calc: power_temp_correct, args: {power: power, cell_temp: cell_temp}}
```

- For: setups become small and portable across projects; the meas/sim
  asymmetry mostly disappears (the sim side is just a different `variables`
  map, `power: {column: E_Grid}`); project fit is "does the project define
  every variable the setup names", checked once per project rather than per
  setup.
- Against: a second config location; some aggregation choices (`sum` vs
  `mean` of inverter power) are test decisions, not site facts, and a rule is
  needed for where they go.

## Current leaning

Options 4 and 7 together (a flat DAG of named derivations over
project-declared variables, encoded as pure data per option 2) are the
structural change with the best payoff: easier to validate than any tree,
matches what `CapData` does, and makes setups short enough that option 5
becomes practical on top — aspects are reusable fragments of the derivation
list, and the preset library becomes a generated product rather than ten
hand-maintained dicts. Option 3 can be layered on afterwards as terser syntax
for derivation entries. Option 6 is dropped.

Proposed next step before a spec: rewrite the two largest presets
(`bifi_power_tc_etotal` and the spectral-corrected E_total variant) in the
4+7 form and check whether they read well and whether any aggregation choice
resists being pushed to the project side.

## Questions for review

- Is the 4+7 leaning right, or does nesting (option 2 alone) keep enough
  locality to be worth its validation cost?
- Where should the aggregation choice live when it is a test decision (e.g.
  `sum` of inverter power for a per-inverter test)? Project default with a
  setup-level override, or always in the setup?
- Is the calculation registry the right boundary for custom calculations, or
  should `module:qualname` be first-class?
- pydantic versus dataclasses given `param` is already a dependency.
- Does option 5 pull its weight, or should the sweep tooling in pft-mono
  own the "generate variants" step and captest only validate and run complete
  setups?
