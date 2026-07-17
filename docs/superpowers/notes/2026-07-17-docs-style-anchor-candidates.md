# Docs-update skill: prose-style anchor candidates

**Date:** 2026-07-17
**Purpose:** Choose an existing user-guide passage (or blend) to serve as the
prose-style *anchor* in `.agents/skills/docs-update/SKILL.md`, so future doc
writing matches the intended voice. For external review before deciding.

## Background / problem being solved

Past `docs-update` runs — even on Opus — did not produce prose at the intended
level: **clear and explicit, aimed at a capacity-test practitioner coming from
an Excel background, not a Python background.**

Diagnosis (from reading the skill + recent output): the miss is **primarily the
skill, secondarily the run prompt, least of all the model.** Specifically:

1. The skill tells every run to *"match the prose tone already in
   `dataload.rst`"* (SKILL.md lines 75, 155). But `dataload.rst` is written as
   competent **developer-to-developer** prose — dense cross-references, assumes
   the reader knows DataFrames, regressions, kwargs. Matching that anchor
   faithfully lands at the wrong level by construction.
2. The intended audience (Excel background, define terms, use analogies) is
   stated **nowhere** in the skill.
3. Almost all concrete style guidance in the skill is RST *mechanics*
   (cross-reference syntax, `~` prefixes, directive indentation) — all of which
   pulls toward terse, developer-register prose.
4. "Match existing style / prefer additive / don't rewrite working
   documentation" (SKILL.md 155–158) actively tells the model to *conform to*
   the surrounding dense prose rather than write at a different level.

**Decisions already made (2026-07-17):**
- New/edited sections should be written at the Excel-user level; **do not**
  re-level existing dense prose now (additive only; tonal mix accepted for now).
- The anchor should be an **existing passage** from the docs (chosen below),
  not one I draft.

**Still to decide:** which passage (or blend) becomes the anchor.

---

## Candidate 1 — `reporting_conditions.rst` opening (why-first teaching voice)

Source: `docs/user_guide/reporting_conditions.rst:1-33`

> An ASTM E2848 capacity test compares the measured capacity to the modeled
> capacity at a single, agreed-upon set of **reporting conditions** (RCs) — the
> representative irradiance, temperature, and wind speed the plant is rated at.
> Because both regressions are evaluated at the *same* point, pvcaptest models
> the reporting conditions as one value owned by the test:
> `CapTest.rc`.
>
> This page explains how the single test RC is established, overridden, tracked,
> used in filtering and results, and persisted across a yaml round-trip. It
> assumes a `CapTest` instance `ct` has been created and set up as described in
> the CapTest page.
>
> **The single test reporting conditions**
>
> `CapTest.rc` is a one-row `pandas.DataFrame` (or `None` before any have been
> established) holding one value per regression variable:
>
> ```python
> >>> ct.rc
>      poa  t_amb  w_vel
> 0  805.1   24.7    2.1
> ```

**Strengths:** Leads with *why the feature exists* before *how* to use it, in
capacity-test terms. Bolds new terms on first use. Shows expected output. The
most "teaching" voice in the guide.

**Weakness:** Still assumes pandas basics (`pandas.DataFrame` cited without
explanation). No explicit Excel bridge.

---

## Candidate 2 — `custom_test_setups.rst` grammar section (scaffolded build-up)

Source: `docs/user_guide/custom_test_setups.rst:16-55`

> Each key in `reg_cols_meas` or `reg_cols_sim` maps a regression term (such as
> `"power"` or `"poa"`) to one of three node forms.
>
> The simplest node is a plain string matching a column name in the `data`
> attribute. For example, this is the approach used in the built-in test setups
> for the PVsyst output:
>
> ```python
> "poa": "GlobInc"
> ```
>
> A **simple aggregation** is a two-element tuple of the column-group id and an
> aggregation function name. This will be aggregated using the
> `CapData.agg_group` method. [...]
>
> ```python
> "poa": ("irr_poa", "mean")
> ```
>
> A **calculated column** is a two-element tuple of a callable and a dict
> mapping the callable's keyword arguments to column names, aggregation tuples,
> or nested calculated-column tuples:
>
> ```python
> "poa": (
>     e_total,
>     {"poa": ("irr_poa", "mean"), ...},
> )
> ```

**Strengths:** Builds complexity gradually — names each concept in bold, gives
the simplest form first, then adds one layer at a time with a complete example
per step. Strong scaffolding pattern for a reader who needs to be walked up.

**Weakness:** The domain here is inherently developer-facing (dict grammar,
callables, tuples), so it leans on programming vocabulary more than a
data-loading or filtering page would need to.

---

## Candidate 3 — `captest.rst` "Creating a CapTest" (task-based structure)

Source: `docs/user_guide/captest.rst` ("Creating a CapTest" section — the
passage originally proposed as the anchor).

> A `CapTest` can be created from file paths, data that has already been loaded,
> or from a yaml file. Using `from_params` will create a CapTest object given
> file paths and is the option recommended for typical usage of pvcaptest to
> interactively run a test in a Jupyter notebook.
>
> **From data paths**
> If you provide paths to your data, `CapTest` will load the data for you.
>
> ```python
> ct = CapTest.from_params(
>     test_setup='bifi_e2848_etotal_rear_shade_sim',
>     meas_path='./data/measured/',
>     sim_path='./data/pvsyst_results.csv',
>     bifaciality=0.15,
>     ac_nameplate=6_000_000,
>     test_tolerance='- 4',
>     meas_load_kwargs={'group_columns': './path-to/column_groups.xlsx'},
> )
> ```
>
> **From loaded data** / **From yaml** — parallel subsections, each with a
> complete copy-pasteable example and `.. note::` boxes for gotchas (e.g. the
> required `setup()` call, needing `meas_load_kwargs`).

**Strengths:** Best *structural* template — task-oriented section headers a
non-programmer can navigate ("From data paths / From loaded data / From yaml"),
complete copy-pasteable examples (not fragments), note boxes catching gotchas,
plain-language framing of *when* to use each path.

**Weakness:** Assumes Python fluency — never defines "yaml," "kwargs," or
"object"; leans on bare cross-references as the explanation of a step. This is
competent developer documentation, **not** the Excel-user level described in the
problem statement.

---

## Candidate 4 — Blend: Candidate 1 (voice) + Candidate 2 (structure)

Have the skill cite **two** exemplars, each for a different job:

1. **Why-first opening** (from `reporting_conditions.rst`): open each feature by
   stating the problem it solves *in capacity-test terms*, bold new terms on
   first use, show expected output.
2. **Scaffolded build-up** (from `custom_test_setups.rst`): introduce the
   simplest usage first, then add one layer of complexity at a time, with a
   complete example at each step.

New prose must satisfy **both** patterns. (Candidate 3's task-based section
structure could optionally be cited as a third, structural, exemplar.)

**Strengths:** Separates "tone model" from "structure model" instead of asking
one passage to carry both.

**Weakness:** More prescriptive; three patterns to satisfy could feel heavy for
short changelog-only updates.

---

## Reviewer's note (honest caveat)

**None** of the existing passages fully hits the "explain it to an Excel user"
bar as originally described — none defines yaml/kwargs/objects on first use or
uses spreadsheet-world analogies. Candidates 1 and 2 are the closest in
*spirit* (why-first, scaffolded, terms-in-bold). If the true target is stricter
than anything currently in the docs, the better path may be to write a *new*
exemplar at that level (with the reviewer or the user drafting it) rather than
anchoring on any existing passage. That option was deferred pending this review.

## Next step after a passage is chosen

Update `.agents/skills/docs-update/SKILL.md`:
1. Add an "Audience & voice" section (target reader = capacity-test practitioner
   from an Excel background; define each Python/stats/yaml term on first use;
   prefer worked examples over bare cross-references; short, plain sentences).
2. Insert the chosen passage as the concrete style exemplar; soften the "match
   `dataload.rst` tone" / "match existing style" lines so the anchor for *new*
   prose becomes the chosen exemplar — while keeping "prefer additive, don't
   rewrite working docs" so existing sections are left alone.
3. Keep all RST-mechanics guidance intact (that part is correct).
