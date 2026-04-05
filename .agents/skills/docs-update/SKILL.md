---
name: docs-update
description: Update pvcaptest user-facing documentation when code changes. Use this skill whenever asked to update docs, review docs, handle documentation changes, or analyze recent commits for documentation needs. Also trigger this skill when the user mentions the user guide, changelog, API docs, docstrings, ReadTheDocs, Sphinx, RST files, or needs to document a new feature, parameter, method, or behavior change in captest/pvcaptest.
---

# pvcaptest Documentation Bot

Automatically review code changes and update user-facing documentation to keep it synchronized with the codebase. The docs are built with Sphinx and published on ReadTheDocs using the `sphinx_rtd_theme`.

## Project Documentation Layout

```
docs/
├── conf.py                  # Sphinx config (extensions: nbsphinx, autodoc, napoleon, recommonmark)
├── index.rst                # Top-level table of contents
├── installation.rst         # Installation instructions
├── changelog.md             # Keep a Changelog format (Markdown)
├── release.rst              # Dev processes, git workflow, versioning
├── examples.rst             # Index of example notebooks
├── examples/                # Jupyter notebook examples (rendered via nbsphinx)
├── user_guide/
│   ├── index.rst            # User guide TOC
│   ├── dataload.rst         # Loading data, column groups, filtering, regression, results
│   └── bifacial.rst         # Bifacial-specific testing
└── source/
    ├── modules.rst           # autodoc module index
    └── captest.rst           # autodoc API reference pages
```

**Key modules and where they're documented:**
- `CapData` (capdata.py) — `user_guide/dataload.rst` for workflow; `source/captest.rst` for API
- `load_data`, `load_pvsyst`, `DataLoader` (io.py) — `user_guide/dataload.rst`; `source/captest.rst`
- `ColumnGroups` (columngroups.py) — `user_guide/dataload.rst`
- `plot`, `residual_plot` (plotting.py) — `user_guide/dataload.rst` (Dashboard section)
- `perf_ratio`, `perf_ratio_temp_corr_nrel` (prtest.py) — no dedicated user guide page yet

## Workflow

### 1. Identify Significant Code Changes

Determine which commits need documentation updates:

- Find recent commits (default: since the last tag or last 24 hours; accept user-specified range)
- Examine each commit's diff to understand what was modified

**Update docs for:**
- New public methods, functions, parameters, or return values on `CapData`, `load_data`, `load_pvsyst`, `DataLoader`, `ColumnGroups`, or `prtest` functions
- Changes to default behavior or argument defaults that affect users
- New optional features or dependencies (e.g., new optional extras)
- New filter methods or changes to existing filter behavior
- Changes to `regression_cols`, `column_groups`, `rc`, or `regression_formula` semantics
- New CLI options or workflow steps
- Breaking changes that require users to update their code

**Skip docs for:**
- Internal refactoring with no API surface changes
- Test-only changes
- Minor bug fixes that don't change documented behavior
- Performance optimizations with no user-visible impact
- Linting, formatting, or CI config changes

When in doubt, skip. Quality over quantity.

### 2. Analyze Documentation Context

Before writing anything, read the relevant existing doc files to match their style. Pay attention to:

- RST cross-reference patterns: `:py:class:`, `:py:meth:`, `:py:func:`, `:py:attr:` with the full dotted path (e.g., `:py:meth:`~captest.capdata.CapData.fit_regression``)
- The tilde prefix (`~`) in cross-references, which shows only the last component in rendered output
- Code block format: `.. code-block:: Python` (capitalized)
- Note/warning directives: `.. note::` and `.. warning::` with blank line and 4-space indent
- Image references: `.. image:: ../_images/<filename>.png`
- Changelog entries follow Keep a Changelog format with `### Added`, `### Changed`, `### Fixed`, `### Removed` subsections under `## [Unreleased]`

The user guide (`dataload.rst`) is the main prose reference for the workflow. It uses a narrative, instructional tone — not a dry API dump. Explain *why* things work the way they do and give real code examples.

### 3. Determine Documentation Updates

Map each significant code change to one or more of these documentation targets:

| Change type | Primary target | Secondary target |
|---|---|---|
| New method/function | `user_guide/dataload.rst` (if workflow-relevant) | `changelog.md` |
| Changed default behavior | `user_guide/dataload.rst` (update affected section) | `changelog.md` |
| New parameter | Docstring in source (autodoc picks this up) | `user_guide/` if usage pattern changes |
| New module/class | `source/captest.rst` (add autodoc directive) | `user_guide/` if user-facing |
| Breaking change | `changelog.md` under `### Changed` | `user_guide/` update affected section |
| New optional dependency | `docs/installation.rst` | `changelog.md` |

**Changelog always gets an entry** for any user-visible change. Add to the `## [Unreleased]` section.

### 4. Write Documentation

**RST style rules:**
- Use NumPy-style docstring cross-references in prose: `:py:class:`, `:py:meth:`, `:py:func:`, `:py:attr:`
- Use `~` prefix to show short names in rendered HTML: `:py:meth:`~captest.capdata.CapData.agg_sensors``
- For inline code: use double backticks: ` ``column_groups`` `
- For code examples, use `.. code-block:: Python` (capital P)
- Preserve heading hierarchy: `=` overline+underline for page title, `-` underline for sections, `~` for subsections
- Add `.. note::` boxes for caveats, optional behavior, and non-obvious gotchas

**Changelog style rules:**
- Add to the `## [Unreleased]` section at the top of `docs/changelog.md`
- Use past tense, imperative mood: "Added `foo` method to ..." / "Changed `bar` to ..."
- Be specific: name the exact method/parameter/class and what changed
- For breaking changes, mention what code to update

**Docstrings** (in source files — only edit if the change is clearly a docstring fix):
- NumPy-style: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` sections
- 88-character line limit (consistent with ruff)
- Keep in sync with actual parameter names and types

### 5. Build and Validate (when in execution mode)

Before committing, verify the docs build without errors:

```bash
just docs
```

Fix any Sphinx warnings about undefined cross-references or missing directives before opening a PR.

### 6. Execute or Report

**Testing mode** (user asks "what would change" or "show me what needs updating"):

Output a plain text summary:
- Which commits contain user-visible changes
- Which doc files would be modified and why
- Draft content for changelog entries
- Any cross-references that would need updating

**Execution mode** (running as automation or user says "do it"):

1. Create a branch: `docs/update-YYYYMMDD` (or more descriptive if triggered by a specific feature)
2. Edit the relevant `.rst` and `.md` files
3. Run `just docs` and fix any build errors
4. Commit with a message like:
   ```
   docs: update user guide and changelog for <feature summary>
   
   Co-Authored-By: Oz <oz-agent@warp.dev>
   ```
5. Push and open a PR, linking to the source commits that triggered the updates

## Edge Cases

- If the API reference is out of date but the change is purely in autodoc (docstrings), note it but don't create a PR just for that — defer to the next substantive docs update
- If an example notebook is affected, note it in the PR description but don't modify notebooks directly (they require kernel execution)
- If no significant user-visible changes are found, report that no docs update is needed
- If the `## [Unreleased]` section already has a relevant subsection, append to it rather than duplicating

## Key Principles

- **Match existing style**: RST syntax, cross-reference patterns, and prose tone must match what's already in `dataload.rst`
- **Changelog first**: When uncertain what to update, the changelog entry is always the minimum viable documentation
- **Don't document internals**: `DataLoader._reindex_loaded_files` is private; `DataLoader.reindex_loaded_files` (if made public) warrants docs
- **Prefer additive changes**: Don't rewrite working documentation; extend it
- **Validate the build**: A Sphinx warning is a bug; run `just docs` before any PR
