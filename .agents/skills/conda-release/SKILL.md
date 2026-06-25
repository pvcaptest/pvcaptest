---
name: conda-release
description: Release a new pvcaptest version to the conda-forge channel after it is published on PyPI. Use this skill whenever the user wants to update/release the conda package, cut a conda-forge release, merge the regro-cf-autotick-bot PR on the pvcaptest-feedstock, or bump the feedstock recipe to a new version. The gating step is always checking for dependency changes before merging.
---

# conda-release

Release a new pvcaptest version to conda-forge. After a version is published on PyPI, the
`regro-cf-autotick-bot` automatically opens a version-bump PR on the feedstock. The
maintainer's job is to **check for dependency changes**, verify the PR, and merge it — which
triggers conda-forge CI to build and upload the package to the `conda-forge` channel.

Reference (sections up through "Updating for a newly released Python versions"):
https://conda-forge.org/docs/maintainer/updating_pkgs/

## Repository layout

- captest source repo (working dir): `~/python/pvcaptest_bt-`
  - Runtime dependencies are defined in `pyproject.toml` (`[project].dependencies` and
    `[project.optional-dependencies].optional`).
- Conda feedstock fork: `~/python/pvcaptest-feedstock/`
  - `origin` → `git@github.com:bt-/pvcaptest-feedstock.git`
  - `upstream` → `git@github.com:conda-forge/pvcaptest-feedstock.git`
  - Recipe: `recipe/meta.yaml`
- `regro-cf-autotick-bot` has its own feedstock fork and opens one PR per new PyPI version
  against `conda-forge/pvcaptest-feedstock`. It is a regular user account, not a GitHub App.

`g` is the shell alias for `git`; the `git -C <path>` form is used to run commands against a
specific repo without changing directory.

---

## Step 1: Find the bot's version-bump PR

The bot author is the plain login `regro-cf-autotick-bot` (do NOT prefix with `app/` — that
filters for GitHub Apps and matches nothing):

```bash
gh pr list --repo conda-forge/pvcaptest-feedstock --author regro-cf-autotick-bot
gh pr view {pr_number} --repo conda-forge/pvcaptest-feedstock
gh pr diff {pr_number} --repo conda-forge/pvcaptest-feedstock
```

The PR bumps `version` and `sha256` in `recipe/meta.yaml` and usually also re-renders the CI
config (`.github/workflows/`, `.scripts/`, `build-locally.py`) — those re-render changes are
routine; focus review on `recipe/meta.yaml`.

If no bot PR exists yet (just published to PyPI), the bot can take time. Either wait, or do a
manual recipe update on a feedstock fork branch (see "Manual recipe update" below).

---

## Step 2: GATING — check for dependency changes (do this BEFORE merging)

This is the required gate. Dependencies in the recipe MUST match the package's actual
dependencies. **Never auto-edit dependencies — flag any change for manual user review first.**

### 2a. Diff the source dependencies between the last conda release and the new version

`{last}` = the current `version` in `recipe/meta.yaml`; `{new}` = the new version from the bot
PR title (e.g. "pvcaptest v0.16.0"). Tags use a `v` prefix. Fetch tags first in case the new
release tag isn't in the local clone:

```bash
git -C ~/python/pvcaptest_bt- fetch --tags
git -C ~/python/pvcaptest_bt- diff v{last} v{new} -- pyproject.toml
```

- **Empty diff** → no dependency changes. Nothing to flag. Proceed to Step 3.
- **Non-empty diff, but no changes inside `[project].dependencies` or
  `[project.optional-dependencies]`** (e.g. only ruff config, classifiers, build metadata) →
  no dependency change. Proceed to Step 3.
- **Non-empty diff that DOES change `dependencies` or `optional-dependencies`** → STOP.
  Summarize the exact dependency changes and ask the user how to reflect them in the recipe
  before editing.

### 2b. Reconcile pyproject deps against the recipe `run:` section

The feedstock `run:` requirements should cover every runtime dep. Naming differences are
expected and correct:

| pyproject.toml          | conda recipe (`run:`)   |
|-------------------------|-------------------------|
| `matplotlib`            | `matplotlib-base`       |
| `pvlib`                 | `pvlib-python`          |
| `numpy>1.24.4`          | `numpy >=1.24.4`        |
| `universal_pathlib`     | `universal_pathlib`     |

Note: the recipe intentionally lists the `optional` extras (`holoviews`, `panel`,
`pvlib-python`, `openpyxl`) as hard `run:` deps. `fsspec[s3]` is currently NOT in the recipe.
These are pre-existing choices — only act on *changes* since the last release, not on
long-standing differences (unless the user asks).

---

## Step 3: Verify the bot PR's recipe changes

In `recipe/meta.yaml` the bot should have:
- bumped `{% set version = "{new}" %}`
- updated `source.sha256`
- left `build.number` at `0` (correct for a **version** change; a metadata-only change instead
  *increments* the build number)

Verify the sha256 matches the real PyPI sdist:

```bash
curl -sL https://pypi.org/pypi/captest/{new}/json | \
  python3 -c "import sys,json; d=json.load(sys.stdin); \
  print([f['digests']['sha256'] for f in d['urls'] if f['filename'].endswith('.tar.gz')][0])"
```

Confirm CI is green:

```bash
gh pr checks {pr_number} --repo conda-forge/pvcaptest-feedstock
```

Both `linux_64_` (build) and `conda-forge-linter` should pass.

---

## Step 4: Surface the gate result, then merge

Report to the user before merging: the dependency-check result, the sha256 verification, and
CI status. Merging publishes a public package, so confirm the user wants to proceed.

Approve (the PR author is the bot, so the maintainer can approve it) and squash-merge:

```bash
gh pr review {pr_number} --repo conda-forge/pvcaptest-feedstock --approve --body "..."
gh pr merge  {pr_number} --repo conda-forge/pvcaptest-feedstock --squash
```

### Merge gotcha (expected)

The `main` branch ruleset returns `mergeStateStatus: BLOCKED` even when the PR is approved,
mergeable, and all checks pass. Conda-forge requires the maintainer to merge with **admin**
privileges (recipe maintainers hold admin on their own feedstock):

```bash
gh pr merge {pr_number} --repo conda-forge/pvcaptest-feedstock --squash --admin
```

The agent's auto-approval classifier will typically **deny** `--admin` (it bypasses branch
protection on a shared community repo) and may also deny a registry-publishing merge. When
blocked, do NOT work around it — stop and have the user either merge in the GitHub UI ("Squash
and merge") or run the admin merge themselves. The command for the **user** to run in their own
shell (the leading `!` is the Claude Code prompt prefix that runs a shell command inline, not
part of the command):

```
! gh pr merge {pr_number} --repo conda-forge/pvcaptest-feedstock --squash --admin
```

---

## Step 5: Sync the feedstock fork

After the merge lands on `upstream/main`:

```bash
g -C ~/python/pvcaptest-feedstock fetch upstream
g -C ~/python/pvcaptest-feedstock checkout main
g -C ~/python/pvcaptest-feedstock merge --ff-only upstream/main
g -C ~/python/pvcaptest-feedstock push origin main
```

---

## Step 6: Monitor the upload build and verify publication

Merging to `main` runs the build with `UPLOAD_PACKAGES: True`, which uploads to the
`conda-forge` channel on anaconda.org.

`{run_id}` below is the `databaseId` from the list output:

```bash
gh run list  --repo conda-forge/pvcaptest-feedstock --branch main --limit 1 \
  --json databaseId,displayTitle,status,conclusion
gh run watch {run_id} --repo conda-forge/pvcaptest-feedstock --exit-status
```

Once the build succeeds, confirm the package is indexed (allow a short delay):

```bash
curl -sL https://api.anaconda.org/package/conda-forge/pvcaptest | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['latest_version'])"
```

`latest_version` should equal the new version. Release complete.

---

## Manual recipe update (no usable bot PR, or dependency changes)

If there is no bot PR, or dependencies changed and the user approved specific recipe edits:

1. Work on a branch in the **fork** (`origin`), never directly on the conda-forge repo (avoids
   wasteful CI and premature publishing).
2. To adjust an existing bot PR, push commits directly to the bot's branch instead of opening a
   new PR.
3. Edit `recipe/meta.yaml`: `version`, `source.sha256`, and `requirements.host` /
   `requirements.run` per the approved dependency changes.
4. Build number: reset to `0` for a version change; increment by `1` for a metadata-only change
   at the same version.
5. Re-render the feedstock (`conda smithy rerender`, or let the bot's re-render stand) so CI
   config stays current.
6. Open/refresh the PR against `conda-forge/pvcaptest-feedstock` and continue from Step 3.

---

## Summary checklist

- [ ] Located the bot PR (or set up a manual update)
- [ ] **Dependency gate**: `pyproject.toml` diff between releases checked; any change flagged
      to the user before editing the recipe
- [ ] Recipe verified: version + sha256 (matches PyPI), build number `0` for version bump
- [ ] CI green; gate result surfaced to user; PR merged (admin merge via user)
- [ ] Feedstock fork synced to `upstream/main`
- [ ] Upload build succeeded; `latest_version` on `conda-forge` matches the new version
