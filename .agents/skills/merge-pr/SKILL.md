---
name: merge-pr
description: Squash-merge a GitHub PR for pvcaptest, clean up branches, and keep all local copies in sync. Use this skill whenever the user says "merge the PR", "merge PR #N", "merge this branch", "squash and merge", "land this PR", or anything about finalizing a feature and cleaning up. Always use this skill for any pvcaptest merge workflow — it runs safety checks before merging so nothing gets lost.
---

# merge-pr

A workflow for safely merging pvcaptest PRs and keeping the fork and local upstream clone in sync.

## Repository layout

- Fork (working directory): `~/python/pvcaptest_bt-`
  - `origin` → `git@github.com:bt-/pvcaptest.git`
  - `upstream` → `git@github.com:pvcaptest/pvcaptest.git`
- Local upstream clone: `~/python/pvcaptest/`
  - `origin` → `git@github.com:pvcaptest/pvcaptest.git`

All `git` commands use the `g` shell alias.

---

## Step 1: Pre-flight checks

Run all four checks before doing anything else. If any check fails, stop and tell the user what needs to be resolved.

### 1a. Verify the GitHub CLI is available

```bash
gh --version
```

If this fails, stop. Tell the user to install the GitHub CLI (`gh`) before proceeding.

### 1b. Check for uncommitted work

```bash
g -C ~/python/pvcaptest_bt- status --porcelain
```

If any output appears (staged or unstaged changes), stop. List the dirty files and tell the user to commit or stash them first — merging with a dirty tree risks losing work.

### 1c. Check that CI is passing

Fetch the status of all checks on the PR:

```bash
gh pr checks {pr_number} --repo bt-/pvcaptest
```

If the PR number isn't known yet, get it first:

```bash
gh pr view --json number,headRefName,title --repo bt-/pvcaptest
```

Evaluate the output of `gh pr checks`:
- **All checks pass** → proceed.
- **Any check is failing** → stop. List the failing checks and tell the user CI must pass before merging.
- **Any check is still pending/in progress** → wait and monitor until checks finish, then check that tests pass.

### 1d. Check whether docs are up to date

Get the list of files changed on the current branch relative to upstream/master:

```bash
g -C ~/python/pvcaptest_bt- diff upstream/master...HEAD --name-only
```

Parse the output:
- If any `src/captest/` Python files appear in the diff **and** no `docs/` files appear, the docs have likely not been updated to reflect the code changes.
- Ask the user: *"It looks like source files changed but the docs haven't been updated. Would you like to run the `docs-update` skill before merging?"*
  - If yes: invoke the `docs-update` skill, then return here and continue.
  - If no: proceed.
- If docs files also appear in the diff, or only non-source files changed, proceed without asking.

---

## Step 2: Get PR and branch info

Determine the PR number and feature branch name if not already known:

```bash
g -C ~/python/pvcaptest_bt- branch --show-current
gh pr view --json number,headRefName,title --repo bt-/pvcaptest
```

Confirm with the user if anything is ambiguous.

---

## Step 3: Merge the PR

```bash
gh pr merge {pr_number} -s -d
```

- `-s` squash-merges all commits on the branch into a single commit on upstream/master
- `-d` deletes the source branch from the remote after merge

If the merge fails, report the error and stop.

---

## Step 4: Delete the remote feature branch from the fork

`gh pr merge -d` should have already removed the branch, but run this as a safety step:

```bash
g -C ~/python/pvcaptest_bt- push origin -d {branch_name}
```

If this errors with "remote ref does not exist", that's fine — it just means `gh` already cleaned it up.

---

## Step 5: Sync local master with upstream and push to fork

After the PR is merged into upstream, the local master branch needs to be updated:

```bash
g -C ~/python/pvcaptest_bt- checkout master
g -C ~/python/pvcaptest_bt- pull upstream master
g -C ~/python/pvcaptest_bt- push origin master
```

This pulls the squash-merge commit from upstream and pushes it to the fork so `bt-/pvcaptest:master` stays in sync.

---

## Step 6: Update the local upstream clone

```bash
g -C ~/python/pvcaptest pull origin master
```

This keeps `~/python/pvcaptest/` current with the squash-merge commit.

---

## Summary

When all steps complete, confirm to the user:
- Which PR was merged (number + title)
- That the feature branch is deleted (local + both remotes)
- That `master` is synced across the fork origin and local upstream clone
