---
name: create-pr
description: Create a GitHub pull request for the pvcaptest fork following project conventions. Use this skill whenever the user asks to create a PR, open a pull request, submit changes for review, or push a branch up for review. Handles pre-flight checks (tests, lint, docs), commit analysis, and PR creation via the gh CLI targeting the upstream pvcaptest repo.
---

# Create Pull Request (pvcaptest)

This skill walks through creating a well-structured pull request for the pvcaptest fork. The repo has a fork/upstream layout — all PRs target `pvcaptest/pvcaptest:master`.

## Repository layout

- Fork (working directory): `~/python/pvcaptest_bt-`
  - `origin` → `git@github.com:bt-/pvcaptest.git`
  - `upstream` → `git@github.com:pvcaptest/pvcaptest.git`
- Local upstream clone: `~/python/pvcaptest/`

All git commands use the `g` shell alias. All recipe commands use `just`.

---

## Step 1: Pre-flight checks

Run all checks before doing anything else. Stop and tell the user what needs fixing if any fail.

### 1a. Verify gh CLI is installed and authenticated

```bash
gh --version
gh auth status
```

If not installed or not authenticated, stop and guide the user accordingly.

### 1b. Check for uncommitted changes

```bash
g -C ~/python/pvcaptest_bt- status --porcelain
```

If output appears, stop. List the dirty files and ask the user to commit or stash before continuing.

### 1c. Check for an existing PR on this branch

```bash
gh pr list --head $(g -C ~/python/pvcaptest_bt- branch --show-current) --repo pvcaptest/pvcaptest --json number,title,url
```

If a PR already exists, show the details and ask if the user wants to view it, push more commits to update it, or close it and open a new one. Only proceed with creation if there is no existing PR.

### 1d. Verify we're not on master

```bash
g -C ~/python/pvcaptest_bt- branch --show-current
```

If the current branch is `master`, stop and ask the user to switch to or create a feature branch.

### 1e. Run tests and linting

These must pass before opening a PR (run in order shown):

```bash
just -f ~/python/pvcaptest_bt-/.justfile lint
just -f ~/python/pvcaptest_bt-/.justfile fmt
just -f ~/python/pvcaptest_bt-/.justfile test
```

If tests or linting fail, stop and report the failures. Do not proceed until they are resolved.

### 1f. Check whether docs need updating

Get the files changed on this branch relative to upstream:

```bash
g -C ~/python/pvcaptest_bt- diff upstream/master...HEAD --name-only
```

If any `src/captest/` Python files are changed but no `docs/` files appear in the diff, the docs likely haven't been updated. Ask the user: *"Source files changed but docs haven't been updated. Would you like to run the `docs-update` skill before creating the PR?"*
- If yes: invoke `docs-update`, then return here and continue.
- If no: proceed.

---

## Step 2: Gather context

### 2a. Analyze commits on this branch

```bash
g -C ~/python/pvcaptest_bt- log upstream/master..HEAD --oneline
```

Review these commits to understand the scope and intent of the changes.

### 2b. Review the diff summary

```bash
g -C ~/python/pvcaptest_bt- diff upstream/master..HEAD --stat
```

This shows which files changed and helps identify the type of change (feature, fix, refactor, docs, etc.).

---

## Step 3: Push the branch

Ensure all commits are pushed to the fork:

```bash
g -C ~/python/pvcaptest_bt- push origin HEAD
```

If the branch was rebased, use:

```bash
g -C ~/python/pvcaptest_bt- push origin HEAD --force-with-lease
```

---

## Step 4: Compose the PR

### PR title

Synthesize a clear, specific title from the commits and diff. Avoid generic titles like "fix", "update", or "changes".

Look for a conventional-commits style in the commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). If present, use that prefix. Otherwise use a plain descriptive title, e.g. "Add pagination to search results" or "Fix race condition in authentication flow."

If a GitHub issue number appears in commit messages or the branch name (patterns like `#123`, `fixes #123`, `issue-123`), append it to the title.

### PR body

There is no `.github/pull_request_template.md` in this repo. Use this structure:

```
## Summary

<What problem does this solve or what feature does it add? 1–3 sentences.>

## Changes

<Bullet list of the meaningful changes — files or behaviors modified.>

## Testing

<How was this tested? Mention test files modified or commands run.>

## Notes (optional)

<Breaking changes, follow-up work, or anything the reviewer should know.>
```

Fill each section from the commit messages and diff. Be specific — reference method names, parameter names, and module names where relevant.

If an issue number was found, add a `Closes #N` or `Fixes #N` line at the bottom of the body.

### Draft vs. ready-for-review

Use `--draft` when:
- Changes are incomplete or you want early feedback
- CI is expected to fail and you need help debugging
- You want to mark progress but aren't ready for merge

Skip `--draft` when tests pass and the work is complete.

---

## Step 5: Create the PR

```bash
gh pr create \
  --title "PR_TITLE" \
  --body "PR_BODY" \
  --base master \
  --repo pvcaptest/pvcaptest
```

For a draft:

```bash
gh pr create \
  --title "PR_TITLE" \
  --body "PR_BODY" \
  --base master \
  --repo pvcaptest/pvcaptest \
  --draft
```

---

## Step 6: Post-creation

Open the PR in the browser to verify it rendered correctly:

```bash
gh pr view --web --repo pvcaptest/pvcaptest
```

Remind the user that the CI workflow (`.github/workflows/pytest.yml`) will run automatically. Monitor it with:

```bash
gh pr checks --repo pvcaptest/pvcaptest
```

---

## Error reference

| Problem | What to do |
|---|---|
| No commits ahead of upstream/master | Ask if the user is on the right branch |
| Branch not pushed to origin | Run `g -C ~/python/pvcaptest_bt- push -u origin HEAD` |
| PR already exists | Show it; ask if they want to update or replace |
| Tests failing | Fix failures before proceeding |
| Merge conflicts with upstream | Guide through `g rebase upstream/master` |

---

## Co-author attribution

All commits should include the co-author line. When committing as part of this workflow:

```
Co-Authored-By: Oz <oz-agent@warp.dev>
```
