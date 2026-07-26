---
name: merge-pr
description: Squash-merge a GitHub PR for pvcaptest, clean up branches, and keep all local copies in sync. Use this skill whenever the user says "merge the PR", "merge PR #N", "merge this branch", "squash and merge", "land this PR", or anything about finalizing a feature and cleaning up. Always use this skill for any pvcaptest merge workflow — it runs safety checks before merging so nothing gets lost.
---

# merge-pr

A workflow for safely merging pvcaptest PRs and keeping the fork and local upstream clone in sync.

## Repository layout

- Fork (working directory): `~/python/pvcaptest-fork`
  - `origin` → `git@github.com:bt-/pvcaptest.git`
  - `upstream` → `git@github.com:pvcaptest/pvcaptest.git`
- Local upstream clone: `~/python/pvcaptest/`
  - `origin` → `git@github.com:pvcaptest/pvcaptest.git`

Use `git -C <path>` in commands. The `g` alias exists only in the user's interactive
shell and is not available to non-interactive tool calls.

## Where the PR lives

Feature branches are pushed to the **fork** (`bt-/pvcaptest`) but the PR is opened
**against upstream master** (`pvcaptest/pvcaptest:master`). This cross-repo shape
determines every `gh` invocation:

- **All `gh pr ...` commands target `--repo pvcaptest/pvcaptest`** — that is where the
  PR object lives. Querying `--repo bt-/pvcaptest` will not find it.
- The **head branch** lives on `bt-/pvcaptest`, so branch deletion happens against
  `origin`, not upstream.
- Merge permissions are on upstream. Some upstream API endpoints are restricted:
  `gh run view --log-failed` returns `HTTP 403: Must have admin rights` even when
  `gh pr checks` works fine. Use `gh pr checks` / `gh run view` (no `--log`) for
  status, and reproduce failures locally rather than chasing CI logs.

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
git -C ~/python/pvcaptest-fork status --porcelain
```

- **Modified or staged tracked files** (`M`/`A`/`D` lines) → stop. List them and tell
  the user to commit or stash first — merging with a dirty tree risks losing work.
- **Untracked files** (`??` lines) → not a blocker, but check whether any belong in the
  PR before merging, since the branch is deleted afterward. Local-only scratch
  (`docs/superpowers/plans/*.md`, `docs/superpowers/specs/*.md`, scratch notebooks)
  is expected to stay untracked — see the "no specs/plans on remotes" convention.
  If an untracked file looks like it belongs to the feature, ask before proceeding.

Also confirm the branch is fully pushed, or the merge will land stale code:

```bash
git -C ~/python/pvcaptest-fork rev-list --left-right --count origin/{branch}...HEAD
```

Anything other than `0	0` means push (or pull) before merging.

### 1c. Check that CI is passing

Fetch the status of all checks on the PR:

```bash
gh pr checks {pr_number} --repo pvcaptest/pvcaptest
```

If the PR number isn't known yet, get it first:

```bash
gh pr view {pr_number} --repo pvcaptest/pvcaptest --json number,headRefName,title,state,mergeable
```

Evaluate the output of `gh pr checks`:
- **All checks pass** → proceed.
- **Any check is failing** → stop. List the failing checks and tell the user CI must pass before merging.
- **Any check is still pending/in progress** → wait and monitor until checks finish, then check that tests pass.

### 1d. Check whether docs are up to date

Get the list of files changed on the current branch relative to upstream/master:

```bash
git -C ~/python/pvcaptest-fork fetch upstream master
git -C ~/python/pvcaptest-fork diff upstream/master...HEAD --name-only
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
git -C ~/python/pvcaptest-fork branch --show-current
gh pr view {pr_number} --repo pvcaptest/pvcaptest --json number,headRefName,title,body
```

Confirm with the user if anything is ambiguous.

---

## Step 3: Compose the squash commit message

**Always author the squash commit message explicitly.** Never accept GitHub's default:
it concatenates every commit on the branch, so a 26-commit branch produces an unreadable
wall of text with the same `Co-Authored-By` trailer repeated once per commit.

Gather source material first:

```bash
git -C ~/python/pvcaptest-fork log upstream/master..HEAD --oneline
gh pr view {pr_number} --repo pvcaptest/pvcaptest --json body -q .body
```

Then write the message to these rules:

- **First line contains the PR number and the head branch name**, after a conventional
  commit prefix. Format: `{type}: {concise description} (#{pr_number}, {branch_name})`
  - Example: `feat: CapTest run_test orchestrator and staged config lifecycle (#165, captest-lifecycle)`
- **Body is concise bullets** summarizing what actually landed, grouped by theme rather
  than one bullet per commit. Condense aggressively — a 26-commit branch should read as
  roughly 10 bullets. Prefer the PR body's "Changes" section, tightened, over the raw
  commit list. Backtick API names (`CapTest.run_test`, `rerun_filters_from`).
- **No duplicate author attributions.** At most one `Co-Authored-By` trailer and one
  `Claude-Session` line, at the very end of the body — never one per squashed commit.
- **Close with what a reader needs**: breaking-change status (e.g. "Breaking changes are
  pre-v1 and recorded in CHANGELOG") and the test result summary, one line each.

Write the body to a scratch file so quoting and newlines survive the shell:

```bash
# write body to $SCRATCH/merge-body.txt, then pass with --body-file
```

---

## Step 4: Merge the PR

```bash
gh pr merge {pr_number} --repo pvcaptest/pvcaptest -s -d \
  --subject "{first line from Step 3}" \
  --body-file "{path to body file}"
```

- `-s` squash-merges all commits on the branch into a single commit on upstream/master
- `-d` requests deletion of the source branch after merge
- `--subject` / `--body-file` supply the Step 3 message instead of the default

A successful merge prints **nothing**. Do not read the silence as failure — verify:

```bash
gh pr view {pr_number} --repo pvcaptest/pvcaptest --json state,mergedAt,mergeCommit \
  -q '.state, .mergedAt, .mergeCommit.oid'
```

If the merge fails, report the error and stop.

---

## Step 5: Sync local master with upstream and push to fork

Do this **before** deleting the local branch — the branch cannot be deleted while checked out.

```bash
git -C ~/python/pvcaptest-fork checkout master
git -C ~/python/pvcaptest-fork pull upstream master
git -C ~/python/pvcaptest-fork push origin master
```

This pulls the squash-merge commit from upstream and pushes it to the fork so `bt-/pvcaptest:master` stays in sync.

---

## Step 6: Delete the feature branch

Because the head branch is on a fork, `gh pr merge -d` frequently does **not** remove it.
Always run the explicit deletes and confirm.

Remote (on the fork):

```bash
git -C ~/python/pvcaptest-fork push origin -d {branch_name}
```

An error of "remote ref does not exist" is fine — it means `gh` already cleaned up.

Local — squash merges leave the branch commits as non-ancestors of master, so `git branch -d`
refuses with "not fully merged" and `-D` is required. **Verify nothing is lost first:**

```bash
git -C ~/python/pvcaptest-fork diff {branch_name} master --stat   # must be empty
git -C ~/python/pvcaptest-fork branch -D {branch_name}
```

Empty diff = the squash captured every change. If it is **not** empty, stop and report —
something on the branch did not make it into the squash commit.

If deleting the branch prints `error: could not lock config file .git/config`, the ref was
still deleted but a stale stanza remains. Clean it up:

```bash
git -C ~/python/pvcaptest-fork config --remove-section branch.{branch_name}
```

---

## Step 7: Update the local upstream clone

```bash
git -C ~/python/pvcaptest pull origin master
```

This keeps `~/python/pvcaptest/` current with the squash-merge commit.

---

## Step 8: Verify and report

Confirm all four copies of master agree:

```bash
git -C ~/python/pvcaptest-fork rev-parse master origin/master upstream/master
git -C ~/python/pvcaptest rev-parse master
```

Then confirm to the user:
- Which PR was merged (number + title) and the squash commit SHA
- The squash commit subject line, so they can see the message convention was applied
- That the feature branch is deleted (local + fork remote)
- That `master` is synced across the fork origin, upstream, and the local upstream clone
- Anything intentionally left alone (untracked scratch files, unrelated local branches)
