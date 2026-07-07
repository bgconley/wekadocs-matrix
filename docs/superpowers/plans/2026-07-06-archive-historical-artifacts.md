# Archive Historical Root Artifacts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This is **Slice A** of the Refactor-Roadmap wrap-up (`docs/repo_audit/10_refactor_roadmap.md`, Phase 5b: "Replace old root context clutter with an index of historical artifacts"). It is deliberately the safest, offline, no-runtime-code slice. Sequence: **A (this) → B (legacy quarantine) → C (AtomicIngestionCoordinator split) → D (release proof, live-gated).**

**Goal:** Move the historical/session/completed-plan clutter out of the repo root into `docs/archive/` behind a single index, and delete the one superseded plan — so the root shows only active docs.

**Architecture:** Pure `git mv` / `git rm` reorganization plus one new index file. **No runtime source is touched.** Every archive candidate was verified to be referenced by zero active code/CI/tooling files, so moves cannot break imports, links, or hooks. The tightened rename guard already excludes `docs/archive/`, so archived WEKA-era history stays out of any live residue scan.

**Tech Stack:** Git, ripgrep. No Python, no tests to author.

**Execution note:** Create/overwrite files with the Write tool and edit with Edit (apply_patch-style), not shell heredoc redirection. **Commit messages must pass gitlint** (enforced at `commit-msg`): title ≤72 chars with a `(pN.N)` numeric scope, a blank line, then a `Phase N.N — <recap>` body whose lines are ≤80 chars. `git commit` examples are shown title-first — append the `-m "Phase N.N — …"` body when committing.

---

## File Structure

`docs/archive/` already exists. Create two subdirectories and one index:

- Create: `docs/archive/session-history/` (dated context/session/progress dumps)
- Create: `docs/archive/cleanup-2026/` (the completed cleanup/refactor roadmap + audit artifacts)
- Create: `docs/archive/README.md` (the historical-artifact index — satisfies the roadmap requirement)

### Keep at root (active — do NOT move)

- `CLAUDE.md` — harness/project instructions (active)
- `AGENTS.md` — GitNexus wrapper (active tooling)
- `ARCHITECTURE.md` — current architecture reference
- `REPO-MAP.md` — current repo map reference

### Archive → `docs/archive/session-history/`

`context-*.md` (27 files), `AGENT_CONTEXT.md`, `COMPREHENSIVE-SUMMARY.md`, `PROGRESS_SUMMARY_2025-10-13.md`, `SESSION_PROGRESS_2025-10-13.md`, `SESSION-SUMMARY.md`, `SESSION_CONTEXT_20260119_ARCTIC_EMBEDDER.md`, `progress.md`, `DEV-HISTORY.md`, `EMBEDDER_CORRECTIONS.md`, `rrf-fusion-no-neo4j-20251205.md`

### Archive → `docs/archive/cleanup-2026/`

`CLEANUP-PLAN.md`, `CLEANUP-EXECUTION-PLAN.md`, `REFACTOR-PLAN.md`, `STATUS.md`, `DEAD-CODE-MAP.md`, `dead_code_audit_report.md`, `AUDIT-VERIFICATION.md`

### Delete (superseded)

- `docs/superpowers/plans/2026-07-04-evidence-package-core.md` — v1, superseded by the shipped `-v2` plan.

### Explicit Out Of Scope

- Do not touch `docs/repo_audit/10_refactor_roadmap.md` (the live wrap-up tracker) or the *shipped* plans: `2026-07-04-evidence-package-core-v2.md`, `2026-07-04-stabilize-nutanix-wip-branch.md`, `2026-07-05-finish-live-weka-residue-rename.md`, and the four `2026-07-06-*` wrap-up plans. **The one sanctioned deletion is the superseded `2026-07-04-evidence-package-core.md` (v1) in Task 4** — everything else in `docs/superpowers/plans/` stays.
- Do not touch `reports/**`, `claude-raw/**`, `repo-analysis-artifacts/**` (already out of the active surface; separate archival decision).
- Do not deduplicate `docs/plans/*copy*.md` here (a later docs-tidy can do that).
- Do not edit any file under `src/`, `tests/`, `config/`, `scripts/`.

---

## Task 1: Preflight — Confirm Branch And Prove The Archive Set Is Unreferenced

**Files:** read-only.

- [ ] **Step 1: Confirm branch and clean tree**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git branch --show-current
git status --porcelain=v1 -uall | head
```

Expected: `wip/weka-to-nutanix-migration`, and no unexpected pre-existing modifications (a clean tree, or only untracked files you recognize).

- [ ] **Step 2: Prove every archive candidate is referenced by zero active files**

This is the safety gate — moving a referenced doc would break a link/hook.

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
ARCHIVE_SET="AGENT_CONTEXT.md COMPREHENSIVE-SUMMARY.md PROGRESS_SUMMARY_2025-10-13.md SESSION_PROGRESS_2025-10-13.md SESSION-SUMMARY.md SESSION_CONTEXT_20260119_ARCTIC_EMBEDDER.md progress.md DEV-HISTORY.md EMBEDDER_CORRECTIONS.md rrf-fusion-no-neo4j-20251205.md CLEANUP-PLAN.md CLEANUP-EXECUTION-PLAN.md REFACTOR-PLAN.md STATUS.md DEAD-CODE-MAP.md dead_code_audit_report.md AUDIT-VERIFICATION.md"
for f in $ARCHIVE_SET context-1.md; do
  hits=$(rg -l --fixed-strings "$f" src tests scripts .github Makefile pyproject.toml docker-compose.yml CLAUDE.md AGENTS.md deploy config 2>/dev/null | wc -l | tr -d ' ')
  [ "$hits" != "0" ] && echo "REFERENCED ($hits): $f"
done
echo "done"
```

Expected: only `done` (no `REFERENCED` lines). If any file is referenced, remove it from the archive set and keep it at root; note why in Task 5's index.

- [ ] **Step 3: Confirm the superseded v1 plan is the dead one**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
ls docs/superpowers/plans/2026-07-04-evidence-package-core.md docs/superpowers/plans/2026-07-04-evidence-package-core-v2.md
```

Expected: both exist. The non-`-v2` file is the rejected v1 to delete.

---

## Task 2: Archive Session/Context History

**Files:** move root session/context/progress docs.

- [ ] **Step 1: Create the target directory**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
mkdir -p docs/archive/session-history
```

- [ ] **Step 2: Move the `context-*.md` dumps**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git mv context-*.md docs/archive/session-history/
```

Expected: empty output. (Globs expand to the 27 tracked `context-N.md` files.)

- [ ] **Step 3: Move the dated session/progress summaries**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git mv \
  AGENT_CONTEXT.md \
  COMPREHENSIVE-SUMMARY.md \
  PROGRESS_SUMMARY_2025-10-13.md \
  SESSION_PROGRESS_2025-10-13.md \
  SESSION-SUMMARY.md \
  SESSION_CONTEXT_20260119_ARCTIC_EMBEDDER.md \
  progress.md \
  DEV-HISTORY.md \
  EMBEDDER_CORRECTIONS.md \
  rrf-fusion-no-neo4j-20251205.md \
  docs/archive/session-history/
```

Expected: empty output. If any single file is missing, drop it from the list and re-run (do not abort the whole move).

- [ ] **Step 4: Verify the moves are staged renames (not delete+add)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git status --short | rg '^R' | wc -l
```

Expected: a count ≥ 37 (27 context files + the summaries), all shown as `R` (rename) — confirming git detected renames and history is preserved.

---

## Task 3: Archive The Completed Cleanup/Refactor Roadmap

**Files:** move the finished Phase 0–7 cleanup docs.

- [ ] **Step 1: Create the target directory and move**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
mkdir -p docs/archive/cleanup-2026
git mv \
  CLEANUP-PLAN.md \
  CLEANUP-EXECUTION-PLAN.md \
  REFACTOR-PLAN.md \
  STATUS.md \
  DEAD-CODE-MAP.md \
  dead_code_audit_report.md \
  AUDIT-VERIFICATION.md \
  docs/archive/cleanup-2026/
```

Expected: empty output. These correspond to the `STATUS.md` "Cleanup Execution Status" roadmap that completed at commit `7745c92 docs(p7.0)`.

> Note: the **active** wrap-up tracker is `docs/repo_audit/10_refactor_roadmap.md` — it is **not** moved. Only the completed *cleanup* roadmap and its audit artifacts are archived.

---

## Task 4: Delete The Superseded v1 Plan

**Files:** `docs/superpowers/plans/2026-07-04-evidence-package-core.md`

- [ ] **Step 1: Remove the rejected v1 evidence plan**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git rm docs/superpowers/plans/2026-07-04-evidence-package-core.md
```

Expected: `rm 'docs/superpowers/plans/2026-07-04-evidence-package-core.md'`. Its content is preserved in git history and superseded by `-v2` (shipped).

---

## Task 5: Write The Historical-Artifact Index

**Files:** Create `docs/archive/README.md`

This satisfies the roadmap's Phase 5b requirement ("replace root clutter with an **index** of historical artifacts").

- [ ] **Step 1: Create the index**

Create `docs/archive/README.md` with this content:

```markdown
# Archived Historical Artifacts

Point-in-time documents moved out of the repo root during the Refactor-Roadmap
wrap-up (Phase 5b). They are **historical records**, not active guidance. For the
current state, see `ARCHITECTURE.md`, `REPO-MAP.md`, and
`docs/repo_audit/10_refactor_roadmap.md`.

These paths are excluded from the live WEKA-residue guard by policy
(`docs/archive/` is in `EXCLUDED_PREFIXES`).

## session-history/
Session logs, context dumps (`context-*.md`), and dated progress/summary notes
from the 2025-10 → 2026-01 build-out. Superseded by later work; kept for provenance.

## cleanup-2026/
The completed "Cleanup Execution" roadmap (`STATUS.md`, `CLEANUP-PLAN.md`,
`CLEANUP-EXECUTION-PLAN.md`, `REFACTOR-PLAN.md`) and its audit artifacts
(`DEAD-CODE-MAP.md`, `dead_code_audit_report.md`, `AUDIT-VERIFICATION.md`).
That roadmap finished at commit `7745c92 docs(p7.0)`.

## Not archived here
- `docs/repo_audit/10_refactor_roadmap.md` — the **active** overarching roadmap.
- `reports/**`, `claude-raw/**`, `repo-analysis-artifacts/**` — separate archival decisions.
```

Expected: `docs/archive/README.md` exists with the index above.

---

## Task 6: Verify, Commit, And Push

- [ ] **Step 1: Confirm the root is clean of the archived set**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
ls context-*.md 2>/dev/null | wc -l          # expect 0
ls STATUS.md CLEANUP-PLAN.md DEV-HISTORY.md 2>/dev/null   # expect: no such files
ls CLAUDE.md AGENTS.md ARCHITECTURE.md REPO-MAP.md        # expect: all still present
```

Expected: 0 context files at root; the archived docs gone from root; the 4 keepers present.

- [ ] **Step 2: Confirm no active reference now dangles**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rg -n --fixed-strings -e "STATUS.md" -e "CLEANUP-PLAN" -e "DEAD-CODE-MAP" -e "context-" \
  src tests scripts .github Makefile pyproject.toml docker-compose.yml CLAUDE.md AGENTS.md 2>/dev/null \
  | rg -v 'docs/archive' || echo "no active references — clean"
```

Expected: `no active references — clean`.

- [ ] **Step 3: Sanity — nothing under `src/`, `tests/`, `config/` changed**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git status --short | rg '^[ MARD].*(src/|tests/|config/|scripts/|\.env|tools/)' || echo "no runtime/config/test changes ✓"
```

Expected: `no runtime/config/test changes ✓` (this slice moves docs only).

- [ ] **Step 4: The residue guard is unaffected (still green)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_nutanix_hard_rename_guard.py -q
```

Expected: `2 passed`. (Root/`docs/archive/` are not in the guard's active surface; this just proves nothing regressed.)

- [ ] **Step 5: GitNexus change detection (docs-only)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git add docs/archive/
# (the git mv renames and the v1 `git rm` are already staged; `git add docs/archive/` picks up
#  the new index. Do NOT `git add -A` — the untracked 2026-07-06-*.md plans must not be swept in.)
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected: no runtime symbol/flow changes (moves/deletes of `.md` files map to no indexed symbols).

- [ ] **Step 6: Commit and push**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit \
  -m "chore(p5.0): archive historical root artifacts + index" \
  -m "Phase 5.0 — archive root context/summary docs + cleanup roadmap; delete v1 plan."
git push origin wip/weka-to-nutanix-migration
```

Expected: a new commit on `wip/weka-to-nutanix-migration`, pushed and in sync.

---

## Acceptance Criteria

- The repo root no longer contains `context-*.md` or the dated session/progress/cleanup/audit docs.
- Root retains only active docs (`CLAUDE.md`, `AGENTS.md`, `ARCHITECTURE.md`, `REPO-MAP.md`) plus standard project files.
- `docs/archive/session-history/` and `docs/archive/cleanup-2026/` contain the moved files as git **renames** (history preserved).
- `docs/archive/README.md` indexes what was archived and why.
- The superseded `2026-07-04-evidence-package-core.md` (v1) is deleted.
- No file under `src/`, `tests/`, `config/`, `scripts/`, `tools/`, or `.env.example` is modified.
- `tests/test_nutanix_hard_rename_guard.py` is still `2 passed`.
- Branch pushed to `origin/wip/weka-to-nutanix-migration`.

## Self-Review

- Spec coverage: preflight reference-safety gate, session-history archive, cleanup-roadmap archive, v1 deletion, index creation, verification, commit/push.
- Placeholder scan: no TBD/TODO; every file list and command is concrete.
- Type consistency: no runtime types/interfaces introduced (docs-only slice).
- Reversibility: all moves are git-tracked renames; nothing is destroyed (v1 lives in history).
