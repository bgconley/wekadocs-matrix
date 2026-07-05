# Stabilize Nutanix WIP Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `wip/weka-to-nutanix-migration` safe to continue by removing local scratch noise, syncing it with current `master`, and producing an evidence-backed WEKA residue surface map.

**Architecture:** This is a branch-hygiene and audit plan, not a product-code refactor. It preserves the migration checkpoint, avoids rewriting the pushed WIP branch, merges current `master` into the WIP branch, and commits only durable audit output. Scratch files are backed up outside the repo before removal.

**Tech Stack:** Git, GitNexus CLI, ripgrep, Python standard library for scan classification, existing pre-commit hooks.

---

## File Structure

This stabilization should touch as little as possible.

- Local-only backup, not committed: `/tmp/wekadocs-nutanix-scratch-<timestamp>/`
- Remove from working tree after backup: `/Users/brennanconley/vibecode/wekadocs-matrix/.pi-tasks/`
- Remove from working tree after backup: `/Users/brennanconley/vibecode/wekadocs-matrix/gitnexus`
- Remove from working tree after backup: `/Users/brennanconley/vibecode/wekadocs-matrix/test_extract_import.py`
- Remove from working tree after backup: `/Users/brennanconley/vibecode/wekadocs-matrix/test_import.py`
- Remove from working tree after backup: `/Users/brennanconley/vibecode/wekadocs-matrix/test_verify_imports.py`
- Create and commit: `/Users/brennanconley/vibecode/wekadocs-matrix/docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md`
- Do not intentionally modify runtime source during this plan. If a merge conflict requires editing a function, class, or method, stop and run `npx gitnexus impact <symbol> --direction upstream --include-tests --repo wekadocs-matrix` before resolving that symbol.

## Task 1: Preflight Branch State

**Files:**
- Read only: repository git metadata

- [ ] **Step 1: Confirm current branch is the migration WIP branch**

Run:

```bash
git -C /Users/brennanconley/vibecode/wekadocs-matrix branch --show-current
```

Expected output:

```text
wip/weka-to-nutanix-migration
```

- [ ] **Step 2: Confirm the branch tracks the pushed WIP branch**

Run:

```bash
git -C /Users/brennanconley/vibecode/wekadocs-matrix status -sb
```

Expected shape:

```text
## wip/weka-to-nutanix-migration...origin/wip/weka-to-nutanix-migration
?? .pi-tasks/
?? gitnexus
?? test_extract_import.py
?? test_import.py
?? test_verify_imports.py
```

If this plan file has not been committed yet, this additional untracked line is also expected:

```text
?? docs/superpowers/plans/2026-07-04-stabilize-nutanix-wip-branch.md
```

If any tracked files appear as modified, stop and inspect them before continuing.

- [ ] **Step 3: Fetch current remote refs**

Run:

```bash
git -C /Users/brennanconley/vibecode/wekadocs-matrix fetch origin
```

Expected output may be empty. A successful exit code is enough.

## Task 2: Back Up And Remove Scratch Files

**Files:**
- Read: `.pi-tasks/`, `gitnexus`, `test_extract_import.py`, `test_import.py`, `test_verify_imports.py`
- Local backup: `/tmp/wekadocs-nutanix-scratch-<timestamp>/`
- Remove from working tree: listed scratch files

- [ ] **Step 1: Create a timestamped scratch backup**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
SCRATCH_BACKUP="/tmp/wekadocs-nutanix-scratch-$(date +%Y%m%d%H%M%S)"
mkdir -p "$SCRATCH_BACKUP"
for scratch_path in .pi-tasks gitnexus test_extract_import.py test_import.py test_verify_imports.py; do
  if [ -e "$scratch_path" ]; then
    cp -a "$scratch_path" "$SCRATCH_BACKUP/"
  fi
done
printf '%s\n' "$SCRATCH_BACKUP"
```

Expected output:

```text
/tmp/wekadocs-nutanix-scratch-<timestamp>
```

- [ ] **Step 2: Record scratch classification in the terminal before removal**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
printf '%s\n' \
  ".pi-tasks/ => historical pi-task worker ledger; backup only, remove from repo checkout" \
  "gitnexus => local standalone analyzer prototype; backup only, remove to avoid confusion with npx gitnexus" \
  "test_extract_import.py => disposable import probe; backup only, remove" \
  "test_import.py => disposable import probe; backup only, remove" \
  "test_verify_imports.py => disposable import probe; backup only, remove"
```

Expected output is exactly the five classification lines above.

- [ ] **Step 3: Remove the scratch files from the working tree**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rm -rf .pi-tasks gitnexus test_extract_import.py test_import.py test_verify_imports.py
```

Expected output is empty.

- [ ] **Step 4: Verify no scratch files remain in git status**

Run:

```bash
git -C /Users/brennanconley/vibecode/wekadocs-matrix status --porcelain=v1 -uall
```

Expected output is empty. If `.pi-tasks/` is recreated by a local tool before commit, add it to `.git/info/exclude` rather than `.gitignore`:

```bash
printf '%s\n' '.pi-tasks/' >> /Users/brennanconley/vibecode/wekadocs-matrix/.git/info/exclude
```

## Task 3: Merge Current Master Into The WIP Branch

**Files:**
- Merge source: `origin/master`
- Merge target: `wip/weka-to-nutanix-migration`
- Possible inherited master changes include `src/mcp_server/mcp_tools.py` and `src/shared/connections.py`.

- [ ] **Step 1: Start the merge without committing**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git merge --no-commit --no-ff origin/master
```

Expected successful shape:

```text
Automatic merge went well; stopped before committing as requested
```

If conflicts occur, run:

```bash
git status --short
```

For every conflicted function, class, or method that requires manual code resolution, run GitNexus impact before editing:

```bash
npx gitnexus impact <symbol-name> --direction upstream --include-tests --repo wekadocs-matrix
```

If GitNexus reports HIGH or CRITICAL risk, report the affected direct callers/processes before continuing.

- [ ] **Step 2: Run GitNexus change detection before the merge commit**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected acceptable result:

```text
No changes detected.
```

If GitNexus reports affected symbols or flows, inspect them and include the summary in the commit notes.

- [ ] **Step 3: Commit the merge**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit \
  -m "chore(p3.5): sync nutanix wip with master" \
  -m "Phase 3.5 syncs current master into the Nutanix migration branch before further hard-rename work."
```

Expected output includes a new merge commit on `wip/weka-to-nutanix-migration`.

- [ ] **Step 4: Verify master is now an ancestor of the WIP branch**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git merge-base --is-ancestor origin/master HEAD
```

Expected output is empty with exit code `0`.

## Task 4: Run The WEKA Residue Scan

**Files:**
- Create raw temporary scan: `/tmp/wekadocs-nutanix-weka-residue-20260704/raw-rg.txt`
- Create committed surface map: `docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md`

- [ ] **Step 1: Generate raw WEKA residue hits**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
SCAN_DIR=/tmp/wekadocs-nutanix-weka-residue-20260704
mkdir -p "$SCAN_DIR"
rg -n --hidden --pcre2 \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  --glob '!__pycache__/**' \
  --glob '!node_modules/**' \
  --glob '!reports/retrieval_diagnostics/**' \
  '(?i)(\bweka\b|wekadocs|weka[-_:]|weka docs|WekaDocs)' \
  . > "$SCAN_DIR/raw-rg.txt" || true
wc -l "$SCAN_DIR/raw-rg.txt"
```

Expected output is a non-negative line count. A zero count is allowed only if the migration is already complete, which is unlikely at this stage.

- [ ] **Step 2: Generate the committed surface map from the raw scan**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
mkdir -p docs/superpowers/findings
python - <<'PY'
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

repo = Path("/Users/brennanconley/vibecode/wekadocs-matrix")
raw_path = Path("/tmp/wekadocs-nutanix-weka-residue-20260704/raw-rg.txt")
out_path = repo / "docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md"

patterns = [
    "(?i)(\\bweka\\b|wekadocs|weka[-_:]|weka docs|WekaDocs)",
]

def category_for(path: str) -> str:
    clean = path.removeprefix("./")
    if clean.startswith(("src/", "scripts/", "tools/")) or clean in {
        "bootstrap_schema.py",
        "setup.py",
        "pyproject.toml",
    }:
        return "runtime_source"
    if clean.startswith((
        "config/",
        "docker/",
        "infra/",
        ".github/",
        "monitoring/",
        "ops/",
    )) or clean in {
        "docker-compose.yml",
        "docker-compose.dev.yml",
        "Makefile",
        ".env.example",
    }:
        return "config_deploy_ops"
    if clean.startswith("tests/"):
        return "tests"
    if clean.startswith(("archive/", "archives/", "docs/archive/")):
        return "archive_historical"
    if clean.startswith(("reports/", "repo-analysis-artifacts/")) or clean in {
        "DEAD-CODE-MAP.md",
    }:
        return "generated_reports"
    if clean.startswith(("docs/",)) or clean.lower().startswith(("readme", "agents", "claude")):
        return "docs_active"
    return "other"

raw_lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines() if raw_path.exists() else []
records: list[tuple[str, int, str, str]] = []
for line in raw_lines:
    parts = line.split(":", 2)
    if len(parts) != 3:
        continue
    file_path, line_no, text = parts
    try:
        line_int = int(line_no)
    except ValueError:
        line_int = 0
    records.append((file_path, line_int, category_for(file_path), text.strip()))

by_category = Counter(category for _, _, category, _ in records)
files_by_category: dict[str, set[str]] = defaultdict(set)
for file_path, _, category, _ in records:
    files_by_category[category].add(file_path)

blocker_categories = {"runtime_source", "config_deploy_ops", "tests"}
blockers = [record for record in records if record[2] in blocker_categories]

top_files = Counter(file_path for file_path, _, _, _ in records)

lines: list[str] = []
lines.append("# WEKA Residue Surface Map - 2026-07-04")
lines.append("")
lines.append("Generated from the Nutanix migration WIP branch after syncing current `origin/master`.")
lines.append("")
lines.append("## Scan Command")
lines.append("")
lines.append("```bash")
lines.append("rg -n --hidden --pcre2 --glob '!.git/**' --glob '!.venv/**' --glob '!__pycache__/**' --glob '!node_modules/**' --glob '!reports/retrieval_diagnostics/**' '(?i)(\\bweka\\b|wekadocs|weka[-_:]|weka docs|WekaDocs)' .")
lines.append("```")
lines.append("")
lines.append("## Summary")
lines.append("")
lines.append(f"- Total residue hits: {len(records)}")
lines.append(f"- Total files with residue: {len(set(file_path for file_path, _, _, _ in records))}")
lines.append(f"- Runtime/config/test blocker hits: {len(blockers)}")
lines.append("")
lines.append("| Category | Hits | Files | Migration Meaning |")
lines.append("| --- | ---: | ---: | --- |")
meaning = {
    "runtime_source": "Active source or scripts; blockers until renamed or intentionally archived.",
    "config_deploy_ops": "Runtime configuration, deployment, CI, or operations surfaces; blockers until renamed.",
    "tests": "Tests and fixtures; blockers unless explicitly historical.",
    "docs_active": "Active docs/plans; rename or mark historical.",
    "generated_reports": "Generated analysis/report artifacts; archive, regenerate, or exclude from runtime residue gates.",
    "archive_historical": "Historical archive candidates; allowed only under explicit archive policy.",
    "other": "Needs manual classification.",
}
for category in [
    "runtime_source",
    "config_deploy_ops",
    "tests",
    "docs_active",
    "generated_reports",
    "archive_historical",
    "other",
]:
    lines.append(
        f"| {category} | {by_category.get(category, 0)} | {len(files_by_category.get(category, set()))} | {meaning[category]} |"
    )
lines.append("")
lines.append("## Top Files By Hit Count")
lines.append("")
lines.append("| Hits | File |")
lines.append("| ---: | --- |")
for file_path, count in top_files.most_common(40):
    lines.append(f"| {count} | `{file_path}` |")
lines.append("")
lines.append("## Runtime/Config/Test Blockers")
lines.append("")
if blockers:
    lines.append("| Category | File | Line | Text |")
    lines.append("| --- | --- | ---: | --- |")
    for file_path, line_no, category, text in blockers[:300]:
        safe_text = text.replace("|", "\\|")
        lines.append(f"| {category} | `{file_path}` | {line_no} | `{safe_text[:220]}` |")
    if len(blockers) > 300:
        lines.append("")
        lines.append(f"Only the first 300 blocker rows are listed here. Full raw scan: `{raw_path}`.")
else:
    lines.append("No runtime/config/test blockers found by this scan.")
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- This map does not claim the migration is complete.")
lines.append("- Because this map is generated after syncing `origin/master`, it includes any WEKA strings reintroduced by that merge.")
lines.append("- Live embedding, reranking, Qdrant, Neo4j, MCP behavior, and quality tuning remain explicitly out of scope for this architecture/code-only pass.")
lines.append("- Historical WEKA material is allowed only when moved to explicit archive paths excluded from runtime residue gates.")
lines.append("")

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path)
PY
```

Expected output:

```text
/Users/brennanconley/vibecode/wekadocs-matrix/docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md
```

- [ ] **Step 3: Inspect the generated surface map**

Run:

```bash
sed -n '1,220p' /Users/brennanconley/vibecode/wekadocs-matrix/docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md
```

Expected output includes:

```text
# WEKA Residue Surface Map - 2026-07-04
## Summary
## Runtime/Config/Test Blockers
```

## Task 5: Verify Stabilized Branch

**Files:**
- Read only except generated diagnostics cleanup if tests create reports

- [ ] **Step 1: Run lightweight static checks**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python -m compileall -q src scripts
ruff check src/mcp_server/mcp_tools.py src/shared/connections.py
```

Expected output:

```text
All checks passed!
```

`compileall` is silent on success.

Do not broaden this lint gate during this stabilization task. The current WIP branch has pre-existing lint failures outside this plan's touched source set in `scripts/verify_dead_code.py`, `src/ingestion/atomic.py`, and `src/mcp_server/mcp_app.py`. Those belong to a separate cleanup slice.

- [ ] **Step 2: Run the focused evidence/MCP tests affected by the master sync**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/evidence tests/mcp_server_tests -q
```

Expected output shape:

```text
56 passed
```

If this test run creates or deletes retrieval diagnostic files, clean those generated artifacts before committing:

```bash
git restore -- reports/retrieval_diagnostics/2026-03-04 reports/retrieval_diagnostics/2026-03-05 2>/dev/null || true
rm -rf reports/retrieval_diagnostics/2026-07-04
```

- [ ] **Step 3: Check final working tree before commit**

Run:

```bash
git -C /Users/brennanconley/vibecode/wekadocs-matrix status -sb
```

Expected shape:

```text
## wip/weka-to-nutanix-migration...origin/wip/weka-to-nutanix-migration [ahead N]
?? docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md
```

If source files are modified unexpectedly, inspect them before continuing.

## Task 6: Commit And Push The Surface Map

**Files:**
- Commit: `docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md`

- [ ] **Step 1: Stage the surface map**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git add docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md
```

Expected output is empty.

- [ ] **Step 2: Run GitNexus change detection before committing**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected acceptable result:

```text
No changes detected.
```

- [ ] **Step 3: Commit the surface map**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit \
  -m "docs(p3.5): map remaining weka residue" \
  -m "Phase 3.5 records the actual remaining WEKA residue surfaces on the Nutanix migration WIP branch before further hard-rename edits."
```

Expected output includes a new docs commit.

- [ ] **Step 4: Push the stabilized WIP branch**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git push origin wip/weka-to-nutanix-migration
```

Expected output shows `wip/weka-to-nutanix-migration` updated on origin.

- [ ] **Step 5: Final status check**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git status -sb
git log --oneline --decorate -3
```

Expected shape:

```text
## wip/weka-to-nutanix-migration...origin/wip/weka-to-nutanix-migration
<surface-map-commit> docs(p3.5): map remaining weka residue
<merge-commit> chore(p3.5): sync nutanix wip with master
192c192 wip(nutanix): checkpoint hard-rename migration
```

## Acceptance Criteria

- The original checkout is on `wip/weka-to-nutanix-migration`.
- The scratch files are backed up under `/tmp/wekadocs-nutanix-scratch-<timestamp>/` and removed from git status.
- `origin/master` is an ancestor of the WIP branch.
- The branch contains a committed `docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md` generated from the raw `rg` scan.
- The surface map separates runtime/config/test blockers from active docs, generated reports, and historical archive candidates.
- The branch is pushed to `origin/wip/weka-to-nutanix-migration`.
- No claim is made that the WEKA->Nutanix migration is complete.

## Self-Review

- Spec coverage: The plan covers scratch classification/removal, master sync into the WIP branch, residue scanning, surface-map creation, verification, commit, and push.
- Placeholder scan: The plan contains no TBD/TODO/fill-in placeholders. Timestamp values are runtime-generated and intentionally variable.
- Type consistency: No new runtime types, functions, or public interfaces are introduced.
