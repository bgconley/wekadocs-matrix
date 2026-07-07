# Quarantine Legacy Modules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> This is **Slice B** of the Refactor-Roadmap wrap-up (`docs/repo_audit/10_refactor_roadmap.md`, Phase 4c: "Delete or quarantine confirmed legacy modules after fresh tests prove no active imports"). Run it **after Slice A** and **before Slice C** (it removes the `patched/` orphan that otherwise confuses GitNexus for the AtomicIngestionCoordinator decomposition).

**Goal:** Delete the confirmed-dead `patched/` shadow directory and correct stale `@status`/`@called-by` annotations on the five active modules — leaving an enforceable "no legacy imports" guard.

**Architecture:** Investigation (from two independent scoping passes) already proved the five `@status`-flagged modules are all **ACTIVE** with real importers, so **none of them are deleted**. The genuinely dead code is `patched/atomic_patched_v3.py`, `patched/chonkie_adapter_patched.py`, `patched/semantic_chunker_patched.py` — orphaned Dec-2025 hotfix copies with zero importers in `src/`, `tests/`, `scripts/`, or `deploy/`. This slice removes those, fixes misleading annotations (docs-only), and adds a small guard test so the win is durable.

**Tech Stack:** Git, ripgrep, pytest, GitNexus.

**Execution note:** Create/overwrite files with the Write tool and edit with Edit (apply_patch-style), not shell heredoc redirection. **Commit messages must pass gitlint** (enforced at `commit-msg`): title ≤72 chars with a `(pN.N)` numeric scope, a blank line, then a `Phase N.N — <recap>` body whose lines are ≤80 chars. `git commit` examples are shown title-first — append the `-m "Phase N.N — …"` body when committing.

---

## Critical Context

- **The GitNexus index is stale** (indexed at an old commit; HEAD is ahead). It under-reports importers for classes/variables (returned LOW/0 for `TokenizerService`, `CrossDocLinker`, `RELATIONSHIP_TYPES` despite many real importers). **Task 1 records the stale state; Slice C performs the authoritative re-index after this slice deletes `patched/`.** Do not trust `gitnexus impact` counts until then; the ground truth in this slice is ripgrep.
- **All five `@status` modules are ACTIVE — do NOT delete them or their tests:** `src/services/cross_doc_linking.py` (imported at `atomic.py:125`, `neo4j_writers.py`, + scripts), `src/providers/tokenizer_service.py` (13 src importers across query/ingestion/mcp/providers), `src/neo/schema.py` (via `graph_service.py`), `src/shared/qdrant_schema.py` (via `hybrid_retrieval.py`), `src/ingestion/atomic.py` (via `worker.py` — it's Slice C's target).
- **The `patched/` files are provenance-checked orphans.** Confirm no deploy/runbook copies them before deletion (Task 2 Step 1).

---

## File Structure

- Delete: `patched/atomic_patched_v3.py`, `patched/chonkie_adapter_patched.py`, `patched/semantic_chunker_patched.py` (and the now-empty `patched/` dir)
- Modify (annotation comments only — no code change): `src/neo/schema.py`, `src/shared/qdrant_schema.py`, `src/providers/tokenizer_service.py`, `src/services/cross_doc_linking.py`
- Create: `tests/test_no_legacy_shadow_modules.py` (durable guard)

### Explicit Out Of Scope

- Do not delete or modify the runtime behavior of the five active modules (annotation comments only).
- Do not touch `src/ingestion/atomic.py` code (Slice C owns it).
- The `build_qdrant_schema` "function has no non-test caller" observation is flagged for the decomposition phase, not acted on here.

---

## Task 1: Record GitNexus State (re-index is deferred to Slice C)

**Files:** none.

- [ ] **Step 1: Note the staleness — do NOT re-index in this slice**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
npx gitnexus status            # expect: ⚠️ stale — record it in the task log
```

Slice B's edits (deleting `patched/`, fixing comments) are **mechanical and not GitNexus-impact-gated**, so a fresh index is not needed here — and any re-index now would just be re-staled by this slice's own Task 2/3/5 commits (the exact muddiness to avoid). The **authoritative re-index is Slice C's prerequisite**, run *after* this slice removes `patched/`, so it resolves `AtomicIngestionCoordinator` without the shadow-copy collision.

---

## Task 2: Delete The Dead `patched/` Shadow Copies

**Files:** `patched/*.py`

- [ ] **Step 1: Prove zero references (the Phase 4c precondition)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rg -n "from patched|import patched|atomic_patched|chonkie_adapter_patched|semantic_chunker_patched" \
   src tests scripts deploy Makefile docker-compose.yml 2>/dev/null || echo "no references anywhere ✓"
git log --oneline -- patched/ | head -3
```

Expected: `no references anywhere ✓`. If ANY reference appears (especially in `deploy/` or `scripts/`), stop and report it — do not delete.

- [ ] **Step 2: Baseline the tests that cover the REAL (`src/`) versions**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/ -k "atomic or chonkie or semantic_chunker" -q -p no:cacheprovider 2>&1 | tail -3
```

Expected: these pass (they test `src/ingestion/*`, unaffected by removing `patched/`). Record the pass count as the baseline.

- [ ] **Step 3: Remove the orphans**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git rm patched/atomic_patched_v3.py patched/chonkie_adapter_patched.py patched/semantic_chunker_patched.py
rmdir patched 2>/dev/null || true
```

Expected: three `rm` lines. Content is preserved in git history.

- [ ] **Step 4: Re-run the baseline tests + a fresh residue sweep**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/ -k "atomic or chonkie or semantic_chunker" -q -p no:cacheprovider 2>&1 | tail -2
rg -n "patched/" src tests scripts deploy 2>/dev/null || echo "no dangling patched/ references ✓"
```

Expected: same pass count as Step 2; no dangling references.

- [ ] **Step 5: Commit (patched-only)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit -m "chore(p4.0): remove orphaned patched/ shadow modules" \
  -m "Phase 4.0 — remove dead patched/ shadow copies (0 importers in src/tests)."
```

---

## Task 3: Correct Stale Module Annotations (docs-only)

**Files:** `src/neo/schema.py`, `src/shared/qdrant_schema.py`, `src/providers/tokenizer_service.py`, `src/services/cross_doc_linking.py`

- [ ] **Step 1: Fix each `@called-by` header to match reality**

These are comment-only edits (no code change). Update the stale header lines:

- `src/neo/schema.py`: `@called-by: graph_service.py, explain_guard.py` → `@called-by: services/graph_service.py (feeds ExplainGuard rel-type allow-list)`. (`explain_guard.py` is not a file.)
- `src/shared/qdrant_schema.py`: `@called-by: build_graph.py, hybrid_retrieval.py` → `@called-by: query/hybrid_retrieval.py (validate_qdrant_schema), scripts/validate_profile_storage.py`. (Drop the stale `build_graph.py`.)
- `src/providers/tokenizer_service.py`: `@called-by: hybrid_retrieval.py, atomic.py` → `@called-by: widely used (13+ importers across query/, ingestion/, mcp_server/, providers/) — see importer graph`.
- `src/services/cross_doc_linking.py`: append `ingestion/neo4j_writers.py, scripts/backfill_cross_doc_edges.py, scripts/batch_crossdoc_link.py` to `@called-by`.

- [ ] **Step 2: Sanity — no behavior changed**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git diff --stat -- src/neo/schema.py src/shared/qdrant_schema.py src/providers/tokenizer_service.py src/services/cross_doc_linking.py
python -m compileall -q src/neo src/shared/qdrant_schema.py src/providers/tokenizer_service.py src/services/cross_doc_linking.py
```

Expected: only comment lines changed; compile clean.

---

## Task 4: Add A Durable "No Legacy Shadow Modules" Guard

**Files:** Create `tests/test_no_legacy_shadow_modules.py`

The roadmap's Phase 4c precondition ("fresh tests prove no active imports") has no existing test. Add one so the `patched/` win can't regress.

- [ ] **Step 1: Write the guard**

```python
# tests/test_no_legacy_shadow_modules.py
"""Guard: no orphaned shadow-module copies re-appear, and nothing imports them."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The real contract: no active surface imports OR otherwise depends on the shadow
# copies — including non-Python references (e.g. a deploy script that `cp`s them).
FORBIDDEN = re.compile(
    r"(from|import)\s+patched\b|atomic_patched|chonkie_adapter_patched|semantic_chunker_patched|(^|[^A-Za-z0-9_])patched/"
)
SCAN_DIRS = ["src", "tests", "scripts", "deploy"]
SCAN_ROOT_FILES = ["Makefile", "docker-compose.yml", "pyproject.toml"]
SCAN_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".txt", ".env", ""}  # "" catches Makefile/Dockerfile


def _iter_active_files():
    guard = Path(__file__).resolve()               # skip this guard's own file (it names the tokens)
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in SCAN_SUFFIXES and p.resolve() != guard:
                yield p
    for name in SCAN_ROOT_FILES:
        p = ROOT / name
        if p.is_file():
            yield p


def test_no_patched_shadow_directory():
    assert not (ROOT / "patched").exists(), "the orphaned patched/ shadow dir must stay deleted"


def test_nothing_references_shadow_modules():
    offenders = [
        str(p.relative_to(ROOT))
        for p in _iter_active_files()
        if FORBIDDEN.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == [], f"legacy shadow-module references found: {offenders}"
```

- [ ] **Step 2: Run it**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_no_legacy_shadow_modules.py -q
```

Expected: `2 passed`.

---

## Task 5: Verify, Commit, Push

- [ ] **Step 1: Focused verification**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python -m compileall -q src tests
pytest tests/test_no_legacy_shadow_modules.py tests/test_nutanix_hard_rename_guard.py -q
git add src/neo/schema.py src/shared/qdrant_schema.py src/providers/tokenizer_service.py \
        src/services/cross_doc_linking.py tests/test_no_legacy_shadow_modules.py
# (the patched/ deletion was already committed by Task 2; do NOT `git add -A` —
#  the four untracked wrap-up plan files must not be swept into this commit)
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected: compile clean; guards `2 passed` + `2 passed`; detect-changes shows only the annotation/doc/test changes (no runtime symbol/flow changes — the deleted `patched/` was never in the graph as an active node).

- [ ] **Step 2: Commit the annotations + guard**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit -m "chore(p4.0): fix @called-by notes + add legacy-import guard" \
  -m "Phase 4.0 — annotation hygiene + guard against patched/ reimport."
git push origin wip/weka-to-nutanix-migration
```

Expected: pushed and in sync.

---

## Acceptance Criteria

- `patched/` is deleted (3 files) with zero remaining references; git history preserves them.
- The five active modules are **unchanged in behavior** (annotation comments only).
- `tests/test_no_legacy_shadow_modules.py` passes and prevents regression.
- `patched/` is removed so Slice C's prerequisite GitNexus re-index resolves `AtomicIngestionCoordinator` without the shadow-copy collision. (Slice B itself does **not** re-index — its edits are not impact-gated.)
- Branch pushed.

## Self-Review

- Spec coverage: GitNexus stale-state recording/deferred re-index, orphan deletion (verify→baseline→rm→re-verify→commit), annotation hygiene, durable guard, verification.
- Correctness: no active module or test is removed; only proven-dead orphans.
- Sequencing: this slice removes the `patched/` GitNexus collision that Slice C depends on.
- Placeholder scan: concrete files/commands throughout.
