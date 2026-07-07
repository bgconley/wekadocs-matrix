# Finish Live WEKA Residue Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove or explicitly justify all runtime/config/test WEKA residue so the Nutanix migration branch has a clean live-surface residue gate.

**Architecture:** Use the existing residue map as the starting point, but re-derive the active worklist from a fresh scan before editing. Fix the two live runtime/config hits first, then tighten the static guard to include active tests and archive exclusions. Convert active WEKA test fixtures through an iterate-to-green loop driven by the guard, using curated Nutanix replacements for eval/doc fixtures and split legacy strings only for intentional negative assertions.

**Tech Stack:** Python, pytest, ripgrep, GitNexus CLI, existing pre-commit hooks.

---

## File Structure

This slice is intentionally narrower than the full hard rename. Do not start the `mcp_tools.py` split here.

### Live Runtime/Config Hits

- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tools/__init__.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/.env.example`

### Guard And Proof

- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_nutanix_hard_rename_guard.py`
- Create: `/Users/brennanconley/vibecode/wekadocs-matrix/docs/superpowers/findings/2026-07-05-live-weka-residue-completion.md`

### Active Test Residue Targets

The stabilized residue map listed 41 test files, but it is now treated as a baseline, not an execution worklist. Re-derive the worklist from the active scan in Task 1 before editing. Current verification also shows `tests/eval/queries.yaml`, which the old map missed. `tests/eval/` is ignored by `.gitignore`; this plan intentionally treats `tests/eval/queries.yaml` as an active fixture, so scan commands must use `--no-ignore` and the final commit must force-add that file.

- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/conftest.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/e2e/test_golden_set.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/e2e_v22_prod/conftest.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/eval/queries.yaml`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/baseline_query_set.yaml`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/canonical_retrieval_benchmark.yaml`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/doc_with_references_a.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/doc_with_references_b.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/doc_with_references_c.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/doc_with_references_d.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/golden_query_set.yaml`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/fixtures/procedure_with_steps.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/__init__.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/prod_docs_pack_noprefix/README.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json`
- Rename: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md`
- Create by rename: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/prod_docs_pack_noprefix/docs/nutanix-files-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_gliner_config.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_gliner_ingestion_flow.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_gliner_live.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_phase1_entity_edges.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_phase7c_integration.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_precision_retrieval_live.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/integration/test_session_tracking.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/mcp_server_tests/test_nutanix_query_reformulation.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p2_t1_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p2_t3_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p2_t4_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p3_t1_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p3_t2_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p4_t1_complex_patterns_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p6_t1_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p6_t3_test.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/query/test_vector_store.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/shared/test_nutanix_domain.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_embedding_field_canonicalization.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_integration_prephase7.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_phase1_foundation.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_tokenizer_service.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/unit/test_markdown_it_parser.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/unit/test_structural_retrieval.py`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/unit/test_tokenizer_overlap.py`

### Explicit Out Of Scope

- Do not edit `reports/**` in this slice.
- Do not edit `docs/wekadocs50_combined.md` in this slice.
- Do not edit historical/session/audit docs except the new proof report.
- Do not fix unrelated lint debt in `scripts/verify_dead_code.py`, `src/ingestion/atomic.py`, or `src/mcp_server/mcp_app.py`.

## Task 1: Prove Raw Active Residue Is Red And The Existing Guard Is Insufficient

**Files:**
- Read: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_nutanix_hard_rename_guard.py`
- Read: `/Users/brennanconley/vibecode/wekadocs-matrix/.env.example`
- Read: `/Users/brennanconley/vibecode/wekadocs-matrix/tools/__init__.py`

- [ ] **Step 1: Confirm branch and clean status**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git branch --show-current
git status -sb
```

Expected output shape:

```text
wip/weka-to-nutanix-migration
## wip/weka-to-nutanix-migration...origin/wip/weka-to-nutanix-migration
```

- [ ] **Step 2: Run the existing guard and record that it is insufficient**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_nutanix_hard_rename_guard.py -q
```

Expected output today:

```text
1 passed
```

This is not a meaningful green for the migration. The current guard is too narrow: it does not scan `.env.example`, `tools/`, or active tests, and its pattern misses broad case-insensitive legacy terms.

- [ ] **Step 3: Capture the meaningful RED with a raw active-surface scan**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rg -n --hidden --pcre2 \
  --no-ignore \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  --glob '!__pycache__/**' \
  --glob '!node_modules/**' \
  --glob '!reports/**' \
  --glob '!docs/**' \
  '(?i)(weka|wekadocs|WekaDocs)' \
  .env.example tools tests > /tmp/nutanix-live-residue-before.txt || true
wc -l /tmp/nutanix-live-residue-before.txt
head -n 20 /tmp/nutanix-live-residue-before.txt
```

Expected output is greater than zero and includes at least:

```text
.env.example
tools/__init__.py
tests/eval/queries.yaml
```

This is the meaningful RED for the slice because it proves active runtime/config/test residue exists in real checked-in surfaces.

## Task 2: Fix The Two Live Runtime/Config Hits

**Files:**
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/.env.example`
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tools/__init__.py`

- [ ] **Step 1: Edit `.env.example` header**

Change the first heading line from:

```text
# WekaDocs Matrix — Environment Configuration
```

to:

```text
# Nutanix Docs Matrix - Environment Configuration
```

Use ASCII hyphen-minus in the new text.

- [ ] **Step 2: Edit `tools/__init__.py` package comment**

Change:

```python
# Tools package for wekadocs-matrix
```

to:

```python
# Tools package for nutanix-docs-matrix
```

- [ ] **Step 3: Run the existing runtime guard again**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_nutanix_hard_rename_guard.py -q
```

Expected output:

```text
1 passed
```

At this point the old guard should be green because it does not yet scan tests.

- [ ] **Step 4: Run GitNexus change detection before the live-hit commit**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git add .env.example tools/__init__.py
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected acceptable result:

```text
No changes detected.
```

If GitNexus reports symbol or flow changes, inspect and include them in the commit note.

- [ ] **Step 5: Commit the live runtime/config fixes**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit \
  -m "fix(p3.5): remove live weka runtime residue" \
  -m "Phase 3.5 renames the remaining runtime and env header residues."
```

Expected output includes a new commit.

## Task 3: Tighten The Static Guard To Cover Active Tests And Archive Policy

**Files:**
- Modify: `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_nutanix_hard_rename_guard.py`

- [ ] **Step 1: Run GitNexus impact before editing the guard test symbol**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
npx gitnexus impact test_runtime_connected_surfaces_do_not_reference_weka --direction upstream --include-tests --repo wekadocs-matrix || true
```

Record the reported risk and impacted files in the task notes. This is a test-only symbol; a low or no-result impact is expected.

- [ ] **Step 2: Replace the guard file with active-surface policy**

Replace `/Users/brennanconley/vibecode/wekadocs-matrix/tests/test_nutanix_hard_rename_guard.py` with:

```python
"""Static guard for runtime-connected legacy vendor residue."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_SURFACE_PATHS = [
    "src",
    "config",
    "data/ingest/nutanix",
    "docker",
    "deploy",
    "scripts",
    "services",
    "tools",
    ".github",
    "docker-compose.yml",
    "Makefile",
    ".env.example",
    "tests",
]

EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    "reports/",
    "docs/archive/",
    "docs/superpowers/findings/",
    "docs/superpowers/plans/",
    "docs/cdx-outputs/",
    "tests/e2e_v22_prod/artifacts/",
    "repo-analysis-artifacts/",
    "claude-raw/",
)

SKIP_DIR_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
SKIP_SUFFIXES = {".bak", ".pyc", ".pyo", ".so", ".dylib"}
LEGACY_FRAGMENT = "we" + "ka"
LEGACY_PATTERN = re.compile(re.escape(LEGACY_FRAGMENT), re.IGNORECASE)


def _relative_posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _should_scan(path: Path) -> bool:
    rel = _relative_posix(path)
    if rel.startswith("tests/eval/") and rel != "tests/eval/queries.yaml":
        return False
    if SKIP_DIR_PARTS & set(Path(rel).parts):
        return False
    if path.suffix in SKIP_SUFFIXES:
        return False
    return not any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def _iter_active_files() -> list[Path]:
    files: list[Path] = []
    for rel_path in ACTIVE_SURFACE_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        if path.is_file():
            if _should_scan(path):
                files.append(path)
            continue
        files.extend(
            child
            for child in path.rglob("*")
            if child.is_file() and _should_scan(child)
        )
    return sorted(files)


def test_generated_and_historical_paths_are_out_of_scope():
    excluded_examples = [
        PROJECT_ROOT / "reports/phase-7/queries/neo4j-test-results.txt",
        PROJECT_ROOT / "docs/archive/legacy-vendor/example.md",
        PROJECT_ROOT / "docs/superpowers/findings/2026-07-04-weka-residue-surface-map.md",
        PROJECT_ROOT / "docs/superpowers/plans/2026-07-05-finish-live-weka-residue-rename.md",
    ]

    for path in excluded_examples:
        assert _should_scan(path) is False


def test_active_runtime_config_and_tests_reject_old_vendor_residue():
    offenders: list[str] = []

    for file_path in _iter_active_files():
        rel = _relative_posix(file_path)
        if LEGACY_PATTERN.search(rel):
            offenders.append(rel)
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LEGACY_PATTERN.search(text):
            offenders.append(rel)

    assert offenders == []
```

- [ ] **Step 3: Run the tightened guard and capture the second meaningful RED**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_nutanix_hard_rename_guard.py -q
```

Expected: FAIL with active test files listed as offenders. This is the RED for the test/fixture conversion work.

## Task 4: Convert Active Test Fixtures Through Guard-Driven Iteration

**Files:**
- Modify all active test residue targets listed in File Structure.
- Rename the one prod docs fixture file from `weka-filesystems...md` to `nutanix-files-and-object-stores...md`.
- Curate eval and document fixtures instead of relying on mechanical phrase substitution.

- [ ] **Step 1: Rename the prod-docs fixture filename**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git mv \
  tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md \
  tests/integration/prod_docs_pack_noprefix/docs/nutanix-files-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md
```

Expected output is empty.

- [ ] **Step 2: Re-derive the active offender worklist after the guard is tightened**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rg -l --hidden --pcre2 \
  --no-ignore \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  --glob '!__pycache__/**' \
  --glob '!node_modules/**' \
  --glob '!reports/**' \
  --glob '!docs/**' \
  '(?i)(weka|wekadocs|WekaDocs)' \
  .env.example tools tests | sort > /tmp/nutanix-active-offenders.txt
cat /tmp/nutanix-active-offenders.txt
```

Expected output includes `tests/eval/queries.yaml`. Use this file as the live worklist. If a file in File Structure is no longer listed, do not edit it just because the stale map named it.

- [ ] **Step 3: Apply only bounded identifier and infrastructure substitutions**

Use `apply_patch` or careful manual edits for enumerable, non-prose residue. Do not use a broad global `WEKA -> Nutanix` replacement across eval/doc fixtures. Do not mechanically replace absolute local checkout paths that contain `wekadocs-matrix`; those paths should become computed repo roots, not `/Users/.../nutanix-docs-matrix`.

Required bounded substitutions:

```text
wekadocs-matrix -> nutanix-docs-matrix only in package comments, service names, labels, or user-facing project metadata
wekadocs-mcp-test -> nutanix-mcp-test
WekaDocs GraphRAG -> Nutanix Docs GraphRAG
WekaDocs Team -> Nutanix Docs Team
WekaDocs -> Nutanix Docs
weka_sections_v2 -> nutanix_sections_v2
weka_sections -> nutanix_sections
weka-neo4j -> nutanix-neo4j
weka-qdrant -> nutanix-qdrant
weka-redis -> nutanix-redis
weka-mcp-server -> nutanix-mcp-server
weka-ingestion-service -> nutanix-ingestion-service
weka-ingestion-worker -> nutanix-ingestion-worker
WEKA_HOME -> NUTANIX_HOME
WEKA_FS -> NUTANIX_FS
weka-filesystems -> nutanix-files-and-object-stores
```

Expected result: some offenders are removed, but the guard is not expected to be green yet.

In `/Users/brennanconley/vibecode/wekadocs-matrix/tests/p6_t3_test.py`, replace the hardcoded command working directory with a computed repo root. Add this near the imports:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
```

Then change the subprocess call from:

```python
        cwd="/Users/brennanconley/vibecode/wekadocs-matrix",
```

to:

```python
        cwd=str(REPO_ROOT),
```

This removes the legacy checkout-name residue without inventing a nonexistent local path.

- [ ] **Step 4: Curate eval and document fixtures with Nutanix-specific concepts**

Edit these files by domain meaning, not mechanical substitution:

```text
tests/eval/queries.yaml
tests/fixtures/baseline_query_set.yaml
tests/fixtures/canonical_retrieval_benchmark.yaml
tests/fixtures/golden_query_set.yaml
tests/fixtures/doc_with_references_a.md
tests/fixtures/doc_with_references_b.md
tests/fixtures/doc_with_references_c.md
tests/fixtures/doc_with_references_d.md
tests/fixtures/procedure_with_steps.md
tests/integration/prod_docs_pack_noprefix/docs/nutanix-files-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md
```

Use these concrete Nutanix replacements as the content vocabulary:

```text
WEKA snapshots / weka fs snapshot -> Nutanix Files snapshots
WEKA tiering / weka fs tier -> Nutanix Objects lifecycle and object storage tiering
WEKA S3 -> Nutanix Objects S3
WEKA NFS -> Nutanix Files NFS
WEKA SMB -> Nutanix Files SMB
WEKA metrics / WEKAmon -> Prism Central and Nutanix observability
WEKA alerts -> Prism Central alerts
WEKA GUI -> Prism Central
WEKA REST API -> Nutanix v4 APIs
WEKA CLI / weka command -> nCLI or a Prism Central workflow
WEKA data protection -> Nutanix Disaster Recovery, snapshots, and replication
WEKA cluster architecture -> Nutanix Cloud Platform, NCI, AOS, AHV, Prism Central
WEKA hardware sizing -> Nutanix Cloud Platform / NCI sizing
Local WEKA Home -> Nutanix support portal or Nutanix Central, depending on fixture intent
```

For each YAML query fixture, update the related `expected_topics`, `expected_keywords`, `expected_titles`, labels, ids, and rationales in the same edit so the fixture remains semantically coherent. Example shape:

```yaml
- id: nutanix_files_snapshot_create
  text: "How do I create a snapshot for a Nutanix Files share?"
  expected_topics: ["Nutanix Files", "snapshot", "data protection"]
```

- [ ] **Step 5: Repair intentional negative assertions without literal legacy residue**

In `/Users/brennanconley/vibecode/wekadocs-matrix/tests/shared/test_nutanix_domain.py`, replace the final assertions in `test_domain_config_prompts_are_nutanix_only` with:

```python
    legacy_upper = "W" + "EKA"
    legacy_lower = "we" + "ka"
    assert legacy_upper not in prompt_text
    assert legacy_lower not in prompt_text
```

In `/Users/brennanconley/vibecode/wekadocs-matrix/tests/mcp_server_tests/test_nutanix_query_reformulation.py`, replace the final assertions with:

```python
    legacy_upper = "W" + "EKA"
    legacy_lower = "we" + "ka"
    assert legacy_upper not in rewritten
    assert legacy_lower not in rewritten
```

These tests still prove the behavior, but they do not keep direct legacy strings in active test files.

- [ ] **Step 6: Repair renamed expectations that should be domain-specific**

Inspect the diff:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git diff -- tests/shared/test_nutanix_domain.py tests/mcp_server_tests/test_nutanix_query_reformulation.py tests/integration/test_gliner_config.py tests/p3_t2_test.py
```

Ensure the following assertions are true after edits:

```python
# tests/integration/test_gliner_config.py
assert any("nutanix" in label.lower() for label in labels)

# tests/p3_t2_test.py
nutanix_commands = [c for c in command_names if "nutanix" in c.lower() or "ncli" in c.lower()]
assert len(nutanix_commands) > 0
assert any("nutanix.conf" in name for name in config_names)
assert "NUTANIX_HOME" in config_names or "MAX_MEMORY" in config_names
```

Edit the files manually to match these assertions. Do not rely on the bounded substitutions from Step 3 to produce these exact shapes.

- [ ] **Step 7: Run the guard and iterate until GREEN**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/test_nutanix_hard_rename_guard.py -q
```

Expected on the first run after curated edits may still be FAIL with a concrete offender list. For each offender:

1. Open the listed file.
2. Decide whether the residue is an unintentional active legacy term or an intentional negative assertion.
3. Replace unintentional residue with Nutanix-specific content.
4. Split intentional negative assertion strings as shown in Step 5.
5. Rerun the guard.

Repeat until output is:

```text
2 passed
```

Do not proceed to Task 5 until the tightened guard is green.

## Task 5: Prove Active Runtime/Config/Test Blockers Are Zero

**Files:**
- Create: `/Users/brennanconley/vibecode/wekadocs-matrix/docs/superpowers/findings/2026-07-05-live-weka-residue-completion.md`

- [ ] **Step 1: Run active-surface residue scan**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
SCAN_DIR=/tmp/nutanix-live-residue-20260705
mkdir -p "$SCAN_DIR"
rg -n --hidden --pcre2 \
  --no-ignore \
  --glob '!.git/**' \
  --glob '!**/.venv/**' \
  --glob '!**/venv/**' \
  --glob '!**/__pycache__/**' \
  --glob '!**/node_modules/**' \
  --glob '!*.bak' \
  --glob '!*.pyc' \
  --glob '!*.pyo' \
  --glob '!*.so' \
  --glob '!*.dylib' \
  --glob '!reports/**' \
  --glob '!docs/**' \
  --glob '!tests/e2e_v22_prod/artifacts/**' \
  --glob '!tests/eval/**' \
  '(?i)(weka|wekadocs|WekaDocs)' \
  src config data/ingest/nutanix docker deploy scripts services tools .github docker-compose.yml Makefile .env.example tests \
  > "$SCAN_DIR/active-rg.txt" || true
rg -n --hidden --no-ignore --pcre2 '(?i)(weka|wekadocs|WekaDocs)' tests/eval/queries.yaml >> "$SCAN_DIR/active-rg.txt" || true
wc -l "$SCAN_DIR/active-rg.txt"
```

Expected output:

```text
0 /tmp/nutanix-live-residue-20260705/active-rg.txt
```

- [ ] **Step 2: Create proof report**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
cat > docs/superpowers/findings/2026-07-05-live-weka-residue-completion.md <<'EOF'
# Live WEKA Residue Completion - 2026-07-05

## Scope

This report covers runtime, config, deploy, and active test surfaces only.
Generated reports and historical docs are excluded from the live residue gate.

## Result

- Active runtime/config/test blocker hits: 0
- Runtime/config live hits fixed:
  - `.env.example`
  - `tools/__init__.py`
- Active tests and fixtures converted to Nutanix examples.
- Eval and document fixtures were curated to Nutanix-specific concepts rather than mechanically renamed.
- Intentional negative assertions now assemble legacy strings at runtime so the tests keep proving the invariant without polluting active residue scans.
- Edited live-only tests are listed below as deferred from runtime execution.

## Curated Fixture Families

- Nutanix Files snapshots and data protection
- Nutanix Files NFS and SMB
- Nutanix Objects S3 and object lifecycle/tiering
- Prism Central alerts, monitoring, and UI workflows
- Nutanix v4 APIs and nCLI examples
- Nutanix Cloud Platform, NCI, AOS, AHV, and Prism Central architecture

## Edited Live-Only Surfaces Deferred From Runtime Execution

- `tests/e2e_v22_prod/conftest.py`
- `tests/integration/test_gliner_live.py`
- `tests/integration/test_precision_retrieval_live.py`
- `tests/integration/test_phase7c_integration.py`
- `tests/integration/prod_docs_pack_noprefix/README.md`
- `tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json`
- `tests/integration/prod_docs_pack_noprefix/docs/nutanix-files-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md`

## Excluded By Policy

- `reports/**`
- `docs/archive/**`
- `docs/superpowers/findings/**`
- `docs/superpowers/plans/**`
- `docs/cdx-outputs/**`
- `repo-analysis-artifacts/**`
- `claude-raw/**`
- `tests/e2e_v22_prod/artifacts/**`
- All files under `tests/eval/**` except `tests/eval/queries.yaml`
- Nested dependency/cache directories named `.git`, `.venv`, `venv`, `node_modules`, or `__pycache__`
- Backup/binary suffixes `.bak`, `.pyc`, `.pyo`, `.so`, and `.dylib`

## Verification

```bash
pytest tests/test_nutanix_hard_rename_guard.py -q
SCAN_DIR=/tmp/nutanix-live-residue-20260705
: > "$SCAN_DIR/active-rg.txt"
rg -n --hidden --no-ignore --pcre2 --glob '!.git/**' --glob '!**/.venv/**' --glob '!**/venv/**' --glob '!**/__pycache__/**' --glob '!**/node_modules/**' --glob '!*.bak' --glob '!*.pyc' --glob '!*.pyo' --glob '!*.so' --glob '!*.dylib' --glob '!reports/**' --glob '!docs/**' --glob '!tests/e2e_v22_prod/artifacts/**' --glob '!tests/eval/**' '(?i)(weka|wekadocs|WekaDocs)' src config data/ingest/nutanix docker deploy scripts services tools .github docker-compose.yml Makefile .env.example tests >> "$SCAN_DIR/active-rg.txt" || true
rg -n --hidden --no-ignore --pcre2 '(?i)(weka|wekadocs|WekaDocs)' tests/eval/queries.yaml >> "$SCAN_DIR/active-rg.txt" || true
wc -l "$SCAN_DIR/active-rg.txt"
```

The active-surface `rg` command returned zero lines.

## Remaining Work Outside This Slice

- Decide whether large historical/generated WEKA docs should be moved under `docs/archive/**`.
- Keep `reports/**` out of live residue gates.
- Handle unrelated lint cleanup separately.
- Run live-only tests after embedders, rerankers, Qdrant, Neo4j, and GLiNER services are available.
EOF
```

Expected output is empty.

## Task 6: Run Focused Tests And Collection Checks

**Files:**
- Read test files modified in Task 4.

- [ ] **Step 1: Run the guard and focused non-live tests**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest \
  tests/test_nutanix_hard_rename_guard.py \
  tests/shared/test_nutanix_domain.py \
  tests/mcp_server_tests/test_nutanix_query_reformulation.py \
  tests/integration/test_gliner_config.py \
  tests/query/test_vector_store.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Validate edited YAML fixtures**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python - <<'PY'
from pathlib import Path
import yaml

paths = [
    Path("tests/eval/queries.yaml"),
    Path("tests/fixtures/baseline_query_set.yaml"),
    Path("tests/fixtures/canonical_retrieval_benchmark.yaml"),
    Path("tests/fixtures/golden_query_set.yaml"),
]

for path in paths:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data is not None, f"{path} parsed to empty data"
    print(f"parsed {path}")
PY
```

Expected output:

```text
parsed tests/eval/queries.yaml
parsed tests/fixtures/baseline_query_set.yaml
parsed tests/fixtures/canonical_retrieval_benchmark.yaml
parsed tests/fixtures/golden_query_set.yaml
```

- [ ] **Step 3: Run collection over edited non-live tests**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest --collect-only -q \
  tests/test_nutanix_hard_rename_guard.py \
  tests/shared/test_nutanix_domain.py \
  tests/mcp_server_tests/test_nutanix_query_reformulation.py \
  tests/integration/test_gliner_config.py \
  tests/query/test_vector_store.py \
  tests/test_embedding_field_canonicalization.py \
  tests/test_integration_prephase7.py \
  tests/test_phase1_foundation.py \
  tests/test_tokenizer_service.py \
  tests/unit/test_markdown_it_parser.py \
  tests/unit/test_structural_retrieval.py \
  tests/unit/test_tokenizer_overlap.py \
  tests/p2_t1_test.py \
  tests/p2_t3_test.py \
  tests/p2_t4_test.py \
  tests/p3_t1_test.py \
  tests/p3_t2_test.py \
  tests/p4_t1_complex_patterns_test.py \
  tests/p6_t1_test.py \
  tests/p6_t3_test.py
```

Expected: collection succeeds. If collection fails from an edited assertion, fixture, or import shape, fix that before continuing. If collection fails from a pre-existing external-service requirement, record the specific module and reason in the proof report.

- [ ] **Step 4: Run syntax and touched-file lint checks**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python -m compileall -q src scripts tests
ruff check \
  tests/test_nutanix_hard_rename_guard.py \
  tests/shared/test_nutanix_domain.py \
  tests/mcp_server_tests/test_nutanix_query_reformulation.py \
  tests/integration/test_gliner_config.py \
  tests/query/test_vector_store.py
```

Expected output includes:

```text
All checks passed!
```

`compileall` is silent on success.

- [ ] **Step 5: Record edited live-only tests as deferred**

Do not run tests requiring live embedders, rerankers, Qdrant, Neo4j, or GLiNER services unless the user explicitly says those services are available.

Record these edited live-only surfaces in the proof report as static/collection-only for this slice:

```text
tests/e2e_v22_prod/conftest.py
tests/integration/test_gliner_live.py
tests/integration/test_precision_retrieval_live.py
tests/integration/test_phase7c_integration.py
tests/integration/prod_docs_pack_noprefix/README.md
tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json
tests/integration/prod_docs_pack_noprefix/docs/nutanix-files-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md
```

Do not claim live model/datastore quality from this slice.

## Task 7: Commit And Push

**Files:**
- Commit all modified runtime/config/test/proof files from this plan.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git status -sb
git diff --stat
```

Expected: only files from this plan are modified/renamed/created.

- [ ] **Step 2: Run GitNexus change detection before committing**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git add .env.example tools/__init__.py tests docs/superpowers/findings/2026-07-05-live-weka-residue-completion.md
git add -f tests/eval/queries.yaml
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected: GitNexus reports no runtime execution-flow changes, or reports only test/documentation symbol changes. If it reports HIGH or CRITICAL runtime risk, stop and report before committing.

- [ ] **Step 3: Commit**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git commit \
  -m "fix(p3.5): finish live nutanix residue rename" \
  -m "Phase 3.5 removes live runtime, config, and active test residue.\n\nGenerated and historical artifacts remain outside the live residue gate."
```

Expected output includes a new commit.

- [ ] **Step 4: Push**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git push origin wip/weka-to-nutanix-migration
```

Expected output shows `wip/weka-to-nutanix-migration` updated on origin.

- [ ] **Step 5: Final status**

Run:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
git status -sb
git log --oneline --decorate -4
```

Expected shape:

```text
## wip/weka-to-nutanix-migration...origin/wip/weka-to-nutanix-migration
<new-commit> fix(p3.5): finish live nutanix residue rename
12298f8 docs(p3.5): map remaining weka residue
121ff1b chore(p3.5): sync nutanix migration with master
```

## Acceptance Criteria

- `.env.example` contains Nutanix wording and no live legacy vendor residue.
- `tools/__init__.py` contains Nutanix wording and no live legacy vendor residue.
- The active target list is re-derived from a fresh scan after the guard is tightened.
- Active test files no longer contain direct legacy vendor residue.
- Ignored active fixture `tests/eval/queries.yaml` is scanned with `--no-ignore` and force-added if it remains part of the active test surface.
- Eval and document fixtures are curated to coherent Nutanix topics, keywords, titles, ids, and rationales.
- Negative assertion tests still prove Nutanix prompts/reformulation do not emit legacy vendor terms, using runtime-assembled strings.
- `tests/test_nutanix_hard_rename_guard.py` scans runtime, config, deploy, tools, scripts, and active tests.
- The guard excludes generated/historical artifacts by path policy.
- The guard-driven iterate-to-green loop has completed with `2 passed`.
- Active-surface residue scan over the same runtime/config/deploy/tools/test surface as the guard returns zero lines.
- Edited non-live tests collect successfully, or any collection exceptions are explicitly documented as pre-existing external-service requirements.
- Edited live-only tests are listed as deferred in the proof report.
- A proof report exists at `docs/superpowers/findings/2026-07-05-live-weka-residue-completion.md`.
- No live model/datastore quality claims are made.

## Self-Review

- Spec coverage: Tasks cover the 2 live hits, fresh active worklist derivation, curated active test conversion, tightened guard behavior, archive/generated exclusions, final scan proof, collection checks, live-only deferrals, commit, and push.
- Placeholder scan: No TBD/TODO/fill-in placeholders remain. All commands and expected outputs are concrete.
- Type consistency: The guard helper names `_should_scan`, `_iter_active_files`, and `LEGACY_PATTERN` are used consistently in the plan.
