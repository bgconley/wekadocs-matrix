# Codebase Pruning and Modernization Plan

**Date:** 2026-03-01
**Branch:** `multi-embedder-reranker`
**Status:** Phase A+B COMPLETE, pre-Phase-C review
**Author:** Architecture review session
**Last updated:** 2026-03-02

---

## Executive Summary

A comprehensive architecture review revealed that the codebase has undergone a significant but incomplete migration from Phase 3's monolithic `GraphBuilder` to Phase 7E's saga-coordinated `AtomicIngestionCoordinator`. The result is:

- **~7100 lines** of combined ingestion code (`build_graph.py` + `atomic.py`) where only ~60% is active
- **16 phantom modules** loaded at every worker startup via `__init__.py` side effects → **reduced to 1** after Phase B
- **21 fully dead modules** with zero active callers → **35 dead** after reclassifying former phantoms
- **~3500 lines** of dead methods within partially-active modules

This plan documents every change needed, organized into phases that can be executed independently.

---

## Table of Contents

1. [Sequencing](#1-sequencing)
2. [Phase A: Annotate](#2-phase-a-annotate)
3. [Phase B: Clean __init__.py Phantom Chains](#3-phase-b-clean-initpy-phantom-chains)
4. [Phase A+B Completion Report](#4-phase-ab-completion-report)
5. [Code Review Findings](#5-code-review-findings)
6. [Phase B.6: Pre-Phase-C Prerequisites](#6-phase-b6-pre-phase-c-prerequisites)
7. [Phase C: Pull Forward Active Logic](#7-phase-c-pull-forward-active-logic)
8. [Phase D: Prune Dead Modules](#8-phase-d-prune-dead-modules)
9. [Phase E: Prune Dead Methods in Active Modules](#9-phase-e-prune-dead-methods-in-active-modules)
10. [Phase F: Retire Dead Tests](#10-phase-f-retire-dead-tests)
11. [Risk Assessment](#11-risk-assessment)
12. [Validation Checklist](#12-validation-checklist)

---

## 1. Sequencing

```
Phase A: Annotate          ✓ DONE (2026-03-01)
  213 @status annotations across all src/ Python files.
  100% module-level coverage.

Phase B: Clean __init__.py phantom chains  ✓ DONE (2026-03-01)
  Removed eager imports from 4 __init__.py files.
  Eliminated 13 of 14 phantom module loads at startup.
  1 remaining: shadow_comparison.py via parsers/__init__.py.

Phase B.6: Resolve remaining issues  ← DO THIS BEFORE PROCEEDING
  See "Code Review Findings" section below.

Phase C: Pull forward active logic   ← NEXT
  Extract active methods from legacy modules into clean new modules.
  Highest value, moderate risk.

Phase D: Prune dead modules
  Delete entire files/directories with zero active callers.
  Low risk — everything was annotated and verified first.

Phase E: Prune dead methods in active modules
  Remove dead methods from build_graph.py, saga.py, etc.
  Moderate risk — must preserve active methods in same file.

Phase F: Retire dead tests
  Remove or skip tests that exercise dead code paths.
  19 test files currently import from DEAD modules.
```

---

## 2. Phase A: Annotate

**Goal:** Add `@status` annotations to every module, class, and method so that any reviewer (human or agent) can instantly see what is active, dead, phantom, or dormant.

**Convention:**
- Module level: `# @status:` header block before docstring
- Class level: `# @status:` comment above class definition (only when status differs from module)
- Method level: `# @status:` comment above method (only when status differs from containing class/module)
- Active symbols in active modules get NO annotation (annotate deviations only)

**Tags:**
| Tag | Meaning |
|-----|---------|
| `@status` | ACTIVE / DEAD / PHANTOM / DORMANT / MIXED / STANDALONE / PHANTOM-SOURCE |
| `@reason` | Why this status |
| `@superseded-by` | What replaced this code |
| `@called-by` | Primary production caller |
| `@loaded-via` | The `__init__.py` chain that loads it |
| `@gated-by` | Config key that enables it |
| `@pull-forward` | What active logic needs extraction |
| `@safe-to-delete` | Explicit go/no-go for deletion |

**Files to annotate (complete list):**

### Dead modules (add module-level @status: DEAD)
| File | @safe-to-delete |
|------|-----------------|
| `src/learning/feedback.py` | Yes |
| `src/learning/ranking_tuner.py` | Yes |
| `src/learning/suggestions.py` | Yes |
| `src/learning/__init__.py` | Yes |
| `src/ops/optimizer.py` | Yes |
| `src/ops/session_cleanup_job.py` | Yes |
| `src/ops/warmers/query_warmer.py` | Yes |
| `src/ops/warmers/__init__.py` | Yes |
| `src/registry/index_registry.py` | Yes |
| `src/registry/__init__.py` | Yes |
| `src/query/diffusion_reranker.py` | Yes |
| `src/query/graph_expansion.py` | Yes |
| `src/query/graph_features.py` | Yes |
| `src/neo/entity_normalization.py` | Yes |
| `src/neo/graph_enhancements.py` | Yes |
| `src/neo/structural_builder.py` | Yes |
| `src/mcp_server/validation.py` | Yes |
| `src/mcp_server/security/__init__.py` | Yes |
| `src/mcp_server/security/auth.py` | Yes |
| `src/mcp_server/security/rate_limiter.py` | Yes |
| `src/ingestion/auto/watcher.py` | Yes |
| `src/shared/feature_flags.py` | Yes |
| `src/ingestion/incremental.py` | Yes (after __init__ cleanup) |
| `src/ingestion/api.py` | Yes (after __init__ cleanup; move to tests/) |
| `src/ingestion/parsers/notion.py` | Yes (after __init__ cleanup) |

### Phantom modules (add module-level @status: PHANTOM)
| File | @loaded-via |
|------|-------------|
| `src/ingestion/auto/orchestrator.py` | auto/__init__.py |
| `src/ingestion/auto/backpressure.py` | auto/__init__.py |
| `src/ingestion/auto/progress.py` | auto/__init__.py |
| `src/ingestion/auto/report.py` | orchestrator.py → auto/__init__.py |
| `src/ingestion/auto/verification.py` | orchestrator.py → auto/__init__.py |
| `src/ingestion/reconcile.py` | orchestrator.py → auto/__init__.py |
| `src/neo/explain_guard.py` | neo/__init__.py |
| `src/neo/defensive_query.py` | neo/__init__.py |
| `src/neo/health.py` | neo/__init__.py |
| `src/shared/audit/logger.py` | shared/__init__.py |
| `src/ingestion/parsers/shadow_comparison.py` | parsers/__init__.py |

### Phantom source files (add module-level @status: PHANTOM-SOURCE)
| File | @impact |
|------|---------|
| `src/ingestion/auto/__init__.py` | Loads 8 phantom modules on every worker start |
| `src/ingestion/__init__.py` | Loads api.py → build_graph.py early |
| `src/neo/__init__.py` | Loads explain_guard, defensive_query, health |
| `src/shared/__init__.py` | Loads audit/logger.py |

### Dormant modules (add module-level @status: DORMANT)
| File | @gated-by |
|------|-----------|
| `src/ingestion/parsers/markdown.py` | config.ingestion.parser.engine == "legacy" OR markdown-it-py import failure |
| `src/ingestion/extract/ner_gliner.py` | config.ner.enabled |

### Mixed modules (add module-level @status: MIXED + class/method annotations)
| File | Active symbols | Dead symbols |
|------|---------------|-------------|
| `src/ingestion/build_graph.py` | `__init__`, `ensure_embedder`, `_build_section_text_for_embedding`, `_build_title_text_for_embedding` + 3 attributes | 39 methods, `ingest_document()` module-level |
| `src/ingestion/saga.py` | `SagaContext`, `ValidationResult`, `IngestionValidator` | `SagaCoordinator`, `IngestionSagaBuilder`, `SagaStep`, `SagaStepResult`, `SagaStatus`, `StepStatus`, `SagaStepFailure`, `SagaCompensationFailure` |
| `src/neo/contract_checks.py` | `__init__`, `find_documents_needing_repair` | `run_all_checks`, `run_contract_checks`, 6 individual check methods |
| `src/shared/connections.py` | Everything except 2 methods | `purge_document`, `create_collection_with_dims` |

### Standalone modules (add module-level @status: STANDALONE)
| File | Purpose |
|------|---------|
| `src/mcp_server/stdio_server.py` | stdio MCP transport for Claude Desktop direct connection |
| `src/ingestion/auto/cli.py` | `ingestctl` operator CLI for queue management |

### Active modules (add module-level @status: ACTIVE)
All remaining modules in src/ — the module header confirms active status.
For fully active modules, the header is brief:
```python
# =============================================================================
# @status: ACTIVE
# @called-by: <primary caller>
# =============================================================================
```

---

## 3. Phase B: Clean __init__.py Phantom Chains

**Goal:** Eliminate 14+ phantom module loads at startup by removing eager imports from 4 `__init__.py` files.

### B.1: `src/ingestion/auto/__init__.py`

**Current (loads 8 phantom modules):**
```python
from .backpressure import BackPressureMonitor
from .orchestrator import JobState, Orchestrator
from .progress import JobStage, ProgressEvent, ProgressReader, ProgressTracker
```

**After (zero phantom loads):**
```python
# Removed eager imports of Orchestrator, BackPressureMonitor, ProgressTracker.
# These were loaded at every worker startup but never instantiated.
# Consumers that need them can import directly:
#   from src.ingestion.auto.orchestrator import Orchestrator
#   from src.ingestion.auto.backpressure import BackPressureMonitor
#   from src.ingestion.auto.progress import ProgressTracker

__version__ = "0.1.0"
```

**Impact:** Eliminates loading of orchestrator.py, backpressure.py, progress.py, report.py, verification.py, incremental.py, reconcile.py, notion.py at worker startup.

**Risk:** LOW — grep confirmed no active code does `from src.ingestion.auto import Orchestrator`. Tests that use these imports would need updating to import directly.

### B.2: `src/ingestion/__init__.py`

**Current (loads api.py → build_graph.py early):**
```python
from .api import ingest_document

__all__ = ["ingest_document"]
```

**After:**
```python
# Removed eager import of api.py (test facade).
# Tests that need ingest_document should import directly:
#   from src.ingestion.api import ingest_document
```

**Risk:** LOW — `api.py` is a test facade. Tests need updating to import directly.

### B.3: `src/neo/__init__.py`

**Current (loads explain_guard, defensive_query, health):**
```python
from .defensive_query import run_defensive_query, run_existence_check
from .explain_guard import ExplainGuard, PlanRejected, PlanTooExpensive, validate_query_plan
from .health import Neo4jHealthStatus, check_neo4j_connectivity, check_neo4j_health
from .schema_validator import SchemaValidationResult, validate_neo4j_schema
```

**After:**
```python
# Keep only symbols that are actually imported by active code:
from .schema_validator import SchemaValidationResult, validate_neo4j_schema

# Removed re-exports that were never used from outside neo/:
#   ExplainGuard, PlanRejected, PlanTooExpensive, validate_query_plan
#   run_defensive_query, run_existence_check
#   Neo4jHealthStatus, check_neo4j_health, check_neo4j_connectivity
# Import directly if needed: from src.neo.explain_guard import ExplainGuard
```

**Risk:** LOW — grep confirmed no active code imports these symbols from `src.neo`. Tests may need updating.

### B.4: `src/shared/__init__.py`

**Current (loads audit/logger.py):**
```python
from .audit import AuditLogger, get_audit_logger
from .config import Config, Settings, get_config, get_settings, init_config
from .connections import ConnectionManager, close_connections, ...
```

**After:**
```python
# Removed AuditLogger/get_audit_logger (never called by any active code)
from .config import Config, Settings, get_config, get_settings, init_config
from .connections import ConnectionManager, close_connections, ...
```

**Risk:** LOW — no active code calls `get_audit_logger()`. Tests may need direct imports.

---

## 4. Phase A+B Completion Report

**Completed:** 2026-03-01

### Annotation Coverage

| Status | Count | Description |
|--------|-------|-------------|
| ACTIVE | 110 | Module headers (107) + entry point markers (3 `ACTIVE -- ENTRY POINT`) |
| DEAD | 93 | 35 module headers + 58 method/class-level annotations within MIXED files |
| MIXED | 4 | `build_graph.py`, `saga.py`, `contract_checks.py`, `connections.py` |
| STANDALONE | 2 | `stdio_server.py`, `cli.py` |
| DORMANT | 2 | `markdown.py` (legacy parser), `ner_gliner.py` (config-gated) |
| TEST_ONLY | 1 | `templates/advanced/schemas.py` |
| PHANTOM | 1 | `shadow_comparison.py` (still eagerly imported by `parsers/__init__.py`) |
| PHANTOM-SOURCE | 0 | All cleaned in Phase B |
| **Total** | **213** | **annotations across all `src/` files** |

### Phase B Impact

| `__init__.py` | Before | After |
|---------------|--------|-------|
| `auto/__init__.py` | Loads 8 phantom modules | Loads nothing — clean package init |
| `ingestion/__init__.py` | Loads `api.py` → `build_graph.py` | Empty — no phantom loading |
| `neo/__init__.py` | Loads 3 phantom modules | Empty — no phantom loading |
| `shared/__init__.py` | Loads `audit/logger.py` | Clean — only `config` + `connections` |

**Net result:** 13 of 14 phantom loading chains eliminated. 1 remaining: `shadow_comparison.py` via `parsers/__init__.py:28`.

### Entry Point Verification (all passing)

```
python -c "from src.mcp_server.main import app"           # OK
python -c "from src.ingestion.worker import process_job"   # OK
python -c "from src.ingestion.auto.service import app"     # OK
python -c "from src.ingestion.auto.queue import IngestJob" # OK
python -c "from src.ingestion.auto.reaper import JobReaper"# OK
```

---

## 5. Code Review Findings

**Date:** 2026-03-02
**Source:** External code review of Phase A+B work

### Finding 1: Plan document is stale — VALID

Plan doc had `Status: PLANNING (annotations phase)` and `← WE ARE HERE` at Phase A despite A+B being complete. Plan also cited "16 phantom modules" when only 1 remains.

**Resolution:** Plan doc updated (this document).

### Finding 2: One phantom load still exists on active path — VALID

`src/ingestion/parsers/__init__.py:28` eagerly imports `ShadowModeError` from `shadow_comparison.py`:
```python
from src.ingestion.parsers.shadow_comparison import ShadowModeError
```
This is a top-level import in an `@status: ACTIVE` module. The shadow comparison logic only executes when `config.ingestion.parser.shadow_mode=true` (defaults false), but the module is loaded every time the parser package is imported.

**Resolution:** Add Phase B.6 step to convert to lazy import or remove.

### Finding 3: STANDALONE CLI depends on DEAD module — VALID

`src/ingestion/auto/cli.py:34` imports from `src/ingestion/auto/progress.py`:
```python
from src.ingestion.auto.progress import JobStage, ProgressReader
```
`cli.py` is marked `@status: STANDALONE` but `progress.py` is marked `@status: DEAD`. This is a classification conflict — if the CLI is a working standalone tool, `progress.py` is a dependency and can't be DEAD.

**Resolution:** Explicit policy decision needed before Phase D. Options:
1. Both are STANDALONE (CLI + its dependency) — keep both
2. Both are DEAD (retire the CLI tool) — delete both

### Finding 4: 19 test files import from DEAD modules — VALID

After Phase B removed the `ingest_document` re-export from `src/ingestion/__init__.py`, one test still uses the package-root import (`tests/p3_t4_integration_test.py:10`). Additionally, 19 unique test files import from modules annotated `@status: DEAD`:

| # | Test File | Dead Import Target |
|---|-----------|-------------------|
| 1 | `tests/p4_t4_test.py` | `src.learning` |
| 2 | `tests/p4_t2_perf_test.py` | `src.ops.optimizer` |
| 3 | `tests/p4_t2_test.py` | `src.ops.optimizer` |
| 4 | `tests/p4_t3_test.py` | `src.ops.warmers` |
| 5 | `tests/test_phase7c_index_registry.py` | `src.registry.index_registry` |
| 6 | `tests/integration/test_session_tracking.py` | `src.ops.session_cleanup_job` |
| 7 | `tests/p1_t4_test.py` | `src.mcp_server.security` + `src.shared.audit` |
| 8 | `tests/p2_t2_test.py` | `src.mcp_server.validation` |
| 9 | `tests/p5_t3_test.py` | `src.mcp_server.validation` + `security.rate_limiter` |
| 10 | `tests/conftest.py` | `src.mcp_server.security` |
| 11 | `tests/e2e/test_golden_set.py` | `src.shared.feature_flags` |
| 12 | `tests/p3_t4_integration_test.py` | `src.ingestion` (package-root) + `src.ingestion.reconcile` |
| 13 | `tests/p3_t4_test.py` | `src.ingestion.incremental` + `reconcile` |
| 14 | `tests/v2_2/test_reconciliation_drift.py` | `src.ingestion.reconcile` |
| 15 | `tests/ingestion/test_per_call_embedding_overrides.py` | `src.ingestion.api` |
| 16 | `tests/unit/test_graph_enhancements.py` | `src.neo.graph_enhancements` |
| 17 | `tests/p6_t1_test.py` | `src.ingestion.auto.watcher` + dead auto modules |
| 18 | `tests/p6_t2_test.py` | `src.ingestion.auto.progress` + dead auto modules |
| 19 | `tests/p6_t4_test.py` | `src.ingestion.auto.verification` + dead auto modules |

**Resolution:** Must split test suite into active vs legacy before Phase D deletions. Do not gate pruning on legacy test pass. Phase F must handle all 19 files.

### Finding 5: Migration script depends on dead symbols — VALID

`scripts/neo4j_structural_migration.py:172-174` imports from three modules marked DEAD:
```python
from src.neo.contract_checks import run_contract_checks
from src.neo.entity_normalization import normalize_entities_backfill
from src.neo.structural_builder import StructuralEdgeBuilder
```

If this script is still operational or might need to run again, Phase D/E deletions would break it.

**Resolution:** Classify `scripts/` before pruning. Options:
1. Script is retired → annotate and ignore
2. Script may run again → its imports block deletion of those DEAD modules

---

## 6. Phase B.6: Pre-Phase-C Prerequisites

These items must be resolved before proceeding with Phases C-F.

### B.6.1: Remove or lazy-bind `shadow_comparison` import

**File:** `src/ingestion/parsers/__init__.py:28`

Convert the top-level import to a lazy import inside the shadow-mode code path:
```python
# Before (line 28 — top-level eager import):
from src.ingestion.parsers.shadow_comparison import ShadowModeError

# After (remove line 28; import only where used in _parse_with_shadow_comparison):
# ShadowModeError is already imported inside _parse_with_shadow_comparison() at line 196-200.
# The top-level import at line 28 is redundant and causes unnecessary phantom loading.
```

After this change, update `shadow_comparison.py` annotation from `@status: PHANTOM` to `@status: DEAD` (or DORMANT if shadow mode is a planned feature).

### B.6.2: Resolve CLI / progress classification conflict

**Decision required:** Is `ingestctl` (cli.py) still an operational tool?

| Decision | Action |
|----------|--------|
| **Keep CLI** | Reclassify `progress.py` from DEAD to STANDALONE. Both survive Phase D. |
| **Retire CLI** | Reclassify `cli.py` from STANDALONE to DEAD. Both deleted in Phase D. |

### B.6.3: Classify migration scripts

**File:** `scripts/neo4j_structural_migration.py`

Determine if this script is retired or may need to run again. Its imports of `run_contract_checks`, `normalize_entities_backfill`, and `StructuralEdgeBuilder` block deletion of:
- `src/neo/contract_checks.py` (specifically `run_contract_checks`)
- `src/neo/entity_normalization.py`
- `src/neo/structural_builder.py`

| Decision | Action |
|----------|--------|
| **Script is retired** | Add `# @status: RETIRED` header to script. Proceed with Phase D/E deletions. |
| **Script may run again** | Remove DEAD annotation from the 3 imported modules (or classify them as STANDALONE). |

### B.6.4: Add CI guard (recommended)

Add a lint check that fails if any ACTIVE or MIXED module imports from a DEAD or PHANTOM module:
```bash
# Example: check for ACTIVE→DEAD import violations
grep -rn "@status: DEAD" src/ | sed 's/:.*//;s|src/||;s|\.py||;s|/|.|g' | \
  while read mod; do
    grep -rn "from src\.$mod" src/ --include="*.py" | \
    grep -v "@status: DEAD" | grep -v "@status: PHANTOM"
  done
```

This prevents regressions where active code accidentally starts depending on dead code.

### B.6.5: Update Phase F test retirement scope

Phase F must now account for all 19 test files identified in Finding 4 (see table above), not just the 7 originally listed.

---

## 7. Phase C: Pull Forward Active Logic

> **Prerequisite:** Complete all B.6 items above before starting Phase C.

**Goal:** Extract the active portions of legacy modules into clean, focused new modules.

### C.1: `build_graph.py` → NEW `src/ingestion/embedding_context.py`

Extract the 4 active methods + 3 attributes into a lightweight class:

```python
class EmbeddingContext:
    """Settings container and text utilities for embedding computation.

    Extracted from GraphBuilder (build_graph.py) to decouple atomic.py
    from the 3100-line legacy monolith. Provides:
    - Embedding provider initialization (ensure_embedder)
    - Embedding settings/plan access
    - Text preparation for embedding (section text, title text)
    """

    def __init__(self, driver, config, qdrant_client=None):
        # Port: GraphBuilder.__init__ settings-only logic (~70 lines)

    def ensure_embedder(self):
        # Port: GraphBuilder.ensure_embedder() (~30 lines)

    def build_section_text_for_embedding(self, section):
        # Port: GraphBuilder._build_section_text_for_embedding (~10 lines)

    def build_title_text_for_embedding(self, section):
        # Port: GraphBuilder._build_title_text_for_embedding (~15 lines)
```

**Then update `atomic.py:1055`:**
```python
# Before:
from src.ingestion.build_graph import GraphBuilder
builder = GraphBuilder(self.neo4j_driver, config, self.qdrant_client)

# After:
from src.ingestion.embedding_context import EmbeddingContext
ctx = EmbeddingContext(self.neo4j_driver, config, self.qdrant_client)
```

**Estimated size:** ~200 lines (vs 3100 in build_graph.py)

### C.2: `saga.py` → NEW `src/ingestion/ingestion_validation.py`

Extract the 3 active symbols:

```python
# src/ingestion/ingestion_validation.py
# Extracted from saga.py — only the validation and context dataclasses
# that AtomicIngestionCoordinator actually uses.

@dataclass
class SagaContext:
    """Shared context for tracking IDs during saga execution."""
    ...

@dataclass
class ValidationResult:
    """Result of pre-commit validation."""
    ...

class IngestionValidator:
    """Validates data integrity before committing to Neo4j and Qdrant."""
    ...
```

**Estimated size:** ~150 lines (vs 672 in saga.py)

### C.3: `contract_checks.py` → TRIM in place

Keep only `GraphContractChecker.__init__` and `find_documents_needing_repair`.
Remove `run_all_checks`, `run_contract_checks`, and 6 individual check methods.

**Estimated reduction:** ~350 lines removed

---

## 8. Phase D: Prune Dead Modules

**Goal:** Delete entire files/directories with zero active callers.

### D.1: Delete full directories
```
rm -rf src/learning/
rm -rf src/ops/
rm -rf src/registry/
rm -rf src/mcp_server/security/
```

### D.2: Delete individual dead files
```
rm src/query/diffusion_reranker.py
rm src/query/graph_expansion.py
rm src/query/graph_features.py
rm src/neo/entity_normalization.py
rm src/neo/graph_enhancements.py
rm src/neo/structural_builder.py
rm src/mcp_server/validation.py
rm src/ingestion/auto/watcher.py
rm src/shared/feature_flags.py
```

### D.3: Delete files only after Phase B (__init__ cleanup)
These files are currently phantom-loaded. Delete only after their `__init__.py` imports are removed:
```
rm src/ingestion/auto/orchestrator.py
rm src/ingestion/auto/backpressure.py
rm src/ingestion/auto/report.py
rm src/ingestion/auto/verification.py
rm src/ingestion/incremental.py
rm src/ingestion/parsers/notion.py
```

### D.4: Move to tests/ (not delete)
```
mv src/ingestion/api.py tests/fixtures/ingestion_api_facade.py
```

### D.5: Decide per project roadmap
| File | Decision needed |
|------|----------------|
| `src/ingestion/reconcile.py` | Keep if reconciliation feature is planned; delete if not |
| `src/neo/explain_guard.py` | Keep if query plan validation will be used; delete if not |
| `src/neo/defensive_query.py` | Keep if needed for future safety features; delete if not |
| `src/neo/health.py` | Keep if separate Neo4j health checks needed vs monitoring/health.py |
| `src/shared/audit/logger.py` | Keep if audit logging is planned; delete if not |
| `src/ingestion/parsers/shadow_comparison.py` | Keep if parser migration validation still needed |
| `src/ingestion/parsers/markdown.py` | Keep as fallback; mark DORMANT |

---

## 9. Phase E: Prune Dead Methods in Active Modules

**Goal:** Remove dead methods from files that have a mix of active and dead code.

### E.1: `build_graph.py`
After Phase C.1 (pulling forward to `embedding_context.py`):
- Delete entire `build_graph.py` OR
- If keeping for reference, remove 39 dead methods and the module-level `ingest_document()`, keeping only comments pointing to `embedding_context.py`

### E.2: `saga.py`
After Phase C.2 (pulling forward to `ingestion_validation.py`):
- Delete entire `saga.py` OR
- Remove `SagaCoordinator`, `IngestionSagaBuilder`, `SagaStep`, `SagaStepResult`, `SagaStatus`, `StepStatus`, `SagaStepFailure`, `SagaCompensationFailure`

### E.3: `contract_checks.py`
- Remove `run_contract_checks()` module-level function
- Remove `GraphContractChecker.run_all_checks()` and all 6 individual check methods
- Keep `__init__` and `find_documents_needing_repair`

### E.4: `connections.py`
- Remove `CompatQdrantClient.purge_document()` and `create_collection_with_dims()`

---

## 10. Phase F: Retire Dead Tests

**Scope:** 19 test files currently import from modules annotated `@status: DEAD` (identified in code review, Finding 4). These must be retired, updated, or isolated before Phase D deletions.

**Strategy:** Split into active vs legacy test suites. Do not gate pruning on legacy test pass.

### Full list of affected test files

| # | Test File | Dead Import Target | Action |
|---|-----------|-------------------|--------|
| 1 | `tests/p4_t4_test.py` | `src.learning` | Delete |
| 2 | `tests/p4_t2_perf_test.py` | `src.ops.optimizer` | Delete |
| 3 | `tests/p4_t2_test.py` | `src.ops.optimizer` | Delete |
| 4 | `tests/p4_t3_test.py` | `src.ops.warmers` | Delete |
| 5 | `tests/test_phase7c_index_registry.py` | `src.registry.index_registry` | Delete |
| 6 | `tests/integration/test_session_tracking.py` | `src.ops.session_cleanup_job` | Review — may have active tests mixed in |
| 7 | `tests/p1_t4_test.py` | `src.mcp_server.security` + `src.shared.audit` | Delete |
| 8 | `tests/p2_t2_test.py` | `src.mcp_server.validation` | Delete |
| 9 | `tests/p5_t3_test.py` | `src.mcp_server.validation` + `security.rate_limiter` | Delete |
| 10 | `tests/conftest.py` | `src.mcp_server.security` | Edit — remove dead fixture, keep active fixtures |
| 11 | `tests/e2e/test_golden_set.py` | `src.shared.feature_flags` | Review — may have active tests mixed in |
| 12 | `tests/p3_t4_integration_test.py` | `src.ingestion` (package-root) + `reconcile` | Delete |
| 13 | `tests/p3_t4_test.py` | `src.ingestion.incremental` + `reconcile` | Delete |
| 14 | `tests/v2_2/test_reconciliation_drift.py` | `src.ingestion.reconcile` | Delete |
| 15 | `tests/ingestion/test_per_call_embedding_overrides.py` | `src.ingestion.api` | Review — update import path or delete |
| 16 | `tests/unit/test_graph_enhancements.py` | `src.neo.graph_enhancements` | Delete |
| 17 | `tests/p6_t1_test.py` | `src.ingestion.auto.watcher` + dead auto modules | Keep watcher tests; remove dead module tests |
| 18 | `tests/p6_t2_test.py` | `src.ingestion.auto.progress` + dead auto modules | Delete (or keep if CLI is maintained — see B.6.2) |
| 19 | `tests/p6_t4_test.py` | `src.ingestion.auto.verification` | Delete |

### Additional test considerations

| Test file | Notes |
|-----------|-------|
| `tests/p6_t3_test.py` | Tests `cli.py` (STANDALONE) — keep if CLI is maintained per B.6.2 |
| `tests/conftest.py` | Shared fixture file — edit to remove dead-module fixtures only |
| `tests/integration/test_session_tracking.py` | May contain active session tests alongside dead `SessionCleanupJob` tests |
| `tests/e2e/test_golden_set.py` | May contain active e2e tests alongside dead `feature_flags` usage |

---

## 11. Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|-----------|
| A (Annotate) | Zero — no code changes | Comments only. **DONE.** |
| B (__init__ cleanup) | Low — may break test imports | Run full test suite; update imports. **DONE.** |
| B.6 (Prerequisites) | Low — classification decisions | Explicit policy on CLI/progress, migration scripts |
| C (Pull forward) | Moderate — new modules must be exact ports | Side-by-side diff of ported logic; run integration tests |
| D (Prune modules) | Moderate — scripts and tests depend on "dead" symbols | Must resolve B.6.2 (CLI), B.6.3 (scripts) first |
| E (Prune methods) | Moderate — must not remove active methods | Annotations + grep double-check before each removal |
| F (Retire tests) | Low — 19 test files for dead code | Split active vs legacy suites; skip rather than delete if uncertain |

---

## 12. Validation Checklist

Before each phase, run:
```bash
# Verify no active code was accidentally annotated DEAD
grep -rn "@status: DEAD" src/ | wc -l   # Should match expected count

# Verify annotations are greppable
grep -rn "@status:" src/ | head -20      # Spot-check format consistency

# After __init__ cleanup (Phase B), verify no import errors
python -c "from src.mcp_server.main import app"
python -c "from src.ingestion.worker import process_job"
python -c "from src.ingestion.auto.service import app"

# After pull-forward (Phase C), verify atomic.py still works
pytest tests/ -k "atomic" -x

# After prune (Phase D/E), full test suite
pytest tests/ --tb=short
```

---

## Appendix: Line Count Impact

| Category | Lines | After pruning |
|----------|-------|--------------|
| Dead modules (full files) | ~4,200 | 0 |
| Dead methods in active modules | ~3,800 | 0 |
| Phantom modules (delete after __init__ fix) | ~3,500 | 0 |
| Pull-forward (new clean modules) | 0 | +350 |
| **Net reduction** | **~11,500** | **~11,150 lines removed** |

Current `src/` line count: ~28,000 lines
After full pruning: ~16,850 lines (~40% reduction)
