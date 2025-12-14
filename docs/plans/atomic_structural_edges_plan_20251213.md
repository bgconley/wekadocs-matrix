# Atomic Structural Edges & Ingestion Observability Plan

**Date:** 2025-12-13
**Branch:** `neo4j-gds-enhancements`
**Status:** Proposed
**Author:** Claude (with user direction)

---

## Executive Summary

This plan addresses two architectural concerns identified during review of the Neo4j structural edge implementation:

1. **Philosophical inconsistency**: The saga pattern ensures atomicity between Neo4j and Qdrant, but structural edges are built "best-effort" after the saga commits, creating silent partial state.

2. **Observability gap**: No consolidated view of ingestion run results; failures and warnings scattered across log lines.

**Solution:**
- Move structural edge building **inside** the atomic saga transaction
- Add end-of-run summary instrumentation to the ingestion worker
- Repurpose reconciliation from "silent repair" to "health check alerting"

---

## Table of Contents

1. [Background & Problem Statement](#background--problem-statement)
2. [Part 1: Atomic Structural Edges](#part-1-atomic-structural-edges)
3. [Part 2: End-of-Run Summary Instrumentation](#part-2-end-of-run-summary-instrumentation)
4. [Part 3: Reconciliation as Health Check](#part-3-reconciliation-as-health-check)
5. [Implementation Phases](#implementation-phases)
6. [Testing Strategy](#testing-strategy)
7. [Rollback Plan](#rollback-plan)

---

## Background & Problem Statement

### The Saga's Purpose

The saga pattern in `atomic.py` exists because we made a deliberate architectural decision:

> "A document that exists in Neo4j but not Qdrant (or vice versa) is **corrupt state**. We will invest in compensation logic to prevent this."

The saga guarantees:
- If Neo4j write fails → no Qdrant write happens
- If Qdrant write fails after Neo4j commit → compensate by deleting from Neo4j
- Result: atomic "all or nothing" semantics

### The Current Contradiction

The structural edge builder runs **after** the saga commits successfully:

```python
# Current implementation in atomic.py
if saga_result["success"]:
    try:
        struct_result = build_structural_edges_for_document(...)  # Best-effort
    except Exception as e:
        logger.warning(...)  # Just log it

    return AtomicIngestionResult(success=True, ...)  # Still success!
```

This creates a **third state** the saga was never designed to prevent:

| State | Neo4j Doc | Qdrant Vectors | Structural Edges | Result |
|-------|-----------|----------------|------------------|--------|
| Valid | ✓ | ✓ | ✓ | ✅ Success |
| Invalid (caught) | ✓ | ✗ | - | ❌ Compensated |
| Invalid (caught) | ✗ | ✓ | - | ❌ Compensated |
| **Invalid (silent)** | ✓ | ✓ | ✗ | ✅ "Success" |

The fourth row is the problem. We've built elaborate machinery to prevent rows 2-3, then casually accept row 4.

### Why "It's Just an Enhancement" Doesn't Hold

The argument for best-effort is: "Structural edges are graph augmentation on top of vector-first retrieval. They're nice-to-have, not essential."

But this framing is misleading:

1. **The canonical plan builds features on these edges**: Context expansion uses `PARENT_HEADING`. Diffusion reranking uses `NEXT_CHUNK` and `shared_entity` edges. These aren't decorations—they're the retrieval enhancements the whole overhaul exists to enable.

2. **Degraded state is not obvious**: If 15% of documents have missing edges, retrieval "works" but returns worse results. There's no error, no empty response—just silently lower quality.

3. **The reconciliation escape hatch normalizes bad state**: "It's fine, reconciliation will fix it later" is eventual consistency by another name. The saga was explicitly built to avoid eventual consistency.

---

## Part 1: Atomic Structural Edges

### Current Saga Flow

```
_execute_atomic_saga():
    Neo4j Transaction {
        1. Upsert Document node
        2. Upsert Chunk nodes + HAS_SECTION/HAS_CHUNK edges
        3. Upsert Entity nodes
        4. Create MENTIONS edges
        5. Create REFERENCES edges
    }  // Transaction still open

    6. Write vectors to Qdrant
    7. If Qdrant succeeds → Commit Neo4j
    8. If Qdrant fails → Rollback Neo4j

Post-saga (current best-effort):
    9. Build structural edges ← OUTSIDE transaction, can fail silently
```

### Proposed Saga Flow

```
_execute_atomic_saga():
    Neo4j Transaction {
        1. Upsert Document node
        2. Upsert Chunk nodes + HAS_SECTION/HAS_CHUNK edges
        3. Upsert Entity nodes
        4. Create MENTIONS edges
        5. Create REFERENCES edges

        // NEW: Structural edges inside transaction
        6. Normalize parent_path → parent_path_norm
        7. Compute parent_chunk_id
        8. Create NEXT_CHUNK edges (sequential adjacency)
        9. Create PARENT_HEADING/CHILD_OF/PARENT_OF edges (hierarchy)
        10. Create NEXT edges (sibling adjacency)
    }  // Transaction still open

    11. Write vectors to Qdrant
    12. If Qdrant succeeds → Commit Neo4j (all edges committed atomically)
    13. If Qdrant fails → Rollback Neo4j (all edges rolled back atomically)
```

### Key Design Decisions

#### Inline the queries vs. use the builder

The structural builder uses `self.session.run()`. To work inside a transaction, it needs the `tx` object from `session.execute_write()`. Two options:

1. **Modify builder to accept transaction context** — More reusable, but adds complexity
2. **Inline structural queries directly in saga** — Self-contained, clearer failure boundaries

**Decision: Inline the queries** because:
- The queries are straightforward Cypher
- Keeps the saga self-contained
- Failure attribution is clearer (no "did the builder fail or the saga fail?")
- The builder can still exist for migration/reconciliation use cases

#### What happens if structural edges fail?

The transaction rolls back. Qdrant write never happens. Ingestion fails. This is the **correct behavior** — we don't want partially-built documents.

#### Performance consideration

For a document with 100 chunks:
- ~100 chunk upserts (already happening)
- ~99 NEXT_CHUNK edges (new)
- ~50-80 hierarchy edges (new)
- ~99 NEXT edges (new)

This adds ~250 more writes to the transaction. But:
- All scoped to single document
- Indexes on `(document_id, order)` make queries fast
- Neo4j handles this fine for bounded transactions

### Structural Edge Queries to Inline

#### Step 6: Normalize parent_path

```cypher
MATCH (c:Chunk {document_id: $document_id})
WHERE c.parent_path IS NOT NULL
WITH c,
     [p IN split(replace(c.parent_path, ' > ', '>'), '>') | trim(p)] AS parts
WITH c,
     reduce(path = '', p IN parts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS norm
WHERE c.parent_path_norm IS NULL OR c.parent_path_norm <> norm
SET c.parent_path_norm = norm
RETURN count(c) AS normalized
```

#### Step 7: Compute parent_chunk_id

```cypher
MATCH (child:Chunk {document_id: $document_id})
WHERE child.parent_path_norm IS NOT NULL
  AND child.parent_path_norm CONTAINS ' > '
WITH child, split(child.parent_path_norm, ' > ') AS parts
WITH child, parts[0..size(parts)-1] AS parentParts
WITH child,
     reduce(path = '', p IN parentParts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS parentPathNorm
MATCH (parent:Chunk {document_id: $document_id, parent_path_norm: parentPathNorm})
WHERE parent.order < child.order
  AND (child.level IS NULL OR parent.level IS NULL OR parent.level < child.level)
WITH child, parent
ORDER BY parent.order DESC
WITH child, head(collect(parent)) AS parent
SET child.parent_chunk_id = parent.chunk_id
RETURN count(child) AS computed
```

#### Step 8: Create NEXT_CHUNK edges

```cypher
MATCH (c:Chunk {document_id: $document_id})
WHERE c.order IS NOT NULL
WITH c ORDER BY c.order
WITH collect(c) AS chunks
UNWIND range(0, size(chunks)-2) AS i
WITH chunks[i] AS c1, chunks[i+1] AS c2
MERGE (c1)-[:NEXT_CHUNK]->(c2)
RETURN count(*) AS created
```

#### Step 9: Create hierarchy edges

```cypher
MATCH (child:Chunk {document_id: $document_id})
WHERE child.parent_chunk_id IS NOT NULL
MATCH (parent:Chunk {chunk_id: child.parent_chunk_id})
MERGE (child)-[ph:PARENT_HEADING]->(parent)
ON CREATE SET ph.level_delta = coalesce(child.level, 0) - coalesce(parent.level, 0)
MERGE (child)-[:CHILD_OF]->(parent)
MERGE (parent)-[:PARENT_OF]->(child)
RETURN count(*) AS created
```

#### Step 10: Create NEXT edges (siblings)

```cypher
MATCH (c:Chunk {document_id: $document_id})
WHERE c.order IS NOT NULL
WITH c.parent_chunk_id AS pid, c.level AS lvl, c
ORDER BY c.order
WITH pid, lvl, collect(c) AS chunks
UNWIND range(0, size(chunks)-2) AS i
WITH chunks[i] AS c1, chunks[i+1] AS c2
MERGE (c1)-[:NEXT]->(c2)
RETURN count(*) AS created
```

### Changes Required in atomic.py

1. **Remove** the post-commit best-effort structural building block (lines ~750-804)
2. **Add** structural edge queries to `_execute_atomic_saga()` after chunk/entity writes
3. **Update** `AtomicIngestionResult` stats to include structural edge counts
4. **Update** error handling to properly fail if structural edges fail

---

## Part 2: End-of-Run Summary Instrumentation

### The Observability Gap Today

Currently, to understand "how did that ingestion run go?", you have to:
1. Grep logs for each job completion
2. Manually count successes/failures
3. Hope you didn't miss any warnings
4. Piece together stats from scattered log lines

### Proposed: Consolidated Run Summary

When the queue empties and worker has been idle for **3 minutes**, emit a single structured log entry with everything:

```json
{
  "event": "ingestion_run_summary",
  "run_id": "run_20251213_143022",
  "duration_seconds": 847,

  "jobs": {
    "processed": 47,
    "succeeded": 45,
    "failed": 2
  },

  "documents": {
    "ingested": 45,
    "chunks_created": 2340,
    "entities_created": 891,
    "vectors_written": 2340
  },

  "edges": {
    "HAS_CHUNK": 2340,
    "NEXT_CHUNK": 2295,
    "PARENT_HEADING": 1420,
    "NEXT": 2180,
    "MENTIONS": 4230
  },

  "failures": [
    {
      "path": "data/ingest/corrupted-doc.md",
      "document_id": "doc_abc123",
      "error": "Missing 'order' property on 3 chunks"
    },
    {
      "path": "data/ingest/timeout-doc.md",
      "document_id": "doc_def456",
      "error": "Neo4j transaction timeout after 30s"
    }
  ],

  "warnings": [
    {"count": 3, "message": "Entity normalization found duplicate names"},
    {"count": 1, "message": "Large document (>500 chunks) may impact performance"}
  ]
}
```

### Implementation Design

#### Run Stats Accumulator Class

```python
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Any
import time

@dataclass
class IngestionRunStats:
    """Accumulates statistics across a batch of ingestion jobs."""

    run_id: str
    start_time: float
    end_time: Optional[float] = None

    # Job counts
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0

    # Document/content counts
    chunks_created: int = 0
    entities_created: int = 0
    vectors_written: int = 0

    # Edge counts by type
    edges: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Failure details
    failed_jobs: List[Dict[str, Any]] = field(default_factory=list)

    # Warnings accumulated during run
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def record_job_result(self, job_path: str, result: "AtomicIngestionResult"):
        """Record the outcome of a single job."""
        self.jobs_processed += 1

        if result.success:
            self.jobs_succeeded += 1
            self._aggregate_success_stats(result.stats)
        else:
            self.jobs_failed += 1
            self.failed_jobs.append({
                "path": job_path,
                "document_id": result.document_id,
                "error": result.error,
                "saga_id": result.saga_id,
            })

    def _aggregate_success_stats(self, stats: Dict[str, Any]):
        """Aggregate stats from a successful ingestion."""
        self.chunks_created += stats.get("chunks_created", 0)
        self.entities_created += stats.get("entities_created", 0)
        self.vectors_written += stats.get("vectors_written", 0)

        # Aggregate edge counts
        edge_stats = stats.get("structural_edges", {})
        for edge_type, count in edge_stats.items():
            self.edges[edge_type] += count

    def record_warning(self, message: str, context: Optional[Dict] = None):
        """Record a warning, deduplicating by message."""
        for w in self.warnings:
            if w["message"] == message:
                w["count"] += 1
                return
        self.warnings.append({
            "count": 1,
            "message": message,
            "context": context,
        })

    def finalize(self) -> Dict[str, Any]:
        """Generate the final summary dict for logging."""
        self.end_time = time.monotonic()

        return {
            "run_id": self.run_id,
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "jobs": {
                "processed": self.jobs_processed,
                "succeeded": self.jobs_succeeded,
                "failed": self.jobs_failed,
            },
            "documents": {
                "ingested": self.jobs_succeeded,
                "chunks_created": self.chunks_created,
                "entities_created": self.entities_created,
                "vectors_written": self.vectors_written,
            },
            "edges": dict(self.edges),
            "failures": self.failed_jobs,
            "warnings": self.warnings,
        }
```

#### Modified Worker Loop

```python
# Constants
SUMMARY_IDLE_SECONDS = 180  # 3 minutes

# Run state
run_stats: Optional[IngestionRunStats] = None
last_job_completed_at: Optional[float] = None

while not shutdown_requested:
    try:
        item = brpoplpush(timeout=1)

        if not item:
            # Queue is empty - check if we should emit summary
            if (run_stats is not None
                and last_job_completed_at is not None
                and (time.monotonic() - last_job_completed_at) >= SUMMARY_IDLE_SECONDS):

                # Emit the summary
                summary = run_stats.finalize()
                log.info("ingestion_run_summary", **summary)

                # Log failures prominently if any
                if run_stats.jobs_failed > 0:
                    log.warning(
                        "ingestion_run_had_failures",
                        failed_count=run_stats.jobs_failed,
                        failures=run_stats.failed_jobs,
                    )

                # Reset for next run
                run_stats = None
                last_job_completed_at = None

            await asyncio.sleep(0.05)
            continue

        # We have a job to process
        raw, job_id = item
        job = IngestJob.from_json(raw)

        # Start new run if needed
        if run_stats is None:
            run_stats = IngestionRunStats(
                run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                start_time=time.monotonic(),
            )
            log.info("ingestion_run_started", run_id=run_stats.run_id)

        # Process the job
        try:
            result = await process_job(job)
            run_stats.record_job_result(job.path, result)

            if result.success:
                ack(raw, job_id)
                log.info("job_completed", job_id=job_id, path=job.path)
            else:
                fail(raw, job_id, reason=result.error, requeue=True)
                log.error("job_failed", job_id=job_id, path=job.path, error=result.error)

            last_job_completed_at = time.monotonic()

        except Exception as e:
            log.error("job_exception", job_id=job_id, error=str(e))
            fail(raw, job_id, reason=str(e), requeue=True)
            run_stats.record_job_result(job.path, AtomicIngestionResult(
                success=False,
                document_id=None,
                saga_id=None,
                error=str(e),
            ))
            last_job_completed_at = time.monotonic()

    except Exception as loop_err:
        log.error("worker_loop_error", error=str(loop_err))
        await asyncio.sleep(0.25)
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Worker restarts mid-run | Stats lost, but per-job logs still exist. Acceptable. |
| Continuous job stream (never idle) | Summary never emitted. Consider adding "every N jobs" fallback. |
| Multiple concurrent workers | Each emits own summary. Fine for observability. |
| Very long idle (hours) | Summary emitted once at 3 min, then silence. Acceptable. |

### Optional Enhancement: Periodic Summaries for Long Streams

For continuous ingestion without idle periods, emit summary every N jobs:

```python
SUMMARY_JOB_INTERVAL = 100  # Emit summary every 100 jobs

# In the job processing section:
if run_stats.jobs_processed % SUMMARY_JOB_INTERVAL == 0:
    log.info("ingestion_run_progress", **run_stats.finalize())
```

---

## Part 3: Reconciliation as Health Check

### Current Role (Silent Repair)

The current reconciliation runs during worker idle and silently repairs documents with missing edges:

```python
def _run_structural_reconciliation_sync(*, max_docs: int):
    doc_ids = checker.find_documents_needing_repair()[:max_docs]
    for doc_id in doc_ids:
        builder.build_for_document(doc_id)  # Silent repair
```

### Proposed Role (Health Check / Alerting)

With structural edges inside the atomic transaction, documents needing repair indicates a **bug** (or migration gap), not normal operation:

```python
def _run_structural_health_check() -> Dict[str, Any]:
    """
    Check for documents needing structural edge repair.

    After atomic ingestion, this should return zero documents.
    Non-zero indicates a bug in ingestion or a migration gap.
    """
    from src.neo.contract_checks import GraphContractChecker
    from src.shared.connections import get_connection_manager

    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()

    with driver.session() as session:
        checker = GraphContractChecker(session)
        doc_ids = checker.find_documents_needing_repair()

        if doc_ids:
            # This is an ERROR condition, not normal operation
            log.error(
                "structural_integrity_violation",
                documents_needing_repair=len(doc_ids),
                sample_doc_ids=doc_ids[:10],
                message=(
                    "Documents found with missing structural edges after atomic ingestion. "
                    "This indicates a bug in ingestion or a migration gap that requires attention."
                ),
            )
            return {
                "status": "unhealthy",
                "documents_needing_repair": len(doc_ids),
                "sample_doc_ids": doc_ids[:10],
            }

        log.info("structural_health_check_passed", documents_checked="all")
        return {
            "status": "healthy",
            "documents_needing_repair": 0,
        }
```

### When Reconciliation IS Needed

| Use Case | Trigger |
|----------|---------|
| One-time migration of historical documents | Manual script run |
| After schema changes that invalidate existing edges | Manual script run |
| Manual repair after identified bugs | Manual script run |

### When Reconciliation Should NOT Be Needed

| Use Case | Why Not |
|----------|---------|
| Normal ongoing operation | Atomic ingestion handles it |
| "Just in case" background repair | Masks bugs that should be fixed |

### Health Check Trigger

The health check should run:
1. At worker startup (to detect pre-existing issues)
2. Periodically (e.g., every hour) as a sanity check
3. On-demand via API/CLI for debugging

```python
# At worker startup
health = _run_structural_health_check()
if health["status"] == "unhealthy":
    log.warning(
        "worker_starting_with_unhealthy_graph",
        documents_needing_repair=health["documents_needing_repair"],
        message="Graph has documents needing repair. Consider running migration script.",
    )
```

---

## Implementation Phases

### Phase 1: Atomic Structural Edges (Priority: High)

**Estimated effort:** 2-3 hours

**Tasks:**
1. Add structural edge queries to `_execute_atomic_saga()` in `atomic.py`
2. Remove the post-commit best-effort structural building block
3. Update `AtomicIngestionResult` stats to include edge counts
4. Update error handling to fail ingestion if structural edges fail
5. Test with sample document ingestion

**Files modified:**
- `src/ingestion/atomic.py`

**Acceptance criteria:**
- [ ] Structural edges created inside Neo4j transaction
- [ ] If structural edges fail, ingestion fails (transaction rolls back)
- [ ] Stats include edge counts by type
- [ ] Existing tests pass

### Phase 2: Run Summary Instrumentation (Priority: Medium)

**Estimated effort:** 2-3 hours

**Tasks:**
1. Create `IngestionRunStats` class in `worker.py` (or separate module)
2. Modify worker loop to track run state
3. Emit summary after 3-minute idle
4. Include failures, edge stats, warnings in summary
5. Test with batch ingestion

**Files modified:**
- `src/ingestion/worker.py`

**Acceptance criteria:**
- [ ] Summary emitted after 3-minute idle
- [ ] Summary includes all specified fields
- [ ] Failed jobs listed with file paths
- [ ] Warnings deduplicated and counted

### Phase 3: Reconciliation as Health Check (Priority: Low)

**Estimated effort:** 1 hour

**Tasks:**
1. Create `_run_structural_health_check()` function
2. Change reconciliation to emit ERROR (not repair) if documents need repair
3. Add health check at worker startup
4. Keep migration script for manual repairs

**Files modified:**
- `src/ingestion/worker.py`

**Acceptance criteria:**
- [ ] Health check emits ERROR if documents need repair
- [ ] No auto-repair in normal operation
- [ ] Startup check logs warning if graph unhealthy

---

## Testing Strategy

### Unit Tests

1. **Atomic structural edges:**
   - Test that structural edges are created in same transaction as chunks
   - Test that transaction rolls back if structural edge query fails
   - Test that Qdrant write doesn't happen if Neo4j fails

2. **Run summary:**
   - Test `IngestionRunStats` accumulation
   - Test summary generation
   - Test warning deduplication

3. **Health check:**
   - Test healthy graph returns "healthy"
   - Test unhealthy graph returns "unhealthy" with details

### Integration Tests

1. **End-to-end ingestion:**
   - Ingest document, verify all edges created
   - Verify summary emitted after idle period

2. **Failure scenarios:**
   - Ingest document with missing `order` property, verify ingestion fails
   - Verify no partial state (no chunks without edges)

### Manual Testing

1. Drop test document in `data/ingest/`
2. Watch worker logs for job completion
3. Wait 3 minutes, verify summary emitted
4. Query Neo4j for structural edges:
   ```cypher
   MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
   OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next)
   OPTIONAL MATCH (c)-[:PARENT_HEADING]->(parent)
   RETURN c.chunk_id, next.chunk_id, parent.chunk_id
   ```

---

## Rollback Plan

If issues arise after deployment:

### Phase 1 Rollback (Atomic Structural Edges)

1. Revert `atomic.py` changes
2. Re-enable post-commit best-effort structural building
3. Documents ingested during issue window may need reconciliation

### Phase 2 Rollback (Run Summary)

1. Revert `worker.py` changes
2. No data impact—this is purely logging

### Phase 3 Rollback (Health Check)

1. Re-enable silent auto-repair if needed
2. No data impact

---

## Summary

This plan aligns the structural edge implementation with the atomicity mindset established by the saga pattern:

| Component | Before | After |
|-----------|--------|-------|
| Structural edges | Best-effort, post-commit | Atomic, inside transaction |
| Failure handling | Silent warning | Ingestion fails |
| Run visibility | Scattered logs | Consolidated summary |
| Reconciliation | Silent repair | Health check alerting |

**The three parts work together:** Atomic edges ensure consistency, the summary provides visibility into what happened, and the health check catches any gaps. No silent failures, no hidden state, no "hope reconciliation fixes it later."

---

*End of Plan*
