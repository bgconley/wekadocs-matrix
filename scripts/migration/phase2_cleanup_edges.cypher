// ==========================================================================
// Phase 2 Migration: Remove Dead Edges (PREV, SAME_HEADING)
// ==========================================================================
//
// Purpose: Delete PREV and SAME_HEADING relationships that were created
//          by previous ingestion runs. These edge types are no longer
//          created (Phase 2 cleanup) but may exist from earlier ingestion.
//
// Run: AFTER code deployment (when ingestion no longer creates these edges)
// Prerequisite: Neo4j 5.x or 2025.x (uses native CALL {} IN TRANSACTIONS)
//
// Safety: This is a one-way operation. To restore edges, re-run ingestion
//         on affected documents (which won't recreate them post-Phase 2).
//
// History:
//   v1: Used APOC apoc.periodic.iterate (APOC dependency)
//   v2: Rewritten to native CALL {} IN TRANSACTIONS (no APOC required)
//
// ==========================================================================

// ---------------------------------------------------------------------------
// STEP 1: Count existing edges (diagnostic - safe to run anytime)
// ---------------------------------------------------------------------------
MATCH ()-[r:PREV]->()
RETURN 'PREV' AS type, count(r) AS count
UNION ALL
MATCH ()-[r:SAME_HEADING]->()
RETURN 'SAME_HEADING' AS type, count(r) AS count;

// ---------------------------------------------------------------------------
// STEP 2: Delete PREV edges (batched using native Neo4j 5.x+ syntax)
// ---------------------------------------------------------------------------
// Native batched deletion - no APOC required
// ON ERROR CONTINUE: Skip failed batches, report at end
// REPORT STATUS: Track which batches succeeded/failed
//
MATCH (a)-[r:PREV]->(b)
CALL (r) {
  DELETE r
} IN TRANSACTIONS OF 10000 ROWS
  ON ERROR CONTINUE
  REPORT STATUS AS status
RETURN 'PREV' AS type,
       count(*) AS total_processed,
       sum(CASE WHEN status.committed THEN 1 ELSE 0 END) AS batches_committed,
       sum(CASE WHEN NOT status.committed THEN 1 ELSE 0 END) AS batches_failed;

// ---------------------------------------------------------------------------
// STEP 3: Delete SAME_HEADING edges (batched using native Neo4j 5.x+ syntax)
// ---------------------------------------------------------------------------
MATCH (a)-[r:SAME_HEADING]->(b)
CALL (r) {
  DELETE r
} IN TRANSACTIONS OF 10000 ROWS
  ON ERROR CONTINUE
  REPORT STATUS AS status
RETURN 'SAME_HEADING' AS type,
       count(*) AS total_processed,
       sum(CASE WHEN status.committed THEN 1 ELSE 0 END) AS batches_committed,
       sum(CASE WHEN NOT status.committed THEN 1 ELSE 0 END) AS batches_failed;

// ---------------------------------------------------------------------------
// STEP 4: Verify cleanup complete
// ---------------------------------------------------------------------------
MATCH ()-[r:PREV|SAME_HEADING]->()
RETURN type(r) AS type, count(r) AS remaining
ORDER BY type;

// ---------------------------------------------------------------------------
// STEP 5: Update RelationshipTypesMarker (if exists)
// ---------------------------------------------------------------------------
MATCH (m:RelationshipTypesMarker {id: 'chunk_rel_types_v1'})
SET m.types = ['NEXT','CHILD_OF','MENTIONS','PARENT_OF'],
    m.updated_at = datetime(),
    m.phase2_cleanup_at = datetime()
RETURN m.types AS updated_types;

// ---------------------------------------------------------------------------
// EXPECTED RESULTS
// ---------------------------------------------------------------------------
// Step 1: Shows count of PREV and SAME_HEADING edges before cleanup
// Step 2: Shows total_processed and batches_committed for PREV deletion
// Step 3: Shows total_processed and batches_committed for SAME_HEADING deletion
// Step 4: Should show 0 remaining for both types (or no rows)
// Step 5: Shows updated marker with PREV/SAME_HEADING removed
//
// ---------------------------------------------------------------------------
// FALLBACK (for Neo4j < 5.0 without CALL {} IN TRANSACTIONS)
// ---------------------------------------------------------------------------
// If running on older Neo4j, use manual loop:
//
// MATCH ()-[r:PREV]->()
// WITH r LIMIT 10000
// DELETE r
// RETURN count(*) AS deleted;
// -- Repeat until 0 rows returned
//
// MATCH ()-[r:SAME_HEADING]->()
// WITH r LIMIT 10000
// DELETE r
// RETURN count(*) AS deleted;
// -- Repeat until 0 rows returned
