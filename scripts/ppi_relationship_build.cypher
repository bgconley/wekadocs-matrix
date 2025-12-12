// ============================================================================
// post_ingest_relationships_v2_2.cypher
// Build graph relationships AFTER data ingest/re-chunking.
// Canonical key is document_id. Legacy alias doc_id is only used as a fallback.
// Idempotent (MERGE + batching). Optional scoped rebuild.
// ============================================================================

// ---------- Parameters (override via --param) ----------
:param document_id => null;   // canonical document_id; null = all documents
:param tenant       => null;  // optional tenant scope; null = all tenants
:param window       => 8;     // SAME_HEADING fan-out window (forward edges)
:param rebuild      => false; // if true, delete edges in scope before rebuilding

// ---------- 0) Optional alias backfill (noop if already done) ----------
MATCH (c:Chunk)
WHERE c.doc_id IS NULL AND c.document_id IS NOT NULL
SET c.doc_id = c.document_id;

// Note: :Section label deprecated - all content nodes are now :Chunk
// Legacy backfill for any remaining Section nodes during migration
MATCH (s:Chunk)
WHERE s.doc_id IS NULL AND s.document_id IS NOT NULL
SET s.doc_id = s.document_id;

// ---------- 1) Optional REBUILD: delete edges in scope ----------
CALL {
  WITH 1 AS _
  WHERE $rebuild = true

  // Delete edges that originate from chunks in scope
  MATCH (c:Chunk)
  WHERE ($document_id IS NULL OR coalesce(c.document_id, c.doc_id) = $document_id)
    AND ($tenant IS NULL OR c.tenant = $tenant)
  OPTIONAL MATCH (c)-[r:NEXT|PREV|SAME_HEADING|CHILD_OF]->()
  WITH r
  WHERE r IS NOT NULL
  CALL {
    WITH r DELETE r
    RETURN 0
  } IN TRANSACTIONS OF 50000 ROWS;

  // Delete edges that originate from parent chunks in scope
  MATCH (s:Chunk)
  WHERE ($document_id IS NULL OR coalesce(s.document_id, s.doc_id) = $document_id)
    AND ($tenant IS NULL OR s.tenant = $tenant)
  OPTIONAL MATCH (s)-[r:PARENT_OF]->()
  WITH r
  WHERE r IS NOT NULL
  CALL {
    WITH r DELETE r
    RETURN 0
  } IN TRANSACTIONS OF 50000 ROWS;

  RETURN 0
};

// ---------- 2) CHILD_OF: (Chunk)-[:CHILD_OF]->(Chunk) ----------
MATCH (c:Chunk)
WHERE c.parent_section_id IS NOT NULL
  AND ($document_id IS NULL OR coalesce(c.document_id, c.doc_id) = $document_id)
  AND ($tenant IS NULL OR c.tenant = $tenant)
CALL {
  WITH c
  MATCH (parent:Chunk {id: c.parent_section_id})
  WHERE ($tenant IS NULL OR parent.tenant = $tenant)
    AND ($document_id IS NULL OR coalesce(parent.document_id, parent.doc_id) = coalesce(c.document_id, c.doc_id))
  MERGE (c)-[:CHILD_OF]->(parent)
  RETURN 0
} IN TRANSACTIONS OF 20000 ROWS;

// ---------- 3) PARENT_OF: (Chunk)-[:PARENT_OF]->(Chunk) ----------
MATCH (child:Chunk)
WHERE child.parent_section_id IS NOT NULL
  AND ($document_id IS NULL OR coalesce(child.document_id, child.doc_id) = $document_id)
  AND ($tenant IS NULL OR child.tenant = $tenant)
CALL {
  WITH child
  MATCH (parent:Chunk {id: child.parent_section_id})
  WHERE ($tenant IS NULL OR parent.tenant = $tenant)
    AND ($document_id IS NULL OR coalesce(parent.document_id, parent.doc_id) = coalesce(child.document_id, child.doc_id))
  MERGE (parent)-[:PARENT_OF]->(child)
  RETURN 0
} IN TRANSACTIONS OF 20000 ROWS;

// ---------- 4) NEXT/PREV within (document_id, parent_section_id) ----------
MATCH (c:Chunk)
WHERE c.order IS NOT NULL
  AND ($document_id IS NULL OR coalesce(c.document_id, c.doc_id) = $document_id)
  AND ($tenant IS NULL OR c.tenant = $tenant)
WITH coalesce(c.document_id, c.doc_id) AS d, c.parent_section_id AS p, c
ORDER BY d, p, c.order
WITH d, p, collect(c) AS chunks
UNWIND range(0, size(chunks)-2) AS i
WITH chunks[i] AS a, chunks[i+1] AS b
CALL {
  WITH a, b
  MERGE (a)-[:NEXT]->(b)
  MERGE (b)-[:PREV]->(a)
  RETURN 0
} IN TRANSACTIONS OF 20000 ROWS;

// ---------- 5) SAME_HEADING (bounded fanout window) ----------
MATCH (c:Chunk)
WHERE c.heading IS NOT NULL
  AND ($document_id IS NULL OR coalesce(c.document_id, c.doc_id) = $document_id)
  AND ($tenant IS NULL OR c.tenant = $tenant)
WITH coalesce(c.document_id, c.doc_id) AS d, c.parent_section_id AS p, c.heading AS h, c
ORDER BY d, p, h, c.order
WITH d, p, h, collect(c) AS chunks
UNWIND range(0, size(chunks)-1) AS i
WITH chunks, i, chunks[i] AS a, size(chunks) AS n
WITH a, CASE
          WHEN $window IS NULL OR $window < 1 THEN 1
          ELSE $window
        END AS W, n, chunks
WITH a, [j IN range(i+1, CASE WHEN i+W < n THEN i+W ELSE n-1 END) | chunks[j]] AS neigh
UNWIND neigh AS b
CALL {
  WITH a, b
  MERGE (a)-[:SAME_HEADING]->(b)
  RETURN 0
} IN TRANSACTIONS OF 20000 ROWS;

// ---------- 6) Diagnostics ----------
MATCH ()-[r:NEXT]->()          RETURN count(r) AS next_count;
MATCH ()-[r:PREV]->()          RETURN count(r) AS prev_count;
MATCH ()-[r:SAME_HEADING]->()  RETURN count(r) AS same_heading_count;
MATCH ()-[r:CHILD_OF]->()      RETURN count(r) AS child_of_count;
MATCH ()-[r:PARENT_OF]->()     RETURN count(r) AS parent_of_count;

// ============================================================================
