-- search_documentation: Semantic search across sections and entities
-- Parameters: $query_text, $filters, $limit, $max_hops
-- Returns: relevant sections and connected entities

-- Version 1: Basic section search
MATCH (s:Section)
WHERE $section_ids IS NULL OR size($section_ids) = 0 OR s.id IN $section_ids
OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
WHERE r.confidence >= 0.5
RETURN s, collect(DISTINCT {entity: e, confidence: r.confidence}) AS entities
ORDER BY s.document_id, s.order
LIMIT $limit;

-- Version 2: Search with document context
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
WHERE $section_ids IS NULL OR size($section_ids) = 0 OR s.id IN $section_ids
OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
WHERE r.confidence >= 0.5
RETURN d, s, collect(DISTINCT {
  entity: e,
  confidence: r.confidence,
  type: labels(e)[0]
}) AS entities
ORDER BY s.document_id, s.order
LIMIT $limit;

-- Version 3: Search with controlled expansion
-- Phase 2 Cleanup: Removed REQUIRES, AFFECTS from filter (never materialized)
-- Phase 3: Added REFERENCES for cross-document traversal
MATCH (s:Section)
WHERE $section_ids IS NULL OR size($section_ids) = 0 OR s.id IN $section_ids
OPTIONAL MATCH path=(s)-[:MENTIONS|:CONTAINS_STEP|:HAS_PARAMETER*1..$max_hops]->(n)
WHERE ALL(r IN relationships(path) WHERE type(r) IN ['MENTIONS', 'CONTAINS_STEP', 'HAS_PARAMETER'])
WITH s, n, min(length(path)) AS dist
ORDER BY dist ASC
RETURN s, collect(DISTINCT {node: n, distance: dist, labels: labels(n)})[0..10] AS expanded
LIMIT $limit;

-- Version 4: Cross-document expansion via REFERENCES (Phase 3)
-- Traverses from chunks to referenced documents, then to their sections
-- Parameters: $chunk_ids, $limit, $min_ref_confidence (default 0.5)
-- M2: Parameterized confidence threshold for query-time tuning
MATCH (c:Chunk)
WHERE $chunk_ids IS NULL OR size($chunk_ids) = 0 OR c.id IN $chunk_ids
OPTIONAL MATCH (c)-[ref:REFERENCES]->(d:Document)-[:HAS_SECTION]->(s:Section)
WHERE ref.confidence >= coalesce($min_ref_confidence, 0.5)
WITH c, d, collect(DISTINCT {
  section: s,
  document_id: d.id,
  document_title: d.title,
  reference_type: ref.type,
  confidence: ref.confidence
})[0..5] AS cross_doc_results
RETURN c, cross_doc_results
LIMIT $limit;
