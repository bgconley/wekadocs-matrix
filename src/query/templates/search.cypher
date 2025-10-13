-- search_documentation: Semantic search across sections and entities
-- Parameters: $query_text, $filters, $limit, $max_hops
-- Returns: relevant sections and connected entities

-- Version 1: Basic section search
MATCH (s:Section)
WHERE s.id IN $section_ids
OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
WHERE r.confidence >= 0.5
RETURN s, collect(DISTINCT {entity: e, confidence: r.confidence}) AS entities
ORDER BY s.document_id, s.order
LIMIT $limit;

-- Version 2: Search with document context
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
WHERE s.id IN $section_ids
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
MATCH (s:Section)
WHERE s.id IN $section_ids
OPTIONAL MATCH path=(s)-[:MENTIONS|:CONTAINS_STEP|:HAS_PARAMETER*1..$max_hops]->(n)
WHERE ALL(r IN relationships(path) WHERE type(r) IN ['MENTIONS', 'CONTAINS_STEP', 'HAS_PARAMETER', 'REQUIRES', 'AFFECTS'])
WITH s, n, min(length(path)) AS dist
ORDER BY dist ASC
RETURN s, collect(DISTINCT {node: n, distance: dist, labels: labels(n)})[0..10] AS expanded
LIMIT $limit;
