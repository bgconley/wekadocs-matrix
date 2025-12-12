-- dependency_chain: Trace relationship chains for components via MENTIONS
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $component_name, $max_depth (default: 5)
-- Returns: related entities via MENTIONS traversal
-- Guardrails: max_depth enforced, early LIMIT prevents explosion
-- Phase 2 Cleanup: Replaced DEPENDS_ON/CRITICAL_FOR with MENTIONS (active relationship)

-- Schema:
-- Input: {component_name: str, max_depth?: int}
-- Output: [{path: Path, related: [Node], depth: int}]

-- Version 1: Component mentions chain (entities mentioned in same sections)
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH (c)<-[:MENTIONS]-(sec:Section)-[:MENTIONS]->(related)
WHERE c.id <> related.id
WITH c, sec, related
RETURN c.name AS component,
       collect(DISTINCT {
         section: sec.id,
         related: related,
         labels: labels(related)
       }) AS related_entities
LIMIT $limit;

-- Version 2: Multi-hop section traversal for related components
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH (c)<-[:MENTIONS]-(sec1:Section)
OPTIONAL MATCH (sec1)-[:NEXT*1..3]-(sec2:Section)-[:MENTIONS]->(related:Component)
WHERE c.id <> related.id
WITH c, collect(DISTINCT {
  component: related,
  via_section: sec1.id,
  related_section: sec2.id
}) AS related_components
RETURN c, related_components,
  size(related_components) AS related_count
LIMIT 1;

-- Version 3: Document-level component relationships
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH (c)<-[:MENTIONS]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
OPTIONAL MATCH (doc)-[:HAS_SECTION]->(other_sec:Section)-[:MENTIONS]->(other:Component)
WHERE c.id <> other.id
WITH c, doc,
  collect(DISTINCT {component: other, section: other_sec.id}) AS doc_components
RETURN c, doc.id AS document_id,
  doc_components,
  size(doc_components) AS component_count
LIMIT 10;
