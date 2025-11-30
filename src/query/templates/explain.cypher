-- explain_architecture: Explain system architecture or component relationships
-- Parameters: $component_name, $max_depth, $limit
-- Returns: component with its dependencies and architecture context
-- Phase 2 Cleanup: Removed patterns using dead relationship types (DEPENDS_ON,
-- REQUIRES, AFFECTS, RELATED_TO). These were never materialized by ingestion.

-- Version 1: Component with structure (using active relationships)
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH (c)-[:MENTIONS]->(entity)
WITH c, collect(DISTINCT {
  entity: entity,
  labels: labels(entity)
}) AS related_entities
RETURN c, related_entities
LIMIT $limit;

-- Version 2: Concept explanation via MENTIONS (active relationship)
MATCH (concept:Concept)
WHERE concept.term = $concept_term OR concept.name = $concept_term
OPTIONAL MATCH (concept)<-[:MENTIONS]-(sec:Section)
RETURN concept,
       collect(DISTINCT {section_id: sec.id, title: sec.title, document_id: sec.document_id}) AS mentioned_in
LIMIT $limit;
