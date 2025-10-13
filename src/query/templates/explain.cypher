-- explain_architecture: Explain system architecture or component relationships
-- Parameters: $component_name, $max_depth, $limit
-- Returns: component with its dependencies and architecture context

-- Version 1: Component dependencies
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH path=(c)-[:DEPENDS_ON*1..$max_depth]->(dep)
WITH c, path, dep, length(path) AS depth
RETURN c, collect(DISTINCT {
  dependency: dep,
  depth: depth,
  labels: labels(dep)
}) AS dependencies
ORDER BY depth ASC
LIMIT $limit;

-- Version 2: Component with configurations and procedures
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH (c)-[:REQUIRES]->(cfg:Configuration)
OPTIONAL MATCH (c)<-[:AFFECTS]-(proc:Procedure)
RETURN c,
       collect(DISTINCT cfg) AS required_configs,
       collect(DISTINCT proc) AS related_procedures
LIMIT $limit;

-- Version 3: Concept explanation with examples
MATCH (concept:Concept)
WHERE concept.term = $concept_term OR concept.name = $concept_term
OPTIONAL MATCH (concept)<-[:MENTIONS]-(sec:Section)
OPTIONAL MATCH (concept)-[:RELATED_TO]->(ex:Example)
OPTIONAL MATCH (concept)-[:RELATED_TO]->(related:Concept)
RETURN concept,
       collect(DISTINCT {section_id: sec.id, title: sec.title, document_id: sec.document_id}) AS mentioned_in,
       collect(DISTINCT ex) AS examples,
       collect(DISTINCT related) AS related_concepts
LIMIT $limit;
