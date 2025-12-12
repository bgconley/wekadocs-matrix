-- impact_assessment: Analyze mentions and relationships of configurations
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $config_name, $max_hops (default: 3)
-- Returns: related entities via MENTIONS relationship
-- Guardrails: max_hops bounded, typed relationships only
-- Phase 2 Cleanup: Replaced AFFECTS/CRITICAL_FOR with MENTIONS (active relationships)

-- Schema:
-- Input: {config_name: str, max_hops?: int}
-- Output: [{config: Node, mentioned_by: [Section], related_count: int}]

-- Version 1: Configuration mentions analysis
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH (cfg)<-[:MENTIONS]-(sec:Section)
WITH cfg, collect(DISTINCT sec) AS mentioned_by_sections
RETURN cfg, mentioned_by_sections,
  size(mentioned_by_sections) AS mention_count,
  CASE
    WHEN size(mentioned_by_sections) = 0 THEN 'NONE'
    WHEN size(mentioned_by_sections) <= 3 THEN 'LOW'
    WHEN size(mentioned_by_sections) <= 10 THEN 'MEDIUM'
    ELSE 'HIGH'
  END AS mention_level
LIMIT 1;

-- Version 2: Configuration with related entities via sections
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH (cfg)<-[:MENTIONS]-(sec:Section)-[:MENTIONS]->(related)
WHERE cfg.id <> related.id
WITH cfg, collect(DISTINCT {
  node: related,
  labels: labels(related),
  section: sec.id
}) AS related_entities
RETURN cfg, related_entities,
  size(related_entities) AS related_count
LIMIT 1;

-- Version 3: Configuration with command associations
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH (cfg)<-[:MENTIONS]-(sec:Section)
OPTIONAL MATCH (sec)-[:MENTIONS]->(cmd:Command)
WITH cfg,
  collect(DISTINCT sec) AS sections,
  collect(DISTINCT cmd) AS commands
RETURN cfg, sections, commands,
  size(sections) AS section_count,
  size(commands) AS command_count
LIMIT 1;
