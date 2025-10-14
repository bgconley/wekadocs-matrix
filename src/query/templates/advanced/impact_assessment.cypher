-- impact_assessment: Analyze impact of configuration changes
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $config_name, $max_hops (default: 3)
-- Returns: affected entities with criticality assessment
-- Guardrails: max_hops bounded, typed relationships only

-- Schema:
-- Input: {config_name: str, max_hops?: int}
-- Output: [{config: Node, affected: [Node], critical_services: [Node], impact_level: str}]

-- Version 1: Direct impact analysis
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH (cfg)-[:AFFECTS*1..3]->(affected)
WITH cfg, collect(DISTINCT affected) AS affected_entities
RETURN cfg, affected_entities,
  size(affected_entities) AS impact_count,
  CASE
    WHEN size(affected_entities) = 0 THEN 'NONE'
    WHEN size(affected_entities) <= 3 THEN 'LOW'
    WHEN size(affected_entities) <= 10 THEN 'MEDIUM'
    ELSE 'HIGH'
  END AS impact_level
LIMIT 1;

-- Version 2: Impact with criticality assessment
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH path=(cfg)-[:AFFECTS*1..3]->(affected)
OPTIONAL MATCH (affected)-[:CRITICAL_FOR]->(svc)
WITH cfg, affected, svc, length(path) AS distance
ORDER BY distance ASC
WITH cfg,
  collect(DISTINCT {
    node: affected,
    labels: labels(affected),
    distance: distance,
    is_critical: svc IS NOT NULL,
    critical_for: svc.name
  }) AS impacts
RETURN cfg, impacts,
  size([i IN impacts WHERE i.is_critical]) AS critical_impact_count,
  CASE
    WHEN size([i IN impacts WHERE i.is_critical]) > 0 THEN 'CRITICAL'
    WHEN size(impacts) > 10 THEN 'HIGH'
    WHEN size(impacts) > 3 THEN 'MEDIUM'
    WHEN size(impacts) > 0 THEN 'LOW'
    ELSE 'NONE'
  END AS impact_level
LIMIT 1;

-- Version 3: Full impact with change propagation
MATCH (cfg:Configuration {name: $config_name})
OPTIONAL MATCH direct=(cfg)-[:AFFECTS]->(direct_impact)
OPTIONAL MATCH indirect=(cfg)-[:AFFECTS*2..3]->(indirect_impact)
OPTIONAL MATCH (cfg)<-[:HAS_PARAMETER]-(cmd:Command)
WITH cfg,
  collect(DISTINCT direct_impact) AS direct,
  collect(DISTINCT indirect_impact) AS indirect,
  collect(DISTINCT cmd) AS commands
RETURN cfg, direct, indirect, commands,
  size(direct) AS direct_count,
  size(indirect) AS indirect_count,
  size([d IN direct WHERE (d)-[:CRITICAL_FOR]->()]) AS critical_direct,
  size([i IN indirect WHERE (i)-[:CRITICAL_FOR]->()]) AS critical_indirect
LIMIT 1;
