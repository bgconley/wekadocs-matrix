-- compare_systems: Compare configurations, commands, or components
-- Parameters: $item_a, $item_b, $comparison_type, $limit
-- Returns: comparison between two entities with their properties and relationships
-- Phase 2 Cleanup: Removed AFFECTS patterns (never materialized)

-- Version 1: Compare two configurations via MENTIONS
MATCH (a:Configuration {name: $item_a})
MATCH (b:Configuration {name: $item_b})
OPTIONAL MATCH (a)<-[:MENTIONS]-(sec_a:Section)
OPTIONAL MATCH (b)<-[:MENTIONS]-(sec_b:Section)
RETURN a, b,
       collect(DISTINCT sec_a.id) AS mentioned_in_a,
       collect(DISTINCT sec_b.id) AS mentioned_in_b
LIMIT $limit;

-- Version 2: Compare two commands
MATCH (a:Command {name: $item_a})
MATCH (b:Command {name: $item_b})
OPTIONAL MATCH (a)-[:HAS_PARAMETER]->(param_a:Parameter)
OPTIONAL MATCH (b)-[:HAS_PARAMETER]->(param_b:Parameter)
RETURN a, b,
       collect(DISTINCT {name: param_a.name, description: param_a.description}) AS params_a,
       collect(DISTINCT {name: param_b.name, description: param_b.description}) AS params_b
LIMIT $limit;

-- Version 3: Generic comparison with shared relationships
MATCH (a {id: $item_a_id})
MATCH (b {id: $item_b_id})
OPTIONAL MATCH (a)-[r_a]->(shared)<-[r_b]-(b)
WITH a, b, collect(DISTINCT {
  shared: shared,
  rel_a: type(r_a),
  rel_b: type(r_b),
  labels: labels(shared)
}) AS common_connections
RETURN a, b, common_connections
LIMIT $limit;
