-- comparison: Compare configurations or procedures between versions
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $entity_type, $entity_name_a, $entity_name_b
-- Returns: side-by-side comparison with differences highlighted
-- Guardrails: specific entity types only, limited depth

-- Schema:
-- Input: {entity_type: str, entity_name_a: str, entity_name_b: str}
-- Output: [{entity_a: Node, entity_b: Node, differences: map, common: map}]

-- Version 1: Simple entity comparison
CALL {
  MATCH (a {name: $entity_name_a})
  WHERE $entity_type IN labels(a)
  RETURN a
}
CALL {
  MATCH (b {name: $entity_name_b})
  WHERE $entity_type IN labels(b)
  RETURN b
}
RETURN a, b,
  properties(a) AS props_a,
  properties(b) AS props_b
LIMIT 1;

-- Version 2: Configuration comparison with dependencies
MATCH (a:Configuration {name: $entity_name_a})
MATCH (b:Configuration {name: $entity_name_b})
OPTIONAL MATCH (a)-[:AFFECTS]->(affected_a)
OPTIONAL MATCH (b)-[:AFFECTS]->(affected_b)
OPTIONAL MATCH (a)<-[:HAS_PARAMETER]-(cmd_a:Command)
OPTIONAL MATCH (b)<-[:HAS_PARAMETER]-(cmd_b:Command)
WITH a, b,
  collect(DISTINCT affected_a.id) AS affects_a,
  collect(DISTINCT affected_b.id) AS affects_b,
  collect(DISTINCT cmd_a.name) AS commands_a,
  collect(DISTINCT cmd_b.name) AS commands_b
RETURN a, b,
  {
    only_in_a: [x IN affects_a WHERE NOT x IN affects_b],
    only_in_b: [x IN affects_b WHERE NOT x IN affects_a],
    common: [x IN affects_a WHERE x IN affects_b]
  } AS affected_diff,
  {
    only_in_a: [x IN commands_a WHERE NOT x IN commands_b],
    only_in_b: [x IN commands_b WHERE NOT x IN commands_a],
    common: [x IN commands_a WHERE x IN commands_b]
  } AS command_diff
LIMIT 1;

-- Version 3: Procedure comparison with step-by-step diff
MATCH (proc_a:Procedure {name: $entity_name_a})
MATCH (proc_b:Procedure {name: $entity_name_b})
OPTIONAL MATCH (proc_a)-[:CONTAINS_STEP]->(step_a:Step)
OPTIONAL MATCH (proc_b)-[:CONTAINS_STEP]->(step_b:Step)
WITH proc_a, proc_b,
  collect(DISTINCT {order: step_a.order, text: step_a.text, id: step_a.id}) AS steps_a,
  collect(DISTINCT {order: step_b.order, text: step_b.text, id: step_b.id}) AS steps_b
RETURN proc_a, proc_b, steps_a, steps_b,
  size(steps_a) AS step_count_a,
  size(steps_b) AS step_count_b,
  abs(size(steps_a) - size(steps_b)) AS step_count_diff
LIMIT 1;
