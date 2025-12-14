-- troubleshooting_path: Advanced error troubleshooting with full resolution paths
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $error_code, $error_name?, $include_related (default: true)
-- Returns: complete troubleshooting paths with procedures, steps, commands, and related configs
-- Guardrails: depth limited to 3 hops, typed relationships only

-- Schema:
-- Input: {error_code: str, error_name?: str, include_related?: bool}
-- Output: [{error: Node, procedures: [Node], steps: [Node], commands: [Node], configs: [Node], concepts: [Node]}]

-- Version 1: Complete resolution path
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH path=(e)<-[:RESOLVES]-(proc:Procedure)-[:CONTAINS_STEP*1..3]->(step:Step)
OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
WITH e, proc, step, cmd, path
ORDER BY step.order ASC
WITH e, proc, collect(DISTINCT {
  step: step,
  order: step.order,
  command: cmd,
  path_length: length(path)
}) AS ordered_steps
RETURN e, proc, ordered_steps,
  size(ordered_steps) AS step_count
ORDER BY proc.name
LIMIT 10;

-- Version 2: Troubleshooting with related configurations via MENTIONS
-- Phase 2 Cleanup: Replaced RELATED_TO with MENTIONS (active relationship)
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
OPTIONAL MATCH (proc)-[:CONTAINS_STEP]->(step:Step)
OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
OPTIONAL MATCH (error_sec:Chunk)-[:MENTIONS]->(e)
OPTIONAL MATCH (error_sec)-[:MENTIONS]->(cfg:Configuration)
OPTIONAL MATCH (cfg)<-[:HAS_PARAMETER]-(param_cmd:Command)
WITH e, proc,
  collect(DISTINCT {step: step, order: step.order, command: cmd}) AS steps,
  collect(DISTINCT {config: cfg, used_by_commands: collect(DISTINCT param_cmd.name)}) AS related_configs
RETURN e, proc, steps, related_configs,
  size(steps) AS step_count,
  size(related_configs) AS config_count
ORDER BY proc.name
LIMIT 10;

-- Version 3: Full troubleshooting context with chunks and concepts
-- Phase 2 Cleanup: Replaced RELATED_TO with MENTIONS (active relationship)
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH (error_sec:Chunk)-[:MENTIONS]->(e)
OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
OPTIONAL MATCH (proc_sec:Chunk)-[:MENTIONS]->(proc)
OPTIONAL MATCH (proc)-[:CONTAINS_STEP]->(step:Step)
OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
OPTIONAL MATCH (error_sec)-[:MENTIONS]->(cfg:Configuration)
OPTIONAL MATCH (error_sec)-[:MENTIONS]->(concept:Concept)
WITH e,
  collect(DISTINCT error_sec)[0..3] AS error_chunks,
  collect(DISTINCT {
    procedure: proc,
    chunk: proc_sec,
    steps: collect(DISTINCT {step: step, order: step.order, command: cmd})
  }) AS resolution_paths,
  collect(DISTINCT cfg) AS related_configs,
  collect(DISTINCT concept) AS related_concepts
RETURN e, error_chunks, resolution_paths, related_configs, related_concepts,
  size(resolution_paths) AS resolution_count,
  size(related_configs) AS config_count,
  size(related_concepts) AS concept_count
LIMIT 1;
