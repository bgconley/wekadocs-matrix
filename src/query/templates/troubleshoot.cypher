-- troubleshoot_error: Find procedures and steps to resolve errors
-- Parameters: $error_code, $error_name, $limit
-- Returns: errors with resolution procedures and steps

-- Version 1: Error to procedure resolution
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
OPTIONAL MATCH (proc)-[:CONTAINS_STEP {order}]->(step:Step)
OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
RETURN e, proc, step, cmd
ORDER BY step.order ASC
LIMIT $limit;

-- Version 2: Error resolution with context sections
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH (sec:Section)-[:MENTIONS]->(e)
OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
OPTIONAL MATCH (proc)-[:CONTAINS_STEP]->(step:Step)
WITH e, collect(DISTINCT sec) AS mentioning_sections,
     collect(DISTINCT {proc: proc, step: step, order: step.order}) AS resolution_steps
RETURN e, mentioning_sections, resolution_steps
LIMIT $limit;

-- Version 3: Error with related configurations
MATCH (e:Error)
WHERE e.code = $error_code OR e.name = $error_name
OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
OPTIONAL MATCH (e)-[:RELATED_TO]->(cfg:Configuration)
OPTIONAL MATCH (e)-[:RELATED_TO]->(concept:Concept)
RETURN e, proc, collect(DISTINCT cfg) AS related_configs,
       collect(DISTINCT concept) AS related_concepts
LIMIT $limit;
