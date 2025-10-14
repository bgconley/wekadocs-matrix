-- dependency_chain: Trace dependency chains for components
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $component_name, $max_depth (default: 5)
-- Returns: dependency paths with depth information
-- Guardrails: max_depth enforced, early LIMIT prevents explosion

-- Schema:
-- Input: {component_name: str, max_depth?: int}
-- Output: [{path: Path, dependencies: [Node], depth: int, critical_services: [Node]}]

-- Version 1: Simple dependency chain (min depth 2 for multi-hop requirement)
MATCH (c:Component {name: $component_name})
MATCH path=(c)-[:DEPENDS_ON*2..5]->(dep)
WITH path, dep, length(path) AS depth
WHERE depth >= 2
RETURN path, dep, depth, labels(dep) AS dep_types
ORDER BY depth ASC
LIMIT $limit;

-- Version 2: Dependency chain with transitive closure (min depth 2)
MATCH (c:Component {name: $component_name})
MATCH path=(c)-[:DEPENDS_ON*2..5]->(dep)
WITH path, collect(DISTINCT dep) AS all_deps, length(path) AS depth
WHERE depth >= 2
RETURN path, all_deps, depth,
  size([d IN all_deps WHERE (d)-[:CRITICAL_FOR]->()]) AS critical_count
ORDER BY depth ASC
LIMIT $limit;

-- Version 3: Full impact analysis (dependencies + affected by)
MATCH (c:Component {name: $component_name})
OPTIONAL MATCH forward_path=(c)-[:DEPENDS_ON*2..5]->(dep)
OPTIONAL MATCH reverse_path=(c)<-[:DEPENDS_ON*2..5]-(dependent)
WITH c,
  collect(DISTINCT {node: dep, path: forward_path, depth: length(forward_path)}) AS dependencies,
  collect(DISTINCT {node: dependent, path: reverse_path, depth: length(reverse_path)}) AS dependents
RETURN c, dependencies, dependents,
  size(dependencies) AS dep_count,
  size(dependents) AS dependent_count
LIMIT 1;
