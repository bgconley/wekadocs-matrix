-- traverse_relationships: Explore relationships from a starting node
-- Parameters: $start_id, $rel_types[], $max_depth, $limit
-- Returns: paths from start node following specified relationship types

-- Version 1: Simple traversal with type filter
MATCH (start {id: $start_id})
MATCH path=(start)-[r*1..$max_depth]->(target)
WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
WITH path, target, length(path) AS depth
RETURN path, target, depth, labels(target) AS target_labels
ORDER BY depth ASC
LIMIT $limit;

-- Version 2: Bidirectional traversal
MATCH (start {id: $start_id})
MATCH path=(start)-[r*1..$max_depth]-(target)
WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
WITH path, target, length(path) AS depth
RETURN path, target, depth, labels(target) AS target_labels
ORDER BY depth ASC
LIMIT $limit;

-- Version 3: Traversal with relationship metadata
MATCH (start {id: $start_id})
MATCH path=(start)-[r*1..$max_depth]->(target)
WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
WITH path, target, relationships(path) AS rels, length(path) AS depth
RETURN path, target, depth,
  [rel IN rels | {type: type(rel), confidence: rel.confidence, source: rel.source_section_id}] AS rel_metadata
ORDER BY depth ASC
LIMIT $limit;
