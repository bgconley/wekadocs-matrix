// Simplified query - using OPTIONAL MATCH and COALESCE

UNWIND ['998846de98e2dea9250161f25c1b28f3087052d075d7127d0bd18545578073fa'] AS start_id
MATCH (start {id: start_id})
OPTIONAL MATCH path=(start)-[r:MENTIONS|HAS_SECTION|CONTAINS_STEP*1..2]->(target)
WITH start,
     COALESCE(target, start) AS node,
     CASE WHEN target IS NULL THEN 0 ELSE min(length(path)) END AS dist,
     CASE WHEN target IS NULL THEN [] ELSE collect(DISTINCT path)[0..10] END AS sample_paths
WITH DISTINCT node, min(dist) AS distance, sample_paths
WHERE distance <= 2
RETURN node.id AS id,
       labels(node)[0] AS label,
       properties(node) AS props,
       distance AS dist,
       sample_paths
ORDER BY dist ASC
LIMIT 200
