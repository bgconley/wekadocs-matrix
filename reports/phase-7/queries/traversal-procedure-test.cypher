// Test with procedure-rich section (has 2912 relationships)
UNWIND ['542ba6fa5939508b285a89d38eeff55b8a73118bbb5a69a4218a67c6b107d11d'] AS start_id
MATCH (start {id: start_id})
RETURN start.id AS id,
       labels(start)[0] AS label,
       properties(start) AS props,
       0 AS dist,
       [] AS sample_paths

UNION ALL

UNWIND ['542ba6fa5939508b285a89d38eeff55b8a73118bbb5a69a4218a67c6b107d11d'] AS start_id
MATCH (start {id: start_id})
MATCH path=(start)-[r:MENTIONS|HAS_SECTION|CONTAINS_STEP*1..2]->(target)
WITH DISTINCT target, min(length(path)) AS dist, collect(DISTINCT path)[0..10] AS sample_paths
WHERE dist <= 2
RETURN target.id AS id,
       labels(target)[0] AS label,
       properties(target) AS props,
       dist,
       sample_paths

ORDER BY dist ASC
LIMIT 10
