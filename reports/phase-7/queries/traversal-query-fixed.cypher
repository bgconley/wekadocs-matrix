// Corrected query - using properties() instead of map comprehension

UNWIND ['998846de98e2dea9250161f25c1b28f3087052d075d7127d0bd18545578073fa'] AS start_id
MATCH (start {id: start_id})

// Return starting nodes at distance 0
WITH start, start_id
CALL {
    WITH start
    RETURN start.id AS id,
           labels(start)[0] AS label,
           properties(start) AS props,
           0 AS dist,
           [] AS sample_paths
}

UNION

// Return reachable nodes at distance 1..max_depth
UNWIND ['998846de98e2dea9250161f25c1b28f3087052d075d7127d0bd18545578073fa'] AS start_id
MATCH (start {id: start_id})
MATCH path=(start)-[r:MENTIONS|HAS_SECTION|CONTAINS_STEP*1..2]->(target)
WITH DISTINCT target, min(length(path)) AS dist,
     collect(DISTINCT path)[0..10] AS sample_paths
WHERE dist <= 2
RETURN target.id AS id,
       labels(target)[0] AS label,
       properties(target) AS props,
       dist,
       sample_paths

ORDER BY dist ASC
LIMIT 200
