-- temporal: Query documentation state as of a specific version
-- Phase 4, Task 4.1 - Complex query pattern
-- Parameters: $version, $entity_type?, $entity_name?
-- Returns: entities and relationships valid in specified version
-- Guardrails: version-bounded queries, indexed lookups

-- Schema:
-- Input: {version: str, entity_type?: str, entity_name?: str}
-- Output: [{entity: Node, introduced_in: str, deprecated_in: str?, valid_in_version: bool}]

-- Version 1: Entities introduced/deprecated by version
MATCH (e)
WHERE ($entity_type IS NULL OR $entity_type IN labels(e))
  AND ($entity_name IS NULL OR e.name = $entity_name)
  AND (e.introduced_in IS NULL OR e.introduced_in <= $version)
  AND (e.deprecated_in IS NULL OR e.deprecated_in > $version)
RETURN e, e.introduced_in AS introduced, e.deprecated_in AS deprecated,
  labels(e) AS entity_types
ORDER BY e.introduced_in DESC
LIMIT 100;

-- Version 2: Configuration state at version (Phase 2 Cleanup: removed AFFECTS)
MATCH (cfg:Configuration)
WHERE (cfg.introduced_in IS NULL OR cfg.introduced_in <= $version)
  AND (cfg.deprecated_in IS NULL OR cfg.deprecated_in > $version)
OPTIONAL MATCH (cfg)<-[r:MENTIONS]-(sec:Section)
RETURN cfg, collect(DISTINCT {
  section: sec,
  relationship: type(r)
}) AS mentioned_by
ORDER BY cfg.name
LIMIT 100;

-- Version 3: Full entity evolution tracking
MATCH (e)
WHERE ($entity_type IN labels(e))
  AND ($entity_name IS NULL OR e.name = $entity_name)
WITH e,
  CASE
    WHEN e.introduced_in IS NULL THEN true
    WHEN e.introduced_in <= $version THEN true
    ELSE false
  END AS was_introduced,
  CASE
    WHEN e.deprecated_in IS NULL THEN false
    WHEN e.deprecated_in <= $version THEN true
    ELSE false
  END AS was_deprecated
WHERE was_introduced AND NOT was_deprecated
OPTIONAL MATCH (e)-[r]->(related)
WHERE (r.introduced_in IS NULL OR r.introduced_in <= $version)
  AND (r.deprecated_in IS NULL OR r.deprecated_in > $version)
RETURN e, e.introduced_in, e.deprecated_in, e.updated_at,
  collect(DISTINCT {
    related: related,
    rel_type: type(r),
    labels: labels(related)
  })[0..20] AS related_entities_at_version
ORDER BY e.name
LIMIT 100;
