"""
Entity normalization utilities for Neo4j graph.

This module implements entity hygiene from the Neo4j Overhaul Plan (Section 6).
Proper entity normalization is critical for:

- Deterministic entity linking at query time
- Cross-document entity matching
- Reducing duplicate entities
- Improving graph traversal quality

The normalization strategy:
1. normalized_name = lower(trim(name))
2. Index on (entity_type, normalized_name) for fast lookup
3. Query-time linking prefers most-connected entities on collision

Usage:
    from src.neo.entity_normalization import normalize_entities_backfill

    # One-time backfill
    stats = normalize_entities_backfill(session)

    # In ingestion (always set normalized_name)
    entity["normalized_name"] = normalize_entity_name(entity["name"])
"""

from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for consistent matching.

    The normalization is intentionally simple and deterministic:
    - Lowercase
    - Strip whitespace

    This matches the Cypher expression: toLower(trim(name))

    Args:
        name: Raw entity name

    Returns:
        Normalized name
    """
    if not name:
        return ""
    return name.lower().strip()


def normalize_entities_backfill(
    session,
    *,
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    One-time backfill of normalized_name for all entities.

    Sets normalized_name = toLower(trim(name)) for entities where it's missing.

    Args:
        session: Neo4j session
        batch_size: Batch size for updates

    Returns:
        Dict with stats: {"updated": n, "already_normalized": n}
    """
    # First, check how many need normalization
    count_query = """
    MATCH (e:Entity)
    WHERE e.name IS NOT NULL AND e.normalized_name IS NULL
    RETURN count(e) AS count
    """
    result = session.run(count_query)
    record = result.single()
    needs_normalization = record["count"] if record else 0

    if needs_normalization == 0:
        logger.info("entity_normalization_skipped", reason="all_already_normalized")
        return {"updated": 0, "already_normalized": True}

    # Perform the backfill
    update_query = """
    MATCH (e:Entity)
    WHERE e.name IS NOT NULL AND e.normalized_name IS NULL
    SET e.normalized_name = toLower(trim(e.name))
    RETURN count(e) AS updated
    """
    result = session.run(update_query)
    record = result.single()
    updated = record["updated"] if record else 0

    logger.info(
        "entity_normalization_complete",
        entities_normalized=updated,
        total_needing_normalization=needs_normalization,
    )

    return {"updated": updated, "already_normalized": False}


def ensure_entity_index(session) -> bool:
    """
    Ensure the composite index on (entity_type, normalized_name) exists.

    This index is critical for fast entity lookup at query time.

    Args:
        session: Neo4j session

    Returns:
        True if index was created or already exists
    """
    query = """
    CREATE RANGE INDEX entity_type_normalized_name IF NOT EXISTS
    FOR (e:Entity) ON (e.entity_type, e.normalized_name)
    """
    try:
        session.run(query)
        logger.info("entity_index_ensured", index_name="entity_type_normalized_name")
        return True
    except Exception as e:
        logger.warning("entity_index_creation_failed", error=str(e))
        return False


def find_entity_by_normalized_name(
    session,
    normalized_name: str,
    entity_type: Optional[str] = None,
    *,
    prefer_most_connected: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Find an entity by normalized name with deterministic selection.

    Implements the query-time entity linking policy from Section 6.3:
    1. Try (entity_type, normalized_name)
    2. If type is missing, try normalized_name only
    3. If multiple hits, prefer entity with most MENTIONS support

    Args:
        session: Neo4j session
        normalized_name: Pre-normalized entity name
        entity_type: Optional entity type to filter by
        prefer_most_connected: If True, prefer entity with most mentions

    Returns:
        Entity dict if found, None otherwise
    """
    if entity_type:
        # Typed lookup (fastest, most precise)
        query = """
        MATCH (e:Entity {entity_type: $entity_type, normalized_name: $normalized_name})
        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
        WITH e, count(c) AS mention_count
        RETURN e {.*, mention_count: mention_count} AS entity
        ORDER BY mention_count DESC
        LIMIT 1
        """
        result = session.run(
            query,
            entity_type=entity_type,
            normalized_name=normalized_name,
        )
    else:
        # Untyped lookup (may return multiple entities of different types)
        query = """
        MATCH (e:Entity {normalized_name: $normalized_name})
        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
        WITH e, count(c) AS mention_count
        RETURN e {.*, mention_count: mention_count} AS entity
        ORDER BY mention_count DESC
        LIMIT 1
        """
        result = session.run(query, normalized_name=normalized_name)

    record = result.single()
    return dict(record["entity"]) if record else None


def find_entities_by_normalized_names(
    session,
    normalized_names: List[str],
    *,
    entity_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch lookup of entities by normalized names.

    Optimized for retrieving multiple entities at once.

    Args:
        session: Neo4j session
        normalized_names: List of normalized entity names
        entity_type: Optional entity type filter

    Returns:
        Dict mapping normalized_name → entity dict
    """
    if not normalized_names:
        return {}

    if entity_type:
        query = """
        UNWIND $names AS name
        MATCH (e:Entity {entity_type: $entity_type, normalized_name: name})
        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
        WITH name, e, count(c) AS mention_count
        ORDER BY mention_count DESC
        WITH name, head(collect(e {.*, mention_count: mention_count})) AS entity
        WHERE entity IS NOT NULL
        RETURN name, entity
        """
        result = session.run(
            query,
            names=normalized_names,
            entity_type=entity_type,
        )
    else:
        query = """
        UNWIND $names AS name
        MATCH (e:Entity {normalized_name: name})
        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
        WITH name, e, count(c) AS mention_count
        ORDER BY mention_count DESC
        WITH name, head(collect(e {.*, mention_count: mention_count})) AS entity
        WHERE entity IS NOT NULL
        RETURN name, entity
        """
        result = session.run(query, names=normalized_names)

    return {record["name"]: dict(record["entity"]) for record in result}


def get_entity_normalization_stats(session) -> Dict[str, Any]:
    """
    Get statistics about entity normalization status.

    Useful for monitoring and debugging.

    Returns:
        Dict with stats about normalized vs unnormalized entities
    """
    query = """
    MATCH (e:Entity)
    WITH
        count(e) AS total,
        sum(CASE WHEN e.normalized_name IS NOT NULL THEN 1 ELSE 0 END) AS normalized,
        sum(CASE WHEN e.normalized_name IS NULL AND e.name IS NOT NULL THEN 1 ELSE 0 END) AS needs_normalization,
        sum(CASE WHEN e.name IS NULL THEN 1 ELSE 0 END) AS missing_name
    RETURN total, normalized, needs_normalization, missing_name,
           CASE WHEN total > 0 THEN normalized * 1.0 / total ELSE 1.0 END AS normalization_rate
    """
    result = session.run(query)
    record = result.single()
    if record:
        return dict(record)
    return {
        "total": 0,
        "normalized": 0,
        "needs_normalization": 0,
        "missing_name": 0,
        "normalization_rate": 1.0,
    }
