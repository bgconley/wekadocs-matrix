"""
Atomic structural edge building for Neo4j graph.

This module provides transaction-aware structural edge building that integrates
directly with the atomic ingestion saga. Unlike the standalone structural_builder,
these functions are designed to be called WITHIN an active Neo4j transaction.

Key design principles:
- Transaction-aware: Uses tx.run() for atomic operation within saga
- Document-scoped: All operations bounded to single document
- Fail-fast: Raises exceptions on failure to trigger saga rollback
- Focused: ~300 lines, single responsibility

Usage within atomic.py:
    from src.ingestion.structural_edges import build_structural_edges_in_tx

    # Inside _execute_atomic_saga(), after chunk writes:
    structural_stats = build_structural_edges_in_tx(neo4j_tx, document_id)
    stats["structural_edges"] = structural_stats
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StructuralEdgeStats:
    """Statistics from structural edge building."""

    parent_path_normalized: int = 0
    parent_chunk_id_computed: int = 0
    edges_cleared: int = 0
    next_chunk_created: int = 0
    parent_heading_created: int = 0
    child_of_created: int = 0
    parent_of_created: int = 0
    next_sibling_created: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for stats reporting."""
        return {
            "parent_path_normalized": self.parent_path_normalized,
            "parent_chunk_id_computed": self.parent_chunk_id_computed,
            "edges_cleared": self.edges_cleared,
            "NEXT_CHUNK": self.next_chunk_created,
            "PARENT_HEADING": self.parent_heading_created,
            "CHILD_OF": self.child_of_created,
            "PARENT_OF": self.parent_of_created,
            "NEXT": self.next_sibling_created,
        }


@dataclass
class StructuralEdgeResult:
    """Result of atomic structural edge building."""

    document_id: str
    success: bool = True
    stats: StructuralEdgeStats = field(default_factory=StructuralEdgeStats)
    warnings: List[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saga stats."""
        return {
            "success": self.success,
            "stats": self.stats.to_dict(),
            "warnings": self.warnings,
            "error": self.error,
        }


def build_structural_edges_in_tx(
    tx,
    document_id: str,
    *,
    skip_has_chunk: bool = True,
) -> Dict[str, Any]:
    """
    Build structural edges for a document within an active transaction.

    This function is designed to be called from _execute_atomic_saga() AFTER
    chunk nodes have been written but BEFORE Qdrant writes. If it fails,
    the saga transaction will roll back.

    Args:
        tx: Active Neo4j transaction (from session.begin_transaction())
        document_id: Document ID to build edges for
        skip_has_chunk: Skip HAS_CHUNK creation (already done in saga)

    Returns:
        Dict with stats and any warnings for inclusion in saga stats

    Raises:
        Exception: On failure, to trigger saga rollback
    """
    result = StructuralEdgeResult(document_id=document_id)

    try:
        # Step 1: Normalize parent_path → parent_path_norm
        result.stats.parent_path_normalized = _normalize_parent_path(tx, document_id)

        # Step 2: Compute parent_chunk_id from parent_path_norm
        result.stats.parent_chunk_id_computed = _compute_parent_chunk_id(
            tx, document_id
        )

        # Step 3: Clear existing structural edges (idempotent rebuild)
        result.stats.edges_cleared = _clear_structural_edges(tx, document_id)

        # Step 4: Create NEXT_CHUNK edges (document-wide sequence)
        result.stats.next_chunk_created = _create_next_chunk_edges(tx, document_id)

        # Step 5: Create hierarchy edges (PARENT_HEADING, CHILD_OF, PARENT_OF)
        hierarchy = _create_hierarchy_edges(tx, document_id)
        result.stats.parent_heading_created = hierarchy["parent_heading"]
        result.stats.child_of_created = hierarchy["child_of"]
        result.stats.parent_of_created = hierarchy["parent_of"]

        # Step 6: Create NEXT edges (sibling adjacency)
        result.stats.next_sibling_created = _create_next_sibling_edges(tx, document_id)

        # Step 7: Validate invariants (warnings only, don't fail)
        result.warnings = _validate_structural_invariants(tx, document_id, result.stats)

        logger.debug(
            "structural_edges_built_in_tx",
            document_id=document_id,
            stats=result.stats.to_dict(),
            warnings=result.warnings,
        )

        return result.to_dict()

    except Exception as e:
        logger.error(
            "structural_edge_building_failed",
            document_id=document_id,
            error=str(e),
        )
        # Re-raise to trigger saga rollback
        raise


def _normalize_parent_path(tx, document_id: str) -> int:
    """
    Normalize parent_path into parent_path_norm.

    Steps:
    1. Strip HTML tags (e.g., <a href="...">...</a>) that contaminate paths
    2. Split by ' > ' separator
    3. Trim each part and rejoin with ' > '

    This two-phase approach handles HTML anchor tags in heading titles
    that would otherwise be mistaken for path separators.
    """
    import re

    # Phase 1: Fetch chunks with parent_path, strip HTML in Python
    fetch_query = """
    MATCH (c:Chunk {document_id: $document_id})
    WHERE c.parent_path IS NOT NULL AND c.parent_path <> ''
    RETURN c.chunk_id AS chunk_id, c.parent_path AS parent_path, c.parent_path_norm AS current_norm
    """
    result = tx.run(fetch_query, document_id=document_id)
    records = list(result)

    if not records:
        return 0

    # HTML tag pattern - matches <tag ...> and </tag>
    html_pattern = re.compile(r"<[^>]+>")

    updates = []
    for record in records:
        chunk_id = record["chunk_id"]
        parent_path = record["parent_path"]
        current_norm = record["current_norm"]

        # Strip HTML tags
        cleaned = html_pattern.sub("", parent_path)

        # Normalize: split by ' > ' or '>', trim parts, rejoin with ' > '
        # Handle both ' > ' and bare '>' separators
        parts = re.split(r"\s*>\s*", cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        norm = " > ".join(parts)

        # Only update if changed
        if current_norm != norm:
            updates.append({"chunk_id": chunk_id, "norm": norm})

    if not updates:
        return 0

    # Phase 2: Batch update normalized paths
    update_query = """
    UNWIND $updates AS upd
    MATCH (c:Chunk {chunk_id: upd.chunk_id})
    SET c.parent_path_norm = upd.norm
    RETURN count(c) AS updated
    """
    result = tx.run(update_query, updates=updates)
    record = result.single()
    return record["updated"] if record else 0


def _compute_parent_chunk_id(tx, document_id: str) -> int:
    """
    Compute parent_chunk_id from parent_path_norm.

    The parent's path is the breadcrumb without the last segment.

    P2: Relaxed order constraint - prefers parent.order < child.order (normal case)
    but falls back to any matching parent if none found (handles edge cases where
    heading chunks are created after their content chunks).

    Priority:
    1. Matching parent with order < child.order (nearest previous)
    2. Matching parent with order > child.order (fallback - nearest following)
    """
    query = """
    MATCH (child:Chunk {document_id: $document_id})
    WHERE child.parent_path_norm IS NOT NULL
      AND child.parent_path_norm CONTAINS ' > '
      AND child.parent_chunk_id IS NULL
    WITH child, split(child.parent_path_norm, ' > ') AS parts
    WITH child, parts[0..size(parts)-1] AS parentParts
    WITH child,
         reduce(path = '', p IN parentParts |
           path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
         ) AS parentPathNorm

    // Find all matching parent candidates
    MATCH (parent:Chunk {document_id: $document_id, parent_path_norm: parentPathNorm})
    WHERE child.level IS NULL OR parent.level IS NULL OR parent.level < child.level

    // Categorize and rank candidates: prefer order < child.order, fallback to order > child.order
    WITH child, parent,
         CASE WHEN parent.order < child.order THEN 0 ELSE 1 END AS priority,
         CASE WHEN parent.order < child.order
              THEN child.order - parent.order
              ELSE parent.order - child.order
         END AS distance
    ORDER BY priority ASC, distance ASC

    // Take the best candidate per child
    WITH child, head(collect(parent)) AS parent
    WHERE parent IS NOT NULL
    SET child.parent_chunk_id = parent.chunk_id,
        child.parent_section_id = parent.chunk_id
    RETURN count(child) AS updated
    """
    result = tx.run(query, document_id=document_id)
    record = result.single()
    return record["updated"] if record else 0


def _clear_structural_edges(tx, document_id: str) -> int:
    """
    Clear existing structural edges for idempotent rebuild.

    Only clears edges between chunks within this document.
    """
    query = """
    MATCH (a:Chunk {document_id: $document_id})
          -[r:NEXT_CHUNK|NEXT|PARENT_HEADING|CHILD_OF|PARENT_OF]-
          (b:Chunk {document_id: $document_id})
    DELETE r
    RETURN count(r) AS deleted
    """
    result = tx.run(query, document_id=document_id)
    record = result.single()
    return record["deleted"] if record else 0


def _create_next_chunk_edges(tx, document_id: str) -> int:
    """
    Create NEXT_CHUNK edges for document-wide sequence.

    Direction: (c1:Chunk)-[:NEXT_CHUNK]->(c2:Chunk)
    where c2 immediately follows c1 in document order.
    """
    query = """
    MATCH (c:Chunk {document_id: $document_id})
    WHERE c.order IS NOT NULL
    WITH c ORDER BY c.order
    WITH collect(c) AS chunks
    UNWIND range(0, size(chunks)-2) AS i
    WITH chunks[i] AS c1, chunks[i+1] AS c2
    MERGE (c1)-[r:NEXT_CHUNK]->(c2)
    RETURN count(r) AS created
    """
    result = tx.run(query, document_id=document_id)
    record = result.single()
    return record["created"] if record else 0


def _create_hierarchy_edges(tx, document_id: str) -> Dict[str, int]:
    """
    Create hierarchy edges: PARENT_HEADING, CHILD_OF, PARENT_OF.

    Directions:
    - (child:Chunk)-[:PARENT_HEADING]->(parent:Chunk)
    - (child:Chunk)-[:CHILD_OF]->(parent:Chunk)
    - (parent:Chunk)-[:PARENT_OF]->(child:Chunk)
    """
    query = """
    MATCH (child:Chunk {document_id: $document_id})
    WHERE child.parent_chunk_id IS NOT NULL
    MATCH (parent:Chunk {chunk_id: child.parent_chunk_id})
    MERGE (child)-[ph:PARENT_HEADING]->(parent)
    ON CREATE SET ph.level_delta = coalesce(child.level, 0) - coalesce(parent.level, 0)
    MERGE (child)-[:CHILD_OF]->(parent)
    MERGE (parent)-[:PARENT_OF]->(child)
    RETURN
        count(DISTINCT ph) AS parent_heading,
        count(DISTINCT child) AS child_of,
        count(DISTINCT parent) AS parent_of
    """
    result = tx.run(query, document_id=document_id)
    record = result.single()
    if record:
        return {
            "parent_heading": record["parent_heading"],
            "child_of": record["child_of"],
            "parent_of": record["parent_of"],
        }
    return {"parent_heading": 0, "child_of": 0, "parent_of": 0}


def _create_next_sibling_edges(tx, document_id: str) -> int:
    """
    Create NEXT edges for sibling adjacency within same parent scope.

    Direction: (c1:Chunk)-[:NEXT]->(c2:Chunk)
    where c1 and c2 have the same parent_chunk_id.
    """
    query = """
    MATCH (c:Chunk {document_id: $document_id})
    WHERE c.order IS NOT NULL
    WITH c ORDER BY c.order
    WITH c.parent_chunk_id AS pid, c.level AS lvl, c
    WITH pid, lvl, collect(c) AS chunks
    UNWIND range(0, size(chunks)-2) AS i
    WITH chunks[i] AS c1, chunks[i+1] AS c2
    MERGE (c1)-[r:NEXT]->(c2)
    RETURN count(r) AS created
    """
    result = tx.run(query, document_id=document_id)
    record = result.single()
    return record["created"] if record else 0


def _validate_structural_invariants(
    tx, document_id: str, stats: StructuralEdgeStats
) -> List[str]:
    """
    Validate structural invariants and return warnings.

    Does NOT fail the transaction - returns warnings for logging.

    Checks:
    1. NEXT_CHUNK count sanity (n-1 edges for n chunks)
    2. Hierarchy coverage (chunks with parent_path should have parent_chunk_id)
    """
    warnings = []

    # Get chunk count
    chunk_query = """
    MATCH (c:Chunk {document_id: $document_id})
    RETURN count(c) AS chunk_count
    """
    result = tx.run(chunk_query, document_id=document_id)
    record = result.single()
    chunk_count = record["chunk_count"] if record else 0

    # Check 1: NEXT_CHUNK count sanity
    expected_next_chunk = max(0, chunk_count - 1)
    if chunk_count > 1 and stats.next_chunk_created != expected_next_chunk:
        warnings.append(
            f"NEXT_CHUNK count mismatch: expected {expected_next_chunk}, "
            f"got {stats.next_chunk_created}"
        )

    # Check 2: Hierarchy coverage
    hierarchy_query = """
    MATCH (c:Chunk {document_id: $document_id})
    WHERE c.parent_path_norm CONTAINS ' > '
    WITH count(c) AS should_have_parent
    MATCH (c:Chunk {document_id: $document_id})
    WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NOT NULL
    WITH should_have_parent, count(c) AS has_parent
    RETURN
      should_have_parent,
      has_parent,
      CASE WHEN should_have_parent = 0 THEN 1.0
           ELSE (has_parent * 1.0 / should_have_parent) END AS coverage
    """
    result = tx.run(hierarchy_query, document_id=document_id)
    record = result.single()
    if record:
        coverage = record["coverage"]
        should_have = record["should_have_parent"]
        has_parent = record["has_parent"]
        if should_have > 0 and coverage < 0.9:
            warnings.append(
                f"Low hierarchy coverage: {coverage:.1%} "
                f"({has_parent}/{should_have} chunks with parent)"
            )

    return warnings
