"""
Neo4j Graph Enhancements for Phase 3 (markdown-it-py Integration).

This module provides functions to enhance the Neo4j graph with structural
metadata from the markdown-it-py parser:

1. PARENT_HEADING relationships based on parent_path (heading hierarchy)
2. Structural labels (CodeSection, TableSection) based on has_code/has_table
3. Enhanced metadata properties on Section nodes

Design Philosophy:
- Minimal touchpoints to existing large modules (build_graph.py, atomic.py)
- Idempotent operations (safe to re-run)
- Defensive coding with validation and logging
- Batch processing for efficiency

Usage:
    from src.neo.graph_enhancements import (
        create_parent_heading_relationships,
        apply_structural_labels,
        update_section_enhanced_metadata,
    )

    # After sections are upserted, enhance the graph:
    with driver.session() as session:
        create_parent_heading_relationships(session, document_id, sections)
        apply_structural_labels(session, document_id)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


def parse_parent_path(parent_path: str) -> List[str]:
    """
    Parse a parent_path string into individual heading titles.

    Args:
        parent_path: Heading hierarchy string (e.g., "Getting Started > Installation")

    Returns:
        List of heading titles in order from root to immediate parent

    Example:
        >>> parse_parent_path("Getting Started > Installation > Prerequisites")
        ['Getting Started', 'Installation', 'Prerequisites']
    """
    if not parent_path or not parent_path.strip():
        return []

    # Split on " > " separator (with spaces around arrow)
    parts = [p.strip() for p in parent_path.split(" > ")]
    # Filter out empty strings
    return [p for p in parts if p]


def get_immediate_parent_title(parent_path: str) -> Optional[str]:
    """
    Extract the immediate parent heading title from parent_path.

    Args:
        parent_path: Heading hierarchy string

    Returns:
        Immediate parent title, or None if no parent

    Example:
        >>> get_immediate_parent_title("Getting Started > Installation")
        'Installation'
    """
    parts = parse_parent_path(parent_path)
    return parts[-1] if parts else None


def create_parent_heading_relationships(
    session,
    document_id: str,
    sections: Optional[List[Dict]] = None,
    batch_size: int = 100,
) -> Dict[str, int]:
    """
    Create PARENT_HEADING relationships based on parent_path hierarchy.

    This builds heading-based parent-child relationships that differ from
    PARENT_OF (which uses parent_section_id). PARENT_HEADING captures the
    markdown heading hierarchy explicitly.

    Relationship direction: (child:Chunk)-[:PARENT_HEADING]->(parent:Chunk)
    This follows the Neo4j best practice of child→parent direction for hierarchies.

    Args:
        session: Neo4j session
        document_id: Document identifier
        sections: Optional list of section dicts with parent_path
                 If None, queries sections directly from Neo4j
        batch_size: Batch size for relationship creation

    Returns:
        Dict with stats: {"matched": n, "created": n, "errors": n}
    """
    stats = {"matched": 0, "created": 0, "errors": 0, "skipped_no_parent": 0}

    # Strategy: For each chunk with a parent_path, find the immediate parent
    # by matching the last element of parent_path to a chunk title at level-1

    query = """
    // Find chunks with parent_path that don't already have PARENT_HEADING
    MATCH (child:Chunk)
    WHERE child.document_id = $document_id
      AND child.parent_path IS NOT NULL
      AND child.parent_path <> ''
      AND NOT EXISTS {
        MATCH (child)-[:PARENT_HEADING]->(:Chunk)
      }

    // Find the parent chunk by matching:
    // 1. Same document
    // 2. Parent's title matches last element of child's parent_path
    // 3. Parent's level is less than child's level
    // 4. Parent's order is less than child's order (parent comes before child)
    WITH child,
         split(child.parent_path, ' > ') AS path_parts,
         child.level AS child_level,
         child.order AS child_order
    WHERE size(path_parts) > 0

    WITH child,
         path_parts[size(path_parts) - 1] AS immediate_parent_title,
         child_level,
         child_order

    MATCH (parent:Chunk)
    WHERE parent.document_id = $document_id
      AND parent.title = immediate_parent_title
      AND parent.level < child_level
      AND parent.order < child_order

    // Take the closest parent (highest order that's still before child)
    WITH child, parent
    ORDER BY parent.order DESC
    WITH child, collect(parent)[0] AS parent
    WHERE parent IS NOT NULL

    // Create the relationship
    MERGE (child)-[r:PARENT_HEADING]->(parent)
    SET r.level_delta = child.level - parent.level,
        r.created_at = datetime()

    RETURN count(r) AS created, count(child) AS matched
    """

    try:
        result = session.run(query, document_id=document_id)
        record = result.single()
        if record:
            stats["matched"] = record.get("matched", 0)
            stats["created"] = record.get("created", 0)

        logger.info(
            "PARENT_HEADING relationships created",
            document_id=document_id,
            matched=stats["matched"],
            created=stats["created"],
        )
    except Exception as exc:
        stats["errors"] = 1
        logger.warning(
            "Failed to create PARENT_HEADING relationships",
            document_id=document_id,
            error=str(exc),
        )

    return stats


def apply_structural_labels(
    session,
    document_id: str,
) -> Dict[str, int]:
    """
    Apply structural labels (CodeChunk, TableChunk) based on content flags.

    Adds secondary labels to Chunk nodes based on:
    - :CodeChunk for chunks with has_code = true
    - :TableChunk for chunks with has_table = true

    These labels enable fast filtering in Cypher queries without property checks.

    Args:
        session: Neo4j session
        document_id: Document identifier

    Returns:
        Dict with stats: {"code_chunks": n, "table_chunks": n}
    """
    stats = {"code_chunks": 0, "table_chunks": 0, "errors": 0}

    # Add :CodeChunk label
    code_query = """
    MATCH (c:Chunk)
    WHERE c.document_id = $document_id
      AND c.has_code = true
      AND NOT c:CodeChunk
    SET c:CodeChunk
    RETURN count(c) AS labeled
    """

    # Add :TableChunk label
    table_query = """
    MATCH (c:Chunk)
    WHERE c.document_id = $document_id
      AND c.has_table = true
      AND NOT c:TableChunk
    SET c:TableChunk
    RETURN count(c) AS labeled
    """

    try:
        # Apply CodeChunk labels
        code_result = session.run(code_query, document_id=document_id)
        code_record = code_result.single()
        if code_record:
            stats["code_chunks"] = code_record.get("labeled", 0)

        # Apply TableChunk labels
        table_result = session.run(table_query, document_id=document_id)
        table_record = table_result.single()
        if table_record:
            stats["table_chunks"] = table_record.get("labeled", 0)

        logger.info(
            "Structural labels applied",
            document_id=document_id,
            code_chunks=stats["code_chunks"],
            table_chunks=stats["table_chunks"],
        )
    except Exception as exc:
        stats["errors"] = 1
        logger.warning(
            "Failed to apply structural labels",
            document_id=document_id,
            error=str(exc),
        )

    return stats


def update_section_enhanced_metadata(
    session,
    document_id: str,
    sections: List[Dict],
    batch_size: int = 100,
) -> Dict[str, int]:
    """
    Update Chunk nodes with enhanced metadata from markdown-it-py parser.

    Adds Phase 2 metadata fields to existing Chunk nodes:
    - line_start, line_end: Source line numbers
    - parent_path: Heading hierarchy string
    - block_types: List of block types in chunk
    - code_ratio: Fraction of code content
    - has_code, has_table: Structural flags

    This is designed to be called after chunks are initially upserted,
    allowing the core _upsert_sections to remain backward compatible.

    Args:
        session: Neo4j session
        document_id: Document identifier
        sections: List of chunk dicts with enhanced metadata
        batch_size: Batch size for updates

    Returns:
        Dict with stats: {"updated": n, "skipped": n, "errors": n}
    """
    stats = {"updated": 0, "skipped": 0, "errors": 0}

    # Filter chunks that have enhanced metadata
    sections_with_metadata = []
    for section in sections:
        # Check if chunk has any enhanced metadata fields
        has_enhanced = any(
            section.get(field) is not None
            for field in ["line_start", "parent_path", "block_types", "has_code"]
        )
        if has_enhanced:
            sections_with_metadata.append(section)
        else:
            stats["skipped"] += 1

    if not sections_with_metadata:
        logger.debug(
            "No chunks with enhanced metadata to update",
            document_id=document_id,
            skipped=stats["skipped"],
        )
        return stats

    # Batch update query
    query = """
    UNWIND $sections AS sec
    MATCH (c:Chunk {id: sec.id})
    WHERE c.document_id = $document_id
    SET c.line_start = sec.line_start,
        c.line_end = sec.line_end,
        c.parent_path = sec.parent_path,
        c.block_types = sec.block_types,
        c.code_ratio = sec.code_ratio,
        c.has_code = sec.has_code,
        c.has_table = sec.has_table
    RETURN count(c) AS updated
    """

    # Process in batches
    for i in range(0, len(sections_with_metadata), batch_size):
        batch = sections_with_metadata[i : i + batch_size]

        # Prepare batch data with defaults
        batch_data = []
        for section in batch:
            batch_data.append(
                {
                    "id": section.get("id"),
                    "line_start": section.get("line_start"),
                    "line_end": section.get("line_end"),
                    "parent_path": section.get("parent_path", ""),
                    "block_types": section.get("block_types", []),
                    "code_ratio": section.get("code_ratio", 0.0),
                    "has_code": section.get("has_code", False),
                    "has_table": section.get("has_table", False),
                }
            )

        try:
            result = session.run(query, sections=batch_data, document_id=document_id)
            record = result.single()
            if record:
                stats["updated"] += record.get("updated", 0)
        except Exception as exc:
            stats["errors"] += 1
            logger.warning(
                "Failed to update enhanced metadata batch",
                document_id=document_id,
                batch_start=i,
                batch_size=len(batch),
                error=str(exc),
            )

    logger.info(
        "Enhanced metadata updated for sections",
        document_id=document_id,
        updated=stats["updated"],
        skipped=stats["skipped"],
        errors=stats["errors"],
    )

    return stats


def enhance_document_graph(
    session,
    document_id: str,
    sections: List[Dict],
) -> Dict[str, Any]:
    """
    Apply all Phase 3 graph enhancements for a document.

    Convenience function that runs all enhancement steps in order:
    1. Update sections with enhanced metadata
    2. Create PARENT_HEADING relationships
    3. Apply structural labels

    Args:
        session: Neo4j session
        document_id: Document identifier
        sections: List of section dicts with enhanced metadata

    Returns:
        Combined stats from all enhancement operations
    """
    results = {
        "document_id": document_id,
        "metadata": {},
        "relationships": {},
        "labels": {},
        "success": True,
    }

    try:
        # Step 1: Update enhanced metadata
        results["metadata"] = update_section_enhanced_metadata(
            session, document_id, sections
        )

        # Step 2: Create PARENT_HEADING relationships
        results["relationships"] = create_parent_heading_relationships(
            session, document_id, sections
        )

        # Step 3: Apply structural labels
        results["labels"] = apply_structural_labels(session, document_id)

        # Check for any errors
        total_errors = (
            results["metadata"].get("errors", 0)
            + results["relationships"].get("errors", 0)
            + results["labels"].get("errors", 0)
        )
        results["success"] = total_errors == 0

        logger.info(
            "Document graph enhancement complete",
            document_id=document_id,
            metadata_updated=results["metadata"].get("updated", 0),
            relationships_created=results["relationships"].get("created", 0),
            code_sections=results["labels"].get("code_sections", 0),
            table_sections=results["labels"].get("table_sections", 0),
            success=results["success"],
        )

    except Exception as exc:
        results["success"] = False
        results["error"] = str(exc)
        logger.error(
            "Document graph enhancement failed",
            document_id=document_id,
            error=str(exc),
        )

    return results
