"""
Per-document structural edge builder for Neo4j graph.

This module implements the canonical graph structure from the Neo4j Overhaul Plan:
- HAS_CHUNK: Document → Chunk ownership
- NEXT_CHUNK: Sequential adjacency within document
- NEXT: Sibling adjacency within same parent scope
- PARENT_HEADING / CHILD_OF / PARENT_OF: Heading hierarchy

The builder is designed to run after each document ingestion for immediate
correctness, with an end-of-run reconciliation pass for targeted repairs.

Key design principles:
- Idempotent: Safe to run multiple times (delete-before-rebuild)
- Document-scoped: O(chunks_in_doc) complexity, no N×N traps
- Versioned: Supports active/run_id gating for incremental updates
- Fail-fast: Validates invariants immediately after build

Usage:
    from src.neo.structural_builder import StructuralEdgeBuilder

    builder = StructuralEdgeBuilder(neo4j_session)
    result = builder.build_for_document(document_id)

    if result.valid:
        logger.info("Structural edges built successfully")
    else:
        logger.error(f"Build failed: {result.violations}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StructuralBuildResult:
    """Result of a structural edge build operation."""

    document_id: str
    valid: bool = True
    stats: Dict[str, int] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class StructuralEdgeBuilder:
    """
    Builds structural relationships for a single document.

    This implements the per-document structural builder from the Neo4j Overhaul Plan
    (Section 4). It runs after ingestion to create:

    1. HAS_CHUNK edges (Document → Chunk)
    2. NEXT_CHUNK edges (document-wide sequence)
    3. NEXT edges (sibling adjacency within parent scope)
    4. PARENT_HEADING / CHILD_OF / PARENT_OF edges (hierarchy)

    The builder is idempotent - it deletes existing structural edges before
    rebuilding to ensure correctness on re-ingestion.
    """

    # Relationship types managed by this builder
    STRUCTURAL_EDGE_TYPES = frozenset(
        ["NEXT_CHUNK", "NEXT", "PARENT_HEADING", "CHILD_OF", "PARENT_OF"]
    )

    def __init__(self, session, *, run_id: Optional[str] = None):
        """
        Initialize the structural edge builder.

        Args:
            session: Neo4j session or driver
            run_id: Optional ingestion run ID for versioning
        """
        self.session = session
        self.run_id = run_id

    def build_for_document(
        self,
        document_id: str,
        *,
        skip_has_chunk: bool = False,
        validate: bool = True,
    ) -> StructuralBuildResult:
        """
        Build all structural edges for a document.

        This is the main entry point that runs:
        1. Normalize parent_path → parent_path_norm
        2. Compute parent_chunk_id from parent_path_norm
        3. Clear existing structural edges (within this document)
        4. Create HAS_CHUNK edges (unless skip_has_chunk=True)
        5. Create hierarchy edges (PARENT_HEADING, CHILD_OF, PARENT_OF)
        6. Create NEXT_CHUNK edges (document-wide sequence)
        7. Create NEXT edges (sibling adjacency)
        8. Validate invariants

        Args:
            document_id: Document identifier
            skip_has_chunk: Skip HAS_CHUNK creation (if already created by ingestion)
            validate: Run post-build validation

        Returns:
            StructuralBuildResult with stats and validation results
        """
        result = StructuralBuildResult(document_id=document_id)

        try:
            # Step 0: Ensure chunk_id is populated (legacy compatibility).
            # Some ingestion paths historically used only `id`; the canonical plan
            # requires `chunk_id` for stable addressing.
            chunk_id_stats = self._ensure_chunk_id(document_id)
            result.stats["chunk_id_backfilled"] = chunk_id_stats.get("updated", 0)

            # Step 1: Normalize parent_path
            norm_stats = self._normalize_parent_path(document_id)
            result.stats["parent_path_normalized"] = norm_stats.get("updated", 0)

            # Step 2: Compute parent_chunk_id
            parent_stats = self._compute_parent_chunk_id(document_id)
            result.stats["parent_chunk_id_set"] = parent_stats.get("updated", 0)

            # Step 3: Clear existing structural edges
            clear_stats = self._clear_structural_edges(document_id)
            result.stats["edges_cleared"] = clear_stats.get("deleted", 0)

            # Step 4: Create HAS_CHUNK edges
            if not skip_has_chunk:
                has_chunk_stats = self._create_has_chunk_edges(document_id)
                result.stats["has_chunk_created"] = has_chunk_stats.get("created", 0)

            # Step 5: Create hierarchy edges
            hierarchy_stats = self._create_hierarchy_edges(document_id)
            result.stats["parent_heading_created"] = hierarchy_stats.get(
                "parent_heading", 0
            )
            result.stats["child_of_created"] = hierarchy_stats.get("child_of", 0)
            result.stats["parent_of_created"] = hierarchy_stats.get("parent_of", 0)

            # Step 6: Create NEXT_CHUNK edges
            next_chunk_stats = self._create_next_chunk_edges(document_id)
            result.stats["next_chunk_created"] = next_chunk_stats.get("created", 0)

            # Step 7: Create NEXT edges (siblings)
            next_stats = self._create_next_sibling_edges(document_id)
            result.stats["next_sibling_created"] = next_stats.get("created", 0)

            # Step 8: Validate if requested
            if validate:
                validation = self.validate_document(document_id)
                result.valid = validation.valid
                result.violations = validation.violations
                result.warnings = validation.warnings

            logger.info(
                "structural_edges_built",
                document_id=document_id,
                stats=result.stats,
                valid=result.valid,
                run_id=self.run_id,
            )

        except Exception as e:
            result.valid = False
            result.error = str(e)
            logger.error(
                "structural_edge_build_failed",
                document_id=document_id,
                error=str(e),
                run_id=self.run_id,
            )

        return result

    def _ensure_chunk_id(self, document_id: str) -> Dict[str, int]:
        """Backfill Chunk.chunk_id from Chunk.id for a single document."""
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.chunk_id IS NULL AND c.id IS NOT NULL
        SET c.chunk_id = c.id
        RETURN count(c) AS updated
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"updated": record["updated"] if record else 0}

    def _normalize_parent_path(self, document_id: str) -> Dict[str, int]:
        """
        Normalize parent_path into parent_path_norm.

        This makes matching stable across delimiter and whitespace variation.
        Pattern: split by '>' or ' > ', trim each part, rejoin with ' > '.
        """
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.parent_path IS NOT NULL AND c.parent_path <> ''
        WITH c,
             [p IN split(replace(c.parent_path, ' > ', '>'), '>') | trim(p)] AS parts
        WITH c,
             reduce(path = '', p IN parts |
               path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
             ) AS norm
        WHERE c.parent_path_norm IS NULL OR c.parent_path_norm <> norm
        SET c.parent_path_norm = norm
        RETURN count(c) AS updated
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"updated": record["updated"] if record else 0}

    def _compute_parent_chunk_id(self, document_id: str) -> Dict[str, int]:
        """
        Compute parent_chunk_id from parent_path_norm.

        The parent's path is the breadcrumb without the last segment.
        We find the nearest earlier chunk in the same document with the
        matching parent breadcrumb.
        """
        query = """
        MATCH (child:Chunk {document_id: $document_id})
        WHERE child.parent_path_norm IS NOT NULL
          AND child.parent_path_norm CONTAINS ' > '
        WITH child, split(child.parent_path_norm, ' > ') AS parts
        WITH child, parts[0..size(parts)-1] AS parentParts
        WITH child,
             reduce(path = '', p IN parentParts |
               path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
             ) AS parentPathNorm
        MATCH (parent:Chunk {document_id: $document_id, parent_path_norm: parentPathNorm})
        WHERE parent.order < child.order
          AND (child.level IS NULL OR parent.level IS NULL OR parent.level < child.level)
        WITH child, parent
        ORDER BY parent.order DESC
        WITH child, head(collect(parent)) AS parent
        WHERE parent IS NOT NULL
          AND (child.parent_chunk_id IS NULL OR child.parent_chunk_id <> parent.chunk_id)
        SET child.parent_chunk_id = parent.chunk_id,
            child.parent_section_id = parent.chunk_id
        RETURN count(child) AS updated
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"updated": record["updated"] if record else 0}

    def _clear_structural_edges(self, document_id: str) -> Dict[str, int]:
        """
        Clear existing structural edges for the document.

        Only clears edges between chunks within this document to avoid
        affecting cross-document relationships.
        """
        query = """
        MATCH (a:Chunk {document_id: $document_id})
              -[r:NEXT_CHUNK|NEXT|PARENT_HEADING|CHILD_OF|PARENT_OF]-
              (b:Chunk {document_id: $document_id})
        DELETE r
        RETURN count(r) AS deleted
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"deleted": record["deleted"] if record else 0}

    def _create_has_chunk_edges(self, document_id: str) -> Dict[str, int]:
        """
        Create HAS_CHUNK edges from Document to Chunk.

        Relationship direction: (:Document)-[:HAS_CHUNK]->(:Chunk)
        """
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c:Chunk {document_id: $document_id})
        WHERE NOT EXISTS { MATCH (d)-[:HAS_CHUNK]->(c) }
        MERGE (d)-[r:HAS_CHUNK]->(c)
        RETURN count(r) AS created
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"created": record["created"] if record else 0}

    def _create_hierarchy_edges(self, document_id: str) -> Dict[str, int]:
        """
        Create hierarchy edges: PARENT_HEADING, CHILD_OF, PARENT_OF.

        Relationship directions:
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
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        if record:
            return {
                "parent_heading": record["parent_heading"],
                "child_of": record["child_of"],
                "parent_of": record["parent_of"],
            }
        return {"parent_heading": 0, "child_of": 0, "parent_of": 0}

    def _create_next_chunk_edges(self, document_id: str) -> Dict[str, int]:
        """
        Create NEXT_CHUNK edges for document-wide sequence.

        Relationship direction: (c1:Chunk)-[:NEXT_CHUNK]->(c2:Chunk)
        where c2 immediately follows c1 in document order.
        """
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.order IS NOT NULL
        WITH collect(c ORDER BY c.order) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH chunks[i] AS c1, chunks[i+1] AS c2
        MERGE (c1)-[r:NEXT_CHUNK]->(c2)
        RETURN count(r) AS created
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"created": record["created"] if record else 0}

    def _create_next_sibling_edges(self, document_id: str) -> Dict[str, int]:
        """
        Create NEXT edges for sibling adjacency within the same parent scope.

        Relationship direction: (c1:Chunk)-[:NEXT]->(c2:Chunk)
        where c1 and c2 have the same parent_chunk_id and c2 immediately follows c1.
        """
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.order IS NOT NULL
        WITH c.parent_chunk_id AS pid, c.level AS lvl, c
        WITH pid, lvl, collect(c ORDER BY c.order) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH chunks[i] AS c1, chunks[i+1] AS c2
        MERGE (c1)-[r:NEXT]->(c2)
        RETURN count(r) AS created
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        return {"created": record["created"] if record else 0}

    def validate_document(self, document_id: str) -> StructuralBuildResult:
        """
        Validate structural invariants for a document.

        Checks:
        1. NEXT_CHUNK outdegree ≤ 1 (no branching)
        2. NEXT_CHUNK has no cycles
        3. Expected edge count sanity (n-1 NEXT_CHUNK edges for n chunks)
        4. Hierarchy coverage (chunks with parent_path should have parent_chunk_id)

        Returns:
            StructuralBuildResult with validation results
        """
        result = StructuralBuildResult(document_id=document_id, valid=True)

        # Check 1: NEXT_CHUNK outdegree ≤ 1
        branching = self._check_next_chunk_branching(document_id)
        if branching:
            result.valid = False
            result.violations.append(
                f"NEXT_CHUNK branching detected: {len(branching)} chunks with outdegree > 1"
            )

        # Check 2: NEXT_CHUNK cycles (limited depth check)
        cycles = self._check_next_chunk_cycles(document_id)
        if cycles:
            result.valid = False
            result.violations.append(f"NEXT_CHUNK cycle detected: {cycles[:5]}")

        # Check 3: Edge count sanity
        edge_counts = self._get_edge_counts(document_id)
        chunk_count = edge_counts.get("chunk_count", 0)
        next_chunk_count = edge_counts.get("next_chunk_count", 0)
        expected_next_chunk = max(0, chunk_count - 1)

        if chunk_count > 1 and next_chunk_count != expected_next_chunk:
            result.warnings.append(
                f"NEXT_CHUNK count mismatch: expected {expected_next_chunk}, "
                f"got {next_chunk_count}"
            )

        # Check 4: Hierarchy coverage
        hierarchy = self._check_hierarchy_coverage(document_id)
        should_have_parent = hierarchy.get("should_have_parent", 0)
        has_parent = hierarchy.get("has_parent", 0)
        coverage = hierarchy.get("coverage", 1.0)

        if should_have_parent > 0 and coverage < 0.9:
            result.warnings.append(
                f"Low hierarchy coverage: {coverage:.1%} "
                f"({has_parent}/{should_have_parent} chunks with parent)"
            )

        result.stats = {
            "chunk_count": chunk_count,
            "next_chunk_count": next_chunk_count,
            "parent_heading_count": edge_counts.get("parent_heading_count", 0),
            "hierarchy_coverage": coverage,
        }

        return result

    def _check_next_chunk_branching(self, document_id: str) -> List[Dict[str, Any]]:
        """Check for chunks with multiple outgoing NEXT_CHUNK edges."""
        query = """
        MATCH (c:Chunk {document_id: $document_id})-[r:NEXT_CHUNK]->()
        WITH c, count(r) AS out_deg
        WHERE out_deg > 1
        RETURN c.chunk_id AS chunk_id, out_deg
        LIMIT 25
        """
        result = self.session.run(query, document_id=document_id)
        return [dict(record) for record in result]

    def _check_next_chunk_cycles(
        self, document_id: str, max_depth: int = 50
    ) -> List[str]:
        """Check for cycles in NEXT_CHUNK relationships."""
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE EXISTS { MATCH (c)-[:NEXT_CHUNK]->() }
        WITH c LIMIT 100
        MATCH p=(c)-[:NEXT_CHUNK*1..$max_depth]->(c)
        RETURN c.chunk_id AS cycle_chunk
        LIMIT 10
        """
        result = self.session.run(query, document_id=document_id, max_depth=max_depth)
        return [record["cycle_chunk"] for record in result]

    def _get_edge_counts(self, document_id: str) -> Dict[str, int]:
        """Get counts of chunks and structural edges for validation."""
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WITH count(c) AS chunk_count
        OPTIONAL MATCH (:Chunk {document_id: $document_id})-[nc:NEXT_CHUNK]->()
        WITH chunk_count, count(nc) AS next_chunk_count
        OPTIONAL MATCH (:Chunk {document_id: $document_id})-[ph:PARENT_HEADING]->()
        RETURN chunk_count, next_chunk_count, count(ph) AS parent_heading_count
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        if record:
            return dict(record)
        return {"chunk_count": 0, "next_chunk_count": 0, "parent_heading_count": 0}

    def _check_hierarchy_coverage(self, document_id: str) -> Dict[str, Any]:
        """Check what percentage of chunks with parent_path have parent_chunk_id."""
        query = """
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.parent_path_norm CONTAINS ' > '
        WITH count(c) AS should_have_parent
        MATCH (c:Chunk {document_id: $document_id})
        WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NOT NULL
        WITH should_have_parent, count(c) AS has_parent
        RETURN
            should_have_parent,
            has_parent,
            CASE
                WHEN should_have_parent = 0 THEN 1.0
                ELSE (has_parent * 1.0 / should_have_parent)
            END AS coverage
        """
        result = self.session.run(query, document_id=document_id)
        record = result.single()
        if record:
            return dict(record)
        return {"should_have_parent": 0, "has_parent": 0, "coverage": 1.0}


def build_structural_edges_for_document(
    session,
    document_id: str,
    *,
    run_id: Optional[str] = None,
    skip_has_chunk: bool = False,
    validate: bool = True,
) -> StructuralBuildResult:
    """
    Convenience function to build structural edges for a document.

    This is the main entry point for use in ingestion pipelines.

    Args:
        session: Neo4j session
        document_id: Document identifier
        run_id: Optional ingestion run ID
        skip_has_chunk: Skip HAS_CHUNK creation if already done
        validate: Run post-build validation

    Returns:
        StructuralBuildResult with stats and validation
    """
    builder = StructuralEdgeBuilder(session, run_id=run_id)
    return builder.build_for_document(
        document_id,
        skip_has_chunk=skip_has_chunk,
        validate=validate,
    )
