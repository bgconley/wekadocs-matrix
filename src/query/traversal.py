"""
Graph Traversal Service (E4)
Implements traverse_relationships MCP tool for multi-turn graph exploration.
See: /docs/feature-spec-enhanced-responses.md
See: /docs/implementation-plan-enhanced-responses.md Task E4
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    mcp_traverse_depth_total,
    mcp_traverse_nodes_found,
)

logger = get_logger(__name__)


@dataclass
class TraversalNode:
    """Node discovered during traversal."""

    id: str
    label: str
    properties: Dict[str, Any]
    distance: int  # Hops from start


@dataclass
class TraversalRelationship:
    """Relationship discovered during traversal."""

    from_id: str
    to_id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class TraversalResult:
    """Result of graph traversal operation."""

    nodes: List[TraversalNode]
    relationships: List[TraversalRelationship]
    paths: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "relationships": [asdict(r) for r in self.relationships],
            "paths": self.paths,
        }


class TraversalService:
    """
    Service for traversing graph relationships.
    Implements depth-limited traversal with relationship type whitelisting.
    """

    # Whitelist of allowed relationship types
    # Includes all relationship types created by ingestion
    # P0: HAS_SECTION deprecated, use HAS_CHUNK
    ALLOWED_REL_TYPES = [
        "MENTIONS",  # Section → Entity (Command, Configuration, Step, Procedure)
        "HAS_CHUNK",  # Document → Chunk (canonical membership edge)
        "CONTAINS_STEP",  # Procedure → Step
    ]

    MAX_DEPTH = 3
    MAX_NODES = 100

    # Payload safety limits
    MAX_TEXT_CHARS = 1000  # Truncate long text fields
    MAX_STRING_LEN = 500  # Truncate long string properties
    MAX_LIST_LEN = 64  # Drop long numeric arrays (e.g., embeddings)

    # Keys that are known to be very large and should be removed
    BLOCKLIST_KEYS = {
        "embedding",
        "embeddings",
        "embedding_values",
        "embedding_vector",
        "vector",
        "dense",
        "dense_vector",
        "sparse_vector",
        "sparse_indices",
        "sparse_values",
    }

    def __init__(self, neo4j_driver):
        """
        Initialize TraversalService.

        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.driver = neo4j_driver

    def _sanitize_properties(
        self, props: Dict[str, Any], include_text: bool
    ) -> Dict[str, Any]:
        """Sanitize node properties for safe MCP payload sizes."""
        if not props:
            return {}

        sanitized: Dict[str, Any] = {}
        for k, v in props.items():
            # Always drop explicit blocklisted keys
            if k in self.BLOCKLIST_KEYS or "embed" in k.lower():
                continue

            # Handle text field separately
            if k == "text":
                if include_text and isinstance(v, str):
                    if len(v) > self.MAX_TEXT_CHARS:
                        sanitized[k] = v[: self.MAX_TEXT_CHARS] + "..."
                    else:
                        sanitized[k] = v
                # If include_text is False, skip adding text entirely
                continue

            # Truncate long strings
            if isinstance(v, str):
                if len(v) > self.MAX_STRING_LEN:
                    sanitized[k] = v[: self.MAX_STRING_LEN] + "..."
                else:
                    sanitized[k] = v
                continue

            # Drop long numeric arrays (typical for embeddings)
            if isinstance(v, list):
                # If it's a list of numbers and long, drop it
                if len(v) > self.MAX_LIST_LEN and all(
                    isinstance(x, (int, float)) for x in v[:10]
                ):
                    continue
                # Otherwise, keep small lists as-is
                sanitized[k] = v
                continue

            # Default: keep small scalars/dicts
            sanitized[k] = v

        return sanitized

    def traverse(
        self,
        start_ids: List[str],
        rel_types: List[str] = None,
        max_depth: int = 2,
        include_text: bool = True,
    ) -> TraversalResult:
        """
        Traverse graph relationships from starting nodes (bi-directional).

        Follows relationships in both directions to discover:
        - Outgoing: MENTIONS, CONTAINS_STEP (Section → Entities)
        - Incoming: HAS_CHUNK (Document → Chunk)
        - Siblings: via Document at depth=2

        Args:
            start_ids: Starting node IDs
            rel_types: Relationship types to follow (default: MENTIONS, HAS_CHUNK, CONTAINS_STEP)
            max_depth: Maximum traversal depth (1-3)
            include_text: Include full text in node properties

        Returns:
            TraversalResult with nodes, relationships, and paths

        Raises:
            ValueError: If max_depth exceeds limit or invalid relationship type
        """
        # Validate max_depth
        if max_depth > self.MAX_DEPTH:
            raise ValueError(f"max_depth cannot exceed {self.MAX_DEPTH}")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        # Default relationship types (must match ALLOWED_REL_TYPES)
        if rel_types is None:
            rel_types = self.ALLOWED_REL_TYPES

        # Validate relationship types
        for rel_type in rel_types:
            if rel_type not in self.ALLOWED_REL_TYPES:
                raise ValueError(
                    f"Invalid relationship type: {rel_type}. "
                    f"Allowed types: {', '.join(self.ALLOWED_REL_TYPES)}"
                )

        # Instrument metrics (E5)
        mcp_traverse_depth_total.labels(depth=str(max_depth)).inc()

        logger.info(
            f"Starting traversal: start_ids={start_ids}, "
            f"rel_types={rel_types}, max_depth={max_depth}"
        )

        try:
            # Build Cypher query using UNION ALL (bi-directional)
            # Part 1: Start nodes at distance 0 (always returned)
            # Part 2: Reachable nodes at distance 1..max_depth (both directions)
            # Why bi-directional: HAS_CHUNK is incoming to Chunks (Document → Chunk)
            rel_pattern = "|".join(rel_types)

            query = f"""
            UNWIND $start_ids AS start_id
            MATCH (start {{id: start_id}})
            RETURN start.id AS id,
                   labels(start)[0] AS label,
                   {{
                       title: coalesce(start.title, start.name, start.heading, ''),
                       heading: start.heading,
                       name: start.name,
                       doc_tag: start.doc_tag,
                       level: start.level,
                       tokens: start.tokens,
                       anchor: start.anchor,
                       text: start.text
                   }} AS props,
                   0 AS dist,
                   [] AS sample_paths

            UNION ALL

            UNWIND $start_ids AS start_id
            MATCH (start {{id: start_id}})
            MATCH path=(start)-[r:{rel_pattern}*1..{max_depth}]-(target)
            WITH DISTINCT target, min(length(path)) AS dist, collect(DISTINCT path)[0..10] AS sample_paths
            WHERE dist <= {max_depth}
            RETURN target.id AS id,
                   labels(target)[0] AS label,
                   {{
                       title: coalesce(target.title, target.name, target.heading, ''),
                       heading: target.heading,
                       name: target.name,
                       doc_tag: target.doc_tag,
                       level: target.level,
                       tokens: target.tokens,
                       anchor: target.anchor,
                       text: target.text
                   }} AS props,
                   dist,
                   sample_paths

            ORDER BY dist ASC
            LIMIT {self.MAX_NODES}
            """

            nodes = []
            relationships = []
            paths = []

            with self.driver.session() as session:
                result = session.run(query, start_ids=start_ids)

                for record in result:
                    # Get properties and sanitize
                    raw_props = dict(record["props"]) if record["props"] else {}
                    props = self._sanitize_properties(
                        raw_props, include_text=include_text
                    )

                    # Add node
                    nodes.append(
                        TraversalNode(
                            id=record["id"],
                            label=record["label"],
                            properties=props,
                            distance=record["dist"],
                        )
                    )

                    # Extract relationships from paths
                    # Normalize edge directions to match path order (not stored direction)
                    if record["sample_paths"]:
                        for path in record["sample_paths"]:
                            # Build edges from path.nodes[i] → path.nodes[i+1]
                            # This ensures arrows match the shown path, even with undirected patterns
                            path_node_ids = [node["id"] for node in path.nodes]
                            for i, rel in enumerate(path.relationships):
                                relationships.append(
                                    TraversalRelationship(
                                        from_id=path_node_ids[i],
                                        to_id=path_node_ids[i + 1],
                                        type=rel.type,
                                        properties=dict(rel),
                                    )
                                )

                        # Add path representation
                        for path in record["sample_paths"][:3]:  # Limit paths per node
                            path_nodes = [node["id"] for node in path.nodes]
                            paths.append(
                                {
                                    "nodes": path_nodes,
                                    "length": len(path_nodes) - 1,
                                }
                            )

            # Deduplicate relationships
            unique_rels = {}
            for rel in relationships:
                key = (rel.from_id, rel.to_id, rel.type)
                if key not in unique_rels:
                    unique_rels[key] = rel
            relationships = list(unique_rels.values())

            # Instrument metrics (E5)
            mcp_traverse_nodes_found.observe(len(nodes))

            logger.info(
                f"Traversal completed: depth={max_depth}, "
                f"nodes_found={len(nodes)}, "
                f"relationships={len(relationships)}, "
                f"paths={len(paths)}"
            )

            return TraversalResult(
                nodes=nodes, relationships=relationships, paths=paths
            )

        except Exception as e:
            logger.error(f"Traversal failed: {e}", exc_info=True)
            raise
