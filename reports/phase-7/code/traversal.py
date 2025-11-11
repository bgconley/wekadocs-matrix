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
    ALLOWED_REL_TYPES = [
        "MENTIONS",  # Section → Entity (Command, Configuration, Step, Procedure)
        "HAS_SECTION",  # Document → Section
        "CONTAINS_STEP",  # Procedure → Step
    ]

    MAX_DEPTH = 3
    MAX_NODES = 100

    def __init__(self, neo4j_driver):
        """
        Initialize TraversalService.

        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.driver = neo4j_driver

    def traverse(
        self,
        start_ids: List[str],
        rel_types: List[str] = None,
        max_depth: int = 2,
        include_text: bool = True,
    ) -> TraversalResult:
        """
        Traverse graph relationships from starting nodes.

        Args:
            start_ids: Starting node IDs
            rel_types: Relationship types to follow (default: MENTIONS, HAS_SECTION)
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
            # Build Cypher query using UNION ALL
            # Part 1: Start nodes at distance 0
            # Part 2: Reachable nodes at distance 1..max_depth
            rel_pattern = "|".join(rel_types)

            query = f"""
            UNWIND $start_ids AS start_id
            MATCH (start {{id: start_id}})
            RETURN start.id AS id,
                   labels(start)[0] AS label,
                   properties(start) AS props,
                   0 AS dist,
                   [] AS sample_paths

            UNION ALL

            UNWIND $start_ids AS start_id
            MATCH (start {{id: start_id}})
            MATCH path=(start)-[r:{rel_pattern}*1..{max_depth}]->(target)
            WITH DISTINCT target, min(length(path)) AS dist, collect(DISTINCT path)[0..10] AS sample_paths
            WHERE dist <= {max_depth}
            RETURN target.id AS id,
                   labels(target)[0] AS label,
                   properties(target) AS props,
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
                    # Get properties and filter text if needed
                    props = dict(record["props"]) if record["props"] else {}
                    if not include_text and "text" in props:
                        props = {k: v for k, v in props.items() if k != "text"}

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
                    if record["sample_paths"]:
                        for path in record["sample_paths"]:
                            for rel in path.relationships:
                                relationships.append(
                                    TraversalRelationship(
                                        from_id=rel.start_node["id"],
                                        to_id=rel.end_node["id"],
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
