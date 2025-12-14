"""
Graph feature extraction for candidate reranking.

This module implements the candidate subgraph extraction from the Neo4j Overhaul
Plan (Section 7.2). It extracts graph-based features for vector search candidates
to enable graph-aware reranking and diffusion.

Features extracted:
- Entity mention counts and confidence
- Query entity overlap
- Entity connectivity (degree statistics)
- Shared entity edges between candidates
- NEXT_CHUNK edges within candidate pool

Usage:
    from src.query.graph_features import extract_candidate_features

    features = extract_candidate_features(
        session,
        candidates=[{"chunk_id": "c1", "score": 0.9}, ...],
        query_entities=["WEKA", "S3"],
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ChunkFeatures:
    """Graph features for a single chunk."""

    chunk_id: str
    document_id: Optional[str] = None
    base_score: float = 0.0

    # Entity mention features
    mention_entity_count: int = 0
    mention_conf_sum: float = 0.0
    mention_conf_mean: float = 0.0

    # Query entity overlap
    query_entity_overlap_count: int = 0
    query_entity_overlap_conf_sum: float = 0.0

    # Entity connectivity
    max_entity_mention_degree: int = 0
    avg_entity_mention_degree: float = 0.0


@dataclass
class CandidateEdge:
    """Edge between two candidate chunks."""

    src: str
    dst: str
    edge_type: str  # "shared_entity" or "next_chunk"
    weight: float = 1.0


@dataclass
class CandidateSubgraph:
    """Subgraph of candidates with features and edges."""

    node_features: List[ChunkFeatures]
    edges: List[CandidateEdge]
    stats: Dict[str, int] = field(default_factory=dict)

    def get_features_by_id(self) -> Dict[str, ChunkFeatures]:
        """Return features indexed by chunk_id."""
        return {f.chunk_id: f for f in self.node_features}

    def get_adjacency_list(self) -> Dict[str, List[str]]:
        """Return adjacency list from edges."""
        adj: Dict[str, List[str]] = {}
        for edge in self.edges:
            if edge.src not in adj:
                adj[edge.src] = []
            adj[edge.src].append(edge.dst)
        return adj


class GraphFeatureExtractor:
    """
    Extracts graph features for candidate chunks.

    Implements the candidate subgraph extraction query from Section 7.2.
    Returns:
    - Node feature rows for each candidate
    - Edges among candidates (shared_entity + next_chunk)
    """

    def __init__(
        self,
        session,
        *,
        mention_conf_min: float = 0.5,
        max_entities_per_chunk: int = 20,
        max_candidate_chunks_per_entity: int = 50,
    ):
        """
        Initialize the feature extractor.

        Args:
            session: Neo4j session
            mention_conf_min: Minimum MENTIONS confidence to consider
            max_entities_per_chunk: Max entities to consider per chunk
            max_candidate_chunks_per_entity: Max chunks per entity for edge building
        """
        self.session = session
        self.mention_conf_min = mention_conf_min
        self.max_entities_per_chunk = max_entities_per_chunk
        self.max_candidate_chunks_per_entity = max_candidate_chunks_per_entity

    def extract(
        self,
        candidates: List[Dict[str, Any]],
        query_entities: Optional[List[str]] = None,
        *,
        return_edges: bool = True,
    ) -> CandidateSubgraph:
        """
        Extract graph features for candidate chunks.

        Args:
            candidates: List of dicts with "chunk_id" and optional "score"
            query_entities: Entity names from query (for overlap scoring)
            return_edges: Whether to compute edges among candidates

        Returns:
            CandidateSubgraph with features and edges
        """
        if not candidates:
            return CandidateSubgraph(node_features=[], edges=[])

        query_entities = query_entities or []
        query_entities_lower = [e.lower() for e in query_entities]

        # Extract node features
        node_features = self._extract_node_features(candidates, query_entities_lower)

        # Extract edges if requested
        edges: List[CandidateEdge] = []
        if return_edges:
            shared_edges = self._extract_shared_entity_edges(candidates)
            next_edges = self._extract_next_chunk_edges(candidates)
            edges = shared_edges + next_edges

        result = CandidateSubgraph(
            node_features=node_features,
            edges=edges,
            stats={
                "candidate_count": len(candidates),
                "features_extracted": len(node_features),
                "shared_entity_edges": len(
                    [e for e in edges if e.edge_type == "shared_entity"]
                ),
                "next_chunk_edges": len(
                    [e for e in edges if e.edge_type == "next_chunk"]
                ),
            },
        )

        logger.debug(
            "graph_features_extracted",
            candidate_count=len(candidates),
            features=len(node_features),
            edges=len(edges),
        )

        return result

    def _extract_node_features(
        self,
        candidates: List[Dict[str, Any]],
        query_entities_lower: List[str],
    ) -> List[ChunkFeatures]:
        """Extract node features for each candidate."""
        chunk_ids = [c["chunk_id"] for c in candidates]
        scores = {c["chunk_id"]: c.get("score", 0.0) for c in candidates}

        query = """
        UNWIND $chunk_ids AS cid
        MATCH (ch:Chunk {chunk_id: cid})
        OPTIONAL MATCH (ch)-[m:MENTIONS]->(e:Entity)
        WHERE coalesce(m.confidence, 1.0) >= $conf_min
        WITH ch, cid,
             collect({e: e, conf: coalesce(m.confidence, 1.0)})[0..$max_entities] AS mentions
        WITH ch, cid,
             [x IN mentions | x.e] AS entities,
             [x IN mentions | x.conf] AS confs,
             [x IN mentions | toLower(coalesce(x.e.normalized_name, x.e.name, ''))] AS entNames
        WITH ch, cid, entities, confs, entNames,
             [i IN range(0, size(entNames)-1)
              WHERE entNames[i] IN $query_entities | i] AS overlapIdx
        RETURN
            cid AS chunk_id,
            ch.document_id AS document_id,
            size(entities) AS mention_entity_count,
            reduce(s=0.0, x IN confs | s + x) AS mention_conf_sum,
            size(overlapIdx) AS query_entity_overlap_count,
            reduce(s=0.0, i IN overlapIdx | s + confs[i]) AS query_entity_overlap_conf_sum
        """
        result = self.session.run(
            query,
            chunk_ids=chunk_ids,
            conf_min=self.mention_conf_min,
            max_entities=self.max_entities_per_chunk,
            query_entities=query_entities_lower,
        )

        features = []
        for record in result:
            chunk_id = record["chunk_id"]
            mention_count = record["mention_entity_count"] or 0
            conf_sum = record["mention_conf_sum"] or 0.0

            features.append(
                ChunkFeatures(
                    chunk_id=chunk_id,
                    document_id=record["document_id"],
                    base_score=scores.get(chunk_id, 0.0),
                    mention_entity_count=mention_count,
                    mention_conf_sum=conf_sum,
                    mention_conf_mean=(
                        conf_sum / mention_count if mention_count else 0.0
                    ),
                    query_entity_overlap_count=record["query_entity_overlap_count"]
                    or 0,
                    query_entity_overlap_conf_sum=record[
                        "query_entity_overlap_conf_sum"
                    ]
                    or 0.0,
                )
            )

        return features

    def _extract_shared_entity_edges(
        self, candidates: List[Dict[str, Any]]
    ) -> List[CandidateEdge]:
        """Extract shared entity edges among candidates."""
        chunk_ids = [c["chunk_id"] for c in candidates]

        query = """
        UNWIND $chunk_ids AS cid
        MATCH (ch:Chunk {chunk_id: cid})-[m:MENTIONS]->(e:Entity)
        WHERE coalesce(m.confidence, 1.0) >= $conf_min
        WITH e, collect(DISTINCT ch.chunk_id) AS chIds
        WHERE size(chIds) > 1 AND size(chIds) <= $max_per_entity
        UNWIND range(0, size(chIds)-2) AS i
        UNWIND range(i+1, size(chIds)-1) AS j
        WITH chIds[i] AS a, chIds[j] AS b
        WITH a, b, count(*) AS weight
        RETURN a AS src, b AS dst, weight
        """
        result = self.session.run(
            query,
            chunk_ids=chunk_ids,
            conf_min=self.mention_conf_min,
            max_per_entity=self.max_candidate_chunks_per_entity,
        )

        return [
            CandidateEdge(
                src=record["src"],
                dst=record["dst"],
                edge_type="shared_entity",
                weight=float(record["weight"]),
            )
            for record in result
        ]

    def _extract_next_chunk_edges(
        self, candidates: List[Dict[str, Any]]
    ) -> List[CandidateEdge]:
        """Extract NEXT_CHUNK edges within the candidate pool."""
        chunk_ids = [c["chunk_id"] for c in candidates]
        chunk_set = set(chunk_ids)

        query = """
        UNWIND $chunk_ids AS cid
        MATCH (a:Chunk {chunk_id: cid})-[:NEXT_CHUNK]->(b:Chunk)
        WHERE b.chunk_id IN $chunk_ids
        RETURN a.chunk_id AS src, b.chunk_id AS dst
        """
        result = self.session.run(query, chunk_ids=chunk_ids)

        return [
            CandidateEdge(
                src=record["src"],
                dst=record["dst"],
                edge_type="next_chunk",
                weight=1.0,
            )
            for record in result
            if record["dst"] in chunk_set
        ]


def extract_candidate_features(
    session,
    candidates: List[Dict[str, Any]],
    query_entities: Optional[List[str]] = None,
    *,
    return_edges: bool = True,
    mention_conf_min: float = 0.5,
) -> CandidateSubgraph:
    """
    Convenience function to extract graph features for candidates.

    Args:
        session: Neo4j session
        candidates: List of dicts with "chunk_id" and optional "score"
        query_entities: Entity names from query
        return_edges: Whether to compute edges
        mention_conf_min: Minimum MENTIONS confidence

    Returns:
        CandidateSubgraph with features and edges
    """
    extractor = GraphFeatureExtractor(
        session,
        mention_conf_min=mention_conf_min,
    )
    return extractor.extract(
        candidates,
        query_entities=query_entities,
        return_edges=return_edges,
    )
