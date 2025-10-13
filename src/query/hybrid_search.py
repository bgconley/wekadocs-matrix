"""
Hybrid Search Engine (Task 2.3)
Combines vector search with graph expansion and path finding.
See: /docs/spec.md §4.1 (Hybrid retrieval)
See: /docs/pseudocode-reference.md Phase 2, Task 2.3
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.shared.config import get_config


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    node_id: str
    node_label: str
    score: float
    distance: int  # Graph distance from seed
    metadata: Dict[str, Any]
    path: Optional[List[str]] = None  # Node IDs in path from seed


@dataclass
class HybridSearchResults:
    """Results from hybrid search with timing info."""

    results: List[SearchResult]
    total_found: int
    vector_time_ms: float
    graph_time_ms: float
    ranking_time_ms: float
    total_time_ms: float


class VectorStore:
    """Abstract interface for vector operations."""

    def search(
        self, vector: List[float], k: int, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for top-k nearest neighbors."""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, qdrant_client, collection_name: str):
        self.client = qdrant_client
        self.collection_name = collection_name

    def search(
        self, vector: List[float], k: int, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for top-k vectors."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Convert to standard format
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "node_id": hit.payload.get("node_id"),
                "node_label": hit.payload.get("node_label", "Section"),
                "document_id": hit.payload.get("document_id"),
                "metadata": hit.payload,
            }
            for hit in results
        ]


class Neo4jVectorStore(VectorStore):
    """Neo4j vector store implementation."""

    def __init__(self, driver, index_name: str):
        self.driver = driver
        self.index_name = index_name

    def search(
        self, vector: List[float], k: int, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search Neo4j vector index for top-k vectors."""
        # Build WHERE clause from filters
        where_clauses = []
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    where_clauses.append(f"node.{key} = '{value}'")
                else:
                    where_clauses.append(f"node.{key} = {value}")

        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node, score
        WHERE node.embedding_version = $embedding_version{where_clause}
        RETURN node.id AS id, score, node.document_id AS document_id,
               labels(node)[0] AS node_label, properties(node) AS metadata
        LIMIT $k
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                index_name=self.index_name,
                k=k,
                vector=vector,
                embedding_version=get_config()
                .get("embedding", {})
                .get("version", "v1"),
            )

            return [
                {
                    "id": record["id"],
                    "score": record["score"],
                    "node_id": record["id"],
                    "node_label": record["node_label"],
                    "document_id": record.get("document_id"),
                    "metadata": record["metadata"],
                }
                for record in result
            ]


class HybridSearchEngine:
    """
    Hybrid search combining vector similarity with graph structure.
    """

    def __init__(self, vector_store: VectorStore, neo4j_driver, embedder):
        self.vector_store = vector_store
        self.neo4j_driver = neo4j_driver
        self.embedder = embedder
        self.config = get_config()

        # Get search configuration
        self.max_hops = self.config.validator.max_depth  # Use validator's max_depth
        self.top_k = self.config.search.hybrid.top_k

    def search(
        self,
        query_text: str,
        k: int = 20,
        filters: Optional[Dict] = None,
        expand_graph: bool = True,
        find_paths: bool = False,
    ) -> HybridSearchResults:
        """
        Perform hybrid search: vector similarity + graph expansion.

        Args:
            query_text: Natural language query
            k: Number of results to return
            filters: Optional filters for vector search
            expand_graph: Whether to expand from seeds via graph
            find_paths: Whether to find connecting paths between top results

        Returns:
            HybridSearchResults with ranked results and timing
        """
        start_time = time.time()

        # Step 1: Vector search for seed nodes
        vector_start = time.time()
        query_vector = self.embedder.encode(query_text)
        vector_seeds = self.vector_store.search(query_vector, k=k, filters=filters)
        vector_time_ms = (time.time() - vector_start) * 1000

        # Convert to SearchResult objects
        results = [
            SearchResult(
                node_id=seed["node_id"],
                node_label=seed["node_label"],
                score=seed["score"],
                distance=0,  # Seeds have distance 0
                metadata=seed.get("metadata", {}),
            )
            for seed in vector_seeds
        ]

        # Step 2: Graph expansion (if enabled)
        graph_time_ms = 0
        if expand_graph and results:
            graph_start = time.time()
            expanded = self._expand_from_seeds(results[: min(10, len(results))])
            results.extend(expanded)
            graph_time_ms = (time.time() - graph_start) * 1000

        # Step 3: Find connecting paths (if enabled)
        if find_paths and len(results) >= 2:
            graph_start = time.time()
            bridged = self._find_connecting_paths(results[: min(5, len(results))])
            results.extend(bridged)
            graph_time_ms += (time.time() - graph_start) * 1000

        # Step 4: Ranking and deduplication happens in ranking.py
        # For now, just remove duplicates by node_id
        seen = set()
        unique_results = []
        for result in results:
            if result.node_id not in seen:
                seen.add(result.node_id)
                unique_results.append(result)

        ranking_time_ms = 0  # Placeholder - actual ranking in ranking.py

        total_time_ms = (time.time() - start_time) * 1000

        return HybridSearchResults(
            results=unique_results[:k],
            total_found=len(unique_results),
            vector_time_ms=vector_time_ms,
            graph_time_ms=graph_time_ms,
            ranking_time_ms=ranking_time_ms,
            total_time_ms=total_time_ms,
        )

    def _expand_from_seeds(self, seeds: List[SearchResult]) -> List[SearchResult]:
        """
        Expand from seed nodes via typed relationships (1-2 hops).
        """
        if not seeds:
            return []

        seed_ids = [s.node_id for s in seeds]

        # Controlled expansion query with depth limit
        # Note: max_hops must be a literal in the pattern, not a parameter
        expansion_query = f"""
        UNWIND $seed_ids AS seed_id
        MATCH (seed {{id: seed_id}})
        OPTIONAL MATCH path=(seed)-[r:MENTIONS|CONTAINS_STEP|HAS_PARAMETER|REQUIRES|AFFECTS*1..{self.max_hops}]->(target)
        WHERE target.id <> seed.id
        WITH DISTINCT target, min(length(path)) AS dist, seed.id AS seed_id
        WHERE dist <= {self.max_hops}
        RETURN target.id AS id, dist, labels(target)[0] AS label,
               properties(target) AS metadata, seed_id
        ORDER BY dist ASC
        LIMIT 50
        """

        expanded_results = []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(expansion_query, seed_ids=seed_ids)

                for record in result:
                    expanded_results.append(
                        SearchResult(
                            node_id=record["id"],
                            node_label=record["label"],
                            score=0.5,  # Lower score for expanded nodes
                            distance=record["dist"],
                            metadata=record["metadata"],
                            path=[record["seed_id"], record["id"]],
                        )
                    )
        except Exception as e:
            # Log error but don't fail the search
            print(f"Graph expansion error: {e}")

        return expanded_results

    def _find_connecting_paths(self, seeds: List[SearchResult]) -> List[SearchResult]:
        """
        Find shortest paths connecting top seed nodes.
        """
        if len(seeds) < 2:
            return []

        seed_ids = [s.node_id for s in seeds[:5]]  # Limit to top 5

        # Find shortest paths between seeds
        path_query = """
        UNWIND $ids AS a
        UNWIND $ids AS b
        WITH DISTINCT a, b WHERE a < b
        MATCH (x {id: a}), (y {id: b})
        MATCH path=shortestPath((x)-[*..3]-(y))
        WITH path, nodes(path) AS path_nodes, length(path) AS len
        WHERE len > 0 AND len <= 3
        UNWIND path_nodes AS node
        WITH DISTINCT node, min(len) AS min_dist
        WHERE node.id NOT IN $ids  // Don't return seeds again
        RETURN node.id AS id, labels(node)[0] AS label,
               properties(node) AS metadata, min_dist AS dist
        LIMIT 30
        """

        bridge_results = []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(path_query, ids=seed_ids)

                for record in result:
                    bridge_results.append(
                        SearchResult(
                            node_id=record["id"],
                            node_label=record["label"],
                            score=0.3,  # Lower score for bridge nodes
                            distance=record["dist"],
                            metadata=record["metadata"],
                        )
                    )
        except Exception as e:
            # Log error but don't fail the search
            print(f"Path finding error: {e}")

        return bridge_results
