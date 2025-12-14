"""
Hybrid Search Engine (Task 2.3)
Combines vector search with graph expansion and path finding.
See: /docs/spec.md §4.1 (Hybrid retrieval)
See: /docs/pseudocode-reference.md Phase 2, Task 2.3

Phase 7C: Integrated with reranking provider for post-ANN refinement.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.providers.rerank.base import RerankProvider
from src.providers.settings import EmbeddingSettings
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)

METADATA_TEXT_LIMIT = 1000
SAFE_METADATA_FIELDS = [
    "chunk_id",
    "section_id",
    "document_id",
    "parent_section_id",
    "doc_tag",
    "doc_status",
    "snapshot_scope",
    "source_path",
    "document_uri",
    "title",
    "heading",
    "name",
    "description",
    "anchor",
    "level",
    "order",
    "tokens",
    "token_count",
    "document_total_tokens",
    "chunk_order",
    "chunk_kind",
    "is_combined",
    "is_split",
    "is_microdoc",
    "doc_is_microdoc",
    "is_microdoc_extra",
    "is_microdoc_stub",
    "expansion_source",
    "is_expanded",
    "citation_labels",
    "original_section_ids",
    "boundaries_json",
    "entity_type",
    "procedure_type",
    "command_type",
    "configuration_type",
]


def build_metadata_projection(alias: str) -> str:
    """
    Construct a Cypher map projection that whitelists metadata fields and truncates text.
    """

    field_expr = ", ".join(f".{field}" for field in SAFE_METADATA_FIELDS)
    text_expr = (
        f"text: CASE WHEN {alias}.text IS NULL THEN NULL "
        f"ELSE substring({alias}.text, 0, $metadata_text_limit) END"
    )
    if field_expr:
        return f"{alias}{{{field_expr}, {text_expr}}}"
    return f"{alias}{{{text_expr}}}"


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
    rerank_time_ms: float  # Phase 7C: Added reranking timing
    graph_time_ms: float
    ranking_time_ms: float
    total_time_ms: float


class VectorStore:
    """Abstract interface for vector operations."""

    def search(
        self,
        vector: Union[List[float], Dict[str, Any]],
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for top-k nearest neighbors."""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        *,
        query_vector_name: str = "content",
        use_named_vectors: bool = False,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.query_vector_name = query_vector_name or "content"
        self.use_named_vectors = use_named_vectors

    def search(
        self,
        vector: Union[List[float], Dict[str, Any]],
        k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for top-k vectors."""

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

        # Pre-Phase 7 (G1): Metrics instrumentation for Qdrant search
        import time

        from src.shared.observability.metrics import (
            qdrant_operation_latency_ms,
            qdrant_schema_mismatch_total,
            qdrant_search_total,
        )

        start_time = time.time()
        status = "success"

        try:
            # Search - extract vector and optional name for query_points API
            query_vec: Any
            using_name: Optional[str] = None

            if isinstance(vector, dict):
                items = list(vector.items())
                if len(items) == 1:
                    name, vec = items[0]
                    query_vec = vec
                    if self.use_named_vectors:
                        using_name = name
                else:
                    # Multi-vector queries not supported in query_points API
                    # Use first vector and log warning
                    logger.warning(
                        "Multi-vector query not supported, using first vector only",
                        num_vectors=len(items),
                    )
                    name, vec = items[0]
                    query_vec = vec
                    if self.use_named_vectors:
                        using_name = name
            else:
                query_vec = vector
                if self.use_named_vectors:
                    using_name = self.query_vector_name

            # Build query_points call with optional using parameter
            query_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "query": query_vec,
                "limit": k,
                "query_filter": qdrant_filter,
                "with_payload": True,
            }
            if using_name:
                query_kwargs["using"] = using_name

            response = self.client.query_points(**query_kwargs)
            results = response.points

            # Record success metrics
            latency_ms = (time.time() - start_time) * 1000
            qdrant_search_total.labels(
                collection_name=self.collection_name, status=status
            ).inc()
            qdrant_operation_latency_ms.labels(
                collection_name=self.collection_name, operation="search"
            ).observe(latency_ms)

        except Exception as exc:
            status = "error"
            qdrant_search_total.labels(
                collection_name=self.collection_name, status=status
            ).inc()
            if self.use_named_vectors:
                message = str(exc).lower()
                if "named vector" in message or "vector name" in message:
                    qdrant_schema_mismatch_total.labels(
                        collection_name=self.collection_name
                    ).inc()
                    logger.error(
                        "Qdrant schema mismatch detected during search",
                        collection=self.collection_name,
                        use_named_vectors=self.use_named_vectors,
                        error=str(exc),
                    )
            raise

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

    def __init__(
        self,
        driver,
        index_name: str,
        *,
        embedding_version: Optional[str] = None,
    ):
        self.driver = driver
        self.index_name = index_name
        self.embedding_version = embedding_version

    def search(
        self, vector: List[float], k: int, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search Neo4j vector index for top-k vectors."""
        version = self.embedding_version or get_config().embedding.version

        where_clauses = ["node.embedding_version = $embedding_version"]
        params: Dict[str, Any] = {
            "index_name": self.index_name,
            "k": k,
            "vector": vector,
            "embedding_version": version,
        }

        if filters:
            dedupe: Dict[str, int] = {}
            for key, value in filters.items():
                safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key)
                dedupe.setdefault(safe_key, 0)
                dedupe[safe_key] += 1
                suffix = dedupe[safe_key]
                param_name = f"filter_{safe_key}_{suffix}"
                where_clauses.append(f"node.{safe_key} = ${param_name}")
                params[param_name] = value

        where_clause = " AND ".join(where_clauses)

        metadata_projection = build_metadata_projection("node")
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node, score
        WHERE {where_clause}
        RETURN node.id AS id, score, node.document_id AS document_id,
               labels(node)[0] AS node_label, {metadata_projection} AS metadata
        LIMIT $k
        """
        params["metadata_text_limit"] = METADATA_TEXT_LIMIT

        with self.driver.session() as session:
            result = session.run(query, **params)

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
    Pre-Phase 7 B5: Modified to use embedding provider API.
    Phase 7C: Integrated with reranking for post-ANN refinement.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        neo4j_driver,
        embedder,
        reranker: Optional[RerankProvider] = None,
        embedding_settings: Optional[EmbeddingSettings] = None,
    ):
        self.vector_store = vector_store
        self.neo4j_driver = neo4j_driver
        self.embedder = embedder
        self.reranker = reranker  # Phase 7C: Optional reranker
        self.embedding_settings = embedding_settings
        self.config = get_config()

        # Get search configuration
        self.max_hops = self.config.validator.max_depth  # Use validator's max_depth
        self.top_k = self.config.search.hybrid.top_k

        # Phase 7C: Reranking configuration
        self.rerank_enabled = reranker is not None
        self.rerank_k = 50  # Get more candidates for reranking
        self.rerank_top_k = 20  # Narrow down to top 20 after reranking

        if (
            self.embedding_settings
            and hasattr(embedder, "dims")
            and embedder.dims != self.embedding_settings.dims
        ):
            raise ValueError(
                f"HybridSearchEngine embedder dims ({getattr(embedder, 'dims', 'unknown')}) "
                f"do not match embedding profile dims ({self.embedding_settings.dims})."
            )

        # Pre-Phase 7: Log embedder info once at init
        if self.embedding_settings:
            logger.info(
                "HybridSearchEngine initialized with embedder",
                profile=self.embedding_settings.profile,
                provider=self.embedding_settings.provider,
                model_id=embedder.model_id,
                dims=embedder.dims,
                tokenizer_backend=self.embedding_settings.tokenizer_backend,
                supports_sparse=getattr(
                    self.embedding_settings.capabilities, "supports_sparse", None
                ),
                supports_colbert=getattr(
                    self.embedding_settings.capabilities, "supports_colbert", None
                ),
            )
        else:
            logger.info(
                f"HybridSearchEngine initialized with embedder: "
                f"model_id={embedder.model_id}, dims={embedder.dims}"
            )

        # Phase 7C: Log reranker info
        if self.reranker:
            logger.info(
                f"Reranking enabled: provider={self.reranker.provider_name}, "
                f"model={self.reranker.model_id}"
            )
        else:
            logger.info("Reranking disabled (no reranker provided)")

        hybrid_cfg = getattr(self.config.search, "hybrid", None)
        vector_fields_cfg = getattr(hybrid_cfg, "vector_fields", {"content": 1.0})
        self.vector_field_weights = dict(vector_fields_cfg or {"content": 1.0})
        qdrant_cfg = getattr(self.config.search.vector, "qdrant", None)
        self.qdrant_query_strategy = getattr(
            qdrant_cfg, "query_strategy", "content_only"
        )
        self._max_vector_field = self._determine_max_vector_field()

    def search(
        self,
        query_text: str,
        k: int = 20,
        filters: Optional[Dict] = None,
        expand_graph: bool = True,
        find_paths: bool = False,
        focused_entity_ids: Optional[List[str]] = None,  # Task 7C.8: Entity focus bias
    ) -> HybridSearchResults:
        """
        Perform hybrid search: vector similarity + reranking + graph expansion.

        Phase 7C workflow:
        1. Vector search (ANN) - get top 50 candidates
        2. Reranking (optional) - refine to top 20 using cross-attention
        3. Entity focus bias (Task 7C.8) - boost sections mentioning focused entities
        4. Graph expansion - expand from reranked results
        5. Final ranking - done in ranking.py

        Args:
            query_text: Natural language query
            k: Number of results to return
            filters: Optional filters for vector search
            expand_graph: Whether to expand from seeds via graph
            find_paths: Whether to find connecting paths between top results
            focused_entity_ids: Entity IDs to bias toward (from session history)

        Returns:
            HybridSearchResults with ranked results and timing
        """
        start_time = time.time()

        # Step 1: Vector search for seed nodes
        # Phase 7C: Get more candidates (50) if reranking enabled
        vector_k = self.rerank_k if self.rerank_enabled else k

        vector_start = time.time()
        # Pre-Phase 7 B5: Use provider's embed_query method for queries
        base_query_vector = self.embedder.embed_query(query_text)
        query_vector_payload = self._build_query_vector_payload(base_query_vector)
        vector_seeds = self.vector_store.search(
            query_vector_payload, k=vector_k, filters=filters
        )
        vector_time_ms = (time.time() - vector_start) * 1000

        # Convert to SearchResult objects
        # Pre-Phase 7 (D3): Tag score kind for ranking normalization
        results = []
        for seed in vector_seeds:
            metadata = seed.get("metadata", {})
            # Tag the score kind as similarity (Qdrant cosine similarity)
            metadata["score_kind"] = "similarity"
            results.append(
                SearchResult(
                    node_id=seed["node_id"],
                    node_label=seed["node_label"],
                    score=seed["score"],
                    distance=0,  # Seeds have distance 0
                    metadata=metadata,
                )
            )

        # Step 2: Reranking (Phase 7C)
        rerank_time_ms = 0.0
        if self.rerank_enabled and results:
            rerank_start = time.time()
            results = self._apply_reranking(query_text, results)
            rerank_time_ms = (time.time() - rerank_start) * 1000

            logger.debug(
                f"Reranking complete: {len(results)} results, {rerank_time_ms:.2f}ms"
            )

        # Task 7C.8: Apply entity focus bias if provided
        if focused_entity_ids:
            bias_start = time.time()
            results = self._apply_entity_focus_bias(results, focused_entity_ids)
            bias_time_ms = (time.time() - bias_start) * 1000
            logger.debug(
                f"Entity focus bias applied: {len(focused_entity_ids)} entities, "
                f"{bias_time_ms:.2f}ms"
            )

        # Step 3: Graph expansion (if enabled)
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

        # Pre-Phase 7 (D2): Enrich results with coverage signals
        # This adds connection_count and mention_count for ranking
        unique_results = self._enrich_with_coverage(unique_results[:k])

        ranking_time_ms = 0  # Placeholder - actual ranking in ranking.py

        total_time_ms = (time.time() - start_time) * 1000

        return HybridSearchResults(
            results=unique_results,
            total_found=len(unique_results),
            vector_time_ms=vector_time_ms,
            rerank_time_ms=rerank_time_ms,  # Phase 7C: Added reranking timing
            graph_time_ms=graph_time_ms,
            ranking_time_ms=ranking_time_ms,
            total_time_ms=total_time_ms,
        )

    def _determine_max_vector_field(self) -> str:
        """Select the dominant named vector for max_field strategy."""
        fallback = getattr(self.vector_store, "query_vector_name", "content")
        field_weights = self.vector_field_weights or {}
        if not field_weights:
            return fallback
        field, weight = max(
            field_weights.items(),
            key=lambda item: item[1] if item[1] is not None else 0.0,
        )
        if weight is None or weight <= 0:
            return fallback
        return field

    def _build_query_vector_payload(
        self, base_vector: List[float]
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Build query vector payload according to configured strategy."""
        if not isinstance(self.vector_store, QdrantVectorStore):
            return base_vector
        if not getattr(self.vector_store, "use_named_vectors", False):
            return base_vector

        strategy_value = getattr(
            self.qdrant_query_strategy, "value", self.qdrant_query_strategy
        )
        base_list = list(base_vector)

        if strategy_value == "weighted":
            payload: Dict[str, List[float]] = {}
            for field, weight in self.vector_field_weights.items():
                if weight is None or weight <= 0:
                    continue
                if weight == 1.0:
                    payload[field] = list(base_list)
                else:
                    payload[field] = [val * weight for val in base_list]
            if payload:
                return payload
            target_field = getattr(self.vector_store, "query_vector_name", "content")
            return {target_field: base_list}

        if strategy_value == "max_field":
            target_field = self._max_vector_field or getattr(
                self.vector_store, "query_vector_name", "content"
            )
            return {target_field: base_list}

        target_field = getattr(self.vector_store, "query_vector_name", "content")
        return {target_field: base_list}

    def _expand_from_seeds(self, seeds: List[SearchResult]) -> List[SearchResult]:
        """
        Expand from seed nodes via typed relationships (1-2 hops).
        """
        if not seeds:
            return []

        seed_ids = [s.node_id for s in seeds]

        # Controlled expansion query with depth limit
        # Note: max_hops must be a literal in the pattern, not a parameter
        metadata_projection = build_metadata_projection("target")
        expansion_query = f"""
        UNWIND $seed_ids AS seed_id
        MATCH (seed {{id: seed_id}})
        OPTIONAL MATCH path=(seed)-[r:MENTIONS|CONTAINS_STEP|HAS_PARAMETER*1..{self.max_hops}]->(target)
        WHERE target.id <> seed.id
        WITH DISTINCT target, min(length(path)) AS dist, seed.id AS seed_id
        WHERE dist <= {self.max_hops}
        RETURN target.id AS id, dist, labels(target)[0] AS label,
               {metadata_projection} AS metadata, seed_id
        ORDER BY dist ASC
        LIMIT 50
        """

        expanded_results = []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    expansion_query,
                    seed_ids=seed_ids,
                    metadata_text_limit=METADATA_TEXT_LIMIT,
                )

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
            logger.warning(
                "Graph expansion error during hybrid search",
                error=str(e),
                seed_count=len(seeds),
            )

        return expanded_results

    def _find_connecting_paths(self, seeds: List[SearchResult]) -> List[SearchResult]:
        """
        Find shortest paths connecting top seed nodes.
        """
        if len(seeds) < 2:
            return []

        seed_ids = [s.node_id for s in seeds[:5]]  # Limit to top 5

        # Find shortest paths between seeds
        metadata_projection = build_metadata_projection("node")
        path_query = (
            """
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
               """
            + metadata_projection
            + """ AS metadata, min_dist AS dist
        LIMIT 30
        """
        )

        bridge_results = []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    path_query,
                    ids=seed_ids,
                    metadata_text_limit=METADATA_TEXT_LIMIT,
                )

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
            logger.warning(
                "Path finding error during hybrid search",
                error=str(e),
                seed_count=len(seeds),
            )

        return bridge_results

    def _enrich_with_coverage(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Pre-Phase 7 (D2): Enrich results with coverage signals.
        Adds connection_count and mention_count via batched Cypher query.

        Args:
            results: List of search results to enrich

        Returns:
            Results with coverage metadata added
        """
        if not results:
            return results

        # Extract node IDs for batch query
        node_ids = [r.node_id for r in results]

        # Batched Cypher to compute coverage signals
        coverage_query = """
        UNWIND $ids AS sid
        MATCH (c:Chunk {id: sid})
        OPTIONAL MATCH (c)-[r]->()
        WITH c, count(DISTINCT r) AS conn_count
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, conn_count, count(DISTINCT e) AS mention_count
        RETURN c.id AS id,
               conn_count AS connection_count,
               mention_count AS mention_count
        """

        # Create mapping of coverage data
        coverage_map = {}
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(coverage_query, ids=node_ids)
                for record in result:
                    coverage_map[record["id"]] = {
                        "connection_count": record["connection_count"],
                        "mention_count": record["mention_count"],
                    }

            logger.debug(f"Enriched {len(coverage_map)} results with coverage signals")
        except Exception as e:
            logger.warning(f"Failed to enrich with coverage signals: {e}")
            # Return original results if enrichment fails
            return results

        # Update results with coverage data
        for result in results:
            if result.node_id in coverage_map:
                result.metadata.update(coverage_map[result.node_id])
            else:
                # Default values if not found
                result.metadata["connection_count"] = 0
                result.metadata["mention_count"] = 0

        return results

    def _apply_reranking(
        self, query_text: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Phase 7C: Apply reranking to refine candidate ordering.

        Uses cross-attention based reranking to improve precision after ANN.

        Args:
            query_text: Original query text
            results: Initial results from vector search

        Returns:
            Reranked results (top_k after reranking)
        """
        if not self.reranker or not results:
            return results

        # Prepare candidates for reranker
        # Reranker expects: [{"id": str, "text": str, ...}]
        candidates = []
        for result in results:
            # Extract text from metadata
            text = result.metadata.get("text", "")
            if not text:
                # Fallback: try title or other fields
                text = result.metadata.get("title", result.metadata.get("name", ""))

            if text:  # Only add if we have text content
                candidates.append(
                    {
                        "id": result.node_id,
                        "text": text,
                        "original_result": result,  # Keep reference
                    }
                )

        if not candidates:
            logger.warning("No text content found in candidates for reranking")
            return results

        try:
            # Call reranker
            reranked = self.reranker.rerank(
                query=query_text,
                candidates=candidates,
                top_k=self.rerank_top_k,  # Default 20
            )

            # Convert back to SearchResult objects
            reranked_results = []
            for reranked_cand in reranked:
                original = reranked_cand["original_result"]

                # Update score with rerank score
                original.score = reranked_cand["rerank_score"]

                # Add reranking metadata
                original.metadata["rerank_score"] = reranked_cand["rerank_score"]
                original.metadata["original_rank"] = reranked_cand.get(
                    "original_rank", 0
                )
                original.metadata["reranker"] = reranked_cand.get("reranker", "unknown")
                original.metadata["score_kind"] = "rerank"  # Update score kind

                reranked_results.append(original)

            logger.debug(
                f"Reranked {len(candidates)} candidates to {len(reranked_results)} results"
            )

            return reranked_results

        except Exception as e:
            # Log error but don't fail the search - fallback to vector results
            logger.error(f"Reranking failed: {e}. Falling back to vector results.")
            return results[: self.rerank_top_k]  # At least limit to top_k

    def _apply_entity_focus_bias(
        self, results: List[SearchResult], focused_entity_ids: List[str]
    ) -> List[SearchResult]:
        """
        Task 7C.8: Boost sections that mention focused entities from session history.

        This implements multi-turn context awareness by biasing retrieval toward
        entities the user has been discussing in recent turns.

        Strategy: Boost score by 20% per focused entity mention
        Example: Section mentioning 2 focused entities gets 40% boost
        Formula: new_score = old_score * (1 + 0.2 * focus_hits)

        Args:
            results: Initial search results
            focused_entity_ids: Entity IDs to bias toward (from session history)

        Returns:
            Reranked results with entity focus boost applied
        """
        if not focused_entity_ids or not results:
            return results

        # Extract section IDs for batch query
        section_ids = [r.node_id for r in results]

        # Query graph to count focused entity mentions per chunk
        # Uses MENTIONS relationship from Chunk to entities
        focus_query = """
        UNWIND $section_ids AS section_id
        MATCH (c:Chunk {id: section_id})-[:MENTIONS]->(e)
        WHERE e.id IN $focused_entity_ids
        RETURN c.id AS section_id, count(DISTINCT e) AS focus_hits, collect(DISTINCT e.id) AS matched_entities
        """

        focus_counts = {}
        matched_entities_map = {}

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    focus_query,
                    section_ids=section_ids,
                    focused_entity_ids=focused_entity_ids,
                )

                for record in result:
                    focus_counts[record["section_id"]] = record["focus_hits"]
                    matched_entities_map[record["section_id"]] = record[
                        "matched_entities"
                    ]

            logger.debug(
                f"Focus bias: {len(focus_counts)} sections mention focused entities"
            )

        except Exception as e:
            # Log error but don't fail - return original results
            logger.warning(f"Entity focus bias query failed: {e}")
            return results

        # Apply boost to scores
        FOCUS_BOOST_WEIGHT = 0.2  # 20% boost per focused entity mention

        for result in results:
            focus_hits = focus_counts.get(result.node_id, 0)

            if focus_hits > 0:
                # Boost score
                original_score = result.score
                result.score = original_score * (1.0 + FOCUS_BOOST_WEIGHT * focus_hits)

                # Add metadata for debugging
                result.metadata["entity_focus_boost"] = focus_hits
                result.metadata["entity_focus_score_original"] = original_score
                result.metadata["entity_focus_entities"] = matched_entities_map.get(
                    result.node_id, []
                )

                logger.debug(
                    f"Boosted {result.node_id}: {original_score:.3f} -> {result.score:.3f} "
                    f"({focus_hits} entities)"
                )

        # Re-sort by boosted scores (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        return results
