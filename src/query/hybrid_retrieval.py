"""
Phase 7E-2: Hybrid Retrieval Implementation
Combines vector search (Qdrant) with BM25/keyword search (Neo4j full-text)
Implements RRF fusion, weighted fusion, bounded adjacency expansion, and context budget enforcement

Phase 7E-4: Enhanced with comprehensive metrics collection and SLO monitoring

Reference: Phase 7E Canonical Spec L1421-1444, L3781-3788
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from neo4j import Driver
from qdrant_client import QdrantClient

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import get_metrics_aggregator
from src.providers.tokenizer_service import TokenizerService
from src.shared.config import get_config
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    retrieval_expansion_chunks_added,
    retrieval_expansion_rate_current,
    retrieval_expansion_total,
)

logger = get_logger(__name__)


class FusionMethod(str, Enum):
    """Available fusion methods for hybrid retrieval."""

    RRF = "rrf"  # Reciprocal Rank Fusion (default, robust)
    WEIGHTED = "weighted"  # Weighted linear combination (requires tuning)


class ExpandWhen(str, Enum):
    """Controls when adjacency expansion is triggered."""

    AUTO = "auto"  # Long query OR scores close (spec-compliant default)
    QUERY_LENGTH_ONLY = "query_length_only"  # Only expand for long queries
    NEVER = "never"  # Disable expansion
    ALWAYS = "always"  # Always expand (for debugging)


@dataclass
class ChunkResult:
    """A chunk retrieval result with all scoring metadata."""

    chunk_id: str  # Phase 7E: canonical 'id' field
    document_id: str
    parent_section_id: str
    order: int
    level: int
    heading: str
    text: str
    token_count: int

    # Scoring information
    vec_score: Optional[float] = None  # Vector similarity score
    bm25_score: Optional[float] = None  # BM25/keyword score
    fused_score: Optional[float] = None  # Final fused score
    fusion_method: Optional[str] = None  # Method used for fusion

    # Metadata
    is_combined: bool = False
    is_split: bool = False
    original_section_ids: List[str] = None
    boundaries_json: str = "{}"

    # Expansion tracking
    is_expanded: bool = False  # Was this chunk added via expansion?
    expansion_source: Optional[str] = None  # Which chunk triggered expansion

    def __post_init__(self):
        """Ensure required fields are populated."""
        if self.original_section_ids is None:
            self.original_section_ids = []


class BM25Retriever:
    """
    BM25/keyword retriever using Neo4j full-text search.
    Neo4j's full-text search uses Lucene under the hood, providing BM25 scoring.
    """

    def __init__(self, neo4j_driver: Driver, index_name: str = "chunk_text_index"):
        self.driver = neo4j_driver
        self.index_name = index_name
        self._ensure_fulltext_index()

    def _ensure_fulltext_index(self):
        """Create full-text index if it doesn't exist."""
        create_index_query = """
        CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text, c.heading]
        """

        with self.driver.session() as session:
            try:
                session.run(create_index_query)
                logger.info(f"Full-text index '{self.index_name}' ensured")
            except Exception as e:
                # Index might already exist with different config
                logger.warning(f"Could not create full-text index: {e}")

    def search(
        self, query: str, top_k: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Perform BM25 search on Chunk.text using Neo4j full-text search.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (e.g., document_id)

        Returns:
            List of ChunkResult objects with BM25 scores
        """
        start_time = time.time()

        # Build WHERE clause for filters
        where_clauses = []
        params = {"query": query, "limit": top_k}

        if filters:
            for key, value in filters.items():
                param_name = f"filter_{key}"
                where_clauses.append(f"c.{key} = ${param_name}")
                params[param_name] = value

        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Full-text search query with BM25 scoring
        search_query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node AS c, score
        WHERE (c:Chunk){where_clause}
        RETURN
            c.id AS chunk_id,
            c.document_id AS document_id,
            c.parent_section_id AS parent_section_id,
            c.order AS `order`,
            c.level AS level,
            c.heading AS heading,
            c.text AS text,
            c.token_count AS token_count,
            c.is_combined AS is_combined,
            c.is_split AS is_split,
            c.original_section_ids AS original_section_ids,
            c.boundaries_json AS boundaries_json,
            score AS bm25_score
        ORDER BY score DESC
        LIMIT $limit
        """

        params["index_name"] = self.index_name

        results = []
        try:
            with self.driver.session() as session:
                result = session.run(search_query, params)

                for record in result:
                    results.append(
                        ChunkResult(
                            chunk_id=record["chunk_id"],
                            document_id=record["document_id"],
                            parent_section_id=record["parent_section_id"],
                            order=record["order"],
                            level=record["level"],
                            heading=record["heading"] or "",
                            text=record["text"],
                            token_count=record["token_count"],
                            bm25_score=record["bm25_score"],
                            is_combined=record["is_combined"],
                            is_split=record["is_split"],
                            original_section_ids=record["original_section_ids"] or [],
                            boundaries_json=record["boundaries_json"] or "{}",
                        )
                    )

                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"BM25 search completed: query='{query[:50]}...', "
                    f"results={len(results)}, time={elapsed_ms:.2f}ms"
                )

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise

        return results


class VectorRetriever:
    """Vector retriever using Qdrant."""

    def __init__(
        self, qdrant_client: QdrantClient, embedder, collection_name: str = "chunks"
    ):
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def search(
        self, query: str, top_k: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Perform vector search on chunks using Qdrant.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (e.g., document_id)

        Returns:
            List of ChunkResult objects with vector scores
        """
        start_time = time.time()

        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Search Qdrant
        results = []
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            for hit in search_results:
                payload = hit.payload
                results.append(
                    ChunkResult(
                        chunk_id=payload.get("id", hit.id),
                        document_id=payload.get("document_id", ""),
                        parent_section_id=payload.get("parent_section_id", ""),
                        order=payload.get("order", 0),
                        level=payload.get("level", 3),
                        heading=payload.get("heading", ""),
                        text=payload.get("text", ""),
                        token_count=payload.get("token_count", 0),
                        vec_score=hit.score,
                        is_combined=payload.get("is_combined", False),
                        is_split=payload.get("is_split", False),
                        original_section_ids=payload.get("original_section_ids", []),
                        boundaries_json=payload.get("boundaries_json", "{}"),
                    )
                )

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Vector search completed: query='{query[:50]}...', "
                f"results={len(results)}, time={elapsed_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

        return results


class HybridRetriever:
    """
    Main hybrid retrieval engine implementing Phase 7E-2 requirements.
    Combines vector and BM25 search with fusion and expansion.
    """

    def __init__(
        self,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
        embedder,
        tokenizer: Optional[TokenizerService] = None,
    ):
        """
        Initialize hybrid retriever with all components.

        Args:
            neo4j_driver: Neo4j driver for BM25 and graph operations
            qdrant_client: Qdrant client for vector search
            embedder: Embedding provider for query vectorization
            tokenizer: Tokenizer service for token counting (optional)
        """
        self.neo4j_driver = neo4j_driver
        self.bm25_retriever = BM25Retriever(neo4j_driver)
        self.vector_retriever = VectorRetriever(qdrant_client, embedder)
        self.tokenizer = tokenizer or TokenizerService()

        # Load configuration
        config = get_config()
        search_config = config.search

        # Fusion configuration
        self.fusion_method = FusionMethod(
            getattr(search_config.hybrid, "method", "rrf")
        )
        self.rrf_k = getattr(search_config.hybrid, "rrf_k", 60)
        self.fusion_alpha = getattr(search_config.hybrid, "fusion_alpha", 0.6)

        # Expansion configuration
        expansion_config = getattr(search_config.hybrid, "expansion", {})
        if hasattr(expansion_config, "enabled"):
            self.expansion_enabled = expansion_config.enabled
            self.expansion_max_neighbors = getattr(expansion_config, "max_neighbors", 1)
            self.expansion_query_min_tokens = getattr(
                expansion_config, "query_min_tokens", 12
            )
            self.expansion_score_delta_max = getattr(
                expansion_config, "score_delta_max", 0.02
            )
        else:
            # Default expansion settings if not configured
            self.expansion_enabled = True
            self.expansion_max_neighbors = 1
            self.expansion_query_min_tokens = 12
            self.expansion_score_delta_max = 0.02

        # Context budget
        self.context_max_tokens = getattr(
            search_config.response, "answer_context_max_tokens", 4500
        )

        logger.info(
            f"HybridRetriever initialized: fusion={self.fusion_method.value}, "
            f"rrf_k={self.rrf_k}, alpha={self.fusion_alpha}, "
            f"expansion={'enabled' if self.expansion_enabled else 'disabled'}, "
            f"context_budget={self.context_max_tokens}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        expand: bool = True,
        expand_when: str = "auto",
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """
        Perform hybrid retrieval with fusion and optional expansion.

        Args:
            query: Search query text
            top_k: Number of final results to return
            filters: Optional filters for search
            expand: Whether to perform adjacency expansion

        Returns:
            Tuple of (results, metrics) where metrics contains timing and diagnostic info
        """
        start_time = time.time()
        metrics = {}

        # Step 1: Parallel BM25 and vector search
        # Retrieve more candidates for fusion (3x top_k)
        candidate_k = min(top_k * 3, 100)

        # BM25 search
        bm25_start = time.time()
        bm25_results = self.bm25_retriever.search(query, candidate_k, filters)
        metrics["bm25_time_ms"] = (time.time() - bm25_start) * 1000
        metrics["bm25_count"] = len(bm25_results)

        # Vector search
        vec_start = time.time()
        vec_results = self.vector_retriever.search(query, candidate_k, filters)
        metrics["vec_time_ms"] = (time.time() - vec_start) * 1000
        metrics["vec_count"] = len(vec_results)

        # Step 2: Fuse rankings
        fusion_start = time.time()
        if self.fusion_method == FusionMethod.RRF:
            fused_results = self._rrf_fusion(bm25_results, vec_results)
        else:
            fused_results = self._weighted_fusion(bm25_results, vec_results)
        metrics["fusion_time_ms"] = (time.time() - fusion_start) * 1000
        metrics["fusion_method"] = self.fusion_method.value

        # Step 3: Take top_k from fusion as seeds
        fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
        seeds = fused_results[:top_k]
        metrics["seed_count"] = len(seeds)

        # Step 4: Gating decision for expansion
        when = ExpandWhen(expand_when)
        triggered, reason, score_delta, query_tokens = self._should_expand(
            query, seeds, when
        )
        metrics["expansion_triggered"] = bool(expand and triggered)
        metrics["expansion_reason"] = reason
        metrics["scores_close_delta"] = score_delta
        metrics["query_tokens"] = query_tokens

        # Step 5: Optional bounded adjacency expansion
        # Note: Expansion ADDS neighbors without re-limiting to top_k
        all_results = list(seeds)
        if expand and triggered and self.expansion_enabled:
            expansion_start = time.time()
            expanded_results = self._bounded_expansion(query, seeds)
            all_results.extend(expanded_results)
            metrics["expansion_time_ms"] = (time.time() - expansion_start) * 1000
            metrics["expansion_count"] = len(expanded_results)
            metrics["expanded_source_count"] = min(
                len(seeds), getattr(self, "max_sources_to_expand", 5)
            )
        else:
            metrics["expansion_count"] = 0
            metrics["expanded_source_count"] = 0

        # Step 6: Dedup and final sort (no top_k cap - budget enforces limit)
        all_results = self._dedup_results(all_results)
        all_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
        final_results = all_results

        # Step 6: Context assembly with budget enforcement
        context_start = time.time()
        final_results = self._enforce_context_budget(final_results)
        metrics["context_assembly_ms"] = (time.time() - context_start) * 1000
        metrics["final_count"] = len(final_results)
        metrics["total_tokens"] = sum(r.token_count for r in final_results)

        metrics["total_time_ms"] = (time.time() - start_time) * 1000

        # Phase 7E-4: Record metrics to aggregator if enabled
        config = get_config()
        if config.monitoring.metrics_aggregation_enabled:
            aggregator = get_metrics_aggregator()
            aggregator.record_retrieval(
                latency_ms=metrics["total_time_ms"],
                chunks_returned=len(final_results),
                expanded=metrics.get("expansion_triggered", False),
                fusion_method=self.fusion_method.value,
            )

            # Emit Prometheus metrics for expansion
            if metrics.get("expansion_triggered", False):
                retrieval_expansion_total.labels(
                    expansion_reason=metrics.get("expansion_reason", "unknown")
                ).inc()
                retrieval_expansion_chunks_added.observe(
                    metrics.get("expansion_count", 0)
                )

            # Update current expansion rate gauge (rolling window)
            window_metrics = aggregator.get_window_metrics(window_seconds=300)
            if "retrieval" in window_metrics:
                expansion_rate = window_metrics["retrieval"].expansion_rate
                retrieval_expansion_rate_current.set(expansion_rate)
                metrics["expansion_rate_5min"] = expansion_rate

            # Phase 7E-4: Check SLOs if monitoring enabled
            if config.monitoring.slo_monitoring_enabled:
                # Get recent p95 from aggregator
                if "retrieval" in window_metrics:
                    p95_latency = window_metrics["retrieval"].p95_latency
                    expansion_rate = window_metrics["retrieval"].expansion_rate

                    # Check retrieval p95 SLO (target: ≤500ms)
                    if p95_latency > config.monitoring.retrieval_p95_target_ms:
                        logger.warning(
                            "Retrieval p95 SLO violation",
                            p95_ms=p95_latency,
                            target_ms=config.monitoring.retrieval_p95_target_ms,
                            query_preview=query[:50],
                        )

                    # Check expansion rate SLO (target: 10-40%)
                    if expansion_rate < config.monitoring.expansion_rate_min:
                        logger.warning(
                            "Expansion rate below minimum",
                            expansion_rate=expansion_rate,
                            min_threshold=config.monitoring.expansion_rate_min,
                        )
                    elif expansion_rate > config.monitoring.expansion_rate_max:
                        logger.warning(
                            "Expansion rate above maximum",
                            expansion_rate=expansion_rate,
                            max_threshold=config.monitoring.expansion_rate_max,
                        )

        logger.info(
            f"Hybrid retrieval complete: query='{query[:50]}...', "
            f"results={len(final_results)}, tokens={metrics['total_tokens']}, "
            f"time={metrics['total_time_ms']:.2f}ms"
        )

        return final_results, metrics

    def _rrf_fusion(
        self, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Reciprocal Rank Fusion (RRF) - robust fusion without parameter tuning.

        RRF formula: score = Σ(1 / (k + rank_i))
        where k is a constant (default 60) and rank_i is the rank in result list i

        Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
        """
        # Build rank dictionaries
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        vec_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vec_results)}

        # Build combined result map
        all_chunks = {}
        for r in bm25_results:
            all_chunks[r.chunk_id] = r
        for r in vec_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r

        # Calculate RRF scores
        for chunk_id, chunk in all_chunks.items():
            bm25_rank = bm25_ranks.get(chunk_id, float("inf"))
            vec_rank = vec_ranks.get(chunk_id, float("inf"))

            # RRF formula
            rrf_score = 0.0
            if bm25_rank != float("inf"):
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            if vec_rank != float("inf"):
                rrf_score += 1.0 / (self.rrf_k + vec_rank)

            chunk.fused_score = rrf_score
            chunk.fusion_method = "rrf"

            # Preserve original scores for diagnostics
            if chunk_id in bm25_ranks:
                chunk.bm25_score = chunk.bm25_score or 0
            if chunk_id in vec_ranks:
                chunk.vec_score = chunk.vec_score or 0

        return list(all_chunks.values())

    def _weighted_fusion(
        self, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Weighted linear combination fusion.

        score = α * vec_score + (1-α) * bm25_score
        where α is the vector weight (default 0.6)

        Note: Requires score normalization since BM25 and vector scores have different ranges.
        """

        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[ChunkResult], score_attr: str):
            scores = [
                getattr(r, score_attr, 0) for r in results if getattr(r, score_attr)
            ]
            if not scores:
                return
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return

            for r in results:
                score = getattr(r, score_attr, 0)
                if score:
                    normalized = (score - min_score) / (max_score - min_score)
                    setattr(r, f"norm_{score_attr}", normalized)

        normalize_scores(bm25_results, "bm25_score")
        normalize_scores(vec_results, "vec_score")

        # Build combined result map
        all_chunks = {}
        for r in bm25_results:
            all_chunks[r.chunk_id] = r
            r.norm_bm25_score = getattr(r, "norm_bm25_score", 0)
        for r in vec_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r
            else:
                # Update vector score if already exists
                all_chunks[r.chunk_id].vec_score = r.vec_score
                all_chunks[r.chunk_id].norm_vec_score = getattr(r, "norm_vec_score", 0)

        # Calculate weighted scores
        for chunk in all_chunks.values():
            norm_vec = getattr(chunk, "norm_vec_score", 0)
            norm_bm25 = getattr(chunk, "norm_bm25_score", 0)

            chunk.fused_score = (
                self.fusion_alpha * norm_vec + (1 - self.fusion_alpha) * norm_bm25
            )
            chunk.fusion_method = "weighted"

        return list(all_chunks.values())

    def _should_expand(
        self, query: str, seeds: List[ChunkResult], when: ExpandWhen
    ) -> Tuple[bool, str, float, int]:
        """
        Determine if expansion should be triggered based on gating conditions.

        Returns: (triggered, reason, score_delta, query_tokens)
        """
        query_tokens = self.tokenizer.count_tokens(query)
        scores = [r.fused_score or 0.0 for r in seeds]

        # Calculate score delta between top results
        if len(scores) >= 2:
            score_delta = abs(scores[0] - scores[1])
        else:
            score_delta = 1.0  # Large delta if only one result

        reason = "none"
        triggered = False

        if when == ExpandWhen.NEVER:
            reason = "disabled"
        elif when == ExpandWhen.ALWAYS:
            triggered, reason = True, "forced"
        elif when == ExpandWhen.QUERY_LENGTH_ONLY:
            if query_tokens >= self.expansion_query_min_tokens:
                triggered, reason = True, "query_long"
        else:  # AUTO (spec-compliant default)
            if query_tokens >= self.expansion_query_min_tokens:
                triggered, reason = True, "query_long"
            elif len(scores) >= 2 and score_delta <= self.expansion_score_delta_max:
                triggered, reason = True, "scores_close"

        return triggered, reason, score_delta, query_tokens

    def _result_id(self, r: ChunkResult) -> Tuple:
        """Get unique identifier for a chunk result."""
        return ("chunk_id", r.chunk_id)

    def _dedup_results(self, results: List[ChunkResult]) -> List[ChunkResult]:
        """Remove duplicate chunks by identity."""
        seen = set()
        deduped = []
        for r in results:
            rid = self._result_id(r)
            if rid not in seen:
                seen.add(rid)
                deduped.append(r)
        return deduped

    def _bounded_expansion(
        self, query: str, seeds: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Bounded adjacency expansion via NEXT_CHUNK relationships.

        Adds ±1 neighbors from the first N seeds (default N=5).
        Neighbors never outrank their source (score = source_score × 0.5).

        Reference: Phase 7E Canonical Spec L1425-1434
        """
        if not seeds:
            return []

        # Limit to first 5 seeds for expansion (bounded)
        max_sources = getattr(self, "max_sources_to_expand", 5)
        eligible = seeds[:max_sources]

        # Track existing chunks to avoid duplicates
        seen_ids = {self._result_id(r) for r in seeds}
        expanded = []

        chunk_ids_to_expand = [r.chunk_id for r in eligible]

        expansion_query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {id: chunk_id})

        // Find previous chunk
        OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)

        // Find next chunk
        OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)

        WITH chunk_id,
             collect(DISTINCT prev) AS prev_chunks,
             collect(DISTINCT next) AS next_chunks

        UNWIND (prev_chunks + next_chunks) AS neighbor
        WITH chunk_id, neighbor
        WHERE neighbor IS NOT NULL

        RETURN DISTINCT
            neighbor.id AS chunk_id,
            neighbor.document_id AS document_id,
            neighbor.parent_section_id AS parent_section_id,
            neighbor.order AS `order`,
            neighbor.level AS level,
            neighbor.heading AS heading,
            neighbor.text AS text,
            neighbor.token_count AS token_count,
            neighbor.is_combined AS is_combined,
            neighbor.is_split AS is_split,
            neighbor.original_section_ids AS original_section_ids,
            neighbor.boundaries_json AS boundaries_json,
            chunk_id AS source_chunk
        """

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(expansion_query, chunk_ids=chunk_ids_to_expand)

                for record in result:
                    neighbor_id = record["chunk_id"]
                    rid = ("chunk_id", neighbor_id)

                    if rid in seen_ids:
                        continue  # Skip duplicates

                    seen_ids.add(rid)

                    # Find source chunk to get its score
                    source_chunk_id = record["source_chunk"]
                    source_score = 0.0
                    for seed in seeds:
                        if seed.chunk_id == source_chunk_id:
                            source_score = seed.fused_score or 0.0
                            break

                    # Neighbor score = source_score × 0.5 (never outrank source)
                    neighbor_score = source_score * 0.5

                    expanded.append(
                        ChunkResult(
                            chunk_id=neighbor_id,
                            document_id=record["document_id"],
                            parent_section_id=record["parent_section_id"],
                            order=record["order"],
                            level=record["level"],
                            heading=record["heading"] or "",
                            text=record["text"],
                            token_count=record["token_count"],
                            is_combined=record["is_combined"],
                            is_split=record["is_split"],
                            original_section_ids=record["original_section_ids"] or [],
                            boundaries_json=record["boundaries_json"] or "{}",
                            is_expanded=True,
                            expansion_source=source_chunk_id,
                            fused_score=neighbor_score,
                        )
                    )

            # Calculate query tokens for logging
            query_token_count = self.tokenizer.count_tokens(query)
            logger.info(
                f"Adjacency expansion: query_tokens={query_token_count}, "
                f"expanded={len(expanded)} chunks"
            )

        except Exception as e:
            logger.error(f"Adjacency expansion failed: {e}")
            # Don't fail the whole search if expansion fails

        return expanded

    def _enforce_context_budget(self, results: List[ChunkResult]) -> List[ChunkResult]:
        """
        Enforce context budget by limiting total tokens.

        Groups by parent_section_id, sorts by order, and trims from tail
        when exceeding budget.

        Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
        """
        if not results:
            return results

        # Group by parent_section_id for proper ordering
        grouped = {}
        for r in results:
            parent = r.parent_section_id or r.chunk_id
            if parent not in grouped:
                grouped[parent] = []
            grouped[parent].append(r)

        # Sort within each group by order
        for chunks in grouped.values():
            chunks.sort(key=lambda x: x.order)

        # Assemble final list, respecting token budget
        final_results = []
        total_tokens = 0

        # Process groups by best score first
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: max(c.fused_score or 0 for c in x[1]),
            reverse=True,
        )

        for parent_id, chunks in sorted_groups:
            for chunk in chunks:
                if total_tokens + chunk.token_count <= self.context_max_tokens:
                    final_results.append(chunk)
                    total_tokens += chunk.token_count
                else:
                    # Budget exceeded
                    logger.info(
                        f"Context budget enforced: {len(final_results)} chunks, "
                        f"{total_tokens} tokens (max: {self.context_max_tokens})"
                    )
                    break

            if total_tokens >= self.context_max_tokens:
                break

        # Sort final results by parent_section_id and order for coherent context
        final_results.sort(key=lambda x: (x.parent_section_id or "", x.order))

        return final_results

    def assemble_context(self, chunks: List[ChunkResult]) -> str:
        """
        Assemble chunks into coherent context string with headings preserved.

        Args:
            chunks: List of chunks to assemble

        Returns:
            Stitched context string
        """
        if not chunks:
            return ""

        # Group by parent and maintain order
        context_parts = []
        current_parent = None

        for chunk in chunks:
            # Add parent section heading if switching contexts
            if chunk.parent_section_id != current_parent:
                if chunk.heading:
                    context_parts.append(f"\n## {chunk.heading}\n")
                current_parent = chunk.parent_section_id

            # Add chunk text
            context_parts.append(chunk.text)

            # Add expansion indicator if applicable
            if chunk.is_expanded:
                context_parts.append(f" [expanded from: {chunk.expansion_source}]")

        return "\n".join(context_parts)
