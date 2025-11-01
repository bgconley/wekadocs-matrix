"""
Query Service for MCP Server
Integrates hybrid search, ranking, and response building.
Provides cached embedder and connection management.
Pre-Phase 7 B4: Modified to use embedding provider abstraction.
Phase 7C: Integrated with reranking provider for post-ANN refinement.
Task 7C.8: Integrated with SessionTracker for multi-turn conversation support.
"""

import json
import time
from typing import Any, Dict, List, Optional

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.factory import ProviderFactory
from src.providers.rerank.base import RerankProvider
from src.providers.tokenizer_service import TokenizerService
from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import ChunkResult, HybridRetriever
from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore, SearchResult
from src.query.planner import QueryPlanner
from src.query.ranking import RankingFeatures, RankedResult, rank_results
from src.query.response_builder import Response, Verbosity, build_response
from src.query.session_tracker import SessionTracker
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    mcp_search_response_size_bytes,
    mcp_search_verbosity_total,
)

logger = get_logger(__name__)


class QueryService:
    """
    Service for executing queries against the knowledge graph.
    Manages connections, embedder caching, and query execution pipeline.
    """

    def __init__(self):
        self.config = get_config()
        self._embedder: Optional[EmbeddingProvider] = None
        self._reranker: Optional[RerankProvider] = None  # Phase 7C: Reranker cache
        self._search_engine: Optional[HybridSearchEngine] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._context_assembler: Optional[ContextAssembler] = None
        self._planner: Optional[QueryPlanner] = None
        self._session_tracker: Optional[SessionTracker] = (
            None  # Task 7C.8: Session tracking
        )

        logger.info("QueryService initialized")

    def _get_embedder(self) -> EmbeddingProvider:
        """
        Get or initialize the cached embedder.
        Phase 7C Task 7C.1: Uses ProviderFactory for ENV-selectable providers.
        """
        if self._embedder is None:
            expected_dims = self.config.embedding.dims

            logger.info(
                f"Loading embedding provider from config: provider={self.config.embedding.provider}, "
                f"model={self.config.embedding.embedding_model}, dims={expected_dims}"
            )

            # Phase 7C: Use provider factory for ENV-based selection
            factory = ProviderFactory()
            self._embedder = factory.create_embedding_provider()

            # Validate dimensions match configuration
            if self._embedder.dims != expected_dims:
                raise ValueError(
                    f"Provider dimension mismatch: expected {expected_dims}, "
                    f"got {self._embedder.dims}"
                )

            logger.info(
                f"Embedding provider loaded successfully: "
                f"provider={self._embedder.provider_name}, "
                f"model={self._embedder.model_id}, "
                f"dims={self._embedder.dims}"
            )
        return self._embedder

    def _get_reranker(self) -> Optional[RerankProvider]:
        """
        Get or initialize the cached reranker.
        Phase 7C: Uses provider factory for ENV-based configuration.

        Returns None if reranking is disabled (RERANK_PROVIDER=none).
        """
        if self._reranker is None:
            try:
                factory = ProviderFactory()
                self._reranker = factory.create_rerank_provider()

                logger.info(
                    f"Reranker loaded successfully: "
                    f"provider={self._reranker.provider_name}, "
                    f"model={self._reranker.model_id}"
                )
            except Exception as e:
                # If reranker initialization fails, log warning but continue without reranking
                logger.warning(
                    f"Reranker initialization failed: {e}. Continuing without reranking."
                )
                self._reranker = None

        return self._reranker

    def _get_search_engine(self) -> HybridSearchEngine:
        """Get or initialize the search engine."""
        if self._search_engine is None:
            manager = get_connection_manager()

            # Get connections
            neo4j_driver = manager.get_neo4j_driver()
            qdrant_client = manager.get_qdrant_client()
            embedder = self._get_embedder()

            # Determine vector store (from config)
            primary = self.config.search.vector.primary

            if primary == "qdrant":
                collection_name = self.config.search.vector.qdrant.collection_name
                vector_store = QdrantVectorStore(qdrant_client, collection_name)
                logger.info(f"Using Qdrant vector store: {collection_name}")
            else:
                # Neo4j vectors (if configured)
                from src.query.hybrid_search import Neo4jVectorStore

                index_name = self.config.search.vector.neo4j.index_name
                vector_store = Neo4jVectorStore(neo4j_driver, index_name)
                logger.info(f"Using Neo4j vector store: {index_name}")

            # Phase 7C: Get reranker (may be None if disabled)
            reranker = self._get_reranker()

            self._search_engine = HybridSearchEngine(
                vector_store=vector_store,
                neo4j_driver=neo4j_driver,
                embedder=embedder,
                reranker=reranker,  # Phase 7C: Pass reranker
            )
            logger.info("Search engine initialized with reranking support")

        return self._search_engine

    def _get_7e_retriever(self) -> HybridRetriever:
        """Get or initialize the Phase 7E hybrid retriever."""
        if self._hybrid_retriever is None:
            manager = get_connection_manager()
            neo4j_driver = manager.get_neo4j_driver()
            qdrant_client = manager.get_qdrant_client()
            embedder = self._get_embedder()

            self._hybrid_retriever = HybridRetriever(
                neo4j_driver=neo4j_driver,
                qdrant_client=qdrant_client,
                embedder=embedder,
                tokenizer=TokenizerService(),
            )

            logger.info("Phase 7E HybridRetriever initialized")

        return self._hybrid_retriever

    def _get_context_assembler(self) -> ContextAssembler:
        """Get or initialize the context assembler for stitched responses."""
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler()
            logger.info("ContextAssembler initialized for stitched responses")
        return self._context_assembler

    def _wrap_chunks_as_ranked(
        self, chunks: List[ChunkResult]
    ) -> List[RankedResult]:
        """Adapt ChunkResult objects to RankedResult instances for response building."""

        ranked: List[RankedResult] = []
        for index, chunk in enumerate(chunks):
            score = float(
                chunk.fused_score
                if chunk.fused_score is not None
                else chunk.vector_score
                if chunk.vector_score is not None
                else chunk.bm25_score
                if chunk.bm25_score is not None
                else 0.0
            )

            metadata: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_uri": getattr(chunk, "document_uri", None),
                "parent_section_id": chunk.parent_section_id,
                "heading": chunk.heading,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "level": chunk.level,
                "is_combined": chunk.is_combined,
                "is_split": chunk.is_split,
                "boundaries_json": chunk.boundaries_json,
                "score_kind": "similarity",
                "fusion_method": chunk.fusion_method,
                "bm25_score": chunk.bm25_score,
                "vector_score": chunk.vector_score,
                "vec_score": chunk.vector_score,
                "bm25_rank": chunk.bm25_rank,
                "vector_rank": chunk.vector_rank,
            }

            metadata["anchor"] = getattr(chunk, "anchor", None)

            search_result = SearchResult(
                node_id=chunk.chunk_id,
                node_label="Chunk",
                score=score,
                distance=0,
                metadata=metadata,
            )

            features = RankingFeatures(
                semantic_score=float(chunk.vector_score or chunk.fused_score or 0.0),
                graph_distance_score=0.0,
                recency_score=0.0,
                entity_priority_score=1.0,
                coverage_score=0.0,
                final_score=score,
            )

            ranked.append(RankedResult(result=search_result, features=features, rank=index + 1))

        return ranked

    def _get_planner(self) -> QueryPlanner:
        """Get or initialize the query planner."""
        if self._planner is None:
            self._planner = QueryPlanner()
            logger.info("Query planner initialized")
        return self._planner

    def _get_session_tracker(self) -> SessionTracker:
        """
        Get or initialize the session tracker.
        Task 7C.8: Manages multi-turn conversation sessions.
        """
        if self._session_tracker is None:
            manager = get_connection_manager()
            neo4j_driver = manager.get_neo4j_driver()
            self._session_tracker = SessionTracker(neo4j_driver)
            logger.info("Session tracker initialized")
        return self._session_tracker

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        expand_graph: bool = True,
        find_paths: bool = False,
        verbosity: str = "graph",
        session_id: Optional[str] = None,  # Task 7C.8: Multi-turn session ID
        turn: Optional[int] = None,  # Task 7C.8: Turn number within session
    ) -> Response:
        """
        Execute a search query and return formatted response.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional filters for vector search
            expand_graph: Whether to expand from seeds via graph
            find_paths: Whether to find connecting paths
            verbosity: Response detail level (full=text only, graph=text+relationships, default=graph)
            session_id: Optional session ID for multi-turn tracking (Task 7C.8)
            turn: Optional turn number within session (Task 7C.8)

        Returns:
            Response with Markdown and JSON including evidence and confidence

        Raises:
            Exception: If search fails
            ValueError: If verbosity is invalid
        """
        start_time = time.time()

        try:
            # Convert verbosity string to enum
            try:
                verb_enum = Verbosity(verbosity)
            except ValueError:
                raise ValueError(
                    f"Invalid verbosity '{verbosity}'. Must be one of: full, graph"
                )

            # Classify intent using planner
            planner = self._get_planner()
            query_plan = planner.plan(query, filters=filters)
            intent = query_plan.intent
            logger.info(f"Query intent classified: {intent}")

            # Task 7C.8: Multi-turn session tracking
            query_id = None
            focused_entity_ids: List[str] = []

            if session_id:
                tracker = self._get_session_tracker()

                # Create query node
                query_id = tracker.create_query(session_id, query, turn or 1)
                logger.info(
                    f"Created query {query_id} for session {session_id}, turn {turn}"
                )

                # Extract focused entities from current query
                focused_entities_current = tracker.extract_focused_entities(
                    query_id, query
                )
                logger.debug(
                    f"Extracted {len(focused_entities_current)} entities from current query"
                )

                # Get focused entities from recent session history (last 3 turns)
                session_focus = tracker.get_session_focused_entities(
                    session_id, last_n_turns=3
                )
                focused_entity_ids = [e["entity_id"] for e in session_focus]

                if focused_entity_ids:
                    logger.info(
                        f"Entity focus bias active: {len(focused_entity_ids)} entities from "
                        f"last 3 turns will boost retrieval"
                    )

            # Pre-Phase 7 B4: Add embedding_version filter to ensure version consistency
            # This ensures we only retrieve vectors created with the current embedding model
            filters = dict(filters or {})

            # Add embedding_version to filters
            filters.setdefault("embedding_version", self.config.embedding.version)
            logger.debug(
                f"Added embedding_version filter: {self.config.embedding.version}"
            )

            use_phase7e = bool(getattr(self.config.search.hybrid, "enabled", True)) and bool(
                getattr(getattr(self.config.search.hybrid, "bm25", None), "enabled", False)
            )

            assembled_md: Optional[str] = None
            ranked_results: List[RankedResult]
            retrieval_sections: List[Dict[str, Any]] = []
            timing: Dict[str, float]

            if use_phase7e:
                retriever = self._get_7e_retriever()
                top_k_value = top_k or getattr(self.config.search.hybrid, "top_k", 20)
                chunks, metrics = retriever.retrieve(
                    query=query,
                    top_k=top_k_value,
                    filters=filters,
                    expand=expand_graph,
                )

                assembler = self._get_context_assembler()
                assembled_context = assembler.assemble(chunks, query=query)
                assembled_md = assembler.format_with_citations(assembled_context)

                ranked_results = self._wrap_chunks_as_ranked(chunks)

                retrieval_sections = [
                    {
                        "section_id": chunk.chunk_id,
                        "rank": idx + 1,
                        "score_vec": float(chunk.vector_score or 0.0),
                        "score_text": float(chunk.bm25_score or 0.0),
                        "score_graph": 0.0,
                        "score_combined": float(
                            chunk.fused_score
                            or chunk.vector_score
                            or chunk.bm25_score
                            or 0.0
                        ),
                        "retrieval_method": "hybrid_phase7e",
                    }
                    for idx, chunk in enumerate(chunks)
                ]

                timing = {
                    "bm25_ms": metrics.get("bm25_time_ms", 0.0),
                    "vector_search_ms": metrics.get("vec_time_ms", 0.0),
                    "fusion_ms": metrics.get("fusion_time_ms", 0.0),
                    "expansion_ms": metrics.get("expansion_time_ms", 0.0),
                    "context_assembly_ms": metrics.get("context_assembly_ms", 0.0),
                    "ranking_ms": 0.0,
                    "total_ms": metrics.get(
                        "total_time_ms", (time.time() - start_time) * 1000
                    ),
                }

                logger.info(
                    "Phase 7E hybrid retrieval completed: results=%d, time=%.1fms",
                    len(chunks),
                    timing["total_ms"],
                )

            else:
                search_start = time.time()
                search_engine = self._get_search_engine()
                search_results = search_engine.search(
                    query_text=query,
                    k=top_k,
                    filters=filters,
                    expand_graph=expand_graph,
                    find_paths=find_paths,
                    focused_entity_ids=(
                        focused_entity_ids if focused_entity_ids else None
                    ),
                )
                search_time = time.time() - search_start

                logger.info(
                    f"Search completed: {search_results.total_found} results in {search_time*1000:.1f}ms"
                )

                retrieval_sections = [
                    {
                        "section_id": result.node_id,
                        "rank": idx + 1,
                        "score_vec": getattr(result, "vector_score", 0.0),
                        "score_text": getattr(result, "text_score", 0.0),
                        "score_graph": getattr(result, "graph_score", 0.0),
                        "score_combined": result.score,
                        "retrieval_method": "hybrid",
                    }
                    for idx, result in enumerate(search_results.results)
                ]

                rank_start = time.time()
                ranked_results = rank_results(search_results.results)
                rank_time = time.time() - rank_start

                logger.info(
                    f"Ranking completed: {len(ranked_results)} results in {rank_time*1000:.1f}ms"
                )

                timing = {
                    "vector_search_ms": search_results.vector_time_ms,
                    "graph_expansion_ms": search_results.graph_time_ms,
                    "ranking_ms": rank_time * 1000,
                    "total_ms": (time.time() - start_time) * 1000,
                }
                assembled_md = None

            # Task 7C.8: Track retrieval if session tracking is active
            if session_id and query_id and retrieval_sections:
                tracker = self._get_session_tracker()
                tracker.track_retrieval(query_id, retrieval_sections)
                logger.debug(
                    f"Tracked {len(retrieval_sections)} retrieved sections for query {query_id}"
                )

            # Build response with verbosity mode
            manager = get_connection_manager()
            neo4j_driver = manager.get_neo4j_driver()

            # Task 7C.8: Pass session tracking info to response builder
            response = build_response(
                query=query,
                intent=intent,
                ranked_results=ranked_results,
                timing=timing,
                filters=filters,
                verbosity=verb_enum,
                neo4j_driver=neo4j_driver,
                query_id=query_id if session_id else None,  # Task 7C.8
                session_tracker=(
                    self._get_session_tracker() if session_id else None
                ),  # Task 7C.8
                assembled_context=assembled_md,
            )

            # Instrument metrics (E5)
            mcp_search_verbosity_total.labels(verbosity=verbosity).inc()

            # Measure response size
            response_json = json.dumps(response.to_dict())
            response_size = len(response_json.encode("utf-8"))
            mcp_search_response_size_bytes.labels(verbosity=verbosity).observe(
                response_size
            )

            logger.info(
                f"Query completed: verbosity={verbosity}, "
                f"response_size_kb={response_size/1024:.1f}, "
                f"confidence={response.answer_json.confidence:.2f}, "
                f"evidence_count={len(response.answer_json.evidence)}, "
                f"total_time={timing['total_ms']:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "embedder_loaded": self._embedder is not None,
            "search_engine_initialized": self._search_engine is not None,
            "planner_initialized": self._planner is not None,
            "model_name": (
                self.config.embedding.embedding_model if self._embedder else None
            ),
        }


# Global query service instance (initialized on demand)
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get or create the global query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service
