"""
Query Service for MCP Server
Integrates hybrid search, ranking, and response building.
Provides cached embedder and connection management.
Pre-Phase 7 B4: Modified to use embedding provider abstraction.
Phase 7C: Integrated with reranking provider for post-ANN refinement.
Task 7C.8: Integrated with SessionTracker for multi-turn conversation support.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.factory import ProviderFactory
from src.providers.rerank.base import RerankProvider
from src.providers.tokenizer_service import TokenizerService
from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import ChunkResult, HybridRetriever
from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore, SearchResult
from src.query.planner import QueryPlanner
from src.query.ranking import (  # Ranker bypassed - hybrid_retrieval handles ranking
    RankedResult,
)
from src.query.response_builder import (
    Response,
    StructuredResponse,
    Verbosity,
    build_response,
)
from src.query.session_tracker import SessionTracker
from src.shared.config import get_config, get_embedding_settings
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    mcp_search_response_size_bytes,
    mcp_search_verbosity_total,
)

# LGTM Phase 4: OTEL tracing for MCP server observability
try:
    import uuid

    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    import uuid

logger = get_logger(__name__)

# LGTM Phase 4: Tracer for MCP server spans
_tracer = trace.get_tracer("wekadocs.mcp") if OTEL_AVAILABLE else None


class QueryService:
    """
    Service for executing queries against the knowledge graph.
    Manages connections, embedder caching, and query execution pipeline.
    """

    def __init__(self):
        self.config = get_config()
        self.embedding_settings = get_embedding_settings()
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

    def _embedding_diag(self) -> Dict[str, Any]:
        diag = {
            "embedding_profile": self.embedding_settings.profile,
            "embedding_provider": self.embedding_settings.provider,
            "embedding_model": self.embedding_settings.model_id,
            "embedding_dims": self.embedding_settings.dims,
            "tokenizer_backend": self.embedding_settings.tokenizer_backend,
            "tokenizer_model_id": self.embedding_settings.tokenizer_model_id,
            "embedding_task": self.embedding_settings.task,
        }
        caps = getattr(self.embedding_settings, "capabilities", None)
        if caps:
            diag.update(
                {
                    "supports_dense": caps.supports_dense,
                    "supports_sparse": caps.supports_sparse,
                    "supports_colbert": caps.supports_colbert,
                    "supports_long_sequences": caps.supports_long_sequences,
                    "normalized_output": caps.normalized_output,
                    "multilingual": caps.multilingual,
                }
            )
        return diag

    def _get_embedder(self) -> EmbeddingProvider:
        """
        Get or initialize the cached embedder.
        Phase 7C Task 7C.1: Uses ProviderFactory for ENV-selectable providers.
        """
        if self._embedder is None:
            settings = self.embedding_settings
            expected_dims = settings.dims

            logger.info(
                "Loading embedding provider from profile",
                profile=settings.profile,
                provider=settings.provider,
                model=settings.model_id,
                dims=expected_dims,
            )

            # Phase 7C: Use provider factory for ENV-based selection / profile overrides
            factory = ProviderFactory()
            self._embedder = factory.create_embedding_provider(settings=settings)

            # Validate dimensions match configuration
            if self._embedder.dims != expected_dims:
                raise ValueError(
                    f"Provider dimension mismatch: expected {expected_dims}, "
                    f"got {self._embedder.dims}"
                )

            logger.info(
                "Embedding provider loaded successfully",
                provider=self._embedder.provider_name,
                model=self._embedder.model_id,
                dims=self._embedder.dims,
                profile=settings.profile,
                supports_sparse=getattr(settings.capabilities, "supports_sparse", None),
                supports_colbert=getattr(
                    settings.capabilities, "supports_colbert", None
                ),
            )
        return self._embedder

    def _get_reranker(self) -> Optional[RerankProvider]:
        """
        Get or initialize the cached reranker.
        Phase 7C: Uses provider factory for ENV-based configuration.

        Returns None if reranking is disabled (RERANK_PROVIDER=none).
        """
        reranker_cfg = getattr(self.config.search.hybrid, "reranker", None)
        if not reranker_cfg or not getattr(reranker_cfg, "enabled", False):
            return None

        if self._reranker is None:
            provider_hint = getattr(reranker_cfg, "provider", None)
            model_hint = getattr(reranker_cfg, "model", None)
            try:
                factory = ProviderFactory()
                self._reranker = factory.create_rerank_provider(
                    provider=provider_hint,
                    model=model_hint,
                )

                logger.info(
                    "Reranker loaded successfully: provider=%s, model=%s",
                    self._reranker.provider_name,
                    self._reranker.model_id,
                )
            except Exception as e:
                # If reranker initialization fails, log warning but continue without reranking
                logger.warning(
                    "Reranker initialization failed: %s. Continuing without reranking.",
                    e,
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
                vector_fields = getattr(
                    self.config.search.hybrid, "vector_fields", {"content": 1.0}
                )
                multi_vector_enabled = any(
                    key != "content" and (weight or 0) > 0
                    for key, weight in vector_fields.items()
                )
                query_vector_name = (
                    getattr(
                        self.config.search.vector.qdrant, "query_vector_name", "content"
                    )
                    or "content"
                )
                vector_store = QdrantVectorStore(
                    qdrant_client,
                    collection_name,
                    query_vector_name=query_vector_name,
                    use_named_vectors=multi_vector_enabled,
                )
                logger.info(
                    "Using Qdrant vector store: %s (named_vectors=%s)",
                    collection_name,
                    "on" if multi_vector_enabled else "off",
                )
            else:
                # Neo4j vectors (if configured)
                from src.query.hybrid_search import Neo4jVectorStore

                index_name = self.config.search.vector.neo4j.index_name
                vector_store = Neo4jVectorStore(
                    neo4j_driver,
                    index_name,
                    embedding_version=self.embedding_settings.version,
                )
                logger.info(f"Using Neo4j vector store: {index_name}")

            # Phase 7C: Get reranker (may be None if disabled)
            reranker = self._get_reranker()

            self._search_engine = HybridSearchEngine(
                vector_store=vector_store,
                neo4j_driver=neo4j_driver,
                embedder=embedder,
                reranker=reranker,  # Phase 7C: Pass reranker
                embedding_settings=self.embedding_settings,
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

            reranker_cfg = getattr(self.config.search.hybrid, "reranker", None)
            if reranker_cfg and getattr(reranker_cfg, "enabled", False):
                provider_hint = getattr(reranker_cfg, "provider", None) or "auto"
                model_hint = getattr(reranker_cfg, "model", None) or "auto"
                logger.info(
                    "config.search.hybrid.reranker enabled (provider=%s, model=%s)",
                    provider_hint,
                    model_hint,
                )

            self._hybrid_retriever = HybridRetriever(
                neo4j_driver=neo4j_driver,
                qdrant_client=qdrant_client,
                embedder=embedder,
                tokenizer=TokenizerService(),
                embedding_settings=self.embedding_settings,
            )

            logger.info("Phase 7E HybridRetriever initialized")

        return self._hybrid_retriever

    def search_sections_light(
        self,
        query: str,
        *,
        fetch_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """
        Lightweight helper that returns raw ChunkResult objects for section-level tools.
        """
        retriever = self._get_7e_retriever()
        chunks, metrics = retriever.retrieve(
            query=query,
            top_k=fetch_k,
            filters=filters or {},
            expand=True,
        )
        return chunks, metrics

    def _get_context_assembler(self) -> ContextAssembler:
        """Get or initialize the context assembler for stitched responses."""
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler()
            logger.info("ContextAssembler initialized for stitched responses")
        return self._context_assembler

    def _wrap_chunks_as_ranked(self, chunks: List[ChunkResult]) -> List[RankedResult]:
        """
        Adapt ChunkResult objects to RankedResult instances for response building.

        Note: hybrid_retrieval.py already performs sophisticated ranking (ColBERT,
        BGE cross-encoder, graph signals). This method simply wraps the pre-ranked
        chunks into RankedResult format without re-ranking.
        """
        from src.query.ranking import RankingFeatures

        ranked_results: List[RankedResult] = []

        for idx, chunk in enumerate(chunks):
            # Use rerank_score if available (from ColBERT/BGE), else fall back to fusion score
            primary_score = float(
                chunk.rerank_score
                if chunk.rerank_score is not None
                else (
                    chunk.fused_score
                    if chunk.fused_score is not None
                    else (
                        chunk.vector_score
                        if chunk.vector_score is not None
                        else chunk.bm25_score if chunk.bm25_score is not None else 0.0
                    )
                )
            )

            fusion_method = chunk.fusion_method or None
            score_kind = (
                "reranked"
                if chunk.rerank_score is not None
                else ("rrf" if fusion_method == "rrf" else "similarity")
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
                "score_kind": score_kind,
                "fusion_method": fusion_method,
                "bm25_score": chunk.bm25_score,
                "vector_score": chunk.vector_score,
                "vec_score": chunk.vector_score,
                "vector_score_kind": chunk.vector_score_kind,
                "bm25_rank": chunk.bm25_rank,
                "vector_rank": chunk.vector_rank,
                "graph_score": chunk.graph_score,
                "graph_distance": chunk.graph_distance,
                "graph_path": chunk.graph_path,
                "connection_count": chunk.connection_count,
                "mention_count": chunk.mention_count,
                "inherited_score": chunk.inherited_score,
                "rerank_score": chunk.rerank_score,
                "rerank_rank": chunk.rerank_rank,
                "reranker": chunk.reranker,
            }

            metadata["anchor"] = getattr(chunk, "anchor", None)

            search_result = SearchResult(
                node_id=chunk.chunk_id,
                node_label="Chunk",
                score=primary_score,
                distance=int(chunk.graph_distance or 0),
                metadata=metadata,
                path=chunk.graph_path,
            )

            # Create minimal RankingFeatures from chunk's existing scores
            # (hybrid_retrieval already computed these)
            features = RankingFeatures(
                semantic_score=primary_score,
                recall_score=float(chunk.vector_score or 0.0),
                rerank_score=float(chunk.rerank_score or 0.0),
                inherited_score=float(chunk.inherited_score or 0.0),
                graph_distance_score=float(chunk.graph_score or 0.0),
                final_score=primary_score,
            )

            ranked_results.append(
                RankedResult(
                    result=search_result,
                    features=features,
                    rank=idx + 1,  # Already ranked by hybrid_retrieval
                )
            )

        return ranked_results

    def _wrap_search_results_as_ranked(
        self, results: List[SearchResult]
    ) -> List[RankedResult]:
        """
        Wrap SearchResult objects as RankedResult without re-ranking.

        Used by the legacy HybridSearchEngine path. The search engine already
        orders results, so we just wrap them in the expected format.
        """
        from src.query.ranking import RankingFeatures

        ranked_results: List[RankedResult] = []

        for idx, result in enumerate(results):
            # Create minimal RankingFeatures from result metadata
            features = RankingFeatures(
                semantic_score=float(result.score or 0.0),
                recall_score=float(result.metadata.get("vector_score", 0.0) or 0.0),
                rerank_score=float(result.metadata.get("rerank_score", 0.0) or 0.0),
                graph_distance_score=float(
                    result.metadata.get("graph_score", 0.0) or 0.0
                ),
                final_score=float(result.score or 0.0),
            )

            ranked_results.append(
                RankedResult(
                    result=result,
                    features=features,
                    rank=idx + 1,
                )
            )

        return ranked_results

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

        # LGTM Phase 4: Generate unique request ID for tracing
        request_id = str(uuid.uuid4())

        # LGTM Phase 4: Verbose log event 1 - mcp_request_received
        logger.info(
            "mcp_request_received",
            request_id=request_id,
            method="query",
            query=query[:100] if query else None,
            top_k=top_k,
            filters=filters,
            verbosity=verbosity,
            session_id=session_id,
            turn=turn,
            expand_graph=expand_graph,
        )

        try:
            requested_verbosity = (
                verbosity if verbosity is not None else Verbosity.GRAPH.value
            )
            legacy_markdown = False
            if isinstance(requested_verbosity, Verbosity):
                verb_enum = requested_verbosity
            else:
                normalized = (
                    requested_verbosity.strip().lower()
                    if isinstance(requested_verbosity, str)
                    else requested_verbosity
                )
                legacy_markdown = isinstance(normalized, str) and normalized in {
                    "markdown",
                    "md",
                }
                target = "full" if legacy_markdown else normalized
                try:
                    verb_enum = Verbosity(target)
                except Exception as e:
                    allowed = ", ".join(option.value for option in Verbosity)
                    raise ValueError(
                        f"Invalid verbosity '{verbosity}'. Must be one of: {allowed}"
                    ) from e

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
                tracker.ensure_session(session_id)

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

            # Detect a doc_tag like REGPACK-08 in the query and scope retrieval
            if isinstance(query, str):
                m = re.search(r"\b([A-Z]+-\d+)\b", query, flags=re.I)
                if m:
                    filters["doc_tag"] = m.group(1).upper()
                    logger.info(
                        "Detected doc_tag from query",
                        doc_tag=filters["doc_tag"],
                        query=query,
                    )

            embedding_version = getattr(self.embedding_settings, "version", None)
            if embedding_version:
                filters.setdefault("embedding_version", embedding_version)
                logger.debug(
                    "Applied embedding_version filter",
                    embedding_version=embedding_version,
                )
            else:
                filters.pop("embedding_version", None)
                logger.debug("Embedding version not set; skipping filter enforcement")

            use_phase7e = bool(
                getattr(self.config.search.hybrid, "enabled", True)
            ) and bool(
                getattr(
                    getattr(self.config.search.hybrid, "bm25", None), "enabled", False
                )
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

                # Strict post-filter by doc_tag or snapshot_scope (defense-in-depth)
                dt = filters.get("doc_tag")
                if dt:
                    before = len(chunks)
                    chunks = [c for c in chunks if getattr(c, "doc_tag", None) == dt]
                    logger.info(
                        "Applied strict doc_tag=%s: kept %d/%d chunks",
                        dt,
                        len(chunks),
                        before,
                    )
                scope = filters.get("snapshot_scope")
                if scope:
                    before = len(chunks)
                    chunks = [
                        c for c in chunks if getattr(c, "snapshot_scope", None) == scope
                    ]
                    logger.info(
                        "Applied strict snapshot_scope=%s: kept %d/%d chunks",
                        scope,
                        len(chunks),
                        before,
                    )

                assembler = self._get_context_assembler()
                assembled_context = assembler.assemble(chunks, query=query)
                assembled_md = assembler.format_with_citations(assembled_context)

                ranked_results = self._wrap_chunks_as_ranked(chunks)

                embedding_diag = self._embedding_diag()

                retrieval_sections = [
                    {
                        "section_id": chunk.chunk_id,
                        "rank": idx + 1,
                        "score_vec": float(chunk.vector_score or 0.0),
                        "score_text": float(chunk.bm25_score or 0.0),
                        "score_graph": float(chunk.graph_score or 0.0),
                        "score_combined": float(
                            chunk.fused_score
                            or chunk.vector_score
                            or chunk.bm25_score
                            or 0.0
                        ),
                        "graph_distance": int(chunk.graph_distance or 0),
                        "retrieval_method": "hybrid_phase7e",
                        "rerank_score": (
                            float(chunk.rerank_score)
                            if chunk.rerank_score is not None
                            else None
                        ),
                        "rerank_rank": chunk.rerank_rank,
                        "reranker": chunk.reranker,
                        **embedding_diag,
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
                timing.update(embedding_diag)

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
                    f"Search completed: {search_results.total_found} results in {search_time * 1000:.1f}ms"
                )

                embedding_diag = self._embedding_diag()

                retrieval_sections = [
                    {
                        "section_id": result.node_id,
                        "rank": idx + 1,
                        "score_vec": getattr(result, "vector_score", 0.0),
                        "score_text": getattr(result, "text_score", 0.0),
                        "score_graph": getattr(result, "graph_score", 0.0),
                        "graph_distance": getattr(result, "distance", 0),
                        "score_combined": result.score,
                        "retrieval_method": "hybrid",
                        **embedding_diag,
                    }
                    for idx, result in enumerate(search_results.results)
                ]

                # Wrap search results as RankedResult without re-ranking
                # (legacy HybridSearchEngine already orders results)
                rank_start = time.time()
                ranked_results = self._wrap_search_results_as_ranked(
                    search_results.results
                )
                rank_time = time.time() - rank_start

                logger.info(
                    f"Results wrapped: {len(ranked_results)} results in {rank_time * 1000:.1f}ms"
                )

                timing = {
                    "vector_search_ms": search_results.vector_time_ms,
                    "graph_expansion_ms": search_results.graph_time_ms,
                    "ranking_ms": rank_time * 1000,
                    "total_ms": (time.time() - start_time) * 1000,
                }
                timing.update(embedding_diag)
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
            neo4j_driver = None
            if expand_graph or verb_enum == Verbosity.GRAPH:
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
            mcp_search_verbosity_total.labels(verbosity=verb_enum.value).inc()

            # Measure response size
            response_json = json.dumps(response.to_dict())
            response_size = len(response_json.encode("utf-8"))
            mcp_search_response_size_bytes.labels(verbosity=verb_enum.value).observe(
                response_size
            )

            # LGTM Phase 4: Verbose log event 2 - mcp_response_sent
            logger.info(
                "mcp_response_sent",
                request_id=request_id,
                verbosity=verb_enum.value,
                response_size_bytes=response_size,
                response_chunks=len(response.answer_json.evidence),
                confidence=round(response.answer_json.confidence, 2),
                total_time_ms=round(timing.get("total_ms", 0), 2),
                session_id=session_id,
                query_id=query_id if session_id else None,
            )

            if legacy_markdown:
                logger.info(
                    "Using legacy 'markdown' verbosity alias (compat path). "
                    "Consider migrating callers to 'full' or 'graph'."
                )
                markdown_answer = (
                    assembled_md
                    or response.answer_markdown
                    or response.answer_json.answer
                    or ""
                )
                return StructuredResponse(
                    answer=markdown_answer,
                    evidence=response.answer_json.evidence,
                    confidence=response.answer_json.confidence,
                    diagnostics=response.answer_json.diagnostics,
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
            "embedding": self._embedding_diag(),
        }


# Global query service instance (initialized on demand)
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get or create the global query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


def reset_query_service() -> None:
    """Reset the cached QueryService so it will be rebuilt with fresh config/settings."""
    global _query_service
    _query_service = None
