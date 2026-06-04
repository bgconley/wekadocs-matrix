# =============================================================================
# @status: ACTIVE
# @called-by: query_service.py, mcp_app.py
# =============================================================================
"""
Phase 7E-2: Hybrid Retrieval Implementation.

Combines vector search (Qdrant) with BM25/keyword search (Neo4j full-text).
Implements RRF fusion, weighted fusion, bounded adjacency expansion,
and context budget enforcement.

Phase 7E-4: Enhanced with comprehensive metrics collection and SLO monitoring.

Reference: Phase 7E Canonical Spec L1421-1444, L3781-3788
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import Driver
from qdrant_client import QdrantClient

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import get_metrics_aggregator
from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.rerank.base import RerankProvider
from src.providers.settings import EmbeddingSettings
from src.providers.tokenizer_service import TokenizerService
from src.query.entity_extraction import EntityExtractor
from src.query.processing.disambiguation import QueryAnalysis, QueryDisambiguator
from src.query.query_intent import QueryIntent, classify_query_intent
from src.query.signal_pool import SignalPoolResult, build_signal_pool
from src.query.structural_retrieval import StructuralRetrievalConfig as StructuralConfig
from src.query.structural_retrieval import (
    apply_structural_boost as _apply_structural_boost_pure,
)
from src.query.structural_retrieval import (
    get_query_type_rrf_weights,
)
from src.shared.config import (
    get_config,
    get_embedding_plan,
    get_embedding_settings,
    get_settings,
)
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    retrieval_expansion_chunks_added,
    retrieval_expansion_rate_current,
    retrieval_expansion_total,
)
from src.shared.qdrant_schema import validate_qdrant_schema
from src.shared.schema import ensure_schema_version

# LGTM Phase 4: OTEL tracing for retrieval pipeline observability
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore

logger = get_logger(__name__)

# LGTM Phase 4: Tracer for retrieval pipeline spans
_tracer = trace.get_tracer("wekadocs.retrieval") if OTEL_AVAILABLE else None
HYBRID_INIT_LOGGED = False


# --- Observability: extracted to retrieval_observability.py ---
# --- Fusion pipeline: extracted to fusion_pipeline.py, re-exported for compatibility ---
# --- Expansion pipeline: extracted to expansion_pipeline.py, re-exported for compatibility ---
# --- Rerank pipeline: extracted to rerank_pipeline.py, re-exported for compatibility ---
# --- Graph pipeline: extracted to graph_pipeline.py, re-exported for compatibility ---
from src.query import expansion_pipeline as _ep  # noqa: E402
from src.query import fusion_pipeline as _fp  # noqa: E402
from src.query import graph_pipeline as _gp  # noqa: E402
from src.query import rerank_pipeline as _rp  # noqa: E402
from src.query import retrieval_observability as _obs  # noqa: E402

# --- Shared types: extracted to retrieval_types.py, re-exported for compatibility ---
from src.query.retrieval_types import (  # noqa: E402,F401
    ChunkResult,
    ExpandWhen,
    FusionMethod,
    _deduplicate_entity_metadata,
    _snapshot_top,
    dedup_chunk_results,
)

# --- Vector backends: extracted to vector_backends.py, re-exported for compatibility ---
from src.query.vector_backends import (  # noqa: E402,F401
    CITATIONUNIT_BOOST,
    BM25Retriever,
    QdrantMultiVectorRetriever,
    VectorRetriever,
)


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
        embedding_settings: Optional[EmbeddingSettings] = None,
    ):
        """
        Initialize hybrid retriever with all components.

        Args:
            neo4j_driver: Neo4j driver for BM25 and graph operations
            qdrant_client: Qdrant client for vector search
            embedder: Embedding provider for query vectorization
            tokenizer: Tokenizer service for token counting (optional)
        """
        config = get_config()
        settings = get_settings()
        self.config = config
        self.embedding_settings = embedding_settings or get_embedding_settings()
        self.embedding_plan = get_embedding_plan()

        self.expected_schema_version = getattr(config.graph_schema, "version", None)
        if self.expected_schema_version:
            ensure_schema_version(neo4j_driver, self.expected_schema_version)

        if (
            self.embedding_settings
            and hasattr(embedder, "dims")
            and embedder.dims != self.embedding_settings.dims
        ):
            raise ValueError(
                f"HybridRetriever embedder dims "
                f"({getattr(embedder, 'dims', 'unknown')}) do not match "
                f"profile dims ({self.embedding_settings.dims})."
            )

        hybrid_config = getattr(config.search, "hybrid", None)
        qdrant_vector_cfg = getattr(config.search.vector, "qdrant", None)
        self.hybrid_mode = getattr(hybrid_config, "mode", "legacy")
        expansion_timeout_ms = getattr(hybrid_config, "expansion_timeout_ms", None)
        self.expansion_timeout_seconds = (
            float(expansion_timeout_ms) / 1000.0
            if expansion_timeout_ms is not None
            else 2.0
        )
        self.allow_index_migration = (
            os.getenv("HYBRID_ALLOW_INDEX_MIGRATION", "false").lower() == "true"
        )

        self.neo4j_driver = neo4j_driver
        search_config = config.search
        qdrant_collection = getattr(
            getattr(search_config.vector, "qdrant", None), "collection_name", None
        )
        namespace_mode = getattr(settings, "embedding_namespace_mode", "none")
        self.namespace_mode = namespace_mode
        self.qdrant_collection_name = qdrant_collection
        global HYBRID_INIT_LOGGED
        if not HYBRID_INIT_LOGGED:
            logger.info(
                "HybridRetriever initialized",
                extra={
                    "embedding_profile": getattr(
                        self.embedding_settings, "profile", None
                    ),
                    "embedding_provider": getattr(
                        self.embedding_settings, "provider", None
                    ),
                    "embedding_model": getattr(
                        self.embedding_settings, "model_id", None
                    ),
                    "embedding_version": getattr(
                        self.embedding_settings, "version", None
                    ),
                    "embedding_namespace_mode": namespace_mode,
                    "qdrant_collection_name": qdrant_collection,
                },
            )
            HYBRID_INIT_LOGGED = True
        self.filter_allowlist: Dict[str, str] = {
            "doc_tag": "doc_tag",
            "snapshot_scope": "snapshot_scope",
            "document_id": "document_id",
            "embedding_version": "embedding_version",
            "tenant": "tenant",
            "lang": "lang",
        }

        # Validate collection schema against active profile
        effective_settings = self.embedding_settings or get_embedding_settings()
        self.embedding_settings = effective_settings
        qdrant_collection = getattr(
            getattr(search_config.vector, "qdrant", None), "collection_name", None
        )
        include_entity_vector = (
            os.getenv("QDRANT_INCLUDE_ENTITY_VECTOR", "true").lower() == "true"
        )
        strict_validation = bool(
            settings.embedding_strict_mode
            and getattr(settings, "env", "development").lower()
            not in ("development", "dev", "test")
        )
        payload_fields = [
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
            "tenant",
            "document_id",
        ]
        if qdrant_collection and qdrant_client:
            validate_qdrant_schema(
                qdrant_client,
                qdrant_collection,
                self.embedding_settings,
                embedding_plan=self.embedding_plan,
                require_sparse=getattr(qdrant_vector_cfg, "enable_sparse", False),
                require_colbert=getattr(qdrant_vector_cfg, "enable_colbert", False),
                include_entity=include_entity_vector,
                require_doc_title_sparse=getattr(
                    qdrant_vector_cfg, "enable_doc_title_sparse", True
                ),
                require_payload_fields=payload_fields,
                strict=strict_validation,
            )

        expansion_timeout_ms = getattr(hybrid_config, "expansion_timeout_ms", None)
        self.expansion_timeout_seconds = (
            float(expansion_timeout_ms) / 1000.0
            if expansion_timeout_ms is not None
            else 2.0
        )
        self.allow_index_migration = (
            os.getenv("HYBRID_ALLOW_INDEX_MIGRATION", "false").lower() == "true"
        )

        self.vector_field_weights = dict(
            getattr(hybrid_config, "vector_fields", {"content": 1.0})
        )
        self.max_sources_to_expand = getattr(
            getattr(hybrid_config, "expansion", {}), "max_sources", 5
        )

        # ── Resolve retrieval plan EARLY (before retriever construction) ──
        # The plan owns use_weighted_fusion which feeds into the retriever.
        from src.query.retrieval_plan import resolve_retrieval_plan

        _profile_name = getattr(hybrid_config, "profile", None)
        _ff = getattr(config, "feature_flags", None)
        self._plan = resolve_retrieval_plan(_profile_name, hybrid_config, _ff)
        logger.info(
            "retrieval_plan_resolved",
            profile=self._plan.profile.value,
            plan={
                f.name: getattr(self._plan, f.name)
                for f in self._plan.__dataclass_fields__.values()
                if f.name != "profile"
            },
        )

        self.vector_retriever = QdrantMultiVectorRetriever(
            qdrant_client,
            embedder,
            collection_name=qdrant_collection
            or config.search.vector.qdrant.collection_name,
            field_weights=self.vector_field_weights,
            rrf_k=getattr(hybrid_config, "rrf_k", 60),
            embedding_settings=self.embedding_settings,
            embedding_plan=self.embedding_plan,
            use_query_api=getattr(qdrant_vector_cfg, "use_query_api", False),
            query_api_weighted_fusion=self._plan.use_weighted_fusion,
            multi_vector_fusion_method=getattr(
                hybrid_config, "multi_vector_fusion_method", "weighted"
            ),
            query_api_dense_limit=getattr(
                qdrant_vector_cfg, "query_api_dense_limit", 200
            ),
            query_api_sparse_limit=getattr(
                qdrant_vector_cfg, "query_api_sparse_limit", 200
            ),
            query_api_candidate_limit=getattr(
                qdrant_vector_cfg, "query_api_candidate_limit", 200
            ),
            primary_vector_name=getattr(
                qdrant_vector_cfg, "query_vector_name", "content"
            ),
            schema_supports_sparse=getattr(qdrant_vector_cfg, "enable_sparse", False),
            schema_supports_colbert=getattr(qdrant_vector_cfg, "enable_colbert", False),
            # Default True to match ingestion pipeline (atomic.py:1148)
            schema_supports_doc_title_sparse=getattr(
                qdrant_vector_cfg, "enable_doc_title_sparse", True
            ),
            # NEW: Lexical heading matching (sparse vector for section titles)
            schema_supports_title_sparse=getattr(
                qdrant_vector_cfg,
                "enable_title_sparse",
                True,  # Default enabled
            ),
            # NEW: Lexical entity name matching (sparse vector for entity names)
            schema_supports_entity_sparse=getattr(
                qdrant_vector_cfg,
                "enable_entity_sparse",
                True,  # Default enabled
            ),
            # NEW: Per-field RRF contribution logging for debugging
            rrf_debug_logging=getattr(hybrid_config, "rrf_debug_logging", False),
            # NEW: Per-field RRF weights for boosting specific signals
            rrf_field_weights=dict(
                getattr(hybrid_config, "rrf_field_weights", {}) or {}
            ),
        )
        self.colbert_rerank_enabled = getattr(
            hybrid_config, "colbert_rerank_enabled", True
        )
        self.colbert_candidate_limit = getattr(
            hybrid_config, "colbert_candidate_limit", 50
        )
        self.colbert_candidate_multiplier = getattr(
            hybrid_config, "colbert_candidate_multiplier", 3
        )
        self.graph_channel_enabled = getattr(
            hybrid_config, "graph_channel_enabled", False
        )
        self.graph_enrichment_enabled = getattr(
            hybrid_config, "graph_enrichment_enabled", False
        )
        self.graph_adaptive_enabled = getattr(
            hybrid_config, "graph_adaptive_enabled", False
        )
        # Master switch: completely disable ALL Neo4j queries in retrieval path
        self.neo4j_disabled = getattr(hybrid_config, "neo4j_disabled", False)
        dense_active = True
        sparse_active = bool(
            self.vector_retriever.supports_sparse
            and self.vector_retriever.schema_supports_sparse
            and (
                not self.vector_retriever.sparse_field_name
                or self.vector_retriever.field_weights.get(
                    self.vector_retriever.sparse_field_name, 0
                )
                > 0
            )
        )
        colbert_active = bool(
            self.vector_retriever.supports_colbert
            and self.vector_retriever.schema_supports_colbert
            and self.vector_retriever.use_query_api
        )
        logger.info(
            "Retrieval modes resolved",
            extra={
                "dense_active": dense_active,
                "sparse_active": sparse_active,
                "colbert_active": colbert_active,
                "colbert_query_api": self.vector_retriever.use_query_api,
                "has_sparse_field": bool(self.vector_retriever.sparse_field_name),
                "neo4j_disabled": self.neo4j_disabled,  # PHASE 1 VECTOR-ONLY
            },
        )
        self._entity_extractor = None
        # Phase 4: GLiNER query disambiguation for entity-aware retrieval
        self._disambiguator: Optional[QueryDisambiguator] = None
        self._last_query_text: str = ""
        self.tokenizer = tokenizer or TokenizerService()
        self.reranker_config = getattr(hybrid_config, "reranker", None)
        self._reranker_enabled = bool(getattr(self.reranker_config, "enabled", False))
        self.rerank_top_n = int(getattr(self.reranker_config, "top_n", 0) or 0)
        self._reranker: Optional[RerankProvider] = None
        self._reranker_available: bool = True
        if self._reranker_enabled:
            logger.info("HybridRetriever reranker enabled via configuration")

        # Signal-diverse rerank pool configuration
        self._signal_pool_config = getattr(hybrid_config, "signal_pool", None)
        self._signal_pool_enabled = bool(
            getattr(self._signal_pool_config, "enabled", False)
            and getattr(
                getattr(self.config, "feature_flags", None),
                "signal_diverse_rerank_pool",
                False,
            )
        )
        if self._signal_pool_enabled:
            _weighted_fusion_on = bool(
                getattr(
                    getattr(self.config, "feature_flags", None),
                    "query_api_weighted_fusion",
                    False,
                )
            )
            logger.info(
                "Signal-diverse rerank pool enabled",
                extra={
                    "pool_size": getattr(self._signal_pool_config, "pool_size", 200),
                    "weighted_fusion": _weighted_fusion_on,
                },
            )
            if not _weighted_fusion_on:
                logger.warning(
                    "Signal pool enabled without weighted fusion; "
                    "pool will degrade to BM25/vector provenance only. "
                    "Enable feature_flags.query_api_weighted_fusion "
                    "for full per-field signal coverage."
                )
        self.context_group_cap = getattr(
            search_config.response, "max_sections_per_parent", 3
        )

        # Fusion configuration (RRF only; weighted fusion removed)
        configured_method = getattr(hybrid_config, "method", "rrf")
        if configured_method != "rrf":
            logger.warning(
                "Weighted fusion is no longer supported; using RRF",
                configured_method=configured_method,
            )
        self.fusion_method = FusionMethod.RRF
        self.rrf_k = getattr(hybrid_config, "rrf_k", 60)
        self.graph_propagation_decay = getattr(
            hybrid_config, "graph_propagation_decay", 0.85
        )

        # Expansion configuration
        expansion_config = getattr(hybrid_config, "expansion", {})
        if hasattr(expansion_config, "enabled"):
            self.expansion_enabled = expansion_config.enabled
            self.expansion_max_neighbors = getattr(expansion_config, "max_neighbors", 1)
            self.expansion_query_min_tokens = getattr(
                expansion_config, "query_min_tokens", 12
            )
            self.expansion_score_delta_max = getattr(
                expansion_config, "score_delta_max", 0.02
            )
            self.expansion_sparse_threshold = float(
                getattr(expansion_config, "sparse_score_threshold", 0.0) or 0.0
            )
        else:
            # Default expansion settings if not configured
            self.expansion_enabled = True
            self.expansion_max_neighbors = 1
            self.expansion_query_min_tokens = 12
            self.expansion_score_delta_max = 0.02
            self.expansion_sparse_threshold = 0.0

        rescoring_cfg = getattr(expansion_config, "rescoring", None)
        self.expansion_rescoring_enabled = bool(
            getattr(rescoring_cfg, "enabled", False)
        )
        self.expansion_rescoring_mode = getattr(rescoring_cfg, "mode", "threshold_only")
        self.expansion_rescoring_weights = getattr(
            rescoring_cfg,
            "weights",
            {"lexical": 0.4, "structural": 0.5, "proximity": 0.1},
        )
        self.expansion_rescoring_normalize = getattr(
            rescoring_cfg, "normalize_method", "min_max"
        )

        # Graph enrichment configuration (Phase 2.3 parity)
        graph_config = getattr(search_config, "graph", None)
        self.graph_max_depth = (
            getattr(graph_config, "max_depth", 3) if graph_config else 3
        )
        self.graph_max_related = (
            getattr(graph_config, "max_related_per_seed", 20) if graph_config else 20
        )
        self.graph_weight = getattr(hybrid_config, "graph_weight", 0.3)
        rels_env = os.getenv("GRAPH_REL_TYPES")
        if rels_env:
            self.graph_relationships = [
                rel.strip() for rel in rels_env.split(",") if rel.strip()
            ]
        else:
            # Phase 3.5: Single canonical direction per Neo4j best practice
            # Direction-agnostic queries work with any edge direction
            self.graph_relationships = [
                "MENTIONS",  # Chunk->Entity: canonical direction (Phase 3.5)
                "DEFINES",  # Entity->Chunk: definition relationships
                "CONTAINS_STEP",
                "HAS_PARAMETER",
            ]
        # Graph enrichment requires explicit enable flag AND valid depth/related settings
        self.graph_enabled = (
            self.graph_enrichment_enabled
            and self.graph_max_related > 0
            and self.graph_max_depth > 0
        )

        # Context budget
        self.context_max_tokens = getattr(
            search_config.response, "answer_context_max_tokens", 4500
        )

        logger.info(
            f"HybridRetriever initialized: "
            f"rrf_k={self.rrf_k}, "
            f"expansion={'enabled' if self.expansion_enabled else 'disabled'}, "
            f"context_budget={self.context_max_tokens}"
        )
        if self.embedding_settings:
            logger.info(
                "HybridRetriever embedding profile",
                profile=self.embedding_settings.profile,
                provider=self.embedding_settings.provider,
                model=self.embedding_settings.model_id,
                dims=self.embedding_settings.dims,
                tokenizer_backend=self.embedding_settings.tokenizer_backend,
                supports_sparse=getattr(
                    self.embedding_settings.capabilities, "supports_sparse", None
                ),
                supports_colbert=getattr(
                    self.embedding_settings.capabilities, "supports_colbert", None
                ),
            )

    def _normalize_filter_value(self, value: Any) -> Optional[Any]:
        """Normalize filter values to scalar or non-empty list."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            items = [v for v in value if v is not None]
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            return list(items)
        return value

    def _normalize_filters(
        self, raw_filters: Optional[Dict[str, Any]], caller: str = "hybrid"
    ) -> Dict[str, Any]:
        """Apply allowlist, value normalization, and embedding/tenant gating."""
        filters = dict(raw_filters or {})
        normalized: Dict[str, Any] = {}

        # Enforce embedding_version from active settings
        active_version = getattr(self.embedding_settings, "version", None)
        incoming_version = filters.get("embedding_version")
        if active_version:
            if incoming_version and incoming_version != active_version:
                logger.error(
                    "Embedding version mismatch; overriding with active profile",
                    extra={
                        "caller": caller,
                        "incoming": incoming_version,
                        "active": active_version,
                    },
                )
            filters["embedding_version"] = active_version

        for key, value in filters.items():
            if key not in self.filter_allowlist:
                logger.warning(
                    "Ignoring unsupported filter key",
                    extra={"caller": caller, "key": key},
                )
                continue
            normalized_value = self._normalize_filter_value(value)
            if normalized_value is None:
                continue
            normalized[key] = normalized_value

        return normalized

    def _classify_query_type(self, query: str) -> str:
        """Unified query classifier for adaptive retrieval behavior.

        Delegates to query_intent.classify_query_intent() for the actual
        classification logic. Returns just the string type for backward
        compatibility with callers that only need the type string.
        """
        return classify_query_intent(query).query_type

    def _relationships_for_query(self, query: str) -> Tuple[List[str], int]:
        """Select relationship set and neighbor cap based on query type."""
        return _gp.relationships_for_query(self, query)

    def _relationships_for_query_type(self, query_type: str) -> Tuple[List[str], int]:
        """Type-safe variant that accepts a pre-classified query type directly."""
        return _gp.relationships_for_query_type(self, query_type)

    def _get_query_type_weights(self, query_type: str) -> Tuple[float, float]:
        return _gp.get_query_type_weights(self, query_type)

    def _compute_graph_signals(
        self,
        entity_names: List[str],
        candidate_chunk_ids: List[str],
        query_type: str,
        doc_tag: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        return _gp.compute_graph_signals(
            self, entity_names, candidate_chunk_ids, query_type, doc_tag
        )

    def _compute_cross_doc_signals(
        self,
        candidate_chunk_ids: List[str],
        query_type: str,
        doc_tag: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        return _gp.compute_cross_doc_signals(
            self, candidate_chunk_ids, query_type, doc_tag
        )

    def _compute_related_to_doc_signals(
        self,
        seed_doc_ids: List[str],
        doc_tag: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        return _gp.compute_related_to_doc_signals(self, seed_doc_ids, doc_tag)

    def _expand_from_related_docs(
        self,
        fused_results: List[ChunkResult],
        query: str,
        lexical_query: Optional[str],
        filters: Optional[Dict[str, Any]],
        doc_tag: Optional[str],
        metrics: Dict[str, Any],
    ) -> List[ChunkResult]:
        return _gp.expand_from_related_docs(
            self, fused_results, query, lexical_query, filters, doc_tag, metrics
        )

    # ── Standalone RELATED_TO blending (delegates to graph_pipeline) ──

    def _blend_related_to_scores(
        self,
        fused_results: List[ChunkResult],
        query_type: str,
        metrics: Dict[str, Any],
    ) -> None:
        return _gp.blend_related_to_scores(self, fused_results, query_type, metrics)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        expand: bool = True,
        expand_when: str = "auto",
        query_original: Optional[str] = None,
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """
        Perform hybrid retrieval with fusion and optional expansion.

        Args:
            query: Search query text (reformulated for dense/reranker)
            top_k: Number of final results to return
            filters: Optional filters for search
            expand: Whether to perform adjacency expansion
            query_original: Original query keywords for BM25/sparse signals.
                When provided, BM25 and sparse search use this instead of
                ``query``. Dense embedding, ColBERT, and reranker still use
                ``query``. When None, all signals use ``query``.

        Returns:
            Tuple of (results, metrics) with timing and diagnostic info
        """
        start_time = time.time()
        self._last_query_text = query
        # Dual-query: lexical signals use original keywords when available
        lexical_query = query_original or query
        metrics: Dict[str, Any] = {
            "namespace_mode": getattr(self, "namespace_mode", None),
            "bm25_index_name": None,
            "qdrant_collection_name": getattr(
                getattr(self, "vector_retriever", None),
                "collection_name",
                getattr(self, "qdrant_collection_name", None),
            ),
            "signal_pool_configured": bool(
                self._signal_pool_enabled and self._signal_pool_config
            ),
            "signal_pool_enabled": False,
            "signal_pool_used": False,
            "signal_pool_size": 0,
            "signal_pool_slot_fills": {},
            "signal_pool_degraded": False,
        }
        metrics["dual_query_active"] = lexical_query != query
        metrics["retrieval_profile"] = self._plan.profile.value
        if self.embedding_settings:
            metrics["embedding_profile"] = self.embedding_settings.profile
            metrics["embedding_provider"] = self.embedding_settings.provider
            metrics["embedding_model"] = self.embedding_settings.model_id
        normalized_filters = self._normalize_filters(
            filters or {}, caller="hybrid.retrieve"
        )
        raw_doc_tag = normalized_filters.get("doc_tag")
        doc_tag = (
            raw_doc_tag[0]
            if isinstance(raw_doc_tag, list) and raw_doc_tag
            else raw_doc_tag
        )

        # LGTM Phase 4: Extract feature flags for observability
        feature_flags = {}
        ff = getattr(self.config, "feature_flags", None)
        if ff:
            feature_flags = {
                "structure_aware_expansion": getattr(
                    ff, "structure_aware_expansion", False
                ),
                "query_api_weighted_fusion": getattr(
                    ff, "query_api_weighted_fusion", False
                ),
                "graph_as_reranker": getattr(ff, "graph_as_reranker", False),
                "dedup_best_score": getattr(ff, "dedup_best_score", False),
            }

        # Phase 4: GLiNER Query Entity Disambiguation
        # Extract entities from query for post-retrieval boosting
        query_analysis: Optional[QueryAnalysis] = None
        boost_terms: List[str] = []
        entity_boost_enabled = getattr(self.config.ner, "enabled", False)

        if entity_boost_enabled:
            try:
                disambiguator = self._get_disambiguator()
                query_analysis = disambiguator.process(query)
                boost_terms = query_analysis.boost_terms if query_analysis else []
            except Exception as e:
                logger.warning(
                    "query_disambiguation_failed",
                    query=query[:50],
                    error=str(e),
                )
                query_analysis = None
                boost_terms = []

        # LGTM Phase 4: Verbose log event 1 - retrieval_started
        logger.info(
            "retrieval_started",
            query=query[:100],
            top_k=top_k,
            filters=normalized_filters,
            doc_tag=doc_tag,
            feature_flags=feature_flags,
            colbert_enabled=self._plan.use_colbert,
            graph_channel_enabled=self._plan.use_entity_graph_channel,
            graph_enrichment_enabled=self._plan.use_graph_enrichment,
            related_to_expansion=self._plan.use_related_to_expansion,
            related_to_blending=self._plan.use_related_to_blending,
            expansion_enabled=self.expansion_enabled,
            # Phase 4: Entity boosting info
            entity_boost_enabled=entity_boost_enabled,
            query_entities=boost_terms[:5] if boost_terms else [],
        )

        # Step 1: Parallel BM25 and vector search
        # Retrieve more candidates for fusion (3x top_k, or 6x if entity boosting)
        # Over-fetch when entities found to ensure good candidates aren't cut before boosting
        entity_overfetch_multiplier = 2 if boost_terms else 1
        candidate_k = min(top_k * 3 * entity_overfetch_multiplier, 200)

        vec_results: List[ChunkResult] = []
        metrics["bm25_time_ms"] = 0.0
        metrics["bm25_count"] = 0

        # Vector search (always)
        # Apply query-type-specific RRF field weights if adaptive weighting is enabled.
        # The query classifier determines the query type (conceptual, cli, config, etc.)
        # and get_query_type_rrf_weights() returns per-field weights tuned for that type.
        intent = classify_query_intent(query)
        query_type = intent.query_type
        metrics["query_type"] = query_type
        metrics["query_intent_precision_mode"] = intent.precision_mode
        metrics["query_intent_has_cloud_cues"] = intent.has_cloud_cues
        metrics["query_intent_subsystem_terms"] = list(intent.subsystem_terms)
        metrics["query_intent_sizing_terms"] = list(intent.sizing_terms)
        base_rrf_weights = dict(self.vector_retriever.rrf_field_weights)
        adaptive_weights = get_query_type_rrf_weights(
            query_type, base_weights=base_rrf_weights
        )
        # Temporarily override the vector retriever's RRF weights for this search call.
        # try/finally ensures base weights are restored even if search throws.
        self.vector_retriever.rrf_field_weights = adaptive_weights
        vec_start = time.time()
        try:
            vec_results = self.vector_retriever.search(
                query,
                candidate_k,
                normalized_filters,
                lexical_query=lexical_query if lexical_query != query else None,
            )
        finally:
            self.vector_retriever.rrf_field_weights = base_rrf_weights
        vector_stats = getattr(self.vector_retriever, "last_stats", {}) or {}
        metrics["vector_path"] = vector_stats.get("path", "legacy")
        metrics["vec_time_ms"] = vector_stats.get(
            "duration_ms", (time.time() - vec_start) * 1000
        )
        metrics["vec_count"] = len(vec_results)
        # Build accurate list of queried vector fields based on schema_supports_* flags
        queried_vectors = self.vector_retriever.get_queried_vector_fields()
        metrics["vector_fields"] = queried_vectors
        if "prefetch_count" in vector_stats:
            metrics["vector_prefetch_count"] = vector_stats["prefetch_count"]
        if "colbert_used" in vector_stats:
            metrics["vector_colbert_used"] = vector_stats["colbert_used"]
        # Sparse coverage metrics from vector path (Phase B.4)
        if "sparse_scored_ratio" in vector_stats:
            metrics["sparse_scored_ratio"] = vector_stats["sparse_scored_ratio"]
        if "sparse_topk_ratio" in vector_stats:
            metrics["sparse_topk_ratio"] = vector_stats["sparse_topk_ratio"]

        # LGTM Phase 4: Verbose log event 3 - dense_search_complete
        logger.info(
            "dense_search_complete",
            query=query[:50],
            results_count=len(vec_results),
            top_scores=[r.vector_score for r in vec_results[:5] if r.vector_score],
            top_doc_ids=[r.document_id for r in vec_results[:5]],
            search_time_ms=round(metrics["vec_time_ms"], 2),
            vector_path=metrics.get("vector_path", "legacy"),
            colbert_used=metrics.get("vector_colbert_used", False),
        )

        # Step 2: Fuse rankings
        fusion_start = time.time()
        fused_results = self._rrf_fusion([], vec_results)
        metrics["fusion_method"] = "rrf"
        metrics["fusion_time_ms"] = (time.time() - fusion_start) * 1000

        fused_results = [r for r in fused_results if not r.is_microdoc_stub]

        # LGTM Phase 4: Verbose log event 4 - rrf_fusion_complete
        # Note: bm25_count = Neo4j full-text search results (legacy BM25 path)
        # qdrant_sparse_* = Qdrant sparse vector search results (query_api_weighted)
        logger.info(
            "rrf_fusion_complete",
            dense_count=len(vec_results),
            bm25_count=0,
            fused_count=len(fused_results),
            fusion_method=metrics.get("fusion_method", "unknown"),
            rrf_k=self.rrf_k,
            fusion_time_ms=round(metrics.get("fusion_time_ms", 0), 2),
            # Qdrant sparse vector metrics (from query_api_weighted path)
            qdrant_sparse_scored=metrics.get("sparse_scored_ratio", 0),
            qdrant_sparse_topk=metrics.get("sparse_topk_ratio", 0),
            vector_path=metrics.get("vector_path", "legacy"),
            fusion_scores=[
                {
                    "chunk_id": r.chunk_id[:8],
                    "rrf_score": round(r.fused_score or 0, 4),
                    "dense_score": round(r.vector_score or 0, 4),
                    "bm25_score": round(r.bm25_score or 0, 4),
                }
                for r in fused_results[:10]
            ],
        )

        # Step 3: Take fused results as seeds and optionally rerank
        fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
        self._log_stage_snapshot("post-fusion", fused_results)
        metrics["snapshot_post_fusion"] = _snapshot_top(fused_results)

        # Phase 4: Apply GLiNER entity boosting (post-retrieval soft filtering)
        entity_boosted_count = 0
        if boost_terms and fused_results:
            entity_boosted_count = self._apply_entity_boost(fused_results, boost_terms)
            if entity_boosted_count > 0:
                # Re-sort after boosting to reflect new scores
                fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
                self._log_stage_snapshot("post-entity-boost", fused_results)
                metrics["snapshot_post_entity_boost"] = _snapshot_top(fused_results)

            logger.info(
                "entity_boost_complete",
                query=query[:50],
                boost_terms=boost_terms,
                chunks_boosted=entity_boosted_count,
                total_chunks=len(fused_results),
            )

        metrics["entity_boost_enabled"] = entity_boost_enabled
        metrics["entity_boost_terms"] = boost_terms[:5] if boost_terms else []
        metrics["entity_boosted_chunks"] = entity_boosted_count

        # Phase 5: Apply structural boosting based on query type
        # Uses markdown-it-py metadata (has_code, has_table, parent_path_depth)
        structural_boosted_count = 0
        # query_type already computed above via classify_query_intent()
        if fused_results:
            structural_boosted_count = self._apply_structural_boost(
                fused_results, query_type
            )
            if structural_boosted_count > 0:
                # Re-sort after boosting to reflect new scores
                fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
                self._log_stage_snapshot("post-structural-boost", fused_results)
                metrics["snapshot_post_structural_boost"] = _snapshot_top(fused_results)

            logger.info(
                "structural_boost_complete",
                query=query[:50],
                query_type=query_type,
                chunks_boosted=structural_boosted_count,
                total_chunks=len(fused_results),
            )

        metrics["structural_boost_query_type"] = query_type
        metrics["structural_boosted_chunks"] = structural_boosted_count

        # RELATED_TO expansion + standalone blending (plan-gated)
        if not self.neo4j_disabled and fused_results:
            if self._plan.use_related_to_expansion:
                fused_results = self._expand_from_related_docs(
                    fused_results=fused_results,
                    query=query,
                    lexical_query=lexical_query,
                    filters=normalized_filters,
                    doc_tag=doc_tag,
                    metrics=metrics,
                )
            if self._plan.use_related_to_blending:
                self._blend_related_to_scores(fused_results, query_type, metrics)

        # Optional graph retrieval channel (entity-anchored, cross-doc allowed)
        graph_channel_stats: Dict[str, Any] = {}
        graph_candidates: List[ChunkResult] = []
        graph_initial_count = len(fused_results)
        if self._plan.use_entity_graph_channel and not self.neo4j_disabled:
            graph_candidates, graph_channel_stats = self._graph_retrieval_channel(
                query, doc_tag, intent=intent
            )
            if graph_candidates:
                fused_results, merge_stats = self._merge_graph_channel_candidates(
                    fused_results,
                    graph_candidates,
                    query_type=query_type,
                )
                graph_channel_stats.update(merge_stats)

            # LGTM Phase 4: Verbose log event 5 - graph_augmentation_complete
            logger.info(
                "graph_augmentation_complete",
                initial_chunks=graph_initial_count,
                nodes_retrieved=graph_channel_stats.get("graph_channel_candidates", 0),
                raw_rows=graph_channel_stats.get("graph_channel_raw_rows", 0),
                post_support_chunks=graph_channel_stats.get(
                    "graph_channel_post_support_chunks", 0
                ),
                post_sparse_chunks=graph_channel_stats.get(
                    "graph_channel_post_sparse_chunks", 0
                ),
                edges_traversed=graph_channel_stats.get("graph_edges_traversed", 0),
                relationship_types_used=graph_channel_stats.get(
                    "graph_relationship_types", []
                ),
                graph_mode="channel",
                entity_anchors_found=graph_channel_stats.get("entity_anchors_found", 0),
                merged_into_existing=graph_channel_stats.get(
                    "graph_channel_merged_into_existing", 0
                ),
                new_graph_chunks_added=graph_channel_stats.get(
                    "graph_channel_new_chunks_added", 0
                ),
                overlap_blended=graph_channel_stats.get(
                    "graph_channel_overlap_blended", 0
                ),
                overlap_boost=graph_channel_stats.get(
                    "graph_channel_overlap_boost", 0.0
                ),
                graph_channel_score_ceiling=graph_channel_stats.get(
                    "graph_channel_score_ceiling", 0.0
                ),
                graph_anchor_sources=graph_channel_stats.get(
                    "graph_anchor_sources", {}
                ),
            )
        metrics.update(graph_channel_stats)

        metrics["signal_pool_before_colbert"] = True

        # Helper: run ColBERT rerank on a candidate list
        def _run_colbert(
            candidates: List[ChunkResult], limit: int
        ) -> List[ChunkResult]:
            colbert_start = time.time()
            query_bundle = None
            try:
                query_bundle = self.vector_retriever._build_query_bundle(query)
            except Exception as exc:
                logger.warning("ColBERT query embedding failed", error=str(exc))
            if query_bundle and query_bundle.multivector:
                pre_rerank_order = {r.chunk_id: idx for idx, r in enumerate(candidates)}
                hydrated = self._hydrate_colbert_vectors(candidates)
                reranked = self._colbert_rerank(candidates, query_bundle, limit)
                metrics["colbert_hydrated"] = len(hydrated)
                metrics["colbert_rerank_applied"] = True
                metrics["colbert_candidates"] = len(reranked)
                metrics["colbert_rerank_time_ms"] = (time.time() - colbert_start) * 1000
                metrics["colbert_runtime_available"] = True
                metrics["colbert_query_embedding_ok"] = True
                metrics["colbert_rank_delta_top10"] = [
                    pre_rerank_order.get(r.chunk_id, idx) - idx
                    for idx, r in enumerate(reranked[:10])
                ]
                logger.info(
                    "colbert_rerank_complete",
                    input_count=limit,
                    output_count=len(reranked),
                    hydrated_count=len(hydrated),
                    rerank_time_ms=round(metrics["colbert_rerank_time_ms"], 2),
                    rerank_details=[
                        {
                            "chunk_id": r.chunk_id[:8],
                            "original_score": round(r.fused_score or 0, 4),
                            "colbert_score": round(r.rerank_score or 0, 4),
                            "rank_change": pre_rerank_order.get(r.chunk_id, 0) - idx,
                        }
                        for idx, r in enumerate(reranked[:10])
                    ],
                )
                metrics["snapshot_post_colbert"] = _snapshot_top(reranked)
                return reranked
            else:
                metrics["colbert_rerank_applied"] = False
                metrics["colbert_runtime_available"] = (
                    self.vector_retriever.supports_colbert
                )
                metrics["colbert_query_embedding_ok"] = (
                    query_bundle is not None and query_bundle.multivector is not None
                )
                return candidates

        colbert_available = (
            self._plan.use_colbert
            and self.vector_retriever.supports_colbert
            and bool(fused_results)
        )

        # ── Signal pool → ColBERT path ──────────────────────────────
        if self._plan.use_signal_pool and self._signal_pool_config:
            metrics["colbert_input_from_pool"] = True

            pre_rerank_structural: List[ChunkResult] = []
            if not intent.precision_mode:
                try:
                    pre_rerank_structural = self._expand_with_structure(
                        query, fused_results[:10], doc_tag, force=True
                    )
                except Exception as e:
                    logger.warning(
                        "pre_rerank_structural_expansion_failed",
                        extra={"error": str(e)},
                    )
            else:
                logger.info(
                    "pre_rerank_structural_expansion_skipped",
                    query_type=intent.query_type,
                    reason="precision_mode",
                )

            pool_result: SignalPoolResult = build_signal_pool(
                fused_results, pre_rerank_structural, self._signal_pool_config
            )

            metrics["signal_pool_enabled"] = True
            metrics["signal_pool_used"] = True
            metrics["signal_pool_size"] = len(pool_result.pool)
            metrics["signal_pool_slot_fills"] = pool_result.slot_fills
            metrics["signal_pool_degraded"] = pool_result.degraded

            if colbert_available:
                rerank_candidates = _run_colbert(
                    pool_result.pool, len(pool_result.pool)
                )
            else:
                rerank_candidates = pool_result.pool
        elif self._reranker_enabled:
            pool_cap = self.rerank_top_n or top_k
            rerank_pool_size = min(pool_cap, len(fused_results))
            rerank_candidates = fused_results[:rerank_pool_size]
            metrics["signal_pool_enabled"] = False
            metrics["signal_pool_used"] = False
            metrics["colbert_input_from_pool"] = False
        else:
            rerank_candidates = fused_results
            metrics["signal_pool_enabled"] = False
            metrics["signal_pool_used"] = False
            metrics["colbert_input_from_pool"] = False

        # ── Cross-encoder reranker ─────────────────────────────────
        reranker_active = False
        seeds: List[ChunkResult]

        # Best-available ordering: if signal pool + ColBERT produced
        # rerank_candidates, those are strictly better than raw fused_results.
        # Use them as the fallback source instead of discarding the work.
        best_available = rerank_candidates if rerank_candidates else fused_results

        if self._reranker_enabled and rerank_candidates:
            # Hydrate parent_path_norm for reranker context enrichment
            self._hydrate_parent_paths(rerank_candidates)

            # Track pre-rerank order for rank change calculation
            pre_bge_order = {r.chunk_id: idx for idx, r in enumerate(rerank_candidates)}
            ordered_candidates = self._apply_reranker(
                query,
                rerank_candidates,
                metrics,
                query_type=query_type,
                intent=intent,
            )
            reranker_active = bool(metrics.get("reranker_applied"))
            if reranker_active:
                ordered_candidates.sort(
                    key=lambda chunk: (
                        chunk.rerank_score or float("-inf"),
                        chunk.fused_score or 0.0,
                    ),
                    reverse=True,
                )

                # Phase B1: post-rerank specificity adjustment
                specificity_flag = self._plan.use_specificity_adjustment
                specificity_applied = False
                specificity_adjustments = 0
                if (
                    specificity_flag
                    and intent is not None
                    and intent.precision_mode
                    and not intent.has_cloud_cues
                    and intent.primary_anchors
                ):
                    specificity_applied, specificity_adjustments = (
                        self._apply_specificity_adjustment(
                            ordered_candidates, intent, metrics
                        )
                    )
                    if specificity_applied:
                        ordered_candidates.sort(
                            key=lambda chunk: (
                                chunk.rerank_score or float("-inf"),
                                chunk.fused_score or 0.0,
                            ),
                            reverse=True,
                        )
                metrics["precision_specificity_applied"] = specificity_applied
                metrics["precision_specificity_adjustments"] = specificity_adjustments

                seeds = ordered_candidates[:top_k]

                # LGTM Phase 4: Verbose log event 7 - rerank_complete
                logger.info(
                    "rerank_complete",
                    input_count=len(rerank_candidates),
                    output_count=len(seeds),
                    reranker_model=metrics.get("reranker_model", "unknown"),
                    rerank_time_ms=round(metrics.get("reranker_time_ms", 0), 2),
                    specificity_applied=specificity_applied,
                    specificity_adjustments=specificity_adjustments,
                    rerank_details=[
                        {
                            "chunk_id": r.chunk_id[:8],
                            "bge_score": round(r.rerank_score or 0, 4),
                            "original_rank": pre_bge_order.get(r.chunk_id, 0),
                            "final_rank": idx,
                        }
                        for idx, r in enumerate(seeds[:10])
                    ],
                )
                metrics["snapshot_post_reranker"] = _snapshot_top(seeds)
            else:
                seeds = best_available[:top_k]
        else:
            seeds = best_available[:top_k]

        metrics["pre_rerank_structural_expansion_applied"] = (
            len(pre_rerank_structural) > 0
        )
        metrics["final_reranker_applied"] = reranker_active

        # Hydrate vector-only winners with citation labels
        self._hydrate_missing_citations(seeds)

        # Prefer chunks that carry richer citation labels only when reranker is inactive
        if not reranker_active:
            seeds.sort(
                key=lambda chunk: (
                    len(chunk.citation_labels or []),
                    chunk.fused_score or 0.0,
                ),
                reverse=True,
            )

        if doc_tag:
            before = len(seeds)
            seeds = [c for c in seeds if getattr(c, "doc_tag", None) == doc_tag]
            logger.info(
                "Filtered seeds by doc_tag=%s: kept %d/%d", doc_tag, len(seeds), before
            )

        metrics["seed_count"] = len(seeds)
        seed_ids: Optional[Set[str]] = None
        if reranker_active:
            seed_ids = {chunk.chunk_id for chunk in seeds}

        metrics["microdoc_extras"] = 0
        metrics["microdoc_tokens"] = 0

        # Step 5: Gating decision for expansion
        when = ExpandWhen(expand_when)
        triggered, reason, score_delta, query_tokens = self._should_expand(
            query, seeds, when
        )
        metrics["expansion_triggered"] = bool(expand and triggered)
        metrics["expansion_reason"] = reason
        metrics["scores_close_delta"] = score_delta
        metrics["query_tokens"] = query_tokens

        # Step 6: Optional bounded adjacency expansion
        # Note: Expansion ADDS neighbors without re-limiting to top_k
        all_results = list(seeds)
        if (
            expand
            and triggered
            and self.expansion_enabled
            and not intent.precision_mode
        ):
            expansion_start = time.time()
            expanded_results = self._bounded_expansion(query, seeds)

            # Enforce doc_tag and same-document continuity for neighbors
            if doc_tag:
                expanded_results = [
                    n
                    for n in expanded_results
                    if getattr(n, "doc_tag", None) == doc_tag
                ]
            seed_docs = {c.chunk_id: c.document_id for c in seeds}
            expanded_results = [
                n
                for n in expanded_results
                if seed_docs.get(getattr(n, "expansion_source", None)) == n.document_id
            ]

            all_results.extend(expanded_results)
            metrics["expansion_time_ms"] = (time.time() - expansion_start) * 1000
            metrics["expansion_count"] = len(expanded_results)
            metrics["expanded_source_count"] = min(
                len(seeds), getattr(self, "max_sources_to_expand", 5)
            )
            metrics["expansion_cap_hit"] = int(
                len(seeds) > getattr(self, "max_sources_to_expand", 5)
            )
        else:
            metrics["expansion_count"] = 0
            metrics["expanded_source_count"] = 0
            metrics["expansion_cap_hit"] = 0

        # Step 6b: Structure-aware expansion (C.4)
        # Adds sibling, parent section, and shared-entity chunks
        structure_start = time.time()
        structure_expanded = (
            self._expand_with_structure(query, seeds, doc_tag)
            if not intent.precision_mode
            else []
        )
        if structure_expanded:
            all_results.extend(structure_expanded)
            metrics["structure_expansion_time_ms"] = (
                time.time() - structure_start
            ) * 1000
            metrics["structure_expansion_count"] = len(structure_expanded)
            # Count by context_source type
            metrics["structure_sibling_count"] = sum(
                1 for r in structure_expanded if r.context_source == "sibling"
            )
            metrics["structure_parent_count"] = sum(
                1 for r in structure_expanded if r.context_source == "parent_section"
            )
            metrics["structure_entity_count"] = sum(
                1 for r in structure_expanded if r.context_source == "shared_entities"
            )
        else:
            metrics["structure_expansion_time_ms"] = (
                time.time() - structure_start
            ) * 1000
            metrics["structure_expansion_count"] = 0

        # Step 6: Dedup, hydrate citations, and maintain deterministic ordering
        all_results = self._dedup_results(all_results)
        all_results = [
            c for c in all_results if not getattr(c, "is_microdoc_stub", False)
        ]
        self._hydrate_missing_citations(all_results)

        graph_chunks, graph_stats = self._apply_graph_enrichment(
            seeds, all_results, doc_tag
        )
        if graph_chunks:
            all_results.extend(graph_chunks)
            all_results = self._dedup_results(all_results)
            all_results = [
                c for c in all_results if not getattr(c, "is_microdoc_stub", False)
            ]
        metrics.update(graph_stats)

        self._annotate_coverage(all_results)

        if doc_tag:
            all_results = [
                c for c in all_results if getattr(c, "doc_tag", None) == doc_tag
            ]

        primaries = [c for c in all_results if not c.is_microdoc_extra]
        extras = [c for c in all_results if c.is_microdoc_extra]

        def _primary_score_key(chunk: ChunkResult) -> Tuple[float, float, float, float]:
            # Sort by rerank_score descending; seeds get priority via element 0
            rerank_val = (
                float(chunk.rerank_score)
                if chunk.rerank_score is not None
                else float(chunk.fused_score or 0.0)
            )
            citation_weight = float(len(chunk.citation_labels or []))
            if reranker_active and seed_ids and chunk.chunk_id in seed_ids:
                # Seeds first (1.0), then sorted by rerank_score (not rank_idx!)
                return (1.0, rerank_val, citation_weight, 0.0)
            return (0.0, rerank_val, citation_weight, float(chunk.fused_score or 0.0))

        primaries.sort(key=_primary_score_key, reverse=True)
        if reranker_active and seed_ids:
            extras.sort(
                key=lambda chunk: (
                    0.0,
                    (
                        float(chunk.rerank_score)
                        if chunk.rerank_score is not None
                        else float(chunk.fused_score or 0.0)
                    ),
                ),
                reverse=True,
            )
        else:
            extras.sort(key=lambda chunk: float(chunk.fused_score or 0.0), reverse=True)

        primaries = self._apply_doc_continuity_boost(primaries, alpha=0.12)
        self._log_stage_snapshot("post-continuity", primaries)

        # Step 6: Context assembly with budget enforcement
        context_start = time.time()
        primary_budget, primary_tokens = self._enforce_context_budget(primaries)
        extra_budget, total_tokens = self._enforce_context_budget(
            extras, starting_tokens=primary_tokens
        )
        final_results = primary_budget + extra_budget
        self._log_stage_snapshot("post-budget", final_results)

        metrics["context_assembly_ms"] = (time.time() - context_start) * 1000
        metrics["primary_count"] = len(primary_budget)
        metrics["microdoc_used"] = len(extra_budget)
        metrics["microdoc_present"] = int(bool(extras))
        metrics["final_count"] = len(final_results)
        metrics["total_tokens"] = total_tokens

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
                sparse_scored_ratio=metrics.get("sparse_scored_ratio"),
                sparse_topk_ratio=metrics.get("sparse_topk_ratio"),
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

        # LGTM Phase 4: Verbose log event 8 - retrieval_complete
        logger.info(
            "retrieval_complete",
            query=query[:100],
            total_chunks_retrieved=len(final_results),
            unique_documents=len(set(r.document_id for r in final_results)),
            total_tokens=metrics.get("total_tokens", 0),
            microdoc_extras=metrics.get("microdoc_extras", 0),
            total_time_ms=round(metrics.get("total_time_ms", 0), 2),
            reranker_applied=metrics.get("reranker_applied", False),
            colbert_applied=metrics.get("colbert_rerank_applied", False),
            expansion_triggered=metrics.get("expansion_triggered", False),
            # Sample retrieved content per canonical plan
            retrieved_chunks=[
                {
                    "rank": i + 1,
                    "chunk_id": r.chunk_id[:8],
                    "document_id": r.document_id[:8] if r.document_id else None,
                    "document_title": (
                        getattr(r, "document_title", "")[:50]
                        if getattr(r, "document_title", None)
                        else None
                    ),
                    "section_title": r.heading[:50] if r.heading else None,
                    "final_score": round(r.fused_score or r.rerank_score or 0, 4),
                    "content_preview": r.text[:150] if r.text else None,
                }
                for i, r in enumerate(final_results[:5])
            ],
        )

        return final_results, metrics

    def _rrf_fusion(
        self, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
    ) -> List[ChunkResult]:
        return _fp.rrf_fusion(self, bm25_results, vec_results)

    def _apply_doc_continuity_boost(
        self, chunks: List[ChunkResult], alpha: float = 0.12
    ) -> List[ChunkResult]:
        return _fp.apply_doc_continuity_boost(chunks, alpha)

    def _hydrate_parent_paths(self, chunks: List[ChunkResult]) -> None:
        """
        Batch-fetch parent_path_norm from Neo4j for rerank candidates.

        Adds full heading hierarchy paths (e.g., "Config > S3 > Buckets")
        to ChunkResult objects. Used by _apply_reranker to prepend structural
        context to reranker input text.

        Skips if Neo4j is disabled. Failures are non-fatal.
        """
        if self.neo4j_disabled or not self.neo4j_driver:
            return

        chunk_ids = [c.chunk_id for c in chunks if c.parent_path_norm is None]
        if not chunk_ids:
            return

        query = """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid})
        WHERE c.parent_path_norm IS NOT NULL
        RETURN c.id AS chunk_id, c.parent_path_norm AS parent_path_norm
        """
        try:
            with self.neo4j_driver.session() as session:
                rows = session.run(query, ids=chunk_ids).data()

            lookup = {row["chunk_id"]: row["parent_path_norm"] for row in rows}
            for chunk in chunks:
                if chunk.parent_path_norm is None:
                    chunk.parent_path_norm = lookup.get(chunk.chunk_id)
        except Exception as e:
            logger.warning(
                "parent_path_norm hydration failed",
                extra={"error": str(e), "chunk_count": len(chunk_ids)},
            )

    def _hydrate_missing_citations(self, chunks: List[ChunkResult]) -> None:
        """
        Ensure chunks from vectors emit citation labels.

        Fetches CitationUnit headings for chunks missing citation data.
        """
        # PHASE 1 VECTOR-ONLY: Skip Neo4j queries when disabled
        if self.neo4j_disabled:
            return

        query_ids = list({chunk.chunk_id for chunk in chunks})

        lookup: Dict[str, List[Tuple[int, str, int]]] = {}
        if query_ids:
            query = """
            UNWIND $ids AS cid
            MATCH (c:Chunk {id: cid})
            OPTIONAL MATCH (u:CitationUnit)-[:IN_CHUNK]->(c)
            WITH c, u ORDER BY u.order ASC
            RETURN c.id AS chunk_id, collect([u.order, u.heading, u.level]) AS labels
            """

            with self.neo4j_driver.session() as session:
                rows = session.run(query, ids=query_ids).data()

            lookup = {
                row["chunk_id"]: [
                    (int(order or 0), heading, int(level or 0))
                    for order, heading, level in (row.get("labels") or [])
                    if heading
                ]
                for row in rows
            }

        pending_sections: List[str] = []
        chunk_by_id: Dict[str, ChunkResult] = {c.chunk_id: c for c in chunks}

        for chunk in chunks:
            labels: List[Tuple[int, str, int]] = []
            if lookup:
                labels.extend(lookup.get(chunk.chunk_id, []) or [])

            existing = getattr(chunk, "citation_labels", None) or []
            if existing:
                for item in existing:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        order_val = int(item[0] or 0)
                        title = (item[1] or "").strip()
                        level_val = (
                            int(item[2]) if len(item) > 2 and item[2] is not None else 0
                        )
                        if title:
                            labels.append((order_val, title, level_val))

            try:
                boundaries = json.loads(chunk.boundaries_json or "{}")
                items = (
                    boundaries
                    if isinstance(boundaries, list)
                    else boundaries.get("sections", [])
                )
                parsed: List[Tuple[int, str, int]] = []
                for section in items or []:
                    heading_val = (
                        section.get("heading") or section.get("title") or ""
                    ).strip()
                    if not heading_val:
                        continue
                    order_val = section.get("order") or 0
                    try:
                        order_int = int(order_val)
                    except (TypeError, ValueError):
                        order_int = 0
                    level_val = int(section.get("level", 0))
                    parsed.append((order_int, heading_val, level_val))
                if parsed:
                    labels.extend(parsed)
            except Exception:
                pass

            if not labels and chunk.original_section_ids:
                pending_sections.append(chunk.chunk_id)
                continue

            if labels:
                self._assign_normalized_citations(chunk, labels)

        if pending_sections:
            query = """
            UNWIND $ids AS cid
            MATCH (c:Chunk {id: cid})
            UNWIND coalesce(c.original_section_ids, []) AS sid
            MATCH (s:Chunk {id: sid})
            RETURN c.id AS chunk_id, collect([s.order, s.heading, s.level]) AS labels
            """
            with self.neo4j_driver.session() as session:
                rows = session.run(query, ids=pending_sections).data()

            fallback_lookup = {
                row["chunk_id"]: [
                    (int(order or 0), heading, int(level or 0))
                    for order, heading, level in (row.get("labels") or [])
                    if heading
                ]
                for row in rows
            }

            for chunk_id, labels in fallback_lookup.items():
                chunk = chunk_by_id.get(chunk_id)
                if chunk and not getattr(chunk, "citation_labels", None):
                    if labels:
                        self._assign_normalized_citations(chunk, labels)

    def _assign_normalized_citations(
        self, chunk: ChunkResult, labels: List[Tuple[int, str, int]]
    ) -> None:
        """Normalize citation labels while preserving document order and hierarchy."""
        entries: List[Tuple[int, str, int]] = []
        seen: Set[Tuple[int, str]] = set()

        for order_val, title, level in labels:
            clean_title = (title or "").strip()
            if not clean_title:
                continue
            order_int = int(order_val or 0)
            level_int = int(level or 0)
            key = (order_int, clean_title.lower())
            if key in seen:
                continue
            seen.add(key)
            entries.append((order_int, clean_title, level_int))

        if not entries:
            return

        entries.sort(key=lambda x: (x[0], x[2], x[1].lower()))

        deep_levels = [level for order, _, level in entries if order > 0 and level >= 3]
        if deep_levels:
            min_level = min(deep_levels)
            entries = [
                (order, title, level)
                for order, title, level in entries
                if order == 0 or level >= min_level
            ]

        heading_lower = (chunk.heading or "").strip().lower()
        final_labels: List[Tuple[int, str, int]] = []
        for order_int, title, level_int in entries:
            if (
                heading_lower
                and order_int == 0
                and title.lower() == heading_lower
                and len(entries) > 1
            ):
                continue
            final_labels.append((order_int, title, level_int))

        if not final_labels and entries:
            final_labels = [entries[0]]

        chunk.citation_labels = final_labels

    def _apply_reranker(
        self,
        query: str,
        seeds: List[ChunkResult],
        metrics: Dict[str, Any],
        *,
        query_type: Optional[str] = None,
        intent: Optional["QueryIntent"] = None,
    ) -> List[ChunkResult]:
        return _rp.apply_reranker(
            self, query, seeds, metrics, query_type=query_type, intent=intent
        )

    # ── Phase B1: Post-rerank specificity adjustment ──────────────

    def _apply_specificity_adjustment(
        self,
        candidates: List[ChunkResult],
        intent: "QueryIntent",
        metrics: Dict[str, Any],
        *,
        anchor_bonus: float = 0.15,
        deploy_penalty: float = 0.10,
    ) -> Tuple[bool, int]:
        return _rp.apply_specificity_adjustment(
            self,
            candidates,
            intent,
            metrics,
            anchor_bonus=anchor_bonus,
            deploy_penalty=deploy_penalty,
        )

    def _get_reranker(self) -> Optional[RerankProvider]:
        return _rp.get_reranker(self)

    def _should_expand(
        self, query: str, seeds: List[ChunkResult], when: ExpandWhen
    ) -> Tuple[bool, str, float, int]:
        return _ep.should_expand(self, query, seeds, when)

    def _result_id(self, r: ChunkResult) -> Tuple:
        return _ep.result_id(r)

    def _dedup_results(self, results: List[ChunkResult]) -> List[ChunkResult]:
        return _ep.dedup_results(self, results)

    def _bounded_expansion(
        self, query: str, seeds: List[ChunkResult]
    ) -> List[ChunkResult]:
        return _ep.bounded_expansion(self, query, seeds)

    def _expand_with_structure(
        self,
        query: str,
        seeds: List[ChunkResult],
        doc_tag: Optional[str] = None,
        *,
        force: bool = False,
    ) -> List[ChunkResult]:
        return _ep.expand_with_structure(self, query, seeds, doc_tag, force=force)

    def _build_expanded_chunk(
        self,
        record: dict,
        source_chunk_id: str,
        context_source: str,
        source_score: float,
    ) -> ChunkResult:
        return _ep.build_expanded_chunk(
            self, record, source_chunk_id, context_source, source_score
        )

    def _gate_expansion_with_sparse(
        self,
        query: str,
        neighbors: List[ChunkResult],
        threshold: float,
    ) -> List[ChunkResult]:
        return _ep.gate_expansion_with_sparse(self, query, neighbors, threshold)

    def _normalize_sparse_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        return _ep.normalize_sparse_scores(self, scores)

    def _fuse_expansion_scores(
        self, neighbors: List[ChunkResult], score_map: Dict[str, float]
    ) -> List[ChunkResult]:
        return _ep.fuse_expansion_scores(self, neighbors, score_map)

    def _rescore_expansion_with_sparse(
        self,
        indices: List[int],
        values: List[float],
        neighbors: List[ChunkResult],
        threshold: float,
    ) -> Optional[Dict[str, float]]:
        return _ep.rescore_expansion_with_sparse(
            self, indices, values, neighbors, threshold
        )

    def _hydrate_colbert_vectors(
        self, candidates: List[ChunkResult]
    ) -> Dict[str, List[List[float]]]:
        return _rp.hydrate_colbert_vectors(self, candidates)

    def _colbert_rerank(
        self,
        candidates: List[ChunkResult],
        query_bundle: QueryEmbeddingBundle,
        limit: int,
    ) -> List[ChunkResult]:
        return _rp.colbert_rerank(self, candidates, query_bundle, limit)

    def _apply_graph_enrichment(
        self,
        seeds: List[ChunkResult],
        current_results: List[ChunkResult],
        doc_tag: Optional[str],
    ) -> Tuple[List[ChunkResult], Dict[str, int]]:
        return _gp.apply_graph_enrichment(self, seeds, current_results, doc_tag)

    def _get_entity_extractor(self) -> EntityExtractor:
        return _gp.get_entity_extractor(self)

    def _get_disambiguator(self) -> QueryDisambiguator:
        """Lazily initialize the GLiNER-based query disambiguator (Phase 4)."""
        if self._disambiguator is None:
            self._disambiguator = QueryDisambiguator()
        return self._disambiguator

    def _apply_entity_boost(
        self,
        results: List[ChunkResult],
        boost_terms: List[str],
        max_boost: float = 0.5,
        per_entity_boost: float = 0.1,
    ) -> int:
        """
        Apply post-retrieval entity boosting to fused results.

        This implements "soft filtering" - entities in the query boost matching
        chunks rather than filtering them out. This is more robust than hard
        filtering since it doesn't exclude potentially relevant results.

        Args:
            results: List of ChunkResult to boost (modified in-place)
            boost_terms: Normalized entity terms from query disambiguation
            max_boost: Maximum total boost factor (default 50%)
            per_entity_boost: Boost per matching entity (default 10%)

        Returns:
            Number of chunks that received a boost
        """
        if not boost_terms:
            return 0

        boost_terms_set = set(boost_terms)
        boosted_count = 0

        for res in results:
            # Get entity values from chunk's entity_metadata
            entity_metadata = res.entity_metadata or {}
            doc_entities = entity_metadata.get("entity_values_normalized", [])

            if not doc_entities:
                continue

            # Count matching entities
            matches = sum(1 for term in boost_terms_set if term in doc_entities)

            if matches > 0:
                # Calculate boost factor (capped at max_boost)
                boost_factor = 1.0 + min(max_boost, matches * per_entity_boost)

                # Apply boost to fused score
                if res.fused_score is not None:
                    res.fused_score *= boost_factor
                    res.entity_boost_applied = True
                    boosted_count += 1

        return boosted_count

    def _apply_structural_boost(
        self,
        results: List[ChunkResult],
        query_type: str,
    ) -> int:
        """
        Apply Phase 5 structural boosting based on query type.

        Uses markdown-it-py enhanced metadata (has_code, has_table, parent_path_depth)
        to adjust scores based on query type. For example:
        - CLI queries get a boost for code-containing chunks
        - Reference queries get a boost for table-containing chunks
        - Conceptual queries get a penalty for deeply nested content

        Args:
            results: List of ChunkResult to boost (modified in-place)
            query_type: Query type from _classify_query_type()

        Returns:
            Number of chunks that received a boost/penalty
        """
        # Get structural config from hybrid search config
        hybrid_config = getattr(self.config.search, "hybrid", None)
        structural_config = (
            getattr(hybrid_config, "structural", None) if hybrid_config else None
        )

        # Convert to StructuralConfig for the pure function
        if structural_config:
            config = StructuralConfig(
                enabled=getattr(structural_config, "enabled", True),
                filter_by_block_type=getattr(
                    structural_config, "filter_by_block_type", False
                ),
                boost_by_structure=getattr(
                    structural_config, "boost_by_structure", True
                ),
            )
        else:
            config = StructuralConfig()

        if not config.enabled or not config.boost_by_structure:
            return 0

        boosted_count = 0

        # Convert ChunkResult to dict format for pure function
        for res in results:
            if not res.structural_metadata:
                continue

            # Build dict in format expected by pure function
            result_dict = {
                "score": res.fused_score or 0.0,
                "payload": res.structural_metadata,
            }

            # Apply boost via pure function (single-item list)
            boosted = _apply_structural_boost_pure([result_dict], query_type, config)

            # Update score if changed
            new_score = boosted[0]["score"]
            if new_score != (res.fused_score or 0.0):
                res.fused_score = new_score
                res.structural_boost_applied = True
                boosted_count += 1

        return boosted_count

    def _graph_retrieval_channel(
        self,
        query: str,
        doc_tag: Optional[str],
        *,
        intent: Optional["QueryIntent"] = None,
    ) -> Tuple[List[ChunkResult], Dict[str, int]]:
        return _gp.graph_retrieval_channel(self, query, doc_tag, intent=intent)

    def _merge_graph_channel_candidates(
        self,
        fused_results: List[ChunkResult],
        graph_candidates: List[ChunkResult],
        *,
        query_type: str,
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """Merge graph channel output with bounded influence.

        Existing vector-ranked chunks get a small positive boost when graph
        evidence overlaps by chunk identity. Novel graph-only chunks are kept
        as recall candidates with a capped fused score so they can be reranked
        without displacing strong vector results prematurely.
        """
        if not graph_candidates:
            return fused_results, {
                "graph_channel_merged_into_existing": 0,
                "graph_channel_new_chunks_added": 0,
                "graph_channel_overlap_blended": 0,
                "graph_channel_score_ceiling": 0.0,
                "graph_channel_overlap_boost": 0.0,
            }

        existing_by_id = {self._result_id(r): r for r in fused_results}
        merged_count = 0
        added_count = 0
        overlap_blended = 0

        score_ceiling = 1e-3
        if fused_results:
            fused_scores = sorted(
                [float(r.fused_score or 0.0) for r in fused_results],
                reverse=True,
            )
            if fused_scores:
                ceiling_idx = min(len(fused_scores) - 1, 60)
                score_ceiling = max(float(fused_scores[ceiling_idx]), 1e-3)

        _, graph_weight = _gp.get_query_type_weights(self, query_type)
        overlap_boost = max(0.05, min(0.15, float(graph_weight) * 0.4))

        for candidate in graph_candidates:
            rid = self._result_id(candidate)
            current = existing_by_id.get(rid)
            if current is not None:
                current.graph_score = max(
                    float(current.graph_score or 0.0),
                    float(candidate.graph_score or 0.0),
                )
                if (
                    current.vector_score_kind in (None, "", "dense")
                    and candidate.vector_score_kind
                ):
                    current.vector_score_kind = candidate.vector_score_kind

                base_fused = (
                    current.fused_score
                    if current.fused_score is not None
                    else (current.vector_score or current.bm25_score or 0.0)
                )
                if base_fused > 0 and current.graph_score > 0:
                    current.fused_score = float(base_fused) * (
                        1.0 + (overlap_boost * float(current.graph_score))
                    )
                    overlap_blended += 1
                merged_count += 1
                continue

            candidate.fused_score = min(
                float(candidate.fused_score or 0.0), score_ceiling
            )
            fused_results.append(candidate)
            existing_by_id[rid] = candidate
            added_count += 1

        return fused_results, {
            "graph_channel_merged_into_existing": merged_count,
            "graph_channel_new_chunks_added": added_count,
            "graph_channel_overlap_blended": overlap_blended,
            "graph_channel_score_ceiling": round(score_ceiling, 6),
            "graph_channel_overlap_boost": round(overlap_boost, 4),
        }

    def _fetch_graph_neighbors(
        self, seeds: List[ChunkResult], doc_tag: Optional[str]
    ) -> List[ChunkResult]:
        return _gp.fetch_graph_neighbors(self, seeds, doc_tag)

    def _annotate_coverage(self, chunks: List[ChunkResult]) -> None:
        return _gp.annotate_coverage(self, chunks)

    def _neighbor_score(self, source_score: float) -> float:
        return _ep.neighbor_score(source_score)

    def _truncate_text(self, text: str, token_budget: int) -> Tuple[str, int]:
        return _ep.truncate_text(self, text, token_budget)

    def _chunk_from_props(self, props: Dict[str, Any]) -> ChunkResult:
        return _gp.chunk_from_props(props)

    def _enforce_context_budget(
        self, results: List[ChunkResult], starting_tokens: int = 0
    ) -> Tuple[List[ChunkResult], int]:
        return _ep.enforce_context_budget(self, results, starting_tokens)

    def _log_context_budget(self, tokens: int, count: int) -> None:
        return _ep.log_context_budget(self, tokens, count)

    def _log_stage_snapshot(
        self, stage: str, chunks: List[ChunkResult], limit: int = 5
    ) -> None:
        return _obs.log_stage_snapshot(stage, chunks, limit)

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
