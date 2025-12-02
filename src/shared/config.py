# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# Configuration loader with environment variable support
# Enhanced for Pre-Phase 7: Added validation for embedding configuration

import logging
import os
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic_settings import BaseSettings

from src.providers.settings import (
    EmbeddingCapabilities as ProviderEmbeddingCapabilities,
)
from src.providers.settings import EmbeddingSettings as ProviderEmbeddingSettings

from .models import WekaBaseModel

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_PROFILE = "bge_m3"
DEFAULT_PROFILE_FILENAME = "embedding_profiles.yaml"


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration - enhanced for Pre-Phase 7.
    This is the single source of truth for all embedding parameters.
    """

    profile: Optional[str] = Field(default=None)
    # Model configuration
    embedding_model: Optional[str] = Field(
        default=None, alias="model_name"
    )  # Support both names for backwards compat
    dims: Optional[int] = Field(default=None)  # Filled via profile/defaults
    similarity: str = Field(default="cosine")
    version: Optional[str] = Field(default=None)  # Filled via profile/defaults

    # Provider configuration (Pre-Phase 7)
    provider: Optional[str] = Field(default=None)
    task: str = Field(default="retrieval.passage")

    # Performance settings
    batch_size: int = Field(default=32, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)

    # Legacy fields for compatibility
    multilingual: bool = False
    tokenizer_backend: str = Field(default="hf")
    tokenizer_model_id: Optional[str] = None
    supports_dense: bool = True
    supports_sparse: bool = False
    supports_colbert: bool = False
    supports_long_sequences: bool = True
    normalized_output: bool = False

    @property
    def model_name(self) -> Optional[str]:
        """Backward-compatible accessor for embedding_model."""
        return self.embedding_model

    @validator("similarity")
    def validate_similarity(cls, v):
        """Validate similarity metric is supported"""
        valid_metrics = {"cosine", "dot", "euclidean"}
        if v not in valid_metrics:
            raise ValueError(f"similarity must be one of {valid_metrics}, got {v}")
        return v

    @validator("dims")
    def validate_dims(cls, v):
        """Validate dimensions are reasonable"""
        if v is None:
            return v
        if v <= 0:
            raise ValueError(f"dims must be positive, got {v}")
        if v > 4096:  # Sanity check
            logger.warning(f"dims={v} is unusually large, typical range is 128-1536")
        return v

    @validator("version")
    def validate_version(cls, v):
        """Ensure version is not empty"""
        if v is None:
            return v
        if not v or v.strip() == "":
            raise ValueError("version cannot be empty - required for provenance")
        return v

    class Config:
        populate_by_name = True  # Allow both embedding_model and model_name


class EmbeddingProfileTokenizer(BaseModel):
    backend: str = Field(default="hf")
    model_id: str

    @validator("backend")
    def _backend_not_empty(cls, value: str):
        if not value or not value.strip():
            raise ValueError(
                "tokenizer backend must be provided for embedding profiles"
            )
        return value.strip()

    @validator("model_id")
    def _tokenizer_model_not_empty(cls, value: str):
        if not value or not value.strip():
            raise ValueError(
                "tokenizer model_id must be provided for embedding profiles"
            )
        return value.strip()


class EmbeddingProfileCapabilities(BaseModel):
    supports_dense: bool = True
    supports_sparse: bool = False
    supports_colbert: bool = False
    supports_long_sequences: bool = True
    normalized_output: bool = True
    multilingual: bool = True


class EmbeddingProfileDefinition(BaseModel):
    description: Optional[str] = None
    provider: str
    model_id: str
    version: Optional[str] = None
    dims: int
    similarity: str = "cosine"
    task: str = "retrieval.passage"
    tokenizer: EmbeddingProfileTokenizer = Field(
        default_factory=EmbeddingProfileTokenizer
    )
    capabilities: EmbeddingProfileCapabilities = Field(
        default_factory=EmbeddingProfileCapabilities
    )
    requirements: List[str] = Field(default_factory=list)

    @validator("provider")
    def _provider_required(cls, value: str):
        if not value or not value.strip():
            raise ValueError(
                "provider must be provided for embedding profile definitions"
            )
        return value.strip()

    @validator("model_id")
    def _model_required(cls, value: str):
        if not value or not value.strip():
            raise ValueError(
                "model_id must be provided for embedding profile definitions"
            )
        return value.strip()

    @validator("task")
    def _task_required(cls, value: str):
        if not value or not value.strip():
            raise ValueError("task must be provided for embedding profile definitions")
        return value.strip()

    @validator("dims")
    def _positive_dims(cls, value: int):
        if value <= 0:
            raise ValueError("dims must be greater than zero")
        return value

    @validator("similarity")
    def _valid_similarity(cls, value: str):
        allowed = {"cosine", "dot", "euclidean"}
        if value not in allowed:
            raise ValueError(f"similarity must be one of {sorted(allowed)}")
        return value

    @validator("requirements", each_item=True)
    def _requirements_not_blank(cls, value: str):
        if not value or not value.strip():
            raise ValueError(
                "requirements entries must be non-empty environment variable names"
            )
        return value.strip()


class QdrantQueryStrategy(str, Enum):
    CONTENT_ONLY = "content_only"
    WEIGHTED = "weighted"
    MAX_FIELD = "max_field"


class QdrantVectorConfig(BaseModel):
    collection_name: str = "weka_sections"
    use_grpc: bool = False
    timeout: int = 30
    allow_recreate: bool = False
    query_vector_name: str = "content"
    query_strategy: QdrantQueryStrategy = QdrantQueryStrategy.CONTENT_ONLY
    enable_sparse: bool = False
    enable_colbert: bool = False
    enable_doc_title_sparse: bool = (
        True  # Enable doc_title-sparse prefetch for title term matching
    )
    use_query_api: bool = False
    query_api_dense_limit: int = 200
    query_api_sparse_limit: int = 200
    query_api_candidate_limit: int = 200
    # Sparse embedding error handling mode
    # If True: fail ingestion when sparse embedding fails (strict)
    # If False: insert None placeholder, continue gracefully (default, B.2 behavior)
    sparse_strict_mode: bool = False


class Neo4jVectorConfig(BaseModel):
    index_name: str = "section_embeddings"


class VectorSearchConfig(BaseModel):
    primary: str = "qdrant"
    dual_write: bool = False
    qdrant: QdrantVectorConfig
    neo4j: Neo4jVectorConfig


class BM25Config(BaseModel):
    """BM25/keyword retrieval configuration"""

    enabled: bool = True
    top_k: int = 50
    index_name: str = "chunk_text_index_v3"


class ExpansionRescoringConfig(BaseModel):
    """Rescoring configuration for expansion neighbors"""

    enabled: bool = False
    mode: str = "threshold_only"  # threshold_only | weighted
    normalize_method: str = "min_max"  # min_max | sigmoid | percentile
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"lexical": 0.4, "structural": 0.5, "proximity": 0.1}
    )


class ExpansionStructureConfig(BaseModel):
    """Structure-aware expansion configuration (Phase C.4)"""

    sibling_limit: int = 3  # Max sibling chunks from same parent_section
    parent_section_limit: int = 2  # Max chunks from parent section
    shared_entity_limit: int = 3  # Max chunks sharing entities with top results
    timeout_ms: int = 100  # Max latency budget for structure expansion


class ExpansionConfig(BaseModel):
    """Bounded adjacency expansion configuration for Phase 7E"""

    enabled: bool = True
    max_neighbors: int = 1
    query_min_tokens: int = 12
    score_delta_max: float = 0.02
    sparse_score_threshold: float = 0.0  # Gating threshold for sparse lexical scores
    rescoring: ExpansionRescoringConfig = Field(
        default_factory=ExpansionRescoringConfig
    )
    structure: ExpansionStructureConfig = Field(
        default_factory=ExpansionStructureConfig
    )


class RerankerConfig(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    top_n: int = 100
    max_pairs: int = 50
    max_tokens_per_pair: int = 1024


class HybridSearchConfig(BaseModel):
    enabled: bool = True
    mode: str = (
        "legacy"  # legacy (BM25+RRF) or bge_reranker (vector-only + cross-encoder)
    )
    method: str = "rrf"  # Phase 7E: rrf or weighted
    # Phase C: Graph channel configuration
    graph_channel_enabled: bool = False  # Enable graph as independent scoring channel
    graph_adaptive_enabled: bool = False  # Enable adaptive graph weight selection
    colbert_rerank_enabled: bool = True  # Enable ColBERT late-interaction reranking
    rrf_k: int = 60  # Phase 7E: RRF constant
    fusion_alpha: float = 0.6  # Phase 7E: Vector weight for weighted fusion
    vector_weight: float = 0.7  # Legacy
    graph_weight: float = 0.3  # Legacy
    # Architecture 2 semantic blend weights (recall vs rerank vs graph)
    semantic_recall_weight: float = 0.4
    semantic_rerank_weight: float = 0.4
    semantic_graph_weight: float = 0.2
    reranker_veto_threshold: float = 0.2
    graph_propagation_decay: float = 0.85
    top_k: int = 20
    vector_fields: Dict[str, float] = Field(
        default_factory=lambda: {
            "content": 1.0,
            "title": 0.35,
            "doc_title": 0.2,  # Document-level title signal for title-matching queries
            "entity": 0.2,
        }
    )
    query_type_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "conceptual": {"vector": 0.7, "graph": 0.3},
            "cli": {"vector": 0.5, "graph": 0.5},
            "config": {"vector": 0.5, "graph": 0.5},
            "procedural": {"vector": 0.6, "graph": 0.4},
            "troubleshooting": {"vector": 0.7, "graph": 0.3},
            "reference": {"vector": 0.7, "graph": 0.3},
        }
    )
    # Query-type adaptive relationship sets (used if graph_adaptive_enabled)
    query_type_relationships: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "conceptual": ["MENTIONED_IN", "DEFINES", "IN_SECTION"],
            "cli": ["MENTIONED_IN", "CONTAINS_STEP", "HAS_PARAMETER"],
            "config": ["MENTIONED_IN", "HAS_PARAMETER", "DEFINES"],
            "procedural": ["MENTIONED_IN", "CONTAINS_STEP", "NEXT_CHUNK", "IN_SECTION"],
            # Phase 2 Cleanup: Removed AFFECTS, CAUSED_BY (never materialized)
            "troubleshooting": ["MENTIONED_IN", "RESOLVES", "NEXT_CHUNK"],
            "reference": ["MENTIONED_IN", "NEXT_CHUNK"],
        }
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)  # Phase 7E
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)  # Phase 7E
    bm25_timeout_ms: int = 2000
    expansion_timeout_ms: int = 2000


class ResponseConfig(BaseModel):
    """Response building configuration for Phase 7E"""

    max_bytes_full: int = 32768  # 32KB limit for FULL mode
    max_sections: int = 10
    include_citations: bool = True
    answer_context_max_tokens: int = 4500  # Phase 7E: Max tokens for LLM context window


class GraphSearchConfig(BaseModel):
    """Graph search configuration"""

    max_depth: int = 3
    max_related_per_seed: int = 20


class SearchConfig(BaseModel):
    vector: VectorSearchConfig
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid: HybridSearchConfig
    graph: GraphSearchConfig = Field(default_factory=GraphSearchConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)  # Phase 7E-2


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10
    window_seconds: int = 60


class AuthConfig(BaseModel):
    enabled: bool = True
    algorithm: str = "HS256"
    expiry_minutes: int = 60


class AuditConfig(BaseModel):
    enabled: bool = True
    log_params: bool = True
    log_results: bool = False
    retention_days: int = 90


class ValidatorConfig(BaseModel):
    max_depth: int = 3
    max_label_scans: int = 2
    max_expand_ops: int = 5
    max_estimated_rows: int = 10000
    timeout_seconds: int = 30
    enforce_parameters: bool = True
    enforce_limits: bool = True


class TelemetryConfig(BaseModel):
    enabled: bool = True


class ReconciliationConfig(BaseModel):
    enabled: bool = True
    schedule: str = "0 2 * * *"
    drift_threshold: float = 0.01


class QueueRecoveryConfig(BaseModel):
    """Queue recovery configuration for Phase 6 job reaper"""

    enabled: bool = True
    job_timeout_seconds: int = 600
    reaper_interval_seconds: int = 30
    max_retries: int = 3
    stale_job_action: str = "requeue"


class SemanticEnrichmentConfig(BaseModel):
    enabled: bool = False
    provider: str = "stub"
    model_name: Optional[str] = None
    timeout_seconds: int = 5
    max_retries: int = 1


class ChunkStructureConfig(BaseModel):
    min_tokens: int = 800
    target_tokens: int = 1500
    hard_tokens: int = 7900
    max_sections: int = 8
    respect_major_levels: bool = True
    stop_at_level: int = 2
    break_keywords: str = (
        "faq|faqs|glossary|reference|api reference|cli reference|changelog|release notes|troubleshooting"
    )


class TokenizerConfig(BaseModel):
    """Tokenizer configuration with backend selection."""

    backend: str = "hf"


class ChunkSplitConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 7900
    overlap_tokens: int = 150


class ChunkMicrodocConfig(BaseModel):
    enabled: bool = True
    doc_token_threshold: int = 2000
    min_split_tokens: int = 400


class ChunkAssemblyConfig(BaseModel):
    assembler: str = "structured"
    structure: ChunkStructureConfig = Field(default_factory=ChunkStructureConfig)
    split: ChunkSplitConfig = Field(default_factory=ChunkSplitConfig)
    microdoc: ChunkMicrodocConfig = Field(default_factory=ChunkMicrodocConfig)
    semantic: SemanticEnrichmentConfig = Field(default_factory=SemanticEnrichmentConfig)


class IngestionConfig(BaseModel):
    batch_size: int = 500
    max_section_tokens: int = 1000
    timeout_seconds: int = 300
    workers: int = 2
    chunk_assembly: ChunkAssemblyConfig = Field(default_factory=ChunkAssemblyConfig)
    queue_recovery: QueueRecoveryConfig = Field(default_factory=QueueRecoveryConfig)
    reconciliation: ReconciliationConfig


class SchemaConfig(BaseModel):
    version: str = "v1"
    auto_migrate: bool = False


class L1CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 1000


class L2CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600
    key_prefix: str = "weka:cache:v1"


class CacheInvalidationConfig(BaseModel):
    """
    Cache invalidation configuration for Phase 7E-3.

    Epoch-based invalidation (preferred): O(1) invalidation by bumping epoch counters.
    Pattern-scan (fallback): Scans and deletes keys matching patterns.

    Reference: Canonical Spec L3184-3281 (epoch), L3116-3183 (scan)
    """

    mode: str = Field(
        default="epoch",
        description="Invalidation mode: 'epoch' (preferred O(1)) or 'scan' (fallback)",
    )
    namespace: str = Field(
        default="rag:v1", description="Cache namespace for epoch/pattern keys"
    )
    redis_uri: Optional[str] = Field(
        default=None,
        description="Redis URI for invalidation (falls back to CACHE_REDIS_URI or REDIS_URL env)",
    )

    @validator("mode")
    def validate_mode(cls, v):
        """Validate invalidation mode"""
        valid_modes = {"epoch", "scan"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {v}")
        return v


class CacheConfig(BaseModel):
    l1: L1CacheConfig
    l2: L2CacheConfig
    invalidation: CacheInvalidationConfig


class MonitoringConfig(BaseModel):
    """
    Monitoring and observability configuration for Phase 7E-4.

    Reference: Canonical Spec L4916-4976, L3513-3528
    """

    # Health checks
    health_checks_enabled: bool = Field(
        default=True, description="Enable health checks at startup"
    )
    health_check_fail_fast: bool = Field(
        default=True, description="Fail startup if health checks fail"
    )

    # SLO monitoring
    slo_monitoring_enabled: bool = Field(
        default=True, description="Enable SLO monitoring and alerting"
    )

    # SLO thresholds (from canonical spec)
    retrieval_p95_target_ms: int = Field(
        default=500, description="Retrieval p95 latency target in milliseconds"
    )
    ingestion_target_seconds: int = Field(
        default=10, description="Ingestion duration target in seconds"
    )
    expansion_rate_min: float = Field(
        default=0.10, description="Minimum expansion rate (10%)"
    )
    expansion_rate_max: float = Field(
        default=0.40, description="Maximum expansion rate (40%)"
    )

    # Metrics collection
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_aggregation_enabled: bool = Field(
        default=True, description="Enable metrics aggregation for dashboards"
    )


class AppConfig(BaseModel):
    name: str = "wekadocs-graphrag-mcp"
    version: str = "0.1.0"
    log_level: str = "INFO"
    environment: str = "development"


class FeatureFlagsConfig(BaseModel):
    """
    Feature flags for Phase 7C migration.
    Controls experimental features and gradual rollout.
    """

    # Phase 7C.4: Dual-write migration flag
    dual_write_1024d: bool = Field(
        default=False,
        description="Enable dual-write to both 384-D and 1024-D collections during migration",
    )

    # Phase 7C.8: Session tracking flag
    session_tracking_enabled: bool = Field(
        default=False, description="Enable multi-turn session tracking"
    )

    # Phase 7C.8: Entity focus bias flag
    entity_focus_bias: bool = Field(
        default=False, description="Enable entity focus bias in retrieval"
    )

    # Query API weighted fusion rollout (Phase A)
    query_api_weighted_fusion: bool = Field(
        default=False,
        description="Enable weighted per-field fusion on Query API path (Strategy 2)",
    )

    # Graph harm-reduction flags (Phase C.0)
    graph_garbage_filter: bool = Field(
        default=False,
        description="Filter short/stopword entities before graph matching",
    )
    graph_rel_types_wired: bool = Field(
        default=False,
        description="Use rel_types instead of hardcoded MENTIONED_IN in graph Cypher",
    )
    dedup_best_score: bool = Field(
        default=False,
        description="Deduplicate by keeping best fused score and merging signals",
    )
    graph_score_normalized: bool = Field(
        default=False,
        description="Normalize graph scores to [0,1] using saturating exponential",
    )

    # Graph reranker / entity embedding rollout (Phase C.2 / C.3)
    graph_as_reranker: bool = Field(
        default=False,
        description="Use graph as reranker over vector candidates instead of channel",
    )
    entity_embedding_fallback: bool = Field(
        default=False,
        description="Enable entity embedding fallback when trie resolution fails",
    )

    # Structure-aware context expansion (Phase C.4)
    structure_aware_expansion: bool = Field(
        default=False,
        description="Enable sibling, parent section, and shared-entity context expansion",
    )


class GitHubConnectorSettings(BaseModel):
    enabled: bool = False
    poll_interval_seconds: int = 300
    batch_size: int = 50
    max_retries: int = 3
    backoff_base_seconds: float = 2.0
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    owner: Optional[str] = None
    repo: Optional[str] = None
    docs_path: str = "docs"
    webhook_secret: Optional[str] = None

    @validator("owner", "repo")
    def _strip_owner_repo(cls, value):
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None


class ConnectorsConfig(BaseModel):
    queue_max_size: int = 10000
    github: Optional[GitHubConnectorSettings] = None


class ReferencesExtractionConfig(BaseModel):
    max_text_length: int = 8192
    window_overlap: int = 512
    confidence_scores: Dict[str, float] = Field(
        default_factory=lambda: {
            "hyperlink": 0.95,
            "see_also": 0.85,
            "related": 0.80,
            "refer_to": 0.70,
        }
    )


class ReferencesResolutionConfig(BaseModel):
    fuzzy_penalty: float = 0.25
    batch_size: int = 100
    min_hint_length: int = 3
    use_fulltext_index: bool = True


class ReferencesQueryConfig(BaseModel):
    enable_cross_doc_signals: bool = True
    cross_doc_weight_ratio: float = 0.3
    max_referencing_docs: int = 3


class ReferencesConfig(BaseModel):
    enabled: bool = False  # Disabled by default until verified
    extraction: ReferencesExtractionConfig = Field(
        default_factory=ReferencesExtractionConfig
    )
    resolution: ReferencesResolutionConfig = Field(
        default_factory=ReferencesResolutionConfig
    )
    query: ReferencesQueryConfig = Field(default_factory=ReferencesQueryConfig)


class Config(WekaBaseModel):
    """Main configuration model"""

    app: AppConfig
    embedding: EmbeddingConfig
    search: SearchConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    rate_limit: RateLimitConfig
    auth: AuthConfig
    audit: AuditConfig
    validator: ValidatorConfig
    telemetry: TelemetryConfig
    ingestion: IngestionConfig
    graph_schema: SchemaConfig = Field(
        alias="schema"
    )  # Renamed from schema to avoid shadowing BaseModel attribute
    cache: CacheConfig
    feature_flags: FeatureFlagsConfig = Field(
        default_factory=FeatureFlagsConfig
    )  # Phase 7C
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)  # Phase 7E-4
    connectors: Optional[ConnectorsConfig] = None
    references: ReferencesConfig = Field(default_factory=ReferencesConfig)

    class Config:
        populate_by_name = True  # Allow both graph_schema and schema


class Settings(BaseSettings):
    """Environment-based settings"""

    # Environment
    env: str = Field(default="development", alias="ENV")
    config_path: Optional[str] = Field(default=None, alias="CONFIG_PATH")

    # Embedding profiles
    embedding_profile: Optional[str] = Field(default=None, alias="EMBEDDINGS_PROFILE")
    embedding_profiles_path: Optional[str] = Field(
        default=None, alias="EMBEDDING_PROFILES_PATH"
    )
    embedding_strict_mode: bool = Field(default=True, alias="EMBEDDING_STRICT_MODE")
    validation_strict_mode: bool = Field(default=False, alias="VALIDATION_STRICT_MODE")
    embedding_namespace_mode: str = Field(
        default="none", alias="EMBEDDING_NAMESPACE_MODE"
    )
    embedding_profile_swappable: bool = Field(
        default=False, alias="EMBEDDING_PROFILE_SWAPPABLE"
    )
    embedding_profile_experiment: Optional[str] = Field(
        default=None, alias="EMBEDDING_PROFILE_EXPERIMENT"
    )

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(
        default="", alias="REDIS_PASSWORD"
    )  # Optional for workers

    # JWT (only required for MCP server, not workers)
    jwt_secret: str = Field(
        default="dev-secret-key", alias="JWT_SECRET"
    )  # Optional for workers
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    # OpenTelemetry
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_name: str = Field(default="weka-mcp-server", alias="OTEL_SERVICE_NAME")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


def _resolve_profiles_path(config_path: Path, settings: Settings) -> Path:
    if settings.embedding_profiles_path:
        return Path(settings.embedding_profiles_path).expanduser().resolve()
    return (config_path.parent / DEFAULT_PROFILE_FILENAME).resolve()


@lru_cache(maxsize=4)
def _load_embedding_profiles(
    manifest_path: str,
) -> Dict[str, EmbeddingProfileDefinition]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding profile manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profiles_data = data.get("profiles", {})
    profiles: Dict[str, EmbeddingProfileDefinition] = {}
    for name, payload in profiles_data.items():
        try:
            profiles[name] = EmbeddingProfileDefinition(**payload)
        except (
            ValidationError
        ) as exc:  # pragma: no cover - exercised via dedicated tests
            raise ValueError(
                f"Embedding profile '{name}' in {path} is invalid: {exc}"
            ) from exc
    return profiles


def _ensure_embedding_resolved(embedding: EmbeddingConfig) -> None:
    missing = []
    if not embedding.embedding_model:
        missing.append("embedding_model")
    if not embedding.dims:
        missing.append("dims")
    if not embedding.version:
        missing.append("version")
    if not embedding.provider:
        missing.append("provider")
    if missing:
        raise ValueError(
            f"Embedding configuration is incomplete after profile resolution: {missing}"
        )


def apply_embedding_profile(config: Config, settings: Settings, config_path: Path):
    profiles_path = _resolve_profiles_path(config_path, settings)
    profiles = _load_embedding_profiles(str(profiles_path))

    configured_profile = config.embedding.profile
    runtime_profile = settings.embedding_profile

    # Compute the baseline profile from config or manifest defaults
    if not configured_profile:
        if DEFAULT_EMBEDDING_PROFILE in profiles:
            configured_profile = DEFAULT_EMBEDDING_PROFILE
            logger.info(
                "No embedding profile specified in config. Defaulting to %s",
                DEFAULT_EMBEDDING_PROFILE,
            )
        elif profiles:
            configured_profile = next(iter(profiles.keys()))
            logger.info(
                "No embedding profile specified in config. Defaulting to first entry: %s",
                configured_profile,
            )

    profile_name = runtime_profile or configured_profile

    if profile_name and profile_name not in profiles:
        raise ValueError(
            f"Embedding profile '{profile_name}' not defined in {profiles_path}"
        )

    if not profile_name:
        logger.warning(
            "Embedding profile not provided and manifest empty. Proceeding with legacy config values."
        )
        _ensure_embedding_resolved(config.embedding)
        return

    # Phase 6: rollout & safety guard for non-dev/test environments
    env = (settings.env or "").lower()
    is_strict_env = env not in ("development", "dev", "test")
    if (
        is_strict_env
        and runtime_profile
        and configured_profile
        and runtime_profile != configured_profile
    ):
        allowed = False

        if settings.embedding_profile_swappable:
            allowed = True
            logger.warning(
                "Embedding profile override from '%s' to '%s' permitted via "
                "EMBEDDING_PROFILE_SWAPPABLE in '%s' environment.",
                configured_profile,
                runtime_profile,
                settings.env,
            )
        elif (
            settings.embedding_profile_experiment
            and settings.embedding_profile_experiment == runtime_profile
        ):
            allowed = True
            logger.warning(
                "Embedding profile override from '%s' to experimental profile '%s' "
                "permitted via EMBEDDING_PROFILE_EXPERIMENT in '%s' environment.",
                configured_profile,
                runtime_profile,
                settings.env,
            )

        if not allowed:
            raise ValueError(
                "Refusing to apply embedding profile override from "
                f"{configured_profile!r} to {runtime_profile!r} in environment "
                f"{settings.env!r} without EMBEDDING_PROFILE_SWAPPABLE=true or "
                f"EMBEDDING_PROFILE_EXPERIMENT={runtime_profile!r}."
            )

    profile = profiles[profile_name]
    strict_env = env not in ("development", "dev", "test")
    overrides = {
        "profile": profile_name,
        "embedding_model": profile.model_id,
        "dims": profile.dims,
        "similarity": profile.similarity,
        "version": profile.version or profile.model_id,
        "provider": profile.provider,
        "task": profile.task,
        "tokenizer_backend": profile.tokenizer.backend,
        "tokenizer_model_id": profile.tokenizer.model_id,
        "supports_dense": profile.capabilities.supports_dense,
        "supports_sparse": profile.capabilities.supports_sparse,
        "supports_colbert": profile.capabilities.supports_colbert,
        "supports_long_sequences": profile.capabilities.supports_long_sequences,
        "normalized_output": profile.capabilities.normalized_output,
        "multilingual": profile.capabilities.multilingual,
    }
    config.embedding = config.embedding.copy(update=overrides)

    suffix_source = _resolve_namespace_suffix(
        profile_name, profile, settings.embedding_namespace_mode
    )
    qdrant_cfg = getattr(config.search.vector, "qdrant", None)
    neo4j_cfg = getattr(config.search.vector, "neo4j", None)
    bm25_cfg = getattr(config.search, "bm25", None)
    if suffix_source:
        if qdrant_cfg and hasattr(qdrant_cfg, "collection_name"):
            qdrant_cfg.collection_name = namespace_identifier(
                qdrant_cfg.collection_name, suffix_source
            )
        if neo4j_cfg and hasattr(neo4j_cfg, "index_name"):
            neo4j_cfg.index_name = namespace_identifier(
                neo4j_cfg.index_name, suffix_source
            )
        if bm25_cfg and hasattr(bm25_cfg, "index_name"):
            bm25_cfg.index_name = namespace_identifier(
                bm25_cfg.index_name, suffix_source
            )

    if qdrant_cfg:
        if getattr(qdrant_cfg, "enable_sparse", False) and not getattr(
            profile.capabilities, "supports_sparse", False
        ):
            message = (
                f"Profile '{profile_name}' does not support sparse embeddings but "
                "enable_sparse is True."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Disabling sparse for this run.", message)
            qdrant_cfg.enable_sparse = False
        if getattr(qdrant_cfg, "enable_colbert", False) and not getattr(
            profile.capabilities, "supports_colbert", False
        ):
            message = (
                f"Profile '{profile_name}' does not support ColBERT but "
                "enable_colbert is True."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Disabling ColBERT for this run.", message)
            qdrant_cfg.enable_colbert = False
        if getattr(qdrant_cfg, "enable_colbert", False) and not getattr(
            qdrant_cfg, "use_query_api", False
        ):
            message = (
                "ColBERT requires search.vector.qdrant.use_query_api=True; it is "
                "currently False."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Auto-enabling Query API for this run.", message)
            qdrant_cfg.use_query_api = True
        # Ensure namespaced names reflect profile slug (not version) to stay consistent with runtime validation
        if suffix_source and hasattr(qdrant_cfg, "collection_name"):
            qdrant_cfg.collection_name = namespace_identifier(
                qdrant_cfg.collection_name, suffix_source
            )

    missing_req = [req for req in profile.requirements if not os.getenv(req)]
    if missing_req:
        logger.warning(
            "Embedding profile '%s' requires environment variables %s which are not set.",
            profile_name,
            missing_req,
        )

    logger.info(
        "Embedding profile applied",
        extra={
            "embedding_profile": profile_name,
            "embedding_namespace_mode": settings.embedding_namespace_mode,
            "bm25_index_name": getattr(bm25_cfg, "index_name", None),
            "qdrant_collection_name": getattr(qdrant_cfg, "collection_name", None),
            "neo4j_vector_index_name": getattr(neo4j_cfg, "index_name", None),
        },
    )

    _ensure_embedding_resolved(config.embedding)


def load_config() -> tuple[Config, Settings]:
    """
    Load configuration from YAML file and environment variables.
    Enhanced for Pre-Phase 7 with startup validation.

    Returns:
        tuple: (Config, Settings) - YAML config and environment settings

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration validation fails
    """
    # Load environment settings
    settings = Settings()

    # Determine config file path
    if settings.config_path:
        config_path = Path(settings.config_path)
    else:
        config_path = (
            Path(__file__).parent.parent.parent / "config" / f"{settings.env}.yaml"
        )

    # Load YAML config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = Config(**config_dict)
    apply_embedding_profile(config, settings, config_path)

    # Pre-Phase 7: Perform startup validation
    validate_config_at_startup(config, settings)

    return config, settings


def _legacy_env_override(
    env_name: str,
    current_value,
    env_parser=lambda v: v,
    env_overrides: Dict[str, str] | None = None,
):
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return current_value
    try:
        parsed_value = env_parser(raw_value)
    except Exception:
        logger.warning(
            "Failed to parse legacy env override %s=%s; keeping value %s",
            env_name,
            raw_value,
            current_value,
        )
        return current_value
    logger.warning(
        "Legacy env %s overriding embedding profile value %s -> %s",
        env_name,
        current_value,
        parsed_value,
    )
    if env_overrides is not None:
        env_overrides[env_name] = raw_value
    return parsed_value


def get_embedding_settings(
    config_override: Optional[Config] = None,
) -> ProviderEmbeddingSettings:
    """
    Build EmbeddingSettings from resolved config + legacy env overrides.
    """
    config = config_override or get_config()
    env_overrides: Dict[str, str] = {}

    embedding = config.embedding
    dims_default = embedding.dims or 0

    provider = _legacy_env_override(
        "EMBEDDINGS_PROVIDER", embedding.provider, env_overrides=env_overrides
    )
    model = _legacy_env_override(
        "EMBEDDINGS_MODEL",
        embedding.embedding_model,
        env_overrides=env_overrides,
    )
    dims = _legacy_env_override(
        "EMBEDDINGS_DIM",
        dims_default,
        env_parser=lambda v: int(v),
        env_overrides=env_overrides,
    )
    task = _legacy_env_override(
        "EMBEDDINGS_TASK", embedding.task, env_overrides=env_overrides
    )

    service_url = None
    if provider == "bge-m3-service":
        service_url = os.getenv("BGE_M3_API_URL")
        if not service_url:
            logger.warning(
                "BGE_M3_API_URL is not set but provider bge-m3-service is active."
            )

    capabilities = ProviderEmbeddingCapabilities(
        supports_dense=embedding.supports_dense,
        supports_sparse=embedding.supports_sparse,
        supports_colbert=embedding.supports_colbert,
        supports_long_sequences=embedding.supports_long_sequences,
        normalized_output=embedding.normalized_output,
        multilingual=embedding.multilingual,
    )

    if env_overrides:
        overrides = ", ".join(f"{k}={v}" for k, v in env_overrides.items())
        logger.warning(
            "Legacy embedding env overrides detected (%s). Prefer EMBEDDINGS_PROFILE to keep settings consistent.",
            overrides,
        )

    return ProviderEmbeddingSettings(
        profile=embedding.profile,
        provider=provider,
        model_id=model,
        version=embedding.version or model,
        dims=dims,
        similarity=embedding.similarity,
        task=task,
        tokenizer_backend=embedding.tokenizer_backend,
        tokenizer_model_id=embedding.tokenizer_model_id,
        service_url=service_url,
        capabilities=capabilities,
        extra=env_overrides,
    )


def validate_config_at_startup(config: Config, settings: Settings) -> None:
    """
    Validate critical configuration at startup.
    Pre-Phase 7 requirement: Fail fast on invalid configuration.

    Args:
        config: Loaded configuration
        settings: Environment settings

    Raises:
        ValueError: If critical validation fails
    """
    # Log embedding configuration (Pre-Phase 7 requirement)
    logger.info(
        "Embedding configuration loaded: profile=%s model=%s dims=%s "
        "version=%s provider=%s tokenizer=%s",
        config.embedding.profile or "legacy",
        config.embedding.embedding_model,
        config.embedding.dims,
        config.embedding.version,
        config.embedding.provider,
        config.embedding.tokenizer_model_id or config.embedding.tokenizer_backend,
    )

    # Validate embedding dimensions are positive
    if not config.embedding.dims or config.embedding.dims <= 0:
        raise ValueError(
            f"embedding.dims must be positive, got {config.embedding.dims}"
        )

    # Validate embedding version is set
    if not config.embedding.version:
        raise ValueError("embedding.version is required for provenance tracking")

    if not config.embedding.provider:
        raise ValueError("embedding.provider must be defined after profile resolution")

    # Validate similarity metric
    valid_similarities = {"cosine", "dot", "euclidean"}
    if config.embedding.similarity not in valid_similarities:
        raise ValueError(
            f"embedding.similarity must be one of {valid_similarities}, "
            f"got {config.embedding.similarity}"
        )

    # Validate Qdrant distance matches embedding similarity
    if hasattr(config.search.vector.qdrant, "distance"):
        qdrant_distance = getattr(config.search.vector.qdrant, "distance", "cosine")
        if qdrant_distance != config.embedding.similarity:
            logger.warning(
                f"Qdrant distance ({qdrant_distance}) != embedding similarity "
                f"({config.embedding.similarity}). This may cause issues."
            )

    # Log feature flags (Pre-Phase 7)
    if hasattr(config, "feature_flags"):
        logger.info(f"Feature flags: {config.feature_flags}")

    logger.info("Configuration validation successful")


# Global config instances (loaded once at startup)
_config: Optional[Config] = None
_settings: Optional[Settings] = None


def get_config() -> Config:
    """Get the global Config instance"""
    global _config
    if _config is None:
        _config, _ = load_config()
    return _config


def get_settings() -> Settings:
    """Get the global Settings instance"""
    global _settings
    if _settings is None:
        _, _settings = load_config()
    return _settings


def init_config() -> tuple[Config, Settings]:
    """Initialize and cache global config instances"""
    global _config, _settings
    _config, _settings = load_config()
    return _config, _settings


def reload_config() -> tuple[Config, Settings]:
    """Force reload of config/settings from disk and environment."""
    global _config, _settings
    _config, _settings = load_config()
    return _config, _settings


def _slugify_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def get_expected_namespace_suffix(
    settings: ProviderEmbeddingSettings, mode: str
) -> str:
    """
    Deterministic namespace suffix (single source of truth).

    Priority:
    1) Explicit mode selection (profile/version/model)
    2) Fallback: profile, else version, else model_id
    3) If namespacing is disabled/empty, return ""
    """

    normalized = (mode or "").lower()
    if normalized in {"", "none", "disabled"}:
        return ""

    if normalized == "profile" and getattr(settings, "profile", None):
        return _slugify_identifier(settings.profile)
    if normalized == "version" and getattr(settings, "version", None):
        return _slugify_identifier(settings.version)
    if normalized == "model" and getattr(settings, "model_id", None):
        return _slugify_identifier(settings.model_id)

    # Fallback safety: prefer profile, then version, then model_id
    if getattr(settings, "profile", None):
        return _slugify_identifier(settings.profile)
    if getattr(settings, "version", None):
        return _slugify_identifier(settings.version)
    if getattr(settings, "model_id", None):
        return _slugify_identifier(settings.model_id)
    return ""


def namespace_identifier(base: str, suffix: Optional[str]) -> str:
    """Append a slugified suffix to the base identifier if provided."""
    if not suffix:
        return base
    slug = _slugify_identifier(suffix)
    if not slug:
        return base
    if base.endswith(f"_{slug}"):
        return base
    return f"{base}_{slug}"


def _resolve_namespace_suffix(
    profile_name: Optional[str],
    profile: Optional[EmbeddingProfileDefinition],
    mode: str,
) -> Optional[str]:
    normalized = (mode or "profile").lower()
    if normalized in {"", "none", "disabled"}:
        return None
    if normalized == "profile":
        return profile_name
    if normalized == "version" and profile:
        return profile.version or profile.model_id
    if normalized == "model" and profile:
        return profile.model_id
    return None
