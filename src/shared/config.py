# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# Configuration loader with environment variable support
# Enhanced for Pre-Phase 7: Added validation for embedding configuration

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from .models import WekaBaseModel

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration - enhanced for Pre-Phase 7.
    This is the single source of truth for all embedding parameters.
    """

    # Model configuration
    embedding_model: str = Field(
        alias="model_name"
    )  # Support both names for backwards compat
    dims: int = Field(..., gt=0)  # Must be positive
    similarity: str = Field(default="cosine")
    version: str = Field(...)  # Required for provenance tracking

    # Provider configuration (Pre-Phase 7)
    provider: str = Field(default="sentence-transformers")
    task: str = Field(default="retrieval.passage")

    # Performance settings
    batch_size: int = Field(default=32, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)

    # Legacy fields for compatibility
    multilingual: bool = False

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
        if v <= 0:
            raise ValueError(f"dims must be positive, got {v}")
        if v > 4096:  # Sanity check
            logger.warning(f"dims={v} is unusually large, typical range is 128-1536")
        return v

    @validator("version")
    def validate_version(cls, v):
        """Ensure version is not empty"""
        if not v or v.strip() == "":
            raise ValueError("version cannot be empty - required for provenance")
        return v

    class Config:
        populate_by_name = True  # Allow both embedding_model and model_name


class QdrantVectorConfig(BaseModel):
    collection_name: str = "weka_sections"
    use_grpc: bool = False
    timeout: int = 30


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


class ExpansionConfig(BaseModel):
    """Bounded adjacency expansion configuration for Phase 7E"""

    enabled: bool = True
    max_neighbors: int = 1
    query_min_tokens: int = 12
    score_delta_max: float = 0.02


class RerankerConfig(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    top_n: int = 100


class HybridSearchConfig(BaseModel):
    enabled: bool = True
    method: str = "rrf"  # Phase 7E: rrf or weighted
    rrf_k: int = 60  # Phase 7E: RRF constant
    fusion_alpha: float = 0.6  # Phase 7E: Vector weight for weighted fusion
    vector_weight: float = 0.7  # Legacy
    graph_weight: float = 0.3  # Legacy
    top_k: int = 20
    vector_fields: Dict[str, float] = Field(
        default_factory=lambda: {"content": 1.0, "title": 0.35, "entity": 0.2}
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)  # Phase 7E
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)  # Phase 7E


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


class Config(WekaBaseModel):
    """Main configuration model"""

    app: AppConfig
    embedding: EmbeddingConfig
    search: SearchConfig
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

    class Config:
        populate_by_name = True  # Allow both graph_schema and schema


class Settings(BaseSettings):
    """Environment-based settings"""

    # Environment
    env: str = Field(default="development", alias="ENV")
    config_path: Optional[str] = Field(default=None, alias="CONFIG_PATH")

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

    # Pre-Phase 7: Perform startup validation
    validate_config_at_startup(config, settings)

    return config, settings


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
        f"Embedding configuration loaded: "
        f"model={config.embedding.embedding_model}, "
        f"dims={config.embedding.dims}, "
        f"version={config.embedding.version}, "
        f"provider={config.embedding.provider}"
    )

    # Validate embedding dimensions are positive
    if config.embedding.dims <= 0:
        raise ValueError(
            f"embedding.dims must be positive, got {config.embedding.dims}"
        )

    # Validate embedding version is set
    if not config.embedding.version:
        raise ValueError("embedding.version is required for provenance tracking")

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
