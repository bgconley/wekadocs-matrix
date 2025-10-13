# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# Configuration loader with environment variable support

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseModel):
    model_name: str
    dims: int
    similarity: str = "cosine"
    multilingual: bool = False
    version: str = "v1"


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


class HybridSearchConfig(BaseModel):
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    top_k: int = 20


class SearchConfig(BaseModel):
    vector: VectorSearchConfig
    hybrid: HybridSearchConfig


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


class IngestionConfig(BaseModel):
    batch_size: int = 500
    max_section_tokens: int = 1000
    timeout_seconds: int = 300
    workers: int = 2
    reconciliation: ReconciliationConfig


class SchemaConfig(BaseModel):
    version: str = "v1"
    auto_migrate: bool = False


class AppConfig(BaseModel):
    name: str = "wekadocs-graphrag-mcp"
    version: str = "0.1.0"
    log_level: str = "INFO"
    environment: str = "development"


class Config(BaseModel):
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
    schema: SchemaConfig


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
    redis_password: str = Field(..., alias="REDIS_PASSWORD")

    # JWT
    jwt_secret: str = Field(..., alias="JWT_SECRET")
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

    Returns:
        tuple: (Config, Settings) - YAML config and environment settings
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

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = Config(**config_dict)

    return config, settings


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
