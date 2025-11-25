from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class EmbeddingCapabilities:
    supports_dense: bool = True
    supports_sparse: bool = False
    supports_colbert: bool = False
    supports_long_sequences: bool = True
    normalized_output: bool = True
    multilingual: bool = True


@dataclass(frozen=True)
class EmbeddingSettings:
    profile: Optional[str]
    provider: str
    model_id: str
    version: str
    dims: int
    similarity: str
    task: str
    tokenizer_backend: str
    tokenizer_model_id: Optional[str]
    service_url: Optional[str]
    capabilities: EmbeddingCapabilities
    extra: Dict[str, str] = field(default_factory=dict)


def build_embedding_telemetry(settings: "EmbeddingSettings") -> Dict[str, str]:
    """Return standardized telemetry tags for embedding settings."""

    return {
        "embedding_profile": (settings.profile or "legacy"),
        "embedding_provider": settings.provider,
        "embedding_model": settings.model_id,
        "embedding_dims": str(settings.dims),
        "tokenizer_backend": settings.tokenizer_backend or "unknown",
        "tokenizer_model": settings.tokenizer_model_id or "unspecified",
    }
