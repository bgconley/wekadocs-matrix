"""
Embedding provider interfaces and implementations.

Providers:
- SentenceTransformersProvider: Local HuggingFace models
- SnowflakeArcticProvider: Dense embeddings via local Arctic service
- BGEM3ServiceProvider: Multi-head (dense + sparse + ColBERT) via BGE-M3 service
- VoyageEmbeddingProvider: Voyage AI contextual embeddings
- JinaEmbeddingProvider: Jina AI embeddings
"""

from .base import EmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider
from .snowflake_arctic import SnowflakeArcticProvider

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "SnowflakeArcticProvider",
]
