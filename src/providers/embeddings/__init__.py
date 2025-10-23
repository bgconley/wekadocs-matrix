"""
Embedding provider interfaces and implementations.
Pre-Phase 7: Creates abstraction layer for embedding generation.
"""

from .base import EmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider

__all__ = ["EmbeddingProvider", "SentenceTransformersProvider"]
