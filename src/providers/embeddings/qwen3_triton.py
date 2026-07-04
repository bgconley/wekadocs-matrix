# =============================================================================
# @status: ACTIVE
# @called-by: factory.py (registered provider, current plan.dense)
# =============================================================================
"""
Qwen3-Embedding-4B provider for dense embeddings via Triton gateway.

Architecture Role:
- Dense embeddings: YES (primary use case)
- Sparse embeddings: NO (use BGE-M3)
- ColBERT embeddings: NO (use BGE-M3)

The gateway at /embed handles tokenization, Triton inference, last-token pooling,
1024-dim truncation, and L2 normalization.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.clients.qwen3_embedding_client import Qwen3EmbeddingClient
from src.providers.embeddings.base import EmbeddingProvider
from src.providers.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class Qwen3TritonProvider(EmbeddingProvider):
    """
    EmbeddingProvider backed by Qwen3-Embedding-4B via FastAPI gateway + Triton.

    Used for:
    - Dense content embeddings (storage and retrieval)
    - Dense title embeddings
    - Chonkie semantic chunking (via separate adapter)

    NOT used for sparse or ColBERT (those use BGE-M3).

    Environment Variables:
        QWEN3_EMBED_URL: Gateway URL (default: http://10.25.0.50:8101)
        QWEN3_EMBED_MODEL: Model identifier (default: Qwen/Qwen3-Embedding-4B)
        QWEN3_EMBED_TIMEOUT_SECONDS: Request timeout (default: 60)
        QWEN3_EMBED_API_KEY: Optional bearer token
    """

    DEFAULT_BASE_URL = "http://10.25.0.50:8101"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Optional[Qwen3EmbeddingClient] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if settings is None:
            raise ValueError(
                "EmbeddingSettings are required for Qwen3TritonProvider. "
                "Use ProviderFactory.create_embedding_provider_for_role() or "
                "provide settings directly."
            )

        self._settings = settings
        self._dims = settings.dims
        self._model_id = settings.model_id
        self._provider_name = settings.provider or "qwen3-triton-service"
        self._capabilities = settings.capabilities

        self._base_url = (
            base_url
            or settings.service_url
            or os.getenv("QWEN3_EMBED_URL")
            or self.DEFAULT_BASE_URL
        )

        self._timeout = timeout or float(
            os.getenv("QWEN3_EMBED_TIMEOUT_SECONDS", str(self.DEFAULT_TIMEOUT))
        )

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            model = os.getenv("QWEN3_EMBED_MODEL") or self._model_id
            api_key = os.getenv("QWEN3_EMBED_API_KEY")
            self._client = Qwen3EmbeddingClient(
                base_url=self._base_url,
                model=model,
                api_key=api_key,
                timeout=self._timeout,
            )
            self._owns_client = True

        logger.info(
            "Qwen3TritonProvider initialized",
            extra={
                "model_id": self._model_id,
                "dims": self._dims,
                "base_url": self._base_url,
                "provider": self._provider_name,
            },
        )

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty list of documents.")

        response = self._client.embed(texts)
        embeddings = response.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Qwen3 gateway returned {len(embeddings)} embeddings "
                f"for {len(texts)} texts"
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query.")

        response = self._client.embed([text])
        embeddings = response.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("No embedding returned from Qwen3 gateway")
        return embeddings[0]

    def validate_dimensions(self, expected_dims: int) -> bool:
        return self._dims == expected_dims

    def close(self) -> None:
        if self._owns_client and self._client:
            self._client.close()

    def __repr__(self) -> str:
        return (
            f"Qwen3TritonProvider("
            f"model={self._model_id}, "
            f"dims={self._dims}, "
            f"url={self._base_url})"
        )

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        raise NotImplementedError(
            "Qwen3TritonProvider does not support sparse embeddings. "
            "Use BGE-M3 (bge_m3 profile) for sparse embeddings."
        )

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        raise NotImplementedError(
            "Qwen3TritonProvider does not support ColBERT embeddings. "
            "Use BGE-M3 (bge_m3 profile) for ColBERT embeddings."
        )


__all__ = ["Qwen3TritonProvider"]
