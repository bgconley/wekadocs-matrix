"""
Snowflake Arctic embedding provider for dense embeddings.

This provider wraps the SnowflakeEmbeddingClient to provide the EmbeddingProvider
interface for use in the multi-embedder architecture.

Architecture Role:
- Dense embeddings: YES (primary use case)
- Sparse embeddings: NO (use BGE-M3)
- ColBERT embeddings: NO (use BGE-M3)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.clients.snowflake_embedding_client import SnowflakeEmbeddingClient
from src.providers.embeddings.base import EmbeddingProvider
from src.providers.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class SnowflakeArcticProvider(EmbeddingProvider):
    """
    EmbeddingProvider backed by local Snowflake Arctic embed service.

    Uses the OpenAI-compatible API exposed by the local Arctic service.
    This provider is used for:
    - Dense content embeddings (storage and retrieval)
    - Dense title embeddings
    - Chonkie semantic chunking (via separate adapter)

    NOT used for sparse or ColBERT (those use BGE-M3).

    Environment Variables:
        CHONKIE_EMBEDDINGS_BASE_URL: Service URL (default: http://127.0.0.1:9010/v1)
        CHONKIE_EMBEDDINGS_API_KEY: Optional bearer token
        CHONKIE_EMBEDDINGS_TIMEOUT_SECONDS: Request timeout (default: 60)
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:9010/v1"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Optional[SnowflakeEmbeddingClient] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize the Snowflake Arctic provider.

        Args:
            settings: EmbeddingSettings from profile configuration
            client: Optional pre-configured client (for testing)
            base_url: Override service URL
            timeout: Override request timeout
        """
        if settings is None:
            raise ValueError(
                "EmbeddingSettings are required for SnowflakeArcticProvider. "
                "Use ProviderFactory.create_embedding_provider_for_role() or "
                "provide settings directly."
            )

        self._settings = settings
        self._dims = settings.dims
        self._model_id = settings.model_id
        self._provider_name = settings.provider or "snowflake-arctic-service"
        self._capabilities = settings.capabilities

        # Resolve service URL: explicit > settings > env > default
        self._base_url = (
            base_url
            or settings.service_url
            or os.getenv("CHONKIE_EMBEDDINGS_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        # Resolve timeout
        self._timeout = timeout or float(
            os.getenv("CHONKIE_EMBEDDINGS_TIMEOUT_SECONDS", str(self.DEFAULT_TIMEOUT))
        )

        # Create or use provided client
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            api_key = os.getenv("CHONKIE_EMBEDDINGS_API_KEY")
            self._client = SnowflakeEmbeddingClient(
                base_url=self._base_url,
                api_key=api_key,
                timeout=self._timeout,
            )
            self._owns_client = True

        logger.info(
            "SnowflakeArcticProvider initialized",
            extra={
                "model_id": self._model_id,
                "dims": self._dims,
                "base_url": self._base_url,
                "provider": self._provider_name,
            },
        )

    @property
    def dims(self) -> int:
        """Embedding dimensionality (1024 for Arctic)."""
        return self._dims

    @property
    def model_id(self) -> str:
        """Model identifier."""
        return self._model_id

    @property
    def provider_name(self) -> str:
        """Provider name for identification."""
        return self._provider_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors (each 1024 floats for Arctic)

        Raises:
            ValueError: If texts is empty
            RuntimeError: If API call fails after retries
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of documents.")

        # Normalize model name for the service
        model_name = self._normalize_model_name(self._model_id)

        response = self._client.embeddings(
            model=model_name,
            input=texts,
            encoding_format="float",
            dimensions=self._dims,
        )

        # Extract and sort by index to maintain order
        data = response.get("data", [])
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate dense embedding for a single query.

        Uses the query-specific model variant (snowflake-arctic-embed-l-v2.0-query)
        which automatically adds the 'query: ' prefix for asymmetric retrieval.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (1024 floats for Arctic)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query.")

        # Use query model variant for proper asymmetric retrieval
        model_name = self._normalize_model_name(self._model_id)
        query_model = f"{model_name}-query"

        response = self._client.embeddings(
            model=query_model,
            input=[text],
            encoding_format="float",
            dimensions=self._dims,
        )

        data = response.get("data", [])
        if not data:
            raise RuntimeError("No embedding returned from Arctic service")
        return data[0]["embedding"]

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """
        Normalize HuggingFace-style model name to service format.

        Examples:
            "Snowflake/snowflake-arctic-embed-l-v2.0" -> "snowflake-arctic-embed-l-v2.0"
            "snowflake-arctic-embed-l-v2.0" -> "snowflake-arctic-embed-l-v2.0"
        """
        if "/" in model_name:
            candidate = model_name.split("/")[-1]
            if candidate.startswith("snowflake-arctic-embed-"):
                return candidate
        return model_name

    def validate_dimensions(self, expected_dims: int) -> bool:
        """Check if provider dimensions match expected."""
        return self._dims == expected_dims

    def close(self) -> None:
        """Release client resources."""
        if self._owns_client and self._client:
            self._client.close()

    def __repr__(self) -> str:
        return (
            f"SnowflakeArcticProvider("
            f"model={self._model_id}, "
            f"dims={self._dims}, "
            f"url={self._base_url})"
        )

    # -------------------------------------------------------------------------
    # Sparse / ColBERT - Explicitly NOT supported
    # -------------------------------------------------------------------------

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Not supported - use BGE-M3 for sparse embeddings.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "SnowflakeArcticProvider does not support sparse embeddings. "
            "Use BGE-M3 (bge_m3 profile) for sparse embeddings."
        )

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """
        Not supported - use BGE-M3 for ColBERT embeddings.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "SnowflakeArcticProvider does not support ColBERT embeddings. "
            "Use BGE-M3 (bge_m3 profile) for ColBERT embeddings."
        )


__all__ = ["SnowflakeArcticProvider"]
