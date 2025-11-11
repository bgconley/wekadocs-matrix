"""
Ollama embedding provider implementation.
Phase 7C: Local BGE-M3 embeddings via native macOS Ollama.

Architecture:
- Runs natively on macOS (not Docker) for direct Metal GPU access
- Docker services connect via host.docker.internal:11434
- Fallback provider when Jina API unavailable or budget exceeded

Model: BAAI/bge-m3 (1024-D)
"""

import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """
    Ollama embedding provider using BGE-M3 model.

    Connects to native macOS Ollama service for Metal GPU acceleration.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        dims: int = 1024,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model: Ollama model name (must be pulled first: `ollama pull bge-m3`)
            dims: Expected embedding dimensions
            base_url: Ollama API base URL (defaults to localhost or host.docker.internal)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If dims invalid
            RuntimeError: If connection to Ollama fails
        """
        self._model_id = model
        self._dims = dims
        self._provider_name = "ollama"
        self._timeout = timeout

        # Determine base URL
        # Default: native macOS = localhost, from Docker = host.docker.internal
        if base_url:
            self._base_url = base_url
        else:
            # Check if we're in Docker (via env var or detection)
            in_docker = os.path.exists("/.dockerenv")
            self._base_url = (
                "http://host.docker.internal:11434"
                if in_docker
                else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )

        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=timeout, base_url=self._base_url, follow_redirects=True
        )

        # Validate connection to Ollama
        self._validate_connection()

        logger.info(
            f"OllamaEmbeddingProvider initialized: "
            f"model={model}, dims={dims}, base_url={self._base_url}"
        )

    def _validate_connection(self):
        """
        Validate connection to Ollama service.

        Raises:
            RuntimeError: If Ollama is unreachable or model not available
        """
        try:
            # Check Ollama health
            response = self._client.get("/api/tags")
            response.raise_for_status()

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if not any(self._model_id in name for name in model_names):
                logger.warning(
                    f"Model {self._model_id} not found in Ollama. "
                    f"Available models: {model_names}. "
                    f"Run: ollama pull {self._model_id}"
                )
                # Don't fail here - model might be pulled later

        except httpx.HTTPError as e:
            logger.error(f"Failed to connect to Ollama at {self._base_url}: {e}")
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Ensure Ollama is running: brew services start ollama"
            )

    @property
    def dims(self) -> int:
        """Get embedding dimensions."""
        return self._dims

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Metrics
        import time

        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        start_time = time.time()

        try:
            # Embed each text (Ollama API doesn't support batching in single call)
            embeddings = []
            for text in texts:
                response = self._client.post(
                    "/api/embeddings",
                    json={
                        "model": self._model_id,
                        "prompt": text,
                    },
                )

                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding")
                if not embedding:
                    raise RuntimeError("Ollama returned no embedding for text")

                # Validate dimensions
                if len(embedding) != self._dims:
                    raise RuntimeError(
                        f"Ollama returned {len(embedding)}-D vector, "
                        f"expected {self._dims}-D. "
                        f"Verify {self._model_id} model produces {self._dims}-D embeddings."
                    )

                embeddings.append(embedding)

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            embedding_request_total.labels(
                model_id=self._model_id, operation="documents"
            ).inc()
            embedding_latency_ms.labels(
                model_id=self._model_id, operation="documents"
            ).observe(latency_ms)

            logger.debug(
                f"Ollama embeddings generated: {len(embeddings)} vectors, "
                f"{latency_ms:.2f}ms"
            )

            return embeddings

        except Exception as e:
            embedding_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to embed documents with Ollama: {e}")
            raise RuntimeError(f"Ollama document embedding failed: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Note: BGE-M3 uses same encoding for queries and documents.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query text")

        # Metrics
        import time

        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        start_time = time.time()

        try:
            response = self._client.post(
                "/api/embeddings",
                json={
                    "model": self._model_id,
                    "prompt": text,
                },
            )

            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError("Ollama returned no embedding")

            # Validate dimensions
            if len(embedding) != self._dims:
                raise RuntimeError(
                    f"Ollama returned {len(embedding)}-D vector, "
                    f"expected {self._dims}-D"
                )

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            embedding_request_total.labels(
                model_id=self._model_id, operation="query"
            ).inc()
            embedding_latency_ms.labels(
                model_id=self._model_id, operation="query"
            ).observe(latency_ms)

            logger.debug(f"Ollama query embedding generated: {latency_ms:.2f}ms")

            return embedding

        except Exception as e:
            embedding_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to embed query with Ollama: {e}")
            raise RuntimeError(f"Ollama query embedding failed: {e}")

    def validate_dimensions(self, expected_dims: int) -> bool:
        """Validate dimensions match expected."""
        return self.dims == expected_dims

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
