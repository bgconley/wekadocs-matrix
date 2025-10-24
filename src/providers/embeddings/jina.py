"""
Jina AI embedding provider implementation.
Phase 7C: Remote API provider for jina-embeddings-v4 @ 1024-D.

Features:
- Task-specific embeddings (retrieval.passage vs retrieval.query)
- Exponential backoff with jitter
- Circuit breaker for API failures
- Comprehensive error handling
"""

import logging
import os
import random
import time
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker for API resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening
            timeout: Seconds to wait before trying half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful call."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker OPEN after {self.failures} consecutive failures"
            )

    def can_attempt(self) -> bool:
        """Check if call can be attempted."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering HALF-OPEN state")
                return True
            return False

        # half-open: allow one attempt
        return True


class JinaEmbeddingProvider:
    """
    Jina AI embedding provider using jina-embeddings-v4.

    Supports task-specific embeddings:
    - retrieval.passage: For document/section embedding during ingestion
    - retrieval.query: For query embedding during search
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model: str = "jina-embeddings-v4",
        dims: int = 1024,
        api_key: Optional[str] = None,
        task: str = "retrieval.passage",
        timeout: int = 30,
    ):
        """
        Initialize Jina embedding provider.

        Args:
            model: Model identifier
            dims: Expected embedding dimensions
            api_key: Jina API key (from env if not provided)
            task: Default task for embed_documents
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is missing or dims invalid
        """
        self._model_id = model
        self._dims = dims
        self._task = task
        self._provider_name = "jina-ai"
        self._timeout = timeout

        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("JINA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "JINA_API_KEY required for jina-ai provider. "
                "Set JINA_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=300)

        logger.info(
            f"JinaEmbeddingProvider initialized: model={model}, dims={dims}, task={task}"
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

        Uses retrieval.passage task by default.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts is empty
            RuntimeError: If API call fails or circuit breaker open
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            raise RuntimeError(
                "Circuit breaker OPEN - Jina API unavailable. "
                "Falling back to local provider."
            )

        # Metrics
        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        start_time = time.time()

        try:
            # Call Jina API with retry logic
            embeddings = self._call_api(
                texts=texts, task=self._task  # retrieval.passage
            )

            # Validate dimensions
            for i, emb in enumerate(embeddings):
                if len(emb) != self._dims:
                    raise RuntimeError(
                        f"Jina returned {len(emb)}-D vector at index {i}, "
                        f"expected {self._dims}-D"
                    )

            # Record success
            self._circuit_breaker.record_success()
            latency_ms = (time.time() - start_time) * 1000
            embedding_request_total.labels(
                model_id=self._model_id, operation="documents"
            ).inc()
            embedding_latency_ms.labels(
                model_id=self._model_id, operation="documents"
            ).observe(latency_ms)

            return embeddings

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            embedding_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to embed documents with Jina: {e}")
            raise RuntimeError(f"Jina document embedding failed: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Uses retrieval.query task for query-specific optimization.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query text")

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            raise RuntimeError(
                "Circuit breaker OPEN - Jina API unavailable. "
                "Falling back to local provider."
            )

        # Metrics
        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        start_time = time.time()

        try:
            # Call Jina API with query task
            embeddings = self._call_api(
                texts=[text], task="retrieval.query"  # Query-specific task
            )

            embedding = embeddings[0]

            # Validate dimensions
            if len(embedding) != self._dims:
                raise RuntimeError(
                    f"Jina returned {len(embedding)}-D vector, expected {self._dims}-D"
                )

            # Record success
            self._circuit_breaker.record_success()
            latency_ms = (time.time() - start_time) * 1000
            embedding_request_total.labels(
                model_id=self._model_id, operation="query"
            ).inc()
            embedding_latency_ms.labels(
                model_id=self._model_id, operation="query"
            ).observe(latency_ms)

            return embedding

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            embedding_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to embed query with Jina: {e}")
            raise RuntimeError(f"Jina query embedding failed: {e}")

    def _call_api(self, texts: List[str], task: str) -> List[List[float]]:
        """
        Call Jina API with exponential backoff.

        Args:
            texts: Texts to embed
            task: Task type (retrieval.passage or retrieval.query)

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If all retries fail
        """
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    self.API_URL,
                    json={
                        "model": self._model_id,
                        "task": task,
                        "dimensions": self._dims,
                        "input": texts,
                    },
                )

                # Handle HTTP errors
                if response.status_code == 429:
                    # Rate limit - use exponential backoff
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Jina API rate limit (429), retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()

                # Parse response
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]

                logger.debug(
                    f"Jina API success: {len(embeddings)} embeddings, task={task}"
                )

                return embeddings

            except httpx.HTTPStatusError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Jina API error {e.response.status_code}, "
                        f"retrying in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Jina API HTTP error: {e.response.status_code}")

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Jina API error: {e}, retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Jina API call failed: {e}")

        raise RuntimeError(f"Jina API failed after {max_retries} retries")

    def validate_dimensions(self, expected_dims: int) -> bool:
        """Validate dimensions match expected."""
        return self.dims == expected_dims

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
