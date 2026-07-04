# =============================================================================
# @status: ACTIVE
# @called-by: providers/embeddings/qwen3_triton.py, qwen3_chonkie_adapter.py
# =============================================================================
"""
HTTP client for the Qwen3-Embedding-4B service via custom FastAPI gateway.

The gateway wraps Triton Inference Server and exposes a simple REST API:
- POST /embed  {"texts": [...], "model": "optional"}
- Response:    {"embeddings": [[...1024 floats...]], "model": "Qwen/Qwen3-Embedding-4B"}

The gateway handles tokenization, Triton inference, last-token pooling,
truncation to 1024 dims, and L2 normalization.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Qwen3EmbeddingClient:
    """
    Lightweight client for the Qwen3-Embedding-4B FastAPI gateway.

    Features:
    - Retry with exponential backoff on transient errors (429, 5xx)
    - Configurable timeout and retry parameters
    - Structured logging for observability
    """

    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    DEFAULT_MAX_RETRIES = 4
    DEFAULT_MIN_BACKOFF = 0.5
    DEFAULT_MAX_BACKOFF = 8.0

    def __init__(
        self,
        base_url: str = "http://10.25.0.50:8101",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_backoff: float = DEFAULT_MIN_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or "Qwen/Qwen3-Embedding-4B"
        self.api_key = api_key
        self._max_retries = max_retries
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Qwen3EmbeddingClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff and jitter."""
        base = min(self._max_backoff, self._min_backoff * (2**attempt))
        jitter = random.uniform(0, base / 4.0)
        delay = base + jitter
        logger.warning(
            "Qwen3 embed gateway retrying after backoff",
            extra={
                "attempt": attempt + 1,
                "max_retries": self._max_retries,
                "delay_sec": f"{delay:.2f}",
            },
        )
        time.sleep(delay)

    def _post_with_retry(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST with retry on transient errors (429, 5xx, timeouts)."""
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.post(url, json=payload, headers=self._headers())

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            "Qwen3 embed gateway retryable error",
                            extra={
                                "status_code": response.status_code,
                                "attempt": attempt + 1,
                            },
                        )
                        self._sleep_backoff(attempt)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response else None

                if status == 400:
                    raise RuntimeError(
                        f"Qwen3 embed gateway rejected request (400): {exc.response.text}"
                    ) from exc
                if status in {401, 403}:
                    raise RuntimeError(
                        f"Qwen3 embed gateway authentication failed ({status})"
                    ) from exc
                raise

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Qwen3 embed gateway timeout",
                        extra={"attempt": attempt + 1},
                    )
                    self._sleep_backoff(attempt)
                    continue
                raise RuntimeError(
                    f"Qwen3 embed gateway timed out after {self._max_retries} attempts"
                ) from exc

            except Exception as exc:
                last_error = exc
                logger.error(
                    "Qwen3 embed gateway unexpected error",
                    extra={"error": str(exc)},
                )
                raise

        raise RuntimeError(
            f"Qwen3 embed gateway request failed after {self._max_retries} attempts: {last_error}"
        )

    def embed(self, texts: List[str]) -> dict[str, Any]:
        """
        Generate embeddings via the Qwen3 gateway.

        Args:
            texts: List of texts to embed.

        Returns:
            Gateway response: {"embeddings": [[...]], "model": "..."}
        """
        payload: dict[str, Any] = {"texts": texts, "model": self.model}
        return self._post_with_retry("/embed", payload)


__all__ = ["Qwen3EmbeddingClient"]
