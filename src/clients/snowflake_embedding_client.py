from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class SnowflakeEmbeddingClient:
    """
    Lightweight client for the Snowflake Arctic embeddings service.

    The service is OpenAI-compatible (/v1/embeddings), so this keeps the
    request/response shape consistent with standard embedding clients.

    Features:
    - Retry with exponential backoff on transient errors (429, 5xx)
    - Configurable timeout and retry parameters
    - Structured logging for observability
    """

    # Retryable HTTP status codes (rate limit + server errors)
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    DEFAULT_MAX_RETRIES = 4
    DEFAULT_MIN_BACKOFF = 0.5
    DEFAULT_MAX_BACKOFF = 8.0

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9010/v1",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_backoff: float = DEFAULT_MIN_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._max_retries = max_retries
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SnowflakeEmbeddingClient":
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
            "Snowflake Arctic API retrying after backoff",
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

                # Check for retryable status codes
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            "Snowflake Arctic API retryable error",
                            extra={
                                "status_code": response.status_code,
                                "attempt": attempt + 1,
                            },
                        )
                        self._sleep_backoff(attempt)
                        continue
                    # Last attempt - raise the error
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response else None

                # Non-retryable client errors
                if status == 400:
                    raise RuntimeError(
                        f"Snowflake Arctic API rejected request (400): {exc.response.text}"
                    ) from exc
                if status in {401, 403}:
                    raise RuntimeError(
                        f"Snowflake Arctic API authentication failed ({status})"
                    ) from exc

                # Retryable errors already handled above via status code check
                raise

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Snowflake Arctic API timeout",
                        extra={"attempt": attempt + 1},
                    )
                    self._sleep_backoff(attempt)
                    continue
                raise RuntimeError(
                    f"Snowflake Arctic API timed out after {self._max_retries} attempts"
                ) from exc

            except Exception as exc:
                last_error = exc
                logger.error(
                    "Snowflake Arctic API unexpected error",
                    extra={"error": str(exc)},
                )
                raise

        raise RuntimeError(
            f"Snowflake Arctic API request failed after {self._max_retries} attempts: {last_error}"
        )

    def embeddings(
        self,
        model: str,
        input: Any,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate embeddings for the given input.

        Args:
            model: Model identifier (e.g., "snowflake-arctic-embed-l-v2.0")
            input: Text or list of texts to embed
            encoding_format: Output format ("float" or "base64")
            dimensions: Optional output dimensions
            user: Optional user identifier for tracking

        Returns:
            OpenAI-compatible embeddings response with "data" array
        """
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user

        return self._post_with_retry("/embeddings", payload)

    def list_models(self) -> dict[str, Any]:
        """
        List available models from the Arctic service.

        Uses retry logic for consistency with embeddings().

        Returns:
            OpenAI-compatible models list response
        """
        return self._get_with_retry("/models")

    def _get_with_retry(self, endpoint: str) -> dict[str, Any]:
        """GET with retry on transient errors (429, 5xx, timeouts)."""
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.get(url, headers=self._headers())

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            "Snowflake Arctic API retryable error (GET)",
                            extra={
                                "status_code": response.status_code,
                                "attempt": attempt + 1,
                                "endpoint": endpoint,
                            },
                        )
                        self._sleep_backoff(attempt)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Snowflake Arctic API timeout (GET)",
                        extra={"attempt": attempt + 1, "endpoint": endpoint},
                    )
                    self._sleep_backoff(attempt)
                    continue
                raise RuntimeError(
                    f"Snowflake Arctic API GET timed out after {self._max_retries} attempts"
                ) from exc

            except Exception as exc:
                last_error = exc
                logger.error(
                    "Snowflake Arctic API unexpected error (GET)",
                    extra={"error": str(exc), "endpoint": endpoint},
                )
                raise

        raise RuntimeError(
            f"Snowflake Arctic API GET failed after {self._max_retries} attempts: {last_error}"
        )


__all__ = ["SnowflakeEmbeddingClient"]
