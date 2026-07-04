# =============================================================================
# @status: ACTIVE
# @called-by: arctic_chonkie_adapter.py, qwen3_chonkie_adapter.py
# =============================================================================
"""
Base class for Chonkie embedding adapters.

Arctic and Qwen3 adapters share ~85% of their code. This base class extracts
all common logic: input sanitization, batch splitting, token counting, and
lifecycle management. Subclasses only implement client creation, batch
embedding, and health checks.
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Any, List, Optional

import httpx
import numpy as np

log = logging.getLogger(__name__)


# Attempt to import chonkie's base class
try:
    from chonkie.embeddings import BaseEmbeddings

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    BaseEmbeddings = object
    log.debug("chonkie not installed; Chonkie adapters will be unavailable")


class BaseChonkieAdapter(BaseEmbeddings if CHONKIE_AVAILABLE else object):
    """
    Base class for embedding adapters that bridge services to Chonkie's interface.

    Subclasses MUST implement:
        _create_client() -> object
        _embed_single_batch(texts: List[str]) -> List[np.ndarray]
        _health_check() -> bool (classmethod)

    Subclasses MUST set class attributes:
        _adapter_name: str  -- used in log messages (e.g. "Snowflake Arctic", "Qwen3")
        _batch_size_env_var: str  -- env var for max batch size (e.g. "CHONKIE_MAX_BATCH_SIZE")
    """

    # Override in subclasses
    _adapter_name: str = "Unknown"
    _batch_size_env_var: str = "CHONKIE_MAX_BATCH_SIZE"

    def __init__(
        self,
        service_url: str,
        model_name: str,
        timeout: float,
        dimension: int,
        api_key: Optional[str] = None,
    ) -> None:
        self._service_url = service_url
        self._model_name = model_name
        self._timeout = timeout
        self._dimension = dimension
        self._api_key = api_key
        self._client: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        log.info(
            f"{self.__class__.__name__} initialized",
            extra={
                "service_url": self._service_url,
                "model_name": self._model_name,
                "timeout": self._timeout,
                "dimension": self._dimension,
            },
        )

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the embedding client."""
        ...

    @abstractmethod
    def _embed_single_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a single batch of texts using the specific service client."""
        ...

    @classmethod
    @abstractmethod
    def _health_check(cls) -> bool:
        """Perform a service-specific health check. Return True if healthy."""
        ...

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        normalized: List[str] = []
        empty_count = 0
        empty_indices: List[int] = []
        for idx, item in enumerate(texts):
            value = None
            if isinstance(item, str):
                value = item
            elif hasattr(item, "text"):
                value = getattr(item, "text")
            elif isinstance(item, bytes):
                try:
                    value = item.decode("utf-8", errors="ignore")
                except Exception:
                    value = str(item)
            else:
                value = str(item)

            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)

            if value.strip() == "":
                empty_count += 1
                empty_indices.append(idx)
                value = " "  # Avoid 400 for empty inputs; keep shape intact.
            normalized.append(value)

        if empty_count:
            log.info(
                f"{self._adapter_name} embedding inputs contained empty strings; sanitized",
                extra={
                    "empty_count": empty_count,
                    "batch_size": len(normalized),
                    "empty_indices_sample": empty_indices[:10],
                },
            )

        try:
            max_batch_size = int(os.getenv(self._batch_size_env_var, "32"))
        except Exception:
            max_batch_size = 32
        if max_batch_size < 1:
            max_batch_size = 32

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(normalized), max_batch_size):
            batch = normalized[i : i + max_batch_size]
            try:
                all_embeddings.extend(self._embed_single_batch(batch))
            except httpx.HTTPError as exc:
                log.error(
                    f"{self._adapter_name} embedding request failed",
                    extra={"error": str(exc), "batch_size": len(batch)},
                )
                raise
        return all_embeddings

    # Alias for clarity
    embed_dense = embed_batch

    def count_tokens(self, text: str) -> int:
        return self._count_tokens_model(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        return [self.count_tokens(t) for t in texts]

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from src.providers.tokenizer_service import TokenizerService

            self._tokenizer = TokenizerService()
        return self._tokenizer

    def _count_tokens_model(self, text: str) -> int:
        try:
            tokenizer_service = self._get_tokenizer()
            backend = getattr(tokenizer_service, "backend", None)
            tok = getattr(backend, "tokenizer", None) if backend else None
            if tok is not None and hasattr(tok, "encode"):
                try:
                    return len(tok.encode(text, add_special_tokens=False))
                except TypeError:
                    return len(tok.encode(text))
        except Exception:
            pass

        # Fallback heuristic: max(word_count, approx_bpe)
        word_count = len(text.split())
        approx_bpe = max(1, len(text) // 4)
        return max(word_count, approx_bpe)

    def get_tokenizer(self) -> Any:
        tokenizer_service = self._get_tokenizer()
        backend = getattr(tokenizer_service, "backend", None)
        if backend and hasattr(backend, "tokenizer"):
            return backend.tokenizer
        log.warning(
            "Could not extract HuggingFace tokenizer from TokenizerService; "
            "Chonkie may not work correctly"
        )
        return tokenizer_service

    @classmethod
    def is_available(cls) -> bool:
        if not CHONKIE_AVAILABLE:
            log.debug("Chonkie not available - CHONKIE_AVAILABLE=False")
            return False
        return cls._health_check()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown / GC


__all__ = ["BaseChonkieAdapter", "CHONKIE_AVAILABLE"]
