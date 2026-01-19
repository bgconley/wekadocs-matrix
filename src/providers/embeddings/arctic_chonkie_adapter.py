"""
Adapter to use Snowflake Arctic embedding service with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges the local Snowflake Arctic OpenAI-compatible service to that interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import httpx
import numpy as np

from src.clients.snowflake_embedding_client import SnowflakeEmbeddingClient

log = logging.getLogger(__name__)


# Attempt to import chonkie's base class
try:
    from chonkie.embeddings import BaseEmbeddings

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    BaseEmbeddings = object  # Stub for type hints when chonkie not installed
    log.debug("chonkie not installed; ArcticChonkieAdapter will be unavailable")


class ArcticChonkieAdapter(BaseEmbeddings if CHONKIE_AVAILABLE else object):
    """
    Bridges Snowflake Arctic embedding service to Chonkie's embedding interface.

    Uses dense embeddings only - chonkie needs these for similarity computation
    to detect semantic boundaries. This adapter calls the OpenAI-compatible
    /v1/embeddings endpoint exposed by the local service.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        model_name: str = "snowflake-arctic-embed-l-v2.0",
        timeout: float = 60.0,
        dimensions: Optional[int] = None,
    ) -> None:
        self._service_url = (
            service_url
            or os.getenv("CHONKIE_EMBEDDINGS_BASE_URL")
            or "http://127.0.0.1:9010/v1"
        )
        raw_model_name = os.getenv("CHONKIE_EMBEDDINGS_MODEL") or model_name
        normalized_model_name = self._normalize_model_name(raw_model_name)
        if normalized_model_name != raw_model_name:
            log.info(
                "Normalized Snowflake Arctic model name",
                extra={"original": raw_model_name, "normalized": normalized_model_name},
            )
        self._model_name = normalized_model_name
        self._timeout = float(
            os.getenv("CHONKIE_EMBEDDINGS_TIMEOUT_SECONDS", str(timeout))
        )
        self._dimension = int(
            os.getenv("CHONKIE_EMBEDDINGS_DIM", str(dimensions or 1024))
        )
        self._api_key = os.getenv("CHONKIE_EMBEDDINGS_API_KEY")
        self._client: Optional[SnowflakeEmbeddingClient] = None
        self._tokenizer: Optional[Any] = None

        log.info(
            "ArcticChonkieAdapter initialized",
            extra={
                "service_url": self._service_url,
                "model_name": self._model_name,
                "timeout": self._timeout,
                "dimension": self._dimension,
            },
        )

    def _get_client(self) -> SnowflakeEmbeddingClient:
        if self._client is None:
            self._client = SnowflakeEmbeddingClient(
                base_url=self._service_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            candidate = model_name.split("/")[-1]
            if candidate.startswith("snowflake-arctic-embed-"):
                return candidate
        return model_name

    def embed(self, text: str) -> np.ndarray:
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def _embed_single_batch(self, texts: List[str]) -> List[np.ndarray]:
        client = self._get_client()
        response = client.embeddings(
            model=self._model_name,
            input=texts,
            encoding_format="float",
            dimensions=self._dimension,
        )
        items = response.get("data", []) if isinstance(response, dict) else response
        embeddings: List[np.ndarray] = []
        for item in sorted(items, key=lambda x: x.get("index", 0)):
            embeddings.append(np.array(item["embedding"], dtype=np.float32))
        return embeddings

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
                "Snowflake Arctic embedding inputs contained empty strings; sanitized",
                extra={
                    "empty_count": empty_count,
                    "batch_size": len(normalized),
                    "empty_indices_sample": empty_indices[:10],
                },
            )

        try:
            max_batch_size = int(os.getenv("CHONKIE_MAX_BATCH_SIZE", "32"))
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
                    "Snowflake Arctic embedding request failed",
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

        base_url = (
            os.getenv("CHONKIE_EMBEDDINGS_BASE_URL") or "http://127.0.0.1:9010/v1"
        )
        health_base = base_url
        if base_url.endswith("/v1"):
            health_base = base_url[: -len("/v1")]
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{health_base}/healthz")
                if r.status_code == 200:
                    health = r.json()
                    is_healthy = health.get("status") == "ok"
                    if is_healthy:
                        log.debug(
                            "Snowflake Arctic service healthy",
                            extra={"service_url": base_url},
                        )
                    return is_healthy
                return False
        except Exception as exc:
            log.debug(
                "Snowflake Arctic service unavailable",
                extra={"service_url": base_url, "error": str(exc)},
            )
            return False

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"ArcticChonkieAdapter(url={self._service_url}, dim={self._dimension})"

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown / GC


__all__ = ["ArcticChonkieAdapter"]
