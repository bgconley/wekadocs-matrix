"""
Adapter to use Qwen3-Embedding-4B gateway with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges the Qwen3 FastAPI gateway (Triton-backed) to that interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import httpx
import numpy as np

from src.clients.qwen3_embedding_client import Qwen3EmbeddingClient

log = logging.getLogger(__name__)


try:
    from chonkie.embeddings import BaseEmbeddings

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    BaseEmbeddings = object
    log.debug("chonkie not installed; Qwen3ChonkieAdapter will be unavailable")


class Qwen3ChonkieAdapter(BaseEmbeddings if CHONKIE_AVAILABLE else object):
    """
    Bridges Qwen3-Embedding-4B gateway to Chonkie's embedding interface.

    Uses dense embeddings only -- chonkie needs these for similarity computation
    to detect semantic boundaries. The gateway handles tokenization, inference,
    pooling, truncation, and normalization.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        timeout: float = 60.0,
        dimensions: Optional[int] = None,
    ) -> None:
        self._service_url = (
            service_url or os.getenv("QWEN3_EMBED_URL") or "http://10.25.0.50:8101"
        )
        self._model_name = os.getenv("QWEN3_EMBED_MODEL") or model_name
        self._timeout = float(os.getenv("QWEN3_EMBED_TIMEOUT_SECONDS", str(timeout)))
        self._dimension = int(os.getenv("QWEN3_EMBED_DIM", str(dimensions or 1024)))
        self._api_key = os.getenv("QWEN3_EMBED_API_KEY")
        self._client: Optional[Qwen3EmbeddingClient] = None
        self._tokenizer: Optional[Any] = None

        log.info(
            "Qwen3ChonkieAdapter initialized",
            extra={
                "service_url": self._service_url,
                "model_name": self._model_name,
                "timeout": self._timeout,
                "dimension": self._dimension,
            },
        )

    def _get_client(self) -> Qwen3EmbeddingClient:
        if self._client is None:
            self._client = Qwen3EmbeddingClient(
                base_url=self._service_url,
                model=self._model_name,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def _embed_single_batch(self, texts: List[str]) -> List[np.ndarray]:
        client = self._get_client()
        response = client.embed(texts)
        raw_embeddings = response.get("embeddings", [])
        return [np.array(vec, dtype=np.float32) for vec in raw_embeddings]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        # Sanitize inputs (same pattern as ArcticChonkieAdapter)
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
                value = " "
            normalized.append(value)

        if empty_count:
            log.info(
                "Qwen3 embedding inputs contained empty strings; sanitized",
                extra={
                    "empty_count": empty_count,
                    "batch_size": len(normalized),
                    "empty_indices_sample": empty_indices[:10],
                },
            )

        try:
            max_batch_size = int(os.getenv("QWEN3_MAX_BATCH_SIZE", "32"))
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
                    "Qwen3 embedding request failed",
                    extra={"error": str(exc), "batch_size": len(batch)},
                )
                raise
        return all_embeddings

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

        base_url = os.getenv("QWEN3_EMBED_URL") or "http://10.25.0.50:8101"
        try:
            with httpx.Client(timeout=5.0) as client:
                # Probe the gateway with a minimal embed request
                r = client.post(
                    f"{base_url}/embed",
                    json={
                        "texts": ["health check"],
                        "model": "Qwen/Qwen3-Embedding-4B",
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    is_healthy = "embeddings" in data and len(data["embeddings"]) == 1
                    if is_healthy:
                        log.debug(
                            "Qwen3 embed gateway healthy",
                            extra={"service_url": base_url},
                        )
                    return is_healthy
                return False
        except Exception as exc:
            log.debug(
                "Qwen3 embed gateway unavailable",
                extra={"service_url": base_url, "error": str(exc)},
            )
            return False

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"Qwen3ChonkieAdapter(url={self._service_url}, dim={self._dimension})"

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["Qwen3ChonkieAdapter"]
