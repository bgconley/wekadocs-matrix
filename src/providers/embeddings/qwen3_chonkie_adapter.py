# =============================================================================
# @status: ACTIVE
# @called-by: semantic_chunker.py
# =============================================================================
"""
Adapter to use Qwen3-Embedding-4B gateway with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges the Qwen3 FastAPI gateway (Triton-backed) to that interface.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx
import numpy as np

from src.clients.qwen3_embedding_client import Qwen3EmbeddingClient
from src.providers.embeddings.base_chonkie_adapter import BaseChonkieAdapter

log = logging.getLogger(__name__)


class Qwen3ChonkieAdapter(BaseChonkieAdapter):
    """
    Bridges Qwen3-Embedding-4B gateway to Chonkie's embedding interface.

    Uses dense embeddings only -- chonkie needs these for similarity computation
    to detect semantic boundaries. The gateway handles tokenization, inference,
    pooling, truncation, and normalization.
    """

    _adapter_name: str = "Qwen3"
    _batch_size_env_var: str = "QWEN3_MAX_BATCH_SIZE"

    def __init__(
        self,
        service_url: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        timeout: float = 60.0,
        dimensions: Optional[int] = None,
    ) -> None:
        resolved_url = (
            service_url or os.getenv("QWEN3_EMBED_URL") or "http://10.25.0.50:8101"
        )
        resolved_model = os.getenv("QWEN3_EMBED_MODEL") or model_name
        resolved_timeout = float(os.getenv("QWEN3_EMBED_TIMEOUT_SECONDS", str(timeout)))
        resolved_dim = int(os.getenv("QWEN3_EMBED_DIM", str(dimensions or 1024)))
        api_key = os.getenv("QWEN3_EMBED_API_KEY")
        super().__init__(
            service_url=resolved_url,
            model_name=resolved_model,
            timeout=resolved_timeout,
            dimension=resolved_dim,
            api_key=api_key,
        )

    def _create_client(self) -> Qwen3EmbeddingClient:
        return Qwen3EmbeddingClient(
            base_url=self._service_url,
            model=self._model_name,
            api_key=self._api_key,
            timeout=self._timeout,
        )

    def _embed_single_batch(self, texts: List[str]) -> List[np.ndarray]:
        client = self._get_client()
        response = client.embed(texts)
        raw_embeddings = response.get("embeddings", [])
        return [np.array(vec, dtype=np.float32) for vec in raw_embeddings]

    @classmethod
    def _health_check(cls) -> bool:
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

    def __repr__(self) -> str:
        return f"Qwen3ChonkieAdapter(url={self._service_url}, dim={self._dimension})"


__all__ = ["Qwen3ChonkieAdapter"]
