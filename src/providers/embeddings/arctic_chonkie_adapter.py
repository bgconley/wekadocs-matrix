# =============================================================================
# @status: ACTIVE
# @called-by: semantic_chunker.py
# =============================================================================
"""
Adapter to use Snowflake Arctic embedding service with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges the local Snowflake Arctic OpenAI-compatible service to that interface.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx
import numpy as np

from src.clients.snowflake_embedding_client import SnowflakeEmbeddingClient
from src.providers.embeddings.base_chonkie_adapter import BaseChonkieAdapter

log = logging.getLogger(__name__)


class ArcticChonkieAdapter(BaseChonkieAdapter):
    """
    Bridges Snowflake Arctic embedding service to Chonkie's embedding interface.

    Uses dense embeddings only - chonkie needs these for similarity computation
    to detect semantic boundaries. This adapter calls the OpenAI-compatible
    /v1/embeddings endpoint exposed by the local service.
    """

    _adapter_name: str = "Snowflake Arctic"
    _batch_size_env_var: str = "CHONKIE_MAX_BATCH_SIZE"

    def __init__(
        self,
        service_url: Optional[str] = None,
        model_name: str = "snowflake-arctic-embed-l-v2.0",
        timeout: float = 60.0,
        dimensions: Optional[int] = None,
    ) -> None:
        resolved_url = (
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
        resolved_timeout = float(
            os.getenv("CHONKIE_EMBEDDINGS_TIMEOUT_SECONDS", str(timeout))
        )
        resolved_dim = int(os.getenv("CHONKIE_EMBEDDINGS_DIM", str(dimensions or 1024)))
        api_key = os.getenv("CHONKIE_EMBEDDINGS_API_KEY")
        super().__init__(
            service_url=resolved_url,
            model_name=normalized_model_name,
            timeout=resolved_timeout,
            dimension=resolved_dim,
            api_key=api_key,
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            candidate = model_name.split("/")[-1]
            if candidate.startswith("snowflake-arctic-embed-"):
                return candidate
        return model_name

    def _create_client(self) -> SnowflakeEmbeddingClient:
        return SnowflakeEmbeddingClient(
            base_url=self._service_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

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

    @classmethod
    def _health_check(cls) -> bool:
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

    def __repr__(self) -> str:
        return f"ArcticChonkieAdapter(url={self._service_url}, dim={self._dimension})"


__all__ = ["ArcticChonkieAdapter"]
