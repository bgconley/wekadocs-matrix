# =============================================================================
# @status: ACTIVE
# @called-by: providers/embeddings/embedding_service.py
# =============================================================================
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class EmbeddingClientError(RuntimeError):
    """Raised when an embedding HTTP call fails."""


class EmbeddingClient:
    """HTTP client for the unified embedding gateway.

    Supports dense (/v1/embeddings), sparse (/v1/embeddings/sparse),
    and ColBERT (/v1/embeddings/colbert) endpoints. Model routing is
    handled server-side based on the model parameter in the payload.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        if not self._base_url:
            raise RuntimeError(
                "EMBEDDING_BASE_URL environment variable is required. "
                "Set it to the unified embedding gateway URL."
            )
        self._model = os.getenv("EMBEDDING_MODEL_ID", model)
        timeout_value = (
            timeout
            if timeout is not None
            else float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
        )
        self._client = client or httpx.Client(
            base_url=self._base_url, timeout=timeout_value
        )

    def close(self) -> None:
        self._client.close()

    def _handle_error(self, response: httpx.Response) -> None:
        try:
            body = response.text
        except Exception:
            body = "<unavailable>"
        raise EmbeddingClientError(
            f"Embedding service HTTP {response.status_code}: {body}"
        )

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Return dense embeddings using /v1/embeddings."""

        payload: Dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        response = self._client.post("/v1/embeddings", json=payload)
        if response.status_code != 200:
            self._handle_error(response)

        data = response.json()["data"]
        return [[float(x) for x in item["embedding"]] for item in data]

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return sparse representations using /v1/embeddings/sparse."""

        payload: Dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        response = self._client.post("/v1/embeddings/sparse", json=payload)
        if response.status_code != 200:
            self._handle_error(response)

        data = response.json()["data"]
        results = []
        for item in data:
            if "indices" in item and "values" in item:
                # Legacy format: {"indices": [...], "values": [...]}
                results.append(
                    {
                        "indices": [int(i) for i in item["indices"]],
                        "values": [float(v) for v in item["values"]],
                    }
                )
            elif "embedding" in item:
                # Unified gateway format: {"embedding": [{"index": int, "value": float}, ...]}
                pairs = item["embedding"]
                results.append(
                    {
                        "indices": [int(p["index"]) for p in pairs],
                        "values": [float(p["value"]) for p in pairs],
                    }
                )
            else:
                raise ValueError(
                    f"Unexpected sparse embedding format: {list(item.keys())}"
                )
        return results

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """Return ColBERT multi-vectors using /v1/embeddings/colbert."""

        payload: Dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        response = self._client.post("/v1/embeddings/colbert", json=payload)
        if response.status_code != 200:
            self._handle_error(response)

        data = response.json()["data"]
        return [[[float(x) for x in row] for row in item["vectors"]] for item in data]


__all__ = ["EmbeddingClient", "EmbeddingClientError"]
