from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class EmbeddingClientError(RuntimeError):
    """Raised when an embedding HTTP call fails."""


class EmbeddingClient:
    """Python client for the BGE-M3 embedding service.

    This implements the interfaces described in the canonical app spec
    (embedding_client_python): embed_dense, embed_sparse, embed_colbert.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "BAAI/bge-m3",
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url or os.getenv(
            "EMBEDDING_BASE_URL", "http://127.0.0.1:9000"
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
        return [
            {
                "indices": [int(i) for i in item["indices"]],
                "values": [float(v) for v in item["values"]],
            }
            for item in data
        ]

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
