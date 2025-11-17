from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple, Type

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.embeddings.contracts import (
    DocumentEmbeddingBundle,
    MultiVectorEmbedding,
    QueryEmbeddingBundle,
    SparseEmbedding,
)
from src.providers.settings import EmbeddingSettings as ProviderEmbeddingSettings

logger = logging.getLogger(__name__)

_EMBEDDING_CLIENT_CACHE: Optional[Tuple[Type[object], Optional[Type[Exception]]]] = None


def _to_sparse_embedding(entry: Optional[dict]) -> Optional[SparseEmbedding]:
    if not entry:
        return None
    indices = entry.get("indices")
    values = entry.get("values")
    if not indices or not values:
        return None
    return SparseEmbedding(
        indices=[int(i) for i in indices], values=[float(v) for v in values]
    )


def _to_multivector(
    entry: Optional[Sequence[Sequence[float]]],
) -> Optional[MultiVectorEmbedding]:
    if not entry:
        return None
    vectors = [[float(value) for value in vector] for vector in entry]
    if not vectors:
        return None
    return MultiVectorEmbedding(vectors=vectors)


def _load_embedding_client_symbols() -> Tuple[Type[object], Optional[Type[Exception]]]:
    """
    Import the EmbeddingClient from the external bge-m3-custom repository.

    Returns:
        Tuple of (EmbeddingClient class, EmbeddingClientError class or None)
    """

    global _EMBEDDING_CLIENT_CACHE
    if _EMBEDDING_CLIENT_CACHE is not None:
        return _EMBEDDING_CLIENT_CACHE

    from src.clients.embedding_client import EmbeddingClient, EmbeddingClientError

    embedding_client_cls = EmbeddingClient
    embedding_client_error = EmbeddingClientError
    _EMBEDDING_CLIENT_CACHE = (embedding_client_cls, embedding_client_error)
    return _EMBEDDING_CLIENT_CACHE


class BGEM3ServiceProvider(EmbeddingProvider):
    """EmbeddingProvider backed by the canonical bge-m3-custom HTTP service."""

    def __init__(
        self,
        settings: ProviderEmbeddingSettings,
        *,
        client: Optional[object] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if settings is None:
            raise ValueError(
                "Embedding settings are required for BGEM3ServiceProvider."
            )

        self._settings = settings
        self._dims = settings.dims
        self._model_id = settings.model_id
        self._provider_name = settings.provider or "bge-m3-service"
        self._capabilities = settings.capabilities

        embedding_client_cls, client_error_cls = _load_embedding_client_symbols()
        self._client_error_cls = client_error_cls

        if client is not None:
            self._client = client
        else:
            client_kwargs = {}
            resolved_url = (
                base_url or settings.service_url or os.getenv("BGE_M3_API_URL")
            )
            if resolved_url:
                client_kwargs["base_url"] = resolved_url
            if self._model_id:
                client_kwargs["model"] = self._model_id
            if timeout is not None:
                client_kwargs["timeout"] = timeout

            try:
                self._client = embedding_client_cls(**client_kwargs)
            except Exception as exc:  # pragma: no cover - defensive rethrow
                raise RuntimeError(
                    f"Failed to initialize BGE-M3 EmbeddingClient: {exc}"
                ) from exc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def supports_sparse(self) -> bool:
        return bool(getattr(self._capabilities, "supports_sparse", False))

    @property
    def supports_colbert(self) -> bool:
        return bool(getattr(self._capabilities, "supports_colbert", False))

    def close(self) -> None:
        """Close the underlying HTTP client."""
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty batch of documents.")
        vectors = self._client.embed_dense(texts)
        self._validate_dimensions(vectors)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Query text must be non-empty.")
        return self.embed_documents([text])[0]

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        """Return sparse representations using the sparse endpoint."""
        if not texts:
            raise ValueError("Cannot embed an empty batch of documents.")
        return self._client.embed_sparse(texts)

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """Return ColBERT multi-vectors for downstream consumers."""
        if not texts:
            raise ValueError("Cannot embed an empty batch of documents.")
        return self._client.embed_colbert(texts)

    def embed_documents_all(self, texts: List[str]) -> List[DocumentEmbeddingBundle]:
        """Return dense + sparse + multivector bundles for each document."""
        dense_vectors = self.embed_documents(texts)
        sparse_batches = (
            self.embed_sparse(texts)
            if self.supports_sparse
            else [None] * len(dense_vectors)
        )
        colbert_batches = (
            self.embed_colbert(texts)
            if self.supports_colbert
            else [None] * len(dense_vectors)
        )

        bundles: List[DocumentEmbeddingBundle] = []
        for idx, dense in enumerate(dense_vectors):
            sparse_entry = sparse_batches[idx] if idx < len(sparse_batches) else None
            colbert_entry = colbert_batches[idx] if idx < len(colbert_batches) else None
            bundles.append(
                DocumentEmbeddingBundle(
                    dense=list(dense),
                    sparse=_to_sparse_embedding(sparse_entry),
                    multivector=_to_multivector(colbert_entry),
                )
            )
        return bundles

    def embed_query_all(self, text: str) -> QueryEmbeddingBundle:
        """Return dense + sparse + multivector bundle for a query."""
        if not text:
            raise ValueError("Query text must be non-empty.")
        dense = self.embed_query(text)
        sparse_entry = self.embed_sparse([text])[0] if self.supports_sparse else None
        colbert_entry = self.embed_colbert([text])[0] if self.supports_colbert else None
        return QueryEmbeddingBundle(
            dense=list(dense),
            sparse=_to_sparse_embedding(sparse_entry),
            multivector=_to_multivector(colbert_entry),
        )

    def _validate_dimensions(self, vectors: Sequence[Sequence[float]]) -> None:
        if self._dims is None:
            return
        for vector in vectors:
            if len(vector) != self._dims:
                raise RuntimeError(
                    f"BGE-M3 service returned {len(vector)}-D vector, expected {self._dims}."
                )
