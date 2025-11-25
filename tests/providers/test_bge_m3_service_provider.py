from __future__ import annotations

from typing import List

import pytest

from src.providers.embeddings.bge_m3_service import BGEM3ServiceProvider
from src.providers.embeddings.contracts import (
    DocumentEmbeddingBundle,
    QueryEmbeddingBundle,
)
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings


class _FakeEmbeddingClient:
    def __init__(self, dims: int) -> None:
        self._dims = dims
        self.closed = False

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        return [[float(idx) for idx in range(self._dims)] for _ in texts]

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        return [{"indices": [0, 1], "values": [0.5, 0.25]} for _ in texts]

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        return [[[0.1, 0.2], [0.3, 0.4]] for _ in texts]

    def close(self) -> None:
        self.closed = True


def _sample_settings(dims: int = 1024) -> EmbeddingSettings:
    return EmbeddingSettings(
        profile="bge_m3",
        provider="bge-m3-service",
        model_id="bge-m3",
        version="bge-m3",
        dims=dims,
        similarity="cosine",
        task="symmetric",
        tokenizer_backend="hf",
        tokenizer_model_id="BAAI/bge-m3",
        service_url="http://127.0.0.1:9000",
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=True,
            supports_colbert=True,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
    )


def test_embed_documents_and_query_with_fake_client():
    fake_client = _FakeEmbeddingClient(dims=4)
    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=fake_client)

    docs = provider.embed_documents(["a", "b"])
    assert len(docs) == 2
    assert len(docs[0]) == 4

    query = provider.embed_query("single")
    assert len(query) == 4
    assert provider.capabilities.supports_sparse is True
    assert provider.supports_sparse is True


def test_sparse_and_colbert_helpers():
    fake_client = _FakeEmbeddingClient(dims=4)
    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=fake_client)

    sparse = provider.embed_sparse(["a"])
    assert sparse[0]["indices"] == [0, 1]

    colbert = provider.embed_colbert(["a"])
    assert len(colbert[0]) == 2


def test_embed_documents_all_returns_bundles():
    fake_client = _FakeEmbeddingClient(dims=4)
    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=fake_client)

    bundles = provider.embed_documents_all(["a", "b"])
    assert all(isinstance(b, DocumentEmbeddingBundle) for b in bundles)
    first = bundles[0]
    assert len(first.dense) == 4
    assert first.sparse is not None
    assert first.sparse.indices == [0, 1]
    assert first.multivector is not None
    assert first.multivector.vectors[0] == [0.1, 0.2]


def test_embed_query_all_returns_bundle():
    fake_client = _FakeEmbeddingClient(dims=4)
    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=fake_client)

    bundle = provider.embed_query_all("hello world")
    assert isinstance(bundle, QueryEmbeddingBundle)
    assert len(bundle.dense) == 4
    assert bundle.sparse is not None
    assert bundle.multivector is not None


def test_close_propagates_to_client():
    fake_client = _FakeEmbeddingClient(dims=4)
    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=fake_client)

    provider.close()
    assert fake_client.closed is True


def test_dimension_validation_raises_on_mismatch():
    class _BadClient(_FakeEmbeddingClient):
        def embed_dense(self, texts: List[str]) -> List[List[float]]:
            return [[0.0, 0.1] for _ in texts]

    provider = BGEM3ServiceProvider(settings=_sample_settings(4), client=_BadClient(2))

    with pytest.raises(RuntimeError, match="expected 4"):
        provider.embed_documents(["oops"])


def test_embed_documents_rejects_empty_batch():
    provider = BGEM3ServiceProvider(
        settings=_sample_settings(4), client=_FakeEmbeddingClient(4)
    )

    with pytest.raises(ValueError, match="empty batch"):
        provider.embed_documents([])
