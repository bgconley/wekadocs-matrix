from types import SimpleNamespace

import pytest

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever


class _ColbertEmbedder:
    provider_name = "dummy-colbert"

    def embed_query(self, query: str):
        return [0.1, 0.2]

    def embed_query_all(self, query: str) -> QueryEmbeddingBundle:
        return QueryEmbeddingBundle(dense=[0.1, 0.2], multivector=None)


class _SparseOnlyEmbedder:
    provider_name = "dummy-sparse"

    def embed_query(self, query: str):
        return [0.1, 0.2]


def test_colbert_requires_query_api(monkeypatch):
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.get_settings",
        lambda: SimpleNamespace(env="production"),
    )
    with pytest.raises(ValueError):
        QdrantMultiVectorRetriever(
            qdrant_client=None,
            embedder=_ColbertEmbedder(),
            collection_name="chunks",
            schema_supports_colbert=True,
            use_query_api=False,
        )


def test_sparse_requires_embed_sparse(monkeypatch):
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.get_settings",
        lambda: SimpleNamespace(env="production"),
    )
    with pytest.raises(ValueError):
        QdrantMultiVectorRetriever(
            qdrant_client=None,
            embedder=_SparseOnlyEmbedder(),
            collection_name="chunks",
            schema_supports_sparse=True,
            use_query_api=True,
        )
