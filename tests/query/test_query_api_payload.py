from src.providers.embeddings.contracts import (
    MultiVectorEmbedding,
    QueryEmbeddingBundle,
    SparseEmbedding,
)
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever


class DummySparseEmbedder:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]

    def embed_sparse(self, queries):
        # Return list with a dict matching SparseEmbedding structure
        return [{"indices": [1, 2], "values": [0.5, 0.6]} for _ in queries]


class DummyColbertEmbedder:
    def embed_query_all(self, query: str):
        return QueryEmbeddingBundle(
            dense=[0.1, 0.2, 0.3],
            sparse=None,
            multivector=MultiVectorEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4]]),
        )


def test_prefetch_payload_dense_and_sparse():
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=None,  # not used for payload build
        embedder=DummySparseEmbedder(),
        collection_name="chunks_multi",
        field_weights={"content": 1.0, "text-sparse": 0.5},
        embedding_settings=None,
        query_api_dense_limit=10,
        query_api_sparse_limit=10,
        schema_supports_sparse=True,
    )

    bundle = QueryEmbeddingBundle(
        dense=[0.1, 0.2, 0.3],
        sparse=SparseEmbedding(indices=[1, 2], values=[0.5, 0.6]),
        multivector=None,
    )

    entries = retriever._build_prefetch_entries(bundle, qdrant_filter=None, top_k=3)
    assert entries, "Prefetch entries should not be empty"

    dense_entry = next(e for e in entries if e.using == "content")
    assert isinstance(dense_entry.query, list)
    assert dense_entry.limit == 10

    sparse_entry = next(e for e in entries if e.using == "text-sparse")
    from qdrant_client.http.models import SparseVector as HttpSparseVector

    assert isinstance(sparse_entry.query, HttpSparseVector)
    assert sparse_entry.query.indices == [1, 2]
    assert sparse_entry.query.values == [0.5, 0.6]


def test_query_payload_colbert_and_dense():
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=None,  # not used for payload build
        embedder=DummyColbertEmbedder(),
        collection_name="chunks_multi",
        field_weights={"content": 1.0},
        embedding_settings=None,
        schema_supports_colbert=True,
        query_api_dense_limit=10,
    )

    bundle = retriever._build_query_bundle("test query")

    payload, using_name = retriever._build_query_api_query(bundle)
    assert using_name == "late-interaction"
    assert isinstance(payload, list)
    assert payload == [[0.1, 0.2], [0.3, 0.4]]
