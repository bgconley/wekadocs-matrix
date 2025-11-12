from types import SimpleNamespace

from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore


def _make_qdrant_store(use_named_vectors: bool) -> QdrantVectorStore:
    store = object.__new__(QdrantVectorStore)
    store.client = SimpleNamespace()
    store.collection_name = "test"
    store.query_vector_name = "content"
    store.use_named_vectors = use_named_vectors
    return store


def _make_engine(store: QdrantVectorStore) -> HybridSearchEngine:
    engine = object.__new__(HybridSearchEngine)
    engine.vector_store = store
    engine.vector_field_weights = {"content": 1.0, "title": 0.5}
    engine.qdrant_query_strategy = "weighted"
    engine._max_vector_field = "content"
    return engine


def test_weighted_strategy_scales_secondary_vectors():
    store = _make_qdrant_store(True)
    engine = _make_engine(store)

    payload = engine._build_query_vector_payload([1.0, 2.0])

    assert payload["content"] == [1.0, 2.0]
    assert payload["title"] == [0.5, 1.0]


def test_max_field_strategy_uses_heaviest_named_vector():
    store = _make_qdrant_store(True)
    engine = _make_engine(store)
    engine.qdrant_query_strategy = "max_field"
    engine._max_vector_field = "title"

    payload = engine._build_query_vector_payload([3.0, 4.0])

    assert payload == {"title": [3.0, 4.0]}


def test_strategy_falls_back_when_named_vectors_disabled():
    store = _make_qdrant_store(False)
    engine = _make_engine(store)

    payload = engine._build_query_vector_payload([5.0, 6.0])

    assert payload == [5.0, 6.0]
