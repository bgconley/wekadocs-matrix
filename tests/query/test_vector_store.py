from src.query.hybrid_search import Neo4jVectorStore


class DummySession:
    def __init__(self, capture):
        self.capture = capture

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.capture["query"] = query
        self.capture["params"] = params
        return []


class DummyDriver:
    def __init__(self, capture):
        self.capture = capture

    def session(self):
        return DummySession(self.capture)


def test_neo4j_vector_store_parameterizes_filters():
    capture = {}
    store = Neo4jVectorStore(DummyDriver(capture), "section_embeddings")

    filters = {"doc_tag": "prod.docs", "tenant": "weka"}

    results = store.search([0.1, 0.2], 5, filters=filters)

    assert results == []
    query = capture["query"]
    params = capture["params"]

    assert "doc_tag" in query
    assert "tenant" in query
    # Ensure literal values were not interpolated directly
    assert "prod.docs" not in query
    assert "weka" not in query
    assert any(name.startswith("filter_doc_tag") for name in params.keys())
    assert any(name.startswith("filter_tenant") for name in params.keys())


def test_neo4j_vector_store_uses_embedding_version_from_config():
    capture = {}
    store = Neo4jVectorStore(DummyDriver(capture), "section_embeddings")

    store.search([0.3, 0.4], 3)

    params = capture["params"]
    from src.shared.config import get_config

    assert params["embedding_version"] == get_config().embedding.version
