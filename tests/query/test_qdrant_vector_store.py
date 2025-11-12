from src.query.hybrid_search import QdrantVectorStore


class FakeQdrantClient:
    def __init__(self):
        self.last_query_vector = None

    def search(self, **kwargs):
        self.last_query_vector = kwargs.get("query_vector")
        return []


def test_qdrant_search_uses_named_vector_when_enabled():
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        client,
        "chunks_multi",
        query_vector_name="content",
        use_named_vectors=True,
    )

    store.search([0.1, 0.2, 0.3], k=5)

    assert client.last_query_vector == {"content": [0.1, 0.2, 0.3]}


def test_qdrant_search_uses_plain_vector_when_disabled():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client, "chunks_multi")

    store.search([0.4, 0.5], k=3)

    assert client.last_query_vector == [0.4, 0.5]
