import types

import pytest


def _make_dummy_driver():
    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, *args, **kwargs):
            class R:
                def single(self):
                    return {"id": "dummy"}

            return R()

    class DummyDriver:
        def session(self):
            return DummySession()

        def close(self):
            pass

    return DummyDriver()


class DummyQdrantCollections:
    def __init__(self, names=None):
        self.collections = [types.SimpleNamespace(name=n) for n in (names or [])]


class DummyQdrant:
    def __init__(self, *args, **kwargs):
        self._collections = DummyQdrantCollections([])

    # Compatibility methods expected by GraphBuilder
    def get_collections(self):
        return self._collections

    def create_collection(self, *args, **kwargs):
        return None

    def update_collection(self, *args, **kwargs):
        return None

    def create_payload_index(self, *args, **kwargs):
        return None

    def delete_compat(self, *args, **kwargs):
        return None

    def upsert_validated(self, *args, **kwargs):
        return None


class FakeProvider:
    def __init__(self, model_id: str, dims: int, provider_name: str = "test-provider"):
        self.model_id = model_id
        self.dims = dims
        self.provider_name = provider_name
        self.task = "retrieval.passage"

    def embed_documents(self, texts):
        # Return a deterministic vector per text
        return [[0.0] * self.dims for _ in texts]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # Stub Neo4j driver
    import neo4j

    monkeypatch.setattr(
        neo4j.GraphDatabase, "driver", lambda *a, **k: _make_dummy_driver()
    )
    yield


def test_per_call_overrides_do_not_mutate_global(monkeypatch):
    # Import late to ensure fixture monkeypatches are applied
    import src.ingestion.build_graph as build_graph

    # Stub Qdrant client to avoid network
    import src.shared.connections as connections
    from src.ingestion.api import ingest_document
    from src.shared.config import get_config

    monkeypatch.setattr(connections, "CompatQdrantClient", DummyQdrant)
    # Bypass schema version checks in tests
    monkeypatch.setattr(build_graph, "ensure_schema_version", lambda *a, **k: None)

    # Monkeypatch GraphBuilder.upsert_document to only initialize the embedder
    # (this exercises ProviderFactory with explicit params without touching DB)
    def fake_upsert(self, document, sections, entities, mentions):
        from src.providers.factory import ProviderFactory

        if not hasattr(self, "embedder") or self.embedder is None:
            self.embedder = ProviderFactory.create_embedding_provider(
                provider=self.config.embedding.provider,
                model=self.config.embedding.embedding_model,
                dims=self.config.embedding.dims,
                task=self.config.embedding.task,
            )
        return {"mock": True}

    monkeypatch.setattr(build_graph.GraphBuilder, "upsert_document", fake_upsert)

    # Capture provider factory invocations
    calls = []

    def fake_create_embedding_provider(
        provider=None, model=None, dims=None, task=None, **kwargs
    ):
        calls.append({"provider": provider, "model": model, "dims": dims, "task": task})
        # Return a provider honoring requested dims (fallback to 128)
        return FakeProvider(model_id=model or "default-model", dims=int(dims or 128))

    import src.providers.factory as factory

    monkeypatch.setattr(
        factory.ProviderFactory,
        "create_embedding_provider",
        staticmethod(fake_create_embedding_provider),
    )

    # Snapshot original global config values
    global_config = get_config()
    orig_model = global_config.embedding.embedding_model
    orig_version = global_config.embedding.version

    # Minimal markdown document
    md = "# Title\n\nHello world."

    # First ingest: no overrides
    ingest_document("file:///tmp/doc1.md", md)
    assert calls, "Factory should have been called"
    first = calls[-1]
    assert first["model"] == orig_model

    # Second ingest: with overrides
    ingest_document(
        "file:///tmp/doc2.md",
        md,
        embedding_model="unit-test-model",
        embedding_version="unit-test-version",
    )
    second = calls[-1]
    assert second["model"] == "unit-test-model"

    # Ensure global config not mutated by overrides
    post_model = get_config().embedding.embedding_model
    post_version = get_config().embedding.version
    assert post_model == orig_model
    assert post_version == orig_version


def test_ingest_document_accepts_html(monkeypatch):
    import src.ingestion.build_graph as build_graph
    import src.shared.connections as connections
    from src.ingestion.api import ingest_document

    # Stub external dependencies
    monkeypatch.setattr(connections, "CompatQdrantClient", DummyQdrant)
    monkeypatch.setattr(build_graph, "ensure_schema_version", lambda *a, **k: None)

    captured = {}

    def fake_upsert(self, document, sections, entities, mentions):
        captured["document"] = document
        captured["sections"] = sections
        return {"mock": True}

    monkeypatch.setattr(build_graph.GraphBuilder, "upsert_document", fake_upsert)

    html = """
    <html>
      <head><title>HTML Sample</title></head>
      <body>
        <h1>Intro</h1>
        <p>Hello from HTML ingestion.</p>
      </body>
    </html>
    """

    ingest_document("file:///tmp/sample.html", html, fmt="html")

    assert captured["document"]["source_type"] == "html"
    assert captured["sections"], "HTML parser should produce at least one section"
