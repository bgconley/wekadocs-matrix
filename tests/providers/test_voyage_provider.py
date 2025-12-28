from __future__ import annotations

from unittest.mock import Mock

import httpx

from src.providers.embeddings import voyage as voyage_module
from src.providers.embeddings.voyage import VoyageEmbeddingProvider
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings


class _DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def count_tokens_batch(self, texts: list[str]) -> int:
        return sum(len(text.split()) for text in texts)


def _settings(*, contextual: bool) -> EmbeddingSettings:
    return EmbeddingSettings(
        profile="voyage_context_3",
        provider="voyage-ai",
        model_id="voyage-context-3",
        version="test-version",
        dims=2,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer_backend="hf",
        tokenizer_model_id="voyageai/voyage-context-3",
        service_url=None,
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=False,
            supports_colbert=False,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
        extra={
            "query_task": "query",
            "document_task": "document",
            "output_dimension": 2,
            "output_dtype": "float",
            "supports_contextualized_chunks": contextual,
            "contextual_limits": {
                "max_inputs": 1000,
                "max_total_tokens": 120000,
                "max_total_chunks": 16000,
            },
        },
    )


def _mock_response(url: str, payload: dict) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(200, json=payload, request=request)


def test_voyage_contextual_request_shape(monkeypatch):
    monkeypatch.setattr(voyage_module, "TokenizerService", lambda: _DummyTokenizer())
    settings = _settings(contextual=True)
    captured = {}

    def _post(url: str, json: dict):
        captured["url"] = url
        captured["payload"] = json
        response_payload = {
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "index": 0,
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                        {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                    ],
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 4},
        }
        return _mock_response(url, response_payload)

    client = Mock()
    client.post = Mock(side_effect=_post)
    provider = VoyageEmbeddingProvider(settings=settings, client=client, api_key="test")

    vectors = provider.embed_contextualized_documents(
        [["chunk_a", "chunk_b"]], input_type="document"
    )

    assert captured["url"].endswith("/contextualizedembeddings")
    assert captured["payload"]["inputs"] == [["chunk_a", "chunk_b"]]
    assert captured["payload"]["model"] == "voyage-context-3"
    assert captured["payload"]["input_type"] == "document"
    assert captured["payload"]["output_dimension"] == 2
    assert vectors == [[[0.1, 0.2], [0.3, 0.4]]]


def test_voyage_standard_request_shape(monkeypatch):
    monkeypatch.setattr(voyage_module, "TokenizerService", lambda: _DummyTokenizer())
    settings = _settings(contextual=False)
    captured = {}

    def _post(url: str, json: dict):
        captured["url"] = url
        captured["payload"] = json
        response_payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 2},
        }
        return _mock_response(url, response_payload)

    client = Mock()
    client.post = Mock(side_effect=_post)
    provider = VoyageEmbeddingProvider(settings=settings, client=client, api_key="test")

    vectors = provider.embed_documents(["doc_a", "doc_b"])

    assert captured["url"].endswith("/embeddings")
    assert captured["payload"]["input"] == ["doc_a", "doc_b"]
    assert captured["payload"]["model"] == "voyage-context-3"
    assert captured["payload"]["input_type"] == "document"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
