"""Tests for the mxbai-reranker service contract.

These tests verify the HTTP API contract without requiring GPU or the
actual mxbai-rerank model. The model is mocked at the library level.

Run:
    pytest tests/services/test_mxbai_reranker_service.py -v
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock helpers -- build a fake mxbai_rerank module so server.py can import it
# ---------------------------------------------------------------------------


@dataclass
class MockRankedResult:
    """Mock result matching mxbai_rerank output shape (.index, .score)."""

    index: int
    score: float
    document: Optional[str] = None


class MockMxbaiRerankV2:
    """Mock for MxbaiRerankV2 -- returns deterministic scores."""

    def __init__(
        self,
        model_id,
        max_length=8192,
        device="cpu",
        torch_dtype="auto",
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.predefined_length = 96
        self.model_max_length = 32768
        self.max_length_padding = 32768
        # Expose a stub .model attribute so device-move logic doesn't crash
        self.model = MagicMock()
        self.model.dtype = torch_dtype
        self.model.device = device
        self.model.config = MagicMock(use_cache=True)
        self.model.generation_config = MagicMock(use_cache=True)
        self.last_rank_kwargs = None

    def rank(
        self,
        query,
        documents,
        instruction=None,
        return_documents=False,
        top_k=None,
        batch_size=32,
    ):
        self.last_rank_kwargs = {
            "instruction": instruction,
            "return_documents": return_documents,
            "top_k": top_k,
            "batch_size": batch_size,
        }
        top_k = top_k or len(documents)
        results = []
        for i in range(min(top_k, len(documents))):
            # Deterministic descending scores: 1.0, 0.9, 0.8, ...
            score = 1.0 - (i * 0.1)
            results.append(MockRankedResult(index=i, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results


def _build_mock_mxbai_module() -> ModuleType:
    """Create a fake ``mxbai_rerank`` module exposing MxbaiRerankV2."""
    mod = ModuleType("mxbai_rerank")
    mod.MxbaiRerankV2 = MockMxbaiRerankV2
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _patched_server():
    """Import server.py with mxbai_rerank mocked out.

    We inject a fake ``mxbai_rerank`` module into sys.modules *before*
    importing server.py, then add the service directory to sys.path so
    ``import server`` resolves to ``services/mxbai-reranker/server.py``.
    """
    service_dir = str(
        Path(__file__).resolve().parents[2] / "services" / "mxbai-reranker"
    )

    # Inject the mock module
    mock_mod = _build_mock_mxbai_module()
    sys.modules["mxbai_rerank"] = mock_mod

    # Temporarily prepend service dir so ``import server`` works
    sys.path.insert(0, service_dir)
    try:
        # Remove any cached import of 'server' that might conflict
        sys.modules.pop("server", None)
        import server as srv  # noqa: E402

        yield srv
    finally:
        sys.path.remove(service_dir)
        sys.modules.pop("mxbai_rerank", None)
        sys.modules.pop("server", None)


@pytest.fixture(scope="module")
def client(_patched_server):
    """TestClient backed by the mocked FastAPI app."""
    from fastapi.testclient import TestClient

    with TestClient(_patched_server.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_expected_fields(self, client):
        data = client.get("/health").json()
        for key in (
            "status",
            "model",
            "device",
            "loaded",
            "warmup_ok",
            "max_length",
            "dtype",
            "batch_size",
            "loaded_dtype",
            "loaded_device",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ):
            assert key in data, f"missing key: {key}"

    def test_health_reports_loaded(self, client):
        data = client.get("/health").json()
        assert data["loaded"] is True

    def test_health_reports_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# /v1/rerank — happy path
# ---------------------------------------------------------------------------


class TestRerankHappyPath:
    def test_rerank_returns_200(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["doc one", "doc two", "doc three"],
            },
        )
        assert resp.status_code == 200

    def test_rerank_returns_all_documents(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["a", "b", "c"],
            },
        )
        assert len(resp.json()["results"]) == 3

    def test_rerank_results_sorted_descending(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["a", "b", "c", "d"],
            },
        )
        scores = [r["score"] for r in resp.json()["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_result_has_index_and_score(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["doc"],
            },
        )
        result = resp.json()["results"][0]
        assert "index" in result
        assert "score" in result
        assert isinstance(result["index"], int)
        assert isinstance(result["score"], float)

    def test_rerank_response_has_model_and_latency(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["doc"],
            },
        )
        data = resp.json()
        assert "model" in data
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert "mxbai" in data["model"]  # sanity check model name

    def test_rerank_with_instruction(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["doc one", "doc two"],
                "instruction": "Prefer technical content",
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_rerank_with_top_k(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["a", "b", "c", "d", "e"],
                "top_k": 2,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_rerank_with_model_field_ignored(self, client):
        """model field in request is accepted but ignored."""
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["doc"],
                "model": "some-other-model",
            },
        )
        assert resp.status_code == 200
        # Response model should be the service's configured model, not the request's
        assert resp.json()["model"] == "mixedbread-ai/mxbai-rerank-large-v2"

    def test_rerank_uses_configured_batch_size(self, client, _patched_server):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["a", "b", "c", "d", "e"],
            },
        )
        assert resp.status_code == 200
        assert _patched_server._model.last_rank_kwargs["batch_size"] == 8


# ---------------------------------------------------------------------------
# /v1/rerank — validation errors
# ---------------------------------------------------------------------------


class TestRerankValidation:
    def test_empty_documents_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": [],
            },
        )
        assert resp.status_code == 422

    def test_empty_query_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "",
                "documents": ["doc"],
            },
        )
        assert resp.status_code == 422

    def test_empty_string_document_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["valid doc", ""],
            },
        )
        assert resp.status_code == 422

    def test_whitespace_only_document_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["valid doc", "   "],
            },
        )
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "documents": ["doc"],
            },
        )
        assert resp.status_code == 422

    def test_missing_documents_returns_422(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic model tests (no client needed)
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Directly test the request/response Pydantic models."""

    def test_rerank_request_accepts_minimal(self, _patched_server):
        req = _patched_server.RerankRequest(
            query="hello",
            documents=["doc one"],
        )
        assert req.query == "hello"
        assert req.top_k is None
        assert req.instruction is None
        assert req.model is None

    def test_rerank_request_accepts_all_fields(self, _patched_server):
        req = _patched_server.RerankRequest(
            query="hello",
            documents=["a", "b"],
            model="m",
            instruction="inst",
            top_k=1,
        )
        assert req.top_k == 1
        assert req.instruction == "inst"

    def test_rerank_result_fields(self, _patched_server):
        r = _patched_server.RerankResult(index=3, score=0.95)
        assert r.index == 3
        assert r.score == 0.95

    def test_rerank_response_fields(self, _patched_server):
        resp = _patched_server.RerankResponse(
            results=[_patched_server.RerankResult(index=0, score=1.0)],
            model="test-model",
            latency_ms=42.5,
        )
        assert len(resp.results) == 1
        assert resp.model == "test-model"
        assert resp.latency_ms == 42.5


class TestRuntimeHelpers:
    def test_resolve_torch_dtype(self, _patched_server):
        assert (
            _patched_server.resolve_torch_dtype("float16")
            == _patched_server.torch.float16
        )
        assert (
            _patched_server.resolve_torch_dtype("bf16")
            == _patched_server.torch.bfloat16
        )
        assert (
            _patched_server.resolve_torch_dtype("float32")
            == _patched_server.torch.float32
        )
        assert _patched_server.resolve_torch_dtype("auto") == "auto"

    def test_optimize_loaded_model_corrects_padding_and_disables_cache(
        self, _patched_server
    ):
        model = MockMxbaiRerankV2(
            "mixedbread-ai/mxbai-rerank-large-v2", max_length=8192
        )
        _patched_server.optimize_loaded_model(model)
        assert model.max_length_padding == 8288
        assert model.model.config.use_cache is False
        assert model.model.generation_config.use_cache is False
