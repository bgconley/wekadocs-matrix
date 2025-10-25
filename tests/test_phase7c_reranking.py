"""
Phase 7C Reranking Integration Tests
Tests for Jina reranker integration into query pipeline.

Tests:
- Provider factory creates reranker correctly
- Reranker integration in HybridSearchEngine
- Query service uses reranker when available
- Graceful fallback when reranker unavailable
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.providers.factory import ProviderFactory
from src.providers.rerank.base import RerankProvider
from src.providers.rerank.jina import JinaRerankProvider
from src.providers.rerank.noop import NoopReranker


class TestRerankProviderFactory:
    """Test reranker provider factory."""

    def test_create_jina_reranker(self):
        """Test Jina reranker creation from factory."""
        # Set API key for test
        os.environ["JINA_API_KEY"] = "test-key-123"  # pragma: allowlist secret

        try:
            factory = ProviderFactory()
            reranker = factory.create_rerank_provider(
                provider="jina-ai", model="jina-reranker-v3"
            )

            assert isinstance(reranker, JinaRerankProvider)
            assert reranker.provider_name == "jina-ai"
            assert reranker.model_id == "jina-reranker-v3"
        finally:
            # Cleanup
            if "JINA_API_KEY" in os.environ:
                del os.environ["JINA_API_KEY"]

    def test_create_noop_reranker(self):
        """Test noop reranker creation."""
        factory = ProviderFactory()
        reranker = factory.create_rerank_provider(provider="noop")

        assert isinstance(reranker, NoopReranker)
        assert reranker.provider_name == "noop"

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        factory = ProviderFactory()

        with pytest.raises(ValueError, match="Unknown rerank provider"):
            factory.create_rerank_provider(provider="invalid-provider")


class TestHybridSearchEngineReranking:
    """Test reranking integration in HybridSearchEngine."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock reranker."""
        reranker = MagicMock(spec=RerankProvider)
        reranker.provider_name = "mock-reranker"
        reranker.model_id = "mock-model"

        # Mock rerank method to return top 3 with scores
        def mock_rerank(query, candidates, top_k=10):
            # Return top N candidates with rerank scores
            results = []
            for i, cand in enumerate(candidates[:top_k]):
                cand_copy = cand.copy()
                cand_copy["rerank_score"] = 0.9 - (i * 0.1)  # Decreasing scores
                cand_copy["original_rank"] = i + 1
                cand_copy["reranker"] = "mock-model"
                results.append(cand_copy)
            return results

        reranker.rerank = mock_rerank
        return reranker

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()

        # Mock search to return 5 candidates
        def mock_search(vector, k, filters=None):
            return [
                {
                    "id": f"result-{i}",
                    "node_id": f"section-{i}",
                    "node_label": "Section",
                    "score": 0.8 - (i * 0.1),
                    "metadata": {
                        "text": f"This is test section {i} with content about the query.",
                        "title": f"Section {i}",
                        "score_kind": "similarity",
                    },
                }
                for i in range(k)
            ]

        store.search = mock_search
        return store

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.model_id = "test-embedder"
        embedder.dims = 384
        embedder.embed_query = MagicMock(return_value=[0.1] * 384)
        return embedder

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        session.run = MagicMock(return_value=[])
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        driver.session = MagicMock(return_value=session)
        return driver

    def test_search_with_reranking_enabled(
        self, mock_vector_store, mock_neo4j_driver, mock_embedder, mock_reranker
    ):
        """Test search with reranking enabled."""
        from src.query.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            neo4j_driver=mock_neo4j_driver,
            embedder=mock_embedder,
            reranker=mock_reranker,  # Reranking enabled
        )

        # Search should use reranking
        results = engine.search(
            query_text="test query",
            k=3,
            expand_graph=False,  # Disable graph expansion for simplicity
        )

        # Verify results have rerank scores (proves reranking was called)
        assert len(results.results) > 0
        assert "rerank_score" in results.results[0].metadata
        assert "reranker" in results.results[0].metadata
        assert results.results[0].metadata["reranker"] == "mock-model"

        # Verify timing includes rerank_time_ms
        assert results.rerank_time_ms >= 0

    def test_search_without_reranking(
        self, mock_vector_store, mock_neo4j_driver, mock_embedder
    ):
        """Test search without reranking (reranker=None)."""
        from src.query.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            neo4j_driver=mock_neo4j_driver,
            embedder=mock_embedder,
            reranker=None,  # No reranking
        )

        results = engine.search(query_text="test query", k=3, expand_graph=False)

        # Verify rerank_time_ms is 0
        assert results.rerank_time_ms == 0.0

        # Verify results don't have rerank scores
        if results.results:
            assert "rerank_score" not in results.results[0].metadata

    def test_reranking_fallback_on_error(
        self, mock_vector_store, mock_neo4j_driver, mock_embedder
    ):
        """Test graceful fallback when reranking fails."""
        from src.query.hybrid_search import HybridSearchEngine

        # Create reranker that raises exception
        failing_reranker = MagicMock(spec=RerankProvider)
        failing_reranker.provider_name = "failing-reranker"
        failing_reranker.model_id = "failing-model"
        failing_reranker.rerank = MagicMock(
            side_effect=RuntimeError("Reranker API failed")
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            neo4j_driver=mock_neo4j_driver,
            embedder=mock_embedder,
            reranker=failing_reranker,
        )

        # Search should not fail, should fallback to vector results
        results = engine.search(query_text="test query", k=3, expand_graph=False)

        # Should still return results (fallback to vector search)
        assert len(results.results) > 0

        # Rerank_time_ms should be recorded (even if it failed)
        assert results.rerank_time_ms >= 0


class TestQueryServiceReranking:
    """Test reranking integration in QueryService."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.embedding.embedding_model = "test-model"
        config.embedding.dims = 384
        config.embedding.version = "v1"
        config.search.vector.primary = "qdrant"
        config.search.vector.qdrant.collection_name = "test_collection"
        config.search.hybrid.top_k = 20
        config.validator.max_depth = 2
        return config

    def test_query_service_initializes_reranker(self, mock_config):
        """Test QueryService initializes reranker from factory."""
        from src.mcp_server.query_service import QueryService

        with patch("src.mcp_server.query_service.get_config", return_value=mock_config):
            with patch.object(ProviderFactory, "create_rerank_provider") as mock_create:
                # Mock reranker creation
                mock_reranker = MagicMock(spec=RerankProvider)
                mock_reranker.provider_name = "test-reranker"
                mock_reranker.model_id = "test-model"
                mock_create.return_value = mock_reranker

                service = QueryService()

                # Trigger reranker initialization by getting search engine
                # (This is lazy-loaded, so we need to trigger it)
                with patch("src.mcp_server.query_service.get_connection_manager"):
                    with patch.object(service, "_get_embedder"):
                        reranker = service._get_reranker()

                # Verify reranker was created
                assert reranker is not None
                assert reranker.provider_name == "test-reranker"

    def test_query_service_handles_reranker_failure(self, mock_config):
        """Test QueryService handles reranker initialization failure gracefully."""
        from src.mcp_server.query_service import QueryService

        with patch("src.mcp_server.query_service.get_config", return_value=mock_config):
            with patch.object(ProviderFactory, "create_rerank_provider") as mock_create:
                # Mock reranker creation failure
                mock_create.side_effect = RuntimeError("API key missing")

                service = QueryService()
                reranker = service._get_reranker()

                # Should return None (graceful fallback)
                assert reranker is None


class TestNoopReranker:
    """Test NoopReranker fallback."""

    def test_noop_reranker_returns_candidates_unchanged(self):
        """Test NoopReranker returns candidates in original order."""
        from src.providers.rerank.noop import NoopReranker

        reranker = NoopReranker()

        candidates = [
            {"id": "1", "text": "first", "score": 0.9},
            {"id": "2", "text": "second", "score": 0.8},
            {"id": "3", "text": "third", "score": 0.7},
        ]

        results = reranker.rerank(query="test query", candidates=candidates, top_k=2)

        # Should return top_k candidates unchanged
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

        # Should preserve original scores
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8

    def test_noop_reranker_metadata(self):
        """Test NoopReranker has correct metadata."""
        from src.providers.rerank.noop import NoopReranker

        reranker = NoopReranker()

        assert reranker.provider_name == "noop"
        assert reranker.model_id == "noop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
