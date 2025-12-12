"""
Unit tests for provider factory.
Phase 7C, Task 7C.1: Test ENV-selectable provider creation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.factory import ProviderFactory, create_default_providers
from src.providers.rerank.base import RerankProvider


class TestProviderFactory:
    """Tests for ProviderFactory."""

    def test_create_jina_embedding_provider(self):
        """Test creating Jina embedding provider."""
        with patch.dict(
            os.environ,
            {"JINA_API_KEY": "test-key"},  # pragma: allowlist secret
        ):
            provider = ProviderFactory.create_embedding_provider(
                provider="jina-ai",
                model="jina-embeddings-v4",
                dims=1024,
                task="retrieval.passage",
            )

            assert isinstance(provider, EmbeddingProvider)
            assert provider.provider_name == "jina-ai"
            assert provider.model_id == "jina-embeddings-v4"
            assert provider.dims == 1024

    def test_create_sentence_transformers_provider(self):
        """Test creating SentenceTransformers provider."""
        with patch(
            "src.providers.embeddings.sentence_transformers.SentenceTransformer"
        ):
            provider = ProviderFactory.create_embedding_provider(
                provider="sentence-transformers",
                model="sentence-transformers/all-MiniLM-L6-v2",
                dims=384,
            )

            assert isinstance(provider, EmbeddingProvider)
            assert provider.provider_name == "sentence-transformers"
            assert provider.dims == 384

    def test_create_bge_m3_service_provider(self):
        """Test creating the BGE-M3 service-backed provider."""
        with patch(
            "src.providers.embeddings.bge_m3_service.BGEM3ServiceProvider"
        ) as mock_provider_cls:
            mock_instance = MagicMock(spec=EmbeddingProvider)
            mock_provider_cls.return_value = mock_instance

            provider = ProviderFactory.create_embedding_provider(
                provider="bge-m3-service",
                model="bge-m3",
                dims=1024,
                task="symmetric",
                base_url="http://127.0.0.1:9000",
            )

            assert provider is mock_instance
            mock_provider_cls.assert_called_once()

    def test_bge_aliases_normalize_to_service(self):
        """Test that bge aliases normalize to the service provider."""
        with patch(
            "src.providers.embeddings.bge_m3_service.BGEM3ServiceProvider"
        ) as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock(spec=EmbeddingProvider)
            provider = ProviderFactory.create_embedding_provider(
                provider="bge_m3", model="bge-m3", dims=1024
            )

            assert provider is mock_provider_cls.return_value
            mock_provider_cls.assert_called_once()

    def test_create_jina_rerank_provider(self):
        """Test creating Jina rerank provider."""
        with patch.dict(
            os.environ,
            {"JINA_API_KEY": "test-key"},  # pragma: allowlist secret
        ):
            provider = ProviderFactory.create_rerank_provider(
                provider="jina-ai", model="jina-reranker-v3"
            )

            assert isinstance(provider, RerankProvider)
            assert provider.provider_name == "jina-ai"
            assert provider.model_id == "jina-reranker-v3"

    def test_create_noop_reranker(self):
        """Test creating noop reranker."""
        provider = ProviderFactory.create_rerank_provider(provider="noop")

        assert isinstance(provider, RerankProvider)
        assert provider.provider_name == "noop"
        assert provider.model_id == "noop"

    def test_unknown_embedding_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            ProviderFactory.create_embedding_provider(
                provider="unknown-provider", model="test", dims=1024
            )

    def test_unknown_rerank_provider_raises(self):
        """Test that unknown rerank provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            ProviderFactory.create_rerank_provider(
                provider="unknown-provider", model="test"
            )

    def test_jina_without_api_key_raises(self):
        """Test that Jina provider without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="JINA_API_KEY required"):
                ProviderFactory.create_embedding_provider(
                    provider="jina-ai", model="jina-embeddings-v4", dims=1024
                )

    def test_provider_from_env_vars(self):
        """Test creating provider from environment variables."""
        env = {
            "EMBEDDINGS_PROVIDER": "jina-ai",
            "EMBEDDINGS_MODEL": "jina-embeddings-v4",
            "EMBEDDINGS_DIM": "1024",
            "EMBEDDINGS_TASK": "retrieval.passage",
            "JINA_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env):
            provider = ProviderFactory.create_embedding_provider()

            assert provider.provider_name == "jina-ai"
            assert provider.model_id == "jina-embeddings-v4"
            assert provider.dims == 1024

    def test_get_provider_info(self):
        """Test getting provider metadata."""
        with patch.dict(os.environ, {"JINA_API_KEY": "test-key"}):
            provider = ProviderFactory.create_embedding_provider(
                provider="jina-ai", model="jina-embeddings-v4", dims=1024
            )

            info = ProviderFactory.get_provider_info(provider)

            assert info["provider"] == "jina-ai"
            assert info["model"] == "jina-embeddings-v4"
            assert info["dims"] == 1024


class TestCreateDefaultProviders:
    """Tests for create_default_providers helper."""

    def test_creates_providers_from_config(self):
        """Test that default providers are created from config."""
        env = {
            "EMBEDDINGS_PROVIDER": "jina-ai",
            "EMBEDDINGS_MODEL": "jina-embeddings-v4",
            "EMBEDDINGS_DIM": "1024",
            "RERANK_PROVIDER": "jina-ai",
            "RERANK_MODEL": "jina-reranker-v3",
            "JINA_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env):
            embedding_provider, rerank_provider = create_default_providers()

            assert isinstance(embedding_provider, EmbeddingProvider)
            assert isinstance(rerank_provider, RerankProvider)
            assert embedding_provider.dims == 1024
            assert rerank_provider.model_id == "jina-reranker-v3"

    def test_fallback_to_noop_reranker(self):
        """Test fallback to noop reranker when Jina unavailable."""
        env = {
            "EMBEDDINGS_PROVIDER": "jina-ai",
            "JINA_API_KEY": "test-key",
            "RERANK_PROVIDER": "noop",
        }

        with patch.dict(os.environ, env):
            _, rerank_provider = create_default_providers()

            assert rerank_provider.provider_name == "noop"
