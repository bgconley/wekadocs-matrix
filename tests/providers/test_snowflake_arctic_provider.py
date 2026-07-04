"""
Unit tests for SnowflakeArcticProvider.

Tests the dense embedding provider backed by local Snowflake Arctic service.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.providers.embeddings.snowflake_arctic import SnowflakeArcticProvider
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings


def _make_settings(
    *,
    profile: str = "snowflake_arctic_v2l",
    provider: str = "snowflake-arctic-service",
    model_id: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    dims: int = 1024,
    service_url: str = "http://localhost:9010/v1",
) -> EmbeddingSettings:
    """Create test settings with sensible defaults."""
    return EmbeddingSettings(
        profile=profile,
        provider=provider,
        model_id=model_id,
        version="snowflake-arctic-embed-l-v2.0",
        dims=dims,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer_backend="hf",
        tokenizer_model_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        service_url=service_url,
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=False,
            supports_colbert=False,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
        extra={},
    )


class TestSnowflakeArcticProviderInit:
    """Test provider initialization."""

    def test_init_with_settings(self):
        """Provider initializes correctly with settings."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.dims == 1024
        assert provider.model_id == "Snowflake/snowflake-arctic-embed-l-v2.0"
        assert provider.provider_name == "snowflake-arctic-service"

    def test_init_requires_settings(self):
        """Provider raises ValueError if settings is None."""
        with pytest.raises(ValueError, match="EmbeddingSettings are required"):
            SnowflakeArcticProvider(settings=None)

    def test_init_uses_service_url_from_settings(self):
        """Provider uses service_url from settings."""
        settings = _make_settings(service_url="http://custom:9999/v1")
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider._base_url == "http://custom:9999/v1"

    @patch.dict("os.environ", {"CHONKIE_EMBEDDINGS_BASE_URL": "http://env:8080/v1"})
    def test_init_falls_back_to_env_var(self):
        """Provider falls back to env var if settings.service_url is None."""
        settings = _make_settings(service_url=None)
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider._base_url == "http://env:8080/v1"


class TestEmbedDocuments:
    """Test embed_documents method."""

    def test_embed_documents_returns_correct_shape(self):
        """embed_documents returns list of vectors with correct shape."""
        settings = _make_settings()
        mock_client = Mock()
        mock_client.embeddings.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * 1024},
                {"index": 1, "embedding": [0.2] * 1024},
            ]
        }

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)
        result = provider.embed_documents(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 1024
        assert len(result[1]) == 1024
        mock_client.embeddings.assert_called_once()

    def test_embed_documents_preserves_order(self):
        """embed_documents preserves input order based on index."""
        settings = _make_settings()
        mock_client = Mock()
        # Return out of order to test sorting
        mock_client.embeddings.return_value = {
            "data": [
                {"index": 1, "embedding": [0.2] * 1024},
                {"index": 0, "embedding": [0.1] * 1024},
            ]
        }

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)
        result = provider.embed_documents(["first", "second"])

        # First result should be index 0 (0.1), second should be index 1 (0.2)
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2

    def test_embed_documents_raises_on_empty_list(self):
        """embed_documents raises ValueError for empty input."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        with pytest.raises(ValueError, match="empty list"):
            provider.embed_documents([])

    def test_embed_documents_normalizes_model_name(self):
        """embed_documents normalizes HuggingFace-style model names."""
        settings = _make_settings(model_id="Snowflake/snowflake-arctic-embed-l-v2.0")
        mock_client = Mock()
        mock_client.embeddings.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * 1024}]
        }

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)
        provider.embed_documents(["test"])

        # Verify the model name was normalized (no "Snowflake/" prefix)
        call_args = mock_client.embeddings.call_args
        assert call_args.kwargs["model"] == "snowflake-arctic-embed-l-v2.0"


class TestEmbedQuery:
    """Test embed_query method."""

    def test_embed_query_returns_single_vector(self):
        """embed_query returns a single vector."""
        settings = _make_settings()
        mock_client = Mock()
        mock_client.embeddings.return_value = {
            "data": [{"index": 0, "embedding": [0.5] * 1024}]
        }

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)
        result = provider.embed_query("test query")

        assert len(result) == 1024
        assert result[0] == 0.5

    def test_embed_query_uses_query_model_variant(self):
        """embed_query uses the -query model variant for asymmetric retrieval."""
        settings = _make_settings(model_id="Snowflake/snowflake-arctic-embed-l-v2.0")
        mock_client = Mock()
        mock_client.embeddings.return_value = {
            "data": [{"index": 0, "embedding": [0.5] * 1024}]
        }

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)
        provider.embed_query("test query")

        # Verify the query model variant was used
        call_args = mock_client.embeddings.call_args
        assert call_args.kwargs["model"] == "snowflake-arctic-embed-l-v2.0-query"

    def test_embed_query_raises_on_empty_string(self):
        """embed_query raises ValueError for empty input."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        with pytest.raises(ValueError, match="empty query"):
            provider.embed_query("")

    def test_embed_query_raises_on_whitespace_only(self):
        """embed_query raises ValueError for whitespace-only input."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        with pytest.raises(ValueError, match="empty query"):
            provider.embed_query("   ")


class TestSparseAndColBERT:
    """Test that sparse/ColBERT methods raise NotImplementedError."""

    def test_embed_sparse_not_implemented(self):
        """embed_sparse raises NotImplementedError."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        with pytest.raises(NotImplementedError, match="BGE-M3"):
            provider.embed_sparse(["text"])

    def test_embed_colbert_not_implemented(self):
        """embed_colbert raises NotImplementedError."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        with pytest.raises(NotImplementedError, match="BGE-M3"):
            provider.embed_colbert(["text"])


class TestModelNameNormalization:
    """Test model name normalization."""

    def test_normalize_huggingface_style(self):
        """Normalizes HuggingFace-style names (org/model)."""
        result = SnowflakeArcticProvider._normalize_model_name(
            "Snowflake/snowflake-arctic-embed-l-v2.0"
        )
        assert result == "snowflake-arctic-embed-l-v2.0"

    def test_normalize_already_normalized(self):
        """Preserves already-normalized names."""
        result = SnowflakeArcticProvider._normalize_model_name(
            "snowflake-arctic-embed-l-v2.0"
        )
        assert result == "snowflake-arctic-embed-l-v2.0"

    def test_normalize_non_arctic_model(self):
        """Preserves non-Arctic model names."""
        result = SnowflakeArcticProvider._normalize_model_name("some-other-model")
        assert result == "some-other-model"

    def test_normalize_non_arctic_with_org(self):
        """Preserves org/model names that aren't Arctic."""
        result = SnowflakeArcticProvider._normalize_model_name("SomeOrg/some-model")
        assert result == "SomeOrg/some-model"


class TestValidateDimensions:
    """Test dimension validation."""

    def test_validate_dimensions_match(self):
        """validate_dimensions returns True when dimensions match."""
        settings = _make_settings(dims=1024)
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.validate_dimensions(1024) is True

    def test_validate_dimensions_mismatch(self):
        """validate_dimensions returns False when dimensions don't match."""
        settings = _make_settings(dims=1024)
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.validate_dimensions(384) is False


class TestProviderProperties:
    """Test provider properties."""

    def test_dims_property(self):
        """dims property returns correct value."""
        settings = _make_settings(dims=1024)
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.dims == 1024

    def test_model_id_property(self):
        """model_id property returns correct value."""
        settings = _make_settings(model_id="custom-model")
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.model_id == "custom-model"

    def test_provider_name_property(self):
        """provider_name property returns correct value."""
        settings = _make_settings(provider="snowflake-arctic-service")
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        assert provider.provider_name == "snowflake-arctic-service"

    def test_repr(self):
        """__repr__ returns informative string."""
        settings = _make_settings()
        mock_client = Mock()

        provider = SnowflakeArcticProvider(settings=settings, client=mock_client)

        repr_str = repr(provider)
        assert "SnowflakeArcticProvider" in repr_str
        assert "1024" in repr_str


class TestFactoryIntegration:
    """Test factory creates provider correctly."""

    def test_factory_creates_provider(self):
        """ProviderFactory can create SnowflakeArcticProvider."""
        from src.providers.factory import ProviderFactory

        settings = _make_settings()

        # Create via factory's creator method directly
        provider = ProviderFactory._create_snowflake_arctic_provider(settings)

        assert isinstance(provider, SnowflakeArcticProvider)
        assert provider.provider_name == "snowflake-arctic-service"

    def test_provider_registered_in_creators(self):
        """snowflake-arctic-service is registered in _EMBEDDING_PROVIDER_CREATORS."""
        from src.providers.factory import ProviderFactory

        assert (
            "snowflake-arctic-service" in ProviderFactory._EMBEDDING_PROVIDER_CREATORS
        )

    def test_aliases_resolve_to_provider(self):
        """Provider aliases resolve correctly."""
        from src.providers.factory import ProviderFactory

        aliases = ProviderFactory._EMBEDDING_PROVIDER_ALIASES
        assert aliases.get("snowflake-arctic") == "snowflake-arctic-service"
        assert aliases.get("snowflake_arctic") == "snowflake-arctic-service"
        assert aliases.get("arctic") == "snowflake-arctic-service"
