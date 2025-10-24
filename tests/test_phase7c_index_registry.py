"""
Unit tests for index registry.
Phase 7C, Task 7C.2: Test dimension enforcement and index management.
"""

import pytest

from src.registry.index_registry import IndexRegistry, create_default_registry


class MockProvider:
    """Mock embedding provider for testing."""

    def __init__(self, provider_name: str, model_id: str, dims: int):
        self.provider_name = provider_name
        self.model_id = model_id
        self.dims = dims


class TestIndexRegistry:
    """Tests for IndexRegistry."""

    def test_register_index(self):
        """Test registering a new index."""
        registry = IndexRegistry()

        registry.register_index(
            name="test_index",
            dims=1024,
            provider="jina-ai",
            model="jina-embeddings-v4",
            version="v4-2025-01-23",
            is_active=True,
        )

        index = registry.get_index("test_index")

        assert index["name"] == "test_index"
        assert index["dims"] == 1024
        assert index["provider"] == "jina-ai"
        assert index["model"] == "jina-embeddings-v4"
        assert index["is_active"] is True

    def test_register_duplicate_index_raises(self):
        """Test that registering duplicate index raises ValueError."""
        registry = IndexRegistry()

        registry.register_index(
            name="test", dims=1024, provider="jina", model="v4", version="1"
        )

        with pytest.raises(ValueError, match="already registered"):
            registry.register_index(
                name="test", dims=1024, provider="jina", model="v4", version="1"
            )

    def test_register_invalid_dims_raises(self):
        """Test that invalid dimensions raise ValueError."""
        registry = IndexRegistry()

        with pytest.raises(ValueError, match="must be positive"):
            registry.register_index(
                name="test", dims=0, provider="jina", model="v4", version="1"
            )

        with pytest.raises(ValueError, match="must be positive"):
            registry.register_index(
                name="test", dims=-100, provider="jina", model="v4", version="1"
            )

    def test_get_index_not_found_raises(self):
        """Test that getting unknown index raises KeyError."""
        registry = IndexRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_index("unknown_index")

    def test_get_active_index(self):
        """Test getting active index."""
        registry = IndexRegistry()

        registry.register_index(
            name="legacy", dims=384, provider="st", model="minilm", version="1"
        )
        registry.register_index(
            name="new",
            dims=1024,
            provider="jina",
            model="v4",
            version="1",
            is_active=True,
        )

        active = registry.get_active_index()

        assert active["name"] == "new"
        assert active["dims"] == 1024

    def test_get_active_index_not_set_raises(self):
        """Test that getting active index when none set raises RuntimeError."""
        registry = IndexRegistry()

        with pytest.raises(RuntimeError, match="No active index"):
            registry.get_active_index()

    def test_set_active_index(self):
        """Test changing active index."""
        registry = IndexRegistry()

        registry.register_index(
            name="index1",
            dims=384,
            provider="st",
            model="minilm",
            version="1",
            is_active=True,
        )
        registry.register_index(
            name="index2", dims=1024, provider="jina", model="v4", version="1"
        )

        # Initially index1 is active
        assert registry.get_active_index()["name"] == "index1"

        # Switch to index2
        registry.set_active_index("index2")

        assert registry.get_active_index()["name"] == "index2"

        # Verify index1 is no longer active
        index1 = registry.get_index("index1")
        assert index1["is_active"] is False

    def test_set_active_index_unknown_raises(self):
        """Test that setting unknown index active raises KeyError."""
        registry = IndexRegistry()

        with pytest.raises(KeyError, match="unknown index"):
            registry.set_active_index("unknown")

    def test_enforce_compatibility_matching_dims(self):
        """Test enforce_compatibility with matching dimensions."""
        registry = IndexRegistry()

        registry.register_index(
            name="test", dims=1024, provider="jina", model="v4", version="1"
        )

        provider = MockProvider("jina-ai", "jina-embeddings-v4", 1024)

        # Should not raise
        registry.enforce_compatibility("test", provider)

    def test_enforce_compatibility_mismatched_dims_raises(self):
        """Test enforce_compatibility with mismatched dimensions raises."""
        registry = IndexRegistry()

        registry.register_index(
            name="test", dims=1024, provider="jina", model="v4", version="1"
        )

        provider = MockProvider("sentence-transformers", "minilm", 384)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            registry.enforce_compatibility("test", provider)

    def test_enforce_compatibility_warns_on_model_mismatch(self, caplog):
        """Test that model mismatch logs warning."""
        import logging

        caplog.set_level(logging.WARNING)

        registry = IndexRegistry()

        registry.register_index(
            name="test", dims=1024, provider="jina", model="v4", version="1"
        )

        provider = MockProvider("jina-ai", "jina-embeddings-v3", 1024)

        # Should not raise but should warn
        registry.enforce_compatibility("test", provider)

        assert "Model mismatch" in caplog.text

    def test_list_indices(self):
        """Test listing all indices."""
        registry = IndexRegistry()

        registry.register_index(
            name="index1", dims=384, provider="st", model="minilm", version="1"
        )
        registry.register_index(
            name="index2", dims=1024, provider="jina", model="v4", version="1"
        )

        indices = registry.list_indices()

        assert len(indices) == 2
        assert any(idx["name"] == "index1" for idx in indices)
        assert any(idx["name"] == "index2" for idx in indices)

    def test_get_index_for_provider(self):
        """Test finding compatible index for provider."""
        registry = IndexRegistry()

        registry.register_index(
            name="legacy", dims=384, provider="st", model="minilm", version="1"
        )
        registry.register_index(
            name="new", dims=1024, provider="jina", model="v4", version="1"
        )

        # Provider with 1024-D
        provider = MockProvider("jina-ai", "jina-embeddings-v4", 1024)

        index = registry.get_index_for_provider(provider)

        assert index is not None
        assert index["name"] == "new"
        assert index["dims"] == 1024

    def test_get_index_for_provider_no_match(self):
        """Test finding compatible index when no match exists."""
        registry = IndexRegistry()

        registry.register_index(
            name="test", dims=1024, provider="jina", model="v4", version="1"
        )

        # Provider with incompatible dimensions
        provider = MockProvider("test", "test-model", 768)

        index = registry.get_index_for_provider(provider)

        assert index is None


class TestCreateDefaultRegistry:
    """Tests for create_default_registry helper."""

    def test_creates_default_indices(self):
        """Test that default registry has both indices."""
        registry = create_default_registry()

        indices = registry.list_indices()

        assert len(indices) == 2

        # Check for legacy index
        legacy = registry.get_index("weka_sections")
        assert legacy["dims"] == 384
        assert legacy["is_active"] is False

        # Check for new index
        new = registry.get_index("weka_sections_v2")
        assert new["dims"] == 1024
        assert new["is_active"] is True

    def test_default_active_is_1024d(self):
        """Test that default active index is 1024-D."""
        registry = create_default_registry()

        active = registry.get_active_index()

        assert active["name"] == "weka_sections_v2"
        assert active["dims"] == 1024
