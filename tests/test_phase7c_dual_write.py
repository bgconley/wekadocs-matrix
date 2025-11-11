"""
Test dual-write functionality for Phase 7C.4.
Validates that ingestion writes to both 384-D and 1024-D collections.

NOTE (Session 06-07): Task 7C.4 (dual-write) SKIPPED - fresh installation scenario.
Database is empty (no production data), so we start with 1024-D directly.
Migration complexity eliminated. These tests preserved for documentation only.
"""

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.ingestion.build_graph import GraphBuilder
from src.shared.config import get_config, get_settings
from src.shared.connections import CompatQdrantClient

# Skip entire module - dual-write not needed for fresh installation
pytestmark = pytest.mark.skip(
    reason="Task 7C.4 SKIPPED (Session 06): Fresh installation - no migration needed. "
    "Database empty, starting with 1024-D directly. Dual-write tests not applicable."
)


class TestDualWriteSetup:
    """Test dual-write infrastructure setup."""

    @pytest.fixture
    def config(self):
        """Get config with dual-write enabled."""
        config = get_config()
        # Override feature flag
        config.feature_flags.dual_write_1024d = True
        return config

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def qdrant_client(self):
        """Get Qdrant client."""
        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        compat_client = CompatQdrantClient(client)
        yield compat_client
        # Don't close - may be shared across tests

    def test_dual_write_flag_enables_providers(
        self, config, neo4j_driver, qdrant_client
    ):
        """Test that dual-write flag initializes both providers."""
        builder = GraphBuilder(neo4j_driver, config, qdrant_client)

        # Process embeddings to trigger provider initialization
        document = {
            "id": "test-doc",
            "source_uri": "test.md",
            "title": "Test",
        }
        sections = [
            {
                "id": "test-section",
                "document_id": "test-doc",
                "text": "Test text for dual-write validation",
                "title": "Test Section",
                "level": 1,
                "order": 0,
                "anchor": "test",
                "tokens": 10,
                "checksum": "test-checksum",
            }
        ]

        stats = builder._process_embeddings(document, sections, {})

        # Verify both providers initialized
        assert (
            builder.legacy_embedder is not None
        ), "Legacy 384-D provider not initialized"
        assert builder.new_embedder is not None, "New 1024-D provider not initialized"

        # Verify dimensions
        assert builder.legacy_embedder.dims == 384
        assert builder.new_embedder.dims == 1024

        # Verify stats show dual-write
        assert stats["dual_write_1024d"] is True
        assert stats["computed"] == 2  # Both 384-D and 1024-D embeddings
        assert stats["upserted"] == 2  # Both collections updated

    def test_both_collections_created(self, config, neo4j_driver, qdrant_client):
        """Test that both 384-D and 1024-D collections are created."""
        # Initialize builder to trigger collection creation
        _ = GraphBuilder(neo4j_driver, config, qdrant_client)

        # Get collections
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        # Verify both collections exist
        assert "weka_sections" in collection_names, "Legacy 384-D collection not found"
        assert "weka_sections_v2" in collection_names, "New 1024-D collection not found"


class TestDualWriteIngestion:
    """Test end-to-end ingestion with dual-write."""

    @pytest.fixture(autouse=True)
    def enable_dual_write(self, monkeypatch):
        """Enable dual-write via environment variable."""
        # Set feature flag via config override
        # Note: This requires config to read from env or be reloadable
        pass  # Handled by config fixture in actual test

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def qdrant_client(self):
        """Get Qdrant client."""
        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        compat_client = CompatQdrantClient(client)
        yield compat_client
        # Don't close - may be shared across tests

    def test_section_written_to_both_collections(self, neo4j_driver, qdrant_client):
        """Test that a section is written to both 384-D and 1024-D collections."""
        # Enable dual-write
        config = get_config()
        config.feature_flags.dual_write_1024d = True

        builder = GraphBuilder(neo4j_driver, config, qdrant_client)

        # Ingest test document
        document = {
            "id": "test-dual-write-doc",
            "source_uri": "test-dual-write.md",
            "source_type": "markdown",
            "title": "Dual-Write Test Document",
            "version": "test",
            "checksum": "test-checksum",
            "last_edited": None,
        }

        sections = [
            {
                "id": "test-dual-write-section-1",
                "document_id": "test-dual-write-doc",
                "text": "This is a test section for dual-write validation. It contains enough text to generate meaningful embeddings.",
                "title": "Dual-Write Test Section",
                "level": 1,
                "order": 0,
                "anchor": "test-section",
                "tokens": 20,
                "checksum": "section-checksum",
            }
        ]

        # Upsert
        stats = builder.upsert_document(document, sections, {}, [])

        # Verify stats
        assert stats["sections_upserted"] == 1
        assert stats["embeddings_computed"] == 2  # 384-D + 1024-D
        assert stats["vectors_upserted"] == 2  # Both collections

        # Verify legacy 384-D collection
        import uuid

        section_id = "test-dual-write-section-1"
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section_id))

        legacy_point = qdrant_client.retrieve(
            collection_name="weka_sections",
            ids=[point_uuid],
        )

        assert len(legacy_point) == 1, "Section not found in legacy collection"
        assert (
            len(legacy_point[0].vector) == 384
        ), "Legacy collection has wrong dimensions"
        assert legacy_point[0].payload["node_id"] == section_id
        assert legacy_point[0].payload["embedding_dimensions"] == 384
        assert legacy_point[0].payload["embedding_provider"] == "sentence-transformers"

        # Verify new 1024-D collection
        new_point = qdrant_client.retrieve(
            collection_name="weka_sections_v2",
            ids=[point_uuid],
        )

        assert len(new_point) == 1, "Section not found in new collection"
        assert len(new_point[0].vector) == 1024, "New collection has wrong dimensions"
        assert new_point[0].payload["node_id"] == section_id
        assert new_point[0].payload["embedding_dimensions"] == 1024
        # Provider may be jina-ai or ollama depending on ENV
        assert new_point[0].payload["embedding_provider"] in [
            "jina-ai",
            "ollama",
            "sentence-transformers",
        ]

    def test_neo4j_metadata_reflects_new_provider(self, neo4j_driver, qdrant_client):
        """Test that Neo4j Section node has metadata from new 1024-D provider."""
        # Enable dual-write
        config = get_config()
        config.feature_flags.dual_write_1024d = True

        builder = GraphBuilder(neo4j_driver, config, qdrant_client)

        # Ingest test document
        document = {
            "id": "test-metadata-doc",
            "source_uri": "test-metadata.md",
            "source_type": "markdown",
            "title": "Metadata Test",
            "version": "test",
            "checksum": "test",
            "last_edited": None,
        }

        sections = [
            {
                "id": "test-metadata-section",
                "document_id": "test-metadata-doc",
                "text": "Test text for metadata validation",
                "title": "Test",
                "level": 1,
                "order": 0,
                "anchor": "test",
                "tokens": 10,
                "checksum": "test",
            }
        ]

        builder.upsert_document(document, sections, {}, [])

        # Query Neo4j for section metadata
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section {id: $section_id})
                RETURN s.embedding_version as version,
                       s.embedding_provider as provider,
                       s.embedding_dimensions as dimensions,
                       s.embedding_timestamp as timestamp,
                       s.embedding_task as task
                """,
                section_id="test-metadata-section",
            )

            record = result.single()
            assert record is not None, "Section not found in Neo4j"

            # Metadata should reflect new provider
            assert record["dimensions"] == 1024
            assert record["provider"] in ["jina-ai", "ollama", "sentence-transformers"]
            assert record["version"] is not None
            assert record["timestamp"] is not None


class TestDualWriteDimensionValidation:
    """Test dimension validation in dual-write mode."""

    @pytest.fixture
    def config(self):
        """Get config with dual-write enabled."""
        config = get_config()
        config.feature_flags.dual_write_1024d = True
        return config

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def qdrant_client(self):
        """Get Qdrant client."""
        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        compat_client = CompatQdrantClient(client)
        yield compat_client
        # Don't close - may be shared across tests

    def test_dimension_mismatch_raises_error(self, config, neo4j_driver, qdrant_client):
        """Test that dimension mismatches are caught."""
        builder = GraphBuilder(neo4j_driver, config, qdrant_client)

        # Mock embedder with wrong dimensions
        from unittest.mock import Mock

        builder.legacy_embedder = Mock()
        builder.legacy_embedder.dims = 384
        builder.legacy_embedder.embed_documents = Mock(
            return_value=[[0.1] * 512]  # Wrong! Should be 384
        )

        builder.new_embedder = Mock()
        builder.new_embedder.dims = 1024
        builder.new_embedder.embed_documents = Mock(return_value=[[0.1] * 1024])
        builder.new_embedder.provider_name = "test-provider"

        document = {"id": "test", "source_uri": "test.md", "title": "Test"}
        sections = [
            {
                "id": "test-section",
                "document_id": "test",
                "text": "Test",
                "title": "Test",
                "level": 1,
                "order": 0,
                "anchor": "test",
                "tokens": 10,
                "checksum": "test",
            }
        ]

        # Should raise ValueError for dimension mismatch
        with pytest.raises(ValueError) as exc_info:
            builder._process_embeddings(document, sections, {})

        assert "dimension mismatch" in str(exc_info.value).lower()


class TestDualWriteDisabled:
    """Test that dual-write can be disabled."""

    @pytest.fixture
    def config(self):
        """Get config with dual-write DISABLED."""
        config = get_config()
        config.feature_flags.dual_write_1024d = False
        return config

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def qdrant_client(self):
        """Get Qdrant client."""
        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        compat_client = CompatQdrantClient(client)
        yield compat_client
        # Don't close - may be shared across tests

    def test_single_write_when_disabled(self, config, neo4j_driver, qdrant_client):
        """Test that only single embedding is generated when dual-write disabled."""
        builder = GraphBuilder(neo4j_driver, config, qdrant_client)

        document = {"id": "test", "source_uri": "test.md", "title": "Test"}
        sections = [
            {
                "id": "test-single-write",
                "document_id": "test",
                "text": "Test text for single-write validation",
                "title": "Test",
                "level": 1,
                "order": 0,
                "anchor": "test",
                "tokens": 10,
                "checksum": "test",
            }
        ]

        stats = builder._process_embeddings(document, sections, {})

        # Should only generate ONE embedding
        assert stats["dual_write_1024d"] is False
        assert stats["computed"] == 1  # Only one embedding
        assert stats["upserted"] == 1  # Only one collection

        # Legacy and new embedders should not be initialized
        assert builder.legacy_embedder is None
        assert builder.new_embedder is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
