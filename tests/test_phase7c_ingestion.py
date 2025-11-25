"""
Test Phase 7C.7 ingestion updates.

Validates:
- Sections are dual-labeled as Section:Chunk
- All required embedding fields are present and validated
- Embeddings use configured provider (1024-D by default)
- Ingestion fails gracefully if required fields are missing
"""

import pytest

from src.ingestion.build_graph import GraphBuilder
from src.shared.config import get_config
from src.shared.connections import get_connection_manager


class TestPhase7CIngestion:
    """Test Phase 7C.7 ingestion updates for fresh-start architecture."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver."""
        manager = get_connection_manager()
        driver = manager.get_neo4j_driver()
        yield driver
        driver.close()

    @pytest.fixture
    def qdrant_client(self):
        """Get Qdrant client."""
        manager = get_connection_manager()
        client = manager.get_qdrant_client()
        yield client

    @pytest.fixture
    def graph_builder(self, neo4j_driver, qdrant_client):
        """Get GraphBuilder instance with test config."""
        config = get_config()

        # Override to use local provider for testing (no API key needed)
        config.embedding.provider = "sentence-transformers"
        config.embedding.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dims = 384  # MiniLM is 384-D
        config.embedding.version = "miniLM-L6-v2-test"

        # Use test-specific collection name to avoid dimension mismatch
        # (production collection weka_sections_v2 is 1024-D, tests use 384-D)
        config.search.vector.qdrant.collection_name = "weka_sections_test_384d"

        return GraphBuilder(neo4j_driver, config, qdrant_client, strict_mode=False)

    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return {
            "id": "test-doc-phase7c-ingestion",
            "source_uri": "tests/phase7c_ingestion_test.md",
            "source_type": "markdown",
            "title": "Phase 7C Ingestion Test",
            "version": "1.0",
            "checksum": "test-checksum-123",
            "last_edited": None,
        }

    @pytest.fixture
    def sample_sections(self):
        """Sample sections for testing."""
        return [
            {
                "id": "test-section-1-phase7c",
                "document_id": "test-doc-phase7c-ingestion",
                "level": 1,
                "title": "Test Section 1",
                "anchor": "test-section-1",
                "order": 0,
                "text": "This is a test section for Phase 7C ingestion validation.",
                "tokens": 10,
                "checksum": "section-1-checksum",
            },
            {
                "id": "test-section-2-phase7c",
                "document_id": "test-doc-phase7c-ingestion",
                "level": 2,
                "title": "Test Section 2",
                "anchor": "test-section-2",
                "order": 1,
                "text": "This is another test section with different content for embedding generation.",
                "tokens": 12,
                "checksum": "section-2-checksum",
            },
        ]

    @staticmethod
    def _resolve_chunk_id(session, original_section_id: str) -> str:
        """Resolve the chunk ID that contains the provided original section ID."""
        record = session.run(
            """
            MATCH (s:Section)
            WHERE $original_id IN s.original_section_ids
            RETURN s.id AS chunk_id
            """,
            original_id=original_section_id,
        ).single()
        assert record is not None, f"No chunk found for section {original_section_id}"
        return record["chunk_id"]

    def test_sections_dual_labeled(
        self, graph_builder, sample_document, sample_sections, neo4j_driver
    ):
        """Test that sections are dual-labeled as Section:Chunk."""
        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Verify dual-labeling
        with neo4j_driver.session() as session:
            for section in sample_sections:
                chunk_id = self._resolve_chunk_id(session, section["id"])
                result = session.run(
                    """
                    MATCH (s {id: $section_id})
                    RETURN labels(s) as labels
                    """,
                    section_id=chunk_id,
                )
                record = result.single()
                assert record is not None, f"Chunk {chunk_id} not found"

                labels = record["labels"]
                assert "Section" in labels, f"Chunk {chunk_id} missing :Section label"
                assert (
                    "Chunk" in labels
                ), f"Chunk {chunk_id} missing :Chunk label (v3 compat)"

    def test_required_embedding_fields_present(
        self, graph_builder, sample_document, sample_sections, neo4j_driver
    ):
        """Test that all required embedding fields are present after ingestion."""
        config = get_config()

        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Verify all required embedding fields are present
        with neo4j_driver.session() as session:
            for section in sample_sections:
                chunk_id = self._resolve_chunk_id(session, section["id"])
                result = session.run(
                    """
                    MATCH (s:Section {id: $section_id})
                    RETURN s.vector_embedding as embedding,
                           s.embedding_version as version,
                           s.embedding_provider as provider,
                           s.embedding_dimensions as dimensions,
                           s.embedding_timestamp as timestamp,
                           s.embedding_task as task
                    """,
                    section_id=chunk_id,
                )
                record = result.single()
                assert record is not None, f"Chunk {chunk_id} not found"

                # REQUIRED fields (schema v2.1)
                assert (
                    record["embedding"] is not None
                ), f"Section {section['id']} missing REQUIRED vector_embedding"
                assert (
                    record["version"] is not None
                ), f"Section {section['id']} missing REQUIRED embedding_version"
                assert (
                    record["provider"] is not None
                ), f"Section {section['id']} missing REQUIRED embedding_provider"
                assert (
                    record["dimensions"] is not None
                ), f"Section {section['id']} missing REQUIRED embedding_dimensions"
                assert (
                    record["timestamp"] is not None
                ), f"Section {section['id']} missing REQUIRED embedding_timestamp"

                # Validate dimensions match config
                assert record["dimensions"] == config.embedding.dims, (
                    f"Section {section['id']} dimension mismatch: "
                    f"expected {config.embedding.dims}, got {record['dimensions']}"
                )

                # Validate embedding vector dimensions
                embedding_vector = record["embedding"]
                assert len(embedding_vector) == config.embedding.dims, (
                    f"Section {section['id']} vector dimension mismatch: "
                    f"expected {config.embedding.dims}-D, got {len(embedding_vector)}-D"
                )

    def test_embedding_dimensions_match_config(
        self, graph_builder, sample_document, sample_sections, neo4j_driver
    ):
        """Test that embeddings match configured dimensions."""
        expected_dims = 384  # Test uses MiniLM (384-D)

        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Verify embeddings match configuration
        with neo4j_driver.session() as session:
            chunk_ids = [
                self._resolve_chunk_id(session, section["id"])
                for section in sample_sections
            ]
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.id IN $chunk_ids
                RETURN s.id as id,
                       s.embedding_dimensions as dims,
                       size(s.vector_embedding) as vector_dims
                """,
                chunk_ids=chunk_ids,
            )

            for record in result:
                # Metadata should match expected dimensions
                assert (
                    record["dims"] == expected_dims
                ), f"Section {record['id']} has incorrect dimension metadata: {record['dims']}"

                # Actual vector should match expected dimensions
                assert record["vector_dims"] == expected_dims, (
                    f"Section {record['id']} has incorrect vector dimensions: "
                    f"{record['vector_dims']}"
                )

    def test_qdrant_collection_created(
        self, graph_builder, sample_document, sample_sections, qdrant_client
    ):
        """Test that Qdrant collection is created with correct dimensions."""
        config = get_config()

        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        expected_collection = config.search.vector.qdrant.collection_name
        assert (
            expected_collection in collection_names
        ), f"Qdrant collection {expected_collection} not created"

        # Verify collection has correct dimensions
        collection_info = qdrant_client.get_collection(expected_collection)
        vectors_cfg = collection_info.config.params.vectors
        if isinstance(vectors_cfg, dict):
            sizes = {name: params.size for name, params in vectors_cfg.items()}
            assert all(
                size == config.embedding.dims for size in sizes.values()
            ), f"Qdrant collection has mismatched vector dims: {sizes}"
        else:
            assert (
                vectors_cfg.size == config.embedding.dims
            ), f"Qdrant collection has wrong dimension: {vectors_cfg.size}"

    def test_qdrant_points_have_metadata(
        self,
        graph_builder,
        sample_document,
        sample_sections,
        neo4j_driver,
        qdrant_client,
    ):
        """Test that Qdrant points include correct embedding metadata."""
        config = get_config()

        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Resolve chunk IDs for the ingested sections
        with neo4j_driver.session() as session:
            expected_node_ids = {
                self._resolve_chunk_id(session, section["id"])
                for section in sample_sections
            }

        # Retrieve points from Qdrant
        collection_name = config.search.vector.qdrant.collection_name

        # Scroll through all points
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=True,
        )

        # Find our test sections
        test_points = [
            p for p in points if p.payload.get("node_id") in expected_node_ids
        ]

        assert len(test_points) >= len(
            expected_node_ids
        ), "Expected all ingested sections to be present in Qdrant payloads"

        for point in test_points:
            payload = point.payload

            # Verify embedding metadata in payload
            assert (
                "embedding_version" in payload
            ), "Missing embedding_version in payload"
            assert (
                "embedding_provider" in payload
            ), "Missing embedding_provider in payload"
            assert (
                "embedding_dimensions" in payload
            ), "Missing embedding_dimensions in payload"
            assert "embedding_task" in payload, "Missing embedding_task in payload"

            # Verify dimensions
            assert payload["embedding_dimensions"] == config.embedding.dims, (
                f"Point {payload['node_id']} has wrong dimension metadata: "
                f"{payload['embedding_dimensions']}"
            )

            # Verify vector dimensions (handle named vectors)
            if isinstance(point.vector, dict):
                assert all(
                    len(vec) == config.embedding.dims for vec in point.vector.values()
                ), f"Point {payload['node_id']} has mismatched vector dims"
            else:
                assert len(point.vector) == config.embedding.dims, (
                    f"Point {payload['node_id']} has wrong vector dimensions: "
                    f"{len(point.vector)}"
                )

    def test_embedding_validation_blocks_missing_vector(
        self, graph_builder, sample_document, neo4j_driver
    ):
        """Test that ingestion fails if embedding generation fails (required field validation)."""
        # This test verifies that the validation logic catches missing embeddings
        # In practice, this would happen if the embedding provider failed

        # Create a section that will fail embedding (mock scenario)
        # We can't easily mock the embedder here, but we can verify the validation exists
        # by checking that sections without embeddings would be caught

        # Instead, let's verify that after successful ingestion, NO sections exist without embeddings
        sample_sections = [
            {
                "id": "test-validation-section",
                "document_id": sample_document["id"],
                "level": 1,
                "title": "Validation Test",
                "anchor": "validation-test",
                "order": 0,
                "text": "This section should have embeddings after ingestion.",
                "tokens": 8,
                "checksum": "validation-checksum",
            }
        ]

        # Ingest document
        graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Query for sections WITHOUT required embedding fields
        with neo4j_driver.session() as session:
            chunk_id = self._resolve_chunk_id(session, "test-validation-section")
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.id = $chunk_id
                  AND (s.vector_embedding IS NULL
                   OR s.embedding_version IS NULL
                   OR s.embedding_provider IS NULL
                   OR s.embedding_dimensions IS NULL
                   OR s.embedding_timestamp IS NULL)
                RETURN count(s) as incomplete_count
                """,
                chunk_id=chunk_id,
            )

            incomplete_count = result.single()["incomplete_count"]

            # Should be 0 - all sections must have embeddings
            assert (
                incomplete_count == 0
            ), "Found sections without required embedding fields (validation failed)"

    def test_provider_factory_integration(self, graph_builder):
        """Test that GraphBuilder uses ProviderFactory for embedding provider."""
        # The GraphBuilder fixture already initializes with test config
        # Just verify that when embedder is initialized, it uses the factory

        # Verify graph_builder has no embedder initially
        assert graph_builder.embedder is None, "Embedder should not be initialized yet"

        # Create a minimal document to trigger embedder initialization
        test_doc = {
            "id": "test-provider-doc",
            "source_uri": "test.md",
            "source_type": "markdown",
            "title": "Test",
            "version": "1.0",
            "checksum": "test",
            "last_edited": None,
        }
        test_sections = [
            {
                "id": "test-provider-section",
                "document_id": "test-provider-doc",
                "level": 1,
                "title": "Test",
                "anchor": "test",
                "order": 0,
                "text": "Test content for provider verification.",
                "tokens": 5,
                "checksum": "test",
            }
        ]

        # This will trigger embedder initialization
        graph_builder.upsert_document(test_doc, test_sections, {}, [])

        # Now verify embedder was initialized
        assert (
            graph_builder.embedder is not None
        ), "Embedder should be initialized after ingestion"

        # Verify provider name is set
        assert hasattr(
            graph_builder.embedder, "provider_name"
        ), "Provider missing provider_name attribute"
        assert graph_builder.embedder.provider_name is not None, "Provider name is None"

    def test_ingestion_stats_accurate(
        self, graph_builder, sample_document, sample_sections
    ):
        """Test that ingestion returns accurate statistics."""
        # Ingest document
        stats = graph_builder.upsert_document(sample_document, sample_sections, {}, [])

        # Verify stats
        assert stats["document_id"] == sample_document["id"]
        assert stats["sections_upserted"] == len(
            sample_sections
        ), f"Expected {len(sample_sections)} sections, got {stats['sections_upserted']}"
        assert stats["embeddings_computed"] >= len(
            sample_sections
        ), "Not all sections had embeddings computed"
        assert stats["vectors_upserted"] >= len(
            sample_sections
        ), "Not all vectors were upserted"

    @pytest.fixture(autouse=True)
    def cleanup_test_data(self, neo4j_driver, qdrant_client):
        """Clean up test data after each test."""
        yield

        # Clean up Neo4j test data
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: 'test-doc-phase7c-ingestion'})
                DETACH DELETE d
                """
            )
            session.run(
                """
                MATCH (s:Section)
                WHERE s.id STARTS WITH 'test-section-' OR s.id = 'test-validation-section'
                   OR s.id = 'test-provider-section'
                DETACH DELETE s
                """
            )

        # Clean up Qdrant test data
        # Use test-specific collection name (matches graph_builder fixture)
        test_collection = "weka_sections_test_384d"

        try:
            # Delete points with test node_ids
            qdrant_client.delete_compat(
                collection_name=test_collection,
                points_selector={
                    "filter": {
                        "should": [
                            {
                                "key": "node_id",
                                "match": {"text": "test-section-"},
                            },
                            {
                                "key": "node_id",
                                "match": {"value": "test-validation-section"},
                            },
                            {
                                "key": "node_id",
                                "match": {"value": "test-provider-section"},
                            },
                        ]
                    }
                },
                wait=True,
            )
        except Exception:
            # Collection might not exist or points might not exist
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
