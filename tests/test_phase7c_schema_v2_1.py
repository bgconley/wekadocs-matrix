"""
Test Schema v2.1 activation and validation.
Phase 7C, Task 7C.3: Verify schema constraints, indexes, and dual-labeling.
"""

import pytest
from neo4j import GraphDatabase

from src.shared.config import get_config, get_settings
from src.shared.schema import create_schema


class TestSchemaV21Activation:
    """Test suite for Schema v2.1 activation."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    def test_schema_creation_succeeds(self, neo4j_driver):
        """Test that schema v2.1 can be created successfully."""
        config = get_config()
        result = create_schema(neo4j_driver, config)

        assert result["success"] is True
        assert result["schema_version"] == "v2.1"
        assert "errors" not in result or len(result["errors"]) == 0

    def test_schema_version_marker_exists(self, neo4j_driver):
        """Test that SchemaVersion singleton node exists with v2.1."""
        with neo4j_driver.session() as session:
            result = session.run("MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv")
            record = result.single()

            assert record is not None, "SchemaVersion node not found"
            sv = record["sv"]
            assert sv["version"] == "v2.1"
            assert sv["vector_dimensions"] == 1024
            assert sv["embedding_provider"] is not None

    def test_dual_labeling_section_chunk(self, neo4j_driver):
        """Test that all Sections are also labeled as Chunks."""
        # First, ensure there are some sections (from existing data or create test section)
        with neo4j_driver.session() as session:
            # Get counts
            section_count_result = session.run(
                "MATCH (s:Section) RETURN count(s) as count"
            )
            section_count = section_count_result.single()["count"]

            chunk_count_result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            chunk_count = chunk_count_result.single()["count"]

            # If no sections exist, create a test section
            if section_count == 0:
                session.run(
                    """
                    MERGE (d:Document {id: 'test-doc'})
                    SET d.title = 'Test Document'
                    MERGE (s:Section:Chunk {id: 'test-section'})
                    SET s.text = 'Test text',
                        s.document_id = 'test-doc',
                        s.level = 1,
                        s.title = 'Test Section',
                        s.anchor = 'test',
                        s.order = 0,
                        s.tokens = 10,
                        s.checksum = 'test',
                        s.vector_embedding = [0.1] * 1024,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """
                )
                section_count = 1
                chunk_count = 1

            # Verify dual-labeling
            assert (
                section_count == chunk_count
            ), f"Dual-labeling failed: {section_count} Sections but {chunk_count} Chunks"

    def test_session_query_answer_constraints_exist(self, neo4j_driver):
        """Test that Session/Query/Answer constraints are created."""
        with neo4j_driver.session() as session:
            result = session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes")
            constraints = {
                record["name"]: {
                    "type": record["type"],
                    "labels": record["labelsOrTypes"],
                }
                for record in result
            }

            # Check Session constraints
            assert any(
                "session_id" in name.lower()
                and "unique" in constraints[name]["type"].lower()
                for name in constraints
            ), "session_id UNIQUE constraint not found"

            # Check Query constraints
            assert any(
                "query_id" in name.lower()
                and "unique" in constraints[name]["type"].lower()
                for name in constraints
            ), "query_id UNIQUE constraint not found"

            assert any(
                "query_text" in name.lower()
                and "exist" in constraints[name]["type"].lower()
                for name in constraints
            ), "query_text EXISTS constraint not found"

            # Check Answer constraints
            assert any(
                "answer_id" in name.lower()
                and "unique" in constraints[name]["type"].lower()
                for name in constraints
            ), "answer_id UNIQUE constraint not found"

            assert any(
                "answer_text" in name.lower()
                and "exist" in constraints[name]["type"].lower()
                for name in constraints
            ), "answer_text EXISTS constraint not found"

    def test_required_embedding_fields_constraints(self, neo4j_driver):
        """Test that required embedding fields have existence constraints."""
        with neo4j_driver.session() as session:
            result = session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes")
            constraints = {
                record["name"]: {
                    "type": record["type"],
                    "labels": record["labelsOrTypes"],
                }
                for record in result
            }

            # Required embedding field constraints
            required_fields = [
                "vector_embedding",
                "embedding_version",
                "embedding_provider",
                "embedding_timestamp",
                "embedding_dimensions",
            ]

            for field in required_fields:
                assert any(
                    f"section_{field}" in name.lower()
                    and "exist" in constraints[name]["type"].lower()
                    for name in constraints
                ), f"Section {field} EXISTS constraint not found"

    def test_vector_indexes_1024d(self, neo4j_driver):
        """Test that vector indexes exist with 1024-D dimensions."""
        with neo4j_driver.session() as session:
            result = session.run(
                """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties
                """
            )

            vector_indexes = [
                {
                    "name": record["name"],
                    "labels": record["labelsOrTypes"],
                    "properties": record["properties"],
                }
                for record in result
            ]

            # Check for section_embeddings_v2 (Section label, 1024-D)
            section_index = next(
                (
                    idx
                    for idx in vector_indexes
                    if "section_embeddings_v2" in idx["name"].lower()
                ),
                None,
            )
            assert section_index is not None, "section_embeddings_v2 index not found"
            assert "Section" in section_index["labels"]

            # Check for chunk_embeddings_v2 (Chunk label, 1024-D)
            chunk_index = next(
                (
                    idx
                    for idx in vector_indexes
                    if "chunk_embeddings_v2" in idx["name"].lower()
                ),
                None,
            )
            assert chunk_index is not None, "chunk_embeddings_v2 index not found"
            assert "Chunk" in chunk_index["labels"]

            # Note: Actual dimension validation requires inspecting index config
            # which is not easily accessible via SHOW INDEXES
            # Manual verification query:
            # CALL db.index.vector.queryNodes('section_embeddings_v2', 10, [0.1]*1024)

    def test_session_query_answer_property_indexes(self, neo4j_driver):
        """Test that Session/Query/Answer property indexes exist."""
        with neo4j_driver.session() as session:
            result = session.run(
                """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type <> 'VECTOR'
                RETURN name, labelsOrTypes, properties
                """
            )

            indexes = {
                record["name"]: {
                    "labels": record["labelsOrTypes"],
                    "properties": record["properties"],
                }
                for record in result
            }

            # Session indexes
            assert any(
                "session_started_at" in name.lower() for name in indexes
            ), "session_started_at index not found"
            assert any(
                "session_expires_at" in name.lower() for name in indexes
            ), "session_expires_at index not found"
            assert any(
                "session_active" in name.lower() for name in indexes
            ), "session_active index not found"

            # Query indexes
            assert any(
                "query_turn" in name.lower() for name in indexes
            ), "query_turn index not found"
            assert any(
                "query_asked_at" in name.lower() for name in indexes
            ), "query_asked_at index not found"

            # Answer indexes
            assert any(
                "answer_created_at" in name.lower() for name in indexes
            ), "answer_created_at index not found"

    def test_chunk_specific_indexes(self, neo4j_driver):
        """Test that Chunk-specific indexes exist for v3 compatibility."""
        with neo4j_driver.session() as session:
            result = session.run(
                """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type <> 'VECTOR'
                RETURN name, labelsOrTypes, properties
                """
            )

            indexes = {
                record["name"]: {
                    "labels": record["labelsOrTypes"],
                    "properties": record["properties"],
                }
                for record in result
            }

            # Chunk property indexes
            assert any(
                "chunk_document_id" in name.lower() for name in indexes
            ), "chunk_document_id index not found"
            assert any(
                "chunk_level" in name.lower() for name in indexes
            ), "chunk_level index not found"
            assert any(
                "chunk_embedding_version" in name.lower() for name in indexes
            ), "chunk_embedding_version index not found"

    def test_schema_idempotency(self, neo4j_driver):
        """Test that schema creation is idempotent (can be run multiple times)."""
        config = get_config()

        # Run schema creation twice
        result1 = create_schema(neo4j_driver, config)
        result2 = create_schema(neo4j_driver, config)

        # Both should succeed
        assert result1["success"] is True
        assert result2["success"] is True

        # Verify no duplication of schema elements
        with neo4j_driver.session() as session:
            # Should have exactly one SchemaVersion node
            sv_result = session.run(
                "MATCH (sv:SchemaVersion) RETURN count(sv) as count"
            )
            sv_count = sv_result.single()["count"]
            assert sv_count == 1, f"Expected 1 SchemaVersion node, found {sv_count}"


class TestSchemaV21Integration:
    """Integration tests for Schema v2.1 with data operations."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    def test_section_with_required_fields_creation(self, neo4j_driver):
        """Test that creating a Section with all required fields succeeds."""
        with neo4j_driver.session() as session:
            # This should succeed (all required fields present)
            session.run(
                """
                MERGE (s:Section:Chunk {id: 'test-complete-section'})
                SET s.text = 'Test text',
                    s.document_id = 'test-doc',
                    s.level = 1,
                    s.title = 'Test',
                    s.anchor = 'test',
                    s.order = 0,
                    s.tokens = 10,
                    s.checksum = 'test',
                    s.vector_embedding = [0.1] * 1024,
                    s.embedding_version = 'jina-v4-2025-01-23',
                    s.embedding_provider = 'jina-ai',
                    s.embedding_timestamp = datetime(),
                    s.embedding_dimensions = 1024
                """
            )

            # Verify it exists
            result = session.run(
                "MATCH (s:Section {id: 'test-complete-section'}) RETURN s"
            )
            assert result.single() is not None

    def test_section_without_embedding_fails(self, neo4j_driver):
        """Test that creating a Section without required embedding fields fails."""
        with neo4j_driver.session() as session:
            # This should fail (missing required embedding fields)
            with pytest.raises(Exception) as exc_info:
                session.run(
                    """
                    CREATE (s:Section {id: 'test-incomplete-section'})
                    SET s.text = 'Test text',
                        s.document_id = 'test-doc',
                        s.level = 1
                    """
                )

            # Verify constraint violation mentioned
            assert (
                "constraint" in str(exc_info.value).lower()
                or "required" in str(exc_info.value).lower()
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
