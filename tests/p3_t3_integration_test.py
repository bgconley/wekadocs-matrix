# Phase 3, Task 3.3 Integration Tests - Graph Construction & Embeddings
# NO MOCKS - Tests against live Neo4j and Qdrant services

import time
from pathlib import Path

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.ingestion.build_graph import ingest_document
from src.shared.config import get_config


class TestGraphConstruction:
    """Integration tests for idempotent graph construction with live services."""

    @pytest.fixture(scope="class")
    def config(self):
        """Load configuration."""
        return get_config()

    @pytest.fixture(scope="class")
    def settings(self):
        """Load settings."""
        from src.shared.config import get_settings

        return get_settings()

    @pytest.fixture(scope="class")
    def neo4j_driver(self, settings):
        """Create Neo4j driver."""
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
        )
        yield driver
        driver.close()

    @pytest.fixture(scope="class")
    def qdrant_client(self, config, settings):
        """Create Qdrant client."""
        if config.search.vector.primary == "qdrant":
            client = QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port, timeout=30
            )
            return client
        return None

    @pytest.fixture(scope="function")
    def clean_test_data(self, neo4j_driver, qdrant_client, config):
        """Clean test data before each test."""
        # Clean Neo4j test documents and orphan sections
        with neo4j_driver.session() as session:
            # First get document IDs to delete
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.source_uri CONTAINS 'getting_started.md'
                RETURN collect(d.id) as doc_ids
                """
            )
            record = result.single()
            doc_ids = record["doc_ids"] if record else []

            # Delete documents
            session.run(
                """
                MATCH (d:Document)
                WHERE d.source_uri CONTAINS 'getting_started.md'
                DETACH DELETE d
                """
            )

            # Delete orphan sections (those not connected to any document)
            if doc_ids:
                session.run(
                    """
                    MATCH (s:Section)
                    WHERE s.document_id IN $doc_ids
                    DETACH DELETE s
                    """,
                    doc_ids=doc_ids,
                )

        # Clean Qdrant test vectors if primary
        if qdrant_client and config.search.vector.primary == "qdrant":
            collection_name = "weka_sections"  # From config/development.yaml
            try:
                # Delete ALL vectors with Section label (test environment - safe to clean all)
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector={
                        "filter": {
                            "must": [
                                {"key": "node_label", "match": {"value": "Section"}}
                            ]
                        }
                    },
                )
            except Exception:
                pass  # Collection may not exist yet

        yield

        # Cleanup after test (optional - could keep for debugging)

    def test_idempotent_graph_build(
        self, neo4j_driver, qdrant_client, config, clean_test_data
    ):
        """
        Test that re-ingesting the same document results in stable counts.

        DoD: Ingest document twice; node/edge counts must be identical.
        """
        # Parse sample document
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        source_uri = str(md_path)

        # First ingestion
        ingest_document(source_uri, content, "markdown")
        time.sleep(1)  # Allow indexing to complete

        # Get counts after first ingestion
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[m:MENTIONS]->(e)
                RETURN
                    count(DISTINCT d) as doc_count,
                    count(DISTINCT s) as section_count,
                    count(DISTINCT m) as mention_count,
                    count(DISTINCT e) as entity_count
                """,
                uri=source_uri,
            )
            first_counts = result.single()

        # Get vector counts if Qdrant is primary
        first_vector_count = 0
        if qdrant_client and config.search.vector.primary == "qdrant":
            collection_name = "weka_sections"
            try:
                result = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [{"key": "node_label", "match": {"value": "Section"}}]
                    },
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                first_vector_count = len(result[0])
            except Exception:
                first_vector_count = 0

        # Second ingestion (should be idempotent)
        ingest_document(source_uri, content, "markdown")
        time.sleep(1)

        # Get counts after second ingestion
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[m:MENTIONS]->(e)
                RETURN
                    count(DISTINCT d) as doc_count,
                    count(DISTINCT s) as section_count,
                    count(DISTINCT m) as mention_count,
                    count(DISTINCT e) as entity_count
                """,
                uri=source_uri,
            )
            second_counts = result.single()

        # Get vector counts after second ingestion
        second_vector_count = 0
        if qdrant_client and config.search.vector.primary == "qdrant":
            collection_name = "weka_sections"
            result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [{"key": "node_label", "match": {"value": "Section"}}]
                },
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            second_vector_count = len(result[0])

        # Assertions: counts must be identical
        assert first_counts["doc_count"] == second_counts["doc_count"] == 1
        assert first_counts["section_count"] == second_counts["section_count"]
        assert first_counts["section_count"] > 0, "Should have sections"
        assert first_counts["mention_count"] == second_counts["mention_count"]
        assert first_counts["entity_count"] == second_counts["entity_count"]

        if qdrant_client and config.search.vector.primary == "qdrant":
            assert first_vector_count == second_vector_count
            assert first_vector_count > 0, "Should have vectors"

        print("\n✓ Idempotency verified:")
        print(f"  Documents: {first_counts['doc_count']}")
        print(f"  Sections: {first_counts['section_count']}")
        print(f"  Entities: {first_counts['entity_count']}")
        print(f"  Mentions: {first_counts['mention_count']}")
        if qdrant_client:
            print(f"  Vectors: {first_vector_count}")

    def test_vector_parity(self, neo4j_driver, qdrant_client, config, clean_test_data):
        """
        Test that graph and vector store have matching Section counts.

        DoD: Section count in Neo4j must match vector point count in Qdrant (±0).
        """
        if not qdrant_client or config.search.vector.primary != "qdrant":
            pytest.skip("Qdrant not configured as primary vector store")

        # Ingest sample document
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        source_uri = str(md_path)
        ingest_document(source_uri, content, "markdown")
        time.sleep(1)

        # Count Sections in Neo4j with embedding_version
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.embedding_version = $emb_version
                RETURN count(s) as neo4j_section_count
                """,
                emb_version=config.embedding.version,
            )
            neo4j_count = result.single()["neo4j_section_count"]

        # Count vectors in Qdrant for Sections
        collection_name = "weka_sections"
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {"key": "node_label", "match": {"value": "Section"}},
                    {
                        "key": "embedding_version",
                        "match": {"value": config.embedding.version},
                    },
                ]
            },
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        qdrant_count = len(result[0])

        # Assertions: parity must be exact
        assert (
            neo4j_count == qdrant_count
        ), f"Vector parity failed: Neo4j={neo4j_count}, Qdrant={qdrant_count}"
        assert neo4j_count > 0, "Should have sections with embeddings"

        print("\n✓ Vector parity verified:")
        print(f"  Neo4j Sections: {neo4j_count}")
        print(f"  Qdrant Vectors: {qdrant_count}")
        print("  Delta: 0 (exact match)")
