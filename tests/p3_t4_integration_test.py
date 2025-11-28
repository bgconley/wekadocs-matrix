# Phase 3, Task 3.4 Integration Tests - Incremental Updates & Reconciliation
# NO MOCKS - Tests against live Neo4j and Qdrant services

import time

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.ingestion import ingest_document
from src.ingestion.reconcile import Reconciler
from src.shared.config import get_config


class TestIncrementalUpdates:
    """Integration tests for incremental updates with live services."""

    @pytest.fixture(scope="class")
    def config(self):
        return get_config()

    @pytest.fixture(scope="class")
    def settings(self):
        from src.shared.config import get_settings

        return get_settings()

    @pytest.fixture(scope="class")
    def neo4j_driver(self, settings):
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
        )
        yield driver
        driver.close()

    @pytest.fixture(scope="class")
    def qdrant_client(self, config, settings):
        if config.search.vector.primary == "qdrant":
            return QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port, timeout=30
            )
        return None

    @pytest.fixture(scope="function")
    def clean_test_data(self, neo4j_driver, qdrant_client, config):
        """Clean test data."""
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document)
                WHERE d.source_uri CONTAINS 'test_incremental.md'
                DETACH DELETE d
                """
            )
            session.run(
                """
                MATCH (s:Section)
                WHERE NOT EXISTS { MATCH (doc:Document)-[:HAS_SECTION]->(s) }
                DETACH DELETE s
                """
            )

        if qdrant_client and config.search.vector.primary == "qdrant":
            collection_name = config.search.vector.qdrant.collection_name
            try:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector={
                        "filter": {
                            "must": [
                                {
                                    "key": "document_uri",
                                    "match": {"value": "test_incremental.md"},
                                }
                            ]
                        }
                    },
                )
            except Exception:
                pass

        yield

    def test_incremental_update_minimal_delta(
        self, neo4j_driver, qdrant_client, config, clean_test_data
    ):
        """
        Test that modifying one section results in minimal graph/vector changes.

        DoD: Change 1 section → only that section (and at most immediate dependents) change.
        """
        # Create initial document
        initial_content = """# Test Document

## Section 1
This is the first section with some content.

## Section 2
This is the second section that will be modified.

## Section 3
This is the third section.
"""

        source_uri = "/tmp/test_incremental.md"

        # Initial ingestion
        ingest_document(source_uri, initial_content, "markdown")
        time.sleep(1)

        # Get initial counts and checksums
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                RETURN s.id as section_id, s.checksum as checksum, s.title as title
                ORDER BY s.order
                """,
                uri=source_uri,
            )
            initial_sections = [dict(r) for r in result]

        initial_section_count = len(initial_sections)
        assert (
            initial_section_count >= 3
        ), f"Should have at least 3 sections initially, got {initial_section_count}"

        # Modify only Section 2
        modified_content = """# Test Document

## Section 1
This is the first section with some content.

## Section 2
This is the MODIFIED second section with UPDATED content and more details.

## Section 3
This is the third section.
"""

        # Re-ingest with modification (tests idempotent upsert behavior)
        ingest_document(source_uri, modified_content, "markdown")
        time.sleep(1)

        # Get updated counts and checksums
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                RETURN s.id as section_id, s.checksum as checksum, s.title as title
                ORDER BY s.order
                """,
                uri=source_uri,
            )
            updated_sections = [dict(r) for r in result]

        updated_section_count = len(updated_sections)

        # Build title→checksum maps for comparison (titles are stable)
        initial_by_title = {s["title"]: s["checksum"] for s in initial_sections}
        updated_by_title = {s["title"]: s["checksum"] for s in updated_sections}

        # Count sections with different checksums
        changed_sections = []
        for title in updated_by_title:
            if (
                title in initial_by_title
                and initial_by_title[title] != updated_by_title[title]
            ):
                changed_sections.append(title)

        # Verify only Section 2 changed
        assert "Section 2" in changed_sections, "Section 2 should have changed"
        assert "Section 1" not in changed_sections, "Section 1 should not have changed"
        assert "Section 3" not in changed_sections, "Section 3 should not have changed"

        # Verify delta is minimal (changed + possibly 1-2 new/adjacent)
        delta = abs(updated_section_count - initial_section_count) + len(
            changed_sections
        )
        assert (
            delta <= 3
        ), f"Delta too large: {delta} (section count delta={updated_section_count - initial_section_count}, checksums changed={len(changed_sections)})"

        print("\n✓ Incremental update verified:")
        print(f"  Initial sections: {initial_section_count}")
        print(f"  Updated sections: {updated_section_count}")
        print(
            f"  Checksums changed: {len(changed_sections)} ({', '.join(changed_sections)})"
        )
        print("  Delta: minimal (O(changed sections))")


class TestReconciliation:
    """Integration tests for drift detection and repair."""

    @pytest.fixture(scope="class")
    def config(self):
        return get_config()

    @pytest.fixture(scope="class")
    def settings(self):
        from src.shared.config import get_settings

        return get_settings()

    @pytest.fixture(scope="class")
    def neo4j_driver(self, settings):
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
        )
        yield driver
        driver.close()

    @pytest.fixture(scope="class")
    def qdrant_client(self, config, settings):
        if config.search.vector.primary == "qdrant":
            return QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port, timeout=30
            )
        return None

    @pytest.fixture(scope="function")
    def setup_test_document(self, neo4j_driver, qdrant_client, config):
        """Setup a test document for reconciliation testing."""
        # Clean first
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document)
                WHERE d.source_uri CONTAINS 'test_reconcile.md'
                DETACH DELETE d
                """
            )

        if qdrant_client and config.search.vector.primary == "qdrant":
            collection_name = config.search.vector.qdrant.collection_name
            try:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector={
                        "filter": {
                            "must": [
                                {
                                    "key": "document_uri",
                                    "match": {"value": "test_reconcile.md"},
                                }
                            ]
                        }
                    },
                )
            except Exception:
                pass

        # Ingest test document
        content = """# Reconciliation Test

## Section A
Content for section A.

## Section B
Content for section B.

## Section C
Content for section C.

## Section D
Content for section D.

## Section E
Content for section E.
"""
        source_uri = "/tmp/test_reconcile.md"
        ingest_document(source_uri, content, "markdown")
        time.sleep(1)

        yield source_uri

        # Cleanup after test
        with neo4j_driver.session() as session:
            session.run(
                "MATCH (d:Document {source_uri: $uri}) DETACH DELETE d", uri=source_uri
            )

    def test_reconciliation_drift_repair(
        self, neo4j_driver, qdrant_client, config, setup_test_document
    ):
        """
        Test drift detection and repair.

        DoD: Delete vectors, run reconciliation, verify drift < 0.5% after repair.
        """
        if not qdrant_client or config.search.vector.primary != "qdrant":
            pytest.skip("Qdrant not configured as primary vector store")

        source_uri = setup_test_document

        # Get all section IDs
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                WHERE s.embedding_version = $emb_version
                RETURN s.id as section_id
                """,
                uri=source_uri,
                emb_version=config.embedding.version,
            )
            section_ids = [r["section_id"] for r in result]

        total_sections = len(section_ids)
        assert total_sections >= 5, "Should have at least 5 sections for drift test"

        # Deliberately delete 2 vectors from Qdrant (create drift)
        collection_name = config.search.vector.qdrant.collection_name
        deleted_ids = section_ids[:2]

        for sid in deleted_ids:
            try:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector={
                        "filter": {
                            "must": [{"key": "node_id", "match": {"value": sid}}]
                        }
                    },
                )
            except Exception as e:
                print(f"Warning: Could not delete vector {sid}: {e}")

        time.sleep(0.5)

        # Measure drift before repair
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
        vector_count_before = len(result[0])

        drift_before_pct = (
            (total_sections - vector_count_before) / total_sections
        ) * 100
        print("\n✓ Drift induced:")
        print(f"  Neo4j sections: {total_sections}")
        print(f"  Qdrant vectors (before repair): {vector_count_before}")
        print(f"  Drift before: {drift_before_pct:.2f}%")

        # Run reconciliation
        reconciler = Reconciler(neo4j_driver, config, qdrant_client)
        reconciler.reconcile()
        time.sleep(1)

        # Measure drift after repair
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
        vector_count_after = len(result[0])

        drift_after_pct = ((total_sections - vector_count_after) / total_sections) * 100

        print("\n✓ Reconciliation complete:")
        print(f"  Qdrant vectors (after repair): {vector_count_after}")
        print(f"  Drift after: {drift_after_pct:.2f}%")

        # Assertions: drift should be < 0.5% after repair
        assert (
            drift_after_pct < 0.5
        ), f"Drift after repair ({drift_after_pct:.2f}%) exceeds 0.5% threshold"
        assert (
            vector_count_after >= total_sections - 1
        ), "Should repair most/all missing vectors"
