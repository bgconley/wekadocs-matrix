"""
Phase 7E-1 Tests: Dual-Label Idempotent Chunk Ingestion

Tests deterministic ID generation, replace-by-set GC, and idempotency.

Acceptance Criteria:
- Re-ingesting a doc yields no stale nodes/points
- Deterministic IDs (same inputs → same ID)
- Constraints/unique keys hold
"""

import pytest

from src.shared.chunk_utils import (
    create_chunk_metadata,
    generate_chunk_id,
    validate_chunk_schema,
)


class TestChunkIDDeterminism:
    """Test deterministic, order-preserving chunk ID generation."""

    def test_same_inputs_produce_same_id(self):
        """Deterministic: same inputs → same ID."""
        doc_id = "doc_123"
        section_ids = ["sec_1", "sec_2", "sec_3"]

        id1 = generate_chunk_id(doc_id, section_ids)
        id2 = generate_chunk_id(doc_id, section_ids)

        assert id1 == id2, "Same inputs must produce identical IDs"
        assert len(id1) == 64, "Combined chunk ID must be 64-char SHA256 hex"

    def test_order_matters(self):
        """CRITICAL: Order preservation - different orders → different IDs."""
        doc_id = "doc_123"
        section_ids_a = ["sec_1", "sec_2", "sec_3"]
        section_ids_b = ["sec_3", "sec_2", "sec_1"]  # Different order

        id_a = generate_chunk_id(doc_id, section_ids_a)
        id_b = generate_chunk_id(doc_id, section_ids_b)

        assert id_a != id_b, "Different orders must produce different IDs"

    def test_different_documents_different_ids(self):
        """Different documents → different IDs."""
        section_ids = ["sec_1"]

        id1 = generate_chunk_id("doc_123", section_ids)
        id2 = generate_chunk_id("doc_456", section_ids)

        assert id1 != id2, "Different documents must have different chunk IDs"

    def test_different_sections_different_ids(self):
        """Different sections → different IDs."""
        doc_id = "doc_123"

        id1 = generate_chunk_id(doc_id, ["sec_1"])
        id2 = generate_chunk_id(doc_id, ["sec_2"])

        assert id1 != id2, "Different sections must produce different IDs"

    def test_combined_chunks_unique_ids(self):
        """Combined chunks (multiple sections) have unique IDs."""
        doc_id = "doc_123"

        id_single_1 = generate_chunk_id(doc_id, ["sec_1"])
        id_single_2 = generate_chunk_id(doc_id, ["sec_2"])
        id_combined = generate_chunk_id(doc_id, ["sec_1", "sec_2"])

        assert id_single_1 != id_combined
        assert id_single_2 != id_combined
        assert id_single_1 != id_single_2

    def test_empty_inputs_raise_error(self):
        """Empty inputs must raise ValueError."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            generate_chunk_id("", ["sec_1"])

        with pytest.raises(ValueError, match="original_section_ids cannot be empty"):
            generate_chunk_id("doc_123", [])


class TestChunkMetadata:
    """Test chunk metadata creation and validation."""

    def test_create_single_section_chunk(self):
        """Create metadata for single-section chunk (Phase 7E-1 default)."""
        meta = create_chunk_metadata(
            section_id="sec_abc123",
            document_id="doc_xyz789",
            level=2,
            order=5,
            heading="Configuration",
            token_count=450,
        )

        # Verify required fields present
        assert meta["id"] is not None
        # Single-section chunks preserve the original section_id
        assert (
            meta["id"] == "sec_abc123"
        ), "Single-section chunk ID should equal original section_id"
        assert meta["document_id"] == "doc_xyz789"
        assert meta["level"] == 2
        assert meta["order"] == 5
        assert meta["heading"] == "Configuration"
        assert meta["token_count"] == 450

        # Verify chunk-specific fields
        assert meta["is_combined"] is False  # Single-section
        assert meta["is_split"] is False
        assert meta["original_section_ids"] == ["sec_abc123"]
        assert isinstance(meta["boundaries_json"], str)

    def test_validate_chunk_schema_valid(self):
        """Valid chunk passes schema validation."""
        chunk = create_chunk_metadata(
            section_id="sec_1",
            document_id="doc_1",
            level=3,
            order=0,
        )

        assert validate_chunk_schema(chunk) is True

    def test_validate_chunk_schema_missing_fields(self):
        """Invalid chunk fails schema validation."""
        incomplete_chunk = {
            "id": "abc123",
            "document_id": "doc_1",
            # Missing: level, order, original_section_ids, etc.
        }

        assert validate_chunk_schema(incomplete_chunk) is False

    def test_validate_chunk_schema_empty_original_section_ids(self):
        """Chunk with empty original_section_ids fails validation."""
        chunk = create_chunk_metadata(
            section_id="sec_1",
            document_id="doc_1",
            level=3,
            order=0,
        )
        chunk["original_section_ids"] = []  # Invalid

        assert validate_chunk_schema(chunk) is False


class TestReplaceBySetSemantics:
    """
    Test replace-by-set GC semantics.

    These tests verify idempotency at the logic level.
    Full integration tests with Neo4j/Qdrant are in test_phase7e1_integration.py.
    """

    def test_chunk_id_stability_on_reingest(self):
        """Re-ingesting same document produces same chunk IDs."""
        doc_id = "doc_stable"
        sections = ["sec_1", "sec_2", "sec_3"]

        # First ingest
        ids_first = [generate_chunk_id(doc_id, [s]) for s in sections]

        # Re-ingest (simulated)
        ids_second = [generate_chunk_id(doc_id, [s]) for s in sections]

        assert ids_first == ids_second, "Chunk IDs must be stable across re-ingests"

    def test_updated_document_generates_new_ids_for_changed_sections(self):
        """Updated document with different sections gets new IDs."""
        doc_id = "doc_updated"

        # Original sections
        original = ["sec_1", "sec_2", "sec_3"]
        ids_original = [generate_chunk_id(doc_id, [s]) for s in original]

        # Updated sections (sec_2 replaced with sec_2_v2)
        updated = ["sec_1", "sec_2_v2", "sec_3"]
        ids_updated = [generate_chunk_id(doc_id, [s]) for s in updated]

        # sec_1 and sec_3 should have same IDs
        assert ids_original[0] == ids_updated[0]  # sec_1 unchanged
        assert ids_original[2] == ids_updated[2]  # sec_3 unchanged

        # sec_2 vs sec_2_v2 should have different IDs
        assert ids_original[1] != ids_updated[1]  # sec_2 changed


@pytest.mark.integration
class TestChunkIngestionIntegration:
    """
    Integration tests for chunk ingestion with real Neo4j/Qdrant.

    NOTE: These require running services. Run with: pytest -m integration
    """

    @pytest.fixture
    def graph_builder(self, neo4j_driver, qdrant_client, config):
        """Provide configured GraphBuilder for testing."""
        from src.ingestion.build_graph import GraphBuilder

        builder = GraphBuilder(neo4j_driver, config, qdrant_client)
        yield builder
        # Cleanup handled by individual tests

    @pytest.fixture
    def test_doc_id(self):
        """Generate unique test document ID to avoid collisions."""
        import uuid

        return f"test_doc_{uuid.uuid4().hex[:8]}"

    def _create_test_document(self, doc_id: str, num_sections: int = 3) -> tuple:
        """
        Create test document with N sections.

        Returns:
            (document_dict, sections_list, entities_dict, mentions_list)
        """
        from datetime import datetime

        document = {
            "id": doc_id,
            "title": f"Test Document {doc_id}",
            "source_uri": f"https://test.com/{doc_id}",
            "source_type": "test",
            "version": "1.0",
            "checksum": f"sha256_{doc_id}",
            "last_edited": datetime.utcnow().isoformat(),
            "token_count": num_sections * 100,
        }

        sections = []
        for i in range(num_sections):
            section_id = f"{doc_id}_sec_{i}"
            sections.append(
                {
                    "id": section_id,
                    "title": f"Section {i}",
                    "text": f"This is test section {i} content. " * 20,  # ~20 tokens
                    "level": 2,
                    "order": i,
                    "token_count": 100,
                    "parent_section_id": None,
                }
            )

        # Empty entities and mentions for simplicity
        entities = {}
        mentions = []

        return document, sections, entities, mentions

    def _count_chunks_neo4j(self, neo4j_driver, doc_id: str) -> int:
        """Count chunks for a document in Neo4j."""
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:Chunk {document_id: $doc_id}) RETURN count(c) as cnt",
                doc_id=doc_id,
            )
            return result.single()["cnt"]

    def _get_chunk_ids_neo4j(self, neo4j_driver, doc_id: str) -> list:
        """Get sorted list of chunk IDs for a document from Neo4j."""
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:Chunk {document_id: $doc_id}) RETURN c.id as id ORDER BY c.id",
                doc_id=doc_id,
            )
            return [record["id"] for record in result]

    def _count_points_qdrant(self, qdrant_client, doc_id: str) -> int:
        """Count points for a document in Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filter_doc = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        )

        # Use scroll to count points (use correct collection name from config)
        collection_name = "chunks"  # Phase 7E-1: New chunks collection
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_doc,
            limit=1000,
            with_payload=False,
            with_vectors=False,
        )

        return len(points)

    def _cleanup_test_document(self, neo4j_driver, qdrant_client, doc_id: str):
        """Clean up test document from both stores."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Delete from Neo4j
        with neo4j_driver.session() as session:
            session.run(
                "MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id
            )
            session.run(
                "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c", doc_id=doc_id
            )

        # Delete from Qdrant (use correct collection name)
        filter_doc = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        )

        collection_name = "chunks"  # Phase 7E-1: New chunks collection
        try:
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=filter_doc,
                wait=True,
            )
        except Exception:
            pass  # Collection might not exist yet

    def test_replace_by_set_no_stale_chunks_neo4j(
        self, graph_builder, neo4j_driver, test_doc_id
    ):
        """
        Replace-by-set GC: Re-ingesting leaves no stale chunks in Neo4j.

        Acceptance: Re-ingesting a doc yields no stale nodes.
        """
        try:
            # Step 1: Ingest document with 3 sections → expect 3 chunks
            doc, sections, entities, mentions = self._create_test_document(
                test_doc_id, 3
            )
            graph_builder.upsert_document(doc, sections, entities, mentions)

            # Verify: 3 chunks exist
            count = self._count_chunks_neo4j(neo4j_driver, test_doc_id)
            assert count == 3, f"Expected 3 chunks, got {count}"

            # Step 2: Re-ingest same document → still expect 3 chunks (no duplicates)
            graph_builder.upsert_document(doc, sections, entities, mentions)

            # Verify: still 3 chunks (idempotent)
            count = self._count_chunks_neo4j(neo4j_driver, test_doc_id)
            assert (
                count == 3
            ), f"After re-ingest: expected 3 chunks, got {count} (duplicates created!)"

            # Step 3: Update document to 2 sections → expect 2 chunks
            doc_v2, sections_v2, entities_v2, mentions_v2 = self._create_test_document(
                test_doc_id, 2
            )
            graph_builder.upsert_document(doc_v2, sections_v2, entities_v2, mentions_v2)

            # Verify: exactly 2 chunks (third chunk was deleted)
            count = self._count_chunks_neo4j(neo4j_driver, test_doc_id)
            assert count == 2, f"After update: expected 2 chunks, got {count}"

            # Step 4: Verify no orphaned chunks remain
            with neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (c:Chunk) WHERE NOT (c)<-[:HAS_SECTION]-() RETURN count(c) as cnt"
                )
                orphans = result.single()["cnt"]
                assert orphans == 0, f"Found {orphans} orphaned chunks"

        finally:
            # Cleanup
            self._cleanup_test_document(neo4j_driver, None, test_doc_id)

    def test_replace_by_set_no_stale_points_qdrant(
        self, graph_builder, neo4j_driver, qdrant_client, test_doc_id
    ):
        """
        Replace-by-set GC: Re-ingesting leaves no stale points in Qdrant.

        Acceptance: Re-ingesting a doc yields no stale points.
        """
        try:
            # Step 1: Ingest document with 3 sections → expect 3 points
            doc, sections, entities, mentions = self._create_test_document(
                test_doc_id, 3
            )
            graph_builder.upsert_document(doc, sections, entities, mentions)

            # Count points for this document
            count = self._count_points_qdrant(qdrant_client, test_doc_id)
            assert count == 3, f"Expected 3 points, got {count}"

            # Step 2: Re-ingest same document → still expect 3 points (no duplicates)
            graph_builder.upsert_document(doc, sections, entities, mentions)

            count = self._count_points_qdrant(qdrant_client, test_doc_id)
            assert (
                count == 3
            ), f"After re-ingest: expected 3 points, got {count} (duplicates created!)"

            # Step 3: Update document to 2 sections → expect 2 points
            doc_v2, sections_v2, entities_v2, mentions_v2 = self._create_test_document(
                test_doc_id, 2
            )
            graph_builder.upsert_document(doc_v2, sections_v2, entities_v2, mentions_v2)

            # Verify: exactly 2 points (third point was deleted)
            count = self._count_points_qdrant(qdrant_client, test_doc_id)
            assert count == 2, f"After update: expected 2 points, got {count}"

        finally:
            # Cleanup
            self._cleanup_test_document(neo4j_driver, qdrant_client, test_doc_id)

    def test_idempotency_multiple_reingests(
        self, graph_builder, neo4j_driver, qdrant_client, test_doc_id
    ):
        """
        Idempotency: Multiple re-ingests produce identical state.

        Acceptance: Re-ingesting is idempotent.
        """
        try:
            # Step 1: Initial ingest
            doc, sections, entities, mentions = self._create_test_document(
                test_doc_id, 3
            )
            graph_builder.upsert_document(doc, sections, entities, mentions)

            # Snapshot state
            chunk_ids_initial = self._get_chunk_ids_neo4j(neo4j_driver, test_doc_id)
            point_count_initial = self._count_points_qdrant(qdrant_client, test_doc_id)

            assert len(chunk_ids_initial) == 3, "Initial state should have 3 chunks"
            assert point_count_initial == 3, "Initial state should have 3 points"

            # Step 2: Re-ingest 5 times
            for i in range(5):
                graph_builder.upsert_document(doc, sections, entities, mentions)

                # Verify state unchanged
                chunk_ids = self._get_chunk_ids_neo4j(neo4j_driver, test_doc_id)
                point_count = self._count_points_qdrant(qdrant_client, test_doc_id)

                assert (
                    chunk_ids == chunk_ids_initial
                ), f"Neo4j state changed on iteration {i + 1}: {chunk_ids} != {chunk_ids_initial}"
                assert (
                    point_count == point_count_initial
                ), f"Qdrant state changed on iteration {i + 1}: {point_count} != {point_count_initial}"

            # Step 3: Final verification - counts stable
            assert len(chunk_ids_initial) == 3
            assert point_count_initial == 3

        finally:
            # Cleanup
            self._cleanup_test_document(neo4j_driver, qdrant_client, test_doc_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
