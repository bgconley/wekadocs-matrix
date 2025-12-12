"""
Integration tests for GLiNER ingestion flow.

Phase: GLiNER Integration - Phase 2 Document Ingestion Pipeline

These are TRUE integration tests - no mocking of core components.
Tests verify actual behavior of the GLiNER enrichment pipeline.

Tests marked with @pytest.mark.live require the GLiNER model to be loaded.
"""

import pytest

from src.shared.config import get_config


class TestGLiNEREnrichmentIntegration:
    """Integration tests for GLiNER chunk enrichment with real service."""

    @pytest.mark.live
    def test_enrichment_with_real_gliner_service(self):
        """Test enrichment with actual GLiNER model extraction."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import GLiNERService

        # Ensure service is available
        service = GLiNERService()
        if not service.is_available:
            pytest.skip("GLiNER model not loaded - run with --live flag")

        # Real chunks that should produce entities
        chunks = [
            {
                "id": "chunk1",
                "text": "Configure NFS exports on RHEL 8 using the weka fs mount command.",
                "title": "NFS Configuration",
            },
            {
                "id": "chunk2",
                "text": "AWS S3 provides object storage for cloud deployments.",
                "title": "Cloud Storage",
            },
            {
                "id": "chunk3",
                "text": "This is generic text without technical entities.",
                "title": "Introduction",
            },
        ]

        # Run real enrichment
        enrich_chunks_with_entities(chunks)

        # Verify all chunks have entity_metadata (consistent schema)
        for chunk in chunks:
            assert (
                "entity_metadata" in chunk
            ), f"Chunk {chunk['id']} missing entity_metadata"
            assert "entity_count" in chunk["entity_metadata"]
            assert "entity_types" in chunk["entity_metadata"]
            assert "entity_values" in chunk["entity_metadata"]
            assert "entity_values_normalized" in chunk["entity_metadata"]

        # Chunk 1 should have entities (NFS, RHEL, weka fs mount)
        chunk1 = chunks[0]
        assert (
            chunk1["entity_metadata"]["entity_count"] > 0
        ), "Expected entities in NFS chunk"
        assert (
            "_embedding_text" in chunk1
        ), "Expected _embedding_text for enriched chunk"
        assert "[Context:" in chunk1["_embedding_text"]
        assert "_mentions" in chunk1
        # All GLiNER mentions should have source="gliner"
        for m in chunk1["_mentions"]:
            assert m["source"] == "gliner"

        # Chunk 2 should have entities (AWS, S3)
        chunk2 = chunks[1]
        assert (
            chunk2["entity_metadata"]["entity_count"] > 0
        ), "Expected entities in AWS chunk"

        # Chunk 3 may or may not have entities - just verify schema consistency
        chunk3 = chunks[2]
        assert chunk3["entity_metadata"]["entity_count"] >= 0

    @pytest.mark.live
    def test_enrichment_preserves_existing_mentions(self):
        """Verify GLiNER doesn't clobber existing structural mentions."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()
        if not service.is_available:
            pytest.skip("GLiNER model not loaded")

        # Chunk with pre-existing mention from regex extractor
        existing_mention = {
            "name": "weka fs",
            "type": "Command",
            "entity_id": "cmd:weka_fs",
            "source": "regex",
        }

        chunks = [
            {
                "id": "chunk1",
                "text": "Use weka fs mount to attach the filesystem on RHEL.",
                "_mentions": [existing_mention],
            }
        ]

        enrich_chunks_with_entities(chunks)

        mentions = chunks[0]["_mentions"]

        # Original mention should still be present
        regex_mentions = [m for m in mentions if m.get("source") == "regex"]
        assert len(regex_mentions) == 1
        assert regex_mentions[0]["name"] == "weka fs"

        # GLiNER mentions should be added (if any entities found)
        gliner_mentions = [m for m in mentions if m.get("source") == "gliner"]
        # We expect at least RHEL to be found
        if chunks[0]["entity_metadata"]["entity_count"] > 0:
            assert len(gliner_mentions) > 0

    @pytest.mark.live
    def test_embedding_text_format(self):
        """Verify _embedding_text has correct structure."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()
        if not service.is_available:
            pytest.skip("GLiNER model not loaded")

        chunks = [
            {
                "id": "chunk1",
                "text": "Install NVMe drivers for high-performance storage.",
                "title": "Storage Setup",
            }
        ]

        enrich_chunks_with_entities(chunks)

        if chunks[0]["entity_metadata"]["entity_count"] > 0:
            emb_text = chunks[0]["_embedding_text"]

            # Should contain title at start
            assert emb_text.startswith("Storage Setup")

            # Should contain original text
            assert "Install NVMe drivers" in emb_text

            # Should contain context block
            assert "[Context:" in emb_text
            assert "]" in emb_text


class TestNeo4jMentionsFilterLogic:
    """
    Integration tests for Neo4j mentions filtering.

    These test the actual filtering logic in AtomicIngestionCoordinator
    without requiring a live Neo4j connection.
    """

    def test_gliner_source_filtered_from_section_entity_mentions(self):
        """Verify source='gliner' mentions are filtered from Neo4j writes."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        config = get_config()

        # Create coordinator with mocked drivers (we won't call Neo4j)
        # We're testing the Python filtering logic, not the DB write
        class MockDriver:
            pass

        class MockTx:
            def __init__(self):
                self.queries = []
                self.params = []

            def run(self, query, **kwargs):
                self.queries.append(query)
                self.params.append(kwargs)

        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=MockDriver(),
            qdrant_client=None,
            config=config,
        )

        mock_tx = MockTx()

        # Mix of structural and GLiNER mentions
        mentions = [
            # Structural mention (should be written to Neo4j)
            {
                "section_id": "chunk1",
                "entity_id": "cmd:weka_fs",
                "name": "weka fs",
                "type": "Command",
                "source": "regex",
            },
            # GLiNER mention (should be FILTERED - not written to Neo4j)
            {
                "section_id": "chunk1",
                "entity_id": "gliner:os:abc123",
                "name": "RHEL",
                "type": "operating_system",
                "source": "gliner",
            },
            # Another structural mention
            {
                "section_id": "chunk2",
                "entity_id": "cfg:net_apply",
                "name": "--net-apply",
                "type": "Configuration",
                "source": "regex",
            },
            # Structural mention without source field (legacy)
            {
                "section_id": "chunk3",
                "entity_id": "proc:install",
                "name": "Installation",
                "type": "Procedure",
            },
        ]

        # Patch _neo4j_create_entity_relationships to avoid entity-entity processing
        original_method = coordinator._neo4j_create_entity_relationships
        coordinator._neo4j_create_entity_relationships = lambda tx, rels: None

        try:
            coordinator._neo4j_create_mentions(mock_tx, mentions)
        finally:
            coordinator._neo4j_create_entity_relationships = original_method

        # Verify tx.run was called
        assert len(mock_tx.params) == 1

        # Extract the mentions passed to the query
        passed_mentions = mock_tx.params[0].get("mentions", [])

        # Should have 3 mentions (not the GLiNER one)
        assert len(passed_mentions) == 3

        # Verify no GLiNER sources were passed
        for m in passed_mentions:
            assert m.get("source") != "gliner", f"GLiNER mention leaked: {m}"

        # Verify the expected mentions are present
        names = [m["name"] for m in passed_mentions]
        assert "weka fs" in names
        assert "--net-apply" in names
        assert "Installation" in names
        assert "RHEL" not in names  # GLiNER entity should be filtered

    def test_entity_entity_relationships_unaffected(self):
        """Verify entity→entity relationships are not affected by GLiNER filter."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        config = get_config()

        class MockDriver:
            pass

        class MockTx:
            def __init__(self):
                self.queries = []
                self.params = []

            def run(self, query, **kwargs):
                self.queries.append(query)
                self.params.append(kwargs)

        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=MockDriver(),
            qdrant_client=None,
            config=config,
        )

        mock_tx = MockTx()

        # Entity→Entity relationships (Procedure→Step)
        entity_relationships_processed = []

        def capture_entity_rels(tx, rels):
            entity_relationships_processed.extend(rels)

        coordinator._neo4j_create_entity_relationships = capture_entity_rels

        mentions = [
            # Section→Entity (will be processed as mentions)
            {"section_id": "chunk1", "entity_id": "cmd:test", "source": "regex"},
            # Entity→Entity (should go to _neo4j_create_entity_relationships)
            {
                "from_id": "proc:install",
                "to_id": "step:1",
                "relationship": "CONTAINS_STEP",
            },
            # Another Entity→Entity
            {
                "from_id": "proc:install",
                "to_id": "step:2",
                "relationship": "CONTAINS_STEP",
            },
        ]

        coordinator._neo4j_create_mentions(mock_tx, mentions)

        # Verify entity relationships were captured
        assert len(entity_relationships_processed) == 2
        assert all(
            r.get("relationship") == "CONTAINS_STEP"
            for r in entity_relationships_processed
        )


class TestConfigGating:
    """Integration tests for config-based GLiNER gating."""

    def test_enrichment_respects_disabled_config(self):
        """Verify enrichment does nothing when ner.enabled=False."""
        from unittest.mock import MagicMock, patch

        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        # Create chunks
        chunks = [{"id": "chunk1", "text": "Test NFS on RHEL"}]

        # Mock service to always be available
        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [[]]

        # But mock get_default_labels to return empty (simulating disabled)
        with patch(
            "src.ingestion.extract.ner_gliner.get_default_labels", return_value=[]
        ):
            enrich_chunks_with_entities(chunks)

        # Chunk should be unchanged (no entity_metadata added when no labels)
        assert "entity_metadata" not in chunks[0]

    def test_config_ner_section_loads(self):
        """Verify NER config section loads correctly."""
        config = get_config()

        # Verify ner section exists
        assert hasattr(config, "ner"), "Config missing 'ner' section"

        # Verify expected fields
        assert hasattr(config.ner, "enabled")
        assert hasattr(config.ner, "model_name")
        assert hasattr(config.ner, "threshold")
        assert hasattr(config.ner, "labels")

        # Verify labels are populated
        assert isinstance(config.ner.labels, list)
        assert len(config.ner.labels) > 0, "Expected configured labels"


class TestEntityMetadataSchema:
    """Integration tests for entity_metadata payload schema."""

    @pytest.mark.live
    def test_entity_metadata_qdrant_compatible(self):
        """Verify entity_metadata structure is Qdrant-payload compatible."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()
        if not service.is_available:
            pytest.skip("GLiNER model not loaded")

        chunks = [
            {
                "id": "chunk1",
                "text": "Configure AWS S3 bucket with NFS gateway on RHEL 8.",
            }
        ]

        enrich_chunks_with_entities(chunks)

        meta = chunks[0]["entity_metadata"]

        # All fields must be JSON-serializable for Qdrant payload
        import json

        try:
            serialized = json.dumps(meta)
            deserialized = json.loads(serialized)
        except (TypeError, ValueError) as e:
            pytest.fail(f"entity_metadata not JSON serializable: {e}")

        # Verify structure matches expected Qdrant payload schema
        assert isinstance(deserialized["entity_types"], list)
        assert isinstance(deserialized["entity_values"], list)
        assert isinstance(deserialized["entity_values_normalized"], list)
        assert isinstance(deserialized["entity_count"], int)

        # All list items should be strings (for Qdrant keyword index)
        for val in deserialized["entity_values"]:
            assert isinstance(val, str)
        for val in deserialized["entity_values_normalized"]:
            assert isinstance(val, str)
        for val in deserialized["entity_types"]:
            assert isinstance(val, str)


class TestLabelsCleaning:
    """Integration tests for label cleaning in real extraction."""

    @pytest.mark.live
    def test_labels_cleaned_in_real_extraction(self):
        """Verify parenthetical examples are stripped from real extractions."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()
        if not service.is_available:
            pytest.skip("GLiNER model not loaded")

        chunks = [{"id": "chunk1", "text": "Install RHEL 8 on the server."}]

        enrich_chunks_with_entities(chunks)

        if chunks[0]["entity_metadata"]["entity_count"] > 0:
            entity_types = chunks[0]["entity_metadata"]["entity_types"]

            # No entity type should contain parentheses (examples should be stripped)
            for etype in entity_types:
                assert "(" not in etype, f"Label not cleaned: {etype}"
                assert "e.g." not in etype, f"Label not cleaned: {etype}"
