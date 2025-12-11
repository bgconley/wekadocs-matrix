"""
Unit tests for GLiNER chunk enrichment module.

Phase: GLiNER Integration - Phase 2 Document Ingestion Pipeline

Tests verify:
- Basic enrichment flow (with mocked service)
- Empty chunks handling
- Service unavailable handling
- No labels configured
- Deduplication of existing mentions
- Entity metadata structure
- _embedding_text format
- _mentions format with source="gliner" marker
"""

from unittest.mock import MagicMock, patch


class TestEnrichChunksBasic:
    """Tests for basic enrich_chunks_with_entities functionality."""

    def test_empty_chunks_list(self):
        """Verify empty list is handled gracefully."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        chunks = []
        enrich_chunks_with_entities(chunks)
        assert chunks == []

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_service_unavailable(self, mock_labels, mock_service_class):
        """Verify graceful degradation when service is unavailable."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        mock_service = MagicMock()
        mock_service.is_available = False
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["test_label"]

        chunks = [{"id": "chunk1", "text": "Test text"}]
        enrich_chunks_with_entities(chunks)

        # Service was checked but batch_extract not called
        assert mock_service.batch_extract_entities.call_count == 0
        # Chunk should be unchanged (no entity_metadata added)
        assert "entity_metadata" not in chunks[0]

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_no_labels_configured(self, mock_labels, mock_service_class):
        """Verify graceful handling when no labels configured."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service_class.return_value = mock_service
        mock_labels.return_value = []  # No labels

        chunks = [{"id": "chunk1", "text": "Test text"}]
        enrich_chunks_with_entities(chunks)

        # batch_extract should not be called
        assert mock_service.batch_extract_entities.call_count == 0

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_batch_extraction_failure(self, mock_labels, mock_service_class):
        """Verify graceful handling when batch extraction fails."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.side_effect = RuntimeError("Model crashed")
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["test_label"]

        chunks = [{"id": "chunk1", "text": "Test text"}]
        # Should not raise
        enrich_chunks_with_entities(chunks)

        # Chunk should be unchanged
        assert "entity_metadata" not in chunks[0]


class TestEnrichChunksWithEntities:
    """Tests for successful entity enrichment."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_enrichment_adds_entity_metadata(self, mock_labels, mock_service_class):
        """Verify entity_metadata is added correctly."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [
                Entity("NFS", "network_or_storage_protocol", 10, 13, 0.85),
                Entity("RHEL", "operating_system", 20, 24, 0.90),
            ]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["operating_system", "network_or_storage_protocol"]

        chunks = [{"id": "chunk1", "text": "Configure NFS on RHEL"}]
        enrich_chunks_with_entities(chunks)

        assert "entity_metadata" in chunks[0]
        meta = chunks[0]["entity_metadata"]
        assert meta["entity_count"] == 2
        assert "NFS" in meta["entity_values"]
        assert "RHEL" in meta["entity_values"]
        assert "nfs" in meta["entity_values_normalized"]
        assert "rhel" in meta["entity_values_normalized"]
        assert set(meta["entity_types"]) == {
            "network_or_storage_protocol",
            "operating_system",
        }

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_enrichment_adds_embedding_text_with_title(
        self, mock_labels, mock_service_class
    ):
        """Verify _embedding_text includes title and entity context."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("NFS", "protocol", 10, 13, 0.85)]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["protocol"]

        chunks = [{"id": "chunk1", "text": "Configure NFS mount", "title": "NFS Setup"}]
        enrich_chunks_with_entities(chunks)

        assert "_embedding_text" in chunks[0]
        emb_text = chunks[0]["_embedding_text"]
        assert "NFS Setup" in emb_text
        assert "Configure NFS mount" in emb_text
        assert "[Context:" in emb_text
        assert "protocol: NFS" in emb_text

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_enrichment_adds_embedding_text_without_title(
        self, mock_labels, mock_service_class
    ):
        """Verify _embedding_text works without title."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("AWS", "cloud_provider", 5, 8, 0.95)]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["cloud_provider"]

        chunks = [{"id": "chunk1", "text": "Use AWS S3 bucket"}]
        enrich_chunks_with_entities(chunks)

        emb_text = chunks[0]["_embedding_text"]
        assert emb_text.startswith("Use AWS S3 bucket")
        assert "[Context: cloud_provider: AWS]" in emb_text

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_enrichment_uses_heading_fallback(self, mock_labels, mock_service_class):
        """Verify heading is used when title is missing."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("NVMe", "hardware", 0, 4, 0.88)]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["hardware"]

        chunks = [
            {"id": "chunk1", "text": "NVMe drives", "heading": "Storage Hardware"}
        ]
        enrich_chunks_with_entities(chunks)

        emb_text = chunks[0]["_embedding_text"]
        assert "Storage Hardware" in emb_text


class TestMentionsEnrichment:
    """Tests for _mentions field enrichment."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_mentions_added_with_gliner_source(self, mock_labels, mock_service_class):
        """Verify _mentions includes source='gliner' marker."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("weka-agent", "weka_software_component", 0, 10, 0.82)]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["weka_software_component"]

        chunks = [{"id": "chunk1", "text": "Install weka-agent"}]
        enrich_chunks_with_entities(chunks)

        assert "_mentions" in chunks[0]
        mentions = chunks[0]["_mentions"]
        assert len(mentions) == 1

        m = mentions[0]
        assert m["name"] == "weka-agent"
        assert m["type"] == "weka_software_component"
        assert m["source"] == "gliner"  # Key marker for Neo4j filtering
        assert m["confidence"] == 0.82
        assert m["entity_id"].startswith("gliner:")

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_mentions_deduplication(self, mock_labels, mock_service_class):
        """Verify GLiNER entities don't duplicate existing mentions."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [
                Entity("NFS", "protocol", 0, 3, 0.85),
                Entity("SMB", "protocol", 10, 13, 0.80),
            ]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["protocol"]

        # Chunk already has NFS mention from regex extractor
        existing_mention = {
            "name": "NFS",
            "type": "protocol",
            "entity_id": "regex:proto:abc123",
            "source": "regex",
        }
        chunks = [
            {
                "id": "chunk1",
                "text": "NFS and SMB protocols",
                "_mentions": [existing_mention],
            }
        ]
        enrich_chunks_with_entities(chunks)

        mentions = chunks[0]["_mentions"]
        # Should have 2: original NFS + new SMB (NFS deduplicated)
        assert len(mentions) == 2

        # Original NFS should still be there with regex source
        nfs_mentions = [m for m in mentions if m["name"] == "NFS"]
        assert len(nfs_mentions) == 1
        assert nfs_mentions[0]["source"] == "regex"

        # SMB should be added with gliner source
        smb_mentions = [m for m in mentions if m["name"] == "SMB"]
        assert len(smb_mentions) == 1
        assert smb_mentions[0]["source"] == "gliner"

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_mentions_case_insensitive_dedup(self, mock_labels, mock_service_class):
        """Verify deduplication is case-insensitive."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("nfs", "protocol", 0, 3, 0.85)]  # lowercase
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["protocol"]

        existing_mention = {"name": "NFS", "type": "PROTOCOL"}  # uppercase
        chunks = [
            {"id": "chunk1", "text": "nfs mount", "_mentions": [existing_mention]}
        ]
        enrich_chunks_with_entities(chunks)

        # Should still be just 1 mention (case-insensitive dedup)
        assert len(chunks[0]["_mentions"]) == 1


class TestEmptyEntityMetadata:
    """Tests for consistent entity_metadata on chunks without entities."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_empty_metadata_for_no_entities(self, mock_labels, mock_service_class):
        """Verify entity_metadata is set even when no entities found."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [[]]  # No entities
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["some_label"]

        chunks = [{"id": "chunk1", "text": "Generic text with no entities"}]
        enrich_chunks_with_entities(chunks)

        # entity_metadata should still be present with empty values
        assert "entity_metadata" in chunks[0]
        meta = chunks[0]["entity_metadata"]
        assert meta["entity_count"] == 0
        assert meta["entity_types"] == []
        assert meta["entity_values"] == []
        assert meta["entity_values_normalized"] == []

        # No _embedding_text for chunks without entities
        assert "_embedding_text" not in chunks[0]


class TestLabelCleaning:
    """Tests for label cleaning (stripping parenthetical examples)."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_label_examples_stripped(self, mock_labels, mock_service_class):
        """Verify parenthetical examples are stripped from labels."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        # Simulate GLiNER returning label with examples (as configured)
        mock_service.batch_extract_entities.return_value = [
            [Entity("RHEL", "operating_system (e.g. RHEL, Ubuntu)", 0, 4, 0.90)]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["operating_system (e.g. RHEL, Ubuntu)"]

        chunks = [{"id": "chunk1", "text": "RHEL 8 server"}]
        enrich_chunks_with_entities(chunks)

        # Label should be cleaned
        assert "operating_system" in chunks[0]["entity_metadata"]["entity_types"]
        assert (
            "operating_system (e.g. RHEL, Ubuntu)"
            not in chunks[0]["entity_metadata"]["entity_types"]
        )


class TestMultipleChunks:
    """Tests for batch processing of multiple chunks."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_multiple_chunks_enriched(self, mock_labels, mock_service_class):
        """Verify multiple chunks are enriched in batch."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("NFS", "protocol", 0, 3, 0.85)],  # chunk1
            [],  # chunk2 - no entities
            [Entity("AWS", "cloud", 0, 3, 0.90)],  # chunk3
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["protocol", "cloud"]

        chunks = [
            {"id": "chunk1", "text": "NFS mount"},
            {"id": "chunk2", "text": "Generic text"},
            {"id": "chunk3", "text": "AWS S3 bucket"},
        ]
        enrich_chunks_with_entities(chunks)

        # All chunks should have entity_metadata
        assert chunks[0]["entity_metadata"]["entity_count"] == 1
        assert chunks[1]["entity_metadata"]["entity_count"] == 0
        assert chunks[2]["entity_metadata"]["entity_count"] == 1

        # Only chunks with entities should have _embedding_text
        assert "_embedding_text" in chunks[0]
        assert "_embedding_text" not in chunks[1]
        assert "_embedding_text" in chunks[2]


class TestEntityExclusions:
    """Tests for entity exclusion filtering (domain stopwords like 'WEKA')."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_weka_entities_excluded(self, mock_labels, mock_service_class):
        """Verify WEKA and variants are filtered from enrichment."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        # GLiNER might extract WEKA as a component - we want to filter it
        mock_service.batch_extract_entities.return_value = [
            [
                Entity("WEKA", "COMPONENT", 0, 4, 0.95),
                Entity("NFS", "PROTOCOL", 10, 13, 0.85),
                Entity("Weka", "COMPONENT", 20, 24, 0.90),
            ]
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["COMPONENT", "PROTOCOL"]

        chunks = [{"id": "chunk1", "text": "WEKA NFS mount with Weka"}]
        enrich_chunks_with_entities(chunks)

        # WEKA/Weka should be filtered out, only NFS remains
        meta = chunks[0]["entity_metadata"]
        assert meta["entity_count"] == 1
        assert "NFS" in meta["entity_values"]
        assert "WEKA" not in meta["entity_values"]
        assert "Weka" not in meta["entity_values"]
        assert "weka" not in meta["entity_values_normalized"]

    def test_is_excluded_entity_function(self):
        """Test the is_excluded_entity helper directly."""
        from src.providers.ner.labels import is_excluded_entity

        # Should be excluded
        assert is_excluded_entity("WEKA") is True
        assert is_excluded_entity("Weka") is True
        assert is_excluded_entity("weka") is True
        assert is_excluded_entity("WekaFS") is True
        assert is_excluded_entity("  WEKA  ") is True  # With whitespace

        # Should NOT be excluded
        assert is_excluded_entity("NFS") is False
        assert is_excluded_entity("AWS") is False
        assert is_excluded_entity("weka-agent") is False  # Compound term
        assert is_excluded_entity("weka fs") is False  # Command with space


class TestEntityIdGeneration:
    """Tests for deterministic entity_id generation."""

    @patch("src.ingestion.extract.ner_gliner.GLiNERService")
    @patch("src.ingestion.extract.ner_gliner.get_default_labels")
    def test_entity_id_is_deterministic(self, mock_labels, mock_service_class):
        """Verify same entity text+label produces same entity_id."""
        from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities
        from src.providers.ner.gliner_service import Entity

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.batch_extract_entities.return_value = [
            [Entity("NFS", "protocol", 0, 3, 0.85)],
            [Entity("NFS", "protocol", 5, 8, 0.90)],  # Same entity, different position
        ]
        mock_service_class.return_value = mock_service
        mock_labels.return_value = ["protocol"]

        chunks = [
            {"id": "chunk1", "text": "NFS mount"},
            {"id": "chunk2", "text": "Use NFS"},
        ]
        enrich_chunks_with_entities(chunks)

        eid1 = chunks[0]["_mentions"][0]["entity_id"]
        eid2 = chunks[1]["_mentions"][0]["entity_id"]

        # Same entity text+label should produce same entity_id
        assert eid1 == eid2
        assert eid1.startswith("gliner:protocol:")
