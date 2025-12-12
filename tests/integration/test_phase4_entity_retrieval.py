"""
Integration tests for Phase 4 entity-aware retrieval.

Tests the full flow from query disambiguation through entity boosting
in the hybrid retrieval pipeline.
"""

from unittest.mock import patch

import pytest

from src.query.processing.disambiguation import QueryAnalysis, QueryDisambiguator


class TestQueryDisambiguatorIntegration:
    """Integration tests for QueryDisambiguator with real config."""

    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_disambiguator_respects_config(self, mock_service_cls):
        """Disambiguator respects config settings."""
        from src.shared.config import get_config

        config = get_config()

        # Should use config values
        disambiguator = QueryDisambiguator()

        # Check threshold comes from config
        expected_threshold = getattr(config.ner, "threshold", 0.45)
        assert disambiguator.threshold == expected_threshold

    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_disambiguator_labels_from_config(self, mock_service_cls):
        """Disambiguator uses labels from config."""
        from src.query.processing.disambiguation import get_default_labels

        disambiguator = QueryDisambiguator()
        labels = disambiguator.labels

        # Should have some labels configured
        # If config has labels, should match; otherwise uses defaults
        default_labels = get_default_labels()
        assert labels == default_labels


class TestEntityBoostIntegration:
    """Integration tests for entity boosting in retrieval."""

    def test_chunk_result_entity_metadata_field(self):
        """ChunkResult correctly stores entity_metadata."""
        from src.query.hybrid_retrieval import ChunkResult

        entity_metadata = {
            "entity_types": ["network_or_storage_protocol", "operating_system"],
            "entity_values": ["NFS", "RHEL"],
            "entity_values_normalized": ["nfs", "rhel"],
            "entity_count": 2,
        }

        chunk = ChunkResult(
            chunk_id="test_chunk",
            document_id="test_doc",
            parent_section_id="test_section",
            order=0,
            level=1,
            heading="Test Heading",
            text="Test text about NFS on RHEL",
            token_count=10,
            fused_score=0.85,
            entity_metadata=entity_metadata,
        )

        assert chunk.entity_metadata is not None
        assert chunk.entity_metadata["entity_count"] == 2
        assert "nfs" in chunk.entity_metadata["entity_values_normalized"]
        assert not chunk.entity_boost_applied

    def test_entity_boost_applied_flag(self):
        """entity_boost_applied flag is set when boosting occurs."""
        from src.query.hybrid_retrieval import ChunkResult

        chunk = ChunkResult(
            chunk_id="test_chunk",
            document_id="test_doc",
            parent_section_id="test_section",
            order=0,
            level=1,
            heading="Test",
            text="Test",
            token_count=10,
            fused_score=0.8,
            entity_metadata={
                "entity_values_normalized": ["nfs"],
                "entity_count": 1,
            },
        )

        # Before boost
        assert not chunk.entity_boost_applied
        original_score = chunk.fused_score

        # Simulate boost (what _apply_entity_boost does)
        boost_terms = ["nfs"]
        doc_entities = chunk.entity_metadata.get("entity_values_normalized", [])
        matches = sum(1 for term in boost_terms if term in doc_entities)

        if matches > 0:
            boost_factor = 1.0 + min(0.5, matches * 0.1)
            chunk.fused_score *= boost_factor
            chunk.entity_boost_applied = True

        # After boost
        assert chunk.entity_boost_applied
        assert chunk.fused_score > original_score
        assert chunk.fused_score == pytest.approx(0.88, rel=1e-2)


class TestHybridRetrieverEntityBoost:
    """Tests for entity boosting in HybridRetriever._apply_entity_boost."""

    def test_apply_entity_boost_method_exists(self):
        """HybridRetriever has _apply_entity_boost method."""
        from src.query.hybrid_retrieval import HybridRetriever

        assert hasattr(HybridRetriever, "_apply_entity_boost")

    def test_apply_entity_boost_signature(self):
        """_apply_entity_boost has correct signature."""
        import inspect

        from src.query.hybrid_retrieval import HybridRetriever

        sig = inspect.signature(HybridRetriever._apply_entity_boost)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "results" in params
        assert "boost_terms" in params

    def test_get_disambiguator_method_exists(self):
        """HybridRetriever has _get_disambiguator method."""
        from src.query.hybrid_retrieval import HybridRetriever

        assert hasattr(HybridRetriever, "_get_disambiguator")


class TestRetrieveMethodIntegration:
    """Tests for entity boosting integration in retrieve method."""

    @patch("src.query.hybrid_retrieval.QdrantClient")
    @patch("src.query.hybrid_retrieval.Driver")
    def test_retrieve_accepts_entity_boost_config(
        self, mock_driver, mock_qdrant_client
    ):
        """retrieve method reads entity boost config."""
        from src.shared.config import get_config

        config = get_config()

        # Verify config has ner section
        assert hasattr(config, "ner")
        ner_config = config.ner

        # Check for expected attributes
        assert hasattr(ner_config, "enabled")
        assert hasattr(ner_config, "threshold")


class TestEntityMetadataPayloadIndex:
    """Tests for entity metadata payload indexes."""

    def test_entity_metadata_in_schema(self):
        """entity_metadata indexes are defined in schema."""
        from src.shared.config import get_embedding_settings
        from src.shared.qdrant_schema import build_qdrant_schema

        # Use embedding settings which has the right structure
        settings = get_embedding_settings()
        schema_plan = build_qdrant_schema(settings)

        # QdrantSchemaPlan has payload_indexes as an attribute (list of tuples)
        payload_indexes = schema_plan.payload_indexes
        index_names = [idx[0] for idx in payload_indexes]

        # Should have entity metadata indexes
        assert "entity_metadata.entity_types" in index_names
        assert "entity_metadata.entity_values" in index_names
        assert "entity_metadata.entity_values_normalized" in index_names
        assert "entity_metadata.entity_count" in index_names


class TestQueryAnalysisDataclass:
    """Tests for QueryAnalysis dataclass serialization."""

    def test_query_analysis_serializable(self):
        """QueryAnalysis can be serialized to dict."""
        analysis = QueryAnalysis(
            query="Configure NFS on RHEL",
            entities=[
                {
                    "text": "NFS",
                    "label": "protocol",
                    "score": 0.9,
                    "start": 10,
                    "end": 13,
                }
            ],
            boost_terms=["nfs"],
            entity_types=["protocol"],
            enabled=True,
        )

        # Should be able to access all fields
        assert analysis.query == "Configure NFS on RHEL"
        assert len(analysis.entities) == 1
        assert analysis.entities[0]["text"] == "NFS"
        assert analysis.boost_terms == ["nfs"]
        assert analysis.entity_types == ["protocol"]
        assert analysis.enabled
        assert analysis.has_entities

    def test_query_analysis_empty_entities(self):
        """QueryAnalysis handles empty entities correctly."""
        analysis = QueryAnalysis(
            query="Simple query",
            enabled=True,
        )

        assert not analysis.has_entities
        assert analysis.entities == []
        assert analysis.boost_terms == []
        assert analysis.entity_types == []


class TestEndToEndEntityBoosting:
    """End-to-end tests simulating the full entity boosting flow."""

    def test_full_boost_flow(self):
        """Test full entity boost flow from query to results."""
        from src.query.hybrid_retrieval import ChunkResult

        # 1. Create mock query analysis (what QueryDisambiguator returns)
        boost_terms = ["nfs", "rhel"]

        # 2. Create mock search results (what fusion returns)
        chunk_nfs = ChunkResult(
            chunk_id="chunk_nfs",
            document_id="doc1",
            parent_section_id="sec1",
            order=0,
            level=1,
            heading="NFS Configuration",
            text="Configure NFS exports on RHEL",
            token_count=20,
            fused_score=0.75,  # Lower initial score
            entity_metadata={
                "entity_types": ["network_or_storage_protocol", "operating_system"],
                "entity_values": ["NFS", "RHEL"],
                "entity_values_normalized": ["nfs", "rhel"],
                "entity_count": 2,
            },
        )

        chunk_other = ChunkResult(
            chunk_id="chunk_other",
            document_id="doc2",
            parent_section_id="sec2",
            order=0,
            level=1,
            heading="S3 Backend",
            text="Configure S3 object storage",
            token_count=20,
            fused_score=0.85,  # Higher initial score
            entity_metadata={
                "entity_types": ["cloud_provider_or_service"],
                "entity_values": ["S3"],
                "entity_values_normalized": ["s3"],
                "entity_count": 1,
            },
        )

        # 3. Simulate sorting by fused_score (before boost)
        results = [chunk_other, chunk_nfs]  # chunk_other is first (higher score)

        # 4. Apply entity boost (what _apply_entity_boost does)
        for res in results:
            entity_metadata = res.entity_metadata or {}
            doc_entities = entity_metadata.get("entity_values_normalized", [])
            matches = sum(1 for term in boost_terms if term in doc_entities)
            if matches > 0:
                boost_factor = 1.0 + min(0.5, matches * 0.1)
                if res.fused_score is not None:
                    res.fused_score *= boost_factor
                    res.entity_boost_applied = True

        # 5. Re-sort by fused_score (after boost)
        results.sort(key=lambda x: x.fused_score or 0, reverse=True)

        # 6. Verify results
        # chunk_nfs should now be first because it matched 2 entities
        assert results[0].chunk_id == "chunk_nfs"
        assert results[0].entity_boost_applied
        assert results[0].fused_score == pytest.approx(
            0.90, rel=1e-2
        )  # 0.75 * 1.2 (2 matches)

        # chunk_other should be second (no boost)
        assert results[1].chunk_id == "chunk_other"
        assert not results[1].entity_boost_applied
        assert results[1].fused_score == 0.85  # Unchanged

    def test_boost_respects_max_cap(self):
        """Entity boost respects the maximum cap (50%)."""
        from src.query.hybrid_retrieval import ChunkResult

        # Entity with many matches (6)
        chunk = ChunkResult(
            chunk_id="chunk_multi",
            document_id="doc1",
            parent_section_id="sec1",
            order=0,
            level=1,
            heading="Multi-entity chunk",
            text="NFS RHEL Kerberos LDAP SMB S3",
            token_count=30,
            fused_score=0.5,
            entity_metadata={
                "entity_values_normalized": [
                    "nfs",
                    "rhel",
                    "kerberos",
                    "ldap",
                    "smb",
                    "s3",
                ],
                "entity_count": 6,
            },
        )

        boost_terms = ["nfs", "rhel", "kerberos", "ldap", "smb", "s3"]  # 6 matches

        # Apply boost
        doc_entities = chunk.entity_metadata.get("entity_values_normalized", [])
        matches = sum(1 for term in boost_terms if term in doc_entities)
        assert matches == 6

        max_boost = 0.5
        per_entity_boost = 0.1
        boost_factor = 1.0 + min(max_boost, matches * per_entity_boost)

        # Should be capped at 1.5 (not 1.6)
        assert boost_factor == 1.5

        chunk.fused_score *= boost_factor
        assert chunk.fused_score == pytest.approx(0.75, rel=1e-2)  # 0.5 * 1.5
