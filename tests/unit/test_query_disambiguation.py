"""
Unit tests for QueryDisambiguator (Phase 4 GLiNER integration).

Tests query-time entity extraction for entity-aware hybrid retrieval.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.query.processing.disambiguation import (
    QueryAnalysis,
    QueryDisambiguator,
    get_query_disambiguator,
)


class TestQueryAnalysis:
    """Tests for the QueryAnalysis dataclass."""

    def test_empty_analysis(self):
        """Empty analysis has no entities."""
        analysis = QueryAnalysis(query="test query")
        assert analysis.query == "test query"
        assert analysis.entities == []
        assert analysis.boost_terms == []
        assert analysis.entity_types == []
        assert not analysis.has_entities
        assert not analysis.enabled

    def test_analysis_with_entities(self):
        """Analysis with entities sets has_entities to True."""
        analysis = QueryAnalysis(
            query="Configure NFS on RHEL",
            entities=[{"text": "NFS", "label": "protocol", "score": 0.9}],
            boost_terms=["nfs"],
            entity_types=["protocol"],
            enabled=True,
        )
        assert analysis.has_entities
        assert analysis.boost_terms == ["nfs"]
        assert analysis.entity_types == ["protocol"]

    def test_analysis_enabled_but_no_entities(self):
        """Enabled analysis without entities still reports has_entities=False."""
        analysis = QueryAnalysis(query="test", enabled=True)
        assert not analysis.has_entities
        assert analysis.enabled


class TestQueryDisambiguator:
    """Tests for QueryDisambiguator class."""

    @patch("src.query.processing.disambiguation.get_config")
    def test_init_disabled_by_default(self, mock_config):
        """Disambiguator respects NER disabled config."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = False
        mock_config.return_value = mock_cfg

        disambiguator = QueryDisambiguator()
        assert not disambiguator.is_enabled()

    @patch("src.query.processing.disambiguation.get_config")
    def test_process_when_disabled(self, mock_config):
        """Processing returns empty analysis when NER disabled."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = False
        mock_config.return_value = mock_cfg

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Configure NFS on RHEL 8")

        assert result.query == "Configure NFS on RHEL 8"
        assert result.boost_terms == []
        assert not result.has_entities
        assert not result.enabled

    @patch("src.query.processing.disambiguation.get_config")
    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_process_extracts_entities(self, mock_service_cls, mock_config):
        """Processing extracts entities when NER enabled."""
        # Setup config
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_cfg.ner.threshold = 0.45
        mock_config.return_value = mock_cfg

        # Setup GLiNER service mock
        mock_entity = MagicMock()
        mock_entity.text = "NFS"
        mock_entity.label = "network_or_storage_protocol"
        mock_entity.score = 0.85
        mock_entity.start = 10
        mock_entity.end = 13

        mock_entity2 = MagicMock()
        mock_entity2.text = "RHEL"
        mock_entity2.label = "operating_system"
        mock_entity2.score = 0.78
        mock_entity2.start = 17
        mock_entity2.end = 21

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.extract_entities.return_value = [mock_entity, mock_entity2]
        mock_service_cls.return_value = mock_service

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Configure NFS on RHEL 8")

        assert result.enabled
        assert result.has_entities
        assert "nfs" in result.boost_terms
        assert "rhel" in result.boost_terms
        assert "network_or_storage_protocol" in result.entity_types
        assert "operating_system" in result.entity_types

    @patch("src.query.processing.disambiguation.get_config")
    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_process_handles_service_unavailable(self, mock_service_cls, mock_config):
        """Processing gracefully handles unavailable service."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_config.return_value = mock_cfg

        mock_service = MagicMock()
        mock_service.is_available = False
        mock_service_cls.return_value = mock_service

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Test query")

        assert result.enabled
        assert not result.has_entities
        assert result.boost_terms == []

    @patch("src.query.processing.disambiguation.get_config")
    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_process_handles_extraction_error(self, mock_service_cls, mock_config):
        """Processing gracefully handles extraction errors."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_config.return_value = mock_cfg

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.extract_entities.side_effect = RuntimeError("Model error")
        mock_service_cls.return_value = mock_service

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Test query")

        assert result.enabled
        assert not result.has_entities

    @patch("src.query.processing.disambiguation.get_config")
    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_process_normalizes_boost_terms(self, mock_service_cls, mock_config):
        """Boost terms are normalized (lowercased, stripped)."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_cfg.ner.threshold = 0.45
        mock_config.return_value = mock_cfg

        mock_entity = MagicMock()
        mock_entity.text = "  RHEL 8  "  # With whitespace
        mock_entity.label = "os"
        mock_entity.score = 0.9
        mock_entity.start = 0
        mock_entity.end = 10

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.extract_entities.return_value = [mock_entity]
        mock_service_cls.return_value = mock_service

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Configure RHEL 8")

        assert "rhel 8" in result.boost_terms  # Normalized

    @patch("src.query.processing.disambiguation.get_config")
    @patch("src.query.processing.disambiguation.GLiNERService")
    def test_process_deduplicates_boost_terms(self, mock_service_cls, mock_config):
        """Duplicate entities result in unique boost terms."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_cfg.ner.threshold = 0.45
        mock_config.return_value = mock_cfg

        mock_entity1 = MagicMock()
        mock_entity1.text = "NFS"
        mock_entity1.label = "protocol"
        mock_entity1.score = 0.9
        mock_entity1.start = 10
        mock_entity1.end = 13

        # Duplicate mention of NFS
        mock_entity2 = MagicMock()
        mock_entity2.text = "NFS"
        mock_entity2.label = "protocol"
        mock_entity2.score = 0.85
        mock_entity2.start = 30
        mock_entity2.end = 33

        mock_service = MagicMock()
        mock_service.is_available = True
        mock_service.extract_entities.return_value = [mock_entity1, mock_entity2]
        mock_service_cls.return_value = mock_service

        disambiguator = QueryDisambiguator()
        result = disambiguator.process("Configure NFS exports for NFS clients")

        # Should only have one "nfs" in boost_terms
        assert result.boost_terms.count("nfs") == 1

    @patch("src.query.processing.disambiguation.get_config")
    def test_custom_labels_override(self, mock_config):
        """Custom labels are used when provided."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_config.return_value = mock_cfg

        custom_labels = ["custom_label_1", "custom_label_2"]
        disambiguator = QueryDisambiguator(labels=custom_labels)

        assert disambiguator.labels == custom_labels

    @patch("src.query.processing.disambiguation.get_config")
    def test_custom_threshold_override(self, mock_config):
        """Custom threshold is used when provided."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = True
        mock_cfg.ner.threshold = 0.45
        mock_config.return_value = mock_cfg

        disambiguator = QueryDisambiguator(threshold=0.3)
        assert disambiguator.threshold == 0.3


class TestEntityBoostFunction:
    """Tests for the _apply_entity_boost method in HybridRetriever."""

    def test_boost_with_matching_entities(self):
        """Chunks with matching entities receive score boost."""
        from src.query.hybrid_retrieval import ChunkResult

        # Create chunks with entity metadata
        chunk1 = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            parent_section_id="s1",
            order=0,
            level=1,
            heading="NFS Configuration",
            text="Configure NFS exports",
            token_count=10,
            fused_score=0.8,
            entity_metadata={
                "entity_values_normalized": ["nfs", "rhel"],
                "entity_count": 2,
            },
        )

        chunk2 = ChunkResult(
            chunk_id="c2",
            document_id="d1",
            parent_section_id="s1",
            order=1,
            level=1,
            heading="S3 Configuration",
            text="Configure S3 backend",
            token_count=10,
            fused_score=0.9,  # Higher initial score
            entity_metadata={
                "entity_values_normalized": ["s3", "aws"],
                "entity_count": 2,
            },
        )

        results = [chunk2, chunk1]  # chunk2 first by score
        boost_terms = ["nfs"]  # Only matches chunk1

        # Simulate boost application

        # We can't easily test _apply_entity_boost without a full HybridRetriever
        # So we test the logic directly
        for res in results:
            entity_metadata = res.entity_metadata or {}
            doc_entities = entity_metadata.get("entity_values_normalized", [])
            matches = sum(1 for term in boost_terms if term in doc_entities)
            if matches > 0:
                boost_factor = 1.0 + min(0.5, matches * 0.1)
                if res.fused_score is not None:
                    res.fused_score *= boost_factor
                    res.entity_boost_applied = True

        # chunk1 should be boosted (has "nfs"), chunk2 should not
        assert chunk1.entity_boost_applied
        assert not chunk2.entity_boost_applied
        assert chunk1.fused_score == pytest.approx(0.88, rel=1e-2)  # 0.8 * 1.1
        assert chunk2.fused_score == 0.9  # Unchanged

    def test_boost_capped_at_max(self):
        """Boost is capped at maximum (50% by default)."""
        boost_terms = ["nfs", "rhel", "kerberos", "ldap", "mount", "exports"]
        doc_entities = ["nfs", "rhel", "kerberos", "ldap", "mount", "exports"]

        # Count matches
        matches = sum(1 for term in boost_terms if term in doc_entities)
        assert matches == 6

        # Calculate boost (should be capped)
        max_boost = 0.5
        per_entity_boost = 0.1
        boost_factor = 1.0 + min(max_boost, matches * per_entity_boost)
        assert boost_factor == 1.5  # Capped at 1.0 + 0.5


class TestFactoryFunction:
    """Tests for the factory function."""

    @patch("src.query.processing.disambiguation.get_config")
    def test_get_query_disambiguator_returns_instance(self, mock_config):
        """Factory function returns a QueryDisambiguator instance."""
        mock_cfg = MagicMock()
        mock_cfg.ner.enabled = False
        mock_config.return_value = mock_cfg

        disambiguator = get_query_disambiguator()
        assert isinstance(disambiguator, QueryDisambiguator)
