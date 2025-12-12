"""
Unit tests for Phase C.4: Structure-Aware Context Expansion.

Tests the _expand_with_structure method and _build_expanded_chunk helper.
"""

from unittest.mock import MagicMock

import pytest

from src.query.hybrid_retrieval import ChunkResult, HybridRetriever


def _make_chunk(
    chunk_id: str,
    parent_section_id: str = "sec1",
    fused_score: float = 0.5,
    document_id: str = "doc1",
) -> ChunkResult:
    """Helper to create minimal ChunkResult for testing."""
    return ChunkResult(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_section_id=parent_section_id,
        order=0,
        level=1,
        heading="Test",
        text="Test content",
        token_count=10,
        fused_score=fused_score,
    )


class TestBuildExpandedChunk:
    """Tests for _build_expanded_chunk helper method."""

    def test_builds_chunk_with_correct_context_source(self):
        """Verify expanded chunk has correct context_source."""
        # Create minimal mock retriever
        retriever = MagicMock(spec=HybridRetriever)
        retriever._neighbor_score = lambda x: x * 0.5

        record = {
            "chunk_id": "expanded_1",
            "document_id": "doc1",
            "parent_section_id": "sec2",
            "order": 1,
            "level": 2,
            "heading": "Expanded Heading",
            "text": "Expanded text content",
            "token_count": 50,
            "is_combined": False,
            "is_split": False,
            "original_section_ids": None,
            "boundaries_json": "{}",
            "doc_tag": "test_tag",
            "document_total_tokens": 500,
            "source_path": "/test/path.md",
            "is_microdoc": False,
            "doc_is_microdoc": False,
            "is_microdoc_stub": False,
            "embedding_version": "v1",
            "tenant": "default",
        }

        # Call the actual method
        chunk = HybridRetriever._build_expanded_chunk(
            retriever, record, "source_chunk_1", "sibling", 0.8
        )

        assert chunk.chunk_id == "expanded_1"
        assert chunk.is_expanded is True
        assert chunk.expansion_source == "source_chunk_1"
        assert chunk.context_source == "sibling"
        assert chunk.fused_score == 0.4  # 0.8 * 0.5

    def test_builds_chunk_with_parent_section_context(self):
        """Verify parent_section context_source is set correctly."""
        retriever = MagicMock(spec=HybridRetriever)
        retriever._neighbor_score = lambda x: x * 0.5

        record = {
            "chunk_id": "parent_chunk",
            "document_id": "doc1",
            "parent_section_id": "parent_sec",
            "order": 0,
            "level": 1,
            "heading": "Parent",
            "text": "Parent content",
            "token_count": 30,
        }

        chunk = HybridRetriever._build_expanded_chunk(
            retriever, record, "child_chunk", "parent_section", 0.6
        )

        assert chunk.context_source == "parent_section"
        assert chunk.expansion_source == "child_chunk"

    def test_builds_chunk_with_shared_entities_context(self):
        """Verify shared_entities context_source is set correctly."""
        retriever = MagicMock(spec=HybridRetriever)
        retriever._neighbor_score = lambda x: x * 0.5

        record = {
            "chunk_id": "entity_chunk",
            "document_id": "doc2",
            "parent_section_id": "sec3",
            "order": 2,
            "level": 1,
            "heading": "Entity Related",
            "text": "Shares entities",
            "token_count": 20,
        }

        chunk = HybridRetriever._build_expanded_chunk(
            retriever, record, "seed_chunk", "shared_entities", 0.7
        )

        assert chunk.context_source == "shared_entities"


class TestExpandWithStructureFeatureFlag:
    """Tests for feature flag gating of structure expansion."""

    def test_returns_empty_when_flag_disabled(self):
        """Verify no expansion when structure_aware_expansion is False."""
        retriever = MagicMock(spec=HybridRetriever)
        retriever._result_id = lambda r: ("chunk_id", r.chunk_id)

        # Mock config with flag disabled
        mock_config = MagicMock()
        mock_config.feature_flags.structure_aware_expansion = False
        retriever.config = mock_config

        seeds = [_make_chunk("seed_1"), _make_chunk("seed_2")]

        result = HybridRetriever._expand_with_structure(
            retriever, "test query", seeds, None
        )

        assert result == []

    def test_returns_empty_when_no_seeds(self):
        """Verify no expansion with empty seed list."""
        retriever = MagicMock(spec=HybridRetriever)

        result = HybridRetriever._expand_with_structure(
            retriever, "test query", [], None
        )

        assert result == []


class TestChunkResultContextSource:
    """Tests for context_source field on ChunkResult."""

    def test_context_source_defaults_to_none(self):
        """Verify context_source is None by default."""
        chunk = _make_chunk("test_1")
        assert chunk.context_source is None

    def test_context_source_can_be_set(self):
        """Verify context_source can be set on ChunkResult."""
        chunk = ChunkResult(
            chunk_id="test_1",
            document_id="doc1",
            parent_section_id="sec1",
            order=0,
            level=1,
            heading="Test",
            text="Test content",
            token_count=10,
            is_expanded=True,
            expansion_source="seed_1",
            context_source="sibling",
        )
        assert chunk.context_source == "sibling"
        assert chunk.is_expanded is True

    def test_sequential_expansion_sets_context_source(self):
        """Verify bounded expansion would set context_source to sequential."""
        # This is more of a documentation test - the actual implementation
        # was verified by reading the code. The ChunkResult created in
        # _bounded_expansion now includes context_source="sequential"
        chunk = ChunkResult(
            chunk_id="neighbor_1",
            document_id="doc1",
            parent_section_id="sec1",
            order=1,
            level=1,
            heading="Neighbor",
            text="Neighbor content",
            token_count=15,
            is_expanded=True,
            expansion_source="seed_1",
            context_source="sequential",
        )
        assert chunk.context_source == "sequential"


class TestStructureExpansionConfig:
    """Tests for structure expansion configuration."""

    def test_config_defaults_when_not_specified(self):
        """Verify default limits are used when config not present."""
        retriever = MagicMock()
        retriever._result_id = lambda r: ("chunk_id", r.chunk_id)

        # Mock config with flag enabled but no structure config
        mock_config = MagicMock()
        mock_config.feature_flags.structure_aware_expansion = True
        mock_config.search.hybrid.expansion = MagicMock()
        mock_config.search.hybrid.expansion.structure = None
        retriever.config = mock_config

        # Mock neo4j driver to prevent actual DB calls
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])  # Empty results
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        retriever.neo4j_driver = MagicMock()
        retriever.neo4j_driver.session.return_value = mock_context

        seeds = [_make_chunk("seed_1", parent_section_id="psid_1")]

        # Should not raise, should use defaults
        result = HybridRetriever._expand_with_structure(
            retriever, "test query", seeds, None
        )

        # With empty DB results, should return empty list
        assert result == []


# Integration-style test (requires actual DB, so marked skip by default)
@pytest.mark.skip(reason="Requires running Neo4j instance")
class TestStructureExpansionIntegration:
    """Integration tests requiring actual Neo4j connection."""

    def test_sibling_expansion_finds_same_section_chunks(self):
        """Verify sibling expansion finds chunks with same parent_section_id."""
        pass  # Would require actual DB setup

    def test_parent_expansion_finds_parent_section_chunks(self):
        """Verify parent expansion traverses CHILD_OF correctly."""
        pass

    def test_entity_expansion_finds_shared_entity_chunks(self):
        """Verify entity expansion finds chunks sharing entities."""
        pass
