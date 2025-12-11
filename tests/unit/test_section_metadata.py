"""
Unit tests for section metadata helpers (Phase 2: markdown-it-py integration).

Tests the section_metadata.py module which provides utilities for extracting
and propagating structural metadata from parsed sections to chunks.
"""

from src.shared.section_metadata import (
    ENHANCED_METADATA_FIELDS,
    FIELD_BLOCK_TYPES,
    FIELD_CODE_RATIO,
    FIELD_HAS_CODE,
    FIELD_HAS_TABLE,
    FIELD_LINE_END,
    FIELD_LINE_START,
    FIELD_PARENT_PATH,
    build_context_prefix,
    extract_enhanced_metadata,
    get_structural_filter_fields,
    has_enhanced_metadata,
    merge_enhanced_metadata_to_chunk,
)


class TestFieldConstants:
    """Test field name constants are defined correctly."""

    def test_field_constants_are_strings(self):
        """All field constants should be non-empty strings."""
        assert isinstance(FIELD_LINE_START, str) and FIELD_LINE_START
        assert isinstance(FIELD_LINE_END, str) and FIELD_LINE_END
        assert isinstance(FIELD_PARENT_PATH, str) and FIELD_PARENT_PATH
        assert isinstance(FIELD_BLOCK_TYPES, str) and FIELD_BLOCK_TYPES
        assert isinstance(FIELD_CODE_RATIO, str) and FIELD_CODE_RATIO
        assert isinstance(FIELD_HAS_CODE, str) and FIELD_HAS_CODE
        assert isinstance(FIELD_HAS_TABLE, str) and FIELD_HAS_TABLE

    def test_enhanced_metadata_fields_list(self):
        """ENHANCED_METADATA_FIELDS should contain all field constants."""
        # Original Phase 2 fields (7)
        assert FIELD_LINE_START in ENHANCED_METADATA_FIELDS
        assert FIELD_LINE_END in ENHANCED_METADATA_FIELDS
        assert FIELD_PARENT_PATH in ENHANCED_METADATA_FIELDS
        assert FIELD_BLOCK_TYPES in ENHANCED_METADATA_FIELDS
        assert FIELD_CODE_RATIO in ENHANCED_METADATA_FIELDS
        assert FIELD_HAS_CODE in ENHANCED_METADATA_FIELDS
        assert FIELD_HAS_TABLE in ENHANCED_METADATA_FIELDS
        # Phase 5 derived fields (2)
        assert "parent_path_depth" in ENHANCED_METADATA_FIELDS
        assert "block_type" in ENHANCED_METADATA_FIELDS
        # Total: 7 original + 2 Phase 5 = 9
        assert len(ENHANCED_METADATA_FIELDS) == 9


class TestExtractEnhancedMetadata:
    """Test extract_enhanced_metadata function."""

    def test_extract_all_fields_present(self):
        """Should extract all fields when present."""
        section = {
            "line_start": 10,
            "line_end": 25,
            "parent_path": "Getting Started > Installation",
            "block_types": ["paragraph", "code", "paragraph"],
            "code_ratio": 0.35,
            "has_code": True,
            "has_table": False,
        }

        result = extract_enhanced_metadata(section)

        assert result[FIELD_LINE_START] == 10
        assert result[FIELD_LINE_END] == 25
        assert result[FIELD_PARENT_PATH] == "Getting Started > Installation"
        assert result[FIELD_BLOCK_TYPES] == ["paragraph", "code", "paragraph"]
        assert result[FIELD_CODE_RATIO] == 0.35
        assert result[FIELD_HAS_CODE] is True
        assert result[FIELD_HAS_TABLE] is False

    def test_extract_with_defaults_for_missing_fields(self):
        """Should return defaults for missing fields."""
        section = {"title": "Test Section", "text": "Some content"}

        result = extract_enhanced_metadata(section)

        assert result[FIELD_LINE_START] is None
        assert result[FIELD_LINE_END] is None
        assert result[FIELD_PARENT_PATH] == ""
        assert result[FIELD_BLOCK_TYPES] == []
        assert result[FIELD_CODE_RATIO] == 0.0
        assert result[FIELD_HAS_CODE] is False
        assert result[FIELD_HAS_TABLE] is False

    def test_extract_empty_section(self):
        """Should handle empty section dict."""
        result = extract_enhanced_metadata({})

        assert result[FIELD_LINE_START] is None
        assert result[FIELD_PARENT_PATH] == ""
        assert result[FIELD_BLOCK_TYPES] == []


class TestHasEnhancedMetadata:
    """Test has_enhanced_metadata detection function."""

    def test_detects_line_start(self):
        """Should detect presence of line_start as marker."""
        section = {"line_start": 5}
        assert has_enhanced_metadata(section) is True

    def test_detects_parent_path(self):
        """Should detect non-empty parent_path as marker."""
        section = {"parent_path": "Getting Started"}
        assert has_enhanced_metadata(section) is True

    def test_no_enhanced_metadata_in_legacy_section(self):
        """Should return False for legacy parser output."""
        section = {
            "title": "Test",
            "text": "Content",
            "level": 2,
            "order": 0,
        }
        assert has_enhanced_metadata(section) is False

    def test_empty_parent_path_not_marker(self):
        """Empty parent_path should not be considered marker."""
        section = {"parent_path": ""}
        assert has_enhanced_metadata(section) is False


class TestBuildContextPrefix:
    """Test build_context_prefix function."""

    def test_full_path_with_heading(self):
        """Should build full path with parent and heading."""
        result = build_context_prefix("Getting Started", "Prerequisites")
        assert result == "[Section: Getting Started > Prerequisites]"

    def test_heading_only(self):
        """Should handle heading without parent."""
        result = build_context_prefix("", "Installation")
        assert result == "[Section: Installation]"

    def test_parent_path_only(self):
        """Should handle parent path without heading."""
        result = build_context_prefix("Getting Started", "")
        assert result == "[Section: Getting Started]"

    def test_empty_both(self):
        """Should return empty string when both empty."""
        result = build_context_prefix("", "")
        assert result == ""

    def test_without_brackets(self):
        """Should return plain path when include_brackets=False."""
        result = build_context_prefix("Parent", "Child", include_brackets=False)
        assert result == "Parent > Child"

    def test_multi_level_hierarchy(self):
        """Should handle multi-level parent path."""
        result = build_context_prefix("Level1 > Level2", "Level3")
        assert result == "[Section: Level1 > Level2 > Level3]"


class TestMergeEnhancedMetadataToChunk:
    """Test merge_enhanced_metadata_to_chunk function."""

    def test_merge_all_fields(self):
        """Should merge all enhanced fields into chunk."""
        chunk = {"id": "chunk_1", "text": "content"}
        section = {
            "line_start": 10,
            "line_end": 20,
            "parent_path": "Parent > Child",
            "block_types": ["code", "paragraph"],
            "code_ratio": 0.5,
            "has_code": True,
            "has_table": False,
        }

        result = merge_enhanced_metadata_to_chunk(chunk, section)

        assert result[FIELD_LINE_START] == 10
        assert result[FIELD_LINE_END] == 20
        assert result[FIELD_PARENT_PATH] == "Parent > Child"
        assert result[FIELD_BLOCK_TYPES] == ["code", "paragraph"]
        assert result[FIELD_CODE_RATIO] == 0.5
        assert result[FIELD_HAS_CODE] is True
        assert result[FIELD_HAS_TABLE] is False
        # Original fields preserved
        assert result["id"] == "chunk_1"
        assert result["text"] == "content"

    def test_merge_modifies_in_place(self):
        """Should modify chunk dict in place."""
        chunk = {"id": "chunk_1"}
        section = {"line_start": 5}

        result = merge_enhanced_metadata_to_chunk(chunk, section)

        assert result is chunk  # Same object
        assert chunk[FIELD_LINE_START] == 5

    def test_line_number_approximation_for_splits(self):
        """Should approximate line numbers for semantic splits."""
        chunk = {}
        section = {"line_start": 10, "line_end": 30}  # 20 lines

        # First chunk of 4
        result = merge_enhanced_metadata_to_chunk(
            chunk, section, semantic_index=0, semantic_total=4
        )
        assert result[FIELD_LINE_START] == 10  # 10 + 0*(20/4) = 10
        assert result[FIELD_LINE_END] == 15  # 10 + 1*(20/4) = 15

        # Third chunk of 4
        chunk2 = {}
        result2 = merge_enhanced_metadata_to_chunk(
            chunk2, section, semantic_index=2, semantic_total=4
        )
        assert result2[FIELD_LINE_START] == 20  # 10 + 2*(20/4) = 20
        assert result2[FIELD_LINE_END] == 25  # 10 + 3*(20/4) = 25

    def test_line_numbers_passthrough_for_single_chunk(self):
        """Should pass through line numbers unchanged for single chunk."""
        chunk = {}
        section = {"line_start": 10, "line_end": 30}

        result = merge_enhanced_metadata_to_chunk(
            chunk, section, semantic_index=0, semantic_total=1
        )

        assert result[FIELD_LINE_START] == 10
        assert result[FIELD_LINE_END] == 30

    def test_handles_missing_line_numbers(self):
        """Should handle None line numbers gracefully."""
        chunk = {}
        section = {"parent_path": "Test"}  # No line numbers

        result = merge_enhanced_metadata_to_chunk(
            chunk, section, semantic_index=1, semantic_total=3
        )

        assert result[FIELD_LINE_START] is None
        assert result[FIELD_LINE_END] is None


class TestGetStructuralFilterFields:
    """Test get_structural_filter_fields function."""

    def test_returns_filterable_fields(self):
        """Should return fields suitable for Qdrant filtering."""
        fields = get_structural_filter_fields()

        assert FIELD_HAS_CODE in fields
        assert FIELD_HAS_TABLE in fields
        assert FIELD_CODE_RATIO in fields
        assert FIELD_PARENT_PATH in fields
        assert FIELD_LINE_START in fields

    def test_returns_list(self):
        """Should return a list of strings."""
        fields = get_structural_filter_fields()

        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)


class TestChunkUtilsIntegration:
    """Test integration with chunk_utils.py."""

    def test_create_chunk_metadata_accepts_enhanced_fields(self):
        """create_chunk_metadata should accept all enhanced field kwargs."""
        from src.shared.chunk_utils import create_chunk_metadata

        chunk = create_chunk_metadata(
            section_id="sec_1",
            document_id="doc_1",
            level=2,
            order=0,
            heading="Test Section",
            # Phase 2 enhanced fields
            line_start=10,
            line_end=25,
            parent_path="Parent > Child",
            block_types=["code", "paragraph"],
            code_ratio=0.4,
            has_code=True,
            has_table=False,
        )

        assert chunk["line_start"] == 10
        assert chunk["line_end"] == 25
        assert chunk["parent_path"] == "Parent > Child"
        assert chunk["block_types"] == ["code", "paragraph"]
        assert chunk["code_ratio"] == 0.4
        assert chunk["has_code"] is True
        assert chunk["has_table"] is False

    def test_create_chunk_metadata_defaults_for_legacy(self):
        """create_chunk_metadata should have sensible defaults for legacy."""
        from src.shared.chunk_utils import create_chunk_metadata

        chunk = create_chunk_metadata(
            section_id="sec_1",
            document_id="doc_1",
            level=2,
            order=0,
        )

        # Should have defaults, not break
        assert chunk["line_start"] is None
        assert chunk["line_end"] is None
        assert chunk["parent_path"] == ""
        assert chunk["block_types"] == []
        assert chunk["code_ratio"] == 0.0
        assert chunk["has_code"] is False
        assert chunk["has_table"] is False
