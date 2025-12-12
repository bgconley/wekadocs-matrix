"""
Unit tests for Phase 4: Parser shadow mode and migration testing.

Tests cover:
    - ParserComparisonResult dataclass behavior
    - compare_parser_results() comparison logic
    - ShadowModeError exception handling
    - fail_on_mismatch integration
    - Shadow mode routing logic
"""

from unittest.mock import patch

import pytest

from src.ingestion.parsers.shadow_comparison import (
    ParserComparisonResult,
    ShadowModeError,
    compare_parser_results,
    log_comparison_result,
)


class TestParserComparisonResult:
    """Tests for ParserComparisonResult dataclass."""

    def test_default_values(self):
        """Result should initialize with sensible defaults."""
        result = ParserComparisonResult(source_uri="test://doc.md")

        assert result.source_uri == "test://doc.md"
        assert result.has_differences is False
        assert result.section_count_legacy == 0
        assert result.section_count_mit == 0
        assert result.title_differences == []
        assert result.document_title_differs is False
        assert result.new_metadata_fields == []

    def test_section_count_differs_property(self):
        """section_count_differs should compare counts correctly."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=5,
        )
        assert result.section_count_differs is False

        result.section_count_mit = 6
        assert result.section_count_differs is True

    def test_section_count_delta_property(self):
        """section_count_delta should be (mit - legacy)."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=7,
        )
        assert result.section_count_delta == 2

        result.section_count_mit = 3
        assert result.section_count_delta == -2

    def test_summary_no_differences(self):
        """summary() should return 'no differences' when parsers match."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=5,
        )
        assert result.summary() == "no differences"

    def test_summary_with_section_count_diff(self):
        """summary() should include section count difference."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=7,
        )
        summary = result.summary()
        assert "section_count: 5 → 7" in summary
        assert "delta: +2" in summary

    def test_summary_with_title_diffs(self):
        """summary() should include title diff count."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=5,
            title_differences=[(0, "Foo", "Bar"), (1, "Baz", "Qux")],
        )
        assert "title_diffs: 2" in result.summary()

    def test_summary_with_doc_title_diff(self):
        """summary() should include document title difference."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            section_count_legacy=5,
            section_count_mit=5,
            document_title_differs=True,
            legacy_doc_title="Old Title",
            mit_doc_title="New Title",
        )
        assert "doc_title: 'Old Title' → 'New Title'" in result.summary()

    def test_to_dict(self):
        """to_dict() should serialize all key metrics."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            has_differences=True,
            section_count_legacy=5,
            section_count_mit=7,
            title_differences=[(0, "A", "B")],
            new_metadata_fields=["line_start", "parent_path"],
        )
        d = result.to_dict()

        assert d["source_uri"] == "test://doc.md"
        assert d["has_differences"] is True
        assert d["section_count_legacy"] == 5
        assert d["section_count_mit"] == 7
        assert d["section_count_delta"] == 2
        assert d["title_differences_count"] == 1
        assert d["new_metadata_fields"] == ["line_start", "parent_path"]


class TestCompareParserResults:
    """Tests for compare_parser_results() function."""

    def test_identical_results(self):
        """Should detect when parsers produce identical output."""
        legacy = {
            "Document": {"title": "Test Doc"},
            "Sections": [
                {"title": "Section 1", "text": "Content 1"},
                {"title": "Section 2", "text": "Content 2"},
            ],
        }
        mit = {
            "Document": {"title": "Test Doc"},
            "Sections": [
                {"title": "Section 1", "text": "Content 1"},
                {"title": "Section 2", "text": "Content 2"},
            ],
        }

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is False
        assert result.section_count_legacy == 2
        assert result.section_count_mit == 2
        assert result.title_differences == []
        assert result.document_title_differs is False

    def test_section_count_difference(self):
        """Should detect different section counts."""
        legacy = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}, {"title": "S2"}],
        }
        mit = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}, {"title": "S2"}, {"title": "S3"}],
        }

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is True
        assert result.section_count_differs is True
        assert result.section_count_delta == 1

    def test_title_differences(self):
        """Should detect title differences at specific positions."""
        legacy = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "Alpha"}, {"title": "Beta"}],
        }
        mit = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "Alpha"}, {"title": "Gamma"}],
        }

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is True
        assert len(result.title_differences) == 1
        assert result.title_differences[0] == (1, "Beta", "Gamma")

    def test_document_title_difference(self):
        """Should detect document title differences."""
        legacy = {
            "Document": {"title": "Old Title"},
            "Sections": [],
        }
        mit = {
            "Document": {"title": "New Title"},
            "Sections": [],
        }

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is True
        assert result.document_title_differs is True
        assert result.legacy_doc_title == "Old Title"
        assert result.mit_doc_title == "New Title"

    def test_new_metadata_fields_detected(self):
        """Should detect new metadata fields in markdown-it-py output."""
        legacy = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}],
        }
        mit = {
            "Document": {"title": "Test"},
            "Sections": [
                {
                    "title": "S1",
                    "line_start": 0,
                    "line_end": 10,
                    "parent_path": "Root",
                    "block_types": ["paragraph"],
                    "code_ratio": 0.0,
                    "has_code": False,
                    "has_table": False,
                }
            ],
        }

        result = compare_parser_results("test://doc.md", legacy, mit)

        # Parsers match on core output
        assert result.has_differences is False
        # But new fields are tracked
        assert "line_start" in result.new_metadata_fields
        assert "parent_path" in result.new_metadata_fields
        assert "block_types" in result.new_metadata_fields
        assert result.sample_metadata.get("line_start") == 0
        assert result.sample_metadata.get("parent_path") == "Root"

    def test_empty_sections(self):
        """Should handle empty section lists gracefully."""
        legacy = {"Document": {"title": "Test"}, "Sections": []}
        mit = {"Document": {"title": "Test"}, "Sections": []}

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is False
        assert result.section_count_legacy == 0
        assert result.section_count_mit == 0

    def test_missing_sections_key(self):
        """Should handle missing Sections key gracefully."""
        legacy = {"Document": {"title": "Test"}}
        mit = {"Document": {"title": "Test"}}

        result = compare_parser_results("test://doc.md", legacy, mit)

        assert result.has_differences is False
        assert result.section_count_legacy == 0


class TestShadowModeError:
    """Tests for ShadowModeError exception."""

    def test_error_message_includes_summary(self):
        """Error message should include comparison summary."""
        comparison = ParserComparisonResult(
            source_uri="test://doc.md",
            has_differences=True,
            section_count_legacy=5,
            section_count_mit=7,
        )
        error = ShadowModeError(comparison)

        assert "Parser mismatch detected" in str(error)
        assert "section_count: 5 → 7" in str(error)

    def test_error_contains_comparison_result(self):
        """Error should contain the comparison result for inspection."""
        comparison = ParserComparisonResult(
            source_uri="test://doc.md",
            has_differences=True,
        )
        error = ShadowModeError(comparison)

        assert error.comparison_result is comparison
        assert error.comparison_result.source_uri == "test://doc.md"


class TestLogComparisonResult:
    """Tests for log_comparison_result() function."""

    def test_logs_differences_at_info_level(self):
        """Should log differences at specified level."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            has_differences=True,
            section_count_legacy=5,
            section_count_mit=7,
        )

        with patch("src.ingestion.parsers.shadow_comparison.logger") as mock_logger:
            log_comparison_result(result, log_level="warning")
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args[1]
            assert call_kwargs["source_uri"] == "test://doc.md"

    def test_logs_match_at_debug_level(self):
        """Should log matching results at debug level."""
        result = ParserComparisonResult(
            source_uri="test://doc.md",
            has_differences=False,
            section_count_legacy=5,
            section_count_mit=5,
        )

        with patch("src.ingestion.parsers.shadow_comparison.logger") as mock_logger:
            log_comparison_result(result)
            mock_logger.debug.assert_called()


class TestShadowModeIntegration:
    """Integration tests for shadow mode in parser __init__.py."""

    def test_get_fail_on_mismatch_default(self):
        """get_fail_on_mismatch should return False by default."""
        from src.ingestion.parsers import get_fail_on_mismatch

        # Patch at the source where get_config is imported from
        with patch("src.shared.config.get_config") as mock_config:
            mock_config.return_value = {"ingestion": {"parser": {}}}
            result = get_fail_on_mismatch()
            assert result is False

    def test_get_fail_on_mismatch_enabled(self):
        """get_fail_on_mismatch should return True when configured."""
        from src.ingestion.parsers import get_fail_on_mismatch

        with patch("src.shared.config.get_config") as mock_config:
            mock_config.return_value = {
                "ingestion": {"parser": {"fail_on_mismatch": True}}
            }
            result = get_fail_on_mismatch()
            assert result is True

    def test_shadow_mode_error_importable(self):
        """ShadowModeError should be importable from parsers module."""
        from src.ingestion.parsers import ShadowModeError

        assert ShadowModeError is not None
        # Should be the same class
        from src.ingestion.parsers.shadow_comparison import (
            ShadowModeError as DirectError,
        )

        assert ShadowModeError is DirectError

    def test_shadow_comparison_raises_on_mismatch(self):
        """_parse_with_shadow_comparison should raise when fail_on_mismatch is True."""
        from src.ingestion.parsers import ShadowModeError

        # Create mock results that differ
        legacy_result = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}],
        }
        mit_result = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}, {"title": "S2"}],  # Extra section
        }

        with patch("src.ingestion.parsers.get_fail_on_mismatch", return_value=True):
            with patch(
                "src.ingestion.parsers.markdown.parse_markdown",
                return_value=legacy_result,
            ):
                with patch(
                    "src.ingestion.parsers.markdown_it_parser.parse_markdown",
                    return_value=mit_result,
                ):
                    from src.ingestion.parsers import _parse_with_shadow_comparison

                    with pytest.raises(ShadowModeError) as exc_info:
                        _parse_with_shadow_comparison(
                            "test://doc.md", "# Test", "markdown-it-py"
                        )

                    assert exc_info.value.comparison_result.section_count_delta == 1

    def test_shadow_comparison_no_raise_when_disabled(self):
        """_parse_with_shadow_comparison should not raise when fail_on_mismatch is False."""
        legacy_result = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}],
        }
        mit_result = {
            "Document": {"title": "Test"},
            "Sections": [{"title": "S1"}, {"title": "S2"}],
        }

        with patch("src.ingestion.parsers.get_fail_on_mismatch", return_value=False):
            with patch(
                "src.ingestion.parsers.markdown.parse_markdown",
                return_value=legacy_result,
            ):
                with patch(
                    "src.ingestion.parsers.markdown_it_parser.parse_markdown",
                    return_value=mit_result,
                ):
                    from src.ingestion.parsers import _parse_with_shadow_comparison

                    # Should not raise, returns mit_result
                    result = _parse_with_shadow_comparison(
                        "test://doc.md", "# Test", "markdown-it-py"
                    )
                    assert len(result["Sections"]) == 2


class TestShadowModeConfig:
    """Tests for shadow mode configuration reading."""

    def test_get_shadow_mode_reads_config(self):
        """get_shadow_mode should read from config correctly."""
        from src.ingestion.parsers import get_shadow_mode

        # Default should be False
        result = get_shadow_mode()
        assert isinstance(result, bool)

    def test_get_parser_engine_reads_config(self):
        """get_parser_engine should read from config correctly."""
        from src.ingestion.parsers import get_parser_engine

        result = get_parser_engine()
        assert result in ["legacy", "markdown-it-py"]
