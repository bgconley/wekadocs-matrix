"""
Unit tests for graph_enhancements.py (Phase 3: markdown-it-py integration).

Tests the Neo4j graph enhancement functions for:
- PARENT_HEADING relationships based on parent_path hierarchy
- Structural labels (CodeSection, TableSection)
- Enhanced metadata updates

Note: Tests that require Neo4j are marked with @pytest.mark.neo4j and
can be skipped when Neo4j is not available.
"""

from unittest.mock import MagicMock, patch

from src.neo.graph_enhancements import (
    apply_structural_labels,
    create_parent_heading_relationships,
    enhance_document_graph,
    get_immediate_parent_title,
    parse_parent_path,
    update_section_enhanced_metadata,
)

# =============================================================================
# Tests for parse_parent_path
# =============================================================================


class TestParseParentPath:
    """Test parent_path parsing into heading hierarchy list."""

    def test_parse_simple_path(self):
        """Should parse simple two-level path."""
        result = parse_parent_path("Getting Started > Installation")
        assert result == ["Getting Started", "Installation"]

    def test_parse_deep_path(self):
        """Should parse deep multi-level path."""
        result = parse_parent_path("A > B > C > D")
        assert result == ["A", "B", "C", "D"]

    def test_parse_single_element(self):
        """Should handle single element path."""
        result = parse_parent_path("Overview")
        assert result == ["Overview"]

    def test_parse_empty_string(self):
        """Should return empty list for empty string."""
        result = parse_parent_path("")
        assert result == []

    def test_parse_none(self):
        """Should handle None gracefully."""
        result = parse_parent_path(None)
        assert result == []

    def test_parse_whitespace_only(self):
        """Should return empty list for whitespace-only string."""
        result = parse_parent_path("   ")
        assert result == []

    def test_parse_trims_whitespace(self):
        """Should trim whitespace around elements."""
        result = parse_parent_path("  A  >  B  >  C  ")
        assert result == ["A", "B", "C"]

    def test_parse_filters_empty_elements(self):
        """Should filter out empty elements from malformed paths."""
        result = parse_parent_path("A >  > B")
        assert result == ["A", "B"]

    def test_parse_preserves_special_characters(self):
        """Should preserve special characters in heading titles."""
        result = parse_parent_path("WEKA® Configuration > S3-Compatible Storage")
        assert result == ["WEKA® Configuration", "S3-Compatible Storage"]


# =============================================================================
# Tests for get_immediate_parent_title
# =============================================================================


class TestGetImmediateParentTitle:
    """Test extraction of immediate parent from parent_path."""

    def test_get_parent_from_path(self):
        """Should return last element of path."""
        result = get_immediate_parent_title("Getting Started > Installation")
        assert result == "Installation"

    def test_get_parent_from_deep_path(self):
        """Should return immediate parent from deep path."""
        result = get_immediate_parent_title("A > B > C > D")
        assert result == "D"

    def test_get_parent_single_element(self):
        """Should return the single element as parent."""
        result = get_immediate_parent_title("Overview")
        assert result == "Overview"

    def test_get_parent_empty_returns_none(self):
        """Should return None for empty path."""
        result = get_immediate_parent_title("")
        assert result is None

    def test_get_parent_none_returns_none(self):
        """Should return None for None input."""
        result = get_immediate_parent_title(None)
        assert result is None


# =============================================================================
# Tests for create_parent_heading_relationships (with mocks)
# =============================================================================


class TestCreateParentHeadingRelationships:
    """Test PARENT_HEADING relationship creation with mocked Neo4j session."""

    def test_creates_relationships_for_document(self):
        """Should execute Cypher query to create PARENT_HEADING edges."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"matched": 5, "created": 5}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        result = create_parent_heading_relationships(
            mock_session, "doc_123", sections=None
        )

        # Verify Cypher was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args

        # Verify document_id parameter was passed
        assert call_args[1]["document_id"] == "doc_123"

        # Verify return stats
        assert result["matched"] == 5
        assert result["created"] == 5
        assert result["errors"] == 0

    def test_handles_no_matches(self):
        """Should handle case where no sections have parent_path."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"matched": 0, "created": 0}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        result = create_parent_heading_relationships(
            mock_session, "doc_empty", sections=None
        )

        assert result["matched"] == 0
        assert result["created"] == 0
        assert result["errors"] == 0

    def test_handles_cypher_exception(self):
        """Should catch and log exceptions without raising."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Neo4j connection failed")

        result = create_parent_heading_relationships(
            mock_session, "doc_fail", sections=None
        )

        assert result["errors"] == 1
        assert result["created"] == 0


# =============================================================================
# Tests for apply_structural_labels (with mocks)
# =============================================================================


class TestApplyStructuralLabels:
    """Test CodeSection/TableSection label application with mocked session."""

    def test_applies_code_and_table_labels(self):
        """Should execute both label queries and return counts."""
        mock_session = MagicMock()

        # Mock code query result
        mock_code_result = MagicMock()
        mock_code_result.single.return_value = {"labeled": 10}

        # Mock table query result
        mock_table_result = MagicMock()
        mock_table_result.single.return_value = {"labeled": 3}

        # Return different results for each call
        mock_session.run.side_effect = [mock_code_result, mock_table_result]

        result = apply_structural_labels(mock_session, "doc_123")

        # Verify both queries executed
        assert mock_session.run.call_count == 2

        # Verify return stats
        assert result["code_sections"] == 10
        assert result["table_sections"] == 3
        assert result["errors"] == 0

    def test_handles_no_matching_sections(self):
        """Should handle documents with no code or table sections."""
        mock_session = MagicMock()

        mock_result = MagicMock()
        mock_result.single.return_value = {"labeled": 0}
        mock_session.run.return_value = mock_result

        result = apply_structural_labels(mock_session, "doc_empty")

        assert result["code_sections"] == 0
        assert result["table_sections"] == 0

    def test_handles_exception(self):
        """Should catch and log exceptions."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Label query failed")

        result = apply_structural_labels(mock_session, "doc_fail")

        assert result["errors"] == 1


# =============================================================================
# Tests for update_section_enhanced_metadata (with mocks)
# =============================================================================


class TestUpdateSectionEnhancedMetadata:
    """Test enhanced metadata updates with mocked Neo4j session."""

    def test_updates_sections_with_metadata(self):
        """Should update sections that have enhanced metadata."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"updated": 3}
        mock_session.run.return_value = mock_result

        sections = [
            {
                "id": "sec_1",
                "line_start": 10,
                "line_end": 25,
                "parent_path": "A > B",
                "block_types": ["paragraph"],
                "code_ratio": 0.0,
                "has_code": False,
                "has_table": False,
            },
            {
                "id": "sec_2",
                "line_start": 26,
                "has_code": True,
            },
            {
                "id": "sec_3",
                "line_start": 50,
                "parent_path": "A",
            },
        ]

        result = update_section_enhanced_metadata(mock_session, "doc_123", sections)

        assert result["updated"] == 3
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_skips_sections_without_enhanced_metadata(self):
        """Should skip sections that lack enhanced metadata."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"updated": 1}
        mock_session.run.return_value = mock_result

        sections = [
            {
                "id": "sec_1",
                "line_start": 10,  # Has enhanced metadata
            },
            {
                "id": "sec_2",
                "title": "Legacy Section",  # No enhanced metadata
                "text": "Content",
            },
        ]

        result = update_section_enhanced_metadata(mock_session, "doc_123", sections)

        # One section should be skipped
        assert result["skipped"] == 1

    def test_returns_early_if_no_sections_to_update(self):
        """Should return immediately if no sections have enhanced metadata."""
        mock_session = MagicMock()

        sections = [
            {"id": "sec_1", "title": "Legacy Only"},
            {"id": "sec_2", "title": "Also Legacy"},
        ]

        result = update_section_enhanced_metadata(mock_session, "doc_123", sections)

        # Should not have called Neo4j
        mock_session.run.assert_not_called()
        assert result["skipped"] == 2
        assert result["updated"] == 0

    def test_handles_empty_sections_list(self):
        """Should handle empty sections list gracefully."""
        mock_session = MagicMock()

        result = update_section_enhanced_metadata(mock_session, "doc_123", [])

        mock_session.run.assert_not_called()
        assert result["updated"] == 0
        assert result["skipped"] == 0


# =============================================================================
# Tests for enhance_document_graph (with mocks)
# =============================================================================


class TestEnhanceDocumentGraph:
    """Test the convenience function that runs all enhancements."""

    @patch("src.neo.graph_enhancements.update_section_enhanced_metadata")
    @patch("src.neo.graph_enhancements.create_parent_heading_relationships")
    @patch("src.neo.graph_enhancements.apply_structural_labels")
    def test_runs_all_enhancement_steps(self, mock_labels, mock_parent, mock_metadata):
        """Should run all three enhancement functions in order."""
        mock_session = MagicMock()

        mock_metadata.return_value = {"updated": 5, "skipped": 0, "errors": 0}
        mock_parent.return_value = {"matched": 3, "created": 3, "errors": 0}
        mock_labels.return_value = {
            "code_sections": 2,
            "table_sections": 1,
            "errors": 0,
        }

        sections = [{"id": "sec_1", "line_start": 10}]

        result = enhance_document_graph(mock_session, "doc_123", sections)

        # Verify all functions called
        mock_metadata.assert_called_once_with(mock_session, "doc_123", sections)
        mock_parent.assert_called_once_with(mock_session, "doc_123", sections)
        mock_labels.assert_called_once_with(mock_session, "doc_123")

        # Verify combined result
        assert result["success"] is True
        assert result["document_id"] == "doc_123"
        assert result["metadata"]["updated"] == 5
        assert result["relationships"]["created"] == 3
        assert result["labels"]["code_sections"] == 2

    @patch("src.neo.graph_enhancements.update_section_enhanced_metadata")
    @patch("src.neo.graph_enhancements.create_parent_heading_relationships")
    @patch("src.neo.graph_enhancements.apply_structural_labels")
    def test_reports_failure_on_errors(self, mock_labels, mock_parent, mock_metadata):
        """Should set success=False if any step has errors."""
        mock_session = MagicMock()

        mock_metadata.return_value = {"updated": 5, "skipped": 0, "errors": 0}
        mock_parent.return_value = {"matched": 0, "created": 0, "errors": 1}  # Error
        mock_labels.return_value = {
            "code_sections": 0,
            "table_sections": 0,
            "errors": 0,
        }

        result = enhance_document_graph(mock_session, "doc_123", [])

        assert result["success"] is False

    @patch("src.neo.graph_enhancements.update_section_enhanced_metadata")
    def test_handles_exception_gracefully(self, mock_metadata):
        """Should catch exceptions and set success=False."""
        mock_session = MagicMock()
        mock_metadata.side_effect = Exception("Unexpected error")

        result = enhance_document_graph(mock_session, "doc_123", [])

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Tests for Cypher query correctness (pattern validation)
# =============================================================================


class TestCypherQueryPatterns:
    """Validate that Cypher queries follow expected patterns."""

    def test_parent_heading_query_uses_correct_relationship_direction(self):
        """PARENT_HEADING should go from child to parent (child)-[:PARENT_HEADING]->(parent)."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"matched": 0, "created": 0}
        mock_session.run.return_value = mock_result

        create_parent_heading_relationships(mock_session, "doc_test")

        # Get the Cypher query
        call_args = mock_session.run.call_args
        query = call_args[0][0]

        # Verify direction: (child)-[:PARENT_HEADING]->(parent)
        assert "(child)-[r:PARENT_HEADING]->(parent)" in query
        # Verify we're matching child.parent_path
        assert "child.parent_path" in query

    def test_structural_label_queries_use_compound_labels(self):
        """Label queries should use SET s:CodeSection pattern."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"labeled": 0}
        mock_session.run.return_value = mock_result

        apply_structural_labels(mock_session, "doc_test")

        # Get both queries
        calls = mock_session.run.call_args_list

        code_query = calls[0][0][0]
        table_query = calls[1][0][0]

        # Verify label application syntax
        assert "SET s:CodeSection" in code_query
        assert "SET s:TableSection" in table_query

        # Verify idempotency check
        assert "NOT s:CodeSection" in code_query
        assert "NOT s:TableSection" in table_query


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and potential regression scenarios."""

    def test_parse_parent_path_handles_arrow_in_heading(self):
        """Should handle headings that contain > character."""
        # This is a known edge case - the separator is " > " (with spaces)
        result = parse_parent_path("Error Codes > Error Code >= 100")
        # The " > " separator should be recognized, not bare ">"
        assert result == ["Error Codes", "Error Code >= 100"]

    def test_update_metadata_batch_size_respected(self):
        """Should process sections in batches of specified size."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"updated": 50}
        mock_session.run.return_value = mock_result

        # Create 150 sections
        sections = [{"id": f"sec_{i}", "line_start": i * 10} for i in range(150)]

        update_section_enhanced_metadata(
            mock_session, "doc_123", sections, batch_size=50
        )

        # Should have been called 3 times (150 / 50 = 3 batches)
        assert mock_session.run.call_count == 3
