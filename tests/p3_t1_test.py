# Phase 3, Task 3.1 Tests - Multi-format Parsers
# NO MOCKS - Tests against real sample documents

from pathlib import Path

import pytest

from src.ingestion.parsers.html import parse_html
from src.ingestion.parsers.markdown import parse_markdown


class TestMarkdownParser:
    """Tests for markdown parser with determinism guarantees."""

    @pytest.fixture
    def sample_docs_path(self):
        return Path(__file__).parent.parent / "data" / "samples"

    def test_parse_getting_started(self, sample_docs_path):
        """Test parsing getting started markdown."""
        md_path = sample_docs_path / "getting_started.md"
        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)

        assert "Document" in result
        assert "Sections" in result

        doc = result["Document"]
        assert doc["source_type"] == "markdown"
        assert doc["title"] == "Getting Started with WekaFS"
        assert len(doc["id"]) == 64  # SHA-256 hash

        sections = result["Sections"]
        assert len(sections) > 0

        # Verify all sections have required fields
        for section in sections:
            assert "id" in section
            assert "title" in section
            assert "text" in section
            assert "checksum" in section
            assert "tokens" in section
            assert "anchor" in section
            assert len(section["id"]) == 64

    def test_parse_determinism(self, sample_docs_path):
        """Test that parsing the same document twice yields identical IDs."""
        md_path = sample_docs_path / "api_guide.md"
        with open(md_path, "r") as f:
            content = f.read()

        # Parse twice
        result1 = parse_markdown(str(md_path), content)
        result2 = parse_markdown(str(md_path), content)

        # Document IDs should be identical
        assert result1["Document"]["id"] == result2["Document"]["id"]

        # Section IDs should be identical
        sections1 = {s["id"]: s for s in result1["Sections"]}
        sections2 = {s["id"]: s for s in result2["Sections"]}

        assert set(sections1.keys()) == set(sections2.keys())

        # Checksums should match
        for sid in sections1.keys():
            assert sections1[sid]["checksum"] == sections2[sid]["checksum"]

    def test_preserves_code_blocks(self, sample_docs_path):
        """Test that code blocks are preserved."""
        md_path = sample_docs_path / "getting_started.md"
        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        sections = result["Sections"]

        # Find section with code blocks
        sections_with_code = [s for s in sections if s.get("code_blocks")]
        assert len(sections_with_code) > 0

        # Verify code blocks contain expected commands
        all_code = "\n".join("\n".join(s["code_blocks"]) for s in sections_with_code)
        assert "weka" in all_code.lower()

    def test_anchor_generation(self, sample_docs_path):
        """Test that anchors are generated for sections."""
        md_path = sample_docs_path / "performance_tuning.md"
        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        sections = result["Sections"]

        # All sections should have anchors
        for section in sections:
            assert section["anchor"]
            assert isinstance(section["anchor"], str)
            assert len(section["anchor"]) > 0

    def test_token_counting(self, sample_docs_path):
        """Test that token counts are computed."""
        md_path = sample_docs_path / "api_guide.md"
        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        sections = result["Sections"]

        # At least some sections should have tokens
        sections_with_tokens = [s for s in sections if s["tokens"] > 0]
        assert len(sections_with_tokens) > 0

        for section in sections_with_tokens:
            # Token count should be reasonable (not excessive)
            assert section["tokens"] < len(section["text"]) + 100


class TestHTMLParser:
    """Tests for HTML parser."""

    @pytest.fixture
    def sample_docs_path(self):
        return Path(__file__).parent.parent / "data" / "samples"

    def test_parse_html_document(self, sample_docs_path):
        """Test parsing HTML document."""
        html_path = sample_docs_path / "sample_doc.html"
        with open(html_path, "r") as f:
            content = f.read()

        result = parse_html(str(html_path), content)

        assert "Document" in result
        assert "Sections" in result

        doc = result["Document"]
        assert doc["source_type"] == "html"
        assert "WekaFS" in doc["title"]

        sections = result["Sections"]
        assert len(sections) > 0

    def test_html_determinism(self, sample_docs_path):
        """Test HTML parser determinism."""
        html_path = sample_docs_path / "sample_doc.html"
        with open(html_path, "r") as f:
            content = f.read()

        result1 = parse_html(str(html_path), content)
        result2 = parse_html(str(html_path), content)

        assert result1["Document"]["id"] == result2["Document"]["id"]

        sections1_ids = {s["id"] for s in result1["Sections"]}
        sections2_ids = {s["id"] for s in result2["Sections"]}
        assert sections1_ids == sections2_ids

    def test_html_preserves_code(self, sample_docs_path):
        """Test that HTML parser preserves code elements."""
        html_path = sample_docs_path / "sample_doc.html"
        with open(html_path, "r") as f:
            content = f.read()

        result = parse_html(str(html_path), content)
        sections = result["Sections"]

        # Should find code blocks
        sections_with_code = [s for s in sections if s.get("code_blocks")]
        assert len(sections_with_code) > 0

    def test_html_preserves_tables(self, sample_docs_path):
        """Test that HTML parser preserves tables."""
        html_path = sample_docs_path / "sample_doc.html"
        with open(html_path, "r") as f:
            content = f.read()

        result = parse_html(str(html_path), content)
        sections = result["Sections"]

        # Should find tables
        sections_with_tables = [s for s in sections if s.get("tables")]
        assert len(sections_with_tables) > 0

        # Verify table content
        for section in sections_with_tables:
            for table in section["tables"]:
                assert len(table) > 0


class TestParserDeterminism:
    """Cross-cutting determinism tests."""

    def test_section_id_uniqueness_within_document(self):
        """Test that section IDs are unique within a document."""
        content = """
# Section 1

Content for section 1.

## Section 2

Content for section 2.

## Section 3

Content for section 3.
"""
        result = parse_markdown("test://doc", content)
        sections = result["Sections"]

        section_ids = [s["id"] for s in sections]
        assert len(section_ids) == len(set(section_ids)), "Section IDs must be unique"

    def test_checksum_changes_with_content(self):
        """Test that checksum changes when content changes."""
        content1 = "# Title\n\nOriginal content"
        content2 = "# Title\n\nModified content"

        result1 = parse_markdown("test://doc", content1)
        result2 = parse_markdown("test://doc", content2)

        # Document checksum should differ
        assert result1["Document"]["checksum"] != result2["Document"]["checksum"]

        # Section checksums should differ
        section1 = result1["Sections"][0]
        section2 = result2["Sections"][0]
        assert section1["checksum"] != section2["checksum"]
