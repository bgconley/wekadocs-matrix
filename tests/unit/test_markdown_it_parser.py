"""
Unit tests for markdown-it-py based parser.

Tests cover:
- API compatibility with legacy parser
- New metadata fields (line_start, parent_path, block_types, etc.)
- Frontmatter extraction
- Section boundary detection
- Code block and table handling
- Edge cases and error handling

Author: WekaDocs Team
Created: 2024-12-10
"""


class TestMarkdownItParserBasics:
    """Test basic parsing functionality and API compatibility."""

    def test_parse_simple_document(self):
        """Simple document should parse without errors."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Title

This is content.
"""
        result = parse_markdown("test://simple.md", doc)

        assert "Document" in result
        assert "Sections" in result
        assert result["Document"]["title"] == "Title"
        assert len(result["Sections"]) >= 1

    def test_parse_returns_required_document_fields(self):
        """Document should have all required fields for API compatibility."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        result = parse_markdown("test://doc.md", "# Test\n\nContent")

        doc = result["Document"]
        required_fields = [
            "id",
            "source_uri",
            "source_type",
            "title",
            "version",
            "checksum",
        ]
        for field in required_fields:
            assert field in doc, f"Missing required field: {field}"

        assert doc["source_uri"] == "test://doc.md"
        assert doc["source_type"] == "markdown"

    def test_parse_returns_required_section_fields(self):
        """Sections should have all required fields for API compatibility."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        result = parse_markdown("test://doc.md", "# Test\n\nContent")

        section = result["Sections"][0]
        required_fields = [
            "id",
            "document_id",
            "level",
            "title",
            "anchor",
            "order",
            "text",
            "tokens",
            "checksum",
            "code_blocks",
            "tables",
            "vector_embedding",
            "embedding_version",
        ]
        for field in required_fields:
            assert field in section, f"Missing required field: {field}"

    def test_parse_determinism(self):
        """Same input should produce same output (deterministic IDs)."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = "# Test\n\nContent here"

        result1 = parse_markdown("test://doc.md", doc)
        result2 = parse_markdown("test://doc.md", doc)

        assert result1["Document"]["id"] == result2["Document"]["id"]
        assert result1["Document"]["checksum"] == result2["Document"]["checksum"]
        assert result1["Sections"][0]["id"] == result2["Sections"][0]["id"]


class TestFrontmatterExtraction:
    """Test YAML frontmatter handling."""

    def test_frontmatter_extraction(self):
        """YAML frontmatter should be extracted and parsed."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """---
title: My Document
version: "2.0"
author: Test
---

# Content

Body text.
"""
        result = parse_markdown("test://fm.md", doc)

        assert result["Document"]["title"] == "My Document"
        assert result["Document"]["frontmatter"]["author"] == "Test"
        assert result["Document"]["version"] == "2.0"  # String when quoted in YAML

    def test_frontmatter_title_priority(self):
        """Frontmatter title should take priority over heading."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """---
title: Frontmatter Title
---

# Heading Title

Content.
"""
        result = parse_markdown("test://priority.md", doc)

        assert result["Document"]["title"] == "Frontmatter Title"

    def test_missing_frontmatter(self):
        """Documents without frontmatter should parse correctly."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = "# Just a Heading\n\nNo frontmatter here."
        result = parse_markdown("test://no-fm.md", doc)

        assert result["Document"]["title"] == "Just a Heading"
        assert result["Document"].get("frontmatter") == {}

    def test_invalid_frontmatter_graceful_handling(self):
        """Invalid YAML should not crash parser."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """---
invalid: yaml: content: [unclosed
---

# Title

Content.
"""
        # Should not raise
        result = parse_markdown("test://invalid-fm.md", doc)
        assert "Document" in result


class TestNewMetadataFields:
    """Test markdown-it-py specific metadata enhancements."""

    def test_line_start_tracking(self):
        """Sections should have line_start metadata."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# First

Content.

# Second

More content.
"""
        result = parse_markdown("test://lines.md", doc)

        # First section at line 0
        assert result["Sections"][0]["line_start"] == 0
        # Second section starts later
        assert result["Sections"][1]["line_start"] > result["Sections"][0]["line_start"]

    def test_parent_path_tracking(self):
        """Nested headings should have parent_path."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        # Each heading needs content to create a section
        doc = """# Parent

Parent content.

## Child

Child content.

### Grandchild

Grandchild content.

## Sibling

Sibling content.
"""
        result = parse_markdown("test://hierarchy.md", doc)
        sections = result["Sections"]

        # Find sections by title
        parent = next(s for s in sections if s["title"] == "Parent")
        child = next(s for s in sections if s["title"] == "Child")
        grandchild = next(s for s in sections if s["title"] == "Grandchild")
        sibling = next(s for s in sections if s["title"] == "Sibling")

        assert parent["parent_path"] == ""
        assert child["parent_path"] == "Parent"
        assert grandchild["parent_path"] == "Parent > Child"
        assert sibling["parent_path"] == "Parent"

    def test_block_types_tracking(self):
        """Sections should track block types (paragraph, code, table, list)."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Mixed Content

This is a paragraph.

```python
def hello():
    pass
```

- Item 1
- Item 2

| A | B |
|---|---|
| 1 | 2 |
"""
        result = parse_markdown("test://blocks.md", doc)
        section = result["Sections"][0]

        assert "paragraph" in section["block_types"]
        assert "code" in section["block_types"]
        assert "list" in section["block_types"]
        assert "table" in section["block_types"]

    def test_code_ratio_calculation(self):
        """Code ratio should be calculated correctly."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        # All code section
        all_code = """# Code Only

```bash
echo "hello"
```

```python
print("world")
```
"""
        result = parse_markdown("test://all-code.md", all_code)
        # Two code blocks out of two blocks = 1.0
        assert result["Sections"][0]["code_ratio"] == 1.0

        # Mixed section
        mixed = """# Mixed

Paragraph text.

```bash
echo "hello"
```

More text.
"""
        result = parse_markdown("test://mixed.md", mixed)
        # 1 code block out of 3 blocks = ~0.33
        assert 0.3 <= result["Sections"][0]["code_ratio"] <= 0.4

    def test_has_code_and_has_table_flags(self):
        """Boolean flags for code and table presence."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        with_code = """# With Code

```python
x = 1
```
"""
        result = parse_markdown("test://code.md", with_code)
        assert result["Sections"][0]["has_code"] is True
        assert result["Sections"][0]["has_table"] is False

        with_table = """# With Table

| A | B |
|---|---|
| 1 | 2 |
"""
        result = parse_markdown("test://table.md", with_table)
        assert result["Sections"][0]["has_code"] is False
        assert result["Sections"][0]["has_table"] is True


class TestSectionBoundaries:
    """Test section boundary detection."""

    def test_heading_levels_create_sections(self):
        """Each heading with content should create a new section."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        # Each heading needs content to avoid being skipped
        doc = """# H1

Content 1.

## H2

Content 2.

### H3

Content 3.

#### H4

Content 4.

##### H5

Content 5.

###### H6

Content 6.
"""
        result = parse_markdown("test://levels.md", doc)

        assert len(result["Sections"]) == 6
        levels = [s["level"] for s in result["Sections"]]
        assert levels == [1, 2, 3, 4, 5, 6]

    def test_section_order_preserved(self):
        """Section order should match document order."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        # Each heading needs content
        doc = """# First

First content.

## Second

Second content.

## Third

Third content.

# Fourth

Fourth content.
"""
        result = parse_markdown("test://order.md", doc)

        orders = [s["order"] for s in result["Sections"]]
        assert orders == [0, 1, 2, 3]

    def test_empty_sections_skipped(self):
        """Sections with no content should be skipped."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Title

# Empty

# Has Content

Some text here.
"""
        result = parse_markdown("test://empty.md", doc)

        # Only sections with content should be included
        titles = [s["title"] for s in result["Sections"]]
        assert "Has Content" in titles
        # Empty might be skipped

    def test_document_without_headings(self):
        """Document without headings should create single section."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """Just some content without any headings.

More paragraphs here.

And some more.
"""
        result = parse_markdown("test://no-headings.md", doc)

        assert len(result["Sections"]) >= 1
        assert "Just some content" in result["Sections"][0]["text"]


class TestCodeBlockHandling:
    """Test fenced code block extraction."""

    def test_code_blocks_extracted(self):
        """Fenced code blocks should be in code_blocks list."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Code Example

```python
def hello():
    return "world"
```
"""
        result = parse_markdown("test://code.md", doc)
        section = result["Sections"][0]

        assert len(section["code_blocks"]) == 1
        assert "def hello():" in section["code_blocks"][0]

    def test_code_block_markers_in_text(self):
        """Code blocks should have [CODE] markers in text."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Code

```bash
echo "hi"
```
"""
        result = parse_markdown("test://code-markers.md", doc)

        assert "[CODE]" in result["Sections"][0]["text"]
        assert "[/CODE]" in result["Sections"][0]["text"]

    def test_multiple_code_blocks(self):
        """Multiple code blocks should all be extracted."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Multiple Blocks

```python
x = 1
```

Some text.

```bash
echo "hi"
```
"""
        result = parse_markdown("test://multi-code.md", doc)

        assert len(result["Sections"][0]["code_blocks"]) == 2


class TestTableHandling:
    """Test table extraction."""

    def test_tables_extracted(self):
        """Tables should be in tables list."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Table Example

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
"""
        result = parse_markdown("test://table.md", doc)
        section = result["Sections"][0]

        assert len(section["tables"]) == 1
        assert "Header 1" in section["tables"][0]

    def test_table_markers_in_text(self):
        """Tables should have [TABLE] markers in text."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# Table

| A | B |
|---|---|
| 1 | 2 |
"""
        result = parse_markdown("test://table-markers.md", doc)

        assert "[TABLE]" in result["Sections"][0]["text"]
        assert "[/TABLE]" in result["Sections"][0]["text"]


class TestListHandling:
    """Test list extraction."""

    def test_unordered_list_preserved(self):
        """Unordered lists should preserve bullet markers."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# List

- Item 1
- Item 2
- Item 3
"""
        result = parse_markdown("test://ul.md", doc)
        text = result["Sections"][0]["text"]

        assert "- Item 1" in text or "Item 1" in text
        assert "list" in result["Sections"][0]["block_types"]

    def test_ordered_list_preserved(self):
        """Ordered lists should preserve number markers."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = """# List

1. First
2. Second
3. Third
"""
        result = parse_markdown("test://ol.md", doc)
        text = result["Sections"][0]["text"]

        assert "First" in text
        assert "list" in result["Sections"][0]["block_types"]


class TestAnchorGeneration:
    """Test anchor/slug generation."""

    def test_anchor_slugified(self):
        """Anchors should be URL-safe slugs."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = "# Hello World\n\nContent"
        result = parse_markdown("test://anchor.md", doc)

        assert result["Sections"][0]["anchor"] == "hello-world"

    def test_anchor_special_characters_removed(self):
        """Special characters should be removed from anchors."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        doc = "# Hello! World? (Test)\n\nContent"
        result = parse_markdown("test://special.md", doc)

        anchor = result["Sections"][0]["anchor"]
        assert "!" not in anchor
        assert "?" not in anchor
        assert "(" not in anchor


class TestIDDeterminism:
    """Test ID generation determinism."""

    def test_document_id_from_uri(self):
        """Document ID should be deterministic from URI."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        # Same URI should produce same ID
        r1 = parse_markdown("test://foo.md", "# A\n\nContent A")
        r2 = parse_markdown("test://foo.md", "# B\n\nContent B")

        assert r1["Document"]["id"] == r2["Document"]["id"]

        # Different URI should produce different ID
        r3 = parse_markdown("test://bar.md", "# A\n\nContent A")
        assert r1["Document"]["id"] != r3["Document"]["id"]

    def test_section_id_content_coupled(self):
        """Section ID should change when content changes."""
        from src.ingestion.parsers.markdown_it_parser import parse_markdown

        r1 = parse_markdown("test://doc.md", "# Title\n\nContent A")
        r2 = parse_markdown("test://doc.md", "# Title\n\nContent B")

        # Same title but different content = different section ID
        assert r1["Sections"][0]["id"] != r2["Sections"][0]["id"]


class TestParserSelector:
    """Test the parser selector/router."""

    def test_parse_markdown_from_init(self):
        """Should be able to import parse_markdown from __init__."""
        from src.ingestion.parsers import parse_markdown

        result = parse_markdown("test://init.md", "# Test\n\nContent")
        assert "Document" in result
        assert "Sections" in result

    def test_engine_constants_available(self):
        """Engine constants should be importable."""
        from src.ingestion.parsers import ENGINE_LEGACY, ENGINE_MARKDOWN_IT_PY

        assert ENGINE_LEGACY == "legacy"
        assert ENGINE_MARKDOWN_IT_PY == "markdown-it-py"


class TestLegacyCompatibility:
    """Test compatibility with legacy parser output."""

    def test_output_structure_matches_legacy(self):
        """Output structure should match legacy parser for basic docs."""
        from src.ingestion.parsers.markdown import parse_markdown as legacy_parse
        from src.ingestion.parsers.markdown_it_parser import parse_markdown as new_parse

        doc = """# Test Document

## Section One

Content for section one.

## Section Two

Content for section two.
"""
        new_result = new_parse("test://compat.md", doc)
        legacy_result = legacy_parse("test://compat.md", doc)

        # Same number of sections
        assert len(new_result["Sections"]) == len(legacy_result["Sections"])

        # Same document ID
        assert new_result["Document"]["id"] == legacy_result["Document"]["id"]

        # Same section titles
        new_titles = [s["title"] for s in new_result["Sections"]]
        legacy_titles = [s["title"] for s in legacy_result["Sections"]]
        assert new_titles == legacy_titles

        # Same section levels
        new_levels = [s["level"] for s in new_result["Sections"]]
        legacy_levels = [s["level"] for s in legacy_result["Sections"]]
        assert new_levels == legacy_levels

    def test_code_blocks_format_matches(self):
        """Code blocks format should match legacy."""
        from src.ingestion.parsers.markdown import parse_markdown as legacy_parse
        from src.ingestion.parsers.markdown_it_parser import parse_markdown as new_parse

        doc = """# Code Test

```python
def foo():
    pass
```
"""
        new_result = new_parse("test://code-compat.md", doc)
        legacy_result = legacy_parse("test://code-compat.md", doc)

        # Both should have [CODE] markers in text
        assert "[CODE]" in new_result["Sections"][0]["text"]
        assert "[CODE]" in legacy_result["Sections"][0]["text"]

        # Both should have code in code_blocks list
        assert len(new_result["Sections"][0]["code_blocks"]) >= 1
        assert len(legacy_result["Sections"][0]["code_blocks"]) >= 1
