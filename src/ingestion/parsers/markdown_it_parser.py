"""
markdown-it-py based parser for WekaDocs RAG pipeline.

This module provides an AST-based markdown parser using markdown-it-py,
replacing the legacy markdown + BeautifulSoup approach.

Key improvements over legacy parser:
- Native AST traversal via SyntaxTreeNode (no HTML intermediate)
- Source line mapping via Token.map for NER correlation
- True heading hierarchy tracking (parent_path)
- Block type metadata for structural queries
- CommonMark + GFM compliance

API Compatibility:
    This module exposes `parse_markdown(source_uri, raw_text)` with the same
    return signature as the legacy parser, enabling drop-in replacement.

New Metadata Fields (per section):
    - line_start: Source line number where section begins
    - line_end: Source line number where section ends
    - parent_path: Heading hierarchy trail (e.g., "Installation > Prerequisites")
    - block_types: List of block types in section (paragraph, code, table, list)
    - code_ratio: Fraction of content that is code (0.0-1.0)
    - has_table: Boolean flag for table-heavy sections
    - has_code: Boolean flag for code-containing sections

Author: WekaDocs Team
Created: 2024-12-10
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml

# markdown-it-py imports
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

# Plugin imports - graceful degradation if not installed
try:
    from mdit_py_plugins.front_matter import front_matter_plugin

    FRONTMATTER_PLUGIN_AVAILABLE = True
except ImportError:
    FRONTMATTER_PLUGIN_AVAILABLE = False
    front_matter_plugin = None

logger = structlog.get_logger(__name__)

# --- Constants ---
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# =============================================================================
# Helper Functions (deterministic ID generation - must match legacy parser)
# =============================================================================


def _slugify(title: str) -> str:
    """Convert text to URL-friendly slug with proper normalization."""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    t = t.lower().strip()
    t = _SLUG_RE.sub("-", t).strip("-")
    return t or "section"


def _normalize_text(s: str) -> str:
    """Normalize text for consistent hashing - trim, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip())


def _section_checksum(text: str) -> str:
    """Compute checksum from normalized text."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _section_id(source_uri: str, anchor: str, checksum: str) -> str:
    """Content-coupled identity; stale sections are removed by stage/swap."""
    raw = f"{source_uri}#{anchor}|{checksum}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_document_id(source_uri: str) -> str:
    """Compute deterministic document ID from source URI."""
    normalized_uri = source_uri.strip().lower()
    return hashlib.sha256(normalized_uri.encode("utf-8")).hexdigest()


def _compute_checksum(text: str) -> str:
    """Compute SHA-256 checksum of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# Parser Creation
# =============================================================================


def create_parser() -> MarkdownIt:
    """
    Create configured markdown-it-py parser with WEKA-appropriate settings.

    Uses 'gfm-like' preset for GitHub Flavored Markdown support:
    - Tables
    - Strikethrough
    - Autolinks (if linkify-it-py installed)

    Returns:
        Configured MarkdownIt instance
    """
    md = MarkdownIt("gfm-like", {"typographer": False, "html": True})

    # Enable frontmatter plugin if available
    if FRONTMATTER_PLUGIN_AVAILABLE and front_matter_plugin is not None:
        md.use(front_matter_plugin)
    else:
        logger.warning(
            "mdit-py-plugins not installed, frontmatter will use regex fallback"
        )

    # Ensure key extensions are enabled
    md.enable("table")

    return md


def parse_to_tokens(raw_text: str) -> List[Any]:
    """
    Parse markdown to flat token stream.

    Args:
        raw_text: Raw markdown content

    Returns:
        List of Token objects with source positions
    """
    md = create_parser()
    return md.parse(raw_text)


def parse_to_ast(raw_text: str) -> SyntaxTreeNode:
    """
    Parse markdown to SyntaxTreeNode for hierarchical traversal.

    The AST collapses opening/closing tokens into single nodes with children,
    making section extraction more intuitive than flat token iteration.

    Args:
        raw_text: Raw markdown content

    Returns:
        SyntaxTreeNode root of the AST
    """
    tokens = parse_to_tokens(raw_text)
    return SyntaxTreeNode(tokens)


# =============================================================================
# Frontmatter Extraction
# =============================================================================


def extract_frontmatter_from_ast(ast: SyntaxTreeNode) -> Dict[str, Any]:
    """
    Extract YAML frontmatter from AST (if present via plugin).

    Args:
        ast: Parsed SyntaxTreeNode

    Returns:
        Dictionary of frontmatter metadata, empty dict if none found
    """
    for node in ast.children:
        if node.type == "front_matter":
            content = node.content if hasattr(node, "content") else ""
            if not content and node.token:
                content = node.token.content
            try:
                return yaml.safe_load(content) or {}
            except yaml.YAMLError as e:
                logger.warning(
                    "Failed to parse YAML frontmatter from AST", error=str(e)
                )
                return {}
    return {}


def extract_frontmatter_regex(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter using regex (fallback method).

    This matches the legacy parser's frontmatter extraction behavior.

    Args:
        raw_text: Raw markdown text with potential frontmatter

    Returns:
        Tuple of (metadata_dict, content_without_frontmatter)
    """
    match = _FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        return {}, raw_text

    yaml_content = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_content)
        if metadata is None:
            metadata = {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse YAML frontmatter", error=str(e))
        return {}, raw_text

    content = raw_text[match.end() :]
    return metadata, content


# =============================================================================
# Text Extraction from AST Nodes
# =============================================================================


def _extract_text_from_node(node: SyntaxTreeNode) -> str:
    """
    Recursively extract plain text content from an AST node.

    Args:
        node: SyntaxTreeNode to extract text from

    Returns:
        Plain text content
    """
    if node.type == "text":
        return node.content if hasattr(node, "content") else ""

    # For inline nodes, get content attribute
    if hasattr(node, "content") and node.content:
        return node.content

    # Recursively gather from children
    parts = []
    for child in node.children or []:
        parts.append(_extract_text_from_node(child))
    return "".join(parts)


def _extract_inline_text(node: SyntaxTreeNode) -> str:
    """
    Extract text from an inline container node.

    Inline nodes contain the actual text content as children.

    Args:
        node: Inline-type SyntaxTreeNode

    Returns:
        Combined text content
    """
    if node.type == "inline":
        # Inline node has children with actual text
        return _extract_text_from_node(node)

    # For other nodes, check children
    for child in node.children or []:
        if child.type == "inline":
            return _extract_text_from_node(child)

    return ""


def _get_heading_text(heading_node: SyntaxTreeNode) -> str:
    """
    Extract the text content of a heading node.

    Args:
        heading_node: A heading-type SyntaxTreeNode

    Returns:
        The heading text
    """
    for child in heading_node.children or []:
        if child.type == "inline":
            return _extract_text_from_node(child)
    return ""


def _get_heading_level(heading_node: SyntaxTreeNode) -> int:
    """
    Get the level (1-6) of a heading node.

    Args:
        heading_node: A heading-type SyntaxTreeNode

    Returns:
        Heading level (1-6)
    """
    # Check tag attribute (h1, h2, etc.)
    if hasattr(heading_node, "tag") and heading_node.tag:
        tag = heading_node.tag
        if tag.startswith("h") and len(tag) == 2:
            try:
                return int(tag[1])
            except ValueError:
                pass

    # Check token markup (# count)
    if hasattr(heading_node, "token") and heading_node.token:
        markup = heading_node.token.markup
        if markup and markup.startswith("#"):
            return len(markup)

    return 2  # Default to h2


def _get_source_lines(node: SyntaxTreeNode) -> Tuple[Optional[int], Optional[int]]:
    """
    Get source line numbers from a node's token.map.

    Args:
        node: SyntaxTreeNode with potential line mapping

    Returns:
        Tuple of (start_line, end_line), None values if unavailable
    """
    # Check node's map attribute
    if hasattr(node, "map") and node.map:
        return node.map[0], node.map[1]

    # Check underlying token
    if hasattr(node, "token") and node.token and node.token.map:
        return node.token.map[0], node.token.map[1]

    return None, None


# =============================================================================
# Section Extraction
# =============================================================================


def _render_code_block(node: SyntaxTreeNode) -> Tuple[str, str]:
    """
    Extract code content and language from a fence/code_block node.

    Args:
        node: Code block SyntaxTreeNode

    Returns:
        Tuple of (code_content, language)
    """
    content = ""
    lang = ""

    if hasattr(node, "content") and node.content:
        content = node.content
    elif hasattr(node, "token") and node.token:
        content = node.token.content or ""
        lang = node.token.info or ""

    # Try to get info/language
    if not lang and hasattr(node, "info"):
        lang = node.info or ""

    return content, lang.strip()


def _render_table_text(node: SyntaxTreeNode) -> str:
    """
    Render table node to readable text representation.

    Args:
        node: Table SyntaxTreeNode

    Returns:
        Text representation of the table
    """
    rows = []

    def collect_cells(n: SyntaxTreeNode, current_row: List[str]) -> None:
        """Recursively collect cell content."""
        if n.type in ("th", "td"):
            current_row.append(_extract_inline_text(n).strip())
        for child in n.children or []:
            collect_cells(child, current_row)

    # Walk table structure
    for child in node.children or []:
        if child.type in ("thead", "tbody"):
            for row_node in child.children or []:
                if row_node.type == "tr":
                    row_cells: List[str] = []
                    collect_cells(row_node, row_cells)
                    if row_cells:
                        rows.append(" | ".join(row_cells))
        elif child.type == "tr":
            row_cells = []
            collect_cells(child, row_cells)
            if row_cells:
                rows.append(" | ".join(row_cells))

    return "\n".join(rows)


def _render_list_text(node: SyntaxTreeNode, ordered: bool = False) -> str:
    """
    Render list node to text with bullets/numbers.

    Args:
        node: List SyntaxTreeNode
        ordered: Whether this is an ordered list

    Returns:
        Text representation with list markers
    """
    items = []
    item_num = 1

    for child in node.children or []:
        if child.type == "list_item":
            # Extract paragraph content from list item
            text_parts = []
            for sub in child.children or []:
                if sub.type == "paragraph":
                    text_parts.append(_extract_inline_text(sub))
                elif sub.type in ("bullet_list", "ordered_list"):
                    # Nested list - recurse with indent
                    nested = _render_list_text(sub, sub.type == "ordered_list")
                    indented = "\n".join("  " + line for line in nested.split("\n"))
                    text_parts.append(indented)

            text = " ".join(text_parts).strip()
            if ordered:
                items.append(f"{item_num}. {text}")
                item_num += 1
            else:
                items.append(f"- {text}")

    return "\n".join(items)


def extract_sections(
    ast: SyntaxTreeNode, source_uri: str, raw_text: str
) -> List[Dict[str, Any]]:
    """
    Extract sections by traversing heading nodes in AST.

    This is the core improvement over BeautifulSoup scraping:
    - Preserves source line numbers (Token.map)
    - Maintains true parent-child heading relationships
    - Captures block type metadata

    Args:
        ast: Parsed SyntaxTreeNode
        source_uri: Source document URI
        raw_text: Original raw text (for fallback title)

    Returns:
        List of section dictionaries with enhanced metadata
    """
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    heading_stack: List[Dict[str, Any]] = []  # For parent_path tracking
    order = 0

    def finalize_current_section() -> None:
        """Finalize and append current section if valid."""
        nonlocal current_section, order

        if current_section is None:
            return

        section = _finalize_section(source_uri, current_section, order)
        if section:  # Only add non-empty sections
            sections.append(section)
            order += 1

        current_section = None

    def process_node(node: SyntaxTreeNode) -> None:
        """Process a single AST node."""
        nonlocal current_section, heading_stack

        if node.type == "heading":
            # Finalize previous section
            finalize_current_section()

            level = _get_heading_level(node)
            title = _get_heading_text(node)
            line_start, line_end = _get_source_lines(node)

            # Update heading stack for parent_path
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            parent_path = " > ".join(h["title"] for h in heading_stack)
            heading_stack.append({"level": level, "title": title})

            current_section = {
                "level": level,
                "title": title,
                "anchor": _slugify(title),
                "parent_path": parent_path,
                "line_start": line_start,
                "line_end": line_end,
                "content_elements": [],
                "code_blocks": [],
                "tables": [],
                "block_types": [],
            }

        elif current_section is not None:
            # Accumulate content with block type tracking
            _process_content_node(node, current_section)

            # Update line_end to track section extent
            _, node_end = _get_source_lines(node)
            if node_end and (
                current_section["line_end"] is None
                or node_end > current_section["line_end"]
            ):
                current_section["line_end"] = node_end

    def _process_content_node(node: SyntaxTreeNode, section: Dict[str, Any]) -> None:
        """Process a content node and add to section."""
        if node.type == "fence":
            code, lang = _render_code_block(node)
            section["code_blocks"].append({"code": code, "lang": lang})
            section["content_elements"].append(f"[CODE]\n{code}\n[/CODE]")
            section["block_types"].append("code")

        elif node.type == "code_block":
            code, lang = _render_code_block(node)
            section["code_blocks"].append({"code": code, "lang": lang})
            section["content_elements"].append(f"[CODE]\n{code}\n[/CODE]")
            section["block_types"].append("code")

        elif node.type == "table":
            table_text = _render_table_text(node)
            section["tables"].append(table_text)
            section["content_elements"].append(f"[TABLE]\n{table_text}\n[/TABLE]")
            section["block_types"].append("table")

        elif node.type == "bullet_list":
            list_text = _render_list_text(node, ordered=False)
            section["content_elements"].append(list_text)
            section["block_types"].append("list")

        elif node.type == "ordered_list":
            list_text = _render_list_text(node, ordered=True)
            section["content_elements"].append(list_text)
            section["block_types"].append("list")

        elif node.type == "paragraph":
            text = _extract_inline_text(node)
            if text.strip():
                section["content_elements"].append(text)
                section["block_types"].append("paragraph")

        elif node.type == "blockquote":
            # Extract blockquote content
            parts = []
            for child in node.children or []:
                if child.type == "paragraph":
                    parts.append(_extract_inline_text(child))
            if parts:
                quote_text = "\n".join(f"> {p}" for p in parts)
                section["content_elements"].append(quote_text)
                section["block_types"].append("blockquote")

    # Walk the AST - only process top-level children (not recursive)
    # Headings define section boundaries
    for child in ast.children:
        process_node(child)

    # Finalize last section
    finalize_current_section()

    # If no sections found (no headings), create a single section
    if not sections:
        title = _extract_title_from_ast(ast) or _extract_title_from_text(raw_text)
        section = _finalize_section(
            source_uri,
            {
                "level": 1,
                "title": title,
                "anchor": "content",
                "parent_path": "",
                "line_start": 0,
                "line_end": None,
                "content_elements": [raw_text],
                "code_blocks": [],
                "tables": [],
                "block_types": ["paragraph"],
            },
            0,
        )
        if section:
            sections.append(section)

    return sections


def _finalize_section(
    source_uri: str, section_data: Dict[str, Any], order: int
) -> Optional[Dict[str, Any]]:
    """
    Finalize section with computed fields.

    Returns None for empty sections to prevent phantom sections.

    Args:
        source_uri: Document source URI
        section_data: Raw section data from extraction
        order: Section order index

    Returns:
        Finalized section dict or None if empty
    """
    # Combine content
    text = "\n\n".join(section_data["content_elements"])

    # Normalize for consistency
    content_norm = _normalize_text(text)

    # P1: Handle heading-only sections (headings with no body content)
    # Instead of skipping empty sections, emit minimal "heading-only" sections
    # when the heading has children (parent_path will be used by child sections).
    # This ensures hierarchy edges can be created for all levels.
    is_heading_only = False
    if (
        not content_norm
        and not section_data["code_blocks"]
        and not section_data["tables"]
    ):
        title = section_data.get("title", "").strip()
        level = section_data.get("level", 1)
        # Only emit heading-only sections for non-empty titles at level > 1
        # (level 1 is document title, handled separately)
        if title and level > 1:
            # Use title as minimal text for embedding
            text = title
            is_heading_only = True
        else:
            return None

    # Compute checksum from normalized content
    checksum = _section_checksum(text)

    # Use slugified anchor for consistency
    anchor = _slugify(section_data["title"])

    # Compute deterministic section ID
    section_id = _section_id(source_uri, anchor, checksum)

    # Count tokens (simple whitespace split)
    tokens = len(text.split())

    # Compute structural metadata
    block_types = section_data.get("block_types", [])
    code_count = block_types.count("code")
    total_blocks = len(block_types) if block_types else 1
    code_ratio = code_count / total_blocks if total_blocks > 0 else 0.0

    # Convert code_blocks to legacy format (list of strings)
    code_blocks_legacy = [cb["code"] for cb in section_data.get("code_blocks", [])]

    return {
        "id": section_id,
        "document_id": _compute_document_id(source_uri),
        "level": section_data["level"],
        "title": section_data["title"],
        "anchor": anchor,
        "order": order,
        "text": text,
        "tokens": tokens,
        "checksum": checksum,
        "code_blocks": code_blocks_legacy,
        "tables": section_data.get("tables", []),
        # Vector embedding fields (populated later by embedding pipeline)
        "vector_embedding": None,
        "embedding_version": None,
        # NEW: markdown-it-py enhanced metadata
        "line_start": section_data.get("line_start"),
        "line_end": section_data.get("line_end"),
        "parent_path": section_data.get("parent_path", ""),
        "block_types": block_types,
        "code_ratio": round(code_ratio, 3),
        "has_table": len(section_data.get("tables", [])) > 0,
        "has_code": len(section_data.get("code_blocks", [])) > 0,
        # P1: Flag for heading-only sections (no body content)
        "is_heading_only": is_heading_only,
    }


# =============================================================================
# Title Extraction
# =============================================================================


def _extract_title_from_ast(ast: SyntaxTreeNode) -> Optional[str]:
    """
    Extract document title from first heading in AST.

    Args:
        ast: Parsed SyntaxTreeNode

    Returns:
        Title string or None if no heading found
    """
    for child in ast.children:
        if child.type == "heading":
            return _get_heading_text(child)
    return None


def _extract_title_from_text(raw_text: str) -> str:
    """
    Extract title from raw text using regex.

    Falls back to first line or "Untitled" if no heading found.

    Args:
        raw_text: Raw markdown text

    Returns:
        Extracted title string
    """
    # Look for first heading
    match = re.search(r"^#+\s*(.+)$", raw_text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to first non-empty line
    for line in raw_text.split("\n"):
        line = line.strip()
        if line:
            return line[:100]  # Truncate long first lines

    return "Untitled"


# =============================================================================
# Main API (compatible with legacy parser)
# =============================================================================


def parse_markdown(source_uri: str, raw_text: str) -> Dict[str, Any]:
    """
    Parse Markdown document into Document and Sections with deterministic IDs.

    This function provides API compatibility with the legacy markdown parser
    while using markdown-it-py internally for improved parsing quality.

    Args:
        source_uri: Source URI/path of the document
        raw_text: Raw markdown text

    Returns:
        Dict with 'Document' and 'Sections' keys, compatible with legacy format
    """
    logger.info("Parsing markdown document (markdown-it-py)", source_uri=source_uri)

    # Extract frontmatter - try plugin first, fall back to regex
    frontmatter, content = extract_frontmatter_regex(raw_text)

    # Parse to AST (from content without frontmatter)
    try:
        ast = parse_to_ast(content)
    except Exception as e:
        logger.error("Failed to parse markdown to AST", error=str(e))
        # Fall back to minimal parsing
        ast = SyntaxTreeNode([])

    # Try to get additional frontmatter from AST (if plugin captured it)
    ast_frontmatter = extract_frontmatter_from_ast(ast)
    if ast_frontmatter and not frontmatter:
        frontmatter = ast_frontmatter

    # Create Document metadata
    document_id = _compute_document_id(source_uri)
    title = (
        frontmatter.get("title")
        or _extract_title_from_ast(ast)
        or _extract_title_from_text(content)
    )
    checksum = _compute_checksum(raw_text)  # Use original for consistency

    document = {
        "id": document_id,
        "source_uri": source_uri,
        "source_type": "markdown",
        "title": title,
        "version": frontmatter.get("version", "1.0"),
        "checksum": checksum,
        "last_edited": frontmatter.get("last_edited"),
        # NEW: Preserve frontmatter for downstream use
        "frontmatter": frontmatter,
    }

    # Extract sections from AST
    sections = extract_sections(ast, source_uri, content)

    logger.info(
        "Markdown parsed successfully (markdown-it-py)",
        source_uri=source_uri,
        sections_count=len(sections),
        has_frontmatter=bool(frontmatter),
    )

    return {"Document": document, "Sections": sections}
