# Implements Phase 3, Task 3.1 (Multi-format parser - Markdown)
# See: /docs/spec.md §3 (Data model - Document/Section)
# See: /docs/implementation-plan.md → Task 3.1
# See: /docs/pseudocode-reference.md → Task 3.1

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional

import markdown
from bs4 import BeautifulSoup

from src.shared.observability import get_logger

logger = get_logger(__name__)

# Fixed slug generation
_SLUG_RE = re.compile(r"[^a-z0-9]+")


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


def parse_markdown(source_uri: str, raw_text: str) -> Dict[str, any]:
    """
    Parse Markdown document into Document and Sections with deterministic IDs.

    Args:
        source_uri: Source URI/path of the document
        raw_text: Raw markdown text

    Returns:
        Dict with Document and Sections
    """
    logger.info("Parsing markdown document", source_uri=source_uri)

    # Create Document metadata
    document_id = _compute_document_id(source_uri)
    title = _extract_title(raw_text)
    checksum = _compute_checksum(raw_text)

    document = {
        "id": document_id,
        "source_uri": source_uri,
        "source_type": "markdown",
        "title": title,
        "version": "1.0",  # Can be enhanced later
        "checksum": checksum,
        "last_edited": None,  # Can be set from file metadata
    }

    # Parse sections
    sections = _parse_sections(source_uri, raw_text)

    logger.info(
        "Markdown parsed successfully",
        source_uri=source_uri,
        sections_count=len(sections),
    )

    return {"Document": document, "Sections": sections}


def _extract_title(raw_text: str) -> str:
    """Extract title from first heading or first line."""
    lines = raw_text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # Remove markdown heading markers
            return re.sub(r"^#+\s*", "", line).strip()
        elif line:
            # Use first non-empty line
            return line[:100]  # Limit length
    return "Untitled"


def _parse_sections(source_uri: str, raw_text: str) -> List[Dict[str, any]]:
    """
    Parse markdown into sections based on headings.

    Each section starts with a heading and includes content until the next heading.
    """
    sections = []

    # Convert markdown to HTML with TOC extension to get anchors
    md = markdown.Markdown(
        extensions=[
            "toc",
            "fenced_code",
            "tables",
            "codehilite",
        ]
    )
    html = md.convert(raw_text)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Build section tree from headings
    current_section = None
    order = 0

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "table", "ul", "ol"]
    ):
        if element.name.startswith("h"):
            # Save previous section if exists
            if current_section:
                finalized = _finalize_section(source_uri, current_section, order)
                if finalized:  # Only add non-empty sections
                    sections.append(finalized)
                    order += 1

            # Start new section
            level = int(element.name[1])  # h1 -> 1, h2 -> 2, etc.
            title = element.get_text().strip()
            anchor = element.get("id", _slugify(title))

            current_section = {
                "level": level,
                "title": title,
                "anchor": anchor,
                "order": order,
                "content_elements": [],
                "code_blocks": [],
                "tables": [],
            }

        elif current_section:
            # Add content to current section
            if element.name == "pre":
                # Code block
                code = element.get_text()
                current_section["code_blocks"].append(code)
                current_section["content_elements"].append(f"[CODE]\n{code}\n[/CODE]")
            elif element.name == "table":
                # Table
                table_text = _extract_table_text(element)
                current_section["tables"].append(table_text)
                current_section["content_elements"].append(
                    f"[TABLE]\n{table_text}\n[/TABLE]"
                )
            elif element.name == "ol":
                # Ordered list - preserve numbers
                list_text = _extract_ordered_list(element)
                current_section["content_elements"].append(list_text)
            elif element.name == "ul":
                # Unordered list - preserve bullets
                list_text = _extract_unordered_list(element)
                current_section["content_elements"].append(list_text)
            else:
                # Regular content
                text = element.get_text()
                current_section["content_elements"].append(text)

    # Save last section
    if current_section:
        finalized = _finalize_section(source_uri, current_section, order)
        if finalized:  # Only add non-empty sections
            sections.append(finalized)

    # If no sections found (no headings), create a single section for the whole doc
    if not sections:
        section = _finalize_section(
            source_uri,
            {
                "level": 1,
                "title": _extract_title(raw_text),
                "anchor": "content",
                "order": 0,
                "content_elements": [raw_text],
                "code_blocks": [],
                "tables": [],
            },
            0,
        )
        if section:
            sections.append(section)

    return sections


def _finalize_section(
    source_uri: str, section_data: Dict, order: int
) -> Optional[Dict[str, any]]:
    """
    Finalize section with computed fields.
    Returns None for empty sections to prevent phantom sections.
    """
    # Combine content
    text = "\n\n".join(section_data["content_elements"])

    # Normalize for consistency
    content_norm = _normalize_text(text)

    # Skip empty sections (no content, no code, no tables)
    if (
        not content_norm
        and not section_data["code_blocks"]
        and not section_data["tables"]
    ):
        return None

    # Compute checksum from normalized content
    checksum = _section_checksum(text)

    # Use slugified anchor for consistency
    anchor = _slugify(section_data["title"])

    # Compute deterministic section ID using content-coupled approach
    section_id = _section_id(source_uri, anchor, checksum)

    # Count tokens (simple whitespace split for now)
    tokens = len(text.split())

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
        "code_blocks": section_data["code_blocks"],
        "tables": section_data["tables"],
        # Vector embedding fields (populated later)
        "vector_embedding": None,
        "embedding_version": None,
    }


def _compute_document_id(source_uri: str) -> str:
    """Compute deterministic document ID from source URI."""
    normalized_uri = source_uri.strip().lower()
    return hashlib.sha256(normalized_uri.encode("utf-8")).hexdigest()


def _compute_checksum(text: str) -> str:
    """Compute SHA-256 checksum of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_table_text(table_element) -> str:
    """Extract text representation of HTML table."""
    rows = []
    for tr in table_element.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text().strip())
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_ordered_list(ol_element) -> str:
    """Extract ordered list with number markers preserved."""
    items = []
    start = int(ol_element.get("start", 1))  # Handle start attribute
    for i, li in enumerate(ol_element.find_all("li", recursive=False)):
        item_num = start + i
        item_text = li.get_text().strip()
        items.append(f"{item_num}. {item_text}")
    return "\n".join(items)


def _extract_unordered_list(ul_element) -> str:
    """Extract unordered list with bullet markers."""
    items = []
    for li in ul_element.find_all("li", recursive=False):
        item_text = li.get_text().strip()
        items.append(f"- {item_text}")
    return "\n".join(items)
