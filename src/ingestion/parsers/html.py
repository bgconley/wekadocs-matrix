# Implements Phase 3, Task 3.1 (Multi-format parser - HTML)
# See: /docs/spec.md §3 (Data model - Document/Section)
# See: /docs/implementation-plan.md → Task 3.1
# See: /docs/pseudocode-reference.md → Task 3.1

import hashlib
import re
from typing import Dict, List

from bs4 import BeautifulSoup

from src.shared.observability import get_logger

logger = get_logger(__name__)


def parse_html(source_uri: str, raw_html: str) -> Dict[str, any]:
    """
    Parse HTML document into Document and Sections with deterministic IDs.

    Args:
        source_uri: Source URI/path of the document
        raw_html: Raw HTML text

    Returns:
        Dict with Document and Sections
    """
    logger.info("Parsing HTML document", source_uri=source_uri)

    # Parse HTML
    soup = BeautifulSoup(raw_html, "html.parser")

    # Create Document metadata
    document_id = _compute_document_id(source_uri)
    title = _extract_title(soup)
    checksum = _compute_checksum(raw_html)

    document = {
        "id": document_id,
        "source_uri": source_uri,
        "source_type": "html",
        "title": title,
        "version": "1.0",
        "checksum": checksum,
        "last_edited": None,
    }

    # Parse sections
    sections = _parse_sections(source_uri, soup)

    logger.info(
        "HTML parsed successfully",
        source_uri=source_uri,
        sections_count=len(sections),
    )

    return {"Document": document, "Sections": sections}


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract title from HTML."""
    # Try <title> tag first
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text().strip()

    # Try first h1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text().strip()

    # Try meta title
    meta_title = soup.find("meta", attrs={"property": "og:title"})
    if meta_title and meta_title.get("content"):
        return meta_title.get("content").strip()

    return "Untitled"


def _parse_sections(source_uri: str, soup: BeautifulSoup) -> List[Dict[str, any]]:
    """
    Parse HTML into sections based on headings.
    """
    sections = []

    # Find main content area (skip nav, header, footer)
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|article|main", re.I))
        or soup.find("body")
    )

    if not main_content:
        main_content = soup

    # Build sections from headings
    current_section = None
    order = 0

    for element in main_content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "pre",
            "code",
            "table",
            "ul",
            "ol",
            "div",
        ]
    ):
        # Skip navigation elements
        if element.find_parent(["nav", "header", "footer"]):
            continue

        if element.name.startswith("h"):
            # Save previous section
            if current_section:
                sections.append(_finalize_section(source_uri, current_section, order))
                order += 1

            # Start new section
            level = int(element.name[1])
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
            if element.name in ["pre", "code"] and not element.find_parent("pre"):
                # Code block
                code = element.get_text()
                if code.strip():
                    current_section["code_blocks"].append(code)
                    current_section["content_elements"].append(
                        f"[CODE]\n{code}\n[/CODE]"
                    )

            elif element.name == "table":
                # Table
                table_text = _extract_table_text(element)
                if table_text.strip():
                    current_section["tables"].append(table_text)
                    current_section["content_elements"].append(
                        f"[TABLE]\n{table_text}\n[/TABLE]"
                    )

            elif element.name in ["p", "ul", "ol", "div"]:
                # Regular content
                text = element.get_text(separator=" ", strip=True)
                if text.strip() and len(text) > 10:  # Skip very short snippets
                    current_section["content_elements"].append(text)

    # Save last section
    if current_section:
        sections.append(_finalize_section(source_uri, current_section, order))

    # If no sections found, create one from all text
    if not sections:
        all_text = main_content.get_text(separator="\n", strip=True)
        sections.append(
            _finalize_section(
                source_uri,
                {
                    "level": 1,
                    "title": _extract_title(soup),
                    "anchor": "content",
                    "order": 0,
                    "content_elements": [all_text],
                    "code_blocks": [],
                    "tables": [],
                },
                0,
            )
        )

    return sections


def _finalize_section(
    source_uri: str, section_data: Dict, order: int
) -> Dict[str, any]:
    """Finalize section with computed fields."""
    text = "\n\n".join(section_data["content_elements"])
    normalized_text = _normalize_text(text)
    section_id = _compute_section_id(
        source_uri, section_data["anchor"], normalized_text
    )
    checksum = _compute_checksum(text)
    tokens = len(text.split())

    return {
        "id": section_id,
        "document_id": _compute_document_id(source_uri),
        "level": section_data["level"],
        "title": section_data["title"],
        "anchor": section_data["anchor"],
        "order": order,
        "text": text,
        "tokens": tokens,
        "checksum": checksum,
        "code_blocks": section_data["code_blocks"],
        "tables": section_data["tables"],
        "vector_embedding": None,
        "embedding_version": None,
    }


def _compute_document_id(source_uri: str) -> str:
    """Compute deterministic document ID from source URI."""
    normalized_uri = source_uri.strip().lower()
    return hashlib.sha256(normalized_uri.encode("utf-8")).hexdigest()


def _compute_section_id(source_uri: str, anchor: str, normalized_text: str) -> str:
    """Compute deterministic section ID."""
    id_input = f"{source_uri.strip()}#{anchor.strip()}{normalized_text}"
    return hashlib.sha256(id_input.encode("utf-8")).hexdigest()


def _compute_checksum(text: str) -> str:
    """Compute SHA-256 checksum of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    """Normalize text for consistent hashing."""
    text = text.strip()
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    return text


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text or "section"


def _extract_table_text(table_element) -> str:
    """Extract text representation of HTML table."""
    rows = []
    for tr in table_element.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text().strip())
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)
