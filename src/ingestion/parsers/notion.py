# Implements Phase 3, Task 3.1 (Multi-format parser - Notion)
# See: /docs/spec.md §3 (Data model - Document/Section)
# See: /docs/implementation-plan.md → Task 3.1
# See: /docs/pseudocode-reference.md → Task 3.1

import hashlib
import os
import re
from typing import Dict, List, Optional

from src.shared.observability import get_logger

logger = get_logger(__name__)


def parse_notion(page_id: str, api_key: Optional[str] = None) -> Dict[str, any]:
    """
    Parse Notion page into Document and Sections with deterministic IDs.

    Note: Requires Notion API credentials. Can be skipped in tests if not configured.

    Args:
        page_id: Notion page ID
        api_key: Notion API key (optional, reads from env if not provided)

    Returns:
        Dict with Document and Sections
    """
    api_key = api_key or os.getenv("NOTION_API_KEY")

    if not api_key:
        logger.warning(
            "Notion API key not configured, skipping Notion parsing",
            page_id=page_id,
        )
        raise ValueError("NOTION_API_KEY not configured")

    try:
        # This would use the official notion-client library
        # For now, provide a stub implementation
        from notion_client import Client

        notion = Client(auth=api_key)

        # Fetch page
        page = notion.pages.retrieve(page_id=page_id)
        blocks = _fetch_all_blocks(notion, page_id)

        # Create document
        source_uri = f"notion://{page_id}"
        document_id = _compute_document_id(source_uri)
        title = _extract_title(page)
        checksum = _compute_checksum(str(blocks))

        document = {
            "id": document_id,
            "source_uri": source_uri,
            "source_type": "notion",
            "title": title,
            "version": "1.0",
            "checksum": checksum,
            "last_edited": page.get("last_edited_time"),
        }

        # Parse sections from blocks
        sections = _parse_sections(source_uri, blocks)

        logger.info(
            "Notion page parsed successfully",
            page_id=page_id,
            sections_count=len(sections),
        )

        return {"Document": document, "Sections": sections}

    except ImportError:
        logger.error("notion-client library not installed")
        raise ImportError("Install notion-client: pip install notion-client")

    except Exception as e:
        logger.error("Failed to parse Notion page", page_id=page_id, error=str(e))
        raise


def _fetch_all_blocks(notion_client, block_id: str) -> List[Dict]:
    """Recursively fetch all blocks from a Notion page."""
    blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        response = notion_client.blocks.children.list(
            block_id=block_id, start_cursor=start_cursor
        )
        blocks.extend(response["results"])
        has_more = response["has_more"]
        start_cursor = response.get("next_cursor")

    return blocks


def _extract_title(page: Dict) -> str:
    """Extract title from Notion page object."""
    try:
        title_property = page.get("properties", {}).get("title", {})
        if title_property:
            title_content = title_property.get("title", [])
            if title_content:
                return title_content[0].get("plain_text", "Untitled")
    except Exception:
        pass
    return "Untitled"


def _parse_sections(source_uri: str, blocks: List[Dict]) -> List[Dict[str, any]]:
    """
    Parse Notion blocks into sections.

    Sections start with heading blocks.
    """
    sections = []
    current_section = None
    order = 0

    for block in blocks:
        block_type = block.get("type")

        # Heading blocks start new sections
        if block_type in ["heading_1", "heading_2", "heading_3"]:
            # Save previous section
            if current_section:
                sections.append(_finalize_section(source_uri, current_section, order))
                order += 1

            # Start new section
            level = int(block_type.split("_")[1])
            title = _extract_block_text(block)
            anchor = _slugify(title)

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
            text = _extract_block_text(block)

            if block_type == "code":
                current_section["code_blocks"].append(text)
                current_section["content_elements"].append(f"[CODE]\n{text}\n[/CODE]")
            elif block_type == "table":
                current_section["tables"].append(text)
                current_section["content_elements"].append(f"[TABLE]\n{text}\n[/TABLE]")
            elif text:
                current_section["content_elements"].append(text)

    # Save last section
    if current_section:
        sections.append(_finalize_section(source_uri, current_section, order))

    return sections


def _extract_block_text(block: Dict) -> str:
    """Extract text from a Notion block."""
    block_type = block.get("type")
    block_content = block.get(block_type, {})

    # Get rich text
    rich_text = block_content.get("rich_text", [])
    if rich_text:
        return "".join([t.get("plain_text", "") for t in rich_text])

    # For code blocks
    if block_type == "code":
        return "".join([t.get("plain_text", "") for t in rich_text])

    return ""


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
