"""
Chunk utilities for Phase 7E-1 (Dual-Label Idempotent Ingestion).

Provides deterministic, order-preserving chunk ID generation and
chunk metadata helpers for GraphRAG v2.1.

Key invariants:
- IDs are deterministic: same inputs → same ID
- Order is preserved: never sort original_section_ids
- IDs are 24-char SHA256 prefixes for Neo4j compatibility
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional


def generate_chunk_id(document_id: str, original_section_ids: List[str]) -> str:
    """
    Generate deterministic, order-preserving chunk ID.

    CRITICAL: Never sort original_section_ids - order is part of the identity.

    Args:
        document_id: Document identifier
        original_section_ids: List of section IDs in preservation order

    Returns:
        24-character deterministic chunk ID (SHA256 prefix)

    Example:
        >>> generate_chunk_id("doc_123", ["sec_1", "sec_2"])
        'a1b2c3d4e5f6a7b8c9d0aa11'  # Same every time  # pragma: allowlist secret
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")

    if not original_section_ids:
        raise ValueError("original_section_ids cannot be empty")

    # CRITICAL: Preserve order - DO NOT sort!
    # Format: document_id|section_1|section_2|...
    material = f"{document_id}|{'|'.join(original_section_ids)}"

    # Generate SHA256 hash and take first 24 chars
    hash_digest = hashlib.sha256(material.encode("utf-8")).hexdigest()

    return hash_digest[:24]


def create_chunk_metadata(
    section_id: str,
    document_id: str,
    level: int,
    order: int,
    heading: Optional[str] = None,
    parent_section_id: Optional[str] = None,
    is_combined: bool = False,
    is_split: bool = False,
    boundaries_json: Optional[str] = None,
    token_count: Optional[int] = None,
) -> Dict:
    """
    Create chunk metadata dict for a single-section chunk (Phase 7E-1 default).

    In Phase 7E-1, each section becomes its own chunk. This helper creates
    the required chunk fields following the canonical schema.

    Args:
        section_id: Original section identifier
        document_id: Parent document identifier
        level: Heading depth (1-6)
        order: Position within parent
        heading: Optional section heading
        parent_section_id: Optional logical parent
        is_combined: Whether chunk combines multiple sections (default False)
        is_split: Whether section was split into multiple chunks (default False)
        boundaries_json: Optional JSON-serialized boundary metadata
        token_count: Optional token count

    Returns:
        Dict with chunk metadata fields
    """
    # For single-section chunks, original_section_ids contains just this section
    original_section_ids = [section_id]

    # Generate deterministic chunk ID
    chunk_id = generate_chunk_id(document_id, original_section_ids)

    return {
        "id": chunk_id,
        "document_id": document_id,
        "level": level,
        "order": order,
        "heading": heading or "",
        "parent_section_id": parent_section_id,
        "original_section_ids": original_section_ids,
        "is_combined": is_combined,
        "is_split": is_split,
        "boundaries_json": boundaries_json or "{}",
        "token_count": token_count or 0,
        "updated_at": datetime.utcnow(),
    }


def validate_chunk_schema(chunk: Dict) -> bool:
    """
    Validate chunk has all required schema fields.

    Required fields per canonical spec:
    - id, document_id, level, order
    - original_section_ids (non-empty list)
    - is_combined, is_split (booleans)
    - token_count (integer)

    Args:
        chunk: Chunk dict to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "id",
        "document_id",
        "level",
        "order",
        "original_section_ids",
        "is_combined",
        "is_split",
        "token_count",
    ]

    for field in required_fields:
        if field not in chunk:
            return False

    # Validate original_section_ids is non-empty list
    if not isinstance(chunk["original_section_ids"], list):
        return False

    if len(chunk["original_section_ids"]) == 0:
        return False

    # Validate booleans
    if not isinstance(chunk["is_combined"], bool):
        return False

    if not isinstance(chunk["is_split"], bool):
        return False

    return True
