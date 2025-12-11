"""
Chunk utilities for Phase 7E-1 (Dual-Label Idempotent Ingestion).

Provides deterministic, order-preserving chunk ID generation and
chunk metadata helpers for GraphRAG v2.1.

Key invariants:
- IDs are deterministic: same inputs → same ID
- Order is preserved: never sort original_section_ids
- IDs are full SHA256 hex (64-char) to stay consistent across parsing, extraction, and storage
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def generate_chunk_id(document_id: str, original_section_ids: List[str]) -> str:
    """
    Generate deterministic, order-preserving chunk ID.

    CRITICAL: Never sort original_section_ids - order is part of the identity.

    Args:
        document_id: Document identifier
        original_section_ids: List of section IDs in preservation order

    Returns:
        64-character deterministic chunk ID (SHA256 hex)

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

    # Preserve original ID for single-section chunks to avoid ID drift
    if len(original_section_ids) == 1 and original_section_ids[0]:
        return original_section_ids[0]

    # Generate SHA256 hash (full 64-char hex)
    hash_digest = hashlib.sha256(material.encode("utf-8")).hexdigest()

    return hash_digest


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
    *,
    doc_id: Optional[str] = None,
    doc_tag: Optional[str] = None,
    tenant: Optional[str] = None,
    lang: Optional[str] = None,
    version: Optional[str] = None,
    text_hash: Optional[str] = None,
    shingle_hash: Optional[str] = None,
    # Phase 2: markdown-it-py enhanced metadata (additive, backward compatible)
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    parent_path: Optional[str] = None,
    block_types: Optional[List[str]] = None,
    code_ratio: Optional[float] = None,
    has_code: Optional[bool] = None,
    has_table: Optional[bool] = None,
    # Phase 5: Derived structural fields for query-type adaptive retrieval
    parent_path_depth: Optional[int] = None,
    block_type: Optional[str] = None,
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
        line_start: Source line number start (Phase 2: markdown-it-py)
        line_end: Source line number end (Phase 2: markdown-it-py)
        parent_path: Heading hierarchy path (Phase 2: markdown-it-py)
        block_types: List of block types in section (Phase 2: markdown-it-py)
        code_ratio: Fraction of content that is code (Phase 2: markdown-it-py)
        has_code: Whether section contains code blocks (Phase 2: markdown-it-py)
        has_table: Whether section contains tables (Phase 2: markdown-it-py)

    Returns:
        Dict with chunk metadata fields
    """
    # For single-section chunks, original_section_ids contains just this section
    original_section_ids = [section_id]

    # Generate deterministic chunk ID (aligned with original 64-char section_id)
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
        "doc_id": doc_id or document_id,
        "doc_tag": doc_tag,
        "tenant": tenant,
        "lang": lang,
        "version": version,
        "text_hash": text_hash,
        "shingle_hash": shingle_hash,
        "updated_at": datetime.utcnow(),
        # Phase 2: markdown-it-py enhanced metadata (None if not available)
        "line_start": line_start,
        "line_end": line_end,
        "parent_path": parent_path or "",
        "block_types": block_types or [],
        "code_ratio": code_ratio if code_ratio is not None else 0.0,
        "has_code": has_code if has_code is not None else False,
        "has_table": has_table if has_table is not None else False,
        # Phase 5: Derived structural fields
        "parent_path_depth": parent_path_depth if parent_path_depth is not None else 0,
        "block_type": block_type if block_type is not None else "paragraph",
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


def create_combined_chunk_metadata(
    document_id: str,
    original_section_ids: List[str],
    level: int,
    order: int,
    heading: Optional[str] = None,
    parent_section_id: Optional[str] = None,
    token_count: Optional[int] = None,
    boundaries: Optional[Dict] = None,
    *,
    doc_id: Optional[str] = None,
    doc_tag: Optional[str] = None,
    tenant: Optional[str] = None,
    lang: Optional[str] = None,
    version: Optional[str] = None,
    text_hash: Optional[str] = None,
    shingle_hash: Optional[str] = None,
) -> Dict:
    """
    Create chunk metadata dict for a combined chunk (multiple sections).

    Phase 7E-2: Used when combining multiple small sections into a single chunk
    to achieve target token counts (800-1500).

    Args:
        document_id: Parent document identifier
        original_section_ids: List of section IDs combined (in order)
        level: Minimum heading depth across combined sections
        order: Position within document
        heading: Heading from first section
        parent_section_id: Logical parent (typically first section's ID)
        token_count: Total token count of combined text
        boundaries: Optional dict with section boundaries metadata

    Returns:
        Dict with chunk metadata fields
    """
    import json

    # Generate deterministic chunk ID from all combined sections
    chunk_id = generate_chunk_id(document_id, original_section_ids)

    # Serialize boundaries if provided
    boundaries_json = (
        json.dumps(boundaries, separators=(",", ":")) if boundaries else "{}"
    )

    return {
        "id": chunk_id,
        "document_id": document_id,
        "level": level,
        "order": order,
        "heading": heading or "",
        "parent_section_id": parent_section_id,
        "original_section_ids": original_section_ids,
        "is_combined": True,
        "is_split": False,
        "boundaries_json": boundaries_json,
        "token_count": token_count or 0,
        "doc_id": doc_id or document_id,
        "doc_tag": doc_tag,
        "tenant": tenant,
        "lang": lang,
        "version": version,
        "text_hash": text_hash,
        "shingle_hash": shingle_hash,
        "updated_at": datetime.utcnow(),
    }


def canonicalize_parent_ids(chunks: List[Dict]) -> Tuple[int, int]:
    """
    Ensure parent references point to canonical chunk IDs.

    Returns:
        Tuple (remapped_count, missing_parent_count)
    """
    if not chunks:
        return 0, 0

    original_to_chunk: Dict[str, str] = {}
    for chunk in chunks:
        originals = chunk.get("original_section_ids") or []
        if not originals:
            originals = [chunk.get("id")]
        for original in originals:
            if original and original not in original_to_chunk:
                original_to_chunk[original] = chunk.get("id")

    remapped = 0
    missing = 0
    for chunk in chunks:
        parent_original = chunk.get("parent_section_id")
        if not parent_original:
            continue

        chunk["parent_section_original_id"] = parent_original
        mapped_parent = original_to_chunk.get(parent_original)
        if mapped_parent:
            chunk["parent_chunk_id"] = mapped_parent
            chunk["parent_section_id"] = mapped_parent
            if mapped_parent != parent_original:
                remapped += 1
        else:
            chunk["parent_chunk_id"] = None
            chunk["parent_section_id"] = None
            missing += 1

    return remapped, missing
