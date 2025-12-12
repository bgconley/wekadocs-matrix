"""
Section metadata helpers for markdown-it-py enhanced parsing.

This module provides utilities for extracting and propagating
structural metadata from parsed sections to chunks.

Phase 2 of markdown-it-py integration: Enhanced metadata for downstream pipeline.
Phase 5 additions: Derived fields for query-type adaptive retrieval.

Design principles:
1. Additive only - new fields don't break existing consumers
2. Passthrough pattern - section metadata flows unchanged where possible
3. Defensive extraction - graceful fallbacks for missing fields
4. Derived fields computed at extraction time for consistency
"""

from typing import Any, Dict, List

# === Field name constants ===
# Use constants to ensure consistency across modules

# Source line mapping (for NER correlation, citations)
FIELD_LINE_START = "line_start"
FIELD_LINE_END = "line_end"

# Heading hierarchy (for context enrichment, graph edges)
FIELD_PARENT_PATH = "parent_path"

# Block composition (for query-type adaptive retrieval)
FIELD_BLOCK_TYPES = "block_types"
FIELD_CODE_RATIO = "code_ratio"
FIELD_HAS_CODE = "has_code"
FIELD_HAS_TABLE = "has_table"

# === Phase 5: Derived fields for structural retrieval ===
# These are computed from existing fields for efficient filtering
FIELD_PARENT_PATH_DEPTH = "parent_path_depth"  # Nesting level (0=root)
FIELD_BLOCK_TYPE = "block_type"  # Dominant block type for filtering

# All enhanced metadata fields (for iteration)
ENHANCED_METADATA_FIELDS = [
    FIELD_LINE_START,
    FIELD_LINE_END,
    FIELD_PARENT_PATH,
    FIELD_BLOCK_TYPES,
    FIELD_CODE_RATIO,
    FIELD_HAS_CODE,
    FIELD_HAS_TABLE,
    # Phase 5 derived fields
    FIELD_PARENT_PATH_DEPTH,
    FIELD_BLOCK_TYPE,
]


# === Phase 5: Derived field computation ===


def compute_parent_path_depth(parent_path: str) -> int:
    """
    Compute nesting depth from parent_path string.

    Depth is the number of heading levels above this section:
    - "" (root/no parent) -> 0
    - "Getting Started" -> 1
    - "Getting Started > Installation" -> 2
    - "Getting Started > Installation > Prerequisites" -> 3

    Args:
        parent_path: Heading hierarchy string with " > " separators

    Returns:
        Integer depth (0 for root sections)

    Example:
        >>> compute_parent_path_depth("")
        0
        >>> compute_parent_path_depth("Getting Started > Installation")
        2
    """
    if not parent_path or not parent_path.strip():
        return 0
    # Count the number of segments (separated by " > ")
    return len([p.strip() for p in parent_path.split(" > ") if p.strip()])


def compute_dominant_block_type(block_types: List[str], code_ratio: float = 0.0) -> str:
    """
    Compute dominant block type from block_types list.

    Returns a single keyword for efficient Qdrant KEYWORD filtering.

    Classification logic:
    1. If code_ratio > 0.5 OR most blocks are "code" -> "code"
    2. If any block is "table" -> "table" (tables are distinctive)
    3. If mixed block types -> "mixed"
    4. Otherwise use most common type or "paragraph" as default

    Args:
        block_types: List of block types (e.g., ["paragraph", "code", "paragraph"])
        code_ratio: Fraction of content that is code (0.0-1.0)

    Returns:
        Single block type keyword: "paragraph", "code", "table", "list", or "mixed"

    Example:
        >>> compute_dominant_block_type(["code", "code", "paragraph"], 0.7)
        "code"
        >>> compute_dominant_block_type(["paragraph", "table"])
        "table"
    """
    if not block_types:
        return "paragraph"

    # Normalize block types
    normalized = [bt.lower().strip() for bt in block_types if bt]
    if not normalized:
        return "paragraph"

    # High code ratio -> code dominant
    if code_ratio > 0.5:
        return "code"

    # Count occurrences
    type_counts: Dict[str, int] = {}
    for bt in normalized:
        type_counts[bt] = type_counts.get(bt, 0) + 1

    # Tables are distinctive - if present, mark as table
    if "table" in type_counts:
        return "table"

    # Find most common type
    most_common = max(type_counts.keys(), key=lambda k: type_counts[k])
    most_common_count = type_counts[most_common]

    # If no single type dominates (>50%), mark as mixed
    total = len(normalized)
    if most_common_count <= total / 2 and len(type_counts) > 1:
        return "mixed"

    return most_common


def extract_enhanced_metadata(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract enhanced metadata fields from a parsed section.

    Defensively extracts all markdown-it-py enhanced fields with
    appropriate defaults for backward compatibility with legacy parser.

    Phase 5: Also computes derived fields (parent_path_depth, block_type)
    for efficient Qdrant filtering.

    Args:
        section: Section dict from parse_markdown()

    Returns:
        Dict containing only the enhanced metadata fields
    """
    # Extract base fields
    parent_path = section.get(FIELD_PARENT_PATH, "")
    block_types = section.get(FIELD_BLOCK_TYPES, [])
    code_ratio = section.get(FIELD_CODE_RATIO, 0.0)

    return {
        FIELD_LINE_START: section.get(FIELD_LINE_START),
        FIELD_LINE_END: section.get(FIELD_LINE_END),
        FIELD_PARENT_PATH: parent_path,
        FIELD_BLOCK_TYPES: block_types,
        FIELD_CODE_RATIO: code_ratio,
        FIELD_HAS_CODE: section.get(FIELD_HAS_CODE, False),
        FIELD_HAS_TABLE: section.get(FIELD_HAS_TABLE, False),
        # Phase 5: Derived fields for structural retrieval
        FIELD_PARENT_PATH_DEPTH: compute_parent_path_depth(parent_path),
        FIELD_BLOCK_TYPE: compute_dominant_block_type(block_types, code_ratio),
    }


def has_enhanced_metadata(section: Dict[str, Any]) -> bool:
    """
    Check if section has any enhanced metadata from markdown-it-py.

    Useful for determining if new parser was used.

    Args:
        section: Section dict from parse_markdown()

    Returns:
        True if any enhanced metadata field is present and non-default
    """
    # Check for definitive marker: line_start is only set by markdown-it-py
    line_start = section.get(FIELD_LINE_START)
    if line_start is not None:
        return True

    # Also check parent_path as secondary marker
    parent_path = section.get(FIELD_PARENT_PATH)
    if parent_path and parent_path != "":
        return True

    return False


def build_context_prefix(
    parent_path: str,
    heading: str,
    include_brackets: bool = True,
) -> str:
    """
    Build context prefix string for chunk enrichment.

    Following md2chunks pattern: prepend structural context to chunk text
    for better semantic understanding during embedding.

    Args:
        parent_path: Heading hierarchy string (e.g., "Getting Started > Installation")
        heading: Current section heading
        include_brackets: Whether to wrap in [Section: ...] format

    Returns:
        Context prefix string, empty if no context available

    Example:
        >>> build_context_prefix("Getting Started", "Prerequisites")
        "[Section: Getting Started > Prerequisites]"
    """
    if parent_path and heading:
        full_path = f"{parent_path} > {heading}"
    elif heading:
        full_path = heading
    elif parent_path:
        full_path = parent_path
    else:
        return ""

    if include_brackets:
        return f"[Section: {full_path}]"
    return full_path


def merge_enhanced_metadata_to_chunk(
    chunk: Dict[str, Any],
    section: Dict[str, Any],
    *,
    semantic_index: int = 0,
    semantic_total: int = 1,
) -> Dict[str, Any]:
    """
    Merge enhanced section metadata into a chunk dict.

    For chunks that are semantic splits of a section, adjusts line numbers
    proportionally (approximate, since we don't have exact positions).

    Phase 5: Also includes derived fields (parent_path_depth, block_type).

    Args:
        chunk: Chunk dict to enhance
        section: Source section with enhanced metadata
        semantic_index: Index of this chunk within semantic split (0-based)
        semantic_total: Total chunks from semantic split

    Returns:
        Enhanced chunk dict (modified in place and returned)
    """
    enhanced = extract_enhanced_metadata(section)

    # Direct passthrough fields
    chunk[FIELD_PARENT_PATH] = enhanced[FIELD_PARENT_PATH]
    chunk[FIELD_BLOCK_TYPES] = enhanced[FIELD_BLOCK_TYPES]
    chunk[FIELD_CODE_RATIO] = enhanced[FIELD_CODE_RATIO]
    chunk[FIELD_HAS_CODE] = enhanced[FIELD_HAS_CODE]
    chunk[FIELD_HAS_TABLE] = enhanced[FIELD_HAS_TABLE]

    # Phase 5: Derived fields for structural retrieval
    chunk[FIELD_PARENT_PATH_DEPTH] = enhanced[FIELD_PARENT_PATH_DEPTH]
    chunk[FIELD_BLOCK_TYPE] = enhanced[FIELD_BLOCK_TYPE]

    # Line numbers: approximate for split chunks
    line_start = enhanced[FIELD_LINE_START]
    line_end = enhanced[FIELD_LINE_END]

    if line_start is not None and line_end is not None and semantic_total > 1:
        # Approximate line range for this chunk within the section
        total_lines = max(1, line_end - line_start)
        lines_per_chunk = total_lines / semantic_total
        chunk[FIELD_LINE_START] = int(line_start + (semantic_index * lines_per_chunk))
        chunk[FIELD_LINE_END] = int(
            line_start + ((semantic_index + 1) * lines_per_chunk)
        )
    else:
        chunk[FIELD_LINE_START] = line_start
        chunk[FIELD_LINE_END] = line_end

    return chunk


def get_structural_filter_fields() -> List[str]:
    """
    Get field names suitable for Qdrant payload filtering.

    Returns fields that are useful for query-time structural filtering:
    - has_code: Filter for code-heavy chunks (CLI queries)
    - has_table: Filter for table-heavy chunks (reference queries)
    - code_ratio: Range filter for code density
    - parent_path: Text search on heading hierarchy
    - parent_path_depth: Integer depth for nesting level filtering (Phase 5)
    - block_type: Keyword for dominant block type filtering (Phase 5)
    - line_start/line_end: Source line range for citation (Phase 5)

    Returns:
        List of field names for Qdrant payload indexes
    """
    return [
        FIELD_HAS_CODE,
        FIELD_HAS_TABLE,
        FIELD_CODE_RATIO,
        FIELD_PARENT_PATH,
        FIELD_LINE_START,
        # Phase 5: Additional structural filter fields
        FIELD_LINE_END,
        FIELD_PARENT_PATH_DEPTH,
        FIELD_BLOCK_TYPE,
    ]
