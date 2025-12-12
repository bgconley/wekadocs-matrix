# Phase 3: Cross-Document Reference Extraction
# Implements (Chunk)-[:REFERENCES]->(Document) edge pattern
# See: /docs/cdx-outputs/2025-11-28-graphrag-architecture-enhancement-plan-v2-CANONICAL.md
#
# Multi-Model Consensus Decisions (2025-11-29):
# - Edge pattern: Chunk→Document (hybrid - source preserves provenance, target is reliable)
# - Target resolution: Neo4j indexed lookup on Document.title (no Redis)
# - Idempotency: MERGE with uniqueness constraint
# - Gate criteria: count >= 10
#
# Known Limitations (documented per multi-model review):
# - H2/M3: Graph traversal architecture is entity-centric; REFERENCES edges won't
#   boost scores via graph reranker until Phase 3.1 adds cross-document expansion
# - L2: Nested brackets in hyperlinks not handled (edge case, rare in practice)
# - L3: refer_to pattern requires specific suffixes (Guide, Manual, Doc, etc.)

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Prefer 'regex' for timeout support; fall back to stdlib re if unavailable
try:
    import regex as re  # type: ignore

    _REGEX_SUPPORTS_TIMEOUT = True
except Exception:  # pragma: no cover
    import re

    _REGEX_SUPPORTS_TIMEOUT = False

try:  # Best-effort config overrides; falls back to module defaults if unavailable
    from src.shared.config import get_config

    _cfg = get_config()
    _ref_cfg = getattr(_cfg, "references", None)
    _extract_cfg = getattr(_ref_cfg, "extraction", None) if _ref_cfg else None
except (
    Exception
):  # pragma: no cover - defensive; avoids import cycles during early init
    _extract_cfg = None

from src.shared.observability import get_logger

logger = get_logger(__name__)

# M6: ReDoS protection - limit text length before regex matching
# Prevents catastrophic backtracking on adversarially crafted or very large chunks
MAX_TEXT_LENGTH_FOR_REGEX = (
    getattr(_extract_cfg, "max_text_length", None) or 8192
)  # 8KB default
WINDOW_OVERLAP = getattr(_extract_cfg, "window_overlap", None) or 512  # default overlap
REGEX_TIMEOUT_SEC = getattr(_extract_cfg, "timeout_sec", None) or 1.0

# Regex for slugify: matches non-alphanumeric sequences
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_for_id(text: str) -> str:
    """
    Create deterministic, ASCII-safe slug for use in IDs (e.g., ghost document IDs).

    Fix #1 (Phase 3): Ensures consistent ghost IDs across OS/locales by normalizing
    Unicode characters before conversion. Without this, titles like "Björk's Guide"
    could produce different IDs on different systems.

    The normalization pipeline:
    1. NFKD normalize - decomposes characters (é → e + combining accent)
    2. ASCII encode with ignore - drops non-ASCII (accents, CJK, etc.)
    3. Lowercase for case-insensitive matching
    4. Replace non-alphanumeric sequences with hyphens
    5. Strip leading/trailing hyphens

    Examples:
        "Snapshot Policies" -> "snapshot-policies"
        "Björk's Naïve Guide" -> "bjorks-naive-guide"
        "日本語ドキュメント" -> "" (returns "unknown" fallback)
        "Café Configuration" -> "cafe-configuration"

    Args:
        text: The text to slugify (typically a document title)

    Returns:
        ASCII-safe, lowercase, hyphenated slug suitable for IDs
    """
    if not text:
        return "unknown"

    # NFKD normalization decomposes characters (e.g., ñ → n + combining tilde)
    normalized = unicodedata.normalize("NFKD", text)
    # Encode to ASCII, dropping non-ASCII characters (accents, CJK, etc.)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    # Check if non-decomposable characters were lost (CJK, emoji, symbols)
    # These could be distinguishing features between similar titles
    # e.g., "Guide 🚀" vs "Guide 🌟" would both become "guide" without hash
    chars_lost = len(normalized) - len(ascii_text)
    if chars_lost > 0:
        slug = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "slugify_hash_fallback",
            original_text=text[:50],
            hash_slug=slug,
            chars_lost=chars_lost,
        )
        return slug

    # Lowercase and strip
    slug = ascii_text.lower().strip()
    # Replace non-alphanumeric sequences with hyphens
    slug = _SLUG_RE.sub("-", slug).strip("-")

    # Fallback for empty result (shouldn't happen with above check, but defensive)
    if not slug:
        slug = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "slugify_hash_fallback_empty",
            original_text=text[:50],
            hash_slug=slug,
        )

    return slug


@dataclass
class Reference:
    """Represents a detected cross-document reference."""

    reference_type: str  # 'hyperlink', 'see_also', 'related', 'refer_to'
    reference_text: str  # The matched text (e.g., "Snapshot Policies")
    target_hint: str  # Target identifier hint (filename, title, or phrase)
    confidence: float  # Extraction confidence (0.0 - 1.0)
    start: int  # Character offset in source text
    end: int  # Character offset end position


@dataclass
class ReferenceEdge:
    """Edge dict for REFERENCES relationship (Chunk → Document)."""

    source_chunk_id: str
    target_doc_id: Optional[str]  # Resolved document ID (may be None if unresolved)
    target_hint: str  # Original hint for logging/debugging
    reference_type: str
    reference_text: str
    confidence: float


# Reference detection patterns (ordered for deterministic priority)
# Priority order: hyperlink (95%), see_also (85%), related (80%), refer_to (70%)
REFERENCE_PATTERNS = [
    (
        "hyperlink",
        re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)", re.IGNORECASE),
    ),
    (
        "see_also",
        re.compile(
            r"see\s+(?:also)?:?\s*\[([^\]]+)\]\([^)]+\)|"
            r"see\s+(?:also)?:?\s+(?P<plain_text>[A-Z][^\n\.\,]{0,200}+)(?:\s*[-–—]|$|\n|\.)",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "related",
        re.compile(
            r"related:?\s*\[([^\]]+)\]\([^)]+\)|"
            r"related:?\s+(?P<plain_text>[A-Z][^\n\.\,]{0,200}+)(?:\s*[-–—]|$|\n)",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "refer_to",
        re.compile(
            r"(?:refer\s+to|see)\s+(?:the\s+)?([A-Z][^\.\n]{0,200}?(?:Guide|Manual|Doc|Documentation|Configuration))",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
]

# Confidence scores by pattern type
CONFIDENCE_SCORES = getattr(_extract_cfg, "confidence_scores", None) or {
    "hyperlink": 0.95,
    "see_also": 0.85,
    "related": 0.80,
    "refer_to": 0.70,
}

# L1: Removed unused _hash16 helper function (was dead code)


def extract_references(text: str, source_chunk_id: str) -> List[Reference]:
    """
    Extract cross-document references from text.

    Args:
        text: The chunk text to analyze
        source_chunk_id: The ID of the source chunk (for context)

    Returns:
        List of detected Reference objects
    """
    if not text:
        return []

    references = []
    seen_targets = set()  # Deduplicate by target hint

    window_size = MAX_TEXT_LENGTH_FOR_REGEX
    step = window_size - WINDOW_OVERLAP
    window_starts = range(0, len(text), step) if len(text) > window_size else [0]

    for start in window_starts:
        window = text[start : start + window_size]

        for ref_type, pattern in REFERENCE_PATTERNS:
            confidence = CONFIDENCE_SCORES[ref_type]

            finditer_kwargs = (
                {"timeout": REGEX_TIMEOUT_SEC} if _REGEX_SUPPORTS_TIMEOUT else {}
            )
            for match in pattern.finditer(window, **finditer_kwargs):
                # Extract the target hint from match groups
                if ref_type == "hyperlink":
                    target_hint = match.group(2)  # filename
                    reference_text = match.group(1)  # anchor text
                else:
                    target_hint = (
                        match.groupdict().get("plain_text")
                        if match.groupdict()
                        else None
                    ) or next(
                        (g.strip() for g in match.groups() if g and g.strip()),
                        None,
                    )
                    reference_text = match.group(0)

                if not target_hint:
                    continue

                normalized_hint = target_hint.lower().strip()
                if normalized_hint in seen_targets:
                    continue
                seen_targets.add(normalized_hint)

                references.append(
                    Reference(
                        reference_type=ref_type,
                        reference_text=reference_text,
                        target_hint=target_hint,
                        confidence=confidence,
                        start=start + match.start(),
                        end=start + match.end(),
                    )
                )

    if references:
        logger.debug(
            "references_extracted",
            source_chunk_id=source_chunk_id,
            count=len(references),
            types=[r.reference_type for r in references],
        )

    return references


def normalize_filename_to_title(filename: str) -> str:
    """
    Convert a markdown filename to a likely document title.

    Preserves acronyms (API, SMB), handles URL fragments/encoding, and mixed case.
    """
    # Strip fragments
    if "#" in filename:
        filename = filename.split("#")[0]

    # Strip path components
    if "/" in filename:
        filename = filename.split("/")[-1]

    # Decode common URL encodings
    try:
        from urllib.parse import unquote

        filename = unquote(filename)
    except Exception:
        pass

    # Remove .md extension (case-insensitive)
    name = re.sub(r"\\.md$", "", filename, flags=re.IGNORECASE)

    # Split on separators
    tokens = re.split(r"[-_]+", name)
    normalized = []
    for token in tokens:
        if not token:
            continue
        if token.isupper() and len(token) > 1:
            normalized.append(token)  # preserve acronym
        elif token.islower():
            normalized.append(token.capitalize())
        else:
            normalized.append(token)

    return " ".join(normalized)


def create_reference_edge(
    source_chunk_id: str,
    target_doc_id: Optional[str],
    target_hint: str,
    reference_type: str,
    reference_text: str,
    confidence: float,
) -> Dict:
    """
    Create a REFERENCES edge dict for atomic pipeline integration.

    The edge structure follows the consensus decision:
    (Chunk)-[:REFERENCES {type, confidence, reference_text}]->(Document)

    Args:
        source_chunk_id: The chunk where the reference was found
        target_doc_id: The resolved target document ID (or None if unresolved)
        target_hint: Original hint (for logging/debugging)
        reference_type: Type of reference (hyperlink, see_also, related, refer_to)
        reference_text: The display text of the reference
        confidence: Extraction confidence score

    Returns:
        Dict suitable for _neo4j_create_references in atomic.py
    """
    return {
        "source_chunk_id": source_chunk_id,
        "target_doc_id": target_doc_id,
        "target_hint": target_hint,
        "reference_type": reference_type,
        "reference_text": reference_text,
        "confidence": confidence,
        "type": "REFERENCES",  # Aligned with codebase convention (other extractors use "type")
    }


def extract_chunk_references(
    chunks: List[Dict],
    known_doc_titles: Optional[Dict[str, str]] = None,
    deduplicate_across_chunks: bool = True,
) -> Tuple[List[Dict], int, int]:
    """
    Extract references from all chunks and create edge dicts.

    This is the main entry point for reference extraction during ingestion.
    It processes all chunks, extracts references, and attempts to resolve
    target documents using the provided title mapping.

    Args:
        chunks: List of chunk dicts with 'id' and 'text' fields
        known_doc_titles: Optional mapping of normalized titles to doc_ids
                         for local resolution (before Neo4j lookup)
        deduplicate_across_chunks: M1 enhancement - if True (default), deduplicate
                         references across all chunks in the document, not just
                         per-chunk. This prevents multiple edges to the same target
                         from different chunks in the same document.

    Returns:
        Tuple of (reference_edges, resolved_count, unresolved_count)
    """
    if known_doc_titles is None:
        known_doc_titles = {}

    reference_edges = []
    resolved_count = 0
    unresolved_count = 0

    # M1: Document-scope deduplication - track (target_hint, chunk_id) pairs
    # to prevent the same reference from being extracted multiple times
    document_seen_targets: set = set() if deduplicate_across_chunks else None

    for chunk in chunks:
        chunk_id = chunk.get("id")
        text = chunk.get("text", "")

        if not chunk_id or not text:
            continue

        # Extract references from this chunk
        refs = extract_references(text, chunk_id)

        for ref in refs:
            # M1: Document-scope deduplication - skip if same target already seen
            normalized_target = ref.target_hint.lower().strip()
            if document_seen_targets is not None:
                if normalized_target in document_seen_targets:
                    logger.debug(
                        "reference_deduplicated_across_chunks",
                        source_chunk_id=chunk_id,
                        target_hint=ref.target_hint,
                        reason="already_referenced_in_document",
                    )
                    continue
                document_seen_targets.add(normalized_target)

            # Attempt local resolution from known titles
            target_doc_id = None

            # For hyperlinks, try to match by filename
            if ref.reference_type == "hyperlink":
                # Convert filename to possible title
                possible_title = normalize_filename_to_title(ref.target_hint)
                normalized_title = possible_title.lower()

                if normalized_title in known_doc_titles:
                    target_doc_id = known_doc_titles[normalized_title]

            # For text-based references, try direct title match
            if target_doc_id is None:
                normalized_hint = ref.target_hint.lower()
                if normalized_hint in known_doc_titles:
                    target_doc_id = known_doc_titles[normalized_hint]

            # Track resolution stats
            if target_doc_id:
                resolved_count += 1
            else:
                unresolved_count += 1
                # Log unresolved for debugging
                logger.debug(
                    "reference_unresolved_locally",
                    source_chunk_id=chunk_id,
                    target_hint=ref.target_hint,
                    reference_type=ref.reference_type,
                )

            # Create edge (even for unresolved - Neo4j will resolve later)
            edge = create_reference_edge(
                source_chunk_id=chunk_id,
                target_doc_id=target_doc_id,
                target_hint=ref.target_hint,
                reference_type=ref.reference_type,
                reference_text=ref.reference_text,
                confidence=ref.confidence,
            )
            reference_edges.append(edge)

    if reference_edges:
        logger.info(
            "chunk_references_extracted",
            total_edges=len(reference_edges),
            resolved=resolved_count,
            unresolved=unresolved_count,
            chunks_processed=len(chunks),
        )

    return reference_edges, resolved_count, unresolved_count


def build_title_lookup(documents: List[Dict]) -> Dict[str, str]:
    """
    Build a normalized title -> doc_id lookup from documents.

    Args:
        documents: List of document dicts with 'doc_id' and 'title' fields

    Returns:
        Dict mapping normalized (lowercase) titles to doc_ids
    """
    lookup = {}
    for doc in documents:
        doc_id = doc.get("doc_id") or doc.get("document_id")
        title = doc.get("title", "")
        if doc_id and title:
            lookup[title.lower()] = doc_id
    return lookup
