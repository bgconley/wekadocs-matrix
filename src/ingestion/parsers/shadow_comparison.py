"""
Shadow mode comparison utilities for parser migration.

This module provides structured comparison between legacy and markdown-it-py
parser outputs. Used during Phase 4 migration to validate the new parser
produces equivalent (or better) results before full rollout.

Key components:
    - ParserComparisonResult: Structured comparison metrics
    - ShadowModeError: Raised when fail_on_mismatch is enabled and parsers differ
    - compare_parser_results(): Detailed comparison with metrics

Usage:
    from src.ingestion.parsers.shadow_comparison import (
        compare_parser_results,
        ShadowModeError,
    )

    result = compare_parser_results(source_uri, legacy_result, mit_result)
    if result.has_differences and fail_on_mismatch:
        raise ShadowModeError(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ShadowModeError(Exception):
    """
    Raised when shadow mode comparison fails and fail_on_mismatch is enabled.

    Contains the comparison result for inspection.
    """

    def __init__(self, comparison_result: "ParserComparisonResult"):
        self.comparison_result = comparison_result
        super().__init__(f"Parser mismatch detected: {comparison_result.summary()}")


@dataclass
class ParserComparisonResult:
    """
    Structured result of comparing legacy and markdown-it-py parser outputs.

    Attributes:
        source_uri: Document being compared
        has_differences: True if any differences were found
        section_count_legacy: Number of sections from legacy parser
        section_count_mit: Number of sections from markdown-it-py parser
        title_differences: List of (index, legacy_title, mit_title) tuples
        document_title_differs: True if document titles differ
        legacy_doc_title: Document title from legacy parser
        mit_doc_title: Document title from markdown-it-py parser
        new_metadata_fields: List of new fields available in markdown-it-py
        sample_metadata: Sample of new metadata from first section
    """

    source_uri: str
    has_differences: bool = False

    # Section comparison
    section_count_legacy: int = 0
    section_count_mit: int = 0
    title_differences: List[tuple] = field(default_factory=list)

    # Document comparison
    document_title_differs: bool = False
    legacy_doc_title: Optional[str] = None
    mit_doc_title: Optional[str] = None

    # New metadata tracking
    new_metadata_fields: List[str] = field(default_factory=list)
    sample_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def section_count_differs(self) -> bool:
        """True if section counts don't match."""
        return self.section_count_legacy != self.section_count_mit

    @property
    def section_count_delta(self) -> int:
        """Difference in section counts (mit - legacy)."""
        return self.section_count_mit - self.section_count_legacy

    def summary(self) -> str:
        """Generate a human-readable summary of differences."""
        parts = []

        if self.section_count_differs:
            parts.append(
                f"section_count: {self.section_count_legacy} → {self.section_count_mit} "
                f"(delta: {self.section_count_delta:+d})"
            )

        if self.title_differences:
            parts.append(f"title_diffs: {len(self.title_differences)}")

        if self.document_title_differs:
            parts.append(
                f"doc_title: '{self.legacy_doc_title}' → '{self.mit_doc_title}'"
            )

        if not parts:
            return "no differences"

        return "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "source_uri": self.source_uri,
            "has_differences": self.has_differences,
            "section_count_legacy": self.section_count_legacy,
            "section_count_mit": self.section_count_mit,
            "section_count_delta": self.section_count_delta,
            "title_differences_count": len(self.title_differences),
            "document_title_differs": self.document_title_differs,
            "new_metadata_fields": self.new_metadata_fields,
        }


def compare_parser_results(
    source_uri: str,
    legacy_result: Dict[str, Any],
    mit_result: Dict[str, Any],
) -> ParserComparisonResult:
    """
    Compare parser outputs and generate structured comparison result.

    This function performs a detailed comparison between legacy and markdown-it-py
    parser outputs, tracking:
    - Section count differences
    - Section title differences (position-wise)
    - Document title differences
    - New metadata fields available in markdown-it-py

    Args:
        source_uri: Document URI for context
        legacy_result: Output from legacy parser
        mit_result: Output from markdown-it-py parser

    Returns:
        ParserComparisonResult with all comparison metrics
    """
    result = ParserComparisonResult(source_uri=source_uri)

    legacy_sections = legacy_result.get("Sections", [])
    mit_sections = mit_result.get("Sections", [])

    # Section count comparison
    result.section_count_legacy = len(legacy_sections)
    result.section_count_mit = len(mit_sections)

    if result.section_count_differs:
        result.has_differences = True

    # Section title comparison (position-wise)
    max_sections = max(len(legacy_sections), len(mit_sections))
    for i in range(max_sections):
        legacy_title = (
            legacy_sections[i].get("title", "") if i < len(legacy_sections) else None
        )
        mit_title = mit_sections[i].get("title", "") if i < len(mit_sections) else None

        if legacy_title != mit_title:
            result.title_differences.append((i, legacy_title, mit_title))
            result.has_differences = True

    # Document title comparison
    legacy_doc = legacy_result.get("Document", {})
    mit_doc = mit_result.get("Document", {})

    result.legacy_doc_title = legacy_doc.get("title")
    result.mit_doc_title = mit_doc.get("title")

    if result.legacy_doc_title != result.mit_doc_title:
        result.document_title_differs = True
        result.has_differences = True

    # Track new metadata fields available in markdown-it-py
    if mit_sections:
        sample = mit_sections[0]
        new_fields = [
            "line_start",
            "line_end",
            "parent_path",
            "block_types",
            "code_ratio",
            "has_code",
            "has_table",
        ]
        result.new_metadata_fields = [
            f for f in new_fields if sample.get(f) is not None
        ]

        # Capture sample metadata for debugging
        result.sample_metadata = {f: sample.get(f) for f in result.new_metadata_fields}

    return result


def log_comparison_result(
    result: ParserComparisonResult,
    log_level: str = "info",
) -> None:
    """
    Log comparison result with appropriate detail level.

    Args:
        result: Comparison result to log
        log_level: One of "debug", "info", "warning"
    """
    log_fn = getattr(logger, log_level, logger.info)

    if result.has_differences:
        log_fn(
            "Shadow mode: parser differences detected",
            source_uri=result.source_uri,
            summary=result.summary(),
            section_count_legacy=result.section_count_legacy,
            section_count_mit=result.section_count_mit,
            title_diff_count=len(result.title_differences),
            # First 3 title diffs for context
            title_diffs_sample=result.title_differences[:3],
        )
    else:
        logger.debug(
            "Shadow mode: parsers match",
            source_uri=result.source_uri,
            section_count=result.section_count_mit,
            new_fields=result.new_metadata_fields,
        )

    # Always log new metadata availability at debug level
    if result.new_metadata_fields:
        logger.debug(
            "Shadow mode: new metadata fields available",
            source_uri=result.source_uri,
            fields=result.new_metadata_fields,
            sample=result.sample_metadata,
        )
