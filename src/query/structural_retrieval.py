"""
Structural retrieval enhancements for query-type adaptive multi-vector search.

Phase 5 of markdown-it-py integration: Leverage structural metadata for improved retrieval.

This module provides:
1. Query-type adaptive RRF field weights
2. Structural filter construction for Qdrant
3. Post-retrieval structural boosting

Design principles:
1. Pure functions with no side effects for testability
2. Configuration-driven via StructuralRetrievalConfig
3. Minimal coupling to hybrid_retrieval.py - import and call
4. Defensive handling of missing metadata

Usage:
    from src.query.structural_retrieval import (
        get_query_type_rrf_weights,
        build_structural_filter,
        apply_structural_boost,
    )

    # Get adaptive field weights for a query type
    weights = get_query_type_rrf_weights("cli", config)

    # Build Qdrant filter for structural constraints
    filter_obj = build_structural_filter("cli", config)

    # Apply post-retrieval boosts
    boosted = apply_structural_boost(results, "cli", config)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# === Default Query Type Configurations ===
# These define how different query types should weight fields and filter results

DEFAULT_QUERY_TYPE_RRF_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Conceptual queries: boost semantic understanding, hierarchy context
    "conceptual": {
        "content": 2.5,  # Strong semantic signal
        "title": 1.0,  # Section titles less important
        "text-sparse": 0.3,  # Lexical less important
        "doc_title-sparse": 0.8,
        "title-sparse": 0.8,
        "entity-sparse": 0.6,  # Concepts may not be named entities
    },
    # CLI/command queries: boost entity matching, code-containing chunks
    "cli": {
        "content": 1.5,  # Semantic still matters
        "title": 1.5,  # CLI often under clear headings
        "text-sparse": 0.8,  # Commands are lexical
        "doc_title-sparse": 0.8,
        "title-sparse": 1.0,
        "entity-sparse": 1.8,  # Commands are entities
    },
    # Configuration queries: balance semantic and lexical
    "config": {
        "content": 1.8,
        "title": 1.2,
        "text-sparse": 0.7,
        "doc_title-sparse": 0.8,
        "title-sparse": 1.0,
        "entity-sparse": 1.5,  # Config params are entities
    },
    # Procedural/how-to queries: boost step-by-step content
    "procedural": {
        "content": 2.0,
        "title": 1.2,  # Procedure titles important
        "text-sparse": 0.5,
        "doc_title-sparse": 0.8,
        "title-sparse": 1.0,
        "entity-sparse": 0.8,
    },
    # Troubleshooting: boost error entities and diagnostic content
    "troubleshooting": {
        "content": 2.0,
        "title": 1.0,
        "text-sparse": 0.6,
        "doc_title-sparse": 0.8,
        "title-sparse": 0.8,
        "entity-sparse": 1.5,  # Error codes are entities
    },
    # Reference/lookup: balanced, slightly boost titles
    "reference": {
        "content": 1.8,
        "title": 1.5,
        "text-sparse": 0.6,
        "doc_title-sparse": 1.0,
        "title-sparse": 1.2,
        "entity-sparse": 1.0,
    },
}

# Structural boost factors by query type
DEFAULT_STRUCTURAL_BOOSTS: Dict[str, Dict[str, float]] = {
    # CLI queries: boost chunks with code
    "cli": {
        "has_code_boost": 1.20,  # 20% boost for code-containing chunks
        "has_table_boost": 1.0,  # No boost for tables
        "deep_nesting_penalty": 0.95,  # 5% penalty for depth > 2
        "max_depth_for_penalty": 2,
    },
    # Config queries: boost code (config files are code)
    "config": {
        "has_code_boost": 1.15,
        "has_table_boost": 1.10,  # Config tables are useful
        "deep_nesting_penalty": 0.95,
        "max_depth_for_penalty": 2,
    },
    # Reference queries: boost tables (specs, parameters)
    "reference": {
        "has_code_boost": 1.05,
        "has_table_boost": 1.20,  # 20% boost for table chunks
        "deep_nesting_penalty": 0.90,  # Stronger penalty for nested content
        "max_depth_for_penalty": 1,
    },
    # Procedural: neutral, steps can be anywhere
    "procedural": {
        "has_code_boost": 1.10,
        "has_table_boost": 1.0,
        "deep_nesting_penalty": 1.0,  # No penalty
        "max_depth_for_penalty": 99,
    },
    # Conceptual: prefer shallow content (overviews)
    "conceptual": {
        "has_code_boost": 1.0,  # No boost
        "has_table_boost": 1.0,
        "deep_nesting_penalty": 0.85,  # 15% penalty for deep content
        "max_depth_for_penalty": 2,
    },
    # Troubleshooting: boost code (error messages, logs)
    "troubleshooting": {
        "has_code_boost": 1.15,
        "has_table_boost": 1.05,
        "deep_nesting_penalty": 0.95,
        "max_depth_for_penalty": 3,
    },
}


@dataclass
class StructuralRetrievalConfig:
    """
    Configuration for structural retrieval enhancements.

    Can be populated from HybridSearchConfig or standalone for testing.

    Attributes:
        query_type_rrf_weights: Per-field RRF weights for each query type
        structural_boosts: Post-retrieval boost factors by query type
        enabled: Master switch for structural enhancements
        filter_by_block_type: Whether to apply block type filters
        boost_by_structure: Whether to apply structural boosts
    """

    query_type_rrf_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: DEFAULT_QUERY_TYPE_RRF_WEIGHTS.copy()
    )
    structural_boosts: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: DEFAULT_STRUCTURAL_BOOSTS.copy()
    )
    enabled: bool = True
    filter_by_block_type: bool = False  # Strict filtering (off by default)
    boost_by_structure: bool = True  # Soft boosting (on by default)
    default_query_type: str = "conceptual"


def get_query_type_rrf_weights(
    query_type: str,
    config: Optional[StructuralRetrievalConfig] = None,
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Get RRF field weights adapted for a specific query type.

    Merges query-type specific weights with base weights, with query-type
    weights taking precedence for fields they specify.

    Args:
        query_type: Type of query (cli, config, procedural, etc.)
        config: Optional structural config, uses defaults if None
        base_weights: Base field weights to merge with (from config.rrf_field_weights)

    Returns:
        Dict mapping field names to RRF weights

    Example:
        >>> weights = get_query_type_rrf_weights("cli")
        >>> weights["entity-sparse"]
        1.8
    """
    if config is None:
        config = StructuralRetrievalConfig()

    if not config.enabled:
        return base_weights or {}

    # Normalize query type
    qtype = query_type.lower().strip() if query_type else config.default_query_type

    # Get query-type specific weights
    type_weights = config.query_type_rrf_weights.get(qtype, {})

    # If no specific weights for this type, fall back to default or base
    if not type_weights:
        type_weights = config.query_type_rrf_weights.get(config.default_query_type, {})

    # Start with base weights if provided
    result = dict(base_weights) if base_weights else {}

    # Override with query-type specific weights
    result.update(type_weights)

    return result


def build_structural_filter(
    query_type: str,
    config: Optional[StructuralRetrievalConfig] = None,
    *,
    require_code: Optional[bool] = None,
    require_table: Optional[bool] = None,
    max_depth: Optional[int] = None,
    block_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build Qdrant filter conditions based on query type and structural requirements.

    Returns a filter dict compatible with Qdrant's Filter model.
    Import and convert to Filter object in the calling code.

    Args:
        query_type: Type of query for automatic filter selection
        config: Structural config, uses defaults if None
        require_code: Explicit override to require has_code=True
        require_table: Explicit override to require has_table=True
        max_depth: Explicit override for max parent_path_depth
        block_types: Explicit list of allowed block_type values

    Returns:
        Dict with "must" conditions list, or None if no filters needed

    Example:
        >>> filter_dict = build_structural_filter("cli", require_code=True)
        >>> filter_dict
        {'must': [{'key': 'has_code', 'match': {'value': True}}]}
    """
    if config is None:
        config = StructuralRetrievalConfig()

    if not config.enabled or not config.filter_by_block_type:
        # Only apply explicit overrides if filtering is disabled
        if (
            require_code is None
            and require_table is None
            and max_depth is None
            and not block_types
        ):
            return None

    conditions = []

    # Explicit has_code filter
    if require_code is True:
        conditions.append({"key": "has_code", "match": {"value": True}})

    # Explicit has_table filter
    if require_table is True:
        conditions.append({"key": "has_table", "match": {"value": True}})

    # Max depth filter
    if max_depth is not None:
        conditions.append({"key": "parent_path_depth", "range": {"lte": max_depth}})

    # Block type filter (explicit list)
    if block_types:
        # Qdrant uses "any" match for keyword arrays
        conditions.append({"key": "block_type", "match": {"any": block_types}})

    # Query-type automatic filters (only if filter_by_block_type enabled)
    if config.filter_by_block_type:
        qtype = query_type.lower().strip() if query_type else config.default_query_type

        if qtype == "cli" and require_code is None:
            # CLI queries strongly prefer code chunks
            conditions.append({"key": "has_code", "match": {"value": True}})

        elif qtype == "reference" and require_table is None:
            # Reference queries prefer table chunks (soft - don't require)
            pass  # Use boosting instead of hard filter

    if not conditions:
        return None

    return {"must": conditions}


def apply_structural_boost(
    results: List[Dict[str, Any]],
    query_type: str,
    config: Optional[StructuralRetrievalConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Apply post-retrieval score adjustments based on structural metadata.

    Modifies scores in place and re-sorts by adjusted score.
    This is a "soft" filtering approach - boosts good matches rather
    than hard-filtering potentially relevant results.

    Args:
        results: List of result dicts with 'payload' and 'score' keys
        query_type: Type of query for boost selection
        config: Structural config, uses defaults if None

    Returns:
        Same list, sorted by adjusted scores (highest first)

    Example:
        >>> results = [{'score': 0.8, 'payload': {'has_code': True}}]
        >>> boosted = apply_structural_boost(results, "cli")
        >>> boosted[0]['score']  # Boosted from 0.8
        0.96
    """
    if config is None:
        config = StructuralRetrievalConfig()

    if not config.enabled or not config.boost_by_structure:
        return results

    # Normalize query type
    qtype = query_type.lower().strip() if query_type else config.default_query_type

    # Get boost factors for this query type
    boosts = config.structural_boosts.get(qtype, {})
    if not boosts:
        boosts = config.structural_boosts.get(config.default_query_type, {})

    has_code_boost = boosts.get("has_code_boost", 1.0)
    has_table_boost = boosts.get("has_table_boost", 1.0)
    deep_nesting_penalty = boosts.get("deep_nesting_penalty", 1.0)
    max_depth_for_penalty = boosts.get("max_depth_for_penalty", 99)

    for result in results:
        score = result.get("score", 0.0)
        payload = result.get("payload", {})

        # Apply has_code boost
        if payload.get("has_code", False) and has_code_boost != 1.0:
            score *= has_code_boost

        # Apply has_table boost
        if payload.get("has_table", False) and has_table_boost != 1.0:
            score *= has_table_boost

        # Apply deep nesting penalty
        depth = payload.get("parent_path_depth", 0)
        if depth > max_depth_for_penalty and deep_nesting_penalty != 1.0:
            score *= deep_nesting_penalty

        result["score"] = score

    # Re-sort by adjusted score
    return sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)


def get_structural_boost_info(
    query_type: str,
    config: Optional[StructuralRetrievalConfig] = None,
) -> Dict[str, Any]:
    """
    Get human-readable info about structural boosts for a query type.

    Useful for debugging and explaining retrieval behavior.

    Args:
        query_type: Type of query
        config: Structural config

    Returns:
        Dict with boost factors and explanations
    """
    if config is None:
        config = StructuralRetrievalConfig()

    qtype = query_type.lower().strip() if query_type else config.default_query_type

    boosts = config.structural_boosts.get(qtype, {})
    weights = config.query_type_rrf_weights.get(qtype, {})

    return {
        "query_type": qtype,
        "enabled": config.enabled,
        "rrf_field_weights": weights,
        "has_code_boost": boosts.get("has_code_boost", 1.0),
        "has_table_boost": boosts.get("has_table_boost", 1.0),
        "deep_nesting_penalty": boosts.get("deep_nesting_penalty", 1.0),
        "max_depth_for_penalty": boosts.get("max_depth_for_penalty", 99),
        "filter_by_block_type": config.filter_by_block_type,
        "boost_by_structure": config.boost_by_structure,
    }
