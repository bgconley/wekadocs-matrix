"""
Query intent classification for precision retrieval.

Single source of truth for query type classification. Called by
HybridRetriever._classify_query_type() and directly by retrieve() for
the full QueryIntent object.

Design: Pure function, no IO, fully testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

# ── Term sets ──────────────────────────────────────────────────────────

CLOUD_CUES: FrozenSet[str] = frozenset(
    {
        "aws",
        "azure",
        "gcp",
        "slurm",
        "cyclecloud",
        "terraform",
        "cloudformation",
        "parallelcluster",
        "sagemaker",
        "eks",
        "aks",
        "gke",
    }
)

SUBSYSTEM_TERMS: FrozenSet[str] = frozenset(
    {
        "metadata",
        "inode",
        "internals",
        "filesystem internals",
        "tiering",
        "snapshots",
        "limitations",
        "architecture",
        "architected",
        "managed",
        "backend",
        "destage",
        "prefetch",
        "data striping",
        "protection scheme",
        "rebuild",
        "failure domain",
        "stripe width",
        "data placement",
        "hot spare",
    }
)

SIZING_TERMS: FrozenSet[str] = frozenset(
    {
        "size",
        "sizing",
        "capacity",
        "drives",
        "compute",
        "frontend",
        "containers",
        "memory",
        "cpu",
        "cores",
        "ram",
        "drives0",
        "compute0",
        "frontend0",
        "container composition",
        "resource allocation",
        "minimum requirements",
        "maximum capacity",
    }
)

# ── CLI patterns (migrated from hybrid_retrieval.py:2593) ────────────

_CLI_PATTERNS = [
    re.compile(r"\bweka\s+\w+"),
    re.compile(r"--[a-z][\w-]+"),
    re.compile(r"\s-[a-z]\b"),
    re.compile(r"\bcli\b"),
    re.compile(r"\bcommand\b"),
    re.compile(r"\brun\b.*\bcommand"),
]

# ── Config patterns (migrated from hybrid_retrieval.py:2607) ─────────

_CONFIG_PATTERNS = [
    re.compile(r"\w+\s*=\s*\w+"),
    re.compile(r"\.ya?ml\b"),
    re.compile(r"\.json\b"),
    re.compile(r"\.conf\b"),
    re.compile(r"\bconfig(?:ure|uration)?\b"),
    re.compile(r"\bsetting\b"),
    re.compile(r"\bparameter\b"),
    re.compile(r"\benvironment\s+variable"),
]


@dataclass(frozen=True)
class QueryIntent:
    """Immutable classification result for a user query."""

    query_type: str  # cli, config, subsystem_architecture,
    # resource_sizing, procedural, troubleshooting,
    # reference, conceptual
    has_cloud_cues: bool = False  # Orthogonal flag — does NOT affect query_type
    subsystem_terms: Tuple[str, ...] = ()  # matched subsystem terms
    sizing_terms: Tuple[str, ...] = ()  # matched sizing terms
    precision_mode: bool = False  # True for subsystem_architecture, resource_sizing


def classify_query_intent(query: str) -> QueryIntent:
    """
    Classify a query into a QueryIntent.

    Priority chain (first match wins):
      1. cli (>= 2 regex signals)
      2. config (any config regex match)
      3. subsystem_architecture (>= 1 subsystem term) -- NEW
      4. resource_sizing (>= 1 sizing term) -- NEW
      5. procedural
      6. troubleshooting
      7. reference
      8. conceptual (default)

    CRITICAL: has_cloud_cues is orthogonal to query_type. A query like
    "how does metadata work on Azure" classifies as subsystem_architecture
    with has_cloud_cues=True. Cloud cues do NOT disqualify subsystem/sizing
    classification.
    """
    q = (query or "").lower()
    words = set(q.split())

    # ── Detect cloud cues (independent of type) ──────────────
    cloud_matches = CLOUD_CUES & words
    has_cloud = bool(cloud_matches)

    # ── Detect subsystem and sizing terms ─────────────────────
    matched_subsystem: List[str] = []
    for term in sorted(SUBSYSTEM_TERMS):  # sorted for deterministic output
        if term in q:
            matched_subsystem.append(term)

    matched_sizing: List[str] = []
    for term in sorted(SIZING_TERMS):  # sorted for deterministic output
        if term in q:
            matched_sizing.append(term)

    # ── Build common kwargs ───────────────────────────────────
    common = dict(
        has_cloud_cues=has_cloud,
        subsystem_terms=tuple(matched_subsystem),
        sizing_terms=tuple(matched_sizing),
    )

    # ── Priority chain ────────────────────────────────────────

    # 1. CLI (requires >= 2 signals)
    cli_signals = sum(1 for pat in _CLI_PATTERNS if pat.search(q))
    if cli_signals >= 2:
        return QueryIntent(query_type="cli", **common)

    # 2. Config
    if any(pat.search(q) for pat in _CONFIG_PATTERNS):
        return QueryIntent(query_type="config", **common)

    # 3. Subsystem architecture (NEW — regardless of cloud cues)
    if matched_subsystem:
        return QueryIntent(
            query_type="subsystem_architecture", precision_mode=True, **common
        )

    # 4. Resource sizing (NEW — regardless of cloud cues)
    if matched_sizing:
        return QueryIntent(query_type="resource_sizing", precision_mode=True, **common)

    # 5. Procedural
    if "how to" in q or "steps to" in q or "configure" in q:
        return QueryIntent(query_type="procedural", **common)

    # 6. Troubleshooting
    if "error" in q or "failed" in q or "not working" in q:
        return QueryIntent(query_type="troubleshooting", **common)

    # 7. Reference
    if "what is" in q or "definition" in q or "meaning" in q:
        return QueryIntent(query_type="reference", **common)

    # 8. Default: conceptual
    return QueryIntent(query_type="conceptual", **common)
