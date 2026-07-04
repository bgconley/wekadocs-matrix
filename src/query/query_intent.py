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
from typing import Dict, FrozenSet, List, Tuple

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
        "aos",
        "ahv",
        "prism",
        "prism central",
        "prism element",
        "nutanix central",
        "nci",
        "ncp",
        "nus",
        "files",
        "objects",
        "volumes",
        "ncm",
        "ndb",
        "nkp",
        "nc2",
        "nai",
        "flow",
        "move",
        "disaster recovery",
        "data services",
        "cloud native aos",
        "data lens",
        "microsegmentation",
        "lifecycle",
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
        "node count",
        "cluster requirements",
        "configuration maximums",
        "license tier",
        "licensing",
        "gpu",
        "usable tib",
    }
)

# ── Anchor / modifier split ───────────────────────────────────────────
# Anchors name the specific Nutanix subsystem or resource.
# Modifiers describe what the user wants to know about them.

SUBSYSTEM_ANCHORS: FrozenSet[str] = frozenset(
    {
        "metadata",
        "inode",
        "aos",
        "ahv",
        "prism",
        "prism central",
        "nutanix central",
        "nci",
        "ncp",
        "nus",
        "files",
        "objects",
        "volumes",
        "ncm",
        "ndb",
        "nkp",
        "nc2",
        "nai",
        "flow",
        "move",
        "disaster recovery",
        "tiering",
        "snapshots",
        "prefetch",
        "destage",
        "rebuild",
        "failure domain",
        "stripe width",
        "data placement",
        "hot spare",
        "protection scheme",
    }
)

SUBSYSTEM_MODIFIERS: FrozenSet[str] = frozenset(
    {
        "architecture",
        "architected",
        "managed",
        "internals",
        "backend",
        "limitations",
        "filesystem internals",
    }
)

SIZING_ANCHORS: FrozenSet[str] = frozenset(
    {
        "drives",
        "compute",
        "frontend",
        "containers",
        "drives0",
        "compute0",
        "frontend0",
        "ram",
        "cpu",
        "cores",
        "node count",
        "cluster requirements",
        "configuration maximums",
        "license tier",
        "licensing",
        "gpu",
        "usable tib",
    }
)

SIZING_MODIFIERS: FrozenSet[str] = frozenset(
    {
        "size",
        "sizing",
        "capacity",
        "memory",
        "resource allocation",
        "minimum requirements",
        "maximum capacity",
        "container composition",
    }
)

# ── CLI patterns (migrated from hybrid_retrieval.py:2593) ────────────

_CLI_PATTERNS = [
    re.compile(r"\b(?:ncli|acli|kubectl|nutanix)\s+\w+"),
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
    primary_anchors: Tuple[
        str, ...
    ] = ()  # Specific topic terms (metadata, drives, ...)
    generic_modifiers: Tuple[str, ...] = ()  # Broad terms (managed, sizing, ...)


def _partition_terms(
    matched: List[str],
    anchors: FrozenSet[str],
    modifiers: FrozenSet[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split matched terms into (primary_anchors, generic_modifiers)."""
    a: List[str] = []
    m: List[str] = []
    for t in matched:
        if t in anchors:
            a.append(t)
        elif t in modifiers:
            m.append(t)
        # Terms in neither set are ignored (they still count for classification)
    return tuple(a), tuple(m)


# Pre-compiled word-boundary patterns for term matching.
# Multi-word terms (e.g. "failure domain") use raw substring matching since
# they span word boundaries naturally.  Single-word terms use \b anchors to
# prevent "ram" matching inside "program".
_TERM_PATTERNS: Dict[str, re.Pattern] = {}


def _term_matches(term: str, text: str) -> bool:
    """Check whether *term* appears in *text* with word-boundary safety."""
    if " " in term:
        # Multi-word: plain substring is safe ("failure domain" won't appear
        # accidentally inside another word).
        return term in text
    pat = _TERM_PATTERNS.get(term)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(term) + r"\b")
        _TERM_PATTERNS[term] = pat
    return pat.search(text) is not None


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
        if _term_matches(term, q):
            matched_subsystem.append(term)

    matched_sizing: List[str] = []
    for term in sorted(SIZING_TERMS):  # sorted for deterministic output
        if _term_matches(term, q):
            matched_sizing.append(term)

    # ── Partition into anchors and modifiers ──────────────────
    sub_anchors, sub_modifiers = _partition_terms(
        matched_subsystem, SUBSYSTEM_ANCHORS, SUBSYSTEM_MODIFIERS
    )
    siz_anchors, siz_modifiers = _partition_terms(
        matched_sizing, SIZING_ANCHORS, SIZING_MODIFIERS
    )

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
        return QueryIntent(
            query_type="cli",
            primary_anchors=sub_anchors or siz_anchors,
            generic_modifiers=sub_modifiers or siz_modifiers,
            **common,
        )

    # 2. Config
    if any(pat.search(q) for pat in _CONFIG_PATTERNS):
        return QueryIntent(
            query_type="config",
            primary_anchors=sub_anchors or siz_anchors,
            generic_modifiers=sub_modifiers or siz_modifiers,
            **common,
        )

    # 3. Subsystem architecture (NEW — regardless of cloud cues)
    if matched_subsystem:
        return QueryIntent(
            query_type="subsystem_architecture",
            precision_mode=True,
            primary_anchors=sub_anchors,
            generic_modifiers=sub_modifiers,
            **common,
        )

    # 4. Resource sizing (NEW — regardless of cloud cues)
    if matched_sizing:
        return QueryIntent(
            query_type="resource_sizing",
            precision_mode=True,
            primary_anchors=siz_anchors,
            generic_modifiers=siz_modifiers,
            **common,
        )

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
