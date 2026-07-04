# =============================================================================
# @status: ACTIVE
# @called-by: disambiguation.py (eager top-level import)
# =============================================================================
"""
Entity label configuration for GLiNER zero-shot NER.

GLiNER supports zero-shot entity extraction using descriptive labels.
Including examples in parentheses (e.g. ...) helps the model understand
what kind of entities to extract.

This module provides:
- Default domain-specific labels for Nutanix documentation
- Utility functions to retrieve labels from config or defaults
"""

import re
from typing import Dict, List

from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)

# Default Nutanix domain-specific entity labels (v2 - refined for retrieval)
# These are used if config.ner.labels is empty
#
# Design rationale:
# - Product/component/platform labels preserve Nutanix-specific semantics.
# - COMMAND/API labels support operational and developer-doc lookup.
# - Storage, deployment, error, metric, and procedure labels improve filters.
DEFAULT_LABELS: List[str] = [
    "PRODUCT (e.g. Nutanix Cloud Platform, NCI, NCM, NUS, NDB, NKP, NC2, NAI)",
    "COMPONENT (e.g. AOS, AHV, Prism Central, Files, Objects, Volumes, Flow)",
    "PLATFORM (e.g. on-premises, AWS, Azure, Google Cloud, OVHcloud, edge)",
    "VERSION (e.g. AOS 7.3, PC 2024.x, NAI 2.7)",
    "COMMAND (e.g. ncli, acli, kubectl, nutanix command-line operations)",
    "API (e.g. Prism v4 API, REST endpoint, category API)",
    "PROTOCOL (e.g. NFS, SMB, S3, iSCSI, CSI, COSI)",
    "STORAGE_CONCEPT (e.g. storage container, snapshot, replication, tiering)",
    "CLOUD_PROVIDER (e.g. AWS, Azure, Google Cloud, OVHcloud)",
    "DEPLOYMENT_MODEL (e.g. NCI, NC2, GC2, NCI-Edge, NCI-VDI)",
    "ERROR (e.g. alert, error code, failed task, health check failure)",
    "METRIC (e.g. IOPS, latency, throughput, CPU, memory, usable TiB)",
    "PROCEDURE_STEP (e.g. click Save, run the command, create a cluster)",
]

# Entities to exclude from enrichment (too common, pollutes queries)
# These are filtered out AFTER extraction to avoid noisy embeddings.
# Case-insensitive matching via is_excluded_entity() — lowercase entries suffice.
ENTITY_EXCLUSIONS: set[str] = {
    # Brand terms
    "nutanix",
    "Nutanix",
    # Generic domain vocabulary — high document frequency, low discriminative value
    "system",
    "server",
    "cluster",
    "node",
    "service",
    "data",
    "file",
    "process",
    "configuration",
    "management",
    "user",
    "group",
    "host",
    "client",
    "network",
    "storage",
    "volume",
    "drive",
    # Over-generic measurement terms
    "performance",
    "capacity",
    "size",
    "time",
    "number",
    # Over-generic procedure words
    "step",
    "click",
    "select",
    "run",
    "enter",
}

# Per-label confidence floors for retrieval signals (entity-sparse, _embedding_text).
# Entities below these thresholds are still recorded in entity_metadata (informational)
# but excluded from retrieval-critical paths (_mentions, _embedding_text).
# Rationale: GLiNER's confidence varies by label type. Abstract/ambiguous labels
# (STORAGE_CONCEPT, CAPACITY_METRIC) need higher thresholds to avoid noise.
RETRIEVAL_CONFIDENCE_FLOORS: Dict[str, float] = {
    "COMMAND": 0.55,
    "PARAMETER": 0.55,
    "PROTOCOL": 0.60,
    "CLOUD_PROVIDER": 0.65,
    "VERSION": 0.55,
    "ERROR": 0.55,
    "COMPONENT": 0.60,
    "PROCEDURE_STEP": 0.70,
    "STORAGE_CONCEPT": 0.70,
    "CAPACITY_METRIC": 0.70,
}
DEFAULT_RETRIEVAL_FLOOR: float = 0.60


def get_default_labels() -> List[str]:
    """
    Get entity labels from config, falling back to defaults.

    Returns:
        List of entity label strings for GLiNER extraction.
    """
    try:
        config = get_config()
        labels = config.ner.labels
        if labels:
            return labels
    except Exception as e:
        logger.warning(f"Failed to load NER labels from config: {e}")

    return DEFAULT_LABELS.copy()


def extract_label_name(label: str) -> str:
    """
    Extract the clean label name from a descriptive label.

    Example:
        "PRODUCT (e.g. NCI, NKP)" -> "PRODUCT"

    Args:
        label: Full label string with optional examples

    Returns:
        Clean label name without examples
    """
    # Split on " (" to remove examples
    return label.split(" (")[0].strip()


def get_label_names() -> List[str]:
    """
    Get clean label names without example descriptions.

    Returns:
        List of clean label names (e.g., ["COMMAND", "PARAMETER", ...])
    """
    return [extract_label_name(label) for label in get_default_labels()]


def is_excluded_entity(entity_text: str) -> bool:
    """
    Check if an entity should be excluded from enrichment.

    Some terms are so common in the domain (e.g., "Nutanix") that including
    them as entities would pollute embeddings and queries. This function
    checks against the exclusion list.

    Args:
        entity_text: The entity text to check

    Returns:
        True if entity should be excluded, False otherwise
    """
    # Normalize and check against exclusions
    normalized = entity_text.strip()
    return normalized in ENTITY_EXCLUSIONS or normalized.lower() in {
        e.lower() for e in ENTITY_EXCLUSIONS
    }


# Structural (regex-extracted) entity noise terms.
# These entities bypass GLiNER gates and can become high-DF hubs that dilute
# entity-sparse vectors and graph priors.
_STRUCTURAL_NOISE_TERMS: frozenset[str] = frozenset(
    {
        # CLI output formatting flags (belt-and-suspenders with extractor gating)
        "color",
        "filter-color",
        "output",
        "format",
        "filter",
        "sort",
        "profile",
        "raw-units",
        "verbose",
        "no-header",
        "json",
        "csv",
        "utf8",
        # Generic procedure/step boilerplate
        "procedure",
        "step",
        "note",
        "example",
        "overview",
        "prerequisites",
        "before you begin",
        "related topics",
        "optional",
        "required",
        # Generic computing terms that leak from structural extractors
        "new-name",
        "path",
        "port",
        "hostname",
        "timeout",
        "password",
        "username",
        "true",
        "false",
        "yes",
        "no",
        "none",
        "default",
    }
)


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for dedupe + filtering.

    Strips common Markdown artifacts (bold/italic markers, backticks),
    collapses whitespace, and trims edge punctuation without mutating
    meaningful internal characters like underscores/hyphens.
    """
    if not name:
        return ""

    s = str(name).strip()

    # Strip Markdown bold/italic wrappers.
    # Do bold/underline first to avoid leaving stray markers behind.
    # Examples: "**Procedure**" -> "Procedure", "__Note__" -> "Note".
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)

    # Remove inline code fences/backticks.
    s = s.replace("`", "")

    # Strip remaining emphasis markers at edges only (avoid nuking underscores
    # inside config keys like memory_mb).
    s = s.strip("*_")

    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()

    # Trim edge punctuation (keep hyphens/underscores inside names).
    s = s.strip(" \t\r\n\"'“”‘’()[]{}<>.,:;!?")

    return s


def is_excluded_structural_entity(name: str) -> bool:
    """Unified quality gate for structural (regex-extracted) entities."""
    normalized = normalize_entity_name(name)
    if not normalized:
        return True

    # Very short lowercase tokens are usually noise (e.g., "of", "to", "it").
    # Preserve short alnum tokens when they contain digits (e.g., "s3", "v4")
    # or are ALLCAPS (e.g., "IP"). Everything else <=2 chars is filtered.
    if len(normalized) <= 2:
        if normalized.isalnum() and any(ch.isdigit() for ch in normalized):
            pass
        elif normalized.isalnum() and normalized.upper() == normalized:
            pass
        else:
            return True

    if is_excluded_entity(normalized):
        return True

    lowered = normalized.lower()
    if lowered in _STRUCTURAL_NOISE_TERMS:
        return True

    return False
