"""
Entity label configuration for GLiNER zero-shot NER.

GLiNER supports zero-shot entity extraction using descriptive labels.
Including examples in parentheses (e.g. ...) helps the model understand
what kind of entities to extract.

This module provides:
- Default domain-specific labels for WEKA documentation
- Utility functions to retrieve labels from config or defaults
"""

from typing import List

from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)

# Default WEKA domain-specific entity labels (v2 - refined for retrieval)
# These are used if config.ner.labels is empty
#
# Design rationale:
# - 10 focused entity types optimized for retrieval value
# - COMMAND: Direct CLI lookup for "how do I..." queries
# - PARAMETER: Configuration queries for flags and settings
# - COMPONENT: Architecture questions (backend, frontend, drives)
# - PROTOCOL: Protocol-specific filtering (NFS, SMB, S3)
# - CLOUD_PROVIDER: Deployment context (AWS, Azure, GCP)
# - STORAGE_CONCEPT: Conceptual queries (tiering, snapshots)
# - VERSION: Version-specific filtering (4.4, 4.4.x)
# - PROCEDURE_STEP: Procedural extraction for how-to guides
# - ERROR: Troubleshooting (error codes, failure messages)
# - CAPACITY_METRIC: Sizing/performance queries (GB, TB, IOPS)
DEFAULT_LABELS: List[str] = [
    "COMMAND (e.g. weka fs, weka nfs permission add, mount)",
    "PARAMETER (e.g. --json, num_cores, memory_mb, stripe-width)",
    "COMPONENT (e.g. backend server, frontend process, drive process)",
    "PROTOCOL (e.g. NFS, SMB, S3, POSIX)",
    "CLOUD_PROVIDER (e.g. AWS, Azure, GCP, OCI)",
    "STORAGE_CONCEPT (e.g. tiering, snapshot, object store, SSD capacity)",
    "VERSION (e.g. 4.4, 4.4.x, v4.3)",
    "PROCEDURE_STEP (e.g. Select Save, Run the command, Click Apply)",
    "ERROR (e.g. error code 10054, Connection refused, timeout)",
    "CAPACITY_METRIC (e.g. GB, TB, IOPS, latency, throughput)",
]

# Entities to exclude from enrichment (too common, pollutes queries)
# These are filtered out AFTER extraction to avoid noisy embeddings
ENTITY_EXCLUSIONS: set[str] = {
    "weka",
    "WEKA",
    "Weka",
    "WekaFS",
    "wekafs",
}


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
        "weka_software_component (e.g. backend, frontend)" -> "weka_software_component"

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

    Some terms are so common in the domain (e.g., "WEKA") that including
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
