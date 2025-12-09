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

# Default WEKA domain-specific entity labels
# These are used if config.ner.labels is empty
DEFAULT_LABELS: List[str] = [
    "weka_software_component (e.g. backend, frontend, agent, client)",
    "operating_system (e.g. RHEL, Ubuntu, Rocky Linux)",
    "hardware_component (e.g. NVMe, NIC, GPU, switch)",
    "filesystem_object (e.g. inode, snapshot, file, directory)",
    "cloud_provider_or_service (e.g. AWS, S3, Azure, EC2)",
    "cli_command (e.g. weka fs, mount, systemctl)",
    "configuration_parameter (e.g. --net-apply, stripe-width)",
    "network_or_storage_protocol (e.g. NFS, SMB, S3, POSIX, TCP)",
    "error_message_or_code (e.g. 10054, Connection refused)",
    "performance_metric (e.g. IOPS, latency, throughput)",
    "file_system_path (e.g. /mnt/weka, /etc/fstab)",
]


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
        List of clean label names (e.g., ["weka_software_component", "operating_system", ...])
    """
    return [extract_label_name(label) for label in get_default_labels()]
