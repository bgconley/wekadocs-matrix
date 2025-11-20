"""
Embedding field canonicalization helpers for GraphRAG v2.1.

This module provides utilities to ensure consistent use of canonical embedding
fields across the codebase, mapping from legacy `embedding_model` to the
canonical `embedding_version` used in persisted data.

Key invariants enforced:
- embedding_version = "jina-embeddings-v3" (not embedding_model)
- embedding_dimensions = 1024
- embedding_provider = "jina-ai"
- embedding_timestamp present (ISO-8601)
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical values for Jina v3 embeddings
CANONICAL_VERSION = "jina-embeddings-v3"
CANONICAL_PROVIDER = "jina-ai"
CANONICAL_DIMENSIONS = 1024
CANONICAL_TASK = "retrieval.passage"


def canonicalize_embedding_metadata(
    embedding_model: str,
    dimensions: int,
    provider: Optional[str] = None,
    task: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    profile: Optional[str] = None,
    namespace_mode: Optional[str] = None,
    namespace_suffix: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create canonical embedding metadata for persistence.

    Maps from code-level `embedding_model` to persisted `embedding_version`.

    Args:
        embedding_model: The model identifier (e.g., "jina-embeddings-v3")
        dimensions: The embedding dimensions
        provider: Optional provider override
        task: Optional task override
        timestamp: Optional timestamp override

    Returns:
        Dict with canonical fields for persistence
    """
    metadata = {
        "embedding_version": embedding_model,  # Map model -> version
        "embedding_dimensions": dimensions,
        "embedding_provider": provider or CANONICAL_PROVIDER,
        "embedding_task": task or CANONICAL_TASK,
        "embedding_timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
    }
    if profile:
        metadata["embedding_profile"] = profile
    if namespace_mode:
        metadata["namespace_mode"] = namespace_mode
    if namespace_suffix:
        metadata["namespace_suffix"] = namespace_suffix
    if collection_name:
        metadata["collection_name"] = collection_name
    return metadata


def read_embedding_version_with_fallback(
    props: Dict[str, Any], log_deprecation: bool = True
) -> Optional[str]:
    """
    Read embedding version with fallback to legacy field.

    This is a transitional shim that should be removed once all data
    is migrated to use embedding_version exclusively.

    Args:
        props: Properties dict from Neo4j node or Qdrant payload
        log_deprecation: Whether to log deprecation warning

    Returns:
        The embedding version string, or None if not found
    """
    # Check canonical field first
    if "embedding_version" in props and props["embedding_version"]:
        return props["embedding_version"]

    # Fallback to legacy field
    legacy = props.get("embedding_model")
    if legacy:
        if log_deprecation:
            logger.warning(
                "DEPRECATION: Found legacy 'embedding_model' field. "
                "Use 'embedding_version' instead. Node/point will be updated.",
                node_id=props.get("id", "unknown"),
            )
        return legacy

    return None


def validate_embedding_metadata(
    metadata: Dict[str, Any],
    expected_dimensions: Optional[int] = None,
    expected_provider: Optional[str] = None,
    expected_version: Optional[str] = None,
) -> bool:
    """
    Validate that embedding metadata meets canonical requirements.

    Args:
        metadata: Metadata dict to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "embedding_version",
        "embedding_dimensions",
        "embedding_provider",
        "embedding_timestamp",
    ]

    # Check all required fields present
    for field in required_fields:
        if field not in metadata or metadata[field] is None:
            logger.error(f"Missing required embedding field: {field}")
            return False

    dimensions = metadata["embedding_dimensions"]
    provider = metadata["embedding_provider"]
    version = metadata["embedding_version"]

    if expected_dimensions and dimensions != expected_dimensions:
        logger.error(
            "Invalid embedding dimensions: expected %s, got %s",
            expected_dimensions,
            dimensions,
        )
        return False

    if expected_provider and provider != expected_provider:
        logger.warning(
            "Embedding provider %s differs from expected %s",
            provider,
            expected_provider,
        )

    if expected_version and version != expected_version:
        logger.warning(
            "Embedding version %s differs from expected %s",
            version,
            expected_version,
        )

    return True


def ensure_no_embedding_model_in_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure payload does not contain legacy embedding_model field.

    This is used as a guardrail when writing to stores.

    Args:
        payload: Payload dict to clean

    Returns:
        Cleaned payload without embedding_model
    """
    if "embedding_model" in payload:
        logger.warning(
            "Removing legacy 'embedding_model' from payload. "
            "Only 'embedding_version' should be persisted.",
            node_id=payload.get("id", "unknown"),
        )
        payload = {k: v for k, v in payload.items() if k != "embedding_model"}

    return payload


def create_write_payload(
    node_id: str,
    embedding_version: str,
    dimensions: int,
    provider: str,
    task: str,
    **extra_fields,
) -> Dict[str, Any]:
    """
    Create a clean payload for writing to Neo4j/Qdrant.

    Ensures only canonical fields are used.

    Args:
        node_id: The node/point ID
        embedding_version: The embedding version (e.g., "jina-embeddings-v3")
        dimensions: Embedding dimensions
        provider: Embedding provider
        task: Embedding task
        **extra_fields: Additional fields to include

    Returns:
        Clean payload dict for persistence
    """
    payload = {
        "id": node_id,
        "embedding_version": embedding_version,
        "embedding_dimensions": dimensions,
        "embedding_provider": provider,
        "embedding_task": task,
        "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Add extra fields but ensure no embedding_model
    for key, value in extra_fields.items():
        if key != "embedding_model":
            payload[key] = value

    return payload
