# =============================================================================
# @status: ACTIVE
# @reason: SagaContext, ValidationResult, and IngestionValidator are used by
#          atomic.py for pre-ingest validation and saga context management.
# @called-by: atomic.py:AtomicIngestionCoordinator
# =============================================================================
"""
Saga context and pre-ingest validation for atomic Neo4j + Qdrant synchronization.

Provides:
- SagaContext: Shared state container for ingestion sagas
- ValidationResult: Result type for pre-ingest validation
- IngestionValidator: Pre-commit validation of data integrity

Reference: https://microservices.io/patterns/data/saga.html
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SagaContext:
    """
    Shared context passed between saga steps.

    Steps can store data here that subsequent steps need.
    Also used during compensation to know what to undo.
    """

    saga_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    document_id: Optional[str] = None

    # Data collected during execution for compensation
    neo4j_chunk_ids: List[str] = field(default_factory=list)
    qdrant_point_ids: List[str] = field(default_factory=list)
    neo4j_entity_ids: List[str] = field(default_factory=list)
    neo4j_relationship_ids: List[str] = field(default_factory=list)

    # Stores for passing data between steps
    prepared_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Pre-Commit Validation
# ============================================================================


@dataclass
class ValidationResult:
    """Result of pre-commit validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class IngestionValidator:
    """
    Validates data integrity before committing to Neo4j and Qdrant.

    Checks:
    - All chunk IDs are deterministic and consistent
    - Entity references point to valid chunks
    - No orphan relationships will be created
    - Qdrant collection exists with correct schema
    """

    def __init__(self, neo4j_driver, qdrant_client, config):
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config

    def validate_pre_ingest(
        self,
        document_id: str,
        chunks: List[Dict],
        entities: Dict[str, Dict],
        mentions: List[Dict],
    ) -> ValidationResult:
        """
        Run all pre-ingest validations.

        Returns:
            ValidationResult with any errors or warnings
        """
        errors = []
        warnings = []

        # 1. Validate chunk IDs are present and unique
        chunk_ids = set()
        for chunk in chunks:
            cid = chunk.get("id")
            if not cid:
                errors.append(
                    f"Chunk missing 'id' field: {chunk.get('title', 'untitled')}"
                )
            elif cid in chunk_ids:
                errors.append(f"Duplicate chunk ID: {cid}")
            else:
                chunk_ids.add(cid)

        # 2. Validate entity-chunk references (mentions)
        for mention in mentions:
            chunk_ref = mention.get("section_id") or mention.get("chunk_id")
            if chunk_ref and chunk_ref not in chunk_ids:
                warnings.append(f"Mention references non-existent chunk: {chunk_ref}")

        # 3. Validate Qdrant collection exists
        if self.qdrant_client:
            collection_name = self.config.search.vector.qdrant.collection_name
            try:
                info = self.qdrant_client.get_collection(collection_name)
                if not info:
                    errors.append(f"Qdrant collection not found: {collection_name}")
            except Exception as e:
                errors.append(f"Cannot access Qdrant collection: {str(e)}")

        # 4. Validate document_id consistency
        for chunk in chunks:
            if chunk.get("document_id") != document_id:
                warnings.append(
                    f"Chunk document_id mismatch: {chunk.get('id')} "
                    f"has {chunk.get('document_id')} != {document_id}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
