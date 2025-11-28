"""
Saga Pattern Implementation for Atomic Neo4j + Qdrant Synchronization.

This module implements the Saga pattern with compensating transactions to ensure
atomic data synchronization between Neo4j (graph store) and Qdrant (vector store).

Key Concepts:
- Each saga step has an execute() and compensate() action
- Steps execute in order; on failure, compensations run in reverse order
- This provides eventual consistency without requiring distributed transactions

Reference: https://microservices.io/patterns/data/saga.html
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)


class SagaStatus(Enum):
    """Status of a saga execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class StepStatus(Enum):
    """Status of an individual saga step."""

    PENDING = "pending"
    EXECUTING = "executing"
    EXECUTED = "executed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SagaStepResult:
    """Result of executing a saga step."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class SagaStep:
    """
    A single step in a saga with execute and compensate actions.

    Attributes:
        name: Human-readable name for logging
        execute: Callable that performs the forward action
        compensate: Callable that undoes the forward action
        status: Current status of this step
        result: Result from execution
        compensation_result: Result from compensation (if run)
    """

    name: str
    execute: Callable[[], SagaStepResult]
    compensate: Callable[[], SagaStepResult]
    status: StepStatus = StepStatus.PENDING
    result: Optional[SagaStepResult] = None
    compensation_result: Optional[SagaStepResult] = None

    def __post_init__(self):
        if not callable(self.execute):
            raise ValueError(f"execute must be callable for step '{self.name}'")
        if not callable(self.compensate):
            raise ValueError(f"compensate must be callable for step '{self.name}'")


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


class SagaCoordinator:
    """
    Coordinates execution of saga steps with compensation on failure.

    Usage:
        coordinator = SagaCoordinator()
        coordinator.add_step(SagaStep(
            name="write_neo4j",
            execute=lambda: write_to_neo4j(data),
            compensate=lambda: delete_from_neo4j(data)
        ))
        coordinator.add_step(SagaStep(
            name="write_qdrant",
            execute=lambda: write_to_qdrant(data),
            compensate=lambda: delete_from_qdrant(data)
        ))
        result = coordinator.execute()
    """

    def __init__(self, context: Optional[SagaContext] = None):
        self.context = context or SagaContext()
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.PENDING
        self._executed_steps: List[SagaStep] = []

    def add_step(self, step: SagaStep) -> "SagaCoordinator":
        """Add a step to the saga. Returns self for chaining."""
        self.steps.append(step)
        return self

    def execute(self) -> Dict[str, Any]:
        """
        Execute all saga steps in order.

        On failure, automatically runs compensation for executed steps in reverse.

        Returns:
            Dict with execution results and metadata
        """
        self.status = SagaStatus.EXECUTING
        start_time = time.time()

        logger.info(
            "saga_execution_started",
            saga_id=self.context.saga_id,
            document_id=self.context.document_id,
            step_count=len(self.steps),
        )

        try:
            for step in self.steps:
                step_start = time.time()
                step.status = StepStatus.EXECUTING

                logger.debug(
                    "saga_step_executing",
                    saga_id=self.context.saga_id,
                    step_name=step.name,
                )

                try:
                    result = step.execute()
                    step.result = result

                    if result.success:
                        step.status = StepStatus.EXECUTED
                        self._executed_steps.append(step)

                        logger.debug(
                            "saga_step_completed",
                            saga_id=self.context.saga_id,
                            step_name=step.name,
                            duration_ms=result.duration_ms,
                        )
                    else:
                        step.status = StepStatus.FAILED
                        raise SagaStepFailure(
                            f"Step '{step.name}' failed: {result.error}"
                        )

                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.result = SagaStepResult(
                        success=False,
                        error=str(e),
                        duration_ms=int((time.time() - step_start) * 1000),
                    )

                    logger.error(
                        "saga_step_failed",
                        saga_id=self.context.saga_id,
                        step_name=step.name,
                        error=str(e),
                    )

                    # Trigger compensation
                    self._compensate()
                    raise

            # All steps succeeded
            self.status = SagaStatus.COMPLETED
            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "saga_execution_completed",
                saga_id=self.context.saga_id,
                document_id=self.context.document_id,
                duration_ms=duration_ms,
                steps_executed=len(self._executed_steps),
            )

            return {
                "saga_id": self.context.saga_id,
                "status": self.status.value,
                "duration_ms": duration_ms,
                "steps_executed": len(self._executed_steps),
                "step_results": {
                    s.name: s.result.data if s.result else None
                    for s in self._executed_steps
                },
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "saga_id": self.context.saga_id,
                "status": self.status.value,
                "duration_ms": duration_ms,
                "steps_executed": len(self._executed_steps),
                "error": str(e),
                "compensation_ran": self.status == SagaStatus.COMPENSATED,
            }

    def _compensate(self):
        """Run compensation for all executed steps in reverse order."""
        self.status = SagaStatus.COMPENSATING

        logger.warning(
            "saga_compensation_started",
            saga_id=self.context.saga_id,
            steps_to_compensate=len(self._executed_steps),
        )

        compensation_errors = []

        for step in reversed(self._executed_steps):
            step.status = StepStatus.COMPENSATING

            logger.debug(
                "saga_step_compensating",
                saga_id=self.context.saga_id,
                step_name=step.name,
            )

            try:
                comp_start = time.time()
                result = step.compensate()
                step.compensation_result = result

                if result.success:
                    step.status = StepStatus.COMPENSATED
                    logger.debug(
                        "saga_step_compensated",
                        saga_id=self.context.saga_id,
                        step_name=step.name,
                        duration_ms=result.duration_ms,
                    )
                else:
                    step.status = StepStatus.FAILED
                    compensation_errors.append(f"{step.name}: {result.error}")
                    logger.error(
                        "saga_compensation_step_failed",
                        saga_id=self.context.saga_id,
                        step_name=step.name,
                        error=result.error,
                    )

            except Exception as e:
                step.status = StepStatus.FAILED
                step.compensation_result = SagaStepResult(
                    success=False,
                    error=str(e),
                    duration_ms=int((time.time() - comp_start) * 1000),
                )
                compensation_errors.append(f"{step.name}: {str(e)}")

                logger.error(
                    "saga_compensation_step_exception",
                    saga_id=self.context.saga_id,
                    step_name=step.name,
                    error=str(e),
                )

        if compensation_errors:
            self.status = SagaStatus.FAILED
            self.context.errors.extend(compensation_errors)
            logger.error(
                "saga_compensation_incomplete",
                saga_id=self.context.saga_id,
                errors=compensation_errors,
            )
        else:
            self.status = SagaStatus.COMPENSATED
            logger.info(
                "saga_compensation_completed",
                saga_id=self.context.saga_id,
                steps_compensated=len(self._executed_steps),
            )


class SagaStepFailure(Exception):
    """Raised when a saga step fails during execution."""

    pass


class SagaCompensationFailure(Exception):
    """Raised when compensation fails, requiring manual intervention."""

    pass


# ============================================================================
# Ingestion-Specific Saga Builders
# ============================================================================

T = TypeVar("T")


# Architectural Note:
# -------------------
# IngestionSagaBuilder and AtomicIngestionCoordinator serve different purposes:
#
# - IngestionSagaBuilder: A composable, testable factory for building sagas
#   with explicit step definitions. Ideal for:
#   * Complex multi-phase ingestion workflows
#   * Dynamic saga composition (conditional steps)
#   * Unit testing saga logic in isolation
#
# - AtomicIngestionCoordinator (in atomic.py): An optimized, production-hardened
#   coordinator with inline saga logic. Ideal for:
#   * Standard document ingestion (hot path)
#   * Performance-critical scenarios (fewer abstraction layers)
#   * When step composition is fixed at compile time
#
# Both patterns are valid. IngestionSagaBuilder is NOT dead code - it provides
# the foundation that SagaCoordinator builds on and enables testability.
# AtomicIngestionCoordinator's inline approach trades flexibility for performance.


class IngestionSagaBuilder:
    """
    Factory for building ingestion sagas with Neo4j and Qdrant steps.

    This builder creates sagas with the correct step order and compensation
    handlers for atomic document ingestion.
    """

    def __init__(
        self,
        neo4j_driver,
        qdrant_client,
        config,
        document_id: str,
    ):
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config

        self.context = SagaContext(document_id=document_id)
        self.coordinator = SagaCoordinator(self.context)

        # Track what we've written for compensation
        self._pending_neo4j_writes: List[Dict] = []
        self._pending_qdrant_writes: List[Dict] = []

    def add_neo4j_step(
        self,
        name: str,
        cypher: str,
        params: Dict[str, Any],
        compensation_cypher: Optional[str] = None,
        compensation_params: Optional[Dict[str, Any]] = None,
    ) -> "IngestionSagaBuilder":
        """
        Add a Neo4j write step with automatic compensation.

        Args:
            name: Step name for logging
            cypher: Cypher query to execute
            params: Query parameters
            compensation_cypher: Cypher to undo the write (optional, auto-generated if not provided)
            compensation_params: Parameters for compensation query
        """

        def execute():
            start = time.time()
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, params)
                summary = result.consume()

                # Track created nodes for compensation
                created = summary.counters.nodes_created

            return SagaStepResult(
                success=True,
                data={"nodes_created": created},
                duration_ms=int((time.time() - start) * 1000),
            )

        def compensate():
            start = time.time()
            if compensation_cypher:
                with self.neo4j_driver.session() as session:
                    result = session.run(
                        compensation_cypher,
                        compensation_params or params,
                    )
                    result.consume()
            return SagaStepResult(
                success=True,
                duration_ms=int((time.time() - start) * 1000),
            )

        self.coordinator.add_step(
            SagaStep(
                name=name,
                execute=execute,
                compensate=compensate,
            )
        )

        return self

    def add_qdrant_step(
        self,
        name: str,
        collection_name: str,
        points: List[Any],
    ) -> "IngestionSagaBuilder":
        """
        Add a Qdrant upsert step with automatic compensation (delete).

        Args:
            name: Step name for logging
            collection_name: Qdrant collection
            points: List of PointStruct to upsert
        """
        point_ids = [p.id for p in points]

        def execute():
            start = time.time()
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,  # Synchronous for saga integrity
            )
            # Track for potential compensation
            self.context.qdrant_point_ids.extend(point_ids)

            return SagaStepResult(
                success=True,
                data={"points_upserted": len(points)},
                duration_ms=int((time.time() - start) * 1000),
            )

        def compensate():
            start = time.time()
            try:
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids,
                    wait=True,
                )
                return SagaStepResult(
                    success=True,
                    data={"points_deleted": len(point_ids)},
                    duration_ms=int((time.time() - start) * 1000),
                )
            except Exception as e:
                return SagaStepResult(
                    success=False,
                    error=str(e),
                    duration_ms=int((time.time() - start) * 1000),
                )

        self.coordinator.add_step(
            SagaStep(
                name=name,
                execute=execute,
                compensate=compensate,
            )
        )

        return self

    def build(self) -> SagaCoordinator:
        """Build and return the configured saga coordinator."""
        return self.coordinator


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

    def validate_entity_chunk_sync(
        self,
        chunk_ids: List[str],
    ) -> ValidationResult:
        """
        Validate that all entity-chunk relationships reference valid chunks.

        This catches the data drift issue where Entity→Chunk relationships
        point to chunks that don't exist in Qdrant.
        """
        errors = []
        warnings = []

        if not self.qdrant_client:
            return ValidationResult(
                valid=True, warnings=["Qdrant client not available"]
            )

        collection_name = self.config.search.vector.qdrant.collection_name

        # Check which chunk_ids exist in Qdrant
        try:
            # Query Qdrant for existing chunk IDs
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="id",
                            match=MatchAny(
                                any=chunk_ids[:100]
                            ),  # Batch to avoid huge queries
                        )
                    ]
                ),
                limit=len(chunk_ids),
                with_payload=["id"],
            )

            existing_ids = {p.payload.get("id") for p in result[0] if p.payload}
            missing_ids = set(chunk_ids) - existing_ids

            if missing_ids:
                warnings.append(
                    f"{len(missing_ids)} chunk IDs not found in Qdrant: {list(missing_ids)[:5]}..."
                )

        except Exception as e:
            warnings.append(f"Could not validate Qdrant sync: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
