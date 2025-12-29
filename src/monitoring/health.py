"""
Phase 7E-4: Health Check System
Verifies Neo4j schema v2.2, Qdrant 1024-D, embedding configuration at startup

Reference: Canonical Spec L3513-3528, L535, L621, L3570
Integration Guide L1905-1918
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List

from neo4j import Driver
from qdrant_client import QdrantClient

from src.shared.config import get_config, get_embedding_plan

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_ok(self) -> bool:
        """Check if result is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_ok(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY and all(
            c.is_ok() for c in self.checks
        )

    def get_failures(self) -> List[HealthCheckResult]:
        """Get list of failed checks."""
        return [c for c in self.checks if not c.is_ok()]


class HealthChecker:
    """
    Comprehensive health check system for GraphRAG v2.2.

    Verifies:
    - Neo4j constraints and indexes exist (v2.2 schema)
    - Vector indexes are 1024-D with cosine distance
    - Qdrant collection exists with 1024-D named vectors
    - SchemaVersion marker is v2.2
    - Embedding configuration matches canonical spec
    """

    # Canonical requirements from Phase 7E spec
    REQUIRED_SCHEMA_VERSION = "v2.2"
    REQUIRED_EMBED_DIM = 1024
    REQUIRED_EMBED_MODEL = "BAAI/bge-m3"
    REQUIRED_EMBED_PROVIDER = "bge-m3-service"
    REQUIRED_DISTANCE = "cosine"

    def __init__(
        self,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
        embed_dim: int,
        embed_model: str,
        embed_provider: str,
        qdrant_collection: str = "chunks_multi",
    ):
        """
        Initialize health checker.

        Args:
            neo4j_driver: Neo4j database driver
            qdrant_client: Qdrant vector store client
            embed_dim: Configured embedding dimensions
            embed_model: Configured embedding model ID
            embed_provider: Configured embedding provider
            qdrant_collection: Qdrant collection name
        """
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.embed_dim = embed_dim
        self.embed_model = embed_model
        self.embed_provider = embed_provider
        self.qdrant_collection = qdrant_collection

    def check_all(self, fail_fast: bool = False) -> SystemHealth:
        """
        Run all health checks.

        Args:
            fail_fast: If True, stop on first failure

        Returns:
            SystemHealth with all check results
        """
        checks = []

        # Run each check
        for check_fn in [
            self._check_neo4j_connection,
            self._check_neo4j_schema_version,
            self._check_neo4j_constraints,
            self._check_neo4j_indexes,
            self._check_neo4j_vector_indexes,
            self._check_qdrant_connection,
            self._check_qdrant_collection,
            self._check_qdrant_dimensions,
            self._check_embedding_config,
        ]:
            result = check_fn()
            checks.append(result)

            # Fail-fast should stop only on hard failures. Degraded indicates a
            # non-blocking issue (e.g., optional constraints missing) that should
            # not prevent the service from starting.
            if fail_fast and result.status == HealthStatus.UNHEALTHY:
                logger.error(f"Health check failed (fail_fast): {result.name}")
                break

        # Determine overall status
        if all(c.is_ok() for c in checks):
            overall_status = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in checks):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        return SystemHealth(status=overall_status, checks=checks)

    def _check_neo4j_connection(self) -> HealthCheckResult:
        """Verify Neo4j connection is alive."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 as num")
                record = result.single()
                if record and record["num"] == 1:
                    return HealthCheckResult(
                        name="neo4j_connection",
                        status=HealthStatus.HEALTHY,
                        message="Neo4j connection is healthy",
                    )
                else:
                    return HealthCheckResult(
                        name="neo4j_connection",
                        status=HealthStatus.UNHEALTHY,
                        message="Neo4j returned unexpected result",
                    )
        except Exception as e:
            logger.exception("Neo4j connection check failed")
            return HealthCheckResult(
                name="neo4j_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Neo4j connection failed: {str(e)}",
            )

    def _check_neo4j_schema_version(self) -> HealthCheckResult:
        """Verify SchemaVersion marker matches the required version."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv.version as version"
                )
                record = result.single()

                if not record:
                    return HealthCheckResult(
                        name="schema_version",
                        status=HealthStatus.UNHEALTHY,
                        message="SchemaVersion node not found - run schema migration",
                    )

                version = record["version"]
                if version == self.REQUIRED_SCHEMA_VERSION:
                    return HealthCheckResult(
                        name="schema_version",
                        status=HealthStatus.HEALTHY,
                        message=f"Schema version is {version}",
                        details={"version": version},
                    )
                else:
                    return HealthCheckResult(
                        name="schema_version",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Schema version mismatch: expected {self.REQUIRED_SCHEMA_VERSION}, got {version}",
                        details={
                            "expected": self.REQUIRED_SCHEMA_VERSION,
                            "actual": version,
                        },
                    )
        except Exception as e:
            logger.exception("Schema version check failed")
            return HealthCheckResult(
                name="schema_version",
                status=HealthStatus.UNHEALTHY,
                message=f"Schema version check failed: {str(e)}",
            )

    def _check_neo4j_constraints(self) -> HealthCheckResult:
        """Verify required constraints exist."""
        # Constraints are important for correctness, but the project has gone
        # through a Section -> Chunk transition. In newer schemas, Section may
        # not exist, and Chunk constraints may be added later.
        required_specs = [
            {"label": "Document", "properties": ["id"]},
        ]
        # At least one of these should exist once the chunk model is fully
        # stabilized. Treat absence as DEGRADED (warning), not UNHEALTHY.
        optional_any_specs = [
            {"label": "Section", "properties": ["id"]},
            {"label": "Chunk", "properties": ["id"]},
            {"label": "Chunk", "properties": ["chunk_id"]},
        ]

        try:
            with self.neo4j_driver.session() as session:
                result = session.run("SHOW CONSTRAINTS")
                existing_specs = set()

                for record in result:
                    label = (record.get("labelsOrTypes") or [None])[0]
                    props = tuple(sorted(record.get("properties") or []))
                    if label and props:
                        existing_specs.add((label, props))

                def _has_spec(spec: dict) -> bool:
                    label = spec["label"]
                    props = tuple(sorted(spec["properties"]))
                    return (label, props) in existing_specs

                missing_required = [s for s in required_specs if not _has_spec(s)]
                has_any_optional = any(_has_spec(s) for s in optional_any_specs)

                if not missing_required and has_any_optional:
                    return HealthCheckResult(
                        name="neo4j_constraints",
                        status=HealthStatus.HEALTHY,
                        message="Required constraints exist (including chunk/section constraint)",
                        details={
                            "required": required_specs,
                            "optional_any": optional_any_specs,
                        },
                    )
                if missing_required:
                    return HealthCheckResult(
                        name="neo4j_constraints",
                        status=HealthStatus.UNHEALTHY,
                        message="Missing required constraints",
                        details={"missing_required": missing_required},
                    )
                return HealthCheckResult(
                    name="neo4j_constraints",
                    status=HealthStatus.DEGRADED,
                    message="Missing chunk/section uniqueness constraint (non-blocking)",
                    details={"optional_any": optional_any_specs},
                )
        except Exception as e:
            logger.exception("Constraint check failed")
            return HealthCheckResult(
                name="neo4j_constraints",
                status=HealthStatus.UNHEALTHY,
                message=f"Constraint check failed: {str(e)}",
            )

    def _check_neo4j_indexes(self) -> HealthCheckResult:
        """Verify required property indexes exist."""
        # The project has transitioned from Section -> Chunk; prefer chunk-focused
        # indexes that support structural building and graph features.
        required_indexes = [
            "chunk_doc_order",
            "chunk_doc_parent_path_norm",
            "chunk_parent_chunk_id",
            "entity_type_normalized_name",
        ]

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "SHOW INDEXES YIELD name, type WHERE type <> 'VECTOR' RETURN name"
                )
                existing_indexes = {record["name"] for record in result}

                missing = [
                    idx for idx in required_indexes if idx not in existing_indexes
                ]

                if not missing:
                    return HealthCheckResult(
                        name="neo4j_indexes",
                        status=HealthStatus.HEALTHY,
                        message=f"All {len(required_indexes)} required property indexes exist",
                        details={"indexes": required_indexes},
                    )
                else:
                    return HealthCheckResult(
                        name="neo4j_indexes",
                        status=HealthStatus.DEGRADED,
                        message=f"Missing optional indexes: {', '.join(missing)} (performance may be impacted)",
                        details={"missing": missing},
                    )
        except Exception as e:
            logger.exception("Index check failed")
            return HealthCheckResult(
                name="neo4j_indexes",
                status=HealthStatus.DEGRADED,
                message=f"Index check failed: {str(e)}",
            )

    def _check_neo4j_vector_indexes(self) -> HealthCheckResult:
        """Verify vector indexes are 1024-D with cosine distance."""
        vector_cfg = get_config().search.vector
        neo4j_vectors_required = vector_cfg.primary == "neo4j" or vector_cfg.dual_write
        if not neo4j_vectors_required:
            return HealthCheckResult(
                name="neo4j_vector_indexes",
                status=HealthStatus.HEALTHY,
                message="Neo4j vector indexes not required (vector primary is qdrant)",
                details={
                    "primary": vector_cfg.primary,
                    "dual_write": vector_cfg.dual_write,
                },
            )

        # Allow namespacing: section index name is derived from config.search.vector.neo4j.index_name
        # while chunk index remains legacy/non-namespaced for now.
        required_vector_indexes = {
            "section": {
                "names": [
                    getattr(
                        get_config().search.vector.neo4j,
                        "index_name",
                        "section_embeddings_v2",
                    )
                ],
                "label": "Section",
                "property": "vector_embedding",
                "dimensions": 1024,
            },
            "chunk": {
                "names": ["chunk_embeddings_v2"],
                "label": "Chunk",
                "property": "vector_embedding",
                "dimensions": 1024,
            },
        }

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, labelsOrTypes, properties, options
                    WHERE type = 'VECTOR'
                    RETURN name, labelsOrTypes, properties, options
                    """
                )

                existing_indexes = {}
                for record in result:
                    name = record["name"]
                    existing_indexes[name] = {
                        "label": (
                            record["labelsOrTypes"][0]
                            if record["labelsOrTypes"]
                            else None
                        ),
                        "property": (
                            record["properties"][0] if record["properties"] else None
                        ),
                        "options": record.get("options", {}),
                    }

                issues = []
                for logical_name, spec in required_vector_indexes.items():
                    names = spec["names"]
                    matched_name = next(
                        (n for n in names if n in existing_indexes), None
                    )
                    if not matched_name:
                        issues.append(
                            f"{logical_name} missing (tried {', '.join(names)})"
                        )
                        continue

                    idx = existing_indexes[matched_name]
                    options = idx.get("options", {})

                    # Check dimensions (in indexConfig)
                    index_config = options.get("indexConfig", {})
                    dims = index_config.get("vector.dimensions")
                    if dims != spec["dimensions"]:
                        issues.append(
                            f"{matched_name} has {dims}D (expected {spec['dimensions']}D)"
                        )

                    # Check similarity (cosine expected)
                    similarity = index_config.get(
                        "vector.similarity_function", ""
                    ).lower()
                    if similarity and "cosine" not in similarity:
                        issues.append(
                            f"{matched_name} uses {similarity} (expected cosine)"
                        )

                if not issues:
                    return HealthCheckResult(
                        name="neo4j_vector_indexes",
                        status=HealthStatus.HEALTHY,
                        message=f"All {len(required_vector_indexes)} vector indexes are 1024-D cosine",
                        details={"indexes": required_vector_indexes},
                    )
                else:
                    return HealthCheckResult(
                        name="neo4j_vector_indexes",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Vector index issues: {'; '.join(issues)}",
                        details={"issues": issues},
                    )
        except Exception as e:
            logger.exception("Vector index check failed")
            return HealthCheckResult(
                name="neo4j_vector_indexes",
                status=HealthStatus.UNHEALTHY,
                message=f"Vector index check failed: {str(e)}",
            )

    def _check_qdrant_connection(self) -> HealthCheckResult:
        """Verify Qdrant connection is alive."""
        try:
            # Simple health check - get collections
            collections = self.qdrant_client.get_collections()
            return HealthCheckResult(
                name="qdrant_connection",
                status=HealthStatus.HEALTHY,
                message="Qdrant connection is healthy",
                details={"collections": [c.name for c in collections.collections]},
            )
        except Exception as e:
            logger.exception("Qdrant connection check failed")
            return HealthCheckResult(
                name="qdrant_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant connection failed: {str(e)}",
            )

    def _check_qdrant_collection(self) -> HealthCheckResult:
        """Verify Qdrant collection exists."""
        try:
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
            # vectors_count is deprecated in qdrant-client 1.16+, use indexed_vectors_count
            vectors_count = (
                collection_info.indexed_vectors_count
                if collection_info.indexed_vectors_count is not None
                else collection_info.vectors_count or 0
            )
            return HealthCheckResult(
                name="qdrant_collection",
                status=HealthStatus.HEALTHY,
                message=f"Collection '{self.qdrant_collection}' exists",
                details={
                    "points_count": collection_info.points_count or 0,
                    "indexed_vectors_count": vectors_count,
                },
            )
        except Exception as e:
            logger.exception(
                f"Qdrant collection '{self.qdrant_collection}' check failed"
            )
            return HealthCheckResult(
                name="qdrant_collection",
                status=HealthStatus.UNHEALTHY,
                message=f"Collection '{self.qdrant_collection}' not found or inaccessible: {str(e)}",
            )

    def _check_qdrant_dimensions(self) -> HealthCheckResult:
        """Verify Qdrant collection uses 1024-D named vector 'content' with cosine distance."""
        try:
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
            vectors_config = collection_info.config.params.vectors

            # Qdrant named vectors config
            if hasattr(vectors_config, "get"):  # Named vectors (dict-like)
                content_config = vectors_config.get("content")
                if not content_config:
                    return HealthCheckResult(
                        name="qdrant_dimensions",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Collection '{self.qdrant_collection}' missing 'content' named vector",
                        details={"available_vectors": list(vectors_config.keys())},
                    )

                size = content_config.size
                distance = content_config.distance.name.lower()

            else:  # Single vector config
                size = vectors_config.size
                distance = vectors_config.distance.name.lower()

            issues = []
            if size != self.REQUIRED_EMBED_DIM:
                issues.append(
                    f"vector size is {size}D (expected {self.REQUIRED_EMBED_DIM}D)"
                )

            if "cosine" not in distance.lower():
                issues.append(f"distance is {distance} (expected cosine)")

            if not issues:
                return HealthCheckResult(
                    name="qdrant_dimensions",
                    status=HealthStatus.HEALTHY,
                    message=f"Collection uses {size}D {distance} vectors",
                    details={"size": size, "distance": distance},
                )
            else:
                return HealthCheckResult(
                    name="qdrant_dimensions",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Qdrant config issues: {'; '.join(issues)}",
                    details={"issues": issues},
                )
        except Exception as e:
            logger.exception("Qdrant dimensions check failed")
            return HealthCheckResult(
                name="qdrant_dimensions",
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant dimensions check failed: {str(e)}",
            )

    def _check_embedding_config(self) -> HealthCheckResult:
        """Verify embedding configuration matches canonical spec."""
        expected_dim = self.REQUIRED_EMBED_DIM
        expected_model = self.REQUIRED_EMBED_MODEL
        expected_provider = self.REQUIRED_EMBED_PROVIDER

        try:
            plan = get_embedding_plan()
        except Exception as exc:
            logger.warning(
                "Embedding plan unavailable; falling back to legacy embedding expectations.",
                exc_info=exc,
            )
            plan = None

        if plan and plan.dense:
            dense_profile = plan.dense.profile
            if getattr(dense_profile, "dims", None):
                expected_dim = dense_profile.dims
            if getattr(dense_profile, "model_id", None):
                expected_model = dense_profile.model_id
            if getattr(dense_profile, "provider", None):
                expected_provider = dense_profile.provider

        issues = []

        if self.embed_dim != expected_dim:
            issues.append(f"EMBED_DIM is {self.embed_dim} (expected {expected_dim})")

        if self.embed_model != expected_model:
            issues.append(
                f"EMBED_MODEL is '{self.embed_model}' (expected '{expected_model}')"
            )

        if self.embed_provider != expected_provider:
            issues.append(
                f"EMBED_PROVIDER is '{self.embed_provider}' (expected '{expected_provider}')"
            )

        if not issues:
            return HealthCheckResult(
                name="embedding_config",
                status=HealthStatus.HEALTHY,
                message=f"Embedding config matches canonical spec: {self.embed_model} @ {self.embed_dim}D",
                details={
                    "model": self.embed_model,
                    "provider": self.embed_provider,
                    "dimensions": self.embed_dim,
                },
            )
        else:
            return HealthCheckResult(
                name="embedding_config",
                status=HealthStatus.UNHEALTHY,
                message=f"Embedding config drift: {'; '.join(issues)}",
                details={"issues": issues},
            )


def run_startup_health_checks(
    neo4j_driver: Driver,
    qdrant_client: QdrantClient,
    embed_dim: int,
    embed_model: str,
    embed_provider: str,
    qdrant_collection: str = "chunks",
    fail_fast: bool = True,
) -> SystemHealth:
    """
    Run health checks at application startup.

    Args:
        neo4j_driver: Neo4j database driver
        qdrant_client: Qdrant vector store client
        embed_dim: Configured embedding dimensions
        embed_model: Configured embedding model ID
        embed_provider: Configured embedding provider
        qdrant_collection: Qdrant collection name
        fail_fast: Stop on first failure

    Returns:
        SystemHealth result

    Raises:
        RuntimeError: If fail_fast=True and checks fail
    """
    checker = HealthChecker(
        neo4j_driver=neo4j_driver,
        qdrant_client=qdrant_client,
        embed_dim=embed_dim,
        embed_model=embed_model,
        embed_provider=embed_provider,
        qdrant_collection=qdrant_collection,
    )

    logger.info("Running startup health checks...")
    health = checker.check_all(fail_fast=fail_fast)

    if health.is_ok():
        logger.info("✅ All health checks passed")
    else:
        failures = health.get_failures()
        if health.status == HealthStatus.UNHEALTHY:
            logger.error(f"❌ {len(failures)} health check(s) failed:")
            for check in failures:
                logger.error(f"  - {check.name}: {check.message}")
            if fail_fast:
                raise RuntimeError(
                    f"Health checks failed: {len(failures)} check(s) unhealthy/degraded"
                )
        else:
            logger.warning(f"⚠️ {len(failures)} health check(s) degraded:")
            for check in failures:
                logger.warning(f"  - {check.name}: {check.message}")

    return health
