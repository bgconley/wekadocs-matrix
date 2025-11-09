# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# See: /docs/expert-coder-guidance.md → 1.2 (connection pools, graceful shutdown)
# Connection pools for Neo4j, Qdrant, and Redis

import string
import uuid
from typing import Any, Dict, Iterable, Optional, Sequence

import redis.asyncio as aioredis
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
)
from redis.asyncio.connection import ConnectionPool

from .config import Settings, get_settings
from .observability import get_logger

logger = get_logger(__name__)


def _as_sequence(points: Iterable[str] | str | None) -> Sequence[str]:
    if points is None:
        return ()
    if isinstance(points, (list, tuple, set)):
        return list(points)
    return [points]


def _convert_point_id(point_id: str) -> str:
    if (
        isinstance(point_id, str)
        and len(point_id) == 64
        and all(ch in string.hexdigits for ch in point_id)
    ):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
    return point_id


def _normalize_points_selector(
    *,
    points: Optional[Iterable[str]] = None,
    points_selector: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Accepts either `points=[ids]` or `points_selector={<filter or ids>}`
    and returns kwargs suitable for Qdrant delete APIs.
    """
    if points_selector is not None:
        ps = points_selector
        if isinstance(ps, PointIdsList):
            converted = [_convert_point_id(pid) for pid in ps.points]
            return {"points_selector": PointIdsList(points=converted)}
        if isinstance(ps, FilterSelector):
            return {"points_selector": ps}
        if hasattr(ps, "dict") or hasattr(ps, "model_fields"):
            return {"points_selector": ps}
        if isinstance(ps, (list, tuple, set)):
            converted = [_convert_point_id(pid) for pid in ps]
            return {"points_selector": PointIdsList(points=converted)}
        if isinstance(ps, dict):
            filt = ps.get("filter")
            if filt:
                must = []
                for cond in filt.get("must", []):
                    match = cond.get("match", {})
                    must.append(
                        FieldCondition(
                            key=cond.get("key"),
                            match=MatchValue(value=match.get("value")),
                        )
                    )
                return {"points_selector": FilterSelector(filter=Filter(must=must))}
            if "points" in ps:
                converted = [
                    _convert_point_id(pid) for pid in _as_sequence(ps["points"])
                ]
                return {"points_selector": PointIdsList(points=converted)}

    if points is not None:
        converted = [_convert_point_id(pid) for pid in _as_sequence(points)]
        return {"points_selector": PointIdsList(points=converted)}

    raise ValueError(
        "Unsupported points selector. Provide `points=[...]` or `points_selector=Filter/PointIdsList`."
    )


class CompatQdrantClient(QdrantClient):
    """
    Subclass of QdrantClient that normalizes delete selectors and converts
    deterministic Section IDs (SHA-256 hex) to UUIDs expected by the vector store.
    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], QdrantClient):
            base_client: QdrantClient = args[0]
            self.__dict__ = base_client.__dict__
        else:
            super().__init__(*args, **kwargs)

    def delete_compat(
        self,
        collection_name: str,
        *,
        points: Optional[Iterable[str]] = None,
        points_selector: Optional[Any] = None,
        wait: bool = True,
    ):
        normalized = _normalize_points_selector(
            points=points, points_selector=points_selector
        )
        return super().delete(collection_name=collection_name, wait=wait, **normalized)

    def delete(self, collection_name: str, wait: bool | None = None, **kwargs):
        normalized = _normalize_points_selector(
            points=kwargs.pop("points", None),
            points_selector=kwargs.pop("points_selector", None),
        )
        wait_arg = wait if wait is not None else kwargs.pop("wait", True)
        return super().delete(
            collection_name=collection_name, wait=wait_arg, **normalized
        )

    def purge_document(self, collection_name: str, document_id: str):
        """Delete all vectors for a specific document."""
        filt = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        )
        return super().delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=filt),
            wait=True,
        )

    def upsert_validated(
        self,
        collection_name: str,
        points: Sequence,
        expected_dim: int | Dict[str, int],
        wait: bool = True,
    ):
        """
        Pre-Phase 7: Upsert points with dimension validation.
        Validates that all vectors match the expected dimension before upserting.

        Args:
            collection_name: The Qdrant collection to upsert to
            points: List of PointStruct objects to upsert
            expected_dim: Expected vector dimension
            wait: Whether to wait for operation completion

        Raises:
            ValueError: If any vector dimension doesn't match expected_dim
        """
        # Pre-Phase 7 (G1): Metrics instrumentation
        import time

        from src.shared.observability.metrics import (
            qdrant_operation_latency_ms,
            qdrant_upsert_total,
        )

        start_time = time.time()
        status = "success"

        try:
            # Validate all point vectors before upsert
            for i, point in enumerate(points):
                vectors = self._extract_point_vectors(point)
                if vectors:
                    if isinstance(vectors, dict):
                        iterator = vectors.items()
                    else:
                        iterator = [(None, vectors)]

                    for name, vec in iterator:
                        if vec is None:
                            continue
                        expected = (
                            expected_dim.get(name)
                            if isinstance(expected_dim, dict)
                            else expected_dim
                        )
                        if expected is None and isinstance(expected_dim, dict):
                            expected = next(iter(expected_dim.values()), None)
                        if expected and len(vec) != expected:
                            label = name or "vector"
                            raise ValueError(
                                f"Dimension mismatch for vector '{label}' in point {i}: "
                                f"expected {expected}, got {len(vec)}. "
                                f"Point ID: {getattr(point, 'id', 'unknown')}"
                            )
                else:
                    # Fall back to treating point.vector as a single unnamed vector
                    raw_vector = getattr(point, "vector", None) or []
                    actual_dim = len(raw_vector)
                    if isinstance(expected_dim, dict):
                        exp_dim = expected_dim.get("content") or next(
                            iter(expected_dim.values())
                        )
                    else:
                        exp_dim = expected_dim
                    if exp_dim and actual_dim != exp_dim:
                        raise ValueError(
                            f"Dimension mismatch in point {i}: expected {exp_dim}, "
                            f"got {actual_dim}. Point ID: {getattr(point, 'id', 'unknown')}"
                        )

            # All dimensions valid, proceed with upsert
            result = super().upsert(
                collection_name=collection_name,
                points=points,
                wait=wait,
            )

            # Record success metrics
            latency_ms = (time.time() - start_time) * 1000
            qdrant_upsert_total.labels(
                collection_name=collection_name, status=status
            ).inc()
            qdrant_operation_latency_ms.labels(
                collection_name=collection_name, operation="upsert"
            ).observe(latency_ms)

            return result

        except Exception:
            status = "error"
            qdrant_upsert_total.labels(
                collection_name=collection_name, status=status
            ).inc()
            raise

    @staticmethod
    def _extract_point_vectors(point: Any):
        """
        Normalize PointStruct vector payloads (single unnamed vectors or named
        vector dictionaries) into a structure we can inspect.
        """
        candidate = None
        if hasattr(point, "vector") and point.vector is not None:
            candidate = point.vector
        elif hasattr(point, "vectors") and point.vectors is not None:
            candidate = point.vectors
        elif isinstance(getattr(point, "payload", None), dict):
            payload_vectors = point.payload.get("vectors")
            if payload_vectors:
                candidate = payload_vectors

        if candidate is None:
            return None

        # Convert NamedVector structures into plain lists so len() reflects
        # dimensionality instead of field count.
        if isinstance(candidate, dict):
            normalized = {}
            for name, vec in candidate.items():
                if hasattr(vec, "vector"):
                    normalized[name] = list(vec.vector)
                elif isinstance(vec, list):
                    normalized[name] = vec
                elif isinstance(vec, tuple):
                    normalized[name] = list(vec)
                else:
                    normalized[name] = vec
            return normalized

        if hasattr(candidate, "vector"):
            return list(candidate.vector)

        return candidate

    def create_collection_with_dims(
        self,
        collection_name: str,
        size: int,
        distance: str = "Cosine",
    ):
        """
        Pre-Phase 7: Helper to create a collection with specific dimensions.
        Supports blue/green collection pattern for dimension changes.

        Args:
            collection_name: Name for the new collection
            size: Vector dimension size
            distance: Distance metric (Cosine, Euclid, Dot)

        Note: This helper prepares for future blue/green migrations
              but does NOT execute them automatically.
        """
        from qdrant_client.models import Distance, VectorParams

        # Map string distance to enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        distance_enum = distance_map.get(distance.lower(), Distance.COSINE)

        # Check if collection already exists
        collections = self.get_collections()
        if collection_name not in [c.name for c in collections.collections]:
            logger.info(
                f"Creating collection '{collection_name}' with "
                f"size={size}, distance={distance}"
            )
            self.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=distance_enum),
            )
        else:
            logger.info(
                f"Collection '{collection_name}' already exists, skipping creation"
            )


class ConnectionManager:
    """Manages connections to Neo4j, Qdrant, and Redis"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._neo4j_driver: Optional[Driver] = None
        self._qdrant_client: Optional[CompatQdrantClient] = None
        self._redis_pool: Optional[ConnectionPool] = None
        self._redis_client: Optional[aioredis.Redis] = None

    # Neo4j
    def get_neo4j_driver(self) -> Driver:
        """Get or create Neo4j driver"""
        if self._neo4j_driver is None:
            logger.info(
                "Initializing Neo4j driver",
                uri=self.settings.neo4j_uri,
                user=self.settings.neo4j_user,
            )
            self._neo4j_driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                connection_timeout=1.5,  # Phase 7a: 1500ms connection timeout
            )
            # Verify connectivity
            self._neo4j_driver.verify_connectivity()
            logger.info("Neo4j driver initialized successfully")
        return self._neo4j_driver

    async def close_neo4j(self) -> None:
        """Close Neo4j driver"""
        if self._neo4j_driver:
            logger.info("Closing Neo4j driver")
            self._neo4j_driver.close()
            self._neo4j_driver = None

    # Qdrant
    def get_qdrant_client(self) -> CompatQdrantClient:
        """Get or create Qdrant client with compatibility wrapper"""
        if self._qdrant_client is None:
            logger.info(
                "Initializing Qdrant client",
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )
            raw_client = CompatQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30,
            )
            self._qdrant_client = raw_client
            logger.info("Qdrant client initialized successfully")
        return self._qdrant_client

    async def close_qdrant(self) -> None:
        """Close Qdrant client"""
        if self._qdrant_client:
            logger.info("Closing Qdrant client")
            self._qdrant_client.close()
            self._qdrant_client = None

    # Redis
    async def get_redis_client(self) -> aioredis.Redis:
        """Get or create Redis client"""
        if self._redis_client is None:
            logger.info(
                "Initializing Redis client",
                host=self.settings.redis_host,
                port=self.settings.redis_port,
            )
            self._redis_pool = ConnectionPool(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=0,
                decode_responses=True,
                max_connections=50,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._redis_client = aioredis.Redis(connection_pool=self._redis_pool)
            # Verify connectivity
            await self._redis_client.ping()
            logger.info("Redis client initialized successfully")
        return self._redis_client

    async def close_redis(self) -> None:
        """Close Redis client"""
        if self._redis_client:
            logger.info("Closing Redis client")
            await self._redis_client.close()
            self._redis_client = None
        if self._redis_pool:
            await self._redis_pool.disconnect()
            self._redis_pool = None

    # Lifecycle management
    async def initialize_all(self) -> None:
        """Initialize all connections"""
        logger.info("Initializing all connections")
        self.get_neo4j_driver()
        self.get_qdrant_client()
        await self.get_redis_client()
        logger.info("All connections initialized")

    async def close_all(self) -> None:
        """Close all connections gracefully"""
        logger.info("Closing all connections")
        await self.close_neo4j()
        await self.close_qdrant()
        await self.close_redis()
        logger.info("All connections closed")


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global ConnectionManager instance"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def initialize_connections() -> ConnectionManager:
    """Initialize and return the global ConnectionManager"""
    manager = get_connection_manager()
    await manager.initialize_all()
    return manager


async def close_connections() -> None:
    """Close all connections in the global ConnectionManager"""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.close_all()
        _connection_manager = None
