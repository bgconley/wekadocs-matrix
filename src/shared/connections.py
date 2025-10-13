# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# See: /docs/expert-coder-guidance.md → 1.2 (connection pools, graceful shutdown)
# Connection pools for Neo4j, Qdrant, and Redis

from typing import Optional

import redis.asyncio as aioredis
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import (FieldCondition, Filter, FilterSelector,
                                  MatchValue, PointIdsList)
from redis.asyncio.connection import ConnectionPool

from .config import Settings, get_settings
from .observability import get_logger

logger = get_logger(__name__)


def _normalize_points_selector(kwargs):
    """
    Accepts either `points=[ids]` or `points_selector={<filter or ids>}` and
    returns a proper selector object.
    """
    if "points_selector" in kwargs:
        ps = kwargs["points_selector"]
        # Already a model type?
        if hasattr(ps, "dict") or hasattr(ps, "model_fields"):
            return ps
        # Plain list (test passes points_selector=[id1, id2, ...])
        if isinstance(ps, list):
            return PointIdsList(points=ps)
        # Dict with 'filter'
        if isinstance(ps, dict) and "filter" in ps:
            f = ps["filter"]
            # Try to map a simple {"must":[{"key":..,"match":{"value":..}}, ...]}
            must = []
            for cond in f.get("must", []):
                must.append(
                    FieldCondition(
                        key=cond["key"], match=MatchValue(value=cond["match"]["value"])
                    )
                )
            return Filter(must=must)
        # Dict with 'points'
        if isinstance(ps, dict) and "points" in ps:
            return PointIdsList(points=ps["points"])

    if "points" in kwargs:
        return PointIdsList(points=kwargs["points"])

    raise ValueError(
        "Unsupported points selector. Provide `points=[...]` or `points_selector=Filter/PointIdsList`."
    )


class CompatQdrantClient:
    """
    Thin adapter to tolerate different delete() call shapes used by tests.
    Handles conversion of section_ids to UUIDs for point operations.
    Other methods are proxied to the real client.
    """

    def __init__(self, client: QdrantClient):
        self._c = client

    def delete(self, collection_name: str, **kwargs):
        """
        Delete points, with automatic conversion of section IDs to UUIDs.
        Accepts points_selector as list of IDs or typed selector.
        """
        selector = _normalize_points_selector(kwargs)

        # If selector is PointIdsList, convert section IDs to UUIDs
        if isinstance(selector, PointIdsList):
            import uuid

            # Convert each ID to UUID using same scheme as build_graph
            uuid_points = []
            for point_id in selector.points:
                # If it looks like a hex hash (64 chars), convert to UUID
                if isinstance(point_id, str) and len(point_id) == 64:
                    uuid_points.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id)))
                else:
                    # Already a UUID or other format, keep as-is
                    uuid_points.append(point_id)
            selector = PointIdsList(points=uuid_points)

        return self._c.delete(collection_name=collection_name, points_selector=selector)

    def purge_document(self, collection_name: str, document_id: str):
        """Delete all vectors for a specific document."""
        filt = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        )
        return self._c.delete(
            collection_name=collection_name, points_selector=FilterSelector(filter=filt)
        )

    def upsert(self, collection_name: str, points, **kwargs):
        """Explicit passthrough for upsert to avoid serialization issues."""
        return self._c.upsert(collection_name=collection_name, points=points, **kwargs)

    def scroll(self, collection_name: str, **kwargs):
        """Explicit passthrough for scroll."""
        return self._c.scroll(collection_name=collection_name, **kwargs)

    def get_collections(self):
        """Explicit passthrough for get_collections."""
        return self._c.get_collections()

    def create_collection(self, collection_name: str, **kwargs):
        """Explicit passthrough for create_collection."""
        return self._c.create_collection(collection_name=collection_name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._c, name)


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
            raw_client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30,
            )
            self._qdrant_client = CompatQdrantClient(raw_client)
            logger.info("Qdrant client initialized successfully")
        return self._qdrant_client

    async def close_qdrant(self) -> None:
        """Close Qdrant client"""
        if self._qdrant_client:
            logger.info("Closing Qdrant client")
            # Close the underlying client
            if hasattr(self._qdrant_client, "_c"):
                self._qdrant_client._c.close()
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
