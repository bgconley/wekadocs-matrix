# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture)
# See: /docs/expert-coder-guidance.md → 1.2 (connection pools, graceful shutdown)
# Connection pools for Neo4j, Qdrant, and Redis

from typing import Optional

import redis.asyncio as aioredis
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from redis.asyncio.connection import ConnectionPool

from .config import Settings, get_settings
from .observability import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages connections to Neo4j, Qdrant, and Redis"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._neo4j_driver: Optional[Driver] = None
        self._qdrant_client: Optional[QdrantClient] = None
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
    def get_qdrant_client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._qdrant_client is None:
            logger.info(
                "Initializing Qdrant client",
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )
            self._qdrant_client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30,
            )
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
