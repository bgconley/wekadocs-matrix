"""
Query Service for MCP Server
Integrates hybrid search, ranking, and response building.
Provides cached embedder and connection management.
Pre-Phase 7 B4: Modified to use embedding provider abstraction.
"""

import json
import time
from typing import Any, Dict, Optional

from src.providers.embeddings import SentenceTransformersProvider
from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore
from src.query.planner import QueryPlanner
from src.query.ranking import rank_results
from src.query.response_builder import Response, Verbosity, build_response
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    mcp_search_response_size_bytes,
    mcp_search_verbosity_total,
)

logger = get_logger(__name__)


class QueryService:
    """
    Service for executing queries against the knowledge graph.
    Manages connections, embedder caching, and query execution pipeline.
    """

    def __init__(self):
        self.config = get_config()
        self._embedder: Optional[SentenceTransformersProvider] = None
        self._search_engine: Optional[HybridSearchEngine] = None
        self._planner: Optional[QueryPlanner] = None

        logger.info("QueryService initialized")

    def _get_embedder(self) -> SentenceTransformersProvider:
        """
        Get or initialize the cached embedder.
        Pre-Phase 7 B4: Uses provider abstraction with dimension validation.
        """
        if self._embedder is None:
            model_name = self.config.embedding.embedding_model
            expected_dims = self.config.embedding.dims

            logger.info(
                f"Loading embedding provider: {model_name}, dims={expected_dims}"
            )

            # Pre-Phase 7: Use provider abstraction
            self._embedder = SentenceTransformersProvider(
                model_name=model_name, expected_dims=expected_dims
            )

            # Validate dimensions match configuration
            if self._embedder.dims != expected_dims:
                raise ValueError(
                    f"Provider dimension mismatch: expected {expected_dims}, "
                    f"got {self._embedder.dims}"
                )

            logger.info(
                f"Embedding provider loaded successfully: "
                f"dims={self._embedder.dims}, model_id={self._embedder.model_id}"
            )
        return self._embedder

    def _get_search_engine(self) -> HybridSearchEngine:
        """Get or initialize the search engine."""
        if self._search_engine is None:
            manager = get_connection_manager()

            # Get connections
            neo4j_driver = manager.get_neo4j_driver()
            qdrant_client = manager.get_qdrant_client()
            embedder = self._get_embedder()

            # Determine vector store (from config)
            primary = self.config.search.vector.primary

            if primary == "qdrant":
                collection_name = self.config.search.vector.qdrant.collection_name
                vector_store = QdrantVectorStore(qdrant_client, collection_name)
                logger.info(f"Using Qdrant vector store: {collection_name}")
            else:
                # Neo4j vectors (if configured)
                from src.query.hybrid_search import Neo4jVectorStore

                index_name = self.config.search.vector.neo4j.index_name
                vector_store = Neo4jVectorStore(neo4j_driver, index_name)
                logger.info(f"Using Neo4j vector store: {index_name}")

            self._search_engine = HybridSearchEngine(
                vector_store=vector_store,
                neo4j_driver=neo4j_driver,
                embedder=embedder,
            )
            logger.info("Search engine initialized")

        return self._search_engine

    def _get_planner(self) -> QueryPlanner:
        """Get or initialize the query planner."""
        if self._planner is None:
            self._planner = QueryPlanner()
            logger.info("Query planner initialized")
        return self._planner

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        expand_graph: bool = True,
        find_paths: bool = False,
        verbosity: str = "graph",
    ) -> Response:
        """
        Execute a search query and return formatted response.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional filters for vector search
            expand_graph: Whether to expand from seeds via graph
            find_paths: Whether to find connecting paths
            verbosity: Response detail level (full=text only, graph=text+relationships, default=graph)

        Returns:
            Response with Markdown and JSON including evidence and confidence

        Raises:
            Exception: If search fails
            ValueError: If verbosity is invalid
        """
        start_time = time.time()

        try:
            # Convert verbosity string to enum
            try:
                verb_enum = Verbosity(verbosity)
            except ValueError:
                raise ValueError(
                    f"Invalid verbosity '{verbosity}'. Must be one of: full, graph"
                )

            # Classify intent using planner
            planner = self._get_planner()
            query_plan = planner.plan(query, filters=filters)
            intent = query_plan.intent
            logger.info(f"Query intent classified: {intent}")

            # Pre-Phase 7 B4: Add embedding_version filter to ensure version consistency
            # This ensures we only retrieve vectors created with the current embedding model
            if filters is None:
                filters = {}

            # Add embedding_version to filters
            filters["embedding_version"] = self.config.embedding.version
            logger.debug(
                f"Added embedding_version filter: {self.config.embedding.version}"
            )

            # Execute hybrid search
            search_start = time.time()
            search_engine = self._get_search_engine()
            search_results = search_engine.search(
                query_text=query,
                k=top_k,
                filters=filters,
                expand_graph=expand_graph,
                find_paths=find_paths,
            )
            search_time = time.time() - search_start

            logger.info(
                f"Search completed: {search_results.total_found} results in {search_time*1000:.1f}ms"
            )

            # Rank results
            rank_start = time.time()
            ranked_results = rank_results(search_results.results)
            rank_time = time.time() - rank_start

            logger.info(
                f"Ranking completed: {len(ranked_results)} results in {rank_time*1000:.1f}ms"
            )

            # Build timing info
            timing = {
                "vector_search_ms": search_results.vector_time_ms,
                "graph_expansion_ms": search_results.graph_time_ms,
                "ranking_ms": rank_time * 1000,
                "total_ms": (time.time() - start_time) * 1000,
            }

            # Build response with verbosity mode
            manager = get_connection_manager()
            neo4j_driver = manager.get_neo4j_driver()

            response = build_response(
                query=query,
                intent=intent,
                ranked_results=ranked_results,
                timing=timing,
                filters=filters,
                verbosity=verb_enum,
                neo4j_driver=neo4j_driver,
            )

            # Instrument metrics (E5)
            mcp_search_verbosity_total.labels(verbosity=verbosity).inc()

            # Measure response size
            response_json = json.dumps(response.to_dict())
            response_size = len(response_json.encode("utf-8"))
            mcp_search_response_size_bytes.labels(verbosity=verbosity).observe(
                response_size
            )

            logger.info(
                f"Query completed: verbosity={verbosity}, "
                f"response_size_kb={response_size/1024:.1f}, "
                f"confidence={response.answer_json.confidence:.2f}, "
                f"evidence_count={len(response.answer_json.evidence)}, "
                f"total_time={timing['total_ms']:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "embedder_loaded": self._embedder is not None,
            "search_engine_initialized": self._search_engine is not None,
            "planner_initialized": self._planner is not None,
            "model_name": (
                self.config.embedding.embedding_model if self._embedder else None
            ),
        }


# Global query service instance (initialized on demand)
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get or create the global query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service
