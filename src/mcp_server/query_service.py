"""
Query Service for MCP Server
Integrates hybrid search, ranking, and response building.
Provides cached embedder and connection management.
"""

import time
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer

from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore
from src.query.planner import QueryPlanner
from src.query.ranking import rank_results
from src.query.response_builder import Response, build_response
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger

logger = get_logger(__name__)


class QueryService:
    """
    Service for executing queries against the knowledge graph.
    Manages connections, embedder caching, and query execution pipeline.
    """

    def __init__(self):
        self.config = get_config()
        self._embedder: Optional[SentenceTransformer] = None
        self._search_engine: Optional[HybridSearchEngine] = None
        self._planner: Optional[QueryPlanner] = None

        logger.info("QueryService initialized")

    def _get_embedder(self) -> SentenceTransformer:
        """Get or initialize the cached embedder."""
        if self._embedder is None:
            model_name = self.config.embedding.embedding_model
            logger.info(f"Loading embedding model: {model_name}")
            self._embedder = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
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
    ) -> Response:
        """
        Execute a search query and return formatted response.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional filters for vector search
            expand_graph: Whether to expand from seeds via graph
            find_paths: Whether to find connecting paths

        Returns:
            Response with Markdown and JSON including evidence and confidence

        Raises:
            Exception: If search fails
        """
        start_time = time.time()

        try:
            # Classify intent using planner
            planner = self._get_planner()
            query_plan = planner.plan(query, filters=filters)
            intent = query_plan.intent
            logger.info(f"Query intent classified: {intent}")

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

            # Build response
            response = build_response(
                query=query,
                intent=intent,
                ranked_results=ranked_results,
                timing=timing,
                filters=filters,
            )

            logger.info(
                f"Query completed: confidence={response.answer_json.confidence:.2f}, "
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
