"""
Phase 2, Task 2.3 Tests: Hybrid Search
Tests vector + graph retrieval, ranking, and performance with NO MOCKS.
Requires seeded graph from scripts/dev/seed_minimal_graph.py
"""

import os
import statistics
import time

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.query.hybrid_search import (HybridSearchEngine, Neo4jVectorStore,
                                     QdrantVectorStore, SearchResult)
from src.query.ranking import rank_results
from src.shared.config import get_config


@pytest.fixture(scope="module")
def config():
    """Load config."""
    return get_config()


@pytest.fixture(scope="module")
def neo4j_driver(config):
    """Get Neo4j driver."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "testpassword123")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


@pytest.fixture(scope="module")
def embedder(config):
    """Get embedder."""
    model_name = config.embedding.model_name
    return SentenceTransformer(model_name)


@pytest.fixture(scope="module")
def vector_store(config, neo4j_driver):
    """Get vector store based on config."""
    vector_primary = config.search.vector.primary

    if vector_primary == "qdrant":
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port)
        collection_name = "weka_sections"  # Default from config
        return QdrantVectorStore(client, collection_name)
    else:
        index_name = "section_embeddings"  # Default from config
        return Neo4jVectorStore(neo4j_driver, index_name)


@pytest.fixture(scope="module")
def search_engine(vector_store, neo4j_driver, embedder):
    """Create hybrid search engine."""
    return HybridSearchEngine(vector_store, neo4j_driver, embedder)


class TestVectorSearch:
    """Test vector search component."""

    def test_vector_search_returns_results(self, vector_store, embedder):
        """Test that vector search returns results."""
        query_vector = embedder.encode("installation prerequisites").tolist()
        results = vector_store.search(query_vector, k=5)

        assert len(results) > 0
        assert all("node_id" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["score"] > 0 for r in results)

    def test_vector_search_ranked_by_score(self, vector_store, embedder):
        """Test that results are ranked by similarity score."""
        query_vector = embedder.encode("cluster configuration").tolist()
        results = vector_store.search(query_vector, k=5)

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            # Scores should be descending
            assert scores == sorted(scores, reverse=True)

    def test_vector_search_respects_k(self, vector_store, embedder):
        """Test that vector search respects k limit."""
        query_vector = embedder.encode("weka").tolist()
        results = vector_store.search(query_vector, k=3)

        assert len(results) <= 3


class TestHybridSearch:
    """Test hybrid search engine."""

    def test_hybrid_search_returns_results(self, search_engine):
        """Test that hybrid search returns results."""
        results = search_engine.search("installation prerequisites", k=5)

        assert results.total_found > 0
        assert len(results.results) > 0
        assert results.vector_time_ms > 0
        assert results.total_time_ms > 0

    def test_hybrid_search_with_expansion(self, search_engine):
        """Test that graph expansion adds more results."""
        results_no_expand = search_engine.search(
            "cluster setup", k=5, expand_graph=False
        )

        results_with_expand = search_engine.search(
            "cluster setup", k=5, expand_graph=True
        )

        # With expansion, we should find more total candidates
        assert results_with_expand.total_found >= results_no_expand.total_found
        assert results_with_expand.graph_time_ms > 0

    def test_hybrid_search_distances(self, search_engine):
        """Test that expanded results have higher distances."""
        results = search_engine.search("installation", k=10, expand_graph=True)

        distances = [r.distance for r in results.results]
        # Should have mix of distance 0 (seeds) and higher (expanded)
        assert 0 in distances
        if len(results.results) > 3:
            assert any(d > 0 for d in distances)

    def test_hybrid_search_deduplicates(self, search_engine):
        """Test that results are deduplicated by node_id."""
        results = search_engine.search("configuration", k=20, expand_graph=True)

        node_ids = [r.node_id for r in results.results]
        # No duplicates
        assert len(node_ids) == len(set(node_ids))


class TestRanking:
    """Test ranking component."""

    def test_ranker_ranks_results(self):
        """Test that ranker produces ranked results."""
        # Create mock results
        results = [
            SearchResult(
                "id1",
                "Section",
                score=0.9,
                distance=0,
                metadata={"updated_at": "2024-01-01"},
            ),
            SearchResult(
                "id2",
                "Section",
                score=0.7,
                distance=1,
                metadata={"updated_at": "2024-01-01"},
            ),
            SearchResult(
                "id3",
                "Command",
                score=0.8,
                distance=0,
                metadata={"updated_at": "2023-01-01"},
            ),
        ]

        ranked = rank_results(results)

        assert len(ranked) == 3
        assert all(hasattr(r, "rank") for r in ranked)
        assert all(hasattr(r, "features") for r in ranked)
        # Ranks should be 1, 2, 3
        assert [r.rank for r in ranked] == [1, 2, 3]

    def test_ranker_features_computed(self):
        """Test that all ranking features are computed."""
        result = SearchResult("id1", "Section", score=0.9, distance=0, metadata={})
        ranked = rank_results([result])

        features = ranked[0].features
        assert features.semantic_score >= 0
        assert features.graph_distance_score >= 0
        assert features.recency_score >= 0
        assert features.entity_priority_score >= 0
        assert features.coverage_score >= 0
        assert features.final_score > 0

    def test_ranker_deterministic_ties(self):
        """Test that ties are broken deterministically."""
        # Same scores, different IDs
        results = [
            SearchResult("id_b", "Section", score=0.9, distance=0, metadata={}),
            SearchResult("id_a", "Section", score=0.9, distance=0, metadata={}),
            SearchResult("id_c", "Section", score=0.9, distance=0, metadata={}),
        ]

        ranked1 = rank_results(results)
        ranked2 = rank_results(results)

        # Order should be deterministic (by node_id)
        order1 = [r.result.node_id for r in ranked1]
        order2 = [r.result.node_id for r in ranked2]
        assert order1 == order2
        # Should be sorted by ID
        assert order1 == sorted(order1)

    def test_ranker_prefers_high_semantic_score(self):
        """Test that higher semantic scores get higher ranks."""
        results = [
            SearchResult("id1", "Section", score=0.5, distance=0, metadata={}),
            SearchResult("id2", "Section", score=0.9, distance=0, metadata={}),
        ]

        ranked = rank_results(results)
        # id2 should rank higher
        assert ranked[0].result.node_id == "id2"
        assert ranked[0].rank == 1

    def test_ranker_prefers_close_distance(self):
        """Test that closer graph distance gets higher rank."""
        results = [
            SearchResult("id1", "Section", score=0.8, distance=3, metadata={}),
            SearchResult("id2", "Section", score=0.8, distance=0, metadata={}),
        ]

        ranked = rank_results(results)
        # id2 (distance 0) should rank higher
        assert ranked[0].result.node_id == "id2"


class TestPerformance:
    """Test performance requirements (P95 < 500ms warmed)."""

    @pytest.mark.slow
    def test_hybrid_search_p95_latency_warmed(self, search_engine):
        """Test that P95 latency < 500ms for warmed queries."""
        queries = [
            "installation prerequisites",
            "cluster configuration",
            "troubleshoot errors",
            "network setup",
            "storage configuration",
        ]

        # Warmup: run each query once
        for query in queries:
            search_engine.search(query, k=20, expand_graph=True)

        # Measure: run each query 5 times
        latencies = []
        for query in queries:
            for _ in range(5):
                start = time.time()
                search_engine.search(query, k=20, expand_graph=True)
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)

        # Calculate percentiles
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

        print(f"\nLatency stats (warmed, n={len(latencies)}):")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P95: {p95:.1f}ms")
        print(f"  P99: {p99:.1f}ms")

        # Gate requirement: P95 < 500ms
        assert p95 < 500, f"P95 latency {p95:.1f}ms exceeds 500ms threshold"

    @pytest.mark.slow
    def test_vector_search_latency(self, vector_store, embedder):
        """Test vector search component latency."""
        query_vector = embedder.encode("test query").tolist()

        # Warmup
        for _ in range(3):
            vector_store.search(query_vector, k=20)

        # Measure
        latencies = []
        for _ in range(10):
            start = time.time()
            vector_store.search(query_vector, k=20)
            latencies.append((time.time() - start) * 1000)

        p95 = statistics.quantiles(latencies, n=20)[18]
        print(f"\nVector search P95: {p95:.1f}ms")

        # Vector search should be fast (< 100ms)
        assert p95 < 100, f"Vector search P95 {p95:.1f}ms > 100ms"

    @pytest.mark.slow
    def test_graph_expansion_bounded(self, search_engine):
        """Test that graph expansion completes in reasonable time."""
        results = search_engine.search("installation", k=20, expand_graph=True)

        # Graph expansion should be < 200ms
        assert (
            results.graph_time_ms < 200
        ), f"Graph expansion {results.graph_time_ms:.1f}ms > 200ms"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, search_engine):
        """Test complete search pipeline."""
        results = search_engine.search(
            "how to troubleshoot installation errors",
            k=10,
            expand_graph=True,
            find_paths=False,  # Paths can be slow, skip for basic test
        )

        assert results.total_found > 0
        assert len(results.results) > 0
        assert all(isinstance(r, SearchResult) for r in results.results)

        # Rank results
        ranked = rank_results(results.results)
        assert len(ranked) > 0
        assert ranked[0].rank == 1

    def test_search_with_filters(self, search_engine):
        """Test search with filters."""
        # This assumes the seeded data has document_id
        doc_id = search_engine.vector_store.search(
            search_engine.embedder.encode("test").tolist(), k=1
        )[0].get("document_id")

        if doc_id:
            results = search_engine.search(
                "configuration", k=5, filters={"document_id": doc_id}
            )

            # All results should be from the same document
            for r in results.results:
                assert r.metadata.get("document_id") == doc_id or doc_id in str(
                    r.metadata
                )
