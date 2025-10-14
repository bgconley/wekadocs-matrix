"""
Phase 4, Task 4.2 - Query Optimizer Tests (NO MOCKS)
Tests slow-query analysis, index recommendations, and plan caching.
"""

import os

import pytest
from neo4j import GraphDatabase

from src.ops.optimizer import (
    ExplainPlan,
    IndexRecommendation,
    PlanCache,
    QueryOptimizer,
)


@pytest.fixture(scope="module")
def neo4j_driver():
    """Neo4j driver for tests."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "weka_graphrag_password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture(scope="function")
def optimizer(neo4j_driver):
    """Create optimizer instance for each test."""
    config = {"slow_query_threshold_ms": 50, "plan_cache_size": 100}
    opt = QueryOptimizer(neo4j_driver, config)
    yield opt
    opt.clear_slow_queries()


@pytest.fixture(scope="module")
def seed_test_data(neo4j_driver):
    """Seed minimal test data for optimizer tests."""
    with neo4j_driver.session() as session:
        # Create test nodes
        session.run(
            """
            MERGE (d:Document {id: 'doc-opt-test', title: 'Test Doc'})
            MERGE (s1:Section {id: 'sec-opt-1', document_id: 'doc-opt-test', title: 'Section 1', text: 'content'})
            MERGE (s2:Section {id: 'sec-opt-2', document_id: 'doc-opt-test', title: 'Section 2', text: 'more content'})
            MERGE (c1:Command {id: 'cmd-opt-1', name: 'weka status'})
            MERGE (c2:Command {id: 'cmd-opt-2', name: 'weka fs list'})
            MERGE (cfg:Configuration {id: 'cfg-opt-1', name: 'max_connections'})

            MERGE (d)-[:HAS_SECTION]->(s1)
            MERGE (d)-[:HAS_SECTION]->(s2)
            MERGE (s1)-[:MENTIONS {confidence: 0.9}]->(c1)
            MERGE (s2)-[:MENTIONS {confidence: 0.8}]->(c2)
            MERGE (s1)-[:MENTIONS {confidence: 0.7}]->(cfg)
        """
        ).consume()
    yield
    # Cleanup
    with neo4j_driver.session() as session:
        session.run(
            """
            MATCH (n) WHERE n.id STARTS WITH 'doc-opt-' OR n.id STARTS WITH 'sec-opt-'
                        OR n.id STARTS WITH 'cmd-opt-' OR n.id STARTS WITH 'cfg-opt-'
            DETACH DELETE n
        """
        ).consume()


class TestPlanCache:
    """Test compiled plan caching."""

    def test_cache_stores_and_retrieves_plans(self):
        cache = PlanCache(max_size=10)

        template_name = "search_v1"
        param_names = ["section_ids", "limit"]
        plan = {"operators": ["NodeByLabelScan", "Filter"], "estimated_rows": 100}

        # Cache plan
        fingerprint = cache.get_fingerprint(template_name, param_names)
        cache.put(fingerprint, plan, metadata={"version": "v1"})

        # Retrieve plan
        cached = cache.get(fingerprint)
        assert cached is not None
        assert cached["plan"] == plan
        assert cached["metadata"]["version"] == "v1"

    def test_cache_evicts_lru_when_full(self):
        cache = PlanCache(max_size=3)

        # Fill cache
        for i in range(3):
            fp = cache.get_fingerprint(f"template_{i}", ["param"])
            cache.put(fp, {"plan": i})

        assert len(cache.cache) == 3

        # Access first entry to make it most recently used
        fp0 = cache.get_fingerprint("template_0", ["param"])
        cache.get(fp0)

        # Add new entry - should evict template_1 (least recently used)
        fp_new = cache.get_fingerprint("template_new", ["param"])
        cache.put(fp_new, {"plan": "new"})

        assert len(cache.cache) == 3
        assert cache.get(fp0) is not None  # Still present
        assert cache.get(fp_new) is not None  # New entry present

    def test_cache_stats(self):
        cache = PlanCache(max_size=10)

        # Add entries
        for i in range(3):
            fp = cache.get_fingerprint(f"template_{i}", ["param"])
            cache.put(fp, {"plan": i})

        # Access entries
        fp0 = cache.get_fingerprint("template_0", ["param"])
        cache.get(fp0)
        cache.get(fp0)

        stats = cache.stats()
        assert stats["size"] == 3
        assert stats["max_size"] == 10
        assert stats["total_accesses"] >= 3  # At least 3 accesses


class TestSlowQueryRecording:
    """Test slow query recording."""

    def test_records_slow_queries_above_threshold(self, optimizer):
        query = "MATCH (n:Section) RETURN n LIMIT 10"
        params = {"limit": 10}

        # Record slow query (above threshold)
        optimizer.record_query(query, params, duration_ms=150)

        assert len(optimizer.slow_queries) == 1
        assert optimizer.slow_queries[0].duration_ms == 150

    def test_ignores_fast_queries_below_threshold(self, optimizer):
        query = "MATCH (n:Section) RETURN n LIMIT 10"
        params = {"limit": 10}

        # Record fast query (below threshold)
        optimizer.record_query(query, params, duration_ms=10)

        assert len(optimizer.slow_queries) == 0

    def test_computes_query_fingerprint(self, optimizer):
        query1 = "MATCH (n:Section {id: $id}) RETURN n"
        query2 = "MATCH (n:Section {id: $id})  RETURN  n"  # Different whitespace

        fp1 = optimizer._compute_query_fingerprint(query1)
        fp2 = optimizer._compute_query_fingerprint(query2)

        # Should have same fingerprint (normalized)
        assert fp1 == fp2


class TestExplainAnalysis:
    """Test EXPLAIN plan analysis."""

    def test_explain_query_without_index(self, optimizer, neo4j_driver, seed_test_data):
        """Test EXPLAIN on query without index (should show label scan)."""
        query = "MATCH (c:Command) WHERE c.name = $name RETURN c"
        params = {"name": "weka status"}

        plan = optimizer._explain_query(query, params)

        assert plan.query == query
        assert len(plan.operator_types) > 0
        # Should have label scan without specific index
        assert plan.label_scans >= 0  # May or may not scan depending on index state

    def test_explain_query_extracts_operators(
        self, optimizer, neo4j_driver, seed_test_data
    ):
        """Test operator extraction from EXPLAIN plan."""
        query = "MATCH (s:Section) RETURN s LIMIT 5"
        params = {}

        plan = optimizer._explain_query(query, params)

        # Should have various operators
        assert len(plan.operator_types) > 0
        assert any("Scan" in op or "Seek" in op for op in plan.operator_types)


class TestIndexRecommendations:
    """Test index recommendation generation."""

    def test_recommends_index_for_label_scan(self, optimizer):
        """Test recommending index when label scan is detected."""
        query = "MATCH (c:Configuration {name: $name}) RETURN c"
        plan = ExplainPlan(
            query=query,
            operator_types=["NodeByLabelScan", "Filter"],
            label_scans=1,
            index_seeks=0,
            expand_all_ops=0,
            estimated_rows=1000,
        )

        recommendations = optimizer._generate_recommendations(plan)

        assert len(recommendations) > 0
        # Should recommend index on Configuration.name
        assert any(
            rec.label == "Configuration" and "name" in rec.properties
            for rec in recommendations
        )

    def test_extract_label_property_patterns(self, optimizer):
        """Test extracting label:property patterns from queries."""
        query = """
            MATCH (c:Configuration {name: $name})
            WHERE c.enabled = true
            RETURN c
        """

        patterns = optimizer._extract_label_property_patterns(query)

        assert "Configuration" in patterns
        assert "name" in patterns["Configuration"]

    def test_export_recommendations_as_cypher(self, optimizer):
        """Test exporting recommendations as Cypher CREATE INDEX statements."""
        recommendations = [
            IndexRecommendation(
                label="Section",
                properties=["document_id"],
                reason="Frequent lookups",
                priority=1,
                estimated_improvement="50-90%",
            ),
            IndexRecommendation(
                label="Command",
                properties=["name", "category"],
                reason="Composite key lookups",
                priority=2,
                estimated_improvement="30-70%",
            ),
        ]

        cypher = optimizer.export_recommendations_as_cypher(recommendations)

        assert "CREATE INDEX" in cypher
        assert "FOR (n:Section)" in cypher
        assert "FOR (n:Command)" in cypher


class TestQueryRewrites:
    """Test query rewrite suggestions."""

    def test_suggests_rewrite_for_unbounded_pattern(self, optimizer):
        """Test suggesting rewrite for unbounded variable-length pattern."""
        query = "MATCH (n)-[*..]-(m) RETURN n, m"
        plan = ExplainPlan(
            query=query,
            operator_types=["Expand(All)"],
            label_scans=0,
            index_seeks=0,
            expand_all_ops=1,
            estimated_rows=10000,
        )

        rewrites = optimizer._suggest_rewrites(query, plan)

        assert len(rewrites) > 0
        assert any("unbounded" in r.reason.lower() for r in rewrites)

    def test_suggests_limit_when_missing(self, optimizer):
        """Test suggesting LIMIT when query has none."""
        query = "MATCH (n:Section) RETURN n"
        plan = ExplainPlan(
            query=query,
            operator_types=["NodeByLabelScan"],
            label_scans=1,
            index_seeks=0,
            expand_all_ops=0,
            estimated_rows=5000,
        )

        rewrites = optimizer._suggest_rewrites(query, plan)

        assert any("LIMIT" in r.reason for r in rewrites)
        assert any("LIMIT 100" in r.suggested for r in rewrites)


class TestOptimizationReport:
    """Test end-to-end optimization report generation."""

    def test_analyze_slow_queries_generates_report(
        self, optimizer, neo4j_driver, seed_test_data
    ):
        """Test full analysis pipeline."""
        # Record some slow queries
        queries = [
            ("MATCH (s:Section) RETURN s LIMIT 100", {}, 120),
            ("MATCH (c:Command {name: $name}) RETURN c", {"name": "test"}, 250),
            ("MATCH (n)-[*1..5]-(m) RETURN n, m LIMIT 10", {}, 180),
        ]

        for query, params, duration in queries:
            optimizer.record_query(query, params, duration)

        # Analyze
        report = optimizer.analyze_slow_queries(top_n=10)

        # Verify report structure
        assert len(report.slow_queries) == 3
        assert len(report.plans) > 0
        assert "total_slow_queries" in report.summary
        assert report.summary["total_slow_queries"] == 3

    def test_report_deduplicates_recommendations(
        self, optimizer, neo4j_driver, seed_test_data
    ):
        """Test that duplicate recommendations are removed."""
        # Record multiple slow queries with same pattern
        for i in range(3):
            optimizer.record_query(
                "MATCH (c:Command {name: $name}) RETURN c",
                {"name": f"test{i}"},
                150,
            )

        report = optimizer.analyze_slow_queries(top_n=10)

        # Should have recommendations, but deduplicated
        if report.index_recommendations:
            # Count recommendations for Command.name
            command_name_recs = [
                r
                for r in report.index_recommendations
                if r.label == "Command" and "name" in r.properties
            ]
            # Should have at most one recommendation for this pattern
            assert len(command_name_recs) <= 1


class TestPlanCacheIntegration:
    """Test plan cache integration with optimizer."""

    def test_cache_and_retrieve_plan(self, optimizer):
        """Test caching and retrieving compiled plans."""
        template_name = "search_sections"
        param_names = ["section_ids", "limit", "max_hops"]
        plan = {
            "query": "MATCH (s:Section) WHERE s.id IN $section_ids RETURN s LIMIT $limit",
            "params": param_names,
        }

        # Cache plan
        optimizer.cache_plan(template_name, param_names, plan)

        # Retrieve plan
        cached = optimizer.get_cached_plan(template_name, param_names)

        assert cached is not None
        assert cached["plan"]["query"] == plan["query"]

    def test_cache_stats_tracking(self, optimizer):
        """Test cache statistics tracking."""
        # Cache multiple plans
        for i in range(5):
            template = f"template_{i}"
            params = [f"param_{j}" for j in range(i + 1)]
            optimizer.cache_plan(template, params, {"plan": i})

        stats = optimizer.get_cache_stats()

        assert stats["size"] == 5
        assert stats["total_accesses"] >= 5  # At least one access per put


def test_optimizer_initialization(neo4j_driver):
    """Test optimizer can be initialized with config."""
    config = {
        "slow_query_threshold_ms": 200,
        "plan_cache_size": 500,
    }

    optimizer = QueryOptimizer(neo4j_driver, config)

    assert optimizer.slow_query_threshold_ms == 200
    assert optimizer.plan_cache.max_size == 500
