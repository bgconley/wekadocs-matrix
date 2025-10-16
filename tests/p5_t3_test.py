# Phase 5, Task 5.3 - Comprehensive Testing Framework
# Streamlined version focusing on critical gate requirements

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests
from neo4j import GraphDatabase

# Config
MCP_BASE = "http://localhost:8000"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "testpassword123")
QDRANT_URL = "http://localhost:6333"


# ============================================================================
# Suite 1: Unit & Integration Tests
# ============================================================================


class TestCoreValidation:
    """Core validation tests (building on Phases 1-4)"""

    def test_validator_blocks_dangerous_operations(self):
        """Verify validator blocks dangerous operations"""
        from src.mcp_server.validation import CypherValidator, ValidationError

        validator = CypherValidator()

        dangerous_queries = [
            "MATCH (n) DELETE n",
            "DROP DATABASE neo4j",
            "MERGE (n:Evil) RETURN n",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValidationError):
                validator.validate(query, {})

    def test_validator_returns_valid_result(self):
        """Verify validator returns ValidationResult for valid queries"""
        from src.mcp_server.validation import CypherValidator

        validator = CypherValidator()
        query = "MATCH (n:Section {id: $sid}) RETURN n LIMIT 10"
        result = validator.validate(query, {"sid": "test-id"})

        assert result.valid is True
        assert result.query is not None
        assert result.params == {"sid": "test-id"}

    def test_parser_determinism(self):
        """Verify parsing is deterministic"""
        from src.ingestion.parsers.markdown import parse_markdown

        content = "# Test\n## Section\nContent here."

        r1 = parse_markdown("test://doc", content)
        r2 = parse_markdown("test://doc", content)

        # Both should have same section IDs
        ids1 = [s["id"] for s in r1["Sections"]]
        ids2 = [s["id"] for s in r2["Sections"]]
        assert ids1 == ids2

    def test_schema_idempotence(self):
        """Verify schema operations are idempotent"""
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

        with driver.session(database="neo4j") as session:
            count_before = session.run(
                "SHOW CONSTRAINTS YIELD name RETURN count(name) AS cnt"
            ).single()["cnt"]

            # Re-create constraints (IF NOT EXISTS)
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE"
            )

            count_after = session.run(
                "SHOW CONSTRAINTS YIELD name RETURN count(name) AS cnt"
            ).single()["cnt"]

            assert count_before == count_after

        driver.close()


# ============================================================================
# Suite 2: E2E Workflows
# ============================================================================


class TestEndToEndWorkflows:
    """End-to-end MCP workflows"""

    def test_health_endpoint(self):
        """Verify health endpoint works"""
        response = requests.get(f"{MCP_BASE}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_mcp_initialize(self):
        """Verify MCP initialize endpoint"""
        response = requests.post(
            f"{MCP_BASE}/mcp/initialize",
            json={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert "capabilities" in result

    def test_mcp_tools_list(self):
        """Verify tools list endpoint"""
        response = requests.get(f"{MCP_BASE}/mcp/tools/list", timeout=5)
        assert response.status_code == 200
        result = response.json()
        assert "tools" in result
        tool_names = [t["name"] for t in result["tools"]]
        assert "search_documentation" in tool_names

    def test_mcp_tools_call(self):
        """Verify tools call endpoint"""
        response = requests.post(
            f"{MCP_BASE}/mcp/tools/call",
            json={
                "name": "search_documentation",
                "arguments": {"query": "test query"},
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert "content" in result


# ============================================================================
# Suite 3: Performance Benchmarks
# ============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance under load"""

    def test_p95_latency_under_500ms(self):
        """P95 latency < 500ms target"""
        latencies = []

        def request():
            start = time.time()
            try:
                r = requests.get(f"{MCP_BASE}/health", timeout=5)
                if r.status_code == 200:
                    return (time.time() - start) * 1000
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=10) as ex:
            results = [ex.submit(request) for _ in range(100)]
            for f in as_completed(results):
                lat = f.result()
                if lat:
                    latencies.append(lat)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        print(f"\nP95: {p95:.1f}ms")
        assert p95 < 500

    def test_concurrent_operations(self):
        """Concurrent operations maintain >95% success rate"""

        def call():
            try:
                r = requests.post(
                    f"{MCP_BASE}/mcp/tools/call",
                    json={
                        "name": "search_documentation",
                        "arguments": {"query": "test"},
                    },
                    timeout=10,
                )
                return r.status_code == 200
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=20) as ex:
            results = [ex.submit(call) for _ in range(20)]
            success = sum([f.result() for f in as_completed(results)])

        rate = success / 20
        print(f"\nSuccess rate: {rate * 100:.1f}%")
        assert rate >= 0.95


# ============================================================================
# Suite 4: Security
# ============================================================================


class TestSecurity:
    """Security controls"""

    def test_injection_blocked(self):
        """Injection attempts blocked"""
        from src.mcp_server.validation import CypherValidator, ValidationError

        validator = CypherValidator()
        attacks = ["'; DROP DATABASE;", "MATCH (n) DELETE n"]

        for attack in attacks:
            with pytest.raises(ValidationError):
                validator.validate(attack, {})

    def test_rate_limiter_configured(self):
        """Rate limiting configured"""
        from src.mcp_server.security.rate_limiter import RateLimiter

        limiter = RateLimiter()
        assert limiter.enabled is not None
        assert limiter.burst_size > 0


# ============================================================================
# Suite 5: Chaos Tests
# ============================================================================


@pytest.mark.chaos
class TestChaos:
    """Chaos engineering scenarios"""

    def test_qdrant_down(self):
        """System degrades gracefully when Qdrant is down"""
        print("\n[CHAOS] Stopping Qdrant...")
        subprocess.run(
            ["docker", "stop", "weka-qdrant"], capture_output=True, timeout=30
        )
        time.sleep(3)

        try:
            # Verify Qdrant is down
            try:
                r = requests.get(f"{QDRANT_URL}/health", timeout=2)
                if r.status_code == 200:
                    pytest.fail("Qdrant should be down")
            except Exception:
                pass

            # MCP should still work
            r = requests.get(f"{MCP_BASE}/health", timeout=5)
            assert r.status_code == 200

            r = requests.get(f"{MCP_BASE}/mcp/tools/list", timeout=5)
            assert r.status_code == 200

            print("[CHAOS] Degraded gracefully ✓")

        finally:
            print("[CHAOS] Restarting Qdrant...")
            subprocess.run(
                ["docker", "start", "weka-qdrant"], capture_output=True, timeout=30
            )
            time.sleep(5)

    def test_redis_down(self):
        """System operates without Redis (L1 cache only)"""
        print("\n[CHAOS] Stopping Redis...")
        subprocess.run(
            ["docker", "stop", "weka-redis"], capture_output=True, timeout=30
        )
        time.sleep(3)

        try:
            # MCP should still work
            r = requests.get(f"{MCP_BASE}/health", timeout=5)
            assert r.status_code == 200

            r = requests.get(f"{MCP_BASE}/mcp/tools/list", timeout=5)
            assert r.status_code == 200

            print("[CHAOS] L1-only operation ✓")

        finally:
            print("[CHAOS] Restarting Redis...")
            subprocess.run(
                ["docker", "start", "weka-redis"], capture_output=True, timeout=30
            )
            time.sleep(5)

    def test_neo4j_backpressure(self):
        """System handles Neo4j backpressure"""
        print("\n[CHAOS] Creating Neo4j load...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

        def load():
            try:
                with driver.session(database="neo4j") as s:
                    s.run("UNWIND range(1, 50000) AS x RETURN x").consume()
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(load) for _ in range(10)]
            time.sleep(2)

            # MCP should remain responsive
            r = requests.get(f"{MCP_BASE}/health", timeout=5)
            assert r.status_code == 200

            for f in futures:
                try:
                    f.result(timeout=20)
                except Exception:
                    pass

        driver.close()
        print("[CHAOS] Backpressure handled ✓")
