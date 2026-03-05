"""
Integration tests for GDS readiness validation (Stage 8).

These tests require a live Neo4j instance and are skipped
in CI without infrastructure.

Run with: pytest tests/integration/test_gds_readiness.py -v
"""

import os

import pytest

# Skip all tests if NEO4J_URI not set (no infrastructure available)
pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="NEO4J_URI not set — requires live Neo4j for GDS readiness tests",
)


@pytest.fixture(scope="module")
def neo4j_driver():
    """Create a Neo4j driver for integration tests."""
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    yield driver
    driver.close()


class TestGDSReadinessGates:
    """Run the GDS readiness gates against a live Neo4j instance."""

    def test_gates_can_execute(self, neo4j_driver):
        """All gate queries execute without error (even if some fail)."""
        from scripts.validate_gds_readiness import run_all_gates

        gates = run_all_gates(neo4j_driver)
        assert len(gates) == 8, f"Expected 8 gates, got {len(gates)}"
        # All gates should execute (not error)
        for g in gates:
            assert "ERROR" not in g.message, f"Gate {g.name} errored: {g.message}"

    def test_metrics_can_execute(self, neo4j_driver):
        """All metric queries execute without error."""
        from scripts.validate_gds_readiness import run_metrics

        metrics = run_metrics(neo4j_driver)
        # Metrics may be empty if no RELATED_TO edges exist,
        # but they should not raise exceptions
        assert isinstance(metrics, list)

    def test_schema_version_gate(self, neo4j_driver):
        """Schema version gate returns a result (pass or fail)."""
        from scripts.validate_gds_readiness import run_all_gates

        gates = run_all_gates(neo4j_driver)
        schema_gate = next((g for g in gates if g.name == "schema_version"), None)
        assert schema_gate is not None, "schema_version gate not found"
        # The gate should have a clear pass/fail, not an error
        assert schema_gate.passed or "FAIL" in schema_gate.message

    def test_indexes_gate(self, neo4j_driver):
        """Index gate returns a result (pass or fail)."""
        from scripts.validate_gds_readiness import run_all_gates

        gates = run_all_gates(neo4j_driver)
        idx_gate = next((g for g in gates if g.name == "indexes_exist"), None)
        assert idx_gate is not None, "indexes_exist gate not found"
        assert idx_gate.passed or "FAIL" in idx_gate.message
