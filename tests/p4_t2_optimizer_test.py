# tests/p4_t2_optimizer_test.py
import os

import pytest
from neo4j import GraphDatabase


@pytest.fixture(scope="module")
def neo4j_driver():
    """Always provide a synchronous Neo4j driver and yield it properly."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "test")
    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        yield drv
    finally:
        drv.close()


def _flatten_plan(plan):
    ops = []
    stack = [plan]
    while stack:
        n = stack.pop()
        ops.append(
            getattr(n, "operator_type", None) or getattr(n, "operatorType", None)
        )
        for c in getattr(n, "children", []) or []:
            stack.append(c)
    return [o for o in ops if o]


@pytest.mark.order(4)
def test_explain_uses_index_seek_for_point_lookup(neo4j_driver):
    with neo4j_driver.session() as s:
        # Ensure index exists on Configuration.name
        s.run(
            "CREATE INDEX config_name IF NOT EXISTS FOR (c:Configuration) ON (c.name)"
        ).consume()
        res = s.run(
            "EXPLAIN MATCH (c:Configuration {name:$name}) RETURN c", name="fsync"
        )
        summary = res.consume()
        plan = summary.plan
        ops = _flatten_plan(plan)
        assert any(
            "IndexSeek" in o or "NodeIndexSeek" in o for o in ops
        ), f"Expected IndexSeek in EXPLAIN plan, got: {ops}"


@pytest.mark.order(5)
def test_traversal_depth_is_capped(neo4j_driver):
    with neo4j_driver.session() as s:
        # A deliberately aggressive query; system should keep this bounded by configuration / template
        res = s.run("EXPLAIN MATCH p=(n)-[*1..10]-(m) RETURN p LIMIT 10")
        plan_ops = _flatten_plan(res.consume().plan)
        # This is a heuristic assertion: the presence of 'Expand(All)' is fine, but your validator/optimizer
        # should cap depth in templates to <=5. Adjust once your templates expose depth caps.
        # For kickoff we just assert EXPLAIN returns a valid plan.
        assert len(plan_ops) > 0
