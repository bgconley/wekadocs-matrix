# tests/p4_t1_complex_patterns_test.py
import os
import uuid

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


TEST_NS = f"p4_{uuid.uuid4().hex[:8]}"


def _seed_advanced_graph(s):
    # Components / dependency chain
    s.run(
        """
    MERGE (a:Component {id:$a, name:'Comp A'})
    MERGE (b:Component {id:$b, name:'Comp B'})
    MERGE (c:Component {id:$c, name:'Comp C'})
    MERGE (a)-[:DEPENDS_ON]->(b)
    MERGE (b)-[:DEPENDS_ON]->(c)
    """,
        a=f"{TEST_NS}:compA",
        b=f"{TEST_NS}:compB",
        c=f"{TEST_NS}:compC",
    )

    # Configuration -> AFFECTS -> Components
    s.run(
        """
    MERGE (cfg:Configuration {id:$cfg, name:'fsync', scope:'cluster', description:'Controls fsync'})
    MERGE (a:Component {id:$a})
    MERGE (cfg)-[:AFFECTS]->(a)
    """,
        cfg=f"{TEST_NS}:cfg1",
        a=f"{TEST_NS}:compA",
    )

    # Troubleshooting path: Error <-RESOLVES- Procedure -CONTAINS_STEP-> Step -EXECUTES-> Command
    s.run(
        """
    MERGE (e:Error {id:$e, code:'E-42', message:'Disk warning', severity:'WARN'})
    MERGE (p:Procedure {id:$p, title:'Fix E-42', description:'Resolve disk warnings'})
    MERGE (p)-[:RESOLVES]->(e)
    MERGE (s1:Step {id:$s1, order:1, instruction:'Run diag'})
    MERGE (s2:Step {id:$s2, order:2, instruction:'Apply fix'})
    MERGE (p)-[:CONTAINS_STEP {order:1}]->(s1)
    MERGE (p)-[:CONTAINS_STEP {order:2}]->(s2)
    MERGE (cmd:Command {id:$cmd, name:'weka diag', cli_syntax:'weka diag'})
    MERGE (s1)-[:EXECUTES]->(cmd)
    """,
        e=f"{TEST_NS}:err",
        p=f"{TEST_NS}:proc",
        s1=f"{TEST_NS}:step1",
        s2=f"{TEST_NS}:step2",
        cmd=f"{TEST_NS}:cmd1",
    )


@pytest.mark.order(1)
def test_dependency_chain_query(neo4j_driver):
    with neo4j_driver.session() as s:
        _seed_advanced_graph(s)
        rec = s.run(
            """
            MATCH path = (start:Component {name:'Comp A'})-[:DEPENDS_ON*1..5]->(end)
            RETURN length(path) AS depth
            LIMIT 1
        """
        ).single()
        assert (
            rec is not None and rec["depth"] >= 2
        ), "Expected multi-hop DEPENDS_ON chain (>=2)."


@pytest.mark.order(2)
def test_impact_assessment_query(neo4j_driver):
    with neo4j_driver.session() as s:
        rec = s.run(
            """
            MATCH (config:Configuration {name:'fsync'})
            MATCH (config)-[:AFFECTS*1..3]->(affected)
            OPTIONAL MATCH (affected)-[:CRITICAL_FOR]->(service)
            RETURN config.name AS config, collect(DISTINCT labels(affected)) AS affected_labels,
                   CASE WHEN service IS NOT NULL THEN 'CRITICAL' ELSE 'NORMAL' END AS impact_level
            LIMIT 1
        """
        ).single()
        assert rec and rec["config"] == "fsync", "Config node not found."
        assert rec["impact_level"] in ("CRITICAL", "NORMAL")
        assert any(
            "Component" in labs for labs in rec["affected_labels"]
        ), "Expected affected components."


@pytest.mark.order(3)
def test_troubleshooting_path_query(neo4j_driver):
    with neo4j_driver.session() as s:
        rec = s.run(
            """
            MATCH (e:Error {code:'E-42'})
            MATCH (p:Procedure)-[:RESOLVES]->(e)
            MATCH (p)-[:CONTAINS_STEP*]->(step:Step)
            OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
            WITH e, p, step, cmd
            ORDER BY step.order
            RETURN e.code AS code, p.title AS proc,
                   collect({order:step.order, cmd:cmd.cli_syntax}) AS steps
        """
        ).single()
        assert rec and rec["code"] == "E-42"
        steps = rec["steps"]
        assert len(steps) >= 2
        assert steps[0]["order"] == 1, "Steps must be in order."
        assert steps[0]["cmd"] == "weka diag", "First step should execute 'weka diag'."
