"""E2E v2.2 – Graph & vector alignment checks (spec-only)."""

from __future__ import annotations

from typing import Dict, List

import pytest
from neo4j import Driver
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration


def _list_sample_points(
    qdrant: QdrantClient, snapshot_scope: str, limit: int = 50
) -> List[Dict]:
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[
            FieldCondition(key="snapshot_scope", match=MatchValue(value=snapshot_scope))
        ]
    )
    points, _ = qdrant.scroll(
        collection_name="chunks_multi",
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=flt,
    )
    out: List[Dict] = []
    for p in points or []:
        payload = p.payload or {}
        out.append({"id": str(p.id), "payload": payload})
    return out


def _run_cypher(driver: Driver, query: str, params: Dict | None = None):
    with driver.session() as sess:
        return list(sess.run(query, params or {}))


def test_zero_orphans_and_adjacency(prod_env):
    driver: Driver = prod_env["neo4j"]
    qdrant: QdrantClient = prod_env["qdrant"]
    snapshot_scope: str = prod_env["snapshot_scope"]

    points = _list_sample_points(qdrant, snapshot_scope, limit=100)
    assert points, "No points found for snapshot_scope"

    # 1) Zero orphans: every point must have a Chunk node
    for p in points:
        payload = p["payload"]
        node_id = payload.get("id") or payload.get("node_id")
        did = payload.get("document_id") or payload.get("doc_id")
        rows = _run_cypher(
            driver,
            """
            MATCH (c:Chunk {id: $node_id})-[:NEXT_CHUNK*0..]->(c2:Chunk)
            WHERE coalesce(c.document_id, c.doc_id) = $did
            RETURN count(c) as c
            """,
            {"node_id": node_id, "did": did},
        )
        assert rows and rows[0]["c"] >= 1, f"Orphan chunk (node_id={node_id})"

    # 2) Adjacency chain not empty per document
    rows = _run_cypher(
        driver,
        """
        MATCH (d:Document {snapshot_scope: $scope})-[:HAS_SECTION*]->(:Section)-[:NEXT_CHUNK]->(:Chunk)
        RETURN count(*) as adj
        """,
        {"scope": snapshot_scope},
    )
    assert (
        rows and rows[0]["adj"] >= 1
    ), "Expected some NEXT_CHUNK edges for snapshot_scope"
