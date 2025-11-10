"""E2E v2.2 – Qdrant named vectors & payload conformance (spec-only)."""

from __future__ import annotations

import math
from typing import Dict

import pytest
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration


def _iter_points(qdrant: QdrantClient, snapshot_scope: str, limit: int = 500):
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[
            FieldCondition(key="snapshot_scope", match=MatchValue(value=snapshot_scope))
        ]
    )
    res = qdrant.scroll(
        collection_name="chunks_multi",
        limit=limit,
        with_payload=True,
        with_vectors=True,
        scroll_filter=flt,
    )
    points = res[0] if isinstance(res, tuple) else res
    return points or []


def _is_nan(v: float) -> bool:
    return math.isnan(v) or math.isinf(v)


def test_named_vectors_dims_and_sanity(prod_env):
    qdrant: QdrantClient = prod_env["qdrant"]
    snapshot_scope: str = prod_env["snapshot_scope"]

    points = _iter_points(qdrant, snapshot_scope, limit=200)
    assert points, "No points found for snapshot_scope"

    for p in points:
        vecs = getattr(p, "vector", None)
        if not isinstance(vecs, dict):
            continue
        for name in ("content", "title", "entity"):
            v = vecs.get(name)
            if v is None:
                # entity may be absent on some points; warn-only
                continue
            assert len(v) == 1024, f"{name} dim != 1024"
            assert not any(_is_nan(x) for x in v), f"NaN in {name} vector"


def test_payload_carries_required_fields(prod_env):
    qdrant: QdrantClient = prod_env["qdrant"]
    snapshot_scope: str = prod_env["snapshot_scope"]

    points = _iter_points(qdrant, snapshot_scope, limit=200)
    assert points, "No points found for snapshot_scope"

    required = {
        "document_id",
        "heading",
        "token_count",
        "text_hash",
        "doc_tag",
        "snapshot_scope",
    }
    for p in points:
        payload: Dict = p.payload or {}
        missing = [k for k in required if k not in payload]
        assert not missing, f"Missing payload keys: {missing}"
