"""E2E v2.2 – Hybrid retrieval checks (spec-only).

Validates that HybridRetriever returns fused results with metrics, respects the token
budget, and excludes stubs. Optionally, if a small goldset is provided, compute
Recall@k uplift.
"""

from __future__ import annotations

import os
from typing import List

import pytest
from qdrant_client import QdrantClient

from src.providers.factory import ProviderFactory
from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import HybridRetriever
from src.shared.connections import get_connection_manager

pytestmark = pytest.mark.integration


def _extract_titles(
    qdrant: QdrantClient, snapshot_scope: str, limit: int = 10
) -> List[str]:
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[
            FieldCondition(key="snapshot_scope", match=MatchValue(value=snapshot_scope))
        ]
    )
    points, _ = qdrant.scroll(
        collection_name="chunks_multi",
        limit=200,
        with_payload=True,
        with_vectors=False,
        scroll_filter=flt,
    )
    titles: List[str] = []
    for p in points or []:
        payload = p.payload or {}
        h = (payload.get("heading") or "").strip()
        if h and h not in titles:
            titles.append(h)
        if len(titles) >= limit:
            break
    # Fallback if no headings
    return titles or ["installation", "configuration", "troubleshooting"]


def test_hybrid_retrieval_returns_results(prod_env, capture_logs):
    capture_logs("retrieval-before")
    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()
    qdrant = manager.get_qdrant_client()

    # Use live embedder via provider factory (respects Jina env configuration)
    factory = ProviderFactory()
    embed_provider = (
        os.environ.get("EMBEDDINGS_PROVIDER") or prod_env["config"].embedding.provider
    )
    if embed_provider == "jina-ai" and not os.getenv("JINA_API_KEY"):
        pytest.skip("JINA_API_KEY required for jina-ai provider in this test")
    embedder = factory.create_embedding_provider()

    retriever = HybridRetriever(
        neo4j_driver=driver,
        qdrant_client=qdrant,
        embedder=embedder,
        tokenizer=TokenizerService(),
    )

    snapshot_scope: str = prod_env["snapshot_scope"]
    queries = _extract_titles(qdrant, snapshot_scope, limit=5)

    for q in queries:
        results, metrics = retriever.retrieve(
            query=q,
            top_k=20,
            filters={"snapshot_scope": snapshot_scope},
        )
        assert results, f"No results for query {q!r}"
        # Required metrics
        for k in (
            "seed_count",
            "fusion_method",
            "vector_fields",
            "microdoc_present",
            "microdoc_used",
            "total_tokens",
        ):
            assert k in metrics, f"Missing metric: {k}"
        assert metrics["total_tokens"] <= 4500, "Token budget exceeded"
        # No stubs in final results
        assert all(not getattr(r, "is_microdoc_stub", False) for r in results)
    capture_logs("retrieval-after")
