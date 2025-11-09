"""Hybrid RAG v2.2 integration tests (sections 8-9 deliverables).

These tests operate against live Neo4j, Qdrant, and Redis instances using the
canonical multi-vector ingestion + retrieval flow described in
`docs/hybrid-rag-v2_2-spec.md` and the context notes.
"""

import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.ingestion.build_graph import GraphBuilder
from src.query.hybrid_retrieval import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_source_document(doc_id: str) -> Tuple[Dict, List[Dict]]:
    now = datetime.utcnow().isoformat() + "Z"
    document = {
        "id": doc_id,
        "source_uri": f"memory://integration/{doc_id}.md",
        "source_type": "markdown",
        "title": "Hybrid v2.2 Integration Fixture",
        "version": "1.0",
        "checksum": doc_id,
        "last_edited": now,
        "doc_tag": f"integration-{doc_id}",
        "tenant": "test-tenant",
        "lang": "en",
    }
    sections = [
        {
            "id": f"{doc_id}-s1",
            "title": "Networking Configuration Overview",
            "text": (
                "Network configuration involves fabric zoning, IP allocations, and "
                "interface orchestration for storage clusters."
            ),
            "tokens": 160,
            "level": 2,
            "order": 0,
        },
        {
            "id": f"{doc_id}-s2",
            "title": "Multi-Vector Retrieval Guardrails",
            "text": (
                "Multi-vector ANN queries combine content, title, and entity vectors to "
                "preserve intent while respecting BM25 fallback semantics."
            ),
            "tokens": 140,
            "level": 2,
            "order": 1,
        },
        {
            "id": f"{doc_id}-s3",
            "title": "Cleanup and Reconciliation",
            "text": (
                "The cleanup workflow preserves SchemaVersion nodes while removing "
                "document data and verifying Qdrant payload parity."
            ),
            "tokens": 150,
            "level": 2,
            "order": 2,
        },
    ]
    return document, sections


def _scroll_qdrant_chunks(qdrant_client, collection: str, document_id: str):
    scroll_filter = Filter(
        must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="node_label", match=MatchValue(value="Section")),
        ]
    )
    points, _ = qdrant_client.scroll(
        collection_name=collection,
        scroll_filter=scroll_filter,
        with_payload=True,
        with_vectors=True,
        limit=32,
    )
    return points


@pytest.fixture(scope="session")
def ingested_document(integration_env):
    document_id = f"doc-v22-{uuid.uuid4().hex[:12]}"
    document, sections = build_source_document(document_id)
    builder = GraphBuilder(
        integration_env["driver"],
        integration_env["config"],
        qdrant_client=integration_env["qdrant"],
    )
    builder.embedder = integration_env["embedder"]
    stats = builder.upsert_document(document, sections, entities={}, mentions=[])
    return {
        "document": document,
        "sections": sections,
        "stats": stats,
    }


@pytest.mark.integration
def test_ingestion_populates_multi_vector_payloads(ingested_document, integration_env):
    doc_id = ingested_document["document"]["id"]
    points = _scroll_qdrant_chunks(
        integration_env["qdrant"],
        integration_env["config"].search.vector.qdrant.collection_name,
        doc_id,
    )

    assert points, "Expected at least one chunk in chunks_multi"

    expected_tag = ingested_document["document"]["doc_tag"]
    expected_tenant = ingested_document["document"]["tenant"]
    total_original_sections = 0
    for point in points:
        payload = point.payload or {}
        original_ids = payload.get("original_section_ids") or []
        total_original_sections += len(original_ids)

        vectors = point.vector or {}
        assert {"content", "title"}.issubset(set(vectors.keys()))
        if "entity" in vectors:
            assert len(vectors["entity"]) == integration_env["embedder"].dims
        assert payload.get("node_label") == "Section"
        assert payload.get("doc_tag") == expected_tag
        assert payload.get("tenant") == expected_tenant
        assert (
            payload.get("embedding_version")
            == integration_env["config"].embedding.version
        )
        assert payload.get("semantic_metadata") == {"entities": [], "topics": []}
        assert payload.get("text_hash")

    assert total_original_sections == len(ingested_document["sections"])


@pytest.mark.integration
def test_hybrid_retrieval_returns_multi_vector_results(
    ingested_document, integration_env
):
    retriever = HybridRetriever(
        neo4j_driver=integration_env["driver"],
        qdrant_client=integration_env["qdrant"],
        embedder=integration_env["embedder"],
        tokenizer=integration_env["tokenizer"],
    )

    query = "How does the multi-vector ANN retriever combine content and title signals?"
    results, metrics = retriever.retrieve(
        query=query,
        top_k=5,
        filters={"doc_tag": ingested_document["document"]["doc_tag"]},
    )

    assert results, "Hybrid retriever should return fused chunks"
    assert metrics["vector_fields"] == list(
        integration_env["config"].search.hybrid.vector_fields.keys()
    )
    doc_ids = {chunk.document_id for chunk in results}
    assert ingested_document["document"]["id"] in doc_ids

    top_hit = results[0]
    assert top_hit.vector_score is not None
    assert top_hit.fused_score is not None
    assert top_hit.heading

    assert not results[0].is_microdoc_extra
    first_micro_idx = next(
        (idx for idx, chunk in enumerate(results) if chunk.is_microdoc_extra),
        len(results),
    )
    for chunk in results[:first_micro_idx]:
        assert chunk.vector_score is not None

    assert metrics["primary_count"] + metrics["microdoc_used"] == len(results)
    assert metrics["microdoc_used"] == sum(
        1 for chunk in results if chunk.is_microdoc_extra
    )


@pytest.mark.integration
def test_cleanup_script_generates_report(integration_env, tmp_path):
    report_dir = tmp_path / "cleanup-reports"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "cleanup-databases.py"),
        "--dry-run",
        "--report-dir",
        str(report_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr

    reports = sorted(report_dir.glob("cleanup-report-*.json"))
    assert reports, "Cleanup script should emit a JSON report"
    data = json.loads(reports[-1].read_text())
    assert data["summary"]["dry_run"] is True
    assert data["summary"]["metadata_preserved"] is True
