"""Integration tests covering ingestion edge cases for v2.2."""

import uuid
from datetime import datetime

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.ingestion.build_graph import GraphBuilder


def _scroll_chunks(qdrant, collection: str, document_id: str):
    flt = Filter(
        must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="node_label", match=MatchValue(value="Section")),
        ]
    )
    points, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=flt,
        with_payload=True,
        with_vectors=False,
        limit=64,
    )
    return [pt.payload for pt in points]


def _builder(env):
    builder = GraphBuilder(env["driver"], env["config"], qdrant_client=env["qdrant"])
    builder.embedder = env["embedder"]
    return builder


def _doc(meta_suffix: str, doc_tag: str, tenant: str, lang: str = "en"):
    doc_id = f"edge-{meta_suffix}-{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat() + "Z"
    document = {
        "id": doc_id,
        "source_uri": f"memory://edge/{doc_id}.md",
        "source_type": "markdown",
        "title": f"Edge Case {meta_suffix}",
        "version": "1.0",
        "checksum": doc_id,
        "last_edited": now,
        "doc_tag": doc_tag,
        "tenant": tenant,
        "lang": lang,
    }
    return doc_id, document


@pytest.mark.integration
def test_microdoc_with_missing_heading_sets_metadata(integration_env):
    doc_id, document = _doc("micro", "edge-tenant", "edge-tenant")
    sections = [
        {
            "id": f"{doc_id}-s1",
            "title": None,
            "text": "Short paragraph one.",
            "tokens": 18,
            "level": 2,
            "order": 0,
        },
        {
            "id": f"{doc_id}-s2",
            "title": "Second",
            "text": "Another short section.",
            "tokens": 22,
            "level": 2,
            "order": 1,
        },
    ]
    builder = _builder(integration_env)
    builder.upsert_document(document, sections, entities={}, mentions=[])

    payloads = _scroll_chunks(
        integration_env["qdrant"],
        integration_env["config"].search.vector.qdrant.collection_name,
        document_id=doc_id,
    )
    assert payloads, "Expected microdoc chunks"
    for payload in payloads:
        assert payload["doc_tag"] == "edge-tenant"
        assert payload["tenant"] == "edge-tenant"
        assert payload["is_microdoc"] is True
        assert payload.get("doc_is_microdoc") is True
        assert payload.get("heading") is not None
        assert payload.get("text_hash")
        assert payload.get("shingle_hash")


@pytest.mark.integration
def test_large_section_split_preserves_split_flags(integration_env):
    doc_id, document = _doc("split", "edge-large", "edge-large")
    sections = [
        {
            "id": f"{doc_id}-s1",
            "title": "Massive",
            "text": " ".join(["long-section"] * 9000),
            "tokens": 9000,
            "level": 2,
            "order": 0,
        }
    ]
    builder = _builder(integration_env)
    stats = builder.upsert_document(document, sections, entities={}, mentions=[])
    assert stats["embeddings_computed"] >= 1

    payloads = _scroll_chunks(
        integration_env["qdrant"],
        integration_env["config"].search.vector.qdrant.collection_name,
        document_id=doc_id,
    )
    assert len(payloads) >= 2, "Expected section split into multiple chunks"
    assert any(p.get("is_split") for p in payloads)
    assert all(
        p.get("semantic_metadata") == {"entities": [], "topics": []} for p in payloads
    )
    assert all(not p.get("doc_is_microdoc", False) for p in payloads)
