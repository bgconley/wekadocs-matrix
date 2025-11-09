"""Tests covering Qdrant/Neo4j reconciliation drift scenarios."""

import uuid
from datetime import datetime

import pytest
from qdrant_client.models import PointStruct

from src.ingestion.build_graph import GraphBuilder
from src.ingestion.reconcile import Reconciler


def _builder(env):
    builder = GraphBuilder(env["driver"], env["config"], qdrant_client=env["qdrant"])
    builder.embedder = env["embedder"]
    return builder


def _ingest_document(env, doc_tag: str):
    doc_id = f"recon-{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat() + "Z"
    document = {
        "id": doc_id,
        "source_uri": f"memory://recon/{doc_id}.md",
        "source_type": "markdown",
        "title": f"Recon Fixture {doc_tag}",
        "version": "1.0",
        "checksum": doc_id,
        "last_edited": now,
        "doc_tag": doc_tag,
        "tenant": doc_tag,
        "lang": "en",
    }
    sections = [
        {
            "id": f"{doc_id}-s1",
            "title": "Recon Section",
            "text": "This section is used for reconciliation drift tests.",
            "tokens": 220,
            "level": 2,
            "order": 0,
        }
    ]
    builder = _builder(env)
    builder.upsert_document(document, sections, entities={}, mentions=[])
    return doc_id


def _collection(env) -> str:
    return env["config"].search.vector.qdrant.collection_name


@pytest.mark.integration
def test_reconciler_backfills_missing_points(integration_env):
    doc_id = _ingest_document(integration_env, "recon-missing")
    qdrant = integration_env["qdrant"]
    collection = _collection(integration_env)
    qdrant.delete(
        collection_name=collection,
        points_selector={
            "filter": {
                "must": [
                    {"key": "document_id", "match": {"value": doc_id}},
                    {"key": "node_label", "match": {"value": "Section"}},
                ]
            }
        },
        wait=True,
    )

    reconciler = Reconciler(
        integration_env["driver"],
        integration_env["config"],
        qdrant_client=integration_env["qdrant"],
    )
    embed_fn = integration_env["embedder"].embed_query
    stats = reconciler.reconcile_sync(embedding_fn=embed_fn)
    assert stats.missing_backfilled >= 1
    assert stats.graph_count == stats.vector_count


@pytest.mark.integration
def test_reconciler_removes_orphan_points(integration_env):
    doc_id = _ingest_document(integration_env, "recon-orphan")
    env = integration_env
    qdrant = env["qdrant"]
    collection = _collection(env)
    dims = env["config"].embedding.dims
    zero_vec = [0.0] * dims
    payload = {
        "node_id": f"orphan-{uuid.uuid4().hex[:6]}",
        "node_label": "Section",
        "document_id": f"orphan-{doc_id}",
        "doc_tag": "orphan",
        "embedding_version": env["config"].embedding.version,
    }
    qdrant.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"content": zero_vec, "title": zero_vec},
                payload=payload,
            )
        ],
        wait=True,
    )

    reconciler = Reconciler(env["driver"], env["config"], qdrant_client=qdrant)
    stats = reconciler.reconcile_sync(embedding_fn=env["embedder"].embed_query)
    assert stats.extras_removed >= 1
    assert stats.graph_count == stats.vector_count
