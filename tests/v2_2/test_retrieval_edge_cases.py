"""Retrieval edge-case coverage for v2.2 hybrid retriever."""

import os
import uuid
from datetime import datetime

import pytest

from src.ingestion.build_graph import GraphBuilder
from src.query.hybrid_retrieval import HybridRetriever


def _builder(env):
    builder = GraphBuilder(env["driver"], env["config"], qdrant_client=env["qdrant"])
    builder.embedder = env["embedder"]
    return builder


def _doc(doc_tag_prefix: str, tenant: str, text: str, repeats: int = 200):
    doc_id = f"retrieval-{uuid.uuid4().hex[:8]}"
    doc_tag = f"{doc_tag_prefix}-{uuid.uuid4().hex[:4]}"
    now = datetime.utcnow().isoformat() + "Z"
    document = {
        "id": doc_id,
        "source_uri": f"memory://retrieval/{doc_id}.md",
        "source_type": "markdown",
        "title": f"Retrieval Fixture {doc_tag}",
        "version": "1.0",
        "checksum": doc_id,
        "last_edited": now,
        "doc_tag": doc_tag,
        "tenant": tenant,
        "lang": "en",
    }
    sections = [
        {
            "id": f"{doc_id}-s1",
            "title": f"{doc_tag} Overview",
            "text": " ".join([text] * repeats),
            "tokens": repeats,
            "level": 1,
            "order": 0,
        },
        {
            "id": f"{doc_id}-s2",
            "title": f"{doc_tag} Deep Dive",
            "text": " ".join([text + " deep"] * repeats),
            "tokens": repeats,
            "level": 1,
            "order": 1,
        },
    ]
    return document, sections


@pytest.fixture(scope="module")
def seeded_docs(integration_env):
    builder = _builder(integration_env)
    documents = []
    payloads = [
        ("alpha", "tenant-alpha", "Alpha fabrics emphasize deterministic routing."),
        (
            "beta",
            "tenant-beta",
            "Beta fabrics rely on adaptive routing for resilience.",
        ),
    ]
    for prefix, tenant, text in payloads:
        document, sections = _doc(prefix, tenant, text)
        builder.upsert_document(document, sections, entities={}, mentions=[])
        documents.append({"document": document, "sections": sections})
    return documents


@pytest.mark.integration
def test_doc_tag_filter_limits_results(integration_env, seeded_docs):
    retriever = HybridRetriever(
        neo4j_driver=integration_env["driver"],
        qdrant_client=integration_env["qdrant"],
        embedder=integration_env["embedder"],
        tokenizer=integration_env["tokenizer"],
    )
    alpha_doc = next(
        item for item in seeded_docs if item["document"]["doc_tag"].startswith("alpha")
    )
    query = "How does the alpha fabric behave?"
    results, metrics = retriever.retrieve(
        query=query,
        top_k=5,
        filters={"doc_tag": alpha_doc["document"]["doc_tag"]},
    )

    assert results, "Expected results for alpha query"
    assert all(chunk.doc_tag == alpha_doc["document"]["doc_tag"] for chunk in results)
    assert metrics["seed_gated"] >= 0
    assert metrics["microdoc_present"] in (0, 1)


@pytest.mark.integration
def test_forced_expansion_adds_neighbors(integration_env, seeded_docs):
    retriever = HybridRetriever(
        neo4j_driver=integration_env["driver"],
        qdrant_client=integration_env["qdrant"],
        embedder=integration_env["embedder"],
        tokenizer=integration_env["tokenizer"],
    )
    beta_doc = next(
        item for item in seeded_docs if item["document"]["doc_tag"].startswith("beta")
    )
    query = "Describe beta fabric routing."
    results, metrics = retriever.retrieve(
        query=query,
        top_k=1,
        expand=True,
        expand_when="always",
        filters={"doc_tag": beta_doc["document"]["doc_tag"]},
    )

    assert metrics["expansion_count"] >= 1
    with integration_env["driver"].session() as session:
        neighbor_count = session.run(
            """
            MATCH (d:Document {doc_tag: $tag})-[:HAS_SECTION]->(c:Section)-[:NEXT_CHUNK]->(:Chunk)
            RETURN count(*) AS neighbors
            """,
            tag=beta_doc["document"]["doc_tag"],
        ).single()["neighbors"]
    assert (
        neighbor_count >= 1
    ), "Expected at least one NEXT_CHUNK neighbor for beta document"
    seed_map = {chunk.chunk_id: chunk for chunk in results if not chunk.is_expanded}
    for chunk in results:
        if not chunk.is_expanded:
            continue
        assert chunk.expansion_source
        assert chunk.expansion_source in seed_map
        assert chunk.document_id == seed_map[chunk.expansion_source].document_id


@pytest.fixture(autouse=True)
def disable_doc_fallback(monkeypatch):
    original = os.getenv("COMBINE_DOC_FALLBACK_ENABLED")
    monkeypatch.setenv("COMBINE_DOC_FALLBACK_ENABLED", "false")
    yield
    if original is not None:
        monkeypatch.setenv("COMBINE_DOC_FALLBACK_ENABLED", original)
    else:
        monkeypatch.delenv("COMBINE_DOC_FALLBACK_ENABLED", raising=False)
