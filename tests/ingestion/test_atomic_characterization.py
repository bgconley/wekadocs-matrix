"""Offline behavior pin for AtomicIngestionCoordinator orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.atomic import AtomicIngestionCoordinator
from src.shared.config import get_config


class DummyDriver:
    def session(self, **_):
        raise AssertionError("offline pin must not open Neo4j sessions")


class DummyQdrant:
    pass


def test_atomic_ingestion_orchestrates_prepare_embed_saga_and_result_contract(
    monkeypatch,
):
    config = get_config().model_copy(deep=True)
    coordinator = AtomicIngestionCoordinator(
        DummyDriver(),
        DummyQdrant(),
        config,
        validate_before_commit=False,
    )
    calls = []

    document = {
        "id": "doc-nutanix-files",
        "doc_id": "doc-nutanix-files",
        "title": "Nutanix Files snapshots",
        "total_tokens": 12,
    }
    sections = [
        {
            "id": "chunk-1",
            "document_id": "doc-nutanix-files",
            "text": "Nutanix Files supports NFS and SMB shares.",
            "token_count": 12,
            "original_section_ids": ["source-section-1"],
            "_mentions": [{"entity_id": "gliner-files", "source": "gliner"}],
        }
    ]
    entities = {
        "struct-files": {"id": "struct-files", "name": "Nutanix Files"},
        "gliner-files": {"id": "gliner-files", "name": "Nutanix Files"},
    }
    mentions = [
        {
            "section_id": "source-section-1",
            "entity_id": "struct-files",
            "source": "structural",
        }
    ]
    references = [{"source_chunk_id": "chunk-1", "target_hint": "Prism Central"}]
    builder = SimpleNamespace(name="fake-builder")

    def fake_prepare(
        source_uri,
        content,
        fmt,
        *,
        embedding_model=None,
        embedding_version=None,
    ):
        calls.append(("prepare", source_uri, fmt, embedding_model, embedding_version))
        assert "Nutanix Files" in content
        return {
            "document": document,
            "sections": sections,
            "entities": entities,
            "mentions": mentions,
            "references": references,
            "builder": builder,
        }

    def fake_compute(doc, stage_sections, stage_entities, stage_builder):
        calls.append(("embed", doc["id"], [s["id"] for s in stage_sections]))
        assert stage_builder is builder
        assert stage_entities is entities
        merged_ids = {m["entity_id"] for m in stage_sections[0]["_mentions"]}
        assert merged_ids == {"gliner-files", "struct-files"}
        return {
            "sections": {
                "chunk-1": {
                    "content": [0.1, 0.2, 0.3],
                    "sparse": {"indices": [1], "values": [0.5]},
                    "colbert": [[0.1, 0.2]],
                }
            },
            "stats": {"sparse_coverage": 1.0},
        }

    def fake_saga(**kwargs):
        calls.append(("saga", kwargs["document"]["id"], len(kwargs["sections"])))
        assert kwargs["references"] is references
        assert kwargs["embeddings"]["sections"]["chunk-1"]["content"] == [
            0.1,
            0.2,
            0.3,
        ]
        return {
            "success": True,
            "stats": {
                "sections_upserted": 1,
                "entities_upserted": 2,
                "vectors_upserted": 1,
                "cross_doc_edges": 0,
            },
        }

    monkeypatch.setattr(coordinator, "_prepare_ingestion", fake_prepare)
    monkeypatch.setattr(coordinator, "_compute_embeddings", fake_compute)
    monkeypatch.setattr(coordinator, "_execute_atomic_saga", fake_saga)

    result = coordinator.ingest_document_atomic(
        "nutanixdocs://files/snapshots",
        "# Nutanix Files\n\nNFS and SMB snapshot workflow.",
        embedding_model="test-model",
        embedding_version="test-version",
    )

    assert result.success is True
    assert result.document_id == "doc-nutanix-files"
    assert result.neo4j_committed is True
    assert result.qdrant_committed is True
    assert result.stats["sections_upserted"] == 1
    assert result.stats["vectors_upserted"] == 1
    assert calls == [
        (
            "prepare",
            "nutanixdocs://files/snapshots",
            "markdown",
            "test-model",
            "test-version",
        ),
        ("embed", "doc-nutanix-files", ["chunk-1"]),
        ("saga", "doc-nutanix-files", 1),
    ]
