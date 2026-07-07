"""Functional tests for extracted ingestion stages."""

from __future__ import annotations

from types import SimpleNamespace

import src.ingestion.stages.embed as embed_stage
from src.ingestion.stages.chunk import assemble_chunks
from src.ingestion.stages.embed import compute_embeddings
from src.ingestion.stages.enrich import merge_section_mentions
from src.ingestion.stages.parse import parse_document
from src.ingestion.stages.write import execute_saga


def test_parse_document_applies_doc_metadata_and_embedding_overrides():
    parsed = parse_document(
        "file:///tmp/data/ingest/nutanix-platform/scope__files.md",
        "# Overview\n\nDocTag: explicit_tag\n\nNutanix Files content.",
        "markdown",
        embedding_model="stage-test-model",
        embedding_version="stage-test-version",
    )

    document = parsed["document"]
    sections = parsed["sections"]
    config = parsed["config"]

    assert document["doc_tag"] == "explicit_tag"
    assert document["doc_category"] is None
    assert document["snapshot_scope"] is None
    assert sections
    assert all(section["doc_tag"] == "explicit_tag" for section in sections)
    assert all(section["doc_category"] is None for section in sections)
    assert config.embedding.embedding_model == "stage-test-model"
    assert config.embedding.version == "stage-test-version"


def test_parse_document_uses_filename_scope_when_no_explicit_doc_tag():
    parsed = parse_document(
        "file:///tmp/data/ingest/nutanix-platform/scope__files.md",
        "# Overview\n\nNutanix Files content.",
        "markdown",
    )

    document = parsed["document"]

    assert document["doc_tag"] == "files"
    assert document["doc_category"] == "nutanix-platform"
    assert document["snapshot_scope"] == "scope"


def test_assemble_chunks_preserves_document_identity_and_token_totals():
    class FakeAssembler:
        def assemble(self, document_id, raw_sections):
            assert document_id == "doc-nutanix-files"
            assert [section["id"] for section in raw_sections] == ["source-section-1"]
            return [
                {"id": "chunk-1", "token_count": 7},
                {"id": "chunk-2", "token_count": 5, "document_id": "existing-doc"},
            ]

    document = {"id": "doc-nutanix-files", "title": "Nutanix Files"}
    sections = [{"id": "source-section-1", "text": "Nutanix Files"}]
    config = SimpleNamespace(
        ingestion=SimpleNamespace(chunk_assembly=SimpleNamespace(assembler="fake"))
    )

    assembled = assemble_chunks(document, sections, config, assembler=FakeAssembler())

    assert assembled == [
        {
            "id": "chunk-1",
            "token_count": 7,
            "document_id": "doc-nutanix-files",
            "doc_id": "doc-nutanix-files",
            "document_total_tokens": 12,
        },
        {
            "id": "chunk-2",
            "token_count": 5,
            "document_id": "existing-doc",
            "doc_id": "doc-nutanix-files",
            "document_total_tokens": 12,
        },
    ]
    assert document["total_tokens"] == 12
    assert document["doc_id"] == "doc-nutanix-files"


def test_merge_section_mentions_preserves_gliner_and_filters_structural_noise():
    sections = [
        {
            "id": "chunk-1",
            "original_section_ids": ["source-section-1"],
            "_mentions": [{"entity_id": "gliner-files", "source": "gliner"}],
        }
    ]
    entities = {
        "struct-files": {"id": "struct-files", "name": "Nutanix Files"},
        "gliner-files": {"id": "gliner-files", "name": "Nutanix Files"},
        "noise-data": {"id": "noise-data", "name": "data"},
    }
    mentions = [
        {"section_id": "source-section-1", "entity_id": "struct-files"},
        {"section_id": "source-section-1", "entity_id": "gliner-files"},
        {"section_id": "chunk-1", "entity_id": "noise-data"},
    ]

    merge_section_mentions(sections, entities, mentions)

    assert sections[0]["_mentions"] == [
        {"entity_id": "gliner-files", "source": "gliner"},
        {"section_id": "source-section-1", "entity_id": "struct-files"},
    ]


def test_execute_saga_commits_neo4j_after_qdrant_and_links_after_commit():
    events = []

    class FakeTx:
        def __init__(self):
            self._closed = False

        def commit(self):
            events.append("neo4j_commit")
            self._closed = True

        def rollback(self):
            events.append("neo4j_rollback")
            self._closed = True

        def closed(self):
            return self._closed

    class FakeSession:
        def __init__(self):
            self.tx = FakeTx()

        def begin_transaction(self):
            events.append("neo4j_begin")
            return self.tx

        def close(self):
            events.append("session_close")

    class FakeDriver:
        def session(self):
            events.append("session_open")
            return FakeSession()

    class FakeNeo4jWriter:
        def _neo4j_upsert_document(self, tx, document):
            assert not tx.closed()
            events.append("neo4j_document")

        def _neo4j_upsert_sections(self, tx, document_id, sections):
            assert not tx.closed()
            events.append("neo4j_sections")
            return len(sections)

        def _neo4j_upsert_entities(self, tx, entities):
            assert not tx.closed()
            events.append("neo4j_entities")
            return len(entities)

        def _neo4j_create_mentions(self, tx, mentions):
            assert not tx.closed()
            events.append("neo4j_mentions")

        def _neo4j_create_references(self, tx, references):
            assert not tx.closed()
            events.append("neo4j_references")
            return len(references)

        def _neo4j_upsert_embedding_metadata(self, tx, sections, embeddings, builder):
            assert not tx.closed()
            events.append("neo4j_embedding_metadata")
            return len(sections)

    class FakeQdrantWriter:
        def _qdrant_upsert_vectors(self, document, sections, embeddings, builder):
            assert "neo4j_commit" not in events
            events.append("qdrant_upsert")
            return len(sections)

        def _compensate_qdrant(self, points, builder):
            events.append("qdrant_compensate")

    def fake_structural_edges(tx, document_id, *, skip_has_chunk):
        assert not tx.closed()
        events.append("structural_edges")
        return {"stats": {"NEXT_CHUNK": 0}, "warnings": []}

    def fake_cross_doc_linker(**kwargs):
        assert "neo4j_commit" in events
        events.append("cross_doc_linking")
        return {"skipped": True, "reason": "test"}

    result = execute_saga(
        saga_id="saga-test",
        document={"id": "doc-nutanix-files"},
        sections=[{"id": "chunk-1", "_mentions": []}],
        entities={},
        mentions=[],
        references=[],
        embeddings={"sections": {"chunk-1": {"content": [0.1]}}},
        builder=SimpleNamespace(collection_name="test-collection"),
        neo4j_driver=FakeDriver(),
        qdrant_client=object(),
        neo4j_writer=FakeNeo4jWriter(),
        qdrant_writer=FakeQdrantWriter(),
        config=SimpleNamespace(
            search=SimpleNamespace(
                vector=SimpleNamespace(
                    primary="qdrant",
                    dual_write=False,
                    qdrant=SimpleNamespace(collection_name="test-collection"),
                )
            )
        ),
        structural_edges_builder=fake_structural_edges,
        cross_doc_linker=fake_cross_doc_linker,
    )

    assert result["success"] is True
    assert events.index("qdrant_upsert") < events.index("neo4j_commit")
    assert events.index("neo4j_commit") < events.index("cross_doc_linking")
    assert events[-1] == "session_close"


def test_compute_embeddings_assembles_dense_vectors_with_batch_stats(monkeypatch):
    monkeypatch.setattr(
        embed_stage,
        "TokenizerService",
        lambda: SimpleNamespace(count_tokens=lambda text: len(text.split())),
    )

    class FakeEmbedder:
        provider_name = "fake"
        task = "retrieval"

        def embed_documents(self, texts):
            return [[float(len(text)), 0.2, 0.3] for text in texts]

    class FakeBuilder:
        embedder = FakeEmbedder()
        embedding_plan = None
        embedding_settings = None
        embedding_dims = 3

        def _build_section_text_for_embedding(self, section):
            return section["text"]

        def _build_title_text_for_embedding(self, section):
            return section["heading"]

    config = SimpleNamespace(
        search=SimpleNamespace(
            vector=SimpleNamespace(
                primary="neo4j",
                dual_write=False,
                qdrant=SimpleNamespace(
                    enable_sparse=False,
                    enable_colbert=False,
                    sparse_strict_mode=False,
                ),
            )
        )
    )
    section = {
        "id": "chunk-1",
        "document_id": "doc-nutanix-files",
        "level": 2,
        "order": 1,
        "original_section_ids": ["source-section-1"],
        "is_combined": False,
        "is_split": False,
        "token_count": 4,
        "heading": "Nutanix Files",
        "text": "Nutanix Files supports SMB.",
    }

    result = compute_embeddings(
        {"id": "doc-nutanix-files", "title": "Nutanix Files"},
        [section],
        {},
        FakeBuilder(),
        config,
    )

    assert result["sections"]["chunk-1"]["content"] == [27.0, 0.2, 0.3]
    assert result["sections"]["chunk-1"]["title"] == [13.0, 0.2, 0.3]
    assert result["sections"]["chunk-1"]["doc_title"] == [13.0, 0.2, 0.3]
    assert result["stats"]["batch_count"] == 1
    assert result["stats"]["total_tokens_processed"] == 4
