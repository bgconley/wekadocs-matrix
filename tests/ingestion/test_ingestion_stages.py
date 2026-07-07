"""Functional tests for extracted ingestion stages."""

from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.stages.chunk import assemble_chunks
from src.ingestion.stages.enrich import merge_section_mentions
from src.ingestion.stages.parse import parse_document


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
