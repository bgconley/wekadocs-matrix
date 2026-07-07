"""Functional tests for extracted ingestion stages."""

from __future__ import annotations

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
