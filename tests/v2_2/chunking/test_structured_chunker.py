"""Unit tests for the structured chunk assembler."""

from prometheus_client import REGISTRY

from src.ingestion.chunk_assembler import StructuredChunker
from src.shared.config import (
    ChunkAssemblyConfig,
    ChunkMicrodocConfig,
    ChunkSplitConfig,
    ChunkStructureConfig,
    SemanticEnrichmentConfig,
)


def _metric_value(metric_name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(metric_name, labels)
    return 0.0 if value is None else value


def _make_config(
    *,
    semantic_enabled: bool = False,
    microdoc_enabled: bool = True,
    **overrides,
) -> ChunkAssemblyConfig:
    structure = ChunkStructureConfig(
        min_tokens=overrides.get("min_tokens", 50),
        target_tokens=overrides.get("target_tokens", 120),
        hard_tokens=overrides.get("hard_tokens", 400),
        max_sections=overrides.get("max_sections", 8),
        respect_major_levels=True,
        stop_at_level=2,
        break_keywords="",
    )
    split = ChunkSplitConfig(
        enabled=True,
        max_tokens=overrides.get("split_max_tokens", 400),
        overlap_tokens=overrides.get("split_overlap_tokens", 50),
    )
    microdoc = ChunkMicrodocConfig(
        enabled=microdoc_enabled,
        doc_token_threshold=overrides.get("microdoc_threshold", 300),
        min_split_tokens=overrides.get("microdoc_min_split", 10),
    )
    semantic = SemanticEnrichmentConfig(
        enabled=semantic_enabled,
        provider="stub",
        model_name="stub-model" if semantic_enabled else None,
        timeout_seconds=1,
        max_retries=1,
    )
    return ChunkAssemblyConfig(
        assembler="structured",
        structure=structure,
        split=split,
        microdoc=microdoc,
        semantic=semantic,
    )


def _section(section_id: str, title: str, text: str, order: int = 0) -> dict:
    return {
        "id": section_id,
        "title": title,
        "text": text,
        "tokens": len(text.split()),
        "level": 2,
        "order": order,
    }


def test_structured_chunker_marks_microdoc_documents():
    config = _make_config(microdoc_threshold=500)
    chunker = StructuredChunker(config)
    sections = [
        _section("s1", "Overview", "alpha beta" * 50, 0),
        _section("s2", "Details", "gamma delta" * 50, 1),
    ]

    chunks = chunker.assemble("doc-1", sections)

    assert chunks, "Expected chunks to be generated"
    assert all(chunk.get("doc_is_microdoc") for chunk in chunks)


def test_structured_chunker_emits_stub_for_single_microdoc_chunk():
    config = _make_config(microdoc_threshold=500, microdoc_min_split=1)
    chunker = StructuredChunker(config)
    sections = [
        _section("s1", "Solo", "word " * 60, 0),
    ]

    chunks = chunker.assemble("doc-2", sections)

    assert len(chunks) >= 2
    assert any(chunk.get("is_microdoc_stub") for chunk in chunks)


def test_semantic_enrichment_disabled_is_noop():
    config = _make_config(microdoc_enabled=False)
    chunker = StructuredChunker(config)
    sections = [
        _section("s1", "Solo", "token " * 40, 0),
    ]

    base_value = _metric_value(
        "semantic_enrichment_total", {"provider": "stub", "status": "success"}
    )

    chunks = chunker.assemble("doc-semantic-disabled", sections)

    assert chunks, "expected chunk output"
    assert chunks[0].get("semantic_metadata") == {}
    assert (
        _metric_value(
            "semantic_enrichment_total", {"provider": "stub", "status": "success"}
        )
        == base_value
    )


def test_semantic_enrichment_enabled_emits_metadata_and_metrics():
    config = _make_config(microdoc_enabled=False, semantic_enabled=True)
    chunker = StructuredChunker(config)
    sections = [
        _section("s1", "Enabled", "alpha beta gamma " * 20, 0),
    ]

    base_success = _metric_value(
        "semantic_enrichment_total", {"provider": "stub", "status": "success"}
    )

    chunks = chunker.assemble("doc-semantic-enabled", sections)

    assert chunks, "expected chunk output"
    assert all(
        chunk.get("semantic_metadata")
        == {"entities": [], "topics": [], "summary": None}
        for chunk in chunks
    )

    expected_success = base_success + len(chunks)
    assert (
        _metric_value(
            "semantic_enrichment_total", {"provider": "stub", "status": "success"}
        )
        == expected_success
    )
