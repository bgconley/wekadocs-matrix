"""
Phase 7E-3 Integration Test
Verifies that stitched context provided to the LLM preserves chunk ordering
and emits sequential citations after ingesting mixed-size markdown documents.
"""

import uuid

import pytest

from src.ingestion.build_graph import ingest_document
from src.mcp_server.query_service import QueryService
from src.shared.config import get_config
from src.shared.connections import get_connection_manager


def _build_long_markdown() -> str:
    """Generate markdown with adjacent steps for chunk stitching verification."""

    step_one_body = (
        "Prepare all prerequisite packages, validate network reachability, and ensure "
        "cluster services are responsive before proceeding. ORDER_STEP_ONE markers "
        "appear to help tests assert ordering."
    )

    step_two_body = (
        "Complete the installation by executing the deployment command twice to "
        "simulate idempotent runs. ORDER_STEP_TWO appears in this section so the "
        "test can assert stitched ordering for Step 2."
    )

    appendix_paragraph = (
        "Extended parameter reference covering dozens of tunables. This paragraph "
        "is repeated to simulate a very long section for chunk assembly testing. "
        "The appendix includes default values, edge cases, and operational notes."
    )

    appendix_body = "\n\n".join([appendix_paragraph for _ in range(40)])

    return f"""# Installation Deep Dive

## Quick Recap

This document expands on the installation workflow and references additional
materials for completeness so that retrieval has multiple documents to fuse.

## Step-by-step Installation

### Step 1: Prepare Environment

{step_one_body}

### Step 2: Complete Installation

{step_two_body}

### Step 3: Validate Installation

Confirm services are healthy and run smoke tests to ensure no regressions occur.

## Appendix A: Extended Reference

{appendix_body}
"""


def _build_short_markdown() -> str:
    return """# Quick Start Overview

## Purpose

This short guide exists to ensure multi-document ingestion paths stay intact.
"""


@pytest.mark.integration
def test_context_stitching_preserves_order_and_citations():
    """Ingest two docs and verify stitched context ordering + citations."""

    short_uri = f"tests://integration/short-context-{uuid.uuid4()}.md"
    long_uri = f"tests://integration/long-context-{uuid.uuid4()}.md"

    short_stats = ingest_document(short_uri, _build_short_markdown())
    long_stats = ingest_document(long_uri, _build_long_markdown())

    document_ids = [short_stats["document_id"], long_stats["document_id"]]

    query_service = QueryService()

    try:
        response = query_service.search(
            query="Detailed instructions for completing step 2 installation",
            top_k=5,
            verbosity="graph",
        )

        markdown = response.answer_markdown
        context_header = "### Context (chunk-stitched, budget-bounded)"
        assert (
            context_header in markdown
        ), "Stitched context header missing from response markdown"

        context_section = markdown.split(context_header, 1)[1].strip()
        body_part, citation_part = context_section.split("\n\n---\n### Citations\n", 1)

        # Verify both steps are retrieved and included in context in document order
        assert "ORDER_STEP_ONE" in body_part, "Step 1 content not present in context"
        assert "ORDER_STEP_TWO" in body_part, "Step 2 content not present in context"
        assert body_part.index("ORDER_STEP_ONE") < body_part.index(
            "ORDER_STEP_TWO"
        ), "Step 1 content should appear before Step 2 in stitched context"

        citation_lines = [
            line.strip() for line in citation_part.splitlines() if line.strip()
        ]
        assert len(citation_lines) >= 2, "Expected at least two citations"

        assert (
            citation_lines[0].startswith("[1]")
            and "Step 1: Prepare Environment" in citation_lines[0]
        ), "First citation should reference Step 1"
        assert (
            citation_lines[1].startswith("[2]")
            and "Step 2: Complete Installation" in citation_lines[1]
        ), "Second citation should reference Step 2"

        for i, line in enumerate(citation_lines, start=1):
            assert line.startswith(f"[{i}]")

    finally:
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()
        with neo4j_driver.session() as session:
            for doc_id in document_ids:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    DETACH DELETE d, s
                    """,
                    doc_id=doc_id,
                )

        qdrant_client = manager.get_qdrant_client()
        collection = get_config().search.vector.qdrant.collection_name
        for doc_id in document_ids:
            try:
                qdrant_client.delete_compat(
                    collection_name=collection,
                    points_selector={
                        "filter": {
                            "must": [{"key": "document_id", "match": {"value": doc_id}}]
                        }
                    },
                    wait=True,
                )
            except Exception:
                # Collection may not exist in certain test modes
                pass
