"""Cross-document link stage for atomic ingestion."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.shared.observability import get_logger

logger = get_logger(__name__)


def _add_trace_event(trace, *, kind: str, message: str, data: Optional[Dict] = None):
    if trace:
        trace.add_event(
            stage="link",
            kind=kind,
            message=message,
            data=data or {},
        )


def get_document_count(neo4j_driver, *, trace=None) -> int:
    """Get total document count for corpus size check."""
    try:
        with neo4j_driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            record = result.single()
            return record["count"] if record else 0
    except Exception as e:
        logger.warning("cross_doc_get_count_failed", error=str(e))
        _add_trace_event(
            trace,
            kind="fallback",
            message="cross_doc_get_count_failed",
            data={"error": str(e)},
        )
        return 0


def create_cross_doc_links(
    *,
    neo4j_driver,
    qdrant_client,
    config,
    document_id: str,
    document: Dict,
    sections: List[Dict],
    embeddings: Dict,
    trace=None,
) -> Optional[Dict[str, Any]]:
    """
    Create cross-document links for a newly ingested document.

    This is called AFTER the Neo4j commit to ensure the document exists.
    Failures here are logged but NEVER fail the ingestion.
    """
    _ = sections

    linking_config = getattr(
        getattr(config, "ingestion", None),
        "cross_doc_linking",
        None,
    )
    if not linking_config:
        _add_trace_event(
            trace,
            kind="skipped",
            message="cross_doc_linking_not_configured",
            data={"document_id": document_id},
        )
        return {"skipped": True, "reason": "not_configured"}

    if not linking_config.enabled:
        _add_trace_event(
            trace,
            kind="skipped",
            message="cross_doc_linking_disabled",
            data={"document_id": document_id},
        )
        return {"skipped": True, "reason": "disabled"}

    if not qdrant_client:
        _add_trace_event(
            trace,
            kind="skipped",
            message="cross_doc_linking_no_qdrant_client",
            data={"document_id": document_id},
        )
        return {"skipped": True, "reason": "no_qdrant_client"}

    doc_count = get_document_count(neo4j_driver, trace=trace)
    if doc_count < linking_config.min_corpus_size:
        logger.debug(
            "cross_doc_linking_skipped_corpus_small",
            document_id=document_id,
            doc_count=doc_count,
            min_required=linking_config.min_corpus_size,
        )
        _add_trace_event(
            trace,
            kind="skipped",
            message="cross_doc_linking_skipped_corpus_small",
            data={
                "document_id": document_id,
                "doc_count": doc_count,
                "min_required": linking_config.min_corpus_size,
            },
        )
        return {"skipped": True, "reason": f"corpus_too_small:{doc_count}"}

    doc_title_vector = None
    doc_title_sparse = None

    section_embeddings = embeddings.get("sections", {})
    if section_embeddings:
        first_section_id = next(iter(section_embeddings.keys()), None)
        if first_section_id:
            section_emb = section_embeddings[first_section_id]
            doc_title_vector = section_emb.get("doc_title")
            doc_title_sparse = section_emb.get("doc_title_sparse")

    if not doc_title_vector:
        logger.debug(
            "cross_doc_linking_skipped_no_vector",
            document_id=document_id,
        )
        _add_trace_event(
            trace,
            kind="skipped",
            message="cross_doc_linking_skipped_no_vector",
            data={"document_id": document_id},
        )
        return {"skipped": True, "reason": "no_doc_title_vector"}

    try:
        from src.services.cross_doc_linking import CrossDocLinker

        start_time = time.time()

        linker = CrossDocLinker(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            config=linking_config,
        )

        result = linker.link_document(
            doc_id=document_id,
            doc_title=document.get("title", ""),
            doc_title_vector=doc_title_vector,
            doc_title_sparse=doc_title_sparse,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        sample_edges = []
        if hasattr(result, "edges") and result.edges:
            sample_edges = [
                {
                    "target": getattr(e, "target_doc_id", None),
                    "score": getattr(e, "score", None),
                    "colbert_score": getattr(e, "colbert_score", None),
                }
                for e in result.edges[:3]
            ]
        logger.info(
            "cross_doc_linking_complete",
            doc_id=document_id,
            edges_created=result.edges_created,
            edges_updated=result.edges_updated,
            candidates_evaluated=result.candidates_found,
            pruned_count=result.candidates_found - result.edges_created,
            method=result.method,
            colbert_reranked=getattr(result, "colbert_reranked", False),
            duration_ms=duration_ms,
            sample_edges=sample_edges,
            skipped=result.skipped,
            skip_reason=result.skip_reason,
        )

        return {
            "edges_created": result.edges_created,
            "edges_updated": result.edges_updated,
            "candidates_found": result.candidates_found,
            "method": result.method,
            "duration_ms": duration_ms,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        }

    except Exception as e:
        logger.warning(
            "cross_doc_linking_failed",
            document_id=document_id,
            error=str(e),
        )
        _add_trace_event(
            trace,
            kind="error",
            message="cross_doc_linking_failed",
            data={"document_id": document_id, "error": str(e)},
        )
        return {"skipped": True, "reason": f"error:{str(e)[:100]}"}
