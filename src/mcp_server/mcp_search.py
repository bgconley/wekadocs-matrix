# =============================================================================
# Search, evidence extraction, and diagnostics
# =============================================================================
"""
Core search candidates retrieval, evidence extraction with retrieval-score blending,
structural graph expansion, and diagnostic emission.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from uuid import uuid4

from src.mcp_server.mcp_utils import (
    _DIAGNOSTIC_EMITTER,
    KB_EVIDENCE_FORCE_METADATA_LIMITATIONS,
    KB_EVIDENCE_GRAPH_SYNTHETIC_FUSED_SCORE,
    KB_EVIDENCE_MAX_QUOTES,
    KB_EVIDENCE_METADATA_LIMITATIONS_MIN_SCORE,
    KB_SEARCH_DEFAULT_SNIPPET_CHARS,
    KB_SEARCH_DEFAULT_TOP_K,
    KB_SEARCH_MAX_PAGE_SIZE,
    KB_SEARCH_MAX_SNIPPET_CHARS,
    KB_SEARCH_MAX_TOP_K,
    Deps,
    _build_preview,
    _config,
    _decode_cursor,
    _dedupe_by_doc,
    _detect_transport,
    _encode_cursor,
    _is_graph_expanded_source,
    _is_metadata_limitations_candidate,
    _merge_scope_filters,
    _neo4j_disabled,
    _normalize_scope,
    _quote_from_passage,
    _tokenize_query,
)
from src.mcp_server.scratch_store import ScratchStore
from src.query.retrieval_types import ChunkResult
from src.shared.observability import get_correlation_id, get_logger

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    StatusCode = None  # type: ignore

logger = get_logger(__name__)


async def _kb_search_candidates(
    *,
    query: str,
    top_k: int,
    cursor: Optional[str],
    page_size: int,
    scope: Optional[dict[str, Any]],
    filters: Optional[dict[str, Any]],
    options: Optional[dict[str, Any]],
    deps: Deps,
    effective_session: str,
    _fetch_k_override: Optional[int] = None,
) -> tuple[dict, dict]:
    normalized_scope = _normalize_scope(scope)
    merged_filters = _merge_scope_filters(filters, normalized_scope)
    options = options or {}

    mode = str(options.get("mode", "auto")).lower()
    include_scores = bool(options.get("include_scores", False))
    include_debug = bool(options.get("include_debug", False))
    max_snippet_chars = int(
        options.get("max_snippet_chars", KB_SEARCH_DEFAULT_SNIPPET_CHARS)
    )
    max_snippet_chars = max(60, min(max_snippet_chars, KB_SEARCH_MAX_SNIPPET_CHARS))
    max_per_doc = int(options.get("max_per_doc", 1))
    max_per_doc = max(1, min(max_per_doc, 5))

    if _fetch_k_override is not None:
        # Evidence pack path: bypass the KB_SEARCH_MAX_TOP_K cap
        top_k = max(1, int(_fetch_k_override))
        page = top_k
    else:
        top_k = max(1, min(int(top_k or KB_SEARCH_DEFAULT_TOP_K), KB_SEARCH_MAX_TOP_K))
        page = max(1, min(int(page_size or top_k), KB_SEARCH_MAX_PAGE_SIZE))

    offset = _decode_cursor(cursor)
    effective_limit = min(page, top_k)
    total_cap = max(page, top_k)
    fetch_k = min(total_cap, offset + effective_limit + 1)

    if mode == "vector_only":
        expand = False
    elif mode == "hybrid_local":
        expand = True
    else:
        expand = not _neo4j_disabled

    chunks, metrics = deps.query.search_sections_light(
        query=query,
        fetch_k=fetch_k,
        filters=merged_filters,
        expand=expand,
    )
    deduped, duplicates = _dedupe_by_doc(chunks, max_per_doc)

    sliced = deduped[offset : offset + effective_limit]
    more = len(deduped) > offset + len(sliced)
    next_cursor = _encode_cursor(offset + len(sliced)) if more else None

    results = []
    for idx, chunk in enumerate(sliced):
        score = (
            chunk.rerank_score
            if chunk.rerank_score is not None
            else (chunk.fused_score or chunk.vector_score or chunk.bm25_score or 0.0)
        )
        source = "hybrid"
        graph_distance = getattr(chunk, "graph_distance", 0) or 0
        graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
        if chunk.rerank_score is not None:
            source = "reranked"
        elif graph_distance > 0 or graph_score > 0:
            source = "graph_expanded"
        elif chunk.fusion_method == "rrf":
            source = "rrf_fusion"
        elif chunk.vector_score is not None and chunk.bm25_score is None:
            source = "vector"
        elif chunk.bm25_score is not None and chunk.vector_score is None:
            source = "bm25"

        preview = _build_preview(chunk.text or "", query, max_snippet_chars)
        passage_id = uuid4().hex
        scratch_payload = {
            "section_id": chunk.chunk_id,
            "doc_tag": chunk.doc_tag,
            "title": chunk.heading,
            "text": chunk.text,
            "source_uri": getattr(chunk, "source_path", None),
            "created_at": datetime.utcnow().isoformat() + "Z",
            # Retrieval scores — preserved for evidence extraction
            "rerank_score": chunk.rerank_score,
            "fused_score": chunk.fused_score,
            "vector_score": chunk.vector_score,
            "bm25_score": chunk.bm25_score,
            "graph_score": getattr(chunk, "graph_score", None),
            "parent_path_norm": chunk.parent_path_norm,
            "rerank_rank": chunk.rerank_rank,
            "fusion_method": chunk.fusion_method,
            "is_expanded": chunk.is_expanded,
            "expansion_source": chunk.expansion_source,
            "source": source,
        }
        size_bytes = await deps.scratch.put(
            effective_session, passage_id, scratch_payload
        )

        result = {
            "passage_id": passage_id,
            "section_id": chunk.chunk_id,
            "doc_tag": chunk.doc_tag,
            "title": chunk.heading,
            "rank": offset + idx + 1,
            "score": float(score),
            "source": source,
            "preview": preview,
            "scratch_uri": ScratchStore.build_uri(effective_session, passage_id),
            "size_bytes": size_bytes,
            "anchor": getattr(chunk, "anchor", None),
        }

        if include_scores and not include_debug:
            result["score_breakdown"] = {
                "fused_score": (
                    float(chunk.fused_score) if chunk.fused_score is not None else None
                ),
                "vector_score": (
                    float(chunk.vector_score)
                    if chunk.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    float(chunk.bm25_score) if chunk.bm25_score is not None else None
                ),
                "rerank_score": (
                    float(chunk.rerank_score)
                    if chunk.rerank_score is not None
                    else None
                ),
                "graph_score": float(graph_score),
            }

        if include_debug:
            result["debug"] = {
                "fusion_method": chunk.fusion_method,
                "fused_score": (
                    float(chunk.fused_score) if chunk.fused_score is not None else None
                ),
                "vector_score": (
                    float(chunk.vector_score)
                    if chunk.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    float(chunk.bm25_score) if chunk.bm25_score is not None else None
                ),
                "rerank_score": (
                    float(chunk.rerank_score)
                    if chunk.rerank_score is not None
                    else None
                ),
                "graph_score": float(graph_score),
                "graph_distance": int(graph_distance),
                "connection_count": int(getattr(chunk, "connection_count", 0) or 0),
                "mention_count": int(getattr(chunk, "mention_count", 0) or 0),
                "entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
            }

        results.append(result)

    payload = {
        "results": results,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "duplicates": duplicates,
        "metrics": metrics,
        "partial": bool(next_cursor),
        "limit_reason": "page_size" if next_cursor else "none",
        "include_debug": include_debug,
    }
    diagnostic_context = {
        "query": query,
        "scope": normalized_scope,
        "filters": merged_filters,
        "metrics": metrics,
        "chunks": list(sliced),
        "duplicates": duplicates,
        "deduped_count": len(deduped),
        "mode": mode,
        "top_k": top_k,
    }
    return payload, diagnostic_context


async def _extract_evidence_from_passages(
    *,
    question: str,
    passage_ids: list[str],
    max_quotes: int,
    max_quote_tokens: int,
    include_context_tokens: int,
    deps: Deps,
    effective_session: str,
) -> list[dict[str, Any]]:
    """Extract evidence quotes using retrieval scores + lexical blending.

    Stage 1: Rank passages by retrieval score (rerank > fused > vector > bm25).
    Stage 2: Within each passage, select the best span using a blended score
    of retrieval quality (70%) and keyword overlap (30%).
    """
    query_tokens = _tokenize_query(question)
    if not passage_ids:
        return []

    # Stage 1: Load passages with retrieval scores and source metadata.
    passages = []
    for rank, passage_id in enumerate(passage_ids):
        entry = await deps.scratch.get(effective_session, passage_id)
        if not entry:
            continue
        retrieval_score = (
            entry.get("rerank_score")
            or entry.get("fused_score")
            or entry.get("vector_score")
            or entry.get("bm25_score")
            or 0.0
        )
        source = entry.get("source", "hybrid")
        passages.append(
            {
                "passage_id": passage_id,
                "entry": entry,
                "retrieval_score": float(retrieval_score),
                "rank": rank + 1,
                "source": source,
                "is_graph_expanded": _is_graph_expanded_source(source),
            }
        )

    if not passages:
        return []

    # Primary candidates first (reranked/rrf/vector/bm25/hybrid), graph-expanded as backfill.
    passages.sort(
        key=lambda p: (
            1 if p["is_graph_expanded"] else 0,
            -p["retrieval_score"],
            p["rank"],
        )
    )
    primary_passages = [p for p in passages if not p["is_graph_expanded"]]
    backfill_passages = [p for p in passages if p["is_graph_expanded"]]

    max_quotes = max(1, min(int(max_quotes or 6), KB_EVIDENCE_MAX_QUOTES))
    max_quote_tokens = max(1, min(int(max_quote_tokens or 80), 200))
    include_context_tokens = max(0, min(int(include_context_tokens or 20), 200))
    context_chars = include_context_tokens * 4
    max_quote_chars = min(max_quote_tokens * 4, 500)

    quotes: list[dict[str, Any]] = []
    seen_passage_ids: set[str] = set()

    def _append_quote(passage: dict[str, Any]) -> bool:
        if len(quotes) >= max_quotes:
            return False
        pid = passage["passage_id"]
        if pid in seen_passage_ids:
            return False
        quote = _quote_from_passage(
            passage=passage,
            query_tokens=query_tokens,
            max_quote_chars=max_quote_chars,
            context_chars=context_chars,
            effective_session=effective_session,
        )
        if quote is None:
            return False
        seen_passage_ids.add(pid)
        quotes.append(quote)
        return True

    # Primary evidence always wins first-pass selection.
    for passage in primary_passages:
        if len(quotes) >= max_quotes:
            break
        _append_quote(passage)

    # Safety guard: retain at least one strong reranked "metadata limitations" quote
    # before any graph-expanded backfill is considered.
    if KB_EVIDENCE_FORCE_METADATA_LIMITATIONS and "metadata" in query_tokens:
        has_metadata_limitations_quote = any(
            "metadata" in (q.get("title") or "").lower()
            and "limitation" in (q.get("title") or "").lower()
            for q in quotes
        )
        if not has_metadata_limitations_quote:
            candidate = next(
                (
                    p
                    for p in primary_passages
                    if p["source"] == "reranked"
                    and p["retrieval_score"]
                    >= KB_EVIDENCE_METADATA_LIMITATIONS_MIN_SCORE
                    and _is_metadata_limitations_candidate(p["entry"], query_tokens)
                ),
                None,
            )
            if candidate:
                forced_quote = _quote_from_passage(
                    passage=candidate,
                    query_tokens=query_tokens,
                    max_quote_chars=max_quote_chars,
                    context_chars=context_chars,
                    effective_session=effective_session,
                )
                if forced_quote:
                    forced_pid = forced_quote["passage_id"]
                    if forced_pid not in seen_passage_ids:
                        # Replace weakest selected primary quote only if we are full.
                        replace_idx = None
                        if len(quotes) >= max_quotes and quotes:
                            retrieval_by_pid = {
                                p["passage_id"]: float(p["retrieval_score"])
                                for p in primary_passages
                            }
                            replace_idx = min(
                                range(len(quotes)),
                                key=lambda idx: retrieval_by_pid.get(
                                    quotes[idx].get("passage_id"), float("inf")
                                ),
                            )
                        if replace_idx is not None and len(quotes) >= max_quotes:
                            old_pid = quotes[replace_idx].get("passage_id")
                            if old_pid:
                                seen_passage_ids.discard(old_pid)
                            quotes[replace_idx] = forced_quote
                            seen_passage_ids.add(forced_pid)
                        elif len(quotes) < max_quotes:
                            quotes.append(forced_quote)
                            seen_passage_ids.add(forced_pid)

    # Backfill only from graph-expanded passages if slots remain.
    if len(quotes) < max_quotes:
        for passage in backfill_passages:
            if len(quotes) >= max_quotes:
                break
            _append_quote(passage)

    return quotes


async def _expand_evidence_with_structure(
    *,
    section_ids: list[str],
    deps: Deps,
    effective_session: str,
    max_neighbors: int = 20,
) -> list[str]:
    """Fetch structural neighbors (NEXT_CHUNK, siblings) and store in scratch.

    Adds sequential and sibling chunks to the evidence extraction candidate
    pool. Returns new passage_ids for expanded neighbors. Skips gracefully
    if Neo4j is unavailable.
    """
    if _neo4j_disabled or not deps.graph or not section_ids:
        return []

    seed_ids = section_ids[:10]  # Top 10 seeds for structural expansion

    cypher = """
    UNWIND $ids AS sid
    MATCH (c:Chunk {id: sid})
    OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(nxt:Chunk)
    OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
    OPTIONAL MATCH (sib:Chunk {parent_section_id: c.parent_section_id})
      WHERE sib.id <> c.id
    WITH c, nxt, prev, collect(DISTINCT sib)[..3] AS sibs
    UNWIND (
      CASE WHEN nxt IS NOT NULL THEN [nxt] ELSE [] END +
      CASE WHEN prev IS NOT NULL THEN [prev] ELSE [] END +
      sibs
    ) AS neighbor
    WITH neighbor WHERE NOT neighbor.id IN $ids
    RETURN DISTINCT neighbor.id AS chunk_id,
           neighbor.heading AS heading,
           neighbor.text AS text,
           neighbor.doc_tag AS doc_tag,
           neighbor.parent_path_norm AS parent_path_norm,
           neighbor.parent_section_id AS parent_section_id
    LIMIT $limit
    """
    try:
        with deps.graph.driver.session() as session:
            result = session.run(cypher, ids=seed_ids, limit=max_neighbors)
            records = list(result)
    except Exception as exc:
        logger.warning(f"evidence_graph_expansion_failed: {exc}")
        return []

    if not records:
        return []

    # Store expanded neighbors in scratch with synthetic scores
    new_passage_ids = []
    existing_section_ids = set(section_ids)
    for record in records:
        cid = record["chunk_id"]
        if cid in existing_section_ids:
            continue
        existing_section_ids.add(cid)

        passage_id = uuid4().hex
        scratch_payload = {
            "section_id": cid,
            "doc_tag": record["doc_tag"],
            "title": record["heading"] or "",
            "text": record["text"] or "",
            "source_uri": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            # Structural neighbors get a low synthetic score so they only
            # backfill evidence slots after primary retrieval candidates.
            "rerank_score": None,
            "fused_score": KB_EVIDENCE_GRAPH_SYNTHETIC_FUSED_SCORE,
            "vector_score": None,
            "bm25_score": None,
            "graph_score": 0.5,
            "parent_path_norm": record["parent_path_norm"],
            "rerank_rank": None,
            "fusion_method": "graph_expansion",
            "is_expanded": True,
            "expansion_source": "evidence_structural",
            "source": "graph_expanded",
        }
        await deps.scratch.put(effective_session, passage_id, scratch_payload)
        new_passage_ids.append(passage_id)

    logger.info(
        f"evidence_graph_expansion: seeds={len(seed_ids)}, neighbors_added={len(new_passage_ids)}"
    )
    return new_passage_ids


def _infer_source(chunk: ChunkResult) -> str:
    source = "hybrid"
    graph_distance = getattr(chunk, "graph_distance", 0) or 0
    graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
    if chunk.rerank_score is not None:
        source = "reranked"
    elif graph_distance > 0 or graph_score > 0:
        source = "graph_expanded"
    elif chunk.fusion_method == "rrf":
        source = "rrf_fusion"
    elif chunk.vector_score is not None and chunk.bm25_score is None:
        source = "vector"
    elif chunk.bm25_score is not None and chunk.vector_score is None:
        source = "bm25"
    return source


def _infer_source_tags(chunk: ChunkResult) -> List[str]:
    """Return all applicable source tags for diagnostics (non-exclusive)."""
    tags: List[str] = []
    graph_distance = getattr(chunk, "graph_distance", 0) or 0
    graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
    if chunk.rerank_score is not None:
        tags.append("reranked")
    if graph_distance > 0 or graph_score > 0:
        tags.append("graph_expanded")
    if chunk.fusion_method == "rrf":
        tags.append("rrf_fusion")
    if chunk.vector_score is not None:
        tags.append("vector")
    if chunk.bm25_score is not None:
        tags.append("bm25")
    return tags if tags else ["hybrid"]


def _diagnostic_results_from_chunks(chunks: list[ChunkResult]) -> list[dict[str, Any]]:
    results = []
    for idx, chunk in enumerate(chunks):
        results.append(
            {
                "rank": idx + 1,
                "chunk_id": chunk.chunk_id,
                "doc_tag": chunk.doc_tag,
                "token_count": chunk.token_count,
                "source": _infer_source(chunk),
                "source_tags": _infer_source_tags(chunk),
                "scores": {
                    "bm25": float(chunk.bm25_score or 0.0),
                    "vector": float(chunk.vector_score or 0.0),
                    "fused": float(chunk.fused_score or 0.0),
                    "rerank": float(chunk.rerank_score or 0.0),
                    "graph": float(getattr(chunk, "graph_score", 0.0) or 0.0),
                },
                "explain": {
                    "rrf_field_contributions": getattr(
                        chunk, "rrf_field_contributions", None
                    ),
                    "entity_boost_applied": getattr(
                        chunk, "entity_boost_applied", False
                    ),
                    "entity_metadata": getattr(chunk, "entity_metadata", None),
                    "graph_distance": int(getattr(chunk, "graph_distance", 0) or 0),
                    "graph_path": getattr(chunk, "graph_path", None),
                },
            }
        )
    return results


async def _emit_diagnostics(
    *,
    tool_name: str,
    ctx: Any | None,
    session_id: str,
    diagnostic_context: dict[str, Any],
    tokens_estimate: int,
    bytes_estimate: int,
    partial: bool,
    limit_reason: str,
) -> Optional[dict[str, Any]]:
    metrics = diagnostic_context.get("metrics") or {}
    scope = diagnostic_context.get("scope") or {}
    filters = diagnostic_context.get("filters") or {}
    chunks = diagnostic_context.get("chunks") or []
    duplicates = diagnostic_context.get("duplicates") or 0
    deduped_count = diagnostic_context.get("deduped_count") or len(chunks)
    doc_tag = None
    raw_doc_tag = filters.get("doc_tag")
    if isinstance(raw_doc_tag, list) and raw_doc_tag:
        doc_tag = raw_doc_tag[0]
    elif isinstance(raw_doc_tag, str):
        doc_tag = raw_doc_tag

    results = _diagnostic_results_from_chunks(chunks)
    candidates_initial = int(metrics.get("vec_count", 0) or 0) + int(
        metrics.get("bm25_count", 0) or 0
    )
    candidates_post_filter = int(metrics.get("primary_count", 0) or 0)
    candidates_post_dedupe = int(metrics.get("final_count", 0) or deduped_count)

    record = {
        "schema_version": 1,
        "diagnostic_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transport": _detect_transport(ctx),
        "tool": tool_name,
        "session_id": session_id,
        "correlation_id": get_correlation_id(),
        "otel": {"trace_id": None, "span_id": None},
        "scope": {
            "project_id": scope.get("project_id"),
            "environment": scope.get("environment"),
            "doc_tag": doc_tag,
            "snapshot_scope": filters.get("snapshot_scope"),
            "embedding_version": metrics.get("embedding_version"),
        },
        "query": {
            "raw": diagnostic_context.get("query"),
            "normalized": metrics.get(
                "query_rewrite_result", diagnostic_context.get("query")
            ),
            "rewritten": {
                "applied": bool(metrics.get("query_rewrite_applied")),
                "result": metrics.get("query_rewrite_result"),
                "reason": metrics.get("query_rewrite_reason"),
            },
        },
        "config": {
            "neo4j_disabled": _neo4j_disabled,
            "hybrid_enabled": getattr(_config.search.hybrid, "enabled", True),
            "reranker": {
                "enabled": bool(
                    getattr(
                        getattr(_config.search.hybrid, "reranker", None),
                        "enabled",
                        False,
                    )
                ),
                "provider": getattr(
                    getattr(_config.search.hybrid, "reranker", None), "provider", None
                ),
                "model": getattr(
                    getattr(_config.search.hybrid, "reranker", None), "model", None
                ),
            },
            "colbert": {
                "enabled": bool(
                    getattr(
                        getattr(_config.search.vector, "qdrant", None),
                        "enable_colbert",
                        False,
                    )
                ),
                "scoring": "maxsim",
            },
            "fusion": {
                "method": metrics.get("fusion_method")
                or getattr(_config.search.hybrid, "method", None),
                "params": {"k": getattr(_config.search.hybrid, "rrf_k", 60)},
            },
        },
        "timing_ms": {
            "bm25": metrics.get("bm25_time_ms", 0.0),
            "vector_search": metrics.get("vec_time_ms", 0.0),
            "fusion": metrics.get("fusion_time_ms", 0.0),
            "rerank": metrics.get("reranker_time_ms", 0.0),
            "graph_expansion": metrics.get("expansion_time_ms", 0.0),
            "context_assembly": metrics.get("context_assembly_ms", 0.0),
            "total": metrics.get("total_time_ms", 0.0),
        },
        "counts": {
            "candidates_initial": candidates_initial,
            "candidates_post_filter": candidates_post_filter,
            "candidates_post_dedupe": candidates_post_dedupe,
            "returned": len(results),
            "dropped": {
                "dedupe": duplicates,
                "veto": 0,
                "scope_mismatch": 0,
            },
        },
        "results": results,
        "budgets": {
            "response_bytes": bytes_estimate,
            "tokens_estimate": tokens_estimate,
            "partial": partial,
            "limit_reason": limit_reason,
        },
    }

    diag = await _DIAGNOSTIC_EMITTER.emit(record)
    if not diag:
        return None

    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("retrieval.diagnostic_id", diag["diagnostic_id"])
            span.set_attribute("retrieval.transport", record["transport"])
            span.set_attribute("retrieval.mode", diagnostic_context.get("mode"))
            span.set_attribute("retrieval.top_k", diagnostic_context.get("top_k"))
            span.set_attribute("retrieval.returned_count", len(results))
            span.set_attribute("retrieval.dedupe_dropped", duplicates)
            span.set_attribute("retrieval.partial", partial)
            span.set_attribute("retrieval.limit_reason", limit_reason)
            span.set_attribute("retrieval.bm25_ms", record["timing_ms"]["bm25"])
            span.set_attribute(
                "retrieval.vector_ms", record["timing_ms"]["vector_search"]
            )
            span.set_attribute("retrieval.fusion_ms", record["timing_ms"]["fusion"])
            span.set_attribute("retrieval.rerank_ms", record["timing_ms"]["rerank"])
            span.set_attribute(
                "retrieval.expansion_ms", record["timing_ms"]["graph_expansion"]
            )
            span.set_attribute("retrieval.total_ms", record["timing_ms"]["total"])
            for item in results[:3]:
                scores = item.get("scores", {}) or {}
                span.add_event(
                    "retrieval.candidate",
                    {
                        "rank": item.get("rank"),
                        "chunk_id": item.get("chunk_id"),
                        "source": item.get("source"),
                        "final_score": scores.get("fused"),
                        "rerank_score": scores.get("rerank"),
                        "doc_tag": item.get("doc_tag"),
                    },
                )

    return diag
