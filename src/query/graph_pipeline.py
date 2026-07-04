# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade wrappers)
# =============================================================================
"""
Graph pipeline: Neo4j/Cypher queries, RELATED_TO signals, graph channel,
and graph enrichment.

Extracted from hybrid_retrieval.py to isolate all graph-related retrieval
logic behind a stable interface.
"""

import json
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.query.entity_extraction import EntityExtractor

if TYPE_CHECKING:
    from src.query.query_intent import QueryIntent
from src.query.retrieval_types import ChunkResult
from src.shared.observability import get_logger

logger = get_logger(__name__)

# Per-type scaling factors for RELATED_TO blending lambda.
# 1.0 = full base_lambda, 0.0 = disabled.
_RELATED_TO_SCALE = {
    "conceptual": 1.0,
    "config": 1.0,
    "procedural": 0.67,
    "troubleshooting": 0.67,
    "reference": 0.67,
    "cli": 0.0,  # CLI queries unaffected
}


# ── 1. chunk_from_props (pure function, no owner) ───────────────────────────


def chunk_from_props(props: Dict[str, Any]) -> ChunkResult:
    boundaries = props.get("boundaries_json", "{}")
    if isinstance(boundaries, dict):
        boundaries_json = json.dumps(boundaries, separators=(",", ":"))
    else:
        boundaries_json = boundaries or "{}"
    path_nodes = props.get("graph_path")
    if path_nodes and not isinstance(path_nodes, list):
        path_nodes = [path_nodes]
    return ChunkResult(
        chunk_id=props.get("id"),
        document_id=props.get("document_id", ""),
        parent_section_id=props.get("parent_section_id", ""),
        order=int(props.get("order", 0)),
        level=int(props.get("level", 3)),
        heading=props.get("heading", ""),
        text=props.get("text", ""),
        token_count=int(props.get("token_count", 0)),
        is_combined=bool(props.get("is_combined", False)),
        is_split=bool(props.get("is_split", False)),
        original_section_ids=props.get("original_section_ids", []),
        boundaries_json=boundaries_json,
        doc_tag=props.get("doc_tag"),
        document_total_tokens=int(props.get("document_total_tokens", 0)),
        source_path=props.get("source_path"),
        is_microdoc=bool(props.get("is_microdoc", False)),
        doc_is_microdoc=bool(props.get("doc_is_microdoc", False)),
        is_microdoc_stub=bool(props.get("is_microdoc_stub", False)),
        embedding_version=props.get("embedding_version"),
        tenant=props.get("tenant"),
        citation_labels=[],
        graph_distance=int(props.get("graph_distance", 0)),
        graph_score=float(props.get("graph_score", 0.0)),
        graph_path=[str(node) for node in path_nodes] if path_nodes else None,
        connection_count=int(props.get("connection_count", 0)),
        mention_count=int(props.get("mention_count", 0)),
    )


# ── 2. relationships_for_query ──────────────────────────────────────────────


def relationships_for_query(owner, query: str) -> Tuple[List[str], int]:
    """Select relationship set and neighbor cap based on query type."""
    qtype = owner._classify_query_type(query)
    return relationships_for_query_type(owner, qtype)


# ── 3. relationships_for_query_type ─────────────────────────────────────────


def relationships_for_query_type(owner, query_type: str) -> Tuple[List[str], int]:
    """Type-safe variant that accepts a pre-classified query type directly.

    Avoids re-classifying an already-classified type string as raw query text.
    """
    rels = list(owner.graph_relationships)
    cap = owner.graph_max_related
    if not owner.graph_adaptive_enabled:
        return rels, cap

    # Optional config-driven relationships per query type
    rels_cfg = getattr(
        getattr(owner.config.search, "hybrid", None), "query_type_relationships", {}
    )
    if rels_cfg:
        rels_override = rels_cfg.get(query_type)
        if rels_override:
            return rels_override, cap

    # Phase 3: Added REFERENCES for cross-document traversal
    # Fix #4: Cross-doc REFERENCES traversal implemented via pattern:
    # (seed:Chunk) → [:REFERENCES] → (doc:Document) → [:HAS_CHUNK] → ...
    #
    # Phase 3.5: MENTIONS replaces MENTIONED_IN (single canonical direction)
    # Direction-agnostic queries work with either direction
    # Weight allocation (70/30 split) keeps entity signals primary.
    if query_type == "conceptual":
        return ["MENTIONS", "DEFINES", "IN_SECTION", "REFERENCES"], cap
    if query_type == "cli":
        # L4: CLI queries excluded from REFERENCES - CLI commands are self-contained
        # and cross-document references provide minimal value for command lookups
        return ["MENTIONS", "CONTAINS_STEP", "HAS_PARAMETER"], cap
    if query_type == "config":
        return ["MENTIONS", "HAS_PARAMETER", "DEFINES", "REFERENCES"], cap
    if query_type == "procedural":
        return [
            "MENTIONS",
            "CONTAINS_STEP",
            "NEXT_CHUNK",
            "IN_SECTION",
            "REFERENCES",
        ], min(cap, 10)
    # Phase 2 Cleanup: Removed AFFECTS, CAUSED_BY (never materialized)
    if query_type == "troubleshooting":
        return ["MENTIONS", "RESOLVES", "NEXT_CHUNK", "REFERENCES"], min(cap, 15)
    if query_type == "reference":
        return ["MENTIONS", "NEXT_CHUNK", "REFERENCES"], min(cap, 5)
    return rels, cap


# ── 4. get_query_type_weights ───────────────────────────────────────────────


def get_query_type_weights(owner, query_type: str) -> Tuple[float, float]:
    weights_cfg = getattr(
        getattr(owner.config.search, "hybrid", None), "query_type_weights", {}
    )
    weights = weights_cfg.get(query_type) or {}
    return float(weights.get("vector", 0.7)), float(weights.get("graph", 0.3))


# ── 5. compute_graph_signals ────────────────────────────────────────────────


def compute_graph_signals(
    owner,
    entity_names: List[str],
    candidate_chunk_ids: List[str],
    query_type: str,
    doc_tag: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute graph signals for chunks using canonical_name matches.

    Uses direction-agnostic traversal (Neo4j best practice) so MENTIONS edges
    work regardless of direction. Neo4j traverses both directions with equal
    performance, so (e)-[r]-(c) finds edges whether stored as e->c or c->e.
    """
    # PHASE 1 VECTOR-ONLY: Skip graph signals when neo4j_disabled
    if not entity_names or not candidate_chunk_ids or owner.neo4j_disabled:
        return {}
    rels, _ = relationships_for_query_type(owner, query_type)
    cypher = """
    UNWIND $entity_names AS ename
    MATCH (e:Entity)
    WHERE toLower(coalesce(e.canonical_name, e.name, '')) = toLower(ename)
    MATCH (e)-[r]-(c:Chunk)
    WHERE type(r) IN $rel_types
      AND c.id IN $chunk_ids
      AND ($doc_tag IS NULL OR c.doc_tag = $doc_tag)
    RETURN c.id AS chunk_id,
           count(DISTINCT e) AS entity_count,
           collect(DISTINCT e.name)[..5] AS entity_names
    """
    signals: Dict[str, Dict[str, Any]] = {}
    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                cypher,
                entity_names=list(set(entity_names)),
                rel_types=rels,
                chunk_ids=list(set(candidate_chunk_ids)),
                doc_tag=doc_tag,
            )
            for record in result:
                chunk_id = str(record["chunk_id"])
                entity_count = float(record.get("entity_count") or 0.0)
                if owner._plan.graph_score_normalized_on:
                    score = 1.0 - math.exp(-entity_count / 3.0)
                else:
                    score = max(1.0, entity_count)
                signals[chunk_id] = {
                    "score": score,
                    "entity_count": int(entity_count),
                    "entities": record.get("entity_names") or [],
                }
    except Exception as exc:
        logger.warning("Graph signal computation failed", error=str(exc))
    return signals


# ── 6. compute_cross_doc_signals ────────────────────────────────────────────


def compute_cross_doc_signals(
    owner,
    candidate_chunk_ids: List[str],
    query_type: str,
    doc_tag: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute cross-document signals via REFERENCES edge traversal.

    Fix #4 (Phase 3): Implements the missing cross-document traversal that enables
    REFERENCES edges to boost relevance scores.

    The traversal pattern is:
    (seed:Chunk) → [:REFERENCES] → (doc:Document) → [:HAS_CHUNK] → (related:Chunk)

    If a candidate chunk's document is referenced by another candidate chunk,
    the related chunk gets a boost. This captures cross-document relevance
    signals that the entity-centric _compute_graph_signals cannot.

    Args:
        candidate_chunk_ids: List of chunk IDs from initial vector search
        query_type: Query classification (conceptual, procedural, etc.)
        doc_tag: Optional doc_tag filter

    Returns:
        Dict mapping chunk_id to {score, ref_count, ref_titles}
    """
    # PHASE 1 VECTOR-ONLY: Skip cross-doc signals when neo4j_disabled
    if not candidate_chunk_ids or owner.neo4j_disabled:
        return {}

    refs_cfg = getattr(owner.config, "references", None)
    refs_query_cfg = getattr(refs_cfg, "query", None) if refs_cfg else None
    if not refs_cfg or not getattr(refs_cfg, "enabled", False):
        return {}
    if refs_query_cfg and not getattr(refs_query_cfg, "enable_cross_doc_signals", True):
        return {}

    # Check if REFERENCES is enabled for this query type
    rels, _ = relationships_for_query_type(owner, query_type)
    if "REFERENCES" not in rels:
        return {}

    # Cypher: find chunks whose docs are referenced by other candidates
    cypher = """
    // Start with candidate chunks that have REFERENCES edges
    UNWIND $chunk_ids AS seed_id
    MATCH (seed:Chunk {id: seed_id})
    // Follow REFERENCES directly from the chunk to target document
    MATCH (seed)-[:REFERENCES]->(ref_doc:Document)
    WHERE NOT ref_doc:GhostDocument
    // Get chunks from referenced document that are also candidates
    MATCH (ref_doc)-[:HAS_CHUNK]->(related:Chunk)
    WHERE related.id IN $chunk_ids
      AND related.id <> seed_id
      AND ($doc_tag IS NULL OR related.doc_tag = $doc_tag)
    // Aggregate: count candidates referencing each related chunk's doc
    RETURN related.id AS chunk_id,
           count(DISTINCT seed) AS ref_count,
           collect(DISTINCT ref_doc.title)[..3] AS referencing_docs
    """
    signals: Dict[str, Dict[str, Any]] = {}
    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                cypher,
                chunk_ids=list(set(candidate_chunk_ids)),
                doc_tag=doc_tag,
            )
            for record in result:
                chunk_id = str(record["chunk_id"])
                ref_count = float(record.get("ref_count") or 0.0)
                # Score: diminishing returns for multiple references
                # 1 ref = 0.5, 2 refs = 0.75, 3+ refs = ~0.875
                score = 1.0 - math.exp(-ref_count / 2.0)
                signals[chunk_id] = {
                    "score": score,
                    "ref_count": int(ref_count),
                    "referencing_docs": record.get("referencing_docs") or [],
                }
            if signals:
                logger.debug(
                    "cross_doc_signals_computed",
                    chunk_count=len(signals),
                    query_type=query_type,
                )
    except Exception as exc:
        logger.warning("Cross-doc signal computation failed", error=str(exc))
    return signals


# ── 7. compute_related_to_doc_signals ───────────────────────────────────────


def compute_related_to_doc_signals(
    owner,
    seed_doc_ids: List[str],
    doc_tag: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute RELATED_TO signals at document level.

    Uses coalesce(score_final, colbert_score, score) for backward compatibility
    with both current (v1) and v2 RELATED_TO edges.

    Args:
        seed_doc_ids: Document IDs from top fused results to use as seeds.
        doc_tag: Optional doc_tag filter for target documents.

    Returns:
        Dict mapping target_doc_id to {edge_score, prior_ref, prior_ent,
        prior_tax, is_mutual, quality_tier, seed_sources}.
    """
    if not seed_doc_ids or owner.neo4j_disabled:
        return {}

    # Gate absorbed by plan (use_related_to_expansion). Caller checks plan.
    refs_cfg = getattr(owner.config, "references", None)
    refs_query_cfg = getattr(refs_cfg, "query", None) if refs_cfg else None

    min_edge_score = (
        getattr(refs_query_cfg, "related_to_min_edge_score", 0.025)
        if refs_query_cfg
        else 0.025
    )
    max_docs = (
        getattr(refs_query_cfg, "related_to_max_docs", 3) if refs_query_cfg else 3
    )

    cypher = """
    UNWIND $seed_doc_ids AS seed_id
    MATCH (seed:Document {id: seed_id})-[r:RELATED_TO]->(target:Document)
    WHERE ($doc_tag IS NULL OR target.doc_tag = $doc_tag)
      AND coalesce(r.score_final, r.colbert_score, r.score, 0.0) >= $min_edge_score
    WITH target,
         max(coalesce(r.score_final, r.colbert_score, r.score, 0.0)) AS edge_score,
         max(coalesce(r.prior_reference, 0.0)) AS prior_ref,
         max(coalesce(r.prior_entity, 0.0)) AS prior_ent,
         max(coalesce(r.prior_taxonomy, 0.0)) AS prior_tax,
         any(x IN collect(r.is_mutual) WHERE x = true) AS is_mutual,
         head(collect(r.quality_tier)) AS quality_tier,
         collect(DISTINCT seed_id) AS seed_sources
    RETURN target.id AS target_doc_id,
           edge_score, prior_ref, prior_ent, prior_tax,
           is_mutual, quality_tier, seed_sources
    ORDER BY edge_score DESC
    LIMIT $max_docs
    """
    signals: Dict[str, Dict[str, Any]] = {}
    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                cypher,
                seed_doc_ids=list(set(seed_doc_ids)),
                doc_tag=doc_tag,
                min_edge_score=min_edge_score,
                max_docs=max_docs,
            )
            for record in result:
                target_doc_id = str(record["target_doc_id"])
                signals[target_doc_id] = {
                    "edge_score": float(record.get("edge_score") or 0.0),
                    "prior_ref": float(record.get("prior_ref") or 0.0),
                    "prior_ent": float(record.get("prior_ent") or 0.0),
                    "prior_tax": float(record.get("prior_tax") or 0.0),
                    "is_mutual": bool(record.get("is_mutual")),
                    "quality_tier": record.get("quality_tier") or "medium",
                    "seed_sources": record.get("seed_sources") or [],
                }
            if signals:
                logger.debug(
                    "related_to_doc_signals_computed",
                    doc_count=len(signals),
                    seed_count=len(seed_doc_ids),
                )
    except Exception as exc:
        logger.warning("RELATED_TO doc signal computation failed", error=str(exc))
    return signals


# ── 8. expand_from_related_docs ─────────────────────────────────────────────


def expand_from_related_docs(
    owner,
    fused_results: List[ChunkResult],
    query: str,
    lexical_query: Optional[str],
    filters: Optional[Dict[str, Any]],
    doc_tag: Optional[str],
    metrics: Dict[str, Any],
) -> List[ChunkResult]:
    """Expand candidates with chunks from RELATED_TO documents.

    1. Pick top N seed docs from fused_results
    2. Fetch related docs via _compute_related_to_doc_signals()
    3. Run vector search constrained to related doc_ids
    4. Annotate each chunk with related_to_* scores
    5. Merge + dedup into fused_results
    """
    # Gate absorbed by plan (use_related_to_expansion). Caller checks plan.
    refs_cfg = getattr(owner.config, "references", None)
    refs_query_cfg = getattr(refs_cfg, "query", None) if refs_cfg else None

    seed_docs_limit = (
        getattr(refs_query_cfg, "related_to_seed_docs", 5) if refs_query_cfg else 5
    )
    max_docs = (
        getattr(refs_query_cfg, "related_to_max_docs", 3) if refs_query_cfg else 3
    )
    chunks_per_doc = (
        getattr(refs_query_cfg, "related_to_chunks_per_doc", 3) if refs_query_cfg else 3
    )

    # 1. Extract unique doc_ids from top fused results
    seed_doc_ids: List[str] = []
    seen_doc_ids: set = set()
    for r in fused_results[:20]:
        if r.document_id and r.document_id not in seen_doc_ids:
            seen_doc_ids.add(r.document_id)
            seed_doc_ids.append(r.document_id)
            if len(seed_doc_ids) >= seed_docs_limit:
                break

    if not seed_doc_ids:
        return fused_results

    # 2. Fetch related docs from Neo4j
    related_docs = compute_related_to_doc_signals(owner, seed_doc_ids, doc_tag)
    metrics["related_to_seed_docs"] = len(seed_doc_ids)
    metrics["related_to_docs_found"] = len(related_docs)

    if not related_docs:
        metrics["related_to_chunks_added"] = 0
        return fused_results

    # 3. Vector search constrained to related doc_ids
    related_doc_ids = list(related_docs.keys())[:max_docs]
    doc_filter = dict(filters or {})
    doc_filter["document_id"] = related_doc_ids
    fetch_limit = chunks_per_doc * len(related_doc_ids)

    try:
        related_chunks = owner.vector_retriever.search(
            query,
            fetch_limit,
            doc_filter,
            lexical_query=lexical_query if lexical_query != query else None,
        )
    except Exception as exc:
        logger.warning("RELATED_TO chunk expansion failed", error=str(exc))
        metrics["related_to_chunks_added"] = 0
        return fused_results

    # 4. Annotate each chunk with RELATED_TO scores
    for chunk in related_chunks:
        doc_signal = related_docs.get(chunk.document_id, {})
        chunk.related_to_edge_score = doc_signal.get("edge_score", 0.0)
        prior = 0.35 * max(
            doc_signal.get("prior_ref", 0.0),
            doc_signal.get("prior_ent", 0.0),
            doc_signal.get("prior_tax", 0.0),
        )
        chunk.related_to_prior_score = prior
        mutual_bonus = 1.1 if doc_signal.get("is_mutual") else 1.0
        quality_bonus = {"high": 1.15, "medium": 1.0, "low": 0.85}.get(
            doc_signal.get("quality_tier", "medium"), 1.0
        )
        chunk.related_to_score = (
            chunk.related_to_edge_score * (1 + prior) * mutual_bonus * quality_bonus
        )
        chunk.related_to_source_doc = chunk.document_id

    # 5. Merge + dedup (existing chunk_ids take priority)
    existing_ids = {r.chunk_id for r in fused_results}
    new_chunks = [c for c in related_chunks if c.chunk_id not in existing_ids]
    fused_results.extend(new_chunks)
    metrics["related_to_chunks_added"] = len(new_chunks)

    if new_chunks:
        avg_edge = sum(c.related_to_edge_score or 0.0 for c in new_chunks) / len(
            new_chunks
        )
        metrics["related_to_avg_edge_score"] = round(avg_edge, 4)
        logger.info(
            "related_to_expansion_complete",
            seed_docs=len(seed_doc_ids),
            related_docs=len(related_doc_ids),
            chunks_added=len(new_chunks),
            avg_edge_score=round(avg_edge, 4),
        )

    return fused_results


# ── 9. blend_related_to_scores ──────────────────────────────────────────────


def blend_related_to_scores(
    owner,
    fused_results: List[ChunkResult],
    query_type: str,
    metrics: Dict[str, Any],
) -> None:
    """Blend RELATED_TO scores into fused_score for related-doc chunks.

    Standalone step extracted from _apply_graph_reranker to decouple
    RELATED_TO blending from the entity graph channel gate.

    Modifies fused_results in-place.
    """
    refs_cfg = getattr(owner.config, "references", None)
    refs_query_cfg = getattr(refs_cfg, "query", None) if refs_cfg else None

    base_lambda = (
        getattr(refs_query_cfg, "related_to_weight_ratio", 0.15)
        if refs_query_cfg
        else 0.15
    )
    related_to_lambda = base_lambda * _RELATED_TO_SCALE.get(query_type, 0.67)

    if related_to_lambda <= 0:
        metrics["related_to_blend_lambda"] = 0.0
        metrics["related_to_blend_count"] = 0
        return

    related_to_blended = 0
    for r in fused_results:
        related = r.related_to_score or 0.0
        if related > 0:
            base_fused = r.fused_score or 0.0
            r.fused_score = (
                1 - related_to_lambda
            ) * base_fused + related_to_lambda * related
            related_to_blended += 1

    metrics["related_to_blend_lambda"] = related_to_lambda
    metrics["related_to_blend_count"] = related_to_blended

    if related_to_blended > 0:
        logger.info(
            "related_to_blending_applied",
            query_type=query_type,
            blend_lambda=round(related_to_lambda, 4),
            chunks_blended=related_to_blended,
        )


# ── 10. apply_graph_reranker ────────────────────────────────────────────────


def apply_graph_reranker(
    owner,
    query: str,
    doc_tag: Optional[str],
    vector_results: List[ChunkResult],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Graph-as-reranker: apply graph signals to vector candidates only.

    Fix #4 (Phase 3): Now includes cross-document REFERENCES traversal signals
    in addition to entity-based signals. The final graph score is a weighted
    combination of both signal types.
    """
    stats: Dict[str, Any] = {
        "graph_channel_candidates": 0,
        "graph_channel_entities": 0,
        "graph_reranker_applied": False,
        "graph_rerank_avg_delta": 0.0,
        "cross_doc_candidates": 0,  # Fix #4: Track cross-doc signal count
    }
    # PHASE 1 VECTOR-ONLY: Skip graph reranker when neo4j_disabled
    if not vector_results or owner.neo4j_disabled:
        return stats
    qtype = owner._classify_query_type(query)

    # Cap candidates to avoid explosion
    candidate_ids = [str(r.chunk_id) for r in vector_results if r.chunk_id][:50]

    # Fix #4: Compute cross-document signals (no entity extraction needed)
    # REFERENCES edges boost relevance even when entity extraction fails
    cross_doc_signals = compute_cross_doc_signals(
        owner,
        candidate_chunk_ids=candidate_ids,
        query_type=qtype,
        doc_tag=doc_tag,
    )
    stats["cross_doc_candidates"] = len(cross_doc_signals)

    # If cross-doc signals are strong enough, skip entity extraction to save time
    if cross_doc_signals and len(cross_doc_signals) >= max(1, len(candidate_ids) // 2):
        entities = []
    else:
        # Extract entities for traditional entity-based graph signals
        extractor = get_entity_extractor(owner)
        entities = extractor.extract_entities(query)
    if owner._plan.graph_garbage_filter_on:
        entities = [
            e
            for e in entities
            if isinstance(e, str)
            and len(e.strip()) >= 4
            and e.lower()
            not in {
                "at",
                "in",
                "id",
                "to",
                "of",
                "for",
                "the",
                "and",
                "or",
                "is",
                "it",
            }
        ]
    stats["graph_channel_entities"] = len(entities)

    # Compute entity-based graph signals (existing behavior)
    graph_signals: Dict[str, Dict[str, Any]] = {}
    if entities:
        # Simple guardrail: if entities*candidates too large, trim candidates
        entity_candidate_ids = candidate_ids
        if len(entities) * len(candidate_ids) > 500:
            entity_candidate_ids = candidate_ids[: max(1, 500 // max(len(entities), 1))]
        graph_signals = compute_graph_signals(
            owner,
            entity_names=entities,
            candidate_chunk_ids=entity_candidate_ids,
            query_type=qtype,
            doc_tag=doc_tag,
        )
    stats["graph_channel_candidates"] = len(graph_signals)

    # If neither entity signals nor cross-doc signals exist, nothing to blend
    if not graph_signals and not cross_doc_signals:
        return stats

    w_vec, w_graph = get_query_type_weights(owner, qtype)

    # Config-driven weight allocation for graph signals
    refs_cfg = getattr(owner.config, "references", None)
    refs_query_cfg = getattr(refs_cfg, "query", None) if refs_cfg else None
    cross_doc_ratio = (
        getattr(refs_query_cfg, "cross_doc_weight_ratio", 0.3)
        if refs_query_cfg
        else 0.3
    )
    w_entity = w_graph * (1.0 - cross_doc_ratio)
    w_cross_doc = w_graph * cross_doc_ratio

    # Track pre-rerank positions to compute delta
    pre_ranks = {str(r.chunk_id): idx for idx, r in enumerate(vector_results)}
    for r in vector_results:
        chunk_id = str(r.chunk_id)
        entity_sig = graph_signals.get(chunk_id)
        cross_doc_sig = cross_doc_signals.get(chunk_id)

        entity_score = entity_sig.get("score", 0.0) if entity_sig else 0.0
        cross_doc_score = cross_doc_sig.get("score", 0.0) if cross_doc_sig else 0.0

        # Combined graph score (for metrics/debugging)
        r.graph_score = entity_score + cross_doc_score

        # Preserve prior fused ranking (which already encodes sparse + dense RRF),
        # then apply graph as a bounded blend. Do NOT fall back to vector-only.
        base_fused = (
            r.fused_score if r.fused_score is not None else (r.vector_score or 0.0)
        )
        if entity_sig or cross_doc_sig:
            graph_component = (w_entity * entity_score) + (
                w_cross_doc * cross_doc_score
            )
            r.fused_score = ((1.0 - w_graph) * base_fused) + graph_component
        else:
            r.fused_score = base_fused

        # NOTE: RELATED_TO blending removed — now a standalone step
        # in _blend_related_to_scores(), called before signal pool.

    vector_results.sort(key=lambda x: x.fused_score or 0.0, reverse=True)

    # Compute simple rerank delta metrics
    deltas = []
    for idx, r in enumerate(vector_results):
        old = pre_ranks.get(str(r.chunk_id))
        if old is not None:
            deltas.append(old - idx)
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
    else:
        avg_delta = 0.0
    stats["graph_reranker_applied"] = True
    stats["graph_rerank_avg_delta"] = avg_delta
    logger.info(
        "graph_reranker_applied",
        extra={
            "query_preview": query[:80],
            "query_type": qtype,
            "entities": len(entities),
            "graph_candidates": len(graph_signals),
            "cross_doc_candidates": len(cross_doc_signals),
            "avg_rank_delta": avg_delta,
        },
    )
    return stats


# ── 11. get_entity_extractor ────────────────────────────────────────────────


def get_entity_extractor(owner) -> EntityExtractor:
    if owner._entity_extractor:
        return owner._entity_extractor
    owner._entity_extractor = EntityExtractor(owner.neo4j_driver)
    return owner._entity_extractor


# ── 12. apply_graph_enrichment ──────────────────────────────────────────────


def apply_graph_enrichment(
    owner,
    seeds: List[ChunkResult],
    current_results: List[ChunkResult],
    doc_tag: Optional[str],
) -> Tuple[List[ChunkResult], Dict[str, int]]:
    """Expand results with graph neighbors and annotate graph scores."""
    stats = {
        "graph_neighbors_considered": 0,
        "graph_neighbors_added": 0,
    }
    # PHASE 1 VECTOR-ONLY: Skip graph enrichment when neo4j_disabled
    if not owner._plan.use_graph_enrichment or not seeds or owner.neo4j_disabled:
        return [], stats

    neighbors = fetch_graph_neighbors(owner, seeds, doc_tag=doc_tag)
    stats["graph_neighbors_considered"] = len(neighbors)
    if not neighbors:
        return [], stats

    existing = {chunk.chunk_id: chunk for chunk in current_results}
    added: List[ChunkResult] = []

    for neighbor in neighbors:
        current = existing.get(neighbor.chunk_id)
        if current:
            current.graph_score = max(current.graph_score, neighbor.graph_score)
            if not current.graph_distance or (
                neighbor.graph_distance
                and neighbor.graph_distance < current.graph_distance
            ):
                current.graph_distance = neighbor.graph_distance
                current.graph_path = neighbor.graph_path
            if not current.expansion_source:
                current.expansion_source = neighbor.expansion_source
            continue
        added.append(neighbor)
        existing[neighbor.chunk_id] = neighbor

    stats["graph_neighbors_added"] = len(added)
    return added, stats


# ── 13. graph_retrieval_channel ─────────────────────────────────────────────


def graph_retrieval_channel(
    owner,
    query: str,
    doc_tag: Optional[str],
    *,
    intent: Optional["QueryIntent"] = None,
) -> Tuple[List[ChunkResult], Dict[str, int]]:
    """Entity-anchored graph retrieval channel (flagged)."""
    stats = {
        "graph_channel_candidates": 0,
        "graph_channel_entities": 0,
        "entity_anchors_found": 0,
        "graph_edges_traversed": 0,
        "graph_relationship_types": [],
        "graph_channel_raw_rows": 0,
        "graph_channel_post_support_chunks": 0,
        "graph_channel_post_sparse_chunks": 0,
        "graph_anchor_sources": {},
    }
    # Decoupled: graph channel only requires its own plan flag + neo4j.
    # No longer coupled to graph_enabled (enrichment).
    if not owner._plan.use_entity_graph_channel or owner.neo4j_disabled:
        return [], stats

    # Build graph anchors from two sources:
    # 1. GLiNER entity extraction (NER-based)
    # 2. QueryIntent.primary_anchors (precision terms like "metadata")
    extractor = get_entity_extractor(owner)
    extracted_entities = extractor.extract_entities(query)

    # Inject precision anchors alongside extracted entities
    precision_anchors: List[str] = []
    if intent is not None and intent.primary_anchors:
        precision_anchors = list(intent.primary_anchors)

    # Union and deduplicate (case-insensitive), then filter domain-generic
    # entities that would match too broadly (e.g., "nutanix", "cluster",
    # "nutanix cluster"). Uses the same exclusion list as ingestion.
    from src.providers.ner.labels import is_excluded_entity

    seen_lower: set = set()
    entities: List[str] = []
    for e in list(extracted_entities) + precision_anchors:
        if not isinstance(e, str):
            continue
        low = e.lower().strip()
        if low and low not in seen_lower:
            seen_lower.add(low)
            # Exclude domain-generic terms and their compounds
            # Split compounds: "nutanix cluster" -> check each word.
            words = low.split()
            if not is_excluded_entity(e) and not any(
                is_excluded_entity(w) for w in words
            ):
                entities.append(e)

    pre_filter_count = len(entities)
    if owner._plan.graph_garbage_filter_on:
        entities = [
            e
            for e in entities
            if isinstance(e, str)
            and len(e.strip()) >= 4
            and e.lower()
            not in {
                "at",
                "in",
                "id",
                "to",
                "of",
                "for",
                "the",
                "and",
                "or",
                "is",
                "it",
            }
        ]
    stats["graph_anchor_sources"] = {
        "extracted": len(extracted_entities),
        "precision_anchors": precision_anchors,
        "merged": len(entities),
        "pre_garbage_filter": pre_filter_count,
        "excluded_count": len(seen_lower) - len(entities),
    }
    stats["graph_channel_entities"] = len(entities)
    if not entities:
        return [], stats
    rels, max_related = relationships_for_query(owner, query)
    limit_per_entity = max(1, min(max_related, 50))
    # Direction-agnostic: MENTIONS is stored as (Chunk)-[:MENTIONS]->(Entity),
    # while DEFINES is (Entity)-[:DEFINES]->(Chunk). Using (e)-[r]-(c)
    # matches both directions without duplicating the query.
    cypher = """
    UNWIND $entities AS name
    WITH trim(toLower(name)) AS anchor
    MATCH (e:Entity)
    WITH anchor, e, toLower(coalesce(e.canonical_name, e.name, '')) AS entity_name
    WHERE entity_name = anchor
       OR (size(anchor) >= 6 AND entity_name CONTAINS anchor)
    MATCH (e)-[r]-(c:Chunk)
    WHERE (
        $use_rel_types = false
        OR type(r) IN $rel_types
    )
    AND ($doc_tag IS NULL OR c.doc_tag = $doc_tag)
    WITH c,
         collect(DISTINCT anchor) AS matched_anchors,
         count(DISTINCT e) AS entity_count,
         count(DISTINCT r) AS edge_count,
         collect(DISTINCT type(r)) AS rel_types_collected
    WITH c, matched_anchors, entity_count, edge_count, rel_types_collected
    RETURN c {
        .id,
        .document_id,
        .parent_section_id,
        .order,
        .level,
        .heading,
        .text,
        token_count: coalesce(c.token_count, c.tokens, 0),
        .doc_tag,
        .document_total_tokens,
        .source_path,
        .is_microdoc,
        .doc_is_microdoc,
        .is_microdoc_stub,
        .embedding_version,
        .tenant
    } AS props,
    matched_anchors,
    size(matched_anchors) AS anchor_count,
    entity_count,
    edge_count,
    rel_types_collected AS rel_types
    ORDER BY anchor_count DESC, entity_count DESC, edge_count DESC, c.id
    LIMIT $limit
    """
    chunks: List[ChunkResult] = []
    entity_list = list(set(entities))
    logger.info(
        "graph_channel_query",
        entities=entity_list,
        rel_types=rels,
        use_rel_types=owner._plan.use_entity_graph_channel,
        doc_tag=doc_tag,
        limit=limit_per_entity,
    )
    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                cypher,
                entities=entity_list,
                doc_tag=doc_tag,
                limit=limit_per_entity,
                rel_types=rels,
                use_rel_types=owner._plan.use_entity_graph_channel,
            )
            records = list(result)
            stats["graph_channel_raw_rows"] = len(records)
            logger.info(
                "graph_channel_raw_results",
                record_count=len(records),
            )
            anchors_found: set = set()
            rel_types_used: set = set()
            edges_traversed = 0
            support_threshold = 2 if owner._plan.graph_garbage_filter_on else 1
            for record in records:
                matched_anchors = [
                    str(anchor).strip().lower()
                    for anchor in (record.get("matched_anchors") or [])
                    if str(anchor).strip()
                ]
                anchors_found.update(matched_anchors)
                anchor_count = int(record.get("anchor_count") or 0)
                entity_count = int(record.get("entity_count") or 0)
                support_count = max(anchor_count, entity_count)
                if support_count < support_threshold:
                    continue
                props = record["props"] or {}
                chunk = chunk_from_props(props)
                edge_count = int(record.get("edge_count") or 0)
                rel_types = record.get("rel_types") or []
                for rel_type in rel_types:
                    if rel_type:
                        rel_types_used.add(str(rel_type))
                edges_traversed += max(0, edge_count)
                if owner._plan.graph_score_normalized_on:
                    chunk.graph_score = 1.0 - math.exp(-support_count / 3.0)
                else:
                    chunk.graph_score = float(max(1, support_count))
                # Do not overwrite vector_score; keep graph score separate
                chunk.fused_score = chunk.graph_score
                chunk.vector_score_kind = "graph_entity"
                chunks.append(chunk)
            stats["entity_anchors_found"] = len(anchors_found)
            stats["graph_edges_traversed"] = edges_traversed
            stats["graph_relationship_types"] = sorted(rel_types_used)
            stats["graph_channel_post_support_chunks"] = len(chunks)
    except Exception as exc:
        logger.warning("Graph retrieval channel failed", error=str(exc))
        return [], stats
    if not chunks:
        return [], stats

    # Sparse gating via kg_id if available
    sparse_query = owner.vector_retriever._build_sparse_query(query)
    sparse_gate_debug = []
    sparse_gate_hits = 0
    if (
        sparse_query
        and isinstance(sparse_query, dict)
        and sparse_query.get("indices")
        and sparse_query.get("values")
    ):
        gated = owner._rescore_expansion_with_sparse(
            sparse_query["indices"],
            sparse_query["values"],
            chunks,
            owner.expansion_sparse_threshold,
        )
        if gated is not None:
            sparse_gate_hits = len(gated)
            allowed = {
                pid for pid, s in gated.items() if s >= owner.expansion_sparse_threshold
            }
            for chunk in chunks[:10]:
                chunk_id = str(chunk.chunk_id) if chunk.chunk_id else None
                score = gated.get(chunk_id) if chunk_id else None
                sparse_gate_debug.append(
                    {
                        "chunk_id": chunk_id,
                        "sparse_score": round(score, 4) if score is not None else None,
                        "passed": bool(chunk_id and chunk_id in allowed),
                    }
                )
            chunks = [c for c in chunks if str(c.chunk_id) in allowed]
    stats["graph_channel_sparse_hits"] = sparse_gate_hits
    stats["graph_channel_post_sparse_chunks"] = len(chunks)
    stats["graph_channel_candidates"] = len(chunks)
    logger.info(
        "graph_channel_invoked",
        extra={
            "query_preview": query[:80],
            "entities": stats.get("graph_channel_entities"),
            "candidates": stats.get("graph_channel_candidates"),
            "raw_rows": stats.get("graph_channel_raw_rows"),
            "post_support_chunks": stats.get("graph_channel_post_support_chunks"),
            "post_sparse_chunks": stats.get("graph_channel_post_sparse_chunks"),
            "sparse_hits": stats.get("graph_channel_sparse_hits", 0),
            "rels": rels,
        },
    )
    if stats.get("graph_channel_post_support_chunks", 0) and not stats.get(
        "graph_channel_post_sparse_chunks", 0
    ):
        logger.warning(
            "graph_channel_sparse_gate_zeroed",
            extra={
                "query_preview": query[:80],
                "threshold": owner.expansion_sparse_threshold,
                "post_support_chunks": stats.get(
                    "graph_channel_post_support_chunks", 0
                ),
                "sparse_hits": stats.get("graph_channel_sparse_hits", 0),
                "debug_candidates": sparse_gate_debug,
            },
        )
    return chunks, stats


# ── 14. fetch_graph_neighbors ───────────────────────────────────────────────


def fetch_graph_neighbors(
    owner, seeds: List[ChunkResult], doc_tag: Optional[str]
) -> List[ChunkResult]:
    """Fetch graph neighbors for the given seed chunks."""
    # PHASE 1 VECTOR-ONLY: Skip graph neighbors when neo4j_disabled
    if not owner._plan.use_graph_enrichment or owner.neo4j_disabled:
        return []

    seed_lookup = {seed.chunk_id: seed for seed in seeds if seed.chunk_id}
    if not seed_lookup:
        return []

    relationships, max_related = relationships_for_query(owner, owner._last_query_text)
    limit_per_seed = max(1, min(max_related, 200))
    rel_pattern = "|".join(relationships)
    query = f"""
    UNWIND $seed_ids AS seed_id
    MATCH (seed:Chunk {{id: seed_id}})
    CALL (seed) {{
        MATCH path=(seed)-[r:{rel_pattern}*1..{owner.graph_max_depth}]-(target:Chunk)
        WHERE seed.id <> target.id
          AND ($doc_tag IS NULL OR target.doc_tag = $doc_tag)
        WITH target, path
        ORDER BY length(path) ASC
        LIMIT $per_seed
        RETURN target, path, length(path) AS dist
    }}
    RETURN seed.id AS seed_id,
           target {{
               .id,
               .document_id,
               .parent_section_id,
               .order,
               .level,
               .heading,
               .text,
               token_count: coalesce(target.token_count, target.tokens, 0),
               .is_combined,
               .is_split,
               .original_section_ids,
               .boundaries_json,
               .doc_tag,
               .document_total_tokens,
               .source_path,
               .is_microdoc,
               .doc_is_microdoc,
               .is_microdoc_stub,
               .embedding_version,
               .tenant
           }} AS props,
           dist,
           [node IN nodes(path) | node.id] AS path_nodes
    """

    best_by_id: Dict[str, ChunkResult] = {}
    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                query,
                seed_ids=list(seed_lookup.keys()),
                per_seed=limit_per_seed,
                doc_tag=doc_tag,
                timeout=owner.expansion_timeout_seconds,
            )
            for record in result:
                seed_id = record["seed_id"]
                source_chunk = seed_lookup.get(seed_id)
                if not source_chunk:
                    continue

                props = record["props"] or {}
                if doc_tag and props.get("doc_tag") != doc_tag:
                    continue
                if (
                    source_chunk.document_id
                    and props.get("document_id")
                    and props["document_id"] != source_chunk.document_id
                ):
                    continue

                chunk = chunk_from_props(props)
                chunk.is_expanded = True
                chunk.expansion_source = seed_id

                distance = int(record.get("dist") or 1)
                chunk.graph_distance = max(1, distance)
                chunk.graph_score = 1.0 / (chunk.graph_distance + 1)
                path_nodes = record.get("path_nodes") or []
                chunk.graph_path = [seed_id] + [
                    str(node_id) for node_id in path_nodes if node_id != seed_id
                ]

                source_score = (
                    source_chunk.fused_score
                    or source_chunk.vector_score
                    or source_chunk.bm25_score
                    or 0.0
                )
                if source_chunk.rerank_score is not None:
                    try:
                        rerank_sem = 1.0 / (
                            1.0 + math.exp(-float(source_chunk.rerank_score))
                        )
                    except OverflowError:
                        rerank_sem = 0.0 if source_chunk.rerank_score < 0 else 1.0
                    source_score = max(source_score, rerank_sem)

                propagated = source_score * owner.graph_propagation_decay
                chunk.inherited_score = propagated
                chunk.fused_score = max(
                    chunk.fused_score or 0.0,
                    propagated,
                )
                chunk.vector_score = chunk.fused_score
                chunk.vector_score_kind = chunk.vector_score_kind or "graph_propagated"

                existing = best_by_id.get(chunk.chunk_id)
                if existing and existing.graph_score >= chunk.graph_score:
                    continue
                best_by_id[chunk.chunk_id] = chunk
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Graph enrichment failed (expansion)",
            extra={"error": str(exc)},
        )
        return []

    return list(best_by_id.values())


# ── 15. annotate_coverage ───────────────────────────────────────────────────


def annotate_coverage(owner, chunks: List[ChunkResult]) -> None:
    """Attach connection/mention counts used by ranker coverage features."""
    # PHASE 1 VECTOR-ONLY: Skip Neo4j queries when disabled
    if owner.neo4j_disabled:
        return

    ids = {chunk.chunk_id for chunk in chunks if chunk.chunk_id}
    if not ids:
        return

    coverage_query = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {id: cid})
    OPTIONAL MATCH (c)-[r]->()
    WITH c, count(DISTINCT r) AS conn_count
    OPTIONAL MATCH (c)-[:MENTIONS]->(e)
    RETURN c.id AS id,
           conn_count AS connection_count,
           count(DISTINCT e) AS mention_count
    """

    coverage = {}
    try:
        with owner.neo4j_driver.session() as session:
            records = session.run(coverage_query, ids=list(ids))
            for record in records:
                coverage[record["id"]] = {
                    "connection_count": record.get("connection_count", 0),
                    "mention_count": record.get("mention_count", 0),
                }
    except Exception as exc:
        logger.warning("Coverage enrichment failed: %s", exc)
        return

    for chunk in chunks:
        data = coverage.get(chunk.chunk_id)
        if not data:
            continue
        chunk.connection_count = int(data.get("connection_count") or 0)
        chunk.mention_count = int(data.get("mention_count") or 0)
