# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade wrappers), graph_pipeline.py (via facade)
# =============================================================================
"""
Expansion pipeline: bounded adjacency expansion, structure-aware expansion,
micro-doc stitching, context budget enforcement, and sparse gating.

Extracted from hybrid_retrieval.py to isolate all post-retrieval shaping
logic behind a stable interface.
"""

import math
import os
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client.http.models import SparseVector as QdrantSparseVector
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter as QdrantFilter
from qdrant_client.models import MatchAny

from src.query.retrieval_types import ChunkResult, _deduplicate_entity_metadata
from src.shared.observability import get_logger

logger = get_logger(__name__)


# ── Pure helpers (no owner) ──────────────────────────────────────────────────


def neighbor_score(source_score: float) -> float:
    if source_score <= 0:
        return 0.0
    epsilon = 1e-4
    half = source_score * 0.5
    return max(0.0, min(half, source_score - epsilon))


def result_id(r: ChunkResult) -> Tuple:
    """Get unique identifier for a chunk result."""
    return ("chunk_id", r.chunk_id)


def path_prefix(source_path: Optional[str], micro_dir_depth: int) -> Optional[str]:
    if not source_path:
        return None
    path = Path(source_path)
    parts = [p for p in path.parts if p not in ("", os.sep)]
    if not parts:
        normalized = source_path.replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
    if not parts:
        return None
    depth = min(micro_dir_depth, len(parts))
    return "/".join(parts[:depth])


# ── Owner-dependent functions ────────────────────────────────────────────────


def should_expand(
    owner, query: str, seeds: List[ChunkResult], when
) -> Tuple[bool, str, float, int]:
    """
    Determine if expansion should be triggered based on gating conditions.

    Returns: (triggered, reason, score_delta, query_tokens)
    """
    from src.query.retrieval_types import ExpandWhen

    query_tokens = owner.tokenizer.count_tokens(query)
    scores = [r.fused_score or 0.0 for r in seeds]

    # Calculate score delta between top results
    if len(scores) >= 2:
        score_delta = abs(scores[0] - scores[1])
    else:
        score_delta = 1.0  # Large delta if only one result

    reason = "none"
    triggered = False

    if when == ExpandWhen.NEVER:
        reason = "disabled"
    elif when == ExpandWhen.ALWAYS:
        triggered, reason = True, "forced"
    elif when == ExpandWhen.QUERY_LENGTH_ONLY:
        if query_tokens >= owner.expansion_query_min_tokens:
            triggered, reason = True, "query_long"
    else:  # AUTO (spec-compliant default)
        if query_tokens >= owner.expansion_query_min_tokens:
            triggered, reason = True, "query_long"
        elif len(scores) >= 2 and score_delta <= owner.expansion_score_delta_max:
            triggered, reason = True, "scores_close"

    return triggered, reason, score_delta, query_tokens


def dedup_results(owner, results: List[ChunkResult]) -> List[ChunkResult]:
    """Remove duplicate chunks by identity, optionally merging best scores."""
    from src.query.retrieval_types import dedup_chunk_results

    if not getattr(
        getattr(owner.config, "feature_flags", None), "dedup_best_score", False
    ):
        # Simple first-wins dedup when feature flag is off
        seen = set()
        deduped = []
        for r in results:
            rid = result_id(r)
            if rid not in seen:
                seen.add(rid)
                deduped.append(r)
        return deduped

    # Use config weights if available, else default to 0.7/0.3
    weights = getattr(
        getattr(owner.config.search, "hybrid", None), "query_type_weights", {}
    )
    wv, wg = 0.7, 0.3
    default_weights = weights.get("conceptual") or {}
    wv = float(default_weights.get("vector", wv))
    wg = float(default_weights.get("graph", wg))

    # Delegate to standalone function for testability
    return dedup_chunk_results(
        results,
        vector_weight=wv,
        graph_weight=wg,
        id_fn=lambda r: result_id(r),
    )


def build_expanded_chunk(
    owner,
    record: dict,
    source_chunk_id: str,
    context_source: str,
    source_score: float,
) -> ChunkResult:
    """Build a ChunkResult from a Neo4j record for structure expansion."""
    nscore = neighbor_score(source_score)
    return ChunkResult(
        chunk_id=record["chunk_id"],
        document_id=record["document_id"],
        parent_section_id=record["parent_section_id"],
        order=record["order"] or 0,
        level=record["level"] or 0,
        heading=record["heading"] or "",
        text=record["text"] or "",
        token_count=record["token_count"] or 0,
        is_combined=record.get("is_combined") or False,
        is_split=record.get("is_split") or False,
        original_section_ids=record.get("original_section_ids") or [],
        boundaries_json=record.get("boundaries_json") or "{}",
        doc_tag=record.get("doc_tag"),
        document_total_tokens=record.get("document_total_tokens", 0),
        source_path=record.get("source_path"),
        is_microdoc=record.get("is_microdoc", False),
        doc_is_microdoc=record.get("doc_is_microdoc", False),
        is_microdoc_stub=record.get("is_microdoc_stub", False),
        embedding_version=record.get("embedding_version"),
        tenant=record.get("tenant"),
        is_expanded=True,
        expansion_source=source_chunk_id,
        context_source=context_source,
        fused_score=nscore,
        citation_labels=[],
        graph_distance=1,
        graph_score=0.5,  # Lower than sequential expansion
    )


def bounded_expansion(owner, query: str, seeds: List[ChunkResult]) -> List[ChunkResult]:
    """
    Bounded adjacency expansion via NEXT_CHUNK relationships.

    Adds ±1 neighbors from the first N seeds (default N=5).
    Neighbors never outrank their source (score = source_score * 0.5).

    Reference: Phase 7E Canonical Spec L1425-1434
    """
    # PHASE 1 VECTOR-ONLY: Skip expansion when neo4j_disabled
    if not seeds or owner.neo4j_disabled:
        return []

    # Limit to first 5 seeds for expansion (bounded)
    max_sources = getattr(owner, "max_sources_to_expand", 5)
    eligible = seeds[:max_sources]

    # Track existing chunks to avoid duplicates
    seen_ids = {result_id(r) for r in seeds}
    expanded = []

    chunk_ids_to_expand = [r.chunk_id for r in eligible]
    seed_lookup = {seed.chunk_id: seed for seed in seeds}

    expansion_query = """
    UNWIND $chunk_ids AS chunk_id
    MATCH (c:Chunk {id: chunk_id})

    // Find previous chunk
    OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)

    // Find next chunk
    OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)

    WITH chunk_id,
         collect(DISTINCT prev) AS prev_chunks,
         collect(DISTINCT next) AS next_chunks

    UNWIND (prev_chunks + next_chunks) AS neighbor
    WITH chunk_id, neighbor
    WHERE neighbor IS NOT NULL

    RETURN DISTINCT
        neighbor.id AS chunk_id,
        neighbor.document_id AS document_id,
        neighbor.parent_section_id AS parent_section_id,
        neighbor.order AS `order`,
        neighbor.level AS level,
        neighbor.heading AS heading,
        neighbor.text AS text,
        neighbor.token_count AS token_count,
        neighbor.is_combined AS is_combined,
        neighbor.is_split AS is_split,
        neighbor.original_section_ids AS original_section_ids,
        neighbor.boundaries_json AS boundaries_json,
        neighbor.doc_tag AS doc_tag,
        neighbor.document_total_tokens AS document_total_tokens,
        neighbor.is_microdoc AS is_microdoc,
        neighbor.doc_is_microdoc AS doc_is_microdoc,
        neighbor.is_microdoc_stub AS is_microdoc_stub,
        neighbor.source_path AS source_path,
        neighbor.embedding_version AS embedding_version,
        neighbor.tenant AS tenant,
        chunk_id AS source_chunk
    """

    try:
        with owner.neo4j_driver.session() as session:
            result = session.run(
                expansion_query,
                chunk_ids=chunk_ids_to_expand,
                timeout=owner.expansion_timeout_seconds,
            )

            for record in result:
                neighbor_id = record["chunk_id"]
                rid = ("chunk_id", neighbor_id)

                if rid in seen_ids:
                    continue  # Skip duplicates

                seen_ids.add(rid)

                # Find source chunk to get its score
                source_chunk_id = record["source_chunk"]
                source_seed = seed_lookup.get(source_chunk_id)
                source_fused = (
                    float(source_seed.fused_score or 0.0) if source_seed else 0.0
                )
                source_rerank = (
                    float(source_seed.rerank_score)
                    if source_seed and source_seed.rerank_score is not None
                    else None
                )

                neighbor_fused_score = neighbor_score(source_fused)
                neighbor_rerank_score = (
                    neighbor_score(source_rerank) if source_rerank is not None else None
                )

                expanded.append(
                    ChunkResult(
                        chunk_id=neighbor_id,
                        document_id=record["document_id"],
                        parent_section_id=record["parent_section_id"],
                        order=record["order"],
                        level=record["level"],
                        heading=record["heading"] or "",
                        text=record["text"],
                        token_count=record["token_count"],
                        is_combined=record["is_combined"],
                        is_split=record["is_split"],
                        original_section_ids=record["original_section_ids"] or [],
                        boundaries_json=record["boundaries_json"] or "{}",
                        doc_tag=record.get("doc_tag"),
                        document_total_tokens=record.get("document_total_tokens", 0),
                        source_path=record.get("source_path"),
                        is_microdoc=record.get("is_microdoc", False),
                        doc_is_microdoc=record.get("doc_is_microdoc", False),
                        is_microdoc_stub=record.get("is_microdoc_stub", False),
                        embedding_version=record.get("embedding_version"),
                        tenant=record.get("tenant"),
                        is_expanded=True,
                        expansion_source=source_chunk_id,
                        context_source="sequential",  # C.4: Track expansion type
                        fused_score=neighbor_fused_score,
                        citation_labels=[],
                        graph_distance=1,
                        graph_score=1.0,
                        graph_path=[source_chunk_id, neighbor_id],
                    )
                )
                if neighbor_rerank_score is not None:
                    expanded[-1].rerank_score = neighbor_rerank_score

        # Calculate query tokens for logging
        query_token_count = owner.tokenizer.count_tokens(query)
        logger.info(
            f"Adjacency expansion: query_tokens={query_token_count}, "
            f"expanded={len(expanded)} chunks"
        )

    except Exception as e:
        logger.error(f"Adjacency expansion failed: {e}")
        # Don't fail the whole search if expansion fails

    if (
        expanded
        and owner.vector_retriever.supports_sparse
        and owner.expansion_sparse_threshold > 0
    ):
        expanded = gate_expansion_with_sparse(
            owner, query, expanded, owner.expansion_sparse_threshold
        )

    return expanded


def expand_with_structure(
    owner,
    query: str,
    seeds: List[ChunkResult],
    doc_tag: Optional[str] = None,
    *,
    force: bool = False,
) -> List[ChunkResult]:
    """
    Structure-aware context expansion (Phase C.4).

    Finds additional context chunks via:
    1. Sibling chunks - same parent_section_id (source="sibling")
    2. Parent section chunks - CHILD_OF traversal (source="parent_section")
    3. Shared-entity chunks - entities with top results (source="shared")

    Each expanded chunk gets:
    - is_expanded=True
    - expansion_source=seed_chunk_id
    - context_source=<type>

    Args:
        force: Bypass the structure_aware_expansion feature flag check.
               Used by the signal-diverse pool to run structural expansion
               pre-rerank without globally enabling post-rerank expansion.

    Returns combined list of expanded chunks (no duplicates).
    """
    # PHASE 1 VECTOR-ONLY: Skip structure expansion when neo4j_disabled
    if not seeds or owner.neo4j_disabled:
        return []

    # Check plan (bypass if force=True for signal pool pre-rerank)
    if not force and not owner._plan.use_structure_expansion:
        return []

    # Get config limits
    expansion_cfg = getattr(
        getattr(owner.config.search, "hybrid", None), "expansion", {}
    )
    structure_cfg = getattr(expansion_cfg, "structure", None) or {}
    if isinstance(structure_cfg, dict):
        sibling_limit = structure_cfg.get("sibling_limit", 3)
        parent_limit = structure_cfg.get("parent_section_limit", 2)
        entity_limit = structure_cfg.get("shared_entity_limit", 3)
        timeout_ms = structure_cfg.get("timeout_ms", 100)
    else:
        sibling_limit = getattr(structure_cfg, "sibling_limit", 3)
        parent_limit = getattr(structure_cfg, "parent_section_limit", 2)
        entity_limit = getattr(structure_cfg, "shared_entity_limit", 3)
        timeout_ms = getattr(structure_cfg, "timeout_ms", 100)

    timeout_seconds = timeout_ms / 1000.0
    seen_ids = {result_id(r) for r in seeds}
    expanded: List[ChunkResult] = []

    # Limit seeds for expansion to control latency
    max_sources = min(5, len(seeds))
    eligible = seeds[:max_sources]
    seed_lookup = {seed.chunk_id: seed for seed in seeds}

    # 1. Sibling expansion - chunks with same parent_section_id
    sibling_query = """
    UNWIND $parent_section_ids AS psid
    MATCH (c:Chunk {parent_section_id: psid})
    WHERE NOT c.id IN $exclude_ids
      AND ($doc_tag IS NULL OR c.doc_tag = $doc_tag)
    WITH psid, c
    ORDER BY c.order
    WITH psid, collect(c)[..$limit] AS siblings
    UNWIND siblings AS sibling
    RETURN DISTINCT
        sibling.id AS chunk_id,
        sibling.document_id AS document_id,
        sibling.parent_section_id AS parent_section_id,
        sibling.order AS `order`,
        sibling.level AS level,
        sibling.heading AS heading,
        sibling.text AS text,
        sibling.token_count AS token_count,
        sibling.is_combined AS is_combined,
        sibling.is_split AS is_split,
        sibling.original_section_ids AS original_section_ids,
        sibling.boundaries_json AS boundaries_json,
        sibling.doc_tag AS doc_tag,
        sibling.document_total_tokens AS document_total_tokens,
        sibling.source_path AS source_path,
        sibling.is_microdoc AS is_microdoc,
        sibling.doc_is_microdoc AS doc_is_microdoc,
        sibling.is_microdoc_stub AS is_microdoc_stub,
        sibling.embedding_version AS embedding_version,
        sibling.tenant AS tenant,
        psid AS source_parent_section
    """

    # 2. Parent chunk expansion - traverse CHILD_OF to find parent's chunks
    parent_query = """
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk {id: cid})
    OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Chunk)
    WITH cid, parent
    WHERE parent IS NOT NULL
    MATCH (sibling:Chunk)-[:IN_SECTION]->(parent)
    WHERE sibling.id <> cid
      AND ($doc_tag IS NULL OR sibling.doc_tag = $doc_tag)
    WITH cid, sibling
    ORDER BY sibling.order
    WITH cid, collect(sibling)[..$limit] AS parent_chunks
    UNWIND parent_chunks AS pc
    RETURN DISTINCT
        pc.id AS chunk_id,
        pc.document_id AS document_id,
        pc.parent_section_id AS parent_section_id,
        pc.order AS `order`,
        pc.level AS level,
        pc.heading AS heading,
        pc.text AS text,
        pc.token_count AS token_count,
        pc.is_combined AS is_combined,
        pc.is_split AS is_split,
        pc.original_section_ids AS original_section_ids,
        pc.boundaries_json AS boundaries_json,
        pc.doc_tag AS doc_tag,
        pc.document_total_tokens AS document_total_tokens,
        pc.source_path AS source_path,
        pc.is_microdoc AS is_microdoc,
        pc.doc_is_microdoc AS doc_is_microdoc,
        pc.is_microdoc_stub AS is_microdoc_stub,
        pc.embedding_version AS embedding_version,
        pc.tenant AS tenant,
        cid AS source_chunk
    """

    # 3. Shared-entity expansion - chunks sharing entities
    # Phase 3.5: Uses MENTIONS (canonical) with direction-agnostic traversal
    entity_query = """
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk {id: cid})-[:MENTIONS|IN_CHUNK]-(e:Entity)
    WHERE e.name IS NOT NULL AND size(e.name) >= 4
    WITH cid, collect(DISTINCT e)[..5] AS entities
    UNWIND entities AS entity
    MATCH (entity)-[:MENTIONS|IN_CHUNK]-(other:Chunk)
    WHERE other.id <> cid
      AND NOT other.id IN $exclude_ids
      AND ($doc_tag IS NULL OR other.doc_tag = $doc_tag)
    WITH cid, other, count(DISTINCT entity) AS shared_count
    ORDER BY shared_count DESC
    WITH cid, collect(other)[..$limit] AS entity_chunks
    UNWIND entity_chunks AS ec
    RETURN DISTINCT
        ec.id AS chunk_id,
        ec.document_id AS document_id,
        ec.parent_section_id AS parent_section_id,
        ec.order AS `order`,
        ec.level AS level,
        ec.heading AS heading,
        ec.text AS text,
        ec.token_count AS token_count,
        ec.is_combined AS is_combined,
        ec.is_split AS is_split,
        ec.original_section_ids AS original_section_ids,
        ec.boundaries_json AS boundaries_json,
        ec.doc_tag AS doc_tag,
        ec.document_total_tokens AS document_total_tokens,
        ec.source_path AS source_path,
        ec.is_microdoc AS is_microdoc,
        ec.doc_is_microdoc AS doc_is_microdoc,
        ec.is_microdoc_stub AS is_microdoc_stub,
        ec.embedding_version AS embedding_version,
        ec.tenant AS tenant,
        cid AS source_chunk
    """

    try:
        with owner.neo4j_driver.session() as session:
            # Collect parent_section_ids and chunk_ids from seeds
            parent_section_ids = list(
                {r.parent_section_id for r in eligible if r.parent_section_id}
            )
            chunk_ids = [r.chunk_id for r in eligible if r.chunk_id]
            exclude_ids = [r.chunk_id for r in seeds if r.chunk_id]

            # Run sibling expansion
            if sibling_limit > 0 and parent_section_ids:
                result = session.run(
                    sibling_query,
                    parent_section_ids=parent_section_ids,
                    exclude_ids=exclude_ids,
                    doc_tag=doc_tag,
                    limit=sibling_limit,
                    timeout=timeout_seconds,
                )
                for record in result:
                    chunk_id = record["chunk_id"]
                    rid = ("chunk_id", chunk_id)
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    # Find source seed by parent_section_id
                    source_psid = record["source_parent_section"]
                    source_seed = next(
                        (s for s in seeds if s.parent_section_id == source_psid),
                        eligible[0],
                    )
                    source_score = source_seed.fused_score or 0.0

                    expanded.append(
                        build_expanded_chunk(
                            owner,
                            record,
                            source_seed.chunk_id,
                            context_source="sibling",
                            source_score=source_score,
                        )
                    )

            # Run parent section expansion
            if parent_limit > 0 and chunk_ids:
                result = session.run(
                    parent_query,
                    chunk_ids=chunk_ids,
                    doc_tag=doc_tag,
                    limit=parent_limit,
                    timeout=timeout_seconds,
                )
                for record in result:
                    chunk_id = record["chunk_id"]
                    rid = ("chunk_id", chunk_id)
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    source_chunk_id = record["source_chunk"]
                    source_seed = seed_lookup.get(source_chunk_id, eligible[0])
                    source_score = source_seed.fused_score or 0.0

                    expanded.append(
                        build_expanded_chunk(
                            owner,
                            record,
                            source_chunk_id,
                            context_source="parent_section",
                            source_score=source_score,
                        )
                    )

            # Run shared-entity expansion
            if entity_limit > 0 and chunk_ids:
                result = session.run(
                    entity_query,
                    chunk_ids=chunk_ids,
                    exclude_ids=list(seen_ids),
                    doc_tag=doc_tag,
                    limit=entity_limit,
                    timeout=timeout_seconds,
                )
                for record in result:
                    chunk_id = record["chunk_id"]
                    rid = ("chunk_id", chunk_id)
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    source_chunk_id = record["source_chunk"]
                    source_seed = seed_lookup.get(source_chunk_id, eligible[0])
                    source_score = source_seed.fused_score or 0.0

                    expanded.append(
                        build_expanded_chunk(
                            owner,
                            record,
                            source_chunk_id,
                            context_source="shared_entities",
                            source_score=source_score,
                        )
                    )

        logger.info(
            "structure_aware_expansion",
            extra={
                "seeds": len(seeds),
                "expanded": len(expanded),
                "sibling_limit": sibling_limit,
                "parent_limit": parent_limit,
                "entity_limit": entity_limit,
            },
        )

    except Exception as e:
        logger.error(f"Structure-aware expansion failed: {e}")
        # Don't fail the whole search if expansion fails

    return expanded


def gate_expansion_with_sparse(
    owner,
    query: str,
    neighbors: List[ChunkResult],
    threshold: float,
) -> List[ChunkResult]:
    """Filter/rescore expanded neighbors using BGE sparse scores in Qdrant."""
    sparse_vector = owner.vector_retriever._build_sparse_query(query)
    if not sparse_vector:
        return neighbors

    indices = sparse_vector.get("indices") if isinstance(sparse_vector, dict) else None
    values = sparse_vector.get("values") if isinstance(sparse_vector, dict) else None
    if not indices or not values:
        return neighbors

    try:
        score_map = rescore_expansion_with_sparse(
            owner, indices, values, neighbors, threshold
        )
        if score_map is None:
            return neighbors

        if (
            not owner.expansion_rescoring_enabled
            or owner.expansion_rescoring_mode == "threshold_only"
        ):
            allowed = {
                pid for pid, s in score_map.items() if threshold == 0 or s >= threshold
            }
            return [n for n in neighbors if str(n.chunk_id) in allowed] or neighbors

        rescored = fuse_expansion_scores(owner, neighbors, score_map)
        return rescored or neighbors
    except Exception as exc:
        logger.debug(
            "Sparse gating for expansion failed; keeping original neighbors",
            error=str(exc),
        )
        return neighbors


def normalize_sparse_scores(owner, scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    min_s, max_s = min(vals), max(vals)
    percentile_map: Dict[float, float] = {}
    if owner.expansion_rescoring_normalize == "percentile":
        sorted_vals = sorted(vals)
        denom = max(len(sorted_vals) - 1, 1)
        percentile_map = {v: idx / denom for idx, v in enumerate(sorted_vals)}

    def _norm(val: float) -> float:
        method = owner.expansion_rescoring_normalize
        if method == "sigmoid":
            return 1.0 / (1.0 + math.exp(-val))
        if method == "percentile":
            return percentile_map.get(val, 0.0)
        if max_s > min_s:
            return (val - min_s) / (max_s - min_s)
        return 0.0

    return {pid: _norm(v) for pid, v in scores.items()}


def fuse_expansion_scores(
    owner, neighbors: List[ChunkResult], score_map: Dict[str, float]
) -> List[ChunkResult]:
    w = owner.expansion_rescoring_weights or {}
    w_lex = float(w.get("lexical", 0.4))
    w_struct = float(w.get("structural", 0.5))
    w_prox = float(w.get("proximity", 0.1))

    rescored: List[Tuple[float, ChunkResult]] = []
    for n in neighbors:
        pid = str(n.chunk_id)
        lex_norm = score_map.get(pid)
        if lex_norm is None:
            continue
        struct = float(n.graph_score or 0.0)
        prox = 1.0 / float((n.graph_distance or 0) + 1)
        final = (w_lex * lex_norm) + (w_struct * struct) + (w_prox * prox)
        n.lexical_vec_score = lex_norm
        n.fused_score = final
        rescored.append((final, n))

    rescored.sort(key=lambda t: t[0], reverse=True)
    return [n for _, n in rescored]


def rescore_expansion_with_sparse(
    owner,
    indices: List[int],
    values: List[float],
    neighbors: List[ChunkResult],
    threshold: float,
) -> Optional[Dict[str, float]]:
    sparse_query = QdrantSparseVector(indices=indices, values=values)
    neighbor_ids = [str(n.chunk_id) for n in neighbors if n.chunk_id]
    if not neighbor_ids:
        return None

    q_filter = QdrantFilter(
        must=[
            FieldCondition(
                key="kg_id",
                match=MatchAny(any=neighbor_ids),
            )
        ],
    )
    result = owner.vector_retriever.client.query_points(
        collection_name=owner.vector_retriever.collection,
        query=sparse_query,
        using=owner.vector_retriever.sparse_query_name,
        query_filter=q_filter,
        limit=len(neighbor_ids),
        with_payload=False,
        with_vectors=False,
        score_threshold=threshold or None,
    )
    hits = result.points
    if not hits:
        return None

    raw_scores = {
        str(hit.id): float(hit.score) for hit in hits if hit.score is not None
    }
    if not raw_scores:
        return None

    if (
        not owner.expansion_rescoring_enabled
        or owner.expansion_rescoring_mode == "threshold_only"
    ):
        return raw_scores

    return normalize_sparse_scores(owner, raw_scores)


def truncate_text(owner, text: str, token_budget: int) -> Tuple[str, int]:
    if token_budget <= 0 or not text:
        return "", 0
    total = owner.tokenizer.count_tokens(text)
    if total <= token_budget:
        return text, total
    if getattr(owner.tokenizer, "supports_decode", False):
        tokens = owner.tokenizer.encode(text)
        truncated_tokens = tokens[:token_budget]
        truncated_text = owner.tokenizer.decode_tokens(truncated_tokens)
        return truncated_text, len(truncated_tokens)

    logger.warning(
        "Tokenizer %s lacks decode; truncating text approximately",
        getattr(owner.tokenizer, "backend_name", "unknown"),
    )
    ratio = token_budget / total if total else 0
    approx_chars = max(1, int(len(text) * ratio))
    truncated_text = text[:approx_chars]
    return truncated_text, min(token_budget, total)


def is_microdoc_candidate(owner, chunk: ChunkResult) -> bool:
    return (chunk.doc_is_microdoc or chunk.token_count < owner.micro_min_tokens) and (
        chunk.document_total_tokens or 0
    ) <= owner.micro_doc_max


def is_microdoc_source(owner, chunk: ChunkResult) -> bool:
    total_tokens = chunk.document_total_tokens or 0
    return chunk.doc_is_microdoc or total_tokens <= owner.micro_doc_max


def microdoc_from_fused(
    owner,
    base: ChunkResult,
    fused_pool: List[ChunkResult],
    used_docs: Set[str],
    limit: int,
) -> List[ChunkResult]:
    if limit <= 0:
        return []
    extras: List[ChunkResult] = []
    for candidate in fused_pool:
        if candidate.document_id in used_docs:
            continue
        if candidate.document_id == base.document_id:
            continue
        if not is_microdoc_source(owner, candidate):
            continue
        extras.append(replace(candidate))
        if len(extras) >= limit:
            break
    return extras


def microdoc_from_directory(
    owner,
    base: ChunkResult,
    used_docs: Set[str],
    limit: int,
    filters: Dict[str, Any],
) -> List[ChunkResult]:
    """
    Find micro-doc candidates from the same directory as the base chunk.

    Applies tenant/doc_tag/snapshot_scope filters to prevent cross-scope
    data leakage when expanding micro-doc results.
    """
    # PHASE 1 VECTOR-ONLY: Skip microdoc expansion when neo4j_disabled
    if limit <= 0 or owner.neo4j_disabled:
        return []
    prefix = path_prefix(base.source_path, owner.micro_dir_depth)
    if not prefix:
        return []

    # Extract filter values with None as default (NULL in Cypher)
    doc_tag = filters.get("doc_tag") if filters else None
    tenant = filters.get("tenant") if filters else None
    snapshot_scope = filters.get("snapshot_scope") if filters else None

    # Build Cypher query with optional filter conditions
    # NULL parameters are handled with IS NULL OR equality checks
    query = """
    MATCH (c:Chunk)
    WHERE c.document_id <> $document_id
      AND c.document_total_tokens <= $doc_max
      AND c.source_path STARTS WITH $prefix
      AND ($doc_tag IS NULL OR c.doc_tag = $doc_tag)
      AND ($tenant IS NULL OR c.tenant = $tenant)
      AND ($snapshot_scope IS NULL OR c.snapshot_scope = $snapshot_scope)
    RETURN c
    ORDER BY c.document_total_tokens ASC, c.token_count ASC
    LIMIT $limit
    """

    extras: List[ChunkResult] = []
    try:
        with owner.neo4j_driver.session() as session:
            records = session.run(
                query,
                document_id=base.document_id,
                doc_max=owner.micro_doc_max,
                prefix=prefix,
                limit=owner.micro_knn_limit,
                doc_tag=doc_tag,
                tenant=tenant,
                snapshot_scope=snapshot_scope,
            )
            for record in records:
                node = record.get("c")
                if not node:
                    continue
                candidate = owner._chunk_from_props(dict(node))
                if candidate.document_id in used_docs:
                    continue
                extras.append(candidate)
                if len(extras) >= limit:
                    break
    except Exception as exc:
        logger.debug(
            "Microdoc directory lookup failed",
            extra={"error": str(exc), "doc": base.document_id},
        )
    return extras


def microdoc_from_knn(
    owner,
    base: ChunkResult,
    used_docs: Set[str],
    limit: int,
    filters: Dict[str, Any],
) -> List[ChunkResult]:
    if limit <= 0:
        return []
    text = (base.text or "").strip()
    if not text:
        return []
    query_vector: Optional[List[float]] = None
    try:
        query_vector = owner.vector_retriever.embedder.embed_query(text)
    except Exception as exc:
        logger.debug(
            "Microdoc query embedding failed; falling back to passage embedding",
            extra={"error": str(exc), "doc": base.document_id},
        )
        try:
            vectors = owner.vector_retriever.embedder.embed_documents([text])
            if vectors:
                query_vector = vectors[0]
        except Exception as inner_exc:
            logger.debug(
                "Microdoc embedding failed",
                extra={"error": str(inner_exc), "doc": base.document_id},
            )
            return []

    if not query_vector:
        return []

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError:
        return []

    must_conditions = []
    must_not_conditions = [
        FieldCondition(key="document_id", match=MatchValue(value=base.document_id))
    ]
    doc_tag = filters.get("doc_tag")
    if doc_tag:
        must_conditions.append(
            FieldCondition(key="doc_tag", match=MatchValue(value=doc_tag))
        )
    snapshot_scope = filters.get("snapshot_scope")
    if snapshot_scope:
        must_conditions.append(
            FieldCondition(key="snapshot_scope", match=MatchValue(value=snapshot_scope))
        )

    qdrant_filter = None
    if must_conditions or must_not_conditions:
        qdrant_filter = Filter(
            must=must_conditions or None, must_not=must_not_conditions or None
        )

    extras: List[ChunkResult] = []
    try:
        hits = owner.vector_retriever.search_named_vector(
            "content",
            query_vector,
            owner.micro_knn_limit,
            query_filter=qdrant_filter,
            score_threshold=owner.micro_sim_threshold,
        )
    except Exception as exc:
        logger.debug(
            "Microdoc kNN search failed",
            extra={"error": str(exc), "doc": base.document_id},
        )
        return []

    for hit in hits:
        payload = hit.payload or {}
        doc_id = payload.get("document_id")
        if not doc_id or doc_id in used_docs or doc_id == base.document_id:
            continue
        doc_total = payload.get("document_total_tokens", 0)
        if doc_total and doc_total > owner.micro_doc_max:
            continue
        candidate = ChunkResult(
            chunk_id=payload.get("id", hit.id),
            document_id=doc_id,
            parent_section_id=payload.get("parent_section_id", ""),
            order=payload.get("order", 0),
            level=payload.get("level", 3),
            heading=payload.get("heading", ""),
            text=payload.get("text", ""),
            token_count=payload.get("token_count", 0),
            is_combined=payload.get("is_combined", False),
            is_split=payload.get("is_split", False),
            original_section_ids=payload.get("original_section_ids", []),
            boundaries_json=payload.get("boundaries_json", "{}"),
            doc_tag=payload.get("doc_tag"),
            snapshot_scope=payload.get("snapshot_scope"),
            document_total_tokens=doc_total,
            source_path=payload.get("source_path"),
            is_microdoc=payload.get("is_microdoc", False),
            doc_is_microdoc=payload.get("doc_is_microdoc", False),
            is_microdoc_stub=payload.get("is_microdoc_stub", False),
            embedding_version=payload.get("embedding_version"),
            tenant=payload.get("tenant"),
            fused_score=hit.score,
            citation_labels=[],
            # Phase 4: Entity metadata for GLiNER-aware retrieval boosting
            # Deduplicate to clean up repeated entity extractions from same chunk
            entity_metadata=_deduplicate_entity_metadata(
                payload.get("entity_metadata")
            ),
        )
        extras.append(candidate)
        if len(extras) >= limit:
            break
    return extras


def expand_microdoc_results(
    owner,
    query: str,
    fused_results: List[ChunkResult],
    seeds: List[ChunkResult],
    filters: Dict[str, Any],
) -> Tuple[List[ChunkResult], int]:
    """Stitch additional micro-doc chunks when base results are inherently small."""
    if not owner.microdoc_enabled or not seeds or owner.micro_max_neighbors <= 0:
        return [], 0

    extras: List[ChunkResult] = []
    stitched_tokens = 0
    used_docs = {r.document_id for r in seeds if r.document_id}

    fused_pool = [
        r for r in fused_results if r.document_id and r.document_id not in used_docs
    ]
    fused_pool.sort(key=lambda x: x.fused_score or 0.0, reverse=True)

    for base in seeds:
        if not is_microdoc_candidate(owner, base):
            continue

        remaining = owner.micro_max_neighbors
        cohort: List[ChunkResult] = []

        cohort.extend(
            microdoc_from_fused(owner, base, fused_pool, used_docs, remaining)
        )
        remaining = owner.micro_max_neighbors - len(cohort)

        if remaining > 0:
            cohort.extend(
                microdoc_from_directory(owner, base, used_docs, remaining, filters)
            )
            remaining = owner.micro_max_neighbors - len(cohort)

        if remaining > 0:
            cohort.extend(microdoc_from_knn(owner, base, used_docs, remaining, filters))

        for candidate in cohort:
            if candidate.document_id in used_docs:
                continue

            truncated_text, truncated_tokens = truncate_text(
                owner, candidate.text, owner.micro_per_doc_budget
            )
            if truncated_tokens == 0:
                continue
            if stitched_tokens + truncated_tokens > owner.micro_total_budget:
                logger.info(
                    "Microdoc stitching budget exhausted",
                    extra={"tokens": stitched_tokens},
                )
                return extras, stitched_tokens

            patched = replace(
                candidate,
                text=truncated_text,
                token_count=truncated_tokens,
                parent_section_id=candidate.chunk_id,
                is_microdoc_extra=True,
                expansion_source=base.chunk_id,
            )
            if base.rerank_score is not None:
                patched.rerank_score = neighbor_score(float(base.rerank_score))
            extras.append(patched)
            used_docs.add(candidate.document_id)
            stitched_tokens += truncated_tokens

    if extras:
        logger.info(
            "Microdoc stitching added extras",
            extra={
                "base_count": len(seeds),
                "extras": len(extras),
                "tokens": stitched_tokens,
            },
        )
    return extras, stitched_tokens


def enforce_context_budget(
    owner, results: List[ChunkResult], starting_tokens: int = 0
) -> Tuple[List[ChunkResult], int]:
    """
    Enforce context budget by limiting total tokens.

    Groups by parent_section_id, sorts by order, and trims from tail
    when exceeding budget.

    Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
    """
    if not results:
        return [], starting_tokens

    total_tokens = starting_tokens
    final_results: List[ChunkResult] = []
    grouped: "OrderedDict[str, List[ChunkResult]]" = OrderedDict()

    for chunk in results:
        parent = chunk.parent_section_id or chunk.chunk_id
        grouped.setdefault(parent, []).append(chunk)

    for parent, chunks in grouped.items():
        taken = 0
        for chunk in chunks:
            tokens = max(0, chunk.token_count or 0)
            if total_tokens + tokens > owner.context_max_tokens:
                log_context_budget(owner, total_tokens, len(final_results))
                return final_results, total_tokens
            final_results.append(chunk)
            total_tokens += tokens
            taken += 1
            if taken >= max(1, owner.context_group_cap):
                break

    return final_results, total_tokens


def log_context_budget(owner, tokens: int, count: int) -> None:
    logger.info(
        "Context budget enforced",
        extra={
            "chunks": count,
            "tokens": tokens,
            "max_tokens": owner.context_max_tokens,
        },
    )
