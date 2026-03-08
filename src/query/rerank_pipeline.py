# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade wrappers)
# =============================================================================
"""
Rerank pipeline: cross-encoder reranking, ColBERT late-interaction scoring,
and specificity adjustment.

Extracted from hybrid_retrieval.py to isolate all reranking logic behind
a stable interface.
"""

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.query.query_intent import QueryIntent

import numpy as np
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter as QdrantFilter
from qdrant_client.models import MatchAny

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.factory import ProviderFactory
from src.providers.rerank.base import RerankProvider
from src.query.retrieval_types import ChunkResult
from src.shared.observability import get_logger

logger = get_logger(__name__)

# Cloud/deploy cues to detect in heading + parent_path (not doc_tag).
# Broader than CLOUD_CUES — includes deployment-workflow terms that
# indicate the chunk is about cloud setup, not the subsystem itself.
_DEPLOY_CUES = frozenset(
    {
        "aws",
        "azure",
        "gcp",
        "slurm",
        "kubernetes",
        "k8s",
        "eks",
        "aks",
        "gke",
        "terraform",
        "cloudformation",
        "parallelcluster",
        "cyclecloud",
        "sagemaker",
        "deploy",
        "deployment",
        "installation",
    }
)


# ── 1. get_reranker ─────────────────────────────────────────────────────────


def get_reranker(owner) -> Optional[RerankProvider]:
    if not owner.reranker_config or not getattr(
        owner.reranker_config, "enabled", False
    ):
        return None

    if owner._reranker is not None:
        return owner._reranker

    cfg_provider = getattr(owner.reranker_config, "provider", None)
    cfg_model = getattr(owner.reranker_config, "model", None)

    try:
        owner._reranker = ProviderFactory.create_rerank_provider(
            provider=cfg_provider,
            model=cfg_model,
        )
        healthy = True
        if hasattr(owner._reranker, "health_check"):
            try:
                healthy = bool(owner._reranker.health_check())
            except Exception:
                healthy = False
        if not healthy:
            logger.warning(
                "Reranker failed health check; disabling reranker",
                provider=cfg_provider,
                model=cfg_model,
            )
            owner._reranker_available = False
            owner._reranker = None
            return None

        logger.info(
            "HybridRetriever reranker initialized: provider=%s, model=%s",
            owner._reranker.provider_name,
            owner._reranker.model_id,
        )
        owner._reranker_available = True
    except Exception as exc:  # pragma: no cover - provider init logging
        logger.warning("Unable to initialize reranker provider: %s", exc)
        owner._reranker = None
        owner._reranker_available = False

    return owner._reranker


# ── 2. apply_reranker ────────────────────────────────────────────────────────


def apply_reranker(
    owner,
    query: str,
    seeds: List[ChunkResult],
    metrics: Dict[str, Any],
    *,
    query_type: Optional[str] = None,
    intent: Optional["QueryIntent"] = None,
) -> List[ChunkResult]:
    cfg = owner.reranker_config
    metrics.setdefault("reranker_input_count", 0)
    metrics.setdefault("reranker_output_count", 0)
    if not cfg or not getattr(cfg, "enabled", False):
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "disabled"
        return seeds

    reranker = get_reranker(owner)
    if reranker is None or not owner._reranker_available:
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "not_available"
        return seeds

    # Select per-query-type instruction if available
    per_call_instruction: Optional[str] = None
    if query_type and cfg:
        instructions_by_type = getattr(cfg, "instructions_by_type", None) or {}
        per_call_instruction = instructions_by_type.get(query_type)

    # Focused reranker text (plan-driven)
    use_focused_text = (
        owner._plan.use_focused_rerank_text
        and intent is not None
        and intent.precision_mode
        and len(intent.primary_anchors) > 0
    )
    metrics["precision_focused_rerank_text"] = use_focused_text
    focused_fallback_count = 0

    def _clean_text(text: str) -> str:
        # Remove simple HTML tags and known markup artifacts, collapse whitespace.
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("[CODE]", " ").replace("[/CODE]", " ")
        text = text.replace("</details>", " ").replace("<details>", " ")
        text = re.sub(r"\b[A-Z]{1,4}\]\s*", " ", text)  # drop stray tokens like 'DE]'
        text = text.replace("####", " ")
        text = text.replace("DE]", " ")
        text = re.sub(r"^[^A-Za-z0-9]+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_focused_text(
        body: str,
        heading: str,
        parent_path: str,
        anchors: Tuple[str, ...],
    ) -> Tuple[str, bool]:
        """Extract lines containing anchor terms with ±1 context.

        Returns (focused_text, used_fallback).
        """
        lines = body.split("\n")
        matched_indices: set = set()
        anchors_lower = [a.lower() for a in anchors]
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(a in line_lower for a in anchors_lower):
                matched_indices.add(idx)
        if not matched_indices:
            # Fallback: heading + first 500 chars of body
            fallback_body = body[:500]
            parts = [p for p in [parent_path, heading, fallback_body] if p]
            return "\n\n".join(parts), True
        # Expand ±1 context window
        expanded: set = set()
        for idx in matched_indices:
            expanded.add(max(0, idx - 1))
            expanded.add(idx)
            expanded.add(min(len(lines) - 1, idx + 1))
        selected = [lines[i] for i in sorted(expanded)]
        focused = "\n".join(selected)
        parts = [p for p in [parent_path, heading, focused] if p]
        return "\n\n".join(parts), False

    candidates: List[Dict[str, Any]] = []
    for chunk in seeds:
        text_body = (chunk.text or "").strip()
        heading = (chunk.heading or "").strip()
        parent_path = (chunk.parent_path_norm or "").strip()

        # Phase A3: Build focused text for precision intents
        if use_focused_text and text_body:
            text, used_fallback = _build_focused_text(
                text_body, heading, parent_path, intent.primary_anchors
            )
            if used_fallback:
                focused_fallback_count += 1
        else:
            # Default: full structural context
            # "path > to > section\n\nHeading\n\nbody text"
            if parent_path and heading and text_body:
                text = f"{parent_path}\n\n{heading}\n\n{text_body}"
            elif heading and text_body:
                text = f"{heading}\n\n{text_body}"
            else:
                text = text_body or heading

        text = _clean_text(text)
        if not text:
            has_tokens = (chunk.token_count or 0) > 0
            if not heading or not has_tokens:
                continue

        candidates.append(
            {
                "id": chunk.chunk_id,
                "text": text,
                "original_result": chunk,
            }
        )

    metrics["focused_rerank_text_fallback_count"] = focused_fallback_count

    if not candidates:
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "no_text"
        metrics["reranker_input_count"] = 0
        metrics["reranker_output_count"] = 0
        return seeds

    metrics["reranker_input_count"] = len(candidates)

    top_n = getattr(cfg, "top_n", None)
    if not top_n or top_n <= 0:
        top_n = len(candidates)
    top_k = min(top_n, len(candidates))

    # Batch candidates to respect reranker service limits
    service_max_batch = 32
    service_max_batch_tokens = 4096  # Qwen3-Reranker-4B supports 8K context

    def _approx_tokens(text: str) -> int:
        # Simple word-count approximation
        return max(1, len(text.split()))

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_tokens = 0

    for idx, cand in enumerate(candidates):
        tcount = _approx_tokens(cand.get("text", ""))
        if current_batch and (
            len(current_batch) + 1 > service_max_batch
            or current_tokens + tcount > service_max_batch_tokens
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        cand["orig_index"] = idx
        current_batch.append(cand)
        current_tokens += tcount

    if current_batch:
        batches.append(current_batch)

    start_time = time.time()
    reranked_payload: List[Dict[str, Any]] = []
    try:
        for batch in batches:
            batch_top_k = min(top_k, len(batch))
            reranked_batch = reranker.rerank(
                query=query,
                candidates=batch,
                top_k=batch_top_k,
                instruction=per_call_instruction,
            )
            reranked_payload.extend(reranked_batch)
        latency_ms = (time.time() - start_time) * 1000
    except Exception as exc:  # pragma: no cover - provider failure logging
        logger.warning("Reranker call failed; returning fused ordering: %s", exc)
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "provider_error"
        return seeds

    reranked_payload = sorted(
        reranked_payload,
        key=lambda c: (c.get("rerank_score") or float("-inf")),
        reverse=True,
    )
    reranked_payload = reranked_payload[:top_k]

    reranked_chunks: List[ChunkResult] = []
    for payload in reranked_payload:
        chunk = payload.get("original_result")
        if not isinstance(chunk, ChunkResult):
            continue

        score = payload.get("rerank_score")
        if score is not None:
            chunk.rerank_score = score

        rank = payload.get("original_rank")
        chunk.rerank_original_rank = None
        if rank is not None:
            try:
                chunk.rerank_original_rank = int(rank)
            except (TypeError, ValueError):
                chunk.rerank_original_rank = None

        chunk.reranker = payload.get("reranker") or reranker.model_id
        chunk.fusion_method = "rerank"
        reranked_chunks.append(chunk)

    try:
        sample_logs = reranked_payload[: min(3, len(reranked_payload))]
        logger.debug(
            "Reranker payload samples",
            extra={
                "sample_count": len(sample_logs),
                "query_snippet": query[:200],
                "doc_snippets": [(p.get("text") or "")[:200] for p in sample_logs],
                "scores": [p.get("rerank_score") for p in sample_logs],
            },
        )
    except Exception:
        pass

    if not reranked_chunks:
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "no_results"
        metrics["reranker_output_count"] = 0
        return seeds

    reranked_chunks.sort(
        key=lambda c: (c.rerank_score or float("-inf"), c.fused_score or 0.0),
        reverse=True,
    )
    for idx, chunk in enumerate(reranked_chunks, start=1):
        chunk.rerank_rank = idx

    # Detect reranker fallback mode from payload markers
    fallback_markers = {"circuit_open", "rerank_failed", "batch_failed"}
    reranker_names = {p.get("reranker") for p in reranked_payload}
    is_fallback = bool(reranker_names & fallback_markers)
    real_scores = sum(
        1
        for p in reranked_payload
        if p.get("reranker") not in fallback_markers
        and (p.get("rerank_score") or 0) > 0
    )

    # Extract batch-level meta from first payload if present
    reranker_meta = {}
    if reranked_payload:
        reranker_meta = reranked_payload[0].pop("_reranker_meta", {})

    if is_fallback and real_scores == 0:
        # Total fallback — reranking did not produce meaningful scores
        metrics["reranker_applied"] = False
        metrics["reranker_reason"] = "fallback_zero_scores"
    else:
        metrics["reranker_applied"] = True
        metrics["reranker_reason"] = "ok"

    metrics["reranker_model"] = reranker.model_id
    metrics["reranker_time_ms"] = latency_ms
    metrics["reranker_output_count"] = len(reranked_chunks)
    metrics["reranker_real_scores_count"] = real_scores
    metrics["reranker_zero_scores_count"] = len(reranked_payload) - real_scores
    metrics["reranker_batch_successes"] = reranker_meta.get("batch_successes", 0)
    metrics["reranker_batch_failures"] = reranker_meta.get("batch_failures", 0)
    metrics["reranker_fallback_mode"] = reranker_meta.get("fallback_mode")
    metrics["reranker_instruction"] = per_call_instruction or getattr(
        cfg, "instruction", None
    )
    return reranked_chunks


# ── 3. hydrate_colbert_vectors ───────────────────────────────────────────────


def hydrate_colbert_vectors(
    owner, candidates: List[ChunkResult]
) -> Dict[str, List[List[float]]]:
    """Fetch ColBERT vectors for candidates that lack them."""
    if not (
        owner.vector_retriever.supports_colbert
        and owner.vector_retriever.schema_supports_colbert
    ):
        return {}
    missing = [c.chunk_id for c in candidates if not c.colbert_vector]
    if not missing:
        return {}
    q_filter = QdrantFilter(
        must=[
            FieldCondition(key="kg_id", match=MatchAny(any=list(missing))),
        ]
    )
    try:
        scroll_res = owner.vector_retriever.client.scroll(
            collection_name=owner.vector_retriever.collection,
            scroll_filter=q_filter,
            with_vectors=["late-interaction"],
            with_payload=["kg_id"],  # FIXED: Need kg_id to match chunk_id
            limit=len(missing),
        )
        points = scroll_res[0] if isinstance(scroll_res, tuple) else scroll_res
    except Exception as exc:
        logger.warning("ColBERT hydration failed", error=str(exc))
        return {}
    hydrated: Dict[str, List[List[float]]] = {}
    for point in points:
        vectors = getattr(point, "vector", None) or getattr(point, "vectors", None)
        if isinstance(vectors, dict):
            colbert_vec = vectors.get("late-interaction")
        else:
            colbert_vec = None
        if colbert_vec:
            # Use kg_id from payload (matches chunk_id), not point.id (Qdrant UUID)
            payload = getattr(point, "payload", None) or {}
            kg_id = payload.get("kg_id") or payload.get("id") or str(point.id)
            hydrated[kg_id] = colbert_vec
    for c in candidates:
        if c.chunk_id in hydrated:
            c.colbert_vector = hydrated[c.chunk_id]
    return hydrated


# ── 4. colbert_rerank ────────────────────────────────────────────────────────


def colbert_rerank(
    owner,
    candidates: List[ChunkResult],
    query_bundle: QueryEmbeddingBundle,
    limit: int,
) -> List[ChunkResult]:
    """Rerank using ColBERT late-interaction MaxSim scores.

    Uses query-length normalization (Vespa production approach) to eliminate
    document length bias. ColBERT pads queries to fixed length (32 tokens),
    and each query token contributes max 1.0, so dividing by QUERY_MAXLEN
    bounds scores to [0, 1]. See: https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/
    """
    # Standard ColBERT query length (queries are padded to this length)
    QUERY_MAXLEN = 32

    if not query_bundle or not query_bundle.multivector:
        return candidates
    q_vectors = query_bundle.multivector.vectors or []
    if not q_vectors:
        return candidates
    q_mat = np.array(q_vectors)
    scored: List[ChunkResult] = []
    for chunk in candidates:
        if not chunk.colbert_vector:
            continue
        d_mat = np.array(chunk.colbert_vector)
        if d_mat.size == 0:
            continue
        sims = q_mat @ d_mat.T
        # MaxSim with query-length normalization to eliminate length bias
        raw_score = float(np.max(sims, axis=1).sum()) if sims.size else 0.0
        score = raw_score / QUERY_MAXLEN
        chunk.rerank_score = score
        chunk.reranker = "colbert"
        scored.append(chunk)
    if not scored:
        return candidates
    scored.sort(key=lambda c: c.rerank_score or float("-inf"), reverse=True)
    return scored[:limit]


# ── 5. apply_specificity_adjustment ──────────────────────────────────────────


def apply_specificity_adjustment(
    owner,
    candidates: List[ChunkResult],
    intent: "QueryIntent",
    metrics: Dict[str, Any],
    *,
    anchor_bonus: float = 0.15,
    deploy_penalty: float = 0.10,
) -> Tuple[bool, int]:
    """Small tie-break adjustment after cross-encoder reranking.

    For precision intents, nudges scores based on:
    - Bonus: primary anchor term appears in heading or parent_path
    - Penalty: cloud/deploy cue in heading or parent_path

    Bounded to ±0.15 on a 9-10 scale: enough to break ties within
    the reranker's score plateau, not enough to override a genuinely
    higher-ranked chunk.

    Returns (applied, adjustment_count).
    """
    if not intent.primary_anchors:
        return False, 0

    anchors_lower = [a.lower() for a in intent.primary_anchors]
    adjustments = 0

    for chunk in candidates:
        if chunk.rerank_score is None:
            continue

        heading_lower = (chunk.heading or "").lower()
        path_lower = (chunk.parent_path_norm or "").lower()
        surface = heading_lower + " " + path_lower

        delta = 0.0

        # Anchor bonus: heading or path contains a primary anchor
        if any(a in surface for a in anchors_lower):
            delta += anchor_bonus

        # Deploy penalty: heading or path contains cloud/deploy cues
        if any(cue in surface for cue in _DEPLOY_CUES):
            delta -= deploy_penalty

        if delta != 0.0:
            chunk.rerank_score = chunk.rerank_score + delta
            adjustments += 1

    if adjustments > 0:
        logger.info(
            "specificity_adjustment_applied",
            anchors=list(intent.primary_anchors),
            adjustments=adjustments,
            anchor_bonus=anchor_bonus,
            deploy_penalty=deploy_penalty,
            total_candidates=len(candidates),
        )

    return adjustments > 0, adjustments
