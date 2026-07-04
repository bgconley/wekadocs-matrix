"""
Shared data model for RELATED_TO v2 edges.

This module defines the canonical data structures and helpers used by both the
incremental linker (src/services/cross_doc_linking.py) and the batch backfill
script (scripts/backfill_cross_doc_edges.py).

Design constraints:
    - Only stdlib / typing imports — no project dependencies so that scripts
      can import this module without pulling in the full application stack.
    - No timestamp fields on EdgePayload — those are set by Cypher datetime()
      at write time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_MODEL_VERSION = "2.0"
DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------------
# CandidateSignals
# ---------------------------------------------------------------------------


@dataclass
class CandidateSignals:
    """Per-document candidate with individual retrieval signal scores.

    Replaces the old ``Tuple[str, float]`` candidate representation so that
    every scoring signal is preserved through fusion, thresholding, and edge
    creation.
    """

    target_doc_id: str

    score_dense: Optional[float] = None
    score_sparse: Optional[float] = None
    score_rrf: Optional[float] = None
    score_colbert: Optional[float] = None
    score_title_ft: Optional[float] = None

    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None

    @property
    def score_final(self) -> float:
        """Return the best available score using a coalesce chain.

        Priority: colbert > rrf > dense > title_ft > sparse > 0.0.
        """
        for val in (
            self.score_colbert,
            self.score_rrf,
            self.score_dense,
            self.score_title_ft,
            self.score_sparse,
        ):
            if val is not None:
                return val
        return 0.0


# ---------------------------------------------------------------------------
# StructuralPriors
# ---------------------------------------------------------------------------


@dataclass
class StructuralPriors:
    """Structural graph priors that can boost or suppress candidate edges.

    These are computed from existing graph topology rather than from vector
    similarity.
    """

    prior_reference: Optional[float] = None
    """Boost derived from explicit REFERENCES edges between documents."""

    prior_entity: Optional[float] = None
    """Boost derived from shared MENTIONS entities (hub-suppressed)."""

    prior_taxonomy: Optional[float] = None
    """Boost derived from doc_tag / doc_category alignment."""


# ---------------------------------------------------------------------------
# EdgePayload
# ---------------------------------------------------------------------------


@dataclass
class EdgePayload:
    """Complete RELATED_TO v2 property set written to Neo4j.

    Timestamps (``created_at``, ``updated_at``, ``last_seen_at``) are **not**
    included here — they are set by the Cypher MERGE statement via
    ``datetime()``.
    """

    # -- Signal scores -------------------------------------------------------
    score_final: float = 0.0
    score: float = 0.0  # legacy alias, always == score_final
    score_dense: Optional[float] = None
    score_sparse: Optional[float] = None
    score_rrf: Optional[float] = None
    score_colbert: Optional[float] = None
    score_title_ft: Optional[float] = None

    # -- Structural priors ---------------------------------------------------
    prior_reference: Optional[float] = None
    prior_entity: Optional[float] = None
    prior_taxonomy: Optional[float] = None

    # -- Provenance ----------------------------------------------------------
    method: str = "rrf_fusion"
    method_version: str = EDGE_MODEL_VERSION
    phase: str = "3.5b"

    # -- Reciprocity ---------------------------------------------------------
    is_mutual: Optional[bool] = None
    mutual_score: Optional[float] = None

    # -- Quality -------------------------------------------------------------
    quality_tier: Optional[str] = None

    # -- Serialisation -------------------------------------------------------

    def to_neo4j_params(self) -> dict:
        """Return a dict of all fields, omitting ``None`` values.

        The ``score`` field is always forced to equal ``score_final`` for
        backward compatibility with consumers that read ``r.score``.
        """
        raw = asdict(self)
        raw["score"] = raw["score_final"]
        return {k: v for k, v in raw.items() if v is not None}

    # -- Quality tier --------------------------------------------------------

    def _compute_quality_tier(self, thresholds: Optional[dict] = None) -> str:
        """Classify ``score_final`` into a quality tier.

        Default thresholds depend on the scoring method:
            - RRF-based methods (``method`` contains 'rrf'):
              high >= 0.040, medium >= 0.028
            - Dense / ColBERT methods:
              high >= 0.80, medium >= 0.65
        """
        if thresholds is None:
            if "rrf" in self.method:
                thresholds = {"high": 0.040, "medium": 0.028}
            else:
                thresholds = {"high": 0.80, "medium": 0.65}

        if self.score_final >= thresholds["high"]:
            return "high"
        if self.score_final >= thresholds["medium"]:
            return "medium"
        return "low"

    # -- Factory -------------------------------------------------------------

    @classmethod
    def from_candidate(
        cls,
        signals: CandidateSignals,
        method: str,
        phase: str,
        priors: Optional[StructuralPriors] = None,
        quality_thresholds: Optional[dict] = None,
    ) -> EdgePayload:
        """Build an ``EdgePayload`` from a scored candidate.

        Parameters
        ----------
        signals:
            The candidate with all retrieval signal scores populated.
        method:
            Scoring method label (e.g. ``"rrf_fusion"``, ``"dense"``).
        phase:
            Pipeline phase identifier (e.g. ``"3.5b"``).
        priors:
            Optional structural priors to attach to the edge.
        quality_thresholds:
            Optional custom thresholds for quality tier classification.
        """
        payload = cls(
            score_final=signals.score_final,
            score=signals.score_final,
            score_dense=signals.score_dense,
            score_sparse=signals.score_sparse,
            score_rrf=signals.score_rrf,
            score_colbert=signals.score_colbert,
            score_title_ft=signals.score_title_ft,
            method=method,
            method_version=EDGE_MODEL_VERSION,
            phase=phase,
        )

        if priors is not None:
            payload.prior_reference = priors.prior_reference
            payload.prior_entity = priors.prior_entity
            payload.prior_taxonomy = priors.prior_taxonomy

        payload.quality_tier = payload._compute_quality_tier(quality_thresholds)
        return payload


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def aggregate_chunks_to_candidates(
    chunk_hits: list,
    exclude_doc_id: str,
    score_field: str = "score_dense",
) -> List[CandidateSignals]:
    """Group chunks by ``document_id``, keeping the max score per document.

    Parameters
    ----------
    chunk_hits:
        List of objects with ``.id``, ``.payload``, and ``.score`` attributes
        (e.g. Qdrant ``ScoredPoint``).
    exclude_doc_id:
        Document ID to exclude (typically the query document itself).
    score_field:
        Which ``CandidateSignals`` field to populate with the chunk score.
        Must be one of ``score_dense``, ``score_sparse``, ``score_colbert``,
        ``score_title_ft``.

    Returns
    -------
    List[CandidateSignals]
        Candidates sorted by score descending.
    """
    doc_best: Dict[str, float] = {}

    for hit in chunk_hits:
        doc_id = hit.payload.get("document_id", "")
        if not doc_id or doc_id == exclude_doc_id:
            continue
        current_best = doc_best.get(doc_id, float("-inf"))
        if hit.score > current_best:
            doc_best[doc_id] = hit.score

    candidates = []
    for doc_id, best_score in doc_best.items():
        sig = CandidateSignals(target_doc_id=doc_id)
        setattr(sig, score_field, best_score)
        candidates.append(sig)

    candidates.sort(key=lambda c: getattr(c, score_field) or 0.0, reverse=True)
    return candidates


def reciprocal_rank_fusion_v2(
    dense_candidates: List[CandidateSignals],
    sparse_candidates: List[CandidateSignals],
    k: int = DEFAULT_RRF_K,
) -> List[CandidateSignals]:
    """RRF fusion that preserves per-signal component scores.

    For each ``doc_id`` present in either candidate list:

    - Computes ``rrf_score = sum(1 / (k + rank_i))`` across lists where the
      document appears.
    - Preserves ``score_dense`` and ``score_sparse`` from the input candidates.
    - Sets ``dense_rank`` and ``sparse_rank``.

    Parameters
    ----------
    dense_candidates:
        Candidates from dense retrieval, assumed sorted by ``score_dense``
        descending.
    sparse_candidates:
        Candidates from sparse retrieval, assumed sorted by ``score_sparse``
        descending.
    k:
        RRF constant (default 60).

    Returns
    -------
    List[CandidateSignals]
        Merged candidates sorted by ``score_rrf`` descending.
    """
    # Build lookup maps keyed by target_doc_id
    dense_lookup: Dict[str, CandidateSignals] = {}
    for rank, cand in enumerate(dense_candidates, start=1):
        dense_lookup[cand.target_doc_id] = cand
        cand.dense_rank = rank

    sparse_lookup: Dict[str, CandidateSignals] = {}
    for rank, cand in enumerate(sparse_candidates, start=1):
        sparse_lookup[cand.target_doc_id] = cand
        cand.sparse_rank = rank

    all_doc_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())

    merged: List[CandidateSignals] = []
    for doc_id in all_doc_ids:
        rrf_score = 0.0

        dense_cand = dense_lookup.get(doc_id)
        sparse_cand = sparse_lookup.get(doc_id)

        dense_rank: Optional[int] = None
        sparse_rank: Optional[int] = None
        score_dense: Optional[float] = None
        score_sparse: Optional[float] = None

        if dense_cand is not None:
            dense_rank = dense_cand.dense_rank
            score_dense = dense_cand.score_dense
            rrf_score += 1.0 / (k + dense_rank)

        if sparse_cand is not None:
            sparse_rank = sparse_cand.sparse_rank
            score_sparse = sparse_cand.score_sparse
            rrf_score += 1.0 / (k + sparse_rank)

        merged.append(
            CandidateSignals(
                target_doc_id=doc_id,
                score_dense=score_dense,
                score_sparse=score_sparse,
                score_rrf=rrf_score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
            )
        )

    merged.sort(key=lambda c: c.score_rrf or 0.0, reverse=True)
    return merged


def build_edge_merge_cypher() -> str:
    """Return the canonical MERGE Cypher for RELATED_TO edge writes.

    Uses ``$now`` parameter for timestamps and ``$props`` for edge properties.
    Returns created/updated detection via ``r.updated_at IS NULL AS was_created``.
    """
    return """
    MATCH (source:Document {id: $source_id})
    MATCH (target:Document {id: $target_id})
    MERGE (source)-[r:RELATED_TO]->(target)
    ON CREATE SET r += $props, r.created_at = $now, r.last_seen_at = $now
    ON MATCH SET r += $props, r.updated_at = $now, r.last_seen_at = $now
    RETURN r.updated_at IS NULL AS was_created
    """
