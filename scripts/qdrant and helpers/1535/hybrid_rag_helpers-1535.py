# -*- coding: utf-8 -*-
"""
hybrid_rag_helpers.py  (v2.2, provider-free)

Drop-in utilities for a Neo4j + Qdrant hybrid RAG pipeline:
  1) budgeted_pack: Greedy context packer with diversity & per-doc caps
  2) QdrantWeightedSearcher: Weighted multi-field (named-vector) search + fusion
  3) expand_graph_neighbors: Neo4j helper to expand neighbors with typed edges
  4) example_retrieve_pack: end-to-end reference (ANN -> graph -> pack)

Schema alignment:
- Canonical: document_id; legacy alias: doc_id (handled & indexed)
- Some deployments dual-label :Section as :Chunk; helpers avoid accidental
  inclusion of Section nodes by checking existence of chunk properties.

Design notes:
- Community Edition friendly (no triggers/APOC required).
- Pluggable knobs: thresholds & weights are parameters.
- No 'provider' logic; use tenant/doc_tag or entity relations if needed.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# Optional imports (guarded)
try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore

try:
    from neo4j import Driver
except Exception:  # pragma: no cover
    Driver = None  # type: ignore


# ------------------------- Data models ----------------------------------------


@dataclass
class Candidate:
    """
    A chunk candidate for inclusion in the LLM context.

    Note: The 'doc_id' field below stores the CANONICAL document_id value.
    It is filled from payload['document_id'] if present, else payload['doc_id'].
    """

    id: str
    text: str
    doc_id: Optional[str] = None  # stores canonical document_id value
    heading: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Component scores (for debugging/analysis)
    vector_score: float = 0.0
    title_vec_score: float = 0.0
    entity_vec_score: float = 0.0
    lexical_score: float = 0.0
    graph_score: float = 0.0

    # Aggregated score used for ranking before packing
    fused_score: float = 0.0

    # Flags
    is_seed: bool = False

    def total_tokens(self, token_counter: Optional[Callable[[str], int]] = None) -> int:
        if self.token_count is not None:
            return int(self.token_count)
        if token_counter:
            return int(token_counter(self.text))
        # simple fallback approximation: ~1.35 words per token
        return max(1, int(len(self.text.split()) * 1.35))


@dataclass
class GraphNeighbor:
    neighbor_id: str
    hop: int
    edge_weight: float
    path_types: Optional[List[str]] = None


# ------------------------- Tokenization / similarity --------------------------

_WORD_RE = re.compile(r"\w+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def default_token_counter(text: str) -> int:
    """
    Lightweight token estimate (~1.33 words per token). Replace with tiktoken if desired.
    """
    return max(1, int(len(_WORD_RE.findall(text)) / 1.33))


def sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard_similarity(a: str, b: str, ngram: int = 8) -> float:
    """
    Approximate near-duplicate detection via shingled Jaccard similarity on tokens.
    """
    ta = _WORD_RE.findall(a.lower())
    tb = _WORD_RE.findall(b.lower())
    if not ta or not tb:
        return 0.0

    def sh(t):
        return {tuple(t[i : i + ngram]) for i in range(0, max(0, len(t) - ngram + 1))}

    sa, sb = sh(ta), sh(tb)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ------------------------- 1) budgeted_pack -----------------------------------


def budgeted_pack(
    candidates: Iterable[Candidate],
    token_budget: int,
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    max_jaccard: float = 0.80,
    per_doc_cap: float = 0.40,
    force_include_seeds: bool = True,
    definition_keywords: Optional[Set[str]] = None,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """
    Greedily packs high-quality candidates into a token budget with:
      - near-duplicate suppression (Jaccard on shingles)
      - per-document token cap (fraction of budget)
      - optional seed forcing (ensure at least one top-seed survives)
      - light coverage heuristic (prefer to include a definition/overview)
    """
    t0 = time.time()
    token_counter = token_counter or default_token_counter
    packed: List[Candidate] = []
    used_tokens = 0
    per_doc_tokens: DefaultDict[str, int] = defaultdict(int)
    doc_token_limit = max(1, int(per_doc_cap * token_budget))

    skipped_diversity = 0
    skipped_over_budget = 0
    skipped_per_doc = 0

    def is_definition_like(c: Candidate) -> bool:
        text = f"{c.heading or ''}\n{c.text or ''}"
        keys = (
            set(definition_keywords)
            if definition_keywords
            else {"overview", "introduction", "what is", "summary", "definition"}
        )
        low = text.lower()
        return any(k in low for k in keys)

    have_definition = False
    cand_list = list(candidates)

    for c in cand_list:
        c_tokens = c.total_tokens(token_counter)
        doc = c.doc_id or "__NODOC__"  # holds canonical document_id value

        if per_doc_tokens[doc] + c_tokens > doc_token_limit and not (
            force_include_seeds and c.is_seed and per_doc_tokens[doc] == 0
        ):
            skipped_per_doc += 1
            continue

        if used_tokens + c_tokens > token_budget:
            skipped_over_budget += 1
            continue

        is_dup = False
        for kept in packed:
            if jaccard_similarity(c.text, kept.text) >= max_jaccard:
                if not (force_include_seeds and c.is_seed and per_doc_tokens[doc] == 0):
                    is_dup = True
                    break
        if is_dup:
            skipped_diversity += 1
            continue

        packed.append(c)
        used_tokens += c_tokens
        per_doc_tokens[doc] += c_tokens
        if not have_definition and is_definition_like(c):
            have_definition = True

        if used_tokens >= token_budget:
            break

    stats = {
        "total_chunks": len(cand_list),
        "kept": len(packed),
        "used_tokens": used_tokens,
        "budget": token_budget,
        "per_doc_tokens": dict(per_doc_tokens),
        "skipped_diversity": skipped_diversity,
        "skipped_over_budget": skipped_over_budget,
        "skipped_per_doc": skipped_per_doc,
        "have_definition": have_definition,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    return packed, stats


# ------------------------- Score fusion helpers --------------------------------


def reciprocal_rank_fusion(
    rankings: Dict[str, List[Tuple[str, float]]],
    weights: Dict[str, float],
    k: int = 60,
) -> Dict[str, float]:
    """
    Weighted Reciprocal Rank Fusion across multiple ranked lists.
    rankings[field] is a list of (id, score) sorted DESC by score.
    Returns id -> fused_score.
    """
    fused: DefaultDict[str, float] = defaultdict(float)
    for vector_name, items in rankings.items():
        w = float(weights.get(vector_name, 1.0))
        for rank, (pid, _score) in enumerate(items, start=1):
            fused[pid] += w * 1.0 / (k + rank)
    return fused


# ------------------------- 2) Qdrant search wrapper ----------------------------


class QdrantWeightedSearcher:
    """
    Weighted multi-field (named-vector) search over a single Qdrant collection,
    with rank-fusion across fields and optional payload extraction.

    Assumes the collection is configured with named vectors, e.g.:
        {
          "content": {"size": 1024, "distance": "Cosine"},
          "title":   {"size": 1024, "distance": "Cosine"},
          # "entity":  {"size": 1024, "distance": "Cosine"}
        }
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        *,
        payload_keys: Optional[List[str]] = None,
    ) -> None:
        if client is None or QdrantClient is None:
            raise RuntimeError(
                "Qdrant client not available. Install `qdrant-client` and pass an initialized QdrantClient."
            )
        self.client: QdrantClient = client
        self.collection = collection
        self.payload_keys = payload_keys or [
            "text",
            "document_id",
            "doc_id",
            "heading",
            "token_count",
            "tenant",
            "lang",
            "version",
            "source_path",
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
        ]

    def _search_single(
        self,
        vector_name: str,
        vector: Sequence[float],
        *,
        top_k: int,
        query_filter: Optional[Any],
        ef: Optional[int],
    ):
        params = None
        try:
            from qdrant_client.http.models import SearchParams

            params = SearchParams(hnsw_ef=ef) if ef else None
        except Exception:
            params = None

        # Note: `using=` selects the named vector.
        return self.client.search(
            collection_name=self.collection,
            query_vector=list(vector),
            limit=top_k,
            query_filter=query_filter,
            search_params=params,
            with_payload=True,
            with_vectors=False,
            using=vector_name,
        )

    def search(
        self,
        *,
        query_vectors: Dict[str, Sequence[float]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 120,
        query_filter: Optional[Any] = None,
        ef: Optional[int] = 256,
        fusion: str = "rrf",
        extra_rankings: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ) -> List[Candidate]:
        """
        Perform weighted multi-field search and fuse scores.
        """
        t0 = time.time()
        weights = weights or {}
        rankings: Dict[str, List[Tuple[str, float]]] = {}
        payload_by_id: Dict[str, Dict[str, Any]] = {}
        vec_score_by_id_field: Dict[Tuple[str, str], float] = {}

        for vector_name, vec in query_vectors.items():
            if weights and weights.get(vector_name, 0.0) <= 0:
                continue
            hits = self._search_single(
                vector_name=vector_name,
                vector=vec,
                top_k=top_k,
                query_filter=query_filter,
                ef=ef,
            )
            hits_sorted = sorted(hits, key=lambda h: h.score or 0.0, reverse=True)
            rankings[vector_name] = []
            for h in hits_sorted:
                pid = str(h.id)
                score = float(h.score or 0.0)
                rankings[vector_name].append((pid, score))
                vec_score_by_id_field[(pid, vector_name)] = score
                if pid not in payload_by_id:
                    payload = (h.payload or {}) if hasattr(h, "payload") else {}
                    if self.payload_keys:
                        payload = {k: payload.get(k) for k in self.payload_keys}
                    payload_by_id[pid] = payload

        if extra_rankings:
            for vector_name, items in extra_rankings.items():
                rankings[vector_name] = items

        if fusion == "rrf":
            fused = reciprocal_rank_fusion(
                rankings, weights={k: float(weights.get(k, 1.0)) for k in rankings}
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion}")

        out: List[Candidate] = []
        for pid, fscore in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            payload = payload_by_id.get(pid, {})
            # Prefer canonical document_id; fall back to legacy doc_id
            doc = payload.get("document_id") or payload.get("doc_id")
            cand = Candidate(
                id=pid,
                text=str(payload.get("text", "")),
                doc_id=doc,  # holds canonical document_id value
                heading=payload.get("heading"),
                token_count=payload.get("token_count"),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {"text", "document_id", "doc_id", "heading", "token_count"}
                },
                fused_score=float(fscore),
                vector_score=vec_score_by_id_field.get((pid, "content"), 0.0),
                title_vec_score=vec_score_by_id_field.get((pid, "title"), 0.0),
                entity_vec_score=vec_score_by_id_field.get((pid, "entity"), 0.0),
            )
            out.append(cand)

        if out:
            out[0].metadata["_search_ms"] = int((time.time() - t0) * 1000)

        return out


# ------------------------- 3) Neo4j expansion helper --------------------------

_DEFAULT_REL_TYPES = (
    "SAME_HEADING",
    "CHILD_OF",
    "NEXT",
    "PREV",
    "MENTIONS",
    "PARENT_OF",
)
_DEFAULT_EDGE_WEIGHTS = {
    "SAME_HEADING": 1.0,
    "CHILD_OF": 0.7,
    "NEXT": 0.5,
    "PREV": 0.5,
    "MENTIONS": 0.4,
    "PARENT_OF": 0.7,
}


def expand_graph_neighbors(
    driver: Any,
    seed_ids: Sequence[str],
    *,
    rel_types: Sequence[str] = _DEFAULT_REL_TYPES,
    max_hops: int = 2,
    limit_per_seed: int = 200,
    edge_weights: Optional[Dict[str, float]] = None,
    same_doc_only: bool = False,
    tenant: Optional[str] = None,
    undirected: bool = True,
) -> Dict[str, List[GraphNeighbor]]:
    """
    Expand neighbors around each seed Chunk using typed relationships and compute edge-weighted scores.
    Guards against Section contamination by requiring chunk properties to exist.
    """
    if driver is None or Driver is None:
        raise RuntimeError(
            "Neo4j driver not available. Install `neo4j` Python driver and pass an initialized Driver."
        )

    edge_weights = edge_weights or dict(_DEFAULT_EDGE_WEIGHTS)
    rel_union = "|".join(rel_types)
    arrow = "-" if undirected else "->"

    cypher = f"""
    MATCH (c:Chunk)
    WHERE c.id = $seed
      AND exists(c.text) AND exists(c.token_count)
      {"AND c.tenant = $tenant" if tenant else ""}
    MATCH p=(c)-[r:{rel_union}*1..$max_hops]{arrow}(n:Chunk)
    WHERE exists(n.text) AND exists(n.token_count)
      AND (
        $same_doc_only = false OR
        coalesce(n.document_id, n.doc_id) = coalesce(c.document_id, c.doc_id)
      )
      {"AND n.tenant = $tenant" if tenant else ""}
    WITH n, [rel IN relationships(p) | type(rel)] AS types
    WITH n, types,
         reduce(w=0.0, t IN types | w + coalesce($edge_weights[t], 0.2)) AS edge_weight,
         size(types) AS hop
    RETURN n.id AS neighbor_id, hop, edge_weight, types AS path_types
    ORDER BY edge_weight DESC
    LIMIT $limit_per_seed
    """

    results: Dict[str, List[GraphNeighbor]] = {}
    with driver.session() as session:
        for seed in seed_ids:
            params = {
                "seed": seed,
                "max_hops": int(max_hops),
                "limit_per_seed": int(limit_per_seed),
                "edge_weights": edge_weights,
                "same_doc_only": bool(same_doc_only),
            }
            if tenant:
                params["tenant"] = tenant

            records = session.run(cypher, **params)
            neighbors: List[GraphNeighbor] = []
            for rec in records:
                neighbors.append(
                    GraphNeighbor(
                        neighbor_id=str(rec["neighbor_id"]),
                        hop=int(rec["hop"]),
                        edge_weight=float(rec["edge_weight"]),
                        path_types=(
                            list(rec["path_types"]) if "path_types" in rec else None
                        ),
                    )
                )
            results[str(seed)] = neighbors
    return results


# ------------------------- 4) Apply graph scores -------------------------------


def apply_graph_scores(
    candidates: List[Candidate],
    graph_expansion: Dict[str, List[GraphNeighbor]],
    *,
    graph_weight: float = 0.6,
) -> None:
    """
    Mutates `candidates` in place, adding graph_score and adjusting fused_score.
    """
    neighbor_score_by_id: Dict[str, float] = defaultdict(float)
    for _seed, neighs in graph_expansion.items():
        for nb in neighs:
            hop_factor = max(1, nb.hop)
            neighbor_score_by_id[nb.neighbor_id] = max(
                neighbor_score_by_id[nb.neighbor_id], nb.edge_weight / hop_factor
            )
    for c in candidates:
        if c.id in neighbor_score_by_id:
            c.graph_score = neighbor_score_by_id[c.id]
            c.fused_score += graph_weight * c.graph_score


# ------------------------- 5) End-to-end example -------------------------------


def example_retrieve_pack(
    *,
    qdrant_searcher: QdrantWeightedSearcher,
    query_vectors: Dict[str, Sequence[float]],
    weights: Dict[str, float],
    token_budget: int = 1600,
    driver: Optional[Any] = None,
    graph_rel_types: Sequence[str] = _DEFAULT_REL_TYPES,
    graph_weight: float = 0.6,
    top_k_per_field: int = 120,
    query_filter: Optional[Any] = None,
    ef: Optional[int] = 256,
    token_counter: Optional[Callable[[str], int]] = None,
    same_doc_only: bool = True,
    tenant: Optional[str] = None,
    undirected: bool = True,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """
    Reference pipeline showing how to:
      1) multi-field ANN search in Qdrant
      2) expand via Neo4j (optional)
      3) fuse graph scores
      4) budgeted packing

    Returns:
      (packed_chunks, stats)
    """
    seeds = qdrant_searcher.search(
        query_vectors=query_vectors,
        weights=weights,
        top_k=top_k_per_field,
        query_filter=query_filter,
        ef=ef,
        fusion="rrf",
    )

    # Mark first hit per canonical document_id as seed
    seen_doc: Set[str] = set()
    for c in seeds:
        if (
            c.doc_id and c.doc_id not in seen_doc
        ):  # c.doc_id holds canonical document_id
            c.is_seed = True
            seen_doc.add(c.doc_id)

    stats = {"search_seeds": len(seeds)}

    if driver is not None and seeds:
        seed_ids = [c.id for c in seeds[: min(40, len(seeds))]]
        expansion = expand_graph_neighbors(
            driver,
            seed_ids,
            rel_types=graph_rel_types,
            max_hops=2,
            limit_per_seed=200,
            same_doc_only=same_doc_only,
            tenant=tenant,
            undirected=undirected,
        )
        apply_graph_scores(seeds, expansion, graph_weight=graph_weight)
        stats["graph_expanded_seeds"] = len(seed_ids)

    seeds.sort(key=lambda c: c.fused_score, reverse=True)

    packed, pack_stats = budgeted_pack(
        seeds,
        token_budget,
        token_counter=token_counter,
        max_jaccard=0.80,
        per_doc_cap=0.40,
        force_include_seeds=True,
    )
    stats.update(pack_stats)
    return packed, stats


if __name__ == "__main__":
    print("hybrid_rag_helpers.py v2.2 (provider-free) provides:")
    print(" - budgeted_pack(candidates, token_budget, ...)")
    print(" - QdrantWeightedSearcher(client, 'chunks_multi').search(...)")
    print(" - expand_graph_neighbors(driver, seed_ids, ...)")
    print(" - example_retrieve_pack(...)")
