"""
Cross-document linking service for Phase 3.5 semantic similarity edges.

This module provides shared logic for creating RELATED_TO edges between
semantically similar documents. Used by both:
- Incremental mode: Called during document ingestion
- Batch mode: Called by backfill scripts for bulk processing

Architecture:
    Stage 1: Query Qdrant for chunks with similar doc_title vectors
    Stage 2: Aggregate chunks to documents (max-score wins)
    Stage 3: RRF fusion of dense and sparse results (if using rrf method)
    Stage 4: Filter by threshold and create RELATED_TO edges

Reference: docs/plans/phase-3.5-cross-doc-linking.md
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog
from qdrant_client import QdrantClient, models

from src.shared.config import CrossDocLinkingConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Vector field names (unlikely to change, kept as constants)
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME = "doc_title"
SPARSE_VECTOR_NAME = "doc_title-sparse"
COLBERT_VECTOR_NAME = "late-interaction"
CHUNK_FULLTEXT_INDEX = "chunk_text_fulltext"

# Default thresholds (can be overridden via config)
DEFAULT_THRESHOLDS = {
    "dense": 0.70,  # Cosine similarity threshold
    "rrf": 0.025,  # RRF score threshold
    "title_ft": 2.0,  # Lucene full-text score threshold
    "colbert": 0.40,  # ColBERT MaxSim threshold
}

# Discovery thresholds (relaxed, we filter after aggregation/fusion)
DISCOVERY_THRESHOLDS = {
    "dense": 0.50,
    "sparse": 0.0,  # Sparse scores can be low but still meaningful
}

# RRF constant (standard value from literature)
DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LinkingResult:
    """Result of linking a single document."""

    source_document_id: str
    edges_created: int = 0
    edges_updated: int = 0
    edges_pruned: int = 0  # ColBERT-pruned edges (Phase 4)
    candidates_found: int = 0
    method: str = "rrf"
    duration_ms: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for stats/logging."""
        return {
            "source_document_id": self.source_document_id,
            "edges_created": self.edges_created,
            "edges_updated": self.edges_updated,
            "edges_pruned": self.edges_pruned,
            "candidates_found": self.candidates_found,
            "method": self.method,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class BatchLinkingStats:
    """Aggregate statistics for batch linking operations."""

    docs_processed: int = 0
    docs_with_edges: int = 0
    docs_skipped_no_vector: int = 0
    docs_skipped_no_candidates: int = 0
    total_edges_created: int = 0
    total_edges_updated: int = 0
    total_candidates_found: int = 0
    dense_candidates: int = 0
    sparse_candidates: int = 0
    start_time: float = field(default_factory=time.time)
    method: str = "rrf"

    @classmethod
    def from_results(
        cls, results: List[LinkingResult], method: str = "rrf"
    ) -> "BatchLinkingStats":
        """Create stats from a list of LinkingResults."""
        stats = cls(method=method)
        for r in results:
            stats.docs_processed += 1
            if r.skipped:
                if r.skip_reason and "no_vector" in r.skip_reason:
                    stats.docs_skipped_no_vector += 1
                elif r.skip_reason and "no_candidates" in r.skip_reason:
                    stats.docs_skipped_no_candidates += 1
            else:
                if r.edges_created > 0:
                    stats.docs_with_edges += 1
                stats.total_edges_created += r.edges_created
                stats.total_edges_updated += r.edges_updated
                stats.total_candidates_found += r.candidates_found
        return stats


# ---------------------------------------------------------------------------
# Pure utility functions (module-level for testability)
# ---------------------------------------------------------------------------


def escape_lucene_query(text: str) -> str:
    """
    Escape special characters for Lucene full-text queries.

    Lucene special characters that need escaping:
    + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /

    We wrap the query in quotes for phrase matching, so we mainly need
    to escape quotes and backslashes within the phrase.
    """
    # Escape backslashes first (important order)
    text = text.replace("\\", "\\\\")
    # Escape quotes
    text = text.replace('"', '\\"')
    return text


def prepare_lucene_phrase_query(
    hint: str,
    min_length: int = 3,
    max_length: int = 512,
) -> str | None:
    """
    Prepare a hint string for safe Lucene fulltext phrase query.

    This function implements the "Hybrid" approach recommended for handling
    user-derived input in Lucene fulltext queries:
    1. Validate: Filter invalid hints (no alphanumeric content, too short)
    2. Sanitize: Strip control characters, normalize whitespace
    3. Escape: Handle backslash and quote (required inside phrase)
    4. Wrap: Enclose in double quotes for literal phrase matching

    Args:
        hint: Raw hint string from user input
        min_length: Minimum length after stripping (default: 3)
        max_length: Maximum length before truncation (default: 512)

    Returns:
        Escaped and quoted phrase string ready for Lucene, or None if invalid.

    Example:
        >>> prepare_lucene_phrase_query('NFS Support')
        '"NFS Support"'
        >>> prepare_lucene_phrase_query('C:\\\\Program Files')
        '"C:\\\\\\\\Program Files"'
        >>> prepare_lucene_phrase_query('---')  # No alphanumeric
        None
    """
    import re

    if not hint:
        return None

    # Strip control characters (ASCII 0x00-0x1f, 0x7f-0x9f)
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", hint)

    # Normalize whitespace and strip
    cleaned = " ".join(cleaned.split()).strip()

    # Check minimum length
    if len(cleaned) < min_length:
        return None

    # Must have at least one alphanumeric character
    if not any(c.isalnum() for c in cleaned):
        return None

    # Truncate to avoid Lucene term length limits
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(" ", 1)[0]  # Truncate at word boundary

    # Escape backslashes first (order matters!)
    escaped = cleaned.replace("\\", "\\\\")
    # Escape double quotes
    escaped = escaped.replace('"', '\\"')

    # Wrap in double quotes for phrase matching
    return f'"{escaped}"'


def aggregate_chunks_to_documents(
    chunk_hits: List[models.ScoredPoint],
    exclude_document_id: str,
) -> List[Tuple[str, float]]:
    """
    Aggregate chunk-level search results to document-level scores.

    Strategy: Max-Score Wins
    - Group chunks by document_id
    - Take the highest chunk score as the document's score
    - This captures the "best match" semantic from any chunk

    Args:
        chunk_hits: List of ScoredPoint from Qdrant search
        exclude_document_id: Document ID to exclude (the source document)

    Returns:
        List of (document_id, max_score) tuples, sorted by score descending
    """
    doc_scores: Dict[str, float] = defaultdict(float)

    for hit in chunk_hits:
        if not hit.payload:
            continue
        doc_id = hit.payload.get("document_id")
        if not doc_id or doc_id == exclude_document_id:
            continue
        doc_scores[doc_id] = max(doc_scores[doc_id], hit.score)

    return sorted(doc_scores.items(), key=lambda x: -x[1])


def reciprocal_rank_fusion(
    dense_docs: List[Tuple[str, float]],
    sparse_docs: List[Tuple[str, float]],
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """
    Combine document rankings using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum(1/(k + rank_i(d)))

    Why RRF works well:
    - Rank-based, not score-based (handles different score scales)
    - Documents appearing in BOTH lists get boosted
    - Robust to outliers in either retrieval method
    - Standard k=60 from literature works well empirically

    Args:
        dense_docs: List of (doc_id, score) from dense search, sorted by score desc
        sparse_docs: List of (doc_id, score) from sparse search, sorted by score desc
        k: RRF constant (default 60)

    Returns:
        List of (doc_id, rrf_score) tuples, sorted by RRF score descending
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    # Add dense contributions
    for rank, (doc_id, _) in enumerate(dense_docs):
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    # Add sparse contributions
    for rank, (doc_id, _) in enumerate(sparse_docs):
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: -x[1])


def compute_maxsim(source: np.ndarray, target: np.ndarray) -> float:
    """
    Compute ColBERT MaxSim score between two sets of token vectors.

    MaxSim algorithm:
    1. For each source token, find maximum cosine similarity to any target token
    2. Average these maximum similarities

    This captures how well each concept in the source is represented in the target,
    providing fine-grained semantic comparison that dense vectors miss.

    Args:
        source: [N, 1024] array of source token embeddings
        target: [M, 1024] array of target token embeddings

    Returns:
        Float score in range [0, 1]

    Complexity:
        O(N × M) dot products, ~40k ops for 200×200 tokens (sub-ms on CPU)

    Reference:
        Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search
        via Contextualized Late Interaction over BERT" (SIGIR 2020)
    """
    # Handle edge cases
    if source.size == 0 or target.size == 0:
        return 0.0

    # Ensure 2D arrays
    if source.ndim == 1:
        source = source.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)

    # Normalize vectors for cosine similarity
    source_norm = source / (np.linalg.norm(source, axis=1, keepdims=True) + 1e-9)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-9)

    # Compute all pairwise similarities: [N, M]
    similarities = source_norm @ target_norm.T

    # MaxSim: max over target tokens for each source token, then average
    max_sims = similarities.max(axis=1)

    return float(max_sims.mean())


def get_document_colbert_vectors(
    qdrant_client: QdrantClient,
    document_id: str,
    collection_name: str,
    max_chunks: int = 3,
    max_tokens: int = 200,
) -> Optional[np.ndarray]:
    """
    Fetch ColBERT vectors from first N chunks of a document.

    ColBERT stores per-token embeddings that enable fine-grained semantic
    comparison via MaxSim. This function retrieves and concatenates vectors
    from the first few chunks to capture the document's key concepts.

    Strategy:
    1. Query Qdrant for chunks where document_id=X, order < max_chunks
    2. Sort by order ascending
    3. Concatenate ColBERT arrays into single [total_tokens, 1024] matrix
    4. Truncate to max_tokens if total exceeds budget

    Why multiple chunks?
    - First chunk alone (~70 tokens) may be just a title or TOC
    - 150-200 tokens captures document's key concepts
    - Balance between coverage and TOC noise

    Args:
        qdrant_client: Qdrant client instance
        document_id: Document ID to fetch vectors for
        collection_name: Qdrant collection name
        max_chunks: Maximum chunks to fetch (default: 3)
        max_tokens: Maximum tokens to return (default: 200)

    Returns:
        numpy array of shape [N, 1024] where N <= max_tokens,
        or None if no vectors found

    Edge Cases:
        - Document has < max_chunks: Use all available chunks
        - No ColBERT vectors found: Return None
        - Empty chunk ColBERT array: Skip and continue
    """
    try:
        # Query for first N chunks ordered by position
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="order",
                        range=models.Range(gte=0, lt=max_chunks),
                    ),
                ]
            ),
            limit=max_chunks,
            with_vectors=[COLBERT_VECTOR_NAME],
            with_payload=["order"],
        )

        if not points:
            logger.debug(
                "colbert_no_chunks_found",
                document_id=document_id[:12] if document_id else "unknown",
            )
            return None

        # Sort by order and collect ColBERT vectors
        points_sorted = sorted(
            points,
            key=lambda p: p.payload.get("order", 0) if p.payload else 0,
        )

        colbert_arrays = []
        total_tokens = 0

        for point in points_sorted:
            if not point.vector:
                continue

            # Handle both dict and direct vector formats
            if isinstance(point.vector, dict):
                colbert_vec = point.vector.get(COLBERT_VECTOR_NAME)
            else:
                colbert_vec = point.vector

            if colbert_vec is None:
                continue

            # Convert to numpy array
            arr = np.array(colbert_vec)

            # Handle both [N, 1024] and flat arrays
            if arr.ndim == 1:
                # Single token - reshape to [1, dim]
                arr = arr.reshape(1, -1)

            # Check token budget
            if total_tokens + len(arr) <= max_tokens:
                colbert_arrays.append(arr)
                total_tokens += len(arr)
            else:
                # Truncate to fit remaining budget
                remaining = max_tokens - total_tokens
                if remaining > 0:
                    colbert_arrays.append(arr[:remaining])
                    total_tokens += remaining
                break

        if not colbert_arrays:
            logger.debug(
                "colbert_no_vectors_in_chunks",
                document_id=document_id[:12] if document_id else "unknown",
                chunks_found=len(points),
            )
            return None

        # Concatenate all arrays
        result = np.vstack(colbert_arrays)

        logger.debug(
            "colbert_vectors_fetched",
            document_id=document_id[:12] if document_id else "unknown",
            chunks_used=len(colbert_arrays),
            total_tokens=len(result),
        )

        return result

    except Exception as e:
        logger.warning(
            "colbert_fetch_failed",
            document_id=document_id[:12] if document_id else "unknown",
            error=str(e),
        )
        return None


# ---------------------------------------------------------------------------
# CrossDocLinker class
# ---------------------------------------------------------------------------


class CrossDocLinker:
    """
    Handles cross-document linking via semantic similarity.

    Supports multiple linking methods:
    - dense: Dense vector similarity only
    - rrf: RRF fusion of dense + sparse vectors
    - title_ft: Full-text search for title mentions

    Usage:
        linker = CrossDocLinker(neo4j_driver, qdrant_client, config)

        # Incremental mode (single document)
        result = linker.link_document(doc_id, doc_title, vectors...)

        # Batch mode (all documents)
        results = linker.link_all_documents(method="rrf", dry_run=False)
    """

    def __init__(
        self,
        neo4j_driver,
        qdrant_client: QdrantClient,
        config: CrossDocLinkingConfig,
    ):
        """
        Initialize the linker with database connections and config.

        Args:
            neo4j_driver: Neo4j driver instance
            qdrant_client: Qdrant client instance
            config: Linking configuration
        """
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config

    # -----------------------------------------------------------------------
    # Qdrant operations
    # -----------------------------------------------------------------------

    def _get_doc_vectors(
        self,
        doc_id: str,
        include_sparse: bool = False,
    ) -> Tuple[Optional[List[float]], Optional[models.SparseVector]]:
        """
        Get the doc_title vectors (dense and optionally sparse) for a document.

        Since all chunks from the same document have identical doc_title vectors,
        we only need to fetch one chunk to get the document's vectors.

        Args:
            doc_id: Document ID
            include_sparse: Whether to also fetch sparse vectors

        Returns:
            Tuple of (dense_vector, sparse_vector)
            sparse_vector is a SparseVector model with 'indices' and 'values' attrs
        """
        vector_names = [DENSE_VECTOR_NAME]
        if include_sparse:
            vector_names.append(SPARSE_VECTOR_NAME)

        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=1,
                with_vectors=vector_names,
                with_payload=False,
            )

            if not points or not points[0].vector:
                return None, None

            vec = points[0].vector
            if isinstance(vec, dict):
                dense = vec.get(DENSE_VECTOR_NAME)
                sparse = vec.get(SPARSE_VECTOR_NAME) if include_sparse else None
                return dense, sparse

            # Single vector format (shouldn't happen with named vectors)
            return vec, None

        except Exception as e:
            logger.warning(
                "cross_doc_get_vectors_failed",
                document_id=doc_id[:12] if doc_id else "unknown",
                error=str(e),
            )
            return None, None

    def _search_similar_chunks_dense(
        self,
        query_vector: List[float],
        exclude_document_id: str,
        limit: Optional[int] = None,
    ) -> List[models.ScoredPoint]:
        """
        Search for chunks with similar doc_title dense vectors.

        Returns up to `limit` chunks, excluding those from the source document.
        Uses query_points() API (qdrant-client >= 1.12).
        """
        limit = limit or self.config.chunk_limit
        try:
            response = self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=exclude_document_id),
                        )
                    ]
                ),
                limit=limit,
                score_threshold=DISCOVERY_THRESHOLDS["dense"],
                with_payload=["document_id"],
            )
            return response.points
        except Exception as e:
            logger.warning(
                "cross_doc_dense_search_failed",
                document_id=(
                    exclude_document_id[:12] if exclude_document_id else "unknown"
                ),
                error=str(e),
            )
            return []

    def _search_similar_chunks_sparse(
        self,
        sparse_vector: models.SparseVector,
        exclude_document_id: str,
        limit: Optional[int] = None,
    ) -> List[models.ScoredPoint]:
        """
        Search for chunks with similar doc_title sparse vectors.

        Sparse vectors enable lexical/keyword matching - documents sharing
        specific terms in their titles will score higher.
        Uses query_points() API (qdrant-client >= 1.12).

        Args:
            sparse_vector: SparseVector model with 'indices' and 'values' attributes

        Returns:
            Up to `limit` chunks, excluding those from the source document.
        """
        limit = limit or self.config.chunk_limit
        try:
            response = self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                query_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=exclude_document_id),
                        )
                    ]
                ),
                limit=limit,
                with_payload=["document_id"],
            )
            return response.points
        except Exception as e:
            logger.warning(
                "cross_doc_sparse_search_failed",
                document_id=(
                    exclude_document_id[:12] if exclude_document_id else "unknown"
                ),
                error=str(e),
            )
            return []

    # -----------------------------------------------------------------------
    # Neo4j operations
    # -----------------------------------------------------------------------

    def _search_title_mentions(
        self,
        source_document_id: str,
        source_title: str,
        min_score: float = 2.0,
        limit: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Search for chunks that explicitly mention this document's title.

        Uses Neo4j full-text index with phrase matching.
        Aggregates chunk results to document level using max-score strategy.

        Args:
            source_document_id: ID of source document (to exclude from results)
            source_title: Title to search for in chunk text
            min_score: Minimum Lucene score threshold
            limit: Maximum chunks to retrieve before aggregation

        Returns:
            List of (document_id, max_score) tuples, sorted by score descending
        """
        # Escape and prepare the title for phrase search
        escaped_title = escape_lucene_query(source_title)
        search_query = f'"{escaped_title}"'

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $search_query, {limit: $fetch_limit})
                    YIELD node, score
                    WHERE node.document_id <> $source_document_id AND score > $min_score
                    WITH node.document_id AS target_doc_id, max(score) AS max_score
                    RETURN target_doc_id, max_score
                    ORDER BY max_score DESC
                    """,
                    index_name=CHUNK_FULLTEXT_INDEX,
                    search_query=search_query,
                    source_document_id=source_document_id,
                    min_score=min_score,
                    fetch_limit=limit,
                )
                return [(r["target_doc_id"], r["max_score"]) for r in result]
        except Exception as e:
            logger.warning(
                "cross_doc_title_search_failed",
                title=source_title[:30] if source_title else "unknown",
                error=str(e),
            )
            return []

    def _get_document_title(self, document_id: str) -> Optional[str]:
        """Get title for a specific document."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (d:Document {id: $document_id}) RETURN d.title AS title",
                    document_id=document_id,
                )
                record = result.single()
                return record["title"] if record else None
        except Exception:
            return None

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        score: float,
        method: str,
        phase: str,
        dry_run: bool = False,
    ) -> Tuple[bool, bool]:
        """
        Create a RELATED_TO edge between two documents.

        Uses MERGE for idempotency - re-running updates scores
        rather than creating duplicate edges.

        Args:
            source_id: Source document ID
            target_id: Target document ID
            score: Similarity score
            method: Method used ('dense_similarity', 'rrf_fusion', 'title_mention')
            phase: Phase identifier for rollback targeting
            dry_run: If True, don't actually create the edge

        Returns:
            Tuple of (created, updated) booleans
        """
        if dry_run:
            return True, False

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (source:Document {id: $source_id})
                    MATCH (target:Document {id: $target_id})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET
                        r.score = $score,
                        r.method = $method,
                        r.phase = $phase,
                        r.created_at = datetime()
                    ON MATCH SET
                        r.score = $score,
                        r.method = $method,
                        r.phase = $phase,
                        r.updated_at = datetime()
                    RETURN
                        r.created_at = r.updated_at AS was_created
                    """,
                    source_id=source_id,
                    target_id=target_id,
                    score=score,
                    method=method,
                    phase=phase,
                )
                record = result.single()
                if record:
                    was_created = record.get("was_created", True)
                    return was_created, not was_created
                return False, False
        except Exception as e:
            logger.warning(
                "cross_doc_create_edge_failed",
                source=source_id[:8] if source_id else "?",
                target=target_id[:8] if target_id else "?",
                error=str(e),
            )
            return False, False

    # -----------------------------------------------------------------------
    # ColBERT Reranking Methods (Phase 4)
    # -----------------------------------------------------------------------

    def _update_edge_colbert_score(
        self,
        source_id: str,
        target_id: str,
        colbert_score: float,
    ) -> bool:
        """
        Update an existing edge with ColBERT score.

        Sets the colbert_score property and updates the method to indicate
        ColBERT reranking was applied.

        Args:
            source_id: Source document ID
            target_id: Target document ID
            colbert_score: MaxSim score from ColBERT comparison

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (s:Document {id: $source_id})-[r:RELATED_TO]->(t:Document {id: $target_id})
                    SET r.colbert_score = $colbert_score,
                        r.method = 'rrf_fusion+colbert',
                        r.reranked_at = datetime()
                    """,
                    source_id=source_id,
                    target_id=target_id,
                    colbert_score=colbert_score,
                )
            return True
        except Exception as e:
            logger.warning(
                "colbert_update_edge_failed",
                source_id=source_id[:12] if source_id else "unknown",
                target_id=target_id[:12] if target_id else "unknown",
                error=str(e),
            )
            return False

    def _prune_edge(self, source_id: str, target_id: str) -> bool:
        """
        Delete an edge that failed the ColBERT threshold.

        Called when ColBERT MaxSim score is below the configured threshold,
        indicating the edge was a false positive from RRF fusion.

        Args:
            source_id: Source document ID
            target_id: Target document ID

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (s:Document {id: $source_id})-[r:RELATED_TO]->(t:Document {id: $target_id})
                    DELETE r
                    """,
                    source_id=source_id,
                    target_id=target_id,
                )
            logger.debug(
                "colbert_edge_pruned",
                source_id=source_id[:12] if source_id else "unknown",
                target_id=target_id[:12] if target_id else "unknown",
            )
            return True
        except Exception as e:
            logger.warning(
                "colbert_prune_edge_failed",
                source_id=source_id[:12] if source_id else "unknown",
                target_id=target_id[:12] if target_id else "unknown",
                error=str(e),
            )
            return False

    def _rerank_edges_colbert(
        self,
        source_document_id: str,
        edges_created: List[Tuple[str, float]],
        source_colbert: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        Rerank newly-created edges using ColBERT MaxSim.

        For each edge:
        1. Fetch target document's ColBERT vectors
        2. Compute MaxSim score against source vectors
        3. If score >= threshold: update edge with colbert_score
        4. If score < threshold: delete edge (prune false positive)

        Args:
            source_document_id: Source document ID
            edges_created: List of (target_id, rrf_score) tuples
            source_colbert: Pre-fetched source ColBERT vectors (optional optimization)

        Returns:
            Tuple of (edges_kept, edges_pruned)
        """
        if not edges_created:
            return 0, 0

        threshold = self.config.colbert_threshold
        edges_kept = 0
        edges_pruned = 0

        # Fetch source ColBERT vectors if not provided
        if source_colbert is None:
            source_colbert = get_document_colbert_vectors(
                self.qdrant_client,
                source_document_id,
                self.config.collection_name,
                self.config.colbert_max_chunks,
                self.config.colbert_max_tokens,
            )

        if source_colbert is None:
            # Can't rerank without source vectors - keep all edges
            logger.warning(
                "colbert_rerank_skipped_no_source",
                document_id=(
                    source_document_id[:12] if source_document_id else "unknown"
                ),
                edges_count=len(edges_created),
            )
            return len(edges_created), 0

        for target_id, rrf_score in edges_created:
            # Fetch target ColBERT vectors
            target_colbert = get_document_colbert_vectors(
                self.qdrant_client,
                target_id,
                self.config.collection_name,
                self.config.colbert_max_chunks,
                self.config.colbert_max_tokens,
            )

            if target_colbert is None:
                # Can't compute score - keep edge (don't punish missing data)
                logger.debug(
                    "colbert_rerank_no_target_vectors",
                    source_id=(
                        source_document_id[:12] if source_document_id else "unknown"
                    ),
                    target_id=target_id[:12] if target_id else "unknown",
                )
                edges_kept += 1
                continue

            # Compute ColBERT MaxSim score
            colbert_score = compute_maxsim(source_colbert, target_colbert)

            if colbert_score >= threshold:
                # Keep edge and update with ColBERT score
                if self._update_edge_colbert_score(
                    source_document_id, target_id, colbert_score
                ):
                    edges_kept += 1
                    logger.debug(
                        "colbert_edge_kept",
                        source_id=(
                            source_document_id[:12] if source_document_id else "unknown"
                        ),
                        target_id=target_id[:12] if target_id else "unknown",
                        rrf_score=round(rrf_score, 4),
                        colbert_score=round(colbert_score, 4),
                    )
                else:
                    edges_kept += 1  # Update failed but edge still exists
            else:
                # Prune edge below threshold
                if self._prune_edge(source_document_id, target_id):
                    edges_pruned += 1
                    logger.debug(
                        "colbert_edge_filtered",
                        source_id=(
                            source_document_id[:12] if source_document_id else "unknown"
                        ),
                        target_id=target_id[:12] if target_id else "unknown",
                        rrf_score=round(rrf_score, 4),
                        colbert_score=round(colbert_score, 4),
                        threshold=threshold,
                    )
                else:
                    edges_kept += 1  # Prune failed, edge still exists

        logger.info(
            "colbert_rerank_complete",
            document_id=source_document_id[:12] if source_document_id else "unknown",
            edges_kept=edges_kept,
            edges_pruned=edges_pruned,
            threshold=threshold,
        )

        return edges_kept, edges_pruned

    # -----------------------------------------------------------------------
    # Main linking methods
    # -----------------------------------------------------------------------

    def link_document(
        self,
        doc_id: str,
        doc_title: str,
        doc_title_vector: Optional[List[float]] = None,
        doc_title_sparse: Optional[Union[models.SparseVector, Dict]] = None,
        dry_run: bool = False,
    ) -> LinkingResult:
        """
        Link a single document to semantically similar documents.

        This is the main entry point for incremental mode during ingestion.
        If ColBERT reranking is enabled, edges will be reranked after creation
        and weak edges will be pruned.

        Args:
            doc_id: Document ID
            doc_title: Document title
            doc_title_vector: Pre-computed dense vector (optional, will fetch if None)
            doc_title_sparse: Pre-computed sparse vector (optional, may be None)
            dry_run: If True, don't create edges

        Returns:
            LinkingResult with statistics
        """
        start_time = time.time()
        result = LinkingResult(source_document_id=doc_id, method=self.config.method)

        # Get vectors if not provided
        if doc_title_vector is None:
            include_sparse = self.config.method == "rrf"
            doc_title_vector, fetched_sparse = self._get_doc_vectors(
                doc_id, include_sparse=include_sparse
            )
            if doc_title_sparse is None:
                doc_title_sparse = fetched_sparse

        # Validate we have vectors
        if doc_title_vector is None:
            result.skipped = True
            result.skip_reason = "no_vector"
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        # Route to appropriate method
        if self.config.method == "dense":
            candidates = self._link_dense_only(doc_id, doc_title_vector)
            threshold = self.config.dense_threshold
            method_name = "dense_similarity"
            phase = "3.5a"
        elif self.config.method == "rrf":
            candidates = self._link_rrf(doc_id, doc_title_vector, doc_title_sparse)
            threshold = self.config.rrf_threshold
            method_name = "rrf_fusion"
            phase = "3.5b"
        elif self.config.method == "title_ft":
            candidates = self._search_title_mentions(
                doc_id, doc_title, min_score=self.config.rrf_threshold
            )
            threshold = DEFAULT_THRESHOLDS["title_ft"]
            method_name = "title_mention"
            phase = "3.5c"
        else:
            result.skipped = True
            result.skip_reason = f"unknown_method:{self.config.method}"
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        result.candidates_found = len(candidates)

        if not candidates:
            result.skipped = True
            result.skip_reason = "no_candidates"
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        # Create edges for candidates above threshold
        edges_created = 0
        edges_updated = 0
        created_edges: List[Tuple[str, float]] = []  # Track for ColBERT reranking

        for target_id, score in candidates:
            if score < threshold:
                break
            if edges_created >= self.config.max_edges_per_doc:
                break

            created, updated = self._create_edge(
                source_id=doc_id,
                target_id=target_id,
                score=score,
                method=method_name,
                phase=phase,
                dry_run=dry_run,
            )

            if created:
                edges_created += 1
                created_edges.append((target_id, score))
            if updated:
                edges_updated += 1
                created_edges.append((target_id, score))

        result.edges_created = edges_created
        result.edges_updated = edges_updated

        # ColBERT reranking (Phase 4)
        if self.config.colbert_rerank and created_edges and not dry_run:
            # Fetch source ColBERT vectors once
            source_colbert = get_document_colbert_vectors(
                self.qdrant_client,
                doc_id,
                self.config.collection_name,
                self.config.colbert_max_chunks,
                self.config.colbert_max_tokens,
            )

            if source_colbert is not None:
                edges_kept, edges_pruned = self._rerank_edges_colbert(
                    doc_id,
                    created_edges,
                    source_colbert,
                )
                result.edges_created = edges_kept
                result.edges_pruned = edges_pruned
                if edges_kept > 0:
                    result.method = "rrf_fusion+colbert"
            else:
                logger.warning(
                    "colbert_rerank_skipped",
                    document_id=doc_id[:12] if doc_id else "unknown",
                    reason="no_source_vectors",
                )

        result.duration_ms = int((time.time() - start_time) * 1000)

        return result

    def _link_dense_only(
        self,
        doc_id: str,
        dense_vector: List[float],
    ) -> List[Tuple[str, float]]:
        """Link using dense vectors only (Phase 3.5a)."""
        chunk_hits = self._search_similar_chunks_dense(dense_vector, doc_id)
        return aggregate_chunks_to_documents(chunk_hits, doc_id)

    def _link_rrf(
        self,
        doc_id: str,
        dense_vector: List[float],
        sparse_vector: Optional[Union[models.SparseVector, Dict]],
    ) -> List[Tuple[str, float]]:
        """Link using RRF fusion of dense + sparse (Phase 3.5b)."""
        # Dense search
        dense_hits = self._search_similar_chunks_dense(dense_vector, doc_id)
        dense_docs = aggregate_chunks_to_documents(dense_hits, doc_id)

        # Sparse search (if vector available)
        sparse_docs: List[Tuple[str, float]] = []
        if sparse_vector is not None:
            # Handle both SparseVector model and dict formats
            if hasattr(sparse_vector, "indices") and sparse_vector.indices:
                sparse_hits = self._search_similar_chunks_sparse(sparse_vector, doc_id)
                sparse_docs = aggregate_chunks_to_documents(sparse_hits, doc_id)
            elif isinstance(sparse_vector, dict) and sparse_vector.get("indices"):
                # Convert dict to SparseVector model
                sv = models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                )
                sparse_hits = self._search_similar_chunks_sparse(sv, doc_id)
                sparse_docs = aggregate_chunks_to_documents(sparse_hits, doc_id)

        # If no sparse results, fall back to dense-only
        if not sparse_docs:
            return dense_docs

        # RRF fusion
        return reciprocal_rank_fusion(dense_docs, sparse_docs, k=self.config.rrf_k)

    def link_all_documents(
        self,
        method: Optional[str] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
    ) -> List[LinkingResult]:
        """
        Link all documents in batch mode.

        This is the main entry point for backfill/batch operations.

        Args:
            method: Override the configured method
            dry_run: If True, don't create edges
            limit: Limit number of documents to process (for testing)

        Returns:
            List of LinkingResult for each document
        """
        # Override method if specified
        original_method = self.config.method
        if method:
            self.config.method = method

        try:
            # Fetch all documents
            documents = self._get_all_documents()
            if limit:
                documents = documents[:limit]

            results: List[LinkingResult] = []

            for doc in documents:
                doc_id = doc["id"]
                doc_title = doc.get("title", "")

                result = self.link_document(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    dry_run=dry_run,
                )
                results.append(result)

            return results
        finally:
            # Restore original method
            self.config.method = original_method

    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """Fetch all documents from Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document)
                    RETURN d.id AS id, d.title AS title
                    ORDER BY d.title
                    """
                )
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("cross_doc_get_all_documents_failed", error=str(e))
            return []


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "CrossDocLinkingConfig",
    "CrossDocLinker",
    "LinkingResult",
    "BatchLinkingStats",
    # Pure functions
    "escape_lucene_query",
    "aggregate_chunks_to_documents",
    "reciprocal_rank_fusion",
    "compute_maxsim",
    "get_document_colbert_vectors",
    # Constants
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "COLBERT_VECTOR_NAME",
    "DEFAULT_THRESHOLDS",
]
