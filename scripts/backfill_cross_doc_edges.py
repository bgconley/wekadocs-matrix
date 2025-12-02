#!/usr/bin/env python3
"""
Phase 3.5 Cross-Document Linking via Vector Similarity

Creates RELATED_TO edges between semantically similar documents using
doc_title vectors (dense and/or sparse) from Qdrant, and full-text search.

Phases:
    3.5a: Dense similarity only (doc_title vectors)
    3.5b: Dense + Sparse with RRF fusion (doc_title + doc_title-sparse)
    3.5c: Full-text title mentions (Neo4j chunk_text_fulltext index)

Architecture (per plan docs/plans/phase-3.5-cross-doc-linking.md):
    Stage 1: Query Qdrant for chunks with similar vectors (limit=100)
    Stage 2: Aggregate chunks to documents (max-score wins)
    Stage 3: RRF fusion of dense and sparse results (if using rrf method)
    Stage 4: Filter by threshold and create RELATED_TO edges

Usage:
    # Dense-only mode (Phase 3.5a)
    python scripts/backfill_cross_doc_edges.py --dry-run --method dense
    python scripts/backfill_cross_doc_edges.py --execute --method dense

    # RRF fusion mode (Phase 3.5b) - combines dense + sparse
    python scripts/backfill_cross_doc_edges.py --dry-run --method rrf
    python scripts/backfill_cross_doc_edges.py --execute --method rrf

    # Full-text title mention mode (Phase 3.5c) - finds explicit title references
    python scripts/backfill_cross_doc_edges.py --dry-run --method title_ft
    python scripts/backfill_cross_doc_edges.py --execute --method title_ft

    # Custom threshold and limits
    python scripts/backfill_cross_doc_edges.py --execute --method rrf --threshold 0.03

    # Test with limited documents
    python scripts/backfill_cross_doc_edges.py --dry-run --limit-docs 5 --verbose

Environment variables:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_PASSWORD: Neo4j password (default: testpassword123)
    QDRANT_HOST: Qdrant host (default: localhost)
    QDRANT_PORT: Qdrant port (default: 6333)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "chunks_multi_bge_m3"
DENSE_VECTOR_NAME = "doc_title"
SPARSE_VECTOR_NAME = "doc_title-sparse"

# Defaults (can be overridden via CLI)
DEFAULT_CHUNK_LIMIT = 100  # Chunks to fetch per query (before aggregation)
DEFAULT_MAX_EDGES_PER_DOC = 5  # Limit edges per source document

# Method-specific thresholds
# Dense similarity scores range 0-1 (cosine similarity)
# RRF scores are much smaller (sum of 1/(k+rank) terms)
# Title FT scores are Lucene BM25-style scores (typically 1.0-5.0 range)
THRESHOLDS = {
    "dense": 0.70,  # Cosine similarity threshold
    "rrf": 0.025,  # RRF score threshold (appears in 2+ methods or high-ranked in 1)
    "title_ft": 2.0,  # Lucene full-text score threshold
}

# Full-text index name for chunk content
CHUNK_FULLTEXT_INDEX = "chunk_text_fulltext"

# Discovery thresholds (relaxed, we filter after aggregation/fusion)
DISCOVERY_THRESHOLDS = {
    "dense": 0.50,
    "sparse": 0.0,  # Sparse scores can be low but still meaningful
}

# RRF constant (standard value from literature)
RRF_K = 60


# ---------------------------------------------------------------------------
# Statistics Tracking
# ---------------------------------------------------------------------------


@dataclass
class BackfillStats:
    """Track statistics for the backfill operation."""

    docs_processed: int = 0
    docs_with_edges: int = 0
    docs_skipped_no_vector: int = 0
    docs_skipped_no_candidates: int = 0
    total_edges_created: int = 0
    total_edges_updated: int = 0  # Edges that already existed (score updated)
    total_candidates_found: int = 0
    dense_candidates: int = 0
    sparse_candidates: int = 0
    score_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "0.90+": 0,
            "0.80-0.90": 0,
            "0.70-0.80": 0,
            "0.60-0.70": 0,
            "<0.60": 0,
        }
    )
    rrf_score_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "0.05+": 0,
            "0.04-0.05": 0,
            "0.03-0.04": 0,
            "0.02-0.03": 0,
            "<0.02": 0,
        }
    )
    title_ft_score_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "4.0+": 0,
            "3.0-4.0": 0,
            "2.5-3.0": 0,
            "2.0-2.5": 0,
            "<2.0": 0,
        }
    )
    title_ft_candidates: int = 0  # Chunks found via full-text search
    start_time: float = field(default_factory=time.time)
    method: str = "dense"

    def record_edge(self, score: float) -> None:
        """Record an edge creation with its score."""
        self.total_edges_created += 1

        if self.method == "rrf":
            # RRF scores are smaller
            if score >= 0.05:
                self.rrf_score_distribution["0.05+"] += 1
            elif score >= 0.04:
                self.rrf_score_distribution["0.04-0.05"] += 1
            elif score >= 0.03:
                self.rrf_score_distribution["0.03-0.04"] += 1
            elif score >= 0.02:
                self.rrf_score_distribution["0.02-0.03"] += 1
            else:
                self.rrf_score_distribution["<0.02"] += 1
        elif self.method == "title_ft":
            # Lucene full-text scores
            if score >= 4.0:
                self.title_ft_score_distribution["4.0+"] += 1
            elif score >= 3.0:
                self.title_ft_score_distribution["3.0-4.0"] += 1
            elif score >= 2.5:
                self.title_ft_score_distribution["2.5-3.0"] += 1
            elif score >= 2.0:
                self.title_ft_score_distribution["2.0-2.5"] += 1
            else:
                self.title_ft_score_distribution["<2.0"] += 1
        else:
            # Dense similarity scores
            if score >= 0.90:
                self.score_distribution["0.90+"] += 1
            elif score >= 0.80:
                self.score_distribution["0.80-0.90"] += 1
            elif score >= 0.70:
                self.score_distribution["0.70-0.80"] += 1
            elif score >= 0.60:
                self.score_distribution["0.60-0.70"] += 1
            else:
                self.score_distribution["<0.60"] += 1

    def summary(self, phase: str) -> str:
        """Generate a summary report."""
        elapsed = time.time() - self.start_time
        lines = [
            "",
            "=" * 60,
            f"Phase {phase} Backfill Complete",
            "=" * 60,
            f"Method:                     {self.method}",
            f"Documents processed:        {self.docs_processed}",
            f"Documents with new edges:   {self.docs_with_edges}",
            f"Documents skipped (no vec): {self.docs_skipped_no_vector}",
            f"Documents skipped (no cand):{self.docs_skipped_no_candidates}",
            f"Total edges created:        {self.total_edges_created}",
            f"Total candidates evaluated: {self.total_candidates_found}",
        ]

        if self.method == "rrf":
            lines.extend(
                [
                    f"  - Dense candidates:       {self.dense_candidates}",
                    f"  - Sparse candidates:      {self.sparse_candidates}",
                    "",
                    "RRF Score distribution of created edges:",
                ]
            )
            for bucket, count in self.rrf_score_distribution.items():
                lines.append(f"  {bucket}: {count}")
        elif self.method == "title_ft":
            lines.extend(
                [
                    f"  - Full-text candidates:   {self.title_ft_candidates}",
                    "",
                    "Lucene Score distribution of created edges:",
                ]
            )
            for bucket, count in self.title_ft_score_distribution.items():
                lines.append(f"  {bucket}: {count}")
        else:
            lines.extend(
                [
                    "",
                    "Score distribution of created edges:",
                ]
            )
            for bucket, count in self.score_distribution.items():
                lines.append(f"  {bucket}: {count}")

        lines.extend(
            [
                "",
                f"Elapsed time: {elapsed:.1f}s",
                "=" * 60,
            ]
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Database Access Functions
# ---------------------------------------------------------------------------


def get_neo4j_driver():
    """Create Neo4j driver from environment variables."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    password = os.getenv("NEO4J_PASSWORD", "testpassword123")
    return GraphDatabase.driver(uri, auth=("neo4j", password))


def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client from environment variables."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def get_all_documents(neo4j_driver) -> List[Dict]:
    """Fetch all documents from Neo4j with their IDs and titles."""
    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            RETURN d.id AS id, d.title AS title
            ORDER BY d.title
        """
        )
        return [dict(r) for r in result]


def get_document_title(neo4j_driver, doc_id: str) -> Optional[str]:
    """Get title for a specific document."""
    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document {id: $doc_id})
            RETURN d.title AS title
        """,
            doc_id=doc_id,
        )
        record = result.single()
        return record["title"] if record else None


# ---------------------------------------------------------------------------
# Qdrant Vector Operations
# ---------------------------------------------------------------------------


def get_doc_vectors(
    qdrant: QdrantClient, doc_id: str, include_sparse: bool = False
) -> Tuple[Optional[List[float]], Optional[models.SparseVector]]:
    """
    Get the doc_title vectors (dense and optionally sparse) for a document.

    Since all chunks from the same document have identical doc_title vectors,
    we only need to fetch one chunk to get the document's vectors.

    Returns:
        Tuple of (dense_vector, sparse_vector)
        sparse_vector is a SparseVector model with 'indices' and 'values' attrs
    """
    vector_names = [DENSE_VECTOR_NAME]
    if include_sparse:
        vector_names.append(SPARSE_VECTOR_NAME)

    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=doc_id)
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
        print(f"  [ERROR] Failed to get vectors for doc {doc_id[:12]}...: {e}")
        return None, None


def search_similar_chunks_dense(
    qdrant: QdrantClient,
    query_vector: List[float],
    exclude_doc_id: str,
    limit: int = DEFAULT_CHUNK_LIMIT,
) -> List[models.ScoredPoint]:
    """
    Search for chunks with similar doc_title dense vectors.

    Returns up to `limit` chunks, excluding those from the source document.
    """
    try:
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name=DENSE_VECTOR_NAME, vector=query_vector
            ),
            query_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=exclude_doc_id)
                    )
                ]
            ),
            limit=limit,
            score_threshold=DISCOVERY_THRESHOLDS["dense"],
            with_payload=["document_id"],
        )
    except Exception as e:
        print(f"  [ERROR] Dense search failed for doc {exclude_doc_id[:12]}...: {e}")
        return []


def search_similar_chunks_sparse(
    qdrant: QdrantClient,
    sparse_vector: models.SparseVector,
    exclude_doc_id: str,
    limit: int = DEFAULT_CHUNK_LIMIT,
) -> List[models.ScoredPoint]:
    """
    Search for chunks with similar doc_title sparse vectors.

    Sparse vectors enable lexical/keyword matching - documents sharing
    specific terms in their titles will score higher.

    Args:
        sparse_vector: SparseVector model with 'indices' and 'values' attributes

    Returns up to `limit` chunks, excluding those from the source document.
    """
    try:
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedSparseVector(
                name=SPARSE_VECTOR_NAME,
                vector=sparse_vector,  # Already a SparseVector model
            ),
            query_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=exclude_doc_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=["document_id"],
        )
    except Exception as e:
        print(f"  [ERROR] Sparse search failed for doc {exclude_doc_id[:12]}...: {e}")
        return []


# ---------------------------------------------------------------------------
# Neo4j Full-Text Search Operations (Phase 3.5c)
# ---------------------------------------------------------------------------


def escape_lucene_query(text: str) -> str:
    """
    Escape special characters for Lucene full-text queries.

    Lucene special characters that need escaping:
    + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /

    We wrap the query in quotes for phrase matching, so we mainly need
    to escape quotes and backslashes within the phrase.
    """
    # Escape backslashes first (important order)
    text = text.replace("\\", "\\\\")
    # Escape quotes
    text = text.replace('"', '\\"')
    return text


def search_title_mentions(
    neo4j_driver,
    source_doc_id: str,
    source_title: str,
    min_score: float = 2.0,
    limit: int = 50,
) -> List[Tuple[str, float]]:
    """
    Search for chunks that explicitly mention this document's title.

    Uses Neo4j full-text index (chunk_text_fulltext) with phrase matching.
    Aggregates chunk results to document level using max-score strategy.

    Args:
        neo4j_driver: Neo4j driver instance
        source_doc_id: ID of source document (to exclude from results)
        source_title: Title to search for in chunk text
        min_score: Minimum Lucene score threshold
        limit: Maximum chunks to retrieve before aggregation

    Returns:
        List of (document_id, max_score) tuples, sorted by score descending
    """
    # Escape and prepare the title for phrase search
    escaped_title = escape_lucene_query(source_title)
    # Wrap in quotes for phrase matching
    search_query = f'"{escaped_title}"'

    try:
        with neo4j_driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes($index_name, $search_query, {limit: $fetch_limit})
                YIELD node, score
                WHERE node.document_id <> $source_doc_id AND score > $min_score
                WITH node.document_id AS target_doc_id, max(score) AS max_score
                RETURN target_doc_id, max_score
                ORDER BY max_score DESC
            """,
                index_name=CHUNK_FULLTEXT_INDEX,
                search_query=search_query,
                source_doc_id=source_doc_id,
                min_score=min_score,
                fetch_limit=limit,
            )
            return [(r["target_doc_id"], r["max_score"]) for r in result]
    except Exception as e:
        print(f"  [ERROR] Full-text search failed for '{source_title[:30]}...': {e}")
        return []


# ---------------------------------------------------------------------------
# Core Algorithms
# ---------------------------------------------------------------------------


def aggregate_chunks_to_documents(
    chunk_hits: List[models.ScoredPoint], exclude_doc_id: str
) -> List[Tuple[str, float]]:
    """
    Aggregate chunk-level search results to document-level scores.

    Strategy: Max-Score Wins
    - Group chunks by document_id
    - Take the highest chunk score as the document's score
    - This captures the "best match" semantic from any chunk

    Returns:
        List of (document_id, max_score) tuples, sorted by score descending
    """
    doc_scores: Dict[str, float] = defaultdict(float)

    for hit in chunk_hits:
        doc_id = hit.payload.get("document_id")
        if not doc_id or doc_id == exclude_doc_id:
            continue
        doc_scores[doc_id] = max(doc_scores[doc_id], hit.score)

    return sorted(doc_scores.items(), key=lambda x: -x[1])


def reciprocal_rank_fusion(
    dense_docs: List[Tuple[str, float]],
    sparse_docs: List[Tuple[str, float]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Combine document rankings using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))

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


# ---------------------------------------------------------------------------
# Edge Creation
# ---------------------------------------------------------------------------


def create_related_to_edge(
    neo4j_driver,
    source_id: str,
    target_id: str,
    score: float,
    method: str,
    phase: str,
    dry_run: bool = False,
) -> bool:
    """
    Create a RELATED_TO edge between two documents.

    Uses MERGE for idempotency - re-running the script updates scores
    rather than creating duplicate edges.

    Edge properties:
    - score: Similarity score (cosine for dense, RRF score for fusion)
    - method: 'dense_similarity' or 'rrf_fusion'
    - phase: '3.5a' or '3.5b' (for rollback targeting)
    - created_at: Timestamp
    """
    if dry_run:
        return True

    try:
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (source:Document {id: $source_id})
                MATCH (target:Document {id: $target_id})
                MERGE (source)-[r:RELATED_TO]->(target)
                SET r.score = $score,
                    r.method = $method,
                    r.phase = $phase,
                    r.created_at = datetime()
                RETURN count(r) AS created
            """,
                source_id=source_id,
                target_id=target_id,
                score=score,
                method=method,
                phase=phase,
            )
            return result.single()["created"] > 0
    except Exception as e:
        print(f"  [ERROR] Failed to create edge {source_id[:8]}→{target_id[:8]}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------


def process_document_dense(
    doc: Dict,
    qdrant: QdrantClient,
    neo4j_driver,
    stats: BackfillStats,
    threshold: float,
    max_edges: int,
    chunk_limit: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Process a document using dense-only similarity (Phase 3.5a)."""
    source_id = doc["id"]
    source_title = doc.get("title", "Unknown")[:50]

    # Get dense vector
    dense_vector, _ = get_doc_vectors(qdrant, source_id, include_sparse=False)
    if not dense_vector:
        stats.docs_skipped_no_vector += 1
        if verbose:
            print(f"  [SKIP] No vector: {source_title}")
        return 0

    # Search for similar chunks
    chunk_hits = search_similar_chunks_dense(
        qdrant, dense_vector, source_id, limit=chunk_limit
    )

    if not chunk_hits:
        stats.docs_skipped_no_candidates += 1
        if verbose:
            print(f"  [SKIP] No candidates: {source_title}")
        return 0

    # Aggregate to documents
    doc_candidates = aggregate_chunks_to_documents(chunk_hits, source_id)
    stats.total_candidates_found += len(doc_candidates)

    # Filter and create edges
    edges_created = 0
    for target_id, score in doc_candidates:
        if score < threshold:
            break
        if edges_created >= max_edges:
            break

        if create_related_to_edge(
            neo4j_driver,
            source_id,
            target_id,
            score,
            method="dense_similarity",
            phase="3.5a",
            dry_run=dry_run,
        ):
            edges_created += 1
            stats.record_edge(score)

            if verbose or dry_run:
                target_title = get_document_title(neo4j_driver, target_id)
                target_title = (target_title or "Unknown")[:40]
                mode = "[DRY-RUN]" if dry_run else "[CREATED]"
                print(
                    f"  {mode} {source_title[:30]}... --[{score:.3f}]--> {target_title}..."
                )

    if edges_created > 0:
        stats.docs_with_edges += 1

    return edges_created


def process_document_rrf(
    doc: Dict,
    qdrant: QdrantClient,
    neo4j_driver,
    stats: BackfillStats,
    threshold: float,
    max_edges: int,
    chunk_limit: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Process a document using RRF fusion of dense + sparse (Phase 3.5b)."""
    source_id = doc["id"]
    source_title = doc.get("title", "Unknown")[:50]

    # Get both dense and sparse vectors
    dense_vector, sparse_vector = get_doc_vectors(
        qdrant, source_id, include_sparse=True
    )

    if not dense_vector:
        stats.docs_skipped_no_vector += 1
        if verbose:
            print(f"  [SKIP] No dense vector: {source_title}")
        return 0

    # Search with dense vectors
    dense_hits = search_similar_chunks_dense(
        qdrant, dense_vector, source_id, limit=chunk_limit
    )
    dense_docs = aggregate_chunks_to_documents(dense_hits, source_id)
    stats.dense_candidates += len(dense_docs)

    # Search with sparse vectors (if available)
    sparse_docs = []
    if sparse_vector and hasattr(sparse_vector, "indices") and sparse_vector.indices:
        sparse_hits = search_similar_chunks_sparse(
            qdrant, sparse_vector, source_id, limit=chunk_limit
        )
        sparse_docs = aggregate_chunks_to_documents(sparse_hits, source_id)
        stats.sparse_candidates += len(sparse_docs)

    if not dense_docs and not sparse_docs:
        stats.docs_skipped_no_candidates += 1
        if verbose:
            print(f"  [SKIP] No candidates: {source_title}")
        return 0

    # RRF Fusion
    fused_candidates = reciprocal_rank_fusion(dense_docs, sparse_docs)
    stats.total_candidates_found += len(fused_candidates)

    # Filter and create edges
    edges_created = 0
    for target_id, rrf_score in fused_candidates:
        if rrf_score < threshold:
            break
        if edges_created >= max_edges:
            break

        if create_related_to_edge(
            neo4j_driver,
            source_id,
            target_id,
            rrf_score,
            method="rrf_fusion",
            phase="3.5b",
            dry_run=dry_run,
        ):
            edges_created += 1
            stats.record_edge(rrf_score)

            if verbose or dry_run:
                target_title = get_document_title(neo4j_driver, target_id)
                target_title = (target_title or "Unknown")[:40]
                mode = "[DRY-RUN]" if dry_run else "[CREATED]"
                # Show RRF score with more decimals since they're smaller
                print(
                    f"  {mode} {source_title[:30]}... --[RRF:{rrf_score:.4f}]--> {target_title}..."
                )

    if edges_created > 0:
        stats.docs_with_edges += 1

    return edges_created


def process_document_title_ft(
    doc: Dict,
    neo4j_driver,
    stats: BackfillStats,
    threshold: float,
    max_edges: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Process a document using full-text title mention search (Phase 3.5c).

    Finds documents whose chunks explicitly mention this document's title.
    This captures explicit references that vector similarity might miss.
    """
    source_id = doc["id"]
    source_title = doc.get("title", "")

    if not source_title:
        stats.docs_skipped_no_vector += 1
        if verbose:
            print(f"  [SKIP] No title: {source_id[:12]}...")
        return 0

    # Search for chunks mentioning this title
    doc_candidates = search_title_mentions(
        neo4j_driver,
        source_id,
        source_title,
        min_score=threshold,
    )

    stats.title_ft_candidates += len(doc_candidates)
    stats.total_candidates_found += len(doc_candidates)

    if not doc_candidates:
        stats.docs_skipped_no_candidates += 1
        if verbose:
            print(f"  [SKIP] No mentions found: {source_title[:50]}")
        return 0

    # Create edges for matching documents
    edges_created = 0
    for target_id, score in doc_candidates:
        if score < threshold:
            break
        if edges_created >= max_edges:
            break

        if create_related_to_edge(
            neo4j_driver,
            source_id,
            target_id,
            score,
            method="title_mention",
            phase="3.5c",
            dry_run=dry_run,
        ):
            edges_created += 1
            stats.record_edge(score)

            if verbose or dry_run:
                target_title = get_document_title(neo4j_driver, target_id)
                target_title = (target_title or "Unknown")[:40]
                mode = "[DRY-RUN]" if dry_run else "[CREATED]"
                print(
                    f"  {mode} {source_title[:30]}... --[FT:{score:.2f}]--> {target_title}..."
                )

    if edges_created > 0:
        stats.docs_with_edges += 1

    return edges_created


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.5: Create RELATED_TO edges between similar documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run", action="store_true", help="Preview edges without creating them"
    )
    mode_group.add_argument(
        "--execute", action="store_true", help="Actually create the edges"
    )

    # Method selection
    parser.add_argument(
        "--method",
        choices=["dense", "rrf", "title_ft"],
        default="rrf",
        help="Similarity method: 'dense' (3.5a), 'rrf' (3.5b, default), 'title_ft' (3.5c)",
    )

    # Configuration options
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum score to create edge (default: 0.70 for dense, 0.025 for rrf, 2.0 for title_ft)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=DEFAULT_MAX_EDGES_PER_DOC,
        help=f"Maximum edges per source document (default: {DEFAULT_MAX_EDGES_PER_DOC})",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=DEFAULT_CHUNK_LIMIT,
        help=f"Chunks to fetch per query before aggregation (default: {DEFAULT_CHUNK_LIMIT})",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only first N documents (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each document",
    )

    args = parser.parse_args()
    dry_run = args.dry_run
    method = args.method

    # Set threshold based on method if not specified
    threshold = args.threshold if args.threshold is not None else THRESHOLDS[method]

    # Determine phase and method description
    phase_map = {"dense": "3.5a", "rrf": "3.5b", "title_ft": "3.5c"}
    method_desc = {
        "dense": "dense vectors only",
        "rrf": "dense + sparse RRF fusion",
        "title_ft": "full-text title mentions",
    }
    phase = phase_map[method]

    # Banner
    print()
    print("=" * 60)
    print(f"Phase {phase}: Cross-Document Linking")
    print("=" * 60)
    print(
        f"Mode:           {'DRY-RUN (no writes)' if dry_run else 'EXECUTE (creating edges)'}"
    )
    print(f"Method:         {method} ({method_desc[method]})")
    print(f"Threshold:      {threshold}")
    print(f"Max edges/doc:  {args.max_edges}")
    if method != "title_ft":
        print(f"Chunk limit:    {args.chunk_limit}")
    if args.limit_docs:
        print(f"Doc limit:      {args.limit_docs}")
    print("=" * 60)
    print()

    # Initialize connections
    print("Connecting to databases...")
    qdrant = None
    try:
        neo4j_driver = get_neo4j_driver()
        print("  Neo4j: Connected")
        # Only connect to Qdrant for vector-based methods
        if method != "title_ft":
            qdrant = get_qdrant_client()
            print("  Qdrant: Connected")
        else:
            print("  Qdrant: Skipped (not needed for title_ft)")
    except Exception as e:
        print(f"[FATAL] Failed to connect: {e}")
        sys.exit(1)

    # Get all documents
    print("\nFetching documents from Neo4j...")
    documents = get_all_documents(neo4j_driver)
    print(f"  Found {len(documents)} documents")

    if args.limit_docs:
        documents = documents[: args.limit_docs]
        print(f"  Limited to first {args.limit_docs} documents")

    # Process documents
    print("\nProcessing documents...")
    print("-" * 60)

    stats = BackfillStats()
    stats.method = method

    for i, doc in enumerate(documents):
        stats.docs_processed += 1

        # Progress indicator
        if (i + 1) % 10 == 0 or args.verbose:
            print(
                f"\n[{i + 1}/{len(documents)}] Processing: {doc.get('title', 'Unknown')[:50]}..."
            )

        # Call appropriate processing function based on method
        if method == "title_ft":
            edges = process_document_title_ft(
                doc=doc,
                neo4j_driver=neo4j_driver,
                stats=stats,
                threshold=threshold,
                max_edges=args.max_edges,
                dry_run=dry_run,
                verbose=args.verbose,
            )
        elif method == "dense":
            edges = process_document_dense(
                doc=doc,
                qdrant=qdrant,
                neo4j_driver=neo4j_driver,
                stats=stats,
                threshold=threshold,
                max_edges=args.max_edges,
                chunk_limit=args.chunk_limit,
                dry_run=dry_run,
                verbose=args.verbose,
            )
        else:  # rrf
            edges = process_document_rrf(
                doc=doc,
                qdrant=qdrant,
                neo4j_driver=neo4j_driver,
                stats=stats,
                threshold=threshold,
                max_edges=args.max_edges,
                chunk_limit=args.chunk_limit,
                dry_run=dry_run,
                verbose=args.verbose,
            )

        if edges > 0 and not args.verbose and not dry_run:
            print(f"  Created {edges} edge(s)")

    # Summary
    print(stats.summary(phase))

    # Cleanup
    neo4j_driver.close()

    # Verification hint
    if not dry_run and stats.total_edges_created > 0:
        print("\nVerification query:")
        print(f"  MATCH ()-[r:RELATED_TO {{phase: '{phase}'}}]->() RETURN count(r)")
        print()


if __name__ == "__main__":
    main()
