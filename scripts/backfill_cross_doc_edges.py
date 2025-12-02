#!/usr/bin/env python3
"""
Phase 3.5a: Dense Similarity Cross-Document Linking (MVP)

Creates RELATED_TO edges between semantically similar documents based on
doc_title vector similarity in Qdrant.

Architecture (per plan docs/plans/phase-3.5-cross-doc-linking.md):
    Stage 1: Query Qdrant for chunks with similar doc_title vectors (limit=100)
    Stage 2: Aggregate chunks to documents (max-score wins)
    Stage 3: Filter by threshold and create RELATED_TO edges

Usage:
    # Preview what edges would be created (no writes)
    python scripts/backfill_cross_doc_edges.py --dry-run

    # Create edges with default settings (threshold=0.70, max 5 edges per doc)
    python scripts/backfill_cross_doc_edges.py --execute

    # Custom threshold and limits
    python scripts/backfill_cross_doc_edges.py --execute --threshold 0.75 --max-edges 3

    # Test with limited documents first
    python scripts/backfill_cross_doc_edges.py --dry-run --limit-docs 5

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

# Defaults (can be overridden via CLI)
DEFAULT_SCORE_THRESHOLD = 0.70  # Minimum similarity to create edge
DEFAULT_CHUNK_LIMIT = 100  # Chunks to fetch per query (before aggregation)
DEFAULT_MAX_EDGES_PER_DOC = 5  # Limit edges per source document
DEFAULT_DISCOVERY_THRESHOLD = 0.50  # Relaxed threshold for initial discovery


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
    total_candidates_found: int = 0
    score_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "0.90+": 0,
            "0.80-0.90": 0,
            "0.70-0.80": 0,
            "0.60-0.70": 0,
            "<0.60": 0,
        }
    )
    start_time: float = field(default_factory=time.time)

    def record_edge(self, score: float) -> None:
        """Record an edge creation with its score."""
        self.total_edges_created += 1
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

    def summary(self) -> str:
        """Generate a summary report."""
        elapsed = time.time() - self.start_time
        lines = [
            "",
            "=" * 60,
            "Phase 3.5a Backfill Complete",
            "=" * 60,
            f"Documents processed:        {self.docs_processed}",
            f"Documents with new edges:   {self.docs_with_edges}",
            f"Documents skipped (no vec): {self.docs_skipped_no_vector}",
            f"Documents skipped (no cand):{self.docs_skipped_no_candidates}",
            f"Total edges created:        {self.total_edges_created}",
            f"Total candidates evaluated: {self.total_candidates_found}",
            "",
            "Score distribution of created edges:",
        ]
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


def get_doc_title_vector(qdrant: QdrantClient, doc_id: str) -> Optional[List[float]]:
    """
    Get the doc_title vector for a document.

    Since all chunks from the same document have identical doc_title vectors,
    we only need to fetch one chunk to get the document's vector.
    """
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
            with_vectors=[DENSE_VECTOR_NAME],
            with_payload=False,
        )

        if points and points[0].vector:
            vec = points[0].vector
            # Handle both dict and direct vector formats
            if isinstance(vec, dict):
                return vec.get(DENSE_VECTOR_NAME)
            return vec
        return None

    except Exception as e:
        print(f"  [ERROR] Failed to get vector for doc {doc_id[:12]}...: {e}")
        return None


def search_similar_chunks(
    qdrant: QdrantClient,
    query_vector: List[float],
    exclude_doc_id: str,
    limit: int = DEFAULT_CHUNK_LIMIT,
    score_threshold: float = DEFAULT_DISCOVERY_THRESHOLD,
) -> List[models.ScoredPoint]:
    """
    Search for chunks with similar doc_title vectors.

    Returns up to `limit` chunks, excluding those from the source document.
    Uses a relaxed score_threshold since we filter more strictly after aggregation.
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
            score_threshold=score_threshold,
            with_payload=["document_id"],  # Only need doc ID for aggregation
        )
    except Exception as e:
        print(f"  [ERROR] Search failed for doc {exclude_doc_id[:12]}...: {e}")
        return []


# ---------------------------------------------------------------------------
# Core Algorithm: Chunk-to-Document Aggregation
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

    Why max-score instead of mean?
    - A document might have 1 highly relevant chunk and 5 irrelevant ones
    - Mean would dilute the signal from the relevant chunk
    - Max preserves the strongest match signal

    Args:
        chunk_hits: Raw Qdrant search results (chunks)
        exclude_doc_id: Source document ID to exclude from results

    Returns:
        List of (document_id, max_score) tuples, sorted by score descending
    """
    doc_scores: Dict[str, float] = defaultdict(float)

    for hit in chunk_hits:
        doc_id = hit.payload.get("document_id")
        if not doc_id or doc_id == exclude_doc_id:
            continue
        # Max-score aggregation: best chunk represents document
        doc_scores[doc_id] = max(doc_scores[doc_id], hit.score)

    return sorted(doc_scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Edge Creation
# ---------------------------------------------------------------------------


def create_related_to_edge(
    neo4j_driver, source_id: str, target_id: str, score: float, dry_run: bool = False
) -> bool:
    """
    Create a RELATED_TO edge between two documents.

    Uses MERGE for idempotency - re-running the script updates scores
    rather than creating duplicate edges.

    Edge properties:
    - score: Similarity score from dense vector search
    - method: 'dense_similarity' (Phase 3.5a)
    - phase: '3.5a' (for rollback targeting)
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
                    r.method = 'dense_similarity',
                    r.phase = '3.5a',
                    r.created_at = datetime()
                RETURN count(r) AS created
            """,
                source_id=source_id,
                target_id=target_id,
                score=score,
            )
            return result.single()["created"] > 0
    except Exception as e:
        print(f"  [ERROR] Failed to create edge {source_id[:8]}→{target_id[:8]}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------


def process_document(
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
    """
    Process a single document: find similar docs and create edges.

    Returns the number of edges created for this document.
    """
    source_id = doc["id"]
    source_title = doc.get("title", "Unknown")[:50]

    # Stage 1: Get source document's doc_title vector
    query_vector = get_doc_title_vector(qdrant, source_id)
    if not query_vector:
        stats.docs_skipped_no_vector += 1
        if verbose:
            print(f"  [SKIP] No vector: {source_title}")
        return 0

    # Stage 2: Search for similar chunks
    chunk_hits = search_similar_chunks(
        qdrant, query_vector, source_id, limit=chunk_limit
    )

    if not chunk_hits:
        stats.docs_skipped_no_candidates += 1
        if verbose:
            print(f"  [SKIP] No candidates: {source_title}")
        return 0

    # Stage 3: Aggregate chunks to documents
    doc_candidates = aggregate_chunks_to_documents(chunk_hits, source_id)
    stats.total_candidates_found += len(doc_candidates)

    # Stage 4: Filter by threshold and create edges
    edges_created = 0
    for target_id, score in doc_candidates:
        if score < threshold:
            break  # Sorted by score, so no more will pass threshold
        if edges_created >= max_edges:
            break

        if create_related_to_edge(neo4j_driver, source_id, target_id, score, dry_run):
            edges_created += 1
            stats.record_edge(score)

            if verbose or dry_run:
                # Get target title for display
                target_title = get_document_title(neo4j_driver, target_id)
                target_title = (target_title or "Unknown")[:40]
                mode = "[DRY-RUN]" if dry_run else "[CREATED]"
                print(
                    f"  {mode} {source_title[:30]}... --[{score:.3f}]--> {target_title}..."
                )

    if edges_created > 0:
        stats.docs_with_edges += 1

    return edges_created


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.5a: Create RELATED_TO edges between similar documents",
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

    # Configuration options
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Minimum similarity score to create edge (default: {DEFAULT_SCORE_THRESHOLD})",
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

    # Banner
    print()
    print("=" * 60)
    print("Phase 3.5a: Dense Similarity Cross-Document Linking")
    print("=" * 60)
    print(
        f"Mode:           {'DRY-RUN (no writes)' if dry_run else 'EXECUTE (creating edges)'}"
    )
    print(f"Threshold:      {args.threshold}")
    print(f"Max edges/doc:  {args.max_edges}")
    print(f"Chunk limit:    {args.chunk_limit}")
    if args.limit_docs:
        print(f"Doc limit:      {args.limit_docs}")
    print("=" * 60)
    print()

    # Initialize connections
    print("Connecting to databases...")
    try:
        neo4j_driver = get_neo4j_driver()
        qdrant = get_qdrant_client()
        print("  Neo4j: Connected")
        print("  Qdrant: Connected")
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

    for i, doc in enumerate(documents):
        stats.docs_processed += 1

        # Progress indicator
        if (i + 1) % 10 == 0 or args.verbose:
            print(
                f"\n[{i + 1}/{len(documents)}] Processing: {doc.get('title', 'Unknown')[:50]}..."
            )

        edges = process_document(
            doc=doc,
            qdrant=qdrant,
            neo4j_driver=neo4j_driver,
            stats=stats,
            threshold=args.threshold,
            max_edges=args.max_edges,
            chunk_limit=args.chunk_limit,
            dry_run=dry_run,
            verbose=args.verbose,
        )

        if edges > 0 and not args.verbose and not dry_run:
            print(f"  Created {edges} edge(s)")

    # Summary
    print(stats.summary())

    # Cleanup
    neo4j_driver.close()

    # Verification hint
    if not dry_run and stats.total_edges_created > 0:
        print("\nVerification query:")
        print("  MATCH ()-[r:RELATED_TO {phase: '3.5a'}]->() RETURN count(r)")
        print()


if __name__ == "__main__":
    main()
