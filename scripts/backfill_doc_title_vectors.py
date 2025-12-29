#!/usr/bin/env python3
"""
Backfill doc_title vectors for existing chunks in Qdrant.

This script:
1. Adds the doc_title named vector to the collection schema (if not exists)
2. Fetches document_id -> title mapping from Neo4j
3. Embeds unique document titles
4. Updates existing Qdrant points with doc_title vectors

Usage:
    python scripts/backfill_doc_title_vectors.py --dry-run
    python scripts/backfill_doc_title_vectors.py --execute
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointVectors,
)


def get_neo4j_doc_titles():
    """Get document_id -> title mapping from Neo4j."""
    from src.shared.connections import get_connection_manager

    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            RETURN d.doc_id AS doc_id, d.title AS title
            """
        )
        return {r["doc_id"]: r["title"] or "" for r in result}


def get_qdrant_points_by_doc():
    """Get all Qdrant points grouped by document_id."""
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "chunks_multi")

    # Scroll through all points
    points_by_doc = defaultdict(list)
    offset = None

    while True:
        results, offset = qdrant.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=["document_id", "doc_id", "doc_title"],
            with_vectors=False,
        )

        for point in results:
            doc_id = point.payload.get("document_id") or point.payload.get("doc_id")
            if doc_id:
                points_by_doc[doc_id].append(point.id)

        if offset is None:
            break

    return dict(points_by_doc), collection


def add_doc_title_vector_to_schema(collection: str, dims: int):
    """Add doc_title named vector to collection if not exists."""
    import requests

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))

    qdrant = QdrantClient(host=host, port=port)

    info = qdrant.get_collection(collection)
    vectors = info.config.params.vectors

    if isinstance(vectors, dict) and "doc_title" in vectors:
        print(f"doc_title vector already exists in {collection}")
        return False

    # Add new named vector via REST API (Qdrant 1.7+)
    # PUT /collections/{collection_name}/vectors/{vector_name}
    print(f"Adding doc_title vector ({dims} dims) to {collection}...")

    url = f"http://{host}:{port}/collections/{collection}/vectors/doc_title"
    payload = {
        "size": dims,
        "distance": "Cosine",
    }

    response = requests.put(url, json=payload)
    if response.status_code == 200:
        print("doc_title vector added successfully")
        return True
    else:
        print(f"Failed to add doc_title vector: {response.status_code} {response.text}")
        raise RuntimeError(f"Failed to add doc_title vector: {response.text}")


def backfill_doc_title_vectors(dry_run: bool = True):
    """Main backfill logic."""
    from src.providers.factory import ProviderFactory

    print(f"\n{'=' * 70}")
    print(f"Doc Title Vector Backfill - {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"{'=' * 70}\n")

    # Get embedding provider
    provider = ProviderFactory.create_embedding_provider()
    dims = provider.dims
    print(f"Embedding provider: {provider.provider_name}, dims={dims}")

    # Get doc_id -> title mapping from Neo4j
    print("\nFetching document titles from Neo4j...")
    doc_titles = get_neo4j_doc_titles()
    print(f"Found {len(doc_titles)} documents")

    # Get points grouped by document
    print("\nFetching Qdrant points...")
    points_by_doc, collection = get_qdrant_points_by_doc()
    total_points = sum(len(pts) for pts in points_by_doc.values())
    print(f"Found {total_points} points across {len(points_by_doc)} documents")

    # Add schema if needed (only in execute mode)
    if not dry_run:
        add_doc_title_vector_to_schema(collection, dims)

    # Embed unique document titles
    unique_titles = list(set(doc_titles.values()))
    unique_titles = [t for t in unique_titles if t]  # Filter empty
    print(f"\nEmbedding {len(unique_titles)} unique document titles...")

    if dry_run:
        print("  [DRY RUN] Would embed titles")
        title_embeddings = {}
    else:
        # Batch embed titles
        embeddings = provider.embed_documents(unique_titles)
        title_embeddings = dict(zip(unique_titles, embeddings))
        print(f"  Embedded {len(title_embeddings)} titles")

    # Update points
    print("\nUpdating Qdrant points...")
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    updated = 0
    skipped = 0
    errors = 0

    for doc_id, point_ids in points_by_doc.items():
        title = doc_titles.get(doc_id, "")
        if not title:
            skipped += len(point_ids)
            continue

        if dry_run:
            print(
                f"  [DRY RUN] Would update {len(point_ids)} points for doc: {title[:50]}..."
            )
            updated += len(point_ids)
            continue

        embedding = title_embeddings.get(title)
        if not embedding:
            skipped += len(point_ids)
            continue

        # Update points with doc_title vector
        try:
            qdrant.update_vectors(
                collection_name=collection,
                points=[
                    PointVectors(id=pid, vector={"doc_title": embedding})
                    for pid in point_ids
                ],
            )
            updated += len(point_ids)
        except Exception as e:
            print(f"  ERROR updating points for {doc_id}: {e}")
            errors += len(point_ids)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total points: {total_points}")
    print(f"Updated: {updated}")
    print(f"Skipped (no title): {skipped}")
    print(f"Errors: {errors}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --execute to apply.")
    else:
        print("\nBackfill complete!")

    return {
        "total": total_points,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill doc_title vectors")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the backfill",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Error: Must specify --dry-run or --execute")
        sys.exit(1)

    if args.dry_run and args.execute:
        print("Error: Cannot specify both --dry-run and --execute")
        sys.exit(1)

    backfill_doc_title_vectors(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
