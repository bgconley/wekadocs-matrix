#!/usr/bin/env python3
"""
Diagnose missing chunks between Neo4j and Qdrant.

Identifies which chunks exist in Neo4j but not in Qdrant,
and analyzes their characteristics to find the root cause.
"""

import json
import os
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from qdrant_client import QdrantClient


def get_neo4j_chunks(driver):
    """Get all chunk IDs and metadata from Neo4j."""
    query = """
    MATCH (c:Chunk)
    OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(s:Section)
    OPTIONAL MATCH (s)<-[:HAS_SECTION]-(d:Document)
    RETURN c.id AS chunk_id,
           c.heading AS heading,
           c.text AS text,
           c.token_count AS token_count,
           c.is_microdoc AS is_microdoc,
           c.microdoc_type AS microdoc_type,
           s.id AS section_id,
           s.title AS section_title,
           d.doc_id AS doc_id,
           d.title AS doc_title
    """
    with driver.session() as session:
        result = session.run(query)
        chunks = []
        for record in result:
            chunks.append(
                {
                    "chunk_id": record["chunk_id"],
                    "heading": record["heading"],
                    "text": (
                        record["text"][:200] if record["text"] else None
                    ),  # First 200 chars
                    "text_length": len(record["text"]) if record["text"] else 0,
                    "token_count": record["token_count"],
                    "is_microdoc": record["is_microdoc"],
                    "microdoc_type": record["microdoc_type"],
                    "section_id": record["section_id"],
                    "section_title": record["section_title"],
                    "doc_id": record["doc_id"],
                    "doc_title": record["doc_title"],
                }
            )
        return chunks


def get_qdrant_point_ids(client, collection_name="chunks_multi_bge_m3"):
    """Get all point IDs from Qdrant."""
    point_ids = set()
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points, next_offset = result

        for point in points:
            point_ids.add(point.id)

        if next_offset is None:
            break
        offset = next_offset

    return point_ids


def analyze_missing_chunks(missing_chunks):
    """Analyze characteristics of missing chunks."""
    analysis = {
        "total_missing": len(missing_chunks),
        "by_microdoc_status": defaultdict(int),
        "by_microdoc_type": defaultdict(int),
        "by_document": defaultdict(int),
        "by_text_length_bucket": defaultdict(int),
        "by_token_count_bucket": defaultdict(int),
        "sample_headings": [],
        "samples": [],
    }

    for chunk in missing_chunks:
        # Microdoc analysis
        is_microdoc = chunk.get("is_microdoc")
        analysis["by_microdoc_status"][str(is_microdoc)] += 1

        microdoc_type = chunk.get("microdoc_type")
        if microdoc_type:
            analysis["by_microdoc_type"][microdoc_type] += 1

        # Document distribution
        doc_title = chunk.get("doc_title") or "Unknown"
        analysis["by_document"][doc_title] += 1

        # Text length buckets
        text_len = chunk.get("text_length", 0)
        if text_len == 0:
            bucket = "0 (empty)"
        elif text_len < 100:
            bucket = "1-99 (tiny)"
        elif text_len < 500:
            bucket = "100-499 (small)"
        elif text_len < 1000:
            bucket = "500-999 (medium)"
        else:
            bucket = "1000+ (large)"
        analysis["by_text_length_bucket"][bucket] += 1

        # Token count buckets
        tokens = chunk.get("token_count") or 0
        if tokens == 0:
            token_bucket = "0 (none)"
        elif tokens < 50:
            token_bucket = "1-49"
        elif tokens < 200:
            token_bucket = "50-199"
        elif tokens < 500:
            token_bucket = "200-499"
        else:
            token_bucket = "500+"
        analysis["by_token_count_bucket"][token_bucket] += 1

        # Collect sample headings
        if len(analysis["sample_headings"]) < 20:
            heading = chunk.get("heading") or chunk.get("section_title") or "No heading"
            analysis["sample_headings"].append(heading)

        # Collect full samples for first 5
        if len(analysis["samples"]) < 5:
            analysis["samples"].append(chunk)

    return analysis


def main():
    # Connection parameters
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    print("=" * 60)
    print("MISSING CHUNK DIAGNOSTIC REPORT")
    print("=" * 60)

    # Connect to Neo4j
    print("\n[1] Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Connect to Qdrant
    print("[2] Connecting to Qdrant...")
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Get data
    print("[3] Fetching Neo4j chunks...")
    neo4j_chunks = get_neo4j_chunks(driver)
    neo4j_chunk_ids = {c["chunk_id"] for c in neo4j_chunks}
    print(f"    Found {len(neo4j_chunks)} chunks in Neo4j")

    print("[4] Fetching Qdrant point IDs...")
    qdrant_ids = get_qdrant_point_ids(qdrant)
    print(f"    Found {len(qdrant_ids)} points in Qdrant")

    # Find differences
    print("\n[5] Comparing...")
    missing_from_qdrant = neo4j_chunk_ids - qdrant_ids
    extra_in_qdrant = qdrant_ids - neo4j_chunk_ids

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Neo4j chunks:        {len(neo4j_chunk_ids)}")
    print(f"Qdrant points:       {len(qdrant_ids)}")
    print(f"Missing from Qdrant: {len(missing_from_qdrant)}")
    print(f"Extra in Qdrant:     {len(extra_in_qdrant)}")

    if missing_from_qdrant:
        # Get full chunk data for missing ones
        missing_chunks = [
            c for c in neo4j_chunks if c["chunk_id"] in missing_from_qdrant
        ]

        print(f"\n{'=' * 60}")
        print("MISSING CHUNK ANALYSIS")
        print(f"{'=' * 60}")

        analysis = analyze_missing_chunks(missing_chunks)

        print("\n--- By Microdoc Status ---")
        for status, count in sorted(analysis["by_microdoc_status"].items()):
            pct = count / analysis["total_missing"] * 100
            print(f"  {status}: {count} ({pct:.1f}%)")

        if analysis["by_microdoc_type"]:
            print("\n--- By Microdoc Type ---")
            for mtype, count in sorted(analysis["by_microdoc_type"].items()):
                print(f"  {mtype}: {count}")

        print("\n--- By Text Length ---")
        for bucket, count in sorted(analysis["by_text_length_bucket"].items()):
            pct = count / analysis["total_missing"] * 100
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        print("\n--- By Token Count ---")
        for bucket, count in sorted(analysis["by_token_count_bucket"].items()):
            pct = count / analysis["total_missing"] * 100
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        print("\n--- By Document ---")
        for doc, count in sorted(analysis["by_document"].items(), key=lambda x: -x[1]):
            print(f"  {doc}: {count}")

        print("\n--- Sample Headings (first 20) ---")
        for i, heading in enumerate(analysis["sample_headings"], 1):
            print(f"  {i}. {heading[:80]}...")

        print("\n--- Full Sample Chunks (first 5) ---")
        for i, sample in enumerate(analysis["samples"], 1):
            print(f"\n  [{i}] Chunk ID: {sample['chunk_id']}")
            print(f"      Heading: {sample.get('heading') or 'None'}")
            print(f"      Section: {sample.get('section_title') or 'None'}")
            print(f"      Document: {sample.get('doc_title') or 'None'}")
            print(f"      Text length: {sample.get('text_length', 0)} chars")
            print(f"      Token count: {sample.get('token_count') or 'None'}")
            print(f"      Is microdoc: {sample.get('is_microdoc')}")
            print(f"      Microdoc type: {sample.get('microdoc_type') or 'None'}")
            if sample.get("text"):
                print(f"      Text preview: {sample['text'][:150]}...")

        # Output missing IDs to file for further analysis
        output_file = "/tmp/missing_chunk_ids.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "missing_count": len(missing_from_qdrant),
                    "missing_ids": list(missing_from_qdrant),
                    "analysis": {
                        "by_microdoc_status": dict(analysis["by_microdoc_status"]),
                        "by_microdoc_type": dict(analysis["by_microdoc_type"]),
                        "by_document": dict(analysis["by_document"]),
                        "by_text_length_bucket": dict(
                            analysis["by_text_length_bucket"]
                        ),
                    },
                    "samples": analysis["samples"],
                },
                f,
                indent=2,
            )
        print(f"\n[!] Full results written to: {output_file}")

    if extra_in_qdrant:
        print(f"\n{'=' * 60}")
        print("EXTRA IN QDRANT (orphan vectors)")
        print(f"{'=' * 60}")
        print(f"  Count: {len(extra_in_qdrant)}")
        print(f"  Sample IDs: {list(extra_in_qdrant)[:5]}")

    driver.close()
    print("\n[Done]")


if __name__ == "__main__":
    main()
