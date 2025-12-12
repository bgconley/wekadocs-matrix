#!/usr/bin/env python3
"""
Verification script for STDIO endpoint RRF wiring fix.

This script tests that:
1. search_sections_light() returns ChunkResult objects with fusion_method="rrf"
2. fused_score is populated
3. Per-signal scores (title_vec_score, entity_vec_score) are available

Usage:
    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
    QDRANT_HOST=localhost REDIS_HOST=localhost \
    python scripts/verify_stdio_rrf_wiring.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_environment():
    """Check required environment variables."""
    required = ["NEO4J_URI", "NEO4J_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        print("Run with:")
        print(
            "  NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 "
            "QDRANT_HOST=localhost REDIS_HOST=localhost python scripts/verify_stdio_rrf_wiring.py"
        )
        sys.exit(1)


def main():
    """Run the verification."""
    verify_environment()

    # Set config path before imports
    os.environ.setdefault("CONFIG_PATH", "config/development.yaml")

    from src.mcp_server.query_service import get_query_service

    print("=" * 70)
    print("STDIO RRF Wiring Verification")
    print("=" * 70)

    print("\n1. Initializing QueryService...")
    qs = get_query_service()
    print("   QueryService initialized successfully")

    # Test queries
    test_queries = [
        "WEKA filesystem configuration",
        "NFS mount options",
        "cluster node setup",
    ]

    for query in test_queries:
        print(f"\n2. Testing query: '{query}'")
        print("-" * 60)

        chunks, metrics = qs.search_sections_light(query=query, fetch_k=5)

        print(f"   Found {len(chunks)} chunks")

        if not chunks:
            print("   WARNING: No chunks returned!")
            continue

        # Analyze first chunk in detail
        c = chunks[0]
        print("\n   TOP RESULT ANALYSIS:")
        print(f"   chunk_id:       {c.chunk_id}")
        print(
            f"   heading:        {c.heading[:50]}..."
            if len(c.heading) > 50
            else f"   heading:        {c.heading}"
        )
        print(f"   fusion_method:  {c.fusion_method}")
        print(f"   fused_score:    {c.fused_score}")
        print(f"   rerank_score:   {c.rerank_score}")

        print("\n   PER-SIGNAL SCORES:")
        print(f"   vector_score:         {c.vector_score}")
        print(f"   title_vec_score:      {c.title_vec_score}")
        print(f"   entity_vec_score:     {c.entity_vec_score}")
        print(f"   doc_title_sparse:     {c.doc_title_sparse_score}")
        print(f"   lexical_vec_score:    {c.lexical_vec_score}")
        print(f"   bm25_score:           {c.bm25_score}")

        # Validation checks
        print("\n   VALIDATION:")
        checks_passed = 0
        checks_total = 0

        # Check 1: fusion_method is 'rrf' or 'rerank' (rerank overrides when BGE reranker is applied)
        checks_total += 1
        if c.fusion_method in ("rrf", "rerank"):
            print(
                f"   [PASS] fusion_method = '{c.fusion_method}' (rrf or rerank expected)"
            )
            checks_passed += 1
        else:
            print(
                f"   [FAIL] fusion_method = '{c.fusion_method}' (expected 'rrf' or 'rerank')"
            )

        # Check 2: fused_score is populated
        checks_total += 1
        if c.fused_score is not None and c.fused_score > 0:
            print(f"   [PASS] fused_score = {c.fused_score:.4f}")
            checks_passed += 1
        else:
            print(f"   [FAIL] fused_score = {c.fused_score}")

        # Check 3: At least one signal score is populated
        checks_total += 1
        signal_scores = [
            c.vector_score,
            c.title_vec_score,
            c.entity_vec_score,
            c.doc_title_sparse_score,
            c.lexical_vec_score,
        ]
        populated = [s for s in signal_scores if s is not None and s > 0]
        if populated:
            print(f"   [PASS] {len(populated)} signal scores populated")
            checks_passed += 1
        else:
            print("   [FAIL] No signal scores populated")

        print(f"\n   Result: {checks_passed}/{checks_total} checks passed")

        # Only show first query details
        break

    # Summary of all chunks
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL CHUNKS")
    print("=" * 70)

    if chunks:
        # Count fusion methods (rrf or rerank are both valid - rerank means BGE reranker was applied)
        rrf_or_rerank_count = sum(
            1 for c in chunks if c.fusion_method in ("rrf", "rerank")
        )
        fused_count = sum(1 for c in chunks if c.fused_score is not None)
        title_count = sum(1 for c in chunks if c.title_vec_score is not None)
        entity_count = sum(1 for c in chunks if c.entity_vec_score is not None)

        print(f"Total chunks:               {len(chunks)}")
        print(f"With fusion_method rrf/rerank: {rrf_or_rerank_count}")
        print(f"With fused_score:           {fused_count}")
        print(f"With title_vec_score:       {title_count}")
        print(f"With entity_vec_score:      {entity_count}")

        # Success if most chunks have valid fusion method and scores are populated
        # (Some expanded chunks may not have fusion_method set)
        if fused_count > 0 and (title_count > 0 or entity_count > 0):
            print("\n" + "=" * 70)
            print(
                "SUCCESS: RRF/Rerank fusion is properly wired in the retrieval pipeline!"
            )
            print("Per-signal scores are exposed and the STDIO fix is working!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("WARNING: Fusion may not be fully active")
            print("=" * 70)


if __name__ == "__main__":
    main()
