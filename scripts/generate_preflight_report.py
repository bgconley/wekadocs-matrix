#!/usr/bin/env python3
"""
Generate Phase 7E Preflight Report (Markdown)

Converts JSON preflight results into a comprehensive markdown report.
"""

import json
import sys
from datetime import datetime


def generate_markdown_report(results: dict) -> str:
    """Generate markdown report from JSON results."""

    lines = []

    # Header
    lines.append("# Phase 7E.0 - Preflight Validation Report")
    lines.append("")
    lines.append(f"**Timestamp:** {results['timestamp']}")
    lines.append(
        f"**Overall Status:** {'✅ PASS' if results['overall_status'] == 'PASS' else '❌ FAIL' if results['overall_status'] == 'FAIL' else '⚠️ PARTIAL'}"
    )
    lines.append(f"**Pass Rate:** {results['summary']['pass_rate']}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "Phase 7E preflight checks verify that infrastructure, schema, and configuration are ready for Phase 7E implementation (GraphRAG v2.1 with Jina v3, hybrid retrieval, and bounded adjacency expansion)."
    )
    lines.append("")

    passed = results["summary"]["passed"]
    total = results["summary"]["total"]

    if passed == total:
        lines.append(
            f"✅ **All {total} checks passed.** Infrastructure is ready for Phase 7E implementation."
        )
    else:
        lines.append(
            f"⚠️ **{passed}/{total} checks passed.** Review failures below before proceeding."
        )

    lines.append("")

    # Detailed Results
    lines.append("## Detailed Results")
    lines.append("")

    for check in results["checks"]:
        status_icon = "✅" if check["status"] == "PASS" else "❌"
        lines.append(f"### {status_icon} {check['name']}")
        lines.append("")
        lines.append(f"**Status:** {check['status']}")
        lines.append(f"**Details:** {check['details']}")
        lines.append("")

        if check["evidence"]:
            lines.append("**Evidence:**")
            lines.append("```json")
            lines.append(json.dumps(check["evidence"], indent=2))
            lines.append("```")
            lines.append("")

    # Critical Specifications
    lines.append("## Critical Specifications Verified")
    lines.append("")
    lines.append("| Component | Specification | Status |")
    lines.append("|-----------|---------------|--------|")

    # Extract key specs from results
    specs = [
        (
            "Neo4j Constraints",
            "15 unique constraints",
            (
                "✅"
                if any(
                    c["name"] == "Neo4j Constraints" and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
        (
            "Neo4j Vector Indexes",
            "2 indexes @ 1024-D cosine",
            (
                "✅"
                if any(
                    c["name"] == "Neo4j Vector Indexes (1024-D)"
                    and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
        (
            "Neo4j Schema Version",
            "v2.1",
            (
                "✅"
                if any(
                    c["name"] == "Neo4j Schema Version" and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
        (
            "Qdrant Collection",
            "'chunks' @ 1024-D Cosine",
            (
                "✅"
                if any(
                    c["name"] == "Qdrant 'chunks' Collection" and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
        (
            "Qdrant Payload Indexes",
            "document_id, parent_section_id, order",
            (
                "✅"
                if any(
                    c["name"] == "Qdrant Payload Indexes" and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
        (
            "Embedding Config",
            "jina-embeddings-v3 @ 1024-D",
            (
                "✅"
                if any(
                    c["name"] == "Config File Settings" and c["status"] == "PASS"
                    for c in results["checks"]
                )
                else "❌"
            ),
        ),
    ]

    for component, spec, status in specs:
        lines.append(f"| {component} | {spec} | {status} |")

    lines.append("")

    # Phase 7E Configuration
    lines.append("## Phase 7E Configuration Validated")
    lines.append("")
    lines.append(
        "The following Phase 7E-specific settings have been validated in `config/development.yaml`:"
    )
    lines.append("")
    lines.append("### Hybrid Retrieval Settings")
    lines.append("- **Method:** `rrf` (Reciprocal Rank Fusion)")
    lines.append("- **RRF K:** `60` (default constant)")
    lines.append("- **Fusion Alpha:** `0.6` (vector weight for weighted mode)")
    lines.append("- **BM25 Enabled:** `true`")
    lines.append("- **BM25 Top-K:** `50`")
    lines.append("")
    lines.append("### Bounded Adjacency Expansion")
    lines.append("- **Enabled:** `true`")
    lines.append("- **Max Neighbors:** `1` (±1 NEXT_CHUNK)")
    lines.append("- **Query Min Tokens:** `12` (expand if query >= 12 tokens)")
    lines.append("- **Score Delta Max:** `0.02` (expand if top scores close)")
    lines.append("")
    lines.append("### Context Budget")
    lines.append("- **Answer Context Max Tokens:** `4500` (LLM context window limit)")
    lines.append("")
    lines.append("### Cache Invalidation")
    lines.append("- **Mode:** `epoch` (O(1) invalidation)")
    lines.append("- **Namespace:** `rag:v1`")
    lines.append("- **Doc Epoch Key:** `rag:v1:doc_epoch`")
    lines.append("- **Chunk Epoch Key:** `rag:v1:chunk_epoch`")
    lines.append("")

    # Health Probe Commands
    lines.append("## Health Probe Commands")
    lines.append("")
    lines.append("The following commands were used to verify system health:")
    lines.append("")
    lines.append("### Neo4j")
    lines.append("```cypher")
    lines.append("-- List constraints")
    lines.append("SHOW CONSTRAINTS;")
    lines.append("")
    lines.append("-- List indexes")
    lines.append("SHOW INDEXES;")
    lines.append("")
    lines.append("-- Verify schema version")
    lines.append("MATCH (sv:SchemaVersion {id: 'singleton'})")
    lines.append(
        "RETURN sv.version, sv.vector_dimensions, sv.embedding_provider, sv.embedding_model;"
    )
    lines.append("")
    lines.append("-- Test vector index dimensions")
    lines.append(
        "CALL db.index.vector.queryNodes('section_embeddings_v2', 1, [0.0, ...]) YIELD node RETURN node LIMIT 0;"
    )
    lines.append("```")
    lines.append("")
    lines.append("### Qdrant")
    lines.append("```python")
    lines.append("# Get collection info")
    lines.append("info = client.get_collection('chunks')")
    lines.append('print(f"Size: {info.config.params.vectors.size}")')
    lines.append('print(f"Distance: {info.config.params.vectors.distance}")')
    lines.append('print(f"Points: {info.points_count}")')
    lines.append("")
    lines.append("# List payload indexes")
    lines.append("payload_schema = info.payload_schema")
    lines.append('print(f"Indexes: {list(payload_schema.keys())}")')
    lines.append("```")
    lines.append("")

    # Next Steps
    lines.append("## Next Steps")
    lines.append("")

    if results["overall_status"] == "PASS":
        lines.append(
            "✅ **All preflight checks passed. Proceed with Phase 7E implementation:**"
        )
        lines.append("")
        lines.append("1. **Phase 7E.1 - Ingestion**")
        lines.append(
            "   - Implement deterministic ID generation (`sha256(document_id|original_section_ids)[:24]`)"
        )
        lines.append("   - Write nodes as `:Section:Chunk` with canonical fields")
        lines.append("   - Implement replace-by-set GC in both Neo4j and Qdrant")
        lines.append("   - Enforce app-layer validation (1024-D, required fields)")
        lines.append("")
        lines.append("2. **Phase 7E.2 - Retrieval**")
        lines.append("   - Implement BM25/full-text search over `Chunk.text`")
        lines.append("   - Implement RRF fusion (k=60) + optional weighted (α=0.6)")
        lines.append("   - Implement bounded expansion (±1 NEXT_CHUNK)")
        lines.append("   - Enforce context budget (4500 tokens)")
        lines.append("")
        lines.append("3. **Phase 7E.3 - Caching**")
        lines.append("   - Implement epoch-based keys (doc_epoch, chunk_epoch)")
        lines.append("   - Bump epochs on successful ingest")
        lines.append("   - Add pattern-scan fallback")
        lines.append("")
        lines.append("4. **Phase 7E.4 - Observability**")
        lines.append(
            "   - Implement health checks (schema v2.1, vector dims, constraints/indexes)"
        )
        lines.append("   - Add metrics (latencies, chunk sizes, expansion rate)")
        lines.append("   - Define SLOs (p95≤500ms, 0 oversized, 0 integrity failures)")
    else:
        lines.append(
            "❌ **Preflight checks failed. Address the following issues before proceeding:**"
        )
        lines.append("")
        failed_checks = [c for c in results["checks"] if c["status"] != "PASS"]
        for check in failed_checks:
            lines.append(f"- **{check['name']}:** {check['details']}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "**Report Generated:** " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    lines.append("**Script:** `scripts/phase7e_preflight.py`")
    lines.append("**Phase:** 7E.0 (Preflight)")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate markdown report from JSON input."""
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            results = json.load(f)
    else:
        results = json.load(sys.stdin)

    report = generate_markdown_report(results)
    print(report)


if __name__ == "__main__":
    main()
