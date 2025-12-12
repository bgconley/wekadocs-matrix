#!/usr/bin/env python3
"""Collect baseline counts for embedding field migration."""

import json
import os
from datetime import datetime

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def get_neo4j_counts():
    """Get baseline counts from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    counts = {}
    with driver.session() as session:
        # Count Section nodes with embedding_model
        result = session.run(
            "MATCH (n:Section) WHERE n.embedding_model IS NOT NULL RETURN count(n) as count"
        )
        counts["sections_with_embedding_model"] = result.single()["count"]

        # Count Chunk nodes with embedding_model
        result = session.run(
            "MATCH (n:Chunk) WHERE n.embedding_model IS NOT NULL RETURN count(n) as count"
        )
        counts["chunks_with_embedding_model"] = result.single()["count"]

        # Count Section nodes with embedding_version
        result = session.run(
            "MATCH (n:Section) WHERE n.embedding_version IS NOT NULL RETURN count(n) as count"
        )
        counts["sections_with_embedding_version"] = result.single()["count"]

        # Count Chunk nodes with embedding_version
        result = session.run(
            "MATCH (n:Chunk) WHERE n.embedding_version IS NOT NULL RETURN count(n) as count"
        )
        counts["chunks_with_embedding_version"] = result.single()["count"]

        # Total counts
        result = session.run("MATCH (n:Section) RETURN count(n) as count")
        counts["total_sections"] = result.single()["count"]

        result = session.run("MATCH (n:Chunk) RETURN count(n) as count")
        counts["total_chunks"] = result.single()["count"]

        # Sample a few nodes to see field values
        result = session.run(
            """
            MATCH (n:Section)
            WHERE n.embedding_model IS NOT NULL OR n.embedding_version IS NOT NULL
            RETURN n.id as id,
                   n.embedding_model as model,
                   n.embedding_version as version,
                   n.embedding_dimensions as dims,
                   n.embedding_provider as provider
            LIMIT 5
        """
        )
        counts["sample_sections"] = [dict(r) for r in result]

    driver.close()
    return counts


def get_qdrant_counts():
    """Get baseline counts from Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    counts = {}

    # Get collection info
    try:
        collection_info = client.get_collection("weka_sections_v2")
        counts["collection_exists"] = True
        counts["vectors_count"] = collection_info.vectors_count
        counts["points_count"] = collection_info.points_count

        # Check vector configuration - handle both dict and object structures
        vec_config = collection_info.config.params.vectors
        if hasattr(vec_config, "size"):
            counts["vector_size"] = vec_config.size
            counts["distance"] = str(vec_config.distance)
        else:
            counts["vector_size"] = "unknown"
            counts["distance"] = "unknown"

        # Sample points to check payload fields
        points, next_offset = client.scroll(
            collection_name="weka_sections_v2",
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        with_embedding_model = 0
        with_embedding_version = 0
        sample_payloads = []

        for point in points[:100]:  # Check first 100 points
            payload = point.payload
            if payload and "embedding_model" in payload:
                with_embedding_model += 1
            if payload and "embedding_version" in payload:
                with_embedding_version += 1

            # Collect sample for first 3 points
            if len(sample_payloads) < 3 and payload:
                sample_payloads.append(
                    {
                        "id": str(point.id),
                        "has_embedding_model": "embedding_model" in payload,
                        "has_embedding_version": "embedding_version" in payload,
                        "embedding_model": payload.get("embedding_model"),
                        "embedding_version": payload.get("embedding_version"),
                        "embedding_dimensions": payload.get("embedding_dimensions"),
                        "embedding_provider": payload.get("embedding_provider"),
                    }
                )

        counts["sample_with_embedding_model"] = with_embedding_model
        counts["sample_with_embedding_version"] = with_embedding_version
        counts["sample_payloads"] = sample_payloads
        counts["total_checked"] = len(points)

    except Exception as e:
        counts["collection_exists"] = False
        counts["error"] = str(e)

    return counts


def main():
    """Collect and save baseline counts."""
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "neo4j": get_neo4j_counts(),
        "qdrant": get_qdrant_counts(),
    }

    # Save as JSON
    with open("migration/baseline_counts.json", "w") as f:
        json.dump(baseline, f, indent=2)

    # Create markdown report
    report = f"""# Baseline Counts for Embedding Field Migration

**Generated:** {baseline["timestamp"]}

## Neo4j Baseline

| Metric | Count |
|--------|-------|
| Sections with `embedding_model` | {baseline["neo4j"]["sections_with_embedding_model"]} |
| Sections with `embedding_version` | {baseline["neo4j"]["sections_with_embedding_version"]} |
| Chunks with `embedding_model` | {baseline["neo4j"]["chunks_with_embedding_model"]} |
| Chunks with `embedding_version` | {baseline["neo4j"]["chunks_with_embedding_version"]} |
| Total Sections | {baseline["neo4j"]["total_sections"]} |
| Total Chunks | {baseline["neo4j"]["total_chunks"]} |

### Sample Section Nodes
```json
{json.dumps(baseline["neo4j"]["sample_sections"], indent=2)}
```

## Qdrant Baseline

| Metric | Value |
|--------|-------|
| Collection exists | {baseline["qdrant"]["collection_exists"]} |
| Points count | {baseline["qdrant"].get("points_count", "N/A")} |
| Vector size | {baseline["qdrant"].get("vector_size", "N/A")} |
| Distance metric | {baseline["qdrant"].get("distance", "N/A")} |
| Sample with `embedding_model` | {baseline["qdrant"].get("sample_with_embedding_model", 0)}/{baseline["qdrant"].get("total_checked", 100)} |
| Sample with `embedding_version` | {baseline["qdrant"].get("sample_with_embedding_version", 0)}/{baseline["qdrant"].get("total_checked", 100)} |

### Sample Payloads
```json
{json.dumps(baseline["qdrant"].get("sample_payloads", []), indent=2)}
```

## Summary

- **Neo4j**: {baseline["neo4j"]["sections_with_embedding_model"] + baseline["neo4j"]["chunks_with_embedding_model"]} nodes need field migration
- **Qdrant**: {baseline["qdrant"].get("sample_with_embedding_model", 0)}% of sampled points have legacy field
- **Target**: Migrate all `embedding_model` → `embedding_version`
"""

    with open("migration/baseline_counts.md", "w") as f:
        f.write(report)

    print(report)
    print(
        "\n✅ Baseline counts saved to migration/baseline_counts.json and migration/baseline_counts.md"
    )


if __name__ == "__main__":
    main()
