#!/usr/bin/env python3
import os

from qdrant_client import QdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    print("Getting collection info...")
    collection_info = client.get_collection("weka_sections_v2")
    print(f"✓ Collection exists with {collection_info.points_count} points")

    print("Getting vector config...")
    vec_config = collection_info.config.params.vectors
    print(f"✓ Vector size: {vec_config.size}, Distance: {vec_config.distance}")

    print("Scrolling points...")
    points, next_offset = client.scroll(
        collection_name="weka_sections_v2",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    print(f"✓ Got {len(points)} points")

    for i, point in enumerate(points[:3]):
        print(f"  Point {i+1}: {point.id}")
        if point.payload:
            print(f"    Has embedding_version: {'embedding_version' in point.payload}")
            print(f"    Has embedding_model: {'embedding_model' in point.payload}")

except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e)}")
    import traceback

    traceback.print_exc()
