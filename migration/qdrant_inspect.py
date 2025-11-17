#!/usr/bin/env python3
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Get collection
collection_info = client.get_collection("weka_sections_v2")
print("Collection exists: True")
print(f"Points count: {collection_info.points_count}")
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance: {collection_info.config.params.vectors.distance}")

# Sample points
points, next_offset = client.scroll(
    collection_name="weka_sections_v2", limit=5, with_payload=True, with_vectors=False
)

print(f"\nFound {len(points)} points")
if points:
    print("\nFirst point payload keys:")
    print(points[0].payload.keys() if points[0].payload else "No payload")

    # Check for embedding fields
    for point in points[:3]:
        print(f"\nPoint {point.id}:")
        print(f"  has embedding_model: {'embedding_model' in point.payload}")
        print(f"  has embedding_version: {'embedding_version' in point.payload}")
        if "embedding_version" in point.payload:
            print(f"  embedding_version: {point.payload['embedding_version']}")
