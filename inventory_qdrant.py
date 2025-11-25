import json
import sys

from qdrant_client import QdrantClient

# --- CONFIGURATION ---
# Ensure this matches your current running v1.7 container
QDRANT_URL = "http://localhost:6333"
OUTPUT_FILE = "qdrant_schema_inventory.json"


def safe_dump(pydantic_obj):
    """
    Helper to convert Pydantic models to dicts across versions.
    Handles both V1 (dict) and V2 (model_dump) Pydantic styles.
    """
    if pydantic_obj is None:
        return None

    try:
        # Try Pydantic V2 style first
        if hasattr(pydantic_obj, "model_dump"):
            return pydantic_obj.model_dump(exclude_none=True)
        # Fallback to Pydantic V1 style
        if hasattr(pydantic_obj, "dict"):
            return pydantic_obj.dict(exclude_none=True)
    except TypeError:
        # Fall through to string representation if direct dumping fails
        pass

    # As a last resort, coerce to string to keep JSON serialization safe
    try:
        return str(pydantic_obj)
    except Exception:
        return None


def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(QDRANT_URL)
        # Simple connectivity check
        client.get_collections()
    except Exception as e:
        print(f"Could not connect to Qdrant: {e}")
        print("Make sure your v1.7 Qdrant instance is running on this host/port.")
        sys.exit(1)

    schema_inventory = {}

    # 1. Get all collections
    try:
        collections_response = client.get_collections()
    except Exception as e:
        print(f"Error fetching collections list: {e}")
        sys.exit(1)

    count = len(collections_response.collections)
    print(f"Found {count} collection(s).")

    for col in collections_response.collections:
        name = col.name
        print(f"  - Introspecting '{name}'...")

        try:
            # 2. Get detailed info (Config + Status)
            info = client.get_collection(name)
            config = info.config
            params = config.params

            # 3. Extract critical components

            # Vectors (Dense) - single vs named
            vectors_config = safe_dump(params.vectors)

            # Sparse Vectors (may be missing on v1.7)
            sparse_vectors_config = None
            if hasattr(params, "sparse_vectors") and params.sparse_vectors:
                sparse_vectors_config = safe_dump(params.sparse_vectors)

            # Payload Indexes / Schema
            payload_schema = {}
            if hasattr(info, "payload_schema") and info.payload_schema:
                payload_schema = {
                    k: safe_dump(v) for k, v in info.payload_schema.items()
                }

            # 4. Build export object
            schema_inventory[name] = {
                "vectors_config": vectors_config,
                "sparse_vectors_config": sparse_vectors_config,
                "shard_number": params.shard_number,
                "replication_factor": params.replication_factor,
                "on_disk_payload": params.on_disk_payload,
                "payload_schema_indexes": payload_schema,
                "hnsw_config": safe_dump(config.hnsw_config),
            }
        except Exception as e:
            print(f"    Error inspecting collection '{name}': {e}")

    # 5. Save to File
    with open(OUTPUT_FILE, "w") as f:
        json.dump(schema_inventory, f, indent=2)

    print(f"\nInventory complete. Saved to: {OUTPUT_FILE}")

    # Quick feedback on sparse config
    print("\nQuick sparse-vector analysis:")
    for name, data in schema_inventory.items():
        print(f"Collection: {name}")
        sparse = data.get("sparse_vectors_config")
        if sparse:
            if isinstance(sparse, dict):
                keys = list(sparse.keys())
            else:
                keys = [type(sparse).__name__]
            print(f"  [OK] Sparse vectors configured: {keys}")
        else:
            print("  [NONE] No sparse vectors configured.")


if __name__ == "__main__":
    main()
