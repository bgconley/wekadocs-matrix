# =============================================================================
# @status: ACTIVE
# @standalone: CLI utility for Qdrant chunk inspection
# =============================================================================
"""
Inspect a chunk by ID from Qdrant.

Usage:
    python -m src.tools.inspect_chunk <chunk_id>
    python -m src.tools.inspect_chunk <chunk_id> --json

Looks up the chunk in Qdrant by point ID and displays the full payload:
text, heading, doc_tag, parent_path_norm, token_count, and metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _get_qdrant_client():
    """Create a Qdrant client from environment config."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("Error: qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def _get_collection_name() -> str:
    """Resolve the Qdrant collection name from config."""
    try:
        from src.shared.config import get_config

        config = get_config()
        return getattr(config.qdrant, "collection_name", "wekadocs")
    except Exception:
        return os.getenv("QDRANT_COLLECTION", "wekadocs")


def inspect_chunk(chunk_id: str, output_json: bool = False) -> None:
    """Look up a chunk by ID and print its payload."""
    client = _get_qdrant_client()
    collection = _get_collection_name()

    try:
        points = client.retrieve(
            collection_name=collection,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        print(f"Error querying Qdrant: {exc}")
        sys.exit(1)

    if not points:
        print(f"Chunk '{chunk_id}' not found in collection '{collection}'")
        sys.exit(1)

    point = points[0]
    payload = point.payload or {}

    if output_json:
        print(json.dumps(payload, indent=2, default=str))
        return

    # Human-readable output
    divider = "=" * 60
    print()
    print(divider)
    print(f"  CHUNK: {chunk_id}")
    print(divider)
    print()
    print(f"  doc_tag:          {payload.get('doc_tag', '—')}")
    print(f"  heading:          {payload.get('heading', '—')}")
    print(f"  parent_path_norm: {payload.get('parent_path_norm', '—')}")
    print(f"  parent_section_id:{payload.get('parent_section_id', '—')}")
    print(f"  level:            {payload.get('level', '—')}")
    print(f"  order:            {payload.get('order', '—')}")
    print(f"  token_count:      {payload.get('token_count', '—')}")
    print(f"  is_combined:      {payload.get('is_combined', False)}")
    print(f"  document_id:      {payload.get('document_id', '—')}")
    print(f"  snapshot_scope:   {payload.get('snapshot_scope', '—')}")
    print()
    print("  TEXT:")
    print("  " + "-" * 56)
    text = payload.get("text", "")
    for line in text.split("\n"):
        print(f"    {line}")
    print()
    print(divider)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a chunk by ID from Qdrant",
        usage="python -m src.tools.inspect_chunk <chunk_id> [--json]",
    )
    parser.add_argument("chunk_id", help="The chunk ID to look up")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    inspect_chunk(args.chunk_id, output_json=args.json)


if __name__ == "__main__":
    main()
