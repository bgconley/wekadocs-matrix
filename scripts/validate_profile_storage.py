#!/usr/bin/env python3
"""Validate that the active embedding profile matches the configured Qdrant collection.

Fails fast when:
- Collection is missing.
- Dense dims mismatch the profile.
- Required sparse/colbert slots are missing.

Intended for CI rollout safety and manual checks.
"""

import sys

from src.shared.config import get_config, get_embedding_settings
from src.shared.connections import get_connection_manager
from src.shared.qdrant_schema import validate_qdrant_schema


def main() -> int:
    config, _ = get_config()
    settings = get_embedding_settings()

    qdrant_cfg = getattr(config.search.vector, "qdrant", None)
    if not qdrant_cfg:
        print("No Qdrant config found; skipping validation")
        return 0

    collection = getattr(qdrant_cfg, "collection_name", None)
    if not collection:
        print("Qdrant collection name not configured; skipping validation")
        return 0

    manager = get_connection_manager()
    try:
        qdrant_client = manager.get_qdrant_client()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to get Qdrant client: {exc}")
        return 1

    require_sparse = getattr(qdrant_cfg, "enable_sparse", False)
    require_colbert = getattr(qdrant_cfg, "enable_colbert", False)

    try:
        validate_qdrant_schema(
            qdrant_client,
            collection,
            settings,
            require_sparse=require_sparse,
            require_colbert=require_colbert,
        )
        print(
            f"✅ Qdrant collection '{collection}' matches profile '{settings.profile}' "
            f"(dims={settings.dims}, sparse={require_sparse}, colbert={require_colbert})"
        )
        return 0
    except Exception as exc:
        print(
            f"❌ Qdrant schema validation failed for collection '{collection}': {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
