#!/usr/bin/env python3
"""Initialize Neo4j schema with config-driven vector indexes"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.config import get_config  # noqa: E402
from src.shared.connections import get_connection_manager  # noqa: E402
from src.shared.schema import create_schema  # noqa: E402


def main():
    config = get_config()

    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()

    print("Creating Neo4j schema with config-driven vector indexes...")
    print(f"  Embedding dimensions: {config.embedding.dims}")
    print(f"  Similarity function: {config.embedding.similarity}")

    result = create_schema(driver, config)

    if result["success"]:
        print("✓ Schema created successfully")
        print(f"  Constraints: {result.get('constraints_created', 0)}")
        print(f"  Indexes: {result.get('indexes_created', 0)}")
        print(f"  Vector indexes: {result.get('vector_indexes_created', 0)}")
    else:
        print("✗ Schema creation failed")
        print(f"  Error: {result.get('error')}")
        sys.exit(1)

    driver.close()


if __name__ == "__main__":
    main()
