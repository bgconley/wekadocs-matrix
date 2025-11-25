import os

from qdrant_client import QdrantClient

from src.ingestion.build_graph import GraphBuilder
from src.shared import config as config_module
from src.shared.config import _slugify_identifier, get_config, get_embedding_settings


def main():
    # Ensure we use the bge_m3 embedding profile
    os.environ.setdefault("EMBEDDINGS_PROFILE", "bge_m3")

    # Enable schema management on GraphBuilder init
    os.environ["MANAGE_QDRANT_SCHEMA_ON_INIT"] = "true"

    # Reset cached config so EMBEDDINGS_PROFILE/env is honored
    config_module._config = None
    config = get_config()
    # Also resolve embedding settings so we can derive namespace suffix
    settings = get_embedding_settings()

    # Enable sparse + ColBERT for this run
    qcfg = config.search.vector.qdrant
    # Use setattr in case fields are missing in some environments
    setattr(qcfg, "enable_sparse", True)
    setattr(qcfg, "enable_colbert", True)

    # Ensure collection_name ends with the expected namespace suffix
    expected_suffix = _slugify_identifier(settings.version or settings.profile or "")
    cname = str(qcfg.collection_name)
    if expected_suffix and not cname.endswith(expected_suffix):
        qcfg.collection_name = f"{cname}_{expected_suffix}"

    # Connect to the upgraded local Qdrant
    client = QdrantClient(url="http://127.0.0.1:6333")

    # This will log initialization and, with MANAGE_QDRANT_SCHEMA_ON_INIT=true,
    # will create/reconcile the Qdrant collection schema for bge_m3.
    GraphBuilder(driver=None, config=config, qdrant_client=client)


if __name__ == "__main__":
    main()
