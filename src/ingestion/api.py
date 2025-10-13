from typing import Any, Dict, Optional

from src.ingestion.build_graph import ingest_document as build_graph_ingest


def ingest_document(
    source_uri: str,
    content: str,
    fmt: str = "markdown",
    *,
    embedding_model: Optional[str] = None,
    embedding_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous façade used by integration tests.
    Delegates to the full graph-building ingestion pipeline while allowing
    optional overrides for embedding model metadata.
    """
    return build_graph_ingest(
        source_uri,
        content,
        format=fmt,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )
