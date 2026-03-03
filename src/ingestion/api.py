# =============================================================================
# @status: DEAD
# @reason: This is a test facade (docstring: "Synchronous facade used by
#          integration tests") that delegates to build_graph.ingest_document.
#          Not called from the production worker path.
#          No longer phantom-loaded after Phase B __init__.py cleanup.
# @loaded-via: (none — removed from ingestion/__init__.py in Phase B)
# @superseded-by: src/ingestion/atomic.py:AtomicIngestionCoordinator
# =============================================================================
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
