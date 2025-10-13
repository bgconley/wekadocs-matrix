from typing import Dict

from src.ingestion.build_graph import ingest_document as build_graph_ingest


def ingest_document(source_uri: str, content: str, fmt: str = "markdown") -> Dict:
    """
    Synchronous façade used by integration tests.
    Delegates to the full graph-building ingestion pipeline.
    """
    return build_graph_ingest(source_uri, content, fmt)
