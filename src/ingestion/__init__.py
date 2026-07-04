# =============================================================================
# @status: ACTIVE
# @reason: Package init for ingestion. Eager import of api.py (test facade)
#          was removed (Phase B cleanup) to prevent early build_graph.py loading.
#          Tests needing ingest_document should import from src.ingestion.api directly.
# =============================================================================
