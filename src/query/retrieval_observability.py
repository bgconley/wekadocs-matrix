# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade wrapper)
# =============================================================================
"""
Retrieval observability: stage snapshots and diagnostic helpers.

Extracted from hybrid_retrieval.py to isolate observability helpers.
Event payload builders remain inline in retrieve() to avoid restructuring
the orchestration flow.
"""

import logging
from typing import List

from src.query.retrieval_types import ChunkResult
from src.shared.observability import get_logger

logger = get_logger(__name__)


def log_stage_snapshot(stage: str, chunks: List[ChunkResult], limit: int = 5) -> None:
    """Log a lightweight snapshot of the top N candidates at a given stage."""
    underlying = getattr(logger, "_logger", None) or getattr(logger, "logger", None)
    if not underlying or not underlying.isEnabledFor(logging.DEBUG):
        return
    sample = [getattr(c, "chunk_id", None) for c in chunks[:limit]]
    logger.debug(
        "Ranking stage snapshot",
        extra={"stage": stage, "count": len(chunks), "sample": sample},
    )
