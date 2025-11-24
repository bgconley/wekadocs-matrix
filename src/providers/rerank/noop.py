"""
No-op reranker implementation.
Phase 7C: Passthrough reranker that preserves original ordering.

Used when:
- Reranking is disabled (RERANK_PROVIDER=none)
- Fallback when Jina/BGE unavailable
- Testing scenarios
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class NoopReranker:
    """
    No-op reranker that preserves original candidate ordering.

    This is a valid fallback strategy when sophisticated reranking
    is unavailable or disabled. It ensures the system continues to
    function without reranking rather than failing.
    """

    def __init__(self):
        """Initialize the no-op reranker."""
        self._model_id = "noop"
        self._provider_name = "noop"

        logger.info("NoopReranker initialized (passthrough mode)")

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Return candidates unchanged (no reranking).

        Args:
            query: Query text (unused)
            candidates: List of candidates
            top_k: Number of results to return

        Returns:
            Top-k candidates from original list with passthrough scores

        Raises:
            ValueError: If candidates is empty
        """
        if not candidates:
            raise ValueError("Cannot rerank empty candidate list")

        # Pre-Phase 7 (G1): Metrics instrumentation
        import time

        from src.shared.observability.metrics import (
            rerank_latency_ms,
            rerank_request_total,
        )

        start_time = time.time()

        # Simply return top_k candidates with passthrough metadata
        result = []
        for idx, candidate in enumerate(candidates[:top_k]):
            result.append(
                {
                    **candidate,
                    "rerank_score": candidate.get(
                        "score", 1.0 - (idx * 0.01)
                    ),  # Preserve or slight decay
                    "original_rank": idx + 1,
                    "reranker": "noop",
                }
            )

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        rerank_request_total.labels(model_id="noop", status="success").inc()
        rerank_latency_ms.labels(model_id="noop").observe(latency_ms)

        logger.debug(
            f"NoopReranker: Returned top {len(result)} candidates (no reranking applied)"
        )

        return result

    def health_check(self) -> bool:
        """Always healthy (no external dependency)."""
        return True
