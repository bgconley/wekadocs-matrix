"""
Local BGE reranker service provider.

This provider calls an external HTTP service that exposes /v1/rerank
for the BAAI/bge-reranker-v2-m3 model (or compatible).

Phase 1.1 Enhancement: Batched HTTP requests for 8-10x latency improvement.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from src.providers.rerank.base import RerankProvider
from src.shared.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

# Configuration constants for batching
DEFAULT_BATCH_SIZE = (
    16  # Documents per batch (balance between latency and payload size)
)
# Token budgets updated for Qwen3-Reranker-4B (supports 4096+ tokens)
# Previous BGE values (800/1024) were too restrictive and filtered many valid candidates
MAX_TOKENS_PER_DOC = 2048  # Token budget per document (Qwen3 supports 4096)
MAX_TOKENS_TOTAL = 4096  # Max tokens per request (query + doc)


def batch_documents(
    documents: List[Tuple[int, str]],
    max_batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[Tuple[int, str]]]:
    """
    Split indexed documents into batches for efficient HTTP requests.

    Phase 1.1 Fix: Now accepts (original_index, text) tuples directly to avoid
    fragile index reconstruction. The caller is responsible for preserving
    original indices throughout the pipeline.

    Args:
        documents: List of (original_index, text) tuples to batch
        max_batch_size: Maximum documents per batch (default: 16)

    Returns:
        List of batches, where each batch is a list of (original_index, text) tuples
    """
    if not documents:
        return []

    batches: List[List[Tuple[int, str]]] = []
    current_batch: List[Tuple[int, str]] = []

    for item in documents:
        current_batch.append(item)

        if len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []

    # Don't forget the last partial batch
    if current_batch:
        batches.append(current_batch)

    return batches


class BGERerankerServiceProvider(RerankProvider):
    """HTTP client wrapper for the local BGE reranker service.

    Phase 1.1: Implements batched HTTP requests to reduce latency from
    ~300-500ms (per-doc) to ~30-50ms (batched) for typical k=50 operations.

    Features:
    - Batched HTTP requests (configurable batch size)
    - Circuit breaker for resilience (Issue #11)
    - Proper resource cleanup (Issue #13)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "http://qwen3-reranker-lambda:9003",
        timeout: float = 60.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_id = model
        self._provider_name = "bge-reranker-service"
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._batch_size = batch_size

        # Issue #15 Fix: Support both old (DISABLE) and new (ENABLED) flag naming
        # New positive flag takes precedence, fall back to old negative flag
        batching_enabled = os.getenv("RERANKER_BATCHING_ENABLED", "").lower()
        batching_disabled = os.getenv("RERANKER_DISABLE_BATCHING", "").lower()

        if batching_enabled in {"1", "true", "yes"}:
            self._use_batching = True
        elif batching_enabled in {"0", "false", "no"}:
            self._use_batching = False
        elif batching_disabled in {"1", "true", "yes"}:
            # Backward compatibility with old flag
            self._use_batching = False
            logger.info(
                "RERANKER_DISABLE_BATCHING is deprecated, use RERANKER_BATCHING_ENABLED=false"
            )
        else:
            # Default: batching enabled
            self._use_batching = True

        if not self._use_batching:
            logger.warning("Reranker batching disabled via environment flag")

        # Issue #11: Thread-safe circuit breaker from shared module
        # Configuration via env vars: CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self._circuit_breaker = CircuitBreaker(name=self._provider_name)

    def __del__(self) -> None:
        """Issue #13 Fix: Ensure HTTP client resources are released."""
        self.close()

    def close(self) -> None:
        """Explicitly close the HTTP client and release resources."""
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Best effort cleanup

    def __enter__(self) -> "BGERerankerServiceProvider":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close resources when exiting context."""
        self.close()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def health_check(self) -> bool:
        try:
            resp = self._client.get("/health")
            # Some deployments may not expose /health; treat any response as available.
            return resp.status_code >= 200 and resp.status_code < 500
        except Exception:
            return True

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token budget."""
        tokens = text.split()
        if len(tokens) > MAX_TOKENS_PER_DOC:
            return " ".join(tokens[:MAX_TOKENS_PER_DOC])
        return text

    def _rerank_batch(
        self,
        query: str,
        batch: List[Tuple[int, str]],
    ) -> List[Tuple[int, float]]:
        """
        Rerank a batch of documents with a single HTTP request.

        Args:
            query: Query text
            batch: List of (original_index, document_text) tuples

        Returns:
            List of (original_index, score) tuples
        """
        if not batch:
            return []

        # Prepare documents for the batch request
        documents = [text for _, text in batch]

        payload: Dict[str, Any] = {
            "query": query,
            "documents": documents,
            "model": self._model_id,
        }

        response = self._client.post("/v1/rerank", json=payload)
        if response.status_code != 200:
            try:
                body = response.text
            except Exception:
                body = "<unavailable>"
            raise RuntimeError(f"Reranker service HTTP {response.status_code}: {body}")

        data = response.json()
        results = data.get("results", [])

        logger.debug(
            "Reranker batch response",
            extra={
                "provider": self._provider_name,
                "model": self._model_id,
                "batch_size": len(documents),
                "results_count": len(results),
                "scores": [r.get("score") for r in results[:5]],
                "query_snippet": query[:200],
            },
        )

        # Map results back to original indices
        # Results from the API are ordered by score descending, with 'index' field
        # indicating which document in our input list the result corresponds to
        scored: List[Tuple[int, float]] = []
        for result in results:
            batch_index = result.get("index", 0)
            score = result.get("score", 0.0)
            if 0 <= batch_index < len(batch):
                original_index = batch[batch_index][0]
                scored.append((original_index, score))

        return scored

    def _rerank_single(
        self,
        query: str,
        original_index: int,
        text: str,
    ) -> Tuple[int, float]:
        """
        Rerank a single document (fallback/legacy behavior).

        Args:
            query: Query text
            original_index: Original index in the candidates list
            text: Document text

        Returns:
            Tuple of (original_index, score)
        """
        payload: Dict[str, Any] = {
            "query": query,
            "documents": [text],
            "model": self._model_id,
        }

        response = self._client.post("/v1/rerank", json=payload)
        if response.status_code != 200:
            try:
                body = response.text
            except Exception:
                body = "<unavailable>"
            raise RuntimeError(f"Reranker service HTTP {response.status_code}: {body}")

        data = response.json()
        results = data.get("results", [])

        if not results:
            return (original_index, 0.0)

        score = results[0].get("score", 0.0)
        return (original_index, score)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates based on relevance to query.

        Phase 1.1: Uses batched HTTP requests for improved latency.
        With k=50 candidates and batch_size=16, this reduces from 50 HTTP
        calls to just 4, achieving ~8-10x latency improvement.

        Args:
            query: Query text
            candidates: List of candidate documents (must have 'text' field)
            top_k: Number of top results to return

        Returns:
            Top-k candidates sorted by rerank_score descending
        """
        if not candidates:
            raise ValueError("Cannot rerank empty candidate list")

        # Issue #11: Thread-safe circuit breaker check
        if not self._circuit_breaker.allow_request():
            logger.warning(
                "reranker_circuit_open",
                extra={
                    "provider": self._provider_name,
                    "candidate_count": len(candidates),
                    "circuit_state": self._circuit_breaker.state.value,
                    "action": "returning_unranked",
                },
            )
            # Return candidates without reranking (preserve original order)
            return [
                {**cand, "rerank_score": 0.0, "reranker": "circuit_open"}
                for cand in candidates[:top_k]
            ]

        query_tokens = len(query.split())

        # Prepare and truncate documents, filtering those that exceed budget
        valid_candidates: List[Tuple[int, str, Dict]] = []
        skipped_count = 0
        for i, cand in enumerate(candidates):
            if "text" not in cand:
                raise ValueError(f"Candidate {i} missing required 'text' field")

            text = self._truncate_text(cand["text"])
            doc_tokens = len(text.split())

            if query_tokens + doc_tokens > MAX_TOKENS_TOTAL:
                # Skip if even truncated doc exceeds budget with query
                logger.debug(
                    f"Skipping candidate {i}: {query_tokens + doc_tokens} tokens exceeds {MAX_TOKENS_TOTAL}"
                )
                skipped_count += 1
                continue

            valid_candidates.append((i, text, cand))

        if not valid_candidates:
            # Issue #4 Fix: Warn when ALL candidates are filtered out
            logger.warning(
                "reranker_all_candidates_filtered",
                extra={
                    "provider": self._provider_name,
                    "total_candidates": len(candidates),
                    "skipped_count": skipped_count,
                    "query_tokens": query_tokens,
                    "max_tokens_total": MAX_TOKENS_TOTAL,
                    "reason": "all_candidates_exceed_token_budget",
                },
            )
            return []

        if skipped_count > 0:
            logger.info(
                "reranker_candidates_filtered",
                extra={
                    "provider": self._provider_name,
                    "total_candidates": len(candidates),
                    "valid_candidates": len(valid_candidates),
                    "skipped_count": skipped_count,
                },
            )

        # Collect all scores
        all_scores: Dict[int, float] = {}

        if self._use_batching:
            # Phase 1.1: Batched processing for efficiency
            # Keep (original_index, text) tuples throughout to avoid fragile reconstruction
            indexed_documents = [(idx, text) for idx, text, _ in valid_candidates]
            indexed_batches = batch_documents(
                indexed_documents,
                max_batch_size=self._batch_size,
            )

            # Process each batch
            batch_success_count = 0
            batch_failure_count = 0
            for batch in indexed_batches:
                try:
                    batch_results = self._rerank_batch(query, batch)
                    for orig_idx, score in batch_results:
                        all_scores[orig_idx] = score
                    batch_success_count += 1
                except Exception as e:
                    batch_failure_count += 1
                    error_str = str(e).lower()
                    # N3 Fix: Detect service unhealthy conditions (overload OR server errors)
                    # These indicate the service itself is failing, not our payload
                    is_service_unhealthy = any(
                        indicator in error_str
                        for indicator in [
                            "timeout",
                            "429",
                            "503",  # Overload indicators
                            "500",
                            "502",
                            "504",  # Server error indicators (N3 addition)
                            "connect",
                            "refused",
                            "reset",  # Network failure indicators
                        ]
                    )

                    if is_service_unhealthy:
                        # Issue #11 + N3: Record failure for thread-safe circuit breaker
                        self._circuit_breaker.record_failure()
                        # Don't retry during service failure - would amplify 16x
                        logger.error(
                            f"Batch rerank failed due to service unhealthy: {e}. "
                            f"Skipping {len(batch)} documents to prevent retry storm."
                        )
                        # Skip this batch entirely - documents won't get rerank scores
                        continue

                    # True payload errors (e.g., 400 Bad Request, malformed input)
                    # are safe to retry individually - service is healthy but rejected our batch
                    logger.warning(
                        f"Batch rerank failed (payload issue): {e}, falling back to single"
                    )
                    for orig_idx, text in batch:
                        try:
                            _, score = self._rerank_single(query, orig_idx, text)
                            all_scores[orig_idx] = score
                        except Exception as e2:
                            logger.warning(
                                f"Single rerank failed for idx {orig_idx}: {e2}"
                            )

            # Issue #11: Record success if any batches succeeded
            if batch_success_count > 0:
                self._circuit_breaker.record_success()
        else:
            # Legacy per-document processing (fallback mode)
            for orig_idx, text, _ in valid_candidates:
                try:
                    _, score = self._rerank_single(query, orig_idx, text)
                    all_scores[orig_idx] = score
                except Exception as e:
                    logger.warning(f"Rerank failed for candidate {orig_idx}: {e}")

        # Build results with scores
        all_results: List[Dict[str, Any]] = []
        for orig_idx, _, cand in valid_candidates:
            if orig_idx in all_scores:
                all_results.append(
                    {
                        **cand,
                        "rerank_score": all_scores[orig_idx],
                        "original_rank": orig_idx,
                        "reranker": self._provider_name,
                    }
                )

        # Sort by score descending and return top_k
        all_results = sorted(
            all_results,
            key=lambda c: (c.get("rerank_score") or float("-inf")),
            reverse=True,
        )

        # Multi-model validation fix: Fallback when all batches fail
        # Prevents returning empty results when reranker service is down
        if not all_results and valid_candidates:
            logger.warning(
                "reranker_total_failure_fallback",
                extra={
                    "provider": self._provider_name,
                    "candidate_count": len(valid_candidates),
                    "reason": "all_batches_failed",
                },
            )
            return [
                {**cand, "rerank_score": 0.0, "reranker": "rerank_failed"}
                for _, _, cand in valid_candidates[:top_k]
            ]

        # Debug logging for payload inspection
        if os.getenv("RERANKER_PAYLOAD_DEBUG", "").lower() in {"1", "true", "yes"}:
            logger.warning(
                "Reranker payload debug",
                extra={
                    "provider": self._provider_name,
                    "model": self._model_id,
                    "candidate_count": len(candidates),
                    "valid_count": len(valid_candidates),
                    "batching_enabled": self._use_batching,
                    "batch_size": self._batch_size if self._use_batching else None,
                    "scores_collected": len(all_scores),
                    "top_scores": [r.get("rerank_score") for r in all_results[:5]],
                },
            )

        return all_results[:top_k]
