"""
Jina AI rerank provider implementation.
Phase 7C: Remote API provider for jina-reranker models (v2/v3).

Features:
- Cross-attention based reranking for high precision
- Rate limiting (500 RPM, 1M TPM - standard API tier)
- Exponential backoff with jitter
- Circuit breaker for API failures
- Comprehensive error handling
"""

import logging
import os
import random
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class JinaRerankProvider:
    """
    Jina AI rerank provider (supports jina-reranker-v2/v3 models).

    Applies cross-attention scoring to refine candidate ordering
    after initial ANN retrieval.

    Key features:
    - Rate limiting: 500 RPM, 1M TPM (Jina API limits - standard tier)
    - Exponential backoff on errors
    - Circuit breaker for resilience
    """

    API_URL = "https://api.jina.ai/v1/rerank"

    # Rate limit constants (Jina rerank API limits - standard tier, same as embeddings)
    MAX_REQUESTS_PER_MINUTE = 500
    MAX_TOKENS_PER_MINUTE = 1_000_000

    def __init__(
        self,
        model: str = "jina-reranker-v3",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize Jina rerank provider.

        Args:
            model: Model identifier
            api_key: Jina API key (from env if not provided)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is missing
        """
        self._model_id = model
        self._provider_name = "jina-ai"
        self._timeout = timeout

        # Get API key
        self._api_key = api_key or os.getenv("JINA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "JINA_API_KEY required for jina-ai reranker. "
                "Set JINA_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        # Initialize circuit breaker (shared with embedding provider)
        from src.providers.embeddings.jina import CircuitBreaker, RateLimiter

        self._circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=300)

        # Initialize rate limiter for rerank API (500 RPM, 1M TPM - standard tier)
        self._rate_limiter = RateLimiter(
            max_requests_per_minute=self.MAX_REQUESTS_PER_MINUTE,
            max_tokens_per_minute=self.MAX_TOKENS_PER_MINUTE,
        )

        logger.info(f"JinaRerankProvider initialized: model={model}")

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
        Rerank candidates using Jina reranker.

        Args:
            query: Query text
            candidates: List of candidate documents
            top_k: Number of top results to return

        Returns:
            List of reranked candidates with scores

        Raises:
            ValueError: If candidates is empty or improperly formatted
            RuntimeError: If API call fails
        """
        if not candidates:
            raise ValueError("Cannot rerank empty candidate list")

        # Validate candidate format
        for i, cand in enumerate(candidates):
            if "text" not in cand:
                raise ValueError(
                    f"Candidate {i} missing required 'text' field. "
                    f"Each candidate must have 'text' key."
                )
            if "id" not in cand:
                raise ValueError(f"Candidate {i} missing required 'id' field")

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            raise RuntimeError(
                "Circuit breaker OPEN - Jina reranker unavailable. "
                "Falling back to noop reranker."
            )

        # Metrics
        from src.shared.observability.metrics import (
            rerank_error_total,
            rerank_latency_ms,
            rerank_request_total,
        )

        start_time = time.time()

        try:
            # Call Jina rerank API
            reranked = self._call_api(query=query, candidates=candidates, top_k=top_k)

            # Record success
            self._circuit_breaker.record_success()
            latency_ms = (time.time() - start_time) * 1000
            rerank_request_total.labels(model_id=self._model_id, status="success").inc()
            rerank_latency_ms.labels(model_id=self._model_id).observe(latency_ms)

            logger.debug(
                f"Jina reranking complete: {len(reranked)}/{len(candidates)} candidates, "
                f"{latency_ms:.2f}ms"
            )

            return reranked

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            rerank_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            rerank_request_total.labels(model_id=self._model_id, status="error").inc()
            logger.error(f"Failed to rerank with Jina: {e}")
            raise RuntimeError(f"Jina reranking failed: {e}")

    def _call_api(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Call Jina rerank API with rate limiting and exponential backoff.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of top results

        Returns:
            Reranked candidates

        Raises:
            RuntimeError: If all retries fail
        """
        max_retries = 3
        base_delay = 1.0

        # Prepare documents for API
        documents = [cand["text"] for cand in candidates]

        # Estimate tokens for rate limiting
        # Query + all documents (conservative: 4 chars per token)
        query_chars = len(query)
        doc_chars = sum(len(doc) for doc in documents)
        estimated_tokens = (query_chars + doc_chars) // 4

        # Rate limiting - wait if needed
        self._rate_limiter.wait_if_needed(estimated_tokens)

        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    self.API_URL,
                    json={
                        "model": self._model_id,
                        "query": query,
                        "documents": documents,
                        "top_n": min(top_k, len(documents)),
                        "return_documents": False,  # We already have documents
                    },
                )

                # Handle HTTP errors
                if response.status_code == 429:
                    # Rate limit
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Jina rerank API rate limit (429), retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()

                # Parse response
                data = response.json()
                results = data.get("results", [])

                # Map back to original candidates with scores
                reranked = []
                for result in results:
                    index = result.get("index")
                    if index is None:
                        index = result.get("document_index")
                    try:
                        index = int(index)
                    except (TypeError, ValueError):
                        index = None
                    if index is None or index >= len(candidates):
                        logger.warning(
                            "Jina rerank response missing valid index: %s", result
                        )
                        continue

                    score = result.get("relevance_score")
                    if score is None:
                        score = result.get("score")
                    if score is None:
                        logger.warning(
                            "Jina rerank response missing score field: %s", result
                        )
                        continue

                    # Get original candidate
                    original = candidates[index].copy()

                    # Add reranking metadata
                    original["rerank_score"] = score
                    original["original_rank"] = index + 1
                    original["reranker"] = self._model_id

                    reranked.append(original)

                return reranked

            except httpx.HTTPStatusError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Jina rerank API error {e.response.status_code}, "
                        f"retrying in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Jina rerank API HTTP error: {e.response.status_code}"
                    )

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Jina rerank API error: {e}, retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Jina rerank API call failed: {e}")

        raise RuntimeError(f"Jina rerank API failed after {max_retries} retries")

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
