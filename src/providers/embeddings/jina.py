"""
Jina AI embedding provider implementation.
Phase 7C: Remote API provider for jina-embeddings-v3 @ 1024-D.

Features:
- Task-specific embeddings (retrieval.passage vs retrieval.query)
- Adaptive batching (count + size limits)
- Rate limiting (500 RPM, 1M TPM)
- Auto-split on 400 errors
- Exponential backoff with jitter
- Circuit breaker for API failures
- Comprehensive error handling
- Phase 7C Hotfix: Accurate token counting with XLM-RoBERTa tokenizer
"""

import logging
import os
import time
from collections import deque
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for Jina API calls.
    Enforces 500 RPM and 1,000,000 TPM limits using sliding window.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 500,
        max_tokens_per_minute: int = 1_000_000,
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Max API calls per 60 seconds
            max_tokens_per_minute: Max tokens per 60 seconds
        """
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        self.request_times = deque()
        self.token_counts = deque()

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Wait if making request would exceed rate limits.

        Uses sliding window: tracks requests in last 60 seconds.
        Proactively throttles before hitting API limits.

        Args:
            estimated_tokens: Estimated tokens for this request
        """
        now = time.time()
        minute_ago = now - 60

        # Remove entries older than 60 seconds
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
            self.token_counts.popleft()

        # Check if we would exceed limits
        requests_in_window = len(self.request_times)
        tokens_in_window = sum(self.token_counts)

        wait_time = 0

        # Check request limit
        if requests_in_window >= self.max_rpm:
            oldest = self.request_times[0]
            wait_time = max(wait_time, 60 - (now - oldest))

        # Check token limit
        if tokens_in_window + estimated_tokens > self.max_tpm:
            oldest = self.request_times[0]
            wait_time = max(wait_time, 60 - (now - oldest))

        if wait_time > 0:
            logger.warning(
                f"Rate limit approaching, throttling for {wait_time:.1f}s "
                f"(requests={requests_in_window}/{self.max_rpm}, "
                f"tokens={tokens_in_window}/{self.max_tpm})"
            )
            time.sleep(wait_time)

        # Record this request
        self.request_times.append(time.time())
        self.token_counts.append(estimated_tokens)


class CircuitBreaker:
    """Simple circuit breaker for API resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening
            timeout: Seconds to wait before trying half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful call."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker OPEN after {self.failures} consecutive failures"
            )

    def can_attempt(self) -> bool:
        """Check if call can be attempted."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering HALF-OPEN state")
                return True
            return False

        # half-open: allow one attempt
        return True


class JinaEmbeddingProvider:
    """
    Jina AI embedding provider using jina-embeddings-v3.

    Supports task-specific embeddings:
    - retrieval.passage: For document/section embedding during ingestion
    - retrieval.query: For query embedding during search

    Key features:
    - Adaptive batching: Respects both count (32) and size (50KB) limits
    - Rate limiting: Enforces 500 RPM and 1M TPM
    - Auto-split: Divides batch in half on 400 errors
    - Circuit breaker: Opens on repeated failures
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    # Adaptive batching constants (from official Jina AI documentation)
    MAX_TEXTS_PER_BATCH = 2_048  # Official Jina limit: 2,048 texts per request
    MAX_TOKENS_PER_TEXT = 8_192  # Official Jina limit: 8,192 tokens per text

    # Empirical payload size limit (conservative, based on observed 400 errors)
    MAX_CHARS_PER_BATCH = 50_000  # ~50KB text + JSON overhead ≈ 62.5KB total
    # Note: Jina doesn't publish exact byte limit, but ~100KB total payload observed
    # We use 50KB text to leave room for JSON structure and safety margin

    # Rate limit constants (Jina API limits)
    MAX_REQUESTS_PER_MINUTE = 500
    MAX_TOKENS_PER_MINUTE = 1_000_000

    # Retry constants
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SEC = 1.0
    BACKOFF_MULTIPLIER = 2.5
    MAX_BACKOFF_SEC = 30.0

    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        dims: int = 1024,
        api_key: Optional[str] = None,
        task: str = "retrieval.passage",
        timeout: int = 60,
    ):
        """
        Initialize Jina embedding provider.

        Args:
            model: Model identifier (v3 defaults to 1024-D)
            dims: Expected embedding dimensions
            api_key: Jina API key (from env if not provided)
            task: Default task for embed_documents
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is missing or dims invalid
            RuntimeError: If tokenizer service initialization fails
        """
        self._model_id = model
        self._dims = dims
        self._task = task
        self._provider_name = "jina-ai"
        self._timeout = timeout

        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("JINA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "JINA_API_KEY required for jina-ai provider. "
                "Set JINA_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize HTTP client with structured timeout
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,  # 10s to establish connection
                read=60.0,  # 60s to read response (Jina is fast, this is safe)
                write=10.0,  # 10s to send request
                pool=5.0,  # 5s to get connection from pool
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=300)

        # Initialize rate limiter
        self._rate_limiter = RateLimiter(
            max_requests_per_minute=self.MAX_REQUESTS_PER_MINUTE,
            max_tokens_per_minute=self.MAX_TOKENS_PER_MINUTE,
        )

        # Phase 7C Hotfix: Initialize tokenizer service for accurate token counting
        try:
            from src.providers.tokenizer_service import create_tokenizer_service

            self._tokenizer_service = create_tokenizer_service()
            logger.info(
                f"Tokenizer service initialized: backend={self._tokenizer_service.backend_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer service: {e}")
            raise RuntimeError(
                f"Tokenizer service initialization failed: {e}. "
                f"This is required for accurate token counting."
            )

        logger.info(
            f"JinaEmbeddingProvider initialized: model={model}, dims={dims}, task={task}"
        )

    @property
    def dims(self) -> int:
        """Get embedding dimensions."""
        return self._dims

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents with adaptive batching.

        Uses retrieval.passage task by default.
        Automatically splits large batches to respect API limits.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts is empty
            RuntimeError: If API call fails or circuit breaker open
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            raise RuntimeError(
                "Circuit breaker OPEN - Jina API unavailable. "
                "Falling back to local provider."
            )

        # Metrics
        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        start_time = time.time()

        try:
            # Create adaptive batches
            batches = self._create_adaptive_batches(texts)

            logger.info(
                f"Created adaptive batches: total_texts={len(texts)}, "
                f"num_batches={len(batches)}, "
                f"avg_batch_size={len(texts) / len(batches):.1f}"
            )

            # Embed each batch
            all_vectors = []
            for batch_idx, batch in enumerate(batches):
                vectors = self._embed_batch_with_retry(batch, batch_idx)
                all_vectors.extend(vectors)

            # Sanity check
            assert len(all_vectors) == len(
                texts
            ), f"Vector count mismatch: {len(all_vectors)} != {len(texts)}"

            # Record success
            self._circuit_breaker.record_success()
            latency_ms = (time.time() - start_time) * 1000
            embedding_request_total.labels(
                model_id=self._model_id, operation="documents"
            ).inc()
            embedding_latency_ms.labels(
                model_id=self._model_id, operation="documents"
            ).observe(latency_ms)

            return all_vectors

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            embedding_error_total.labels(
                model_id=self._model_id, error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to embed documents with Jina: {e}")
            raise RuntimeError(f"Jina document embedding failed: {e}")

    def _create_adaptive_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches respecting both count and size limits.

        Phase 7C Hotfix: Uses ACCURATE token counting (not character estimation).
        Texts exceeding token limit are logged as errors - they should be split
        at the document processing level, not truncated here.

        Algorithm:
        1. Iterate through texts
        2. Count EXACT tokens using XLM-RoBERTa tokenizer
        3. Add to current batch if:
           - Batch count < MAX_TEXTS_PER_BATCH
           - Total chars + new text < MAX_CHARS_PER_BATCH
        4. Otherwise, start new batch

        Args:
            texts: List of text strings to batch

        Returns:
            List of batches (each batch is a list of texts)
        """
        batches = []
        current_batch = []
        current_chars = 0
        texts_truncated = 0

        for idx, text in enumerate(texts):
            text_len = len(text)

            # Phase 7C Hotfix: Use EXACT token counting (not character estimation)
            token_count = self._tokenizer_service.count_tokens(text)

            # Check if text exceeds token limit
            if token_count > self.MAX_TOKENS_PER_TEXT:
                logger.error(
                    f"Text {idx} exceeds token limit: "
                    f"original={text_len} chars, {token_count} tokens "
                    f"(limit={self.MAX_TOKENS_PER_TEXT}). "
                    f"This text should be split at document processing level. "
                    f"Truncating to prevent API error, but CONTENT WILL BE LOST."
                )
                # Truncate to exact token limit (last resort to prevent 400 error)
                text = self._tokenizer_service.truncate_to_token_limit(
                    text, self.MAX_TOKENS_PER_TEXT
                )
                text_len = len(text)
                token_count = self.MAX_TOKENS_PER_TEXT
                texts_truncated += 1

            # Check if adding would exceed limits
            would_exceed_count = len(current_batch) >= self.MAX_TEXTS_PER_BATCH
            would_exceed_size = current_chars + text_len > self.MAX_CHARS_PER_BATCH

            if current_batch and (would_exceed_count or would_exceed_size):
                # Save current batch and start new one
                batches.append(current_batch)
                logger.debug(
                    f"Batch {len(batches)} filled: texts={len(current_batch)}, "
                    f"chars={current_chars}, "
                    f"reason={'count' if would_exceed_count else 'size'}"
                )
                current_batch = []
                current_chars = 0

            current_batch.append(text)
            current_chars += text_len

        # Don't forget last batch
        if current_batch:
            batches.append(current_batch)
            logger.debug(
                f"Final batch {len(batches)}: texts={len(current_batch)}, "
                f"chars={current_chars}"
            )

        # Log truncation summary
        if texts_truncated > 0:
            logger.error(
                f"CONTENT LOSS: {texts_truncated}/{len(texts)} texts truncated. "
                f"Implement document-level splitting to prevent data loss."
            )

        return batches

    def _embed_batch_with_retry(
        self, texts: List[str], batch_idx: int
    ) -> List[List[float]]:
        """
        Embed batch with retry logic and auto-split on 400.

        Retry logic:
        - 400 error: Try splitting batch in half (payload too large)
        - Timeout: Retry with exponential backoff
        - 429 rate limit: Retry with backoff
        - Other errors: Fail immediately

        Args:
            texts: List of texts to embed (pre-batched)
            batch_idx: Batch number (for logging)

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: After max retries or unrecoverable error
        """
        batch_size = len(texts)
        total_chars = sum(len(t) for t in texts)

        # Phase 7C Hotfix: Use EXACT token counting for rate limiting
        estimated_tokens = sum(self._tokenizer_service.count_tokens(t) for t in texts)

        # Rate limiting - wait if needed
        self._rate_limiter.wait_if_needed(estimated_tokens)

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(
                    f"Embedding batch {batch_idx}: attempt={attempt + 1}, "
                    f"size={batch_size}, chars={total_chars}"
                )

                start_time = time.time()

                response = self._client.post(
                    self.API_URL,
                    json={
                        "model": self._model_id,
                        "task": self._task,  # MUST be "retrieval.passage" for documents
                        "input": texts,
                        "truncate": False,  # WE handle truncation (client-side, with visibility)
                        "normalized": True,  # Return normalized embeddings
                        "embedding_type": "float",  # Float embeddings (default)
                    },
                )

                # Handle specific HTTP errors
                if response.status_code == 400:
                    # Payload too large - try splitting
                    logger.error(
                        f"Batch {batch_idx} rejected (400 Bad Request), "
                        f"size={batch_size}, chars={total_chars}"
                    )

                    if batch_size > 1:
                        logger.info(f"Splitting batch {batch_idx} in half and retrying")
                        mid = batch_size // 2
                        vectors_1 = self._embed_batch_with_retry(
                            texts[:mid], f"{batch_idx}a"
                        )
                        vectors_2 = self._embed_batch_with_retry(
                            texts[mid:], f"{batch_idx}b"
                        )
                        return vectors_1 + vectors_2
                    else:
                        raise RuntimeError(
                            f"Single text too large ({total_chars} chars) - "
                            f"cannot split further"
                        )

                elif response.status_code == 429:
                    # Rate limit - backoff and retry
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = min(
                            self.INITIAL_BACKOFF_SEC
                            * (self.BACKOFF_MULTIPLIER**attempt),
                            self.MAX_BACKOFF_SEC,
                        )
                        logger.warning(
                            f"Rate limit (429), retrying in {backoff:.1f}s "
                            f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError("Rate limit exceeded after max retries")

                # Raise for other HTTP errors
                response.raise_for_status()

                # Parse response
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]

                # Validate dimensions
                for i, vec in enumerate(embeddings):
                    if len(vec) != self._dims:
                        raise ValueError(
                            f"Dimension mismatch at index {i}: "
                            f"got {len(vec)}-D, expected {self._dims}-D"
                        )

                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Batch {batch_idx} embedded successfully: "
                    f"{batch_size} texts in {duration_ms:.0f}ms "
                    f"({total_chars / (duration_ms / 1000):.0f} chars/s)"
                )

                return embeddings

            except httpx.ReadTimeout:
                if attempt < self.MAX_RETRIES - 1:
                    backoff = min(
                        self.INITIAL_BACKOFF_SEC * (self.BACKOFF_MULTIPLIER**attempt),
                        self.MAX_BACKOFF_SEC,
                    )
                    logger.warning(
                        f"Timeout, retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                else:
                    raise RuntimeError("Timeout after max retries")

            except httpx.HTTPStatusError as e:
                # Already handled 400 and 429 above
                logger.error(
                    f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
                )
                raise RuntimeError(
                    f"Jina API error {e.response.status_code}: {e.response.text[:100]}"
                )

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.INITIAL_BACKOFF_SEC * (
                        self.BACKOFF_MULTIPLIER**attempt
                    )
                    logger.warning(
                        f"Error: {type(e).__name__}: {e}, retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                else:
                    raise RuntimeError(
                        f"Failed after {self.MAX_RETRIES} attempts: {type(e).__name__}: {e}"
                    )

        raise RuntimeError("Retry loop exited unexpectedly")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query with full error handling and retry logic.

        Uses retrieval.query task for query-specific optimization.
        IMPORTANT: Rate limiting is SHARED with embed_documents() (same 500 RPM, 1M TPM pool).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If API call fails or circuit breaker open
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query text")

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            raise RuntimeError(
                "Circuit breaker OPEN - Jina API unavailable. "
                "Falling back to local provider."
            )

        # Metrics
        from src.shared.observability.metrics import (
            embedding_error_total,
            embedding_latency_ms,
            embedding_request_total,
        )

        # Phase 7C Hotfix: Use EXACT token counting (not character estimation)
        token_count = self._tokenizer_service.count_tokens(text)

        if token_count > self.MAX_TOKENS_PER_TEXT:
            logger.warning(
                f"Query text exceeds token limit: "
                f"original={len(text)} chars, {token_count} tokens "
                f"(limit={self.MAX_TOKENS_PER_TEXT}). "
                f"Truncating to exact token limit. Query context will be incomplete."
            )
            text = self._tokenizer_service.truncate_to_token_limit(
                text, self.MAX_TOKENS_PER_TEXT
            )

        start_time = time.time()

        # Retry loop with exponential backoff (same as embed_documents)
        for attempt in range(self.MAX_RETRIES):
            try:
                # Phase 7C Hotfix: Use EXACT token counting for rate limiting
                estimated_tokens = self._tokenizer_service.count_tokens(text)
                self._rate_limiter.wait_if_needed(estimated_tokens)

                logger.debug(
                    f"Embedding query: attempt={attempt + 1}, chars={len(text)}"
                )

                response = self._client.post(
                    self.API_URL,
                    json={
                        "model": self._model_id,
                        "task": "retrieval.query",  # MUST be "retrieval.query" for search queries (NOT passage!)
                        "input": [text],
                        "truncate": False,  # WE handle truncation (client-side, with visibility)
                        "normalized": True,
                        "embedding_type": "float",
                    },
                )

                # Handle 429 rate limit error
                if response.status_code == 429:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = min(
                            self.INITIAL_BACKOFF_SEC
                            * (self.BACKOFF_MULTIPLIER**attempt),
                            self.MAX_BACKOFF_SEC,
                        )
                        logger.warning(f"Rate limit (429), retrying in {backoff:.1f}s")
                        time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError("Rate limit exceeded after max retries")

                response.raise_for_status()
                data = response.json()
                embedding = data["data"][0]["embedding"]

                # Validate dimensions
                if len(embedding) != self._dims:
                    raise ValueError(
                        f"Dimension mismatch: got {len(embedding)}-D, expected {self._dims}-D"
                    )

                # Record success
                self._circuit_breaker.record_success()
                latency_ms = (time.time() - start_time) * 1000

                embedding_request_total.labels(
                    model_id=self._model_id, operation="query"
                ).inc()
                embedding_latency_ms.labels(
                    model_id=self._model_id, operation="query"
                ).observe(latency_ms)

                logger.debug(f"Query embedded successfully in {latency_ms:.0f}ms")
                return embedding

            except httpx.ReadTimeout:
                if attempt < self.MAX_RETRIES - 1:
                    backoff = min(
                        self.INITIAL_BACKOFF_SEC * (self.BACKOFF_MULTIPLIER**attempt),
                        self.MAX_BACKOFF_SEC,
                    )
                    logger.warning(
                        f"Timeout on query embedding, retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                else:
                    raise RuntimeError("Timeout after max retries")

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.INITIAL_BACKOFF_SEC * (
                        self.BACKOFF_MULTIPLIER**attempt
                    )
                    logger.warning(
                        f"Error embedding query: {e}, retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                else:
                    # Record failure and re-raise
                    self._circuit_breaker.record_failure()
                    embedding_error_total.labels(
                        model_id=self._model_id, error_type=type(e).__name__
                    ).inc()
                    logger.error(
                        f"Failed to embed query after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise RuntimeError(f"Jina query embedding failed: {e}")

        raise RuntimeError("Retry loop exited unexpectedly")

    def validate_dimensions(self, expected_dims: int) -> bool:
        """Validate dimensions match expected."""
        return self.dims == expected_dims

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
