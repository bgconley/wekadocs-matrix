"""
Adapter to use BGE-M3 embedding service with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges our BGE-M3 HTTP service (http://127.0.0.1:9000) to that interface.

Key points:
- Uses dense embeddings only (chonkie needs these for similarity computation)
- Sparse/ColBERT embeddings are used later during the main embedding phase
- CONSENSUS REFINEMENT: 60s timeout for boundary detection latency (o3 concern)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

log = logging.getLogger(__name__)


class OversizeEmbeddingInputError(ValueError):
    """Raised when an embedding request would exceed the model's input token limit."""

    def __init__(
        self,
        *,
        max_tokens: int,
        safe_tokens: int,
        oversize: List[dict],
        message: Optional[str] = None,
    ):
        self.max_tokens = max_tokens
        self.safe_tokens = safe_tokens
        self.oversize = oversize
        if message is None:
            msg = (
                f"Embedding input exceeds safe token limit (safe={safe_tokens}, max={max_tokens}). "
                f"Oversize items: {oversize}"
            )
        else:
            msg = message
        super().__init__(msg)


# Attempt to import chonkie's base class
try:
    from chonkie.embeddings import BaseEmbeddings

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    BaseEmbeddings = object  # Stub for type hints when chonkie not installed
    log.debug("chonkie not installed; BgeM3ChonkieAdapter will be unavailable")


class BgeM3ChonkieAdapter(BaseEmbeddings if CHONKIE_AVAILABLE else object):
    """
    Bridges BGE-M3 HTTP service to Chonkie's embedding interface.

    Uses dense embeddings only - chonkie needs these for similarity computation
    to detect semantic boundaries. The sparse/colbert embeddings are used later
    during the main embedding phase (not during chunking).

    Attributes:
        _service_url: URL of BGE-M3 service (default: http://127.0.0.1:9000)
        _model_name: Model name for API requests (BAAI/bge-m3)
        _timeout: HTTP timeout in seconds (CONSENSUS: 60s for boundary detection)
        _dimension: Dense embedding dimension (1024 for BGE-M3)
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        timeout: float = 60.0,  # CONSENSUS REFINEMENT: Increased from 30s
    ):
        """
        Initialize the BGE-M3 Chonkie adapter.

        Args:
            service_url: BGE-M3 service URL. Defaults to BGE_M3_API_URL env var
                        or http://127.0.0.1:9000
            model_name: Model identifier for API requests
            timeout: HTTP request timeout in seconds (60s for semantic boundary detection)
        """
        self._service_url = (
            service_url or os.getenv("BGE_M3_API_URL") or "http://127.0.0.1:9000"
        )
        self._model_name = model_name
        self._timeout = timeout
        self._dimension = 1024  # BGE-M3 dense embedding dimension
        self._client: Optional[httpx.Client] = None
        self._tokenizer: Optional[Any] = None  # Lazy-loaded TokenizerService

        # Hard model constraint (BGE-M3 input window). Defaults are safe for BGE-M3.
        # NOTE: "safe" is what we enforce to avoid off-by-some issues across tokenizers.
        env_max = os.getenv("BGE_M3_MAX_INPUT_TOKENS")
        env_safe = os.getenv("BGE_M3_SAFE_INPUT_TOKENS")
        try:
            self._max_input_tokens = int(env_max) if env_max else 8192
        except Exception:
            self._max_input_tokens = 8192

        try:
            self._safe_input_tokens = int(env_safe) if env_safe else 8000
        except Exception:
            self._safe_input_tokens = 8000

        # Clamp safe<=max and ensure sane lower bound.
        if self._max_input_tokens < 256:
            self._max_input_tokens = 8192
        if self._safe_input_tokens > self._max_input_tokens:
            self._safe_input_tokens = max(1, self._max_input_tokens - 1)

        # If the user hasn't explicitly set BGE_M3_SAFE_INPUT_TOKENS, align the
        # per-input safe limit with the batching cap to avoid sending a single
        # text that our own batching logic already treats as risky.
        try:
            max_batch_tokens = int(os.getenv("BGE_M3_MAX_BATCH_TOKENS", "7500"))
        except Exception:
            max_batch_tokens = 7500
        if env_safe is None and self._safe_input_tokens > max_batch_tokens:
            log.info(
                "Clamping BGE-M3 safe input tokens to batch cap",
                extra={
                    "previous_safe_input_tokens": self._safe_input_tokens,
                    "max_batch_tokens": max_batch_tokens,
                },
            )
            self._safe_input_tokens = max_batch_tokens

        # How to handle oversize inputs that slip through.
        # - raise (recommended): fail fast and let the caller pre-split.
        # - truncate: truncate to safe limit (can hide data loss).
        # - split_and_pool: split to windows, embed each, mean-pool back to one vector.
        self._oversize_policy = os.getenv("BGE_M3_OVERSIZE_POLICY", "raise").lower()

        # When split_and_pool, this overlap helps retain boundary information.
        try:
            self._pooling_overlap_tokens = int(
                os.getenv("BGE_M3_POOLING_OVERLAP_TOKENS", "128")
            )
        except Exception:
            self._pooling_overlap_tokens = 128

        log.info(
            "BgeM3ChonkieAdapter initialized",
            extra={
                "service_url": self._service_url,
                "model_name": self._model_name,
                "timeout": self._timeout,
                "dimension": self._dimension,
                "max_input_tokens": self._max_input_tokens,
                "safe_input_tokens": self._safe_input_tokens,
                "oversize_policy": self._oversize_policy,
            },
        )

    def _get_client(self) -> httpx.Client:
        """Lazy client initialization for connection reuse."""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _get_tokenizer(self):
        """Lazy tokenizer initialization to avoid circular imports."""
        if self._tokenizer is None:
            from src.providers.tokenizer_service import TokenizerService

            self._tokenizer = TokenizerService()
        return self._tokenizer

    @property
    def dimension(self) -> int:
        """Return embedding dimension (1024 for BGE-M3)."""
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            1024-dimensional numpy array (float32)
        """
        embeddings = self.embed_batch([text])
        return embeddings[0]

    # -------------------------
    # Token accounting & guards
    # -------------------------
    def _get_hf_tokenizer(self) -> Optional[Any]:
        """Best-effort access to the underlying HF tokenizer (encode/decode)."""
        try:
            tok = self.get_tokenizer()
        except Exception:
            return None
        if hasattr(tok, "encode") and hasattr(tok, "decode"):
            return tok
        return None

    def _count_tokens_model(self, text: str) -> int:
        """Count tokens using the model-aligned tokenizer where possible."""
        tok = self._get_hf_tokenizer()
        if tok is not None:
            try:
                return len(tok.encode(text, add_special_tokens=False))
            except TypeError:
                # Some tokenizers don't accept add_special_tokens
                return len(tok.encode(text))
        # Fallback: TokenizerService count
        return self._get_tokenizer().count_tokens(text)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to at most max_tokens using HF tokenizer decode."""
        tok = self._get_hf_tokenizer()
        if tok is None:
            raise OversizeEmbeddingInputError(
                max_tokens=self._max_input_tokens,
                safe_tokens=self._safe_input_tokens,
                oversize=[{"error": "no_hf_tokenizer", "required": max_tokens}],
                message="Cannot truncate: underlying HF tokenizer unavailable",
            )
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tok.encode(text)
        ids = ids[:max_tokens]
        return tok.decode(ids, skip_special_tokens=True)

    def _split_to_token_windows(
        self, text: str, *, window_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """Split text into decoded token windows (<=window_tokens) with overlap."""
        if window_tokens <= 0:
            return [text]

        tok = self._get_hf_tokenizer()
        if tok is None:
            # As a last resort, return the original text; caller should handle.
            return [text]

        try:
            ids = tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tok.encode(text)

        if len(ids) <= window_tokens:
            return [text]

        windows: List[str] = []
        start = 0
        overlap = max(0, min(overlap_tokens, window_tokens - 1))
        while start < len(ids):
            end = min(start + window_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                windows.append(chunk_text)
            if end >= len(ids):
                break
            start = max(0, end - overlap)

        return windows

    # -------------------------
    # HTTP embedding calls
    # -------------------------
    def _embed_batch_http(self, texts: List[str]) -> List[np.ndarray]:
        """
        Raw HTTP embedding call with automatic sub-batching for total token limits.

        BGE-M3 service has max_batch_tokens=8192 for the TOTAL tokens across all
        texts in a single request. This method splits large batches into sub-batches
        to stay under the limit.
        """
        if not texts:
            return []

        # BGE-M3 service has max_batch_tokens=8192 - use 7500 for safety margin
        try:
            max_batch_tokens = int(os.getenv("BGE_M3_MAX_BATCH_TOKENS", "7500"))
        except Exception:
            max_batch_tokens = 7500

        # Some BGE-M3 service implementations also cap batch size by item count.
        # Keep a conservative default, overridable via env.
        try:
            max_batch_size = int(os.getenv("BGE_M3_MAX_BATCH_SIZE", "32"))
        except Exception:
            max_batch_size = 32
        if max_batch_size < 1:
            max_batch_size = 32

        def _embed_with_split_retry(
            batch_texts: List[str], batch_idx: str = "0"
        ) -> List[np.ndarray]:
            """Retry on HTTP 400 by splitting the batch to isolate bad inputs."""
            try:
                return self._embed_single_batch_http(batch_texts)
            except httpx.HTTPStatusError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 400 and len(batch_texts) > 1:
                    log.warning(
                        "Embedding batch rejected (HTTP 400); splitting batch",
                        extra={
                            "batch_idx": batch_idx,
                            "size": len(batch_texts),
                        },
                    )
                    mid = len(batch_texts) // 2
                    head = _embed_with_split_retry(batch_texts[:mid], f"{batch_idx}a")
                    tail = _embed_with_split_retry(batch_texts[mid:], f"{batch_idx}b")
                    return head + tail
                raise

        # Calculate token counts for all texts
        token_counts = [self._count_tokens_model(t) for t in texts]
        total_tokens = sum(token_counts)

        # If total is under limit and count is reasonable, send in one batch
        if total_tokens <= max_batch_tokens and len(texts) <= max_batch_size:
            return _embed_with_split_retry(texts)

        # Split into sub-batches respecting the total token and size limits
        log.info(
            "Splitting batch for token/size limit",
            extra={
                "total_texts": len(texts),
                "total_tokens": total_tokens,
                "max_batch_tokens": max_batch_tokens,
                "max_batch_size": max_batch_size,
            },
        )

        all_embeddings: List[np.ndarray] = []
        current_batch: List[str] = []
        current_tokens = 0

        for text, tc in zip(texts, token_counts):
            # If single text exceeds limit, it goes in its own batch
            # (individual text guards should have caught this, but be safe)
            if tc > max_batch_tokens:
                # Flush current batch first
                if current_batch:
                    all_embeddings.extend(
                        _embed_with_split_retry(
                            current_batch, f"flush{len(all_embeddings)}"
                        )
                    )
                    current_batch = []
                    current_tokens = 0
                # Send oversized text alone (will likely fail, but let it error cleanly)
                all_embeddings.extend(
                    _embed_with_split_retry([text], f"single{len(all_embeddings)}")
                )
                continue

            # Would adding this text exceed the limit (tokens or count)?
            would_exceed_tokens = current_tokens + tc > max_batch_tokens
            would_exceed_size = len(current_batch) >= max_batch_size
            if would_exceed_tokens or would_exceed_size:
                if current_batch:
                    all_embeddings.extend(
                        _embed_with_split_retry(
                            current_batch, f"flush{len(all_embeddings)}"
                        )
                    )
                current_batch = [text]
                current_tokens = tc
            else:
                current_batch.append(text)
                current_tokens += tc

        # Flush final batch
        if current_batch:
            all_embeddings.extend(
                _embed_with_split_retry(current_batch, f"flush{len(all_embeddings)}")
            )

        return all_embeddings

    def _embed_single_batch_http(self, texts: List[str]) -> List[np.ndarray]:
        """Execute a single HTTP embedding request (no batching logic)."""
        if not texts:
            return []

        client = self._get_client()
        response = client.post(
            f"{self._service_url}/v1/embeddings",
            json={"model": self._model_name, "input": texts},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            if response.status_code == 400:
                # Log the service response body to help diagnose invalid requests.
                body_preview = ""
                try:
                    body_preview = (response.text or "")[:1000]
                except Exception:
                    body_preview = ""
                log.warning(
                    "BGE-M3 embedding request rejected (HTTP 400)",
                    extra={
                        "batch_size": len(texts),
                        "response_preview": body_preview,
                    },
                )
            raise
        data = response.json()

        embeddings: List[np.ndarray] = []
        for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
            embeddings.append(np.array(item["embedding"], dtype=np.float32))
        return embeddings

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch of texts via BGE-M3 service (DENSE embeddings).

        This is the primary method used by Chonkie's SemanticChunker
        for semantic boundary detection.

        Args:
            texts: List of texts to embed

        Returns:
            List of 1024-dimensional numpy arrays (float32)

        Raises:
            OversizeEmbeddingInputError: If an input exceeds the safe token limit and
                BGE_M3_OVERSIZE_POLICY=raise (default).
            httpx.HTTPError: If embedding request fails
        """
        if not texts:
            return []

        # Guard against oversize inputs (BGE-M3 rejects >8192 tokens).
        token_counts = [self._count_tokens_model(t) for t in texts]
        oversize: List[Dict[str, Any]] = [
            {"index": i, "tokens": tc}
            for i, tc in enumerate(token_counts)
            if tc > self._safe_input_tokens
        ]

        if oversize:
            policy = (self._oversize_policy or "raise").lower().strip()

            if policy == "truncate":
                log.warning(
                    "Oversize embedding inputs; truncating to safe limit",
                    extra={
                        "safe_input_tokens": self._safe_input_tokens,
                        "oversize": oversize,
                        "batch_size": len(texts),
                    },
                )
                texts = [
                    (
                        self._truncate_to_tokens(t, self._safe_input_tokens)
                        if tc > self._safe_input_tokens
                        else t
                    )
                    for t, tc in zip(texts, token_counts)
                ]
                return self._embed_batch_http(texts)

            if policy == "split_and_pool":
                log.warning(
                    "Oversize embedding inputs; splitting into windows and pooling",
                    extra={
                        "safe_input_tokens": self._safe_input_tokens,
                        "oversize": oversize,
                        "batch_size": len(texts),
                    },
                )

                # Embed safe items in one batch.
                out: List[Optional[np.ndarray]] = [None] * len(texts)
                safe_texts: List[str] = []
                safe_indices: List[int] = []
                for i, (t, tc) in enumerate(zip(texts, token_counts)):
                    if tc <= self._safe_input_tokens:
                        safe_indices.append(i)
                        safe_texts.append(t)

                if safe_texts:
                    safe_embs = self._embed_batch_http(safe_texts)
                    for idx, emb in zip(safe_indices, safe_embs):
                        out[idx] = emb

                # For oversize items: split into safe windows, embed, mean-pool.
                for item in oversize:
                    i = int(item["index"])
                    windows = self._split_to_token_windows(
                        texts[i],
                        window_tokens=self._safe_input_tokens,
                        overlap_tokens=self._pooling_overlap_tokens,
                    )
                    window_embs = self._embed_batch_http(windows)
                    pooled = np.mean(np.stack(window_embs, axis=0), axis=0).astype(
                        np.float32
                    )
                    # Normalize to unit length (common for retrieval embeddings).
                    norm = float(np.linalg.norm(pooled))
                    if norm > 0:
                        pooled = (pooled / norm).astype(np.float32)
                    out[i] = pooled

                # mypy: we ensure all are filled
                return [
                    e if e is not None else np.zeros(self._dimension, dtype=np.float32)
                    for e in out
                ]

            # Default: raise (fail fast) and force caller to pre-split.
            raise OversizeEmbeddingInputError(
                max_tokens=self._max_input_tokens,
                safe_tokens=self._safe_input_tokens,
                oversize=oversize,
            )

        # No oversize; perform normal embedding.
        try:
            return self._embed_batch_http(texts)
        except httpx.HTTPError as e:
            log.error(
                "BGE-M3 embedding request failed",
                extra={"error": str(e), "batch_size": len(texts)},
            )
            raise

    # Alias for clarity
    embed_dense = embed_batch

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Generate SPARSE (BM25-style lexical) embeddings via BGE-M3 service.

        Maintains continuity with BGEM3ServiceProvider for full multi-vector support.
        Sparse embeddings capture lexical/keyword signals for hybrid retrieval.

        Args:
            texts: List of texts to embed

        Returns:
            List of sparse embedding dicts with 'indices' and 'values' keys

        Raises:
            httpx.HTTPError: If embedding request fails
        """
        if not texts:
            return []

        client = self._get_client()
        try:
            response = client.post(
                f"{self._service_url}/v1/embeddings/sparse",
                json={"model": self._model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            log.error(
                "BGE-M3 sparse embedding request failed",
                extra={"error": str(e), "batch_size": len(texts)},
            )
            raise

        # Response format: {"data": [{"index": 0, "indices": [...], "values": [...], ...}]}
        # Extract indices and values from each item, sorted by index
        results = []
        items = data.get("data", []) if isinstance(data, dict) else data
        for item in sorted(items, key=lambda x: x.get("index", 0)):
            results.append(
                {
                    "indices": item.get("indices", []),
                    "values": item.get("values", []),
                }
            )
        return results

    def embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """
        Generate ColBERT (late-interaction) multi-vector embeddings via BGE-M3 service.

        Maintains continuity with BGEM3ServiceProvider for full multi-vector support.
        ColBERT embeddings are token-level vectors for MaxSim scoring.

        Args:
            texts: List of texts to embed

        Returns:
            List of ColBERT embeddings, each is a list of token vectors (1024-D each)

        Raises:
            httpx.HTTPError: If embedding request fails
        """
        if not texts:
            return []

        client = self._get_client()
        try:
            response = client.post(
                f"{self._service_url}/v1/embeddings/colbert",
                json={"model": self._model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            log.error(
                "BGE-M3 ColBERT embedding request failed",
                extra={"error": str(e), "batch_size": len(texts)},
            )
            raise

        # Response format: {"data": [{"index": 0, "vectors": [[...], ...], ...}, ...]}
        # Extract vectors from each item, sorted by index
        results = []
        items = data.get("data", []) if isinstance(data, dict) else data
        for item in sorted(items, key=lambda x: x.get("index", 0)):
            vectors = item.get("vectors", [])
            results.append(vectors)
        return results

    def embed_all(self, texts: List[str]) -> List[dict]:
        """
        Generate all three embedding types (dense, sparse, ColBERT) in one call.

        Provides full continuity with the existing 8-vector pipeline.
        Useful when you need all modalities for a set of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of dicts with 'dense', 'sparse', and 'colbert' keys

        Note:
            Makes three separate HTTP calls. For high-volume use, consider
            using BGEM3ServiceProvider.embed_documents_all() which may have
            optimizations.
        """
        if not texts:
            return []

        dense = self.embed_batch(texts)
        sparse = self.embed_sparse(texts)
        colbert = self.embed_colbert(texts)

        results = []
        for i in range(len(texts)):
            results.append(
                {
                    "dense": dense[i] if i < len(dense) else None,
                    "sparse": sparse[i] if i < len(sparse) else None,
                    "colbert": colbert[i] if i < len(colbert) else None,
                }
            )
        return results

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using our tokenizer service.

        Uses the profile-aware TokenizerService (HuggingFace backend)
        for accurate token counting with the BGE-M3 tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        # Prefer the model-aligned HF tokenizer when available.
        return self._count_tokens_model(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            List of token counts
        """
        return [self.count_tokens(t) for t in texts]

    def get_tokenizer(self) -> Any:
        """
        Return the underlying HuggingFace tokenizer for chonkie's internal use.

        Required by Chonkie's BaseEmbeddings abstract class.
        Chonkie's AutoTokenizer expects a HuggingFace PreTrainedTokenizer,
        not our TokenizerService wrapper. We return the actual tokenizer.

        Returns:
            HuggingFace PreTrainedTokenizer (from TokenizerService's backend)
        """
        tokenizer_service = self._get_tokenizer()
        # Return the underlying HuggingFace tokenizer, not our wrapper
        # TokenizerService.backend is a HuggingFaceTokenizerBackend with .tokenizer attribute
        backend = getattr(tokenizer_service, "backend", None)
        if backend and hasattr(backend, "tokenizer"):
            return backend.tokenizer
        # Fallback: return the service itself and hope Chonkie can handle it
        log.warning(
            "Could not extract HuggingFace tokenizer from TokenizerService; "
            "Chonkie may not work correctly"
        )
        return tokenizer_service

    # Alias for backward compatibility
    def get_tokenizer_or_token_counter(self) -> Any:
        """Alias for get_tokenizer() - backward compatibility."""
        return self.get_tokenizer()

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if BGE-M3 service is reachable.

        This is used to determine whether to use semantic chunking
        or fall back to simpler approaches.

        Returns:
            True if chonkie is installed AND BGE-M3 service is healthy
        """
        if not CHONKIE_AVAILABLE:
            log.debug("Chonkie not available - CHONKIE_AVAILABLE=False")
            return False

        service_url = os.getenv("BGE_M3_API_URL") or "http://127.0.0.1:9000"
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{service_url}/healthz")
                if r.status_code == 200:
                    health = r.json()
                    is_healthy = health.get("status") == "ok"
                    if is_healthy:
                        log.debug(
                            "BGE-M3 service healthy",
                            extra={"service_url": service_url},
                        )
                    return is_healthy
                return False
        except Exception as e:
            log.debug(
                "BGE-M3 service unavailable",
                extra={"service_url": service_url, "error": str(e)},
            )
            return False

    def close(self):
        """Close the HTTP client and release resources."""
        if self._client:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"BgeM3ChonkieAdapter(url={self._service_url}, dim={self._dimension})"

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()
