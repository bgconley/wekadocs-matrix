"""
Tokenizer service for accurate token counting and text splitting.

CRITICAL: This module uses the EXACT tokenizer for jina-embeddings-v3 (XLM-RoBERTa family).
DO NOT use tiktoken or cl100k_base - those are for OpenAI models and will give wrong counts.

Dual-backend architecture:
- Primary: HuggingFace local tokenizer (fast, exact, no network)
- Secondary: Jina Segmenter API (validation, fallback, FREE - not billed)

Features:
- Exact token counting using model-specific tokenizer
- Lossless text splitting with configurable overlap
- SHA256 integrity verification
- Environment-based backend selection
- Comprehensive error handling
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from src.providers.settings import build_embedding_telemetry
from src.shared.config import get_config, get_embedding_plan, get_embedding_settings

logger = logging.getLogger(__name__)


class TokenizerBackend(ABC):
    """Abstract base class for tokenizer backends."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using model-specific tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            Exact token count

        Raises:
            RuntimeError: If tokenization fails
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs

        Raises:
            RuntimeError: If encoding fails
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text

        Raises:
            RuntimeError: If decoding fails
        """
        pass


class HuggingFaceTokenizerBackend(TokenizerBackend):
    """
    HuggingFace local tokenizer backend (PRIMARY).

    Uses the exact tokenizer for jina-embeddings-v3.
    Fast (<5ms per section), deterministic, works offline.
    """

    def __init__(self, *, model_id: Optional[str] = None):
        """
        Initialize HuggingFace tokenizer.

        Loads jinaai/jina-embeddings-v3 tokenizer from cache.
        Expects tokenizer to be prefetched during Docker build.

        Raises:
            RuntimeError: If tokenizer cannot be loaded
        """
        try:
            from transformers import AutoTokenizer

            embedding_settings = get_embedding_settings()
            default_model_id = (
                model_id
                or embedding_settings.tokenizer_model_id
                or embedding_settings.model_id
            )
            model_id = os.getenv(
                "HF_TOKENIZER_ID", default_model_id or "jinaai/jina-embeddings-v3"
            )
            cache_dir = os.getenv("HF_CACHE", "/opt/hf-cache")
            offline = os.getenv("TRANSFORMERS_OFFLINE", "true").lower() == "true"

            telemetry = build_embedding_telemetry(embedding_settings)
            log_extra = {
                **telemetry,
                "hf_model_id": model_id,
                "hf_cache": cache_dir,
                "hf_offline": offline,
            }
            logger.info(
                "Loading HuggingFace tokenizer with profile-aware settings",
                extra=log_extra,
            )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=offline,
                )
            except Exception as first_err:
                if offline:
                    raise RuntimeError(
                        f"HuggingFace tokenizer cache miss for {model_id!r} in offline mode. "
                        "Prefetch the tokenizer into HF_CACHE, or set TRANSFORMERS_OFFLINE=false."
                    ) from first_err
                raise

            # Test tokenizer works
            test_tokens = self.tokenizer.encode("test", add_special_tokens=False)
            logger.info(
                f"HuggingFace tokenizer loaded successfully "
                f"(test: 'test' -> {len(test_tokens)} tokens)"
            )

        except Exception as e:
            logger.error(f"Failed to load HuggingFace tokenizer: {e}")
            raise RuntimeError(
                f"HuggingFace tokenizer initialization failed: {e}. "
                f"Ensure transformers and tokenizer cache are available."
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer."""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise RuntimeError(f"HuggingFace token counting failed: {e}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise RuntimeError(f"HuggingFace encoding failed: {e}")

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise RuntimeError(f"HuggingFace decoding failed: {e}")


class JinaSegmenterBackend(TokenizerBackend):
    """
    Jina Segmenter API backend (SECONDARY/VALIDATION).

    Uses Jina's free Segmenter API for token counting.
    IMPORTANT: Token counting is FREE (not billed as token usage)!
    Slower (~300ms + network) but useful for validation and fallback.
    """

    def __init__(self, *, tokenizer_name: Optional[str] = None):
        """
        Initialize Jina Segmenter API client.

        Uses JINA_API_KEY from environment (optional but improves rate limits).

        Rate limits:
        - Free tier: 20 requests/min
        - With API key: 200 requests/min
        - Premium: 1000 requests/min

        Raises:
            RuntimeError: If client initialization fails
        """
        self.base_url = os.getenv(
            "JINA_SEGMENTER_BASE_URL", "https://api.jina.ai/v1/segment"
        )
        self.api_key = os.getenv("JINA_API_KEY")

        # Determine tokenizer name from profile/env (no remapping here)
        embedding_settings = get_embedding_settings()
        default_model_id = (
            tokenizer_name
            or embedding_settings.tokenizer_model_id
            or embedding_settings.model_id
        )
        self.tokenizer_name = os.getenv(
            "SEGMENTER_TOKENIZER_NAME", default_model_id or "xlm-roberta-base"
        )

        timeout_ms = int(os.getenv("SEGMENTER_TIMEOUT_MS", "5000"))

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.Client(
            timeout=timeout_ms / 1000.0,
            headers=headers,
        )

        logger.info(
            f"Jina Segmenter backend initialized: "
            f"tokenizer={self.tokenizer_name}, "
            f"has_key={bool(self.api_key)}"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Jina Segmenter API.

        NOTE: This is FREE - not billed as token usage!
        Rate limits still apply.
        """
        try:
            response = self.client.post(
                self.base_url,
                json={
                    "content": text,
                    "tokenizer": self.tokenizer_name,
                    "return_tokens": True,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Response includes num_tokens or tokens array
            token_count = data.get("num_tokens", len(data.get("tokens", [])))
            return token_count

        except httpx.TimeoutException:
            raise RuntimeError("Jina Segmenter API timeout")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Jina Segmenter API error {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
        except Exception as e:
            raise RuntimeError(f"Jina Segmenter API failed: {e}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text using Segmenter API.

        Returns token IDs from Segmenter response.
        """
        try:
            response = self.client.post(
                self.base_url,
                json={
                    "content": text,
                    "tokenizer": self.tokenizer_name,
                    "return_tokens": True,
                    "return_ids": True,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Return token IDs if available, otherwise empty list
            return data.get("token_ids", [])

        except Exception as e:
            raise RuntimeError(f"Jina Segmenter encoding failed: {e}")

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs using Segmenter API.

        Note: Jina Segmenter may not support decoding directly.
        This is a best-effort implementation.
        """
        # Segmenter API doesn't provide decode endpoint
        # This is a limitation of the secondary backend
        raise NotImplementedError(
            "Jina Segmenter does not support decoding. "
            "Use HuggingFace backend for splitting operations."
        )

    def __del__(self):
        """Close HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


class VoyageTokenCounterBackend:
    """
    Voyage exact token counting backend (count-only).

    Uses voyageai.Client.count_tokens for model-specific counts.
    """

    def __init__(self, *, model_id: str, api_key: Optional[str] = None) -> None:
        try:
            import voyageai
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "voyageai package is required for Voyage token counting."
            ) from exc

        self._model_id = model_id
        self._client = voyageai.Client(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        return int(self._client.count_tokens([text], model=self._model_id))

    def count_tokens_batch(self, texts: List[str]) -> int:
        return int(self._client.count_tokens(texts, model=self._model_id))


class TokenizerService:
    """
    Tokenizer service with dual-backend support.

    Provides accurate token counting and lossless text splitting
    for jina-embeddings-v3 (XLM-RoBERTa tokenizer).

    Features:
    - Exact token counting (no estimation)
    - Lossless splitting with overlap
    - SHA256 integrity verification
    - Backend selection via environment variable
    """

    def __init__(self):
        """
        Initialize tokenizer service.

        Selects backend based on TOKENIZER_BACKEND environment variable:
        - 'hf' (default): HuggingFace local tokenizer
        - 'segmenter': Jina Segmenter API

        Raises:
            ValueError: If backend is invalid
            RuntimeError: If backend initialization fails
        """
        embedding_settings = get_embedding_settings()
        embedding_profile = (getattr(embedding_settings, "profile", "") or "").lower()
        dense_profile_name = embedding_profile
        dense_model_id = embedding_settings.model_id
        tokenizer_model_id = embedding_settings.tokenizer_model_id
        token_counting = None

        try:
            embedding_plan = get_embedding_plan()
            dense_profile_name = embedding_plan.dense.profile_name.lower()
            dense_model_id = embedding_plan.dense.profile.model_id
            tokenizer_model_id = embedding_plan.dense.profile.tokenizer.model_id
            token_counting = embedding_plan.dense.profile.token_counting
        except Exception:
            embedding_plan = None

        backend_name = os.getenv("TOKENIZER_BACKEND")
        backend_source = "env" if backend_name else None

        if not backend_name:
            # Profile-driven routing
            if dense_profile_name in {"bge_m3", "bge-m3", "bge-m3-unpad"}:
                backend_name = "hf"
            elif "jina" in dense_profile_name:
                backend_name = "segmenter"
            else:
                backend_name = (
                    embedding_settings.tokenizer_backend
                    if embedding_settings.tokenizer_backend
                    else "hf"
                )
            backend_source = "profile"

        if not backend_name:
            backend_source = "config"
            try:
                config = get_config()
                tokenizer_config = getattr(config, "tokenizer", None)
                if tokenizer_config:
                    backend_name = getattr(tokenizer_config, "backend", None)
            except Exception:
                backend_name = None

        if not backend_name:
            backend_source = "default"
            backend_name = "hf"

        backend_name = backend_name.lower()

        # Fallback allowance: disallow segmenter fallback for BGE profiles unless explicitly enabled
        allow_segmenter_fallback = (
            os.getenv("TOKENIZER_ALLOW_SEGMENTER_FALLBACK", "").lower() == "true"
        ) or dense_profile_name not in {
            "bge_m3",
            "bge-m3",
            "bge-m3-service",
            "bge-m3-unpad",
        }

        if backend_name == "hf":
            try:
                self.backend = HuggingFaceTokenizerBackend(model_id=tokenizer_model_id)
                self.backend_name = "huggingface"
            except Exception as exc:
                if allow_segmenter_fallback:
                    logger.warning(
                        "HuggingFace tokenizer initialization failed; falling back to Jina Segmenter",
                        exc_info=exc,
                    )
                    self.backend = JinaSegmenterBackend(
                        tokenizer_name=tokenizer_model_id
                    )
                    self.backend_name = "jina-segmenter"
                else:
                    raise RuntimeError(
                        "HuggingFace tokenizer initialization failed and segmenter fallback is disabled "
                        f"for profile '{dense_profile_name}'"
                    ) from exc
        elif backend_name == "segmenter":
            self.backend = JinaSegmenterBackend(tokenizer_name=tokenizer_model_id)
            self.backend_name = "jina-segmenter"
        else:
            raise ValueError(
                f"Invalid tokenizer backend: {backend_name}. "
                f"Must be 'hf' or 'segmenter'."
            )

        count_backend_name = os.getenv("TOKENIZER_COUNT_BACKEND")
        count_backend_source = "env" if count_backend_name else None
        if not count_backend_name:
            if token_counting and token_counting.backend:
                count_backend_name = token_counting.backend
            else:
                count_backend_name = backend_name
            count_backend_source = "profile"
        count_backend_name = count_backend_name.lower() if count_backend_name else None

        self.count_backend = None
        self.count_backend_name = None
        if count_backend_name == "voyage":
            allow_voyage_fallback = (
                os.getenv("TOKENIZER_ALLOW_VOYAGE_FALLBACK", "").lower() == "true"
            )
            try:
                count_model_id = (
                    token_counting.model_id
                    if token_counting and token_counting.model_id
                    else dense_model_id
                )
                api_key = os.getenv("VOYAGE_API_KEY")
                self.count_backend = VoyageTokenCounterBackend(
                    model_id=count_model_id, api_key=api_key
                )
                self.count_backend_name = "voyage"
            except Exception as exc:
                if allow_voyage_fallback:
                    logger.warning(
                        "Voyage token counter initialization failed; falling back to HuggingFace",
                        exc_info=exc,
                    )
                else:
                    raise RuntimeError(
                        "Voyage token counting initialization failed and fallback is disabled."
                    ) from exc
        elif count_backend_name == "hf":
            if backend_name == "hf":
                self.count_backend = self.backend
            else:
                self.count_backend = HuggingFaceTokenizerBackend(
                    model_id=tokenizer_model_id
                )
            self.count_backend_name = "huggingface"
        elif count_backend_name == "segmenter":
            if backend_name == "segmenter":
                self.count_backend = self.backend
            else:
                self.count_backend = JinaSegmenterBackend(
                    tokenizer_name=tokenizer_model_id
                )
            self.count_backend_name = "jina-segmenter"

        if self.count_backend is None:
            self.count_backend = self.backend
            self.count_backend_name = self.backend_name

        # Token limits (from Jina API specifications)
        self.max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "8192"))
        self.target_tokens = int(os.getenv("EMBED_TARGET_TOKENS", "7900"))
        self.overlap_tokens = int(os.getenv("EMBED_OVERLAP_TOKENS", "200"))
        self.min_tokens = int(os.getenv("SPLIT_MIN_TOKENS", "1000"))

        # Observability flags
        self.log_split_decisions = (
            os.getenv("LOG_SPLIT_DECISIONS", "true").lower() == "true"
        )
        self.integrity_check_rate = float(
            os.getenv("INTEGRITY_CHECK_SAMPLE_RATE", "0.05")
        )

        logger.info(
            "TokenizerService initialized",
            extra={
                "backend": self.backend_name,
                "backend_source": backend_source,
                "count_backend": self.count_backend_name,
                "count_backend_source": count_backend_source,
                "max_tokens": self.max_tokens,
                "target_tokens": self.target_tokens,
                "overlap_tokens": self.overlap_tokens,
            },
        )

    def count_tokens(self, text: str) -> int:
        """
        Count exact tokens in text.

        Args:
            text: Input text

        Returns:
            Exact token count using model-specific tokenizer

        Raises:
            RuntimeError: If token counting fails
        """
        if self.count_backend:
            return self.count_backend.count_tokens(text)
        return self.backend.count_tokens(text)

    def count_tokens_batch(self, texts: List[str]) -> int:
        if not texts:
            return 0
        if self.count_backend and hasattr(self.count_backend, "count_tokens_batch"):
            return self.count_backend.count_tokens_batch(texts)
        return sum(self.count_tokens(text) for text in texts)

    @property
    def supports_decode(self) -> bool:
        return hasattr(self.backend, "decode") and not isinstance(
            self.backend, JinaSegmenterBackend
        )

    def encode(self, text: str) -> List[int]:
        if not hasattr(self.backend, "encode"):
            raise RuntimeError(
                f"Tokenizer backend {self.backend_name} does not support encode()."
            )
        return self.backend.encode(text)

    def decode_tokens(self, token_ids: List[int]) -> str:
        if not self.supports_decode:
            raise NotImplementedError(
                f"Tokenizer backend {self.backend_name} does not support decode()."
            )
        return self.backend.decode(token_ids)

    def needs_splitting(self, text: str) -> bool:
        """
        Check if text exceeds token limit.

        Args:
            text: Input text

        Returns:
            True if text exceeds max_tokens limit
        """
        return self.count_tokens(text) > self.max_tokens

    def compute_integrity_hash(self, text: str) -> str:
        """
        Compute SHA256 hash for integrity verification.

        Args:
            text: Input text

        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def split_to_chunks(
        self,
        text: str,
        section_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split text into token-sized chunks with overlap (LOSSLESS).

        Algorithm:
        1. Check if splitting needed (token count > max_tokens)
        2. If no split needed, return single chunk
        3. Otherwise, encode to tokens and split with overlap
        4. Maintain chunk relationships and metadata
        5. Verify integrity with SHA256 hash

        Args:
            text: Text to split (may be large)
            section_id: Optional parent section identifier

        Returns:
            List of chunk dictionaries with metadata:
            [
                {
                    'text': str,              # Chunk text content
                    'chunk_index': int,       # 0-based chunk index
                    'total_chunks': int,      # Total chunks created
                    'token_count': int,       # Exact token count for this chunk
                    'char_count': int,        # Character count
                    'overlap_start': bool,    # Has overlap from previous
                    'overlap_end': bool,      # Has overlap with next
                    'integrity_hash': str,    # SHA256 of chunk content
                    'parent_section_id': str  # Original section ID (if provided)
                }
            ]

        Raises:
            RuntimeError: If backend doesn't support splitting (e.g., Segmenter)
        """
        if not isinstance(self.backend, HuggingFaceTokenizerBackend):
            raise RuntimeError(
                f"Splitting requires HuggingFace backend (current: {self.backend_name}). "
                f"Set TOKENIZER_BACKEND=hf"
            )

        token_count = self.count_tokens(text)

        # No splitting needed if under target (safe chunking size)
        # Note: We split at target_tokens, not max_tokens, for efficient batching
        if token_count <= self.target_tokens:
            return [
                {
                    "text": text,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "token_count": token_count,
                    "char_count": len(text),
                    "overlap_start": False,
                    "overlap_end": False,
                    "overlap_chars": 0,  # No overlap for single chunk
                    "overlap_tokens": 0,  # No overlap for single chunk
                    "integrity_hash": self.compute_integrity_hash(text),
                    "parent_section_id": section_id,
                }
            ]

        # Split required
        if self.log_split_decisions:
            logger.info(
                f"Splitting text: {token_count} tokens -> chunks of {self.target_tokens} "
                f"with {self.overlap_tokens} overlap"
            )

        # Encode entire text to tokens
        tokens = self.encode(text)
        total_tokens = len(tokens)

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < total_tokens:
            # Calculate chunk end position
            end_idx = min(start_idx + self.target_tokens, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode chunk tokens back to text
            chunk_text = self.decode_tokens(chunk_tokens)

            # Calculate precise overlap for this chunk (Plan 3 enhancement)
            # For chunks after the first, the overlap is the tokens that were at the
            # END of the previous chunk and now appear at the START of this chunk.
            # These are tokens[start_idx : start_idx + overlap_tokens] if we have overlap.
            if chunk_index > 0 and start_idx > 0:
                # How many tokens of overlap do we actually have?
                # It's min(overlap_tokens, start_idx) in case we're near the beginning
                actual_overlap_tokens = min(self.overlap_tokens, start_idx)
                # The overlap region is at the BEGINNING of this chunk
                overlap_token_slice = chunk_tokens[:actual_overlap_tokens]
                overlap_text = self.decode_tokens(overlap_token_slice)
                overlap_chars = len(overlap_text)
            else:
                overlap_chars = 0
                actual_overlap_tokens = 0

            # Create chunk metadata
            chunk = {
                "text": chunk_text,
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will update after loop
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "overlap_start": start_idx > 0,
                "overlap_end": end_idx < total_tokens,
                "overlap_chars": overlap_chars,  # NEW: Exact character count of overlap
                "overlap_tokens": actual_overlap_tokens,  # NEW: Token count of overlap
                "integrity_hash": self.compute_integrity_hash(chunk_text),
                "parent_section_id": section_id,
            }

            chunks.append(chunk)

            # Move to next chunk with overlap
            if end_idx < total_tokens:
                start_idx = end_idx - self.overlap_tokens
            else:
                start_idx = end_idx

            chunk_index += 1

        # Update total_chunks in all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        # Integrity verification (sample-based)
        if total_chunks > 1 and self.integrity_check_rate > 0:
            import random

            if random.random() < self.integrity_check_rate:
                self._verify_chunk_integrity(text, chunks)

        if self.log_split_decisions:
            logger.info(
                f"Split complete: {total_tokens} tokens -> {total_chunks} chunks, "
                f"avg_tokens={total_tokens / total_chunks:.0f}"
            )

        return chunks

    def _verify_chunk_integrity(
        self, original: str, chunks: List[Dict], strict: bool = False
    ) -> Dict[str, Any]:
        """
        Verify chunks can be reassembled to original (with overlap removal).

        Enhanced with precise overlap tracking (Plan 3). Uses stored overlap_chars
        when available, with precise fallback for legacy chunks.

        Args:
            original: Original text before splitting
            chunks: List of chunk dictionaries
            strict: If True, cross-validate stored vs calculated overlap

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "reassembled_length": int,
                "original_length": int,
                "length_diff": int,
                "length_diff_percent": float,
                "overlap_mismatches": List[Dict]
            }
        """
        results = {
            "valid": True,
            "reassembled_length": 0,
            "original_length": len(original),
            "overlap_mismatches": [],
        }

        try:
            # Reassemble without overlaps
            reassembled = chunks[0]["text"]
            prev_chunk = chunks[0]

            for i in range(1, len(chunks)):
                chunk = chunks[i]
                if chunk.get("overlap_start", False):
                    # Try to use stored overlap_chars (Plan 3 enhancement)
                    stored_overlap = chunk.get("overlap_chars")

                    if stored_overlap is not None:
                        # Use stored value directly
                        overlap_chars = stored_overlap

                        # Cross-validate if strict mode
                        if strict and prev_chunk:
                            calculated = self._calculate_overlap_chars_precise(
                                chunk["text"],
                                prev_chunk.get("text", ""),
                                chunk.get("overlap_tokens", self.overlap_tokens),
                            )
                            if abs(calculated - stored_overlap) > 5:
                                results["overlap_mismatches"].append(
                                    {
                                        "chunk_index": i,
                                        "stored": stored_overlap,
                                        "calculated": calculated,
                                        "diff": abs(calculated - stored_overlap),
                                    }
                                )
                                logger.error(
                                    "overlap_mismatch_detected",
                                    extra={
                                        "chunk_index": i,
                                        "stored": stored_overlap,
                                        "calculated": calculated,
                                    },
                                )
                    else:
                        # Legacy chunk: calculate precisely
                        overlap_chars = self._calculate_overlap_chars_precise(
                            chunk["text"],
                            prev_chunk.get("text", "") if prev_chunk else "",
                            chunk.get("overlap_tokens", self.overlap_tokens),
                        )

                    reassembled += chunk["text"][overlap_chars:]
                else:
                    reassembled += chunk["text"]

                prev_chunk = chunk

            results["reassembled_length"] = len(reassembled)

            # Compare lengths (should be very close with precise overlap)
            len_diff = abs(len(reassembled) - len(original))
            len_threshold = len(original) * 0.001  # 0.1% tolerance with precise overlap

            results["length_diff"] = len_diff
            results["length_diff_percent"] = (
                (len_diff / len(original)) * 100 if len(original) > 0 else 0
            )

            if len_diff > len_threshold:
                results["valid"] = False
                logger.warning(
                    f"Integrity check: length difference {len_diff} chars "
                    f"({results['length_diff_percent']:.2f}%), "
                    f"original={len(original)}, reassembled={len(reassembled)}"
                )

            # Hash comparison
            original_hash = self.compute_integrity_hash(original)
            reassembled_hash = self.compute_integrity_hash(reassembled)

            if original_hash != reassembled_hash:
                logger.debug(
                    f"Integrity check: hash mismatch, "
                    f"chunks={len(chunks)}, len_diff={len_diff}"
                )

            if results["overlap_mismatches"]:
                results["valid"] = False

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            results["valid"] = False
            results["error"] = str(e)

        return results

    def _estimate_overlap_chars(self, text: str, overlap_tokens: int) -> int:
        """
        Estimate character count for overlap_tokens.

        DEPRECATED: This uses a rough 3-chars/token heuristic which is incorrect
        for XLM-RoBERTa (actual range: 2.2 to 4.75 chars/token). Use stored
        overlap_chars from split_to_chunks() or _calculate_overlap_chars_precise()
        for legacy chunks. See Plan 3 for details.

        Args:
            text: Chunk text
            overlap_tokens: Number of overlap tokens

        Returns:
            Estimated character count for overlap (INACCURATE)
        """
        # Conservative estimate: 3 chars per token
        estimated_chars = overlap_tokens * 3
        # Don't exceed chunk length
        return min(estimated_chars, len(text) // 2)

    def _calculate_overlap_chars_precise(
        self,
        chunk_text: str,
        prev_chunk_text: str,
        overlap_tokens: int,
    ) -> int:
        """
        Calculate precise character count for overlap by re-encoding.

        Used for legacy chunks that don't have stored overlap_chars.
        Re-encodes the previous chunk, decodes its tail, and finds the
        exact boundary in the current chunk.

        Args:
            chunk_text: Current chunk text
            prev_chunk_text: Previous chunk text
            overlap_tokens: Number of overlap tokens

        Returns:
            Exact character count of overlap region
        """
        if not prev_chunk_text or overlap_tokens <= 0:
            return 0

        try:
            # Re-tokenize the previous chunk
            prev_tokens = self.encode(prev_chunk_text)

            if len(prev_tokens) < overlap_tokens:
                # Previous chunk is smaller than overlap - use entire chunk
                overlap_token_slice = prev_tokens
            else:
                overlap_token_slice = prev_tokens[-overlap_tokens:]

            # Decode the overlap tokens to get the expected overlap text
            expected_overlap_text = self.decode_tokens(overlap_token_slice)

            # Verify: current chunk should start with this text
            if chunk_text.startswith(expected_overlap_text):
                return len(expected_overlap_text)

            # Handle tokenizer normalization edge cases
            return self._fuzzy_find_overlap_boundary(chunk_text, expected_overlap_text)

        except Exception as e:
            logger.warning(f"Precise overlap calculation failed, using heuristic: {e}")
            return self._estimate_overlap_chars(chunk_text, overlap_tokens)

    def _fuzzy_find_overlap_boundary(
        self,
        chunk_text: str,
        expected_overlap: str,
    ) -> int:
        """
        Handle edge cases where tokenizer normalization causes slight mismatches.

        Common causes:
        - Leading/trailing whitespace normalization
        - Unicode normalization (NFKC vs NFC)
        - Special token insertions

        Args:
            chunk_text: Current chunk text
            expected_overlap: Expected overlap text from decoding tokens

        Returns:
            Best-effort character count for overlap boundary
        """
        if not expected_overlap:
            return 0

        # Try with stripped whitespace
        stripped_expected = expected_overlap.strip()
        stripped_chunk = chunk_text.lstrip()

        if stripped_chunk.startswith(stripped_expected):
            # Account for stripped leading whitespace
            leading_ws = len(chunk_text) - len(stripped_chunk)
            return len(stripped_expected) + leading_ws

        # Fallback: find longest common prefix
        for length in range(len(expected_overlap), 0, -1):
            if chunk_text[:length] == expected_overlap[:length]:
                if length < len(expected_overlap):
                    logger.warning(
                        "fuzzy_overlap_match",
                        extra={
                            "expected_len": len(expected_overlap),
                            "matched_len": length,
                            "diff": len(expected_overlap) - length,
                        },
                    )
                return length

        logger.error(
            "overlap_boundary_not_found",
            extra={
                "expected_start": expected_overlap[:50],
                "actual_start": chunk_text[:50],
            },
        )
        # Last resort: use heuristic
        return self._estimate_overlap_chars(chunk_text, len(expected_overlap) // 3)

    def verify_chunks(
        self,
        chunks: List[Dict],
        original_text: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Public API for verifying chunk integrity.

        Reassembles chunks by stripping overlap regions and validates that
        the result matches the original text (if provided).

        Args:
            chunks: List of chunk dictionaries from split_to_chunks()
            original_text: Optional original text for comparison
            strict: If True, cross-validate stored vs calculated overlap

        Returns:
            Dict with validation results including reassembled text
        """
        if not chunks:
            return {"valid": True, "reassembled": "", "chunks": 0}

        # Build reassembled text
        reassembled_parts = [chunks[0]["text"]]
        prev_chunk = chunks[0]

        for i in range(1, len(chunks)):
            chunk = chunks[i]
            overlap_chars = chunk.get("overlap_chars", 0)

            # Use stored value or calculate precisely
            if overlap_chars == 0 and chunk.get("overlap_start", False):
                overlap_chars = self._calculate_overlap_chars_precise(
                    chunk["text"],
                    prev_chunk.get("text", ""),
                    chunk.get("overlap_tokens", self.overlap_tokens),
                )

            reassembled_parts.append(chunk["text"][overlap_chars:])
            prev_chunk = chunk

        reassembled = "".join(reassembled_parts)

        result = {
            "valid": True,
            "reassembled": reassembled,
            "reassembled_length": len(reassembled),
            "chunks": len(chunks),
        }

        if original_text:
            result["original_length"] = len(original_text)
            result["length_diff"] = abs(len(reassembled) - len(original_text))
            result["length_diff_percent"] = (
                (result["length_diff"] / len(original_text)) * 100
                if len(original_text) > 0
                else 0
            )
            result["valid"] = (
                result["length_diff_percent"] < 1.0
            )  # 1% tolerance (improved from 6.72%)

        return result

    def truncate_to_token_limit(
        self, text: str, max_tokens: Optional[int] = None
    ) -> str:
        """
        Truncate text to exact token count.

        Use this only when lossless splitting is not desired (e.g., queries).
        For documents, prefer split_to_chunks() to preserve all content.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (defaults to self.max_tokens)

        Returns:
            Truncated text with exact token count

        Raises:
            RuntimeError: If backend doesn't support truncation
        """
        if not isinstance(self.backend, HuggingFaceTokenizerBackend):
            raise RuntimeError(
                f"Truncation requires HuggingFace backend (current: {self.backend_name})"
            )

        target = max_tokens or self.max_tokens

        tokens = self.backend.encode(text)
        if len(tokens) <= target:
            return text

        truncated_tokens = tokens[:target]
        return self.backend.decode(truncated_tokens)


def create_tokenizer_service() -> TokenizerService:
    """
    Factory function to create TokenizerService.

    Reads TOKENIZER_BACKEND from environment and initializes appropriate backend.

    Returns:
        TokenizerService instance

    Raises:
        ValueError: If backend is invalid
        RuntimeError: If initialization fails
    """
    return TokenizerService()
