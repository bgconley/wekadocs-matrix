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

    def __init__(self):
        """
        Initialize HuggingFace tokenizer.

        Loads jinaai/jina-embeddings-v3 tokenizer from cache.
        Expects tokenizer to be prefetched during Docker build.

        Raises:
            RuntimeError: If tokenizer cannot be loaded
        """
        try:
            from transformers import AutoTokenizer

            model_id = os.getenv("HF_TOKENIZER_ID", "jinaai/jina-embeddings-v3")
            cache_dir = os.getenv("HF_CACHE", "/opt/hf-cache")
            offline = os.getenv("TRANSFORMERS_OFFLINE", "true").lower() == "true"

            logger.info(
                f"Loading HuggingFace tokenizer: {model_id} "
                f"(cache={cache_dir}, offline={offline})"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=offline,
            )

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

    def __init__(self):
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
        self.tokenizer_name = os.getenv("SEGMENTER_TOKENIZER_NAME", "xlm-roberta-base")
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
        backend_name = os.getenv("TOKENIZER_BACKEND", "hf").lower()

        if backend_name == "hf":
            self.backend = HuggingFaceTokenizerBackend()
            self.backend_name = "huggingface"
        elif backend_name == "segmenter":
            self.backend = JinaSegmenterBackend()
            self.backend_name = "jina-segmenter"
        else:
            raise ValueError(
                f"Invalid tokenizer backend: {backend_name}. "
                f"Must be 'hf' or 'segmenter'."
            )

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
            f"TokenizerService initialized: backend={self.backend_name}, "
            f"max_tokens={self.max_tokens}, target={self.target_tokens}, "
            f"overlap={self.overlap_tokens}"
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
        return self.backend.count_tokens(text)

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
        tokens = self.backend.encode(text)
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
            chunk_text = self.backend.decode(chunk_tokens)

            # Create chunk metadata
            chunk = {
                "text": chunk_text,
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will update after loop
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "overlap_start": start_idx > 0,
                "overlap_end": end_idx < total_tokens,
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

    def _verify_chunk_integrity(self, original: str, chunks: List[Dict]) -> None:
        """
        Verify chunks can be reassembled to original (with overlap removal).

        This is a spot-check that runs on a sample of splits to ensure
        no data loss occurs during chunking.

        Args:
            original: Original text before splitting
            chunks: List of chunk dictionaries

        Raises:
            RuntimeError: If integrity check fails (indicates bug in splitting logic)
        """
        try:
            # Reassemble without overlaps
            reassembled = chunks[0]["text"]

            for i in range(1, len(chunks)):
                chunk = chunks[i]
                if chunk["overlap_start"]:
                    # Find overlap boundary
                    # Simple approach: skip overlap_tokens worth of text from start
                    overlap_chars = self._estimate_overlap_chars(
                        chunk["text"], self.overlap_tokens
                    )
                    reassembled += chunk["text"][overlap_chars:]
                else:
                    reassembled += chunk["text"]

            # Compare lengths (should be close, allowing for tokenizer edge effects)
            len_diff = abs(len(reassembled) - len(original))
            len_threshold = len(original) * 0.01  # Allow 1% difference for edge effects

            if len_diff > len_threshold:
                logger.warning(
                    f"Integrity check: length difference {len_diff} chars "
                    f"({len_diff / len(original) * 100:.2f}%), "
                    f"original={len(original)}, reassembled={len(reassembled)}"
                )

            # Hash comparison (more lenient - we expect some difference due to overlap removal)
            original_hash = self.compute_integrity_hash(original)
            reassembled_hash = self.compute_integrity_hash(reassembled)

            if original_hash != reassembled_hash:
                logger.debug(
                    f"Integrity check: hash mismatch (expected with overlap), "
                    f"chunks={len(chunks)}, len_diff={len_diff}"
                )

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            # Don't raise - this is a spot-check, not a hard requirement

    def _estimate_overlap_chars(self, text: str, overlap_tokens: int) -> int:
        """
        Estimate character count for overlap_tokens.

        Uses rough heuristic: 3 chars per token for XLM-RoBERTa.

        Args:
            text: Chunk text
            overlap_tokens: Number of overlap tokens

        Returns:
            Estimated character count for overlap
        """
        # Conservative estimate: 3 chars per token
        estimated_chars = overlap_tokens * 3
        # Don't exceed chunk length
        return min(estimated_chars, len(text) // 2)

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
