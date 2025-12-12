# src/ingestion/semantic_chunker.py
"""
Semantic-first chunking using Chonkie.

This module implements the SemanticChunkerAssembler which replaces greedy combination
with semantic boundary detection, creating smaller coherent chunks that align with
research best practices (Anthropic, Pinecone, NVIDIA, Chroma 2024-2025).

Key differences from GreedyCombinerV2:
- No forced minimum token count (allows 100+ token chunks)
- Semantic boundary detection using BGE-M3 embeddings
- Preserves heading structure as hard boundaries
- Target ~400 tokens (research optimal) instead of 500+

Design Philosophy:
- Semantic coherence over token count
- Let embeddings handle context (GLiNER + multi-vector compensates)
- Respect structural boundaries (headings as hard stops)

See: docs/plans/chonkie_semantic_chunking_integration_plan.md
See: docs/plans/chunking_architecture_analysis.md
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.shared.chunk_utils import create_chunk_metadata
from src.shared.config import ChunkAssemblyConfig, SemanticChunkingConfig
from src.shared.logging import get_logger
from src.shared.section_metadata import (
    FIELD_BLOCK_TYPE,
    FIELD_BLOCK_TYPES,
    FIELD_CODE_RATIO,
    FIELD_HAS_CODE,
    FIELD_HAS_TABLE,
    FIELD_LINE_END,
    FIELD_LINE_START,
    FIELD_PARENT_PATH,
    FIELD_PARENT_PATH_DEPTH,
    extract_enhanced_metadata,
)

log = get_logger(__name__)

# Attempt to import chonkie for semantic chunking
try:
    from chonkie import SemanticChunker

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    SemanticChunker = None  # type: ignore
    log.warning("chonkie not installed; SemanticChunkerAssembler will use fallback")

# Attempt to import RecursiveCharacterTextSplitter for fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore
    log.debug(
        "langchain-text-splitters not installed; fallback may create oversized chunks"
    )


def _text_hash(text: str) -> str:
    """Generate SHA-256 hash of text for deduplication."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shingle_hash(text: str, n: int = 8) -> str:
    """Generate shingle hash for near-duplicate detection."""
    if not text:
        return ""
    tokens = text.split()
    if not tokens:
        return ""
    shingles = []
    limit = 64
    for i in range(0, max(0, len(tokens) - n + 1)):
        shingles.append(" ".join(tokens[i : i + n]))
        if len(shingles) >= limit:
            break
    if not shingles:
        shingles = [" ".join(tokens)]
    return hashlib.sha256("||".join(shingles).encode("utf-8")).hexdigest()


class SemanticChunkerAssembler:
    """
    Semantic-first chunk assembler using Chonkie.

    Implements the ChunkAssembler protocol but uses semantic boundary
    detection instead of greedy token-based combination.

    Design principles (from research):
    1. Allow naturally small chunks if semantically coherent
    2. Never merge across heading boundaries
    3. Target ~400 tokens (not forced minimum)
    4. Let downstream pipeline (GLiNER, multi-vector) handle context

    Attributes:
        config: SemanticChunkingConfig with threshold, token limits, etc.
        _chunker: Chonkie SemanticChunker instance (or None if unavailable)
        _adapter: BGE-M3 embedding adapter for Chonkie
        _tokenizer: TokenizerService for token counting
    """

    def __init__(self, assembly_config: Optional[ChunkAssemblyConfig] = None):
        """
        Initialize the semantic chunker.

        Args:
            assembly_config: Optional ChunkAssemblyConfig containing semantic_chunking settings.
                            If None, uses default SemanticChunkingConfig.
        """
        self.config = (
            assembly_config.semantic_chunking
            if assembly_config
            else SemanticChunkingConfig()
        )
        self._chunker: Optional[Any] = None
        self._adapter: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        # Embedding-model input window guard.
        # BGE-M3 hard max is 8192 tokens; we enforce a "safe" limit to avoid
        # off-by-some differences across tokenizers/special tokens.
        try:
            self._embed_max_input_tokens = int(
                os.getenv("BGE_M3_MAX_INPUT_TOKENS", "8192")
            )
        except Exception:
            self._embed_max_input_tokens = 8192

        env_safe = os.getenv("BGE_M3_SAFE_INPUT_TOKENS")
        try:
            self._embed_safe_input_tokens = int(env_safe) if env_safe else 8000
        except Exception:
            self._embed_safe_input_tokens = 8000

        if self._embed_max_input_tokens < 256:
            self._embed_max_input_tokens = 8192
        if self._embed_safe_input_tokens > self._embed_max_input_tokens:
            self._embed_safe_input_tokens = max(1, self._embed_max_input_tokens - 1)

        # If safe limit isn't explicitly configured, keep it at or below the
        # batching cap to avoid oversize single-input requests.
        try:
            max_batch_tokens = int(os.getenv("BGE_M3_MAX_BATCH_TOKENS", "7500"))
        except Exception:
            max_batch_tokens = 7500
        if env_safe is None and self._embed_safe_input_tokens > max_batch_tokens:
            log.info(
                "Clamping embed safe input tokens to batch cap",
                extra={
                    "previous_safe_input_tokens": self._embed_safe_input_tokens,
                    "max_batch_tokens": max_batch_tokens,
                },
            )
            self._embed_safe_input_tokens = max_batch_tokens

        # Overlap when we pre-split oversized sections into guard chunks.
        try:
            self._guard_overlap_tokens = int(
                os.getenv("BGE_M3_GUARD_OVERLAP_TOKENS", "200")
            )
        except Exception:
            self._guard_overlap_tokens = 200

        # Overlap for fallback fixed-size chunking (when semantic chunking fails).
        try:
            self._fallback_overlap_tokens = int(
                os.getenv("SEMANTIC_FALLBACK_OVERLAP_TOKENS", "80")
            )
        except Exception:
            self._fallback_overlap_tokens = 80

        # Allow environment variable override
        force_semantic = os.getenv("FORCE_SEMANTIC_CHUNKING", "").lower()
        if force_semantic == "true":
            self.config = SemanticChunkingConfig(enabled=True)
            log.info("Semantic chunking force-enabled via FORCE_SEMANTIC_CHUNKING")
        elif force_semantic == "false":
            self.config = SemanticChunkingConfig(enabled=False)
            log.info("Semantic chunking force-disabled via FORCE_SEMANTIC_CHUNKING")

        if not CHONKIE_AVAILABLE:
            log.warning(
                "SemanticChunkerAssembler: chonkie unavailable, will use fallback"
            )
            return

        if not self.config.enabled:
            log.info("SemanticChunkerAssembler: disabled in config")
            return

        self._initialize_chunker()

    def _initialize_chunker(self):
        """Initialize Chonkie with BGE-M3 adapter."""
        from src.providers.embeddings.chonkie_adapter import BgeM3ChonkieAdapter

        # Check if BGE-M3 service is available
        if not BgeM3ChonkieAdapter.is_available():
            log.warning(
                "BGE-M3 service unavailable; semantic chunking disabled",
                extra={
                    "service_url": os.getenv("BGE_M3_API_URL", "http://127.0.0.1:9000")
                },
            )
            return

        try:
            self._adapter = BgeM3ChonkieAdapter()
            self._chunker = SemanticChunker(
                embedding_model=self._adapter,
                threshold=self.config.similarity_threshold,
                chunk_size=self.config.target_tokens,
            )
            log.info(
                "SemanticChunkerAssembler initialized",
                extra={
                    "similarity_threshold": self.config.similarity_threshold,
                    "target_tokens": self.config.target_tokens,
                    "min_tokens": self.config.min_tokens,
                    "max_tokens": self.config.max_tokens,
                    "skip_code_blocks": self.config.skip_code_blocks,
                },
            )
        except Exception as e:
            log.error(
                "Failed to initialize Chonkie SemanticChunker",
                extra={"error": str(e)},
                exc_info=True,
            )
            self._chunker = None
            self._adapter = None

    def _get_tokenizer(self):
        """Lazy tokenizer initialization."""
        if self._tokenizer is None:
            from src.providers.tokenizer_service import TokenizerService

            self._tokenizer = TokenizerService()
        return self._tokenizer

    def _get_hf_tokenizer(self) -> Optional[Any]:
        """Best-effort access to an underlying HF tokenizer (encode/decode).

        We prefer the tokenizer that the embedding adapter exposes (should match
        BGE-M3). If unavailable, we attempt to extract it from TokenizerService.
        """
        if self._adapter and hasattr(self._adapter, "get_tokenizer"):
            try:
                tok = self._adapter.get_tokenizer()
                if hasattr(tok, "encode") and hasattr(tok, "decode"):
                    return tok
            except Exception:
                pass

        try:
            tokenizer_service = self._get_tokenizer()
            backend = getattr(tokenizer_service, "backend", None)
            tok = getattr(backend, "tokenizer", None) if backend else None
            if tok and hasattr(tok, "encode") and hasattr(tok, "decode"):
                return tok
        except Exception:
            pass

        return None

    def _count_tokens_model(self, text: str) -> int:
        """Token count aligned with the embedding model where possible."""
        tok = self._get_hf_tokenizer()
        if tok is not None:
            try:
                return len(tok.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tok.encode(text))
        return self._get_tokenizer().count_tokens(text)

    def _split_text_by_tokens(
        self, text: str, *, max_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """Hard split text into <=max_tokens windows using token boundaries."""
        if not text:
            return []
        if max_tokens <= 0:
            return [text]

        tok = self._get_hf_tokenizer()
        if tok is None:
            # Fall back to a char-based splitter if HF tokenizer unavailable.
            # This is less exact but still prevents catastrophic oversize when
            # combined with conservative sizes.
            if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_tokens * 4,
                    chunk_overlap=max(0, min(200, overlap_tokens)) * 4,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                return [t for t in splitter.split_text(text) if t.strip()]
            return [text]

        try:
            ids = tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tok.encode(text)

        if len(ids) <= max_tokens:
            return [text]

        overlap = max(0, min(overlap_tokens, max_tokens - 1))
        pieces: List[str] = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunk_text = tok.decode(ids[start:end], skip_special_tokens=True)
            if chunk_text.strip():
                pieces.append(chunk_text)
            if end >= len(ids):
                break
            start = max(0, end - overlap)

        return pieces

    @staticmethod
    def _ensure_continuation_heading(heading: str) -> str:
        """Avoid duplicating '(cont.)' suffix when nesting splits."""
        h = (heading or "").strip()
        if not h:
            return ""
        return h if h.endswith("(cont.)") else f"{h} (cont.)"

    def _merge_tiny_chunks(
        self, chunks: List[Tuple[str, int]], *, min_tokens: int
    ) -> List[Tuple[str, int]]:
        """Merge, never drop, sub-minimum chunks.

        Chonkie can emit small tail fragments. Dropping them loses content.
        We instead merge small fragments into adjacent chunks.
        """
        if not chunks:
            return []
        if min_tokens <= 0:
            return chunks

        merged: List[Tuple[str, int]] = []
        cur_text = ""
        cur_tokens = 0

        for text, tc in chunks:
            if not text.strip():
                continue

            if not cur_text:
                cur_text, cur_tokens = text, tc
                continue

            # If either side is too small, merge.
            if cur_tokens < min_tokens or tc < min_tokens:
                cur_text = f"{cur_text}\n\n{text}" if text else cur_text
                cur_tokens += tc
                continue

            merged.append((cur_text, cur_tokens))
            cur_text, cur_tokens = text, tc

        if cur_text:
            merged.append((cur_text, cur_tokens))

        # If the last chunk ended up tiny, merge it backwards.
        if len(merged) >= 2 and merged[-1][1] < min_tokens:
            prev_text, prev_tokens = merged[-2]
            last_text, last_tokens = merged[-1]
            merged[-2] = (f"{prev_text}\n\n{last_text}", prev_tokens + last_tokens)
            merged.pop()

        return merged

    def _fallback_split_to_target(self, text: str) -> List[Tuple[str, int]]:
        """Fallback chunking that always returns multiple chunks when needed."""
        if not text.strip():
            return []

        target = max(1, int(self.config.target_tokens))
        # Always cap by the embedding model safe limit (hard safety guard).
        max_per_chunk = max(1, min(target, int(self._embed_safe_input_tokens) - 8))

        tc = self._count_tokens_model(text)
        if tc <= max_per_chunk:
            return [(text, tc)]

        pieces = self._split_text_by_tokens(
            text,
            max_tokens=max_per_chunk,
            overlap_tokens=self._fallback_overlap_tokens,
        )
        out: List[Tuple[str, int]] = []
        for p in pieces:
            out.append((p, self._count_tokens_model(p)))
        return out

    def assemble(self, document_id: str, sections: List[Dict]) -> List[Dict]:
        """
        Assemble sections into semantically coherent chunks.

        Unlike GreedyCombinerV2, this:
        - Does NOT force combination to reach min_tokens
        - Detects semantic boundaries within sections
        - Preserves heading boundaries as hard splits

        Args:
            document_id: The document identifier
            sections: List of section dicts from parse_markdown()

        Returns:
            List of chunk dicts ready for embedding
        """
        if not sections:
            return []

        # Use fallback if semantic chunking not available
        if not self._chunker:
            log.debug(
                "Chonkie unavailable; using fallback chunking",
                extra={"document_id": document_id, "sections": len(sections)},
            )
            return self._fallback_section_chunks(document_id, sections)

        chunks: List[Dict] = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                document_id=document_id,
                section=section,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Calculate statistics
        if chunks:
            avg_tokens = sum(c.get("token_count", 0) for c in chunks) / len(chunks)
            min_tokens = min(c.get("token_count", 0) for c in chunks)
            max_tokens = max(c.get("token_count", 0) for c in chunks)
        else:
            avg_tokens = min_tokens = max_tokens = 0

        log.info(
            "semantic_chunking_complete",
            extra={
                "document_id": document_id,
                "input_sections": len(sections),
                "output_chunks": len(chunks),
                "avg_tokens": round(avg_tokens, 1),
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
            },
        )

        return chunks

    def _chunk_section(
        self,
        document_id: str,
        section: Dict,
        start_index: int,
    ) -> List[Dict]:
        """
        Chunk a single section using semantic boundary detection.

        Headings are preserved as metadata in each resulting chunk.

        CONSENSUS REFINEMENT: Includes code block detection to skip semantic
        splitting for code-heavy sections (o3 concern about technical docs).

        Args:
            document_id: Document identifier
            section: Section dict with body, heading, level, etc.
            start_index: Starting chunk index for this section

        Returns:
            List of chunk dicts for this section
        """
        text = section.get("body", section.get("text", ""))
        heading = section.get("heading", section.get("title", ""))
        level = section.get("level", 2)

        if not text.strip():
            return []

        # Determine whether we should prepend heading context for boundary detection.
        heading_prefix = ""
        if self.config.heading_context_in_chunks and heading:
            heading_prefix = f"## {heading}\n\n"

        # CONSENSUS REFINEMENT: Skip semantic chunking for code-heavy sections,
        # but still enforce size constraints (never emit an oversized single chunk).
        if self.config.skip_code_blocks:
            code_ratio = self._calculate_code_block_ratio(text)
            if code_ratio > 0.5:  # Section is >50% code
                log.debug(
                    "skipping_semantic_for_code_block",
                    extra={
                        "heading": heading[:50] if heading else "(none)",
                        "code_ratio": round(code_ratio, 2),
                    },
                )
                pieces = self._fallback_split_to_target(text)
                if not pieces:
                    return []
                total = len(pieces)
                out: List[Dict] = []
                for i, (p_text, p_tc) in enumerate(pieces):
                    out.append(
                        self._create_chunk(
                            document_id=document_id,
                            text=p_text,
                            heading=(
                                heading
                                if i == 0
                                else self._ensure_continuation_heading(heading)
                            ),
                            level=level,
                            index=start_index + i,
                            section=section,
                            semantic_index=i,
                            semantic_total=total,
                            token_count=p_tc,
                        )
                    )
                return out

        # Hard guard: never call semantic boundary detection on text that would
        # exceed the embed model's input window.
        prefix_tokens = (
            self._count_tokens_model(heading_prefix) if heading_prefix else 0
        )
        body_tokens = self._count_tokens_model(text)
        total_for_semantic = body_tokens + prefix_tokens

        if total_for_semantic > self._embed_safe_input_tokens:
            # Pre-split oversized sections into "guard chunks" that are guaranteed
            # to fit in the embedding model context window. Then apply semantic chunking
            # within each guard chunk.
            max_body_tokens = max(1, self._embed_safe_input_tokens - prefix_tokens - 8)
            guard_pieces = self._split_text_by_tokens(
                text,
                max_tokens=max_body_tokens,
                overlap_tokens=self._guard_overlap_tokens,
            )

            log.info(
                "oversized_section_guard_split",
                extra={
                    "heading": heading[:50] if heading else "(none)",
                    "body_tokens": body_tokens,
                    "safe_input_tokens": self._embed_safe_input_tokens,
                    "guard_pieces": len(guard_pieces),
                },
            )

            all_pairs: List[Tuple[str, int]] = []
            for gi, piece in enumerate(guard_pieces):
                piece_for_semantic = (
                    f"{heading_prefix}{piece}" if heading_prefix else piece
                )
                try:
                    piece_semantic = self._chunker.chunk(piece_for_semantic)
                except Exception as e:
                    log.warning(
                        "semantic_chunking_failed_guard_piece",
                        extra={
                            "heading": heading[:50] if heading else "(none)",
                            "guard_index": gi,
                            "error": str(e),
                        },
                    )
                    piece_semantic = None

                if piece_semantic:
                    for sc in piece_semantic:
                        sc_text = sc.text if hasattr(sc, "text") else str(sc)
                        if heading_prefix and sc_text.startswith(heading_prefix):
                            sc_text = sc_text[len(heading_prefix) :]
                        sc_tc = (
                            sc.token_count
                            if hasattr(sc, "token_count")
                            else self._count_tokens_model(sc_text)
                        )
                        all_pairs.append((sc_text, int(sc_tc)))
                else:
                    # If semantic chunking fails for a guard piece, fall back to
                    # size-enforced token chunking (never emit an oversized chunk).
                    all_pairs.extend(self._fallback_split_to_target(piece))

            pairs = self._merge_tiny_chunks(
                all_pairs, min_tokens=int(self.config.min_tokens)
            )
            if not pairs:
                pairs = self._fallback_split_to_target(text)

        else:
            # Normal sized: apply semantic chunking directly.
            chunk_text = f"{heading_prefix}{text}" if heading_prefix else text
            try:
                semantic_chunks = self._chunker.chunk(chunk_text)
            except Exception as e:
                log.warning(
                    "semantic_chunking_failed",
                    extra={
                        "heading": heading[:50] if heading else "(none)",
                        "error": str(e),
                    },
                )
                semantic_chunks = None

            if not semantic_chunks:
                pairs = self._fallback_split_to_target(text)
            else:
                raw_pairs: List[Tuple[str, int]] = []
                for sc in semantic_chunks:
                    sc_text = sc.text if hasattr(sc, "text") else str(sc)
                    if heading_prefix and sc_text.startswith(heading_prefix):
                        sc_text = sc_text[len(heading_prefix) :]
                    sc_tc = (
                        sc.token_count
                        if hasattr(sc, "token_count")
                        else self._count_tokens_model(sc_text)
                    )
                    raw_pairs.append((sc_text, int(sc_tc)))

                pairs = self._merge_tiny_chunks(
                    raw_pairs, min_tokens=int(self.config.min_tokens)
                )
                if not pairs:
                    pairs = self._fallback_split_to_target(text)

        # Final safety: ensure no returned chunk exceeds the embedding safe limit.
        final_pairs: List[Tuple[str, int]] = []
        for p_text, p_tc in pairs:
            if p_tc <= self._embed_safe_input_tokens:
                final_pairs.append((p_text, p_tc))
                continue

            # Should be rare; split again conservatively.
            re_pieces = self._split_text_by_tokens(
                p_text,
                max_tokens=max(1, self._embed_safe_input_tokens - 8),
                overlap_tokens=self._fallback_overlap_tokens,
            )
            for rp in re_pieces:
                final_pairs.append((rp, self._count_tokens_model(rp)))

        if not final_pairs:
            return []

        total = len(final_pairs)
        result: List[Dict] = []
        for i, (sc_text, sc_tc) in enumerate(final_pairs):
            chunk = self._create_chunk(
                document_id=document_id,
                text=sc_text,
                heading=(
                    heading if i == 0 else self._ensure_continuation_heading(heading)
                ),
                level=level,
                index=start_index + i,
                section=section,
                semantic_index=i,
                semantic_total=total,
                token_count=sc_tc,
            )
            result.append(chunk)

        return result

    def _create_chunk(
        self,
        document_id: str,
        text: str,
        heading: str,
        level: int,
        index: int,
        section: Dict,
        semantic_index: int,
        semantic_total: int,
        token_count: Optional[int] = None,
    ) -> Dict:
        """
        Create a chunk dict with all required metadata.

        Uses create_chunk_metadata() for consistency with existing pipeline.

        Args:
            document_id: Document identifier
            text: Chunk text content
            heading: Section heading
            level: Heading level (1-6)
            index: Global chunk index
            section: Original section dict
            semantic_index: Index within semantic split (0-based)
            semantic_total: Total chunks from semantic split
            token_count: Pre-calculated token count (or None to calculate)

        Returns:
            Chunk dict ready for embedding pipeline
        """
        # Calculate token count if not provided
        if token_count is None:
            token_count = self._get_tokenizer().count_tokens(text)

        # Generate deterministic chunk ID using content hash
        content_hash = hashlib.sha256(
            f"{document_id}:{heading}:{semantic_index}:{text[:100]}".encode()
        ).hexdigest()[:16]

        # Get section metadata
        section_id = section.get("id", f"{document_id}_sec_{index}")
        order = int(section.get("order", index))
        parent_section_id = section.get("parent_section_id")
        doc_id = section.get("doc_id") or document_id
        doc_tag = section.get("doc_tag")
        tenant = section.get("tenant")
        lang = section.get("lang")
        version = section.get("version")

        # Build boundaries for provenance tracking
        boundaries = {
            "combined": False,
            "semantic_split": semantic_total > 1,
            "semantic_split_index": semantic_index,
            "semantic_split_total": semantic_total,
            "sections": [
                {
                    "id": section_id,
                    "order": order,
                    "level": level,
                    "tokens": token_count,
                    "title": heading,
                }
            ],
        }

        # Extract enhanced metadata from section (Phase 2: markdown-it-py)
        enhanced = extract_enhanced_metadata(section)

        # Calculate approximate line numbers for semantic splits
        line_start = enhanced.get(FIELD_LINE_START)
        line_end = enhanced.get(FIELD_LINE_END)
        if line_start is not None and line_end is not None and semantic_total > 1:
            total_lines = max(1, line_end - line_start)
            lines_per_chunk = total_lines / semantic_total
            chunk_line_start = int(line_start + (semantic_index * lines_per_chunk))
            chunk_line_end = int(line_start + ((semantic_index + 1) * lines_per_chunk))
        else:
            chunk_line_start = line_start
            chunk_line_end = line_end

        # Create metadata using existing helper
        meta = create_chunk_metadata(
            section_id=section_id,
            document_id=document_id,
            level=level,
            order=order * 1000 + semantic_index,  # Preserve order within splits
            heading=heading,
            parent_section_id=parent_section_id,
            is_combined=False,
            is_split=semantic_total > 1,
            boundaries_json=json.dumps(boundaries, separators=(",", ":")),
            token_count=token_count,
            doc_id=doc_id,
            doc_tag=doc_tag,
            tenant=tenant,
            lang=lang,
            version=version,
            text_hash=_text_hash(text),
            shingle_hash=_shingle_hash(text),
            # Phase 2: markdown-it-py enhanced metadata
            line_start=chunk_line_start,
            line_end=chunk_line_end,
            parent_path=enhanced.get(FIELD_PARENT_PATH, ""),
            block_types=enhanced.get(FIELD_BLOCK_TYPES, []),
            code_ratio=enhanced.get(FIELD_CODE_RATIO, 0.0),
            has_code=enhanced.get(FIELD_HAS_CODE, False),
            has_table=enhanced.get(FIELD_HAS_TABLE, False),
            # Phase 5: Derived structural fields
            parent_path_depth=enhanced.get(FIELD_PARENT_PATH_DEPTH, 0),
            block_type=enhanced.get(FIELD_BLOCK_TYPE, "paragraph"),
        )

        # Override ID with our semantic-aware ID
        meta["id"] = f"{document_id}_chunk_{index}_{content_hash}"

        # Add text and additional metadata
        meta["text"] = text
        meta["tokens"] = token_count
        meta["token_count"] = token_count
        meta["checksum"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta["anchor"] = section.get("anchor", "")
        meta["doc_tag"] = doc_tag

        # Semantic split provenance
        meta["is_semantic_split"] = semantic_total > 1
        meta["semantic_split_index"] = semantic_index
        meta["semantic_split_total"] = semantic_total

        # Section provenance
        meta["section_index"] = section.get("index", index)

        return meta

    def _calculate_code_block_ratio(self, text: str) -> float:
        """
        CONSENSUS REFINEMENT: Calculate what fraction of text is code blocks.

        Used to skip semantic chunking for code-heavy sections where
        semantic similarity is unreliable (o3 concern about technical docs).

        Args:
            text: Section text content

        Returns:
            Float between 0 and 1 representing code block ratio
        """
        if not text:
            return 0.0

        # Match fenced code blocks (```...```)
        pattern = self.config.code_block_pattern
        try:
            matches = re.findall(pattern, text)
        except re.error:
            log.warning(
                "Invalid code_block_pattern regex",
                extra={"pattern": pattern},
            )
            return 0.0

        if not matches:
            return 0.0

        code_chars = sum(len(m) for m in matches)
        return code_chars / len(text)

    def _fallback_section_chunks(
        self,
        document_id: str,
        sections: List[Dict],
    ) -> List[Dict]:
        """
        CONSENSUS REFINEMENT: Fallback with size enforcement.

        Both o3 and Gemini 3 Pro identified that a naive fallback could create
        oversized chunks. This fallback enforces strict token caps (aligned to
        the embedding model safe limit) and splits toward the configured target
        chunk size.

        Args:
            document_id: Document identifier
            sections: List of sections to chunk

        Returns:
            List of chunks with size enforcement
        """
        # This fallback is used when Chonkie semantic chunking is unavailable.
        # We still enforce strict size limits so the embedding stage never sees
        # an oversized chunk.
        chunks: List[Dict] = []
        chunk_index = 0

        for section in sections:
            text = section.get("body", section.get("text", ""))
            if not text.strip():
                continue

            heading = section.get("heading", section.get("title", ""))
            level = section.get("level", 2)

            pieces = self._fallback_split_to_target(text)
            if not pieces:
                continue

            total = len(pieces)
            for i, (p_text, p_tc) in enumerate(pieces):
                chunks.append(
                    self._create_chunk(
                        document_id=document_id,
                        text=p_text,
                        heading=(
                            heading
                            if i == 0
                            else self._ensure_continuation_heading(heading)
                        ),
                        level=level,
                        index=chunk_index,
                        section=section,
                        semantic_index=i,
                        semantic_total=total,
                        token_count=p_tc,
                    )
                )
                chunk_index += 1

        log.info(
            "fallback_chunking_complete",
            extra={
                "document_id": document_id,
                "input_sections": len(sections),
                "output_chunks": len(chunks),
            },
        )
        return chunks

    def close(self):
        """Release resources."""
        if self._adapter and hasattr(self._adapter, "close"):
            self._adapter.close()
        self._chunker = None
        self._adapter = None
