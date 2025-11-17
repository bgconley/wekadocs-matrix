"""
Phase 7E-2: Context Assembly and Budget Enforcement
Integrates with hybrid retrieval to assemble coherent context within token budget

Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import ChunkResult
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)

_STEP_RE = re.compile(r"^\s*step\s*\d+", re.I)


def _normalize_and_order_citations(chunk) -> List[Tuple[int, str]]:
    """
    Accepts chunk.citation_labels as:
      - list[(order:int, title:str)] or
      - list[(order:int, title:str, level:int)]
    Returns a normalized, ordered list of (order, title).
    """
    labels = getattr(chunk, "citation_labels", None) or []

    # 1) Normalize to triples (order, title, level)
    norm: List[Tuple[int, str, int]] = []
    for item in labels:
        if not item:
            continue
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            order = int(item[0] or 0)
            title = (item[1] or "").strip()
            level = int(item[2]) if len(item) > 2 and item[2] is not None else 0
            if title:
                norm.append((order, title, level))

    if not norm:
        return []

    # 2) Case-insensitive dedupe by (order, title)
    seen = set()
    dedup: List[Tuple[int, str, int]] = []
    for order_val, title_val, level_val in norm:
        key = (order_val, title_val.casefold())
        if key in seen:
            continue
        seen.add(key)
        dedup.append((order_val, title_val, level_val))

    # 3) If >1 labels, drop the top-level document heading (order==0) that equals chunk.heading
    heading_ci = ((getattr(chunk, "heading", "") or "").strip()).casefold()
    if len(dedup) > 1:
        filtered = [
            (order_val, title_val, level_val)
            for (order_val, title_val, level_val) in dedup
            if order_val != 0
        ]
        if filtered:
            dedup = filtered
        elif len(dedup) > 1 and heading_ci:
            filtered = [
                (order_val, title_val, level_val)
                for (order_val, title_val, level_val) in dedup
                if not (order_val == 0 and title_val.strip().casefold() == heading_ci)
            ]
            if filtered:
                dedup = filtered

    # 4) Stable ordering: order asc, level asc (parents first), Step* bias, then lexical
    def _sort_key(item: Tuple[int, str, int]):
        order_val, title_val, level_val = item
        step_priority = 0 if _STEP_RE.match(title_val) else 1
        return (order_val, level_val, step_priority, title_val.casefold())

    dedup.sort(key=_sort_key)

    # 5) Return pairs
    return [(order_val, title_val) for (order_val, title_val, _level_val) in dedup]


@dataclass
class AssembledContext:
    """Context assembled from chunks with metadata."""

    text: str  # The assembled context text
    chunks: List[ChunkResult]  # Chunks included in context
    total_tokens: int  # Total token count
    truncated: bool  # Whether context was truncated for budget

    # Provenance tracking
    chunk_ids: List[str]  # IDs of chunks in order
    document_ids: List[str]  # Unique document IDs represented
    parent_section_ids: List[str]  # Unique parent sections represented

    # Statistics
    expansion_count: int = 0  # Number of expanded chunks included
    combined_count: int = 0  # Number of combined chunks
    split_count: int = 0  # Number of split chunks


class ContextAssembler:
    """
    Assembles coherent context from retrieved chunks with budget enforcement.

    Phase 7E-2 requirements:
    - Stitch chunks by (parent_section_id, order)
    - Enforce context budget (default 4500 tokens)
    - Preserve headings and structure
    - Track provenance for citations
    """

    def __init__(self, tokenizer: Optional[TokenizerService] = None):
        """
        Initialize context assembler.

        Args:
            tokenizer: Tokenizer service for accurate token counting
        """
        self.tokenizer = tokenizer or get_default_tokenizer()

        # Load configuration
        config = get_config()
        self.max_tokens = getattr(
            config.search.response, "answer_context_max_tokens", 4500
        )
        self.include_citations = getattr(
            config.search.response, "include_citations", True
        )

        logger.info(f"ContextAssembler initialized: max_tokens={self.max_tokens}")

        try:
            newline_cost = self.tokenizer.count_tokens("\n")
        except Exception:
            newline_cost = 1
        self._newline_token_cost = newline_cost if newline_cost > 0 else 1

    def assemble(
        self, chunks: List[ChunkResult], query: Optional[str] = None
    ) -> AssembledContext:
        """
        Assemble chunks into coherent context with budget enforcement.

        Args:
            chunks: List of chunks to assemble (already sorted by relevance)
            query: Optional query for context (used for logging)

        Returns:
            AssembledContext with assembled text and metadata
        """
        if not chunks:
            return AssembledContext(
                text="",
                chunks=[],
                total_tokens=0,
                truncated=False,
                chunk_ids=[],
                document_ids=[],
                parent_section_ids=[],
            )

        start_time = time.time()

        # Step 1: Group chunks by parent section for coherence
        grouped = self._group_by_parent(chunks)

        # Step 2: Order groups by best chunk score
        ordered_groups = self._order_groups(grouped)

        # Step 3: Assemble with budget enforcement
        included_chunks = []
        context_parts = []
        total_tokens = 0
        truncated = False

        current_document = None
        current_parent = None

        for parent_id, group_chunks in ordered_groups:
            # Sort chunks within group by order
            group_chunks.sort(key=lambda x: x.order)

            # Check if we can fit this group
            group_text = self._format_group(
                group_chunks, current_document, current_parent
            )
            group_tokens = self.tokenizer.count_tokens(group_text)

            if total_tokens + group_tokens <= self.max_tokens:
                # Include the entire group
                context_parts.append(group_text)
                included_chunks.extend(group_chunks)
                total_tokens += group_tokens

                # Update current context
                if group_chunks:
                    current_document = group_chunks[0].document_id
                    current_parent = parent_id
            else:
                # Try to include partial group
                remaining_budget = self.max_tokens - total_tokens
                if remaining_budget > 100:  # Only try if we have reasonable space
                    partial_text, partial_chunks = self._fit_partial_group(
                        group_chunks, remaining_budget, current_document, current_parent
                    )
                    if partial_text:
                        context_parts.append(partial_text)
                        included_chunks.extend(partial_chunks)
                        partial_tokens = self.tokenizer.count_tokens(partial_text)
                        total_tokens += partial_tokens
                        truncated = True
                        break
                else:
                    # No more room
                    truncated = True
                    break

        # Step 4: Build final context
        assembled_text = "\n".join(context_parts)

        # Step 5: Collect statistics
        chunk_ids = [c.chunk_id for c in included_chunks]
        document_ids = list(set(c.document_id for c in included_chunks))
        parent_section_ids = list(set(c.parent_section_id for c in included_chunks))

        expansion_count = sum(1 for c in included_chunks if c.is_expanded)
        combined_count = sum(1 for c in included_chunks if c.is_combined)
        split_count = sum(1 for c in included_chunks if c.is_split)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Context assembled: chunks={len(included_chunks)}, "
            f"tokens={total_tokens}/{self.max_tokens}, "
            f"truncated={truncated}, expanded={expansion_count}, "
            f"time={elapsed_ms:.2f}ms"
        )

        return AssembledContext(
            text=assembled_text,
            chunks=included_chunks,
            total_tokens=total_tokens,
            truncated=truncated,
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            parent_section_ids=parent_section_ids,
            expansion_count=expansion_count,
            combined_count=combined_count,
            split_count=split_count,
        )

    def _group_by_parent(
        self, chunks: List[ChunkResult]
    ) -> Dict[str, List[ChunkResult]]:
        """Group chunks by parent_section_id for coherence."""
        grouped = {}
        for chunk in chunks:
            parent = chunk.parent_section_id or chunk.chunk_id
            if getattr(chunk, "is_microdoc_extra", False):
                parent = f"microdoc:{chunk.chunk_id}"
            if parent not in grouped:
                grouped[parent] = []
            grouped[parent].append(chunk)
        return grouped

    def _order_groups(
        self, grouped: Dict[str, List[ChunkResult]]
    ) -> List[Tuple[str, List[ChunkResult]]]:
        """Order groups by best chunk score in each group."""
        # Calculate prioritization metrics per group
        group_metrics = {}
        for parent_id, chunks in grouped.items():
            best_score = max((c.fused_score or 0.0) for c in chunks)
            richest_citations = max(len(c.citation_labels or []) for c in chunks)
            min_order = min((c.order or 0) for c in chunks)
            group_metrics[parent_id] = {
                "best_score": best_score,
                "richest_citations": richest_citations,
                "min_order": min_order,
            }

        # Sort groups by document order first, then citation richness, then score
        ordered = sorted(
            grouped.items(),
            key=lambda item: (
                group_metrics[item[0]]["min_order"],
                -group_metrics[item[0]]["richest_citations"],
                -float(group_metrics[item[0]]["best_score"] or 0.0),
            ),
        )
        return ordered

    def _format_group(
        self,
        chunks: List[ChunkResult],
        current_document: Optional[str],
        current_parent: Optional[str],
    ) -> str:
        """
        Format a group of chunks with appropriate headers.

        Args:
            chunks: Chunks in the group (already sorted by order)
            current_document: Current document ID for context
            current_parent: Current parent section ID for context

        Returns:
            Formatted text for the group
        """
        if not chunks:
            return ""

        parts = []
        first_chunk = chunks[0]
        parts.extend(
            self._build_group_prefix_parts(
                first_chunk, current_document, current_parent
            )
        )

        for idx, chunk in enumerate(chunks):
            parts.extend(
                self._build_chunk_parts(chunk, is_last=(idx == len(chunks) - 1))
            )

        return "\n".join(parts)

    def _build_group_prefix_parts(
        self,
        first_chunk: ChunkResult,
        current_document: Optional[str],
        current_parent: Optional[str],
    ) -> List[str]:
        parts: List[str] = []
        if current_document and first_chunk.document_id != current_document:
            parts.append("\n---\n")

        is_microdoc_extra = getattr(first_chunk, "is_microdoc_extra", False)
        if first_chunk.heading and first_chunk.parent_section_id != current_parent:
            if is_microdoc_extra:
                display_heading = first_chunk.heading or "Related Document"
                doc_hint = (
                    first_chunk.document_id[:8] if first_chunk.document_id else ""
                )
                header_line = f"#### Related: {display_heading}"
                if doc_hint:
                    header_line += f"  \n*(Doc: {doc_hint})*"
                if current_parent is None:
                    parts.append(f"{header_line}\n")
                else:
                    parts.append(f"\n{header_line}\n")
            else:
                heading_prefix = "#" * min(first_chunk.level + 1, 4)
                if current_parent is None:
                    parts.append(f"{heading_prefix} {first_chunk.heading}\n")
                else:
                    parts.append(f"\n{heading_prefix} {first_chunk.heading}\n")
        return parts

    def _build_chunk_parts(self, chunk: ChunkResult, *, is_last: bool) -> List[str]:
        parts = [chunk.text or ""]
        if self.include_citations and getattr(chunk, "is_expanded", False):
            source = getattr(chunk, "expansion_source", "seed")
            parts.append(f" [↪ expanded from {source}]")
        if chunk.is_split and not is_last:
            parts.append(" [...] ")
        return [part for part in parts if part]

    def _fit_partial_group(
        self,
        chunks: List[ChunkResult],
        budget: int,
        current_document: Optional[str],
        current_parent: Optional[str],
    ) -> Tuple[str, List[ChunkResult]]:
        """
        Try to fit as many chunks from a group as possible within budget.

        Args:
            chunks: Chunks in the group (already sorted by order)
            budget: Remaining token budget
            current_document: Current document ID for context
            current_parent: Current parent section ID for context

        Returns:
            Tuple of (formatted text, included chunks)
        """
        if not chunks:
            return "", []

        included: List[ChunkResult] = []
        rendered_blocks: List[str] = []
        tokens_used = 0

        prefix_parts = self._build_group_prefix_parts(
            chunks[0], current_document, current_parent
        )
        prefix_text = self._join_parts(prefix_parts)
        if prefix_text:
            prefix_tokens = self.tokenizer.count_tokens(prefix_text)
            if prefix_tokens > budget:
                return "", []
            tokens_used += prefix_tokens
            rendered_blocks.append(prefix_text)

        for idx, chunk in enumerate(chunks):
            chunk_parts = self._build_chunk_parts(
                chunk, is_last=(idx == len(chunks) - 1)
            )
            chunk_text = self._join_parts(chunk_parts)
            if not chunk_text:
                continue

            chunk_tokens = self._estimate_chunk_tokens(
                chunk, chunk_parts, has_existing_blocks=bool(rendered_blocks)
            )
            if tokens_used + chunk_tokens > budget:
                break

            tokens_used += chunk_tokens
            rendered_blocks.append(chunk_text)
            included.append(chunk)

        if not included:
            return "", []

        return "\n".join(rendered_blocks), included

    @staticmethod
    def _join_parts(parts: List[str]) -> str:
        cleaned = [part for part in parts if part]
        return "\n".join(cleaned) if cleaned else ""

    def _estimate_chunk_tokens(
        self,
        chunk: ChunkResult,
        parts: List[str],
        *,
        has_existing_blocks: bool,
    ) -> int:
        """Estimate tokens for a chunk using cached metadata."""
        tokens = 0
        has_segments = has_existing_blocks
        for idx, part in enumerate(parts):
            if not part:
                continue
            if has_segments:
                tokens += self._newline_token_cost
            if idx == 0:
                chunk_tokens = int(chunk.token_count or 0)
                if chunk_tokens <= 0:
                    chunk_tokens = self.tokenizer.count_tokens(part)
                tokens += chunk_tokens
            else:
                tokens += self.tokenizer.count_tokens(part)
            has_segments = True
        return tokens

    def format_with_citations(self, context: AssembledContext) -> str:
        """
        Format assembled context with inline citations.

        Args:
            context: Assembled context with chunks

        Returns:
            Formatted text with citations
        """
        if not self.include_citations or not context.chunks:
            return context.text

        parts = [context.text]
        parts.append("\n\n---\n### Citations\n")

        citation_index = 1
        for chunk in context.chunks:
            labels = _normalize_and_order_citations(chunk)
            lines_emitted = 0

            for _order, title in labels:
                parts.append(f"[{citation_index}] {title}\n")
                citation_index += 1
                lines_emitted += 1

            if lines_emitted == 0:
                fallback = getattr(chunk, "heading", None) or "Section"
                if getattr(chunk, "is_microdoc_extra", False):
                    fallback = f"Related: {fallback}"
                parts.append(f"[{citation_index}] {fallback}\n")
                citation_index += 1

        return "".join(parts)


_DEFAULT_TOKENIZER: Optional[TokenizerService] = None


def get_default_tokenizer() -> TokenizerService:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = TokenizerService()
    return _DEFAULT_TOKENIZER
