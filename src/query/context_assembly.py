"""
Phase 7E-2: Context Assembly and Budget Enforcement
Integrates with hybrid retrieval to assemble coherent context within token budget

Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import ChunkResult
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)


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
        self.tokenizer = tokenizer or TokenizerService()

        # Load configuration
        config = get_config()
        self.max_tokens = getattr(
            config.search.response, "answer_context_max_tokens", 4500
        )
        self.include_citations = getattr(
            config.search.response, "include_citations", True
        )

        logger.info(f"ContextAssembler initialized: max_tokens={self.max_tokens}")

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
                        total_tokens = self.max_tokens
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
            group_metrics[parent_id] = (richest_citations, best_score)

        # Sort groups by citation richness first, then best score
        ordered = sorted(
            grouped.items(),
            key=lambda item: group_metrics[item[0]],
            reverse=True,
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

        # Add document separator if switching documents
        if current_document and first_chunk.document_id != current_document:
            parts.append("\n---\n")

        # Add section heading if available and different from current
        if first_chunk.heading and first_chunk.parent_section_id != current_parent:
            # Determine heading level based on chunk level
            heading_prefix = "#" * min(first_chunk.level + 1, 4)  # Cap at ####
            # Don't add leading newline for the very first heading
            if current_parent is None:
                parts.append(f"{heading_prefix} {first_chunk.heading}\n")
            else:
                parts.append(f"\n{heading_prefix} {first_chunk.heading}\n")

        # Add chunk texts
        for chunk in chunks:
            parts.append(chunk.text)

            # Add expansion indicator if configured
            if self.include_citations and chunk.is_expanded:
                parts.append(f" [↪ expanded from {chunk.expansion_source}]")

            # Add chunk boundary indicator for split chunks
            if chunk.is_split and chunk != chunks[-1]:
                parts.append(" [...] ")

        return "\n".join(parts)

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
        included = []
        parts = []

        # Try to include chunks one by one
        for i, chunk in enumerate(chunks):
            # Format just this chunk
            chunk_group = chunks[: i + 1]
            text = self._format_group(chunk_group, current_document, current_parent)
            tokens = self.tokenizer.count_tokens(text)

            if tokens <= budget:
                # Can fit this chunk
                included = chunk_group
                parts = [text]
            else:
                # Can't fit any more
                break

        if included:
            return "\n".join(parts), included
        else:
            return "", []

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
            labels = getattr(chunk, "citation_labels", None) or []
            lines_emitted = 0

            if labels:
                labels = sorted(labels, key=lambda x: (x[0], x[1]))
                seen_titles = set()
                for order_val, title in labels:
                    cleaned_title = title or (chunk.heading or "Section")
                    key = (order_val, cleaned_title)
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    parts.append(
                        f"[{citation_index}] {cleaned_title} "
                        f"(Doc: {chunk.document_id[:20]}..., "
                        f"Tokens: {chunk.token_count})"
                    )
                    if chunk.is_expanded and lines_emitted == 0:
                        parts.append(" [expanded]")
                    parts.append("\n")
                    citation_index += 1
                    lines_emitted += 1

            if lines_emitted == 0:
                fallback_title = chunk.heading or "Section"
                parts.append(
                    f"[{citation_index}] {fallback_title} "
                    f"(Doc: {chunk.document_id[:20]}..., "
                    f"Tokens: {chunk.token_count})"
                )
                if chunk.is_expanded:
                    parts.append(" [expanded]")
                parts.append("\n")
                citation_index += 1

        return "".join(parts)
