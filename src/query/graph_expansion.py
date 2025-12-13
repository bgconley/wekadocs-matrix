"""
Graph-based context expansion for retrieval.

This module implements the bounded context expansion from the Neo4j Overhaul Plan
(Section 7.3). It adds structural context to vector search results without
letting expansion outrank the original seeds.

Key principles:
1. Expansion is context-only, not ranking material
2. Bounded to ±1 adjacency and 1 hop hierarchy
3. Preserves seed ordering, appends context as attachments
4. Safe defaults: no expansion if relationships don't exist

Usage:
    from src.query.graph_expansion import expand_context

    # After vector search returns candidates
    expanded = expand_context(
        session,
        chunk_ids=["chunk1", "chunk2"],
        include_parent=True,
        include_adjacent=True,
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExpandedChunk:
    """A chunk with its expansion context."""

    chunk_id: str
    is_seed: bool = True
    expansion_type: Optional[str] = None  # "parent", "prev", "next"
    parent_of: Optional[str] = None  # chunk_id this is parent of
    adjacent_to: Optional[str] = None  # chunk_id this is adjacent to
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionResult:
    """Result of context expansion."""

    seed_chunks: List[ExpandedChunk]
    context_chunks: List[ExpandedChunk]
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def all_chunks(self) -> List[ExpandedChunk]:
        """All chunks in expansion order (seeds first, then context)."""
        return self.seed_chunks + self.context_chunks

    @property
    def all_chunk_ids(self) -> List[str]:
        """All chunk IDs in expansion order."""
        return [c.chunk_id for c in self.all_chunks]


class ContextExpander:
    """
    Expands vector search results with graph-based context.

    Implements the conservative expansion policy from Section 7.3:
    - Include PARENT_HEADING (1 hop) if present
    - Include NEXT_CHUNK previous + next (bounded to ±1)
    - Do NOT add expansions back into the ranking pool

    The expansion is designed to provide structural context for LLM consumption
    without disrupting the ranking from vector search.
    """

    def __init__(self, session):
        """
        Initialize the context expander.

        Args:
            session: Neo4j session or driver
        """
        self.session = session

    def expand(
        self,
        chunk_ids: List[str],
        *,
        include_parent: bool = True,
        include_adjacent: bool = True,
        adjacent_window: int = 1,
        include_chunk_data: bool = True,
    ) -> ExpansionResult:
        """
        Expand chunk IDs with structural context.

        Args:
            chunk_ids: Seed chunk IDs from vector search
            include_parent: Include parent heading chunks
            include_adjacent: Include adjacent chunks (NEXT_CHUNK ±1)
            adjacent_window: How many adjacent chunks to include (default 1)
            include_chunk_data: Fetch full chunk data or just IDs

        Returns:
            ExpansionResult with seeds and context chunks
        """
        if not chunk_ids:
            return ExpansionResult(seed_chunks=[], context_chunks=[])

        seed_set = set(chunk_ids)
        context_chunks: List[ExpandedChunk] = []
        seen_context: Set[str] = set()

        # Collect parents
        if include_parent:
            parents = self._get_parent_chunks(chunk_ids)
            for parent in parents:
                if (
                    parent["chunk_id"] not in seed_set
                    and parent["chunk_id"] not in seen_context
                ):
                    seen_context.add(parent["chunk_id"])
                    context_chunks.append(
                        ExpandedChunk(
                            chunk_id=parent["chunk_id"],
                            is_seed=False,
                            expansion_type="parent",
                            parent_of=parent["child_chunk_id"],
                            data=parent.get("data", {}),
                        )
                    )

        # Collect adjacent chunks
        if include_adjacent:
            adjacent = self._get_adjacent_chunks(chunk_ids, window=adjacent_window)
            for adj in adjacent:
                if (
                    adj["chunk_id"] not in seed_set
                    and adj["chunk_id"] not in seen_context
                ):
                    seen_context.add(adj["chunk_id"])
                    context_chunks.append(
                        ExpandedChunk(
                            chunk_id=adj["chunk_id"],
                            is_seed=False,
                            expansion_type=adj["direction"],  # "prev" or "next"
                            adjacent_to=adj["seed_chunk_id"],
                            data=adj.get("data", {}),
                        )
                    )

        # Build seed chunks
        if include_chunk_data:
            seed_data = self._get_chunk_data(chunk_ids)
        else:
            seed_data = {cid: {} for cid in chunk_ids}

        seed_chunks = [
            ExpandedChunk(
                chunk_id=cid,
                is_seed=True,
                data=seed_data.get(cid, {}),
            )
            for cid in chunk_ids
        ]

        result = ExpansionResult(
            seed_chunks=seed_chunks,
            context_chunks=context_chunks,
            stats={
                "seed_count": len(seed_chunks),
                "parent_count": sum(
                    1 for c in context_chunks if c.expansion_type == "parent"
                ),
                "prev_count": sum(
                    1 for c in context_chunks if c.expansion_type == "prev"
                ),
                "next_count": sum(
                    1 for c in context_chunks if c.expansion_type == "next"
                ),
                "total_context": len(context_chunks),
            },
        )

        logger.debug(
            "context_expansion_complete",
            seed_count=len(chunk_ids),
            context_added=len(context_chunks),
            stats=result.stats,
        )

        return result

    def _get_parent_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get parent heading chunks for the given chunk IDs."""
        query = """
        UNWIND $chunk_ids AS cid
        MATCH (child:Chunk {chunk_id: cid})-[:PARENT_HEADING]->(parent:Chunk)
        RETURN DISTINCT
            parent.chunk_id AS chunk_id,
            cid AS child_chunk_id,
            parent {
                .chunk_id, .document_id, .heading, .level, .order,
                .text, .token_count
            } AS data
        """
        result = self.session.run(query, chunk_ids=chunk_ids)
        return [dict(record) for record in result]

    def _get_adjacent_chunks(
        self, chunk_ids: List[str], window: int = 1
    ) -> List[Dict[str, Any]]:
        """Get adjacent chunks (previous and next) for the given chunk IDs."""
        # Get previous chunks
        prev_query = """
        UNWIND $chunk_ids AS cid
        MATCH (seed:Chunk {chunk_id: cid})
        MATCH (prev:Chunk)-[:NEXT_CHUNK]->(seed)
        RETURN DISTINCT
            prev.chunk_id AS chunk_id,
            cid AS seed_chunk_id,
            'prev' AS direction,
            prev {
                .chunk_id, .document_id, .heading, .level, .order,
                .text, .token_count
            } AS data
        """
        prev_result = self.session.run(prev_query, chunk_ids=chunk_ids)
        prev_chunks = [dict(record) for record in prev_result]

        # Get next chunks
        next_query = """
        UNWIND $chunk_ids AS cid
        MATCH (seed:Chunk {chunk_id: cid})
        MATCH (seed)-[:NEXT_CHUNK]->(next:Chunk)
        RETURN DISTINCT
            next.chunk_id AS chunk_id,
            cid AS seed_chunk_id,
            'next' AS direction,
            next {
                .chunk_id, .document_id, .heading, .level, .order,
                .text, .token_count
            } AS data
        """
        next_result = self.session.run(next_query, chunk_ids=chunk_ids)
        next_chunks = [dict(record) for record in next_result]

        return prev_chunks + next_chunks

    def _get_chunk_data(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch chunk data for the given IDs."""
        query = """
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {chunk_id: cid})
        RETURN c.chunk_id AS chunk_id,
               c {
                   .chunk_id, .document_id, .heading, .level, .order,
                   .text, .token_count, .parent_chunk_id
               } AS data
        """
        result = self.session.run(query, chunk_ids=chunk_ids)
        return {record["chunk_id"]: dict(record["data"]) for record in result}


def expand_context(
    session,
    chunk_ids: List[str],
    *,
    include_parent: bool = True,
    include_adjacent: bool = True,
    adjacent_window: int = 1,
    include_chunk_data: bool = True,
) -> ExpansionResult:
    """
    Convenience function to expand context for chunk IDs.

    Args:
        session: Neo4j session
        chunk_ids: Seed chunk IDs from vector search
        include_parent: Include parent heading chunks
        include_adjacent: Include adjacent chunks
        adjacent_window: Adjacent chunk window size
        include_chunk_data: Fetch full chunk data

    Returns:
        ExpansionResult with seeds and context
    """
    expander = ContextExpander(session)
    return expander.expand(
        chunk_ids,
        include_parent=include_parent,
        include_adjacent=include_adjacent,
        adjacent_window=adjacent_window,
        include_chunk_data=include_chunk_data,
    )
