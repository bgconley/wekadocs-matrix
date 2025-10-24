"""
Response Builder (Task 2.4)
Generates Markdown + JSON responses with evidence and confidence.
See: /docs/spec.md §5 (Responses & explainability)
See: /docs/pseudocode-reference.md Phase 2, Task 2.4

Enhanced Response Features (E1-E7):
- Verbosity modes: full (text only), graph (text + relationships, default)
- Graph mode provides complete context for better LLM reasoning
- Supports multi-turn exploration via traverse_relationships
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from src.query.ranking import RankedResult


class Verbosity(str, Enum):
    """Response verbosity levels for search results."""

    FULL = "full"  # Complete section text only (32KB limit, faster)
    GRAPH = "graph"  # Full text + related entities (default, better answers)


@dataclass
class Evidence:
    """Evidence supporting an answer."""

    section_id: Optional[str] = None
    node_id: Optional[str] = None
    node_label: Optional[str] = None
    snippet: Optional[str] = None
    path: Optional[List[str]] = None
    confidence: float = 1.0

    # Enhanced fields for full and graph modes (E1)
    title: Optional[str] = None
    full_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    related_entities: Optional[List[Dict[str, Any]]] = None
    related_sections: Optional[List[Dict[str, Any]]] = None


@dataclass
class Diagnostics:
    """Diagnostic information about result ranking."""

    ranking_features: Dict[str, Any]
    timing: Dict[str, float]
    total_candidates: int
    filters_applied: Optional[Dict[str, Any]] = None


@dataclass
class StructuredResponse:
    """Structured JSON response."""

    answer: str
    evidence: List[Evidence]
    confidence: float
    diagnostics: Diagnostics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "evidence": [asdict(e) for e in self.evidence],
            "confidence": self.confidence,
            "diagnostics": asdict(self.diagnostics),
        }


@dataclass
class Response:
    """Complete response with Markdown and JSON."""

    answer_markdown: str
    answer_json: StructuredResponse

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer_markdown": self.answer_markdown,
            "answer_json": self.answer_json.to_dict(),
        }


class ResponseBuilder:
    """
    Builds dual-format responses (Markdown + JSON) from ranked results.
    """

    def __init__(self, neo4j_driver=None, max_text_bytes: int = 32768):
        """
        Initialize ResponseBuilder.

        Args:
            neo4j_driver: Optional Neo4j driver for graph mode queries (E3)
            max_text_bytes: Maximum bytes for full text truncation (E1, default: 32KB)
        """
        self.neo4j_driver = neo4j_driver
        self.max_text_bytes = max_text_bytes

    def _truncate_to_bytes(self, text: str, max_bytes: Optional[int] = None) -> str:
        """
        Truncate text to max bytes (UTF-8) without breaking multi-byte sequences.

        Pre-Phase 7 (E1): Byte-accurate truncation for FULL mode 32KB cap.
        Uses UTF-8 encoding and safe decoding to avoid splitting multi-byte chars.

        Args:
            text: Text to truncate
            max_bytes: Maximum bytes (defaults to self.max_text_bytes)

        Returns:
            Truncated text with ellipsis if truncated
        """
        if max_bytes is None:
            max_bytes = self.max_text_bytes

        # Encode to UTF-8 bytes
        text_bytes = text.encode("utf-8")

        if len(text_bytes) <= max_bytes:
            return text

        # Truncate at byte boundary
        truncated_bytes = text_bytes[:max_bytes]

        # Decode with error handling (ignores incomplete multi-byte sequences)
        truncated_text = truncated_bytes.decode("utf-8", errors="ignore")

        return truncated_text + "...[truncated]"

    def format_citations(
        self, results: List[RankedResult], max_citations: int = 5
    ) -> str:
        """
        Format citations from ranked results as Markdown.

        Pre-Phase 7 (E3): Citation scaffolding - currently unused.
        Will be integrated in Phase 7 for answer provenance tracking.

        Args:
            results: Ranked search results to cite
            max_citations: Maximum citations to include (default: 5)

        Returns:
            Markdown-formatted citation block

        Example output:
            [1] Installation Prerequisites (prerequisites.md#network-requirements)
            [2] Network Configuration Guide (network-config.md#interface-setup)
        """
        if not results:
            return ""

        lines = []
        for i, ranked in enumerate(results[:max_citations], 1):
            metadata = ranked.result.metadata

            # Extract citation components
            title = metadata.get("title") or metadata.get("name") or "Untitled"
            document_uri = metadata.get("document_uri") or metadata.get(
                "source_uri", ""
            )
            anchor = metadata.get("anchor", "")

            # Format citation
            if document_uri:
                if anchor:
                    citation = f"[{i}] {title} ({document_uri}#{anchor})"
                else:
                    citation = f"[{i}] {title} ({document_uri})"
            else:
                citation = f"[{i}] {title}"

            lines.append(citation)

        return "\n".join(lines)

    def build_response(
        self,
        query: str,
        intent: str,
        ranked_results: List[RankedResult],
        timing: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None,
        verbosity: Verbosity = Verbosity.GRAPH,
    ) -> Response:
        # Pre-Phase 7 (G1): Metrics instrumentation for response builder
        import time

        from src.shared.observability.metrics import (
            response_builder_evidence_count,
            response_builder_latency_ms,
        )

        start_time = time.time()
        """
        Build complete response from ranked results.

        Args:
            query: Original query text
            intent: Classified intent
            ranked_results: Ranked search results
            timing: Timing information
            filters: Optional filters applied
            verbosity: Response detail level (full=text only, graph=text+relationships, default=graph)

        Returns:
            Response with Markdown and JSON
        """
        # Extract top evidence with verbosity mode
        evidence = self._extract_evidence(ranked_results[:5], verbosity)

        # Estimate confidence
        confidence = self._estimate_confidence(ranked_results, evidence)

        # Build diagnostics
        diagnostics = self._build_diagnostics(ranked_results, timing, filters)

        # Render Markdown
        markdown = self._render_markdown(
            query, intent, ranked_results, evidence, confidence
        )

        # Build structured answer
        answer_text = self._extract_answer_text(intent, ranked_results)

        structured = StructuredResponse(
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            diagnostics=diagnostics,
        )

        # Record metrics
        latency = (time.time() - start_time) * 1000
        response_builder_latency_ms.labels(verbosity=verbosity.value).observe(latency)
        response_builder_evidence_count.labels(verbosity=verbosity.value).observe(
            len(evidence)
        )

        return Response(answer_markdown=markdown, answer_json=structured)

    def _extract_evidence(
        self, top_results: List[RankedResult], verbosity: Verbosity = Verbosity.GRAPH
    ) -> List[Evidence]:
        """
        Extract evidence from top results based on verbosity mode.

        Args:
            top_results: Top ranked results
            verbosity: Response detail level (full or graph, default=graph)

        Returns:
            List of Evidence with fields populated based on verbosity
        """
        evidence_list = []

        for ranked in top_results:
            result = ranked.result
            metadata = result.metadata

            # Determine section_id and node_id
            section_id = None
            node_id = result.node_id

            if result.node_label == "Section":
                section_id = result.node_id
            else:
                # For non-sections, try to get section from metadata
                section_id = metadata.get("section_id")

            # Mode-specific evidence extraction
            if verbosity == Verbosity.FULL:
                # Full mode: complete section text (32KB byte limit)
                # Pre-Phase 7 (E1): Byte-accurate UTF-8 truncation
                full_text, node_title, tokens = self._fetch_full_text_from_neo4j(
                    node_id
                )

                # Apply byte cap (32KB)
                full_text = self._truncate_to_bytes(full_text, max_bytes=32768)

                evidence_list.append(
                    Evidence(
                        section_id=section_id,
                        node_id=node_id,
                        node_label=result.node_label,
                        snippet=self._extract_snippet(metadata, max_length=200),
                        title=node_title or metadata.get("title"),
                        full_text=full_text,
                        metadata={
                            "document_id": metadata.get("document_id"),
                            "level": metadata.get("level"),
                            "anchor": metadata.get("anchor"),
                            "tokens": tokens,
                        },
                        path=result.path,
                        confidence=ranked.features.semantic_score,
                    )
                )

            elif verbosity == Verbosity.GRAPH:
                # Graph mode: full text + related entities
                # Pre-Phase 7 (E1, E2): Byte-accurate truncation + safety caps
                full_text, node_title, tokens = self._fetch_full_text_from_neo4j(
                    node_id
                )

                # Apply byte cap (32KB)
                full_text = self._truncate_to_bytes(full_text, max_bytes=32768)

                # Get related entities from Neo4j with safety caps (E2)
                # Max 20 related entities per seed (prevents explosion)
                related = (
                    self._get_related_entities(node_id, max_entities=20)
                    if self.neo4j_driver
                    else {
                        "entities": [],
                        "sections": [],
                    }
                )

                evidence_list.append(
                    Evidence(
                        section_id=section_id,
                        node_id=node_id,
                        node_label=result.node_label,
                        snippet=self._extract_snippet(metadata, max_length=200),
                        title=node_title or metadata.get("title"),
                        full_text=full_text,
                        metadata={
                            "document_id": metadata.get("document_id"),
                            "level": metadata.get("level"),
                            "anchor": metadata.get("anchor"),
                            "tokens": tokens,
                        },
                        related_entities=related["entities"],
                        related_sections=related["sections"],
                        path=result.path,
                        confidence=ranked.features.semantic_score,
                    )
                )

        return evidence_list

    def _get_related_entities(
        self, node_id: str, max_entities: int = 20, max_depth: int = 1
    ) -> Dict[str, List]:
        """
        Fetch related entities and sections from Neo4j (E2, E3 implementation).

        Pre-Phase 7 (E2): Safety caps prevent graph explosion
        - max_entities: Limit entities per seed (default: 20)
        - max_depth: Limit traversal depth (default: 1, currently enforced)

        Query: 1-hop neighbors via MENTIONS, CONTAINS_STEP, REQUIRES, AFFECTS.
        Filter: Only entity labels (Command, Configuration, Step, Error, Concept).

        Args:
            node_id: Starting node ID
            max_entities: Maximum entities to return (default: 20, E2 cap)
            max_depth: Maximum traversal depth (default: 1, E2 cap - future expansion)

        Returns:
            Dict with "entities" and "sections" lists
        """
        from src.shared.observability import get_logger

        logger = get_logger(__name__)
        logger.info(
            f"_get_related_entities called for node_id={node_id[:20]}..., neo4j_driver={self.neo4j_driver is not None}"
        )

        if not self.neo4j_driver:
            logger.warning("Neo4j driver is None, returning empty relationships")
            return {"entities": [], "sections": []}

        try:
            # Parameterized Cypher query for 1-hop neighbors (bi-directional)
            # Pre-Phase 7 (E2): max_depth=1 enforced (prevents deep traversal)
            # Why bi-directional: Sections may have incoming HAS_SECTION from Documents
            query = """
            MATCH (n {id: $node_id})-[r:MENTIONS|CONTAINS_STEP|REQUIRES|AFFECTS]-(e)
            WHERE labels(e)[0] IN ['Command', 'Configuration', 'Step', 'Error', 'Concept']
            RETURN DISTINCT
                e.id AS entity_id,
                labels(e)[0] AS label,
                e.name AS name,
                type(r) AS relationship,
                COALESCE(r.confidence, 1.0) AS confidence
            ORDER BY confidence DESC
            LIMIT $max_entities
            """

            entities = []
            with self.neo4j_driver.session() as session:
                result = session.run(query, node_id=node_id, max_entities=max_entities)

                for record in result:
                    entities.append(
                        {
                            "entity_id": record["entity_id"],
                            "label": record["label"],
                            "name": record["name"],
                            "relationship": record["relationship"],
                            "confidence": record["confidence"],
                        }
                    )

            # Pre-Phase 7 (E2): Payload size early-stop (defensive guard)
            # Check if accumulated entities exceed safe payload size
            MAX_PAYLOAD_BYTES = 50000  # 50KB safety limit (defensive guard)
            payload_size = sum(len(str(entity)) for entity in entities)

            if payload_size > MAX_PAYLOAD_BYTES:
                logger.warning(
                    f"Payload size {payload_size} bytes exceeds limit {MAX_PAYLOAD_BYTES}, "
                    f"truncating from {len(entities)} entities"
                )
                # Truncate to approximately half to get under limit
                entities = entities[: len(entities) // 2]
                logger.info(f"Truncated to {len(entities)} entities")

            # TODO: Add related sections query (optional enhancement)
            sections = []

            return {"entities": entities, "sections": sections}

        except Exception as e:
            # Log error but don't fail the entire response
            from src.shared.observability import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to fetch related entities for {node_id}: {e}")
            return {"entities": [], "sections": []}

    def _fetch_full_text_from_neo4j(self, node_id: str) -> tuple[str, str, int]:
        """
        Fetch full text, title, and token count from Neo4j for a given node.

        Args:
            node_id: The node ID to fetch

        Returns:
            Tuple of (full_text, title, tokens)
        """
        if not self.neo4j_driver:
            return ("", "", 0)

        try:
            query = """
            MATCH (n {id: $node_id})
            RETURN n.text AS text, n.title AS title, n.tokens AS tokens
            LIMIT 1
            """

            with self.neo4j_driver.session() as session:
                result = session.run(query, node_id=node_id)
                record = result.single()

                if record:
                    text = record["text"] or ""
                    title = record["title"] or ""
                    tokens = record["tokens"] or 0
                    return (text, title, tokens)
                else:
                    return ("", "", 0)

        except Exception as e:
            from src.shared.observability import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to fetch full text for {node_id}: {e}")
            return ("", "", 0)

    def _extract_snippet(self, metadata: Dict[str, Any], max_length: int = 200) -> str:
        """Extract a text snippet from metadata."""
        # Try common text fields
        text = (
            metadata.get("text")
            or metadata.get("description")
            or metadata.get("title")
            or metadata.get("name")
            or ""
        )

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    def _estimate_confidence(
        self, ranked_results: List[RankedResult], evidence: List[Evidence]
    ) -> float:
        """
        Estimate overall confidence in the answer.

        Factors:
        - Top result score
        - Evidence strength (multiple high-confidence pieces)
        - Coverage (how well results cover query)
        - Path coherence (are results well-connected)
        """
        if not ranked_results:
            return 0.0

        # Top result semantic score (weight: 0.3)
        top_score = ranked_results[0].features.semantic_score if ranked_results else 0.0

        # Evidence strength: average confidence of top 3 (weight: 0.2)
        evidence_scores = [e.confidence for e in evidence[:3]]
        evidence_strength = (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        )

        # Coverage: how many high-confidence results (weight: 0.2)
        high_conf_count = sum(
            1 for r in ranked_results[:5] if r.features.semantic_score > 0.7
        )
        coverage = min(1.0, high_conf_count / 3.0)

        # Path coherence: results with paths are more coherent (weight: 0.3)
        path_count = sum(1 for r in ranked_results[:5] if r.result.path)
        path_coherence = (
            path_count / min(5, len(ranked_results)) if ranked_results else 0.0
        )

        # Weighted combination
        confidence = (
            0.3 * top_score
            + 0.2 * evidence_strength
            + 0.2 * coverage
            + 0.3 * path_coherence
        )

        return min(1.0, max(0.0, confidence))

    def _build_diagnostics(
        self,
        ranked_results: List[RankedResult],
        timing: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Diagnostics:
        """Build diagnostic information."""
        # Extract ranking features from top result
        ranking_features = {}
        if ranked_results:
            top_features = ranked_results[0].features
            ranking_features = {
                "semantic_score": top_features.semantic_score,
                "graph_distance_score": top_features.graph_distance_score,
                "recency_score": top_features.recency_score,
                "entity_priority_score": top_features.entity_priority_score,
                "coverage_score": top_features.coverage_score,
                "final_score": top_features.final_score,
            }

        return Diagnostics(
            ranking_features=ranking_features,
            timing=timing,
            total_candidates=len(ranked_results),
            filters_applied=filters,
        )

    def _extract_answer_text(
        self, intent: str, ranked_results: List[RankedResult]
    ) -> str:
        """Extract concise answer text based on intent."""
        if not ranked_results:
            return "No relevant information found."

        # Intent-specific answer extraction
        if intent == "search":
            return self._answer_search(ranked_results)
        elif intent == "troubleshoot":
            return self._answer_troubleshoot(ranked_results)
        elif intent == "compare":
            return self._answer_compare(ranked_results)
        elif intent == "explain":
            return self._answer_explain(ranked_results)
        elif intent == "traverse":
            return self._answer_traverse(ranked_results)
        else:
            return self._answer_default(ranked_results)

    def _answer_search(self, results: List[RankedResult]) -> str:
        """Answer for search intent."""
        count = min(5, len(results))
        return f"Found {len(results)} relevant sections. Top {count} results show information about this topic."

    def _answer_troubleshoot(self, results: List[RankedResult]) -> str:
        """Answer for troubleshoot intent."""
        # Look for procedures and steps in results
        procedures = [r for r in results if r.result.node_label == "Procedure"]
        if procedures:
            return f"Found {len(procedures)} resolution procedures. Follow the steps in the linked sections."
        return "Found related documentation. Review the sections for troubleshooting guidance."

    def _answer_compare(self, results: List[RankedResult]) -> str:
        """Answer for compare intent."""
        return f"Comparison found {len(results)} relevant nodes showing differences and similarities."

    def _answer_explain(self, results: List[RankedResult]) -> str:
        """Answer for explain intent."""
        return f"Explanation includes {len(results)} related concepts and components."

    def _answer_traverse(self, results: List[RankedResult]) -> str:
        """Answer for traverse intent."""
        return f"Traversal found {len(results)} connected nodes across relationships."

    def _answer_default(self, results: List[RankedResult]) -> str:
        """Default answer."""
        return f"Found {len(results)} relevant results."

    def _render_markdown(
        self,
        query: str,
        intent: str,
        ranked_results: List[RankedResult],
        evidence: List[Evidence],
        confidence: float,
    ) -> str:
        """Render human-readable Markdown response."""
        lines = []

        # Header
        lines.append("# Query Results\n")
        lines.append(f"**Query:** {query}\n")
        lines.append(f"**Intent:** {intent}\n")
        lines.append(f"**Confidence:** {confidence:.2f}\n")

        # Answer summary
        answer = self._extract_answer_text(intent, ranked_results)
        lines.append("## Answer\n")
        lines.append(f"{answer}\n")

        # Evidence
        if evidence:
            lines.append(f"## Evidence ({len(evidence)} sources)\n")
            for i, ev in enumerate(evidence[:5], 1):
                lines.append(f"### {i}. {ev.node_label}: {ev.node_id}\n")
                if ev.snippet:
                    lines.append(f"> {ev.snippet}\n")
                if ev.section_id:
                    lines.append(f"**Section:** `{ev.section_id}`\n")
                if ev.path:
                    lines.append(f"**Path:** {' → '.join(ev.path)}\n")
                lines.append(f"**Confidence:** {ev.confidence:.2f}\n")

        # Top results table
        lines.append("## Top Results\n")
        lines.append("| Rank | Type | ID | Score |\n")
        lines.append("|------|------|-------|-------|\n")
        for r in ranked_results[:10]:
            lines.append(
                f"| {r.rank} | {r.result.node_label} | "
                f"`{r.result.node_id[:20]}...` | {r.features.final_score:.3f} |\n"
            )

        # Why these results?
        lines.append("\n## Why These Results?\n")
        if ranked_results:
            top = ranked_results[0]
            lines.append("Top result selected based on:\n")
            lines.append(
                f"- **Semantic similarity:** {top.features.semantic_score:.2f}\n"
            )
            lines.append(
                f"- **Graph proximity:** {top.features.graph_distance_score:.2f}\n"
            )
            lines.append(f"- **Recency:** {top.features.recency_score:.2f}\n")
            lines.append(
                f"- **Entity priority:** {top.features.entity_priority_score:.2f}\n"
            )
            lines.append(f"- **Coverage:** {top.features.coverage_score:.2f}\n")

        return "".join(lines)


def build_response(
    query: str,
    intent: str,
    ranked_results: List[RankedResult],
    timing: Dict[str, float],
    filters: Optional[Dict[str, Any]] = None,
    verbosity: Verbosity = Verbosity.GRAPH,
    neo4j_driver=None,
) -> Response:
    """
    Convenience function to build a response.

    Args:
        query: Original query text
        intent: Classified intent
        ranked_results: Ranked search results
        timing: Timing information
        filters: Optional filters applied
        verbosity: Response detail level (full=text only, graph=text+relationships, default=graph)
        neo4j_driver: Optional Neo4j driver for graph mode

    Returns:
        Response with Markdown and JSON
    """
    builder = ResponseBuilder(neo4j_driver=neo4j_driver)
    return builder.build_response(
        query, intent, ranked_results, timing, filters, verbosity
    )
