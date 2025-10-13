"""
Response Builder (Task 2.4)
Generates Markdown + JSON responses with evidence and confidence.
See: /docs/spec.md §5 (Responses & explainability)
See: /docs/pseudocode-reference.md Phase 2, Task 2.4
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.query.ranking import RankedResult


@dataclass
class Evidence:
    """Evidence supporting an answer."""

    section_id: Optional[str] = None
    node_id: Optional[str] = None
    node_label: Optional[str] = None
    snippet: Optional[str] = None
    path: Optional[List[str]] = None
    confidence: float = 1.0


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

    def build_response(
        self,
        query: str,
        intent: str,
        ranked_results: List[RankedResult],
        timing: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Build complete response from ranked results.

        Args:
            query: Original query text
            intent: Classified intent
            ranked_results: Ranked search results
            timing: Timing information
            filters: Optional filters applied

        Returns:
            Response with Markdown and JSON
        """
        # Extract top evidence
        evidence = self._extract_evidence(ranked_results[:5])

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

        return Response(answer_markdown=markdown, answer_json=structured)

    def _extract_evidence(self, top_results: List[RankedResult]) -> List[Evidence]:
        """Extract evidence from top results."""
        evidence_list = []

        for ranked in top_results:
            result = ranked.result

            # Determine section_id and node_id
            section_id = None
            node_id = result.node_id

            if result.node_label == "Section":
                section_id = result.node_id
            else:
                # For non-sections, try to get section from metadata
                section_id = result.metadata.get("section_id")

            # Extract snippet
            snippet = self._extract_snippet(result.metadata)

            evidence_list.append(
                Evidence(
                    section_id=section_id,
                    node_id=node_id,
                    node_label=result.node_label,
                    snippet=snippet,
                    path=result.path,
                    confidence=ranked.features.semantic_score,
                )
            )

        return evidence_list

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
) -> Response:
    """Convenience function to build a response."""
    builder = ResponseBuilder()
    return builder.build_response(query, intent, ranked_results, timing, filters)
