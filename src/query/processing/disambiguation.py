"""
Query disambiguation using GLiNER entity extraction.

Phase 4 of GLiNER integration: Extract entities from user queries
to enable post-retrieval boosting of chunks with matching entities.

Reference: gliner_rag_implementation_plan_gemini_mods_apple.md (v1.7)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """Result of query entity disambiguation."""

    query: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    boost_terms: List[str] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    enabled: bool = False

    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return bool(self.boost_terms)


class QueryDisambiguator:
    """
    Extracts entities from user queries using GLiNER for entity-aware retrieval.

    This class provides query-time entity extraction to enable:
    1. Post-retrieval boosting based on entity matches
    2. Over-fetching candidates when entities are found
    3. Logging and observability of query entity extraction

    Usage:
        disambiguator = QueryDisambiguator()
        analysis = disambiguator.process("How do I configure NFS on RHEL?")
        if analysis.has_entities:
            # Over-fetch and apply boosting
            boost_terms = analysis.boost_terms  # ["nfs", "rhel"]
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Initialize the query disambiguator.

        Args:
            labels: Entity labels to extract (uses config default if None)
            threshold: Confidence threshold for extraction (uses config default if None)
        """
        self.config = get_config()
        self._service: Optional[GLiNERService] = None
        self._labels = labels
        self._threshold = threshold
        self._enabled = getattr(self.config.ner, "enabled", False)

    @property
    def labels(self) -> List[str]:
        """Get entity labels, preferring explicit config over defaults."""
        if self._labels:
            return self._labels
        return get_default_labels()

    @property
    def threshold(self) -> float:
        """Get confidence threshold from config or override."""
        if self._threshold is not None:
            return self._threshold
        return getattr(self.config.ner, "threshold", 0.45)

    def _get_service(self) -> GLiNERService:
        """Lazily initialize the GLiNER service."""
        if self._service is None:
            self._service = GLiNERService()
        return self._service

    def process(self, query: str) -> QueryAnalysis:
        """
        Extract entities from a query for retrieval boosting.

        Args:
            query: User query text to analyze

        Returns:
            QueryAnalysis with extracted entities and boost terms
        """
        # Early return if NER disabled
        if not self._enabled:
            return QueryAnalysis(query=query, enabled=False)

        try:
            service = self._get_service()

            # Check if service is available
            if not service.is_available:
                logger.debug("GLiNER service not available for query disambiguation")
                return QueryAnalysis(query=query, enabled=True)

            # Extract entities from query
            entities = service.extract_entities(
                text=query,
                labels=self.labels,
                threshold=self.threshold,
            )

            if not entities:
                logger.debug(
                    "query_disambiguation_complete",
                    query=query[:50],
                    entities_found=0,
                )
                return QueryAnalysis(query=query, enabled=True)

            # Build boost terms (normalized for case-insensitive matching)
            boost_terms = list(
                set(e.text.lower().strip() for e in entities if e.text.strip())
            )
            entity_types = list(set(e.label for e in entities))

            # Convert Entity objects to dicts for serialization
            entity_dicts = [
                {
                    "text": e.text,
                    "label": e.label,
                    "score": e.score,
                    "start": e.start,
                    "end": e.end,
                }
                for e in entities
            ]

            logger.info(
                "query_disambiguation_complete",
                query=query[:50],
                entities_found=len(entities),
                boost_terms=boost_terms,
                entity_types=entity_types,
            )

            return QueryAnalysis(
                query=query,
                entities=entity_dicts,
                boost_terms=boost_terms,
                entity_types=entity_types,
                enabled=True,
            )

        except Exception as e:
            logger.warning(
                "query_disambiguation_failed",
                query=query[:50],
                error=str(e),
            )
            return QueryAnalysis(query=query, enabled=True)

    def is_enabled(self) -> bool:
        """Check if disambiguation is enabled."""
        return self._enabled


def get_query_disambiguator() -> QueryDisambiguator:
    """Factory function for QueryDisambiguator singleton-like access."""
    return QueryDisambiguator()
