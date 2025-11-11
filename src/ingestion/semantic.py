"""Semantic enrichment provider abstractions for chunk post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.shared.config import SemanticEnrichmentConfig


@dataclass
class SemanticEnrichmentResult:
    metadata: Dict
    error: str | None = None


class SemanticEnricher:
    """Base interface for semantic enrichment providers."""

    def __init__(self, config: SemanticEnrichmentConfig):
        self.config = config

    def enrich(self, chunk: Dict) -> SemanticEnrichmentResult:
        raise NotImplementedError


class StubSemanticEnricher(SemanticEnricher):
    """No-op enricher used until a real provider is wired."""

    def enrich(self, chunk: Dict) -> SemanticEnrichmentResult:
        return SemanticEnrichmentResult(
            metadata={"entities": [], "topics": [], "summary": None}
        )


PROVIDERS = {
    "stub": StubSemanticEnricher,
}


def get_semantic_enricher(config: SemanticEnrichmentConfig) -> SemanticEnricher:
    provider_cls = PROVIDERS.get(config.provider.lower())
    if not provider_cls:
        provider_cls = StubSemanticEnricher
    return provider_cls(config)
