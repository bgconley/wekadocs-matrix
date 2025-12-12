"""
NER (Named Entity Recognition) provider package.
GLiNER integration for zero-shot entity extraction.

This package provides domain-specific entity extraction for WEKA documentation
to enhance retrieval quality through entity-aware embeddings and boosting.
"""

from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels

__all__ = ["GLiNERService", "get_default_labels"]
