"""GLiNER entity extraction for document ingestion enrichment.

Phase 2 of GLiNER integration: Document Ingestion Pipeline.
Enriches chunks with named entities extracted via GLiNER zero-shot NER.

The enrichment adds three fields to each chunk:
1. entity_metadata - Stored in Qdrant payload for filtering/boosting
2. _embedding_text - Transient field for entity-enriched embedding generation
3. _mentions - Appended with GLiNER entities (source="gliner") for entity-sparse

GLiNER entities are marked with source="gliner" to distinguish them from
structural entities (regex-extracted). This marker is used to filter them
from Neo4j MENTIONS edges (they only enrich vectors, not the graph).

See: /docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md
"""

import hashlib
from typing import Any, Dict, List

from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import extract_label_name, get_default_labels
from src.shared.observability import get_logger

logger = get_logger(__name__)


def enrich_chunks_with_entities(chunks: List[Dict[str, Any]]) -> None:
    """
    Enrich chunks with GLiNER entity extraction.

    Modifies chunks in-place to add:
    - entity_metadata: Stored in Qdrant payload for filtering/boosting
    - _embedding_text: Transient field for entity-enriched embedding generation
    - _mentions: Appended with GLiNER entities (source="gliner") for entity-sparse

    This function is designed to be non-blocking: if GLiNER is unavailable
    or fails, chunks pass through unchanged and ingestion continues.

    Args:
        chunks: List of chunk dicts to enrich (modified in-place)
    """
    if not chunks:
        return

    service = GLiNERService()

    # Check if service is available (circuit breaker not tripped)
    if not service.is_available:
        logger.warning(
            "gliner_service_unavailable_skipping_enrichment",
            reason="circuit_breaker_tripped_or_model_not_loaded",
        )
        return

    labels = get_default_labels()
    if not labels:
        logger.warning(
            "gliner_no_labels_configured_skipping_enrichment",
            hint="Check config.ner.labels in development.yaml",
        )
        return

    # Extract texts for batch processing
    texts = [chunk.get("text", "") for chunk in chunks]

    # Batch extract entities
    try:
        entities_list = service.batch_extract_entities(texts, labels)
    except Exception as e:
        logger.warning(
            "gliner_batch_extraction_failed",
            error=str(e),
            chunk_count=len(chunks),
        )
        return

    enriched_count = 0
    total_entities = 0
    entity_type_counts: Dict[str, int] = {}

    for chunk, entities in zip(chunks, entities_list):
        # Always set entity_metadata for consistent Qdrant payload schema
        if entities:
            # Clean labels (strip parenthetical examples like "(e.g. ...)")
            entity_types = list(set(extract_label_name(e.label) for e in entities))
            entity_values = [e.text for e in entities]
            entity_values_normalized = [e.text.lower().strip() for e in entities]

            chunk["entity_metadata"] = {
                "entity_types": entity_types,
                "entity_values": entity_values,
                "entity_values_normalized": entity_values_normalized,
                "entity_count": len(entities),
            }

            # Track entity type distribution for logging
            for etype in entity_types:
                entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

            # Build transient embedding text with entity context
            # Format: "{title}\n\n{text}\n\n[Context: type1: val1; type2: val2]"
            entity_context = "; ".join(
                f"{extract_label_name(e.label)}: {e.text}" for e in entities
            )
            base_text = chunk.get("text", "")
            title = chunk.get("title", "") or chunk.get("heading", "")

            if title:
                chunk["_embedding_text"] = (
                    f"{title}\n\n{base_text}\n\n[Context: {entity_context}]"
                )
            else:
                chunk["_embedding_text"] = f"{base_text}\n\n[Context: {entity_context}]"

            # Append to _mentions for entity-sparse vector generation
            # Deduplicate against existing mentions to prevent double-counting
            existing_mentions = chunk.get("_mentions", [])
            seen_keys = {
                (m.get("name", "").lower(), m.get("type", "").lower())
                for m in existing_mentions
            }

            new_mentions = []
            for e in entities:
                clean_label = extract_label_name(e.label)
                key = (e.text.lower(), clean_label.lower())
                if key not in seen_keys:
                    # Hash-based entity_id for deterministic identification
                    eid_hash = hashlib.sha256(
                        f"{e.text}:{clean_label}".encode()
                    ).hexdigest()[:16]
                    new_mentions.append(
                        {
                            "name": e.text,
                            "type": clean_label,
                            "entity_id": f"gliner:{clean_label}:{eid_hash}",
                            "source": "gliner",  # Marker for Neo4j filtering
                            "confidence": e.score,
                        }
                    )
                    seen_keys.add(key)

            chunk["_mentions"] = existing_mentions + new_mentions
            enriched_count += 1
            total_entities += len(entities)
        else:
            # Set empty entity_metadata for consistent Qdrant payload schema
            # This ensures all chunks have the same fields for indexing
            chunk["entity_metadata"] = {
                "entity_types": [],
                "entity_values": [],
                "entity_values_normalized": [],
                "entity_count": 0,
            }

    logger.info(
        "gliner_enrichment_complete",
        chunks_processed=len(chunks),
        chunks_enriched=enriched_count,
        total_entities=total_entities,
        entity_type_distribution=entity_type_counts,
    )
