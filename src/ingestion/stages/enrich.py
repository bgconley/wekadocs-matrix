"""Entity, reference, and mention enrichment stages for atomic ingestion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.shared.observability import get_logger

logger = get_logger(__name__)


def extract_and_enrich(
    *,
    document: Dict[str, Any],
    sections: List[Dict[str, Any]],
    content: str,
    format: str,
    config,
    trace=None,
) -> Dict[str, Any]:
    """Extract structural entities and cross-document references."""
    _ = trace

    from src.ingestion.extract import extract_entities
    from src.ingestion.extract.references import (
        create_reference_edge,
        extract_chunk_references,
        extract_references,
    )

    entities, mentions = extract_entities(sections)

    raw_content_refs = []
    if format == "markdown" and content:
        doc_chunk_id = document["id"]
        raw_refs = extract_references(content, doc_chunk_id)

        for ref in raw_refs:
            if ref.reference_type == "hyperlink":
                edge = create_reference_edge(
                    source_chunk_id=doc_chunk_id,
                    target_doc_id=None,
                    target_hint=ref.target_hint,
                    reference_type=ref.reference_type,
                    reference_text=ref.reference_text,
                    confidence=ref.confidence,
                )
                raw_content_refs.append(edge)

        logger.debug(
            "hyperlinks_extracted_from_raw_markdown",
            hyperlink_count=len(raw_content_refs),
            document_id=document["id"],
        )

    references_cfg = getattr(config, "references", None)
    if references_cfg and getattr(references_cfg, "enabled", False):
        reference_edges, ref_resolved, ref_unresolved = extract_chunk_references(
            sections,
            known_doc_titles=None,
        )
    else:
        reference_edges, ref_resolved, ref_unresolved = [], 0, 0

    existing_hints = {edge.get("target_hint", "").lower() for edge in reference_edges}
    for edge in raw_content_refs:
        if edge.get("target_hint", "").lower() not in existing_hints:
            reference_edges.append(edge)
            existing_hints.add(edge.get("target_hint", "").lower())

    logger.debug(
        "references_extracted_from_sections",
        total_references=len(reference_edges),
        hyperlinks_from_raw=len(raw_content_refs),
        local_resolved=ref_resolved,
        pending_neo4j_resolution=ref_unresolved,
    )

    return {
        "entities": entities,
        "mentions": mentions,
        "references": reference_edges,
    }


def enrich_chunks_with_gliner(
    *,
    document: Dict[str, Any],
    sections: List[Dict[str, Any]],
    config,
    trace=None,
) -> None:
    """Apply optional GLiNER enrichment to assembled chunks."""
    _ = trace

    if getattr(config, "ner", None) and getattr(config.ner, "enabled", False):
        try:
            from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

            enrich_chunks_with_entities(sections)
        except Exception as e:
            logger.warning(
                "gliner_enrichment_failed_non_blocking",
                error=str(e),
                document_id=document.get("id"),
                section_count=len(sections),
            )


def merge_section_mentions(
    sections: List[Dict[str, Any]],
    entities: Dict[str, Dict[str, Any]],
    mentions: List[Dict[str, Any]],
) -> None:
    """Attach merged structural and GLiNER mentions to assembled sections."""
    from src.providers.ner.labels import is_excluded_structural_entity

    mentions_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        section_id = mention.get("section_id")
        if section_id:
            mentions_by_section[section_id].append(mention)

    for section in sections:
        section_mentions = []
        section_id = section.get("id")
        if section_id and section_id in mentions_by_section:
            section_mentions.extend(mentions_by_section[section_id])

        original_ids = section.get("original_section_ids", [])
        for orig_id in original_ids:
            if orig_id in mentions_by_section:
                section_mentions.extend(mentions_by_section[orig_id])

        existing_gliner_mentions = section.get("_mentions", [])
        seen_entity_ids = set()
        merged_mentions = []

        for mention in existing_gliner_mentions:
            entity_id = mention.get("entity_id")
            if entity_id and entity_id not in seen_entity_ids:
                seen_entity_ids.add(entity_id)
                merged_mentions.append(mention)

        for mention in section_mentions:
            entity_id = mention.get("entity_id")
            if not entity_id or entity_id in seen_entity_ids:
                continue

            entity_name = ""
            entity_data = (
                entities.get(entity_id) if isinstance(entities, dict) else None
            )
            if isinstance(entity_data, dict):
                entity_name = entity_data.get("name", "") or ""

            if is_excluded_structural_entity(entity_name):
                continue

            seen_entity_ids.add(entity_id)
            merged_mentions.append(mention)

        section["_mentions"] = merged_mentions
