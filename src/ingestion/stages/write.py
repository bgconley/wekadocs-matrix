"""Atomic write saga stage for ingestion."""

from __future__ import annotations

from typing import Any, Dict, List

from src.ingestion.saga import SagaContext
from src.shared.observability import get_logger

logger = get_logger(__name__)


def execute_saga(
    *,
    saga_id: str,
    document: Dict[str, Any],
    sections: List[Dict[str, Any]],
    entities: Dict[str, Any],
    mentions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    embeddings: Dict[str, Any],
    builder,
    neo4j_driver,
    qdrant_client,
    neo4j_writer,
    qdrant_writer,
    config,
    trace=None,
    structural_edges_builder=None,
    cross_doc_linker=None,
) -> Dict[str, Any]:
    """Execute Neo4j and Qdrant writes with deferred Neo4j commit."""
    _ = trace
    document_id = document["id"]
    context = SagaContext(saga_id=saga_id, document_id=document_id)

    written_neo4j_chunks: List[str] = []
    written_qdrant_points: List[str] = []
    neo4j_tx = None
    neo4j_session = None

    stats = {
        "sections_upserted": 0,
        "entities_upserted": 0,
        "vectors_upserted": 0,
    }

    try:
        qdrant_required = bool(
            qdrant_client
            and (
                config.search.vector.primary == "qdrant"
                or config.search.vector.dual_write
            )
        )
        if (
            config.search.vector.primary == "qdrant" or config.search.vector.dual_write
        ) and not qdrant_client:
            raise RuntimeError(
                "Qdrant client is required for primary or dual-write modes."
            )
        if qdrant_required and not embeddings.get("sections"):
            raise RuntimeError(
                "Embeddings are required for Qdrant write but are missing."
            )

        neo4j_session = neo4j_driver.session()
        neo4j_tx = neo4j_session.begin_transaction()

        logger.debug(
            "neo4j_transaction_started",
            saga_id=saga_id,
            document_id=document_id,
        )

        neo4j_writer._neo4j_upsert_document(neo4j_tx, document)
        chunk_count = neo4j_writer._neo4j_upsert_sections(
            neo4j_tx, document_id, sections
        )
        stats["sections_upserted"] = chunk_count
        written_neo4j_chunks = [s["id"] for s in sections if "id" in s]

        if entities and isinstance(entities, dict):
            mentioned_entity_ids = set()
            for section in sections:
                for mention in section.get("_mentions", []):
                    entity_id = mention.get("entity_id")
                    if entity_id and entity_id in entities:
                        mentioned_entity_ids.add(entity_id)

            if mentioned_entity_ids:
                original_count = len(entities)
                entities = {
                    entity_id: entity_data
                    for entity_id, entity_data in entities.items()
                    if entity_id in mentioned_entity_ids
                }
                pruned_count = original_count - len(entities)
                if pruned_count:
                    logger.info(
                        "structural_entities_pruned_zero_mentions",
                        pruned_count=pruned_count,
                        remaining=len(entities),
                        document_id=document_id,
                    )
            else:
                logger.info(
                    "structural_entities_pruned_zero_mentions",
                    pruned_count=len(entities),
                    remaining=0,
                    document_id=document_id,
                )
                entities = {}

        entity_count = neo4j_writer._neo4j_upsert_entities(neo4j_tx, entities)
        stats["entities_upserted"] = entity_count

        all_mentions = []
        structural_count = 0
        gliner_count = 0
        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue
            for mention in section.get("_mentions", []):
                entity_id = mention.get("entity_id")
                if not entity_id:
                    continue
                mention_dict = {
                    "section_id": section_id,
                    "entity_id": entity_id,
                    "name": mention.get("name", ""),
                    "type": mention.get("type", ""),
                    "confidence": mention.get("confidence", 0.5),
                    "source": mention.get("source", "structural"),
                }
                all_mentions.append(mention_dict)
                if mention.get("source") == "gliner":
                    gliner_count += 1
                else:
                    structural_count += 1

        logger.debug(
            "mentions_collected_for_neo4j",
            structural_count=structural_count,
            gliner_count=gliner_count,
            total_count=len(all_mentions),
        )

        neo4j_writer._neo4j_create_mentions(neo4j_tx, all_mentions)

        references_count = neo4j_writer._neo4j_create_references(neo4j_tx, references)
        stats["references_created"] = references_count

        embedding_meta_count = neo4j_writer._neo4j_upsert_embedding_metadata(
            neo4j_tx, sections, embeddings, builder
        )
        stats["embedding_metadata_upserted"] = embedding_meta_count

        logger.info(
            "neo4j_write_complete",
            doc_id=document_id,
            saga_id=saga_id,
            nodes_created=chunk_count + entity_count + 1,
            relationships_created=references_count + len(mentions),
            node_types={
                "Document": 1,
                "Section": chunk_count,
                "Entity": entity_count,
            },
            relationship_types={
                "HAS_CHUNK": chunk_count,
                "MENTIONS": len(mentions),
                "REFERENCES": references_count,
            },
            embedding_metadata=embedding_meta_count,
        )

        if structural_edges_builder is None:
            from src.ingestion.structural_edges import build_structural_edges_in_tx

            structural_edges_builder = build_structural_edges_in_tx

        structural_result = structural_edges_builder(
            neo4j_tx, document_id, skip_has_chunk=True
        )
        stats["structural_edges"] = structural_result

        logger.info(
            "structural_edges_built",
            doc_id=document_id,
            saga_id=saga_id,
            edges=structural_result.get("stats", {}),
            warnings=structural_result.get("warnings", []),
        )

        qdrant_count = 0
        if qdrant_client and embeddings.get("sections"):
            qdrant_count = qdrant_writer._qdrant_upsert_vectors(
                document, sections, embeddings, builder
            )
            stats["vectors_upserted"] = qdrant_count
            written_qdrant_points = [s["id"] for s in sections if "id" in s]

            collection_name = getattr(builder, "collection_name", None)
            if not collection_name:
                collection_name = getattr(
                    config.search.vector.qdrant, "collection_name", None
                )
            if not collection_name:
                collection_name = getattr(
                    config.search.vector, "collection", "chunks_multi"
                )
            logger.info(
                "qdrant_upsert_complete",
                doc_id=document_id,
                saga_id=saga_id,
                points_upserted=qdrant_count,
                collection=collection_name,
                vector_types=[
                    "content",
                    "title",
                    "doc_title",
                    "text-sparse",
                    "title-sparse",
                    "entity-sparse",
                    "late-interaction",
                ],
            )

        if (
            config.search.vector.primary == "qdrant" or config.search.vector.dual_write
        ) and qdrant_count == 0:
            raise RuntimeError("Qdrant write required but produced zero vectors.")

        neo4j_tx.commit()
        context.neo4j_chunk_ids = written_neo4j_chunks

        logger.info(
            "atomic_saga_committed",
            saga_id=saga_id,
            document_id=document_id,
            stats=stats,
        )

        if cross_doc_linker is None:
            from src.ingestion.stages.link import create_cross_doc_links

            cross_doc_linker = create_cross_doc_links

        cross_doc_stats = cross_doc_linker(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            config=config,
            document_id=document_id,
            document=document,
            sections=sections,
            embeddings=embeddings,
            trace=trace,
        )
        if cross_doc_stats:
            stats["cross_doc_linking"] = cross_doc_stats

        return {
            "success": True,
            "stats": stats,
            "neo4j_committed": True,
            "qdrant_committed": qdrant_count > 0,
        }

    except Exception as e:
        logger.error(
            "atomic_saga_failed",
            saga_id=saga_id,
            document_id=document_id,
            error=str(e),
        )

        neo4j_rolled_back = False
        qdrant_cleaned_up = False

        if neo4j_tx and not neo4j_tx.closed():
            try:
                neo4j_tx.rollback()
                logger.info(
                    "neo4j_transaction_rolled_back",
                    saga_id=saga_id,
                    document_id=document_id,
                )
                if trace:
                    trace.add_event(
                        stage="write",
                        kind="compensation",
                        message="neo4j_transaction_rolled_back",
                        data={"saga_id": saga_id, "document_id": document_id},
                    )
                neo4j_rolled_back = True
            except Exception as rollback_err:
                logger.error(
                    "neo4j_rollback_failed",
                    saga_id=saga_id,
                    error=str(rollback_err),
                )
                if trace:
                    trace.add_event(
                        stage="write",
                        kind="error",
                        message="neo4j_rollback_failed",
                        data={"saga_id": saga_id, "error": str(rollback_err)},
                    )

        if written_qdrant_points:
            try:
                qdrant_writer._compensate_qdrant(written_qdrant_points, builder)
                qdrant_cleaned_up = True
                logger.info(
                    "qdrant_compensation_completed",
                    saga_id=saga_id,
                    points_cleaned=len(written_qdrant_points),
                )
                if trace:
                    trace.add_event(
                        stage="write",
                        kind="compensation",
                        message="qdrant_compensation_completed",
                        data={
                            "saga_id": saga_id,
                            "points_cleaned": len(written_qdrant_points),
                        },
                    )
            except Exception as qdrant_err:
                logger.error(
                    "qdrant_compensation_failed",
                    saga_id=saga_id,
                    error=str(qdrant_err),
                )
                if trace:
                    trace.add_event(
                        stage="write",
                        kind="error",
                        message="qdrant_compensation_failed",
                        data={"saga_id": saga_id, "error": str(qdrant_err)},
                    )

        compensated = neo4j_rolled_back or qdrant_cleaned_up

        return {
            "success": False,
            "stats": stats,
            "error": str(e),
            "neo4j_committed": False,
            "qdrant_committed": len(written_qdrant_points) > 0,
            "compensated": compensated,
        }

    finally:
        if neo4j_session:
            try:
                neo4j_session.close()
            except Exception as session_err:
                logger.warning(
                    "neo4j_session_close_failed",
                    saga_id=saga_id,
                    error=str(session_err),
                )
                if trace:
                    trace.add_event(
                        stage="write",
                        kind="error",
                        message="neo4j_session_close_failed",
                        data={"saga_id": saga_id, "error": str(session_err)},
                    )
