"""Parse stage for atomic ingestion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.shared.observability import get_logger

logger = get_logger(__name__)


def parse_document(
    source_uri: str,
    content: str,
    format: str,
    *,
    embedding_model: Optional[str] = None,
    embedding_version: Optional[str] = None,
    trace=None,
) -> Dict[str, Any]:
    """Parse a source document and attach ingestion metadata."""
    from src.ingestion.parsers import parse_markdown
    from src.ingestion.parsers.html import parse_html
    from src.shared.config import get_config, get_settings

    config = get_config().model_copy(deep=True)
    _ = get_settings()

    if embedding_model:
        try:
            config.embedding.embedding_model = embedding_model
            logger.debug("embedding_model_override_applied", model=embedding_model)
        except AttributeError as e:
            logger.warning(
                "embedding_model_override_failed",
                model=embedding_model,
                error=str(e),
            )
    if embedding_version:
        try:
            config.embedding.version = embedding_version
            logger.debug(
                "embedding_version_override_applied", version=embedding_version
            )
        except AttributeError as e:
            logger.warning(
                "embedding_version_override_failed",
                version=embedding_version,
                error=str(e),
            )

    if format == "markdown":
        result = parse_markdown(source_uri, content)
    elif format == "html":
        result = parse_html(source_uri, content)
    else:
        raise ValueError(f"Unsupported format: {format}")

    document = result["Document"]
    sections = result["Sections"]

    doc_tag = None
    snapshot_scope = None
    doc_category = None

    m = re.search(r"DocTag:\s*([A-Za-z0-9_\-]+)", content or "", flags=re.I)
    if m:
        doc_tag = m.group(1)
    else:
        try:
            source_path = Path(source_uri.replace("file://", "") if source_uri else "")
            fname = source_path.name
            stem = Path(fname).stem

            path_parts = source_path.parts
            for i, part in enumerate(path_parts):
                if part == "ingest" or part.endswith("ingest"):
                    if i + 1 < len(path_parts) - 1:
                        doc_category = path_parts[i + 1]
                    break

            if "__" in stem:
                scope_part, slug_part = stem.split("__", 1)
                snapshot_scope = scope_part
                doc_tag = slug_part
            else:
                doc_tag = doc_category if doc_category else stem
        except (ValueError, AttributeError) as e:
            logger.debug(
                "doc_tag_extraction_fallback",
                source_uri=source_uri,
                error=str(e),
            )
            if trace:
                trace.add_event(
                    stage="parse",
                    kind="fallback",
                    message="doc_tag_extraction_fallback",
                    data={"source_uri": source_uri, "error": str(e)},
                )

    document["doc_tag"] = doc_tag
    document["doc_category"] = doc_category
    document["snapshot_scope"] = snapshot_scope

    for section in sections:
        section["doc_tag"] = doc_tag
        section["doc_category"] = doc_category
        section["snapshot_scope"] = snapshot_scope

    return {
        "config": config,
        "document": document,
        "sections": sections,
    }
