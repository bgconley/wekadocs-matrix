"""Stage 6 -- assemble cached pages into one Markdown file per document.

Reads the ``ok`` page artifacts in order, runs per-page QA (advisory), stitches
and cleans them, then writes a single ``.md`` in the ingestion-ready shape. A
document with any failed/missing page is NOT written by default (partial docs
must not silently enter the corpus) -- pass ``allow_incomplete=True`` to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import artifacts
from .clean import RunMeta, clean_document
from .config import Config
from .log import get_logger
from .manifest import DocRecord
from .rasterize import page_text
from .stitch import stitch_pages
from .validate import assess_page, document_coverage

logger = get_logger("output")


@dataclass
class DocResult:
    sha256: str
    slug: str
    title: Optional[str] = None
    page_count: int = 0
    ok_count: int = 0
    complete: bool = False
    written: bool = False
    out_path: Optional[str] = None
    suspect_pages: list[int] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    missing_pages: list[int] = field(default_factory=list)


def output_path(config: Config, slug: str) -> Path:
    base = Path(config.output.dir)
    if config.output.layout == "flat":
        return base / f"{slug}.md"
    return base / slug / f"{slug}.md"


def assemble_document(
    config: Config,
    record: DocRecord,
    model_id: str,
    *,
    dpi: Optional[int] = None,
    allow_incomplete: bool = False,
    qa_overlap: bool = True,
) -> DocResult:
    work_dir = Path(config.work_dir)
    dpi = dpi or config.rasterize.dpi
    cov = document_coverage(work_dir, record, dpi, model_id)

    result = DocResult(
        sha256=record.sha256,
        slug=record.slug,
        page_count=record.page_count,
        ok_count=cov.ok_count,
        complete=cov.complete,
        failed_pages=cov.failed_pages,
        missing_pages=cov.missing_pages,
    )

    if not cov.complete and not allow_incomplete:
        logger.warning(
            "incomplete doc not written",
            extra={
                "fields": {
                    "doc": record.slug,
                    "ok": cov.ok_count,
                    "of": record.page_count,
                    "failed": len(cov.failed_pages),
                    "missing": len(cov.missing_pages),
                }
            },
        )
        return result

    # Collect ok page markdown in order; run advisory QA.
    page_md: list[str] = []
    endpoints_used: set[str] = set()
    for page_no in cov.ok_pages:
        md = artifacts.read_md(work_dir, record.sha256, page_no, dpi, model_id) or ""
        page_md.append(md)
        meta = artifacts.read_meta(work_dir, record.sha256, page_no, dpi, model_id)
        if meta and meta.endpoint:
            endpoints_used.add(meta.endpoint)
        tl = None
        if qa_overlap:
            try:
                tl = page_text(record.pdf_path, page_no)
            except Exception as exc:
                logger.warning(
                    "page text unavailable for QA",
                    extra={
                        "fields": {
                            "doc": record.slug,
                            "page": page_no,
                            "err": str(exc),
                        }
                    },
                )
        qa = assess_page(md, tl)
        if "empty" in qa.flags:
            result.empty_pages.append(page_no)
        if qa.suspect:
            result.suspect_pages.append(page_no)

    body = stitch_pages(page_md)
    run = RunMeta(
        model_id=model_id,
        endpoint=",".join(sorted(endpoints_used))
        or ",".join(e.name for e in config.active_endpoints),
        dpi=dpi,
    )
    final_md, title = clean_document(body, record, run)
    result.title = title

    out = output_path(config, record.slug)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_md, encoding="utf-8")
    result.written = True
    result.out_path = str(out)
    logger.info(
        "wrote doc",
        extra={
            "fields": {
                "doc": record.slug,
                "title": title,
                "pages": cov.ok_count,
                "suspect": len(result.suspect_pages),
                "path": str(out),
            }
        },
    )
    return result


def assemble_all(
    config: Config,
    records: list[DocRecord],
    model_id: str,
    *,
    dpi: Optional[int] = None,
    allow_incomplete: bool = False,
    qa_overlap: bool = True,
) -> list[DocResult]:
    results: list[DocResult] = []
    for rec in records:
        try:
            result = assemble_document(
                config,
                rec,
                model_id,
                dpi=dpi,
                allow_incomplete=allow_incomplete,
                qa_overlap=qa_overlap,
            )
        except Exception as exc:
            logger.error(
                "doc assembly failed",
                extra={"fields": {"doc": rec.slug, "err": str(exc)}},
            )
            result = DocResult(
                sha256=rec.sha256,
                slug=rec.slug,
                page_count=rec.page_count,
                failed_pages=list(range(1, rec.page_count + 1)),
            )
        results.append(result)
    return results
