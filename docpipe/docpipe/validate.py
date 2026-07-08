"""Stage 5 -- validation & QA.

Two levels:
  * Page-level heuristics flag empty, garbled (degenerate repetition / low
    alnum ratio), or suspiciously-short output, plus an ADVISORY token-overlap
    score against the PDF's own text layer (never gating -- a legitimately
    image-only page has low overlap).
  * Document-level coverage confirms every page produced an ``ok`` artifact and
    surfaces the pages that failed or were never attempted.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import artifacts
from .manifest import DocRecord

_WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")


@dataclass
class PageQA:
    flags: list[str] = field(default_factory=list)
    overlap: Optional[float] = None  # fraction of text-layer words present in output

    @property
    def suspect(self) -> bool:
        return any(f.startswith(("garbled", "short")) for f in self.flags)


def _alnum_ratio(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    alnum = sum(c.isalnum() or c.isspace() for c in stripped)
    return alnum / len(stripped)


def _repetition_flag(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return None
    counts = Counter(lines)
    top_line, top_n = counts.most_common(1)[0]
    if len(top_line) > 3 and top_n / len(lines) > 0.5:
        return "garbled:repetition"
    # Long run of identical consecutive lines (classic decode loop).
    run = best = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 8:
        return "garbled:loop"
    return None


def token_overlap(output: str, text_layer: str) -> Optional[float]:
    ref = set(_WORD_RE.findall(text_layer.lower()))
    if not ref:
        return None
    got = set(_WORD_RE.findall(output.lower()))
    return len(got & ref) / len(ref)


def assess_page(markdown: str, text_layer: Optional[str] = None) -> PageQA:
    qa = PageQA()
    body = markdown.strip()
    if not body:
        qa.flags.append("empty")
        return qa
    if _alnum_ratio(body) < 0.55:
        qa.flags.append("garbled:low_alnum")
    rep = _repetition_flag(body)
    if rep:
        qa.flags.append(rep)
    if text_layer is not None:
        tl = text_layer.strip()
        if len(tl) > 200 and len(body) < 0.15 * len(tl) and not qa.suspect:
            qa.flags.append("short_vs_textlayer")
        qa.overlap = token_overlap(body, tl)
    return qa


@dataclass
class DocCoverage:
    sha256: str
    slug: str
    page_count: int
    ok_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    missing_pages: list[int] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.failed_pages and not self.missing_pages

    @property
    def ok_count(self) -> int:
        return len(self.ok_pages)


def document_coverage(
    work_dir: Path,
    record: DocRecord,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> DocCoverage:
    cov = DocCoverage(
        sha256=record.sha256, slug=record.slug, page_count=record.page_count
    )
    for page_no in range(1, record.page_count + 1):
        meta = artifacts.read_meta(
            work_dir, record.sha256, page_no, dpi, model_id, prompt_version
        )
        if meta is None:
            cov.missing_pages.append(page_no)
        elif (
            meta.status == "ok"
            and artifacts.read_md(
                work_dir, record.sha256, page_no, dpi, model_id, prompt_version
            )
            is not None
        ):
            cov.ok_pages.append(page_no)
        elif meta.status == "ok":
            cov.missing_pages.append(page_no)
        else:
            cov.failed_pages.append(page_no)
    return cov
