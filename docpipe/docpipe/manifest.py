"""Stage 0 -- discovery and the JSONL manifest.

Walks the input directory for PDFs and records, per document, a content sha256,
page count, byte size, and any embedded title/version/product metadata. The
manifest (``<work_dir>/manifest.jsonl``) is the backbone every later stage keys
off: resume, status, and the coverage report all read it, and a document is
re-scanned only when its sha256 changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from .log import get_logger

logger = get_logger("manifest")

# Filename / title keyword -> product area. First match wins.
_PRODUCT_KEYWORDS: list[tuple[str, str]] = [
    ("kubernetes", "Nutanix Kubernetes Engine"),
    ("nke", "Nutanix Kubernetes Engine"),
    ("enterprise-ai", "Nutanix Enterprise AI"),
    ("enterprise ai", "Nutanix Enterprise AI"),
    ("cloud-clusters", "Nutanix Cloud Clusters"),
    ("nc2", "Nutanix Cloud Clusters"),
    ("azure", "Nutanix Cloud Clusters"),
    ("book-of-dr", "Data Protection / DR"),
    ("dr-services", "Data Protection / DR"),
    ("book-of-ahv", "AHV"),
    ("ahv", "AHV"),
    ("advanced-admin-aos", "AOS"),
    ("aos", "AOS"),
    ("prism", "Prism"),
    ("files", "Nutanix Files"),
    ("objects", "Nutanix Objects"),
]

_VERSION_RE = re.compile(r"[vV]?(\d+)[._](\d+)")


@dataclass
class DocRecord:
    pdf_path: str  # absolute path on disk
    rel_path: str  # path relative to input_dir (stable identity for layout)
    sha256: str
    size_bytes: int
    page_count: int
    slug: str
    title: Optional[str] = None  # embedded metadata title, if any
    author: Optional[str] = None
    producer: Optional[str] = None
    product: Optional[str] = None
    doc_version: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "DocRecord":
        return cls(**json.loads(line))


def slugify(text: str) -> str:
    """ASCII kebab-case slug (safe filename; no ``__`` so scope__slug is not triggered)."""

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)  # collapse runs so no accidental "__"-like joins
    return text or "untitled"


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _infer_product(*texts: Optional[str]) -> Optional[str]:
    hay = " ".join(t.lower() for t in texts if t)
    for needle, product in _PRODUCT_KEYWORDS:
        if needle in hay:
            return product
    return None


def _infer_version(*texts: Optional[str]) -> Optional[str]:
    for t in texts:
        if not t:
            continue
        m = _VERSION_RE.search(t)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return None


def scan_pdf(pdf_path: Path, input_dir: Path) -> DocRecord:
    """Read one PDF's metadata into a :class:`DocRecord` (no rasterization)."""

    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
        page_count = doc.page_count
    stem = pdf_path.stem
    title = (meta.get("title") or "").strip() or None
    author = (meta.get("author") or "").strip() or None
    producer = (meta.get("producer") or "").strip() or None
    try:
        rel = str(pdf_path.relative_to(input_dir))
    except ValueError:
        rel = pdf_path.name
    return DocRecord(
        pdf_path=str(pdf_path.resolve()),
        rel_path=rel,
        sha256=sha256_file(pdf_path),
        size_bytes=pdf_path.stat().st_size,
        page_count=page_count,
        slug=slugify(stem),
        title=title,
        author=author,
        producer=producer,
        product=_infer_product(stem, title),
        doc_version=_infer_version(stem, title),
    )


def discover(input_dir: Path) -> list[Path]:
    """Recursively find PDFs (case-insensitive extension)."""

    seen: set[Path] = set()
    for pattern in ("*.pdf", "*.PDF"):
        for p in input_dir.rglob(pattern):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def manifest_path(work_dir: Path) -> Path:
    return work_dir / "manifest.jsonl"


def load_manifest(work_dir: Path) -> dict[str, DocRecord]:
    """Load an existing manifest keyed by sha256 (empty dict if none)."""

    path = manifest_path(work_dir)
    out: dict[str, DocRecord] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = DocRecord.from_json(line)
        out[rec.sha256] = rec
    return out


def write_manifest(work_dir: Path, records: list[DocRecord]) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(work_dir)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
    return path


def build(input_dir: Path, work_dir: Path) -> list[DocRecord]:
    """Discover PDFs and (re)build the manifest, reusing unchanged records.

    Reuse is keyed by content sha256, so a moved-but-identical file is rescanned
    (cheap) but its converted page artifacts -- also keyed by sha256 -- stay valid.
    """

    prev_by_path = {r.rel_path: r for r in load_manifest(work_dir).values()}
    records: list[DocRecord] = []
    for pdf in discover(input_dir):
        try:
            rel = str(pdf.relative_to(input_dir.resolve()))
        except ValueError:
            rel = pdf.name
        # Fast-path reuse only when same rel_path, size, and content hash match.
        prev = prev_by_path.get(rel)
        size_bytes = pdf.stat().st_size
        if prev is not None and prev.size_bytes == size_bytes:
            try:
                current_sha = sha256_file(pdf)
            except Exception as exc:
                logger.warning(
                    "skipping pdf after hash failure",
                    extra={"fields": {"path": str(pdf), "error": str(exc)}},
                )
                continue
            if prev.sha256 == current_sha:
                records.append(
                    replace(
                        prev,
                        pdf_path=str(pdf.resolve()),
                        rel_path=rel,
                        size_bytes=size_bytes,
                    )
                )
                continue
        try:
            rec = scan_pdf(pdf, input_dir.resolve())
        except Exception as exc:
            logger.warning(
                "skipping pdf after scan failure",
                extra={"fields": {"path": str(pdf), "error": str(exc)}},
            )
            continue
        logger.info(
            "scanned",
            extra={
                "fields": {
                    "doc": rec.slug,
                    "pages": rec.page_count,
                    "mb": round(rec.size_bytes / 1e6, 2),
                }
            },
        )
        records.append(rec)
    write_manifest(work_dir, records)
    return records
