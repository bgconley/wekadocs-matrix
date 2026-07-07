"""Per-page artifact cache -- the fine-grained resume + retry substrate.

Each converted page is persisted as a Markdown file plus a JSON sidecar under::

    <work_dir>/pages/<sha256>/d<dpi>_<model-slug>/<page:05d>.md
                                                 /<page:05d>.json

Encoding ``dpi`` and ``model_id`` into the path makes them part of the cache key,
so a DPI bump or a model change is a natural cache MISS (re-convert) while an
unchanged re-run is a HIT (instant). A page in ``status="failed"`` state is what
``retry-failed`` scans for.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

from .manifest import slugify
from .prompts import PROMPT_VERSION


@dataclass
class PageMeta:
    sha256: str
    page_no: int
    dpi: int
    model_id: str
    status: str  # "ok" | "failed"
    char_len: int = 0
    endpoint: Optional[str] = None
    image_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    updated_at: float = 0.0


def _page_dir(work_dir: Path, sha256: str, dpi: int, model_id: str) -> Path:
    # PROMPT_VERSION is part of the key so a prompt change is a cache miss.
    return work_dir / "pages" / sha256 / f"d{dpi}_{slugify(model_id)}_p{PROMPT_VERSION}"


def md_path(work_dir: Path, sha256: str, page_no: int, dpi: int, model_id: str) -> Path:
    return _page_dir(work_dir, sha256, dpi, model_id) / f"{page_no:05d}.md"


def meta_path(
    work_dir: Path, sha256: str, page_no: int, dpi: int, model_id: str
) -> Path:
    return _page_dir(work_dir, sha256, dpi, model_id) / f"{page_no:05d}.json"


def read_meta(
    work_dir: Path, sha256: str, page_no: int, dpi: int, model_id: str
) -> Optional[PageMeta]:
    p = meta_path(work_dir, sha256, page_no, dpi, model_id)
    if not p.is_file():
        return None
    try:
        return PageMeta(**json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def is_done(work_dir: Path, sha256: str, page_no: int, dpi: int, model_id: str) -> bool:
    """True iff the page has an ``ok`` artifact for exactly this (dpi, model_id)."""

    m = read_meta(work_dir, sha256, page_no, dpi, model_id)
    if m is None or m.status != "ok":
        return False
    return md_path(work_dir, sha256, page_no, dpi, model_id).is_file()


def read_md(
    work_dir: Path, sha256: str, page_no: int, dpi: int, model_id: str
) -> Optional[str]:
    p = md_path(work_dir, sha256, page_no, dpi, model_id)
    return p.read_text(encoding="utf-8") if p.is_file() else None


def write_success(
    work_dir: Path,
    *,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    markdown: str,
    endpoint: str,
    image_tokens: Optional[int],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    attempts: int,
) -> PageMeta:
    d = _page_dir(work_dir, sha256, dpi, model_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{page_no:05d}.md").write_text(markdown, encoding="utf-8")
    meta = PageMeta(
        sha256=sha256,
        page_no=page_no,
        dpi=dpi,
        model_id=model_id,
        status="ok",
        char_len=len(markdown),
        endpoint=endpoint,
        image_tokens=image_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        attempts=attempts,
        error=None,
        updated_at=time.time(),
    )
    (d / f"{page_no:05d}.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False), encoding="utf-8"
    )
    return meta


def write_failure(
    work_dir: Path,
    *,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    error: str,
    attempts: int,
) -> PageMeta:
    d = _page_dir(work_dir, sha256, dpi, model_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = PageMeta(
        sha256=sha256,
        page_no=page_no,
        dpi=dpi,
        model_id=model_id,
        status="failed",
        char_len=0,
        error=error[:2000],
        attempts=attempts,
        updated_at=time.time(),
    )
    (d / f"{page_no:05d}.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False), encoding="utf-8"
    )
    # A stale success .md must not linger next to a failure marker.
    stale = d / f"{page_no:05d}.md"
    if stale.is_file():
        stale.unlink()
    return meta


def iter_page_meta(
    work_dir: Path, sha256: str, dpi: int, model_id: str
) -> Iterator[PageMeta]:
    d = _page_dir(work_dir, sha256, dpi, model_id)
    if not d.is_dir():
        return
    for jp in sorted(d.glob("*.json")):
        try:
            yield PageMeta(**json.loads(jp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            continue


def discover_keysets(work_dir: Path, sha256: str) -> list[tuple[int, str]]:
    """Return the (dpi, model_id) keysets present for a doc (for ``status`` without probing)."""

    d = work_dir / "pages" / sha256
    out: list[tuple[int, str]] = []
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        for jp in sorted(sub.glob("*.json")):
            try:
                m = PageMeta(**json.loads(jp.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                continue
            out.append((m.dpi, m.model_id))
            break  # one meta is enough to identify the keyset
    return out
