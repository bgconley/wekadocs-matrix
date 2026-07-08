"""Per-page artifact cache -- the fine-grained resume + retry substrate.

Each converted page is persisted as a Markdown file plus a JSON sidecar under::

    <work_dir>/pages/<sha256>/d<dpi>_<model-slug>_p<prompt-version>/<page:05d>.md
                                                                 /<page:05d>.json

Encoding ``dpi``, ``model_id``, and ``prompt_version`` into the path makes them
part of the cache key, so a DPI bump, model change, or prompt change is a natural
cache MISS (re-convert) while an unchanged re-run is a HIT (instant). A page in
``status="failed"`` state is what ``retry-failed`` scans for.
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
    prompt_version: str = PROMPT_VERSION
    char_len: int = 0
    endpoint: Optional[str] = None
    image_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    updated_at: float = 0.0


def _page_dir(
    work_dir: Path,
    sha256: str,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Path:
    # PROMPT_VERSION is part of the key so a prompt change is a cache miss.
    version = prompt_version or PROMPT_VERSION
    return work_dir / "pages" / sha256 / f"d{dpi}_{slugify(model_id)}_p{version}"


def md_path(
    work_dir: Path,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Path:
    return (
        _page_dir(work_dir, sha256, dpi, model_id, prompt_version) / f"{page_no:05d}.md"
    )


def meta_path(
    work_dir: Path,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Path:
    return (
        _page_dir(work_dir, sha256, dpi, model_id, prompt_version)
        / f"{page_no:05d}.json"
    )


def read_meta(
    work_dir: Path,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Optional[PageMeta]:
    p = meta_path(work_dir, sha256, page_no, dpi, model_id, prompt_version)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("prompt_version", prompt_version or PROMPT_VERSION)
        return PageMeta(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def is_done(
    work_dir: Path,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> bool:
    """True iff the page has an ``ok`` artifact for exactly this keyset."""

    m = read_meta(work_dir, sha256, page_no, dpi, model_id, prompt_version)
    if m is None or m.status != "ok":
        return False
    return read_md(work_dir, sha256, page_no, dpi, model_id, prompt_version) is not None


def read_md(
    work_dir: Path,
    sha256: str,
    page_no: int,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Optional[str]:
    p = md_path(work_dir, sha256, page_no, dpi, model_id, prompt_version)
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    meta = read_meta(work_dir, sha256, page_no, dpi, model_id, prompt_version)
    if meta is None or meta.status != "ok":
        return None
    if not text.strip():
        return None
    if len(text) != meta.char_len:
        return None
    return text


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
        prompt_version=PROMPT_VERSION,
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
        prompt_version=PROMPT_VERSION,
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
    work_dir: Path,
    sha256: str,
    dpi: int,
    model_id: str,
    prompt_version: Optional[str] = None,
) -> Iterator[PageMeta]:
    d = _page_dir(work_dir, sha256, dpi, model_id, prompt_version)
    if not d.is_dir():
        return
    for jp in sorted(d.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            data.setdefault("prompt_version", prompt_version or PROMPT_VERSION)
            yield PageMeta(**data)
        except (json.JSONDecodeError, TypeError):
            continue


def _prompt_version_from_keyset_dir(path: Path) -> Optional[str]:
    marker = "_p"
    if marker not in path.name:
        return None
    return path.name.rsplit(marker, 1)[1] or None


def discover_keysets(work_dir: Path, sha256: str) -> list[tuple[int, str, str]]:
    """Return the (dpi, model_id, prompt_version) keysets present for a doc."""

    d = work_dir / "pages" / sha256
    out: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        dir_prompt = _prompt_version_from_keyset_dir(sub)
        for jp in sorted(sub.glob("*.json")):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                prompt_version = data.get("prompt_version") or dir_prompt
                if not prompt_version:
                    continue
                data.setdefault("prompt_version", prompt_version)
                m = PageMeta(**data)
            except (json.JSONDecodeError, TypeError):
                continue
            key = (m.dpi, m.model_id, prompt_version)
            if key not in seen:
                seen.add(key)
                out.append(key)
            break  # one meta is enough to identify the keyset
    return out
