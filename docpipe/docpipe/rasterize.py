"""Stage 1 -- rasterize a PDF page to a PNG for the vision model.

Rendered with PyMuPDF at a target DPI, with dimensions aligned to Qwen's 28 px
effective patch grid so the served processor does not apply a second resize that
can blur small CLI/table glyphs.

Everything here is synchronous/CPU-bound and is expected to be called inside a
thread executor by the async converter. A ``fitz.Document`` is opened per call so
nothing is shared across threads (PyMuPDF documents are not thread-safe).
"""

from __future__ import annotations

import base64
from pathlib import Path

import fitz  # PyMuPDF

_PATCH_GRID = 28


def _zoom_for(page: "fitz.Page", dpi: int, max_long_px: int) -> float:
    """Zoom factor that honours ``dpi`` but never exceeds ``max_long_px`` on the long side."""

    base = dpi / 72.0
    rect = page.rect
    long_pt = max(rect.width, rect.height) or 1.0
    if long_pt * base > max_long_px:
        return max_long_px / long_pt
    return base


def _align_px(value: float) -> int:
    return max(_PATCH_GRID, int(round(value / _PATCH_GRID)) * _PATCH_GRID)


def _target_size_for(page: "fitz.Page", dpi: int, max_long_px: int) -> tuple[int, int]:
    base = _zoom_for(page, dpi, max_long_px)
    rect = page.rect
    width = _align_px(max(1.0, rect.width * base))
    height = _align_px(max(1.0, rect.height * base))
    return width, height


def render_page_png(
    pdf_path: str | Path, page_no: int, dpi: int, max_long_px: int
) -> bytes:
    """Render 1-indexed ``page_no`` of ``pdf_path`` to PNG bytes (no alpha)."""

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_no - 1)  # fitz is 0-indexed
        target_w, target_h = _target_size_for(page, dpi, max_long_px)
        rect = page.rect
        zoom_x = target_w / (rect.width or 1.0)
        zoom_y = target_h / (rect.height or 1.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=False)
        return pix.tobytes("png")


def png_to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def render_page_data_url(
    pdf_path: str | Path, page_no: int, dpi: int, max_long_px: int
) -> str:
    return png_to_data_url(render_page_png(pdf_path, page_no, dpi, max_long_px))


def page_text(pdf_path: str | Path, page_no: int) -> str:
    """Extract the page's native text layer (reading-order sorted).

    Advisory only: used as a fallback continuity cue and as a QA cross-check
    signal. These PDFs have real (non-scanned) text layers, so this is reliable.
    """

    with fitz.open(pdf_path) as doc:
        return str(doc.load_page(page_no - 1).get_text("text", sort=True))


def render_page_to_file(
    pdf_path: str | Path, page_no: int, dpi: int, max_long_px: int, out_path: str | Path
) -> Path:
    """Render a page to a PNG file on disk (used by the ``inspect`` command)."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_page_png(pdf_path, page_no, dpi, max_long_px))
    return out
