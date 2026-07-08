"""Stage 3 -- stitch per-page Markdown into one document and repair page seams.

Page-at-a-time transcription leaves artifacts at page boundaries: running
headers/footers that slipped through, end-of-line hyphenation, and tables / code
fences / paragraphs split across the break. These functions repair those seams.
Each helper is pure (str/list in, str/list out) so the heuristics can be pinned
by unit tests.
"""

from __future__ import annotations

import re
from collections import Counter

from .fences import open_fence_at_end

_FENCE_RE = re.compile(r"^```[\w+-]*\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$")  # |---|:--:| separator row
_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d+\s*(of\s+\d+)?\s*$", re.I)
# Furniture band like "AHV | Host Network Management | 60"
_FURNITURE_BAND_RE = re.compile(r"^\s*\S.*\|.*\|\s*\d+\s*$")

_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s")
_SENTENCE_END = tuple(".?!:;)]\"'")


def _nonempty(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.strip()]


def _furniture_band_key(line: str) -> str | None:
    if not _FURNITURE_BAND_RE.match(line):
        return None
    return re.sub(r"\d+\s*$", "#", line.strip())


def strip_page_furniture(pages: list[str]) -> list[str]:
    """Remove recurring running headers/footers and standalone page numbers.

    A short line that appears in the top-2 or bottom-3 of many pages is treated
    as furniture and dropped everywhere. Pure page-number lines are always dropped.
    """

    if len(pages) < 2:
        # Still drop bare page-number lines even for a single page.
        return [_drop_page_numbers(p) for p in pages]

    edge_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()
    page_edge_indices: list[set[int]] = []
    for p in pages:
        raw_lines = p.splitlines()
        nonempty_indices = [i for i, ln in enumerate(raw_lines) if ln.strip()]
        edge_indices = set(nonempty_indices[:2] + nonempty_indices[-3:])
        page_edge_indices.append(edge_indices)
        # Count each distinct edge line ONCE per page: on a short page the top-2
        # and bottom-3 windows overlap, and we must not mistake a page's own
        # heading for recurring furniture just because it appears in both windows.
        page_edges = {
            raw_lines[i].strip()
            for i in edge_indices
            if 0 < len(raw_lines[i].strip()) <= 90
        }
        for s in page_edges:
            edge_counter[s] += 1
            key = _furniture_band_key(s)
            if key is not None:
                band_counter[key] += 1

    threshold = max(2, int(len(pages) * 0.3))
    furniture = {
        s for s, c in edge_counter.items() if c >= threshold and (len(s) <= 90)
    }
    furniture_bands = {key for key, c in band_counter.items() if c >= threshold}

    cleaned: list[str] = []
    for p, edge_indices in zip(pages, page_edge_indices):
        out_lines = []
        for i, ln in enumerate(p.splitlines()):
            s = ln.strip()
            if s in furniture:
                continue
            if _PAGE_NUM_RE.match(s):
                continue
            band_key = _furniture_band_key(s)
            if (
                i in edge_indices
                and band_key is not None
                and band_key in furniture_bands
            ):
                continue
            out_lines.append(ln)
        cleaned.append("\n".join(out_lines).strip("\n"))
    return cleaned


def _drop_page_numbers(page: str) -> str:
    kept = []
    for ln in page.splitlines():
        s = ln.strip()
        if _PAGE_NUM_RE.match(s):
            continue
        if _FURNITURE_BAND_RE.match(s) and not _is_table_row(s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip("\n")


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") or (s.count("|") >= 2 and not s.startswith("```"))


def _merge_seam(acc: str, nxt: str) -> str:
    """Join two consecutive page bodies, repairing a split structure at the seam."""

    nxt = nxt.strip("\n")
    if not acc.strip():
        return nxt
    if not nxt.strip():
        return acc

    # 0) Seam inside an open code fence (finding #31): the next page continues the
    # code (rule 8 forbids re-opening the fence). Join with a single newline and
    # skip every prose heuristic below -- they would weld command lines together
    # or de-hyphenate code.
    if open_fence_at_end(acc):
        return acc.rstrip("\n") + "\n" + nxt

    acc_lines = acc.rstrip("\n").split("\n")
    nxt_lines = nxt.split("\n")
    acc_last = acc_lines[-1]
    nxt_first = nxt_lines[0]

    def normalized_table_row(line: str) -> tuple[str, ...]:
        cells = line.strip().strip("|").split("|")
        return tuple(cell.strip() for cell in cells)

    def current_table_header() -> str | None:
        for idx in range(len(acc_lines) - 2, 0, -1):
            if _TABLE_SEP_RE.match(acc_lines[idx].strip()) and _is_table_row(
                acc_lines[idx - 1]
            ):
                return acc_lines[idx - 1]
        return None

    # 1) Table continued across the break with a repeated header: only drop the
    # repeat when the next page's header is the same table header. A different
    # header+separator pair is a new table and must remain intact.
    if _is_table_row(acc_last) and _is_table_row(nxt_first):
        nxt_starts_with_header = len(nxt_lines) >= 2 and _TABLE_SEP_RE.match(
            nxt_lines[1].strip()
        )
        if nxt_starts_with_header:
            acc_header = current_table_header()
            if acc_header is not None and normalized_table_row(
                acc_header
            ) == normalized_table_row(nxt_first):
                return "\n".join(acc_lines + nxt_lines[2:])
            return acc.rstrip("\n") + "\n\n" + nxt
        return "\n".join(acc_lines + nxt_lines)

    # 2) Word split by a trailing hyphen at the seam (finding #40): join the
    # fragments onto one line but KEEP the hyphen. At a page boundary a trailing
    # hyphen is far more often a real compound ("read-only", "high-availability")
    # than a soft line-wrap, and corrupting a real term ("readonly") is worse than
    # leaving a soft-wrap hyphenated ("con-figuration").
    if re.search(r"[A-Za-z]-$", acc_last) and re.match(r"^[a-z]", nxt_first):
        joined_first = acc_last + nxt_first
        merged = acc_lines[:-1] + [joined_first] + nxt_lines[1:]
        return "\n".join(merged)

    # 3) Paragraph split mid-sentence -> rejoin with a space (no blank line).
    if (
        _is_prose(acc_last)
        and not acc_last.rstrip().endswith(_SENTENCE_END)
        and _is_prose(nxt_first)
        and re.match(r"^[a-z0-9(]", nxt_first.strip())
    ):
        joined_first = acc_last.rstrip() + " " + nxt_first.lstrip()
        merged = acc_lines[:-1] + [joined_first] + nxt_lines[1:]
        return "\n".join(merged)

    # Default: normal block separation. Closed fence pairs deliberately land here:
    # a closed block at the bottom of one page followed by a new fence at the top
    # of the next page is a separate block, not a continuation.
    return acc.rstrip("\n") + "\n\n" + nxt


def _is_prose(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(("#", ">", "|", "```")) or _LIST_RE.match(s):
        return False
    return True


def stitch_pages(pages: list[str]) -> str:
    """Strip furniture, then fold pages together repairing each seam."""

    cleaned = strip_page_furniture(pages)
    cleaned = [p for p in cleaned if p.strip()]
    if not cleaned:
        return ""
    acc = cleaned[0].strip("\n")
    for nxt in cleaned[1:]:
        acc = _merge_seam(acc, nxt)
    return acc.strip() + "\n"
