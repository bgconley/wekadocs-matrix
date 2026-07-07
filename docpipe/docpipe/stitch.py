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


def strip_page_furniture(pages: list[str]) -> list[str]:
    """Remove recurring running headers/footers and standalone page numbers.

    A short line that appears in the top-2 or bottom-3 of many pages is treated
    as furniture and dropped everywhere. Pure page-number lines are always dropped.
    """

    if len(pages) < 2:
        # Still drop bare page-number lines even for a single page.
        return [_drop_page_numbers(p) for p in pages]

    edge_counter: Counter[str] = Counter()
    for p in pages:
        lines = _nonempty(p.splitlines())
        # Count each distinct edge line ONCE per page: on a short page the top-2
        # and bottom-3 windows overlap, and we must not mistake a page's own
        # heading for recurring furniture just because it appears in both windows.
        page_edges = {
            s for s in (ln.strip() for ln in lines[:2] + lines[-3:]) if 0 < len(s) <= 90
        }
        for s in page_edges:
            edge_counter[s] += 1

    threshold = max(2, int(len(pages) * 0.3))
    furniture = {
        s for s, c in edge_counter.items() if c >= threshold and (len(s) <= 90)
    }

    cleaned: list[str] = []
    for p in pages:
        out_lines = []
        for ln in p.splitlines():
            s = ln.strip()
            if s in furniture:
                continue
            if _PAGE_NUM_RE.match(s) or _FURNITURE_BAND_RE.match(s):
                continue
            out_lines.append(ln)
        cleaned.append("\n".join(out_lines).strip("\n"))
    return cleaned


def _drop_page_numbers(page: str) -> str:
    return "\n".join(
        ln
        for ln in page.splitlines()
        if not (_PAGE_NUM_RE.match(ln.strip()) or _FURNITURE_BAND_RE.match(ln.strip()))
    ).strip("\n")


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

    # 1) Code fence closed at page bottom and reopened at page top -> one block.
    if _FENCE_RE.match(acc_last.strip()) and _FENCE_RE.match(nxt_first.strip()):
        merged = acc_lines[:-1] + nxt_lines[1:]
        return "\n".join(merged)

    # 2) Table continued across the break with a repeated header -> drop the repeat.
    if (
        _is_table_row(acc_last)
        and len(nxt_lines) >= 2
        and _is_table_row(nxt_first)
        and _TABLE_SEP_RE.match(nxt_lines[1])
    ):
        merged = acc_lines + nxt_lines[2:]
        return "\n".join(merged)

    # 3) Word split by a trailing hyphen at the seam (finding #40): join the
    # fragments onto one line but KEEP the hyphen. At a page boundary a trailing
    # hyphen is far more often a real compound ("read-only", "high-availability")
    # than a soft line-wrap, and corrupting a real term ("readonly") is worse than
    # leaving a soft-wrap hyphenated ("con-figuration").
    if re.search(r"[A-Za-z]-$", acc_last) and re.match(r"^[a-z]", nxt_first):
        joined_first = acc_last + nxt_first
        merged = acc_lines[:-1] + [joined_first] + nxt_lines[1:]
        return "\n".join(merged)

    # 4) Paragraph split mid-sentence -> rejoin with a space (no blank line).
    if (
        _is_prose(acc_last)
        and not acc_last.rstrip().endswith(_SENTENCE_END)
        and _is_prose(nxt_first)
        and re.match(r"^[a-z0-9(]", nxt_first.strip())
    ):
        joined_first = acc_last.rstrip() + " " + nxt_first.lstrip()
        merged = acc_lines[:-1] + [joined_first] + nxt_lines[1:]
        return "\n".join(merged)

    # Default: normal block separation.
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
