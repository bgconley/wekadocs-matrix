"""Deterministic fallbacks for born-digital PDF table text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import fitz

PdfWord = tuple[float, float, float, float, str]
_KEY_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+$")


def _word5(raw: Sequence[Any]) -> PdfWord:
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), str(raw[4]))


def _same_line(a: PdfWord, b: PdfWord, tolerance: float = 3.0) -> bool:
    return abs(a[1] - b[1]) <= tolerance


def _line_text(words: Sequence[PdfWord]) -> str:
    return " ".join(word[4] for word in sorted(words, key=lambda item: item[0]))


def _header(words: Sequence[PdfWord]) -> tuple[PdfWord, PdfWord, PdfWord] | None:
    for key in words:
        if key[4].lower() != "key":
            continue
        same_line = [word for word in words if _same_line(key, word)]
        desc = next(
            (word for word in same_line if word[4].lower() == "description"), None
        )
        default = next(
            (word for word in same_line if word[4].lower() == "default"), None
        )
        value = next((word for word in same_line if word[4].lower() == "value"), None)
        if desc and default and value and key[0] < desc[0] < default[0] < value[0]:
            return key, desc, default
    return None


def _title(words: Sequence[PdfWord], header_y: float) -> str | None:
    above = [word for word in words if word[1] < header_y - 4]
    line_ys = sorted({round(word[1], 1) for word in above})
    for y in line_ys:
        line = [word for word in above if abs(word[1] - y) <= 3.0]
        text = _line_text(line).strip()
        if text.lower().startswith("table "):
            return text
    return None


def _join_wrapped(lines: list[list[str]]) -> str:
    parts: list[str] = []
    for words in lines:
        text = " ".join(words).strip()
        if not text:
            continue
        if parts and parts[-1].endswith("-"):
            parts[-1] += text
        else:
            parts.append(text)
    return " ".join(parts)


def _escape_cell(text: str) -> str:
    return text.replace("|", r"\|")


def key_value_table_markdown_from_words(
    words: Sequence[PdfWord], *, page_height: float
) -> str | None:
    normalized = [_word5(word) for word in words]
    header = _header(normalized)
    if header is None:
        return None

    key_word, desc_word, default_word = header
    header_y = key_word[1]
    footer_y = page_height - 60
    row_starts: list[float] = []
    seen: set[float] = set()
    for word in normalized:
        x0, y0, _x1, _y1, text = word
        rounded_y = round(y0, 1)
        if y0 <= header_y + 8 or y0 >= footer_y or rounded_y in seen:
            continue
        if x0 < desc_word[0] - 20 and _KEY_PATH_RE.match(text):
            row_starts.append(y0)
            seen.add(rounded_y)

    if len(row_starts) < 2:
        return None

    rows: list[list[str]] = []
    for index, start_y in enumerate(row_starts):
        end_y = row_starts[index + 1] if index + 1 < len(row_starts) else footer_y
        line_words: list[dict[float, list[tuple[float, str]]]] = [{}, {}, {}]
        for word in normalized:
            x0, y0, _x1, _y1, text = word
            if y0 < start_y - 0.1 or y0 >= end_y - 0.1 or y0 >= footer_y:
                continue
            if text == "|":
                continue
            if x0 < desc_word[0] - 5:
                col = 0
            elif x0 < default_word[0] - 5:
                col = 1
            else:
                col = 2
            if col == 0 and not (abs(y0 - start_y) < 1 and _KEY_PATH_RE.match(text)):
                col = 1
            y_key = round(y0, 1)
            line_words[col].setdefault(y_key, []).append((x0, text))

        row: list[str] = []
        for col in range(3):
            lines = [
                [text for _x, text in sorted(line_words[col][y])]
                for y in sorted(line_words[col])
            ]
            row.append(_join_wrapped(lines))
        rows.append(row)

    title = _title(normalized, header_y) or "Table"
    lines = [
        title,
        "",
        "| Key | Description | Default Value |",
        "|---|---|---|",
    ]
    for key, description, default in rows:
        lines.append(
            "| "
            + " | ".join(
                [_escape_cell(key), _escape_cell(description), _escape_cell(default)]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def page_key_value_table_markdown(pdf_path: str | Path, page_no: int) -> str | None:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_no - 1)
        words = [_word5(word) for word in page.get_text("words", sort=True)]
        return key_value_table_markdown_from_words(words, page_height=page.rect.height)
