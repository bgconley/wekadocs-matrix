"""Stage 4 -- clean, normalize, and front-matter a stitched document.

Guarantees that matter to the downstream ingestion parser:
  * The document body starts with exactly one H1 (its title). Content before the
    first heading is dropped by the parser, so we hoist the title to the top.
  * A single H1 per document (extra H1s are demoted to H2).
  * YAML front matter at byte 0 with the parser-consumed keys (title, version,
    last_edited) plus provenance keys (inert to the parser, useful to humans/tools).
  * Collapsed blank runs and stripped trailing whitespace.

The title is taken FAITHFULLY from the document's own first H1 when present
(what the model transcribed from the page), falling back to embedded PDF metadata,
then a humanized filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from markdown_it import MarkdownIt

from . import PIPELINE_VERSION
from .fences import is_fence_line, iter_lines_with_fence_state
from .manifest import DocRecord

_H1_RE = re.compile(r"^# +(\S.*?)\s*$")
_ANY_H1_RE = re.compile(r"^# +(?=\S)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_H2PLUS_RE = re.compile(r"^#{2,6}\s")
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_TOC_HEADING_RE = re.compile(r"^#{1,6}\s+(?:table\s+of\s+contents|contents)\s*$", re.I)
_TOC_DOT_LEADER_RE = re.compile(r"\.{4,}|(?:^|[ \t])(?:\.\s*){6,}$")
_TOC_SPACED_DOT_PAGE_RE = re.compile(r"(?:\.\s*){6,}\*{0,2}\d{1,4}\*{0,2}\s*$")
_TOC_SYMBOL_LEADER_RE = re.compile(r"(?:^|[ \t])(?:[*!@#$%^~]\s*){4,}$")
_TOC_TRAILING_PAGE_RE = re.compile(r"(?:\.{2,}|\s{3,})\s*\*{0,2}\d{1,4}\*{0,2}\s*$")
_TOC_TITLE_PAGE_RE = re.compile(r"^\S.{2,120}\.\s+\*{0,2}\d{1,4}\*{0,2}\s*$")
_TOC_STANDALONE_PAGE_RE = re.compile(r"^-?\d{1,4}$")
_TOC_PIPE_GARBAGE_RE = re.compile(r"\|\|---|(?:\|\s*){8,}|\.\.\.\|")
_BARE_CLI_RE = re.compile(r"^\s*(?:nutanix@|<acropolis>|ncli\s|acli\s|ncli>|\$\s).+")
_INDENTED_BLOCK_RE = re.compile(r"^(?: {4,}|\t+)")
_INDENTED_FENCE_RE = re.compile(r"^(?P<indent>[ \t]+)(?P<fence>```.*)$")
_BLOCKQUOTE_FENCE_RE = re.compile(r"^\s*>+\s*```(?P<suffix>.*)$")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_TABLE_SEPARATORISH_CELL_RE = re.compile(r"^[\s:.-]+$")
_KEY_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+$")
_KNOWN_FENCE_LANGS = (
    "bash",
    "shell",
    "sh",
    "text",
    "python",
    "yaml",
    "json",
    "console",
)
_MD = MarkdownIt("gfm-like")


@dataclass
class RunMeta:
    model_id: str
    endpoint: str  # host(s) that produced the doc, comma-joined
    dpi: int
    extracted_at: Optional[str] = None  # ISO8601; filled at build time if None


def first_h1_text(body: str) -> Optional[str]:
    # Skip lines inside a fenced code block: a '#' comment there is not a heading.
    for line, in_code in iter_lines_with_fence_state(body):
        if in_code:
            continue
        m = _H1_RE.match(line)
        if m:
            return m.group(1).strip()
    return None


def humanize(slug: str) -> str:
    return re.sub(r"-+", " ", slug).strip().title() or "Untitled"


def resolve_title(body: str, record: DocRecord) -> str:
    return first_h1_text(body) or (record.title or "").strip() or humanize(record.slug)


def demote_extra_h1s(body: str) -> str:
    """Keep the first H1; demote any later ``# `` headings to ``## `` -- but never
    touch a ``# `` line inside a fenced code block (it is a shell/config comment)."""

    seen = False
    out: list[str] = []
    for line, in_code in iter_lines_with_fence_state(body):
        if not in_code and _ANY_H1_RE.match(line):
            if seen:
                out.append("#" + line)  # "# X" -> "## X"
                continue
            seen = True
        out.append(line)
    return "\n".join(out)


def ensure_title_h1(body: str, title: str) -> str:
    stripped = body.lstrip("\n")
    lines = stripped.split("\n")
    if lines and _ANY_H1_RE.match(lines[0]):
        body2 = stripped  # already opens with the (title) H1
    else:
        body2 = f"# {title}\n\n{stripped}"
    return demote_extra_h1s(body2)


def collapse_blanks(body: str) -> str:
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return _MULTI_BLANK_RE.sub("\n\n", body).strip("\n")


def _has_toc_marker(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _TOC_STANDALONE_PAGE_RE.match(stripped):
        return True
    if _TOC_DOT_LEADER_RE.search(stripped):
        return True
    if _TOC_SPACED_DOT_PAGE_RE.search(stripped):
        return True
    if _TOC_SYMBOL_LEADER_RE.search(stripped):
        return True
    if _TOC_TRAILING_PAGE_RE.search(stripped):
        return True
    if _TOC_TITLE_PAGE_RE.match(stripped):
        return True
    if _TOC_PIPE_GARBAGE_RE.search(stripped):
        return True
    return False


def _next_nonblank_index(lines: list[str], start: int) -> Optional[int]:
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return idx
    return None


def _is_toc_heading_entry(lines: list[str], idx: int) -> bool:
    if not _ATX_HEADING_RE.match(lines[idx].strip()):
        return False
    next_idx = _next_nonblank_index(lines, idx + 1)
    return next_idx is not None and _has_toc_marker(lines[next_idx])


def _is_plain_toc_heading_entry(lines: list[str], idx: int) -> bool:
    stripped = lines[idx].strip()
    if (
        not stripped
        or _ATX_HEADING_RE.match(stripped)
        or stripped.startswith(("|", ">", "```", "-", "*", "+"))
        or len(stripped) > 160
    ):
        return False
    next_idx = _next_nonblank_index(lines, idx + 1)
    return next_idx is not None and _has_toc_marker(lines[next_idx])


def _is_toc_continuation(lines: list[str], idx: int) -> bool:
    stripped = lines[idx].strip()
    if not stripped:
        return True
    if _TOC_HEADING_RE.match(stripped):
        return True
    if _has_toc_marker(stripped):
        return True
    return _is_toc_heading_entry(lines, idx) or _is_plain_toc_heading_entry(lines, idx)


def _is_toc_evidence(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped and not _TOC_HEADING_RE.match(stripped) and _has_toc_marker(line)
    )


def strip_leading_table_of_contents(body: str) -> str:
    """Drop transcribed PDF table-of-contents pages from the leading matter."""

    lines = body.splitlines()
    nonempty_seen = 0
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            if _TOC_HEADING_RE.match(stripped) and nonempty_seen <= 40:
                start = idx
                break
            nonempty_seen += 1

    if start is None:
        return body

    end = start + 1
    evidence = False
    while end < len(lines) and _is_toc_continuation(lines, end):
        evidence = evidence or _is_toc_evidence(lines[end])
        end += 1

    if not evidence:
        return body

    prefix = lines[:start]
    suffix = lines[end:]
    while prefix and not prefix[-1].strip():
        prefix.pop()
    while suffix and not suffix[0].strip():
        suffix.pop(0)
    if prefix and suffix:
        return "\n".join([*prefix, "", *suffix])
    return "\n".join([*prefix, *suffix])


def _split_fence_suffix(suffix: str) -> tuple[str, str]:
    raw = suffix.strip()
    if not raw:
        return "", ""
    for lang in _KNOWN_FENCE_LANGS:
        if raw == lang:
            return lang, ""
        if raw.startswith(lang) and not raw[len(lang)].isspace():
            return lang, raw[len(lang) :].strip()
    parts = raw.split(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1].strip()
    return raw, ""


def _strip_blockquote_prefix(line: str) -> str:
    stripped = line.lstrip(" \t")
    if not stripped.startswith(">"):
        return line
    return stripped[1:].lstrip(" ")


def _strip_indent_prefix(line: str, indent: str) -> str:
    if not line.strip():
        return ""
    if line.startswith(indent):
        return line[len(indent) :]
    return line.lstrip(" \t")


def _lift_blockquoted_fences(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    in_lifted = False
    for line in lines:
        if in_lifted:
            content = _strip_blockquote_prefix(line)
            if is_fence_line(content):
                out.append(content.lstrip(" \t"))
                in_lifted = False
            else:
                out.append(content)
            continue

        match = _BLOCKQUOTE_FENCE_RE.match(line)
        if match:
            lang, inline = _split_fence_suffix(match.group("suffix"))
            out.append(f"```{lang}" if lang else "```")
            if inline:
                out.append(inline)
            in_lifted = True
        else:
            out.append(line)
    return "\n".join(out)


def _lift_indented_fences(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _INDENTED_FENCE_RE.match(line)
        if match is None:
            out.append(line)
            index += 1
            continue

        indent = match.group("indent")
        out.append(match.group("fence").lstrip(" \t"))
        index += 1
        while index < len(lines):
            current = lines[index]
            stripped = current.lstrip(" \t")
            if is_fence_line(stripped):
                out.append(stripped)
                index += 1
                break
            out.append(_strip_indent_prefix(current, indent))
            index += 1
    return "\n".join(out)


def _collapse_duplicated_fences(body: str) -> str:
    out: list[str] = []
    for line in body.splitlines():
        if out and is_fence_line(line) and is_fence_line(out[-1]):
            if line.strip() == out[-1].strip():
                continue
        out.append(line)
    return "\n".join(out)


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    for index, char in enumerate(stripped):
        if char == "|" and (index == 0 or stripped[index - 1] != "\\"):
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def _is_pipe_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 1


def _is_separator_row(line: str) -> bool:
    cells = _split_pipe_row(line)
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _is_separatorish_row(line: str) -> bool:
    cells = _split_pipe_row(line)
    return bool(cells) and all(
        _TABLE_SEPARATORISH_CELL_RE.match(cell) for cell in cells
    )


def _format_pipe_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _format_separator_row(width: int, style: str) -> str:
    return _format_pipe_row([style] * width)


def _fit_table_cells(cells: list[str], width: int, header: list[str]) -> list[str]:
    if len(cells) < width:
        return cells + [""] * (width - len(cells))
    if len(cells) <= width:
        return cells
    if width >= 2 and header[-1].strip().lower() == "replicas":
        merged = r" \| ".join(cells[width - 2 : -1])
        return cells[: width - 2] + [merged, cells[-1]]
    merged = r" \| ".join(cells[width - 1 :])
    return cells[: width - 1] + [merged]


def _infer_header(rows: list[list[str]], width: int) -> list[str]:
    if width == 3 and rows and all(_KEY_PATH_RE.match(row[0]) for row in rows):
        return ["Key", "Description", "Default Value"]
    return [f"Column {index}" for index in range(1, width + 1)]


def _normalize_pipe_block(block: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(block):
        if index + 1 >= len(block) or not _is_separatorish_row(block[index + 1]):
            raw_rows = [_split_pipe_row(line) for line in block[index:]]
            width = max((len(row) for row in raw_rows), default=0)
            if width < 2:
                out.extend(block[index:])
                break
            header = _infer_header(raw_rows, width)
            if out and out[-1] != "":
                out.append("")
            out.append(_format_pipe_row(header))
            out.append(_format_separator_row(width, ":---"))
            for row in raw_rows:
                out.append(_format_pipe_row(_fit_table_cells(row, width, header)))
            break

        if out and out[-1] != "":
            out.append("")
        header = _split_pipe_row(block[index])
        width = len(header)
        sep_cells = _split_pipe_row(block[index + 1])
        style = (
            sep_cells[0]
            if sep_cells and _TABLE_SEPARATOR_CELL_RE.match(sep_cells[0])
            else ":---"
        )
        out.append(_format_pipe_row(header))
        out.append(_format_separator_row(width, style))
        index += 2

        while index < len(block):
            if index + 1 < len(block) and _is_separatorish_row(block[index + 1]):
                break
            out.append(
                _format_pipe_row(
                    _fit_table_cells(_split_pipe_row(block[index]), width, header)
                )
            )
            index += 1
    return out


def normalize_pipe_tables(body: str) -> str:
    out: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        nonlocal block
        if block:
            out.extend(_normalize_pipe_block(block))
            block = []

    for line, in_code in iter_lines_with_fence_state(body):
        current = line
        if in_code or not _is_pipe_row(current):
            flush_block()
            out.append(current)
            continue
        if not current.strip().endswith("|"):
            current = current.rstrip() + " |"
        block.append(current)
    flush_block()
    return "\n".join(out)


def normalize_top_level_blocks(body: str) -> str:
    """Make fenced/code-like blocks corpus-safe before heading/title passes.

    The VLM often preserves PDF procedure indentation. Markdown then parses
    indented fences as nested code and ordinary four-space prose as indented code
    blocks, both of which the downstream corpus contract rejects. Normalize these
    mechanical layout artifacts deterministically while preserving fenced content.
    """

    body = _lift_blockquoted_fences(body)
    body = _lift_indented_fences(body)
    body = _collapse_duplicated_fences(body)
    out: list[str] = []
    for line, in_code in iter_lines_with_fence_state(body):
        if not in_code and _INDENTED_BLOCK_RE.match(line):
            out.append(line.lstrip(" \t"))
        else:
            out.append(line)
    return "\n".join(out)


def repair_indented_code_blocks(body: str) -> str:
    """Outdent any remaining Markdown indented-code blocks.

    This is deliberately parser-backed: after the line-oriented repairs have run,
    we ask the same Markdown parser family used by the contract where unfenced
    indented code remains, then strip only those mapped line ranges. Stripping one
    code block can reveal a later indented block that was previously swallowed by
    a malformed fence, so reparse until the parser reaches a fixed point.
    """

    lines = body.splitlines()
    while True:
        changed = False
        for token in _MD.parse("\n".join(lines)):
            if token.type != "code_block" or token.map is None:
                continue
            start, end = token.map
            for idx in range(start, min(end, len(lines))):
                if _INDENTED_BLOCK_RE.match(lines[idx]):
                    lines[idx] = lines[idx].lstrip(" \t")
                    changed = True
        if not changed:
            return "\n".join(lines)


def balance_fences(body: str) -> str:
    """Repair a genuinely dangling (unclosed) code fence.

    A model occasionally opens a ``` fence and never closes it, which makes the
    downstream parser swallow the rest of the document as code. We act ONLY when
    the total fence count is odd -- i.e. there is a real dangling fence. Its
    unmatched opener is then the last fence line (every earlier fence pairs up),
    and we close it before the next H2+ heading after it (the "forgot to close
    before the next section" case), or at end-of-document. When fences are already
    balanced we change nothing, so a legitimate ``#``/``##`` comment inside a
    properly-closed block is never mistaken for a heading (finding #3).
    """

    lines = body.splitlines()
    fence_idxs = [i for i, line in enumerate(lines) if is_fence_line(line)]
    if len(fence_idxs) % 2 == 0:
        return body
    dangling_open = fence_idxs[-1]
    insert_at = len(lines)
    for j in range(dangling_open + 1, len(lines)):
        if _H2PLUS_RE.match(lines[j]):
            insert_at = j
            break
    lines.insert(insert_at, "```")
    return "\n".join(lines)


def fence_bare_cli_commands(body: str) -> str:
    """Fence unfenced command-looking lines for downstream command extraction."""

    out: list[str] = []
    in_cli_block = False
    for line, in_code in iter_lines_with_fence_state(body):
        is_cli = (
            not in_code
            and not is_fence_line(line)
            and _BARE_CLI_RE.match(line) is not None
        )
        if is_cli and not in_cli_block:
            out.append("```bash")
            in_cli_block = True
        elif in_cli_block and not is_cli:
            out.append("```")
            in_cli_block = False
        out.append(line)
    if in_cli_block:
        out.append("```")
    return "\n".join(out)


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    s = str(value)
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_frontmatter(record: DocRecord, title: str, run: RunMeta) -> str:
    """Emit YAML front matter (byte 0). Only title/version/last_edited are read by
    the parser; the rest is provenance."""

    timestamp = run.extracted_at or _now_iso()
    fields: list[tuple[str, object]] = [
        ("title", title),
        ("version", record.doc_version or "1.0"),
        ("last_edited", timestamp),
    ]
    fields += [
        ("product", record.product or "Nutanix"),
        ("source_pdf", record.rel_path),
        ("sha256", record.sha256),
        ("page_count", record.page_count),
        ("dpi", run.dpi),
        ("model_id", run.model_id),
        ("endpoint", run.endpoint),
        ("pipeline_version", PIPELINE_VERSION),
        ("extracted_at", timestamp),
    ]
    lines = ["---"]
    lines += [f"{k}: {_yaml_scalar(v)}" for k, v in fields]
    lines.append("---")
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_document(
    stitched_body: str, record: DocRecord, run: RunMeta
) -> tuple[str, str]:
    """Return ``(final_markdown, resolved_title)`` ready to write to disk."""

    if run.extracted_at is None:
        run.extracted_at = _now_iso()
    body = collapse_blanks(stitched_body)
    body = normalize_top_level_blocks(body)
    body = strip_leading_table_of_contents(body)
    title = resolve_title(body, record)
    body = ensure_title_h1(body, title)
    body = fence_bare_cli_commands(body)
    body = _collapse_duplicated_fences(body)
    body = balance_fences(body)
    body = normalize_top_level_blocks(body)
    body = fence_bare_cli_commands(body)
    body = _collapse_duplicated_fences(body)
    body = demote_extra_h1s(body)
    body = balance_fences(body)
    body = repair_indented_code_blocks(body)
    body = fence_bare_cli_commands(body)
    body = _collapse_duplicated_fences(body)
    body = demote_extra_h1s(body)
    body = balance_fences(body)
    body = normalize_pipe_tables(body)
    body = collapse_blanks(body)
    frontmatter = build_frontmatter(record, title, run)
    return f"{frontmatter}\n\n{body}\n", title
