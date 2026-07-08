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

from . import PIPELINE_VERSION
from .fences import is_fence_line, iter_lines_with_fence_state
from .manifest import DocRecord

_H1_RE = re.compile(r"^# +(\S.*?)\s*$")
_ANY_H1_RE = re.compile(r"^# +(?=\S)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_H2PLUS_RE = re.compile(r"^#{2,6}\s")
_BARE_CLI_RE = re.compile(r"^\s*(?:nutanix@|<acropolis>|ncli\s|acli\s|ncli>|\$\s).+")


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

    fields: list[tuple[str, object]] = [
        ("title", title),
        ("version", record.doc_version or "1.0"),
    ]
    # last_edited is parser-consumed but optional; only emit when we have it.
    fields += [
        ("product", record.product or "Nutanix"),
        ("source_pdf", record.rel_path),
        ("sha256", record.sha256),
        ("page_count", record.page_count),
        ("dpi", run.dpi),
        ("model_id", run.model_id),
        ("endpoint", run.endpoint),
        ("pipeline_version", PIPELINE_VERSION),
        ("extracted_at", run.extracted_at or _now_iso()),
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
    title = resolve_title(body, record)
    body = ensure_title_h1(body, title)
    body = fence_bare_cli_commands(body)
    body = balance_fences(body)
    body = collapse_blanks(body)
    frontmatter = build_frontmatter(record, title, run)
    return f"{frontmatter}\n\n{body}\n", title
