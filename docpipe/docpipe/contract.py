"""Final Markdown contract validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml
from markdown_it import MarkdownIt

from .fences import is_fence_line, iter_lines_with_fence_state, open_fence_at_end

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_SETEXT_RE = re.compile(r"^[ \t]*(=+|-+)[ \t]*$")
_RAW_HTML_RE = re.compile(r"<\s*/?\s*(table|thead|tbody|tr|td|th|pre|code)\b", re.I)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD = MarkdownIt("gfm-like")


@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str
    line: int | None = None


@dataclass
class ContractResult:
    violations: list[ContractViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


class ContractError(ValueError):
    def __init__(self, violations: list[ContractViolation]) -> None:
        self.violations = violations
        joined = ", ".join(v.code for v in violations) or "unknown"
        super().__init__(f"Tier-0 contract failed: {joined}")


def validate_contract(markdown: str) -> ContractResult:
    violations: list[ContractViolation] = []
    title: str | None = None
    body = markdown
    body_line_offset = 0

    frontmatter_match = _FRONTMATTER_RE.match(markdown)
    if frontmatter_match is None:
        violations.append(
            ContractViolation(
                "frontmatter:missing",
                "YAML frontmatter must start at byte 0 and close before the body.",
                1,
            )
        )
    else:
        raw_frontmatter = frontmatter_match.group(1)
        body = markdown[frontmatter_match.end() :]
        body_line_offset = markdown[: frontmatter_match.end()].count("\n")
        try:
            parsed = yaml.safe_load(raw_frontmatter)
        except yaml.YAMLError as exc:
            violations.append(
                ContractViolation(
                    "frontmatter:yaml",
                    f"YAML frontmatter is not parseable: {exc}",
                    1,
                )
            )
            parsed = None
        if isinstance(parsed, dict):
            raw_title = parsed.get("title")
            if isinstance(raw_title, str) and raw_title.strip():
                title = raw_title.strip()
            else:
                violations.append(
                    ContractViolation(
                        "frontmatter:title",
                        "Frontmatter title must be a non-empty string.",
                        1,
                    )
                )
        elif parsed is not None:
            violations.append(
                ContractViolation(
                    "frontmatter:yaml",
                    "YAML frontmatter must parse to a mapping.",
                    1,
                )
            )
        else:
            violations.append(
                ContractViolation(
                    "frontmatter:title",
                    "Frontmatter title must be a non-empty string.",
                    1,
                )
            )

    _scan_lines(body, body_line_offset, title, violations)
    _scan_ast(body, body_line_offset, violations)
    return ContractResult(violations)


def _scan_lines(
    body: str,
    body_line_offset: int,
    title: str | None,
    violations: list[ContractViolation],
) -> None:
    if open_fence_at_end(body):
        violations.append(
            ContractViolation(
                "fence:unbalanced",
                "Fenced code blocks must be balanced.",
                body_line_offset + len(body.splitlines()) or 1,
            )
        )

    headings: list[tuple[int, str, int]] = []
    first_heading_seen = False
    previous_nonblank: tuple[str, int] | None = None

    for index, (line, in_fence) in enumerate(
        iter_lines_with_fence_state(body), start=1
    ):
        line_no = body_line_offset + index
        stripped = line.strip()
        leading = len(line) - len(line.lstrip(" \t"))

        if is_fence_line(line) and leading:
            violations.append(
                ContractViolation(
                    "code:not_top_level",
                    "Fenced code blocks must be top-level, not indented.",
                    line_no,
                )
            )

        if in_fence:
            continue

        if _RAW_HTML_RE.search(line):
            violations.append(
                ContractViolation(
                    "html:raw_table_or_code",
                    "Raw HTML table/code blocks are silently dropped downstream.",
                    line_no,
                )
            )

        if _TABLE_ROW_RE.match(line) and leading:
            violations.append(
                ContractViolation(
                    "table:not_top_level",
                    "GFM tables must be top-level, not indented.",
                    line_no,
                )
            )

        heading = _ATX_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            headings.append((level, text, line_no))
            first_heading_seen = True
        elif stripped and not first_heading_seen:
            violations.append(
                ContractViolation(
                    "body:pre_heading_content",
                    "Body content before the first heading is dropped downstream.",
                    line_no,
                )
            )

        if (
            previous_nonblank is not None
            and _SETEXT_RE.match(line)
            and not previous_nonblank[0].startswith("|")
        ):
            violations.append(
                ContractViolation(
                    "heading:setext",
                    "Only ATX headings are accepted by the corpus contract.",
                    line_no,
                )
            )

        if stripped:
            previous_nonblank = (stripped, line_no)

    h1s = [(text, line_no) for level, text, line_no in headings if level == 1]
    if len(h1s) != 1:
        line = h1s[0][1] if h1s else None
        violations.append(
            ContractViolation(
                "heading:h1_count",
                "Final Markdown must contain exactly one H1.",
                line,
            )
        )
    elif title is not None and h1s[0][0] != title:
        violations.append(
            ContractViolation(
                "heading:title_mismatch",
                "The single H1 must match the frontmatter title.",
                h1s[0][1],
            )
        )

    if headings and headings[0][0] != 1:
        violations.append(
            ContractViolation(
                "heading:first_not_h1",
                "The first body block must be the document H1.",
                headings[0][2],
            )
        )

    previous_level: int | None = None
    for level, _text, line_no in headings:
        if previous_level is not None and level > previous_level + 1:
            violations.append(
                ContractViolation(
                    "heading:level_jump",
                    "Heading levels must not increase by more than one.",
                    line_no,
                )
            )
        previous_level = level


def _scan_ast(
    body: str, body_line_offset: int, violations: list[ContractViolation]
) -> None:
    for token in _MD.parse(body):
        line_no = body_line_offset + token.map[0] + 1 if token.map is not None else None
        if token.type == "code_block":
            violations.append(
                ContractViolation(
                    "code:not_fenced",
                    "Indented code blocks are not corpus-safe; use fences.",
                    line_no,
                )
            )
        elif token.type == "fence" and token.level > 0:
            violations.append(
                ContractViolation(
                    "code:not_top_level",
                    "Fenced code blocks must be top-level.",
                    line_no,
                )
            )
        elif token.type == "table_open" and token.level > 0:
            violations.append(
                ContractViolation(
                    "table:not_top_level",
                    "GFM tables must be top-level.",
                    line_no,
                )
            )
        elif token.type in {"html_block", "html_inline"} and _RAW_HTML_RE.search(
            token.content
        ):
            violations.append(
                ContractViolation(
                    "html:raw_table_or_code",
                    "Raw HTML table/code blocks are silently dropped downstream.",
                    line_no,
                )
            )


def enforce_contract(markdown: str) -> None:
    result = validate_contract(markdown)
    if not result.ok:
        raise ContractError(result.violations)
