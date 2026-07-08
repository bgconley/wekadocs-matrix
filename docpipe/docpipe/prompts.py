"""The VLM page-conversion prompt and OpenAI chat-message builder.

This module is the single most important quality lever in the pipeline. The
instructions here are tuned to the DOWNSTREAM ingestion parser's real behaviour
(``src/ingestion/parsers/markdown_it_parser.py``), verified against source:

  * Raw block-level HTML is silently DROPPED -> forbid HTML tables/blocks; force
    GFM pipe tables even for merged cells (flatten by repeating values).
  * Fenced code / tables NESTED INSIDE a list item are silently DROPPED -> for
    numbered procedures, emit the command/table as a TOP-LEVEL block after the
    step, never indented under it.
  * Content before the first heading is DROPPED -> headings carry the content.
  * Fenced ``` blocks with a language hint earn code metadata -> always fence
    CLI/console/config, never inline backticks or 4-space indentation.

Figure handling is configurable (``figures`` = enriched | minimal | skip); the
default is "enriched" because this corpus feeds a retrieval system and the source
guides are screenshot-dense.
"""

from __future__ import annotations

from typing import Literal

FiguresMode = Literal["enriched", "minimal", "skip"]

# Bump when the prompt text changes: it is folded into the page-artifact cache key
# so a prompt change invalidates cached pages instead of serving stale output.
PROMPT_VERSION = "3"

# --- Figure instruction variants -------------------------------------------------

_FIGURE_ENRICHED = (
    "6. FIGURES: For every figure, screenshot, UI capture, or diagram, output an "
    "italic bracketed description on its own line: *[Figure: ...]*. Make it "
    "SEARCHABLE: transcribe verbatim any legible on-screen text -- menu paths, "
    "field and column labels, button text, tab names, dialog titles, and the "
    "concrete values shown. For a diagram, name the components and the labelled "
    "relationships between them. Describe only what you can actually read or "
    "clearly see; never invent labels, numbers, or detail that is not legible."
)

_FIGURE_MINIMAL = (
    "6. FIGURES: For every figure, screenshot, or diagram, output only "
    "*[Figure: brief factual description of what is visibly shown]* on its own "
    "line. Do not fabricate any detail you cannot clearly see."
)

_FIGURE_SKIP = (
    "6. FIGURES: Omit figures, screenshots, and decorative images entirely. Still "
    "transcribe any real heading, paragraph, table, or code that sits near them."
)

_FIGURE_BY_MODE: dict[str, str] = {
    "enriched": _FIGURE_ENRICHED,
    "minimal": _FIGURE_MINIMAL,
    "skip": _FIGURE_SKIP,
}


def system_prompt(figures: FiguresMode = "enriched") -> str:
    """Build the system prompt for a given figure-handling mode."""

    figure_rule = _FIGURE_BY_MODE.get(figures, _FIGURE_ENRICHED)
    return (
        "You convert a single page image from Nutanix product documentation into "
        "faithful GitHub-Flavored Markdown (GFM). Transcribe exactly what is on the "
        "page. Never invent, summarize, paraphrase, infer, or omit content.\n"
        "\n"
        "Rules:\n"
        "1. HEADINGS: Preserve the heading hierarchy using ATX markers (#, ##, ###). "
        "Use a level-1 heading (#) ONLY for a document's main title, and only on the "
        "page where that title first appears; use ## and ### for sections and "
        "subsections. Do not invent or repeat a title on continuation pages.\n"
        "2. TABLES: Reproduce every table as a GFM pipe table. NEVER emit an HTML "
        "table or any raw HTML -- HTML is discarded downstream and the data would be "
        "lost. If a table has merged or spanning cells, flatten them into GFM by "
        "repeating the spanned value across each affected cell so that no information "
        "is lost.\n"
        "3. CODE & COMMANDS: Put all CLI, console, config-file, and code content in "
        "fenced code blocks (triple backticks) with a language hint. Use `bash` for "
        "shell / nCLI / aCLI (e.g. lines like `nutanix@cvm$ ncli ...` or "
        "`<acropolis> net.list`), `powershell` for PowerShell, and `text` for command "
        "output, REPL sessions, or unknown syntax. Never represent a command with "
        "inline single backticks or with indentation alone. Fence EVERY command -- "
        "including a lone single-line command and each command in a command-reference "
        "list or table; never leave a command as bare unfenced text. Always CLOSE "
        "each fence with ``` before the next heading or paragraph. When a page lists "
        "Description/Command pairs, write the description as prose and put the command "
        "in its own fenced block immediately after it.\n"
        "4. PROCEDURES: When a numbered or bulleted step contains a command, code "
        "block, or table, write the step's prose as the list item, preserving the "
        "visible ordinal in prose (for example, `Step 2: Configure the host`). Then "
        "place the fenced code block or table as a TOP-LEVEL block immediately AFTER "
        "that prose -- never indented inside it. (Indented code and tables inside list "
        "items are discarded downstream.)\n"
        "5. CALLOUTS: Render Note / Tip / Important / Caution / Warning admonitions as "
        "blockquotes that begin with a bold label, e.g. `> **Note:** ...` or "
        "`> **Warning:** ...`.\n"
        f"{figure_rule}\n"
        "7. PAGE FURNITURE: Skip running headers, footers, page numbers, and "
        "watermarks -- e.g. a repeated band like 'AHV | Host Network Management | 60'. "
        "Do not transcribe them.\n"
        "8. CONTINUATION: If the previous-page context provided by the user ends in "
        "the middle of a structure (an open table, list, or code block), continue that "
        "same structure on this page -- do not restart its header row or re-open its "
        "code fence.\n"
        "9. LISTS: Keep list markers consistent ('1.' for ordered, '-' for unordered). "
        "Only paragraphs and nested lists survive inside a list item downstream, so "
        "keep list items to prose plus nested lists.\n"
        "10. OUTPUT: Emit ONLY the Markdown for THIS page. No preamble, no commentary, "
        "no explanation, and do NOT wrap the whole page in a code fence."
    )


def build_messages(
    *,
    image_data_url: str,
    page_no: int,
    total_pages: int,
    prev_tail: str | None,
    anchor_text: str | None = None,
    figures: FiguresMode = "enriched",
) -> list[dict]:
    """Assemble the OpenAI ``messages`` array for one page request.

    ``prev_tail`` is the trailing slice of the PREVIOUS page's generated Markdown
    (or a text-layer fallback) used purely for cross-page structural continuity.
    ``anchor_text`` is THIS page's born-digital text layer, used only to
    disambiguate glyphs visible in the page image.
    """

    parts: list[str] = []
    if page_no <= 1:
        parts.append(
            f"This is page 1 of {total_pages} of the document. "
            "Transcribe it into Markdown per the rules."
        )
    else:
        parts.append(
            f"Continuing the same document: this is page {page_no} of {total_pages}."
        )
        if prev_tail:
            parts.append(
                "For continuity only (do NOT repeat it), the Markdown produced for "
                "the previous page ended as follows:\n"
                "<<<PREV_PAGE_TAIL\n"
                f"{prev_tail.strip()}\n"
                "PREV_PAGE_TAIL>>>"
            )
        parts.append("Now transcribe THIS page per the rules.")

    if anchor_text:
        parts.append(
            "Born-digital text layer for THIS page, provided as a reference only. "
            "Use this text only to disambiguate glyphs visible in the image; do not "
            "invent text that is not present in the image or this reference. If the "
            "image and reference disagree, the image is authoritative.\n"
            "<<<PAGE_TEXT_ANCHOR\n"
            f"{anchor_text.strip()}\n"
            "PAGE_TEXT_ANCHOR>>>"
        )

    user_text = "\n\n".join(parts)

    return [
        {"role": "system", "content": system_prompt(figures)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
