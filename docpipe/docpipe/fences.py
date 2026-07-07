"""Shared fenced-code-block state tracking.

Several clean/stitch passes must know whether a line sits INSIDE a ``` fenced
code block, so that a shell/config comment like ``# do this`` or ``## section``
is treated as code content rather than as a Markdown heading. Centralizing the
fence-state logic here keeps every consumer consistent (findings #2, #3, #22,
#31, #40 all stemmed from passes that lacked it).
"""

from __future__ import annotations

import re
from typing import Iterator

# A fence delimiter: a line that starts (ignoring leading whitespace) with ```.
_FENCE_RE = re.compile(r"^\s*```")


def is_fence_line(line: str) -> bool:
    return bool(_FENCE_RE.match(line))


def iter_lines_with_fence_state(text: str) -> Iterator[tuple[str, bool]]:
    """Yield ``(line, in_code)`` for each line of ``text``.

    ``in_code`` is True only for the CONTENT lines strictly inside an open ```
    fence; the delimiter lines themselves report False (they are boundaries, not
    content). An unterminated fence keeps the trailing lines ``in_code=True``.
    """

    in_code = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_code = not in_code
            yield line, False
        else:
            yield line, in_code


def open_fence_at_end(text: str) -> bool:
    """True if ``text`` ends inside an unclosed fence (odd number of delimiters)."""

    return sum(1 for line in text.splitlines() if is_fence_line(line)) % 2 == 1
