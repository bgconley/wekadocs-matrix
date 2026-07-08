from docpipe.contract import validate_contract

GOOD_MD = """---
title: Good Title
version: "7.5"
---

# Good Title

## Commands

| Command | Purpose |
|---|---|
| ncli cluster get | Inspect the cluster |

```bash
ncli cluster get
```
"""


def _codes(markdown: str) -> set[str]:
    return {violation.code for violation in validate_contract(markdown).violations}


def test_contract_accepts_corpus_ready_markdown():
    assert validate_contract(GOOD_MD).ok


def test_contract_requires_frontmatter_at_byte_zero_and_title():
    leading_blank = "\n" + GOOD_MD
    missing_title = GOOD_MD.replace("title: Good Title\n", "")

    assert "frontmatter:missing" in _codes(leading_blank)
    assert "frontmatter:title" in _codes(missing_title)


def test_contract_rejects_extra_h1_and_heading_level_jump():
    extra_h1 = GOOD_MD + "\n# Another Title\n"
    skipped_h2 = GOOD_MD.replace("## Commands", "### Commands")

    assert "heading:h1_count" in _codes(extra_h1)
    assert "heading:level_jump" in _codes(skipped_h2)


def test_contract_rejects_true_setext_headings():
    setext_h2 = GOOD_MD.replace("## Commands", "Commands\n---")
    setext_h1 = GOOD_MD.replace("## Commands", "Commands\n===")

    assert "heading:setext" in _codes(setext_h2)
    assert "heading:setext" in _codes(setext_h1)


def test_contract_allows_parser_non_setext_rule_lines_after_blank():
    horizontal_rule = GOOD_MD.replace("## Commands", "## Commands\n\nDivider\n\n---")
    equals_paragraph = GOOD_MD.replace("## Commands", "## Commands\n\nDivider\n\n===")

    assert "heading:setext" not in _codes(horizontal_rule)
    assert "heading:setext" not in _codes(equals_paragraph)


def test_contract_rejects_orphan_content_before_first_heading():
    orphan = GOOD_MD.replace("# Good Title", "orphan intro\n\n# Good Title")

    assert "body:pre_heading_content" in _codes(orphan)


def test_contract_rejects_nested_blocks_raw_html_and_unbalanced_fences():
    nested_fence = GOOD_MD.replace("```bash", "  ```bash")
    nested_table = GOOD_MD.replace("| Command | Purpose |", "  | Command | Purpose |")
    raw_html = GOOD_MD.replace("## Commands", "## Commands\n\n<table><tr></tr></table>")
    unbalanced = GOOD_MD + "\n```bash\nncli cluster get\n"

    assert "code:not_top_level" in _codes(nested_fence)
    assert "table:not_top_level" in _codes(nested_table)
    assert "html:raw_table_or_code" in _codes(raw_html)
    assert "fence:unbalanced" in _codes(unbalanced)
