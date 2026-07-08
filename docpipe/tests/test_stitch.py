from docpipe.stitch import _merge_seam, stitch_pages, strip_page_furniture


def test_dehyphenation_keeps_real_hyphen():
    # Finding #40: at a page seam a trailing hyphen is far more likely a real
    # compound than a soft-wrap, so keep it rather than corrupt the term.
    assert (
        _merge_seam("Configure the read-", "only volume.")
        == "Configure the read-only volume."
    )
    assert "highavailability" not in _merge_seam("Enable high-", "availability mode.")
    # fragments are still joined onto one line (not split into two paragraphs)
    assert "\n\n" not in _merge_seam("Set the con-", "figuration value.")


def test_open_fence_seam_joins_with_newline():
    # Finding #31: commands continued inside an OPEN fence across a page seam must
    # not be prose-welded onto one line.
    merged = _merge_seam("```bash\nncli storage list", "ncli storage get\n```")
    assert "ncli storage list ncli storage get" not in merged  # not welded
    assert "ncli storage list\nncli storage get" in merged  # separate lines
    assert merged.count("```") == 2


def test_closed_fence_reopen_preserves_separate_blocks_and_languages():
    acc = "```bash\nncli storage list\n```"
    nxt = "```text\nName    Size\npool0   10TB\n```"

    merged = _merge_seam(acc, nxt)

    assert "```bash\nncli storage list\n```" in merged
    assert "```text\nName    Size\npool0   10TB\n```" in merged
    assert merged.count("```") == 4  # two separate fenced blocks


def test_table_header_dedup_across_seam():
    acc = "| A | B |\n|---|---|\n| 1 | 2 |"
    nxt = "| A | B |\n|---|---|\n| 3 | 4 |"
    merged = _merge_seam(acc, nxt)
    assert merged.count("| A | B |") == 1
    assert "| 1 | 2 |" in merged and "| 3 | 4 |" in merged


def test_table_header_dedup_keeps_different_next_table_header():
    acc = "| Command | Description |\n|---|---|\n| ncli | old |"
    nxt = "| Flag | Meaning |\n|---|---|\n| -v | verbose |"

    merged = _merge_seam(acc, nxt)

    assert "| Command | Description |" in merged
    assert "| Flag | Meaning |" in merged
    assert merged.count("|---|---|") == 2
    assert "| ncli | old |\n\n| Flag | Meaning |" in merged


def test_split_table_continuation_rows_join_without_blank_line():
    acc = "| Node | RAM |\n|---|---|\n| a | 64 |"
    nxt = "| b | 128 |\n| c | 256 |"

    merged = _merge_seam(acc, nxt)

    assert "| a | 64 |\n| b | 128 |\n| c | 256 |" in merged
    assert "| a | 64 |\n\n| b | 128 |" not in merged


def test_paragraph_join_midsentence():
    assert (
        _merge_seam("This sentence continues", "onto the next page.")
        == "This sentence continues onto the next page."
    )


def test_heading_boundary_not_joined():
    # A heading on the next page must remain its own block.
    merged = _merge_seam("Some prose that trails off", "## Next Section")
    assert "## Next Section" in merged
    assert merged.endswith("## Next Section")
    assert "\n\n## Next Section" in merged


def test_furniture_and_page_numbers_stripped():
    pages = [
        "Nutanix AHV Guide\n# Host Networking\nReal body one.\nAHV | Networking | 11",
        "Nutanix AHV Guide\n## Details\nReal body two.\nAHV | Networking | 12",
        "Nutanix AHV Guide\n## More\nReal body three.\nAHV | Networking | 13",
        "Nutanix AHV Guide\n## End\nReal body four.\nAHV | Networking | 14",
    ]
    cleaned = strip_page_furniture(pages)
    joined = "\n".join(cleaned)
    assert "Nutanix AHV Guide" not in joined  # repeated header removed
    assert "AHV | Networking |" not in joined  # footer band removed
    assert "Real body one." in joined  # real content kept
    assert "# Host Networking" in joined


def test_numeric_table_rows_are_not_stripped_as_footer_furniture():
    pages = [
        "# Capacity\n\n| Metric | Scope | Value |\n| --- | --- | --- |\nReplication factor | Minimum | 2\n| Max nodes | Cluster | 32\nAHV | Networking | 11",
        "# Ports\n\nPort | Protocol | 9440\nAHV | Networking | 12",
    ]

    cleaned = strip_page_furniture(pages)
    joined = "\n".join(cleaned)

    assert "Replication factor | Minimum | 2" in joined
    assert "| Max nodes | Cluster | 32" in joined
    assert "Port | Protocol | 9440" in joined
    assert "AHV | Networking |" not in joined


def test_single_page_numeric_table_rows_are_not_stripped_as_footer_furniture():
    pages = [
        "# Capacity\n\n| Metric | Scope | Value |\n| --- | --- | --- |\nReplication factor | Minimum | 2\nAHV | Networking | 11"
    ]

    cleaned = strip_page_furniture(pages)

    assert "Replication factor | Minimum | 2" in cleaned[0]
    assert "AHV | Networking | 11" in cleaned[0]


def test_stitch_pages_end_to_end():
    out = stitch_pages(["# Title\n\nIntro para", "## Section\n\nBody"])
    assert out.startswith("# Title")
    assert "## Section" in out
    assert out.endswith("\n")
