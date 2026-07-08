import pytest

from docpipe.validate import assess_page, token_overlap


def test_empty_page_flagged():
    assert assess_page("   \n  ").flags == ["empty"]


def test_repetition_flagged():
    qa = assess_page("\n".join(["the same line here"] * 10))
    assert any(f.startswith("garbled") for f in qa.flags)
    assert qa.suspect


def test_consecutive_loop_flagged():
    # 10 distinct lines keep the repeated block under the 50% repetition trigger,
    # so the consecutive-run (loop) detector is what should fire.
    body = "\n".join(
        [f"distinct line number {i}" for i in range(10)] + ["dup dup dup"] * 8
    )
    flags = assess_page(body).flags
    assert "garbled:loop" in flags
    assert "garbled:repetition" not in flags


def test_clean_page_not_suspect():
    md = "# Heading\n\nThis is a normal paragraph with several distinct words.\n\n- one\n- two"
    qa = assess_page(md)
    assert not qa.suspect
    assert "empty" not in qa.flags


def test_ascii_table_not_flagged_low_alnum():
    table = """
+----------------------+----------------------+
| Field                | Value                |
+----------------------+----------------------+
| Nutanix CVM address  | 10.10.10.10          |
| acli command         | <acropolis> net.list |
+----------------------+----------------------+
"""

    assert "garbled:low_alnum" not in assess_page(table).flags


def test_symbol_noise_still_flagged_low_alnum():
    noise = "\n".join(["@@@@ !!!! ???? ~~~~"] * 4)

    assert "garbled:low_alnum" in assess_page(noise).flags


def test_short_vs_textlayer():
    text_layer = "word " * 200  # long real text layer
    qa = assess_page("# H\n\ntiny", text_layer)
    assert "short_vs_textlayer" in qa.flags


def test_token_overlap():
    assert token_overlap(
        "the quick brown fox", "the quick brown fox jumps"
    ) == pytest.approx(0.8)
    assert token_overlap("anything", "") is None
