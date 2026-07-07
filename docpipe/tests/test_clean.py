from dataclasses import replace

from docpipe.clean import (
    RunMeta,
    balance_fences,
    clean_document,
    demote_extra_h1s,
    ensure_title_h1,
    first_h1_text,
    resolve_title,
)
from docpipe.manifest import DocRecord

_BASE = DocRecord(
    pdf_path="x.pdf",
    rel_path="x.pdf",
    sha256="deadbeef",
    size_bytes=10,
    page_count=2,
    slug="ahv-admin-guide",
    title=None,
    doc_version="7.5",
    product="AOS",
)


def _rec(**kw) -> DocRecord:
    return replace(_BASE, **kw)


def test_prepend_h1_when_missing():
    out = ensure_title_h1("some text\n## Section", "My Title")
    assert out.startswith("# My Title\n")
    assert "## Section" in out


def test_keep_existing_leading_h1():
    out = ensure_title_h1("# Existing\n\nbody", "Ignored")
    assert out.startswith("# Existing")
    assert "# Ignored" not in out


def test_single_h1_enforced():
    assert demote_extra_h1s("# A\n# B\n# C") == "# A\n## B\n## C"


def test_resolve_title_prefers_body_h1():
    assert (
        resolve_title("# NFS Networking\n\nx", _rec(title="Embedded Title"))
        == "NFS Networking"
    )
    assert (
        resolve_title("no heading here", _rec(title="Embedded Title"))
        == "Embedded Title"
    )
    assert resolve_title("no heading", _rec(title=None)) == "Ahv Admin Guide"


def test_balance_fences_closes_before_heading():
    out = balance_fences("## A\n\n```text\nsome code\n## B\n\nmore")
    assert out.count("```") == 2
    lines = out.splitlines()
    assert "```" in lines[: lines.index("## B")]  # closed before B


def test_balance_fences_closes_at_eof():
    out = balance_fences("intro\n```bash\ncmd")
    assert out.count("```") == 2
    assert out.rstrip().endswith("```")


def test_balance_fences_ignores_shell_comment():
    body = "```bash\n# a shell comment\nls\n```\n\ntext"
    assert balance_fences(body) == body  # single-# is not a heading; nothing changes


# --- Cluster 1: fence-awareness (findings #2, #3, #22) -------------------------


def test_demote_extra_h1s_skips_fenced_comments():
    # Finding #2: a '#' comment inside a code fence must NOT be demoted to '##'
    # (which would leak the command out of the block and inject a false heading).
    body = "# Title\n\n```bash\n# a comment\ncmd\n```\n\n# Second"
    out = demote_extra_h1s(body).splitlines()
    assert "# a comment" in out  # unchanged, still fenced content
    assert "## a comment" not in out
    assert "## Second" in out  # a real second H1 is still demoted


def test_balance_fences_ignores_h2_comment_in_balanced_block():
    # Finding #3: a '##' comment inside a properly-closed fence must not trigger
    # a premature close. Balanced (even) fence count => leave the doc untouched.
    body = (
        "## Section\n\n```bash\n## configure interface\n"
        "ip addr add 1.2.3.4/24 dev eth0\n```\n"
    )
    assert balance_fences(body) == body


def test_first_h1_text_skips_fenced_comment():
    # Finding #22: a '#' comment in a leading code block must not become the title.
    body = (
        "```bash\n# Deploy the production cluster\n"
        "ncli cluster create name=prod\n```\n\n## Overview\nbody"
    )
    assert first_h1_text(body) is None
    assert first_h1_text("# Real Title\n\n```bash\n# c\n```") == "Real Title"


def test_clean_document_preserves_fenced_comments_end_to_end():
    # Finding #2/#3 together, through the real clean_document pipeline order
    # (demote_extra_h1s -> balance_fences): the fenced block stays intact.
    body = (
        "## Host Networking\n\n```bash\n# create the vlan-backed network\n"
        "acli net.create vlan0 vlan=0\n# verify it\nacli net.list\n```\n"
    )
    md, _ = clean_document(body, _rec(), RunMeta(model_id="m", endpoint="e", dpi=200))
    assert md.count("```") == 2  # exactly one intact block
    assert "# create the vlan-backed network" in md  # comment kept as-is
    assert "## create the vlan-backed network" not in md
    assert "acli net.create vlan0 vlan=0" in md  # command still inside code


def test_clean_document_contract():
    md, title = clean_document(
        "# Hello World\n\nbody\n\n\n\n\nmore",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="blackbird", dpi=200),
    )
    assert title == "Hello World"
    assert md.startswith("---\n")  # front matter at byte 0
    assert 'title: "Hello World"' in md
    assert 'version: "7.5"' in md
    assert 'sha256: "deadbeef"' in md
    assert "pipeline_version:" in md
    body = md.split("---\n", 2)[2]
    assert body.lstrip().startswith("# Hello World")  # body opens with the single H1
    assert sum(1 for ln in md.splitlines() if ln.startswith("# ")) == 1
    assert "\n\n\n" not in md  # blank runs collapsed
