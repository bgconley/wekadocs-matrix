from pathlib import Path

from docpipe import artifacts
from docpipe.manifest import DocRecord
from docpipe.validate import document_coverage


def _rec() -> DocRecord:
    return DocRecord(
        pdf_path="sample.pdf",
        rel_path="sample.pdf",
        sha256="deadbeef",
        size_bytes=10,
        page_count=1,
        slug="sample",
        title=None,
        doc_version="7.5",
        product="AOS",
    )


def _write_ok(work_dir: Path) -> None:
    artifacts.write_success(
        work_dir,
        sha256="deadbeef",
        page_no=1,
        dpi=200,
        model_id="model",
        markdown="real page content",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )


def test_is_done_rejects_char_len_mismatch(tmp_path):
    _write_ok(tmp_path)
    md = artifacts.md_path(tmp_path, "deadbeef", 1, 200, "model")
    md.write_text("", encoding="utf-8")

    assert artifacts.read_md(tmp_path, "deadbeef", 1, 200, "model") is None
    assert not artifacts.is_done(tmp_path, "deadbeef", 1, 200, "model")


def test_document_coverage_requires_ok_markdown_file(tmp_path):
    _write_ok(tmp_path)
    artifacts.md_path(tmp_path, "deadbeef", 1, 200, "model").unlink()

    cov = document_coverage(tmp_path, _rec(), 200, "model")

    assert cov.ok_pages == []
    assert cov.missing_pages == [1]
    assert not cov.complete


def test_empty_ok_markdown_is_not_done(tmp_path):
    artifacts.write_success(
        tmp_path,
        sha256="deadbeef",
        page_no=1,
        dpi=200,
        model_id="model",
        markdown="",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )

    assert artifacts.read_md(tmp_path, "deadbeef", 1, 200, "model") is None
    assert not artifacts.is_done(tmp_path, "deadbeef", 1, 200, "model")
    assert document_coverage(tmp_path, _rec(), 200, "model").missing_pages == [1]


def test_discover_keysets_keeps_prompt_version_dimension(tmp_path, monkeypatch):
    model_id = "qwen36-27b-fp8-oxcart"
    monkeypatch.setattr(artifacts, "PROMPT_VERSION", "2")
    artifacts.write_success(
        tmp_path,
        sha256="deadbeef",
        page_no=1,
        dpi=200,
        model_id=model_id,
        markdown="prompt two content",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )
    monkeypatch.setattr(artifacts, "PROMPT_VERSION", "3")
    artifacts.write_success(
        tmp_path,
        sha256="deadbeef",
        page_no=1,
        dpi=200,
        model_id=model_id,
        markdown="prompt three content",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )

    assert artifacts.discover_keysets(tmp_path, "deadbeef") == [
        (200, model_id, "2"),
        (200, model_id, "3"),
    ]
    assert document_coverage(
        tmp_path, _rec(), 200, model_id, prompt_version="2"
    ).ok_pages == [1]
