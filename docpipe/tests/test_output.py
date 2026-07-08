from pathlib import Path

from docpipe import artifacts
from docpipe.config import default_config
from docpipe.manifest import DocRecord
from docpipe.output import DocResult, assemble_all, assemble_document


def _cfg(tmp_path: Path):
    cfg = default_config()
    cfg.work_dir = str(tmp_path / "work")
    cfg.output.dir = str(tmp_path / "out")
    return cfg


def _rec(tmp_path: Path, slug: str, sha256: str = "deadbeef") -> DocRecord:
    pdf = tmp_path / f"{slug}.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    return DocRecord(
        pdf_path=str(pdf),
        rel_path=f"{slug}.pdf",
        sha256=sha256,
        size_bytes=pdf.stat().st_size,
        page_count=1,
        slug=slug,
        title=None,
        doc_version="7.5",
        product="AOS",
    )


def _write_ok_page(cfg, rec: DocRecord, markdown: str = "# Page\n\nbody") -> None:
    artifacts.write_success(
        Path(cfg.work_dir),
        sha256=rec.sha256,
        page_no=1,
        dpi=cfg.rasterize.dpi,
        model_id="model",
        markdown=markdown,
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )


def test_assemble_document_tolerates_page_text_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    rec = _rec(tmp_path, "good")
    _write_ok_page(cfg, rec)

    def fail_page_text(pdf_path: str, page_no: int) -> str:
        raise RuntimeError("source pdf moved")

    monkeypatch.setattr("docpipe.output.page_text", fail_page_text)

    result = assemble_document(cfg, rec, "model")

    assert result.written is True
    assert result.out_path is not None
    assert Path(result.out_path).is_file()


def test_assemble_all_isolates_single_document_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    bad = _rec(tmp_path, "bad", "badbad")
    good = _rec(tmp_path, "good", "goodgood")

    def fake_assemble_document(config, record, model_id, **kwargs):
        if record.slug == "bad":
            raise RuntimeError("write failed")
        return DocResult(
            sha256=record.sha256,
            slug=record.slug,
            page_count=record.page_count,
            ok_count=record.page_count,
            complete=True,
            written=True,
        )

    monkeypatch.setattr("docpipe.output.assemble_document", fake_assemble_document)

    results = assemble_all(cfg, [bad, good], "model")

    assert [result.slug for result in results] == ["bad", "good"]
    assert results[0].written is False
    assert results[0].failed_pages == [1]
    assert results[1].written is True
