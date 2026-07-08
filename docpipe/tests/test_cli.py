import argparse
import json
from pathlib import Path

import pytest

from docpipe import artifacts, cli
from docpipe.config import default_config
from docpipe.manifest import DocRecord, write_manifest


def _rec(tmp_path: Path) -> DocRecord:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    return DocRecord(
        pdf_path=str(pdf),
        rel_path="sample.pdf",
        sha256="deadbeef",
        size_bytes=pdf.stat().st_size,
        page_count=1,
        slug="sample",
        title=None,
        doc_version="7.5",
        product="AOS",
    )


def _args(**overrides):
    defaults = dict(
        config=None,
        in_dir=None,
        out=None,
        work=None,
        dpi=None,
        figures=None,
        model=None,
        endpoint=None,
        only=None,
        limit=None,
        allow_incomplete=False,
        json=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_endpoint_selection_rejects_empty_names():
    with pytest.raises(SystemExit, match="no endpoints selected"):
        cli._load_config(_args(endpoint=","))


def test_select_records_limit_zero_selects_no_records():
    records = [object(), object()]

    assert cli._select_records(records, None, 0) == []


@pytest.mark.asyncio
async def test_run_pipeline_rejects_zero_active_endpoints_before_pool(
    tmp_path, monkeypatch
):
    cfg = default_config()
    cfg.work_dir = str(tmp_path / "work")
    cfg.input_dir = str(tmp_path / "input")
    for endpoint in cfg.endpoints:
        endpoint.enabled = False
    rec = _rec(tmp_path)
    write_manifest(Path(cfg.work_dir), [rec])

    class FailIfConstructed:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("VLMPool should not be constructed")

    monkeypatch.setattr(cli, "VLMPool", FailIfConstructed)

    with pytest.raises(SystemExit, match="no active endpoints"):
        await cli._run_pipeline(cfg, _args(), rebuild_manifest=False)


def test_status_json_uses_discovered_prompt_version_and_full_model_id(
    tmp_path, monkeypatch, capsys
):
    work = tmp_path / "work"
    rec = _rec(tmp_path)
    write_manifest(work, [rec])
    model_id = "qwen36-27b-fp8-oxcart"

    monkeypatch.setattr(artifacts, "PROMPT_VERSION", "2")
    artifacts.write_success(
        work,
        sha256=rec.sha256,
        page_no=1,
        dpi=200,
        model_id=model_id,
        markdown="real page content",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )
    monkeypatch.setattr(artifacts, "PROMPT_VERSION", "3")

    assert cli.cmd_status(_args(work=str(work), json=True)) == 0

    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "slug": "sample",
            "pages": 1,
            "ok": 1,
            "failed": 0,
            "missing": 0,
            "dpi": 200,
            "model": model_id,
            "prompt_version": "2",
        }
    ]


def test_status_text_shows_full_model_id(tmp_path, capsys):
    work = tmp_path / "work"
    rec = _rec(tmp_path)
    write_manifest(work, [rec])
    model_id = "qwen36-27b-fp8-oxcart"
    artifacts.write_success(
        work,
        sha256=rec.sha256,
        page_no=1,
        dpi=200,
        model_id=model_id,
        markdown="real page content",
        endpoint="oxcart",
        image_tokens=None,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=1,
    )

    assert cli.cmd_status(_args(work=str(work))) == 0

    out = capsys.readouterr().out
    assert "model" in out.splitlines()[0]
    assert model_id in out
