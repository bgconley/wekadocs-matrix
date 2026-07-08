from pathlib import Path

import pytest

from docpipe import artifacts
from docpipe.config import default_config
from docpipe.convert import Converter
from docpipe.manifest import DocRecord
from docpipe.vlm_client import ChatResult, EndpointStats


def _rec(tmp_path: Path) -> DocRecord:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    return DocRecord(
        pdf_path=str(pdf),
        rel_path="sample.pdf",
        sha256="deadbeef",
        size_bytes=10,
        page_count=1,
        slug="sample",
        title=None,
        doc_version="7.5",
        product="AOS",
    )


def _cfg(tmp_path: Path):
    cfg = default_config()
    cfg.work_dir = str(tmp_path / "work")
    cfg.convert.max_retries = 1
    cfg.convert.backoff_base_s = 0
    for ep in cfg.endpoints:
        ep.enabled = ep.name == "oxcart"
        ep.inflight = 1
    return cfg


def _result(content: str, finish_reason: str = "stop") -> ChatResult:
    return ChatResult(
        content=content,
        endpoint="oxcart",
        finish_reason=finish_reason,
        image_tokens=10,
        prompt_tokens=20,
        completion_tokens=30,
        latency_s=0.01,
    )


class FakePool:
    def __init__(self, results: list[ChatResult]):
        self.results = list(results)
        self.calls: list[int | None] = []

    async def chat(self, endpoint: str, messages: list[dict], *, max_tokens=None):
        self.calls.append(max_tokens)
        assert endpoint == "oxcart"
        assert messages
        return self.results.pop(0)

    def stats(self):
        return {"oxcart": EndpointStats(requests=len(self.calls))}


@pytest.mark.asyncio
async def test_empty_model_response_is_failed_not_cached_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    pool = FakePool([_result("")])
    conv = Converter(cfg, pool, "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 0
    assert summary.failed == 1
    meta = artifacts.read_meta(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert meta is not None
    assert meta.status == "failed"
    assert not artifacts.md_path(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    ).exists()


@pytest.mark.asyncio
async def test_still_truncated_bumped_response_is_failed_not_cached_ok(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    pool = FakePool([_result("cut one", "length"), _result("cut two", "length")])
    conv = Converter(cfg, pool, "model")

    summary = await conv.run([_rec(tmp_path)])

    assert pool.calls == [None, 9600]
    assert summary.truncated_bumps == 1
    assert summary.converted == 0
    assert summary.failed == 1
    meta = artifacts.read_meta(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert meta is not None
    assert meta.status == "failed"
    assert "truncated" in (meta.error or "")
