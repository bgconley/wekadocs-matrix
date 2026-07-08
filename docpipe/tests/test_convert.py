import asyncio
from pathlib import Path

import pytest

from docpipe import artifacts
from docpipe.config import default_config
from docpipe.convert import Converter, PageJob
from docpipe.manifest import DocRecord
from docpipe.vlm_client import ChatResult, EndpointStats, VLMHTTPError


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


class ScriptedPool:
    def __init__(self, steps: list[ChatResult | Exception]):
        self.steps = list(steps)
        self.calls: list[int | None] = []

    async def chat(self, endpoint: str, messages: list[dict], *, max_tokens=None):
        self.calls.append(max_tokens)
        assert endpoint == "oxcart"
        assert messages
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def stats(self):
        return {"oxcart": EndpointStats(requests=len(self.calls))}


def _http_error(
    status: int, *, retryable: bool = True, retry_after_s: float | None = None
) -> VLMHTTPError:
    return VLMHTTPError(
        status,
        "http failure",
        retryable=retryable,
        retry_after_s=retry_after_s,
    )


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


@pytest.mark.asyncio
async def test_run_with_pending_pages_and_no_active_endpoints_fails_fast(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    for ep in cfg.endpoints:
        ep.enabled = False
    conv = Converter(cfg, FakePool([]), "model")

    with pytest.raises(RuntimeError, match="no active endpoints"):
        await asyncio.wait_for(conv.run([_rec(tmp_path)]), timeout=0.2)


@pytest.mark.asyncio
async def test_non_retryable_http_error_fails_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 4
    pool = ScriptedPool([_http_error(401, retryable=False)])
    conv = Converter(cfg, pool, "model")

    summary = await conv.run([_rec(tmp_path)])

    assert len(pool.calls) == 1
    assert summary.converted == 0
    assert summary.failed == 1
    meta = artifacts.read_meta(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert meta is not None
    assert meta.status == "failed"
    assert "non-retryable" in (meta.error or "")


@pytest.mark.asyncio
async def test_retryable_http_error_honors_retry_after_delay(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("docpipe.convert.asyncio.sleep", fake_sleep)
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 2
    cfg.convert.backoff_base_s = 0
    pool = ScriptedPool(
        [_http_error(429, retryable=True, retry_after_s=1.5), _result("# ok")]
    )
    conv = Converter(cfg, pool, "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 1
    assert summary.failed == 0
    assert sleeps == [1.5]
    assert len(pool.calls) == 2


@pytest.mark.asyncio
async def test_worker_survives_failure_sidecar_write_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conv = Converter(cfg, FakePool([]), "model")
    conv.dpi = cfg.rasterize.dpi
    conv._loop = asyncio.get_running_loop()
    queue: asyncio.Queue[PageJob] = asyncio.Queue()
    rec = _rec(tmp_path)
    queue.put_nowait(PageJob(rec.sha256, rec.pdf_path, 1, rec.page_count))
    queue.put_nowait(PageJob(rec.sha256, rec.pdf_path, 2, rec.page_count))

    async def fail_process(job, endpoint, queue):
        raise RuntimeError("boom")

    def fail_write_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(conv, "_process", fail_process)
    monkeypatch.setattr("docpipe.convert.artifacts.write_failure", fail_write_failure)

    worker = asyncio.create_task(conv._worker("w0", "oxcart", queue))
    try:
        await asyncio.wait_for(queue.join(), timeout=0.2)
        assert not worker.done()
        assert conv._summary.failed == 2
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
