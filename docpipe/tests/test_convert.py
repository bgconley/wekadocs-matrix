import asyncio
from pathlib import Path
from typing import cast

import pytest

from docpipe import artifacts
from docpipe.config import default_config
from docpipe.convert import Converter, PageJob
from docpipe.manifest import DocRecord
from docpipe.vlm_client import ChatResult, EndpointStats, VLMHTTPError, VLMPool


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


def _rec_pages(tmp_path: Path, page_count: int) -> DocRecord:
    rec = _rec(tmp_path)
    rec.page_count = page_count
    return rec


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


class MultiEndpointPool:
    def __init__(self, scripts: dict[str, list[ChatResult | Exception]]):
        self.scripts = {name: list(steps) for name, steps in scripts.items()}
        self.calls: list[tuple[str, int | None]] = []

    async def chat(self, endpoint: str, messages: list[dict], *, max_tokens=None):
        self.calls.append((endpoint, max_tokens))
        assert messages
        step = self.scripts[endpoint].pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def stats(self):
        return {
            name: EndpointStats(
                requests=sum(1 for endpoint, _ in self.calls if endpoint == name)
            )
            for name in self.scripts
        }


class RecordingPool(FakePool):
    def __init__(self, results: list[ChatResult]):
        super().__init__(results)
        self.messages: list[list[dict]] = []

    async def chat(self, endpoint: str, messages: list[dict], *, max_tokens=None):
        self.messages.append(messages)
        return await super().chat(endpoint, messages, max_tokens=max_tokens)


def _pool(fake: FakePool | ScriptedPool | MultiEndpointPool) -> VLMPool:
    return cast(VLMPool, fake)


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
    conv = Converter(cfg, _pool(pool), "model")

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
async def test_converter_keeps_only_latest_page_markdown_in_memory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    pool = FakePool([_result("# Page 1\n\nbody"), _result("# Page 2\n\nbody")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec_pages(tmp_path, 2)])

    assert summary.converted == 2
    assert set(conv._outputs) == {("deadbeef", 2)}


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
    conv = Converter(cfg, _pool(pool), "model")

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
    conv = Converter(cfg, _pool(FakePool([])), "model")

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
    conv = Converter(cfg, _pool(pool), "model")

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
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 1
    assert summary.failed == 0
    assert sleeps == [1.5]
    assert len(pool.calls) == 2


@pytest.mark.asyncio
async def test_worker_survives_failure_sidecar_write_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conv = Converter(cfg, _pool(FakePool([])), "model")
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


@pytest.mark.asyncio
async def test_truncated_retry_uses_higher_token_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_tokens = 14_000
    pool = FakePool([_result("cut one", "length"), _result("cut two", "length")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert pool.calls == [None, 22_400]
    assert summary.converted == 0
    assert summary.failed == 1


@pytest.mark.asyncio
async def test_shorter_truncation_retry_is_failed_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    first = "# good but cut\n" + ("preserved detail\n" * 20)
    pool = FakePool([_result(first, "length"), _result("# short", "stop")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert pool.calls == [None, 9600]
    assert summary.converted == 0
    assert summary.failed == 1
    assert not artifacts.md_path(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    ).exists()


@pytest.mark.asyncio
async def test_prev_tail_read_fault_falls_back_to_native_pdf_text(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    conv = Converter(cfg, _pool(FakePool([])), "model")
    conv.dpi = cfg.rasterize.dpi
    conv._loop = asyncio.get_running_loop()
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 2, 2)

    def fail_read_md(*args, **kwargs):
        raise OSError("sidecar unreadable")

    monkeypatch.setattr("docpipe.convert.artifacts.read_md", fail_read_md)
    monkeypatch.setattr(
        "docpipe.convert.page_text", lambda *args, **kwargs: "native previous page text"
    )

    assert await conv._prev_tail(job) == "native previous page text"


@pytest.mark.asyncio
async def test_prev_tail_starts_at_open_fence_boundary(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.convert.prev_tail_chars = 30
    conv = Converter(cfg, _pool(FakePool([])), "model")
    conv._loop = asyncio.get_running_loop()
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 2, 2)
    conv._outputs[(rec.sha256, 1)] = (
        "# Previous\n\n"
        + "\n".join(f"setup line {i}" for i in range(20))
        + "\n\n```bash\nncli cluster get\nncli storage list\n"
    )

    tail = await conv._prev_tail(job)

    assert tail is not None
    assert tail.startswith("```bash")
    assert "ncli cluster get" in tail
    assert len(tail) > cfg.convert.prev_tail_chars


@pytest.mark.asyncio
async def test_prev_tail_keeps_open_table_header(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.convert.prev_tail_chars = 45
    conv = Converter(cfg, _pool(FakePool([])), "model")
    conv._loop = asyncio.get_running_loop()
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 2, 2)
    conv._outputs[(rec.sha256, 1)] = (
        "# Previous\n\n"
        + "\n".join(f"prose line {i}" for i in range(12))
        + "\n\n| Name | Value |\n|---|---|\n| CVM memory | 32 GiB |\n| Disk count | 6 |\n"
    )

    tail = await conv._prev_tail(job)

    assert tail is not None
    assert tail.startswith("| Name | Value |")
    assert "|---|---|" in tail
    assert "| Disk count | 6 |" in tail


@pytest.mark.asyncio
async def test_prev_tail_keeps_dangling_list_boundary(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.convert.prev_tail_chars = 40
    conv = Converter(cfg, _pool(FakePool([])), "model")
    conv._loop = asyncio.get_running_loop()
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 2, 2)
    conv._outputs[(rec.sha256, 1)] = (
        "Intro\n\n"
        + "\n".join(f"background {i}" for i in range(15))
        + "\n\n1. Open Prism Element.\n2. Select the VM.\n3. Review the alerts.\n"
    )

    tail = await conv._prev_tail(job)

    assert tail is not None
    assert tail.startswith("1. Open Prism Element.")
    assert "3. Review the alerts." in tail


@pytest.mark.asyncio
async def test_convert_injects_current_page_text_layer_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    monkeypatch.setattr(
        "docpipe.convert.page_text",
        lambda *args, **kwargs: "ANCHOR_ONLY_CLI ncli cluster get --redundancy-factor",
    )
    cfg = _cfg(tmp_path)
    pool = RecordingPool([_result("# ok\n\nANCHOR_ONLY_CLI")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 1
    user_text = pool.messages[0][1]["content"][0]["text"]
    assert "ANCHOR_ONLY_CLI ncli cluster get --redundancy-factor" in user_text
    assert "Use this text only to disambiguate glyphs" in user_text


@pytest.mark.asyncio
async def test_low_text_layer_overlap_retries_before_caching(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    anchor = " ".join(f"anchorword{i}" for i in range(80))
    monkeypatch.setattr("docpipe.convert.page_text", lambda *args, **kwargs: anchor)
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 2
    pool = FakePool(
        [
            _result("# Wrong\n\nunrelated hallucinated content only"),
            _result("# Correct\n\n" + anchor),
        ]
    )
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert len(pool.calls) == 2
    assert summary.converted == 1
    stored = artifacts.read_md(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert stored is not None
    assert "anchorword79" in stored


@pytest.mark.asyncio
async def test_exhausted_low_overlap_can_use_pdf_table_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    anchor = " ".join(f"anchorword{i}" for i in range(80))
    monkeypatch.setattr("docpipe.convert.page_text", lambda *args, **kwargs: anchor)
    fallback = "Table 1: Parameters\n\n" + anchor

    import docpipe.pdf_tables as pdf_tables

    monkeypatch.setattr(
        pdf_tables,
        "page_key_value_table_markdown",
        lambda *args, **kwargs: fallback,
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 1
    pool = FakePool([_result("# Wrong\n\nunrelated hallucinated content only")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert len(pool.calls) == 1
    assert summary.converted == 1
    assert summary.failed == 0
    stored = artifacts.read_md(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert stored == fallback
    meta = artifacts.read_meta(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert meta is not None
    assert meta.endpoint == "pdf-text-fallback"
    assert meta.attempts == 1


@pytest.mark.asyncio
async def test_exhausted_garbled_page_can_use_text_layer_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    anchor = "Cluster Name Cluster A Cluster B\nStorage RF3 capable RF2 capable"
    monkeypatch.setattr("docpipe.convert.page_text", lambda *args, **kwargs: anchor)

    import docpipe.pdf_tables as pdf_tables

    monkeypatch.setattr(
        pdf_tables,
        "page_key_value_table_markdown",
        lambda *args, **kwargs: None,
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 1
    pool = FakePool([_result("# Wrong\n\n| key | " + ("loopword " * 8) + "|")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 1
    assert summary.failed == 0
    stored = artifacts.read_md(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert stored == f"```text\n{anchor}\n```"
    meta = artifacts.read_meta(
        Path(cfg.work_dir), "deadbeef", 1, cfg.rasterize.dpi, "model"
    )
    assert meta is not None
    assert meta.endpoint == "pdf-text-fallback"


@pytest.mark.asyncio
async def test_garbled_page_retries_with_escalated_raster_budget(tmp_path, monkeypatch):
    renders: list[tuple[int, int]] = []

    def fake_render(pdf_path, page_no, dpi, max_long_px):
        renders.append((dpi, max_long_px))
        return "data:image/png;base64,AAA="

    monkeypatch.setattr("docpipe.convert.render_page_data_url", fake_render)
    monkeypatch.setattr("docpipe.convert.page_text", lambda *args, **kwargs: "")
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 2
    cfg.rasterize.dpi = 218
    cfg.rasterize.max_long_px = 2408
    garbled = "\n".join(["DECODE LOOP"] * 8)
    pool = FakePool([_result(garbled), _result("# ok\n\nRecovered dense table")])
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert renders == [(218, 2408), (300, 3304)]
    assert summary.converted == 1
    assert summary.failed == 0
    stored = artifacts.read_md(Path(cfg.work_dir), "deadbeef", 1, 218, "model")
    assert stored is not None
    assert "Recovered dense table" in stored


@pytest.mark.asyncio
async def test_retry_backoff_requeues_without_blocking_worker_slot(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 2
    conv = Converter(cfg, _pool(FakePool([])), "model")
    queue: asyncio.Queue[PageJob] = asyncio.Queue()
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 1, rec.page_count)
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        sleep_started.set()
        await release_sleep.wait()

    monkeypatch.setattr("docpipe.convert.asyncio.sleep", fake_sleep)

    await asyncio.wait_for(
        conv._retry_or_fail(
            job, "oxcart", queue, _http_error(429, retryable=True, retry_after_s=1.5)
        ),
        timeout=0.05,
    )
    assert queue.empty()
    await asyncio.wait_for(sleep_started.wait(), timeout=0.05)
    assert sleeps == [1.5]

    release_sleep.set()
    queued = await asyncio.wait_for(queue.get(), timeout=0.05)
    assert queued is job


@pytest.mark.asyncio
async def test_retryable_endpoint_failure_can_fail_over_to_second_endpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 1
    for ep in cfg.endpoints:
        ep.enabled = ep.name in {"oxcart", "blackbird"}
        ep.inflight = 1
    pool = MultiEndpointPool(
        {
            "oxcart": [_http_error(503, retryable=True)],
            "blackbird": [_result("# ok")],
        }
    )
    conv = Converter(cfg, _pool(pool), "model")

    summary = await conv.run([_rec(tmp_path)])

    assert summary.converted == 1
    assert summary.failed == 0
    assert pool.calls == [("oxcart", None), ("blackbird", None)]


def test_retry_deferral_stops_after_every_endpoint_has_tried_page(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 4
    conv = Converter(cfg, _pool(FakePool([])), "model")
    conv._active_endpoint_names = {"oxcart", "blackbird"}
    rec = _rec(tmp_path)
    job = PageJob(rec.sha256, rec.pdf_path, 1, rec.page_count)
    job.failed_endpoints.update({"oxcart", "blackbird"})
    job.endpoint_attempts.update({"oxcart": 1, "blackbird": 1})

    assert not conv._should_defer_to_untried_endpoint(job, "oxcart")
    assert not conv._should_defer_to_untried_endpoint(job, "blackbird")


@pytest.mark.asyncio
async def test_retryable_endpoint_failures_do_not_livelock_after_all_endpoints_tried(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "docpipe.convert.render_page_data_url",
        lambda *args, **kwargs: "data:image/png;base64,AAA=",
    )
    cfg = _cfg(tmp_path)
    cfg.convert.max_retries = 2
    cfg.convert.backoff_base_s = 0
    for ep in cfg.endpoints:
        ep.enabled = ep.name in {"oxcart", "blackbird"}
        ep.inflight = 1
    pool = MultiEndpointPool(
        {
            "oxcart": [_http_error(503, retryable=True), _result("# ok")],
            "blackbird": [_http_error(503, retryable=True), _result("# ok")],
        }
    )
    conv = Converter(cfg, _pool(pool), "model")

    summary = await asyncio.wait_for(conv.run([_rec(tmp_path)]), timeout=0.2)

    assert summary.converted == 1
    assert summary.failed == 0
    assert len(pool.calls) == 3
    assert {endpoint for endpoint, _ in pool.calls[:2]} == {"oxcart", "blackbird"}
