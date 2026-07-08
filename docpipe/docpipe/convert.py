"""Stage 2 -- VLM page conversion (one page per request), concurrent and resumable.

Concurrency model: a single ``asyncio.Queue`` of page jobs is drained by a fixed
pool of workers, ``inflight`` of them bound to each endpoint. Because any free
worker grabs the next page, whichever endpoint has spare capacity does the next
page -- that IS least-outstanding-requests routing, and it keeps both endpoints
saturated with Blackbird (more workers) taking proportionally more. A page that
fails on one worker is re-enqueued, so a struggling endpoint's share simply drains
to the healthy one (cross-endpoint failover).

Resume is free: a page whose ``ok`` artifact already exists for this (dpi, model)
is never enqueued, and a re-run of ``run`` also re-attempts any page left in a
``failed`` state.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import artifacts
from .config import Config
from .log import get_logger
from .manifest import DocRecord
from .prompts import build_messages
from .rasterize import page_text, render_page_data_url
from .vlm_client import VLMError, VLMPool

logger = get_logger("convert")

ProgressFn = Callable[[int, int], None]


_TRUNCATION_BUMP_LIMIT = 262_144


@dataclass
class PageJob:
    sha256: str
    pdf_path: str
    page_no: int
    total_pages: int
    attempts: int = 0
    endpoint_attempts: dict[str, int] = field(default_factory=dict)
    failed_endpoints: set[str] = field(default_factory=set)


@dataclass
class ConvertSummary:
    total_pages: int = 0
    already_done: int = 0
    converted: int = 0
    failed: int = 0
    truncated_bumps: int = 0
    endpoint_requests: dict[str, int] = field(default_factory=dict)

    @property
    def pending_at_start(self) -> int:
        return self.converted + self.failed


def _tail(text: str, n: int) -> str:
    """Last ~n chars, trimmed to a clean line boundary when possible."""

    if len(text) <= n:
        return text.strip()
    slice_ = text[-n:]
    nl = slice_.find("\n")
    if 0 <= nl < len(slice_) - 1:
        slice_ = slice_[nl + 1 :]
    return slice_.strip()


class Converter:
    def __init__(self, config: Config, pool: VLMPool, model_id: str):
        self.config = config
        self.pool = pool
        self.model_id = model_id
        self.work_dir = Path(config.work_dir)
        self.dpi = config.rasterize.dpi
        self._outputs: dict[tuple[str, int], str] = {}  # (sha, page) -> generated md
        self._summary = ConvertSummary()
        self._done_count = 0
        self._pending_total = 0
        self._progress: Optional[ProgressFn] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._complete_event: Optional[asyncio.Event] = None
        self._retry_tasks: set[asyncio.Task[None]] = set()
        self._active_endpoint_names: set[str] = set()

    async def run(
        self,
        records: list[DocRecord],
        *,
        dpi: Optional[int] = None,
        only_sha: Optional[set[str]] = None,
        progress: Optional[ProgressFn] = None,
    ) -> ConvertSummary:
        """Convert all pending pages across ``records``. Idempotent / resumable."""

        self.dpi = dpi or self.config.rasterize.dpi
        self._progress = progress
        self._loop = asyncio.get_running_loop()
        self._complete_event = asyncio.Event()

        queue: asyncio.Queue[PageJob] = asyncio.Queue()
        pending: list[PageJob] = []
        for rec in records:
            if only_sha is not None and rec.sha256 not in only_sha:
                continue
            for page_no in range(1, rec.page_count + 1):
                self._summary.total_pages += 1
                if artifacts.is_done(
                    self.work_dir, rec.sha256, page_no, self.dpi, self.model_id
                ):
                    self._summary.already_done += 1
                    continue
                pending.append(
                    PageJob(rec.sha256, rec.pdf_path, page_no, rec.page_count)
                )

        self._pending_total = len(pending)
        if not pending:
            logger.info(
                "nothing to convert (all pages cached)",
                extra={"fields": {"total": self._summary.total_pages}},
            )
            return self._summary

        active_endpoints = [
            ep for ep in self.config.active_endpoints if ep.inflight > 0
        ]
        if not active_endpoints:
            raise RuntimeError("no active endpoints configured")
        self._active_endpoint_names = {ep.name for ep in active_endpoints}

        logger.info(
            "starting conversion",
            extra={
                "fields": {
                    "pending": self._pending_total,
                    "cached": self._summary.already_done,
                    "dpi": self.dpi,
                    "model": self.model_id,
                }
            },
        )

        for job in pending:
            queue.put_nowait(job)

        workers: list[asyncio.Task[None]] = []
        for ep in active_endpoints:
            for i in range(ep.inflight):
                workers.append(
                    asyncio.create_task(
                        self._worker(f"{ep.name}#{i}", ep.name, queue),
                        name=f"worker-{ep.name}-{i}",
                    )
                )

        try:
            await self._complete_event.wait()
        finally:
            for task in list(self._retry_tasks):
                task.cancel()
            if self._retry_tasks:
                await asyncio.gather(*self._retry_tasks, return_exceptions=True)
                self._retry_tasks.clear()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        self._summary.endpoint_requests = {
            n: s.requests for n, s in self.pool.stats().items()
        }
        logger.info(
            "conversion complete",
            extra={
                "fields": {
                    "converted": self._summary.converted,
                    "failed": self._summary.failed,
                    "cached": self._summary.already_done,
                }
            },
        )
        return self._summary

    async def _worker(
        self, wid: str, endpoint: str, queue: "asyncio.Queue[PageJob]"
    ) -> None:
        while True:
            job = await queue.get()
            try:
                if self._should_defer_to_untried_endpoint(job, endpoint):
                    await queue.put(job)
                    await asyncio.sleep(0)
                    continue
                await self._process(job, endpoint, queue)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # defensive: a bug must not wedge the run
                logger.error(
                    "worker crash", extra={"fields": {"wid": wid, "err": str(exc)}}
                )
                try:
                    artifacts.write_failure(
                        self.work_dir,
                        sha256=job.sha256,
                        page_no=job.page_no,
                        dpi=self.dpi,
                        model_id=self.model_id,
                        error=f"worker crash: {exc}",
                        attempts=job.attempts,
                    )
                except Exception as write_exc:
                    logger.error(
                        "failed to persist worker crash",
                        extra={
                            "fields": {
                                "wid": wid,
                                "sha": job.sha256[:8],
                                "page": job.page_no,
                                "err": str(write_exc),
                            }
                        },
                    )
                self._summary.failed += 1
                self._tick()
            finally:
                queue.task_done()

    def _should_defer_to_untried_endpoint(self, job: PageJob, endpoint: str) -> bool:
        if endpoint not in job.failed_endpoints:
            return False
        return any(
            other not in job.failed_endpoints
            for other in self._active_endpoint_names
            if other != endpoint
        )

    async def _process(
        self, job: PageJob, endpoint: str, queue: "asyncio.Queue[PageJob]"
    ) -> None:
        loop = self._loop
        assert loop is not None

        # Rasterize (CPU-bound) off the event loop.
        try:
            data_url = await loop.run_in_executor(
                None,
                render_page_data_url,
                job.pdf_path,
                job.page_no,
                self.dpi,
                self.config.rasterize.max_long_px,
            )
        except Exception as exc:
            self._fail(job, f"rasterize error: {exc}")
            return

        prev_tail = await self._prev_tail(job)
        messages = build_messages(
            image_data_url=data_url,
            page_no=job.page_no,
            total_pages=job.total_pages,
            prev_tail=prev_tail,
            figures=self.config.figures,
        )

        try:
            result = await self.pool.chat(endpoint, messages)
            # A page that hit the token ceiling is truncated -> one higher-budget retry.
            if result.truncated:
                bumped = min(
                    int(self.config.convert.max_tokens * 1.6), _TRUNCATION_BUMP_LIMIT
                )
                if bumped <= self.config.convert.max_tokens:
                    await self._retry_or_fail(
                        job,
                        endpoint,
                        queue,
                        VLMError(
                            "model response truncated with no higher token budget"
                        ),
                    )
                    return
                self._summary.truncated_bumps += 1
                retry_result = await self.pool.chat(
                    endpoint, messages, max_tokens=bumped
                )
                if retry_result.truncated:
                    await self._retry_or_fail(
                        job, endpoint, queue, VLMError("model response still truncated")
                    )
                    return
                if len(retry_result.content.strip()) <= len(result.content.strip()):
                    await self._retry_or_fail(
                        job,
                        endpoint,
                        queue,
                        VLMError("truncation retry did not improve response"),
                    )
                    return
                result = retry_result
            if not result.content.strip():
                await self._retry_or_fail(
                    job, endpoint, queue, VLMError("empty model response")
                )
                return
        except VLMError as exc:
            await self._retry_or_fail(job, endpoint, queue, exc)
            return

        artifacts.write_success(
            self.work_dir,
            sha256=job.sha256,
            page_no=job.page_no,
            dpi=self.dpi,
            model_id=self.model_id,
            markdown=result.content,
            endpoint=result.endpoint,
            image_tokens=result.image_tokens,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            attempts=job.attempts + 1,
        )
        self._outputs[(job.sha256, job.page_no)] = result.content
        self._summary.converted += 1
        self._tick()

    async def _retry_or_fail(
        self,
        job: PageJob,
        endpoint: str,
        queue: "asyncio.Queue[PageJob]",
        exc: VLMError,
    ) -> None:
        job.attempts += 1
        job.endpoint_attempts[endpoint] = job.endpoint_attempts.get(endpoint, 0) + 1
        job.failed_endpoints.add(endpoint)
        if not getattr(exc, "retryable", True):
            self._fail(
                job, f"non-retryable failure after {job.attempts} attempts: {exc}"
            )
            return
        max_attempts = self.config.convert.max_retries
        active_endpoints = self._active_endpoint_names or set(job.endpoint_attempts)
        has_retry_capacity = any(
            job.endpoint_attempts.get(name, 0) < max_attempts
            for name in active_endpoints
        )
        if not has_retry_capacity:
            self._fail(job, f"gave up after {job.attempts} attempts: {exc}")
            return
        retry_after_s = getattr(exc, "retry_after_s", None)
        if retry_after_s is not None:
            delay = max(0.0, float(retry_after_s))
        else:
            delay = min(
                self.config.convert.backoff_base_s * (2 ** (job.attempts - 1)), 30.0
            )
            delay += random.uniform(0, delay * 0.25)  # jitter to de-synchronize retries
        logger.warning(
            "page failed; retrying",
            extra={
                "fields": {
                    "sha": job.sha256[:8],
                    "page": job.page_no,
                    "attempt": job.attempts,
                    "endpoint": endpoint,
                    "delay": round(delay, 1),
                    "err": str(exc)[:120],
                }
            },
        )
        task = asyncio.create_task(self._delayed_requeue(job, queue, delay))
        self._retry_tasks.add(task)
        task.add_done_callback(lambda done: self._retry_tasks.discard(done))

    async def _delayed_requeue(
        self, job: PageJob, queue: "asyncio.Queue[PageJob]", delay: float
    ) -> None:
        await asyncio.sleep(delay)
        await queue.put(job)

    def _fail(self, job: PageJob, error: str) -> None:
        artifacts.write_failure(
            self.work_dir,
            sha256=job.sha256,
            page_no=job.page_no,
            dpi=self.dpi,
            model_id=self.model_id,
            error=error,
            attempts=job.attempts,
        )
        self._summary.failed += 1
        logger.error(
            "page failed permanently",
            extra={
                "fields": {
                    "sha": job.sha256[:8],
                    "page": job.page_no,
                    "err": error[:160],
                }
            },
        )
        self._tick()

    async def _prev_tail(self, job: PageJob) -> Optional[str]:
        if job.page_no <= 1:
            return None
        prev = job.page_no - 1
        n = self.config.convert.prev_tail_chars
        try:
            md = self._outputs.get((job.sha256, prev))
            if md is None:
                loop = self._loop
                assert loop is not None
                md = await loop.run_in_executor(
                    None,
                    artifacts.read_md,
                    self.work_dir,
                    job.sha256,
                    prev,
                    self.dpi,
                    self.model_id,
                )
            if md:
                return _tail(md, n)
        except Exception:
            pass
        # Fallback: previous page's native text layer (keeps continuity even when
        # the generated prior page isn't available, e.g. it failed or is out of order).
        try:
            loop = self._loop
            assert loop is not None
            txt = await loop.run_in_executor(None, page_text, job.pdf_path, prev)
            return _tail(txt, n) if txt.strip() else None
        except Exception:
            return None

    def _tick(self) -> None:
        self._done_count += 1
        if self._progress is not None:
            self._progress(self._done_count, self._pending_total)
        if self._complete_event is not None and self._done_count >= self._pending_total:
            self._complete_event.set()
        if self._done_count % 25 == 0 or self._done_count == self._pending_total:
            logger.info(
                "progress",
                extra={
                    "fields": {
                        "done": self._done_count,
                        "of": self._pending_total,
                        "ok": self._summary.converted,
                        "failed": self._summary.failed,
                    }
                },
            )
