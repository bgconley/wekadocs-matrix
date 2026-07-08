"""docpipe command-line interface.

Commands:
  convert       PDFs -> Markdown corpus (full pipeline; resumable)
  status        show per-document conversion coverage from the work dir
  retry-failed  re-attempt failed/missing pages, then re-assemble
  inspect       render + convert a single page for debugging / DPI calibration
  doctor        probe both endpoints (/v1/models + a tiny image round-trip)

Run as ``docpipe <cmd>`` (installed) or ``python -m docpipe <cmd>``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from . import PIPELINE_VERSION, artifacts
from . import config as config_mod
from . import manifest as manifest_mod
from . import output as output_mod
from .config import Config
from .convert import Converter
from .evaluation import evaluate_corpus
from .log import get_logger, setup_logging
from .prompts import build_messages
from .rasterize import render_page_to_file
from .report import build_report, report_dict
from .validate import document_coverage
from .vlm_client import VLMPool

logger = get_logger("cli")


# --------------------------------------------------------------------------- config
def _load_config(args: argparse.Namespace) -> Config:
    cfg = config_mod.load(getattr(args, "config", None))
    if getattr(args, "in_dir", None):
        cfg.input_dir = args.in_dir
    if getattr(args, "out", None):
        cfg.output.dir = args.out
    if getattr(args, "work", None):
        cfg.work_dir = args.work
    if getattr(args, "dpi", None):
        cfg.rasterize.dpi = args.dpi
    if getattr(args, "figures", None):
        cfg.figures = args.figures
    if getattr(args, "model", None):
        cfg.model_id = args.model
    if getattr(args, "endpoint", None):
        target = args.endpoint.strip().lower()
        known = {e.name for e in cfg.endpoints}
        if target in ("all", "*"):
            for e in cfg.endpoints:
                e.enabled = True
        else:
            names = {n.strip() for n in target.split(",") if n.strip()}
            if not names:
                raise SystemExit(
                    f"no endpoints selected by --endpoint; known: {sorted(known)}"
                )
            unknown = names - known
            if unknown:
                raise SystemExit(
                    f"unknown endpoint(s) {sorted(unknown)}; known: {sorted(known)}"
                )
            for e in cfg.endpoints:
                e.enabled = e.name in names
    return cfg


def _select_records(records: list, only: Optional[str], limit: Optional[int]) -> list:
    if only:
        needle = only.lower()
        records = [
            r for r in records if needle in r.slug.lower() or r.sha256.startswith(only)
        ]
    if limit is not None:
        records = records[:limit]
    return records


# --------------------------------------------------------------------------- convert
async def _run_pipeline(
    cfg: Config, args: argparse.Namespace, *, rebuild_manifest: bool
) -> int:
    work = Path(cfg.work_dir)
    if rebuild_manifest:
        records = manifest_mod.build(Path(cfg.input_dir), work)
    else:
        records = list(manifest_mod.load_manifest(work).values())
        if not records:
            records = manifest_mod.build(Path(cfg.input_dir), work)
    records = _select_records(
        records, getattr(args, "only", None), getattr(args, "limit", None)
    )
    if not records:
        print("no PDFs found to process", file=sys.stderr)
        return 2
    if not cfg.active_endpoints:
        raise SystemExit(
            "no active endpoints configured; enable an endpoint or remove DOCPIPE_DISABLE"
        )

    t0 = time.monotonic()
    async with VLMPool(cfg) as pool:
        model_id = await pool.resolve_model_id()
        conv = Converter(cfg, pool, model_id)
        summary = await conv.run(records, dpi=cfg.rasterize.dpi)
        endpoint_stats = pool.stats()
    wall = time.monotonic() - t0

    results = output_mod.assemble_all(
        cfg,
        records,
        model_id,
        dpi=cfg.rasterize.dpi,
        allow_incomplete=getattr(args, "allow_incomplete", False),
    )

    if getattr(args, "json", False):
        print(
            json.dumps(
                report_dict(
                    convert_summary=summary,
                    doc_results=results,
                    endpoint_stats=endpoint_stats,
                    wall_s=wall,
                    model_id=model_id,
                    dpi=cfg.rasterize.dpi,
                ),
                indent=2,
            )
        )
    else:
        print(
            build_report(
                convert_summary=summary,
                doc_results=results,
                endpoint_stats=endpoint_stats,
                wall_s=wall,
                model_id=model_id,
                dpi=cfg.rasterize.dpi,
            )
        )
    return 0 if summary.failed == 0 else 1


def cmd_convert(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    return asyncio.run(_run_pipeline(cfg, args, rebuild_manifest=True))


def cmd_retry_failed(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    return asyncio.run(_run_pipeline(cfg, args, rebuild_manifest=False))


# --------------------------------------------------------------------------- status
def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    work = Path(cfg.work_dir)
    records = list(manifest_mod.load_manifest(work).values())
    if not records:
        print("no manifest found; run `docpipe convert` first", file=sys.stderr)
        return 2
    records = _select_records(records, getattr(args, "only", None), None)

    rows: list[tuple[str, int, int, int, int, int | str, str, str]] = []
    for rec in records:
        keysets = artifacts.discover_keysets(work, rec.sha256)
        if not keysets:
            rows.append((rec.slug, rec.page_count, 0, 0, rec.page_count, "-", "-", "-"))
            continue
        for dpi, model_id, prompt_version in keysets:
            cov = document_coverage(
                work, rec, dpi, model_id, prompt_version=prompt_version
            )
            rows.append(
                (
                    rec.slug,
                    rec.page_count,
                    cov.ok_count,
                    len(cov.failed_pages),
                    len(cov.missing_pages),
                    dpi,
                    prompt_version,
                    model_id,
                )
            )

    if getattr(args, "json", False):
        print(
            json.dumps(
                [
                    {
                        "slug": slug,
                        "pages": pages,
                        "ok": ok,
                        "failed": fail,
                        "missing": miss,
                        "dpi": dpi,
                        "prompt_version": prompt_version,
                        "model": model,
                    }
                    for slug, pages, ok, fail, miss, dpi, prompt_version, model in rows
                ],
                indent=2,
            )
        )
        return 0

    print(
        f"{'document':<40} {'pages':>6} {'ok':>6} {'fail':>5} "
        f"{'miss':>5} {'dpi':>5} {'prompt':>8} model"
    )
    print("-" * 118)
    for slug, pages, ok, fail, miss, dpi, prompt_version, model in rows:
        print(
            f"{slug:<40} {pages:>6} {ok:>6} {fail:>5} "
            f"{miss:>5} {str(dpi):>5} {prompt_version:>8} {model}"
        )
    return 0


# --------------------------------------------------------------------------- inspect
async def _inspect(cfg: Config, pdf: str, page: int, dpi: int) -> int:
    work = Path(cfg.work_dir) / "inspect"
    png = work / f"{Path(pdf).stem}_p{page}_d{dpi}.png"
    render_page_to_file(pdf, page, dpi, cfg.rasterize.max_long_px, png)
    print(f"[rendered] {png}", file=sys.stderr)

    from .rasterize import render_page_data_url

    data_url = render_page_data_url(pdf, page, dpi, cfg.rasterize.max_long_px)
    messages = build_messages(
        image_data_url=data_url,
        page_no=page,
        total_pages=page,
        prev_tail=None,
        figures=cfg.figures,
    )
    async with VLMPool(cfg) as pool:
        model_id = await pool.resolve_model_id()
        endpoint = pool.endpoint_names[0]
        result = await pool.chat(endpoint, messages)
    md_path = png.with_suffix(".md")
    md_path.write_text(result.content, encoding="utf-8")
    print(
        f"[model] {model_id} via {endpoint}  latency={result.latency_s:.2f}s  "
        f"image_tokens={result.image_tokens}  completion_tokens={result.completion_tokens}",
        file=sys.stderr,
    )
    print(f"[markdown] {md_path}", file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    print(result.content)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    return asyncio.run(_inspect(cfg, args.pdf, args.page, cfg.rasterize.dpi))


# --------------------------------------------------------------------------- doctor
async def _doctor(cfg: Config) -> int:
    for e in cfg.endpoints:
        e.enabled = (
            True  # always health-check every configured endpoint, enabled or not
        )
    print(f"docpipe {PIPELINE_VERSION} -- endpoint check")
    print(
        f"  figures={cfg.figures}  dpi={cfg.rasterize.dpi}  "
        f"total_inflight={cfg.total_inflight}"
    )
    ok = True
    async with VLMPool(cfg) as pool:
        try:
            served = await pool.probe_models()
        except Exception as exc:
            print(f"  [FAIL] /v1/models probe: {exc}")
            return 1
        # health_image issues a real chat call, which needs a resolved model id.
        pool.model_id = pool.model_id or next(iter(served.values()))
        for ep in cfg.active_endpoints:
            model = served.get(ep.name, "?")
            print(
                f"  [{ep.name}] {ep.base_url}  model={model}  "
                f"auth={'bearer' if ep.bearer is not None else 'none'}  inflight={ep.inflight}"
            )
            try:
                res = await pool.health_image(ep.name)
                print(
                    f"           image round-trip: '{res.content.strip()}'  "
                    f"latency={res.latency_s:.2f}s  image_tokens={res.image_tokens}"
                )
            except Exception as exc:
                ok = False
                print(f"           [FAIL] image round-trip: {exc}")
    print("  OK" if ok else "  DEGRADED")
    return 0 if ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    return asyncio.run(_doctor(cfg))


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    root = Path(cfg.output.dir)
    spec_path = Path(args.spec) if getattr(args, "spec", None) else None
    report = evaluate_corpus(root, spec_path)
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            f"pass_fraction={report.pass_fraction:.3f} "
            f"passed={report.passed}/{report.total}"
        )
        print(
            f"tier0={report.tier0_pass_rate:.3f} "
            f"boilerplate_leaks={report.boilerplate_leaks} "
            f"table_validity={report.table_validity_rate}"
        )
        for result in report.case_results:
            if not result.passed:
                print(f"FAIL {result.id} {result.path}: {result.message}")
    return 0 if report.pass_fraction == 1.0 else 1


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docpipe", description=__doc__)
    p.add_argument(
        "--config", help="path to docpipe.toml (else ./docpipe.toml if present)"
    )
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--in", dest="in_dir", help="input PDF directory")
        sp.add_argument("--out", help="output corpus directory")
        sp.add_argument("--work", help="work dir for manifest + page cache")
        sp.add_argument("--dpi", type=int, help="rasterization DPI")
        sp.add_argument("--model", help="override served model id (else probed)")
        sp.add_argument(
            "--endpoint", help="restrict to one endpoint by name (e.g. oxcart)"
        )

    c = sub.add_parser("convert", help="convert a directory of PDFs to Markdown")
    common(c)
    c.add_argument("--figures", choices=["enriched", "minimal", "skip"])
    c.add_argument("--only", help="restrict to docs whose slug/sha matches")
    c.add_argument("--limit", type=int, help="process only the first N documents")
    c.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write docs even if some pages failed",
    )
    c.add_argument("--json", action="store_true", help="emit the run report as JSON")
    c.set_defaults(func=cmd_convert)

    r = sub.add_parser("retry-failed", help="re-attempt failed/missing pages")
    common(r)
    r.add_argument("--figures", choices=["enriched", "minimal", "skip"])
    r.add_argument("--only", help="restrict to docs whose slug/sha matches")
    r.add_argument("--allow-incomplete", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_retry_failed)

    s = sub.add_parser("status", help="show conversion coverage")
    s.add_argument("--work", help="work dir")
    s.add_argument("--only", help="restrict to docs whose slug/sha matches")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    i = sub.add_parser("inspect", help="render + convert one page (debug)")
    i.add_argument("pdf", help="path to a PDF")
    i.add_argument("page", type=int, help="1-indexed page number")
    i.add_argument("--dpi", type=int, help="rasterization DPI")
    i.add_argument("--work", help="work dir")
    i.add_argument("--figures", choices=["enriched", "minimal", "skip"])
    i.add_argument("--model", help="override served model id")
    i.add_argument("--endpoint", help="restrict to one endpoint by name (e.g. oxcart)")
    i.set_defaults(func=cmd_inspect)

    d = sub.add_parser("doctor", help="probe both endpoints")
    d.set_defaults(func=cmd_doctor)

    e = sub.add_parser("eval", help="run offline corpus quality gates")
    e.add_argument("--out", help="output corpus directory")
    e.add_argument("--spec", help="optional JSON eval spec")
    e.add_argument("--json", action="store_true", help="emit eval report as JSON")
    e.set_defaults(func=cmd_eval)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "log_level", "INFO"))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print(
            "\ninterrupted -- progress is checkpointed; re-run to resume.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
