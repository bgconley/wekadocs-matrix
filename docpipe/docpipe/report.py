"""End-of-run reporting: pages, tokens, throughput, coverage, and QA warnings."""

from __future__ import annotations

from .convert import ConvertSummary
from .output import DocResult
from .vlm_client import EndpointStats


def humanize_secs(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def build_report(
    *,
    convert_summary: ConvertSummary,
    doc_results: list[DocResult],
    endpoint_stats: dict[str, EndpointStats],
    wall_s: float,
    model_id: str,
    dpi: int,
) -> str:
    cs = convert_summary
    written = [d for d in doc_results if d.written]
    incomplete = [d for d in doc_results if not d.complete]
    suspect_total = sum(len(d.suspect_pages) for d in doc_results)
    empty_total = sum(len(d.empty_pages) for d in doc_results)
    contract_failed = [d for d in doc_results if not d.contract_ok]

    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("  docpipe run report")
    lines.append("=" * 68)
    lines.append(f"  model={model_id}  dpi={dpi}  wall={humanize_secs(wall_s)}")
    lines.append("")
    lines.append("  Pages")
    lines.append(f"    total .......... {cs.total_pages}")
    lines.append(f"    converted (run)  {cs.converted}")
    lines.append(f"    cached (resume)  {cs.already_done}")
    lines.append(f"    failed ......... {cs.failed}")
    if cs.truncated_bumps:
        lines.append(f"    max_tokens bumps {cs.truncated_bumps}")
    lines.append("")

    lines.append("  Endpoints")
    tot_prompt = tot_completion = tot_req = tot_fail = 0
    for name, st in endpoint_stats.items():
        avg_lat = (st.latency_s / st.requests) if st.requests else 0.0
        thru = (st.requests / wall_s) if wall_s else 0.0
        tot_prompt += st.prompt_tokens
        tot_completion += st.completion_tokens
        tot_req += st.requests
        tot_fail += st.failures
        lines.append(
            f"    {name:<10} req={st.requests:<5} fail={st.failures:<4} "
            f"avg={avg_lat:5.2f}s  {thru:4.2f} req/s  "
            f"tok in/out={st.prompt_tokens}/{st.completion_tokens}"
        )
    lines.append(
        f"    {'TOTAL':<10} req={tot_req:<5} fail={tot_fail:<4} "
        f"tok in/out={tot_prompt}/{tot_completion}"
    )
    lines.append("")

    lines.append("  Documents")
    lines.append(f"    written ........ {len(written)} / {len(doc_results)}")
    if incomplete:
        written_incomplete = [d for d in incomplete if d.written]
        skipped_incomplete = [d for d in incomplete if not d.written]
        lines.append(
            f"    incomplete ..... {len(incomplete)} "
            f"({len(written_incomplete)} written with --allow-incomplete, "
            f"{len(skipped_incomplete)} skipped)"
        )
        for d in incomplete:
            miss = len(d.missing_pages) + len(d.failed_pages)
            status = "written" if d.written else "skipped"
            lines.append(
                f"        - {d.slug}: {d.ok_count}/{d.page_count} ok, "
                f"{miss} bad, {status}"
            )
    if suspect_total or empty_total:
        lines.append(
            f"    QA: {suspect_total} suspect page(s), {empty_total} empty page(s)"
        )
        for d in doc_results:
            if d.suspect_pages:
                shown = ", ".join(map(str, d.suspect_pages[:12]))
                more = (
                    ""
                    if len(d.suspect_pages) <= 12
                    else f" (+{len(d.suspect_pages) - 12})"
                )
                lines.append(f"        ! {d.slug}: pages {shown}{more}")
    if contract_failed:
        lines.append(f"    Tier-0 contract failed {len(contract_failed)} document(s)")
        for d in contract_failed:
            shown = ", ".join(d.contract_violations[:8])
            more = (
                ""
                if len(d.contract_violations) <= 8
                else f" (+{len(d.contract_violations) - 8})"
            )
            lines.append(f"        x {d.slug}: {shown}{more}")
    lines.append("")
    for d in written:
        lines.append(f"    -> {d.out_path}")
    lines.append("=" * 68)
    return "\n".join(lines)


def report_dict(
    *,
    convert_summary: ConvertSummary,
    doc_results: list[DocResult],
    endpoint_stats: dict[str, EndpointStats],
    wall_s: float,
    model_id: str,
    dpi: int,
) -> dict:
    return {
        "model_id": model_id,
        "dpi": dpi,
        "wall_s": round(wall_s, 2),
        "pages": {
            "total": convert_summary.total_pages,
            "converted": convert_summary.converted,
            "cached": convert_summary.already_done,
            "failed": convert_summary.failed,
            "truncated_bumps": convert_summary.truncated_bumps,
        },
        "endpoints": {
            name: {
                "requests": st.requests,
                "failures": st.failures,
                "prompt_tokens": st.prompt_tokens,
                "completion_tokens": st.completion_tokens,
                "latency_s": round(st.latency_s, 2),
            }
            for name, st in endpoint_stats.items()
        },
        "documents": [
            {
                "slug": d.slug,
                "title": d.title,
                "sha256": d.sha256,
                "page_count": d.page_count,
                "ok_count": d.ok_count,
                "complete": d.complete,
                "written": d.written,
                "out_path": d.out_path,
                "suspect_pages": d.suspect_pages,
                "empty_pages": d.empty_pages,
                "failed_pages": d.failed_pages,
                "missing_pages": d.missing_pages,
                "contract_ok": d.contract_ok,
                "contract_violations": d.contract_violations,
            }
            for d in doc_results
        ],
    }
