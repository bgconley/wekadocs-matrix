from docpipe.convert import ConvertSummary
from docpipe.output import DocResult
from docpipe.report import build_report, report_dict
from docpipe.vlm_client import EndpointStats


def test_build_report_distinguishes_written_and_skipped_incomplete_docs():
    text = build_report(
        convert_summary=ConvertSummary(total_pages=4, converted=2, failed=2),
        doc_results=[
            DocResult(
                sha256="aaa",
                slug="written-partial",
                page_count=3,
                ok_count=1,
                complete=False,
                written=True,
                out_path="/tmp/written.md",
                missing_pages=[2],
                failed_pages=[3],
            ),
            DocResult(
                sha256="bbb",
                slug="skipped-partial",
                page_count=1,
                ok_count=0,
                complete=False,
                written=False,
                missing_pages=[1],
            ),
        ],
        endpoint_stats={"oxcart": EndpointStats(requests=2, failures=1)},
        wall_s=10.0,
        model_id="model",
        dpi=218,
    )

    assert "written ........ 1 / 2" in text
    assert "incomplete ..... 2 (1 written with --allow-incomplete, 1 skipped)" in text
    assert "- written-partial: 1/3 ok, 2 bad, written" in text
    assert "- skipped-partial: 0/1 ok, 1 bad, skipped" in text
    assert "not written unless --allow-incomplete" not in text


def test_report_dict_preserves_doc_status_lists():
    data = report_dict(
        convert_summary=ConvertSummary(
            total_pages=3,
            already_done=1,
            converted=1,
            failed=1,
            truncated_bumps=1,
        ),
        doc_results=[
            DocResult(
                sha256="abc",
                slug="guide",
                title="Guide",
                page_count=3,
                ok_count=2,
                complete=False,
                written=False,
                suspect_pages=[1],
                empty_pages=[2],
                failed_pages=[3],
                contract_ok=False,
                contract_violations=["table:not_top_level"],
            )
        ],
        endpoint_stats={
            "oxcart": EndpointStats(
                requests=2,
                failures=1,
                prompt_tokens=30,
                completion_tokens=40,
                latency_s=3.25,
            )
        },
        wall_s=12.345,
        model_id="model",
        dpi=218,
    )

    assert data["wall_s"] == 12.35
    assert data["pages"] == {
        "total": 3,
        "converted": 1,
        "cached": 1,
        "failed": 1,
        "truncated_bumps": 1,
    }
    assert data["endpoints"]["oxcart"]["latency_s"] == 3.25
    assert data["documents"][0]["suspect_pages"] == [1]
    assert data["documents"][0]["empty_pages"] == [2]
    assert data["documents"][0]["failed_pages"] == [3]
    assert data["documents"][0]["contract_ok"] is False
