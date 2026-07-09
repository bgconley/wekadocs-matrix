from scripts.eval.check_evidence_citations import score_evidence


def test_score_counts_cited_vs_uncited():
    fake = [
        {"id": "direct", "quotes": [{"quote": "x", "doc_tag": "nutanix/files"}]},
        {
            "id": "nested",
            "structured_content": {
                "quotes": [{"text": "y", "docTag": "nutanix/prism"}],
            },
        },
        {"id": "uncited", "quotes": [{"quote": "z"}]},
    ]

    score = score_evidence(fake)

    assert score["cited"] == 2
    assert score["uncited"] == 1
    assert score["pass_rate"] == 2 / 3
    assert [detail["id"] for detail in score["details"]] == [
        "direct",
        "nested",
        "uncited",
    ]
