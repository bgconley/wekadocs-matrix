import json
from pathlib import Path

DOCPIPE_ROOT = Path(__file__).resolve().parents[1]


def test_nutanix_gold_eval_spec_is_populated_and_well_formed():
    spec_path = DOCPIPE_ROOT / "docs" / "eval" / "nutanix_gold.json"

    data = json.loads(spec_path.read_text(encoding="utf-8"))
    tests = data["tests"]
    ids = [case["id"] for case in tests]
    paths = {case["path"] for case in tests}

    assert 30 <= len(tests) <= 60
    assert len(ids) == len(set(ids))
    assert len(paths) == 9
    assert {case["type"] for case in tests} <= {
        "presence",
        "absence",
        "reading_order",
        "table_rectangular",
        "contract",
        "baseline",
    }
    assert all(case["path"].endswith(".md") for case in tests)
    for case in tests:
        if case["type"] == "presence":
            assert case["text"]
        if case["type"] == "absence":
            assert case.get("text") or case.get("regex")
        if case["type"] == "reading_order":
            assert case["before"]
            assert case["after"]


def test_citation_eval_questions_are_populated_and_source_grounded():
    spec_path = DOCPIPE_ROOT / "docs" / "eval" / "citation_questions.json"

    data = json.loads(spec_path.read_text(encoding="utf-8"))
    questions = data["questions"]
    ids = [case["id"] for case in questions]

    assert len(questions) >= 12
    assert len(ids) == len(set(ids))
    for case in questions:
        assert case["question"].strip().endswith("?")
        assert case["expected_sources"]
        assert case["must_answer_contain"]
        for source in case["expected_sources"]:
            assert source["source_pdf"].endswith(".pdf")
            assert source["slug"]
