import json
from pathlib import Path

import pytest

import docpipe.citation_eval as citation_eval
from docpipe.citation_eval import (
    CitationCaseResult,
    CitationQuestion,
    CitationReport,
    SourceExpectation,
    evaluate_payload,
    load_questions,
    main,
    tool_arguments,
)

SPEC = Path("docs/eval/citation_questions.json")


def test_load_questions_accepts_docpipe_citation_spec():
    questions = load_questions(SPEC)

    assert len(questions) == 14
    first = questions[0]
    assert first.id == "cite-ahv-admin-show-uplinks"
    assert first.question == "Which command shows OVS uplinks on the local AHV host?"
    assert first.must_answer_contain == ["manage_ovs show_uplinks"]
    assert first.expected_sources[0].slug == "5c-book-of-ahv-administration"


def test_nai_requests_summary_fixture_uses_source_term_api_keys():
    questions = {question.id: question for question in load_questions(SPEC)}
    question = questions["cite-nai-requests-summary"]

    terms = question.must_answer_contain

    assert "API keys" in question.question
    assert "Requests Summary" in question.question
    assert "API keys" in terms
    assert "API tokens" not in terms


def test_evaluate_payload_passes_when_answer_and_expected_source_quote_match():
    question = CitationQuestion(
        id="cite-ahv-admin-show-uplinks",
        question="Which command shows OVS uplinks on the local AHV host?",
        expected_sources=[
            SourceExpectation(
                source_pdf="5c-book-of-ahv-administration.pdf",
                slug="5c-book-of-ahv-administration",
                must_cite_text="manage_ovs show_uplinks",
            )
        ],
        must_answer_contain=["manage_ovs show_uplinks"],
    )
    payload = {
        "structuredContent": {
            "quotes": [
                {
                    "doc_tag": "5c-book-of-ahv-administration",
                    "title": "AHV - AHV Administration",
                    "uri": "nutanixdocs://5c-book-of-ahv-administration",
                    "quote": "Run manage_ovs show_uplinks to display uplinks.",
                }
            ],
            "answer_draft": {
                "markdown": "Use `manage_ovs show_uplinks` on the local AHV host."
            },
        }
    }

    result = evaluate_payload(question, payload)

    assert result.passed
    assert result.missing_answer_terms == []
    assert result.missing_sources == []
    assert result.quote_count == 1


def test_evaluate_payload_fails_without_expected_citation_text():
    question = CitationQuestion(
        id="cite-aos-genesis-log-tail",
        question="Which command tails the latest genesis.out log from a Controller VM?",
        expected_sources=[
            SourceExpectation(
                source_pdf="Advanced-Admin-AOS-v7_5.pdf",
                slug="advanced-admin-aos-v7-5",
                must_cite_text="tail -F ~/data/logs/genesis.out",
            )
        ],
        must_answer_contain=["tail -F ~/data/logs/genesis.out"],
    )
    payload = {
        "structuredContent": {
            "quotes": [
                {
                    "doc_tag": "advanced-admin-aos-v7-5",
                    "quote": "The genesis log is under ~/data/logs.",
                }
            ],
            "answer_draft": {"markdown": "Use `tail -F ~/data/logs/genesis.out`."},
        }
    }

    result = evaluate_payload(question, payload)

    assert not result.passed
    assert result.missing_answer_terms == []
    assert result.missing_sources == [
        "advanced-admin-aos-v7-5:tail -F ~/data/logs/genesis.out"
    ]


def test_evaluate_payload_reads_legacy_search_documentation_shape():
    question = CitationQuestion(
        id="cite-nke-kubeconfig-expiry",
        question="How long does an NKE kubeconfig token last?",
        expected_sources=[
            SourceExpectation(
                source_pdf="Nutanix-Kubernetes-Engine-v2_10-2.pdf",
                slug="nutanix-kubernetes-engine-v2-10-2",
                must_cite_text="The kubeconfig token expires after 24 hours",
            )
        ],
        must_answer_contain=["24 hours"],
    )
    payload = {
        "content": [
            {"type": "text", "text": "The kubeconfig token lasts 24 hours."},
            {
                "type": "json",
                "json": {
                    "answer": "The kubeconfig token expires after 24 hours.",
                    "evidence": [
                        {
                            "doc_tag": "nutanix-kubernetes-engine-v2-10-2",
                            "text": "The kubeconfig token expires after 24 hours.",
                        }
                    ],
                },
            },
        ]
    }

    assert evaluate_payload(question, payload).passed


def test_evaluate_payload_uses_quotes_when_evidence_tool_returns_status_text():
    question = CitationQuestion(
        id="cite-ahv-admin-show-uplinks",
        question="Which command shows OVS uplinks on the local AHV host?",
        expected_sources=[
            SourceExpectation(
                source_pdf="5c-book-of-ahv-administration.pdf",
                slug="5c-book-of-ahv-administration",
                must_cite_text="manage_ovs show_uplinks",
            )
        ],
        must_answer_contain=["manage_ovs show_uplinks"],
    )
    payload = {
        "content": [
            {
                "type": "text",
                "text": "kb_retrieve_evidence returned 1 quotes.",
            }
        ],
        "structuredContent": {
            "quotes": [
                {
                    "doc_tag": "nutanix",
                    "uri": "file:///app/data/ingest/nutanix/"
                    "5c-book-of-ahv-administration/5c-book-of-ahv-administration.md",
                    "quote": "Run manage_ovs show_uplinks to display uplinks.",
                }
            ]
        },
    }

    result = evaluate_payload(question, payload)

    assert result.passed
    assert result.answer_source == "quotes"


def test_load_questions_rejects_missing_question_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"questions": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="questions"):
        load_questions(bad)


def test_report_dict_exposes_pass_fraction_and_failures():
    report = CitationReport(
        [
            CitationCaseResult("ok", "question", True, quote_count=2),
            CitationCaseResult(
                "bad",
                "question",
                False,
                missing_answer_terms=["term"],
                missing_sources=["slug:text"],
            ),
        ]
    )

    data = report.to_dict()

    assert data["pass_fraction"] == pytest.approx(0.5)
    assert data["passed"] == 1
    assert data["total"] == 2
    assert data["cases"][1]["missing_sources"] == ["slug:text"]


def test_tool_arguments_match_evidence_and_legacy_tools():
    evidence_args = tool_arguments(
        tool_name="kb_retrieve_evidence",
        question="What is the command?",
        top_k=8,
        max_quotes=9,
        retrieval_depth=80,
    )
    legacy_args = tool_arguments(
        tool_name="search_documentation",
        question="What is the command?",
        top_k=8,
        max_quotes=9,
        retrieval_depth=80,
    )

    assert evidence_args == {
        "question": "What is the command?",
        "top_k": 8,
        "max_quotes": 9,
        "retrieval_depth": 80,
    }
    assert legacy_args == {
        "query": "What is the command?",
        "top_k": 8,
        "verbosity": "graph",
    }


def test_main_dry_run_validates_spec_without_gateway(capsys):
    rc = main(["--spec", str(SPEC), "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "questions=14" in out
    assert "cite-ahv-admin-show-uplinks" in out


def test_main_writes_report_from_mcp_payloads(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "citation.json"
    spec.write_text(
        """
{
  "questions": [
    {
      "id": "cite-ahv-admin-show-uplinks",
      "question": "Which command shows OVS uplinks on the local AHV host?",
      "expected_sources": [
        {
          "source_pdf": "5c-book-of-ahv-administration.pdf",
          "slug": "5c-book-of-ahv-administration",
          "must_cite_text": "manage_ovs show_uplinks"
        }
      ],
      "must_answer_contain": ["manage_ovs show_uplinks"]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    def fake_call_tool(**kwargs):
        assert kwargs["base_url"] == "http://rag.example"
        assert kwargs["tool_name"] == "kb_retrieve_evidence"
        assert kwargs["arguments"]["max_quotes"] == 4
        return {
            "structuredContent": {
                "quotes": [
                    {
                        "doc_tag": "5c-book-of-ahv-administration",
                        "quote": "Run manage_ovs show_uplinks.",
                    }
                ],
                "answer_draft": {
                    "markdown": "Use `manage_ovs show_uplinks` on the local AHV host."
                },
            }
        }

    monkeypatch.setattr(citation_eval, "mcp_tool_call", fake_call_tool, raising=False)

    rc = main(
        [
            "--spec",
            str(spec),
            "--base-url",
            "http://rag.example",
            "--max-quotes",
            "4",
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    assert "passed=1/1" in capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass_fraction"] == 1.0
    assert report["cases"][0]["id"] == "cite-ahv-admin-show-uplinks"


def test_main_treats_mcp_tool_error_as_gateway_failure(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "citation.json"
    spec.write_text(
        """
{
  "questions": [
    {
      "id": "cite-ahv-admin-show-uplinks",
      "question": "Which command shows OVS uplinks on the local AHV host?",
      "expected_sources": [
        {
          "source_pdf": "5c-book-of-ahv-administration.pdf",
          "slug": "5c-book-of-ahv-administration",
          "must_cite_text": "manage_ovs show_uplinks"
        }
      ],
      "must_answer_contain": ["manage_ovs show_uplinks"]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    def fake_call_tool(**_kwargs):
        return {
            "content": [{"type": "text", "text": "[Errno 111] Connection refused"}],
            "isError": True,
        }

    monkeypatch.setattr(citation_eval, "mcp_tool_call", fake_call_tool)

    rc = main(["--spec", str(spec), "--report", str(report_path)])

    assert rc == 2
    assert not report_path.exists()
    err = capsys.readouterr().err
    assert "citation-eval gateway error" in err
    assert "Connection refused" in err
