from src.evidence.quotes import normalize_quote_payloads


def test_stable_ids_parent_path_and_legacy_fields():
    quotes = normalize_quote_payloads(
        [
            {
                "rank": 1,
                "passage_id": "p1",
                "section_id": "s1",
                "doc_tag": "nci/aos",
                "title": "AOS Storage",
                "uri": "nutanixdocs://scratch/sess/p1",
                "parent_path": "NCI > AOS Storage",
                "quote": "AOS provides distributed storage.",
                "confidence": 0.81,
                "score": 0.9,
                "source": "retrieval",
            }
        ]
    )

    assert quotes[0].quote_id == "q_0001"
    assert quotes[0].parent_path == ["NCI", "AOS Storage"]
    assert quotes[0].retrieval_signals == {"score": 0.9}
    assert quotes[0].title == "AOS Storage"
    assert quotes[0].source_uri == "nutanixdocs://scratch/sess/p1"


def test_empty_text_is_skipped_and_ids_stay_contiguous():
    quotes = normalize_quote_payloads(
        [
            {"rank": 1, "passage_id": "p1", "quote": "   ", "confidence": 0.9},
            {"rank": 2, "passage_id": "p2", "quote": "Useful.", "confidence": 0.7},
        ]
    )

    assert [q.quote_id for q in quotes] == ["q_0001"]
    assert quotes[0].passage_id == "p2"
