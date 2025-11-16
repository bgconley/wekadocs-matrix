from __future__ import annotations

from typing import Any, Iterable, List

from src.services.context_budget_manager import ContextBudgetManager
from src.services.graph_service import GraphService
from src.services.text_service import TextService


class RecordingSession:
    def __init__(self, driver: "RecordingDriver") -> None:
        self._driver = driver
        self._rows: Iterable[dict[str, Any]] = []

    def __enter__(self) -> "RecordingSession":
        self._rows = self._driver.next_rows()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def run(
        self, query: str, **params: Any
    ) -> Iterable[dict[str, Any]]:  # noqa: ANN001
        return list(self._rows)


class RecordingDriver:
    def __init__(self, responses: List[List[dict[str, Any]]]) -> None:
        self._responses = responses
        self._index = -1

    def session(self) -> RecordingSession:
        return RecordingSession(self)

    def next_rows(self) -> List[dict[str, Any]]:
        if self._index + 1 < len(self._responses):
            self._index += 1
        return self._responses[self._index]


def make_node(idx: int) -> dict[str, Any]:
    return {
        "id": f"node-{idx}",
        "label": "Section",
        "title": f"Node {idx}",
        "level": 2,
        "tokens": 100 + idx,
        "doc_tag": "DOC",
        "anchor": f"#anchor-{idx}",
        "raw_snippet": f"snippet-{idx}",
        "sample_edges": [{"src": "seed", "dst": f"node-{idx}", "type": "MENTIONS"}],
    }


def test_expand_neighbors_enforces_cursor_and_dedupe() -> None:
    rows = [make_node(1), make_node(2)]
    driver = RecordingDriver([rows, rows])
    service = GraphService(driver)

    first = service.expand_neighbors(node_ids=["seed"], page_size=1, session_id="sess")
    assert first.partial is True
    assert first.limit_reason == "page_size"
    assert first.payload["next_cursor"] is not None
    assert first.payload["dedupe_applied"] is True

    second = service.expand_neighbors(
        node_ids=["seed"],
        page_size=1,
        session_id="sess",
        cursor=first.payload["next_cursor"],
    )
    assert second.duplicates_suppressed > 0
    assert second.payload["nodes"] == []


def test_text_service_truncates_and_marks_budget_partial() -> None:
    rows = [
        {
            "id": "sec-1",
            "title": "Title",
            "text": "x" * 40,
            "doc_tag": "DOC",
            "blob_id": "blob-1",
        }
    ]
    driver = RecordingDriver([rows])
    service = TextService(driver)
    budget = ContextBudgetManager(token_budget=1, byte_budget=10_000)

    result = service.get_section_text(["sec-1"], max_bytes_per=10, budget=budget)

    assert result.results[0]["bytes"] == 10
    assert result.results[0]["truncated"] is True
    assert result.partial is True
    assert result.limit_reason == "token_cap"
