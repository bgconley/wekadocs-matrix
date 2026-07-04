from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src.mcp_server.mcp_utils import Deps
from src.query.retrieval_types import ChunkResult


class EvidenceFakeScratch:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def put(self, session: str, pid: str, payload: dict[str, Any]) -> int:
        self._store[pid] = payload
        return len(str(payload))

    async def get(self, session: str, pid: str) -> Optional[dict[str, Any]]:
        return self._store.get(pid)


class EvidenceFakeQuery:
    def search_sections_light(
        self,
        query: str,
        *,
        fetch_k: int,
        filters: Optional[dict[str, Any]] = None,
        expand: bool = True,
    ) -> tuple[list[ChunkResult], dict[str, Any]]:
        chunk = ChunkResult(
            chunk_id="auth-config-1",
            document_id="doc-auth",
            parent_section_id="sec-auth",
            order=1,
            level=2,
            heading="Authentication Configuration",
            text=(
                "Configure authentication by setting the identity provider, "
                "assigning roles, and validating access before production use."
            ),
            token_count=18,
            doc_tag="nutanix_docs/authentication/configuration",
            parent_path_norm="Security > Authentication > Configuration",
            fusion_method="rrf",
            fused_score=0.91,
            vector_score=0.88,
            rerank_score=0.94,
            rerank_rank=1,
        )
        metrics = {
            "reranker_applied": True,
            "signal_pool_used": True,
            "signal_pool_size": 1,
            "query_rewrite_applied": False,
            "query_rewrite_original": query,
            "query_rewrite_result": query,
            "primary_count": 1,
            "final_count": 1,
            "vec_count": 1,
            "bm25_count": 0,
        }
        return [chunk], metrics


@pytest.fixture
def evidence_fake_ctx():
    deps = Deps()
    deps.query = EvidenceFakeQuery()
    deps.graph = None
    deps.text = None
    deps.summarizer = None
    deps.assembler = None
    deps.scratch = EvidenceFakeScratch()
    deps._initialized = True
    request_context = SimpleNamespace(lifespan_context=deps, request=None)
    return SimpleNamespace(request_context=request_context)
