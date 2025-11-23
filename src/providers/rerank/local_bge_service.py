"""
Local BGE reranker service provider.

This provider calls an external HTTP service that exposes /v1/rerank
for the BAAI/bge-reranker-v2-m3 model (or compatible).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from src.providers.rerank.base import RerankProvider

logger = logging.getLogger(__name__)


class BGERerankerServiceProvider(RerankProvider):
    """HTTP client wrapper for the local BGE reranker service."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "http://127.0.0.1:9001",
        timeout: float = 60.0,
    ) -> None:
        self._model_id = model
        self._provider_name = "bge-reranker-service"
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates:
            raise ValueError("Cannot rerank empty candidate list")

        # Prepare payload for service
        docs: List[str] = []
        for i, cand in enumerate(candidates):
            if "text" not in cand:
                raise ValueError(f"Candidate {i} missing required 'text' field")
            docs.append(cand["text"])

        payload: Dict[str, Any] = {
            "query": query,
            "documents": docs,
            "model": self._model_id,
        }

        response = self._client.post("/v1/rerank", json=payload)
        if response.status_code != 200:
            try:
                body = response.text
            except Exception:
                body = "<unavailable>"
            raise RuntimeError(f"Reranker service HTTP {response.status_code}: {body}")

        data = response.json()
        results = data.get("results", [])
        log_docs = [d[:200] for d in docs[:2]]
        logger.debug(
            "Reranker response",
            extra={
                "provider": self._provider_name,
                "model": self._model_id,
                "doc_count": len(docs),
                "top_k": top_k,
                "results_count": len(results),
                "scores": [r.get("score") for r in results[: min(5, len(results))]],
                "query_snippet": query[:200],
                "doc_snippets": log_docs,
            },
        )

        # Emit a higher-visibility log so we can inspect payload text and scores during smoke tests.
        logger.warning(
            "Reranker payload debug",
            extra={
                "provider": self._provider_name,
                "model": self._model_id,
                "doc_count": len(docs),
                "top_k": top_k,
                "scores": [r.get("score") for r in results[: min(5, len(results))]],
                "query_snippet": query[:200],
                "doc_snippets": log_docs,
            },
        )
        # Print a concise payload snapshot to stdout for quick inspection in smoke tests.
        try:
            print(
                "RERANKER_PAYLOAD_DEBUG",
                {
                    "query": repr(query[:200]),
                    "docs": [repr(d) for d in log_docs],
                    "scores": [r.get("score") for r in results[: min(5, len(results))]],
                },
            )
        except Exception:
            pass

        # Map scores back to candidates, preserving original order
        reranked: List[Dict[str, Any]] = []
        for idx, entry in enumerate(results):
            score = entry.get("score")
            # Preserve original candidate info
            original = candidates[idx] if idx < len(candidates) else {}
            reranked.append(
                {
                    **original,
                    "rerank_score": score,
                    "original_rank": idx,
                    "reranker": self._provider_name,
                }
            )

        # Sort by rerank_score desc and truncate to top_k
        reranked = sorted(
            reranked,
            key=lambda c: (c.get("rerank_score") or float("-inf")),
            reverse=True,
        )
        return reranked[:top_k]
