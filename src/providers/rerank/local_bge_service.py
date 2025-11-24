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

        max_tokens_total = 1024  # approximate max tokens per request (query + doc)
        max_tokens_per_doc = 800  # keep doc under this to allow query tokens

        def _truncate(text: str) -> str:
            tokens = text.split()
            if len(tokens) > max_tokens_per_doc:
                return " ".join(tokens[:max_tokens_per_doc])
            return text

        all_results: List[Dict[str, Any]] = []
        for i, cand in enumerate(candidates):
            if "text" not in cand:
                raise ValueError(f"Candidate {i} missing required 'text' field")
            text = _truncate(cand["text"])
            tcount = max(1, len(text.split()))
            qcount = max(1, len(query.split()))
            if qcount + tcount > max_tokens_total:
                # Skip if even truncated doc exceeds budget with query
                continue

            payload: Dict[str, Any] = {
                "query": query,
                "documents": [text],
                "model": self._model_id,
            }
            response = self._client.post("/v1/rerank", json=payload)
            if response.status_code != 200:
                try:
                    body = response.text
                except Exception:
                    body = "<unavailable>"
                raise RuntimeError(
                    f"Reranker service HTTP {response.status_code}: {body}"
                )

            data = response.json()
            results = data.get("results", [])
            log_docs = [text[:200]]
            logger.debug(
                "Reranker response",
                extra={
                    "provider": self._provider_name,
                    "model": self._model_id,
                    "doc_count": 1,
                    "top_k": top_k,
                    "results_count": len(results),
                    "scores": [r.get("score") for r in results[: min(5, len(results))]],
                    "query_snippet": query[:200],
                    "doc_snippets": log_docs,
                },
            )
            logger.warning(
                "Reranker payload debug",
                extra={
                    "provider": self._provider_name,
                    "model": self._model_id,
                    "doc_count": 1,
                    "top_k": top_k,
                    "scores": [r.get("score") for r in results[: min(5, len(results))]],
                    "query_snippet": query[:200],
                    "doc_snippets": log_docs,
                },
            )
            try:
                print(
                    "RERANKER_PAYLOAD_DEBUG",
                    {
                        "query": repr(query[:200]),
                        "docs": [repr(d) for d in log_docs],
                        "scores": [
                            r.get("score") for r in results[: min(5, len(results))]
                        ],
                    },
                )
            except Exception:
                pass

            if not results:
                continue
            score = results[0].get("score")
            all_results.append(
                {
                    **cand,
                    "rerank_score": score,
                    "original_rank": i,
                    "reranker": self._provider_name,
                }
            )

        all_results = sorted(
            all_results,
            key=lambda c: (c.get("rerank_score") or float("-inf")),
            reverse=True,
        )
        return all_results[:top_k]
