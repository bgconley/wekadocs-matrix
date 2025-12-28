from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.embeddings.contracts import (
    DocumentEmbeddingBundle,
    QueryEmbeddingBundle,
)
from src.providers.settings import EmbeddingSettings as ProviderEmbeddingSettings
from src.providers.tokenizer_service import TokenizerService

logger = logging.getLogger(__name__)


class VoyageEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider for Voyage embeddings (contextual + standard)."""

    DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
    CONTEXTUAL_ENDPOINT = "/contextualizedembeddings"
    EMBEDDINGS_ENDPOINT = "/embeddings"

    def __init__(
        self,
        settings: ProviderEmbeddingSettings,
        *,
        client: Optional[httpx.Client] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if settings is None:
            raise ValueError(
                "Embedding settings are required for VoyageEmbeddingProvider."
            )

        self._settings = settings
        self._dims = settings.dims
        self._model_id = settings.model_id
        self._provider_name = settings.provider or "voyage-ai"
        self._capabilities = settings.capabilities
        self._supports_contextual = bool(
            settings.extra.get("supports_contextualized_chunks")
        )

        self._query_task = settings.extra.get("query_task")
        self._document_task = settings.extra.get("document_task")
        self._output_dimension = settings.extra.get("output_dimension") or settings.dims
        self._output_dtype = settings.extra.get("output_dtype") or "float"
        self._contextual_limits = settings.extra.get("contextual_limits") or {}

        if str(self._output_dtype).lower() != "float":
            raise ValueError(
                "VoyageEmbeddingProvider currently expects output_dtype=float. "
                f"Got {self._output_dtype!r}."
            )

        self._api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self._api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for voyage-ai embeddings.")

        self._base_url = (
            base_url or os.getenv("VOYAGE_API_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        self._timeout = timeout or float(os.getenv("VOYAGE_TIMEOUT", "60"))
        self._max_retries = max_retries or int(os.getenv("VOYAGE_MAX_RETRIES", "4"))
        self._min_backoff = float(os.getenv("VOYAGE_RETRY_BACKOFF_MIN_SEC", "0.5"))
        self._max_backoff = float(os.getenv("VOYAGE_RETRY_BACKOFF_MAX_SEC", "8.0"))

        self._client = client or httpx.Client(
            timeout=httpx.Timeout(self._timeout),
            headers=self._headers,
        )
        self._tokenizer_service = TokenizerService()

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def close(self) -> None:
        if self._client:
            self._client.close()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty list of documents.")
        input_type = self._resolve_input_type(self._document_task)
        if self._supports_contextual:
            inputs = [[text] for text in texts]
            batched = self._embed_contextual(inputs, input_type=input_type)
            return [vector for doc_vectors in batched for vector in doc_vectors]
        return self._embed_standard(texts, input_type=input_type)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Query text must be non-empty.")
        input_type = self._resolve_input_type(self._query_task)
        if self._supports_contextual:
            batched = self._embed_contextual([[text]], input_type=input_type)
            return batched[0][0]
        return self._embed_standard([text], input_type=input_type)[0]

    def embed_documents_all(self, texts: List[str]) -> List[DocumentEmbeddingBundle]:
        dense_vectors = self.embed_documents(texts)
        return [DocumentEmbeddingBundle(dense=list(vec)) for vec in dense_vectors]

    def embed_query_all(self, text: str) -> QueryEmbeddingBundle:
        dense = self.embed_query(text)
        return QueryEmbeddingBundle(dense=list(dense))

    def embed_contextualized_documents(
        self, inputs: List[List[str]], *, input_type: Optional[str] = None
    ) -> List[List[List[float]]]:
        if not inputs:
            raise ValueError("Contextual embedding inputs must be non-empty.")
        resolved_type = self._resolve_input_type(input_type or self._document_task)
        return self._embed_contextual(inputs, input_type=resolved_type)

    def _resolve_input_type(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in {"query", "document"}:
            return normalized
        return None

    def _embed_standard(
        self, texts: List[str], *, input_type: Optional[str]
    ) -> List[List[float]]:
        batches = self._batch_standard_inputs(texts)
        results: List[List[float]] = []
        for batch_idx, batch in enumerate(batches):
            results.extend(
                self._embed_standard_batch(
                    batch, input_type=input_type, batch_idx=str(batch_idx)
                )
            )
        self._validate_dimensions(results)
        return results

    def _embed_standard_batch(
        self, batch: List[str], *, input_type: Optional[str], batch_idx: str
    ) -> List[List[float]]:
        payload: Dict[str, Any] = {
            "input": batch,
            "model": self._model_id,
            "input_type": input_type,
            "truncation": False,
        }
        if self._output_dimension:
            payload["output_dimension"] = int(self._output_dimension)
        if self._output_dtype:
            payload["output_dtype"] = self._output_dtype
        payload = {key: value for key, value in payload.items() if value is not None}
        try:
            response = self._post_json(
                self.EMBEDDINGS_ENDPOINT, payload, batch_idx=str(batch_idx)
            )
        except RuntimeError as exc:
            if "rejected request (400)" in str(exc) and len(batch) > 1:
                mid = len(batch) // 2
                head = self._embed_standard_batch(
                    batch[:mid], input_type=input_type, batch_idx=f"{batch_idx}a"
                )
                tail = self._embed_standard_batch(
                    batch[mid:], input_type=input_type, batch_idx=f"{batch_idx}b"
                )
                return head + tail
            raise
        return self._parse_standard_embeddings(response)

    def _embed_contextual(
        self, inputs: List[List[str]], *, input_type: Optional[str]
    ) -> List[List[List[float]]]:
        self._validate_contextual_inputs(inputs, input_type)
        batches = self._batch_contextual_inputs(inputs)
        results: List[List[List[float]]] = []
        for batch_idx, batch in enumerate(batches):
            payload: Dict[str, Any] = {
                "inputs": batch,
                "model": self._model_id,
                "input_type": input_type,
            }
            if self._output_dimension:
                payload["output_dimension"] = int(self._output_dimension)
            if self._output_dtype:
                payload["output_dtype"] = self._output_dtype
            payload = {
                key: value for key, value in payload.items() if value is not None
            }
            response = self._post_json(
                self.CONTEXTUAL_ENDPOINT, payload, batch_idx=str(batch_idx)
            )
            results.extend(self._parse_contextual_embeddings(response))
        if self._output_dimension:
            self._validate_dimensions(
                [vec for doc in results for vec in doc],
                expected_dims=int(self._output_dimension),
            )
        return results

    def _post_json(
        self, endpoint: str, payload: Dict[str, Any], *, batch_idx: str
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.post(url, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Voyage API retryable error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response else None
                if status == 400:
                    detail = self._extract_error_detail(exc.response)
                    raise RuntimeError(
                        f"Voyage API rejected request (400): {detail}"
                    ) from exc
                if status in {401, 403}:
                    detail = self._extract_error_detail(exc.response)
                    raise RuntimeError(
                        f"Voyage API authentication failed ({status}): {detail}"
                    ) from exc
                if status in {429, 500, 502, 503, 504}:
                    self._sleep_backoff(attempt, batch_idx)
                    continue
                detail = self._extract_error_detail(exc.response)
                raise RuntimeError(f"Voyage API error ({status}): {detail}") from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                self._sleep_backoff(attempt, batch_idx)
            except Exception as exc:
                last_error = exc
                break
        raise RuntimeError(
            f"Voyage API request failed after {self._max_retries} attempts: {last_error}"
        )

    def _sleep_backoff(self, attempt: int, batch_idx: str) -> None:
        base = min(self._max_backoff, self._min_backoff * (2**attempt))
        jitter = random.uniform(0, base / 4.0)
        delay = base + jitter
        logger.warning(
            "Voyage API retrying after backoff",
            extra={
                "attempt": attempt + 1,
                "batch_idx": batch_idx,
                "delay_sec": f"{delay:.2f}",
            },
        )
        time.sleep(delay)

    def _extract_error_detail(self, response: Optional[httpx.Response]) -> str:
        if response is None:
            return "no response body"
        try:
            payload = response.json()
        except Exception:
            return response.text[:200]
        return str(payload.get("detail") or payload)

    def _parse_standard_embeddings(self, payload: Dict[str, Any]) -> List[List[float]]:
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise RuntimeError("Voyage embeddings response missing 'data' list.")
        sorted_items = sorted(data, key=lambda item: item.get("index", 0))
        embeddings: List[List[float]] = []
        for item in sorted_items:
            embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError("Voyage embeddings response missing embedding data.")
            embeddings.append([float(value) for value in embedding])
        return embeddings

    def _parse_contextual_embeddings(
        self, payload: Dict[str, Any]
    ) -> List[List[List[float]]]:
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise RuntimeError(
                "Voyage contextual embeddings response missing top-level data list."
            )
        sorted_docs = sorted(data, key=lambda item: item.get("index", 0))
        results: List[List[List[float]]] = []
        for doc_entry in sorted_docs:
            doc_data = doc_entry.get("data") or []
            if not isinstance(doc_data, list):
                raise RuntimeError(
                    "Voyage contextual embeddings response missing nested data list."
                )
            sorted_chunks = sorted(doc_data, key=lambda item: item.get("index", 0))
            chunk_vectors: List[List[float]] = []
            for chunk in sorted_chunks:
                embedding = chunk.get("embedding")
                if embedding is None:
                    raise RuntimeError(
                        "Voyage contextual embeddings response missing embedding data."
                    )
                chunk_vectors.append([float(value) for value in embedding])
            results.append(chunk_vectors)
        return results

    def _batch_standard_inputs(self, texts: List[str]) -> List[List[str]]:
        max_inputs = int(os.getenv("VOYAGE_MAX_INPUTS", "1000"))
        if max_inputs <= 0:
            max_inputs = 1000
        batches: List[List[str]] = []
        current: List[str] = []
        for text in texts:
            current.append(text)
            if len(current) >= max_inputs:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def _batch_contextual_inputs(
        self, inputs: List[List[str]]
    ) -> List[List[List[str]]]:
        limits = self._contextual_limits or {}
        max_inputs = int(limits.get("max_inputs") or 1000)
        max_total_tokens = int(limits.get("max_total_tokens") or 120000)
        max_total_chunks = int(limits.get("max_total_chunks") or 16000)

        batches: List[List[List[str]]] = []
        current: List[List[str]] = []
        current_tokens = 0
        current_chunks = 0

        for doc_chunks in inputs:
            doc_token_count = self._count_tokens(doc_chunks)
            doc_chunk_count = len(doc_chunks)
            if doc_chunk_count > max_total_chunks:
                raise RuntimeError(
                    f"Document has {doc_chunk_count} chunks; exceeds contextual limit "
                    f"{max_total_chunks}."
                )
            if doc_token_count > max_total_tokens:
                raise RuntimeError(
                    f"Document has {doc_token_count} tokens; exceeds contextual limit "
                    f"{max_total_tokens}."
                )
            if current and (
                len(current) + 1 > max_inputs
                or current_tokens + doc_token_count > max_total_tokens
                or current_chunks + doc_chunk_count > max_total_chunks
            ):
                batches.append(current)
                current = []
                current_tokens = 0
                current_chunks = 0

            current.append(doc_chunks)
            current_tokens += doc_token_count
            current_chunks += doc_chunk_count

        if current:
            batches.append(current)
        return batches

    def _count_tokens(self, texts: Iterable[str]) -> int:
        items = list(texts)
        if not items:
            return 0
        try:
            return self._tokenizer_service.count_tokens_batch(items)
        except Exception as exc:
            logger.warning(
                "Voyage token counting failed; falling back to per-text counts",
                extra={"error": str(exc)},
            )
        return sum(self._tokenizer_service.count_tokens(text) for text in items)

    def _validate_contextual_inputs(
        self, inputs: List[List[str]], input_type: Optional[str]
    ) -> None:
        if input_type == "query":
            for idx, entry in enumerate(inputs):
                if len(entry) != 1:
                    raise ValueError(
                        "Contextual query inputs must be length-1 lists. "
                        f"Input {idx} has {len(entry)} elements."
                    )
        for idx, entry in enumerate(inputs):
            if not entry:
                raise ValueError(
                    f"Contextual embedding input {idx} must contain at least one text."
                )

    def _validate_dimensions(
        self, vectors: Sequence[Sequence[float]], expected_dims: Optional[int] = None
    ) -> None:
        dims = expected_dims or self._dims
        if not dims:
            return
        for vector in vectors:
            if len(vector) != dims:
                raise RuntimeError(
                    f"Voyage API returned {len(vector)}-D vector, expected {dims}."
                )
