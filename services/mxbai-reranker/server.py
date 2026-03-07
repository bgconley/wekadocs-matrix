"""
mxbai-rerank HTTP service for cross-encoder reranking.

Serves mixedbread-ai/mxbai-rerank-large-v2 (or base-v2) behind the
standard /v1/rerank contract expected by LocalRerankerServiceProvider.

Env vars:
    MXBAI_MODEL_ID   — HuggingFace model ID (default: mixedbread-ai/mxbai-rerank-large-v2)
    MXBAI_DEVICE     — cuda, mps, cpu, or auto (default: auto)
    MXBAI_DTYPE      — float16, float32, or auto (default: float16)
    MXBAI_MAX_LENGTH — max input length (default: 8192)
    MXBAI_MAX_CONCURRENCY — max concurrent requests (default: 1)
    MXBAI_PORT       — listen port (default: 9006)
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class ServiceConfig:
    model_id: str = os.getenv("MXBAI_MODEL_ID", "mixedbread-ai/mxbai-rerank-large-v2")
    device: str = os.getenv("MXBAI_DEVICE", "auto")
    dtype: str = os.getenv("MXBAI_DTYPE", "float16")
    max_length: int = int(os.getenv("MXBAI_MAX_LENGTH", "8192"))
    max_concurrency: int = int(os.getenv("MXBAI_MAX_CONCURRENCY", "1"))
    port: int = int(os.getenv("MXBAI_PORT", "9006"))


_config = ServiceConfig()
_model = None
_semaphore: Optional[asyncio.Semaphore] = None
_warmup_ok = False


# ── Device detection ──────────────────────────────────────────────────


def detect_device(device_config: str) -> str:
    if device_config != "auto":
        return device_config
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model loading ────────────────────────────────────────────────────


def load_model(model_id: str, device: str, max_length: int):
    """Load the MxbaiRerankV2 model."""
    from mxbai_rerank import MxbaiRerankV2

    logger.info(f"Loading model {model_id} (max_length={max_length})...")
    model = MxbaiRerankV2(model_id, max_length=max_length)

    # Move to device if needed
    if device == "cuda" and hasattr(model, "model"):
        model.model.to("cuda")
    elif device == "mps" and hasattr(model, "model"):
        model.model.to("mps")

    logger.info(f"Model loaded on {device}")
    return model


# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _semaphore, _warmup_ok

    device = detect_device(_config.device)
    _model = load_model(_config.model_id, device, _config.max_length)
    _semaphore = asyncio.Semaphore(_config.max_concurrency)

    # Warmup with dummy data
    try:
        logger.info("Running warmup rerank...")
        _model.rank("warmup query", ["warmup document one", "warmup document two"])
        _warmup_ok = True
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        _warmup_ok = False

    logger.info(
        f"mxbai-rerank service ready: model={_config.model_id} "
        f"device={device} max_length={_config.max_length} port={_config.port}"
    )
    yield
    _model = None
    logger.info("mxbai-rerank service shutting down")


# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(title="mxbai-rerank", lifespan=lifespan)


# ── Request/Response models ──────────────────────────────────────────


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text")
    documents: List[str] = Field(..., min_length=1, description="Documents to rerank")
    model: Optional[str] = Field(
        default=None, description="Model ID (ignored, single-model service)"
    )
    instruction: Optional[str] = Field(
        default=None, description="Optional reranking instruction"
    )
    top_k: Optional[int] = Field(
        default=None, description="Number of top results to return"
    )

    @field_validator("documents")
    @classmethod
    def documents_not_empty(cls, v):
        if not v:
            raise ValueError("documents must not be empty")
        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f"document at index {i} must not be empty")
        return v


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str
    latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _model is not None else "loading",
        "model": _config.model_id,
        "device": detect_device(_config.device),
        "dtype": _config.dtype,
        "max_length": _config.max_length,
        "loaded": _model is not None,
        "warmup_ok": _warmup_ok,
    }


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()
    top_k = request.top_k or len(request.documents)

    async with _semaphore:
        try:
            # mxbai-rerank library call
            kwargs = {"return_documents": False, "top_k": top_k}
            if request.instruction:
                kwargs["instruction"] = request.instruction

            results = _model.rank(request.query, request.documents, **kwargs)
        except Exception as e:
            logger.error(f"Rerank failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Rerank failed: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    # Map library output to contract response
    # mxbai-rerank returns list of objects with .index and .score attributes
    response_results = [
        RerankResult(index=r.index, score=float(r.score)) for r in results
    ]

    logger.info(
        f"rerank_complete: docs={len(request.documents)} top_k={top_k} "
        f"latency_ms={latency_ms:.1f} top_score={response_results[0].score:.4f}"
        if response_results
        else f"rerank_complete: docs={len(request.documents)} no_results"
    )

    return RerankResponse(
        results=response_results,
        model=_config.model_id,
        latency_ms=round(latency_ms, 1),
    )


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_config.port)
