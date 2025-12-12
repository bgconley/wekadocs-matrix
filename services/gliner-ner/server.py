#!/usr/bin/env python3
"""
GLiNER NER Service - Native macOS service with MPS acceleration.

Provides HTTP API for Named Entity Recognition using GLiNER zero-shot NER.
Designed to run natively on macOS to leverage Metal Performance Shaders (MPS)
for 5-10x faster inference on Apple Silicon.

Usage:
    python server.py                    # Auto-detect device (MPS preferred)
    python server.py --device cpu       # Force CPU mode
    python server.py --port 9002        # Custom port

API Endpoints:
    GET  /healthz              - Health check with device info
    POST /v1/extract           - Extract entities from texts
    GET  /v1/config            - Current model configuration

Thread Safety:
    The service runs as a single-process uvicorn worker. Concurrent HTTP
    requests are queued and processed serially through Python's GIL.
    This architecture provides implicit thread safety for model inference
    and is optimal for typical ingestion workloads (2-4 concurrent workers).
"""

import argparse
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from gliner import GLiNER
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gliner-ner-service")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_THRESHOLD = 0.45
DEFAULT_PORT = 9002


@dataclass
class ServiceConfig:
    model_id: str = DEFAULT_MODEL
    threshold: float = DEFAULT_THRESHOLD
    device: str = "auto"
    port: int = DEFAULT_PORT


# Global state
_config: Optional[ServiceConfig] = None
_model: Optional[GLiNER] = None
_device: Optional[str] = None


# ============================================================================
# Device Detection
# ============================================================================


def detect_device(device_config: str) -> str:
    """Auto-detect best available compute device."""
    if device_config != "auto":
        logger.info(f"Using explicitly configured device: {device_config}")
        return device_config

    try:
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected and available")
            return "mps"
        if torch.cuda.is_available():
            logger.info("CUDA detected and available")
            return "cuda"
    except Exception as e:
        logger.warning(f"Device detection error: {e}")

    logger.info("Falling back to CPU")
    return "cpu"


# ============================================================================
# Model Loading
# ============================================================================


def load_model(model_id: str, device: str) -> GLiNER:
    """Load GLiNER model onto specified device."""
    logger.info(f"Loading GLiNER model: {model_id} on {device}")
    start = time.time()

    model = GLiNER.from_pretrained(model_id)

    # Move to device
    if device == "mps":
        model = model.to("mps")
    elif device == "cuda":
        model = model.to("cuda")
    # CPU is default, no explicit move needed

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.2f}s on {device}")
    return model


# ============================================================================
# FastAPI App
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global _model, _device, _config

    _device = detect_device(_config.device)
    _model = load_model(_config.model_id, _device)

    logger.info(
        f"GLiNER NER Service ready on port {_config.port} "
        f"(device={_device}, model={_config.model_id})"
    )
    yield

    # Cleanup
    _model = None
    logger.info("GLiNER NER Service shutting down")


app = FastAPI(
    title="GLiNER NER Service",
    description="Native macOS NER service with MPS acceleration",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Request/Response Models
# ============================================================================


class ExtractRequest(BaseModel):
    """Request for entity extraction."""

    texts: list[str] = Field(..., description="List of texts to extract entities from")
    labels: list[str] = Field(..., description="Entity labels to extract")
    threshold: Optional[float] = Field(
        None, description="Confidence threshold (default: 0.45)"
    )
    flat_ner: bool = Field(True, description="Use flat NER (no nested entities)")


class Entity(BaseModel):
    """Extracted entity."""

    text: str
    label: str
    score: float
    start: int
    end: int


class TextEntities(BaseModel):
    """Entities for a single text."""

    text_hash: str
    entities: list[Entity]


class ExtractResponse(BaseModel):
    """Response from entity extraction."""

    results: list[TextEntities]
    device: str
    model: str
    elapsed_ms: float
    total_entities: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device: str
    model: str
    mps_available: bool
    cuda_available: bool


class ConfigResponse(BaseModel):
    """Configuration response."""

    model_id: str
    threshold: float
    device: str
    port: int


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check with device information."""
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        device=_device or "unknown",
        model=_config.model_id,
        mps_available=(
            torch.backends.mps.is_available()
            if hasattr(torch.backends, "mps")
            else False
        ),
        cuda_available=torch.cuda.is_available(),
    )


@app.get("/v1/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        model_id=_config.model_id,
        threshold=_config.threshold,
        device=_device or _config.device,
        port=_config.port,
    )


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    """Extract entities from texts."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    threshold = request.threshold or _config.threshold

    # Clean labels (strip parenthetical examples)
    clean_labels = []
    for label in request.labels:
        if "(" in label:
            clean_labels.append(label.split("(")[0].strip())
        else:
            clean_labels.append(label.strip())

    # Batch predict
    try:
        # GLiNER batch_predict_entities returns list of list of entities
        all_entities = _model.batch_predict_entities(
            request.texts,
            clean_labels,
            threshold=threshold,
            flat_ner=request.flat_ner,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Format results
    results = []
    total_entities = 0

    for text, entities in zip(request.texts, all_entities):
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        formatted_entities = [
            Entity(
                text=e["text"],
                label=e["label"],
                score=e["score"],
                start=e["start"],
                end=e["end"],
            )
            for e in entities
        ]
        results.append(TextEntities(text_hash=text_hash, entities=formatted_entities))
        total_entities += len(formatted_entities)

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        f"Extracted {total_entities} entities from {len(request.texts)} texts "
        f"in {elapsed_ms:.1f}ms"
    )

    return ExtractResponse(
        results=results,
        device=_device,
        model=_config.model_id,
        elapsed_ms=elapsed_ms,
        total_entities=total_entities,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    global _config

    parser = argparse.ArgumentParser(description="GLiNER NER Service")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    _config = ServiceConfig(
        model_id=args.model,
        threshold=args.threshold,
        device=args.device,
        port=args.port,
    )

    logger.info(f"Starting GLiNER NER Service on {args.host}:{args.port}")
    logger.info(
        f"Model: {args.model}, Device: {args.device}, Threshold: {args.threshold}"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
