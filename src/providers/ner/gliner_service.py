"""
GLiNER Service for zero-shot Named Entity Recognition.

This module provides a singleton service for extracting domain-specific
entities from text using the GLiNER model. It supports two modes:

1. HTTP Mode (preferred): Calls external GLiNER service with MPS acceleration
2. Local Mode (fallback): Loads model in-process (CPU in Docker)

The service automatically tries HTTP first if configured, falling back
to local mode if the external service is unavailable.

Usage:
    from src.providers.ner import GLiNERService

    service = GLiNERService()
    entities = service.extract_entities("Configure NFS mount on RHEL", labels)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional

import httpx
from prometheus_client import Counter, Histogram

from src.shared.config import get_config
from src.shared.observability import get_logger, get_tracer

if TYPE_CHECKING:
    from gliner import GLiNER

logger = get_logger(__name__)
tracer = get_tracer(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

GLINER_EXTRACTION_DURATION = Histogram(
    "gliner_extraction_duration_seconds",
    "Time spent extracting entities with GLiNER",
    ["operation", "mode"],  # single/batch, http/local
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

GLINER_ENTITIES_EXTRACTED = Counter(
    "gliner_entities_extracted_total",
    "Total entities extracted by GLiNER",
    ["label"],
)

GLINER_EXTRACTION_ERRORS = Counter(
    "gliner_extraction_errors_total",
    "Total GLiNER extraction errors",
    ["error_type", "mode"],
)

GLINER_CACHE_HITS = Counter(
    "gliner_cache_hits_total",
    "Total cache hits for query entity extraction",
)

GLINER_CACHE_MISSES = Counter(
    "gliner_cache_misses_total",
    "Total cache misses for query entity extraction",
)

GLINER_HTTP_FALLBACKS = Counter(
    "gliner_http_fallbacks_total",
    "Times HTTP service failed and fell back to local model",
)


# ============================================================================
# Entity Data Class
# ============================================================================


@dataclass(frozen=True)
class Entity:
    """
    Represents an extracted named entity.

    Attributes:
        text: The entity text as it appears in the source
        label: The entity type/label
        start: Character offset where entity starts
        end: Character offset where entity ends
        score: Confidence score from GLiNER (0.0-1.0)
    """

    text: str
    label: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


# ============================================================================
# GLiNER Service (Singleton)
# ============================================================================


class GLiNERService:
    """
    Singleton service for GLiNER entity extraction.

    Supports two modes:
    - HTTP Mode: Calls external service (MPS-accelerated, 5-10x faster)
    - Local Mode: In-process GLiNER model (CPU fallback in Docker)

    The service tries HTTP first if configured, falling back to local.

    Thread Safety:
        The singleton pattern uses Python's GIL for thread safety during
        initialization. GLiNER inference is generally thread-safe for reads.
    """

    _instance: Optional["GLiNERService"] = None
    _initialized: bool = False

    # HTTP client settings
    HTTP_TIMEOUT = 60.0  # seconds (GLiNER can be slow on first call)
    HTTP_CONNECT_TIMEOUT = 5.0  # seconds

    def __new__(cls) -> "GLiNERService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        config = get_config()
        self._config = config.ner
        self._model_name = self._config.model_name
        self._threshold = self._config.threshold
        self._default_batch_size = self._config.batch_size
        self._service_url = self._config.service_url

        # Mode tracking
        self._http_available: Optional[bool] = None  # None = not checked yet
        self._mode: str = "unknown"  # http, local, or disabled

        # Local model state (lazy loaded)
        self._device: Optional[str] = None
        self._model: Optional["GLiNER"] = None
        self._model_load_failed: bool = False

        # HTTP client (lazy initialized)
        self._http_client: Optional[httpx.Client] = None

        self._initialized = True

        # Log initialization
        if self._service_url:
            logger.info(
                "GLiNERService initialized (HTTP mode preferred)",
                extra={
                    "service_url": self._service_url,
                    "model": self._model_name,
                    "threshold": self._threshold,
                    "fallback": "local",
                },
            )
        else:
            logger.info(
                "GLiNERService initialized (local mode only)",
                extra={
                    "model": self._model_name,
                    "device": "auto",
                    "threshold": self._threshold,
                    "batch_size": self._default_batch_size,
                },
            )

    # ========================================================================
    # HTTP Client Mode
    # ========================================================================

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(
                    self.HTTP_TIMEOUT,
                    connect=self.HTTP_CONNECT_TIMEOUT,
                ),
            )
        return self._http_client

    def _check_http_health(self) -> bool:
        """Check if external GLiNER service is healthy."""
        if not self._service_url:
            return False

        try:
            client = self._get_http_client()
            response = client.get(f"{self._service_url}/healthz")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    device = data.get("device", "unknown")
                    logger.info(
                        f"GLiNER HTTP service healthy (device={device})",
                        extra={"service_url": self._service_url, "device": device},
                    )
                    return True
        except httpx.ConnectError:
            logger.debug(f"GLiNER HTTP service not reachable: {self._service_url}")
        except Exception as e:
            logger.warning(f"GLiNER HTTP health check failed: {e}")

        return False

    def _http_batch_extract(
        self,
        texts: List[str],
        labels: List[str],
        threshold: float,
    ) -> Optional[List[List[Entity]]]:
        """
        Extract entities via HTTP service.

        Returns:
            List of entity lists if successful, None if HTTP failed
        """
        if not self._service_url:
            return None

        try:
            client = self._get_http_client()
            response = client.post(
                f"{self._service_url}/v1/extract",
                json={
                    "texts": texts,
                    "labels": labels,
                    "threshold": threshold,
                    "flat_ner": True,
                },
            )

            if response.status_code != 200:
                logger.warning(
                    f"GLiNER HTTP extraction failed: {response.status_code}",
                    extra={"response": response.text[:200]},
                )
                return None

            data = response.json()
            results = []

            for text_result in data.get("results", []):
                entities = []
                for e in text_result.get("entities", []):
                    entities.append(
                        Entity(
                            text=e["text"],
                            label=e["label"],
                            start=e["start"],
                            end=e["end"],
                            score=e["score"],
                        )
                    )
                results.append(entities)

            # Log success
            elapsed_ms = data.get("elapsed_ms", 0)
            device = data.get("device", "unknown")
            logger.debug(
                f"HTTP extraction: {len(texts)} texts, "
                f"{data.get('total_entities', 0)} entities in {elapsed_ms:.1f}ms "
                f"(device={device})"
            )

            return results

        except httpx.ConnectError:
            logger.warning("GLiNER HTTP service connection failed")
            return None
        except httpx.TimeoutException:
            logger.warning("GLiNER HTTP service timeout")
            GLINER_EXTRACTION_ERRORS.labels(error_type="timeout", mode="http").inc()
            return None
        except Exception as e:
            logger.warning(f"GLiNER HTTP extraction error: {e}")
            GLINER_EXTRACTION_ERRORS.labels(error_type="http_error", mode="http").inc()
            return None

    # ========================================================================
    # Local Model Mode
    # ========================================================================

    def _detect_device(self, device_config: str) -> str:
        """
        Auto-detect the best available compute device.

        Priority: MPS (Apple Silicon) → CUDA → CPU
        """
        if device_config != "auto":
            return device_config

        try:
            import torch

            if torch.backends.mps.is_available():
                logger.info("GLiNER: Using MPS (Apple Silicon)")
                return "mps"
            if torch.cuda.is_available():
                logger.info("GLiNER: Using CUDA")
                return "cuda"
        except ImportError:
            logger.warning("PyTorch not available, GLiNER will use CPU")
        except Exception as e:
            logger.warning(f"Device detection failed: {e}")

        logger.info("GLiNER: Using CPU")
        return "cpu"

    def _load_model(self) -> bool:
        """
        Lazy-load the GLiNER model for local inference.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model is not None:
            return True

        if self._model_load_failed:
            # Circuit breaker: don't retry failed loads
            return False

        # Detect device on first load
        if self._device is None:
            self._device = self._detect_device(self._config.device)

        with tracer.start_as_current_span("gliner_load_model") as span:
            span.set_attribute("model_name", self._model_name)
            span.set_attribute("device", self._device)

            try:
                from gliner import GLiNER

                logger.info(f"Loading GLiNER model: {self._model_name}")
                self._model = GLiNER.from_pretrained(self._model_name)
                self._model = self._model.to(self._device)
                logger.info(
                    "GLiNER model loaded successfully",
                    extra={"model": self._model_name, "device": self._device},
                )
                return True

            except ImportError as e:
                logger.error(f"GLiNER package not installed: {e}")
                GLINER_EXTRACTION_ERRORS.labels(
                    error_type="import_error", mode="local"
                ).inc()
                self._model_load_failed = True
                span.record_exception(e)
                return False

            except Exception as e:
                logger.error(f"Failed to load GLiNER model: {e}")
                GLINER_EXTRACTION_ERRORS.labels(
                    error_type="model_load_error", mode="local"
                ).inc()
                self._model_load_failed = True
                span.record_exception(e)
                return False

    def _local_batch_extract(
        self,
        texts: List[str],
        labels: List[str],
        threshold: float,
    ) -> List[List[Entity]]:
        """
        Extract entities using local GLiNER model.

        Returns:
            List of entity lists (empty lists if model unavailable)
        """
        if not self._load_model():
            return [[] for _ in texts]

        all_entities: List[List[Entity]] = []
        batch_size = self._default_batch_size

        with tracer.start_as_current_span("gliner_local_batch_extract") as span:
            span.set_attribute("text_count", len(texts))
            span.set_attribute("batch_size", batch_size)
            span.set_attribute("device", self._device)

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                try:
                    with GLINER_EXTRACTION_DURATION.labels(
                        operation="batch", mode="local"
                    ).time():
                        batch_results = self._model.batch_predict_entities(
                            batch, labels, threshold=threshold
                        )

                    for raw_entities in batch_results:
                        entities = []
                        for e in raw_entities:
                            entity = Entity(
                                text=e["text"],
                                label=e["label"],
                                start=e["start"],
                                end=e["end"],
                                score=e.get("score", threshold),
                            )
                            entities.append(entity)
                            GLINER_ENTITIES_EXTRACTED.labels(label=entity.label).inc()
                        all_entities.append(entities)

                except Exception as e:
                    logger.error(f"Local batch extraction failed for batch {i}: {e}")
                    GLINER_EXTRACTION_ERRORS.labels(
                        error_type="batch_extraction_error", mode="local"
                    ).inc()
                    all_entities.extend([[] for _ in batch])
                    span.record_exception(e)

            total_entities = sum(len(ents) for ents in all_entities)
            span.set_attribute("total_entities_found", total_entities)

        return all_entities

    # ========================================================================
    # Public API
    # ========================================================================

    @property
    def is_available(self) -> bool:
        """Check if GLiNER service is available (HTTP or local)."""
        # Check HTTP first
        if self._service_url:
            if self._http_available is None:
                self._http_available = self._check_http_health()
            if self._http_available:
                return True

        # Fall back to local model check
        if self._model is not None:
            return True
        if self._model_load_failed:
            return False
        return self._load_model()

    @property
    def mode(self) -> str:
        """Return current operating mode: http, local, or disabled."""
        return self._mode

    def extract_entities(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
    ) -> List[Entity]:
        """
        Extract entities from a single text.

        For short texts (<200 chars), results are cached via LRU.

        Args:
            text: Input text to extract entities from
            labels: List of entity labels to extract
            threshold: Optional confidence threshold override

        Returns:
            List of extracted Entity objects
        """
        if not text or not labels:
            return []

        # Use cache for short texts (typical queries)
        if len(text) < 200:
            return self._extract_cached(text, tuple(sorted(labels)), threshold)

        return self._extract_single_impl(text, labels, threshold)

    @lru_cache(maxsize=1000)
    def _extract_cached(
        self,
        text: str,
        labels_tuple: tuple,
        threshold: Optional[float],
    ) -> List[Entity]:
        """Cached extraction for short texts (queries)."""
        GLINER_CACHE_MISSES.inc()
        return self._extract_single_impl(text, list(labels_tuple), threshold)

    def _extract_single_impl(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float],
    ) -> List[Entity]:
        """Single text extraction (delegates to batch)."""
        results = self.batch_extract_entities([text], labels, threshold)
        return results[0] if results else []

    def batch_extract_entities(
        self,
        texts: List[str],
        labels: List[str],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts efficiently.

        Tries HTTP service first (MPS-accelerated), falls back to local model.

        Args:
            texts: List of input texts
            labels: Entity labels to extract
            threshold: Confidence threshold override
            batch_size: Batch size override (only used for local mode)

        Returns:
            List of entity lists, one per input text
        """
        if not texts or not labels:
            return [[] for _ in texts]

        threshold = threshold or self._threshold
        start_time = time.time()

        # Try HTTP service first if configured
        if self._service_url:
            # Check health on first call
            if self._http_available is None:
                self._http_available = self._check_http_health()

            if self._http_available:
                with GLINER_EXTRACTION_DURATION.labels(
                    operation="batch", mode="http"
                ).time():
                    result = self._http_batch_extract(texts, labels, threshold)

                if result is not None:
                    self._mode = "http"
                    return result

                # HTTP failed, mark as unavailable and try local
                logger.warning("HTTP extraction failed, falling back to local model")
                self._http_available = False
                GLINER_HTTP_FALLBACKS.inc()

        # Local model extraction
        self._mode = "local"
        result = self._local_batch_extract(texts, labels, threshold)

        elapsed = time.time() - start_time
        total_entities = sum(len(ents) for ents in result)
        logger.debug(
            f"Batch extraction complete: {len(texts)} texts, "
            f"{total_entities} entities in {elapsed * 1000:.1f}ms (mode={self._mode})"
        )

        return result

    def reset(self) -> None:
        """
        Reset the service state (for testing).

        This clears the model, HTTP state, and caches.
        """
        self._model = None
        self._model_load_failed = False
        self._http_available = None
        self._mode = "unknown"
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self._extract_cached.cache_clear()
        logger.info("GLiNERService reset")


def get_gliner_service() -> GLiNERService:
    """
    Factory function to get the GLiNER service singleton.

    Returns:
        The GLiNERService singleton instance
    """
    return GLiNERService()
