# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §7 (Observability & SLOs)
# OpenTelemetry tracing setup

from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import Settings
from .logging import get_logger

logger = get_logger(__name__)


def setup_tracing(app, settings: Settings) -> Optional[TracerProvider]:
    """
    Setup OpenTelemetry tracing for FastAPI and instrumented libraries.

    Args:
        app: FastAPI application instance
        settings: Application settings

    Returns:
        TracerProvider if tracing is enabled, None otherwise
    """
    if not settings.otel_exporter_otlp_endpoint:
        logger.info("OpenTelemetry tracing disabled (no endpoint configured)")
        return None

    try:
        logger.info(
            "Setting up OpenTelemetry tracing",
            endpoint=settings.otel_exporter_otlp_endpoint,
            service=settings.otel_service_name,
        )

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": "0.1.0",
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces"
        )

        # Add batch span processor
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        # Instrument Redis
        RedisInstrumentor().instrument()

        logger.info("OpenTelemetry tracing enabled successfully")
        return provider

    except Exception as e:
        logger.error("Failed to setup OpenTelemetry tracing", error=str(e))
        return None


def get_tracer(name: str):
    """Get a tracer instance"""
    return trace.get_tracer(name)
