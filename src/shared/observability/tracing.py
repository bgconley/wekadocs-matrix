# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §7 (Observability & SLOs)
# OpenTelemetry tracing setup

from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
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
        TracerProvider (always created, even without exporter)
    """
    try:
        logger.info(
            "Setting up OpenTelemetry tracing",
            endpoint=settings.otel_exporter_otlp_endpoint or "in-memory",
            service=settings.otel_service_name,
        )

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": "0.1.0",
            }
        )

        # Create tracer provider (always create, even without exporter)
        provider = TracerProvider(resource=resource)

        # Add exporter if endpoint configured
        if settings.otel_exporter_otlp_endpoint:
            # Create OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces"
            )
            # Add batch span processor
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP exporter configured")
        else:
            logger.info("OpenTelemetry tracing enabled (in-memory, no exporter)")

        # Set global tracer provider (important: do this even without exporter)
        trace.set_tracer_provider(provider)

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        # Instrument Redis
        RedisInstrumentor().instrument()

        logger.info("OpenTelemetry tracing enabled successfully")
        return provider

    except Exception as e:
        logger.error("Failed to setup OpenTelemetry tracing", error=str(e))
        # Still create a basic provider as fallback
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        return provider


def get_tracer(name: str):
    """Get a tracer instance"""
    return trace.get_tracer(name)
