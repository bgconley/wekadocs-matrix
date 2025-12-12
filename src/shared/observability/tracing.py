# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §7 (Observability & SLOs)
# OpenTelemetry tracing setup
# Enhanced for LGTM observability stack (Phase LGTM)

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGrpc,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import Settings
from .logging import get_logger

logger = get_logger(__name__)

# Service version - should be updated on releases
SERVICE_VERSION_VALUE = "1.0.0"


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


def init_tracing(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    instrument_redis: bool = True,
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing for standalone services (non-FastAPI).

    This function is designed for background workers like ingestion-worker
    that don't have a FastAPI application to instrument.

    Configuration via environment variables:
      - OTEL_SERVICE_NAME: Service name (default: from service_name param)
      - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://alloy:4318)
      - OTEL_TRACES_EXPORTER: Export protocol (default: otlp)

    Args:
        service_name: Service identifier (defaults to OTEL_SERVICE_NAME env var)
        service_version: Service version (defaults to SERVICE_VERSION_VALUE)
        instrument_redis: Whether to auto-instrument Redis client

    Returns:
        Configured TracerProvider

    Example:
        >>> from src.shared.observability.tracing import init_tracing, get_tracer
        >>> init_tracing("weka-ingestion-worker")
        >>> tracer = get_tracer("ingestion")
        >>> with tracer.start_as_current_span("process_document") as span:
        ...     span.set_attribute("doc_id", "abc123")
    """
    # Resolve service name from env or parameter
    resolved_service_name = service_name or os.getenv(
        "OTEL_SERVICE_NAME", "unknown-service"
    )
    resolved_version = service_version or SERVICE_VERSION_VALUE
    environment = os.getenv("ENV", "development")

    logger.info(
        "Initializing OpenTelemetry tracing",
        service=resolved_service_name,
        version=resolved_version,
        environment=environment,
    )

    # Create resource with service info
    resource = Resource.create(
        {
            SERVICE_NAME: resolved_service_name,
            SERVICE_VERSION: resolved_version,
            "deployment.environment": environment,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter based on OTEL_EXPORTER_OTLP_ENDPOINT
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            # Use HTTP exporter (more widely compatible)
            if endpoint.startswith("http"):
                exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            else:
                # Assume gRPC for non-HTTP endpoints
                exporter = OTLPSpanExporterGrpc(endpoint=endpoint, insecure=True)

            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTLP trace exporter configured", endpoint=endpoint)
        except Exception as e:
            logger.warning("Failed to configure OTLP exporter", error=str(e))
    else:
        logger.info("OpenTelemetry tracing enabled (in-memory, no exporter)")

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument Redis if requested
    if instrument_redis:
        try:
            RedisInstrumentor().instrument()
            logger.debug("Redis auto-instrumentation enabled")
        except Exception as e:
            logger.warning("Failed to instrument Redis", error=str(e))

    logger.info("OpenTelemetry tracing initialized successfully")
    return provider
