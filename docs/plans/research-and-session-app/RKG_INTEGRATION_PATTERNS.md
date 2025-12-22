# RKG Integration Implementation Patterns

## Document Overview

This document provides detailed implementation patterns for:
1. LangFuse - Complete tracing and prompt management integration
2. NewRelic - APM, distributed tracing, and infrastructure monitoring
3. Infisical - Secrets management with tiered features

---

# 1. LangFuse Integration Patterns

## 1.1 Full Tracing Implementation

### Context Manager Pattern for Complex Operations

```python
# src/rkg_mcp/observability/langfuse_patterns.py

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from langfuse import get_client
import time

class LangFuseObservability:
    """
    Complete LangFuse observability patterns for RKG.
    """

    def __init__(self):
        self._client = None
        self._enabled = False

    async def initialize(self, settings):
        """Initialize with settings."""
        if not settings.langfuse_enabled:
            return

        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url
            )
            self._enabled = self._client.auth_check()
        except Exception as e:
            logger.warning(f"LangFuse initialization failed: {e}")

    @asynccontextmanager
    async def trace_mcp_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: list[str] = None
    ) -> AsyncGenerator:
        """
        Trace an MCP tool invocation.

        Usage:
            async with langfuse.trace_mcp_tool("rkg_search", {"query": "..."}) as trace:
                result = await perform_search()
                trace.set_output(result)
        """
        if not self._enabled:
            yield _NoOpTrace()
            return

        trace = self._client.trace(
            name=f"mcp_tool:{tool_name}",
            input=input_data,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [tool_name, "mcp"]
        )

        wrapper = _TraceWrapper(trace)
        start_time = time.time()

        try:
            yield wrapper
        except Exception as e:
            trace.update(
                level="ERROR",
                status_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            trace.update(
                output=wrapper._output,
                metadata={
                    **(wrapper._metadata or {}),
                    "duration_ms": duration_ms
                }
            )

    @asynccontextmanager
    async def trace_embedding(
        self,
        provider: str,
        model: str,
        input_type: str,
        batch_size: int = 1
    ) -> AsyncGenerator:
        """Trace embedding generation."""
        if not self._enabled:
            yield _NoOpTrace()
            return

        with self._client.start_as_current_observation(
            as_type="generation",
            name=f"embed:{provider}",
            model=model,
            input={"input_type": input_type, "batch_size": batch_size},
            model_parameters={"model": model, "input_type": input_type}
        ) as generation:
            wrapper = _GenerationWrapper(generation)
            yield wrapper

    @asynccontextmanager
    async def trace_search(
        self,
        search_type: str,  # "semantic" | "lexical" | "hybrid"
        query: str,
        filters: Dict[str, Any] = None
    ) -> AsyncGenerator:
        """Trace a search operation."""
        if not self._enabled:
            yield _NoOpTrace()
            return

        with self._client.start_as_current_observation(
            as_type="span",
            name=f"search:{search_type}",
            input={"query": query, "filters": filters}
        ) as span:
            wrapper = _SpanWrapper(span)
            yield wrapper

    @asynccontextmanager
    async def trace_reranking(
        self,
        model: str,
        query: str,
        num_candidates: int
    ) -> AsyncGenerator:
        """Trace reranking operation."""
        if not self._enabled:
            yield _NoOpTrace()
            return

        with self._client.start_as_current_observation(
            as_type="generation",
            name="rerank",
            model=model,
            input={"query": query, "num_candidates": num_candidates}
        ) as generation:
            wrapper = _GenerationWrapper(generation)
            yield wrapper

    def log_event(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
        level: str = "DEFAULT"
    ):
        """Log a discrete event."""
        if not self._enabled:
            return

        self._client.event(
            name=name,
            metadata=metadata,
            level=level
        )

    async def flush(self):
        """Ensure all traces are sent."""
        if self._enabled:
            self._client.flush()


class _TraceWrapper:
    """Wrapper to collect trace data."""
    def __init__(self, trace):
        self._trace = trace
        self._output = None
        self._metadata = {}

    def set_output(self, output: Any):
        self._output = output

    def add_metadata(self, key: str, value: Any):
        self._metadata[key] = value

    def add_span(self, name: str, input_data: Any = None):
        return self._trace.span(name=name, input=input_data)


class _GenerationWrapper:
    """Wrapper for generation traces."""
    def __init__(self, generation):
        self._generation = generation

    def set_output(self, output: Any):
        self._generation.update(output=output)

    def set_usage(self, input_tokens: int, output_tokens: int = 0, total_cost: float = None):
        usage = {"input": input_tokens, "output": output_tokens}
        if total_cost:
            usage["total_cost"] = total_cost
        self._generation.update(usage=usage)


class _SpanWrapper:
    """Wrapper for span traces."""
    def __init__(self, span):
        self._span = span

    def set_output(self, output: Any):
        self._span.update(output=output)

    def add_event(self, name: str, metadata: Dict = None):
        self._span.event(name=name, metadata=metadata)


class _NoOpTrace:
    """No-op trace for when LangFuse is disabled."""
    def set_output(self, *args, **kwargs): pass
    def add_metadata(self, *args, **kwargs): pass
    def add_span(self, *args, **kwargs): return self
    def set_usage(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
```

### Instrumented MCP Tool Example

```python
# src/rkg_mcp/tools/search_instrumented.py

from ..observability.langfuse_patterns import LangFuseObservability

langfuse = LangFuseObservability()

@mcp.tool()
async def rkg_semantic_search(
    query: str,
    limit: int = 10,
    filters: Dict[str, Any] = None,
    use_reranker: bool = True
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph with full LangFuse tracing.
    """
    async with langfuse.trace_mcp_tool(
        "rkg_semantic_search",
        {"query": query, "limit": limit, "filters": filters, "use_reranker": use_reranker}
    ) as trace:

        # Trace embedding generation
        async with langfuse.trace_embedding(
            provider="voyage",
            model="voyage-3-large",
            input_type="query"
        ) as embed_trace:
            query_embedding = await embedder.embed_text(query, "query")
            embed_trace.set_usage(input_tokens=len(query.split()))

        # Trace search
        async with langfuse.trace_search("hybrid", query, filters) as search_trace:
            candidates = await hybrid_search(query_embedding, filters, limit * 3)
            search_trace.set_output({"candidate_count": len(candidates)})

        # Trace reranking (if enabled)
        if use_reranker and candidates:
            async with langfuse.trace_reranking(
                model="rerank-2.5",
                query=query,
                num_candidates=len(candidates)
            ) as rerank_trace:
                results = await reranker.rerank(query, candidates, limit)
                rerank_trace.set_output({"result_count": len(results)})
        else:
            results = candidates[:limit]

        # Enrich with graph context
        enriched = await enrich_with_graph_context(results)

        trace.set_output(enriched)
        trace.add_metadata("result_count", len(enriched))

        return enriched
```

## 1.2 LangFuse Scores and Evaluation

```python
# src/rkg_mcp/observability/evaluation.py

class LangFuseEvaluator:
    """Record evaluation scores for search results."""

    def __init__(self, langfuse: LangFuseObservability):
        self.langfuse = langfuse

    async def score_search_relevance(
        self,
        trace_id: str,
        query: str,
        results: List[Dict],
        relevance_scores: List[float]  # Human or automated scores
    ):
        """Score search result relevance."""
        if not self.langfuse._enabled:
            return

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        self.langfuse._client.score(
            trace_id=trace_id,
            name="search_relevance",
            value=avg_relevance,
            comment=f"Average relevance for query: {query[:50]}..."
        )

        # Also score individual results
        for i, (result, score) in enumerate(zip(results, relevance_scores)):
            self.langfuse._client.score(
                trace_id=trace_id,
                observation_id=result.get("observation_id"),
                name="result_relevance",
                value=score,
                comment=f"Result {i+1}: {result.get('title', 'Unknown')[:30]}"
            )

    async def score_entity_extraction(
        self,
        trace_id: str,
        precision: float,
        recall: float,
        f1: float
    ):
        """Score entity extraction quality."""
        if not self.langfuse._enabled:
            return

        self.langfuse._client.score(
            trace_id=trace_id,
            name="entity_precision",
            value=precision
        )
        self.langfuse._client.score(
            trace_id=trace_id,
            name="entity_recall",
            value=recall
        )
        self.langfuse._client.score(
            trace_id=trace_id,
            name="entity_f1",
            value=f1
        )
```

---

# 2. NewRelic Integration Patterns

## 2.1 Application Initialization

```python
# src/rkg_mcp/main.py

import newrelic.agent
from .observability.newrelic_agent import NewRelicAgent
from .config import get_settings

async def initialize_application():
    """Initialize application with NewRelic."""
    settings = get_settings()

    # Initialize NewRelic first (before other imports)
    if settings.newrelic_enabled:
        NewRelicAgent.initialize(settings.newrelic_config_file)

    # Load secrets from Infisical
    settings = await load_secrets_from_infisical(settings)

    # Initialize other components
    await initialize_storage(settings)
    await initialize_observability(settings)

    return settings
```

## 2.2 Custom Transaction Naming

```python
# src/rkg_mcp/observability/newrelic_patterns.py

import newrelic.agent
from typing import Callable
from functools import wraps

class NewRelicPatterns:
    """NewRelic integration patterns for RKG."""

    @staticmethod
    def set_transaction_name(name: str):
        """Set custom transaction name."""
        if NewRelicAgent.enabled:
            newrelic.agent.set_transaction_name(name)

    @staticmethod
    def web_transaction(name: str):
        """Decorator for web transactions."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                NewRelicPatterns.set_transaction_name(name)
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def record_mcp_tool_metrics(
        tool_name: str,
        duration_ms: float,
        success: bool,
        result_count: int = 0
    ):
        """Record custom metrics for MCP tools."""
        if not NewRelicAgent.enabled:
            return

        # Record duration
        newrelic.agent.record_custom_metric(
            f"Custom/MCP/Tool/{tool_name}/Duration",
            duration_ms
        )

        # Record success/failure
        status = "Success" if success else "Failure"
        newrelic.agent.record_custom_metric(
            f"Custom/MCP/Tool/{tool_name}/{status}",
            1
        )

        # Record result count for search tools
        if result_count > 0:
            newrelic.agent.record_custom_metric(
                f"Custom/MCP/Tool/{tool_name}/ResultCount",
                result_count
            )

    @staticmethod
    def record_storage_metrics(
        storage_type: str,  # "qdrant" | "neo4j"
        operation: str,     # "read" | "write" | "search"
        duration_ms: float,
        item_count: int = 0
    ):
        """Record storage operation metrics."""
        if not NewRelicAgent.enabled:
            return

        newrelic.agent.record_custom_metric(
            f"Custom/Storage/{storage_type}/{operation}/Duration",
            duration_ms
        )

        if item_count > 0:
            newrelic.agent.record_custom_metric(
                f"Custom/Storage/{storage_type}/{operation}/ItemCount",
                item_count
            )

    @staticmethod
    def record_embedding_metrics(
        provider: str,
        model: str,
        batch_size: int,
        duration_ms: float,
        token_count: int = 0
    ):
        """Record embedding generation metrics."""
        if not NewRelicAgent.enabled:
            return

        newrelic.agent.record_custom_metric(
            f"Custom/Embedding/{provider}/{model}/Duration",
            duration_ms
        )
        newrelic.agent.record_custom_metric(
            f"Custom/Embedding/{provider}/{model}/BatchSize",
            batch_size
        )
        if token_count > 0:
            newrelic.agent.record_custom_metric(
                f"Custom/Embedding/{provider}/{model}/Tokens",
                token_count
            )


# Decorator for automatic metrics collection
def track_mcp_tool(tool_name: str):
    """Decorator to automatically track MCP tool metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result_count = 0

            try:
                result = await func(*args, **kwargs)
                if isinstance(result, list):
                    result_count = len(result)
                return result
            except Exception as e:
                success = False
                NewRelicAgent.notice_error(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                NewRelicPatterns.record_mcp_tool_metrics(
                    tool_name,
                    duration_ms,
                    success,
                    result_count
                )
        return wrapper
    return decorator
```

## 2.3 Distributed Tracing Headers

```python
# src/rkg_mcp/observability/distributed_tracing.py

import newrelic.agent
from typing import Dict, Optional

class DistributedTracingHelper:
    """Helper for distributed tracing across services."""

    @staticmethod
    def get_trace_headers() -> Dict[str, str]:
        """Get headers for outbound requests."""
        headers = {}
        if NewRelicAgent.enabled:
            # W3C Trace Context headers
            transaction = newrelic.agent.current_transaction()
            if transaction:
                dt_headers = transaction.get_linking_metadata()
                headers["traceparent"] = dt_headers.get("trace.id", "")
                headers["tracestate"] = dt_headers.get("span.id", "")
        return headers

    @staticmethod
    def accept_trace_headers(headers: Dict[str, str]):
        """Accept trace context from inbound request."""
        if NewRelicAgent.enabled:
            transaction = newrelic.agent.current_transaction()
            if transaction:
                # Accept distributed trace payload
                transaction.accept_distributed_trace_headers(headers)

    @staticmethod
    def create_span(
        name: str,
        span_type: str = "generic"
    ):
        """Create a custom span for tracing."""
        if not NewRelicAgent.enabled:
            return _NoOpSpan()

        if span_type == "datastore":
            return newrelic.agent.DatastoreTrace(
                product="neo4j",
                target="",
                operation=name
            )
        elif span_type == "external":
            return newrelic.agent.ExternalTrace(
                library="httpx",
                url=name
            )
        else:
            return newrelic.agent.FunctionTrace(name=name)


class _NoOpSpan:
    """No-op span when NewRelic is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
```

## 2.4 Infrastructure Monitoring Dashboard Queries

```sql
-- NewRelic NRQL queries for RKG dashboards

-- MCP Tool Performance
SELECT average(duration) as 'Avg Duration (ms)',
       count(*) as 'Calls',
       percentage(count(*), WHERE error IS NULL) as 'Success Rate'
FROM Transaction
WHERE name LIKE 'Custom/MCP/Tool/%'
FACET name
SINCE 1 hour ago

-- Storage Performance
SELECT average(Custom.Storage.Duration) as 'Avg Duration',
       sum(Custom.Storage.ItemCount) as 'Total Items'
FROM Metric
WHERE metricName LIKE 'Custom/Storage/%'
FACET storage_type, operation
SINCE 1 hour ago

-- Embedding Generation
SELECT average(Custom.Embedding.Duration) as 'Avg Duration',
       sum(Custom.Embedding.Tokens) as 'Total Tokens',
       sum(Custom.Embedding.BatchSize) as 'Total Batches'
FROM Metric
WHERE metricName LIKE 'Custom/Embedding/%'
FACET provider, model
SINCE 1 hour ago

-- Error Rate by Component
SELECT count(*)
FROM TransactionError
FACET error.class, error.message
SINCE 1 hour ago

-- Apdex Score
SELECT apdex(duration, t: 0.5) as 'Apdex'
FROM Transaction
WHERE appName = 'rkg-mcp-server'
SINCE 1 hour ago
TIMESERIES
```

---

# 3. Infisical Integration Patterns

## 3.1 Complete Secret Management

```python
# src/rkg_mcp/secrets/patterns.py

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

class SecretManager:
    """
    Comprehensive secret management with Infisical.
    Supports Pro and Free tier features.
    """

    def __init__(self, infisical: InfisicalManager):
        self.infisical = infisical
        self._refresh_interval = 300  # 5 minutes
        self._refresh_task: asyncio.Task | None = None
        self._secrets: Dict[str, str] = {}

    async def start_auto_refresh(self):
        """Start background secret refresh task."""
        if self._refresh_task:
            return
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop_auto_refresh(self):
        """Stop background refresh."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

    async def _refresh_loop(self):
        """Background loop to refresh secrets."""
        while True:
            try:
                await self._refresh_all_secrets()
            except Exception as e:
                logger.error(f"Secret refresh failed: {e}")
            await asyncio.sleep(self._refresh_interval)

    async def _refresh_all_secrets(self):
        """Refresh all cached secrets."""
        self.infisical.clear_cache()

        # Re-fetch known secrets
        for key in list(self._secrets.keys()):
            try:
                value = await self.infisical.get_secret(key, use_cache=False)
                if value:
                    self._secrets[key] = value
            except Exception as e:
                logger.warning(f"Failed to refresh secret '{key}': {e}")

    async def get(self, key: str, default: str | None = None) -> str | None:
        """Get a secret value with caching."""
        if key in self._secrets:
            return self._secrets[key]

        value = await self.infisical.get_secret(key)
        if value:
            self._secrets[key] = value
            return value
        return default

    async def require(self, key: str) -> str:
        """Get a required secret (raises if missing)."""
        value = await self.get(key)
        if value is None:
            raise SecretNotFoundError(f"Required secret '{key}' not found")
        return value

    # Pro tier features
    async def get_with_version(
        self,
        key: str,
        version: int
    ) -> str | None:
        """Get a specific version of a secret (Pro tier)."""
        return await self.infisical.get_secret(key, version=version)

    async def rollback_secret(
        self,
        key: str,
        to_version: int
    ) -> bool:
        """Rollback a secret to a previous version (Pro tier)."""
        if self.infisical.tier != InfisicalTier.PRO:
            raise FeatureNotAvailableError("Rollback requires Pro tier")

        # Get the old version value
        old_value = await self.get_with_version(key, to_version)
        if not old_value:
            return False

        # Rotate to that value
        return await self.infisical.rotate_secret(key, old_value)

    async def audit_secret_access(
        self,
        key: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get audit log for secret access (Pro tier)."""
        if self.infisical.tier != InfisicalTier.PRO:
            raise FeatureNotAvailableError("Audit logs require Pro tier")

        # Pro tier API endpoint
        # Note: Implementation depends on Infisical API
        return []


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass
```

## 3.2 Environment-Specific Configuration

```python
# src/rkg_mcp/secrets/environments.py

from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

class EnvironmentSecretManager:
    """
    Manage secrets across environments.
    """

    # Secret paths by category
    SECRET_PATHS = {
        "api_keys": "/api-keys",
        "database": "/database",
        "observability": "/observability",
        "auth": "/auth"
    }

    def __init__(self, infisical: InfisicalManager, environment: Environment):
        self.infisical = infisical
        self.environment = environment
        self._loaded = False

    async def load_all(self) -> Dict[str, str]:
        """Load all secrets for current environment."""
        all_secrets = {}

        for category, path in self.SECRET_PATHS.items():
            try:
                secrets = await self.infisical.get_secrets_by_path(path)
                all_secrets.update(secrets)
                logger.info(f"Loaded {len(secrets)} secrets from {path}")
            except Exception as e:
                logger.warning(f"Failed to load secrets from {path}: {e}")

        self._loaded = True
        return all_secrets

    async def get_database_config(self) -> Dict[str, str]:
        """Get database connection configuration."""
        return {
            "neo4j_uri": await self.infisical.get_secret("NEO4J_URI", path="/database"),
            "neo4j_user": await self.infisical.get_secret("NEO4J_USER", path="/database"),
            "neo4j_password": await self.infisical.get_secret("NEO4J_PASSWORD", path="/database"),
            "qdrant_url": await self.infisical.get_secret("QDRANT_URL", path="/database"),
            "qdrant_api_key": await self.infisical.get_secret("QDRANT_API_KEY", path="/database"),
        }

    async def get_api_keys(self) -> Dict[str, str]:
        """Get API keys."""
        return {
            "voyage_api_key": await self.infisical.get_secret("VOYAGE_API_KEY", path="/api-keys"),
            "openai_api_key": await self.infisical.get_secret("OPENAI_API_KEY", path="/api-keys"),
        }

    async def get_observability_config(self) -> Dict[str, str]:
        """Get observability configuration."""
        return {
            "langfuse_public_key": await self.infisical.get_secret("LANGFUSE_PUBLIC_KEY", path="/observability"),
            "langfuse_secret_key": await self.infisical.get_secret("LANGFUSE_SECRET_KEY", path="/observability"),
            "newrelic_license_key": await self.infisical.get_secret("NEW_RELIC_LICENSE_KEY", path="/observability"),
        }
```

## 3.3 Secret Rotation Automation

```python
# src/rkg_mcp/secrets/rotation.py

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Awaitable

class SecretRotationManager:
    """
    Automated secret rotation (Pro tier feature).
    """

    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
        self._rotation_schedule: Dict[str, timedelta] = {}
        self._rotation_handlers: Dict[str, Callable[[str, str], Awaitable[bool]]] = {}
        self._last_rotation: Dict[str, datetime] = {}
        self._task: asyncio.Task | None = None

    def register_rotation(
        self,
        secret_key: str,
        interval: timedelta,
        rotation_handler: Callable[[str, str], Awaitable[bool]]
    ):
        """
        Register a secret for automatic rotation.

        Args:
            secret_key: The secret to rotate
            interval: How often to rotate
            rotation_handler: Async function that generates new secret value
                             and updates dependent systems
        """
        self._rotation_schedule[secret_key] = interval
        self._rotation_handlers[secret_key] = rotation_handler
        self._last_rotation[secret_key] = datetime.now()

    async def start(self):
        """Start rotation monitoring."""
        self._task = asyncio.create_task(self._rotation_loop())

    async def stop(self):
        """Stop rotation monitoring."""
        if self._task:
            self._task.cancel()

    async def _rotation_loop(self):
        """Check for secrets needing rotation."""
        while True:
            for key, interval in self._rotation_schedule.items():
                last = self._last_rotation.get(key, datetime.min)
                if datetime.now() - last >= interval:
                    await self._rotate_secret(key)

            await asyncio.sleep(60)  # Check every minute

    async def _rotate_secret(self, key: str):
        """Rotate a single secret."""
        handler = self._rotation_handlers.get(key)
        if not handler:
            return

        logger.info(f"Rotating secret: {key}")

        try:
            # Get current value
            current = await self.secret_manager.get(key)

            # Generate new value via handler
            new_value = await handler(key, current)

            if new_value:
                # Update in Infisical
                success = await self.secret_manager.infisical.rotate_secret(key, new_value)

                if success:
                    self._last_rotation[key] = datetime.now()
                    logger.info(f"Secret '{key}' rotated successfully")
                else:
                    logger.error(f"Failed to rotate secret '{key}'")

        except Exception as e:
            logger.error(f"Error rotating secret '{key}': {e}")


# Example rotation handler for database passwords
async def rotate_database_password(key: str, current: str) -> str:
    """Generate and apply new database password."""
    import secrets

    # Generate new password
    new_password = secrets.token_urlsafe(32)

    # Update database user password
    # This would use the database admin connection
    # await update_neo4j_password(current, new_password)

    return new_password
```

---

# 4. Unified Observability Facade

## 4.1 Combined Observability Interface

```python
# src/rkg_mcp/observability/facade.py

from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

class ObservabilityFacade:
    """
    Unified interface for all observability integrations.
    Routes to appropriate backend based on feature flags.
    """

    def __init__(
        self,
        langfuse: Optional[LangFuseObservability] = None,
        newrelic: bool = False
    ):
        self.langfuse = langfuse
        self.newrelic_enabled = newrelic

    @asynccontextmanager
    async def trace_operation(
        self,
        name: str,
        operation_type: str,  # "tool" | "search" | "embedding" | "storage"
        input_data: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Unified tracing that routes to appropriate backends.
        """
        # Start traces in all enabled backends
        traces = []

        if self.langfuse:
            if operation_type == "tool":
                ctx = self.langfuse.trace_mcp_tool(name, input_data or {}, **kwargs)
            elif operation_type == "search":
                ctx = self.langfuse.trace_search(name, input_data.get("query", ""), input_data.get("filters"))
            elif operation_type == "embedding":
                ctx = self.langfuse.trace_embedding(
                    input_data.get("provider", "unknown"),
                    input_data.get("model", "unknown"),
                    input_data.get("input_type", "document"),
                    input_data.get("batch_size", 1)
                )
            else:
                ctx = self.langfuse.trace_mcp_tool(name, input_data or {})
            traces.append(("langfuse", ctx))

        # NewRelic is handled via decorators/middleware
        # but we can add custom attributes
        if self.newrelic_enabled:
            NewRelicAgent.add_custom_attribute("rkg.operation", name)
            NewRelicAgent.add_custom_attribute("rkg.operation_type", operation_type)

        # Execute all context managers
        wrapper = _MultiTraceWrapper()

        async with AsyncExitStack() as stack:
            for backend, ctx in traces:
                trace = await stack.enter_async_context(ctx)
                wrapper.add_trace(backend, trace)

            yield wrapper

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Record a metric to all backends."""
        if self.langfuse:
            self.langfuse.log_event(
                f"metric:{name}",
                metadata={"value": value, **(tags or {})}
            )

        if self.newrelic_enabled:
            NewRelicAgent.record_custom_metric(f"Custom/RKG/{name}", value)

    def record_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ):
        """Record an error to all backends."""
        if self.langfuse:
            self.langfuse.log_event(
                "error",
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    **(context or {})
                },
                level="ERROR"
            )

        if self.newrelic_enabled:
            NewRelicAgent.notice_error(error)


class _MultiTraceWrapper:
    """Wrapper that forwards to multiple trace backends."""

    def __init__(self):
        self._traces: Dict[str, Any] = {}

    def add_trace(self, backend: str, trace: Any):
        self._traces[backend] = trace

    def set_output(self, output: Any):
        for trace in self._traces.values():
            if hasattr(trace, 'set_output'):
                trace.set_output(output)

    def add_metadata(self, key: str, value: Any):
        for trace in self._traces.values():
            if hasattr(trace, 'add_metadata'):
                trace.add_metadata(key, value)

    def set_usage(self, **kwargs):
        for trace in self._traces.values():
            if hasattr(trace, 'set_usage'):
                trace.set_usage(**kwargs)
```

## 4.2 Application Startup Integration

```python
# src/rkg_mcp/server.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from .observability.facade import ObservabilityFacade
from .observability.langfuse_patterns import LangFuseObservability
from .observability.newrelic_agent import NewRelicAgent
from .secrets.patterns import SecretManager
from .config import get_settings, load_secrets_from_infisical

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with full observability setup."""

    # 1. Initialize NewRelic first (needs to happen before imports)
    settings = get_settings()
    if settings.newrelic_enabled:
        NewRelicAgent.initialize()

    # 2. Initialize Infisical and load secrets
    infisical = InfisicalManager(
        client_id=settings.infisical_client_id,
        client_secret=settings.infisical_client_secret,
        project_id=settings.infisical_project_id,
        tier=InfisicalTier(settings.infisical_tier)
    )
    await infisical.initialize()

    # 3. Load secrets into settings
    settings = await load_secrets_from_infisical(settings)

    # 4. Initialize LangFuse
    langfuse = LangFuseObservability()
    await langfuse.initialize(settings)

    # 5. Create unified observability facade
    observability = ObservabilityFacade(
        langfuse=langfuse,
        newrelic=settings.newrelic_enabled
    )

    # 6. Initialize secret manager with auto-refresh
    secret_manager = SecretManager(infisical)
    await secret_manager.start_auto_refresh()

    # 7. Initialize storage and other components
    # ... (storage initialization)

    # Store in app state
    app.state.observability = observability
    app.state.secrets = secret_manager
    app.state.settings = settings

    logger.info("Application initialized successfully")

    yield

    # Cleanup
    await secret_manager.stop_auto_refresh()
    await langfuse.flush()

    logger.info("Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="RKG MCP Server",
    lifespan=lifespan
)

# Add NewRelic middleware
if get_settings().newrelic_enabled:
    from .observability.middleware import NewRelicMiddleware
    app.add_middleware(NewRelicMiddleware)
```

---

# 5. Testing Integration

## 5.1 Mock Implementations

```python
# tests/mocks/observability.py

class MockLangFuse:
    """Mock LangFuse for testing."""

    def __init__(self):
        self.traces = []
        self.events = []
        self.scores = []
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    def trace(self, **kwargs):
        trace = MockTrace(**kwargs)
        self.traces.append(trace)
        return trace

    def event(self, **kwargs):
        self.events.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        pass


class MockTrace:
    def __init__(self, **kwargs):
        self.data = kwargs
        self.spans = []
        self.output = None

    def span(self, **kwargs):
        span = MockSpan(**kwargs)
        self.spans.append(span)
        return span

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockInfisical:
    """Mock Infisical for testing."""

    def __init__(self, secrets: Dict[str, str] = None):
        self._secrets = secrets or {}
        self.tier = InfisicalTier.FREE

    async def get_secret(self, key: str, **kwargs) -> str | None:
        return self._secrets.get(key)

    async def get_secrets_by_path(self, path: str) -> Dict[str, str]:
        return {k: v for k, v in self._secrets.items()}

    async def rotate_secret(self, key: str, new_value: str) -> bool:
        self._secrets[key] = new_value
        return True

    def clear_cache(self):
        pass


# tests/conftest.py

import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_langfuse():
    return MockLangFuse()

@pytest.fixture
def mock_infisical():
    return MockInfisical({
        "VOYAGE_API_KEY": "test-voyage-key",  # pragma: allowlist secret
        "NEO4J_PASSWORD": "test-password",  # pragma: allowlist secret
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",  # pragma: allowlist secret
        "LANGFUSE_SECRET_KEY": "sk-lf-test",  # pragma: allowlist secret
    })

@pytest.fixture
def mock_newrelic():
    with patch('newrelic.agent.initialize'):
        with patch('newrelic.agent.record_custom_metric'):
            with patch('newrelic.agent.add_custom_attribute'):
                yield
```

## 5.2 Integration Tests

```python
# tests/integration/test_observability.py

import pytest
from rkg_mcp.observability.facade import ObservabilityFacade

@pytest.mark.asyncio
async def test_trace_operation_with_langfuse(mock_langfuse):
    """Test that operations are traced to LangFuse."""
    facade = ObservabilityFacade(langfuse=mock_langfuse)

    async with facade.trace_operation(
        "test_search",
        "search",
        {"query": "test query", "filters": {}}
    ) as trace:
        trace.set_output({"results": []})
        trace.add_metadata("result_count", 0)

    assert len(mock_langfuse.traces) == 1

@pytest.mark.asyncio
async def test_secret_loading_from_infisical(mock_infisical):
    """Test that secrets are loaded correctly."""
    manager = SecretManager(mock_infisical)

    value = await manager.get("VOYAGE_API_KEY")
    assert value == "test-voyage-key"

    value = await manager.get("MISSING_KEY", "default")
    assert value == "default"

@pytest.mark.asyncio
async def test_pro_feature_guard(mock_infisical):
    """Test that Pro features are guarded."""
    mock_infisical.tier = InfisicalTier.FREE
    manager = SecretManager(mock_infisical)

    with pytest.raises(FeatureNotAvailableError):
        await manager.rollback_secret("key", 1)
```

---

# Summary

This implementation patterns document provides:

1. **LangFuse Patterns**
   - Context managers for tracing MCP tools, searches, embeddings
   - Evaluation and scoring integration
   - Prompt management with fallbacks

2. **NewRelic Patterns**
   - Custom metrics for all operations
   - Distributed tracing helpers
   - NRQL dashboard queries
   - Middleware integration

3. **Infisical Patterns**
   - Multi-environment secret management
   - Auto-refresh with caching
   - Pro tier rotation automation
   - Graceful fallback to env vars

4. **Unified Facade**
   - Single interface for all observability
   - Automatic routing to enabled backends
   - Consistent error handling

5. **Testing Support**
   - Mock implementations for all integrations
   - Integration test examples
   - Feature flag testing
