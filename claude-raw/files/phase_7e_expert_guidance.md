# Phase-7E GraphRAG v2.1 Expert Coder Guidance

## Introduction

This document provides expert-level guidance for implementing the GraphRAG v2.1 system, offering deep insights into best practices, performance optimizations, and common pitfalls to avoid. As a senior sales engineer with technical expertise, you'll appreciate that this guidance focuses on production-ready patterns that ensure scalability, reliability, and maintainability.

## Core Technology Expertise

### Neo4j Graph Database Best Practices

#### Connection Management

The most critical aspect of Neo4j integration involves proper connection pooling and session management. When working with Neo4j, always use the driver's built-in connection pool rather than creating new connections for each operation. The driver maintains a pool of connections that can be reused, significantly reducing overhead.

```python
# CORRECT: Single driver instance with connection pooling
class Neo4jConnection:
    _driver = None
    
    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                os.getenv('NEO4J_URI'),
                auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')),
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
                connection_timeout=30,
                max_retry_time=30
            )
        return cls._driver
    
    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

# INCORRECT: Creating new driver for each operation
def bad_query():
    driver = GraphDatabase.driver(...)  # This creates overhead
    result = driver.session().run(query)
    driver.close()  # Connection thrashing
```

#### Transaction Patterns

Neo4j transactions should follow the Unit of Work pattern. Group related operations into single transactions to ensure consistency and improve performance. Use explicit transactions for write operations and auto-commit transactions for simple reads.

```python
# Write operations with explicit transaction
def create_chunks_transactional(chunks):
    driver = Neo4jConnection.get_driver()
    
    with driver.session() as session:
        def create_work(tx):
            # All operations in single transaction
            for chunk in chunks:
                tx.run(
                    "MERGE (c:Chunk {id: $id}) "
                    "SET c += $properties",
                    id=chunk['id'],
                    properties=chunk
                )
            
            # Create relationships in same transaction
            tx.run(
                "MATCH (c1:Chunk), (c2:Chunk) "
                "WHERE c1.order = c2.order - 1 "
                "AND c1.parent_section_id = c2.parent_section_id "
                "MERGE (c1)-[:NEXT_CHUNK]->(c2)"
            )
        
        session.execute_write(create_work)
```

#### Query Optimization

Understanding Neo4j's query planner is essential for performance. Always use parameters instead of string concatenation to enable query plan caching. Profile queries using `EXPLAIN` and `PROFILE` to understand execution plans.

```cypher
// GOOD: Parameterized query with index hints
MATCH (c:Chunk {document_id: $doc_id})
USING INDEX c:Chunk(document_id)
WHERE c.token_count > $min_tokens
RETURN c
ORDER BY c.order
LIMIT $limit

// BAD: String concatenation prevents plan caching
MATCH (c:Chunk {document_id: '" + doc_id + "'})
WHERE c.token_count > " + str(min_tokens) + "
RETURN c
```

#### Index Strategy

Create indexes based on query patterns, not data model. Composite indexes are powerful but must match query predicates exactly. Monitor index usage with `SHOW INDEX` and `db.indexes()`.

```cypher
// Composite index for common query pattern
CREATE INDEX chunk_retrieval_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id, c.parent_section_id, c.order);

// Monitor index usage
CALL db.index.fulltext.queryNodes('chunk_text_idx', 'search term')
YIELD node, score
RETURN node, score
ORDER BY score DESC;
```

### Qdrant Vector Database Optimization

#### Collection Configuration

Qdrant's performance depends heavily on proper collection configuration. The HNSW (Hierarchical Navigable Small World) index parameters significantly impact search speed and accuracy.

```python
def create_optimized_collection(client, collection_name):
    """Create Qdrant collection with optimized settings"""
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(
                size=1024,
                distance=Distance.COSINE,
                on_disk=False  # Keep in RAM for speed
            )
        },
        # HNSW parameters - critical for performance
        hnsw_config=HnswConfigDiff(
            m=16,  # Number of connections per node (higher = better quality, more memory)
            ef_construct=200,  # Construction time quality (higher = better index)
            full_scan_threshold=10000  # Switch to exact search below this
        ),
        # Optimizer settings
        optimizers_config=OptimizersConfigDiff(
            memmap_threshold=50000,  # Use memory mapping above this
            indexing_threshold=20000,  # Start indexing after this many vectors
            flush_interval_sec=5  # Persist to disk interval
        ),
        # Write-ahead log for durability
        wal_config=WalConfigDiff(
            wal_capacity_mb=32,
            wal_segments_ahead=0
        )
    )
```

#### Batch Operations

Always use batch operations for Qdrant. Single-point operations have significant overhead due to network latency and index updates.

```python
def upsert_embeddings_batch(client, collection_name, chunks, embeddings):
    """Efficiently upsert embeddings in batches"""
    
    batch_size = 100  # Optimal batch size for most scenarios
    points = []
    
    for chunk, embedding in zip(chunks, embeddings):
        point = {
            "id": stable_uuid(chunk['id']),
            "vector": {"content": embedding},
            "payload": {
                "id": chunk['id'],
                "document_id": chunk['document_id'],
                "text": chunk['text'],
                # Index-optimized fields
                "document_id_keyword": chunk['document_id'],  # For exact matching
                "token_count_int": chunk['token_count'],  # For range queries
                "updated_at_timestamp": int(datetime.utcnow().timestamp())
            }
        }
        points.append(point)
        
        # Upsert when batch is full
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=False  # Don't wait for indexing
            )
            points = []
    
    # Upsert remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Wait on last batch
        )
```

#### Search Optimization

Vector search performance depends on the `ef` parameter at search time. Balance between speed and accuracy based on your requirements.

```python
def optimized_vector_search(client, collection_name, query_vector, top_k=10):
    """Perform optimized vector search"""
    
    # Search with tuned parameters
    results = client.search(
        collection_name=collection_name,
        query_vector=("content", query_vector),
        limit=top_k * 2,  # Oversample for filtering
        search_params={
            "ef": 128  # Search-time quality (higher = better but slower)
        },
        # Filter optimization
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_id_keyword",
                    match=MatchValue(value=doc_id)
                )
            ]
        ) if doc_id else None,
        with_payload=True,  # Include full payload
        score_threshold=0.7  # Minimum similarity threshold
    )
    
    return results
```

### Python Performance Patterns

#### Memory Management

Python's garbage collector can cause latency spikes. For large-scale processing, manage memory explicitly and use generators to avoid loading everything into memory.

```python
def process_documents_memory_efficient(file_paths):
    """Process documents with minimal memory footprint"""
    
    def document_generator():
        """Generator to yield one document at a time"""
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                yield content
            # File closed and memory released after each iteration
    
    # Process using generator
    for content in document_generator():
        # Process one document at a time
        sections = parse_document(content)
        chunks = create_chunks(sections)
        
        # Process in batches to control memory
        for batch in batch_iterator(chunks, batch_size=100):
            embeddings = generate_embeddings(batch)
            store_batch(batch, embeddings)
            
            # Explicit garbage collection after batch
            import gc
            gc.collect()

def batch_iterator(items, batch_size):
    """Create batches from iterable"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

#### Async/Await for I/O Operations

Use asyncio for I/O-bound operations to maximize throughput. This is especially important when calling external APIs or databases.

```python
import asyncio
import httpx
from typing import List

class AsyncEmbeddingService:
    """Asynchronous embedding service for high throughput"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20)
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate single embedding asynchronously"""
        
        response = await self.client.post(
            os.getenv('JINA_BASE_URL'),
            json={"input": text, "model": "jina-embeddings-v3"},
            headers={"Authorization": f"Bearer {os.getenv('JINA_API_KEY')}"}
        )
        
        return response.json()['data'][0]['embedding']
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts concurrently"""
        
        # Create tasks for concurrent execution
        tasks = [self.generate_embedding(text) for text in texts]
        
        # Execute concurrently with semaphore to limit parallelism
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        embeddings = await asyncio.gather(*bounded_tasks)
        
        return embeddings
    
    async def close(self):
        """Clean up client"""
        await self.client.aclose()
```

#### Caching Strategy

Implement multi-level caching to reduce computation and API calls. Use Redis for distributed cache and local LRU cache for frequently accessed data.

```python
from functools import lru_cache
import redis
import pickle
import hashlib

class MultiLevelCache:
    """Multi-level caching system"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            db=int(os.getenv('REDIS_DB')),
            decode_responses=False  # Binary data for pickle
        )
        
        # Local LRU cache
        self.local_cache_size = 1000
    
    @lru_cache(maxsize=1000)
    def _local_get(self, key: str):
        """Local LRU cache layer"""
        return None  # Placeholder for LRU decorator
    
    def generate_cache_key(self, prefix: str, content: str) -> str:
        """Generate stable cache key"""
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{prefix}:{content_hash}"
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        
        cache_key = self.generate_cache_key("embedding", text)
        
        # Try local cache first
        local_result = self._local_get(cache_key)
        if local_result:
            return local_result
        
        # Try Redis cache
        redis_result = self.redis_client.get(cache_key)
        if redis_result:
            embedding = pickle.loads(redis_result)
            # Update local cache
            self._local_get.__wrapped__(self, cache_key, embedding)
            return embedding
        
        return None
    
    def set_embedding(self, text: str, embedding: List[float], ttl: int = 3600):
        """Store embedding in cache"""
        
        cache_key = self.generate_cache_key("embedding", text)
        
        # Store in Redis
        self.redis_client.setex(
            cache_key,
            ttl,
            pickle.dumps(embedding)
        )
        
        # Update local cache
        self._local_get.__wrapped__(self, cache_key, embedding)
```

### Error Handling and Resilience

#### Circuit Breaker Pattern

Implement circuit breakers to prevent cascading failures when external services are down.

```python
from enum import Enum
from datetime import datetime, timedelta
import threading

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to retry"""
        
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
```

#### Retry Strategy with Exponential Backoff

Implement intelligent retry logic that adapts to service conditions.

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

class RetryableOperation:
    """Wrapper for operations that need retry logic"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def call_external_api(self, url: str, payload: dict):
        """Call external API with automatic retry"""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=30.0
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                raise httpx.HTTPStatusError("Rate limited", request=None, response=response)
            
            response.raise_for_status()
            return response.json()
```

### Database Migration and Versioning

#### Schema Evolution Strategy

Implement a robust schema versioning system that allows for safe migrations without downtime.

```python
class SchemaManager:
    """Manage schema versions and migrations"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.migrations = {
            'v2.0': self._migrate_v20_to_v21,
            'v2.1': self._verify_v21
        }
    
    def get_current_version(self) -> str:
        """Get current schema version"""
        
        with self.driver.session() as session:
            result = session.run(
                "MATCH (sv:SchemaVersion {id: 'singleton'}) "
                "RETURN sv.version as version"
            ).single()
            
            return result['version'] if result else 'v1.0'
    
    def migrate_to_latest(self):
        """Migrate schema to latest version"""
        
        current = self.get_current_version()
        target = 'v2.1'
        
        if current == target:
            logger.info(f"Schema already at {target}")
            return
        
        # Execute migrations in order
        versions = ['v2.0', 'v2.1']
        start_idx = versions.index(current) + 1 if current in versions else 0
        
        for version in versions[start_idx:]:
            logger.info(f"Migrating to {version}")
            self.migrations[version]()
            self._update_version(version)
    
    def _migrate_v20_to_v21(self):
        """Migrate from v2.0 to v2.1"""
        
        with self.driver.session() as session:
            # Add dual labeling
            session.run("MATCH (s:Section) WHERE NOT s:Chunk SET s:Chunk")
            
            # Add new indexes
            session.run(
                "CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS "
                "FOR (c:Chunk) ON c.vector_embedding "
                "OPTIONS {indexConfig: {`vector.dimensions`: 1024, "
                "`vector.similarity_function`: 'cosine'}}"
            )
    
    def _verify_v21(self):
        """Verify v2.1 schema integrity"""
        
        with self.driver.session() as session:
            # Check dual labeling
            result = session.run(
                "MATCH (s:Section) WHERE NOT s:Chunk RETURN count(s) as count"
            ).single()
            
            if result['count'] > 0:
                raise Exception(f"Found {result['count']} sections without Chunk label")
```

### Monitoring and Observability

#### Metrics Collection

Implement comprehensive metrics collection for production monitoring.

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.ingestion_counter = Counter(
            'graphrag_documents_ingested_total',
            'Total documents ingested',
            ['status'],
            registry=self.registry
        )
        
        self.embedding_histogram = Histogram(
            'graphrag_embedding_duration_seconds',
            'Embedding generation duration',
            ['provider'],
            registry=self.registry
        )
        
        self.chunk_gauge = Gauge(
            'graphrag_chunks_total',
            'Total number of chunks',
            ['document_id'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Counter(
            'graphrag_cache_hits_total',
            'Cache hit/miss counts',
            ['cache_level', 'hit'],
            registry=self.registry
        )
    
    def record_ingestion(self, success: bool):
        """Record document ingestion"""
        
        status = 'success' if success else 'failure'
        self.ingestion_counter.labels(status=status).inc()
    
    def record_embedding_time(self, provider: str, duration: float):
        """Record embedding generation time"""
        
        self.embedding_histogram.labels(provider=provider).observe(duration)
    
    def update_chunk_count(self, document_id: str, count: int):
        """Update chunk count for document"""
        
        self.chunk_gauge.labels(document_id=document_id).set(count)
    
    def record_cache_access(self, level: str, hit: bool):
        """Record cache hit/miss"""
        
        self.cache_hit_rate.labels(
            cache_level=level,
            hit='hit' if hit else 'miss'
        ).inc()
```

#### Structured Logging

Use structured logging for better debugging and analysis.

```python
import structlog
from structlog.processors import JSONRenderer, TimeStamper

def configure_logging():
    """Configure structured logging"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()

# Usage
logger = configure_logging()

def process_document_with_logging(document):
    """Process document with structured logging"""
    
    log = logger.bind(
        document_id=document.id,
        document_uri=document.uri,
        operation='ingestion'
    )
    
    log.info("Starting document processing")
    
    try:
        sections = parse_document(document)
        log.info("Document parsed", section_count=len(sections))
        
        chunks = create_chunks(sections)
        log.info("Chunks created", chunk_count=len(chunks))
        
        embeddings = generate_embeddings(chunks)
        log.info("Embeddings generated")
        
        store_results(chunks, embeddings)
        log.info("Processing complete")
        
    except Exception as e:
        log.error("Processing failed", error=str(e), exc_info=True)
        raise
```

### Security Best Practices

#### Input Validation and Sanitization

Always validate and sanitize inputs to prevent injection attacks.

```python
import re
from typing import Optional

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_document_id(doc_id: str) -> bool:
        """Validate document ID format"""
        
        # Document IDs should be 24-char hex strings
        pattern = r'^[a-f0-9]{24}$'
        return bool(re.match(pattern, doc_id))
    
    @staticmethod
    def sanitize_cypher_parameter(value: str) -> str:
        """Sanitize parameter for Cypher queries"""
        
        # Remove any Cypher special characters
        sanitized = re.sub(r'[`\'"\\\n\r\t]', '', value)
        
        # Limit length
        return sanitized[:1000]
    
    @staticmethod
    def validate_embedding_dimensions(embedding: List[float]) -> bool:
        """Validate embedding dimensions and values"""
        
        if len(embedding) != 1024:
            return False
        
        # Check for valid float values
        for val in embedding:
            if not isinstance(val, float) or not (-1 <= val <= 1):
                return False
        
        return True
```

#### API Key Management

Implement secure API key rotation and management.

```python
import os
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class APIKeyManager:
    """Secure API key management"""
    
    def __init__(self):
        # Use environment variable for encryption key
        self.cipher = Fernet(os.getenv('ENCRYPTION_KEY').encode())
        self.keys = {}
        self.rotation_interval = timedelta(days=30)
    
    def store_api_key(self, service: str, key: str):
        """Store encrypted API key"""
        
        encrypted_key = self.cipher.encrypt(key.encode())
        
        self.keys[service] = {
            'key': encrypted_key,
            'created_at': datetime.now(),
            'last_rotated': datetime.now()
        }
    
    def get_api_key(self, service: str) -> str:
        """Retrieve and decrypt API key"""
        
        if service not in self.keys:
            raise ValueError(f"No key found for {service}")
        
        # Check if rotation needed
        key_data = self.keys[service]
        if datetime.now() - key_data['last_rotated'] > self.rotation_interval:
            logger.warning(f"API key for {service} needs rotation")
        
        return self.cipher.decrypt(key_data['key']).decode()
    
    def rotate_key(self, service: str, new_key: str):
        """Rotate API key"""
        
        old_key = self.get_api_key(service)
        
        # Store new key
        self.store_api_key(service, new_key)
        self.keys[service]['last_rotated'] = datetime.now()
        
        # Log rotation
        logger.info(f"API key rotated for {service}")
        
        return old_key  # Return old key for graceful transition
```

### Performance Testing and Optimization

#### Load Testing Framework

Implement comprehensive load testing to identify bottlenecks.

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

class LoadTester:
    """Load testing framework for GraphRAG"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=50)
        self.results = []
    
    async def run_load_test(
        self,
        operation,
        num_requests: int,
        concurrent_requests: int
    ):
        """Run load test on operation"""
        
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_operation():
            async with semaphore:
                start = time.time()
                try:
                    await operation()
                    duration = time.time() - start
                    return {'success': True, 'duration': duration}
                except Exception as e:
                    duration = time.time() - start
                    return {'success': False, 'duration': duration, 'error': str(e)}
        
        # Run operations concurrently
        tasks = [bounded_operation() for _ in range(num_requests)]
        self.results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate performance statistics"""
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        if successful:
            durations = [r['duration'] for r in successful]
            
            stats = {
                'total_requests': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'mean_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'p95_duration': statistics.quantiles(durations, n=20)[18],
                'p99_duration': statistics.quantiles(durations, n=100)[98]
            }
            
            logger.info("Load test results", **stats)
            return stats
```

## Common Pitfalls and Solutions

### Pitfall 1: Memory Leaks in Long-Running Processes

Python's reference counting can miss circular references. Always use context managers and explicitly close resources.

```python
# PROBLEM: Circular reference causes memory leak
class Document:
    def __init__(self):
        self.sections = []
        
class Section:
    def __init__(self, document):
        self.document = document  # Circular reference
        document.sections.append(self)

# SOLUTION: Use weak references
import weakref

class Section:
    def __init__(self, document):
        self.document = weakref.ref(document)  # Weak reference
        document.sections.append(self)
```

### Pitfall 2: N+1 Query Problem

Fetching related data in loops causes exponential database queries.

```python
# PROBLEM: N+1 queries
def get_chunks_with_documents():
    chunks = neo4j_session.run("MATCH (c:Chunk) RETURN c").values()
    
    for chunk in chunks:
        # This causes a query for each chunk!
        doc = neo4j_session.run(
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk {id: $id}) RETURN d",
            id=chunk['id']
        ).single()
        chunk['document'] = doc

# SOLUTION: Single query with relationships
def get_chunks_with_documents():
    results = neo4j_session.run(
        "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
        "RETURN c, d"
    ).values()
    
    # Process results efficiently
    return [{'chunk': record[0], 'document': record[1]} for record in results]
```

### Pitfall 3: Blocking I/O in Async Context

Mixing synchronous and asynchronous code incorrectly blocks the event loop.

```python
# PROBLEM: Blocking call in async function
async def process_document(document):
    sections = parse_document(document)  # Synchronous parsing
    embeddings = requests.post(...)  # Blocking HTTP call
    return embeddings

# SOLUTION: Use async throughout or run in executor
async def process_document(document):
    loop = asyncio.get_event_loop()
    
    # Run CPU-bound operation in thread pool
    sections = await loop.run_in_executor(None, parse_document, document)
    
    # Use async HTTP client
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
        embeddings = response.json()
    
    return embeddings
```

## Advanced Optimization Techniques

### Vector Index Optimization

Understanding how vector indexes work enables significant performance improvements.

```python
def optimize_vector_index(client, collection_name):
    """Optimize vector index for production workload"""
    
    # Get collection info
    info = client.get_collection(collection_name)
    num_vectors = info.points_count
    
    # Adjust HNSW parameters based on collection size
    if num_vectors < 10000:
        # Small collection - prioritize accuracy
        hnsw_config = HnswConfigDiff(m=32, ef_construct=400)
    elif num_vectors < 100000:
        # Medium collection - balanced
        hnsw_config = HnswConfigDiff(m=16, ef_construct=200)
    else:
        # Large collection - prioritize speed
        hnsw_config = HnswConfigDiff(m=8, ef_construct=100)
    
    # Update configuration
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=hnsw_config
    )
```

### Query Plan Analysis

Analyze and optimize query execution plans for complex graph queries.

```cypher
// Use PROFILE to understand query execution
PROFILE
MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
WHERE c.token_count > 500
WITH c
ORDER BY c.order
LIMIT 10
MATCH (c)-[:NEXT_CHUNK*0..2]->(next:Chunk)
RETURN c, collect(next) as context

// Optimize with index hints and pattern comprehension
PROFILE
MATCH (c:Chunk {document_id: $doc_id})
USING INDEX c:Chunk(document_id)
WHERE c.token_count > 500
WITH c
ORDER BY c.order
LIMIT 10
RETURN c, 
       [(c)-[:NEXT_CHUNK*0..2]->(next:Chunk) | next] as context
```

## Conclusion

This expert guidance provides the deep technical knowledge needed to build a production-grade GraphRAG system. The patterns and practices outlined here come from real-world experience building high-performance, scalable systems. Remember that optimization is an iterative process - measure first, optimize based on data, and always maintain code clarity and maintainability alongside performance improvements.

The key to success lies in understanding not just what to do, but why certain approaches work better than others. This understanding enables you to adapt these patterns to your specific use cases and requirements, ensuring your GraphRAG implementation delivers exceptional performance and reliability in production.
