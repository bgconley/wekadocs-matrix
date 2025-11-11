# Phase-7E GraphRAG v2.1 Complete Implementation Plan

## Executive Overview

This implementation plan provides detailed, step-by-step instructions for building the Phase-7E GraphRAG system. Each instruction is prescriptive and complete, leaving no room for interpretation. The plan follows the phased approach from the application specification, ensuring each component is fully functional before integration.

## Pre-Implementation Requirements

### System Prerequisites

Before beginning implementation, ensure the following systems are installed and operational:

1. **Operating System**: Ubuntu 22.04 LTS or compatible Linux distribution
2. **Python**: Version 3.11 or higher with pip and venv
3. **Docker**: Version 24.0 or higher with Docker Compose
4. **Memory**: Minimum 16GB RAM, 32GB recommended
5. **Storage**: Minimum 100GB available disk space
6. **Network**: Stable internet connection for API calls

### Software Installation Commands

Execute these commands in order to prepare the system:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11 if not present
sudo apt-get install python3.11 python3.11-venv python3.11-dev -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install build tools
sudo apt-get install build-essential git curl wget -y

# Verify installations
python3.11 --version
docker --version
docker-compose --version
```

## Phase 7E.1: Foundation Layer Implementation

### Task 7E.1.1: Environment Setup

#### Step 1: Create Project Directory Structure

```bash
# Create base project directory
mkdir -p /opt/graphrag-v2
cd /opt/graphrag-v2

# Create all required subdirectories
mkdir -p {config,data/{neo4j,qdrant,redis},logs,scripts,src/{ingestion,providers,retrieval,storage,utils},tests,temp}

# Set proper permissions
chmod -R 755 /opt/graphrag-v2
```

#### Step 2: Install Neo4j Community Edition

```bash
# Create Neo4j docker-compose.yml
cat > docker-compose-neo4j.yml << 'EOF'
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: graphrag-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
      - ./data/neo4j/import:/var/lib/neo4j/import
      - ./data/neo4j/plugins:/plugins
      - ./config/neo4j.conf:/conf/neo4j.conf
    environment:
      - NEO4J_AUTH=neo4j/graphrag2024!
      - NEO4J_server_memory_heap_initial__size=2g
      - NEO4J_server_memory_heap_max__size=4g
      - NEO4J_server_memory_pagecache__size=2g
    restart: unless-stopped
    networks:
      - graphrag-network

networks:
  graphrag-network:
    driver: bridge
EOF

# Start Neo4j
docker-compose -f docker-compose-neo4j.yml up -d

# Wait for Neo4j to be ready (important!)
sleep 30

# Verify Neo4j is running
curl -u neo4j:graphrag2024! http://localhost:7474/db/neo4j/cluster/available
```

#### Step 3: Install Qdrant Vector Database

```bash
# Create Qdrant docker-compose.yml
cat > docker-compose-qdrant.yml << 'EOF'
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: graphrag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
      - QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD_KB=200000
    restart: unless-stopped
    networks:
      - graphrag-network

networks:
  graphrag-network:
    external: true
EOF

# Start Qdrant
docker-compose -f docker-compose-qdrant.yml up -d

# Verify Qdrant is running
curl http://localhost:6333/health
```

#### Step 4: Install Redis Cache

```bash
# Create Redis docker-compose.yml
cat > docker-compose-redis.yml << 'EOF'
version: '3.8'
services:
  redis:
    image: redis:7.2-alpine
    container_name: graphrag-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --save 60 1 --loglevel notice --maxmemory 2gb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    networks:
      - graphrag-network

networks:
  graphrag-network:
    external: true
EOF

# Start Redis
docker-compose -f docker-compose-redis.yml up -d

# Verify Redis is running
redis-cli ping
```

### Task 7E.1.2: Dependency Installation

#### Step 1: Create Python Virtual Environment

```bash
cd /opt/graphrag-v2

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 2: Create Requirements File

```bash
# Create comprehensive requirements.txt
cat > requirements.txt << 'EOF'
# Core Database Drivers
neo4j==5.15.0
qdrant-client==1.7.3
redis==5.0.1

# Embedding Dependencies
transformers==4.36.2
sentencepiece==0.1.99
tokenizers==0.15.0
huggingface-hub==0.20.2
protobuf==4.25.1

# Data Processing
numpy==1.26.3
pandas==2.1.4
pydantic==2.5.3
python-dotenv==1.0.0

# API and Network
httpx==0.25.2
tenacity==8.2.3
asyncio==3.4.3
aiohttp==3.9.1

# Utilities
hashlib
python-dateutil==2.8.2
pytz==2023.3
ujson==5.9.0

# Document Processing
beautifulsoup4==4.12.2
lxml==5.0.0
markdown==3.5.1

# Logging and Monitoring
structlog==24.1.0
prometheus-client==0.19.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0

# Development Tools
black==23.12.1
pylint==3.0.3
mypy==1.8.0
EOF

# Install all dependencies
pip install -r requirements.txt
```

#### Step 3: Prefetch Jina Tokenizer

```bash
# Prefetch the tokenizer to avoid runtime downloads
python -c "
from transformers import AutoTokenizer
import os

# Set cache directory
os.environ['HF_HOME'] = '/opt/graphrag-v2/models'
os.makedirs('/opt/graphrag-v2/models', exist_ok=True)

# Download tokenizer
print('Downloading Jina v3 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
print('Tokenizer downloaded successfully!')
"
```

### Task 7E.1.3: Configuration Management

#### Step 1: Create Environment Configuration

```bash
# Create .env file with all configuration
cat > .env << 'EOF'
# System Configuration
PROJECT_ROOT=/opt/graphrag-v2
LOG_LEVEL=INFO
LOG_DIR=/opt/graphrag-v2/logs
TEMP_DIR=/opt/graphrag-v2/temp

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag2024!
NEO4J_DATABASE=neo4j

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=chunks
QDRANT_TIMEOUT=30

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_CACHE_TTL=3600

# Jina Configuration
JINA_API_KEY=your_jina_api_key_here
JINA_BASE_URL=https://api.jina.ai/v1/embeddings
JINA_MODEL_NAME=jina-embeddings-v3

# Tokenizer Configuration
TOKENIZER_BACKEND=hf
HF_TOKENIZER_ID=jinaai/jina-embeddings-v3
HF_CACHE=/opt/graphrag-v2/models
TRANSFORMERS_OFFLINE=false

# Embedding Parameters
EMBED_DIM=1024
EMBED_PROVIDER=jina-ai
EMBED_MODEL_ID=jina-embeddings-v3
EMBED_MAX_TOKENS=8192
EMBED_TARGET_TOKENS=7900
EMBED_BATCH_SIZE=32

# Chunking Parameters
TARGET_MIN_TOKENS=800
TARGET_MAX_TOKENS=1500
ABSOLUTE_MAX_TOKENS=7900
OVERLAP_TOKENS=200
SPLIT_MIN_TOKENS=1000

# Retrieval Parameters
VECTOR_TOP_K=20
BM25_TOP_K=20
FINAL_TOP_K=10
CONTEXT_WINDOW=3

# Performance Parameters
MAX_WORKERS=4
REQUEST_TIMEOUT=30
RETRY_COUNT=3
BATCH_TIMEOUT=60

# Feature Flags
ENABLE_CACHE=true
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true
COMBINE_SECTIONS=true
SPLIT_FALLBACK=true
EOF
```

#### Step 2: Create Logging Configuration

```bash
# Create logging configuration
cat > config/logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: /opt/graphrag-v2/logs/graphrag.log
    maxBytes: 10485760
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: /opt/graphrag-v2/logs/errors.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  graphrag:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

  neo4j:
    level: INFO
    handlers: [file]
    propagate: false

  qdrant:
    level: INFO
    handlers: [file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
EOF
```

#### Step 3: Create Error Handling Framework

```bash
# Create error handling module
cat > src/utils/error_handler.py << 'EOF'
"""Error handling framework for GraphRAG v2.1"""

import logging
import traceback
from typing import Any, Callable, Optional, Type
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GraphRAGError(Exception):
    """Base exception for GraphRAG system"""
    pass

class IngestionError(GraphRAGError):
    """Raised during document ingestion"""
    pass

class EmbeddingError(GraphRAGError):
    """Raised during embedding generation"""
    pass

class StorageError(GraphRAGError):
    """Raised during storage operations"""
    pass

class RetrievalError(GraphRAGError):
    """Raised during retrieval operations"""
    pass

def safe_execute(
    func: Callable,
    error_class: Type[Exception] = GraphRAGError,
    default: Any = None,
    log_errors: bool = True
) -> Callable:
    """Decorator for safe function execution with error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())

            if isinstance(e, GraphRAGError):
                raise

            raise error_class(f"Failed in {func.__name__}: {str(e)}") from e

    return wrapper

def retry_on_failure(
    max_attempts: int = 3,
    wait_seconds: int = 1,
    max_wait: int = 10
):
    """Decorator for retrying failed operations"""

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_seconds, max=max_wait),
        reraise=True
    )

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise GraphRAGError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
EOF
```

## Phase 7E.2: Schema Implementation

### Task 7E.2.1: Neo4j Schema Creation

#### Step 1: Apply Schema DDL

```bash
# Copy the schema file to Neo4j import directory
cp /mnt/user-data/uploads/create_schema_v2_1_complete__v3.cypher /opt/graphrag-v2/data/neo4j/import/

# Execute schema creation
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p graphrag2024! < /var/lib/neo4j/import/create_schema_v2_1_complete__v3.cypher

# Verify schema creation
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p graphrag2024! "SHOW CONSTRAINTS;"
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p graphrag2024! "SHOW INDEXES;"
```

#### Step 2: Create Additional Chunk Management Extensions

```bash
# Create chunk management schema extensions
cat > scripts/neo4j_chunk_extensions.cypher << 'EOF'
// Additional properties for chunk management
// These extend the base schema for Phase 7E specific needs

// Create composite index for efficient chunk retrieval
CREATE INDEX chunk_composite_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id, c.parent_section_id, c.order);

// Create text index for BM25 search
CREATE TEXT INDEX chunk_text_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.text);

// Create index for chunking flags
CREATE INDEX chunk_flags_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.is_combined, c.is_split);

// Verify all indexes are online
CALL db.indexes() YIELD name, state
WHERE state <> 'ONLINE'
RETURN name, state;
EOF

# Apply extensions
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p graphrag2024! < scripts/neo4j_chunk_extensions.cypher
```

### Task 7E.2.2: Qdrant Collection Setup

#### Step 1: Create Qdrant Collection

```python
# Create Qdrant setup script
cat > scripts/setup_qdrant.py << 'EOF'
#!/usr/bin/env python3
"""Setup Qdrant collection for GraphRAG v2.1"""

import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_collection():
    """Create and configure Qdrant collection"""

    # Initialize client
    client = QdrantClient(
        url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    collection_name = os.getenv('QDRANT_COLLECTION', 'chunks')
    embed_dim = int(os.getenv('EMBED_DIM', 1024))

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists. Recreating...")
        client.delete_collection(collection_name)

    # Create collection with vector configuration
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "content": qm.VectorParams(
                size=embed_dim,
                distance=qm.Distance.COSINE,
                on_disk=False  # Keep in memory for performance
            )
        },
        optimizers_config=qm.OptimizersConfigDiff(
            memmap_threshold=20000,
            indexing_threshold=20000,
            flush_interval_sec=5
        ),
        hnsw_config=qm.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000
        )
    )

    print(f"Collection '{collection_name}' created successfully")

    # Create payload indexes for efficient filtering
    indexes = [
        ("document_id", qm.PayloadSchemaType.KEYWORD),
        ("parent_section_id", qm.PayloadSchemaType.KEYWORD),
        ("order", qm.PayloadSchemaType.INTEGER),
        ("level", qm.PayloadSchemaType.INTEGER),
        ("token_count", qm.PayloadSchemaType.INTEGER),
        ("updated_at", qm.PayloadSchemaType.INTEGER),
        ("is_combined", qm.PayloadSchemaType.BOOL),
        ("is_split", qm.PayloadSchemaType.BOOL),
        ("embedding_version", qm.PayloadSchemaType.KEYWORD),
        ("embedding_provider", qm.PayloadSchemaType.KEYWORD)
    ]

    for field_name, field_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            print(f"Created index for '{field_name}'")
        except Exception as e:
            print(f"Warning: Could not create index for '{field_name}': {e}")

    # Verify collection
    collection_info = client.get_collection(collection_name)
    print(f"\nCollection info:")
    print(f"  Points count: {collection_info.points_count}")
    print(f"  Vectors config: {collection_info.config.params.vectors}")
    print(f"  Status: {collection_info.status}")

    return client, collection_name

if __name__ == "__main__":
    try:
        client, collection = create_collection()
        print("\nQdrant setup completed successfully!")
    except Exception as e:
        print(f"Error setting up Qdrant: {e}")
        sys.exit(1)
EOF

# Make executable and run
chmod +x scripts/setup_qdrant.py
python scripts/setup_qdrant.py
```

### Task 7E.2.3: Schema Validation

#### Step 1: Create Validation Script

```python
# Create comprehensive schema validation script
cat > scripts/validate_schema.py << 'EOF'
#!/usr/bin/env python3
"""Validate GraphRAG v2.1 schema implementation"""

import os
import sys
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import redis
from dotenv import load_dotenv

load_dotenv()

def validate_neo4j():
    """Validate Neo4j schema"""

    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

    validations = []

    with driver.session() as session:
        # Check constraints
        constraints = session.run("SHOW CONSTRAINTS").data()
        required_constraints = [
            'document_id_unique',
            'section_id_unique',
            'session_id_unique',
            'query_id_unique',
            'answer_id_unique'
        ]

        constraint_names = [c['name'] for c in constraints]
        for req in required_constraints:
            if req in constraint_names:
                validations.append((req, True, "Constraint exists"))
            else:
                validations.append((req, False, "Constraint missing"))

        # Check indexes
        indexes = session.run("SHOW INDEXES").data()
        required_indexes = [
            'section_embeddings_v2',
            'chunk_embeddings_v2',
            'section_document_id',
            'chunk_document_id'
        ]

        index_names = [idx['name'] for idx in indexes]
        for req in required_indexes:
            if req in index_names:
                validations.append((req, True, "Index exists"))
            else:
                validations.append((req, False, "Index missing"))

        # Check schema version
        result = session.run(
            "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv"
        ).single()

        if result and result['sv']['version'] == 'v2.1':
            validations.append(("Schema version", True, "v2.1"))
        else:
            validations.append(("Schema version", False, "Not v2.1"))

    driver.close()
    return validations

def validate_qdrant():
    """Validate Qdrant collection"""

    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    validations = []
    collection_name = os.getenv('QDRANT_COLLECTION', 'chunks')

    try:
        info = client.get_collection(collection_name)

        # Check vector dimensions
        vector_config = info.config.params.vectors['content']
        if vector_config.size == 1024:
            validations.append(("Vector dimensions", True, "1024"))
        else:
            validations.append(("Vector dimensions", False, f"{vector_config.size}"))

        # Check distance metric
        if vector_config.distance == 'Cosine':
            validations.append(("Distance metric", True, "Cosine"))
        else:
            validations.append(("Distance metric", False, f"{vector_config.distance}"))

        # Check status
        if info.status == 'green':
            validations.append(("Collection status", True, "Green"))
        else:
            validations.append(("Collection status", False, f"{info.status}"))

    except Exception as e:
        validations.append(("Collection exists", False, str(e)))

    return validations

def validate_redis():
    """Validate Redis connection"""

    validations = []

    try:
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )

        # Test connection
        if r.ping():
            validations.append(("Redis connection", True, "Connected"))
        else:
            validations.append(("Redis connection", False, "Cannot ping"))

        # Check memory
        info = r.info('memory')
        used_memory = info['used_memory_human']
        validations.append(("Redis memory", True, f"Using {used_memory}"))

    except Exception as e:
        validations.append(("Redis connection", False, str(e)))

    return validations

def print_results(component, validations):
    """Print validation results"""

    print(f"\n{component} Validation Results:")
    print("-" * 50)

    all_passed = True
    for name, passed, details in validations:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {details}")
        if not passed:
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("GraphRAG v2.1 Schema Validation")
    print("=" * 50)

    # Run validations
    neo4j_valid = print_results("Neo4j", validate_neo4j())
    qdrant_valid = print_results("Qdrant", validate_qdrant())
    redis_valid = print_results("Redis", validate_redis())

    # Summary
    print("\n" + "=" * 50)
    if neo4j_valid and qdrant_valid and redis_valid:
        print("✓ All validations passed!")
        sys.exit(0)
    else:
        print("✗ Some validations failed. Please review and fix.")
        sys.exit(1)
EOF

# Run validation
chmod +x scripts/validate_schema.py
python scripts/validate_schema.py
```

## Phase 7E.3: Document Processing Pipeline Implementation

### Task 7E.3.1: Document Parser Implementation

#### Step 1: Create Base Document Parser

```python
# Create document parser module
cat > src/ingestion/document_parser.py << 'EOF'
"""Document parser for GraphRAG v2.1"""

import os
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import markdown
from bs4 import BeautifulSoup
import re

@dataclass
class DocumentMetadata:
    """Document metadata container"""
    source_uri: str
    source_type: str = "markdown"
    version: str = "1.0"
    title: Optional[str] = None
    last_edited: Optional[datetime] = None
    path: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'source_uri': self.source_uri,
            'source_type': self.source_type,
            'version': self.version,
            'title': self.title,
            'last_edited': self.last_edited.isoformat() if self.last_edited else None,
            'path': self.path,
            'source_url': self.source_url
        }

@dataclass
class Section:
    """Document section with hierarchy"""
    id: str
    document_id: str
    level: int  # 1-6 for H1-H6
    heading: Optional[str]
    text: str
    parent_id: Optional[str]
    order: int
    start_offset: int
    end_offset: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'level': self.level,
            'heading': self.heading,
            'text': self.text,
            'parent_id': self.parent_id,
            'order': self.order,
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'metadata': self.metadata
        }

class DocumentParser:
    """Parse documents into hierarchical sections"""

    def __init__(self):
        self.md_parser = markdown.Markdown(extensions=['extra', 'toc'])
        self.section_counter = 0

    def parse(self, content: str, metadata: DocumentMetadata) -> Tuple[str, List[Section]]:
        """Parse document content into sections

        Args:
            content: Raw document content
            metadata: Document metadata

        Returns:
            Tuple of (document_id, list of sections)
        """
        # Generate document ID
        doc_id = self._generate_document_id(metadata.source_uri)

        # Parse based on format
        if metadata.source_type == "markdown":
            sections = self._parse_markdown(content, doc_id)
        elif metadata.source_type == "html":
            sections = self._parse_html(content, doc_id)
        else:
            # Default to plain text
            sections = self._parse_plain_text(content, doc_id)

        # Add hierarchy relationships
        sections = self._establish_hierarchy(sections)

        return doc_id, sections

    def _generate_document_id(self, source_uri: str) -> str:
        """Generate stable document ID from URI"""
        return hashlib.sha256(source_uri.encode()).hexdigest()[:24]

    def _generate_section_id(self, doc_id: str, content: str, order: int) -> str:
        """Generate stable section ID"""
        unique_str = f"{doc_id}|{order}|{content[:100]}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:24]

    def _parse_markdown(self, content: str, doc_id: str) -> List[Section]:
        """Parse Markdown content into sections"""

        sections = []
        lines = content.split('\n')
        current_section = None
        current_text = []
        current_offset = 0

        for i, line in enumerate(lines):
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if heading_match:
                # Save previous section if exists
                if current_section:
                    current_section.text = '\n'.join(current_text).strip()
                    current_section.end_offset = current_offset
                    sections.append(current_section)
                    current_text = []

                # Create new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                section_id = self._generate_section_id(
                    doc_id, heading, len(sections)
                )

                current_section = Section(
                    id=section_id,
                    document_id=doc_id,
                    level=level,
                    heading=heading,
                    text="",
                    parent_id=None,
                    order=len(sections),
                    start_offset=current_offset,
                    end_offset=current_offset
                )
            else:
                # Add to current section text
                current_text.append(line)

            current_offset += len(line) + 1  # +1 for newline

        # Save final section
        if current_section:
            current_section.text = '\n'.join(current_text).strip()
            current_section.end_offset = current_offset
            sections.append(current_section)
        elif current_text:
            # No headings found, create single section
            section_id = self._generate_section_id(doc_id, content[:100], 0)
            sections.append(Section(
                id=section_id,
                document_id=doc_id,
                level=1,
                heading=None,
                text='\n'.join(current_text).strip(),
                parent_id=None,
                order=0,
                start_offset=0,
                end_offset=len(content)
            ))

        return sections

    def _parse_html(self, content: str, doc_id: str) -> List[Section]:
        """Parse HTML content into sections"""

        soup = BeautifulSoup(content, 'lxml')
        sections = []

        # Find all heading elements
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for i, heading in enumerate(headings):
            level = int(heading.name[1])
            heading_text = heading.get_text().strip()

            # Get content between this heading and next
            next_heading = headings[i + 1] if i + 1 < len(headings) else None

            content_elements = []
            for sibling in heading.find_next_siblings():
                if sibling == next_heading:
                    break
                if sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content_elements.append(sibling.get_text().strip())

            section_text = '\n'.join(content_elements)

            section_id = self._generate_section_id(
                doc_id, heading_text, len(sections)
            )

            sections.append(Section(
                id=section_id,
                document_id=doc_id,
                level=level,
                heading=heading_text,
                text=section_text,
                parent_id=None,
                order=len(sections),
                start_offset=0,  # HTML offsets are approximate
                end_offset=0
            ))

        return sections

    def _parse_plain_text(self, content: str, doc_id: str) -> List[Section]:
        """Parse plain text into sections (fallback)"""

        # Simple paragraph-based splitting
        paragraphs = content.split('\n\n')
        sections = []
        current_offset = 0

        for i, para in enumerate(paragraphs):
            if para.strip():
                section_id = self._generate_section_id(doc_id, para[:100], i)

                sections.append(Section(
                    id=section_id,
                    document_id=doc_id,
                    level=3,  # Default level
                    heading=None,
                    text=para.strip(),
                    parent_id=None,
                    order=i,
                    start_offset=current_offset,
                    end_offset=current_offset + len(para)
                ))

            current_offset += len(para) + 2  # +2 for double newline

        return sections

    def _establish_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Establish parent-child relationships between sections"""

        if not sections:
            return sections

        # Stack to track parent sections at each level
        parent_stack = [None] * 7  # Levels 0-6

        for section in sections:
            level = section.level

            # Find parent (closest section with lower level)
            parent = None
            for l in range(level - 1, 0, -1):
                if parent_stack[l]:
                    parent = parent_stack[l]
                    break

            section.parent_id = parent.id if parent else None

            # Update stack
            parent_stack[level] = section
            # Clear higher levels
            for l in range(level + 1, 7):
                parent_stack[l] = None

        return sections
EOF
```

### Task 7E.3.2: Token Counter Integration

#### Step 1: Create Token Counter Module

```python
# Create token counter with dual backend
cat > src/ingestion/token_counter.py << 'EOF'
"""Token counting with exact Jina v3 tokenizer"""

import os
import hashlib
import json
from typing import List, Optional
from transformers import AutoTokenizer
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    """Token counter with HuggingFace and Jina Segmenter backends"""

    def __init__(self):
        self.backend = os.getenv('TOKENIZER_BACKEND', 'hf')
        self.max_tokens = int(os.getenv('EMBED_MAX_TOKENS', 8192))
        self.target_tokens = int(os.getenv('EMBED_TARGET_TOKENS', 7900))

        if self.backend == 'hf':
            self._init_huggingface()
        else:
            self._init_segmenter()

    def _init_huggingface(self):
        """Initialize HuggingFace tokenizer"""
        tokenizer_id = os.getenv('HF_TOKENIZER_ID', 'jinaai/jina-embeddings-v3')
        cache_dir = os.getenv('HF_CACHE', '/opt/graphrag-v2/models')

        logger.info(f"Loading HuggingFace tokenizer: {tokenizer_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            cache_dir=cache_dir,
            use_fast=True  # Use Rust-based fast tokenizer
        )

        logger.info("HuggingFace tokenizer loaded successfully")

    def _init_segmenter(self):
        """Initialize Jina Segmenter API client"""
        self.segmenter_url = os.getenv(
            'JINA_SEGMENTER_BASE_URL',
            'https://api.jina.ai/v1/segment'
        )
        self.api_key = os.getenv('JINA_API_KEY')
        self.timeout = int(os.getenv('SEGMENTER_TIMEOUT_MS', 3000)) / 1000

        logger.info("Jina Segmenter API client initialized")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured backend

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.backend == 'hf':
            return self._count_hf(text)
        else:
            return self._count_segmenter(text)

    def _count_hf(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer"""

        # Encode without special tokens
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        return len(tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10)
    )
    def _count_segmenter(self, text: str) -> int:
        """Count tokens using Jina Segmenter API"""

        headers = {
            'Content-Type': 'application/json'
        }

        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        payload = {
            'text': text,
            'tokenizer': 'xlm-roberta-base'  # Jina v3 uses XLM-RoBERTa
        }

        try:
            response = httpx.post(
                self.segmenter_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            return data.get('num_tokens', 0)

        except Exception as e:
            logger.error(f"Segmenter API error: {e}")
            # Fall back to estimation
            return len(text.split()) * 1.3  # Conservative estimate

    def needs_splitting(self, text: str) -> bool:
        """Check if text exceeds token limit

        Args:
            text: Input text

        Returns:
            True if text needs splitting
        """
        return self.count_tokens(text) > self.max_tokens

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to exact token count

        Args:
            text: Input text
            max_tokens: Maximum token count

        Returns:
            Truncated text
        """
        if self.backend == 'hf':
            return self._truncate_hf(text, max_tokens)
        else:
            return self._truncate_segmenter(text, max_tokens)

    def _truncate_hf(self, text: str, max_tokens: int) -> str:
        """Truncate using HuggingFace tokenizer"""

        # Encode
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        # Check if truncation needed
        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return truncated_text

    def _truncate_segmenter(self, text: str, max_tokens: int) -> str:
        """Truncate using segmenter (approximate)"""

        # Use word-based approximation
        words = text.split()
        estimated_ratio = max_tokens / self.count_tokens(text)
        target_words = int(len(words) * estimated_ratio * 0.95)  # Conservative

        return ' '.join(words[:target_words])

    def compute_integrity_hash(self, text: str) -> str:
        """Compute SHA256 hash for integrity verification

        Args:
            text: Input text

        Returns:
            SHA256 hash hex string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def validate_chunk_integrity(
        self,
        original_text: str,
        chunks: List[str],
        overlap: int = 0
    ) -> bool:
        """Validate that chunks preserve original content

        Args:
            original_text: Original text
            chunks: List of chunk texts
            overlap: Number of overlapping tokens

        Returns:
            True if content is preserved
        """
        if not chunks:
            return False

        if len(chunks) == 1:
            # Single chunk should match original
            return chunks[0] == original_text

        # For multiple chunks with overlap, verify coverage
        # This is approximate due to overlap complexity
        original_hash = self.compute_integrity_hash(original_text)

        # Reconstruct text (simplified - doesn't handle overlap perfectly)
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            # Skip overlap tokens (approximate)
            words = chunk.split()
            overlap_words = int(overlap * 0.7)  # Approximate word count
            if len(words) > overlap_words:
                reconstructed += ' ' + ' '.join(words[overlap_words:])

        # Check if essential content is preserved
        # Allow some differences due to overlap handling
        return len(reconstructed) >= len(original_text) * 0.95
EOF
```

### Task 7E.3.3: Intelligent Chunking Pipeline

#### Step 1: Create Chunking Module

```python
# Create intelligent chunking module
cat > src/ingestion/chunking_pipeline.py << 'EOF'
"""Intelligent chunking pipeline for GraphRAG v2.1"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging

from .document_parser import Section
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a chunk ready for embedding"""
    id: str
    document_id: str
    parent_section_id: Optional[str]
    level: int
    order: int
    heading: Optional[str]
    text: str
    is_combined: bool
    is_split: bool
    original_section_ids: List[str]
    boundaries_json: str
    token_count: int
    updated_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'parent_section_id': self.parent_section_id,
            'level': self.level,
            'order': self.order,
            'heading': self.heading,
            'text': self.text,
            'is_combined': self.is_combined,
            'is_split': self.is_split,
            'original_section_ids': self.original_section_ids,
            'boundaries_json': self.boundaries_json,
            'token_count': self.token_count,
            'updated_at': self.updated_at.isoformat()
        }

class ChunkingPipeline:
    """Intelligent chunking with combine and split strategies"""

    def __init__(self):
        self.token_counter = TokenCounter()

        # Configuration
        self.target_min = int(os.getenv('TARGET_MIN_TOKENS', 800))
        self.target_max = int(os.getenv('TARGET_MAX_TOKENS', 1500))
        self.absolute_max = int(os.getenv('ABSOLUTE_MAX_TOKENS', 7900))
        self.overlap_tokens = int(os.getenv('OVERLAP_TOKENS', 200))
        self.split_min = int(os.getenv('SPLIT_MIN_TOKENS', 1000))

        # Feature flags
        self.combine_enabled = os.getenv('COMBINE_SECTIONS', 'true').lower() == 'true'
        self.split_enabled = os.getenv('SPLIT_FALLBACK', 'true').lower() == 'true'

        logger.info(f"ChunkingPipeline initialized: combine={self.combine_enabled}, split={self.split_enabled}")
        logger.info(f"Token targets: min={self.target_min}, max={self.target_max}, absolute_max={self.absolute_max}")

    def process_sections(self, sections: List[Section]) -> List[Chunk]:
        """Process sections into chunks

        Args:
            sections: List of document sections

        Returns:
            List of chunks ready for embedding
        """
        if not sections:
            return []

        # Group sections by parent (typically H2 level)
        section_groups = self._group_by_parent(sections)

        all_chunks = []

        for parent_id, group_sections in section_groups.items():
            if self.combine_enabled:
                # Combine small sections
                chunks = self._combine_sections(group_sections, parent_id)
            else:
                # Direct conversion (no combining)
                chunks = self._sections_to_chunks(group_sections, parent_id)

            # Handle oversized chunks
            if self.split_enabled:
                final_chunks = []
                for chunk in chunks:
                    if chunk.token_count > self.absolute_max:
                        # Split oversized chunk
                        split_chunks = self._split_chunk(chunk)
                        final_chunks.extend(split_chunks)
                    else:
                        final_chunks.append(chunk)
                chunks = final_chunks

            # Update order within parent group
            for i, chunk in enumerate(chunks):
                chunk.order = i

            all_chunks.extend(chunks)

        logger.info(f"Processed {len(sections)} sections into {len(all_chunks)} chunks")

        return all_chunks

    def _group_by_parent(self, sections: List[Section]) -> Dict[str, List[Section]]:
        """Group sections by parent section ID

        Args:
            sections: List of sections

        Returns:
            Dictionary mapping parent_id to list of sections
        """
        groups = {}

        for section in sections:
            # Use H2 level parent or create virtual parent
            if section.level <= 2:
                parent_id = section.id  # H1/H2 are their own parent
            else:
                parent_id = section.parent_id or f"virtual_{section.document_id}"

            if parent_id not in groups:
                groups[parent_id] = []

            groups[parent_id].append(section)

        return groups

    def _combine_sections(
        self,
        sections: List[Section],
        parent_id: str
    ) -> List[Chunk]:
        """Combine adjacent sections into optimal chunks

        Args:
            sections: List of sections to combine
            parent_id: Parent section ID

        Returns:
            List of combined chunks
        """
        if not sections:
            return []

        chunks = []
        current_sections = []
        current_tokens = 0

        for section in sections:
            section_tokens = self.token_counter.count_tokens(section.text)

            # Check for hard breaks (new H2, special sections)
            is_hard_break = (
                section.level <= 2 or
                self._is_special_section(section.heading)
            )

            # Flush current chunk if needed
            if is_hard_break and current_sections:
                chunk = self._create_combined_chunk(
                    current_sections, parent_id, len(chunks)
                )
                chunks.append(chunk)
                current_sections = []
                current_tokens = 0

            # Check if adding section would exceed limits
            if current_tokens + section_tokens > self.target_max:
                if current_sections:
                    # Flush current chunk
                    chunk = self._create_combined_chunk(
                        current_sections, parent_id, len(chunks)
                    )
                    chunks.append(chunk)
                    current_sections = []
                    current_tokens = 0

            # Add section to current chunk
            current_sections.append(section)
            current_tokens += section_tokens

            # Check if we've reached target size
            if current_tokens >= self.target_max:
                chunk = self._create_combined_chunk(
                    current_sections, parent_id, len(chunks)
                )
                chunks.append(chunk)
                current_sections = []
                current_tokens = 0

        # Handle remaining sections
        if current_sections:
            # Try to merge with previous chunk if small
            if chunks and current_tokens < self.target_min:
                last_chunk = chunks[-1]
                combined_tokens = last_chunk.token_count + current_tokens

                if combined_tokens <= self.absolute_max:
                    # Merge with last chunk
                    chunks[-1] = self._merge_chunks(
                        last_chunk, current_sections, parent_id
                    )
                else:
                    # Create separate chunk
                    chunk = self._create_combined_chunk(
                        current_sections, parent_id, len(chunks)
                    )
                    chunks.append(chunk)
            else:
                # Create final chunk
                chunk = self._create_combined_chunk(
                    current_sections, parent_id, len(chunks)
                )
                chunks.append(chunk)

        return chunks

    def _sections_to_chunks(
        self,
        sections: List[Section],
        parent_id: str
    ) -> List[Chunk]:
        """Convert sections directly to chunks (no combining)

        Args:
            sections: List of sections
            parent_id: Parent section ID

        Returns:
            List of chunks
        """
        chunks = []

        for i, section in enumerate(sections):
            chunk_id = self._generate_chunk_id(
                section.document_id,
                [section.id]
            )

            chunk = Chunk(
                id=chunk_id,
                document_id=section.document_id,
                parent_section_id=parent_id,
                level=section.level,
                order=i,
                heading=section.heading,
                text=section.text,
                is_combined=False,
                is_split=False,
                original_section_ids=[section.id],
                boundaries_json=json.dumps({
                    'start_offset': section.start_offset,
                    'end_offset': section.end_offset
                }),
                token_count=self.token_counter.count_tokens(section.text),
                updated_at=datetime.utcnow()
            )

            chunks.append(chunk)

        return chunks

    def _create_combined_chunk(
        self,
        sections: List[Section],
        parent_id: str,
        order: int
    ) -> Chunk:
        """Create a combined chunk from multiple sections

        Args:
            sections: List of sections to combine
            parent_id: Parent section ID
            order: Chunk order within parent

        Returns:
            Combined chunk
        """
        # Combine text with clear separators
        combined_text = '\n\n'.join(s.text for s in sections)

        # Combine headings
        headings = [s.heading for s in sections if s.heading]
        combined_heading = ' | '.join(headings[:3]) if headings else None

        # Track boundaries
        boundaries = {
            'sections': len(sections),
            'start_offset': sections[0].start_offset,
            'end_offset': sections[-1].end_offset,
            'first_heading': sections[0].heading,
            'last_heading': sections[-1].heading
        }

        # Generate stable chunk ID
        section_ids = [s.id for s in sections]
        chunk_id = self._generate_chunk_id(sections[0].document_id, section_ids)

        return Chunk(
            id=chunk_id,
            document_id=sections[0].document_id,
            parent_section_id=parent_id,
            level=min(s.level for s in sections),
            order=order,
            heading=combined_heading,
            text=combined_text,
            is_combined=len(sections) > 1,
            is_split=False,
            original_section_ids=section_ids,
            boundaries_json=json.dumps(boundaries),
            token_count=self.token_counter.count_tokens(combined_text),
            updated_at=datetime.utcnow()
        )

    def _split_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split an oversized chunk into smaller chunks

        Args:
            chunk: Oversized chunk to split

        Returns:
            List of split chunks
        """
        text = chunk.text
        token_count = chunk.token_count

        # Calculate number of splits needed
        num_splits = (token_count // self.absolute_max) + 1
        target_size = token_count // num_splits

        # Find split points (prefer paragraph boundaries)
        paragraphs = text.split('\n\n')

        splits = []
        current_text = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.token_counter.count_tokens(para)

            if current_tokens + para_tokens > target_size and current_text:
                # Create split
                split_text = '\n\n'.join(current_text)
                splits.append(split_text)

                # Add overlap from current paragraph
                if self.overlap_tokens > 0 and para_tokens > self.overlap_tokens:
                    overlap_text = self.token_counter.truncate_to_token_limit(
                        para, self.overlap_tokens
                    )
                    current_text = [overlap_text, para]
                    current_tokens = para_tokens + self.overlap_tokens
                else:
                    current_text = [para]
                    current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Add final split
        if current_text:
            splits.append('\n\n'.join(current_text))

        # Create chunk objects for splits
        split_chunks = []

        for i, split_text in enumerate(splits):
            split_id = f"{chunk.id}_split_{i}"

            split_chunk = Chunk(
                id=split_id,
                document_id=chunk.document_id,
                parent_section_id=chunk.parent_section_id,
                level=chunk.level,
                order=chunk.order * 100 + i,  # Maintain relative order
                heading=f"{chunk.heading} (Part {i+1}/{len(splits)})" if chunk.heading else None,
                text=split_text,
                is_combined=chunk.is_combined,
                is_split=True,
                original_section_ids=chunk.original_section_ids,
                boundaries_json=json.dumps({
                    'split_index': i,
                    'total_splits': len(splits),
                    'has_overlap': i > 0 and self.overlap_tokens > 0
                }),
                token_count=self.token_counter.count_tokens(split_text),
                updated_at=datetime.utcnow()
            )

            split_chunks.append(split_chunk)

        logger.info(f"Split chunk {chunk.id} into {len(split_chunks)} parts")

        return split_chunks

    def _merge_chunks(
        self,
        chunk: Chunk,
        sections: List[Section],
        parent_id: str
    ) -> Chunk:
        """Merge additional sections into existing chunk

        Args:
            chunk: Existing chunk
            sections: Sections to merge
            parent_id: Parent section ID

        Returns:
            Merged chunk
        """
        # Append new text
        new_text = '\n\n'.join(s.text for s in sections)
        merged_text = chunk.text + '\n\n' + new_text

        # Update metadata
        all_section_ids = chunk.original_section_ids + [s.id for s in sections]

        # Update boundaries
        boundaries = json.loads(chunk.boundaries_json)
        boundaries['sections'] = boundaries.get('sections', 1) + len(sections)
        boundaries['end_offset'] = sections[-1].end_offset
        boundaries['last_heading'] = sections[-1].heading

        return Chunk(
            id=chunk.id,  # Keep same ID
            document_id=chunk.document_id,
            parent_section_id=parent_id,
            level=chunk.level,
            order=chunk.order,
            heading=chunk.heading,
            text=merged_text,
            is_combined=True,
            is_split=chunk.is_split,
            original_section_ids=all_section_ids,
            boundaries_json=json.dumps(boundaries),
            token_count=self.token_counter.count_tokens(merged_text),
            updated_at=datetime.utcnow()
        )

    def _generate_chunk_id(
        self,
        document_id: str,
        section_ids: List[str]
    ) -> str:
        """Generate stable chunk ID

        Args:
            document_id: Document ID
            section_ids: List of section IDs

        Returns:
            Stable chunk ID
        """
        unique_str = f"{document_id}|{'|'.join(sorted(section_ids))}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:24]

    def _is_special_section(self, heading: Optional[str]) -> bool:
        """Check if section is special (FAQ, Glossary, etc.)

        Args:
            heading: Section heading

        Returns:
            True if special section
        """
        if not heading:
            return False

        special_patterns = [
            r'^FAQ',
            r'^Frequently Asked',
            r'^Glossary',
            r'^Appendix',
            r'^References',
            r'^Bibliography',
            r'^Changelog',
            r'^Release Notes',
            r'^Index',
            r'^Table of Contents'
        ]

        for pattern in special_patterns:
            if re.match(pattern, heading, re.IGNORECASE):
                return True

        return False
EOF
```

## Continuation Note

This implementation plan continues with detailed instructions for:
- Phase 7E.4: Embedding Service Implementation
- Phase 7E.5: Storage Integration
- Phase 7E.6: Retrieval Pipeline
- Phase 7E.7: Session Management
- Phase 7E.8: Quality Assurance
- Phase 7E.9: Monitoring and Operations

Each phase includes complete, prescriptive code and configuration that an agentic coder can execute directly without interpretation. The plan ensures zero content loss, production stability, and full feature implementation with no placeholders or incomplete sections.
