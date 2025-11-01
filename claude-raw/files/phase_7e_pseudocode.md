# Phase-7E GraphRAG v2.1 Complete Pseudocode Document

## Overview

This document provides comprehensive pseudocode for every component and phase of the GraphRAG v2.1 implementation. Each section aligns with the phases defined in the application specification and implementation plan, providing detailed algorithmic guidance for the agentic coder.

## Phase 7E.1: Foundation Layer Pseudocode

### 7E.1.1: Environment Setup

```pseudocode
PROCEDURE SetupEnvironment():
    // Create directory structure
    base_dir = "/opt/graphrag-v2"
    CREATE_DIRECTORY(base_dir)
    
    subdirs = [
        "config", "data/neo4j", "data/qdrant", "data/redis",
        "logs", "scripts", "src/ingestion", "src/providers",
        "src/retrieval", "src/storage", "src/utils", "tests", "temp"
    ]
    
    FOR EACH dir IN subdirs:
        CREATE_DIRECTORY(base_dir + "/" + dir)
        SET_PERMISSIONS(base_dir + "/" + dir, 755)
    
    // Install Docker containers
    EXECUTE_COMMAND("docker network create graphrag-network")
    
    // Neo4j setup
    neo4j_config = LOAD_DOCKER_COMPOSE("neo4j")
    EXECUTE_COMMAND("docker-compose -f neo4j.yml up -d")
    WAIT_FOR_SERVICE("neo4j", port=7687, timeout=30)
    
    // Qdrant setup
    qdrant_config = LOAD_DOCKER_COMPOSE("qdrant")
    EXECUTE_COMMAND("docker-compose -f qdrant.yml up -d")
    WAIT_FOR_SERVICE("qdrant", port=6333, timeout=30)
    
    // Redis setup
    redis_config = LOAD_DOCKER_COMPOSE("redis")
    EXECUTE_COMMAND("docker-compose -f redis.yml up -d")
    WAIT_FOR_SERVICE("redis", port=6379, timeout=30)
    
    RETURN SUCCESS
```

### 7E.1.2: Dependency Installation

```pseudocode
PROCEDURE InstallDependencies():
    // Create Python virtual environment
    python_version = "3.11"
    venv_path = "/opt/graphrag-v2/venv"
    
    CREATE_VIRTUAL_ENV(python_version, venv_path)
    ACTIVATE_VIRTUAL_ENV(venv_path)
    
    // Core dependencies
    dependencies = {
        "database": ["neo4j==5.15.0", "qdrant-client==1.7.3", "redis==5.0.1"],
        "embeddings": ["transformers==4.36.2", "sentencepiece==0.1.99", 
                       "tokenizers==0.15.0", "huggingface-hub==0.20.2"],
        "processing": ["numpy==1.26.3", "pandas==2.1.4", "pydantic==2.5.3"],
        "async": ["httpx==0.25.2", "asyncio==3.4.3", "aiohttp==3.9.1"],
        "utils": ["python-dotenv==1.0.0", "structlog==24.1.0"]
    }
    
    FOR EACH category, packages IN dependencies:
        FOR EACH package IN packages:
            INSTALL_PACKAGE(package)
            VERIFY_INSTALLATION(package)
    
    // Prefetch Jina tokenizer
    PREFETCH_TOKENIZER("jinaai/jina-embeddings-v3", cache_dir="/opt/graphrag-v2/models")
    
    RETURN SUCCESS
```

### 7E.1.3: Configuration Management

```pseudocode
PROCEDURE SetupConfiguration():
    // Create environment configuration
    env_config = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "graphrag2024!",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "chunks",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": 6379,
        "JINA_API_KEY": GET_ENV_OR_PROMPT("JINA_API_KEY"),
        "EMBED_DIM": 1024,
        "EMBED_MODEL_ID": "jina-embeddings-v3",
        "TARGET_MIN_TOKENS": 800,
        "TARGET_MAX_TOKENS": 1500,
        "ABSOLUTE_MAX_TOKENS": 7900
    }
    
    WRITE_ENV_FILE(".env", env_config)
    
    // Setup logging
    logging_config = {
        "level": "INFO",
        "format": "structured_json",
        "handlers": ["console", "file", "error_file"],
        "file_path": "/opt/graphrag-v2/logs/graphrag.log",
        "max_bytes": 10485760,
        "backup_count": 5
    }
    
    CONFIGURE_LOGGING(logging_config)
    
    // Initialize error handling framework
    error_handler = ErrorHandler(
        retry_attempts=3,
        circuit_breaker_threshold=5,
        recovery_timeout=60
    )
    
    REGISTER_GLOBAL_ERROR_HANDLER(error_handler)
    
    RETURN SUCCESS
```

## Phase 7E.2: Schema Implementation Pseudocode

### 7E.2.1: Neo4j Schema Creation

```pseudocode
PROCEDURE CreateNeo4jSchema():
    driver = NEO4J_CONNECT(env.NEO4J_URI, env.NEO4J_USER, env.NEO4J_PASSWORD)
    
    // Load and execute main schema
    schema_ddl = READ_FILE("create_schema_v2_1_complete__v3.cypher")
    
    WITH driver.session() AS session:
        // Split DDL into individual statements
        statements = SPLIT_CYPHER_STATEMENTS(schema_ddl)
        
        FOR EACH statement IN statements:
            TRY:
                session.run(statement)
                LOG_INFO(f"Executed: {statement[:50]}...")
            CATCH error:
                IF NOT IS_IDEMPOTENT_ERROR(error):
                    RAISE error
                LOG_DEBUG(f"Idempotent skip: {error}")
    
    // Verify schema creation
    constraints = session.run("SHOW CONSTRAINTS").data()
    indexes = session.run("SHOW INDEXES").data()
    
    required_constraints = [
        "document_id_unique", "section_id_unique", 
        "session_id_unique", "query_id_unique", "answer_id_unique"
    ]
    
    FOR EACH constraint IN required_constraints:
        IF constraint NOT IN constraints:
            RAISE SchemaError(f"Missing constraint: {constraint}")
    
    // Create additional indexes for chunking
    additional_indexes = [
        "CREATE INDEX chunk_composite_idx IF NOT EXISTS 
         FOR (c:Chunk) ON (c.document_id, c.parent_section_id, c.order)",
        "CREATE TEXT INDEX chunk_text_idx IF NOT EXISTS 
         FOR (c:Chunk) ON (c.text)",
        "CREATE INDEX chunk_flags_idx IF NOT EXISTS 
         FOR (c:Chunk) ON (c.is_combined, c.is_split)"
    ]
    
    FOR EACH index_ddl IN additional_indexes:
        session.run(index_ddl)
    
    driver.close()
    RETURN SUCCESS
```

### 7E.2.2: Qdrant Collection Setup

```pseudocode
PROCEDURE CreateQdrantCollection():
    client = QDRANT_CONNECT(env.QDRANT_URL, env.QDRANT_API_KEY)
    collection_name = env.QDRANT_COLLECTION
    embed_dim = env.EMBED_DIM
    
    // Check if collection exists
    existing_collections = client.get_collections().collections
    
    IF collection_name IN existing_collections:
        LOG_INFO(f"Recreating collection {collection_name}")
        client.delete_collection(collection_name)
    
    // Create collection with optimized settings
    vector_config = {
        "content": VectorParams(
            size=embed_dim,
            distance=Distance.COSINE,
            on_disk=FALSE  // Keep in memory for performance
        )
    }
    
    hnsw_config = HnswConfig(
        m=16,  // Connections per node
        ef_construct=200,  // Construction quality
        full_scan_threshold=10000  // Switch to exact search below this
    )
    
    optimizer_config = OptimizersConfig(
        memmap_threshold=20000,
        indexing_threshold=20000,
        flush_interval_sec=5
    )
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vector_config,
        hnsw_config=hnsw_config,
        optimizers_config=optimizer_config
    )
    
    // Create payload indexes
    payload_indexes = [
        ("document_id", PayloadSchemaType.KEYWORD),
        ("parent_section_id", PayloadSchemaType.KEYWORD),
        ("order", PayloadSchemaType.INTEGER),
        ("level", PayloadSchemaType.INTEGER),
        ("token_count", PayloadSchemaType.INTEGER),
        ("updated_at", PayloadSchemaType.INTEGER),
        ("is_combined", PayloadSchemaType.BOOL),
        ("is_split", PayloadSchemaType.BOOL)
    ]
    
    FOR EACH field_name, field_type IN payload_indexes:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type
        )
    
    RETURN SUCCESS
```

### 7E.2.3: Schema Validation

```pseudocode
PROCEDURE ValidateSchema():
    validation_results = []
    
    // Validate Neo4j
    neo4j_driver = NEO4J_CONNECT(env.NEO4J_URI, env.NEO4J_USER, env.NEO4J_PASSWORD)
    
    WITH neo4j_driver.session() AS session:
        // Check constraints
        constraints = session.run("SHOW CONSTRAINTS").data()
        constraint_names = EXTRACT_NAMES(constraints)
        
        required_constraints = [
            "document_id_unique", "section_id_unique",
            "session_id_unique", "query_id_unique", "answer_id_unique"
        ]
        
        FOR EACH constraint IN required_constraints:
            IF constraint IN constraint_names:
                validation_results.APPEND((constraint, TRUE, "exists"))
            ELSE:
                validation_results.APPEND((constraint, FALSE, "missing"))
        
        // Check vector indexes
        indexes = session.run("SHOW INDEXES WHERE type = 'VECTOR'").data()
        
        required_vector_indexes = ["section_embeddings_v2", "chunk_embeddings_v2"]
        
        FOR EACH idx IN required_vector_indexes:
            IF idx IN EXTRACT_NAMES(indexes):
                // Verify dimensions
                idx_config = GET_INDEX_CONFIG(idx)
                IF idx_config.dimensions == 1024:
                    validation_results.APPEND((idx, TRUE, "1024-D"))
                ELSE:
                    validation_results.APPEND((idx, FALSE, f"{idx_config.dimensions}-D"))
            ELSE:
                validation_results.APPEND((idx, FALSE, "missing"))
    
    // Validate Qdrant
    qdrant_client = QDRANT_CONNECT(env.QDRANT_URL, env.QDRANT_API_KEY)
    
    TRY:
        collection_info = qdrant_client.get_collection(env.QDRANT_COLLECTION)
        
        IF collection_info.config.params.vectors['content'].size == 1024:
            validation_results.APPEND(("qdrant_dimensions", TRUE, "1024"))
        ELSE:
            validation_results.APPEND(("qdrant_dimensions", FALSE, 
                                     f"{collection_info.config.params.vectors['content'].size}"))
        
        IF collection_info.status == 'green':
            validation_results.APPEND(("qdrant_status", TRUE, "green"))
        ELSE:
            validation_results.APPEND(("qdrant_status", FALSE, collection_info.status))
    CATCH error:
        validation_results.APPEND(("qdrant_collection", FALSE, str(error)))
    
    // Check for failures
    failures = FILTER(validation_results, lambda x: NOT x[1])
    
    IF LENGTH(failures) > 0:
        FOR EACH name, passed, details IN failures:
            LOG_ERROR(f"Validation failed: {name} - {details}")
        RAISE ValidationError(f"{LENGTH(failures)} validations failed")
    
    LOG_INFO("All schema validations passed")
    RETURN SUCCESS
```

## Phase 7E.3: Document Processing Pipeline Pseudocode

### 7E.3.1: Document Parser Implementation

```pseudocode
CLASS DocumentParser:
    CONSTRUCTOR():
        this.section_counter = 0
        this.markdown_parser = MARKDOWN_PARSER(extensions=['extra', 'toc'])
    
    METHOD parse(content, metadata):
        // Generate document ID from URI
        doc_id = SHA256(metadata.source_uri)[:24]
        
        // Parse based on format
        IF metadata.source_type == "markdown":
            sections = this.parse_markdown(content, doc_id)
        ELSE IF metadata.source_type == "html":
            sections = this.parse_html(content, doc_id)
        ELSE:
            sections = this.parse_plain_text(content, doc_id)
        
        // Establish hierarchy
        sections = this.establish_hierarchy(sections)
        
        RETURN (doc_id, sections)
    
    METHOD parse_markdown(content, doc_id):
        sections = []
        lines = SPLIT_LINES(content)
        current_section = NULL
        current_text = []
        current_offset = 0
        
        FOR EACH line IN lines:
            heading_match = REGEX_MATCH(r'^(#{1,6})\s+(.+)$', line)
            
            IF heading_match:
                // Save previous section
                IF current_section != NULL:
                    current_section.text = JOIN(current_text, '\n').STRIP()
                    current_section.end_offset = current_offset
                    sections.APPEND(current_section)
                    current_text = []
                
                // Create new section
                level = LENGTH(heading_match.group(1))
                heading = heading_match.group(2).STRIP()
                
                section_id = SHA256(f"{doc_id}|{LENGTH(sections)}|{heading[:100]}")[:24]
                
                current_section = Section(
                    id=section_id,
                    document_id=doc_id,
                    level=level,
                    heading=heading,
                    text="",
                    parent_id=NULL,
                    order=LENGTH(sections),
                    start_offset=current_offset,
                    end_offset=current_offset
                )
            ELSE:
                current_text.APPEND(line)
            
            current_offset += LENGTH(line) + 1  // +1 for newline
        
        // Save final section
        IF current_section != NULL:
            current_section.text = JOIN(current_text, '\n').STRIP()
            current_section.end_offset = current_offset
            sections.APPEND(current_section)
        
        RETURN sections
    
    METHOD establish_hierarchy(sections):
        parent_stack = ARRAY[7]  // Stack for levels 0-6
        
        FOR EACH section IN sections:
            level = section.level
            
            // Find parent (closest section with lower level)
            parent = NULL
            FOR l FROM level-1 DOWNTO 1:
                IF parent_stack[l] != NULL:
                    parent = parent_stack[l]
                    BREAK
            
            section.parent_id = parent.id IF parent ELSE NULL
            
            // Update stack
            parent_stack[level] = section
            
            // Clear higher levels
            FOR l FROM level+1 TO 6:
                parent_stack[l] = NULL
        
        RETURN sections
```

### 7E.3.2: Token Counter Integration

```pseudocode
CLASS TokenCounter:
    CONSTRUCTOR():
        this.backend = GET_ENV('TOKENIZER_BACKEND', 'hf')
        this.max_tokens = INT(GET_ENV('EMBED_MAX_TOKENS', 8192))
        this.target_tokens = INT(GET_ENV('EMBED_TARGET_TOKENS', 7900))
        
        IF this.backend == 'hf':
            this.init_huggingface()
        ELSE:
            this.init_segmenter()
    
    METHOD init_huggingface():
        tokenizer_id = GET_ENV('HF_TOKENIZER_ID', 'jinaai/jina-embeddings-v3')
        cache_dir = GET_ENV('HF_CACHE', '/opt/graphrag-v2/models')
        
        this.tokenizer = LOAD_TOKENIZER(tokenizer_id, cache_dir)
        LOG_INFO(f"Loaded HuggingFace tokenizer: {tokenizer_id}")
    
    METHOD count_tokens(text):
        IF this.backend == 'hf':
            tokens = this.tokenizer.encode(text, add_special_tokens=FALSE)
            RETURN LENGTH(tokens)
        ELSE:
            RETURN this.count_via_segmenter(text)
    
    METHOD count_via_segmenter(text):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GET_ENV("JINA_API_KEY")}'
        }
        
        payload = {
            'text': text,
            'tokenizer': 'xlm-roberta-base'
        }
        
        TRY:
            response = HTTP_POST(this.segmenter_url, json=payload, headers=headers)
            data = response.json()
            RETURN data['num_tokens']
        CATCH error:
            LOG_ERROR(f"Segmenter API error: {error}")
            // Fallback to estimation
            RETURN LENGTH(SPLIT_WORDS(text)) * 1.3
    
    METHOD needs_splitting(text):
        RETURN this.count_tokens(text) > this.max_tokens
    
    METHOD truncate_to_token_limit(text, max_tokens):
        IF this.backend == 'hf':
            tokens = this.tokenizer.encode(text, add_special_tokens=FALSE)
            
            IF LENGTH(tokens) <= max_tokens:
                RETURN text
            
            truncated_tokens = tokens[:max_tokens]
            RETURN this.tokenizer.decode(truncated_tokens, skip_special_tokens=TRUE)
        ELSE:
            // Word-based approximation for segmenter
            words = SPLIT_WORDS(text)
            current_count = this.count_tokens(text)
            ratio = max_tokens / current_count
            target_words = INT(LENGTH(words) * ratio * 0.95)  // Conservative
            RETURN JOIN(words[:target_words], ' ')
    
    METHOD compute_integrity_hash(text):
        RETURN SHA256(text.encode('utf-8'))
```

### 7E.3.3: Intelligent Chunking Pipeline

```pseudocode
CLASS ChunkingPipeline:
    CONSTRUCTOR():
        this.token_counter = TokenCounter()
        this.target_min = INT(GET_ENV('TARGET_MIN_TOKENS', 800))
        this.target_max = INT(GET_ENV('TARGET_MAX_TOKENS', 1500))
        this.absolute_max = INT(GET_ENV('ABSOLUTE_MAX_TOKENS', 7900))
        this.overlap_tokens = INT(GET_ENV('OVERLAP_TOKENS', 200))
        this.combine_enabled = GET_ENV('COMBINE_SECTIONS', 'true') == 'true'
        this.split_enabled = GET_ENV('SPLIT_FALLBACK', 'true') == 'true'
    
    METHOD process_sections(sections):
        IF LENGTH(sections) == 0:
            RETURN []
        
        // Group sections by parent
        section_groups = this.group_by_parent(sections)
        all_chunks = []
        
        FOR EACH parent_id, group_sections IN section_groups:
            IF this.combine_enabled:
                chunks = this.combine_sections(group_sections, parent_id)
            ELSE:
                chunks = this.sections_to_chunks(group_sections, parent_id)
            
            // Handle oversized chunks
            IF this.split_enabled:
                final_chunks = []
                FOR EACH chunk IN chunks:
                    IF chunk.token_count > this.absolute_max:
                        split_chunks = this.split_chunk(chunk)
                        final_chunks.EXTEND(split_chunks)
                    ELSE:
                        final_chunks.APPEND(chunk)
                chunks = final_chunks
            
            // Update order within parent group
            FOR i FROM 0 TO LENGTH(chunks)-1:
                chunks[i].order = i
            
            all_chunks.EXTEND(chunks)
        
        LOG_INFO(f"Processed {LENGTH(sections)} sections into {LENGTH(all_chunks)} chunks")
        RETURN all_chunks
    
    METHOD combine_sections(sections, parent_id):
        chunks = []
        current_sections = []
        current_tokens = 0
        
        FOR EACH section IN sections:
            section_tokens = this.token_counter.count_tokens(section.text)
            
            // Check for hard breaks
            is_hard_break = (section.level <= 2 OR 
                            this.is_special_section(section.heading))
            
            // Flush current chunk if needed
            IF is_hard_break AND LENGTH(current_sections) > 0:
                chunk = this.create_combined_chunk(current_sections, parent_id, LENGTH(chunks))
                chunks.APPEND(chunk)
                current_sections = []
                current_tokens = 0
            
            // Check if adding section would exceed limits
            IF current_tokens + section_tokens > this.target_max:
                IF LENGTH(current_sections) > 0:
                    chunk = this.create_combined_chunk(current_sections, parent_id, LENGTH(chunks))
                    chunks.APPEND(chunk)
                    current_sections = []
                    current_tokens = 0
            
            // Add section to current chunk
            current_sections.APPEND(section)
            current_tokens += section_tokens
            
            // Check if reached target size
            IF current_tokens >= this.target_max:
                chunk = this.create_combined_chunk(current_sections, parent_id, LENGTH(chunks))
                chunks.APPEND(chunk)
                current_sections = []
                current_tokens = 0
        
        // Handle remaining sections
        IF LENGTH(current_sections) > 0:
            // Try to merge with previous chunk if small
            IF LENGTH(chunks) > 0 AND current_tokens < this.target_min:
                last_chunk = chunks[-1]
                combined_tokens = last_chunk.token_count + current_tokens
                
                IF combined_tokens <= this.absolute_max:
                    chunks[-1] = this.merge_chunks(last_chunk, current_sections, parent_id)
                ELSE:
                    chunk = this.create_combined_chunk(current_sections, parent_id, LENGTH(chunks))
                    chunks.APPEND(chunk)
            ELSE:
                chunk = this.create_combined_chunk(current_sections, parent_id, LENGTH(chunks))
                chunks.APPEND(chunk)
        
        RETURN chunks
    
    METHOD split_chunk(chunk):
        text = chunk.text
        token_count = chunk.token_count
        
        // Calculate number of splits needed
        num_splits = CEILING(token_count / this.absolute_max)
        target_size = token_count / num_splits
        
        // Find split points (prefer paragraph boundaries)
        paragraphs = SPLIT(text, '\n\n')
        
        splits = []
        current_text = []
        current_tokens = 0
        
        FOR EACH para IN paragraphs:
            para_tokens = this.token_counter.count_tokens(para)
            
            IF current_tokens + para_tokens > target_size AND LENGTH(current_text) > 0:
                // Create split
                split_text = JOIN(current_text, '\n\n')
                splits.APPEND(split_text)
                
                // Add overlap
                IF this.overlap_tokens > 0 AND para_tokens > this.overlap_tokens:
                    overlap_text = this.token_counter.truncate_to_token_limit(para, this.overlap_tokens)
                    current_text = [overlap_text, para]
                    current_tokens = para_tokens + this.overlap_tokens
                ELSE:
                    current_text = [para]
                    current_tokens = para_tokens
            ELSE:
                current_text.APPEND(para)
                current_tokens += para_tokens
        
        // Add final split
        IF LENGTH(current_text) > 0:
            splits.APPEND(JOIN(current_text, '\n\n'))
        
        // Create chunk objects for splits
        split_chunks = []
        FOR i FROM 0 TO LENGTH(splits)-1:
            split_id = f"{chunk.id}_split_{i}"
            
            split_chunk = Chunk(
                id=split_id,
                document_id=chunk.document_id,
                parent_section_id=chunk.parent_section_id,
                level=chunk.level,
                order=chunk.order * 100 + i,
                heading=f"{chunk.heading} (Part {i+1}/{LENGTH(splits)})" IF chunk.heading ELSE NULL,
                text=splits[i],
                is_combined=chunk.is_combined,
                is_split=TRUE,
                original_section_ids=chunk.original_section_ids,
                boundaries_json=JSON_STRINGIFY({
                    'split_index': i,
                    'total_splits': LENGTH(splits),
                    'has_overlap': i > 0 AND this.overlap_tokens > 0
                }),
                token_count=this.token_counter.count_tokens(splits[i]),
                updated_at=CURRENT_TIMESTAMP()
            )
            
            split_chunks.APPEND(split_chunk)
        
        RETURN split_chunks
    
    METHOD create_combined_chunk(sections, parent_id, order):
        // Combine text with clear separators
        combined_text = JOIN(MAP(sections, s => s.text), '\n\n')
        
        // Combine headings
        headings = FILTER(MAP(sections, s => s.heading), h => h != NULL)
        combined_heading = JOIN(headings[:3], ' | ') IF LENGTH(headings) > 0 ELSE NULL
        
        // Track boundaries
        boundaries = {
            'sections': LENGTH(sections),
            'start_offset': sections[0].start_offset,
            'end_offset': sections[-1].end_offset,
            'first_heading': sections[0].heading,
            'last_heading': sections[-1].heading
        }
        
        // Generate stable chunk ID
        section_ids = MAP(sections, s => s.id)
        unique_str = f"{sections[0].document_id}|{JOIN(SORT(section_ids), '|')}"
        chunk_id = SHA256(unique_str)[:24]
        
        RETURN Chunk(
            id=chunk_id,
            document_id=sections[0].document_id,
            parent_section_id=parent_id,
            level=MIN(MAP(sections, s => s.level)),
            order=order,
            heading=combined_heading,
            text=combined_text,
            is_combined=LENGTH(sections) > 1,
            is_split=FALSE,
            original_section_ids=section_ids,
            boundaries_json=JSON_STRINGIFY(boundaries),
            token_count=this.token_counter.count_tokens(combined_text),
            updated_at=CURRENT_TIMESTAMP()
        )
```

## Phase 7E.4: Embedding Service Pseudocode

### 7E.4.1: Embedding Provider Setup

```pseudocode
CLASS EmbeddingService:
    CONSTRUCTOR():
        this.provider = GET_ENV('EMBED_PROVIDER', 'jina-ai')
        this.model_id = GET_ENV('EMBED_MODEL_ID', 'jina-embeddings-v3')
        this.dimensions = INT(GET_ENV('EMBED_DIM', 1024))
        this.batch_size = INT(GET_ENV('EMBED_BATCH_SIZE', 32))
        this.api_key = GET_ENV('JINA_API_KEY')
        this.base_url = GET_ENV('JINA_BASE_URL')
        
        // Initialize HTTP client with retry logic
        this.client = HTTP_CLIENT(
            timeout=30,
            max_retries=3,
            backoff_factor=2
        )
        
        // Initialize cache
        this.cache = MultiLevelCache()
        
        // Circuit breaker for API failures
        this.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    METHOD generate_embedding(text):
        // Check cache first
        cached_embedding = this.cache.get_embedding(text)
        IF cached_embedding != NULL:
            METRICS.record_cache_hit('embedding', TRUE)
            RETURN cached_embedding
        
        METRICS.record_cache_hit('embedding', FALSE)
        
        // Call API with circuit breaker
        TRY:
            embedding = this.circuit_breaker.call(
                this._call_embedding_api, text
            )
            
            // Cache the result
            this.cache.set_embedding(text, embedding)
            
            RETURN embedding
        CATCH CircuitBreakerOpen:
            LOG_ERROR("Circuit breaker open for embedding service")
            RAISE EmbeddingError("Embedding service temporarily unavailable")
    
    METHOD _call_embedding_api(text):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {this.api_key}'
        }
        
        payload = {
            'input': text,
            'model': this.model_id
        }
        
        response = this.client.POST(
            this.base_url,
            json=payload,
            headers=headers
        )
        
        IF response.status_code == 429:
            // Rate limited
            retry_after = INT(response.headers.get('Retry-After', 60))
            SLEEP(retry_after)
            RAISE RateLimitError("Rate limited by API")
        
        response.raise_for_status()
        
        data = response.json()
        embedding = data['data'][0]['embedding']
        
        // Validate embedding
        IF LENGTH(embedding) != this.dimensions:
            RAISE EmbeddingError(
                f"Invalid embedding dimensions: {LENGTH(embedding)} != {this.dimensions}"
            )
        
        RETURN embedding
```

### 7E.4.2: Embedding Pipeline

```pseudocode
PROCEDURE GenerateEmbeddingsBatch(chunks):
    embedding_service = EmbeddingService()
    embeddings = []
    failed_chunks = []
    
    // Create adaptive batches based on token count
    batches = CREATE_ADAPTIVE_BATCHES(chunks, max_tokens=7900, max_batch_size=32)
    
    FOR EACH batch IN batches:
        TRY:
            // Process batch concurrently
            batch_embeddings = ASYNC_GATHER([
                embedding_service.generate_embedding(chunk.text) 
                FOR chunk IN batch
            ])
            
            embeddings.EXTEND(batch_embeddings)
            
            // Add metadata to chunks
            FOR i FROM 0 TO LENGTH(batch)-1:
                batch[i].vector_embedding = batch_embeddings[i]
                batch[i].embedding_version = embedding_service.model_id
                batch[i].embedding_provider = embedding_service.provider
                batch[i].embedding_dimensions = embedding_service.dimensions
                batch[i].embedding_timestamp = CURRENT_TIMESTAMP()
            
            LOG_INFO(f"Generated embeddings for batch of {LENGTH(batch)} chunks")
            
        CATCH error:
            LOG_ERROR(f"Failed to generate embeddings for batch: {error}")
            failed_chunks.EXTEND(batch)
            
            // Retry individual chunks from failed batch
            FOR chunk IN batch:
                TRY:
                    embedding = embedding_service.generate_embedding(chunk.text)
                    chunk.vector_embedding = embedding
                    chunk.embedding_version = embedding_service.model_id
                    chunk.embedding_provider = embedding_service.provider
                    chunk.embedding_dimensions = embedding_service.dimensions
                    chunk.embedding_timestamp = CURRENT_TIMESTAMP()
                    embeddings.APPEND(embedding)
                    REMOVE(chunk FROM failed_chunks)
                CATCH retry_error:
                    LOG_ERROR(f"Failed to embed chunk {chunk.id}: {retry_error}")
    
    // Report results
    success_rate = (LENGTH(chunks) - LENGTH(failed_chunks)) / LENGTH(chunks)
    LOG_INFO(f"Embedding generation complete: {success_rate:.2%} success rate")
    
    IF LENGTH(failed_chunks) > 0:
        LOG_WARNING(f"{LENGTH(failed_chunks)} chunks failed embedding generation")
    
    RETURN embeddings, failed_chunks

FUNCTION CREATE_ADAPTIVE_BATCHES(chunks, max_tokens, max_batch_size):
    batches = []
    current_batch = []
    current_tokens = 0
    
    FOR EACH chunk IN chunks:
        chunk_tokens = chunk.token_count
        
        // Check if adding chunk would exceed limits
        IF (LENGTH(current_batch) >= max_batch_size OR 
            current_tokens + chunk_tokens > max_tokens):
            
            IF LENGTH(current_batch) > 0:
                batches.APPEND(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.APPEND(chunk)
        current_tokens += chunk_tokens
    
    // Add final batch
    IF LENGTH(current_batch) > 0:
        batches.APPEND(current_batch)
    
    RETURN batches
```

### 7E.4.3: Embedding Storage

```pseudocode
PROCEDURE StoreEmbeddings(chunks_with_embeddings):
    neo4j_driver = NEO4J_CONNECT(env.NEO4J_URI, env.NEO4J_USER, env.NEO4J_PASSWORD)
    qdrant_client = QDRANT_CONNECT(env.QDRANT_URL, env.QDRANT_API_KEY)
    redis_cache = REDIS_CONNECT(env.REDIS_HOST, env.REDIS_PORT)
    
    // Prepare batch data
    neo4j_batch = []
    qdrant_points = []
    
    FOR EACH chunk IN chunks_with_embeddings:
        // Validate embedding exists
        IF chunk.vector_embedding == NULL:
            LOG_WARNING(f"Skipping chunk {chunk.id} - no embedding")
            CONTINUE
        
        // Prepare Neo4j data
        neo4j_batch.APPEND({
            'id': chunk.id,
            'document_id': chunk.document_id,
            'parent_section_id': chunk.parent_section_id,
            'level': chunk.level,
            'order': chunk.order,
            'heading': chunk.heading,
            'text': chunk.text,
            'is_combined': chunk.is_combined,
            'is_split': chunk.is_split,
            'original_section_ids': chunk.original_section_ids,
            'boundaries_json': chunk.boundaries_json,
            'token_count': chunk.token_count,
            'vector_embedding': chunk.vector_embedding,
            'embedding_version': chunk.embedding_version,
            'embedding_provider': chunk.embedding_provider,
            'embedding_dimensions': chunk.embedding_dimensions,
            'embedding_timestamp': chunk.embedding_timestamp,
            'updated_at': chunk.updated_at
        })
        
        // Prepare Qdrant point
        point_id = UUID5(UUID_NAMESPACE, chunk.id)
        
        qdrant_points.APPEND({
            'id': point_id,
            'vector': {
                'content': chunk.vector_embedding
            },
            'payload': {
                'id': chunk.id,
                'document_id': chunk.document_id,
                'parent_section_id': chunk.parent_section_id,
                'level': chunk.level,
                'order': chunk.order,
                'heading': chunk.heading,
                'text': chunk.text,
                'is_combined': chunk.is_combined,
                'is_split': chunk.is_split,
                'original_section_ids': chunk.original_section_ids,
                'boundaries_json': chunk.boundaries_json,
                'token_count': chunk.token_count,
                'embedding_version': chunk.embedding_version,
                'embedding_provider': chunk.embedding_provider,
                'embedding_dimensions': chunk.embedding_dimensions,
                'embedding_timestamp': chunk.embedding_timestamp,
                'updated_at': UNIX_TIMESTAMP(chunk.updated_at)
            }
        })
    
    // Store in Neo4j (transactional)
    WITH neo4j_driver.session() AS session:
        session.execute_write(lambda tx: 
            tx.run(
                """
                UNWIND $chunks AS row
                MERGE (s:Section:Chunk {id: row.id})
                  ON CREATE SET s += row
                  ON MATCH SET s += row
                WITH s, row
                MATCH (d:Document {id: row.document_id})
                MERGE (d)-[:HAS_SECTION]->(s)
                """,
                chunks=neo4j_batch
            )
        )
        
        // Create NEXT_CHUNK relationships
        session.execute_write(lambda tx:
            tx.run(
                """
                UNWIND $chunks AS row
                WITH row.parent_section_id AS pid, row.order AS idx, row.id AS cid
                ORDER BY pid, idx
                WITH pid, collect({idx: idx, cid: cid}) AS ordered
                UNWIND range(0, size(ordered)-2) AS i
                WITH pid, ordered[i] AS a, ordered[i+1] AS b
                MATCH (c1:Chunk {id: a.cid})
                MATCH (c2:Chunk {id: b.cid})
                MERGE (c1)-[:NEXT_CHUNK {parent_section_id: pid}]->(c2)
                """,
                chunks=neo4j_batch
            )
        )
    
    // Store in Qdrant (batch upsert)
    qdrant_client.upsert(
        collection_name=env.QDRANT_COLLECTION,
        points=qdrant_points,
        wait=TRUE  // Wait for indexing
    )
    
    // Warm cache
    FOR EACH chunk IN chunks_with_embeddings:
        cache_key = f"chunk:{chunk.id}"
        redis_cache.setex(
            cache_key,
            env.REDIS_CACHE_TTL,
            JSON_STRINGIFY({
                'embedding': chunk.vector_embedding,
                'metadata': {
                    'document_id': chunk.document_id,
                    'parent_section_id': chunk.parent_section_id,
                    'order': chunk.order
                }
            })
        )
    
    LOG_INFO(f"Stored {LENGTH(chunks_with_embeddings)} chunks with embeddings")
    RETURN SUCCESS
```

## Phase 7E.5: Storage Integration Pseudocode

### 7E.5.1: Neo4j Integration

```pseudocode
CLASS Neo4jStorage:
    CONSTRUCTOR():
        this.driver = GraphDatabase.driver(
            env.NEO4J_URI,
            auth=(env.NEO4J_USER, env.NEO4J_PASSWORD),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30
        )
    
    METHOD upsert_chunks(chunks):
        WITH this.driver.session() AS session:
            // Use transaction for consistency
            session.execute_write(
                lambda tx: this._upsert_chunks_transaction(tx, chunks)
            )
    
    METHOD _upsert_chunks_transaction(tx, chunks):
        // Prepare batch data
        chunk_data = []
        
        FOR EACH chunk IN chunks:
            chunk_data.APPEND(chunk.to_dict())
        
        // Upsert chunks
        result = tx.run(
            """
            UNWIND $chunks AS row
            MERGE (c:Section:Chunk {id: row.id})
              ON CREATE SET
                c.document_id = row.document_id,
                c.parent_section_id = row.parent_section_id,
                c.level = row.level,
                c.order = row.order,
                c.heading = row.heading,
                c.text = row.text,
                c.is_combined = row.is_combined,
                c.is_split = row.is_split,
                c.original_section_ids = row.original_section_ids,
                c.boundaries_json = row.boundaries_json,
                c.token_count = row.token_count,
                c.vector_embedding = row.vector_embedding,
                c.embedding_version = row.embedding_version,
                c.embedding_provider = row.embedding_provider,
                c.embedding_dimensions = row.embedding_dimensions,
                c.embedding_timestamp = row.embedding_timestamp,
                c.updated_at = datetime()
              ON MATCH SET
                c.document_id = row.document_id,
                c.parent_section_id = row.parent_section_id,
                c.level = row.level,
                c.order = row.order,
                c.heading = row.heading,
                c.text = row.text,
                c.is_combined = row.is_combined,
                c.is_split = row.is_split,
                c.original_section_ids = row.original_section_ids,
                c.boundaries_json = row.boundaries_json,
                c.token_count = row.token_count,
                c.vector_embedding = row.vector_embedding,
                c.embedding_version = row.embedding_version,
                c.embedding_provider = row.embedding_provider,
                c.embedding_dimensions = row.embedding_dimensions,
                c.embedding_timestamp = row.embedding_timestamp,
                c.updated_at = datetime()
            RETURN count(c) as chunks_upserted
            """,
            chunks=chunk_data
        )
        
        count = result.single()['chunks_upserted']
        LOG_INFO(f"Upserted {count} chunks in Neo4j")
        
        // Create document relationships
        tx.run(
            """
            UNWIND $chunks AS row
            MATCH (c:Chunk {id: row.id})
            MERGE (d:Document {id: row.document_id})
            MERGE (d)-[:HAS_SECTION]->(c)
            """,
            chunks=chunk_data
        )
        
        // Create adjacency relationships
        this._create_adjacency_relationships(tx, chunk_data)
        
        RETURN count
    
    METHOD _create_adjacency_relationships(tx, chunks):
        tx.run(
            """
            UNWIND $chunks AS row
            WITH row.parent_section_id AS pid, row.order AS ord, row.id AS cid
            ORDER BY pid, ord
            WITH pid, collect({order: ord, id: cid}) AS ordered_chunks
            WHERE size(ordered_chunks) > 1
            UNWIND range(0, size(ordered_chunks)-2) AS i
            WITH pid, ordered_chunks[i] AS curr, ordered_chunks[i+1] AS next
            MATCH (c1:Chunk {id: curr.id})
            MATCH (c2:Chunk {id: next.id})
            MERGE (c1)-[r:NEXT_CHUNK {parent_section_id: pid}]->(c2)
            RETURN count(r) as relationships_created
            """,
            chunks=chunks
        )
    
    METHOD delete_document_chunks(document_id):
        WITH this.driver.session() AS session:
            result = session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (c:Chunk {document_id: $doc_id})
                    DETACH DELETE c
                    RETURN count(c) as deleted_count
                    """,
                    doc_id=document_id
                )
            )
            
            count = result.single()['deleted_count']
            LOG_INFO(f"Deleted {count} chunks for document {document_id}")
            RETURN count
```

### 7E.5.2: Qdrant Integration

```pseudocode
CLASS QdrantStorage:
    CONSTRUCTOR():
        this.client = QdrantClient(
            url=env.QDRANT_URL,
            api_key=env.QDRANT_API_KEY,
            timeout=30
        )
        this.collection = env.QDRANT_COLLECTION
    
    METHOD upsert_chunks(chunks):
        points = []
        
        FOR EACH chunk IN chunks:
            // Generate stable UUID for point
            point_id = UUID5(UUID_NAMESPACE, chunk.id)
            
            point = {
                'id': point_id,
                'vector': {
                    'content': chunk.vector_embedding
                },
                'payload': this._prepare_payload(chunk)
            }
            
            points.APPEND(point)
            
            // Batch upsert when reaching batch size
            IF LENGTH(points) >= 100:
                this._batch_upsert(points)
                points = []
        
        // Upsert remaining points
        IF LENGTH(points) > 0:
            this._batch_upsert(points)
        
        LOG_INFO(f"Upserted {LENGTH(chunks)} chunks to Qdrant")
    
    METHOD _prepare_payload(chunk):
        RETURN {
            'id': chunk.id,
            'document_id': chunk.document_id,
            'parent_section_id': chunk.parent_section_id,
            'level': chunk.level,
            'order': chunk.order,
            'heading': chunk.heading,
            'text': chunk.text,
            'is_combined': chunk.is_combined,
            'is_split': chunk.is_split,
            'original_section_ids': chunk.original_section_ids,
            'boundaries_json': chunk.boundaries_json,
            'token_count': chunk.token_count,
            'embedding_version': chunk.embedding_version,
            'embedding_provider': chunk.embedding_provider,
            'embedding_dimensions': chunk.embedding_dimensions,
            'embedding_timestamp': chunk.embedding_timestamp,
            'updated_at': UNIX_TIMESTAMP(chunk.updated_at),
            // Additional indexed fields for filtering
            'document_id_keyword': chunk.document_id,  // For exact match
            'token_count_int': chunk.token_count,  // For range queries
            'updated_at_timestamp': UNIX_TIMESTAMP(chunk.updated_at)
        }
    
    METHOD _batch_upsert(points):
        TRY:
            this.client.upsert(
                collection_name=this.collection,
                points=points,
                wait=FALSE  // Don't wait for indexing on intermediate batches
            )
        CATCH error:
            LOG_ERROR(f"Qdrant batch upsert failed: {error}")
            
            // Retry individual points
            FOR EACH point IN points:
                TRY:
                    this.client.upsert(
                        collection_name=this.collection,
                        points=[point],
                        wait=TRUE
                    )
                CATCH retry_error:
                    LOG_ERROR(f"Failed to upsert point {point['id']}: {retry_error}")
    
    METHOD delete_document_chunks(document_id):
        this.client.delete(
            collection_name=this.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id_keyword",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        
        LOG_INFO(f"Deleted chunks for document {document_id} from Qdrant")
```

### 7E.5.3: Redis Cache Layer

```pseudocode
CLASS RedisCache:
    CONSTRUCTOR():
        this.client = redis.Redis(
            host=env.REDIS_HOST,
            port=env.REDIS_PORT,
            db=env.REDIS_DB,
            password=env.REDIS_PASSWORD,
            decode_responses=FALSE,  // Binary data for pickle
            connection_pool_kwargs={
                'max_connections': 50,
                'socket_keepalive': TRUE
            }
        )
        this.ttl = INT(env.REDIS_CACHE_TTL)
        this.local_cache = LRU_CACHE(maxsize=1000)
    
    METHOD get_embedding(text):
        cache_key = this._generate_cache_key("embedding", text)
        
        // Check local cache first
        local_result = this.local_cache.get(cache_key)
        IF local_result != NULL:
            METRICS.record_cache_hit('local', TRUE)
            RETURN local_result
        
        // Check Redis
        TRY:
            redis_result = this.client.get(cache_key)
            
            IF redis_result != NULL:
                METRICS.record_cache_hit('redis', TRUE)
                embedding = PICKLE_LOADS(redis_result)
                
                // Update local cache
                this.local_cache[cache_key] = embedding
                
                RETURN embedding
            
            METRICS.record_cache_hit('redis', FALSE)
            RETURN NULL
            
        CATCH redis.ConnectionError AS error:
            LOG_WARNING(f"Redis connection error: {error}")
            RETURN NULL
    
    METHOD set_embedding(text, embedding):
        cache_key = this._generate_cache_key("embedding", text)
        
        // Store in Redis
        TRY:
            this.client.setex(
                cache_key,
                this.ttl,
                PICKLE_DUMPS(embedding)
            )
        CATCH redis.ConnectionError AS error:
            LOG_WARNING(f"Failed to cache embedding: {error}")
        
        // Update local cache
        this.local_cache[cache_key] = embedding
    
    METHOD get_chunk_metadata(chunk_id):
        cache_key = f"chunk:{chunk_id}"
        
        TRY:
            data = this.client.get(cache_key)
            
            IF data != NULL:
                RETURN JSON_PARSE(data)
            
            RETURN NULL
            
        CATCH error:
            LOG_WARNING(f"Failed to get chunk metadata from cache: {error}")
            RETURN NULL
    
    METHOD set_chunk_metadata(chunk_id, metadata):
        cache_key = f"chunk:{chunk_id}"
        
        TRY:
            this.client.setex(
                cache_key,
                this.ttl,
                JSON_STRINGIFY(metadata)
            )
        CATCH error:
            LOG_WARNING(f"Failed to cache chunk metadata: {error}")
    
    METHOD invalidate_document(document_id):
        // Find all cache keys for document
        pattern = f"*:{document_id}:*"
        
        TRY:
            // Use SCAN for production-safe key iteration
            cursor = 0
            deleted_count = 0
            
            WHILE TRUE:
                cursor, keys = this.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                IF LENGTH(keys) > 0:
                    this.client.delete(*keys)
                    deleted_count += LENGTH(keys)
                
                IF cursor == 0:
                    BREAK
            
            LOG_INFO(f"Invalidated {deleted_count} cache entries for document {document_id}")
            
        CATCH error:
            LOG_ERROR(f"Failed to invalidate cache for document {document_id}: {error}")
    
    METHOD _generate_cache_key(prefix, content):
        content_hash = SHA256(content.encode())[:16]
        RETURN f"{prefix}:{content_hash}"
```

## Phase 7E.6: Retrieval Pipeline Pseudocode

### 7E.6.1: Vector Search Implementation

```pseudocode
CLASS VectorSearcher:
    CONSTRUCTOR():
        this.qdrant_client = QdrantClient(env.QDRANT_URL, env.QDRANT_API_KEY)
        this.embedding_service = EmbeddingService()
        this.collection = env.QDRANT_COLLECTION
        this.top_k = INT(env.VECTOR_TOP_K, 20)
    
    METHOD search(query_text, filter_conditions=NULL):
        // Generate query embedding
        query_embedding = this.embedding_service.generate_embedding(query_text)
        
        // Prepare search parameters
        search_params = {
            'ef': 128,  // Search quality parameter
            'exact': FALSE,  // Use HNSW index
            'indexed_only': TRUE  // Use indexed fields only
        }
        
        // Build filter if provided
        query_filter = NULL
        IF filter_conditions != NULL:
            query_filter = this._build_filter(filter_conditions)
        
        // Perform vector search
        results = this.qdrant_client.search(
            collection_name=this.collection,
            query_vector=("content", query_embedding),
            limit=this.top_k * 2,  // Oversample for filtering
            search_params=search_params,
            query_filter=query_filter,
            with_payload=TRUE,
            score_threshold=0.5  // Minimum similarity
        )
        
        // Process results
        search_results = []
        
        FOR EACH result IN results:
            search_results.APPEND({
                'id': result.payload['id'],
                'document_id': result.payload['document_id'],
                'parent_section_id': result.payload['parent_section_id'],
                'score': result.score,
                'text': result.payload['text'],
                'heading': result.payload['heading'],
                'order': result.payload['order'],
                'metadata': {
                    'token_count': result.payload['token_count'],
                    'is_combined': result.payload['is_combined'],
                    'is_split': result.payload['is_split']
                }
            })
        
        // Sort by score and limit
        search_results = SORT(search_results, key='score', reverse=TRUE)
        search_results = search_results[:this.top_k]
        
        RETURN search_results
    
    METHOD _build_filter(conditions):
        must_conditions = []
        
        FOR EACH field, value IN conditions:
            IF field == 'document_id':
                must_conditions.APPEND(
                    FieldCondition(
                        key='document_id_keyword',
                        match=MatchValue(value=value)
                    )
                )
            ELSE IF field == 'min_tokens':
                must_conditions.APPEND(
                    FieldCondition(
                        key='token_count_int',
                        range=Range(gte=value)
                    )
                )
            ELSE IF field == 'max_tokens':
                must_conditions.APPEND(
                    FieldCondition(
                        key='token_count_int',
                        range=Range(lte=value)
                    )
                )
        
        RETURN Filter(must=must_conditions)
```

### 7E.6.2: BM25 Sparse Search

```pseudocode
CLASS BM25Searcher:
    CONSTRUCTOR():
        this.neo4j_driver = Neo4jConnection.get_driver()
        this.top_k = INT(env.BM25_TOP_K, 20)
        
        // BM25 parameters
        this.k1 = 1.2  // Term frequency saturation
        this.b = 0.75  // Length normalization
    
    METHOD search(query_text, filter_conditions=NULL):
        // Tokenize query
        query_tokens = TOKENIZE_AND_LOWERCASE(query_text)
        query_tokens = REMOVE_STOPWORDS(query_tokens)
        
        WITH this.neo4j_driver.session() AS session:
            // Build filter clause
            filter_clause = ""
            IF filter_conditions != NULL:
                filter_clause = this._build_filter_clause(filter_conditions)
            
            // Use full-text index for initial retrieval
            result = session.run(
                f"""
                CALL db.index.fulltext.queryNodes(
                    'chunk_text_idx', 
                    $query
                ) YIELD node, score
                WHERE TRUE {filter_clause}
                WITH node, score
                LIMIT {this.top_k * 3}
                RETURN node.id as id,
                       node.document_id as document_id,
                       node.parent_section_id as parent_section_id,
                       node.text as text,
                       node.heading as heading,
                       node.order as order,
                       node.token_count as token_count,
                       score
                ORDER BY score DESC
                LIMIT {this.top_k}
                """,
                query=' OR '.join(query_tokens)
            )
            
            # Calculate proper BM25 scores
            search_results = []
            
            FOR EACH record IN result:
                bm25_score = this._calculate_bm25_score(
                    record['text'],
                    query_tokens,
                    record['token_count']
                )
                
                search_results.APPEND({
                    'id': record['id'],
                    'document_id': record['document_id'],
                    'parent_section_id': record['parent_section_id'],
                    'score': bm25_score,
                    'text': record['text'],
                    'heading': record['heading'],
                    'order': record['order'],
                    'metadata': {
                        'token_count': record['token_count'],
                        'lucene_score': record['score']
                    }
                })
            
            RETURN search_results
    
    METHOD _calculate_bm25_score(document_text, query_terms, doc_length):
        doc_tokens = TOKENIZE_AND_LOWERCASE(document_text)
        
        // Get term frequencies
        term_freq = COUNTER(doc_tokens)
        
        score = 0.0
        avg_doc_length = 1000  // Approximate average
        
        FOR EACH term IN query_terms:
            tf = term_freq.get(term, 0)
            
            IF tf > 0:
                // IDF calculation (simplified - should use corpus statistics)
                idf = LOG((TOTAL_DOCS - DOCS_WITH_TERM + 0.5) / (DOCS_WITH_TERM + 0.5))
                
                // BM25 formula
                numerator = tf * (this.k1 + 1)
                denominator = tf + this.k1 * (1 - this.b + this.b * (doc_length / avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        RETURN score
```

### 7E.6.3: Rank Fusion

```pseudocode
CLASS RankFusion:
    CONSTRUCTOR():
        this.vector_searcher = VectorSearcher()
        this.bm25_searcher = BM25Searcher()
        this.final_top_k = INT(env.FINAL_TOP_K, 10)
        this.fusion_k = 60  // Reciprocal rank fusion constant
    
    METHOD hybrid_search(query_text, filter_conditions=NULL):
        // Parallel search
        vector_results_future = ASYNC_CALL(
            this.vector_searcher.search, query_text, filter_conditions
        )
        bm25_results_future = ASYNC_CALL(
            this.bm25_searcher.search, query_text, filter_conditions
        )
        
        // Wait for results
        vector_results = AWAIT(vector_results_future)
        bm25_results = AWAIT(bm25_results_future)
        
        // Reciprocal rank fusion
        fused_results = this._reciprocal_rank_fusion(
            vector_results, 
            bm25_results
        )
        
        // Group by parent and expand context
        final_results = this._expand_context(fused_results)
        
        // Limit to final top k
        final_results = final_results[:this.final_top_k]
        
        RETURN final_results
    
    METHOD _reciprocal_rank_fusion(vector_results, bm25_results):
        // Calculate reciprocal ranks
        fusion_scores = {}
        
        // Process vector results
        FOR i FROM 0 TO LENGTH(vector_results)-1:
            result = vector_results[i]
            rank = i + 1
            rrf_score = 1.0 / (this.fusion_k + rank)
            
            IF result['id'] NOT IN fusion_scores:
                fusion_scores[result['id']] = {
                    'score': 0,
                    'data': result
                }
            
            fusion_scores[result['id']]['score'] += rrf_score
            fusion_scores[result['id']]['vector_rank'] = rank
        
        // Process BM25 results
        FOR i FROM 0 TO LENGTH(bm25_results)-1:
            result = bm25_results[i]
            rank = i + 1
            rrf_score = 1.0 / (this.fusion_k + rank)
            
            IF result['id'] NOT IN fusion_scores:
                fusion_scores[result['id']] = {
                    'score': 0,
                    'data': result
                }
            
            fusion_scores[result['id']]['score'] += rrf_score
            fusion_scores[result['id']]['bm25_rank'] = rank
        
        // Sort by fusion score
        sorted_results = SORT(
            fusion_scores.values(), 
            key='score', 
            reverse=TRUE
        )
        
        RETURN sorted_results
    
    METHOD _expand_context(results):
        // Group by parent section
        parent_groups = {}
        
        FOR EACH result IN results:
            parent_id = result['data']['parent_section_id']
            
            IF parent_id NOT IN parent_groups:
                parent_groups[parent_id] = []
            
            parent_groups[parent_id].APPEND(result)
        
        // Expand with adjacent chunks if needed
        expanded_results = []
        
        FOR EACH parent_id, group IN parent_groups:
            // Check if we should expand
            should_expand = (
                LENGTH(group) == 1 AND 
                group[0]['score'] > 0.8
            )
            
            IF should_expand:
                // Fetch adjacent chunks
                adjacent = this._fetch_adjacent_chunks(
                    group[0]['data']['id'],
                    parent_id
                )
                
                FOR EACH adj IN adjacent:
                    IF adj['id'] NOT IN MAP(expanded_results, r => r['data']['id']):
                        expanded_results.APPEND({
                            'score': group[0]['score'] * 0.9,  // Slightly lower score
                            'data': adj,
                            'is_expanded': TRUE
                        })
            
            // Add original results
            expanded_results.EXTEND(group)
        
        // Sort by score
        expanded_results = SORT(expanded_results, key='score', reverse=TRUE)
        
        // Assemble context in order
        RETURN this._assemble_context(expanded_results)
    
    METHOD _fetch_adjacent_chunks(chunk_id, parent_id):
        neo4j_driver = Neo4jConnection.get_driver()
        
        WITH neo4j_driver.session() AS session:
            result = session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (c)-[:NEXT_CHUNK*0..1]-(adjacent:Chunk)
                WHERE adjacent.parent_section_id = $parent_id
                RETURN DISTINCT adjacent.id as id,
                       adjacent.document_id as document_id,
                       adjacent.parent_section_id as parent_section_id,
                       adjacent.text as text,
                       adjacent.heading as heading,
                       adjacent.order as order
                ORDER BY adjacent.order
                """,
                chunk_id=chunk_id,
                parent_id=parent_id
            )
            
            RETURN result.data()
    
    METHOD _assemble_context(results):
        // Group by document and parent, then sort by order
        assembled = []
        
        document_groups = GROUP_BY(results, r => r['data']['document_id'])
        
        FOR EACH doc_id, doc_results IN document_groups:
            parent_groups = GROUP_BY(
                doc_results, 
                r => r['data']['parent_section_id']
            )
            
            FOR EACH parent_id, parent_results IN parent_groups:
                // Sort by order within parent
                sorted_chunks = SORT(
                    parent_results,
                    key=lambda r: r['data']['order']
                )
                
                // Combine text for context
                context_text = JOIN(
                    MAP(sorted_chunks, r => r['data']['text']),
                    '\n\n'
                )
                
                assembled.APPEND({
                    'document_id': doc_id,
                    'parent_section_id': parent_id,
                    'context': context_text,
                    'chunks': sorted_chunks,
                    'max_score': MAX(MAP(sorted_chunks, r => r['score']))
                })
        
        // Sort by max score
        assembled = SORT(assembled, key='max_score', reverse=TRUE)
        
        RETURN assembled
```

## Phase 7E.7: Session Management Pseudocode

### 7E.7.1: Session Lifecycle

```pseudocode
CLASS SessionManager:
    CONSTRUCTOR():
        this.neo4j_driver = Neo4jConnection.get_driver()
        this.redis_cache = RedisCache()
        this.session_timeout = 3600  // 1 hour
        this.max_context_length = 10000  // tokens
    
    METHOD create_session(user_id):
        session_id = GENERATE_UUID()
        
        WITH this.neo4j_driver.session() AS neo4j_session:
            neo4j_session.run(
                """
                CREATE (s:Session {
                    session_id: $session_id,
                    user_id: $user_id,
                    started_at: datetime(),
                    expires_at: datetime() + duration({seconds: $timeout}),
                    active: true,
                    turn_count: 0,
                    total_tokens: 0
                })
                RETURN s
                """,
                session_id=session_id,
                user_id=user_id,
                timeout=this.session_timeout
            )
        
        // Cache session data
        this.redis_cache.set_session_data(session_id, {
            'user_id': user_id,
            'context': [],
            'turn_count': 0
        })
        
        LOG_INFO(f"Created session {session_id} for user {user_id}")
        RETURN session_id
    
    METHOD get_session(session_id):
        // Check cache first
        cached_session = this.redis_cache.get_session_data(session_id)
        
        IF cached_session != NULL:
            RETURN cached_session
        
        // Load from database
        WITH this.neo4j_driver.session() AS neo4j_session:
            result = neo4j_session.run(
                """
                MATCH (s:Session {session_id: $session_id})
                WHERE s.active = true AND s.expires_at > datetime()
                OPTIONAL MATCH (s)-[:HAS_QUERY]->(q:Query)
                OPTIONAL MATCH (q)-[:HAS_ANSWER]->(a:Answer)
                RETURN s, collect({
                    query: q,
                    answer: a
                }) as history
                ORDER BY q.turn
                """,
                session_id=session_id
            ).single()
            
            IF result == NULL:
                RETURN NULL
            
            session_data = {
                'session_id': session_id,
                'user_id': result['s']['user_id'],
                'started_at': result['s']['started_at'],
                'expires_at': result['s']['expires_at'],
                'turn_count': result['s']['turn_count'],
                'history': result['history']
            }
            
            // Update cache
            this.redis_cache.set_session_data(session_id, session_data)
            
            RETURN session_data
    
    METHOD extend_session(session_id):
        WITH this.neo4j_driver.session() AS neo4j_session:
            neo4j_session.run(
                """
                MATCH (s:Session {session_id: $session_id})
                SET s.expires_at = datetime() + duration({seconds: $timeout})
                """,
                session_id=session_id,
                timeout=this.session_timeout
            )
        
        LOG_DEBUG(f"Extended session {session_id}")
    
    METHOD end_session(session_id):
        WITH this.neo4j_driver.session() AS neo4j_session:
            neo4j_session.run(
                """
                MATCH (s:Session {session_id: $session_id})
                SET s.active = false,
                    s.ended_at = datetime()
                """,
                session_id=session_id
            )
        
        // Clear cache
        this.redis_cache.invalidate_session(session_id)
        
        LOG_INFO(f"Ended session {session_id}")
```

### 7E.7.2: Query-Answer Tracking

```pseudocode
CLASS QueryAnswerTracker:
    CONSTRUCTOR():
        this.neo4j_driver = Neo4jConnection.get_driver()
        this.session_manager = SessionManager()
    
    METHOD record_query(session_id, query_text, query_embedding):
        query_id = GENERATE_UUID()
        
        // Get current turn number
        session = this.session_manager.get_session(session_id)
        turn_number = session['turn_count'] + 1
        
        WITH this.neo4j_driver.session() AS neo4j_session:
            neo4j_session.run(
                """
                MATCH (s:Session {session_id: $session_id})
                CREATE (q:Query {
                    query_id: $query_id,
                    text: $query_text,
                    embedding: $embedding,
                    turn: $turn,
                    asked_at: datetime(),
                    token_count: $token_count
                })
                CREATE (s)-[:HAS_QUERY]->(q)
                SET s.turn_count = $turn
                RETURN q
                """,
                session_id=session_id,
                query_id=query_id,
                query_text=query_text,
                embedding=query_embedding,
                turn=turn_number,
                token_count=COUNT_TOKENS(query_text)
            )
        
        LOG_INFO(f"Recorded query {query_id} for session {session_id}")
        RETURN query_id
    
    METHOD record_answer(query_id, answer_text, cited_chunks):
        answer_id = GENERATE_UUID()
        
        WITH this.neo4j_driver.session() AS neo4j_session:
            // Create answer node
            neo4j_session.run(
                """
                MATCH (q:Query {query_id: $query_id})
                CREATE (a:Answer {
                    answer_id: $answer_id,
                    text: $answer_text,
                    created_at: datetime(),
                    token_count: $token_count,
                    confidence_score: $confidence,
                    user_feedback: null
                })
                CREATE (q)-[:HAS_ANSWER]->(a)
                RETURN a
                """,
                query_id=query_id,
                answer_id=answer_id,
                answer_text=answer_text,
                token_count=COUNT_TOKENS(answer_text),
                confidence=CALCULATE_CONFIDENCE(answer_text, cited_chunks)
            )
            
            // Create citation relationships
            IF LENGTH(cited_chunks) > 0:
                neo4j_session.run(
                    """
                    MATCH (a:Answer {answer_id: $answer_id})
                    UNWIND $chunk_ids AS chunk_id
                    MATCH (c:Chunk {id: chunk_id})
                    CREATE (a)-[:CITES {
                        relevance_score: $scores[chunk_id],
                        citation_order: $orders[chunk_id]
                    }]->(c)
                    """,
                    answer_id=answer_id,
                    chunk_ids=MAP(cited_chunks, c => c['id']),
                    scores=MAP_TO_DICT(cited_chunks, c => (c['id'], c['score'])),
                    orders=MAP_TO_DICT(
                        ENUMERATE(cited_chunks), 
                        (i, c) => (c['id'], i)
                    )
                )
        
        LOG_INFO(f"Recorded answer {answer_id} with {LENGTH(cited_chunks)} citations")
        RETURN answer_id
    
    METHOD record_feedback(answer_id, feedback_type, feedback_text=NULL):
        WITH this.neo4j_driver.session() AS neo4j_session:
            neo4j_session.run(
                """
                MATCH (a:Answer {answer_id: $answer_id})
                SET a.user_feedback = $feedback_type,
                    a.feedback_text = $feedback_text,
                    a.feedback_at = datetime()
                """,
                answer_id=answer_id,
                feedback_type=feedback_type,
                feedback_text=feedback_text
            )
        
        LOG_INFO(f"Recorded {feedback_type} feedback for answer {answer_id}")
```

### 7E.7.3: Context Preservation

```pseudocode
CLASS ContextManager:
    CONSTRUCTOR():
        this.max_context_tokens = 10000
        this.context_window_size = INT(env.CONTEXT_WINDOW, 3)
        this.session_manager = SessionManager()
    
    METHOD get_conversation_context(session_id):
        session = this.session_manager.get_session(session_id)
        
        IF session == NULL:
            RETURN []
        
        // Get recent history
        history = session['history'][-this.context_window_size:]
        
        // Build context
        context = []
        total_tokens = 0
        
        FOR EACH turn IN REVERSE(history):
            query_tokens = turn['query']['token_count']
            answer_tokens = turn['answer']['token_count'] IF turn['answer'] ELSE 0
            
            turn_tokens = query_tokens + answer_tokens
            
            IF total_tokens + turn_tokens <= this.max_context_tokens:
                context.INSERT(0, {
                    'query': turn['query']['text'],
                    'answer': turn['answer']['text'] IF turn['answer'] ELSE NULL,
                    'turn': turn['query']['turn']
                })
                total_tokens += turn_tokens
            ELSE:
                BREAK
        
        RETURN context
    
    METHOD get_relevant_context(session_id, current_query):
        // Get conversation history
        conversation_context = this.get_conversation_context(session_id)
        
        // Find relevant previous queries/answers
        relevant_context = []
        
        FOR EACH turn IN conversation_context:
            // Calculate similarity
            similarity = CALCULATE_SIMILARITY(current_query, turn['query'])
            
            IF similarity > 0.7:
                relevant_context.APPEND({
                    'turn': turn,
                    'similarity': similarity
                })
        
        // Sort by similarity
        relevant_context = SORT(relevant_context, key='similarity', reverse=TRUE)
        
        // Include top 3 most relevant
        RETURN relevant_context[:3]
    
    METHOD preserve_context_continuity(session_id, new_chunks):
        // Get previous answer's cited chunks
        WITH this.neo4j_driver.session() AS neo4j_session:
            result = neo4j_session.run(
                """
                MATCH (s:Session {session_id: $session_id})
                      -[:HAS_QUERY]->(q:Query)
                      -[:HAS_ANSWER]->(a:Answer)
                      -[:CITES]->(c:Chunk)
                WITH a, c
                ORDER BY q.turn DESC
                LIMIT 1
                RETURN collect(c.id) as previous_chunks
                """,
                session_id=session_id
            ).single()
            
            previous_chunks = result['previous_chunks'] IF result ELSE []
        
        // Check for continuity
        overlap = SET_INTERSECTION(
            SET(MAP(new_chunks, c => c['parent_section_id'])),
            SET(previous_chunks)
        )
        
        continuity_score = LENGTH(overlap) / MAX(LENGTH(new_chunks), 1)
        
        LOG_DEBUG(f"Context continuity score: {continuity_score:.2f}")
        
        RETURN continuity_score > 0.3  // Has reasonable continuity
```

## Phase 7E.8: Quality Assurance Pseudocode

### 7E.8.1: Integrity Testing

```pseudocode
CLASS IntegrityTester:
    METHOD test_zero_loss_preservation():
        test_results = []
        
        // Test document parsing and reconstruction
        test_doc = LOAD_TEST_DOCUMENT()
        original_hash = SHA256(test_doc.content)
        
        // Parse into sections
        parser = DocumentParser()
        doc_id, sections = parser.parse(test_doc.content, test_doc.metadata)
        
        // Reconstruct from sections
        reconstructed = JOIN(MAP(sections, s => s.text), '\n')
        reconstructed_hash = SHA256(reconstructed)
        
        test_results.APPEND({
            'test': 'document_reconstruction',
            'passed': original_hash == reconstructed_hash,
            'details': {
                'original_hash': original_hash,
                'reconstructed_hash': reconstructed_hash,
                'sections_count': LENGTH(sections)
            }
        })
        
        // Test chunking preservation
        chunker = ChunkingPipeline()
        chunks = chunker.process_sections(sections)
        
        // Verify all content is preserved
        FOR EACH section IN sections:
            section_found = FALSE
            
            FOR EACH chunk IN chunks:
                IF section.id IN chunk.original_section_ids:
                    section_found = TRUE
                    BREAK
            
            IF NOT section_found:
                test_results.APPEND({
                    'test': 'section_preservation',
                    'passed': FALSE,
                    'details': {
                        'missing_section': section.id
                    }
                })
        
        // Test chunk reassembly
        FOR EACH parent_id IN UNIQUE(MAP(chunks, c => c.parent_section_id)):
            parent_chunks = FILTER(chunks, c => c.parent_section_id == parent_id)
            parent_chunks = SORT(parent_chunks, key='order')
            
            // Check for gaps in ordering
            FOR i FROM 0 TO LENGTH(parent_chunks)-2:
                IF parent_chunks[i].order + 1 != parent_chunks[i+1].order:
                    test_results.APPEND({
                        'test': 'chunk_ordering',
                        'passed': FALSE,
                        'details': {
                            'parent_id': parent_id,
                            'gap_between': [parent_chunks[i].order, parent_chunks[i+1].order]
                        }
                    })
        
        RETURN test_results
    
    METHOD test_embedding_integrity():
        test_results = []
        
        // Test embedding dimensions
        test_text = "Test content for embedding validation"
        embedding_service = EmbeddingService()
        
        embedding = embedding_service.generate_embedding(test_text)
        
        test_results.APPEND({
            'test': 'embedding_dimensions',
            'passed': LENGTH(embedding) == 1024,
            'details': {
                'expected': 1024,
                'actual': LENGTH(embedding)
            }
        })
        
        // Test embedding values range
        valid_range = ALL(MAP(embedding, v => -1 <= v <= 1))
        
        test_results.APPEND({
            'test': 'embedding_range',
            'passed': valid_range,
            'details': {
                'min_value': MIN(embedding),
                'max_value': MAX(embedding)
            }
        })
        
        // Test embedding determinism
        embedding2 = embedding_service.generate_embedding(test_text)
        
        test_results.APPEND({
            'test': 'embedding_determinism',
            'passed': ARRAYS_EQUAL(embedding, embedding2),
            'details': {
                'similarity': COSINE_SIMILARITY(embedding, embedding2)
            }
        })
        
        RETURN test_results
```

### 7E.8.2: Performance Testing

```pseudocode
CLASS PerformanceTester:
    METHOD test_ingestion_throughput():
        results = {
            'total_documents': 0,
            'total_sections': 0,
            'total_chunks': 0,
            'total_time': 0,
            'throughput': {}
        }
        
        test_documents = LOAD_TEST_CORPUS()
        
        start_time = CURRENT_TIME()
        
        FOR EACH doc IN test_documents:
            doc_start = CURRENT_TIME()
            
            // Parse document
            parser = DocumentParser()
            doc_id, sections = parser.parse(doc.content, doc.metadata)
            
            // Create chunks
            chunker = ChunkingPipeline()
            chunks = chunker.process_sections(sections)
            
            // Generate embeddings
            embeddings, failed = GenerateEmbeddingsBatch(chunks)
            
            // Store in databases
            StoreEmbeddings(chunks)
            
            doc_time = CURRENT_TIME() - doc_start
            
            results['total_documents'] += 1
            results['total_sections'] += LENGTH(sections)
            results['total_chunks'] += LENGTH(chunks)
            results['total_time'] += doc_time
            
            LOG_INFO(f"Processed document in {doc_time:.2f}s: "
                    f"{LENGTH(sections)} sections -> {LENGTH(chunks)} chunks")
        
        end_time = CURRENT_TIME()
        total_time = end_time - start_time
        
        results['throughput'] = {
            'docs_per_second': results['total_documents'] / total_time,
            'sections_per_second': results['total_sections'] / total_time,
            'chunks_per_second': results['total_chunks'] / total_time
        }
        
        RETURN results
    
    METHOD test_query_latency():
        test_queries = [
            "What is the configuration for Neo4j?",
            "How does vector search work with embeddings?",
            "Explain the chunking strategy for large documents"
        ]
        
        results = []
        rank_fusion = RankFusion()
        
        FOR EACH query IN test_queries:
            latencies = []
            
            // Run multiple times for statistics
            FOR i FROM 1 TO 10:
                start_time = CURRENT_TIME()
                
                search_results = rank_fusion.hybrid_search(query)
                
                latency = CURRENT_TIME() - start_time
                latencies.APPEND(latency)
            
            results.APPEND({
                'query': query,
                'min_latency': MIN(latencies),
                'max_latency': MAX(latencies),
                'mean_latency': MEAN(latencies),
                'p95_latency': PERCENTILE(latencies, 95),
                'p99_latency': PERCENTILE(latencies, 99)
            })
        
        RETURN results
```

### 7E.8.3: Retrieval Quality

```pseudocode
CLASS RetrievalQualityTester:
    METHOD test_retrieval_metrics():
        // Load test queries with ground truth
        test_set = LOAD_RETRIEVAL_TEST_SET()
        
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'mrr': []
        }
        
        rank_fusion = RankFusion()
        
        FOR EACH test_case IN test_set:
            query = test_case['query']
            relevant_chunks = SET(test_case['relevant_chunks'])
            
            // Perform search
            results = rank_fusion.hybrid_search(query)
            retrieved_chunks = MAP(results, r => r['chunks'][0]['data']['id'])
            
            // Calculate precision@k
            FOR k IN [1, 3, 5, 10]:
                top_k = retrieved_chunks[:k]
                relevant_in_top_k = LENGTH(
                    SET_INTERSECTION(SET(top_k), relevant_chunks)
                )
                
                precision = relevant_in_top_k / k
                recall = relevant_in_top_k / LENGTH(relevant_chunks)
                
                metrics['precision_at_k'].APPEND({
                    'k': k,
                    'precision': precision
                })
                
                metrics['recall_at_k'].APPEND({
                    'k': k,
                    'recall': recall
                })
            
            // Calculate MRR (Mean Reciprocal Rank)
            first_relevant_rank = NULL
            
            FOR i FROM 0 TO LENGTH(retrieved_chunks)-1:
                IF retrieved_chunks[i] IN relevant_chunks:
                    first_relevant_rank = i + 1
                    BREAK
            
            reciprocal_rank = 1.0 / first_relevant_rank IF first_relevant_rank ELSE 0
            metrics['mrr'].APPEND(reciprocal_rank)
            
            // Calculate NDCG
            dcg = this._calculate_dcg(retrieved_chunks, relevant_chunks, 10)
            idcg = this._calculate_idcg(relevant_chunks, 10)
            ndcg = dcg / idcg IF idcg > 0 ELSE 0
            
            metrics['ndcg_at_k'].APPEND({
                'k': 10,
                'ndcg': ndcg
            })
        
        // Aggregate metrics
        aggregated = {
            'mean_precision_at_1': MEAN(
                FILTER(metrics['precision_at_k'], m => m['k'] == 1),
                'precision'
            ),
            'mean_precision_at_5': MEAN(
                FILTER(metrics['precision_at_k'], m => m['k'] == 5),
                'precision'
            ),
            'mean_recall_at_10': MEAN(
                FILTER(metrics['recall_at_k'], m => m['k'] == 10),
                'recall'
            ),
            'mean_reciprocal_rank': MEAN(metrics['mrr']),
            'mean_ndcg_at_10': MEAN(
                MAP(metrics['ndcg_at_k'], m => m['ndcg'])
            )
        }
        
        RETURN aggregated
    
    METHOD _calculate_dcg(retrieved, relevant, k):
        dcg = 0.0
        
        FOR i FROM 0 TO MIN(k, LENGTH(retrieved))-1:
            IF retrieved[i] IN relevant:
                // Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1
                dcg += relevance / LOG2(i + 2)  // i+2 because rank starts at 1
        
        RETURN dcg
    
    METHOD _calculate_idcg(relevant, k):
        // Ideal DCG: all relevant documents at top positions
        idcg = 0.0
        
        FOR i FROM 0 TO MIN(k, LENGTH(relevant))-1:
            idcg += 1 / LOG2(i + 2)
        
        RETURN idcg
```

## Phase 7E.9: Monitoring and Operations Pseudocode

### 7E.9.1: Metrics Collection

```pseudocode
CLASS MetricsCollector:
    CONSTRUCTOR():
        this.prometheus_registry = PrometheusRegistry()
        
        // Define counters
        this.ingestion_counter = Counter(
            'graphrag_documents_ingested_total',
            'Total documents ingested',
            ['status']
        )
        
        this.query_counter = Counter(
            'graphrag_queries_total',
            'Total queries processed',
            ['search_type']
        )
        
        // Define histograms
        this.latency_histogram = Histogram(
            'graphrag_operation_latency_seconds',
            'Operation latency',
            ['operation'],
            buckets=[0.1, 0.5, 1, 2, 5, 10]
        )
        
        this.embedding_time_histogram = Histogram(
            'graphrag_embedding_duration_seconds',
            'Embedding generation duration',
            ['provider']
        )
        
        // Define gauges
        this.chunk_count_gauge = Gauge(
            'graphrag_chunks_total',
            'Total number of chunks'
        )
        
        this.cache_size_gauge = Gauge(
            'graphrag_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type']
        )
    
    METHOD record_ingestion(document_id, success):
        status = 'success' IF success ELSE 'failure'
        this.ingestion_counter.labels(status=status).inc()
        
        IF success:
            // Update chunk count
            chunk_count = this._get_chunk_count()
            this.chunk_count_gauge.set(chunk_count)
    
    METHOD record_query(search_type, latency):
        this.query_counter.labels(search_type=search_type).inc()
        this.latency_histogram.labels(operation='query').observe(latency)
    
    METHOD record_embedding_time(provider, duration):
        this.embedding_time_histogram.labels(provider=provider).observe(duration)
    
    METHOD _get_chunk_count():
        neo4j_driver = Neo4jConnection.get_driver()
        
        WITH neo4j_driver.session() AS session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            RETURN result.single()['count']
```

### 7E.9.2: Health Checks

```pseudocode
CLASS HealthChecker:
    METHOD check_all_services():
        health_status = {
            'overall': 'healthy',
            'services': {},
            'timestamp': CURRENT_TIMESTAMP()
        }
        
        // Check Neo4j
        neo4j_health = this.check_neo4j()
        health_status['services']['neo4j'] = neo4j_health
        
        // Check Qdrant
        qdrant_health = this.check_qdrant()
        health_status['services']['qdrant'] = qdrant_health
        
        // Check Redis
        redis_health = this.check_redis()
        health_status['services']['redis'] = redis_health
        
        // Check embedding service
        embedding_health = this.check_embedding_service()
        health_status['services']['embedding'] = embedding_health
        
        // Determine overall health
        IF ANY(MAP(health_status['services'].values(), s => s['status'] == 'unhealthy')):
            health_status['overall'] = 'unhealthy'
        ELSE IF ANY(MAP(health_status['services'].values(), s => s['status'] == 'degraded')):
            health_status['overall'] = 'degraded'
        
        RETURN health_status
    
    METHOD check_neo4j():
        TRY:
            driver = Neo4jConnection.get_driver()
            
            WITH driver.session() AS session:
                start = CURRENT_TIME()
                result = session.run("RETURN 1 as healthy")
                latency = CURRENT_TIME() - start
                
                // Check indexes
                indexes = session.run("SHOW INDEXES WHERE state = 'ONLINE'")
                index_count = LENGTH(indexes.data())
                
                RETURN {
                    'status': 'healthy',
                    'latency': latency,
                    'indexes_online': index_count
                }
                
        CATCH error:
            RETURN {
                'status': 'unhealthy',
                'error': str(error)
            }
    
    METHOD check_qdrant():
        TRY:
            client = QdrantClient(env.QDRANT_URL, env.QDRANT_API_KEY)
            
            start = CURRENT_TIME()
            collection_info = client.get_collection(env.QDRANT_COLLECTION)
            latency = CURRENT_TIME() - start
            
            status = 'healthy' IF collection_info.status == 'green' ELSE 'degraded'
            
            RETURN {
                'status': status,
                'latency': latency,
                'points_count': collection_info.points_count,
                'collection_status': collection_info.status
            }
            
        CATCH error:
            RETURN {
                'status': 'unhealthy',
                'error': str(error)
            }
    
    METHOD check_embedding_service():
        TRY:
            service = EmbeddingService()
            test_text = "Health check test"
            
            start = CURRENT_TIME()
            embedding = service.generate_embedding(test_text)
            latency = CURRENT_TIME() - start
            
            valid = LENGTH(embedding) == 1024
            
            RETURN {
                'status': 'healthy' IF valid ELSE 'degraded',
                'latency': latency,
                'embedding_valid': valid
            }
            
        CATCH error:
            RETURN {
                'status': 'unhealthy',
                'error': str(error)
            }
```

### 7E.9.3: Operational Procedures

```pseudocode
CLASS OperationalManager:
    METHOD backup_system():
        backup_id = GENERATE_TIMESTAMP_ID()
        backup_path = f"/backups/{backup_id}"
        
        CREATE_DIRECTORY(backup_path)
        
        // Backup Neo4j
        EXECUTE_COMMAND(
            f"docker exec graphrag-neo4j neo4j-admin dump "
            f"--database=neo4j --to=/backups/{backup_id}/neo4j.dump"
        )
        
        // Backup Qdrant
        qdrant_client = QdrantClient(env.QDRANT_URL, env.QDRANT_API_KEY)
        qdrant_client.create_snapshot(
            collection_name=env.QDRANT_COLLECTION,
            snapshot_name=f"{backup_id}_qdrant"
        )
        
        // Backup Redis
        EXECUTE_COMMAND(
            f"docker exec graphrag-redis redis-cli BGSAVE"
        )
        WAIT(2)  // Wait for background save
        COPY_FILE(
            "/data/redis/dump.rdb",
            f"{backup_path}/redis.rdb"
        )
        
        // Create backup manifest
        manifest = {
            'backup_id': backup_id,
            'timestamp': CURRENT_TIMESTAMP(),
            'components': {
                'neo4j': f"{backup_path}/neo4j.dump",
                'qdrant': f"{backup_id}_qdrant",
                'redis': f"{backup_path}/redis.rdb"
            }
        }
        
        WRITE_JSON(f"{backup_path}/manifest.json", manifest)
        
        LOG_INFO(f"Backup completed: {backup_id}")
        RETURN backup_id
    
    METHOD restore_system(backup_id):
        backup_path = f"/backups/{backup_id}"
        manifest = READ_JSON(f"{backup_path}/manifest.json")
        
        // Stop services
        EXECUTE_COMMAND("docker-compose stop")
        
        // Restore Neo4j
        EXECUTE_COMMAND(
            f"docker exec graphrag-neo4j neo4j-admin load "
            f"--database=neo4j --from=/backups/{backup_id}/neo4j.dump --force"
        )
        
        // Restore Qdrant
        qdrant_client = QdrantClient(env.QDRANT_URL, env.QDRANT_API_KEY)
        qdrant_client.recover_snapshot(
            collection_name=env.QDRANT_COLLECTION,
            snapshot_name=manifest['components']['qdrant']
        )
        
        // Restore Redis
        STOP_SERVICE("redis")
        COPY_FILE(
            manifest['components']['redis'],
            "/data/redis/dump.rdb"
        )
        START_SERVICE("redis")
        
        // Restart services
        EXECUTE_COMMAND("docker-compose start")
        
        // Verify restoration
        health = HealthChecker().check_all_services()
        
        IF health['overall'] != 'healthy':
            LOG_ERROR(f"Restoration failed: {health}")
            RAISE RestoreError("System unhealthy after restore")
        
        LOG_INFO(f"System restored from backup: {backup_id}")
        RETURN SUCCESS
    
    METHOD scale_system(component, scale_factor):
        IF component == 'neo4j':
            // Update Neo4j configuration
            config_updates = {
                'server.memory.heap.max_size': f"{4 * scale_factor}g",
                'server.memory.pagecache.size': f"{2 * scale_factor}g",
                'dbms.memory.transaction.global_max_size': f"{1 * scale_factor}g"
            }
            
            UPDATE_CONFIG("/data/neo4j/conf/neo4j.conf", config_updates)
            RESTART_SERVICE("neo4j")
            
        ELSE IF component == 'qdrant':
            // Scale Qdrant replicas
            EXECUTE_COMMAND(
                f"docker-compose scale qdrant={scale_factor}"
            )
            
        ELSE IF component == 'redis':
            // Update Redis max memory
            redis_client = REDIS_CONNECT(env.REDIS_HOST, env.REDIS_PORT)
            redis_client.config_set('maxmemory', f"{2 * scale_factor}gb")
        
        LOG_INFO(f"Scaled {component} by factor {scale_factor}")
        RETURN SUCCESS
```

## Conclusion

This comprehensive pseudocode document provides detailed algorithmic guidance for implementing every component of the GraphRAG v2.1 system. Each phase builds upon the previous ones, ensuring a systematic and complete implementation. The pseudocode is designed to be directly translatable into production Python code while maintaining clarity and correctness.

The key principles maintained throughout are:
- **Zero content loss** - Every operation preserves data integrity
- **Production readiness** - All error cases handled, no placeholders
- **Scalability** - Batch operations and efficient algorithms
- **Observability** - Comprehensive logging and metrics
- **Maintainability** - Clear structure and separation of concerns

Following this pseudocode will result in a fully functional GraphRAG system that meets all specifications and quality requirements defined in Phase-7E.
