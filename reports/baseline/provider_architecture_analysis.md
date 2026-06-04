# Provider Architecture Analysis
## wekadocs-matrix - Baseline Assessment

---

## 1. Provider Factory Design Summary

### Registration Pattern

The `ProviderFactory` (in `src/providers/factory.py`) uses a **class-level registry** pattern:

```python
 EMBEDDING_PROVIDER_CREATORS = {
    "jina-ai": ProviderFactory._create_jina_embedding_provider,
    "sentence-transformers": ProviderFactory._create_sentence_transformers_provider,
    "embedding-service": ProviderFactory._create_embedding_service_provider,
    "voyage-ai": ProviderFactory._create_voyage_provider,
    "snowflake-arctic-service": ProviderFactory._create_snowflake_arctic_provider,
    "qwen3-triton-service": ProviderFactory._create_qwen3_triton_provider,
}
```

**Key Design Features:**

1. **Alias Resolution**: The factory maintains `_EMBEDDING_PROVIDER_ALIASES` dict that maps 20+ legacy names to canonical providers:
   - `"bge-m3"`, `"bge_m3"`, `"bge-m3-service"` → `"embedding-service"`
   - `"st_minilm"`, `"huggingface"`, `"hf"` → `"sentence-transformers"`
   - `"snowflake-arctic"`, `"arctic"` → `"snowflake-arctic-service"`
   - `"qwen3-triton"`, `"qwen3_4b"` → `"qwen3-triton-service"`

2. **Profile-Driven Instantiation**: The factory uses `EmbeddingSettings` dataclass from profiles:
   ```python
   def create_embedding_provider_for_role(cls, role_plan) -> EmbeddingProvider:
       settings = cls._build_settings_from_profile(
           role_plan.profile_name, role_plan.profile
       )
       return cls.create_embedding_provider(settings=settings)
   ```

3. **Environment Variable Integration**: Service URLs resolved from:
   - `EMBEDDING_BASE_URL` (unified gateway)
   - `BGE_M3_API_URL` (legacy)
   - `CHONKIE_EMBEDDINGS_BASE_URL` (Snowflake Arctic)
   - `QWEN3_EMBED_URL` (Qwen3 Triton)
   - `JINA_API_KEY`, `VOYAGE_API_KEY` (cloud providers)

4. **Lazy Imports**: Creator methods use `from src.providers.embeddings.X import Y` to avoid loading unused providers.

**Rerank Provider Registration**: Uses inline conditional logic (not registry pattern):
```python
if provider == "jina-ai": ...
elif provider == "local-reranker-service": ...
elif provider in {"noop", "none", "disabled"}: ...
```

---

## 2. Provider Abstraction Pattern

### EmbeddingProvider Protocol (`src/providers/embeddings/base.py`)

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dims(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...

    def validate_dimensions(self, expected_dims: int) -> bool:
        return self.dims == expected_dims
```

**Additional Methods in Full Implementations:**
- `embed_sparse(texts) → List[dict]` - Sparse vector embeddings (BGE-M3 only)
- `embed_colbert(texts) → List[List[List[float]]]` - ColBERT multi-vectors (BGE-M3 only)
- `embed_documents_all(texts) → List[DocumentEmbeddingBundle]` - Dense + sparse + multivector
- `embed_query_all(text) → QueryEmbeddingBundle` - Full query embedding bundle
- `close()` - Resource cleanup

**Return Contract:**
- All providers return `List[List[float]]` (JSON-serializable, no numpy arrays)
- Dimension validation ensures consistency across ingestion and query

### RerankProvider Protocol (`src/providers/rerank/base.py`)

```python
@runtime_checkable
class RerankProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10,
               *, instruction: Optional[str] = None) -> List[Dict]: ...

    def health_check(self) -> bool: ...
```

**Enables Clean Swapping:** The Protocol pattern allows:
- Type checking without inheritance
- Runtime `isinstance()` checks via `@runtime_checkable`
- Mock providers for testing
- Multiple implementations coexisting

---

## 3. Chonkie Adapter Pattern

### What Chonkie Adapters Do

Chonkie is a semantic chunking library that requires an embedding interface to detect semantic boundaries. The adapters bridge the gap between:
- **Chonkie's `BaseEmbeddings` interface** (expects `embed(text) → np.ndarray`, `embed_batch(texts) → List[np.ndarray]`)
- **wekadocs-matrix embedding services** (HTTP APIs returning JSON with embeddings)

**All Three Adapters Implement:**
```python
class XChonkieAdapter(BaseEmbeddings):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]: ...
    def count_tokens(self, text: str) -> int: ...
    def get_tokenizer(self) -> Any: ...
    @classmethod
    def is_available(cls) -> bool: ...  # Health check
```

### Why There Are 3 Adapters

**Adapter Selection Logic** (from `semantic_chunker.py:236-314`):

```python
adapter_name = (self.config.embedding_adapter or "bge_m3").lower()

if adapter_name in {"bge_m3", "embedding-service", "qwen3_0_6b", ...}:
    adapter = BgeM3ChonkieAdapter(...)  # Unified gateway

elif adapter_name in {"snowflake_arctic", "snowflake_arctic_v2l", ...}:
    adapter = ArcticChonkieAdapter()  # Local Arctic service

elif adapter_name in {"qwen3_4b", "qwen3_triton", ...}:
    adapter = Qwen3ChonkieAdapter()  # Triton gateway at :8101
```

**Rationale:**

1. **BgeM3ChonkieAdapter** (`src/providers/embeddings/chonkie_adapter.py`):
   - **Target**: Unified embedding service at `EMBEDDING_BASE_URL` or `BGE_M3_API_URL`
   - **Use case**: Default adapter for BGE-M3 and Qwen3-0.6B via gateway
   - **Evidence**: Called by `semantic_chunker.py:248`, supports both `bge_m3` and `embedding-service` adapter names
   - **Special features**: Handles OversizeEmbeddingInputError, token counting, batch splitting

2. **ArcticChonkieAdapter** (`src/providers/embeddings/arctic_chonkie_adapter.py`):
   - **Target**: Local Snowflake Arctic service at `CHONKIE_EMBEDDINGS_BASE_URL` (default `:9010`)
   - **Use case**: Semantic chunking when using `snowflake_arctic` profile
   - **Evidence**: Called by `semantic_chunker.py:286`, health check at `/healthz`
   - **Client**: Uses `SnowflakeEmbeddingClient` (OpenAI-compatible API)

3. **Qwen3ChonkieAdapter** (`src/providers/embeddings/qwen3_chonkie_adapter.py`):
   - **Target**: Triton gateway at `QWEN3_EMBED_URL` (default `:8101`)
   - **Use case**: Semantic chunking with Qwen3-Embedding-4B via FastAPI gateway
   - **Evidence**: Called by `semantic_chunker.py:308`, health check via embed request
   - **Client**: Uses `Qwen3EmbeddingClient`

**Why Not One Adapter?**
- **Different service protocols**: BGE-M3 uses `/embed_dense`, Arctic uses `/v1/embeddings`, Qwen3 uses `/embed`
- **Different authentication**: API keys, bearer tokens, or none
- **Different health checks**: `/healthz` vs embed probe vs TCP check
- **Different error handling**: Retry logic, batch splitting, timeout strategies

**Design Intent**: **Modular experimentation** - each adapter represents a different embedding strategy that can be tested independently.

---

## 4. Provider Status Assessment

### Embedding Providers (6 total)

| Provider | Status | Evidence | Notes |
|----------|--------|----------|-------|
| **EmbeddingServiceProvider** | ✅ **ACTIVE** | Registered in factory, supports sparse/ColBERT, called by `embedding-service` profile | Primary provider for BGE-M3 gateway |
| **Qwen3TritonProvider** | ✅ **ACTIVE** | Registered in factory, marked `@status: ACTIVE`, `@called-by: factory.py` | Dense-only provider for Qwen3-4B Triton |
| **JinaEmbeddingProvider** | ✅ **ACTIVE** | Registered in factory, comprehensive implementation (752 LOC) | Cloud API with rate limiting, circuit breaker |
| **VoyageEmbeddingProvider** | ✅ **ACTIVE** | Registered in factory, supports contextual embeddings | Cloud API for Voyage AI |
| **SnowflakeArcticProvider** | ✅ **ACTIVE** | Registered in factory, full implementation (260 LOC) | Dense-only, OpenAI-compatible API |
| **SentenceTransformersProvider** | ⚠️ **LEGACY** | Registered but described as "Pre-Phase 7" provider | Local inference fallback, likely deprecated |

### Rerank Providers (3 total)

| Provider | Status | Evidence | Notes |
|----------|--------|----------|-------|
| **LocalRerankerServiceProvider** | ✅ **ACTIVE** | Default reranker in factory, requires `RERANKER_BASE_URL` | Qwen3-Reranker-4B via service |
| **JinaRerankProvider** | ✅ **ACTIVE** | Registered in factory, requires `JINA_API_KEY` | Cloud API reranker |
| **NoopReranker** | ✅ **ACTIVE** | Registered for `noop`/`disabled` | Pass-through for testing |

### Chonkie Adapters (3 total)

| Adapter | Status | Evidence | Notes |
|---------|--------|----------|-------|
| **BgeM3ChonkieAdapter** | ✅ **ACTIVE** | Called by `semantic_chunker.py:248`, default adapter | Gateway-based, supports BGE-M3 and Qwen3-0.6B |
| **ArcticChonkieAdapter** | ✅ **ACTIVE** | Called by `semantic_chunker.py:286` | For Snowflake Arctic profile |
| **Qwen3ChonkieAdapter** | ⚠️ **POTENTIALLY DEAD** | Session notes mention "dead Qwen3ChonkieAdapter (Triton port 8101)" | May be unreachable if Triton not deployed |

**Evidence for Qwen3ChonkieAdapter Status:**
From `docs/session-notes/2026-03-04-integration-deployment-retrieval-tuning.md:141`:
> `production.yaml` set `embedding_adapter: "qwen3_4b"` which routed to the dead `Qwen3ChonkieAdapter` (Triton port 8101). Added `"qwen3_0_6b"` as a recognized alias in `semantic_chunker.py:239` that routes through `BgeM3ChonkieAdapter` with the correct model name and service URL.

**Interpretation**: The Qwen3ChonkieAdapter is **code-functional** but **operationally dead** if the Triton gateway at `:8101` is not running. The code includes a health check (`is_available()`) that will disable it if unreachable.

---

## 5. Assessment: Modular Design vs Redundancy

### Verdict: **INTENTIONAL MODULAR DESIGN** (Keep All)

**Evidence:**

1. **Profile-Driven Selection**: The factory uses `EmbeddingSettings` profiles that specify:
   ```python
   profile: "qwen3_4b"
   provider: "qwen3-triton-service"
   model_id: "Qwen/Qwen3-Embedding-4B"
   dims: 1024
   ```
   This enables swapping providers via YAML config without code changes.

2. **Alias System for Backward Compatibility**: 20+ aliases ensure old configs still work:
   ```python
   "bge-m3" → "embedding-service"  # Legacy name still works
   "snowflake-arctic" → "snowflake-arctic-service"
   ```

3. **Distinct Use Cases**:
   - **EmbeddingServiceProvider**: Full-featured (dense + sparse + ColBERT) for hybrid search
   - **Qwen3TritonProvider**: Dense-only, optimized for Qwen3-4B
   - **SnowflakeArcticProvider**: Dense-only, local service
   - **JinaEmbeddingProvider**: Cloud API, task-specific embeddings
   - **VoyageEmbeddingProvider**: Contextual embeddings
   - **SentenceTransformersProvider**: Local fallback (legacy)

4. **Chonkie Adapter Separation**: Each adapter targets a **different service endpoint** with **different protocols**:
   - BgeM3ChonkieAdapter → Unified gateway (`/embed_dense`)
   - ArcticChonkieAdapter → Local Arctic (`/v1/embeddings`)
   - Qwen3ChonkieAdapter → Triton gateway (`/embed`)

5. **Health Checks Enable Graceful Degradation**:
   ```python
   if not Qwen3ChonkieAdapter.is_available():
       log.warning("Qwen3 embed gateway unavailable; semantic chunking disabled")
       return
   ```

6. **Documentation Confirms Intent**:
   - Factory docstring: "ENV-selectable embedding and rerank providers... docker-compose friendly configuration"
   - Provider modules: `@status: ACTIVE`, `@called-by: factory.py`

### Why Not Redundancy?

The providers are **not redundant** because:

1. **Different Backends**: Each targets a different service/API with unique capabilities
2. **Different Roles**: Some support sparse/ColBERT, others dense-only
3. **Different Deployment Scenarios**: Cloud (Jina, Voyage) vs local (Arctic, BGE-M3) vs hybrid (Qwen3 Triton)
4. **Profile-Based Selection**: Config-driven, not hardcoded

**The Only Dead Code:**
- **Qwen3ChonkieAdapter** may be operationally unreachable if Triton gateway is down, but the code itself is functional and includes health checks to gracefully disable.
- **SentenceTransformersProvider** is described as "Pre-Phase 7" but still registered and usable as a local fallback.

---

## 6. Specific Evidence

### Configuration Examples

**Profile Selection** (from `factory.py:573`):
```python
embedding_plan = get_embedding_plan()
embedding_provider = factory.create_embedding_provider_for_role(
    embedding_plan.dense
)
```

**Legacy Environment Override** (from `factory.py:95-98`):
```python
if any([provider, model, dims, task]):
    settings = cls._apply_legacy_overrides(
        settings, provider, model, dims, task
    )
```

**Service URL Resolution** (from `embedding_service.py:96-108`):
```python
resolved_url = (
    base_url
    or settings.service_url
    or os.getenv("EMBEDDING_BASE_URL")
    or os.getenv("BGE_M3_API_URL")  # legacy fallback
)
```

### Usage Patterns

**Chonkie Adapter Selection** (from `semantic_chunker.py:236-314`):
- `embedding_adapter: "bge_m3"` → BgeM3ChonkieAdapter
- `embedding_adapter: "snowflake_arctic"` → ArcticChonkieAdapter
- `embedding_adapter: "qwen3_4b"` → Qwen3ChonkieAdapter

**Sparse/ColBERT Support**:
- Only `EmbeddingServiceProvider` implements `embed_sparse()` and `embed_colbert()`
- Others raise `NotImplementedError` with guidance: "Use BGE-M3 for sparse embeddings"

### Documentation References

1. **Factory Docstring** (`factory.py:5-22`):
   > "Provider factory for ENV-selectable embedding and rerank providers. Phase 7C, Task 7C.1: Factory pattern for docker-compose friendly configuration."

2. **Provider Status Annotations**:
   - All providers marked `# @status: ACTIVE`
   - All list `# @called-by: factory.py`

3. **Session Notes** confirm operational history:
   - `2026-03-04-integration-deployment-retrieval-tuning.md`: Documents Qwen3ChonkieAdapter routing issue
   - `2026-03-03-signal-pool-provider-cleanup.md`: Notes "stale BGE-M3 references" are cosmetic only

---

## Conclusion

The provider architecture is a **well-designed modular system** that:

✅ **Enables experimentation**: Swap providers via profiles without code changes
✅ **Maintains backward compatibility**: 20+ aliases preserve old configs
✅ **Supports diverse deployments**: Cloud, local, hybrid scenarios
✅ **Gracefully degrades**: Health checks disable unreachable providers
✅ **Separates concerns**: Chonkie adapters isolated from main providers

**Recommendation**: **Do not delete any providers.** The architecture is intentionally modular for experimentation across multiple embedding strategies. The only operational concern is Qwen3ChonkieAdapter, which requires the Triton gateway at `:8101` to be running.
