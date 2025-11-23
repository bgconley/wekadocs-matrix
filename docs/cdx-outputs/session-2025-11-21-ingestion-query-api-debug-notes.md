# WekaDocs Matrix – Phase 7.0/7E Ingestion and Query API Debug Session

_Session date: 2025‑11‑20 → 2025‑11‑21_

This document captures a highly detailed, end‑to‑end record of the recent debugging and refactor
session around the WekaDocs Matrix ingestion and retrieval stack, with a strong focus on:

- Neo4j + Qdrant schema and namespace alignment
- Ingestion cleanup and safety ("only wipe ingested data")
- BGE‑M3 provider wiring and environment configuration
- Query‑time hybrid retrieval behavior, including Qdrant Query API usage for dense, sparse, and
  ColBERT (multi‑vector) embeddings
- Regression tests, smoke tests, container rebuilds, and verification steps

The goal is to give future maintainers an accurate, high‑fidelity picture of what changed, why it
changed, and how to verify that the system is operating correctly.

---

## 1. High‑Level Summary of Changes

At a high level, this session produced five major categories of changes:

1. **Neo4j / Qdrant cleanup script hardening**
   - `scripts/cleanup-databases.py` was refined to delete only ingestion‑generated data while
     preserving all metadata nodes (especially `SchemaVersion` v2.2), indexes, and collection
     schemas.

2. **Namespace suffix refactor for embedding profiles**
   - Introduced `get_expected_namespace_suffix` in `src/shared/config.py` as the singular source
     of truth for Qdrant/Neo4j namespace suffixes.
   - Wired both ingestion (`GraphBuilder`) and retrieval (`HybridRetriever`) guards to this helper
     to eliminate profile vs. version divergence.

3. **Ingestion pipeline robustness improvements**
   - Removed redundant `_enrich_chunk` calls from `_balance_small_tails` in
     `src/ingestion/chunk_assembler.py` to avoid double enrichment and duplicate error logging.
   - Hardened `parse_file_uri` in `src/ingestion/worker.py` to better map host paths to container
     paths, anchoring on the repo root or the last `/data/` segment.

4. **Qdrant Query API payload correctness for dense + sparse + ColBERT**
   - Fixed how `QdrantMultiVectorRetriever` builds Query API payloads, both for prefetch and main
     query, so that Qdrant's Pydantic models accept dense, sparse, and multivector representations
     without validation errors.
   - Added focused regression tests to lock in expected payload shapes.

5. **Verification tooling and smoke tests**
   - Added `scripts/smoke_test_query.py` to exercise `QueryService` and downstream hybrid
     retrieval inside the `mcp-server` container.
   - Adjusted the script to avoid relying on `model_dump_json` and instead use the stable
     `.json()` API or a simple print fallback.

Each of these categories is detailed in subsequent sections, with file‑by‑file notes, rationale,
and verification steps.

---

## 2. Environment and Stack Context

This work is performed in the `wekadocs-matrix` repository with the standard Docker Compose stack:

- **Neo4j**: `weka-neo4j` container, running Neo4j 2025.10.1 community edition
  - Bolted as `bolt://weka-neo4j:7687` from other containers
  - Exposed to the host on `localhost:7687` via port mapping
  - Vector index of interest:
    - `section_embeddings_v2_bge_m3` – a VECTOR index on `Section.vector_embedding` for the BGE‑M3
      profile.
  - Important metadata node:
    - Label: `SchemaVersion`
    - Properties include `version: "v2.2"`, `embedding_provider: "bge-m3-service"`,
      `embedding_model: "BAAI/bge-m3"`, `vector_dimensions: 1024`, `id: "singleton"`, etc.

- **Qdrant**: `weka-qdrant` container, v1.16.0
  - HTTP: `http://weka-qdrant:6333` inside Docker, `http://localhost:6333` from host.
  - Relevant collection:
    - `chunks_multi_bge_m3` – multi‑vector collection with named vectors `content`, `title`,
      `entity`, and `late-interaction`, plus a sparse vector `text-sparse`.

- **Redis**: `weka-redis` container, used for ingestion queues and cache invalidation.

- **MCP server**: `weka-mcp-server` container, running FastAPI + Uvicorn at `http://localhost:8000`.
  - Uses `QueryService`, `HybridRetriever`, Neo4j, Qdrant, Redis, and embedding providers.

- **Ingestion worker and service**: `weka-ingestion-worker` and `weka-ingestion-service` containers.
  - Worker pulls jobs from Redis, reads files under `/app/data/ingest`, and runs the graph builder
    + embedding pipeline.

### 2.1 Embedding Profile and Environment

Embedding profile is configured for BGE‑M3 via `.env` and `config/development.yaml`:

- `.env`:
  - `EMBEDDINGS_PROFILE=bge_m3`
  - `EMBEDDINGS_PROVIDER=bge-m3-service`
  - `EMBEDDINGS_MODEL=BAAI/bge-m3`
  - `EMBEDDINGS_DIM=1024`
  - `EMBEDDINGS_TASK=symmetric`
  - `EMBEDDING_NAMESPACE_MODE=profile` (critical for namespaced index/collection names)
  - `BGE_M3_API_URL=http://host.docker.internal:9000` (host‑side BGE‑M3 service)

- `config/development.yaml`:
  - Embedding profile default is `bge_m3` with `model_name: BAAI/bge-m3`, `dims: 1024`,
    `version: BAAI/bge-m3`, and `provider: bge-m3-service`.
  - Search configuration:
    - Qdrant collection base name: `chunks_multi` (namespaced by profile to
      `chunks_multi_bge_m3`).
    - Neo4j vector index base: `section_embeddings_v2` (namespaced to
      `section_embeddings_v2_bge_m3`).
    - Hybrid search: `rrf` with BM25 index `chunk_text_index_v3_bge_m3`.

These settings are resolved and normalized by `src/shared/config.py`, which also enforces
capability constraints (e.g., `enable_colbert` requires Query API, sparse capability, etc.).

---

## 3. Cleanup Script: Ingestion‑Only Deletion with Schema Preservation

### 3.1 File: `scripts/cleanup-databases.py`

We tightened the behavior of the cleanup script to satisfy the requirement:

> Only wipe data created as a result of ingested documents, while preserving all schema, metadata,
> vector indexes, and collection configurations.

Key behaviors now:

1. **Preserved labels (Neo4j)**
   - `PRESERVED_LABELS` includes:
     - `SchemaVersion` – the v2.2 singleton that drives schema validation and health checks.
     - `SystemMetadata`, `MigrationHistory`, `RelationshipTypesMarker`.
     - Any labels ending in `_metadata` or `_system`.
   - Counts of preserved labels are recorded before and after cleanup to ensure no metadata loss.

2. **Data labels vs. unknown labels (Neo4j)**
   - `DATA_LABELS` includes ingestion‑linked labels such as `Document`, `Section`, `Chunk`,
     `Entity`, `CitationUnit`, `Configuration`, `Command`, `Procedure`, `Step`, etc.
   - **Unknown labels default to preserved now**: earlier versions treated unknown labels as
     deletable. The updated logic classifies any label not in `PRESERVED_LABELS` and not explicitly
     in `DATA_LABELS` as preserved. This prevents accidentally deleting new/experimental metadata
     labels added in the future.

3. **Deletion strategy (Neo4j)**
   - For each label in `deletable_labels`, the script runs batched `MATCH (n:Label) WITH n LIMIT
     10000 DETACH DELETE n RETURN count(n) AS deleted` until no nodes remain.
   - This ensures ingestion‑generated nodes and relationships are removed without touching indexes,
     constraints, or metadata nodes.
   - Schema verification:
     - Before cleanup: `SHOW CONSTRAINTS` and `SHOW INDEXES` counts are captured.
     - After cleanup: counts are re‑queried and compared; any discrepancy would be treated as a
       schema error.

4. **Qdrant cleanup**
   - `self.qdrant_allowed_collections` is derived from configuration plus known bases:
     - The configured collection name (e.g., `chunks_multi_bge_m3`).
     - Legacy base collections: `chunks`, `chunks_multi`.
   - The script iterates all collections and only deletes points from those in the allowed set.
     Collection schemas are preserved.
   - Deletion uses `scroll` to gather batches of point IDs and `delete` to remove them.

5. **Redis cleanup**
   - Keys are partitioned into "system" vs. "data" based on simple prefixes
     (`schema:*`, `metadata:*`, `system:*`, `_version`, `_config`).
   - Only data keys are deleted.

### 3.2 Verified Results of Cleanup

We ran a full cleanup with:

```bash
PYTHONPATH=/Users/brennanconley/vibecode/wekadocs-matrix \
  NEO4J_URI=bolt://localhost:7687 \
  QDRANT_HOST=localhost QDRANT_PORT=6333 \
  REDIS_HOST=localhost REDIS_PASSWORD=testredis123 \
  python scripts/cleanup-databases.py
```

Key observations from the log:

- Neo4j before cleanup:
  - `Total: 258 nodes, 702 relationships`.
  - `Schema: 16 constraints, 70 indexes`.
  - `PRESERVED (Metadata/System)` showed:
    - `SchemaVersion: 1 nodes`
    - `RelationshipTypesMarker: 1 nodes`
  - `TO DELETE (Data)` included `Step`, `Section`, `Chunk`, `Configuration`, `CitationUnit`,
    `Procedure`, `Document`, `Command` nodes.

- Neo4j after cleanup:
  - `After: 2 nodes, 0 relationships`.
  - All metadata preserved; schema (constraints, indexes) unchanged.

- Qdrant:
  - Target collections: `chunks`, `chunks_multi`, `chunks_multi_bge_m3`.
  - At the time of cleanup, both `chunks_multi` and `chunks_multi_bge_m3` had `0` vectors, so no
    deletions were required.

- Redis:
  - DB `1` had `0` keys, so nothing was deleted.

Cleanup reports are written under `reports/cleanup/cleanup-report-*.json` with detailed before/after
statistics, which can be used for future audits.

---

## 4. Namespace Suffix Refactor and Alignment

### 4.1 Motivation

We previously observed a critical mismatch:

- Ingestion guard expected Qdrant collections to end with a suffix derived from the embedding
  profile or version.
- Retrieval guard also derived a suffix, but the rules were reversed in order of priority
  (version vs. profile), leading to situations where ingestion would expect `_baai_bge_m3` while the
  collection created from config + profile naming was `_bge_m3`.

This "Split Brain" between ingestion and retrieval caused runtime errors such as:

> `Qdrant collection 'chunks_multi_bge_m3' does not match expected namespace suffix 'baai_bge_m3'`

The long‑term fix is to centralize namespace suffix derivation in one place and ensure both sides use
the same logic.

### 4.2 New Helper: `get_expected_namespace_suffix`

File: `src/shared/config.py`

We added:

```python
def get_expected_namespace_suffix(
    settings: ProviderEmbeddingSettings, mode: str
) -> str:
    """Deterministic namespace suffix (single source of truth).

    Priority:
    1) Explicit mode selection (profile/version/model)
    2) Fallback: profile, then version, then model_id
    3) If namespacing is disabled/empty, return ""
    """

    normalized = (mode or "").lower()
    if normalized in {"", "none", "disabled"}:
        return ""

    if normalized == "profile" and getattr(settings, "profile", None):
        return _slugify_identifier(settings.profile)
    if normalized == "version" and getattr(settings, "version", None):
        return _slugify_identifier(settings.version)
    if normalized == "model" and getattr(settings, "model_id", None):
        return _slugify_identifier(settings.model_id)

    if getattr(settings, "profile", None):
        return _slugify_identifier(settings.profile)
    if getattr(settings, "version", None):
        return _slugify_identifier(settings.version)
    if getattr(settings, "model_id", None):
        return _slugify_identifier(settings.model_id)
    return ""
```

The helper is intentionally defensive:

- Honors explicit `EMBEDDING_NAMESPACE_MODE` values (`profile`, `version`, `model`).
- If mode is unrecognized, it defaults to preferring `profile`, then `version`, then `model_id`,
  to avoid surprising flips.
- When mode is `none`/`disabled`, it returns an empty string, effectively disabling suffix‑based
  enforcement.

### 4.3 Wiring: Ingestion Guard (GraphBuilder)

File: `src/ingestion/build_graph.py`

We updated two places in `GraphBuilder` to use the helper:

1. `_ensure_qdrant_collection`:

   ```python
   collection = self.config.search.vector.qdrant.collection_name
   expected_suffix = get_expected_namespace_suffix(
       self.embedding_settings, self.namespace_mode
   )
   if expected_suffix and isinstance(collection, str):
       if not collection.endswith(expected_suffix):
           raise RuntimeError(
               f"Qdrant collection {collection!r} does not match expected namespace suffix {expected_suffix!r}"
           )
   ```

2. `_upsert_to_qdrant` (before actually writing vectors):

   ```python
   collection = self.config.search.vector.qdrant.collection_name
   expected_suffix = get_expected_namespace_suffix(
       self.embedding_settings, self.namespace_mode
   )
   if expected_suffix and isinstance(collection, str):
       if not collection.endswith(expected_suffix):
           raise RuntimeError(
               f"Qdrant collection {collection!r} does not match expected namespace suffix {expected_suffix!r}"
           )

   # later, when building embedding_metadata
   embedding_metadata = canonicalize_embedding_metadata(
       embedding_model=self.embedding_settings.version,
       dimensions=len(content_vector),
       provider=self.embedder.provider_name,
       task=getattr(self.embedder, "task", self.embedding_settings.task),
       profile=self.embedding_settings.profile,
       timestamp=datetime.utcnow(),
       namespace_mode=self.namespace_mode,
       namespace_suffix=expected_suffix,
       collection_name=collection,
   )
   ```

This ensures ingestion both validates and records the same suffix, based on the active profile and
`EMBEDDING_NAMESPACE_MODE`.

### 4.4 Wiring: Retrieval Guard (HybridRetriever)

File: `src/query/hybrid_retrieval.py`

The `HybridRetriever` constructor previously computed a `namespace_suffix` using a slightly
different rule and then checked BM25 index and Qdrant collection names for consistency.

We updated it to call the same helper:

```python
config = get_config()
settings = get_settings()
self.embedding_settings = embedding_settings or get_embedding_settings()

search_config = config.search
qdrant_collection = getattr(
    getattr(search_config.vector, "qdrant", None), "collection_name", None
)
namespace_mode = getattr(settings, "embedding_namespace_mode", "none")
namespace_suffix = get_expected_namespace_suffix(
    self.embedding_settings, namespace_mode
)
bm25_namespaced = bool(
    namespace_suffix
    and self.bm25_retriever.index_name
    and str(self.bm25_retriever.index_name).endswith(namespace_suffix)
)
qdrant_namespaced = bool(
    namespace_suffix
    and qdrant_collection
    and str(qdrant_collection).endswith(namespace_suffix)
)
```

The rest of the guard logic remains as before: if Qdrant is namespaced but BM25 is not, and the
environment is strict, it raises a runtime error unless an override is set. In non‑strict
environments, it logs a warning and proceeds.

### 4.5 Namespace Mode Choice

After some experimentation, we **returned `EMBEDDING_NAMESPACE_MODE` to `profile`** in `.env` to
align with existing Neo4j index names and Qdrant collections that are already suffixed with `_bge_m3`.

- Index: `section_embeddings_v2_bge_m3`.
- Qdrant collection: `chunks_multi_bge_m3`.

By keeping `profile` mode and centralizing the suffix logic, we now have:

- Ingestion and retrieval both expecting `_bge_m3`.
- Schema and actual names aligned with the profile, not the fully qualified version string
  `BAAI/bge-m3`.

---

## 5. Ingestion Pipeline Robustness Improvements

### 5.1 Redundant Enrichment Fix: `_balance_small_tails`

File: `src/ingestion/chunk_assembler.py`

`GreedyCombinerV2._balance_small_tails` performs a second‑pass merge on small "tail" chunks within a
document. Previously, it behaved roughly as follows (simplified):

```python
if nxt["token_count"] < self.min_tokens and same_block and ...:
    # merge cur + nxt, then enrich merged
    cur = self._enrich_chunk(cur)
    cur = post_process_chunk(cur)
    out.append(cur)
    i += 2
else:
    # no merge – but still enrich cur
    cur = self._enrich_chunk(cur)
    cur = post_process_chunk(cur)
    out.append(cur)
    i += 1
```

The main assembly loop had already enriched the chunks, so calling `_enrich_chunk` again in the
"no merge" path was unnecessary and potentially resulted in duplicate error messages or logging.

We changed the `else` block to avoid re‑enrichment when there is no merge:

```python
else:
    # No merge: chunk was already enriched in the main assembly; avoid duplicate work
    out.append(cur)
    i += 1
```

For the merge case, re‑enrichment is still appropriate because the merged text and section metadata
have changed, and we want fresh canonicalization and error checks on the combined chunk.

### 5.2 Robust Host → Container Path Mapping: `parse_file_uri`

File: `src/ingestion/worker.py`

The ingestion worker receives job paths that may be:

- `file://` URIs pointing to host files.
- Direct host paths.
- Already container‑relative paths (e.g., `/app/data/ingest/...`).

The previous implementation of `parse_file_uri` used the **first** occurrence of `/data/` in the
path to derive a container path:

```python
if "/data/" in path:
    data_idx = path.index("/data/")
    container_path = "/app" + path[data_idx:]
    return container_path
```

This fails in cases where the host path contains `/data/` earlier, e.g.:

- `/home/user/data/project/wekadocs-matrix/data/ingest/file.md`

In that case, using the first `/data/` would produce `/app/data/project/wekadocs-matrix/data/ingest/file.md`,
which does not match how the repo is mounted inside the container.

We revised the function to anchor either on the repo root or the **last** `/data/` segment:

```python
def parse_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
    else:
        path = uri

    if path.startswith("/app/"):
        return path

    # Prefer anchoring on the repo root
    repo_marker = "/wekadocs-matrix/"
    if repo_marker in path:
        idx = path.index(repo_marker)
        return "/app" + path[idx + len(repo_marker) - 1 :]

    # Fallback: use the last /data/ segment
    if "/data/" in path:
        data_idx = path.rfind("/data/")
        return "/app" + path[data_idx:]

    # If no markers, assume already container-relative
    return path
```

This ensures typical host paths under the repo root are mapped to `/app/...` correctly, and
multi‑`/data/` host paths are no longer mis‑parsed.

---

## 6. BGE‑M3 Service Endpoint and Env Interaction

### 6.1 Container vs. Host Scripts

The BGE‑M3 service is running on the host at `http://127.0.0.1:9000`. Containers cannot reach the
host via `localhost`, so we expose it to containers via `host.docker.internal:9000`.

In `.env` we have:

```bash
BGE_M3_API_URL=http://host.docker.internal:9000
```

This value is picked up by:

```python
if provider == "bge-m3-service":
    service_url = os.getenv("BGE_M3_API_URL")
    if not service_url:
        logger.warning("BGE_M3_API_URL is not set but provider bge-m3-service is active.")
```

When we run **host‑side scripts** (like `cleanup-databases.py`) via `python` directly, we do not
automatically export the `.env` values into the host shell, so these scripts log a warning that
`BGE_M3_API_URL` is missing. That warning is harmless for cleanup but useful as a reminder.

To avoid the warning in host scripts, we can either:

- Export `BGE_M3_API_URL` in the shell before running the script, or
- Use `env $(grep -v '^#' .env | xargs) python ...` to load env vars from `.env`.

Containers themselves get the value from Docker Compose and do not exhibit this warning during
runtime operations.

---

## 7. Qdrant Query API Payload Fixes for Dense, Sparse, and ColBERT

### 7.1 Problem: Prefetch Validation Errors

The smoke test initially revealed that Query API‑based retrieval was failing with a Pydantic
validation error and falling back to the legacy search path:

> `Query API search failed; falling back to legacy search error=16 validation errors for Prefetch ...`

The error showed Qdrant's `Prefetch` model complaining that the `query` field, particularly the
`SparseVector` representation, did not match the expected schema. We were passing higher‑level
objects (e.g., `NamedSparseVector` or dicts with extraneous fields), which Qdrant's client did not
accept as valid `query` payloads.

### 7.2 Fix: `QdrantMultiVectorRetriever` Query API Builders

File: `src/query/hybrid_retrieval.py`

The main components involved are:

- `_build_query_bundle` – builds a `QueryEmbeddingBundle` from the embedder.
- `_build_prefetch_entries` – builds the `Prefetch` list for dense and sparse legs.
- `_build_query_api_query` – builds the main query payload, including ColBERT multivector when
  available.
- `_search_via_query_api` – coordinates `query_points` with the above payloads and prefetch.

#### 7.2.1 `_build_query_bundle`

We standardize the query bundle structure so that downstream code can rely on a common shape:

```python
def _build_query_bundle(self, query: str) -> QueryEmbeddingBundle:
    if hasattr(self.embedder, "embed_query_all"):
        bundle = self.embedder.embed_query_all(query)
    else:
        dense = self.embedder.embed_query(query)
        return QueryEmbeddingBundle(dense=list(dense))
    return QueryEmbeddingBundle(
        dense=list(bundle.dense),
        sparse=bundle.sparse,
        multivector=bundle.multivector,
    )
```

`QueryEmbeddingBundle` uses:

- `dense: List[float]`
- `sparse: Optional[SparseEmbedding]` (with `indices: List[int]`, `values: List[float]`)
- `multivector: Optional[MultiVectorEmbedding]` (with `vectors: List[List[float]]`).

#### 7.2.2 `_build_prefetch_entries`

We now build `Prefetch` objects for dense and sparse legs using types that Qdrant expects:

```python
entries: List[Prefetch] = []
candidate_limit = max(top_k, self.query_api_candidate_limit)
dense_limit = min(candidate_limit, self.query_api_dense_limit)
dense_vector = list(bundle.dense)

for field_name in self.dense_vector_names:
    entries.append(
        Prefetch(
            query=dense_vector,  # Qdrant expects a List[float] here
            using=field_name,
            limit=dense_limit,
            filter=qdrant_filter,
        )
    )

if (
    self.schema_supports_sparse
    and bundle.sparse
    and bundle.sparse.indices
    and bundle.sparse.values
):
    sparse_query = QdrantSparseVector(
        indices=list(bundle.sparse.indices),
        values=list(bundle.sparse.values),
    )
    entries.append(
        Prefetch(
            query=sparse_query,          # SparseVector model
            using=self.sparse_query_name,
            limit=min(candidate_limit, self.query_api_sparse_limit),
            filter=qdrant_filter,
        )
    )
return entries
```

Key points:

- Dense `Prefetch.query` is a simple `List[float]`, which is what Qdrant's `Prefetch` model expects
  for dense queries.
- Sparse `Prefetch.query` is an instance of `qdrant_client.http.models.SparseVector`, which
  matches the `SparseVector` type recognized by Qdrant.

#### 7.2.3 `_build_query_api_query`

For the main `query` passed to `client.query_points`, we differentiate between ColBERT and the
fallback dense vector:

```python
def _build_query_api_query(
    self, bundle: QueryEmbeddingBundle
) -> Tuple[Sequence[Sequence[float]] | Sequence[float], str]:
    if (
        self.schema_supports_colbert
        and bundle.multivector
        and bundle.multivector.vectors
    ):
        return (
            {
                "name": "late-interaction",
                "vector": [list(vec) for vec in bundle.multivector.vectors],
            },
            "late-interaction",
        )
    return {"name": self.primary_vector_name, "vector": list(bundle.dense)}, self.primary_vector_name
```

Points to note:

- When ColBERT is available (`schema_supports_colbert` and `bundle.multivector` not empty), we pass
  a named query payload where `vector` is a list of dense vectors (one per token). The `using` name
  is `"late-interaction"` and must match the multivector configuration in Qdrant.
- For the default dense case, we pass `{"name": primary_vector_name, "vector": dense_list}`. This
  matches how Qdrant's Query API expects named vector queries.

#### 7.2.4 `_search_via_query_api`

The main Query API search now combines the above components:

```python
prefetch_entries = self._build_prefetch_entries(bundle, qdrant_filter, top_k)
prefetch_arg = None
if prefetch_entries:
    prefetch_arg = Prefetch(
        prefetch=prefetch_entries,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=max(top_k, self.query_api_candidate_limit),
        filter=qdrant_filter,
    )
query_payload, using_name = self._build_query_api_query(bundle)
response = self.client.query_points(
    collection_name=self.collection,
    query=query_payload,
    using=using_name,
    prefetch=prefetch_arg,
    query_filter=qdrant_filter,
    with_payload=True,
    with_vectors=False,
    limit=top_k,
)
```

If any error occurs, `_search` still falls back to `_search_legacy`, but the intent is for Query API
to be the default path in non‑error conditions.

### 7.3 Regression Tests for Payload Shape

File: `tests/query/test_query_api_payload.py`

We added tests that validate the **structure** of the payloads constructed, without requiring a live
Qdrant instance:

1. **Dense + Sparse prefetch**:

   ```python
   from src.providers.embeddings.contracts import (
       MultiVectorEmbedding,
       QueryEmbeddingBundle,
       SparseEmbedding,
   )
   from src.query.hybrid_retrieval import QdrantMultiVectorRetriever

   class DummySparseEmbedder:
       def embed_query(self, query: str):
           return [0.1, 0.2, 0.3]

       def embed_sparse(self, queries):
           return [{"indices": [1, 2], "values": [0.5, 0.6]} for _ in queries]

   def test_prefetch_payload_dense_and_sparse():
       retriever = QdrantMultiVectorRetriever(
           qdrant_client=None,
           embedder=DummySparseEmbedder(),
           collection_name="chunks_multi",
           field_weights={"content": 1.0, "text-sparse": 0.5},
           embedding_settings=None,
           query_api_dense_limit=10,
           query_api_sparse_limit=10,
           schema_supports_sparse=True,
       )

       bundle = QueryEmbeddingBundle(
           dense=[0.1, 0.2, 0.3],
           sparse=SparseEmbedding(indices=[1, 2], values=[0.5, 0.6]),
           multivector=None,
       )

       entries = retriever._build_prefetch_entries(bundle, qdrant_filter=None, top_k=3)
       dense_entry = next(e for e in entries if e.using == "content")
       sparse_entry = next(e for e in entries if e.using == "text-sparse")

       assert isinstance(dense_entry.query, list)

       from qdrant_client.http.models import SparseVector as HttpSparseVector

       assert isinstance(sparse_entry.query, HttpSparseVector)
       assert sparse_entry.query.indices == [1, 2]
       assert sparse_entry.query.values == [0.5, 0.6]
   ```

2. **ColBERT multivector query payload**:

   ```python
   class DummyColbertEmbedder:
       def embed_query_all(self, query: str):
           return QueryEmbeddingBundle(
               dense=[0.1, 0.2, 0.3],
               sparse=None,
               multivector=MultiVectorEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4]]),
           )

   def test_query_payload_colbert_and_dense():
       retriever = QdrantMultiVectorRetriever(
           qdrant_client=None,
           embedder=DummyColbertEmbedder(),
           collection_name="chunks_multi",
           field_weights={"content": 1.0},
           embedding_settings=None,
           schema_supports_colbert=True,
           query_api_dense_limit=10,
       )

       bundle = retriever._build_query_bundle("test query")
       payload, using_name = retriever._build_query_api_query(bundle)

       assert using_name == "late-interaction"
       assert payload["name"] == "late-interaction"
       assert payload["vector"] == [[0.1, 0.2], [0.3, 0.4]]
   ```

Running these tests along with `tests/shared/test_namespace_suffix.py` yields:

```bash
pytest tests/query/test_query_api_payload.py tests/shared/test_namespace_suffix.py

11 passed in ~2 seconds
```

This provides confidence that payloads are structurally correct before hitting Qdrant.

---

## 8. Smoke Test Script and Verification Tooling

### 8.1 File: `scripts/smoke_test_query.py`

We added a dedicated smoke test script to exercise the complete retrieval path via `QueryService`:

```python
import sys
import os
import asyncio

sys.path.append(os.getcwd())

from src.mcp_server.query_service import get_query_service
from src.shared.connections import initialize_connections, close_connections


async def run_test():
    print("Initializing connections...")
    await initialize_connections()

    qs = get_query_service()
    query = "What are the prerequisites for storage expansion?"

    print(f"\nExecuting search for: '{query}'")
    try:
        response = qs.search(
            query=query,
            top_k=3,
            verbosity="graph",
        )

        print("\n=== Search Results ===")
        print(f"Confidence: {response.answer_json.confidence}")
        print(f"Evidence Count: {len(response.answer_json.evidence)}")

        print("\n--- Answer (markdown) ---")
        ans = response.answer_markdown or ""
        if len(ans) > 2000:
            print(ans[:2000] + "... [truncated]")
        else:
            print(ans)

        print("\n--- Answer JSON ---")
        try:
            print(response.answer_json.json(indent=2))
        except Exception:
            print(response.answer_json)

        print("\n--- Top Evidence ---")
        for i, ev in enumerate(response.answer_json.evidence[:3]):
            print(
                f"{i+1}. heading={ev.get('heading')} score={ev.get('score')} "
                f"source={ev.get('source_uri')}"
            )
    except Exception as e:
        print(f"\nERROR: Search failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await close_connections()


if __name__ == "__main__":
    asyncio.run(run_test())
```

Key points:

- The script initializes connections (Neo4j, Qdrant, Redis) via `initialize_connections()`.
- It obtains `QueryService` via `get_query_service()` and performs a search with verbosity `graph`.
- It prints:
  - Overall confidence and evidence count.
  - A truncated or full markdown answer.
  - The JSON representation of `answer_json` using `.json(indent=2)` where available.
  - A short summary of the top evidence entries (heading, score, source URI).
- We intentionally avoid `model_dump_json`, which was not available on the `StructuredResponse`
  type in this codebase.

### 8.2 Running the Smoke Test

We run it inside the `mcp-server` container to reuse the same environment and connections as the
live service:

```bash
docker compose exec mcp-server python scripts/smoke_test_query.py
```

During a test run after ingestion, logs showed:

- Connections initialized successfully.
- QueryService and HybridRetriever initialized with:
  - `embedding_profile: bge_m3`
  - `embedding_provider: bge-m3-service`
  - `embedding_model: BAAI/bge-m3`
  - `embedding_namespace_mode: profile`
  - `bm25_index_name: chunk_text_index_v3_bge_m3`
  - `qdrant_collection_name: chunks_multi_bge_m3`
- BM25 search completed with several candidates.
- Query API initially logged a validation error, but after payload fixes, it is expected to run
  cleanly without fallback.
- The final answer showed relevant chunks from the storage expansion best practice documents, with
  correct headings and source URIs.

This script is our main "quick verification" tool for downstream retrieval behavior.

---

## 9. Ingestion and Retrieval Verification

### 9.1 Ingestion Logs After Fixes

After rebuilding containers and cleaning databases, we re‑ran ingestion by dropping documents into
`/app/data/ingest` (inside the container) and watching `weka-ingestion-worker` logs. For example:

- Job processing for `best-practice-guides_storage-expansion-best-practice.md`:
  - Markdown parsing succeeded with `sections_count=7`.
  - Entity extraction logs showed 0 commands/configs for some sections (expected for textual intro
    content).
  - `GraphBuilder initialized` with the correct embedding profile and collection/index names.
  - `Ensured Neo4j vector index exists` for `section_embeddings_v2_bge_m3` with `dims=1024`.
  - `Updated SchemaVersion embedding metadata` to reflect `version=v2.2` and `dimensions=1024`.
  - Graph upsert logs showed:
    - Document node upserted.
    - Chunk batch upserted (3 chunks) with dual‑labels under the canonical schema.
    - NEXT_CHUNK relationships created.
  - Embedding provider initialization:
    - `Initializing embedding provider dims=1024 model=BAAI/bge-m3 profile=bge_m3 provider=bge-m3-service`.
    - `Embedding provider initialized actual_dims=1024 model_id=BAAI/bge-m3 provider_name=bge-m3-service`.
  - Qdrant operations:
    - `Deleted stale Qdrant chunks (replace-by-set GC) collection=chunks_multi_bge_m3`.
    - `Chunk vector upserted to Qdrant with canonical schema` for each chunk, confirming successful
      writes.
  - Neo4j embedding metadata updates per section.
  - Cache invalidation via Redis epoch bump.
  - Ingestion completed with stats summarizing sections, chunks, tokens, and SLOs.

### 9.2 Retrieval Logs After Payload Fixes

When running the smoke test, hybrid retrieval logs showed:

- Query classified as `search` intent.
- Embedding provider loaded for BGE‑M3.
- HybridRetriever initialized with Query API support and sparse/ColBERT enabled based on profile
  capabilities.
- BM25 and multi‑vector retrieval executed, followed by context assembly.
- Final logs indicated hybrid retrieval completed with:
  - `results=3`
  - `context` assembled within token budget.
  - SLO check warnings about latency or expansion rate (informational for now).

The key improvement is that the Query API path should now be structurally valid and no longer emit
Prefetch validation errors that force fallback to legacy.

---

## 10. Files Touched and Commits

### 10.1 Files Touched

- `docker-compose.yml`
  - Ensured `EMBEDDING_NAMESPACE_MODE` is passed into embedding‑aware services (MCP server,
    ingestion worker, ingestion service), though the primary behavior is still driven from `.env`.

- `.env`
  - Confirmed and restored `EMBEDDING_NAMESPACE_MODE=profile`.
  - Confirmed `BGE_M3_API_URL=http://host.docker.internal:9000`.

- `scripts/cleanup-databases.py`
  - Classified unknown labels as preserved by default.
  - Scoped Qdrant cleanup to allowed ingestion collections.
  - Strengthened reporting and before/after checks for metadata and schema.

- `src/shared/config.py`
  - Added `get_expected_namespace_suffix` and wired it into `apply_embedding_profile` / namespace
    logic.

- `src/ingestion/build_graph.py`
  - `_ensure_qdrant_collection`: replaced manual suffix logic with `get_expected_namespace_suffix`.
  - `_upsert_to_qdrant`: validated `collection` name using `get_expected_namespace_suffix` and
    propagated `namespace_mode` and `namespace_suffix` into `canonicalize_embedding_metadata`.

- `src/query/hybrid_retrieval.py`
  - `HybridRetriever.__init__`: replaced ad hoc suffix logic with
    `get_expected_namespace_suffix`.
  - `QdrantMultiVectorRetriever._build_query_bundle`: normalized `QueryEmbeddingBundle` creation.
  - `QdrantMultiVectorRetriever._build_prefetch_entries`: constructed `Prefetch` objects with dense
    `List[float]` and `SparseVector` for sparse prefetch.
  - `QdrantMultiVectorRetriever._build_query_api_query`: built named payloads for dense and
    multivector (ColBERT) queries.
  - `_search_via_query_api`: integrated prefetch and query payloads with `query_points`.

- `src/ingestion/chunk_assembler.py`
  - `_balance_small_tails`: removed redundant `_enrich_chunk` calls in the no‑merge path.

- `src/ingestion/worker.py`
  - `parse_file_uri`: anchored host‑to‑container mapping on repo root or last `/data/` segment.

- `tests/shared/test_namespace_suffix.py`
  - New tests for `get_expected_namespace_suffix`, covering different modes and conflicts between
    profile/version/model.

- `tests/query/test_query_api_payload.py`
  - New regression tests for Query API dense/sparse prefetch payloads and ColBERT multivector
    payloads.

- `scripts/smoke_test_query.py`
  - New verification tool to exercise `QueryService` and hybrid retrieval within the MCP server
    container.

### 10.2 Commits (Branch: `jina-reranker-implement`)

Key commits associated with this work (hashes truncated for brevity):

- `5ee8b21` – **fix: unify namespace suffix handling and cleanup**
  - Added `get_expected_namespace_suffix` and wired ingestion/retrieval guards.
  - Updated cleanup script classification.

- `2d6e6e1` – **fix: avoid duplicate enrichment and robust path mapping**
  - Removed redundant `_enrich_chunk` in `_balance_small_tails`.
  - Hardened `parse_file_uri`.

- `adb66a3` – **fix: build query API payloads for sparse/colbert**
  - Adjusted `_build_prefetch_entries` and `_build_query_api_query` to construct valid
    dense/sparse/multivector payloads for Qdrant Query API.
  - Added `tests/query/test_query_api_payload.py`.
  - Added `scripts/smoke_test_query.py` and made minor formatting/ruff/isort adjustments.

These commits were rebased and pushed to `origin/jina-reranker-implement`, with pre‑commit hooks
(black, ruff, isort, gitlint, detect‑secrets) enforced at each step.

---

## 11. Future Considerations and Follow‑Ups

1. **Query API vs. Legacy Path Telemetry**
   - While payloads are now structurally correct and we have unit tests for them, it's worth
     capturing more runtime metrics about how often Query API is used vs. legacy, and any remaining
     fallbacks.
   - The `last_stats` field in `QdrantMultiVectorRetriever` already tracks `path` and
     `fallback_reason`; we can expose these in higher‑level metrics.

2. **Performance and SLO Tuning**
   - Some smoke test runs reported SLO warnings (e.g., `p95_ms` > target, or expansion rate above
     threshold). While these do not indicate correctness issues, they should feed into performance
     tuning.

3. **Embedding Profile Evolution**
   - As additional profiles are added, we may need to extend `get_expected_namespace_suffix` or the
     embedding capabilities schema, but the centralized design should minimize the risk of drift.

4. **Path Mapping and Multi‑Root Repos**
   - `parse_file_uri` currently assumes the repo root marker `/wekadocs-matrix/`. If deployments
     start to run this stack under different directory names or in multi‑project monorepos, it may
     be useful to parameterize the repo marker or derive it from config.

5. **Additional Smoke Tests**
   - The current `smoke_test_query.py` focuses on a single query. It could be extended into a small
     suite of smoke tests (e.g., multiple queries, different verbosity modes, and error simulation)
     to provide broader coverage of common retrieval patterns.

---

_End of session notes._
