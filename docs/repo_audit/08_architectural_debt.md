# Architectural Debt

## God Modules

| Module | Debt | Why It Matters |
|---|---|---|
| `src/ingestion/atomic.py` | Coordinates parse, reference extraction, chunking, NER merge, embeddings, validation, writes, structural edges, cross-doc linking | Too much blast radius for any ingestion change; hard to reason about stage contracts independently. |
| `src/query/hybrid_retrieval.py` | Large orchestrator with plan resolution, graph/cross-doc signals, expansion, reranking, dedupe, citations | Active but overgrown; fallback behavior and feature flags are hard to audit. |
| `src/query/vector_backends.py` | Query API, legacy search, dense/sparse/ColBERT field handling, fusion details | Config/profile complexity is concentrated here; fallback path can diverge. |
| `src/mcp_server/mcp_tools.py` | Tool schemas, instructions, handlers, aliases, scratch/evidence behavior | Domain text and tool behavior are coupled. |
| `src/shared/config.py` | Large Pydantic config surface with defaults that drift from YAML/env | Hard to identify the true default and which layer wins. |

## Config Debt

- Model/profile source-of-truth split between `development.yaml`, `embedding_profiles.yaml`, `.env.*`, compose env, and code defaults.
- Reranker model differs across layers (`mixedbread-ai/mxbai-rerank-large-v2` vs `Qwen/Qwen3-Reranker-4B`).
- `CHUNK_ASSEMBLER` compose default is `structured`, while YAML enables semantic chunking; active behavior depends on path/config resolution.
- Some config reads appear suspicious, such as MCP utilities reading `_config.hybrid.neo4j_disabled` instead of `config.search.hybrid.neo4j_disabled`.

## Specific Config Contradictions To Resolve

| Concern | Evidence | Risk |
|---|---|---|
| Dense profile identity | `src/shared/config.py` defaults to `bge_m3`; `config/development.yaml` says `bge_m3`; `config/embedding_profiles.yaml` top-level plan says dense `qwen3_0_6b`; env files also override `EMBEDDINGS_PROFILE` | A reader cannot tell which collection/vector namespace is live without a resolved-config report. |
| Reranker identity | `config/development.yaml` names mixedbread; compose and `.env.docker` default to Qwen3; `.env.local` returns to mixedbread | Query quality and token limits differ by actual service. |
| Reranker token limits | Config/env mention 8K-style totals; `src/query/rerank_pipeline.py` has hardcoded service caps in the batching path | Long passages may degrade differently from config expectations. |
| Graph disabled flag location | MCP utilities appear to inspect a top-level `hybrid` attribute while config nests hybrid under `search.hybrid` | Graph expansion may be considered available even when config intended otherwise. |
| Domain prompts | MCP, embedding, reranker, and NER labels all carry domain language in different files | Domain migration cannot be validated by checking only one prompt file. |
| Corpus tag extraction | Ingestion derives `doc_tag` from paths under `data/ingest`, with comments/examples around WEKA categories | Nutanix source layout needs explicit domain rules. |

## Legacy Debt

- Historical docs and scripts still dominate search results and mental model.
- Duplicate implementations and patch files remain in the tree.
- Existing tests have not been proven aligned with the active path.

## Observability Debt

- Many external failures are logged and fail open, but there is no single quality-degradation metric across sparse, ColBERT, graph, reranker, and cross-doc paths.
- Fallbacks should be surfaced in query responses/evidence metadata for operator visibility.
