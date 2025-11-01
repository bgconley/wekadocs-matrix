# Integration Findings — Canonical Check (Generated 2025-10-28T21:20:21.345087Z)

- **Repository root:** `/mnt/data/repo`
- **Canonical model:** `jina-embeddings-v3` @ **1024‑D**
- **Qdrant collection:** `chunks`, named vector: `content`

> Found **838** items. See the table below.

| Severity | File | Line | Check | Match |
|---|---|---:|---|---|
| error | scripts/eval/run_eval.py | 56 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/eval/run_eval.py | 468 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/neo4j/create_schema_v2_1.cypher | 148 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | scripts/neo4j/create_schema_v2_1.cypher | 148 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/neo4j/create_schema_v2_1.cypher | 182 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 231 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 308 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 308 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 318 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 318 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/neo4j/create_schema_v2_1_complete.cypher | 363 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | scripts/test_jina_integration.py | 27 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | scripts/test_jina_integration.py | 27 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/api.py | 11 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/api.py | 23 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/api.py | 23 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/auto/orchestrator.py | 433 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/auto/orchestrator.py | 435 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/build_graph.py | 499 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/build_graph.py | 880 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/build_graph.py | 905 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/build_graph.py | 907 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/ingestion/build_graph.py | 911 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/mcp_server/query_service.py | 61 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/mcp_server/query_service.py | 364 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/providers/embeddings/sentence_transformers.py | 47 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/providers/factory.py | 80 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/providers/factory.py | 199 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/providers/factory.py | 200 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/providers/factory.py | 212 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/registry/index_registry.py | 236 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/shared/config.py | 26 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/shared/config.py | 69 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | src/shared/config.py | 324 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/e2e/test_golden_set.py | 189 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/e2e/test_golden_set.py | 398 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/p6_t4_test.py | 69 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_integration_prephase7.py | 47 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_integration_prephase7.py | 59 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_integration_prephase7.py | 271 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_phase1_foundation.py | 27 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_phase1_foundation.py | 57 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_phase7c_index_registry.py | 31 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_index_registry.py | 41 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_index_registry.py | 149 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_index_registry.py | 215 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_ingestion.py | 43 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_phase7c_provider_factory.py | 26 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 33 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 111 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 118 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 128 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 135 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 141 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_provider_factory.py | 152 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| error | tests/test_phase7c_reranking.py | 220 | Use of deprecated property name 'embedding_model' (use 'embedding_version') | `embedding_model` |
| error | tests/test_phase7c_schema_v2_1.py | 388 | Found reference to deprecated model v4 (should be v3) | `jina-embeddings-v4` |
| info | config/development.yaml | 16 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | config/development.yaml | 23 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 140 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 150 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 222 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 231 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 299 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker-compose.yml | 308 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/ingestion-service.Dockerfile | 20 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/ingestion-service.Dockerfile | 24 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/ingestion-service.Dockerfile | 24 | HF tokenizer for Jina v3 | `from_pretrained('jinaai/jina-embeddings-v3` |
| info | docker/ingestion-worker.Dockerfile | 19 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/ingestion-worker.Dockerfile | 23 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/ingestion-worker.Dockerfile | 23 | HF tokenizer for Jina v3 | `from_pretrained('jinaai/jina-embeddings-v3` |
| info | docker/mcp-server.Dockerfile | 19 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/mcp-server.Dockerfile | 23 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | docker/mcp-server.Dockerfile | 23 | HF tokenizer for Jina v3 | `from_pretrained('jinaai/jina-embeddings-v3` |
| info | requirements.txt | 74 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | scripts/apply_complete_schema_v2_1.py | 79 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/backfill_document_tokens.py | 51 | Presence of :Section label | `:Section` |
| info | scripts/backfill_document_tokens.py | 67 | Presence of :Section label | `:Section` |
| info | scripts/backfill_document_tokens.py | 138 | Presence of :Section label | `:Section` |
| info | scripts/baseline_distribution_analysis.py | 50 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 60 | Qdrant cosine distance | `Distance.COSINE` |
| info | scripts/dev/seed_minimal_graph.py | 66 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/dev/seed_minimal_graph.py | 181 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | scripts/dev/seed_minimal_graph.py | 198 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/dev/seed_minimal_graph.py | 199 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/dev/seed_minimal_graph.py | 199 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/dev/seed_minimal_graph.py | 204 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 215 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 270 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 280 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 290 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 300 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 310 | Presence of :Section label | `:Section` |
| info | scripts/dev/seed_minimal_graph.py | 351 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/dev/seed_minimal_graph.py | 351 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema.cypher | 19 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema.cypher | 65 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema.cypher | 68 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema.cypher | 71 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema.cypher | 100 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema.cypher | 101 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema.cypher | 102 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema.cypher | 103 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema.cypher | 104 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema.cypher | 105 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 21 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 27 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 28 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 29 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 64 | Vector index creation | `CREATE VECTOR INDEX section_embeddings_v2` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 65 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 66 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 75 | Vector index creation | `CREATE VECTOR INDEX chunk_embeddings_v2` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 76 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 77 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 126 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 129 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 132 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 132 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 157 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 159 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 188 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 189 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 190 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 192 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | scripts/neo4j/create_schema_v2_1.cypher | 193 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 28 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 49 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 52 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 114 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 115 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 117 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 118 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 147 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 150 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 153 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 211 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 212 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 213 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 217 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 220 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 223 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 223 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 246 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 252 | Vector index creation | `CREATE VECTOR INDEX section_embeddings_v2` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 253 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 254 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 263 | Vector index creation | `CREATE VECTOR INDEX chunk_embeddings_v2` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 264 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 265 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 276 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 276 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 277 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 279 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 283 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 283 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 286 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 288 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 292 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 293 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 294 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 331 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 333 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 379 | Presence of :Section label | `:Section` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 380 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 381 | Use of canonical 'embedding_version' | `embedding_version` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 383 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | scripts/neo4j/create_schema_v2_1_complete.cypher | 384 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | scripts/test/debug_explain.py | 21 | Presence of :Section label | `:Section` |
| info | scripts/test/debug_explain.py | 26 | Presence of :Section label | `:Section` |
| info | scripts/test_jina_payload_limits.py | 30 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | scripts/test_jina_payload_limits.py | 85 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | scripts/test_jina_payload_limits.py | 146 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | scripts/validate_token_accounting.py | 65 | Presence of :Section label | `:Section` |
| info | scripts/validate_token_accounting.py | 181 | Presence of :Section label | `:Section` |
| info | src/ingestion/api.py | 12 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/api.py | 24 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/api.py | 24 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/cli.py | 418 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | src/ingestion/auto/cli.py | 828 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 350 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 449 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/orchestrator.py | 450 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 794 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 832 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/orchestrator.py | 850 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 869 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/orchestrator.py | 874 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/orchestrator.py | 875 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/orchestrator.py | 876 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 885 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/orchestrator.py | 888 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/orchestrator.py | 892 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/orchestrator.py | 893 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 32 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 137 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/report.py | 170 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 170 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 177 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/report.py | 178 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/report.py | 179 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 182 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 190 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 190 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 199 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 199 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/report.py | 230 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/verification.py | 43 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/verification.py | 99 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/verification.py | 100 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/verification.py | 103 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/verification.py | 121 | Presence of :Section label | `:Section` |
| info | src/ingestion/auto/verification.py | 122 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/auto/verification.py | 123 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/auto/verification.py | 126 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 44 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 141 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | src/ingestion/build_graph.py | 158 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | src/ingestion/build_graph.py | 167 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | src/ingestion/build_graph.py | 167 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 170 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | src/ingestion/build_graph.py | 170 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 218 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 259 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 276 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 396 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 529 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 530 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 592 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/build_graph.py | 610 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 612 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 616 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 666 | Qdrant cosine distance | `Distance.COSINE` |
| info | src/ingestion/build_graph.py | 745 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 747 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 749 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 774 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/build_graph.py | 775 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 784 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 794 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 797 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 805 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 830 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 832 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 833 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 843 | Presence of :Section label | `:Section` |
| info | src/ingestion/build_graph.py | 844 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/build_graph.py | 844 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/build_graph.py | 845 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 845 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 847 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 847 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 848 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 848 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 857 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/build_graph.py | 858 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 858 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 860 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 860 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 861 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 861 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | src/ingestion/build_graph.py | 869 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | src/ingestion/build_graph.py | 881 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 913 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 915 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/build_graph.py | 919 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/extract/commands.py | 257 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/extract/commands.py | 258 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/extract/configs.py | 252 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/extract/configs.py | 253 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/incremental.py | 28 | Presence of :Section label | `:Section` |
| info | src/ingestion/incremental.py | 38 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/incremental.py | 44 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/incremental.py | 55 | Presence of :Section label | `:Section` |
| info | src/ingestion/incremental.py | 140 | Presence of :Section label | `:Section` |
| info | src/ingestion/incremental.py | 162 | Presence of :Section label | `:Section` |
| info | src/ingestion/incremental.py | 167 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/incremental.py | 169 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | src/ingestion/incremental.py | 196 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/parsers/html.py | 219 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/parsers/html.py | 220 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/parsers/markdown.py | 252 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/parsers/markdown.py | 253 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/parsers/notion.py | 211 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/ingestion/parsers/notion.py | 212 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 17 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 27 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 36 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 46 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 64 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 67 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 69 | Presence of :Section label | `:Section` |
| info | src/ingestion/reconcile.py | 70 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 87 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 125 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 182 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 210 | Presence of :Section label | `:Section` |
| info | src/ingestion/reconcile.py | 240 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/ingestion/reconcile.py | 260 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/mcp_server/query_service.py | 247 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/mcp_server/query_service.py | 252 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/mcp_server/query_service.py | 253 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/mcp_server/query_service.py | 255 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/providers/embeddings/base.py | 42 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/embeddings/jina.py | 3 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/embeddings/jina.py | 150 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/embeddings/jina.py | 186 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/tokenizer_service.py | 4 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/tokenizer_service.py | 87 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/tokenizer_service.py | 95 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/tokenizer_service.py | 104 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/providers/tokenizer_service.py | 286 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/query/hybrid_search.py | 156 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/query/hybrid_search.py | 156 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/query/hybrid_search.py | 168 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/query/hybrid_search.py | 469 | Presence of :Section label | `:Section` |
| info | src/query/hybrid_search.py | 615 | Presence of :Section label | `:Section` |
| info | src/query/planner.py | 269 | Presence of :Section label | `:Section` |
| info | src/query/session_tracker.py | 340 | Presence of :Section label | `:Section` |
| info | src/query/session_tracker.py | 429 | Presence of :Section label | `:Section` |
| info | src/query/session_tracker.py | 489 | Presence of :Section label | `:Section` |
| info | src/query/session_tracker.py | 499 | Presence of :Section label | `:Section` |
| info | src/query/templates/advanced/troubleshooting_path.cypher | 49 | Presence of :Section label | `:Section` |
| info | src/query/templates/advanced/troubleshooting_path.cypher | 51 | Presence of :Section label | `:Section` |
| info | src/query/templates/explain.cypher | 29 | Presence of :Section label | `:Section` |
| info | src/query/templates/search.cypher | 6 | Presence of :Section label | `:Section` |
| info | src/query/templates/search.cypher | 15 | Presence of :Section label | `:Section` |
| info | src/query/templates/search.cypher | 28 | Presence of :Section label | `:Section` |
| info | src/query/templates/troubleshoot.cypher | 18 | Presence of :Section label | `:Section` |
| info | src/registry/index_registry.py | 54 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | src/shared/cache.py | 9 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 210 | Pattern-scan deletion (fallback) | `SCAN` |
| info | src/shared/cache.py | 246 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 255 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 259 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 259 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 289 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 295 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 377 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 384 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 390 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 390 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 416 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/cache.py | 416 | Use of canonical 'embedding_version' | `embedding_version` |
| info | src/shared/connections.py | 237 | Qdrant cosine distance | `Distance.COSINE` |
| info | src/shared/connections.py | 241 | Qdrant cosine distance | `Distance.COSINE` |
| info | src/shared/schema.py | 128 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | src/shared/schema.py | 237 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | src/shared/schema.py | 302 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/shared/schema.py | 303 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/shared/schema.py | 304 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/shared/schema.py | 305 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/shared/schema.py | 306 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | src/shared/schema.py | 307 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_jina_large_batches.py | 31 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/integration/test_jina_large_batches.py | 42 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/integration/test_jina_large_batches.py | 268 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/integration/test_jina_large_batches.py | 357 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/integration/test_phase7c_integration.py | 133 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 214 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/integration/test_phase7c_integration.py | 468 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/integration/test_phase7c_integration.py | 468 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 476 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_phase7c_integration.py | 477 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/integration/test_phase7c_integration.py | 479 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/integration/test_phase7c_integration.py | 480 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/integration/test_phase7c_integration.py | 495 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 523 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 548 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/integration/test_phase7c_integration.py | 548 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 556 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_phase7c_integration.py | 557 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/integration/test_phase7c_integration.py | 559 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/integration/test_phase7c_integration.py | 560 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/integration/test_phase7c_integration.py | 586 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 622 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 715 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/integration/test_phase7c_integration.py | 742 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/integration/test_phase7c_integration.py | 742 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 750 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_phase7c_integration.py | 751 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/integration/test_phase7c_integration.py | 753 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/integration/test_phase7c_integration.py | 754 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/integration/test_phase7c_integration.py | 785 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 910 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/integration/test_phase7c_integration.py | 922 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/integration/test_phase7c_integration.py | 922 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 930 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_phase7c_integration.py | 931 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/integration/test_phase7c_integration.py | 933 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/integration/test_phase7c_integration.py | 934 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/integration/test_phase7c_integration.py | 952 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 977 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 997 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 1015 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/integration/test_phase7c_integration.py | 1028 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/integration/test_phase7c_integration.py | 1028 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 1036 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/integration/test_phase7c_integration.py | 1037 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/integration/test_phase7c_integration.py | 1039 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/integration/test_phase7c_integration.py | 1040 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/integration/test_phase7c_integration.py | 1072 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 1100 | Presence of :Section label | `:Section` |
| info | tests/integration/test_phase7c_integration.py | 1130 | Presence of :Section label | `:Section` |
| info | tests/p1_t3_test.py | 129 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/p1_t3_test.py | 156 | Presence of :Section label | `:Section` |
| info | tests/p1_t3_test.py | 172 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 69 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 145 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 154 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 167 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 177 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 211 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 231 | Presence of :Section label | `:Section` |
| info | tests/p2_t2_test.py | 242 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_integration_test.py | 80 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_integration_test.py | 135 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_integration_test.py | 174 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_integration_test.py | 240 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t3_integration_test.py | 244 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_integration_test.py | 245 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t3_integration_test.py | 260 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t3_test.py | 101 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 130 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 166 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 216 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 248 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 250 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/p3_t3_test.py | 265 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t3_test.py | 278 | Presence of :Section label | `:Section` |
| info | tests/p3_t3_test.py | 280 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t4_integration_test.py | 59 | Presence of :Section label | `:Section` |
| info | tests/p3_t4_integration_test.py | 117 | Presence of :Section label | `:Section` |
| info | tests/p3_t4_integration_test.py | 151 | Presence of :Section label | `:Section` |
| info | tests/p3_t4_integration_test.py | 306 | Presence of :Section label | `:Section` |
| info | tests/p3_t4_integration_test.py | 307 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t4_integration_test.py | 344 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t4_integration_test.py | 375 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p3_t4_test.py | 54 | Presence of :Section label | `:Section` |
| info | tests/p3_t4_test.py | 140 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_perf_test.py | 42 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |       
| info | tests/p4_t2_perf_test.py | 51 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_perf_test.py | 232 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_perf_test.py | 289 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 48 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/p4_t2_test.py | 49 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 50 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 139 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 149 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 158 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 159 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 187 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 256 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 282 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 307 | Presence of :Section label | `:Section` |
| info | tests/p4_t2_test.py | 358 | Presence of :Section label | `:Section` |
| info | tests/p4_t3_test.py | 219 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p4_t4_test.py | 27 | Presence of :Section label | `:Section` |
| info | tests/p4_t4_test.py | 172 | Presence of :Section label | `:Section` |
| info | tests/p4_t4_test.py | 261 | Presence of :Section label | `:Section` |
| info | tests/p4_t4_test.py | 317 | Presence of :Section label | `:Section` |
| info | tests/p4_t4_test.py | 353 | Presence of :Section label | `:Section` |
| info | tests/p4_t4_test.py | 441 | Presence of :Section label | `:Section` |
| info | tests/p5_t2_test.py | 170 | Presence of :Section label | `:Section` |
| info | tests/p5_t3_test.py | 48 | Presence of :Section label | `:Section` |
| info | tests/p5_t3_test.py | 83 | Presence of :Section label | `:Section` |
| info | tests/p6_t1_test.py | 66 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t1_test.py | 70 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t1_test.py | 616 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 343 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 361 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 436 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 476 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 550 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 562 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 600 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 824 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 902 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 986 | Presence of :Section label | `:Section` |
| info | tests/p6_t2_test.py | 1098 | Presence of :Section label | `:Section` |
| info | tests/p6_t3_test.py | 64 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t3_test.py | 68 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t3_test.py | 265 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t3_test.py | 274 | Pattern-scan deletion (fallback) | `scan_iter` |
| info | tests/p6_t4_test.py | 524 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/p6_t4_test.py | 542 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 291 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 293 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_integration_prephase7.py | 295 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_integration_prephase7.py | 300 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 302 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_integration_prephase7.py | 360 | Presence of :Section label | `:Section` |
| info | tests/test_integration_prephase7.py | 379 | Presence of :Section label | `:Section` |
| info | tests/test_integration_prephase7.py | 389 | Presence of :Section label | `:Section` |
| info | tests/test_integration_prephase7.py | 449 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 453 | Presence of :Section label | `:Section` |
| info | tests/test_integration_prephase7.py | 454 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 461 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_integration_prephase7.py | 463 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_jina_adaptive_batching.py | 32 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 49 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 72 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 97 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 155 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 190 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 236 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 390 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 405 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 422 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 467 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_jina_adaptive_batching.py | 499 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_phase2_provider_wiring.py | 138 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase2_provider_wiring.py | 140 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase2_provider_wiring.py | 142 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase2_provider_wiring.py | 147 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase2_provider_wiring.py | 149 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase2_provider_wiring.py | 151 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase2_provider_wiring.py | 158 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase2_provider_wiring.py | 163 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase2_provider_wiring.py | 165 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase5_response_schema.py | 103 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_dual_write.py | 208 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_dual_write.py | 220 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_dual_write.py | 267 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_dual_write.py | 268 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_dual_write.py | 270 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_dual_write.py | 271 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_index_registry.py | 179 | Reference to canonical model v3 | `jina-embeddings-v3` |
| info | tests/test_phase7c_ingestion.py | 5 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_ingestion.py | 97 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_ingestion.py | 117 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_ingestion.py | 120 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_ingestion.py | 136 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_ingestion.py | 137 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_ingestion.py | 138 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_ingestion.py | 140 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 141 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_ingestion.py | 152 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_ingestion.py | 155 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_ingestion.py | 161 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 164 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_ingestion.py | 192 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_ingestion.py | 195 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 196 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_ingestion.py | 273 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_ingestion.py | 274 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_ingestion.py | 279 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 280 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 284 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 286 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 328 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_ingestion.py | 330 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_ingestion.py | 331 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_ingestion.py | 333 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_ingestion.py | 334 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_ingestion.py | 426 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 63 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 67 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_schema_v2_1.py | 75 | MERGE Document by id (canonical) | `MERGE (d:Document {id: ` |
| info | tests/test_phase7c_schema_v2_1.py | 77 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_schema_v2_1.py | 77 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 86 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 87 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 89 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_schema_v2_1.py | 90 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_schema_v2_1.py | 157 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_schema_v2_1.py | 157 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 165 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 166 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 168 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_schema_v2_1.py | 169 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_schema_v2_1.py | 177 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 178 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 179 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 181 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_schema_v2_1.py | 182 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_schema_v2_1.py | 193 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 378 | Presence of :Chunk label (dual-label support) | `:Chunk` |
| info | tests/test_phase7c_schema_v2_1.py | 378 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 387 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 388 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 390 | Use of canonical 'embedding_timestamp' | `embedding_timestamp` |
| info | tests/test_phase7c_schema_v2_1.py | 391 | Use of canonical 'embedding_dimensions' | `embedding_dimensions` |
| info | tests/test_phase7c_schema_v2_1.py | 399 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 400 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 424 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 432 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 432 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 438 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 445 | Presence of :Section label | `:Section` |
| info | tests/test_phase7c_schema_v2_1.py | 446 | Use of canonical 'vector_embedding' (Neo4j) | `vector_embedding` |
| info | tests/test_phase7c_schema_v2_1.py | 447 | Use of canonical 'embedding_version' | `embedding_version` |
| info | tests/test_phase7c_schema_v2_1.py | 462 | Presence of :Section label | `:Section` |
| info | tests/test_phase7e_phase0.py | 511 | Presence of :Section label | `:Section` |
| info | tests/test_tokenizer_service.py | 15 | Reference to canonical model v3 | `jina-embeddings-v3` |
| warn | requirements.txt | 74 | Use of tiktoken (NON-canonical for Jina v3) | `tiktoken` |
| warn | scripts/backfill_document_tokens.py | 53 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/backfill_document_tokens.py | 56 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/backfill_document_tokens.py | 120 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/backfill_document_tokens.py | 141 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 51 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 52 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | scripts/baseline_distribution_analysis.py | 131 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 134 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 143 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 143 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 162 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 166 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 182 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 182 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 199 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 199 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/baseline_distribution_analysis.py | 282 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/dev/seed_minimal_graph.py | 214 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/dev/seed_minimal_graph.py | 218 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/neo4j/create_schema_v2_1.cypher | 69 | Neo4j vector dimensions | ``vector.dimensions`: 1024` |
| warn | scripts/neo4j/create_schema_v2_1.cypher | 80 | Neo4j vector dimensions | ``vector.dimensions`: 1024` |
| warn | scripts/neo4j/create_schema_v2_1_complete.cypher | 257 | Neo4j vector dimensions | ``vector.dimensions`: 1024` |
| warn | scripts/neo4j/create_schema_v2_1_complete.cypher | 268 | Neo4j vector dimensions | ``vector.dimensions`: 1024` |
| warn | scripts/perf/test_traversal_latency.py | 43 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | scripts/perf/test_traversal_latency.py | 43 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | scripts/run_baseline_queries.py | 109 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | scripts/validate_token_accounting.py | 66 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 71 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 80 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 152 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 182 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 185 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | scripts/validate_token_accounting.py | 248 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | src/ingestion/auto/orchestrator.py | 836 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/auto/orchestrator.py | 874 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/auto/orchestrator.py | 880 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/auto/orchestrator.py | 892 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/auto/orchestrator.py | 897 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 228 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 248 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 250 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 362 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 396 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 721 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/build_graph.py | 737 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/__init__.py | 33 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 28 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 33 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 39 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 44 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 54 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 54 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 63 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 84 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 92 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 103 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 120 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 126 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 137 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 158 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 164 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 263 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 267 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 267 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/commands.py | 272 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 26 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 29 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 34 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 39 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 46 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 57 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 57 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 65 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 95 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 102 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 128 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 134 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 158 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 165 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 189 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 208 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 258 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 262 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 262 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/configs.py | 267 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/procedures.py | 103 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/procedures.py | 127 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/extract/procedures.py | 155 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/html.py | 201 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/html.py | 208 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/markdown.py | 234 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/markdown.py | 240 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/notion.py | 193 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/ingestion/parsers/notion.py | 200 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/mcp_server/query_service.py | 282 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/providers/tokenizer_service.py | 5 | Use of tiktoken (NON-canonical for Jina v3) | `tiktoken` |
| warn | src/providers/tokenizer_service.py | 383 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/providers/tokenizer_service.py | 397 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/providers/tokenizer_service.py | 404 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/providers/tokenizer_service.py | 432 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/providers/tokenizer_service.py | 439 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/providers/tokenizer_service.py | 456 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/providers/tokenizer_service.py | 471 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/providers/tokenizer_service.py | 471 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/providers/tokenizer_service.py | 478 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/providers/tokenizer_service.py | 489 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | src/query/hybrid_search.py | 614 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/hybrid_search.py | 615 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/hybrid_search.py | 617 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/hybrid_search.py | 632 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/hybrid_search.py | 633 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 39 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 262 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 263 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 264 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 339 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 340 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 344 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 347 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 347 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 362 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 362 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 402 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 402 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 723 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/response_builder.py | 724 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/session_tracker.py | 323 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/session_tracker.py | 340 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/session_tracker.py | 429 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/session_tracker.py | 438 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | src/query/templates/explain.cypher | 33 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/fixtures/baseline_query_set.yaml | 9 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 443 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 452 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 482 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 483 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 498 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 591 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 608 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 609 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 715 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 719 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 735 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 736 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 742 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 744 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 756 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 762 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 765 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 765 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 767 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 775 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 784 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 790 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 910 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 913 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 917 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 922 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 924 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 936 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 939 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 939 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 941 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 952 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 956 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 996 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1000 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1015 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1018 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1022 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1023 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1028 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1030 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1042 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1045 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1045 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1047 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1072 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1080 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_phase7c_integration.py | 1129 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_phase7c_integration.py | 1136 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/integration/test_session_tracking.py | 230 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_session_tracking.py | 239 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/integration/test_session_tracking.py | 535 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p1_t3_test.py | 126 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p1_t3_test.py | 136 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p1_t3_test.py | 144 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p1_t3_test.py | 153 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p1_t3_test.py | 157 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p1_t3_test.py | 163 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p1_t3_test.py | 164 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p1_t3_test.py | 172 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p2_t3_test.py | 336 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p2_t3_test.py | 340 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p2_t3_test.py | 342 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p2_t3_test.py | 347 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p2_t3_test.py | 347 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p2_t4_test.py | 87 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p2_t4_test.py | 92 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t2_test.py | 54 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t2_test.py | 263 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t3_test.py | 101 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 104 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 130 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 133 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 167 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 174 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 217 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 220 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 249 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 253 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 279 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t3_test.py | 283 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t4_integration_test.py | 118 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t4_integration_test.py | 152 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t4_integration_test.py | 308 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t4_integration_test.py | 313 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/p3_t4_test.py | 141 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p3_t4_test.py | 144 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p4_t2_perf_test.py | 289 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p4_t2_perf_test.py | 290 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/p4_t4_test.py | 261 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/test_phase7c_dual_write.py | 195 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_dual_write.py | 196 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_dual_write.py | 207 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_dual_write.py | 219 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_dual_write.py | 267 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_dual_write.py | 274 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_ingestion.py | 106 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_ingestion.py | 109 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_ingestion.py | 136 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_ingestion.py | 144 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_phase7c_schema_v2_1.py | 234 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | tests/test_phase7c_schema_v2_1.py | 242 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | tests/test_phase7c_schema_v2_1.py | 243 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | tests/test_phase7e_phase0.py | 514 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/test_phase7e_phase0.py | 514 | Use of non-canonical 'doc_id' (should be 'document_id') | `doc_id` |
| warn | tests/test_tokenizer_service.py | 274 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_tokenizer_service.py | 278 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | tests/test_tokenizer_service.py | 293 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |
| warn | tests/test_tokenizer_service.py | 300 | Use of non-canonical 'chunk_index' (should be 'order') | `chunk_index` |
| warn | tests/test_tokenizer_service.py | 494 | Use of non-canonical 'section_id' (should be 'id') | `section_id` |

## Context
### scripts/eval/run_eval.py:56 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    54 |         logger.info("Quality Evaluator initialized")
    55 |         logger.info(f"Embedding provider: {self.config.embedding.provider}")
    56 |         logger.info(f"Embedding model: {self.config.embedding.embedding_model}")
    57 |         logger.info(f"Embedding dimensions: {self.config.embedding.dims}")
    58 | 
```

### scripts/eval/run_eval.py:468 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   466 |             "system_config": {
   467 |                 "provider": self.config.embedding.provider,
   468 |                 "model": self.config.embedding.embedding_model,
   469 |                 "dimensions": self.config.embedding.dims,
   470 |                 "version": self.config.embedding.version,
```

### scripts/neo4j/create_schema_v2_1.cypher:148 — Found reference to deprecated model v4 (should be v3) (error)
```text
   146 |     sv.vector_dimensions = 1024,
   147 |     sv.embedding_provider = 'jina-ai',
   148 |     sv.embedding_model = 'jina-embeddings-v4',
   149 |     sv.validation_note = 'Property existence constraints enforced in application layer'
   150 | ;
```

### scripts/neo4j/create_schema_v2_1.cypher:148 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   146 |     sv.vector_dimensions = 1024,
   147 |     sv.embedding_provider = 'jina-ai',
   148 |     sv.embedding_model = 'jina-embeddings-v4',
   149 |     sv.validation_note = 'Property existence constraints enforced in application layer'
   150 | ;
```

### scripts/neo4j/create_schema_v2_1.cypher:182 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   180 | //        sv.vector_dimensions as dims,
   181 | //        sv.embedding_provider as provider,
   182 | //        sv.embedding_model as model,
   183 | //        sv.updated_at as updated,
   184 | //        sv.description as description,
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:231 — Found reference to deprecated model v4 (should be v3) (error)
```text
   229 | // CRITICAL SPECIFICATION: 1024 dimensions (Jina v4, not 384-D or 768-D)
   230 | // Provider: jina-ai (default), bge-m3 (fallback)
   231 | // Model: jina-embeddings-v4
   232 | // Similarity: cosine
   233 | //
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:308 — Found reference to deprecated model v4 (should be v3) (error)
```text
   306 | // - vector_dimensions: Embedding dimensionality (1024)
   307 | // - embedding_provider: Default provider (jina-ai)
   308 | // - embedding_model: Model identifier (jina-embeddings-v4)
   309 | // - updated_at: Timestamp of schema creation/update
   310 | // - description: Human-readable summary
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:308 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   306 | // - vector_dimensions: Embedding dimensionality (1024)
   307 | // - embedding_provider: Default provider (jina-ai)
   308 | // - embedding_model: Model identifier (jina-embeddings-v4)
   309 | // - updated_at: Timestamp of schema creation/update
   310 | // - description: Human-readable summary
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:318 — Found reference to deprecated model v4 (should be v3) (error)
```text
   316 |     sv.vector_dimensions = 1024,
   317 |     sv.embedding_provider = 'jina-ai',
   318 |     sv.embedding_model = 'jina-embeddings-v4',
   319 |     sv.updated_at = datetime(),
   320 |     sv.description = 'Phase 7C: Complete v2.1 schema with 1024-D vectors, dual-labeling, session tracking',
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:318 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   316 |     sv.vector_dimensions = 1024,
   317 |     sv.embedding_provider = 'jina-ai',
   318 |     sv.embedding_model = 'jina-embeddings-v4',
   319 |     sv.updated_at = datetime(),
   320 |     sv.description = 'Phase 7C: Complete v2.1 schema with 1024-D vectors, dual-labeling, session tracking',
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:363 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   361 | //        sv.vector_dimensions as dimensions,
   362 | //        sv.embedding_provider as provider,
   363 | //        sv.embedding_model as model,
   364 | //        sv.updated_at as updated,
   365 | //        sv.description as description;
```

### scripts/test_jina_integration.py:27 — Found reference to deprecated model v4 (should be v3) (error)
```text
    25 |     config = get_config()
    26 |     config.embedding.provider = "jina-ai"
    27 |     config.embedding.embedding_model = "jina-embeddings-v4"
    28 |     config.embedding.dims = 1024
    29 | 
```

### scripts/test_jina_integration.py:27 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    25 |     config = get_config()
    26 |     config.embedding.provider = "jina-ai"
    27 |     config.embedding.embedding_model = "jina-embeddings-v4"
    28 |     config.embedding.dims = 1024
    29 | 
```

### src/ingestion/api.py:11 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
     9 |     fmt: str = "markdown",
    10 |     *,
    11 |     embedding_model: Optional[str] = None,
    12 |     embedding_version: Optional[str] = None,
    13 | ) -> Dict[str, Any]:
```

### src/ingestion/api.py:23 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    21 |         content,
    22 |         format=fmt,
    23 |         embedding_model=embedding_model,
    24 |         embedding_version=embedding_version,
    25 |     )
```

### src/ingestion/api.py:23 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    21 |         content,
    22 |         format=fmt,
    23 |         embedding_model=embedding_model,
    24 |         embedding_version=embedding_version,
    25 |     )
```

### src/ingestion/auto/orchestrator.py:433 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   431 |         if not self.embedder:
   432 |             logger.info(
   433 |                 "Loading embedding model", model=self.config.embedding.embedding_model
   434 |             )
   435 |             self.embedder = SentenceTransformer(self.config.embedding.embedding_model)
```

### src/ingestion/auto/orchestrator.py:435 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   433 |                 "Loading embedding model", model=self.config.embedding.embedding_model
   434 |             )
   435 |             self.embedder = SentenceTransformer(self.config.embedding.embedding_model)
   436 | 
   437 |         # Compute embeddings for sections
```

### src/ingestion/build_graph.py:499 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   497 |                 "Initializing embedding provider",
   498 |                 provider=self.config.embedding.provider,
   499 |                 model=self.config.embedding.embedding_model,
   500 |                 dims=self.config.embedding.dims,
   501 |             )
```

### src/ingestion/build_graph.py:880 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   878 |     format: str = "markdown",
   879 |     *,
   880 |     embedding_model: Optional[str] = None,
   881 |     embedding_version: Optional[str] = None,
   882 | ) -> Dict:
```

### src/ingestion/build_graph.py:905 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   903 | 
   904 |     # Allow optional overrides prior to ingestion
   905 |     if embedding_model:
   906 |         try:
   907 |             config.embedding.model_name = embedding_model
```

### src/ingestion/build_graph.py:907 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   905 |     if embedding_model:
   906 |         try:
   907 |             config.embedding.model_name = embedding_model
   908 |         except Exception:
   909 |             logger.warning(
```

### src/ingestion/build_graph.py:911 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   909 |             logger.warning(
   910 |                 "Failed to override embedding model via ingest_document",
   911 |                 requested_model=embedding_model,
   912 |             )
   913 |     if embedding_version:
```

### src/mcp_server/query_service.py:61 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    59 |             logger.info(
    60 |                 f"Loading embedding provider from config: provider={self.config.embedding.provider}, "
    61 |                 f"model={self.config.embedding.embedding_model}, dims={expected_dims}"
    62 |             )
    63 | 
```

### src/mcp_server/query_service.py:364 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   362 |             "planner_initialized": self._planner is not None,
   363 |             "model_name": (
   364 |                 self.config.embedding.embedding_model if self._embedder else None
   365 |             ),
   366 |         }
```

### src/providers/embeddings/sentence_transformers.py:47 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    45 | 
    46 |         # Use provided values or fall back to config
    47 |         self._model_name = model_name or config.embedding.embedding_model
    48 |         self._expected_dims = expected_dims or config.embedding.dims
    49 |         self._provider_name = "sentence-transformers"
```

### src/providers/factory.py:80 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    78 |             "EMBEDDINGS_PROVIDER", config.embedding.provider
    79 |         )
    80 |         model = model or os.getenv("EMBEDDINGS_MODEL", config.embedding.embedding_model)
    81 |         dims = dims or int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))
    82 |         task = task or os.getenv("EMBEDDINGS_TASK", config.embedding.task)
```

### src/providers/factory.py:199 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   197 |         # Get actual ENV values (may override config)
   198 |         embedding_provider = os.getenv("EMBEDDINGS_PROVIDER", config.embedding.provider)
   199 |         embedding_model = os.getenv(
   200 |             "EMBEDDINGS_MODEL", config.embedding.embedding_model
   201 |         )
```

### src/providers/factory.py:200 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   198 |         embedding_provider = os.getenv("EMBEDDINGS_PROVIDER", config.embedding.provider)
   199 |         embedding_model = os.getenv(
   200 |             "EMBEDDINGS_MODEL", config.embedding.embedding_model
   201 |         )
   202 |         embedding_dims = int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))
```

### src/providers/factory.py:212 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   210 |             "=" * 60 + "\n"
   211 |             f"Embedding Provider: {embedding_provider}\n"
   212 |             f"  Model: {embedding_model}\n"
   213 |             f"  Dimensions: {embedding_dims}\n"
   214 |             f"  Task: {config.embedding.task}\n"
```

### src/registry/index_registry.py:236 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   234 |     # Get current provider config
   235 |     provider_name = os.getenv("EMBEDDINGS_PROVIDER", config.embedding.provider)
   236 |     model_name = os.getenv("EMBEDDINGS_MODEL", config.embedding.embedding_model)
   237 |     dims = int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))
   238 |     version = config.embedding.version
```

### src/shared/config.py:26 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    24 | 
    25 |     # Model configuration
    26 |     embedding_model: str = Field(
    27 |         alias="model_name"
    28 |     )  # Support both names for backwards compat
```

### src/shared/config.py:69 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    67 | 
    68 |     class Config:
    69 |         populate_by_name = True  # Allow both embedding_model and model_name
    70 | 
    71 | 
```

### src/shared/config.py:324 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   322 |     logger.info(
   323 |         f"Embedding configuration loaded: "
   324 |         f"model={config.embedding.embedding_model}, "
   325 |         f"dims={config.embedding.dims}, "
   326 |         f"version={config.embedding.version}, "
```

### tests/e2e/test_golden_set.py:189 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   187 | 
   188 |         # Initialize embedder
   189 |         embedder = SentenceTransformer(config.embedding.embedding_model)
   190 | 
   191 |         # Initialize vector store
```

### tests/e2e/test_golden_set.py:398 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   396 | 
   397 |         # Initialize embedder
   398 |         embedder = SentenceTransformer(config.embedding.embedding_model)
   399 | 
   400 |         # Initialize vector store
```

### tests/p6_t4_test.py:69 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    67 | def embedder(config):
    68 |     """Create embedder for encoding queries."""
    69 |     model_name = config.embedding.embedding_model
    70 |     return SentenceTransformer(model_name)
    71 | 
```

### tests/test_integration_prephase7.py:47 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    45 |     try:
    46 |         provider = SentenceTransformersProvider(
    47 |             model_name=config.embedding.embedding_model, expected_dims=expected_dims
    48 |         )
    49 |         assert provider.dims == expected_dims
```

### tests/test_integration_prephase7.py:59 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    57 |     try:
    58 |         bad_provider = SentenceTransformersProvider(
    59 |             model_name=config.embedding.embedding_model, expected_dims=999  # Wrong!
    60 |         )
    61 |         print(f"  ✗ FAIL: Should have raised ValueError, got dims={bad_provider.dims}")
```

### tests/test_integration_prephase7.py:271 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   269 |         print("Test 2: Provider initialization...")
   270 |         provider = SentenceTransformersProvider(
   271 |             model_name=config.embedding.embedding_model,
   272 |             expected_dims=config.embedding.dims,
   273 |         )
```

### tests/test_phase1_foundation.py:27 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    25 |         print("✓ Configuration loaded successfully")
    26 |         print(f"  - Environment: {settings.env}")
    27 |         print(f"  - Embedding model: {config.embedding.embedding_model}")
    28 |         print(f"  - Dimensions: {config.embedding.dims}")
    29 |         print(f"  - Version: {config.embedding.version}")
```

### tests/test_phase1_foundation.py:57 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    55 | 
    56 |         # Initialize provider
    57 |         print(f"Initializing provider with model: {config.embedding.embedding_model}")
    58 |         provider = SentenceTransformersProvider()
    59 | 
```

### tests/test_phase7c_index_registry.py:31 — Found reference to deprecated model v4 (should be v3) (error)
```text
    29 |             dims=1024,
    30 |             provider="jina-ai",
    31 |             model="jina-embeddings-v4",
    32 |             version="v4-2025-01-23",
    33 |             is_active=True,
```

### tests/test_phase7c_index_registry.py:41 — Found reference to deprecated model v4 (should be v3) (error)
```text
    39 |         assert index["dims"] == 1024
    40 |         assert index["provider"] == "jina-ai"
    41 |         assert index["model"] == "jina-embeddings-v4"
    42 |         assert index["is_active"] is True
    43 | 
```

### tests/test_phase7c_index_registry.py:149 — Found reference to deprecated model v4 (should be v3) (error)
```text
   147 |         )
   148 | 
   149 |         provider = MockProvider("jina-ai", "jina-embeddings-v4", 1024)
   150 | 
   151 |         # Should not raise
```

### tests/test_phase7c_index_registry.py:215 — Found reference to deprecated model v4 (should be v3) (error)
```text
   213 | 
   214 |         # Provider with 1024-D
   215 |         provider = MockProvider("jina-ai", "jina-embeddings-v4", 1024)
   216 | 
   217 |         index = registry.get_index_for_provider(provider)
```

### tests/test_phase7c_ingestion.py:43 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
    41 |         # Override to use local provider for testing (no API key needed)
    42 |         config.embedding.provider = "sentence-transformers"
    43 |         config.embedding.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    44 |         config.embedding.dims = 384  # MiniLM is 384-D
    45 |         config.embedding.version = "miniLM-L6-v2-test"
```

### tests/test_phase7c_provider_factory.py:26 — Found reference to deprecated model v4 (should be v3) (error)
```text
    24 |             provider = ProviderFactory.create_embedding_provider(
    25 |                 provider="jina-ai",
    26 |                 model="jina-embeddings-v4",
    27 |                 dims=1024,
    28 |                 task="retrieval.passage",
```

### tests/test_phase7c_provider_factory.py:33 — Found reference to deprecated model v4 (should be v3) (error)
```text
    31 |             assert isinstance(provider, EmbeddingProvider)
    32 |             assert provider.provider_name == "jina-ai"
    33 |             assert provider.model_id == "jina-embeddings-v4"
    34 |             assert provider.dims == 1024
    35 | 
```

### tests/test_phase7c_provider_factory.py:111 — Found reference to deprecated model v4 (should be v3) (error)
```text
   109 |             with pytest.raises(ValueError, match="JINA_API_KEY required"):
   110 |                 ProviderFactory.create_embedding_provider(
   111 |                     provider="jina-ai", model="jina-embeddings-v4", dims=1024
   112 |                 )
   113 | 
```

### tests/test_phase7c_provider_factory.py:118 — Found reference to deprecated model v4 (should be v3) (error)
```text
   116 |         env = {
   117 |             "EMBEDDINGS_PROVIDER": "jina-ai",
   118 |             "EMBEDDINGS_MODEL": "jina-embeddings-v4",
   119 |             "EMBEDDINGS_DIM": "1024",
   120 |             "EMBEDDINGS_TASK": "retrieval.passage",
```

### tests/test_phase7c_provider_factory.py:128 — Found reference to deprecated model v4 (should be v3) (error)
```text
   126 | 
   127 |             assert provider.provider_name == "jina-ai"
   128 |             assert provider.model_id == "jina-embeddings-v4"
   129 |             assert provider.dims == 1024
   130 | 
```

### tests/test_phase7c_provider_factory.py:135 — Found reference to deprecated model v4 (should be v3) (error)
```text
   133 |         with patch.dict(os.environ, {"JINA_API_KEY": "test-key"}):
   134 |             provider = ProviderFactory.create_embedding_provider(
   135 |                 provider="jina-ai", model="jina-embeddings-v4", dims=1024
   136 |             )
   137 | 
```

### tests/test_phase7c_provider_factory.py:141 — Found reference to deprecated model v4 (should be v3) (error)
```text
   139 | 
   140 |             assert info["provider"] == "jina-ai"
   141 |             assert info["model"] == "jina-embeddings-v4"
   142 |             assert info["dims"] == 1024
   143 | 
```

### tests/test_phase7c_provider_factory.py:152 — Found reference to deprecated model v4 (should be v3) (error)
```text
   150 |         env = {
   151 |             "EMBEDDINGS_PROVIDER": "jina-ai",
   152 |             "EMBEDDINGS_MODEL": "jina-embeddings-v4",
   153 |             "EMBEDDINGS_DIM": "1024",
   154 |             "RERANK_PROVIDER": "jina-ai",
```

### tests/test_phase7c_reranking.py:220 — Use of deprecated property name 'embedding_model' (use 'embedding_version') (error)
```text
   218 |         """Mock configuration."""
   219 |         config = MagicMock()
   220 |         config.embedding.embedding_model = "test-model"
   221 |         config.embedding.dims = 384
   222 |         config.embedding.version = "v1"
```

### tests/test_phase7c_schema_v2_1.py:388 — Found reference to deprecated model v4 (should be v3) (error)
```text
   386 |                     s.checksum = 'test',
   387 |                     s.vector_embedding = $vector,
   388 |                     s.embedding_version = 'jina-embeddings-v4',
   389 |                     s.embedding_provider = 'jina-ai',
   390 |                     s.embedding_timestamp = datetime(),
```

### config/development.yaml:16 — Reference to canonical model v3 (info)
```text
    14 | embedding:
    15 |   # Model configuration (ENV overrideable via EMBEDDINGS_MODEL)
    16 |   model_name: "jina-embeddings-v3"  # Phase 7C: Stable v3 model
    17 |   dims: 1024  # Phase 7C: v3 defaults to 1024-D
    18 |   similarity: "cosine"  # Options: cosine, dot, euclidean
```

### config/development.yaml:23 — Reference to canonical model v3 (info)
```text
    21 |   # This ensures we can filter vectors by the model version used
    22 |   # For Jina models, the model name IS the version (they don't use dated tags)
    23 |   version: "jina-embeddings-v3"  # Phase 7C: Actual Jina model identifier
    24 | 
    25 |   # Provider configuration (ENV overrideable via EMBEDDINGS_PROVIDER)
```

### docker-compose.yml:140 — Reference to canonical model v3 (info)
```text
   138 |       # Phase 7C: Embedding Provider Configuration
   139 |       - EMBEDDINGS_PROVIDER=${EMBEDDINGS_PROVIDER:-jina-ai}
   140 |       - EMBEDDINGS_MODEL=${EMBEDDINGS_MODEL:-jina-embeddings-v3}
   141 |       - EMBEDDINGS_DIM=${EMBEDDINGS_DIM:-1024}
   142 |       - EMBEDDINGS_TASK=${EMBEDDINGS_TASK:-retrieval.passage}
```

### docker-compose.yml:150 — Reference to canonical model v3 (info)
```text
   148 |       # Phase 7C Hotfix: Tokenizer Service Configuration
   149 |       - TOKENIZER_BACKEND=hf
   150 |       - HF_TOKENIZER_ID=jinaai/jina-embeddings-v3
   151 |       - HF_CACHE=/opt/hf-cache
   152 |       - TRANSFORMERS_OFFLINE=true
```

### docker-compose.yml:222 — Reference to canonical model v3 (info)
```text
   220 |       # Phase 7C: Embedding Provider Configuration
   221 |       - EMBEDDINGS_PROVIDER=${EMBEDDINGS_PROVIDER:-jina-ai}
   222 |       - EMBEDDINGS_MODEL=${EMBEDDINGS_MODEL:-jina-embeddings-v3}
   223 |       - EMBEDDINGS_DIM=${EMBEDDINGS_DIM:-1024}
   224 |       - EMBEDDINGS_TASK=${EMBEDDINGS_TASK:-retrieval.passage}
```

### docker-compose.yml:231 — Reference to canonical model v3 (info)
```text
   229 |       # Phase 7C Hotfix: Tokenizer Service Configuration
   230 |       - TOKENIZER_BACKEND=hf
   231 |       - HF_TOKENIZER_ID=jinaai/jina-embeddings-v3
   232 |       - HF_CACHE=/opt/hf-cache
   233 |       - TRANSFORMERS_OFFLINE=true
```

### docker-compose.yml:299 — Reference to canonical model v3 (info)
```text
   297 |       # Phase 7C: Embedding Provider Configuration
   298 |       - EMBEDDINGS_PROVIDER=${EMBEDDINGS_PROVIDER:-jina-ai}
   299 |       - EMBEDDINGS_MODEL=${EMBEDDINGS_MODEL:-jina-embeddings-v3}
   300 |       - EMBEDDINGS_DIM=${EMBEDDINGS_DIM:-1024}
   301 |       - EMBEDDINGS_TASK=${EMBEDDINGS_TASK:-retrieval.passage}
```

### docker-compose.yml:308 — Reference to canonical model v3 (info)
```text
   306 |       # Phase 7C Hotfix: Tokenizer Service Configuration
   307 |       - TOKENIZER_BACKEND=hf
   308 |       - HF_TOKENIZER_ID=jinaai/jina-embeddings-v3
   309 |       - HF_CACHE=/opt/hf-cache
   310 |       - TRANSFORMERS_OFFLINE=true
```

### docker/ingestion-service.Dockerfile:20 — Reference to canonical model v3 (info)
```text
    18 | RUN pip install --no-cache-dir -r requirements.txt
    19 | 
    20 | # Prefetch jina-embeddings-v3 tokenizer during build (Phase 7C hotfix)
    21 | # This eliminates runtime downloads and enables offline operation
    22 | ENV HF_HOME=/opt/hf-cache
```

### docker/ingestion-service.Dockerfile:24 — Reference to canonical model v3 (info)
```text
    22 | ENV HF_HOME=/opt/hf-cache
    23 | RUN mkdir -p /opt/hf-cache && \
    24 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    25 |     echo "Tokenizer prefetched successfully"
    26 | 
```

### docker/ingestion-service.Dockerfile:24 — HF tokenizer for Jina v3 (info)
```text
    22 | ENV HF_HOME=/opt/hf-cache
    23 | RUN mkdir -p /opt/hf-cache && \
    24 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    25 |     echo "Tokenizer prefetched successfully"
    26 | 
```

### docker/ingestion-worker.Dockerfile:19 — Reference to canonical model v3 (info)
```text
    17 | RUN pip install --no-cache-dir -r requirements.txt
    18 | 
    19 | # Prefetch jina-embeddings-v3 tokenizer during build (Phase 7C hotfix)
    20 | # This eliminates runtime downloads and enables offline operation
    21 | ENV HF_HOME=/opt/hf-cache
```

### docker/ingestion-worker.Dockerfile:23 — Reference to canonical model v3 (info)
```text
    21 | ENV HF_HOME=/opt/hf-cache
    22 | RUN mkdir -p /opt/hf-cache && \
    23 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    24 |     echo "Tokenizer prefetched successfully"
    25 | 
```

### docker/ingestion-worker.Dockerfile:23 — HF tokenizer for Jina v3 (info)
```text
    21 | ENV HF_HOME=/opt/hf-cache
    22 | RUN mkdir -p /opt/hf-cache && \
    23 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    24 |     echo "Tokenizer prefetched successfully"
    25 | 
```

### docker/mcp-server.Dockerfile:19 — Reference to canonical model v3 (info)
```text
    17 | RUN pip install --no-cache-dir -r requirements.txt
    18 | 
    19 | # Prefetch jina-embeddings-v3 tokenizer during build (Phase 7C hotfix)
    20 | # This eliminates runtime downloads and enables offline operation
    21 | ENV HF_HOME=/opt/hf-cache
```

### docker/mcp-server.Dockerfile:23 — Reference to canonical model v3 (info)
```text
    21 | ENV HF_HOME=/opt/hf-cache
    22 | RUN mkdir -p /opt/hf-cache && \
    23 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    24 |     echo "Tokenizer prefetched successfully"
    25 | 
```

### docker/mcp-server.Dockerfile:23 — HF tokenizer for Jina v3 (info)
```text
    21 | ENV HF_HOME=/opt/hf-cache
    22 | RUN mkdir -p /opt/hf-cache && \
    23 |     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    24 |     echo "Tokenizer prefetched successfully"
    25 | 
```

### requirements.txt:74 — Reference to canonical model v3 (info)
```text
    72 | 
    73 | # Phase 7C Hotfix: Tokenizer service for accurate token counting
    74 | # CRITICAL: Use jina-embeddings-v3 tokenizer (XLM-RoBERTa), NOT tiktoken (OpenAI)
    75 | transformers>=4.43.0  # HuggingFace tokenizer (primary backend)
    76 | tokenizers>=0.15.0    # Fast Rust backend for transformers
```

### scripts/apply_complete_schema_v2_1.py:79 — Presence of :Chunk label (dual-label support) (info)
```text
    77 |                 elif "MERGE (sv:SchemaVersion" in stmt:
    78 |                     results["schema_version_set"] = True
    79 |                 elif "SET s:Chunk" in stmt:
    80 |                     summary = result.consume()
    81 |                     results["dual_labeled_sections"] = summary.counters.labels_added
```

### scripts/backfill_document_tokens.py:51 — Presence of :Section label (info)
```text
    49 |         MATCH (d:Document)
    50 |         WHERE d.token_count IS NULL OR d.token_count = 0
    51 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
    52 |         WITH d, count(s) as section_count
    53 |         RETURN d.id as doc_id,
```

### scripts/backfill_document_tokens.py:67 — Presence of :Section label (info)
```text
    65 |         query = """
    66 |         MATCH (d:Document)
    67 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
    68 |         WITH d,
    69 |              count(s) as section_count,
```

### scripts/backfill_document_tokens.py:138 — Presence of :Section label (info)
```text
   136 |         logger.info("Executing backfill...")
   137 |         backfill_query = """
   138 |         MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
   139 |         WITH d, sum(s.tokens) AS section_tokens
   140 |         SET d.token_count = section_tokens
```

### scripts/baseline_distribution_analysis.py:50 — Presence of :Section label (info)
```text
    48 |         """Fetch all sections with metadata"""
    49 |         query = """
    50 |         MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
    51 |         RETURN d.id as doc_id,
    52 |                s.id as section_id,
```

### scripts/dev/seed_minimal_graph.py:60 — Qdrant cosine distance (info)
```text
    58 |         qdrant_client.create_collection(
    59 |             collection_name=collection_name,
    60 |             vectors_config=VectorParams(size=embedding_dims, distance=Distance.COSINE),
    61 |         )
    62 | 
```

### scripts/dev/seed_minimal_graph.py:66 — Use of canonical 'embedding_version' (info)
```text
    64 |     model_name = config.embedding.model_name
    65 |     embedder = SentenceTransformer(model_name)
    66 |     embedding_version = config.embedding.version
    67 | 
    68 |     print("🌱 Seeding minimal graph...")
```

### scripts/dev/seed_minimal_graph.py:181 — MERGE Document by id (canonical) (info)
```text
   179 |             session.run(
   180 |                 """
   181 |                 MERGE (d:Document {id: $id})
   182 |                 SET d += $props, d.updated_at = datetime()
   183 |             """,
```

### scripts/dev/seed_minimal_graph.py:198 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   196 |             props = section.copy()
   197 |             if vector_primary == "neo4j":
   198 |                 props["vector_embedding"] = vector
   199 |                 props["embedding_version"] = embedding_version
   200 |             props["checksum"] = hashlib.md5(section["text"].encode()).hexdigest()
```

### scripts/dev/seed_minimal_graph.py:199 — Use of canonical 'embedding_version' (info)
```text
   197 |             if vector_primary == "neo4j":
   198 |                 props["vector_embedding"] = vector
   199 |                 props["embedding_version"] = embedding_version
   200 |             props["checksum"] = hashlib.md5(section["text"].encode()).hexdigest()
   201 | 
```

### scripts/dev/seed_minimal_graph.py:199 — Use of canonical 'embedding_version' (info)
```text
   197 |             if vector_primary == "neo4j":
   198 |                 props["vector_embedding"] = vector
   199 |                 props["embedding_version"] = embedding_version
   200 |             props["checksum"] = hashlib.md5(section["text"].encode()).hexdigest()
   201 | 
```

### scripts/dev/seed_minimal_graph.py:204 — Presence of :Section label (info)
```text
   202 |             session.run(
   203 |                 """
   204 |                 MERGE (s:Section {id: $id})
   205 |                 SET s += $props, s.updated_at = datetime()
   206 |             """,
```

### scripts/dev/seed_minimal_graph.py:215 — Presence of :Section label (info)
```text
   213 |                 """
   214 |                 MATCH (d:Document {id: $doc_id})
   215 |                 MATCH (s:Section {id: $sec_id})
   216 |                 MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
   217 |             """,
```

### scripts/dev/seed_minimal_graph.py:270 — Presence of :Section label (info)
```text
   268 |         session.run(
   269 |             """
   270 |             MATCH (s:Section {id: $sec_id})
   271 |             MATCH (c:Command {id: $cmd_id})
   272 |             MERGE (s)-[:MENTIONS {confidence: 0.95, source_section_id: $sec_id}]->(c)
```

### scripts/dev/seed_minimal_graph.py:280 — Presence of :Section label (info)
```text
   278 |         session.run(
   279 |             """
   280 |             MATCH (s:Section {id: $sec_id})
   281 |             MATCH (c:Command {id: $cmd_id})
   282 |             MERGE (s)-[:MENTIONS {confidence: 0.90, source_section_id: $sec_id}]->(c)
```

### scripts/dev/seed_minimal_graph.py:290 — Presence of :Section label (info)
```text
   288 |         session.run(
   289 |             """
   290 |             MATCH (s:Section {id: $sec_id})
   291 |             MATCH (e:Error {id: $err_id})
   292 |             MERGE (s)-[:MENTIONS {confidence: 0.98, source_section_id: $sec_id}]->(e)
```

### scripts/dev/seed_minimal_graph.py:300 — Presence of :Section label (info)
```text
   298 |         session.run(
   299 |             """
   300 |             MATCH (s:Section {id: $sec_id})
   301 |             MATCH (c:Configuration {id: $cfg_id})
   302 |             MERGE (s)-[:MENTIONS {confidence: 0.92, source_section_id: $sec_id}]->(c)
```

### scripts/dev/seed_minimal_graph.py:310 — Presence of :Section label (info)
```text
   308 |         session.run(
   309 |             """
   310 |             MATCH (s:Section {id: $sec_id})
   311 |             MATCH (c:Configuration {id: $cfg_id})
   312 |             MERGE (s)-[:MENTIONS {confidence: 0.88, source_section_id: $sec_id}]->(c)
```

### scripts/dev/seed_minimal_graph.py:351 — Use of canonical 'embedding_version' (info)
```text
   349 |                         "anchor": section["anchor"],
   350 |                         "updated_at": "2024-01-15T00:00:00Z",
   351 |                         "embedding_version": embedding_version,
   352 |                     },
   353 |                 )
```

### scripts/dev/seed_minimal_graph.py:351 — Use of canonical 'embedding_version' (info)
```text
   349 |                         "anchor": section["anchor"],
   350 |                         "updated_at": "2024-01-15T00:00:00Z",
   351 |                         "embedding_version": embedding_version,
   352 |                     },
   353 |                 )
```

### scripts/neo4j/create_schema.cypher:19 — Presence of :Section label (info)
```text
    17 | // Section constraints
    18 | CREATE CONSTRAINT section_id_unique IF NOT EXISTS
    19 | FOR (s:Section) REQUIRE s.id IS UNIQUE;
    20 | 
    21 | // Domain entity constraints
```

### scripts/neo4j/create_schema.cypher:65 — Presence of :Section label (info)
```text
    63 | // Section indexes
    64 | CREATE INDEX section_document_id IF NOT EXISTS
    65 | FOR (s:Section) ON (s.document_id);
    66 | 
    67 | CREATE INDEX section_level IF NOT EXISTS
```

### scripts/neo4j/create_schema.cypher:68 — Presence of :Section label (info)
```text
    66 | 
    67 | CREATE INDEX section_level IF NOT EXISTS
    68 | FOR (s:Section) ON (s.level);
    69 | 
    70 | CREATE INDEX section_order IF NOT EXISTS
```

### scripts/neo4j/create_schema.cypher:71 — Presence of :Section label (info)
```text
    69 | 
    70 | CREATE INDEX section_order IF NOT EXISTS
    71 | FOR (s:Section) ON (s.order);
    72 | 
    73 | // Domain entity indexes (common properties)
```

### scripts/neo4j/create_schema.cypher:100 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
    98 | 
    99 | // The following indexes will be created by src/shared/schema.py:
   100 | // - section_embeddings (on Section.vector_embedding)
   101 | // - command_embeddings (on Command.vector_embedding)
   102 | // - configuration_embeddings (on Configuration.vector_embedding)
```

### scripts/neo4j/create_schema.cypher:101 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
    99 | // The following indexes will be created by src/shared/schema.py:
   100 | // - section_embeddings (on Section.vector_embedding)
   101 | // - command_embeddings (on Command.vector_embedding)
   102 | // - configuration_embeddings (on Configuration.vector_embedding)
   103 | // - procedure_embeddings (on Procedure.vector_embedding)
```

### scripts/neo4j/create_schema.cypher:102 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   100 | // - section_embeddings (on Section.vector_embedding)
   101 | // - command_embeddings (on Command.vector_embedding)
   102 | // - configuration_embeddings (on Configuration.vector_embedding)
   103 | // - procedure_embeddings (on Procedure.vector_embedding)
   104 | // - error_embeddings (on Error.vector_embedding)
```

### scripts/neo4j/create_schema.cypher:103 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   101 | // - command_embeddings (on Command.vector_embedding)
   102 | // - configuration_embeddings (on Configuration.vector_embedding)
   103 | // - procedure_embeddings (on Procedure.vector_embedding)
   104 | // - error_embeddings (on Error.vector_embedding)
   105 | // - concept_embeddings (on Concept.vector_embedding)
```

### scripts/neo4j/create_schema.cypher:104 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   102 | // - configuration_embeddings (on Configuration.vector_embedding)
   103 | // - procedure_embeddings (on Procedure.vector_embedding)
   104 | // - error_embeddings (on Error.vector_embedding)
   105 | // - concept_embeddings (on Concept.vector_embedding)
   106 | 
```

### scripts/neo4j/create_schema.cypher:105 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   103 | // - procedure_embeddings (on Procedure.vector_embedding)
   104 | // - error_embeddings (on Error.vector_embedding)
   105 | // - concept_embeddings (on Concept.vector_embedding)
   106 | 
   107 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1.cypher:21 — Presence of :Chunk label (dual-label support) (info)
```text
    19 | 
    20 | // ============================================================================
    21 | // Part 1: Dual-label existing Sections as :Chunk
    22 | // ============================================================================
    23 | // Purpose: v3 tool compatibility while preserving v2 queries
```

### scripts/neo4j/create_schema_v2_1.cypher:27 — Presence of :Section label (info)
```text
    25 | // Idempotent: SET is idempotent (no-op if label already exists)
    26 | 
    27 | MATCH (s:Section)
    28 | WHERE NOT s:Chunk
    29 | SET s:Chunk;
```

### scripts/neo4j/create_schema_v2_1.cypher:28 — Presence of :Chunk label (dual-label support) (info)
```text
    26 | 
    27 | MATCH (s:Section)
    28 | WHERE NOT s:Chunk
    29 | SET s:Chunk;
    30 | 
```

### scripts/neo4j/create_schema_v2_1.cypher:29 — Presence of :Chunk label (dual-label support) (info)
```text
    27 | MATCH (s:Section)
    28 | WHERE NOT s:Chunk
    29 | SET s:Chunk;
    30 | 
    31 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1.cypher:64 — Vector index creation (info)
```text
    62 | 
    63 | // Section vector index (primary retrieval)
    64 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
    65 | FOR (s:Section)
    66 | ON s.vector_embedding
```

### scripts/neo4j/create_schema_v2_1.cypher:65 — Presence of :Section label (info)
```text
    63 | // Section vector index (primary retrieval)
    64 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
    65 | FOR (s:Section)
    66 | ON s.vector_embedding
    67 | OPTIONS {
```

### scripts/neo4j/create_schema_v2_1.cypher:66 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
    64 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
    65 | FOR (s:Section)
    66 | ON s.vector_embedding
    67 | OPTIONS {
    68 |   indexConfig: {
```

### scripts/neo4j/create_schema_v2_1.cypher:75 — Vector index creation (info)
```text
    73 | 
    74 | // Chunk vector index (v3 compatibility - same data, dual-labeled)
    75 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
    76 | FOR (c:Chunk)
    77 | ON c.vector_embedding
```

### scripts/neo4j/create_schema_v2_1.cypher:76 — Presence of :Chunk label (dual-label support) (info)
```text
    74 | // Chunk vector index (v3 compatibility - same data, dual-labeled)
    75 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
    76 | FOR (c:Chunk)
    77 | ON c.vector_embedding
    78 | OPTIONS {
```

### scripts/neo4j/create_schema_v2_1.cypher:77 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
    75 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
    76 | FOR (c:Chunk)
    77 | ON c.vector_embedding
    78 | OPTIONS {
    79 |   indexConfig: {
```

### scripts/neo4j/create_schema_v2_1.cypher:126 — Presence of :Chunk label (dual-label support) (info)
```text
   124 | // Chunk-specific indices (mirror Section indices for v3 compatibility)
   125 | CREATE INDEX chunk_document_id IF NOT EXISTS
   126 | FOR (c:Chunk) ON (c.document_id);
   127 | 
   128 | CREATE INDEX chunk_level IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1.cypher:129 — Presence of :Chunk label (dual-label support) (info)
```text
   127 | 
   128 | CREATE INDEX chunk_level IF NOT EXISTS
   129 | FOR (c:Chunk) ON (c.level);
   130 | 
   131 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1.cypher:132 — Use of canonical 'embedding_version' (info)
```text
   130 | 
   131 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
   132 | FOR (c:Chunk) ON (c.embedding_version);
   133 | 
   134 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1.cypher:132 — Presence of :Chunk label (dual-label support) (info)
```text
   130 | 
   131 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
   132 | FOR (c:Chunk) ON (c.embedding_version);
   133 | 
   134 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1.cypher:157 — Presence of :Section label (info)
```text
   155 | 
   156 | // -- 1. Verify dual-labeling (Section and Chunk counts should match)
   157 | // MATCH (s:Section)
   158 | // WITH count(s) as section_count
   159 | // MATCH (c:Chunk)
```

### scripts/neo4j/create_schema_v2_1.cypher:159 — Presence of :Chunk label (dual-label support) (info)
```text
   157 | // MATCH (s:Section)
   158 | // WITH count(s) as section_count
   159 | // MATCH (c:Chunk)
   160 | // RETURN section_count, count(c) as chunk_count,
   161 | //        CASE WHEN section_count = count(c) THEN 'PASS' ELSE 'FAIL' END as test;
```

### scripts/neo4j/create_schema_v2_1.cypher:188 — Presence of :Section label (info)
```text
   186 | 
   187 | // -- 5. Count Sections with missing embedding fields (should be 0)
   188 | // MATCH (s:Section)
   189 | // WHERE s.vector_embedding IS NULL
   190 | //    OR s.embedding_version IS NULL
```

### scripts/neo4j/create_schema_v2_1.cypher:189 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   187 | // -- 5. Count Sections with missing embedding fields (should be 0)
   188 | // MATCH (s:Section)
   189 | // WHERE s.vector_embedding IS NULL
   190 | //    OR s.embedding_version IS NULL
   191 | //    OR s.embedding_provider IS NULL
```

### scripts/neo4j/create_schema_v2_1.cypher:190 — Use of canonical 'embedding_version' (info)
```text
   188 | // MATCH (s:Section)
   189 | // WHERE s.vector_embedding IS NULL
   190 | //    OR s.embedding_version IS NULL
   191 | //    OR s.embedding_provider IS NULL
   192 | //    OR s.embedding_timestamp IS NULL
```

### scripts/neo4j/create_schema_v2_1.cypher:192 — Use of canonical 'embedding_timestamp' (info)
```text
   190 | //    OR s.embedding_version IS NULL
   191 | //    OR s.embedding_provider IS NULL
   192 | //    OR s.embedding_timestamp IS NULL
   193 | //    OR s.embedding_dimensions IS NULL
   194 | // RETURN count(s) as sections_missing_embeddings;
```

### scripts/neo4j/create_schema_v2_1.cypher:193 — Use of canonical 'embedding_dimensions' (info)
```text
   191 | //    OR s.embedding_provider IS NULL
   192 | //    OR s.embedding_timestamp IS NULL
   193 | //    OR s.embedding_dimensions IS NULL
   194 | // RETURN count(s) as sections_missing_embeddings;
   195 | 
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:28 — Presence of :Chunk label (dual-label support) (info)
```text
    26 | // COMPATIBILITY:
    27 | // - Section queries work unchanged
    28 | // - Dual-labeled as :Chunk for v3 tool compatibility
    29 | // - All 12 domain entity types preserved
    30 | // - Backward compatible with v2.0 queries
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:49 — Presence of :Chunk label (dual-label support) (info)
```text
    47 | 
    48 | // Section constraints
    49 | // NOTE: Sections are dual-labeled as :Chunk for v3 compatibility
    50 | // Both labels share the same uniqueness constraint
    51 | CREATE CONSTRAINT section_id_unique IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:52 — Presence of :Section label (info)
```text
    50 | // Both labels share the same uniqueness constraint
    51 | CREATE CONSTRAINT section_id_unique IF NOT EXISTS
    52 | FOR (s:Section) REQUIRE s.id IS UNIQUE;
    53 | 
    54 | // Domain entity constraints (12 types preserved from v2.0)
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:114 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   112 | // REQUIRED FIELDS (enforced in application code):
   113 | // Section nodes:
   114 | //   - vector_embedding (List<Float>) - CRITICAL for hybrid search
   115 | //   - embedding_version (String) - Provenance tracking
   116 | //   - embedding_provider (String) - Provider identification
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:115 — Use of canonical 'embedding_version' (info)
```text
   113 | // Section nodes:
   114 | //   - vector_embedding (List<Float>) - CRITICAL for hybrid search
   115 | //   - embedding_version (String) - Provenance tracking
   116 | //   - embedding_provider (String) - Provider identification
   117 | //   - embedding_dimensions (Integer) - Dimension validation
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:117 — Use of canonical 'embedding_dimensions' (info)
```text
   115 | //   - embedding_version (String) - Provenance tracking
   116 | //   - embedding_provider (String) - Provider identification
   117 | //   - embedding_dimensions (Integer) - Dimension validation
   118 | //   - embedding_timestamp (DateTime) - Freshness tracking
   119 | //
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:118 — Use of canonical 'embedding_timestamp' (info)
```text
   116 | //   - embedding_provider (String) - Provider identification
   117 | //   - embedding_dimensions (Integer) - Dimension validation
   118 | //   - embedding_timestamp (DateTime) - Freshness tracking
   119 | //
   120 | // Query nodes:
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:147 — Presence of :Section label (info)
```text
   145 | // Section indexes (primary retrieval path)
   146 | CREATE INDEX section_document_id IF NOT EXISTS
   147 | FOR (s:Section) ON (s.document_id);
   148 | 
   149 | CREATE INDEX section_level IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:150 — Presence of :Section label (info)
```text
   148 | 
   149 | CREATE INDEX section_level IF NOT EXISTS
   150 | FOR (s:Section) ON (s.level);
   151 | 
   152 | CREATE INDEX section_order IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:153 — Presence of :Section label (info)
```text
   151 | 
   152 | CREATE INDEX section_order IF NOT EXISTS
   153 | FOR (s:Section) ON (s.order);
   154 | 
   155 | // Domain entity indexes (for entity-specific queries)
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:211 — Presence of :Chunk label (dual-label support) (info)
```text
   209 | // PART 5: CHUNK PROPERTY INDEXES (NEW in v2.1 - Dual-label Support)
   210 | // ============================================================================
   211 | // Purpose: v3 tool compatibility - mirrors Section indexes for :Chunk label
   212 | // Impact: Queries using :Chunk label perform equally well
   213 | // Note: Same physical nodes as Section, just accessible via :Chunk label
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:212 — Presence of :Chunk label (dual-label support) (info)
```text
   210 | // ============================================================================
   211 | // Purpose: v3 tool compatibility - mirrors Section indexes for :Chunk label
   212 | // Impact: Queries using :Chunk label perform equally well
   213 | // Note: Same physical nodes as Section, just accessible via :Chunk label
   214 | // Idempotent: IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:213 — Presence of :Chunk label (dual-label support) (info)
```text
   211 | // Purpose: v3 tool compatibility - mirrors Section indexes for :Chunk label
   212 | // Impact: Queries using :Chunk label perform equally well
   213 | // Note: Same physical nodes as Section, just accessible via :Chunk label
   214 | // Idempotent: IF NOT EXISTS
   215 | 
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:217 — Presence of :Chunk label (dual-label support) (info)
```text
   215 | 
   216 | CREATE INDEX chunk_document_id IF NOT EXISTS
   217 | FOR (c:Chunk) ON (c.document_id);
   218 | 
   219 | CREATE INDEX chunk_level IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:220 — Presence of :Chunk label (dual-label support) (info)
```text
   218 | 
   219 | CREATE INDEX chunk_level IF NOT EXISTS
   220 | FOR (c:Chunk) ON (c.level);
   221 | 
   222 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:223 — Use of canonical 'embedding_version' (info)
```text
   221 | 
   222 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
   223 | FOR (c:Chunk) ON (c.embedding_version);
   224 | 
   225 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:223 — Presence of :Chunk label (dual-label support) (info)
```text
   221 | 
   222 | CREATE INDEX chunk_embedding_version IF NOT EXISTS
   223 | FOR (c:Chunk) ON (c.embedding_version);
   224 | 
   225 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:246 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   244 | // 2. chunk_embeddings_v2 - v3 compatibility path (Chunk label)
   245 | //
   246 | // Both indexes point to the same vector_embedding property on the same nodes.
   247 | // This enables backward compatibility and v3 tool support without data duplication.
   248 | //
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:252 — Vector index creation (info)
```text
   250 | 
   251 | // Section vector index (primary retrieval path)
   252 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
   253 | FOR (s:Section)
   254 | ON s.vector_embedding
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:253 — Presence of :Section label (info)
```text
   251 | // Section vector index (primary retrieval path)
   252 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
   253 | FOR (s:Section)
   254 | ON s.vector_embedding
   255 | OPTIONS {
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:254 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   252 | CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
   253 | FOR (s:Section)
   254 | ON s.vector_embedding
   255 | OPTIONS {
   256 |   indexConfig: {
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:263 — Vector index creation (info)
```text
   261 | 
   262 | // Chunk vector index (v3 compatibility - same data, dual-labeled)
   263 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
   264 | FOR (c:Chunk)
   265 | ON c.vector_embedding
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:264 — Presence of :Chunk label (dual-label support) (info)
```text
   262 | // Chunk vector index (v3 compatibility - same data, dual-labeled)
   263 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
   264 | FOR (c:Chunk)
   265 | ON c.vector_embedding
   266 | OPTIONS {
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:265 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   263 | CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
   264 | FOR (c:Chunk)
   265 | ON c.vector_embedding
   266 | OPTIONS {
   267 |   indexConfig: {
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:276 — Presence of :Chunk label (dual-label support) (info)
```text
   274 | // PART 7: DUAL-LABELING EXISTING SECTIONS (Migration-safe)
   275 | // ============================================================================
   276 | // Purpose: Add :Chunk label to existing :Section nodes for v3 compatibility
   277 | // Impact: Enables v3 tools to access sections via :Chunk label
   278 | // Pattern: Dual-labeling (not renaming) - both labels coexist
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:276 — Presence of :Section label (info)
```text
   274 | // PART 7: DUAL-LABELING EXISTING SECTIONS (Migration-safe)
   275 | // ============================================================================
   276 | // Purpose: Add :Chunk label to existing :Section nodes for v3 compatibility
   277 | // Impact: Enables v3 tools to access sections via :Chunk label
   278 | // Pattern: Dual-labeling (not renaming) - both labels coexist
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:277 — Presence of :Chunk label (dual-label support) (info)
```text
   275 | // ============================================================================
   276 | // Purpose: Add :Chunk label to existing :Section nodes for v3 compatibility
   277 | // Impact: Enables v3 tools to access sections via :Chunk label
   278 | // Pattern: Dual-labeling (not renaming) - both labels coexist
   279 | // Idempotent: WHERE NOT s:Chunk prevents redundant labeling
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:279 — Presence of :Chunk label (dual-label support) (info)
```text
   277 | // Impact: Enables v3 tools to access sections via :Chunk label
   278 | // Pattern: Dual-labeling (not renaming) - both labels coexist
   279 | // Idempotent: WHERE NOT s:Chunk prevents redundant labeling
   280 | //
   281 | // IMPORTANT:
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:283 — Presence of :Chunk label (dual-label support) (info)
```text
   281 | // IMPORTANT:
   282 | // For FRESH INSTALLATIONS, this statement is a no-op (no Sections exist yet).
   283 | // For MIGRATIONS, this adds :Chunk label to existing :Section nodes.
   284 | //
   285 | // Section queries continue working unchanged:
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:283 — Presence of :Section label (info)
```text
   281 | // IMPORTANT:
   282 | // For FRESH INSTALLATIONS, this statement is a no-op (no Sections exist yet).
   283 | // For MIGRATIONS, this adds :Chunk label to existing :Section nodes.
   284 | //
   285 | // Section queries continue working unchanged:
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:286 — Presence of :Section label (info)
```text
   284 | //
   285 | // Section queries continue working unchanged:
   286 | //   MATCH (s:Section) WHERE ... - works as before
   287 | // Chunk queries now also work:
   288 | //   MATCH (c:Chunk) WHERE ... - accesses same nodes
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:288 — Presence of :Chunk label (dual-label support) (info)
```text
   286 | //   MATCH (s:Section) WHERE ... - works as before
   287 | // Chunk queries now also work:
   288 | //   MATCH (c:Chunk) WHERE ... - accesses same nodes
   289 | //
   290 | // This is safe to run even on fresh installations (no harm, just no effect).
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:292 — Presence of :Section label (info)
```text
   290 | // This is safe to run even on fresh installations (no harm, just no effect).
   291 | 
   292 | MATCH (s:Section)
   293 | WHERE NOT s:Chunk
   294 | SET s:Chunk;
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:293 — Presence of :Chunk label (dual-label support) (info)
```text
   291 | 
   292 | MATCH (s:Section)
   293 | WHERE NOT s:Chunk
   294 | SET s:Chunk;
   295 | 
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:294 — Presence of :Chunk label (dual-label support) (info)
```text
   292 | MATCH (s:Section)
   293 | WHERE NOT s:Chunk
   294 | SET s:Chunk;
   295 | 
   296 | // ============================================================================
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:331 — Presence of :Section label (info)
```text
   329 | 
   330 | // -- 1. Verify dual-labeling (Section and Chunk counts should match)
   331 | // MATCH (s:Section)
   332 | // WITH count(s) as section_count
   333 | // MATCH (c:Chunk)
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:333 — Presence of :Chunk label (dual-label support) (info)
```text
   331 | // MATCH (s:Section)
   332 | // WITH count(s) as section_count
   333 | // MATCH (c:Chunk)
   334 | // RETURN section_count, count(c) as chunk_count,
   335 | //        CASE WHEN section_count = count(c) THEN 'PASS ✓' ELSE 'FAIL ✗' END as dual_label_test;
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:379 — Presence of :Section label (info)
```text
   377 | 
   378 | // -- 7. Validate Section embedding completeness (should return 0)
   379 | // MATCH (s:Section)
   380 | // WHERE s.vector_embedding IS NULL
   381 | //    OR s.embedding_version IS NULL
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:380 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   378 | // -- 7. Validate Section embedding completeness (should return 0)
   379 | // MATCH (s:Section)
   380 | // WHERE s.vector_embedding IS NULL
   381 | //    OR s.embedding_version IS NULL
   382 | //    OR s.embedding_provider IS NULL
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:381 — Use of canonical 'embedding_version' (info)
```text
   379 | // MATCH (s:Section)
   380 | // WHERE s.vector_embedding IS NULL
   381 | //    OR s.embedding_version IS NULL
   382 | //    OR s.embedding_provider IS NULL
   383 | //    OR s.embedding_timestamp IS NULL
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:383 — Use of canonical 'embedding_timestamp' (info)
```text
   381 | //    OR s.embedding_version IS NULL
   382 | //    OR s.embedding_provider IS NULL
   383 | //    OR s.embedding_timestamp IS NULL
   384 | //    OR s.embedding_dimensions IS NULL
   385 | // RETURN count(s) as sections_missing_required_embedding_fields,
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:384 — Use of canonical 'embedding_dimensions' (info)
```text
   382 | //    OR s.embedding_provider IS NULL
   383 | //    OR s.embedding_timestamp IS NULL
   384 | //    OR s.embedding_dimensions IS NULL
   385 | // RETURN count(s) as sections_missing_required_embedding_fields,
   386 | //        CASE WHEN count(s) = 0 THEN 'PASS ✓' ELSE 'FAIL ✗' END as validation_test;
```

### scripts/test/debug_explain.py:21 — Presence of :Section label (info)
```text
    19 |             session.run(
    20 |                 """
    21 |                 MERGE (s:Section {id: 'debug-sec-1', text: 'test'})
    22 |             """
    23 |             ).consume()
```

### scripts/test/debug_explain.py:26 — Presence of :Section label (info)
```text
    24 | 
    25 |             # Run EXPLAIN
    26 |             result = session.run("EXPLAIN MATCH (s:Section) RETURN s LIMIT 5")
    27 |             summary = result.consume()
    28 |             plan = summary.plan
```

### scripts/test_jina_payload_limits.py:30 — Reference to canonical model v3 (info)
```text
    28 | 
    29 |     payload = {
    30 |         "model": "jina-embeddings-v3",
    31 |         "task": "retrieval.passage",
    32 |         "input": ["Short test text for validation"],
```

### scripts/test_jina_payload_limits.py:85 — Reference to canonical model v3 (info)
```text
    83 | 
    84 |     payload = {
    85 |         "model": "jina-embeddings-v3",
    86 |         "task": "retrieval.passage",
    87 |         "input": texts,
```

### scripts/test_jina_payload_limits.py:146 — Reference to canonical model v3 (info)
```text
   144 | 
   145 |     payload = {
   146 |         "model": "jina-embeddings-v3",
   147 |         "task": "retrieval.passage",
   148 |         "input": texts,
```

### scripts/validate_token_accounting.py:65 — Presence of :Section label (info)
```text
    63 |         query = """
    64 |         MATCH (d:Document)
    65 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
    66 |         WITH d.id AS doc_id,
    67 |              d.token_count AS doc_tokens,
```

### scripts/validate_token_accounting.py:181 — Presence of :Section label (info)
```text
   179 |         MATCH (d:Document)
   180 |         WHERE d.token_count IS NULL OR d.token_count = 0
   181 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   182 |         RETURN d.id as doc_id,
   183 |                d.token_count as token_count,
```

### src/ingestion/api.py:12 — Use of canonical 'embedding_version' (info)
```text
    10 |     *,
    11 |     embedding_model: Optional[str] = None,
    12 |     embedding_version: Optional[str] = None,
    13 | ) -> Dict[str, Any]:
    14 |     """
```

### src/ingestion/api.py:24 — Use of canonical 'embedding_version' (info)
```text
    22 |         format=fmt,
    23 |         embedding_model=embedding_model,
    24 |         embedding_version=embedding_version,
    25 |     )
```

### src/ingestion/api.py:24 — Use of canonical 'embedding_version' (info)
```text
    22 |         format=fmt,
    23 |         embedding_model=embedding_model,
    24 |         embedding_version=embedding_version,
    25 |     )
```

### src/ingestion/auto/cli.py:418 — Pattern-scan deletion (fallback) (info)
```text
   416 |         # Scan for all state keys
   417 |         job_states = []
   418 |         for key in redis_client.scan_iter(f"{queue.STATE_PREFIX}*", count=100):
   419 |             job_id = key.split(":")[-1]
   420 |             state = queue.get_state(job_id)
```

### src/ingestion/auto/cli.py:828 — Use of canonical 'embedding_version' (info)
```text
   826 |             vectors = report.get("vector", {})
   827 |             print(f"  Sections indexed: {vectors.get('sections_indexed', 0)}")
   828 |             print(f"  Embedding version: {vectors.get('embedding_version', 'N/A')}")
   829 | 
   830 |             print(f"\nDrift: {report.get('drift_pct', 0):.2f}%")
```

### src/ingestion/auto/orchestrator.py:350 — Use of canonical 'embedding_version' (info)
```text
   348 |             self.qdrant,
   349 |             collection_name=self.config.search.vector.qdrant.collection_name,
   350 |             embedding_version=self.config.embedding.version,
   351 |         )
   352 | 
```

### src/ingestion/auto/orchestrator.py:449 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   447 | 
   448 |             # Store in section (not yet persisted to vector store)
   449 |             section["vector_embedding"] = embedding
   450 |             section["embedding_version"] = self.config.embedding.version
   451 |             embeddings_computed += 1
```

### src/ingestion/auto/orchestrator.py:450 — Use of canonical 'embedding_version' (info)
```text
   448 |             # Store in section (not yet persisted to vector store)
   449 |             section["vector_embedding"] = embedding
   450 |             section["embedding_version"] = self.config.embedding.version
   451 |             embeddings_computed += 1
   452 | 
```

### src/ingestion/auto/orchestrator.py:794 — Use of canonical 'embedding_version' (info)
```text
   792 |             {"key": "node_label", "match": {"value": "Section"}},
   793 |             {
   794 |                 "key": "embedding_version",
   795 |                 "match": {"value": self.config.embedding.version},
   796 |             },
```

### src/ingestion/auto/orchestrator.py:832 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   830 |         points = []
   831 |         for section in sections:
   832 |             embedding = section.get("vector_embedding")
   833 |             if not embedding:
   834 |                 continue
```

### src/ingestion/auto/orchestrator.py:850 — Use of canonical 'embedding_version' (info)
```text
   848 |                     "title": section.get("title"),
   849 |                     "anchor": section.get("anchor"),
   850 |                     "embedding_version": self.config.embedding.version,
   851 |                 },
   852 |             )
```

### src/ingestion/auto/orchestrator.py:869 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   867 |         with self.neo4j.session() as session:
   868 |             for section in sections:
   869 |                 embedding = section.get("vector_embedding")
   870 |                 if not embedding:
   871 |                     continue
```

### src/ingestion/auto/orchestrator.py:874 — Presence of :Section label (info)
```text
   872 | 
   873 |                 query = """
   874 |                 MATCH (s:Section {id: $section_id})
   875 |                 SET s.vector_embedding = $embedding,
   876 |                     s.embedding_version = $version
```

### src/ingestion/auto/orchestrator.py:875 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   873 |                 query = """
   874 |                 MATCH (s:Section {id: $section_id})
   875 |                 SET s.vector_embedding = $embedding,
   876 |                     s.embedding_version = $version
   877 |                 """
```

### src/ingestion/auto/orchestrator.py:876 — Use of canonical 'embedding_version' (info)
```text
   874 |                 MATCH (s:Section {id: $section_id})
   875 |                 SET s.vector_embedding = $embedding,
   876 |                     s.embedding_version = $version
   877 |                 """
   878 |                 session.run(
```

### src/ingestion/auto/orchestrator.py:885 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   883 |                 )
   884 | 
   885 |         return len([s for s in sections if s.get("vector_embedding")])
   886 | 
   887 |     def _set_embedding_metadata_in_neo4j(self, sections: List[Dict]):
```

### src/ingestion/auto/orchestrator.py:888 — Use of canonical 'embedding_version' (info)
```text
   886 | 
   887 |     def _set_embedding_metadata_in_neo4j(self, sections: List[Dict]):
   888 |         """Set embedding_version metadata in Neo4j without storing vectors."""
   889 |         with self.neo4j.session() as session:
   890 |             for section in sections:
```

### src/ingestion/auto/orchestrator.py:892 — Presence of :Section label (info)
```text
   890 |             for section in sections:
   891 |                 query = """
   892 |                 MATCH (s:Section {id: $section_id})
   893 |                 SET s.embedding_version = $version
   894 |                 """
```

### src/ingestion/auto/orchestrator.py:893 — Use of canonical 'embedding_version' (info)
```text
   891 |                 query = """
   892 |                 MATCH (s:Section {id: $section_id})
   893 |                 SET s.embedding_version = $version
   894 |                 """
   895 |                 session.run(
```

### src/ingestion/auto/report.py:32 — Use of canonical 'embedding_version' (info)
```text
    30 |         self.config = config
    31 |         self.qdrant_client = qdrant_client
    32 |         self.embedding_version = config.embedding.version
    33 |         self.vector_primary = config.search.vector.primary
    34 | 
```

### src/ingestion/auto/report.py:137 — Presence of :Section label (info)
```text
   135 |                         count(n) AS total_nodes,
   136 |                         count{(n)-[]->()} AS total_rels,
   137 |                         count{(n:Section)} AS sections,
   138 |                         count{(n:Document)} AS documents
   139 |                     """
```

### src/ingestion/auto/report.py:170 — Use of canonical 'embedding_version' (info)
```text
   168 |                     "sot": "qdrant",
   169 |                     "sections_indexed": coll_info.points_count,
   170 |                     "embedding_version": self.embedding_version,
   171 |                 }
   172 | 
```

### src/ingestion/auto/report.py:170 — Use of canonical 'embedding_version' (info)
```text
   168 |                     "sot": "qdrant",
   169 |                     "sections_indexed": coll_info.points_count,
   170 |                     "embedding_version": self.embedding_version,
   171 |                 }
   172 | 
```

### src/ingestion/auto/report.py:177 — Presence of :Section label (info)
```text
   175 |                     result = session.run(
   176 |                         """
   177 |                         MATCH (s:Section)
   178 |                         WHERE s.vector_embedding IS NOT NULL
   179 |                           AND s.embedding_version = $version
```

### src/ingestion/auto/report.py:178 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   176 |                         """
   177 |                         MATCH (s:Section)
   178 |                         WHERE s.vector_embedding IS NOT NULL
   179 |                           AND s.embedding_version = $version
   180 |                         RETURN count(s) AS count
```

### src/ingestion/auto/report.py:179 — Use of canonical 'embedding_version' (info)
```text
   177 |                         MATCH (s:Section)
   178 |                         WHERE s.vector_embedding IS NOT NULL
   179 |                           AND s.embedding_version = $version
   180 |                         RETURN count(s) AS count
   181 |                         """,
```

### src/ingestion/auto/report.py:182 — Use of canonical 'embedding_version' (info)
```text
   180 |                         RETURN count(s) AS count
   181 |                         """,
   182 |                         version=self.embedding_version,
   183 |                     )
   184 |                     record = result.single()
```

### src/ingestion/auto/report.py:190 — Use of canonical 'embedding_version' (info)
```text
   188 |                     "sot": "neo4j",
   189 |                     "sections_indexed": count,
   190 |                     "embedding_version": self.embedding_version,
   191 |                 }
   192 | 
```

### src/ingestion/auto/report.py:190 — Use of canonical 'embedding_version' (info)
```text
   188 |                     "sot": "neo4j",
   189 |                     "sections_indexed": count,
   190 |                     "embedding_version": self.embedding_version,
   191 |                 }
   192 | 
```

### src/ingestion/auto/report.py:199 — Use of canonical 'embedding_version' (info)
```text
   197 |             "sot": self.vector_primary,
   198 |             "sections_indexed": 0,
   199 |             "embedding_version": self.embedding_version,
   200 |         }
   201 | 
```

### src/ingestion/auto/report.py:199 — Use of canonical 'embedding_version' (info)
```text
   197 |             "sot": self.vector_primary,
   198 |             "sections_indexed": 0,
   199 |             "embedding_version": self.embedding_version,
   200 |         }
   201 | 
```

### src/ingestion/auto/report.py:230 — Use of canonical 'embedding_version' (info)
```text
   228 |             f"- **Primary:** {report['vector']['sot']}",
   229 |             f"- **Sections Indexed:** {report['vector']['sections_indexed']}",
   230 |             f"- **Embedding Version:** `{report['vector']['embedding_version']}`",
   231 |             "",
   232 |             "## Drift Analysis",
```

### src/ingestion/auto/verification.py:43 — Use of canonical 'embedding_version' (info)
```text
    41 |         self.qdrant_client = qdrant_client
    42 |         self.search_engine = search_engine
    43 |         self.embedding_version = config.embedding.version
    44 |         self.vector_primary = config.search.vector.primary
    45 | 
```

### src/ingestion/auto/verification.py:99 — Presence of :Section label (info)
```text
    97 |                 result = session.run(
    98 |                     """
    99 |                     MATCH (s:Section)
   100 |                     WHERE s.embedding_version = $version
   101 |                     RETURN count(s) AS graph_count
```

### src/ingestion/auto/verification.py:100 — Use of canonical 'embedding_version' (info)
```text
    98 |                     """
    99 |                     MATCH (s:Section)
   100 |                     WHERE s.embedding_version = $version
   101 |                     RETURN count(s) AS graph_count
   102 |                     """,
```

### src/ingestion/auto/verification.py:103 — Use of canonical 'embedding_version' (info)
```text
   101 |                     RETURN count(s) AS graph_count
   102 |                     """,
   103 |                     version=self.embedding_version,
   104 |                 )
   105 |                 record = result.single()
```

### src/ingestion/auto/verification.py:121 — Presence of :Section label (info)
```text
   119 |                     result = session.run(
   120 |                         """
   121 |                         MATCH (s:Section)
   122 |                         WHERE s.vector_embedding IS NOT NULL
   123 |                           AND s.embedding_version = $version
```

### src/ingestion/auto/verification.py:122 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   120 |                         """
   121 |                         MATCH (s:Section)
   122 |                         WHERE s.vector_embedding IS NOT NULL
   123 |                           AND s.embedding_version = $version
   124 |                         RETURN count(s) AS vector_count
```

### src/ingestion/auto/verification.py:123 — Use of canonical 'embedding_version' (info)
```text
   121 |                         MATCH (s:Section)
   122 |                         WHERE s.vector_embedding IS NOT NULL
   123 |                           AND s.embedding_version = $version
   124 |                         RETURN count(s) AS vector_count
   125 |                         """,
```

### src/ingestion/auto/verification.py:126 — Use of canonical 'embedding_version' (info)
```text
   124 |                         RETURN count(s) AS vector_count
   125 |                         """,
   126 |                         version=self.embedding_version,
   127 |                     )
   128 |                     record = result.single()
```

### src/ingestion/build_graph.py:44 — Use of canonical 'embedding_version' (info)
```text
    42 | 
    43 |         self.embedder = None
    44 |         self.embedding_version = config.embedding.version
    45 |         self.vector_primary = config.search.vector.primary
    46 |         self.dual_write = config.search.vector.dual_write
```

### src/ingestion/build_graph.py:141 — MERGE Document by id (canonical) (info)
```text
   139 |         """Upsert Document node."""
   140 |         query = """
   141 |         MERGE (d:Document {id: $id})
   142 |         SET d.source_uri = $source_uri,
   143 |             d.source_type = $source_type,
```

### src/ingestion/build_graph.py:158 — Presence of :Chunk label (dual-label support) (info)
```text
   156 |         Upsert Section nodes with dual-labeling and HAS_SECTION relationships in batches.
   157 | 
   158 |         Phase 7C.7: Dual-label as Section:Chunk for v3 compatibility (Session 06-08).
   159 |         Embedding metadata will be set later in _process_embeddings after vectors are generated.
   160 |         """
```

### src/ingestion/build_graph.py:167 — Presence of :Chunk label (dual-label support) (info)
```text
   165 |             batch = sections[i : i + batch_size]
   166 | 
   167 |             # Phase 7C.7: MERGE with dual-label :Section:Chunk
   168 |             query = """
   169 |             UNWIND $sections as sec
```

### src/ingestion/build_graph.py:167 — Presence of :Section label (info)
```text
   165 |             batch = sections[i : i + batch_size]
   166 | 
   167 |             # Phase 7C.7: MERGE with dual-label :Section:Chunk
   168 |             query = """
   169 |             UNWIND $sections as sec
```

### src/ingestion/build_graph.py:170 — Presence of :Chunk label (dual-label support) (info)
```text
   168 |             query = """
   169 |             UNWIND $sections as sec
   170 |             MERGE (s:Section:Chunk {id: sec.id})
   171 |             SET s.document_id = sec.document_id,
   172 |                 s.level = sec.level,
```

### src/ingestion/build_graph.py:170 — Presence of :Section label (info)
```text
   168 |             query = """
   169 |             UNWIND $sections as sec
   170 |             MERGE (s:Section:Chunk {id: sec.id})
   171 |             SET s.document_id = sec.document_id,
   172 |                 s.level = sec.level,
```

### src/ingestion/build_graph.py:218 — Presence of :Section label (info)
```text
   216 |         # Step 1: Find orphaned sections (not in current document version)
   217 |         find_orphans_query = """
   218 |         MATCH (d:Document {id: $document_id})-[r:HAS_SECTION]->(s:Section)
   219 |         WHERE NOT s.id IN $section_ids
   220 | 
```

### src/ingestion/build_graph.py:259 — Presence of :Section label (info)
```text
   257 |         if to_delete:
   258 |             delete_query = """
   259 |             MATCH (s:Section)
   260 |             WHERE s.id IN $section_ids
   261 |             DETACH DELETE s
```

### src/ingestion/build_graph.py:276 — Presence of :Section label (info)
```text
   274 |         if to_mark_stale:
   275 |             mark_stale_query = """
   276 |             MATCH (s:Section)
   277 |             WHERE s.id IN $section_ids
   278 |             SET s.is_stale = true,
```

### src/ingestion/build_graph.py:396 — Presence of :Section label (info)
```text
   394 |             query = """
   395 |             UNWIND $mentions as m
   396 |             MATCH (s:Section {id: m.section_id})
   397 |             MATCH (e {id: m.entity_id})
   398 |             MERGE (s)-[r:MENTIONS {entity_id: m.entity_id}]->(e)
```

### src/ingestion/build_graph.py:529 — Use of canonical 'embedding_version' (info)
```text
   527 |                 {"key": "node_label", "match": {"value": "Section"}},
   528 |                 {
   529 |                     "key": "embedding_version",
   530 |                     "match": {"value": self.embedding_version},
   531 |                 },
```

### src/ingestion/build_graph.py:530 — Use of canonical 'embedding_version' (info)
```text
   528 |                 {
   529 |                     "key": "embedding_version",
   530 |                     "match": {"value": self.embedding_version},
   531 |                 },
   532 |             ]
```

### src/ingestion/build_graph.py:592 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   590 |                 if not embedding or len(embedding) == 0:
   591 |                     raise ValueError(
   592 |                         f"Section {section['id']} missing REQUIRED vector_embedding. "
   593 |                         "Ingestion blocked - embeddings are mandatory in hybrid system."
   594 |                     )
```

### src/ingestion/build_graph.py:610 — Use of canonical 'embedding_version' (info)
```text
   608 |                         embedding,
   609 |                         {
   610 |                             "embedding_version": self.config.embedding.version,
   611 |                             "embedding_provider": self.embedder.provider_name,
   612 |                             "embedding_dimensions": len(embedding),
```

### src/ingestion/build_graph.py:612 — Use of canonical 'embedding_dimensions' (info)
```text
   610 |                             "embedding_version": self.config.embedding.version,
   611 |                             "embedding_provider": self.embedder.provider_name,
   612 |                             "embedding_dimensions": len(embedding),
   613 |                             "embedding_task": getattr(
   614 |                                 self.embedder, "task", "retrieval.passage"
```

### src/ingestion/build_graph.py:616 — Use of canonical 'embedding_timestamp' (info)
```text
   614 |                                 self.embedder, "task", "retrieval.passage"
   615 |                             ),
   616 |                             "embedding_timestamp": datetime.utcnow(),
   617 |                         },
   618 |                     )
```

### src/ingestion/build_graph.py:666 — Qdrant cosine distance (info)
```text
   664 |                         size=dimensions,
   665 |                         distance=(
   666 |                             Distance.COSINE
   667 |                             if self.config.embedding.similarity == "cosine"
   668 |                             else Distance.EUCLID
```

### src/ingestion/build_graph.py:745 — Use of canonical 'embedding_version' (info)
```text
   743 |                 "anchor": section.get("anchor"),
   744 |                 # Phase 7C.7: Embedding metadata for provenance tracking
   745 |                 "embedding_version": emb_version,
   746 |                 "embedding_provider": emb_provider,
   747 |                 "embedding_dimensions": emb_dimensions,
```

### src/ingestion/build_graph.py:747 — Use of canonical 'embedding_dimensions' (info)
```text
   745 |                 "embedding_version": emb_version,
   746 |                 "embedding_provider": emb_provider,
   747 |                 "embedding_dimensions": emb_dimensions,
   748 |                 "embedding_task": emb_task,
   749 |                 "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
```

### src/ingestion/build_graph.py:749 — Use of canonical 'embedding_timestamp' (info)
```text
   747 |                 "embedding_dimensions": emb_dimensions,
   748 |                 "embedding_task": emb_task,
   749 |                 "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
   750 |                 "updated_at": datetime.utcnow().isoformat() + "Z",
   751 |             },
```

### src/ingestion/build_graph.py:774 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   772 |         query = f"""
   773 |         MATCH (n:{label} {{id: $node_id}})
   774 |         SET n.vector_embedding = $embedding,
   775 |             n.embedding_version = $version
   776 |         RETURN n.id as id
```

### src/ingestion/build_graph.py:775 — Use of canonical 'embedding_version' (info)
```text
   773 |         MATCH (n:{label} {{id: $node_id}})
   774 |         SET n.vector_embedding = $embedding,
   775 |             n.embedding_version = $version
   776 |         RETURN n.id as id
   777 |         """
```

### src/ingestion/build_graph.py:784 — Use of canonical 'embedding_version' (info)
```text
   782 |                 node_id=node_id,
   783 |                 embedding=embedding,
   784 |                 version=self.embedding_version,
   785 |             )
   786 | 
```

### src/ingestion/build_graph.py:794 — Use of canonical 'embedding_version' (info)
```text
   792 | 
   793 |     def _set_embedding_version_in_neo4j(self, node_id: str, label: str):
   794 |         """Set embedding_version metadata in Neo4j without storing vector."""
   795 |         query = f"""
   796 |         MATCH (n:{label} {{id: $node_id}})
```

### src/ingestion/build_graph.py:797 — Use of canonical 'embedding_version' (info)
```text
   795 |         query = f"""
   796 |         MATCH (n:{label} {{id: $node_id}})
   797 |         SET n.embedding_version = $version
   798 |         RETURN n.id as id
   799 |         """
```

### src/ingestion/build_graph.py:805 — Use of canonical 'embedding_version' (info)
```text
   803 |                 query,
   804 |                 node_id=node_id,
   805 |                 version=self.embedding_version,
   806 |             )
   807 | 
```

### src/ingestion/build_graph.py:830 — Use of canonical 'embedding_version' (info)
```text
   828 |         # Phase 7C.7: Validate all required fields are present
   829 |         required_fields = [
   830 |             "embedding_version",
   831 |             "embedding_provider",
   832 |             "embedding_dimensions",
```

### src/ingestion/build_graph.py:832 — Use of canonical 'embedding_dimensions' (info)
```text
   830 |             "embedding_version",
   831 |             "embedding_provider",
   832 |             "embedding_dimensions",
   833 |             "embedding_timestamp",
   834 |         ]
```

### src/ingestion/build_graph.py:833 — Use of canonical 'embedding_timestamp' (info)
```text
   831 |             "embedding_provider",
   832 |             "embedding_dimensions",
   833 |             "embedding_timestamp",
   834 |         ]
   835 |         missing_fields = [f for f in required_fields if f not in metadata]
```

### src/ingestion/build_graph.py:843 — Presence of :Section label (info)
```text
   841 | 
   842 |         query = """
   843 |         MATCH (s:Section {id: $node_id})
   844 |         SET s.vector_embedding = $vector_embedding,
   845 |             s.embedding_version = $embedding_version,
```

### src/ingestion/build_graph.py:844 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   842 |         query = """
   843 |         MATCH (s:Section {id: $node_id})
   844 |         SET s.vector_embedding = $vector_embedding,
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
```

### src/ingestion/build_graph.py:844 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   842 |         query = """
   843 |         MATCH (s:Section {id: $node_id})
   844 |         SET s.vector_embedding = $vector_embedding,
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
```

### src/ingestion/build_graph.py:845 — Use of canonical 'embedding_version' (info)
```text
   843 |         MATCH (s:Section {id: $node_id})
   844 |         SET s.vector_embedding = $vector_embedding,
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
```

### src/ingestion/build_graph.py:845 — Use of canonical 'embedding_version' (info)
```text
   843 |         MATCH (s:Section {id: $node_id})
   844 |         SET s.vector_embedding = $vector_embedding,
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
```

### src/ingestion/build_graph.py:847 — Use of canonical 'embedding_dimensions' (info)
```text
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
   848 |             s.embedding_timestamp = $embedding_timestamp,
   849 |             s.embedding_task = $embedding_task
```

### src/ingestion/build_graph.py:847 — Use of canonical 'embedding_dimensions' (info)
```text
   845 |             s.embedding_version = $embedding_version,
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
   848 |             s.embedding_timestamp = $embedding_timestamp,
   849 |             s.embedding_task = $embedding_task
```

### src/ingestion/build_graph.py:848 — Use of canonical 'embedding_timestamp' (info)
```text
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
   848 |             s.embedding_timestamp = $embedding_timestamp,
   849 |             s.embedding_task = $embedding_task
   850 |         RETURN s.id as id
```

### src/ingestion/build_graph.py:848 — Use of canonical 'embedding_timestamp' (info)
```text
   846 |             s.embedding_provider = $embedding_provider,
   847 |             s.embedding_dimensions = $embedding_dimensions,
   848 |             s.embedding_timestamp = $embedding_timestamp,
   849 |             s.embedding_task = $embedding_task
   850 |         RETURN s.id as id
```

### src/ingestion/build_graph.py:857 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   855 |                 query,
   856 |                 node_id=node_id,
   857 |                 vector_embedding=embedding,
   858 |                 embedding_version=metadata["embedding_version"],
   859 |                 embedding_provider=metadata["embedding_provider"],
```

### src/ingestion/build_graph.py:858 — Use of canonical 'embedding_version' (info)
```text
   856 |                 node_id=node_id,
   857 |                 vector_embedding=embedding,
   858 |                 embedding_version=metadata["embedding_version"],
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
```

### src/ingestion/build_graph.py:858 — Use of canonical 'embedding_version' (info)
```text
   856 |                 node_id=node_id,
   857 |                 vector_embedding=embedding,
   858 |                 embedding_version=metadata["embedding_version"],
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
```

### src/ingestion/build_graph.py:860 — Use of canonical 'embedding_dimensions' (info)
```text
   858 |                 embedding_version=metadata["embedding_version"],
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
   861 |                 embedding_timestamp=metadata["embedding_timestamp"],
   862 |                 embedding_task=metadata.get("embedding_task", "retrieval.passage"),
```

### src/ingestion/build_graph.py:860 — Use of canonical 'embedding_dimensions' (info)
```text
   858 |                 embedding_version=metadata["embedding_version"],
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
   861 |                 embedding_timestamp=metadata["embedding_timestamp"],
   862 |                 embedding_task=metadata.get("embedding_task", "retrieval.passage"),
```

### src/ingestion/build_graph.py:861 — Use of canonical 'embedding_timestamp' (info)
```text
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
   861 |                 embedding_timestamp=metadata["embedding_timestamp"],
   862 |                 embedding_task=metadata.get("embedding_task", "retrieval.passage"),
   863 |             )
```

### src/ingestion/build_graph.py:861 — Use of canonical 'embedding_timestamp' (info)
```text
   859 |                 embedding_provider=metadata["embedding_provider"],
   860 |                 embedding_dimensions=metadata["embedding_dimensions"],
   861 |                 embedding_timestamp=metadata["embedding_timestamp"],
   862 |                 embedding_task=metadata.get("embedding_task", "retrieval.passage"),
   863 |             )
```

### src/ingestion/build_graph.py:869 — Use of canonical 'embedding_dimensions' (info)
```text
   867 |             node_id=node_id,
   868 |             provider=metadata["embedding_provider"],
   869 |             dimensions=metadata["embedding_dimensions"],
   870 |             vector_stored=True,
   871 |         )
```

### src/ingestion/build_graph.py:881 — Use of canonical 'embedding_version' (info)
```text
   879 |     *,
   880 |     embedding_model: Optional[str] = None,
   881 |     embedding_version: Optional[str] = None,
   882 | ) -> Dict:
   883 |     """
```

### src/ingestion/build_graph.py:913 — Use of canonical 'embedding_version' (info)
```text
   911 |                 requested_model=embedding_model,
   912 |             )
   913 |     if embedding_version:
   914 |         try:
   915 |             config.embedding.version = embedding_version
```

### src/ingestion/build_graph.py:915 — Use of canonical 'embedding_version' (info)
```text
   913 |     if embedding_version:
   914 |         try:
   915 |             config.embedding.version = embedding_version
   916 |         except Exception:
   917 |             logger.warning(
```

### src/ingestion/build_graph.py:919 — Use of canonical 'embedding_version' (info)
```text
   917 |             logger.warning(
   918 |                 "Failed to override embedding version via ingest_document",
   919 |                 requested_version=embedding_version,
   920 |             )
   921 | 
```

### src/ingestion/extract/commands.py:257 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   255 |         "deprecated_in": None,
   256 |         "updated_at": None,
   257 |         "vector_embedding": None,
   258 |         "embedding_version": None,
   259 |     }
```

### src/ingestion/extract/commands.py:258 — Use of canonical 'embedding_version' (info)
```text
   256 |         "updated_at": None,
   257 |         "vector_embedding": None,
   258 |         "embedding_version": None,
   259 |     }
   260 | 
```

### src/ingestion/extract/configs.py:252 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   250 |         "deprecated_in": None,
   251 |         "updated_at": None,
   252 |         "vector_embedding": None,
   253 |         "embedding_version": None,
   254 |     }
```

### src/ingestion/extract/configs.py:253 — Use of canonical 'embedding_version' (info)
```text
   251 |         "updated_at": None,
   252 |         "vector_embedding": None,
   253 |         "embedding_version": None,
   254 |     }
   255 | 
```

### src/ingestion/incremental.py:28 — Presence of :Section label (info)
```text
    26 |     - Stage new/modified sections as :StagedSection
    27 |     - Delete removed sections
    28 |     - Promote staged to :Section (atomic swap)
    29 |     - Keeps counts stable for 'minimal delta' scenarios
    30 |     """
```

### src/ingestion/incremental.py:38 — Use of canonical 'embedding_version' (info)
```text
    36 |         qdrant_client: QdrantClient = None,
    37 |         collection_name: str = "weka_sections",
    38 |         embedding_version: str = "v1",
    39 |     ):
    40 |         self.neo4j = neo4j_driver
```

### src/ingestion/incremental.py:44 — Use of canonical 'embedding_version' (info)
```text
    42 |         self.qdrant = qdrant_client
    43 |         self.collection = collection_name
    44 |         self.version = embedding_version
    45 |         if config and hasattr(config, "search") and hasattr(config.search, "vector"):
    46 |             if hasattr(config.search.vector, "qdrant"):
```

### src/ingestion/incremental.py:55 — Presence of :Section label (info)
```text
    53 |         """Get existing sections from the database (synchronous)."""
    54 |         cypher = """
    55 |         MATCH (:Document {id: $doc})-[:HAS_SECTION]->(s:Section)
    56 |         RETURN s.id AS id, coalesce(s.checksum, '') AS checksum, s.title AS title
    57 |         """
```

### src/ingestion/incremental.py:140 — Presence of :Section label (info)
```text
   138 |                     """
   139 |                     UNWIND $ids AS sid
   140 |                     MATCH (d:Document {id: $doc})-[:HAS_SECTION]->(s:Section {id: sid})
   141 |                     DETACH DELETE s
   142 |                     """,
```

### src/ingestion/incremental.py:162 — Presence of :Section label (info)
```text
   160 |                     """
   161 |                     UNWIND $rows AS row
   162 |                     MERGE (s:Section {id: row.id})
   163 |                     SET s.title = row.title,
   164 |                         s.content = coalesce(row.content, row.text),
```

### src/ingestion/incremental.py:167 — Use of canonical 'embedding_version' (info)
```text
   165 |                         s.checksum = row.checksum,
   166 |                         s.document_id = $doc,
   167 |                         s.embedding_version = $v,
   168 |                         s.updated_at = datetime()
   169 |                     MERGE (d:Document {id: $doc})
```

### src/ingestion/incremental.py:169 — MERGE Document by id (canonical) (info)
```text
   167 |                         s.embedding_version = $v,
   168 |                         s.updated_at = datetime()
   169 |                     MERGE (d:Document {id: $doc})
   170 |                     MERGE (d)-[:HAS_SECTION]->(s)
   171 |                     """,
```

### src/ingestion/incremental.py:196 — Use of canonical 'embedding_version' (info)
```text
   194 |                                 "document_uri": sec.get("source_uri")
   195 |                                 or sec.get("document_uri"),
   196 |                                 "embedding_version": self.version,
   197 |                             },
   198 |                         )
```

### src/ingestion/parsers/html.py:219 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   217 |         "code_blocks": section_data["code_blocks"],
   218 |         "tables": section_data["tables"],
   219 |         "vector_embedding": None,
   220 |         "embedding_version": None,
   221 |     }
```

### src/ingestion/parsers/html.py:220 — Use of canonical 'embedding_version' (info)
```text
   218 |         "tables": section_data["tables"],
   219 |         "vector_embedding": None,
   220 |         "embedding_version": None,
   221 |     }
   222 | 
```

### src/ingestion/parsers/markdown.py:252 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   250 |         "tables": section_data["tables"],
   251 |         # Vector embedding fields (populated later)
   252 |         "vector_embedding": None,
   253 |         "embedding_version": None,
   254 |     }
```

### src/ingestion/parsers/markdown.py:253 — Use of canonical 'embedding_version' (info)
```text
   251 |         # Vector embedding fields (populated later)
   252 |         "vector_embedding": None,
   253 |         "embedding_version": None,
   254 |     }
   255 | 
```

### src/ingestion/parsers/notion.py:211 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   209 |         "code_blocks": section_data["code_blocks"],
   210 |         "tables": section_data["tables"],
   211 |         "vector_embedding": None,
   212 |         "embedding_version": None,
   213 |     }
```

### src/ingestion/parsers/notion.py:212 — Use of canonical 'embedding_version' (info)
```text
   210 |         "tables": section_data["tables"],
   211 |         "vector_embedding": None,
   212 |         "embedding_version": None,
   213 |     }
   214 | 
```

### src/ingestion/reconcile.py:17 — Use of canonical 'embedding_version' (info)
```text
    15 | @dataclass
    16 | class DriftStats:
    17 |     embedding_version: str
    18 |     graph_count: int
    19 |     vector_count: int
```

### src/ingestion/reconcile.py:27 — Use of canonical 'embedding_version' (info)
```text
    25 | class Reconciler:
    26 |     """
    27 |     Keeps Qdrant strictly in sync with graph Section nodes for a given embedding_version.
    28 |     """
    29 | 
```

### src/ingestion/reconcile.py:36 — Use of canonical 'embedding_version' (info)
```text
    34 |         qdrant_client: Optional[QdrantClient] = None,
    35 |         collection_name: str = "weka_sections",
    36 |         embedding_version: str = "v1",
    37 |     ):
    38 |         self.neo4j = neo4j_driver
```

### src/ingestion/reconcile.py:46 — Use of canonical 'embedding_version' (info)
```text
    44 |             self.config = None
    45 |             self.collection = collection_name
    46 |             self.version = embedding_version
    47 |         elif hasattr(config_or_qdrant, "embedding"):
    48 |             # New signature: Reconciler(neo4j, config, qdrant)
```

### src/ingestion/reconcile.py:64 — Use of canonical 'embedding_version' (info)
```text
    62 |             self.qdrant = qdrant_client or config_or_qdrant
    63 |             self.collection = collection_name
    64 |             self.version = embedding_version
    65 | 
    66 |     def _graph_section_ids(self) -> Set[str]:
```

### src/ingestion/reconcile.py:67 — Use of canonical 'embedding_version' (info)
```text
    65 | 
    66 |     def _graph_section_ids(self) -> Set[str]:
    67 |         """Get all Section IDs from Neo4j with matching embedding_version (synchronous)."""
    68 |         cypher = """
    69 |         MATCH (s:Section)
```

### src/ingestion/reconcile.py:69 — Presence of :Section label (info)
```text
    67 |         """Get all Section IDs from Neo4j with matching embedding_version (synchronous)."""
    68 |         cypher = """
    69 |         MATCH (s:Section)
    70 |         WHERE s.embedding_version = $v
    71 |         RETURN s.id AS id
```

### src/ingestion/reconcile.py:70 — Use of canonical 'embedding_version' (info)
```text
    68 |         cypher = """
    69 |         MATCH (s:Section)
    70 |         WHERE s.embedding_version = $v
    71 |         RETURN s.id AS id
    72 |         """
```

### src/ingestion/reconcile.py:87 — Use of canonical 'embedding_version' (info)
```text
    85 |             "must": [
    86 |                 {"key": "node_label", "match": {"value": "Section"}},
    87 |                 {"key": "embedding_version", "match": {"value": self.version}},
    88 |             ]
    89 |         }
```

### src/ingestion/reconcile.py:125 — Use of canonical 'embedding_version' (info)
```text
   123 |                                 {"key": "node_id", "match": {"value": nid}},
   124 |                                 {
   125 |                                     "key": "embedding_version",
   126 |                                     "match": {"value": self.version},
   127 |                                 },
```

### src/ingestion/reconcile.py:182 — Use of canonical 'embedding_version' (info)
```text
   180 |     ) -> DriftStats:
   181 |         """
   182 |         Make Qdrant contain exactly the Section nodes at embedding_version=self.version.
   183 |         If embedding_fn is None, uses SentenceTransformers(all-MiniLM-L6-v2) if available.
   184 |         """
```

### src/ingestion/reconcile.py:210 — Presence of :Section label (info)
```text
   208 |             cypher = """
   209 |             UNWIND $ids AS sid
   210 |             MATCH (d:Document)-[:HAS_SECTION]->(s:Section {id: sid})
   211 |             RETURN
   212 |                 s.id AS id,
```

### src/ingestion/reconcile.py:240 — Use of canonical 'embedding_version' (info)
```text
   238 |                     "document_uri": document_uri,
   239 |                     "source_uri": source_uri,
   240 |                     "embedding_version": self.version,
   241 |                 }
   242 |                 upserts.append((sid, vec, payload))
```

### src/ingestion/reconcile.py:260 — Use of canonical 'embedding_version' (info)
```text
   258 | 
   259 |         return DriftStats(
   260 |             embedding_version=self.version,
   261 |             graph_count=graph_count,
   262 |             vector_count=final_vector_count,
```

### src/mcp_server/query_service.py:247 — Use of canonical 'embedding_version' (info)
```text
   245 |                     )
   246 | 
   247 |             # Pre-Phase 7 B4: Add embedding_version filter to ensure version consistency
   248 |             # This ensures we only retrieve vectors created with the current embedding model
   249 |             if filters is None:
```

### src/mcp_server/query_service.py:252 — Use of canonical 'embedding_version' (info)
```text
   250 |                 filters = {}
   251 | 
   252 |             # Add embedding_version to filters
   253 |             filters["embedding_version"] = self.config.embedding.version
   254 |             logger.debug(
```

### src/mcp_server/query_service.py:253 — Use of canonical 'embedding_version' (info)
```text
   251 | 
   252 |             # Add embedding_version to filters
   253 |             filters["embedding_version"] = self.config.embedding.version
   254 |             logger.debug(
   255 |                 f"Added embedding_version filter: {self.config.embedding.version}"
```

### src/mcp_server/query_service.py:255 — Use of canonical 'embedding_version' (info)
```text
   253 |             filters["embedding_version"] = self.config.embedding.version
   254 |             logger.debug(
   255 |                 f"Added embedding_version filter: {self.config.embedding.version}"
   256 |             )
   257 | 
```

### src/providers/embeddings/base.py:42 — Reference to canonical model v3 (info)
```text
    40 | 
    41 |         Returns:
    42 |             str: Model identifier (e.g., "all-MiniLM-L6-v2", "jina-embeddings-v3")
    43 |         """
    44 |         ...
```

### src/providers/embeddings/jina.py:3 — Reference to canonical model v3 (info)
```text
     1 | """
     2 | Jina AI embedding provider implementation.
     3 | Phase 7C: Remote API provider for jina-embeddings-v3 @ 1024-D.
     4 | 
     5 | Features:
```

### src/providers/embeddings/jina.py:150 — Reference to canonical model v3 (info)
```text
   148 | class JinaEmbeddingProvider:
   149 |     """
   150 |     Jina AI embedding provider using jina-embeddings-v3.
   151 | 
   152 |     Supports task-specific embeddings:
```

### src/providers/embeddings/jina.py:186 — Reference to canonical model v3 (info)
```text
   184 |     def __init__(
   185 |         self,
   186 |         model: str = "jina-embeddings-v3",
   187 |         dims: int = 1024,
   188 |         api_key: Optional[str] = None,
```

### src/providers/tokenizer_service.py:4 — Reference to canonical model v3 (info)
```text
     2 | Tokenizer service for accurate token counting and text splitting.
     3 | 
     4 | CRITICAL: This module uses the EXACT tokenizer for jina-embeddings-v3 (XLM-RoBERTa family).
     5 | DO NOT use tiktoken or cl100k_base - those are for OpenAI models and will give wrong counts.
     6 | 
```

### src/providers/tokenizer_service.py:87 — Reference to canonical model v3 (info)
```text
    85 |     HuggingFace local tokenizer backend (PRIMARY).
    86 | 
    87 |     Uses the exact tokenizer for jina-embeddings-v3.
    88 |     Fast (<5ms per section), deterministic, works offline.
    89 |     """
```

### src/providers/tokenizer_service.py:95 — Reference to canonical model v3 (info)
```text
    93 |         Initialize HuggingFace tokenizer.
    94 | 
    95 |         Loads jinaai/jina-embeddings-v3 tokenizer from cache.
    96 |         Expects tokenizer to be prefetched during Docker build.
    97 | 
```

### src/providers/tokenizer_service.py:104 — Reference to canonical model v3 (info)
```text
   102 |             from transformers import AutoTokenizer
   103 | 
   104 |             model_id = os.getenv("HF_TOKENIZER_ID", "jinaai/jina-embeddings-v3")
   105 |             cache_dir = os.getenv("HF_CACHE", "/opt/hf-cache")
   106 |             offline = os.getenv("TRANSFORMERS_OFFLINE", "true").lower() == "true"
```

### src/providers/tokenizer_service.py:286 — Reference to canonical model v3 (info)
```text
   284 | 
   285 |     Provides accurate token counting and lossless text splitting
   286 |     for jina-embeddings-v3 (XLM-RoBERTa tokenizer).
   287 | 
   288 |     Features:
```

### src/query/hybrid_search.py:156 — Use of canonical 'embedding_version' (info)
```text
   154 |         CALL db.index.vector.queryNodes($index_name, $k, $vector)
   155 |         YIELD node, score
   156 |         WHERE node.embedding_version = $embedding_version{where_clause}
   157 |         RETURN node.id AS id, score, node.document_id AS document_id,
   158 |                labels(node)[0] AS node_label, properties(node) AS metadata
```

### src/query/hybrid_search.py:156 — Use of canonical 'embedding_version' (info)
```text
   154 |         CALL db.index.vector.queryNodes($index_name, $k, $vector)
   155 |         YIELD node, score
   156 |         WHERE node.embedding_version = $embedding_version{where_clause}
   157 |         RETURN node.id AS id, score, node.document_id AS document_id,
   158 |                labels(node)[0] AS node_label, properties(node) AS metadata
```

### src/query/hybrid_search.py:168 — Use of canonical 'embedding_version' (info)
```text
   166 |                 k=k,
   167 |                 vector=vector,
   168 |                 embedding_version=get_config()
   169 |                 .get("embedding", {})
   170 |                 .get("version", "v1"),
```

### src/query/hybrid_search.py:469 — Presence of :Section label (info)
```text
   467 |         coverage_query = """
   468 |         UNWIND $ids AS sid
   469 |         MATCH (s:Section {id: sid})
   470 |         OPTIONAL MATCH (s)-[r]->()
   471 |         WITH s, count(DISTINCT r) AS conn_count
```

### src/query/hybrid_search.py:615 — Presence of :Section label (info)
```text
   613 |         focus_query = """
   614 |         UNWIND $section_ids AS section_id
   615 |         MATCH (s:Section {id: section_id})-[:MENTIONS]->(e)
   616 |         WHERE e.id IN $focused_entity_ids
   617 |         RETURN s.id AS section_id, count(DISTINCT e) AS focus_hits, collect(DISTINCT e.id) AS matched_entities
```

### src/query/planner.py:269 — Presence of :Section label (info)
```text
   267 |         # Use basic search template as fallback
   268 |         fallback_cypher = """
   269 |         MATCH (s:Section)
   270 |         WHERE s.id IN $section_ids
   271 |         RETURN s
```

### src/query/session_tracker.py:340 — Presence of :Section label (info)
```text
   338 |         MATCH (q:Query {query_id: $query_id})
   339 |         UNWIND $sections as section
   340 |         MATCH (s:Section {id: section.section_id})
   341 |         MERGE (q)-[r:RETRIEVED]->(s)
   342 |         ON CREATE SET
```

### src/query/session_tracker.py:429 — Presence of :Section label (info)
```text
   427 |                 MATCH (a:Answer {answer_id: $answer_id})
   428 |                 UNWIND $citations as citation
   429 |                 MATCH (s:Section {id: citation.section_id})
   430 |                 CREATE (a)-[r:SUPPORTED_BY {
   431 |                     rank: citation.rank,
```

### src/query/session_tracker.py:489 — Presence of :Section label (info)
```text
   487 | 
   488 |         // Collect retrieved sections
   489 |         OPTIONAL MATCH (q)-[r:RETRIEVED]->(sec:Section)
   490 |         WITH q, s, focused_entities, collect(DISTINCT {
   491 |             id: sec.id,
```

### src/query/session_tracker.py:499 — Presence of :Section label (info)
```text
   497 |         // Collect answer and citations
   498 |         OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
   499 |         OPTIONAL MATCH (a)-[c:SUPPORTED_BY]->(cited:Section)
   500 |         WITH q, s, focused_entities, retrieved_sections,
   501 |              a.answer_id as answer_id,
```

### src/query/templates/advanced/troubleshooting_path.cypher:49 — Presence of :Section label (info)
```text
    47 | MATCH (e:Error)
    48 | WHERE e.code = $error_code OR e.name = $error_name
    49 | OPTIONAL MATCH (error_sec:Section)-[:MENTIONS]->(e)
    50 | OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
    51 | OPTIONAL MATCH (proc_sec:Section)-[:MENTIONS]->(proc)
```

### src/query/templates/advanced/troubleshooting_path.cypher:51 — Presence of :Section label (info)
```text
    49 | OPTIONAL MATCH (error_sec:Section)-[:MENTIONS]->(e)
    50 | OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
    51 | OPTIONAL MATCH (proc_sec:Section)-[:MENTIONS]->(proc)
    52 | OPTIONAL MATCH (proc)-[:CONTAINS_STEP]->(step:Step)
    53 | OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
```

### src/query/templates/explain.cypher:29 — Presence of :Section label (info)
```text
    27 | MATCH (concept:Concept)
    28 | WHERE concept.term = $concept_term OR concept.name = $concept_term
    29 | OPTIONAL MATCH (concept)<-[:MENTIONS]-(sec:Section)
    30 | OPTIONAL MATCH (concept)-[:RELATED_TO]->(ex:Example)
    31 | OPTIONAL MATCH (concept)-[:RELATED_TO]->(related:Concept)
```

### src/query/templates/search.cypher:6 — Presence of :Section label (info)
```text
     4 | 
     5 | -- Version 1: Basic section search
     6 | MATCH (s:Section)
     7 | WHERE s.id IN $section_ids
     8 | OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
```

### src/query/templates/search.cypher:15 — Presence of :Section label (info)
```text
    13 | 
    14 | -- Version 2: Search with document context
    15 | MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
    16 | WHERE s.id IN $section_ids
    17 | OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
```

### src/query/templates/search.cypher:28 — Presence of :Section label (info)
```text
    26 | 
    27 | -- Version 3: Search with controlled expansion
    28 | MATCH (s:Section)
    29 | WHERE s.id IN $section_ids
    30 | OPTIONAL MATCH path=(s)-[:MENTIONS|:CONTAINS_STEP|:HAS_PARAMETER*1..$max_hops]->(n)
```

### src/query/templates/troubleshoot.cypher:18 — Presence of :Section label (info)
```text
    16 | MATCH (e:Error)
    17 | WHERE e.code = $error_code OR e.name = $error_name
    18 | OPTIONAL MATCH (sec:Section)-[:MENTIONS]->(e)
    19 | OPTIONAL MATCH (e)<-[:RESOLVES]-(proc:Procedure)
    20 | OPTIONAL MATCH (proc)-[:CONTAINS_STEP]->(step:Step)
```

### src/registry/index_registry.py:54 — Reference to canonical model v3 (info)
```text
    52 |             dims: Vector dimensions
    53 |             provider: Provider name (e.g., "jina-ai", "ollama")
    54 |             model: Model identifier (e.g., "jina-embeddings-v3", "ollama/nomic-embed")
    55 |             version: Model version for provenance
    56 |             collection_name: Qdrant collection name (defaults to name)
```

### src/shared/cache.py:9 — Use of canonical 'embedding_version' (info)
```text
     7 | See: /docs/pseudocode-reference.md → Phase 4, Task 4.3
     8 | 
     9 | Cache keys are prefixed with {schema_version}:{embedding_version} to ensure
    10 | automatic invalidation when the model or schema changes.
    11 | """
```

### src/shared/cache.py:210 — Pattern-scan deletion (fallback) (info)
```text
   208 |         """Remove all keys with given prefix. Returns count invalidated."""
   209 |         try:
   210 |             # Use SCAN to find keys with prefix
   211 |             count = 0
   212 |             cursor = 0
```

### src/shared/cache.py:246 — Use of canonical 'embedding_version' (info)
```text
   244 |     Two-tier cache: L1 (in-process) + L2 (Redis).
   245 | 
   246 |     Keys are automatically prefixed with {schema_version}:{embedding_version}
   247 |     to ensure cache invalidation when versions change.
   248 |     """
```

### src/shared/cache.py:255 — Use of canonical 'embedding_version' (info)
```text
   253 |         redis_client,
   254 |         schema_version: str,
   255 |         embedding_version: str,
   256 |     ):
   257 |         self.config = config
```

### src/shared/cache.py:259 — Use of canonical 'embedding_version' (info)
```text
   257 |         self.config = config
   258 |         self.schema_version = schema_version
   259 |         self.embedding_version = embedding_version
   260 | 
   261 |         # Initialize L1 cache
```

### src/shared/cache.py:259 — Use of canonical 'embedding_version' (info)
```text
   257 |         self.config = config
   258 |         self.schema_version = schema_version
   259 |         self.embedding_version = embedding_version
   260 | 
   261 |         # Initialize L1 cache
```

### src/shared/cache.py:289 — Use of canonical 'embedding_version' (info)
```text
   287 |         Generate cache key with version prefixes.
   288 | 
   289 |         Format: {base_prefix}:{schema_version}:{embedding_version}:{key_prefix}:{params_hash}
   290 |         """
   291 |         params_hash = hashlib.sha256(
```

### src/shared/cache.py:295 — Use of canonical 'embedding_version' (info)
```text
   293 |         ).hexdigest()[:16]
   294 | 
   295 |         return f"{self.key_prefix_base}:{self.schema_version}:{self.embedding_version}:{key_prefix}:{params_hash}"
   296 | 
   297 |     def get(self, key_prefix: str, params: Dict[str, Any]) -> Optional[Any]:
```

### src/shared/cache.py:377 — Use of canonical 'embedding_version' (info)
```text
   375 |         self,
   376 |         schema_version: Optional[str] = None,
   377 |         embedding_version: Optional[str] = None,
   378 |     ) -> Dict[str, int]:
   379 |         """
```

### src/shared/cache.py:384 — Use of canonical 'embedding_version' (info)
```text
   382 |         Args:
   383 |             schema_version: Schema version to invalidate (or None for current)
   384 |             embedding_version: Embedding version to invalidate (or None for current)
   385 | 
   386 |         Returns:
```

### src/shared/cache.py:390 — Use of canonical 'embedding_version' (info)
```text
   388 |         """
   389 |         sv = schema_version or self.schema_version
   390 |         ev = embedding_version or self.embedding_version
   391 | 
   392 |         prefix = f"{self.key_prefix_base}:{sv}:{ev}"
```

### src/shared/cache.py:390 — Use of canonical 'embedding_version' (info)
```text
   388 |         """
   389 |         sv = schema_version or self.schema_version
   390 |         ev = embedding_version or self.embedding_version
   391 | 
   392 |         prefix = f"{self.key_prefix_base}:{sv}:{ev}"
```

### src/shared/cache.py:416 — Use of canonical 'embedding_version' (info)
```text
   414 |         stats = {
   415 |             "schema_version": self.schema_version,
   416 |             "embedding_version": self.embedding_version,
   417 |         }
   418 | 
```

### src/shared/cache.py:416 — Use of canonical 'embedding_version' (info)
```text
   414 |         stats = {
   415 |             "schema_version": self.schema_version,
   416 |             "embedding_version": self.embedding_version,
   417 |         }
   418 | 
```

### src/shared/connections.py:237 — Qdrant cosine distance (info)
```text
   235 |         # Map string distance to enum
   236 |         distance_map = {
   237 |             "cosine": Distance.COSINE,
   238 |             "euclid": Distance.EUCLID,
   239 |             "dot": Distance.DOT,
```

### src/shared/connections.py:241 — Qdrant cosine distance (info)
```text
   239 |             "dot": Distance.DOT,
   240 |         }
   241 |         distance_enum = distance_map.get(distance.lower(), Distance.COSINE)
   242 | 
   243 |         # Check if collection already exists
```

### src/shared/schema.py:128 — Presence of :Chunk label (dual-label support) (info)
```text
   126 |                     elif "MERGE (sv:SchemaVersion" in stmt:
   127 |                         results["schema_version_set"] = True
   128 |                     elif "SET s:Chunk" in stmt:
   129 |                         summary = result.consume()
   130 |                         results["dual_labeled_sections"] = summary.counters.labels_added
```

### src/shared/schema.py:237 — Presence of :Chunk label (dual-label support) (info)
```text
   235 | 
   236 |                 # Track dual-labeling
   237 |                 if "SET s:Chunk" in stmt:
   238 |                     # Get count from execution result
   239 |                     summary = execution_result.consume()
```

### src/shared/schema.py:302 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   300 |     # Define vector indexes to create
   301 |     vector_index_definitions = [
   302 |         ("section_embeddings", "Section", "vector_embedding"),
   303 |         ("command_embeddings", "Command", "vector_embedding"),
   304 |         ("configuration_embeddings", "Configuration", "vector_embedding"),
```

### src/shared/schema.py:303 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   301 |     vector_index_definitions = [
   302 |         ("section_embeddings", "Section", "vector_embedding"),
   303 |         ("command_embeddings", "Command", "vector_embedding"),
   304 |         ("configuration_embeddings", "Configuration", "vector_embedding"),
   305 |         ("procedure_embeddings", "Procedure", "vector_embedding"),
```

### src/shared/schema.py:304 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   302 |         ("section_embeddings", "Section", "vector_embedding"),
   303 |         ("command_embeddings", "Command", "vector_embedding"),
   304 |         ("configuration_embeddings", "Configuration", "vector_embedding"),
   305 |         ("procedure_embeddings", "Procedure", "vector_embedding"),
   306 |         ("error_embeddings", "Error", "vector_embedding"),
```

### src/shared/schema.py:305 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   303 |         ("command_embeddings", "Command", "vector_embedding"),
   304 |         ("configuration_embeddings", "Configuration", "vector_embedding"),
   305 |         ("procedure_embeddings", "Procedure", "vector_embedding"),
   306 |         ("error_embeddings", "Error", "vector_embedding"),
   307 |         ("concept_embeddings", "Concept", "vector_embedding"),
```

### src/shared/schema.py:306 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   304 |         ("configuration_embeddings", "Configuration", "vector_embedding"),
   305 |         ("procedure_embeddings", "Procedure", "vector_embedding"),
   306 |         ("error_embeddings", "Error", "vector_embedding"),
   307 |         ("concept_embeddings", "Concept", "vector_embedding"),
   308 |     ]
```

### src/shared/schema.py:307 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   305 |         ("procedure_embeddings", "Procedure", "vector_embedding"),
   306 |         ("error_embeddings", "Error", "vector_embedding"),
   307 |         ("concept_embeddings", "Concept", "vector_embedding"),
   308 |     ]
   309 | 
```

### tests/integration/test_jina_large_batches.py:31 — Reference to canonical model v3 (info)
```text
    29 |     """Create Jina provider for testing."""
    30 |     return JinaEmbeddingProvider(
    31 |         model="jina-embeddings-v3",
    32 |         dims=1024,
    33 |         api_key=os.getenv("JINA_API_KEY"),
```

### tests/integration/test_jina_large_batches.py:42 — Reference to canonical model v3 (info)
```text
    40 |     """Create Jina provider for query embedding."""
    41 |     return JinaEmbeddingProvider(
    42 |         model="jina-embeddings-v3",
    43 |         dims=1024,
    44 |         api_key=os.getenv("JINA_API_KEY"),
```

### tests/integration/test_jina_large_batches.py:268 — Reference to canonical model v3 (info)
```text
   266 |         # Create provider expecting wrong dimensions
   267 |         provider = JinaEmbeddingProvider(
   268 |             model="jina-embeddings-v3",
   269 |             dims=512,  # Wrong: v3 produces 1024-D
   270 |             api_key=os.getenv("JINA_API_KEY"),
```

### tests/integration/test_jina_large_batches.py:357 — Reference to canonical model v3 (info)
```text
   355 |         # Required fields
   356 |         assert "model" in request
   357 |         assert request["model"] == "jina-embeddings-v3"
   358 | 
   359 |         assert "task" in request
```

### tests/integration/test_phase7c_integration.py:133 — Presence of :Section label (info)
```text
   131 |                     """
   132 |                     MATCH (d:Document {source_uri: $uri})
   133 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   134 |                     DETACH DELETE d, s
   135 |                     """,
```

### tests/integration/test_phase7c_integration.py:214 — Reference to canonical model v3 (info)
```text
   212 |         ), f"Wrong provider: {embedder.provider_name}"
   213 |         assert (
   214 |             embedder.model_id == "jina-embeddings-v3"
   215 |         ), f"Wrong model: {embedder.model_id}"
   216 |         assert embedder.dims == 1024, f"Wrong dimensions: {embedder.dims}"
```

### tests/integration/test_phase7c_integration.py:468 — Presence of :Chunk label (dual-label support) (info)
```text
   466 |                 session.run(
   467 |                     """
   468 |                     MERGE (s:Section:Chunk {id: $id})
   469 |                     SET s.text = $text,
   470 |                         s.document_id = 'test-doc',
```

### tests/integration/test_phase7c_integration.py:468 — Presence of :Section label (info)
```text
   466 |                 session.run(
   467 |                     """
   468 |                     MERGE (s:Section:Chunk {id: $id})
   469 |                     SET s.text = $text,
   470 |                         s.document_id = 'test-doc',
```

### tests/integration/test_phase7c_integration.py:476 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   474 |                         s.order = 0,
   475 |                         s.tokens = 10,
   476 |                         s.vector_embedding = $vector,
   477 |                         s.embedding_version = 'test-v1',
   478 |                         s.embedding_provider = 'test',
```

### tests/integration/test_phase7c_integration.py:477 — Use of canonical 'embedding_version' (info)
```text
   475 |                         s.tokens = 10,
   476 |                         s.vector_embedding = $vector,
   477 |                         s.embedding_version = 'test-v1',
   478 |                         s.embedding_provider = 'test',
   479 |                         s.embedding_timestamp = datetime(),
```

### tests/integration/test_phase7c_integration.py:479 — Use of canonical 'embedding_timestamp' (info)
```text
   477 |                         s.embedding_version = 'test-v1',
   478 |                         s.embedding_provider = 'test',
   479 |                         s.embedding_timestamp = datetime(),
   480 |                         s.embedding_dimensions = 1024
   481 |                     """,
```

### tests/integration/test_phase7c_integration.py:480 — Use of canonical 'embedding_dimensions' (info)
```text
   478 |                         s.embedding_provider = 'test',
   479 |                         s.embedding_timestamp = datetime(),
   480 |                         s.embedding_dimensions = 1024
   481 |                     """,
   482 |                     id=sec["section_id"],
```

### tests/integration/test_phase7c_integration.py:495 — Presence of :Section label (info)
```text
   493 |                 result = session.run(
   494 |                     """
   495 |                     MATCH (q:Query {query_id: $qid})-[r:RETRIEVED]->(s:Section)
   496 |                     RETURN count(r) as count,
   497 |                            collect({
```

### tests/integration/test_phase7c_integration.py:523 — Presence of :Section label (info)
```text
   521 |                 session.run(
   522 |                     """
   523 |                     MATCH (s:Section)
   524 |                     WHERE s.id STARTS WITH 'test-section-'
   525 |                     DETACH DELETE s
```

### tests/integration/test_phase7c_integration.py:548 — Presence of :Chunk label (dual-label support) (info)
```text
   546 |                 session.run(
   547 |                     """
   548 |                     MERGE (s:Section:Chunk {id: $id})
   549 |                     SET s.text = $text,
   550 |                         s.document_id = 'test-doc',
```

### tests/integration/test_phase7c_integration.py:548 — Presence of :Section label (info)
```text
   546 |                 session.run(
   547 |                     """
   548 |                     MERGE (s:Section:Chunk {id: $id})
   549 |                     SET s.text = $text,
   550 |                         s.document_id = 'test-doc',
```

### tests/integration/test_phase7c_integration.py:556 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   554 |                         s.order = 0,
   555 |                         s.tokens = 10,
   556 |                         s.vector_embedding = $vector,
   557 |                         s.embedding_version = 'test-v1',
   558 |                         s.embedding_provider = 'test',
```

### tests/integration/test_phase7c_integration.py:557 — Use of canonical 'embedding_version' (info)
```text
   555 |                         s.tokens = 10,
   556 |                         s.vector_embedding = $vector,
   557 |                         s.embedding_version = 'test-v1',
   558 |                         s.embedding_provider = 'test',
   559 |                         s.embedding_timestamp = datetime(),
```

### tests/integration/test_phase7c_integration.py:559 — Use of canonical 'embedding_timestamp' (info)
```text
   557 |                         s.embedding_version = 'test-v1',
   558 |                         s.embedding_provider = 'test',
   559 |                         s.embedding_timestamp = datetime(),
   560 |                         s.embedding_dimensions = 1024
   561 |                     """,
```

### tests/integration/test_phase7c_integration.py:560 — Use of canonical 'embedding_dimensions' (info)
```text
   558 |                         s.embedding_provider = 'test',
   559 |                         s.embedding_timestamp = datetime(),
   560 |                         s.embedding_dimensions = 1024
   561 |                     """,
   562 |                     id=sec_id,
```

### tests/integration/test_phase7c_integration.py:586 — Presence of :Section label (info)
```text
   584 |                     """
   585 |                     MATCH (q:Query {query_id: $qid})-[:ANSWERED_AS]->(a:Answer {answer_id: $aid})
   586 |                     MATCH (a)-[c:SUPPORTED_BY]->(s:Section)
   587 |                     RETURN a.text as answer_text,
   588 |                            a.model as model,
```

### tests/integration/test_phase7c_integration.py:622 — Presence of :Section label (info)
```text
   620 |                 session.run(
   621 |                     """
   622 |                     MATCH (s:Section)
   623 |                     WHERE s.id STARTS WITH 'test-cite-section-'
   624 |                     DETACH DELETE s
```

### tests/integration/test_phase7c_integration.py:715 — MERGE Document by id (canonical) (info)
```text
   713 |             session.run(
   714 |                 """
   715 |                 MERGE (d:Document {id: $doc_id})
   716 |                 SET d.title = 'Test NFS Document',
   717 |                     d.source_uri = 'test-nfs.md'
```

### tests/integration/test_phase7c_integration.py:742 — Presence of :Chunk label (dual-label support) (info)
```text
   740 |                 session.run(
   741 |                     """
   742 |                     MERGE (s:Section:Chunk {id: $section_id})
   743 |                     SET s.text = $text,
   744 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:742 — Presence of :Section label (info)
```text
   740 |                 session.run(
   741 |                     """
   742 |                     MERGE (s:Section:Chunk {id: $section_id})
   743 |                     SET s.text = $text,
   744 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:750 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   748 |                         s.order = $order,
   749 |                         s.tokens = 100,
   750 |                         s.vector_embedding = $vector,
   751 |                         s.embedding_version = 'test-v1',
   752 |                         s.embedding_provider = 'test',
```

### tests/integration/test_phase7c_integration.py:751 — Use of canonical 'embedding_version' (info)
```text
   749 |                         s.tokens = 100,
   750 |                         s.vector_embedding = $vector,
   751 |                         s.embedding_version = 'test-v1',
   752 |                         s.embedding_provider = 'test',
   753 |                         s.embedding_timestamp = datetime(),
```

### tests/integration/test_phase7c_integration.py:753 — Use of canonical 'embedding_timestamp' (info)
```text
   751 |                         s.embedding_version = 'test-v1',
   752 |                         s.embedding_provider = 'test',
   753 |                         s.embedding_timestamp = datetime(),
   754 |                         s.embedding_dimensions = 1024
   755 |                     WITH s
```

### tests/integration/test_phase7c_integration.py:754 — Use of canonical 'embedding_dimensions' (info)
```text
   752 |                         s.embedding_provider = 'test',
   753 |                         s.embedding_timestamp = datetime(),
   754 |                         s.embedding_dimensions = 1024
   755 |                     WITH s
   756 |                     MATCH (d:Document {id: $doc_id})
```

### tests/integration/test_phase7c_integration.py:785 — Presence of :Section label (info)
```text
   783 |                 """
   784 |                 MATCH (d:Document {id: $doc_id})
   785 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   786 |                 DETACH DELETE d, s
   787 |                 MATCH (e:Configuration {id: $entity_id})
```

### tests/integration/test_phase7c_integration.py:910 — MERGE Document by id (canonical) (info)
```text
   908 |             session.run(
   909 |                 """
   910 |                 MERGE (d:Document {id: $doc_id})
   911 |                 SET d.title = 'Test Orphan Document'
   912 |                 """,
```

### tests/integration/test_phase7c_integration.py:922 — Presence of :Chunk label (dual-label support) (info)
```text
   920 |                 session.run(
   921 |                     """
   922 |                     MERGE (s:Section:Chunk {id: $section_id})
   923 |                     SET s.text = $text,
   924 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:922 — Presence of :Section label (info)
```text
   920 |                 session.run(
   921 |                     """
   922 |                     MERGE (s:Section:Chunk {id: $section_id})
   923 |                     SET s.text = $text,
   924 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:930 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   928 |                         s.order = $order,
   929 |                         s.tokens = 10,
   930 |                         s.vector_embedding = $vector,
   931 |                         s.embedding_version = 'test-v1',
   932 |                         s.embedding_provider = 'test',
```

### tests/integration/test_phase7c_integration.py:931 — Use of canonical 'embedding_version' (info)
```text
   929 |                         s.tokens = 10,
   930 |                         s.vector_embedding = $vector,
   931 |                         s.embedding_version = 'test-v1',
   932 |                         s.embedding_provider = 'test',
   933 |                         s.embedding_timestamp = datetime(),
```

### tests/integration/test_phase7c_integration.py:933 — Use of canonical 'embedding_timestamp' (info)
```text
   931 |                         s.embedding_version = 'test-v1',
   932 |                         s.embedding_provider = 'test',
   933 |                         s.embedding_timestamp = datetime(),
   934 |                         s.embedding_dimensions = 1024
   935 |                     WITH s
```

### tests/integration/test_phase7c_integration.py:934 — Use of canonical 'embedding_dimensions' (info)
```text
   932 |                         s.embedding_provider = 'test',
   933 |                         s.embedding_timestamp = datetime(),
   934 |                         s.embedding_dimensions = 1024
   935 |                     WITH s
   936 |                     MATCH (d:Document {id: $doc_id})
```

### tests/integration/test_phase7c_integration.py:952 — Presence of :Section label (info)
```text
   950 |                 result = session.run(
   951 |                     """
   952 |                     MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   953 |                     RETURN collect(s.id) as section_ids
   954 |                     ORDER BY s.order
```

### tests/integration/test_phase7c_integration.py:977 — Presence of :Section label (info)
```text
   975 |                 result = session.run(
   976 |                     """
   977 |                     MATCH (s:Section {id: $sid})
   978 |                     RETURN count(s) as count
   979 |                     """,
```

### tests/integration/test_phase7c_integration.py:997 — Presence of :Section label (info)
```text
   995 |                     """
   996 |                     MATCH (d:Document {id: $doc_id})
   997 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   998 |                     DETACH DELETE d, s
   999 |                     """,
```

### tests/integration/test_phase7c_integration.py:1015 — MERGE Document by id (canonical) (info)
```text
  1013 |             session.run(
  1014 |                 """
  1015 |                 MERGE (d:Document {id: $doc_id})
  1016 |                 SET d.title = 'Test Stale Document'
  1017 |                 """,
```

### tests/integration/test_phase7c_integration.py:1028 — Presence of :Chunk label (dual-label support) (info)
```text
  1026 |                 session.run(
  1027 |                     """
  1028 |                     MERGE (s:Section:Chunk {id: $section_id})
  1029 |                     SET s.text = $text,
  1030 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:1028 — Presence of :Section label (info)
```text
  1026 |                 session.run(
  1027 |                     """
  1028 |                     MERGE (s:Section:Chunk {id: $section_id})
  1029 |                     SET s.text = $text,
  1030 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:1036 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
  1034 |                         s.order = $order,
  1035 |                         s.tokens = 10,
  1036 |                         s.vector_embedding = $vector,
  1037 |                         s.embedding_version = 'test-v1',
  1038 |                         s.embedding_provider = 'test',
```

### tests/integration/test_phase7c_integration.py:1037 — Use of canonical 'embedding_version' (info)
```text
  1035 |                         s.tokens = 10,
  1036 |                         s.vector_embedding = $vector,
  1037 |                         s.embedding_version = 'test-v1',
  1038 |                         s.embedding_provider = 'test',
  1039 |                         s.embedding_timestamp = datetime(),
```

### tests/integration/test_phase7c_integration.py:1039 — Use of canonical 'embedding_timestamp' (info)
```text
  1037 |                         s.embedding_version = 'test-v1',
  1038 |                         s.embedding_provider = 'test',
  1039 |                         s.embedding_timestamp = datetime(),
  1040 |                         s.embedding_dimensions = 1024
  1041 |                     WITH s
```

### tests/integration/test_phase7c_integration.py:1040 — Use of canonical 'embedding_dimensions' (info)
```text
  1038 |                         s.embedding_provider = 'test',
  1039 |                         s.embedding_timestamp = datetime(),
  1040 |                         s.embedding_dimensions = 1024
  1041 |                     WITH s
  1042 |                     MATCH (d:Document {id: $doc_id})
```

### tests/integration/test_phase7c_integration.py:1072 — Presence of :Section label (info)
```text
  1070 |                     })
  1071 |                     WITH q
  1072 |                     MATCH (sec:Section {id: $section_id})
  1073 |                     CREATE (q)-[:RETRIEVED {
  1074 |                         rank: 1,
```

### tests/integration/test_phase7c_integration.py:1100 — Presence of :Section label (info)
```text
  1098 |                 result = session.run(
  1099 |                     """
  1100 |                     MATCH (s:Section {id: $sid})
  1101 |                     RETURN s.is_stale as is_stale,
  1102 |                            s.stale_since as stale_since,
```

### tests/integration/test_phase7c_integration.py:1130 — Presence of :Section label (info)
```text
  1128 |                     """
  1129 |                     MATCH (d:Document {id: $doc_id})
  1130 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
  1131 |                     DETACH DELETE d, s
  1132 |                     MATCH (sess:Session {session_id: $session_id})
```

### tests/p1_t3_test.py:129 — MERGE Document by id (canonical) (info)
```text
   127 |         result = session.run(
   128 |             """
   129 |             MERGE (d:Document {id: $id})
   130 |             SET d.source_uri = $uri,
   131 |                 d.source_type = 'markdown',
```

### tests/p1_t3_test.py:156 — Presence of :Section label (info)
```text
   154 |         result = session.run(
   155 |             """
   156 |             MERGE (s:Section {id: $id})
   157 |             SET s.document_id = $doc_id,
   158 |                 s.level = 1,
```

### tests/p1_t3_test.py:172 — Presence of :Section label (info)
```text
   170 | 
   171 |         # Cleanup
   172 |         session.run("MATCH (s:Section {id: $id}) DELETE s", id=section_id)
```

### tests/p2_t2_test.py:69 — Presence of :Section label (info)
```text
    67 |         # Should not raise
    68 |         result = validator.validate(
    69 |             "MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "abc123"}
    70 |         )
    71 |         assert result.valid
```

### tests/p2_t2_test.py:145 — Presence of :Section label (info)
```text
   143 |         validator.enforce_limits = True
   144 | 
   145 |         result = validator.validate("MATCH (n:Section) RETURN n", {"limit": 50})
   146 |         assert result.valid
   147 |         assert "LIMIT" in result.query.upper()
```

### tests/p2_t2_test.py:154 — Presence of :Section label (info)
```text
   152 |         validator.enforce_limits = True
   153 | 
   154 |         result = validator.validate("MATCH (n:Section) RETURN n LIMIT 20", {})
   155 |         assert result.valid
   156 |         assert "LIMIT" in result.query.upper()
```

### tests/p2_t2_test.py:167 — Presence of :Section label (info)
```text
   165 |         # Safe query with index lookup
   166 |         result = validator.validate(
   167 |             "MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "test123"}
   168 |         )
   169 |         assert result.valid
```

### tests/p2_t2_test.py:177 — Presence of :Section label (info)
```text
   175 |         # Query that requires multiple label scans
   176 |         query = """
   177 |         MATCH (a:Section), (b:Command), (c:Configuration)
   178 |         RETURN a, b, c
   179 |         LIMIT 10
```

### tests/p2_t2_test.py:211 — Presence of :Section label (info)
```text
   209 |         # Typical template query
   210 |         query = """
   211 |         MATCH (s:Section)
   212 |         WHERE s.id IN $section_ids
   213 |         OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
```

### tests/p2_t2_test.py:231 — Presence of :Section label (info)
```text
   229 |         # Collection of valid queries that should pass
   230 |         valid_queries = [
   231 |             ("MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "x"}),
   232 |             (
   233 |                 "MATCH (n)-[r:MENTIONS]->(m) WHERE n.id = $id RETURN m LIMIT 20",
```

### tests/p2_t2_test.py:242 — Presence of :Section label (info)
```text
   240 |             ("MATCH (n:Error {code: $code}) RETURN n LIMIT 1", {"code": "E100"}),
   241 |             ("MATCH (a)-[:REQUIRES*1..2]->(b) RETURN b LIMIT 10", {}),
   242 |             ("MATCH (s:Section) RETURN s ORDER BY s.order LIMIT 50", {}),
   243 |             ("MATCH (d:Document)-[:HAS_SECTION]->(s) RETURN d, s LIMIT 30", {}),
   244 |             ("MATCH (n) WHERE n.id IN $ids RETURN n LIMIT 20", {"ids": ["a", "b"]}),
```

### tests/p3_t3_integration_test.py:80 — Presence of :Section label (info)
```text
    78 |                 session.run(
    79 |                     """
    80 |                     MATCH (s:Section)
    81 |                     WHERE s.document_id IN $doc_ids
    82 |                     DETACH DELETE s
```

### tests/p3_t3_integration_test.py:135 — Presence of :Section label (info)
```text
   133 |                 """
   134 |                 MATCH (d:Document {source_uri: $uri})
   135 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   136 |                 OPTIONAL MATCH (s)-[m:MENTIONS]->(e)
   137 |                 RETURN
```

### tests/p3_t3_integration_test.py:174 — Presence of :Section label (info)
```text
   172 |                 """
   173 |                 MATCH (d:Document {source_uri: $uri})
   174 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   175 |                 OPTIONAL MATCH (s)-[m:MENTIONS]->(e)
   176 |                 RETURN
```

### tests/p3_t3_integration_test.py:240 — Use of canonical 'embedding_version' (info)
```text
   238 |         time.sleep(1)
   239 | 
   240 |         # Count Sections in Neo4j with embedding_version
   241 |         with neo4j_driver.session() as session:
   242 |             result = session.run(
```

### tests/p3_t3_integration_test.py:244 — Presence of :Section label (info)
```text
   242 |             result = session.run(
   243 |                 """
   244 |                 MATCH (s:Section)
   245 |                 WHERE s.embedding_version = $emb_version
   246 |                 RETURN count(s) as neo4j_section_count
```

### tests/p3_t3_integration_test.py:245 — Use of canonical 'embedding_version' (info)
```text
   243 |                 """
   244 |                 MATCH (s:Section)
   245 |                 WHERE s.embedding_version = $emb_version
   246 |                 RETURN count(s) as neo4j_section_count
   247 |                 """,
```

### tests/p3_t3_integration_test.py:260 — Use of canonical 'embedding_version' (info)
```text
   258 |                     {"key": "node_label", "match": {"value": "Section"}},
   259 |                     {
   260 |                         "key": "embedding_version",
   261 |                         "match": {"value": config.embedding.version},
   262 |                     },
```

### tests/p3_t3_test.py:101 — Presence of :Section label (info)
```text
    99 |             result = session.run(
   100 |                 """
   101 |                 MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   102 |                 RETURN count(s) as section_count
   103 |                 """,
```

### tests/p3_t3_test.py:130 — Presence of :Section label (info)
```text
   128 |             result = session.run(
   129 |                 """
   130 |                 MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   131 |                 RETURN count(s) as section_count
   132 |                 """,
```

### tests/p3_t3_test.py:166 — Presence of :Section label (info)
```text
   164 |             result = session.run(
   165 |                 """
   166 |                 MATCH (s:Section)-[m:MENTIONS]->(e)
   167 |                 WHERE s.document_id = $doc_id
   168 |                 RETURN m.confidence as confidence,
```

### tests/p3_t3_test.py:216 — Presence of :Section label (info)
```text
   214 |             result = session.run(
   215 |                 """
   216 |                 MATCH (s:Section)
   217 |                 WHERE s.document_id = $doc_id
   218 |                 RETURN count(s) as count
```

### tests/p3_t3_test.py:248 — Presence of :Section label (info)
```text
   246 |                 result = session.run(
   247 |                     """
   248 |                     MATCH (s:Section)
   249 |                     WHERE s.document_id = $doc_id
   250 |                       AND s.vector_embedding IS NOT NULL
```

### tests/p3_t3_test.py:250 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   248 |                     MATCH (s:Section)
   249 |                     WHERE s.document_id = $doc_id
   250 |                       AND s.vector_embedding IS NOT NULL
   251 |                     RETURN count(s) as count
   252 |                     """,
```

### tests/p3_t3_test.py:265 — Use of canonical 'embedding_version' (info)
```text
   263 |         self, graph_builder, neo4j_driver, config, sample_document
   264 |     ):
   265 |         """Test that embedding_version is set on sections."""
   266 |         graph_builder.upsert_document(
   267 |             sample_document["document"],
```

### tests/p3_t3_test.py:278 — Presence of :Section label (info)
```text
   276 |                 result = session.run(
   277 |                     """
   278 |                     MATCH (s:Section)
   279 |                     WHERE s.document_id = $doc_id
   280 |                     RETURN s.embedding_version as version
```

### tests/p3_t3_test.py:280 — Use of canonical 'embedding_version' (info)
```text
   278 |                     MATCH (s:Section)
   279 |                     WHERE s.document_id = $doc_id
   280 |                     RETURN s.embedding_version as version
   281 |                     LIMIT 1
   282 |                     """,
```

### tests/p3_t4_integration_test.py:59 — Presence of :Section label (info)
```text
    57 |             session.run(
    58 |                 """
    59 |                 MATCH (s:Section)
    60 |                 WHERE NOT EXISTS { MATCH (doc:Document)-[:HAS_SECTION]->(s) }
    61 |                 DETACH DELETE s
```

### tests/p3_t4_integration_test.py:117 — Presence of :Section label (info)
```text
   115 |             result = session.run(
   116 |                 """
   117 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   118 |                 RETURN s.id as section_id, s.checksum as checksum, s.title as title
   119 |                 ORDER BY s.order
```

### tests/p3_t4_integration_test.py:151 — Presence of :Section label (info)
```text
   149 |             result = session.run(
   150 |                 """
   151 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   152 |                 RETURN s.id as section_id, s.checksum as checksum, s.title as title
   153 |                 ORDER BY s.order
```

### tests/p3_t4_integration_test.py:306 — Presence of :Section label (info)
```text
   304 |             result = session.run(
   305 |                 """
   306 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   307 |                 WHERE s.embedding_version = $emb_version
   308 |                 RETURN s.id as section_id
```

### tests/p3_t4_integration_test.py:307 — Use of canonical 'embedding_version' (info)
```text
   305 |                 """
   306 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   307 |                 WHERE s.embedding_version = $emb_version
   308 |                 RETURN s.id as section_id
   309 |                 """,
```

### tests/p3_t4_integration_test.py:344 — Use of canonical 'embedding_version' (info)
```text
   342 |                     {"key": "node_label", "match": {"value": "Section"}},
   343 |                     {
   344 |                         "key": "embedding_version",
   345 |                         "match": {"value": config.embedding.version},
   346 |                     },
```

### tests/p3_t4_integration_test.py:375 — Use of canonical 'embedding_version' (info)
```text
   373 |                     {"key": "node_label", "match": {"value": "Section"}},
   374 |                     {
   375 |                         "key": "embedding_version",
   376 |                         "match": {"value": config.embedding.version},
   377 |                     },
```

### tests/p3_t4_test.py:54 — Presence of :Section label (info)
```text
    52 |             session.run(
    53 |                 """
    54 |                 MATCH (s:Section)
    55 |                 WHERE s.document_id STARTS WITH '535b6ac'
    56 |                     OR s.document_id STARTS WITH 'c1c5dce'
```

### tests/p3_t4_test.py:140 — Presence of :Section label (info)
```text
   138 |             result = session.run(
   139 |                 """
   140 |                 MATCH (s:Section)
   141 |                 WHERE s.document_id = $doc_id
   142 |                 RETURN s.id as id, s.updated_at as updated_at
```

### tests/p4_t2_perf_test.py:42 — MERGE Document by id (canonical) (info)
```text
    40 |             // Create documents
    41 |             UNWIND range(1, 10) AS doc_num
    42 |             MERGE (d:Document {
    43 |                 id: 'perf-doc-' + toString(doc_num),
    44 |                 title: 'Performance Test Doc ' + toString(doc_num),
```

### tests/p4_t2_perf_test.py:51 — Presence of :Section label (info)
```text
    49 |             WITH d, doc_num
    50 |             UNWIND range(1, 20) AS sec_num
    51 |             MERGE (s:Section {
    52 |                 id: 'perf-sec-' + toString(doc_num) + '-' + toString(sec_num),
    53 |                 document_id: d.id,
```

### tests/p4_t2_perf_test.py:232 — Presence of :Section label (info)
```text
   230 |         template_name = "search_sections"
   231 |         param_names = ["section_ids", "limit"]
   232 |         query = "MATCH (s:Section) WHERE s.id IN $section_ids RETURN s LIMIT $limit"
   233 | 
   234 |         # Measure without cache
```

### tests/p4_t2_perf_test.py:289 — Presence of :Section label (info)
```text
   287 |         {
   288 |             "name": "section_lookup",
   289 |             "query": "MATCH (s:Section) WHERE s.document_id = $doc_id RETURN s",
   290 |             "params": {"doc_id": "perf-doc-1"},
   291 |             "optimize": {"label": "Section", "property": "document_id"},
```

### tests/p4_t2_test.py:48 — MERGE Document by id (canonical) (info)
```text
    46 |         session.run(
    47 |             """
    48 |             MERGE (d:Document {id: 'doc-opt-test', title: 'Test Doc'})
    49 |             MERGE (s1:Section {id: 'sec-opt-1', document_id: 'doc-opt-test', title: 'Section 1', text: 'content'})
    50 |             MERGE (s2:Section {id: 'sec-opt-2', document_id: 'doc-opt-test', title: 'Section 2', text: 'more content'})
```

### tests/p4_t2_test.py:49 — Presence of :Section label (info)
```text
    47 |             """
    48 |             MERGE (d:Document {id: 'doc-opt-test', title: 'Test Doc'})
    49 |             MERGE (s1:Section {id: 'sec-opt-1', document_id: 'doc-opt-test', title: 'Section 1', text: 'content'})
    50 |             MERGE (s2:Section {id: 'sec-opt-2', document_id: 'doc-opt-test', title: 'Section 2', text: 'more content'})
    51 |             MERGE (c1:Command {id: 'cmd-opt-1', name: 'weka status'})
```

### tests/p4_t2_test.py:50 — Presence of :Section label (info)
```text
    48 |             MERGE (d:Document {id: 'doc-opt-test', title: 'Test Doc'})
    49 |             MERGE (s1:Section {id: 'sec-opt-1', document_id: 'doc-opt-test', title: 'Section 1', text: 'content'})
    50 |             MERGE (s2:Section {id: 'sec-opt-2', document_id: 'doc-opt-test', title: 'Section 2', text: 'more content'})
    51 |             MERGE (c1:Command {id: 'cmd-opt-1', name: 'weka status'})
    52 |             MERGE (c2:Command {id: 'cmd-opt-2', name: 'weka fs list'})
```

### tests/p4_t2_test.py:139 — Presence of :Section label (info)
```text
   137 | 
   138 |     def test_records_slow_queries_above_threshold(self, optimizer):
   139 |         query = "MATCH (n:Section) RETURN n LIMIT 10"
   140 |         params = {"limit": 10}
   141 | 
```

### tests/p4_t2_test.py:149 — Presence of :Section label (info)
```text
   147 | 
   148 |     def test_ignores_fast_queries_below_threshold(self, optimizer):
   149 |         query = "MATCH (n:Section) RETURN n LIMIT 10"
   150 |         params = {"limit": 10}
   151 | 
```

### tests/p4_t2_test.py:158 — Presence of :Section label (info)
```text
   156 | 
   157 |     def test_computes_query_fingerprint(self, optimizer):
   158 |         query1 = "MATCH (n:Section {id: $id}) RETURN n"
   159 |         query2 = "MATCH (n:Section {id: $id})  RETURN  n"  # Different whitespace
   160 | 
```

### tests/p4_t2_test.py:159 — Presence of :Section label (info)
```text
   157 |     def test_computes_query_fingerprint(self, optimizer):
   158 |         query1 = "MATCH (n:Section {id: $id}) RETURN n"
   159 |         query2 = "MATCH (n:Section {id: $id})  RETURN  n"  # Different whitespace
   160 | 
   161 |         fp1 = optimizer._compute_query_fingerprint(query1)
```

### tests/p4_t2_test.py:187 — Presence of :Section label (info)
```text
   185 |     ):
   186 |         """Test operator extraction from EXPLAIN plan."""
   187 |         query = "MATCH (s:Section) RETURN s LIMIT 5"
   188 |         params = {}
   189 | 
```

### tests/p4_t2_test.py:256 — Presence of :Section label (info)
```text
   254 | 
   255 |         assert "CREATE INDEX" in cypher
   256 |         assert "FOR (n:Section)" in cypher
   257 |         assert "FOR (n:Command)" in cypher
   258 | 
```

### tests/p4_t2_test.py:282 — Presence of :Section label (info)
```text
   280 |     def test_suggests_limit_when_missing(self, optimizer):
   281 |         """Test suggesting LIMIT when query has none."""
   282 |         query = "MATCH (n:Section) RETURN n"
   283 |         plan = ExplainPlan(
   284 |             query=query,
```

### tests/p4_t2_test.py:307 — Presence of :Section label (info)
```text
   305 |         # Record some slow queries
   306 |         queries = [
   307 |             ("MATCH (s:Section) RETURN s LIMIT 100", {}, 120),
   308 |             ("MATCH (c:Command {name: $name}) RETURN c", {"name": "test"}, 250),
   309 |             ("MATCH (n)-[*1..5]-(m) RETURN n, m LIMIT 10", {}, 180),
```

### tests/p4_t2_test.py:358 — Presence of :Section label (info)
```text
   356 |         param_names = ["section_ids", "limit", "max_hops"]
   357 |         plan = {
   358 |             "query": "MATCH (s:Section) WHERE s.id IN $section_ids RETURN s LIMIT $limit",
   359 |             "params": param_names,
   360 |         }
```

### tests/p4_t3_test.py:219 — Use of canonical 'embedding_version' (info)
```text
   217 |             redis_client=redis_client,
   218 |             schema_version="v1",
   219 |             embedding_version="v1",
   220 |         )
   221 | 
```

### tests/p4_t4_test.py:27 — Presence of :Section label (info)
```text
    25 |             query_text="How to configure TLS?",
    26 |             intent="search",
    27 |             cypher_query="MATCH (s:Section) WHERE s.text CONTAINS 'TLS' RETURN s LIMIT 10",
    28 |             result_ids=["sec1", "sec2", "sec3"],
    29 |             ranking_features={
```

### tests/p4_t4_test.py:172 — Presence of :Section label (info)
```text
   170 |                 f"Query {i}",
   171 |                 "search",
   172 |                 "MATCH (s:Section) RETURN s LIMIT 10",
   173 |                 [f"r{i}"],
   174 |                 {
```

### tests/p4_t4_test.py:261 — Presence of :Section label (info)
```text
   259 | 
   260 |         # Create similar queries (same pattern)
   261 |         base_query = "MATCH (s:Section {document_id: $doc_id}) RETURN s LIMIT 10"
   262 | 
   263 |         for i in range(8):
```

### tests/p4_t4_test.py:317 — Presence of :Section label (info)
```text
   315 | 
   316 |         # Create slow queries accessing unindexed properties
   317 |         slow_query = "MATCH (s:Section) WHERE s.checksum = $cs RETURN s LIMIT 10"
   318 | 
   319 |         for i in range(12):
```

### tests/p4_t4_test.py:353 — Presence of :Section label (info)
```text
   351 |                 f"Report test {i}",
   352 |                 "search",
   353 |                 "MATCH (n:Section) RETURN n LIMIT 10",
   354 |                 [f"r{i}"],
   355 |                 {},
```

### tests/p4_t4_test.py:441 — Presence of :Section label (info)
```text
   439 |                 f"Training query {i}",
   440 |                 "search",
   441 |                 "MATCH (s:Section) RETURN s LIMIT 10",
   442 |                 [f"r{i}"],
   443 |                 {
```

### tests/p5_t2_test.py:170 — Presence of :Section label (info)
```text
   168 |         with trace_cypher_query(
   169 |             "test_template",
   170 |             "MATCH (n:Section) RETURN n LIMIT 1",
   171 |             {"param": "value"},
   172 |         ) as span:
```

### tests/p5_t3_test.py:48 — Presence of :Section label (info)
```text
    46 | 
    47 |         validator = CypherValidator()
    48 |         query = "MATCH (n:Section {id: $sid}) RETURN n LIMIT 10"
    49 |         result = validator.validate(query, {"sid": "test-id"})
    50 | 
```

### tests/p5_t3_test.py:83 — Presence of :Section label (info)
```text
    81 |             )
    82 |             session.run(
    83 |                 "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE"
    84 |             )
    85 | 
```

### tests/p6_t1_test.py:66 — Pattern-scan deletion (fallback) (info)
```text
    64 | 
    65 |     # Clear checksum sets (all tags)
    66 |     for key in redis_sync_client.scan_iter("ingest:checksums:*", count=100):
    67 |         redis_sync_client.delete(key)
    68 | 
```

### tests/p6_t1_test.py:70 — Pattern-scan deletion (fallback) (info)
```text
    68 | 
    69 |     # Clean up any old state keys (from orchestrator)
    70 |     for key in redis_sync_client.scan_iter("ingest:state:*", count=100):
    71 |         redis_sync_client.delete(key)
    72 | 
```

### tests/p6_t1_test.py:616 — Presence of :Section label (info)
```text
   614 |                     """
   615 |                     MATCH (d:Document {source_uri: $uri})
   616 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   617 |                     RETURN count(DISTINCT s) as sections
   618 |                 """,
```

### tests/p6_t2_test.py:343 — Presence of :Section label (info)
```text
   341 |         # Get counts after first run
   342 |         with neo4j_driver.session() as session:
   343 |             result = session.run("MATCH (s:Section) RETURN count(s) as count")
   344 |             sections_count = result.single()["count"]
   345 | 
```

### tests/p6_t2_test.py:361 — Presence of :Section label (info)
```text
   359 |         # Verify no duplication
   360 |         with neo4j_driver.session() as session:
   361 |             result = session.run("MATCH (s:Section) RETURN count(s) as count")
   362 |             sections_count_after = result.single()["count"]
   363 | 
```

### tests/p6_t2_test.py:436 — Presence of :Section label (info)
```text
   434 |                 """
   435 |                 MATCH (d:Document {source_uri: $uri})
   436 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   437 |                 RETURN count(DISTINCT s) as sections
   438 |             """,
```

### tests/p6_t2_test.py:476 — Presence of :Section label (info)
```text
   474 |                 """
   475 |                 MATCH (d:Document {source_uri: $uri})
   476 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   477 |                 RETURN count(DISTINCT s) as sections
   478 |             """,
```

### tests/p6_t2_test.py:550 — Presence of :Section label (info)
```text
   548 |             result = session.run(
   549 |                 """
   550 |                 MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   551 |                 RETURN s.id as id
   552 |                 ORDER BY s.order
```

### tests/p6_t2_test.py:562 — Presence of :Section label (info)
```text
   560 |             session.run(
   561 |                 """
   562 |                 MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   563 |                 DETACH DELETE s
   564 |             """,
```

### tests/p6_t2_test.py:600 — Presence of :Section label (info)
```text
   598 |             result = session.run(
   599 |                 """
   600 |                 MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   601 |                 RETURN s.id as id
   602 |                 ORDER BY s.order
```

### tests/p6_t2_test.py:824 — Presence of :Section label (info)
```text
   822 |             result = session.run(
   823 |                 """
   824 |                 MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   825 |                 RETURN s.title as title, s.anchor as anchor, s.text as text
   826 |                 ORDER BY s.order
```

### tests/p6_t2_test.py:902 — Presence of :Section label (info)
```text
   900 |                 """
   901 |                 MATCH (d:Document {source_uri: $uri})
   902 |                 MATCH (s:Section {document_id: d.id})-[m:MENTIONS]->(e)
   903 |                 RETURN count(m) as mention_count
   904 |             """,
```

### tests/p6_t2_test.py:986 — Presence of :Section label (info)
```text
   984 |             result = session.run(
   985 |                 """
   986 |                 MATCH (d:Document {source_uri: $uri})-[r:HAS_SECTION]->(s:Section)
   987 |                 RETURN count(r) as rel_count
   988 |             """,
```

### tests/p6_t2_test.py:1098 — Presence of :Section label (info)
```text
  1096 |                 """
  1097 |                 MATCH (d:Document {source_uri: $uri})
  1098 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
  1099 |                 RETURN count(DISTINCT s) as sections
  1100 |             """,
```

### tests/p6_t3_test.py:64 — Pattern-scan deletion (fallback) (info)
```text
    62 |     )
    63 |     # Clear ingestion-related keys before each test to avoid duplicate detection
    64 |     for key in client.scan_iter("ingest:*", count=1000):
    65 |         client.delete(key)
    66 | 
```

### tests/p6_t3_test.py:68 — Pattern-scan deletion (fallback) (info)
```text
    66 | 
    67 |     # Also clear checksum sets
    68 |     for key in client.scan_iter("ingest:checksums:*", count=1000):
    69 |         client.delete(key)
    70 | 
```

### tests/p6_t3_test.py:265 — Pattern-scan deletion (fallback) (info)
```text
   263 |         """
   264 |         # Count jobs before
   265 |         keys_before = list(redis_client.scan_iter("ingest:state:*", count=1000))
   266 | 
   267 |         result = run_cli(["ingest", str(sample_markdown), "--dry-run"])
```

### tests/p6_t3_test.py:274 — Pattern-scan deletion (fallback) (info)
```text
   272 | 
   273 |         # Count jobs after
   274 |         keys_after = list(redis_client.scan_iter("ingest:state:*", count=1000))
   275 | 
   276 |         # No new jobs should be created
```

### tests/p6_t4_test.py:524 — Use of canonical 'embedding_version' (info)
```text
   522 |         assert stats["sot"] == "qdrant"
   523 |         assert "sections_indexed" in stats
   524 |         assert "embedding_version" in stats
   525 |         assert isinstance(stats["sections_indexed"], int)
   526 | 
```

### tests/p6_t4_test.py:542 — Use of canonical 'embedding_version' (info)
```text
   540 |         assert stats["sot"] == "neo4j"
   541 |         assert "sections_indexed" in stats
   542 |         assert "embedding_version" in stats
   543 | 
   544 |     def test_render_markdown_format(self, report_gen, sample_parsed_doc):
```

### tests/test_integration_prephase7.py:291 — Use of canonical 'embedding_version' (info)
```text
   289 |             "node_label": "Section",
   290 |             "text": test_text,
   291 |             "embedding_version": config.embedding.version,
   292 |             "embedding_provider": config.embedding.provider,
   293 |             "embedding_dimensions": len(embeddings[0]),
```

### tests/test_integration_prephase7.py:293 — Use of canonical 'embedding_dimensions' (info)
```text
   291 |             "embedding_version": config.embedding.version,
   292 |             "embedding_provider": config.embedding.provider,
   293 |             "embedding_dimensions": len(embeddings[0]),
   294 |             "embedding_task": config.embedding.task,
   295 |             "embedding_timestamp": "2025-01-23T00:00:00Z",
```

### tests/test_integration_prephase7.py:295 — Use of canonical 'embedding_timestamp' (info)
```text
   293 |             "embedding_dimensions": len(embeddings[0]),
   294 |             "embedding_task": config.embedding.task,
   295 |             "embedding_timestamp": "2025-01-23T00:00:00Z",
   296 |         }
   297 | 
```

### tests/test_integration_prephase7.py:300 — Use of canonical 'embedding_version' (info)
```text
   298 |         # Verify all required fields present
   299 |         required_fields = [
   300 |             "embedding_version",
   301 |             "embedding_provider",
   302 |             "embedding_dimensions",
```

### tests/test_integration_prephase7.py:302 — Use of canonical 'embedding_dimensions' (info)
```text
   300 |             "embedding_version",
   301 |             "embedding_provider",
   302 |             "embedding_dimensions",
   303 |             "embedding_task",
   304 |         ]
```

### tests/test_integration_prephase7.py:360 — Presence of :Section label (info)
```text
   358 |             # Check if we have any data
   359 |             with neo4j.session() as session:
   360 |                 result = session.run("MATCH (s:Section) RETURN count(s) as count")
   361 |                 section_count = result.single()["count"]
   362 | 
```

### tests/test_integration_prephase7.py:379 — Presence of :Section label (info)
```text
   377 |         test_node_ids = []
   378 |         with neo4j.session() as session:
   379 |             result = session.run("MATCH (s:Section) RETURN s.id as id LIMIT 3")
   380 |             test_node_ids = [record["id"] for record in result]
   381 | 
```

### tests/test_integration_prephase7.py:389 — Presence of :Section label (info)
```text
   387 |         coverage_query = """
   388 |         UNWIND $ids AS sid
   389 |         MATCH (s:Section {id: sid})
   390 |         OPTIONAL MATCH (s)-[r]->()
   391 |         WITH s, count(DISTINCT r) AS conn_count
```

### tests/test_integration_prephase7.py:449 — Use of canonical 'embedding_version' (info)
```text
   447 |         print("Test 4: Embedding version filter...")
   448 | 
   449 |         # Check that embedding_version field exists on sections
   450 |         with neo4j.session() as session:
   451 |             result = session.run(
```

### tests/test_integration_prephase7.py:453 — Presence of :Section label (info)
```text
   451 |             result = session.run(
   452 |                 """
   453 |                 MATCH (s:Section)
   454 |                 WHERE s.embedding_version IS NOT NULL
   455 |                 RETURN count(s) as count
```

### tests/test_integration_prephase7.py:454 — Use of canonical 'embedding_version' (info)
```text
   452 |                 """
   453 |                 MATCH (s:Section)
   454 |                 WHERE s.embedding_version IS NOT NULL
   455 |                 RETURN count(s) as count
   456 |             """
```

### tests/test_integration_prephase7.py:461 — Use of canonical 'embedding_version' (info)
```text
   459 | 
   460 |         if sections_with_version > 0:
   461 |             print(f"  ✓ {sections_with_version} sections have embedding_version field")
   462 |         else:
   463 |             print("  ⚠️  No sections have embedding_version (legacy data?)")
```

### tests/test_integration_prephase7.py:463 — Use of canonical 'embedding_version' (info)
```text
   461 |             print(f"  ✓ {sections_with_version} sections have embedding_version field")
   462 |         else:
   463 |             print("  ⚠️  No sections have embedding_version (legacy data?)")
   464 | 
   465 |         print("✓ H4: All search with coverage tests passed")
```

### tests/test_jina_adaptive_batching.py:32 — Reference to canonical model v3 (info)
```text
    30 |         """Test that small batches are not split."""
    31 |         provider = JinaEmbeddingProvider(
    32 |             model="jina-embeddings-v3",
    33 |             dims=1024,
    34 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:49 — Reference to canonical model v3 (info)
```text
    47 |         """Test that batches exceeding MAX_TEXTS_PER_BATCH are split."""
    48 |         provider = JinaEmbeddingProvider(
    49 |             model="jina-embeddings-v3",
    50 |             dims=1024,
    51 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:72 — Reference to canonical model v3 (info)
```text
    70 |         """Test that batches exceeding MAX_CHARS_PER_BATCH are split by size."""
    71 |         provider = JinaEmbeddingProvider(
    72 |             model="jina-embeddings-v3",
    73 |             dims=1024,
    74 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:97 — Reference to canonical model v3 (info)
```text
    95 |         """Test that single oversized text is truncated with warning."""
    96 |         provider = JinaEmbeddingProvider(
    97 |             model="jina-embeddings-v3",
    98 |             dims=1024,
    99 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:155 — Reference to canonical model v3 (info)
```text
   153 | 
   154 |         provider = JinaEmbeddingProvider(
   155 |             model="jina-embeddings-v3",
   156 |             dims=1024,
   157 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:190 — Reference to canonical model v3 (info)
```text
   188 | 
   189 |         provider = JinaEmbeddingProvider(
   190 |             model="jina-embeddings-v3",
   191 |             dims=1024,
   192 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:236 — Reference to canonical model v3 (info)
```text
   234 | 
   235 |         provider = JinaEmbeddingProvider(
   236 |             model="jina-embeddings-v3",
   237 |             dims=1024,
   238 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:390 — Reference to canonical model v3 (info)
```text
   388 | 
   389 |         provider = JinaEmbeddingProvider(
   390 |             model="jina-embeddings-v3",
   391 |             dims=1024,
   392 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:405 — Reference to canonical model v3 (info)
```text
   403 |             request_body = call_args.kwargs["json"]
   404 |             assert request_body["task"] == "retrieval.query"
   405 |             assert request_body["model"] == "jina-embeddings-v3"
   406 |             assert request_body["truncate"] is False
   407 | 
```

### tests/test_jina_adaptive_batching.py:422 — Reference to canonical model v3 (info)
```text
   420 | 
   421 |         provider = JinaEmbeddingProvider(
   422 |             model="jina-embeddings-v3",
   423 |             dims=1024,
   424 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:467 — Reference to canonical model v3 (info)
```text
   465 | 
   466 |         provider = JinaEmbeddingProvider(
   467 |             model="jina-embeddings-v3",
   468 |             dims=1024,
   469 |             api_key="test-key",
```

### tests/test_jina_adaptive_batching.py:499 — Reference to canonical model v3 (info)
```text
   497 | 
   498 |         provider = JinaEmbeddingProvider(
   499 |             model="jina-embeddings-v3",
   500 |             dims=1024,
   501 |             api_key="test-key",
```

### tests/test_phase2_provider_wiring.py:138 — Use of canonical 'embedding_version' (info)
```text
   136 |         # Simulate metadata that should be added
   137 |         metadata = {
   138 |             "embedding_version": "miniLM-L6-v2-2024-01-01",
   139 |             "embedding_provider": "sentence-transformers",
   140 |             "embedding_dimensions": 384,
```

### tests/test_phase2_provider_wiring.py:140 — Use of canonical 'embedding_dimensions' (info)
```text
   138 |             "embedding_version": "miniLM-L6-v2-2024-01-01",
   139 |             "embedding_provider": "sentence-transformers",
   140 |             "embedding_dimensions": 384,
   141 |             "embedding_task": "retrieval.passage",
   142 |             "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
```

### tests/test_phase2_provider_wiring.py:142 — Use of canonical 'embedding_timestamp' (info)
```text
   140 |             "embedding_dimensions": 384,
   141 |             "embedding_task": "retrieval.passage",
   142 |             "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
   143 |         }
   144 | 
```

### tests/test_phase2_provider_wiring.py:147 — Use of canonical 'embedding_version' (info)
```text
   145 |         # Verify all required fields are present
   146 |         required_fields = [
   147 |             "embedding_version",
   148 |             "embedding_provider",
   149 |             "embedding_dimensions",
```

### tests/test_phase2_provider_wiring.py:149 — Use of canonical 'embedding_dimensions' (info)
```text
   147 |             "embedding_version",
   148 |             "embedding_provider",
   149 |             "embedding_dimensions",
   150 |             "embedding_task",
   151 |             "embedding_timestamp",
```

### tests/test_phase2_provider_wiring.py:151 — Use of canonical 'embedding_timestamp' (info)
```text
   149 |             "embedding_dimensions",
   150 |             "embedding_task",
   151 |             "embedding_timestamp",
   152 |         ]
   153 | 
```

### tests/test_phase2_provider_wiring.py:158 — Use of canonical 'embedding_timestamp' (info)
```text
   156 | 
   157 |         # Verify timestamp format
   158 |         assert metadata["embedding_timestamp"].endswith(
   159 |             "Z"
   160 |         ), "Timestamp should be ISO-8601 UTC with Z suffix"
```

### tests/test_phase2_provider_wiring.py:163 — Use of canonical 'embedding_version' (info)
```text
   161 | 
   162 |         print("✓ Embedding metadata structure correct")
   163 |         print(f"  - Version: {metadata['embedding_version']}")
   164 |         print(f"  - Provider: {metadata['embedding_provider']}")
   165 |         print(f"  - Dimensions: {metadata['embedding_dimensions']}")
```

### tests/test_phase2_provider_wiring.py:165 — Use of canonical 'embedding_dimensions' (info)
```text
   163 |         print(f"  - Version: {metadata['embedding_version']}")
   164 |         print(f"  - Provider: {metadata['embedding_provider']}")
   165 |         print(f"  - Dimensions: {metadata['embedding_dimensions']}")
   166 |         print(f"  - Task: {metadata['embedding_task']}")
   167 | 
```

### tests/test_phase5_response_schema.py:103 — Presence of :Chunk label (dual-label support) (info)
```text
   101 | 
   102 |         content = ddl_path.read_text()
   103 |         assert "SET s:Chunk" in content
   104 |         assert "session_id_unique" in content
   105 |         assert "query_id_unique" in content
```

### tests/test_phase7c_dual_write.py:208 — Use of canonical 'embedding_dimensions' (info)
```text
   206 |         ), "Legacy collection has wrong dimensions"
   207 |         assert legacy_point[0].payload["node_id"] == section_id
   208 |         assert legacy_point[0].payload["embedding_dimensions"] == 384
   209 |         assert legacy_point[0].payload["embedding_provider"] == "sentence-transformers"
   210 | 
```

### tests/test_phase7c_dual_write.py:220 — Use of canonical 'embedding_dimensions' (info)
```text
   218 |         assert len(new_point[0].vector) == 1024, "New collection has wrong dimensions"
   219 |         assert new_point[0].payload["node_id"] == section_id
   220 |         assert new_point[0].payload["embedding_dimensions"] == 1024
   221 |         # Provider may be jina-ai or ollama depending on ENV
   222 |         assert new_point[0].payload["embedding_provider"] in [
```

### tests/test_phase7c_dual_write.py:267 — Presence of :Section label (info)
```text
   265 |             result = session.run(
   266 |                 """
   267 |                 MATCH (s:Section {id: $section_id})
   268 |                 RETURN s.embedding_version as version,
   269 |                        s.embedding_provider as provider,
```

### tests/test_phase7c_dual_write.py:268 — Use of canonical 'embedding_version' (info)
```text
   266 |                 """
   267 |                 MATCH (s:Section {id: $section_id})
   268 |                 RETURN s.embedding_version as version,
   269 |                        s.embedding_provider as provider,
   270 |                        s.embedding_dimensions as dimensions,
```

### tests/test_phase7c_dual_write.py:270 — Use of canonical 'embedding_dimensions' (info)
```text
   268 |                 RETURN s.embedding_version as version,
   269 |                        s.embedding_provider as provider,
   270 |                        s.embedding_dimensions as dimensions,
   271 |                        s.embedding_timestamp as timestamp,
   272 |                        s.embedding_task as task
```

### tests/test_phase7c_dual_write.py:271 — Use of canonical 'embedding_timestamp' (info)
```text
   269 |                        s.embedding_provider as provider,
   270 |                        s.embedding_dimensions as dimensions,
   271 |                        s.embedding_timestamp as timestamp,
   272 |                        s.embedding_task as task
   273 |                 """,
```

### tests/test_phase7c_index_registry.py:179 — Reference to canonical model v3 (info)
```text
   177 |         )
   178 | 
   179 |         provider = MockProvider("jina-ai", "jina-embeddings-v3", 1024)
   180 | 
   181 |         # Should not raise but should warn
```

### tests/test_phase7c_ingestion.py:5 — Presence of :Chunk label (dual-label support) (info)
```text
     3 | 
     4 | Validates:
     5 | - Sections are dual-labeled as Section:Chunk
     6 | - All required embedding fields are present and validated
     7 | - Embeddings use configured provider (1024-D by default)
```

### tests/test_phase7c_ingestion.py:97 — Presence of :Chunk label (dual-label support) (info)
```text
    95 |         self, graph_builder, sample_document, sample_sections, neo4j_driver
    96 |     ):
    97 |         """Test that sections are dual-labeled as Section:Chunk."""
    98 |         # Ingest document
    99 |         graph_builder.upsert_document(sample_document, sample_sections, {}, [])
```

### tests/test_phase7c_ingestion.py:117 — Presence of :Section label (info)
```text
   115 |                 assert (
   116 |                     "Section" in labels
   117 |                 ), f"Section {section['id']} missing :Section label"
   118 |                 assert (
   119 |                     "Chunk" in labels
```

### tests/test_phase7c_ingestion.py:120 — Presence of :Chunk label (dual-label support) (info)
```text
   118 |                 assert (
   119 |                     "Chunk" in labels
   120 |                 ), f"Section {section['id']} missing :Chunk label (v3 compat)"
   121 | 
   122 |     def test_required_embedding_fields_present(
```

### tests/test_phase7c_ingestion.py:136 — Presence of :Section label (info)
```text
   134 |                 result = session.run(
   135 |                     """
   136 |                     MATCH (s:Section {id: $section_id})
   137 |                     RETURN s.vector_embedding as embedding,
   138 |                            s.embedding_version as version,
```

### tests/test_phase7c_ingestion.py:137 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   135 |                     """
   136 |                     MATCH (s:Section {id: $section_id})
   137 |                     RETURN s.vector_embedding as embedding,
   138 |                            s.embedding_version as version,
   139 |                            s.embedding_provider as provider,
```

### tests/test_phase7c_ingestion.py:138 — Use of canonical 'embedding_version' (info)
```text
   136 |                     MATCH (s:Section {id: $section_id})
   137 |                     RETURN s.vector_embedding as embedding,
   138 |                            s.embedding_version as version,
   139 |                            s.embedding_provider as provider,
   140 |                            s.embedding_dimensions as dimensions,
```

### tests/test_phase7c_ingestion.py:140 — Use of canonical 'embedding_dimensions' (info)
```text
   138 |                            s.embedding_version as version,
   139 |                            s.embedding_provider as provider,
   140 |                            s.embedding_dimensions as dimensions,
   141 |                            s.embedding_timestamp as timestamp,
   142 |                            s.embedding_task as task
```

### tests/test_phase7c_ingestion.py:141 — Use of canonical 'embedding_timestamp' (info)
```text
   139 |                            s.embedding_provider as provider,
   140 |                            s.embedding_dimensions as dimensions,
   141 |                            s.embedding_timestamp as timestamp,
   142 |                            s.embedding_task as task
   143 |                     """,
```

### tests/test_phase7c_ingestion.py:152 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   150 |                 assert (
   151 |                     record["embedding"] is not None
   152 |                 ), f"Section {section['id']} missing REQUIRED vector_embedding"
   153 |                 assert (
   154 |                     record["version"] is not None
```

### tests/test_phase7c_ingestion.py:155 — Use of canonical 'embedding_version' (info)
```text
   153 |                 assert (
   154 |                     record["version"] is not None
   155 |                 ), f"Section {section['id']} missing REQUIRED embedding_version"
   156 |                 assert (
   157 |                     record["provider"] is not None
```

### tests/test_phase7c_ingestion.py:161 — Use of canonical 'embedding_dimensions' (info)
```text
   159 |                 assert (
   160 |                     record["dimensions"] is not None
   161 |                 ), f"Section {section['id']} missing REQUIRED embedding_dimensions"
   162 |                 assert (
   163 |                     record["timestamp"] is not None
```

### tests/test_phase7c_ingestion.py:164 — Use of canonical 'embedding_timestamp' (info)
```text
   162 |                 assert (
   163 |                     record["timestamp"] is not None
   164 |                 ), f"Section {section['id']} missing REQUIRED embedding_timestamp"
   165 | 
   166 |                 # Validate dimensions match config
```

### tests/test_phase7c_ingestion.py:192 — Presence of :Section label (info)
```text
   190 |             result = session.run(
   191 |                 """
   192 |                 MATCH (s:Section)
   193 |                 WHERE s.id STARTS WITH 'test-section-'
   194 |                 RETURN s.id as id,
```

### tests/test_phase7c_ingestion.py:195 — Use of canonical 'embedding_dimensions' (info)
```text
   193 |                 WHERE s.id STARTS WITH 'test-section-'
   194 |                 RETURN s.id as id,
   195 |                        s.embedding_dimensions as dims,
   196 |                        size(s.vector_embedding) as vector_dims
   197 |                 """
```

### tests/test_phase7c_ingestion.py:196 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   194 |                 RETURN s.id as id,
   195 |                        s.embedding_dimensions as dims,
   196 |                        size(s.vector_embedding) as vector_dims
   197 |                 """
   198 |             )
```

### tests/test_phase7c_ingestion.py:273 — Use of canonical 'embedding_version' (info)
```text
   271 |             # Verify embedding metadata in payload
   272 |             assert (
   273 |                 "embedding_version" in payload
   274 |             ), "Missing embedding_version in payload"
   275 |             assert (
```

### tests/test_phase7c_ingestion.py:274 — Use of canonical 'embedding_version' (info)
```text
   272 |             assert (
   273 |                 "embedding_version" in payload
   274 |             ), "Missing embedding_version in payload"
   275 |             assert (
   276 |                 "embedding_provider" in payload
```

### tests/test_phase7c_ingestion.py:279 — Use of canonical 'embedding_dimensions' (info)
```text
   277 |             ), "Missing embedding_provider in payload"
   278 |             assert (
   279 |                 "embedding_dimensions" in payload
   280 |             ), "Missing embedding_dimensions in payload"
   281 |             assert "embedding_task" in payload, "Missing embedding_task in payload"
```

### tests/test_phase7c_ingestion.py:280 — Use of canonical 'embedding_dimensions' (info)
```text
   278 |             assert (
   279 |                 "embedding_dimensions" in payload
   280 |             ), "Missing embedding_dimensions in payload"
   281 |             assert "embedding_task" in payload, "Missing embedding_task in payload"
   282 | 
```

### tests/test_phase7c_ingestion.py:284 — Use of canonical 'embedding_dimensions' (info)
```text
   282 | 
   283 |             # Verify dimensions
   284 |             assert payload["embedding_dimensions"] == config.embedding.dims, (
   285 |                 f"Point {payload['node_id']} has wrong dimension metadata: "
   286 |                 f"{payload['embedding_dimensions']}"
```

### tests/test_phase7c_ingestion.py:286 — Use of canonical 'embedding_dimensions' (info)
```text
   284 |             assert payload["embedding_dimensions"] == config.embedding.dims, (
   285 |                 f"Point {payload['node_id']} has wrong dimension metadata: "
   286 |                 f"{payload['embedding_dimensions']}"
   287 |             )
   288 | 
```

### tests/test_phase7c_ingestion.py:328 — Presence of :Section label (info)
```text
   326 |             result = session.run(
   327 |                 """
   328 |                 MATCH (s:Section)
   329 |                 WHERE s.id = 'test-validation-section'
   330 |                   AND (s.vector_embedding IS NULL
```

### tests/test_phase7c_ingestion.py:330 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   328 |                 MATCH (s:Section)
   329 |                 WHERE s.id = 'test-validation-section'
   330 |                   AND (s.vector_embedding IS NULL
   331 |                    OR s.embedding_version IS NULL
   332 |                    OR s.embedding_provider IS NULL
```

### tests/test_phase7c_ingestion.py:331 — Use of canonical 'embedding_version' (info)
```text
   329 |                 WHERE s.id = 'test-validation-section'
   330 |                   AND (s.vector_embedding IS NULL
   331 |                    OR s.embedding_version IS NULL
   332 |                    OR s.embedding_provider IS NULL
   333 |                    OR s.embedding_dimensions IS NULL
```

### tests/test_phase7c_ingestion.py:333 — Use of canonical 'embedding_dimensions' (info)
```text
   331 |                    OR s.embedding_version IS NULL
   332 |                    OR s.embedding_provider IS NULL
   333 |                    OR s.embedding_dimensions IS NULL
   334 |                    OR s.embedding_timestamp IS NULL)
   335 |                 RETURN count(s) as incomplete_count
```

### tests/test_phase7c_ingestion.py:334 — Use of canonical 'embedding_timestamp' (info)
```text
   332 |                    OR s.embedding_provider IS NULL
   333 |                    OR s.embedding_dimensions IS NULL
   334 |                    OR s.embedding_timestamp IS NULL)
   335 |                 RETURN count(s) as incomplete_count
   336 |                 """
```

### tests/test_phase7c_ingestion.py:426 — Presence of :Section label (info)
```text
   424 |             session.run(
   425 |                 """
   426 |                 MATCH (s:Section)
   427 |                 WHERE s.id STARTS WITH 'test-section-' OR s.id = 'test-validation-section'
   428 |                    OR s.id = 'test-provider-section'
```

### tests/test_phase7c_schema_v2_1.py:63 — Presence of :Section label (info)
```text
    61 |             # Get counts
    62 |             section_count_result = session.run(
    63 |                 "MATCH (s:Section) RETURN count(s) as count"
    64 |             )
    65 |             section_count = section_count_result.single()["count"]
```

### tests/test_phase7c_schema_v2_1.py:67 — Presence of :Chunk label (dual-label support) (info)
```text
    65 |             section_count = section_count_result.single()["count"]
    66 | 
    67 |             chunk_count_result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    68 |             chunk_count = chunk_count_result.single()["count"]
    69 | 
```

### tests/test_phase7c_schema_v2_1.py:75 — MERGE Document by id (canonical) (info)
```text
    73 |                 session.run(
    74 |                     """
    75 |                     MERGE (d:Document {id: 'test-doc'})
    76 |                     SET d.title = 'Test Document'
    77 |                     MERGE (s:Section:Chunk {id: 'test-section'})
```

### tests/test_phase7c_schema_v2_1.py:77 — Presence of :Chunk label (dual-label support) (info)
```text
    75 |                     MERGE (d:Document {id: 'test-doc'})
    76 |                     SET d.title = 'Test Document'
    77 |                     MERGE (s:Section:Chunk {id: 'test-section'})
    78 |                     SET s.text = 'Test text',
    79 |                         s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:77 — Presence of :Section label (info)
```text
    75 |                     MERGE (d:Document {id: 'test-doc'})
    76 |                     SET d.title = 'Test Document'
    77 |                     MERGE (s:Section:Chunk {id: 'test-section'})
    78 |                     SET s.text = 'Test text',
    79 |                         s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:86 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
    84 |                         s.tokens = 10,
    85 |                         s.checksum = 'test',
    86 |                         s.vector_embedding = $vector,
    87 |                         s.embedding_version = 'test-v1',
    88 |                         s.embedding_provider = 'test',
```

### tests/test_phase7c_schema_v2_1.py:87 — Use of canonical 'embedding_version' (info)
```text
    85 |                         s.checksum = 'test',
    86 |                         s.vector_embedding = $vector,
    87 |                         s.embedding_version = 'test-v1',
    88 |                         s.embedding_provider = 'test',
    89 |                         s.embedding_timestamp = datetime(),
```

### tests/test_phase7c_schema_v2_1.py:89 — Use of canonical 'embedding_timestamp' (info)
```text
    87 |                         s.embedding_version = 'test-v1',
    88 |                         s.embedding_provider = 'test',
    89 |                         s.embedding_timestamp = datetime(),
    90 |                         s.embedding_dimensions = 1024
    91 |                     MERGE (d)-[:HAS_SECTION]->(s)
```

### tests/test_phase7c_schema_v2_1.py:90 — Use of canonical 'embedding_dimensions' (info)
```text
    88 |                         s.embedding_provider = 'test',
    89 |                         s.embedding_timestamp = datetime(),
    90 |                         s.embedding_dimensions = 1024
    91 |                     MERGE (d)-[:HAS_SECTION]->(s)
    92 |                     """,
```

### tests/test_phase7c_schema_v2_1.py:157 — Presence of :Chunk label (dual-label support) (info)
```text
   155 |             session.run(
   156 |                 """
   157 |                 MERGE (s:Section:Chunk {id: 'test-validation-complete'})
   158 |                 SET s.text = 'Test',
   159 |                     s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:157 — Presence of :Section label (info)
```text
   155 |             session.run(
   156 |                 """
   157 |                 MERGE (s:Section:Chunk {id: 'test-validation-complete'})
   158 |                 SET s.text = 'Test',
   159 |                     s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:165 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   163 |                     s.order = 0,
   164 |                     s.tokens = 10,
   165 |                     s.vector_embedding = $vector,
   166 |                     s.embedding_version = 'test-v1',
   167 |                     s.embedding_provider = 'test',
```

### tests/test_phase7c_schema_v2_1.py:166 — Use of canonical 'embedding_version' (info)
```text
   164 |                     s.tokens = 10,
   165 |                     s.vector_embedding = $vector,
   166 |                     s.embedding_version = 'test-v1',
   167 |                     s.embedding_provider = 'test',
   168 |                     s.embedding_timestamp = datetime(),
```

### tests/test_phase7c_schema_v2_1.py:168 — Use of canonical 'embedding_timestamp' (info)
```text
   166 |                     s.embedding_version = 'test-v1',
   167 |                     s.embedding_provider = 'test',
   168 |                     s.embedding_timestamp = datetime(),
   169 |                     s.embedding_dimensions = 1024
   170 |             """,
```

### tests/test_phase7c_schema_v2_1.py:169 — Use of canonical 'embedding_dimensions' (info)
```text
   167 |                     s.embedding_provider = 'test',
   168 |                     s.embedding_timestamp = datetime(),
   169 |                     s.embedding_dimensions = 1024
   170 |             """,
   171 |                 vector=test_vector,
```

### tests/test_phase7c_schema_v2_1.py:177 — Presence of :Section label (info)
```text
   175 |             result = session.run(
   176 |                 """
   177 |                 MATCH (s:Section {id: 'test-validation-complete'})
   178 |                 WHERE s.vector_embedding IS NULL
   179 |                    OR s.embedding_version IS NULL
```

### tests/test_phase7c_schema_v2_1.py:178 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   176 |                 """
   177 |                 MATCH (s:Section {id: 'test-validation-complete'})
   178 |                 WHERE s.vector_embedding IS NULL
   179 |                    OR s.embedding_version IS NULL
   180 |                    OR s.embedding_provider IS NULL
```

### tests/test_phase7c_schema_v2_1.py:179 — Use of canonical 'embedding_version' (info)
```text
   177 |                 MATCH (s:Section {id: 'test-validation-complete'})
   178 |                 WHERE s.vector_embedding IS NULL
   179 |                    OR s.embedding_version IS NULL
   180 |                    OR s.embedding_provider IS NULL
   181 |                    OR s.embedding_timestamp IS NULL
```

### tests/test_phase7c_schema_v2_1.py:181 — Use of canonical 'embedding_timestamp' (info)
```text
   179 |                    OR s.embedding_version IS NULL
   180 |                    OR s.embedding_provider IS NULL
   181 |                    OR s.embedding_timestamp IS NULL
   182 |                    OR s.embedding_dimensions IS NULL
   183 |                 RETURN count(s) as missing_count
```

### tests/test_phase7c_schema_v2_1.py:182 — Use of canonical 'embedding_dimensions' (info)
```text
   180 |                    OR s.embedding_provider IS NULL
   181 |                    OR s.embedding_timestamp IS NULL
   182 |                    OR s.embedding_dimensions IS NULL
   183 |                 RETURN count(s) as missing_count
   184 |             """
```

### tests/test_phase7c_schema_v2_1.py:193 — Presence of :Section label (info)
```text
   191 |             # Cleanup
   192 |             session.run(
   193 |                 "MATCH (s:Section {id: 'test-validation-complete'}) DETACH DELETE s"
   194 |             )
   195 | 
```

### tests/test_phase7c_schema_v2_1.py:378 — Presence of :Chunk label (dual-label support) (info)
```text
   376 |             session.run(
   377 |                 """
   378 |                 MERGE (s:Section:Chunk {id: 'test-complete-section'})
   379 |                 SET s.text = 'Test text',
   380 |                     s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:378 — Presence of :Section label (info)
```text
   376 |             session.run(
   377 |                 """
   378 |                 MERGE (s:Section:Chunk {id: 'test-complete-section'})
   379 |                 SET s.text = 'Test text',
   380 |                     s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:387 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   385 |                     s.tokens = 10,
   386 |                     s.checksum = 'test',
   387 |                     s.vector_embedding = $vector,
   388 |                     s.embedding_version = 'jina-embeddings-v4',
   389 |                     s.embedding_provider = 'jina-ai',
```

### tests/test_phase7c_schema_v2_1.py:388 — Use of canonical 'embedding_version' (info)
```text
   386 |                     s.checksum = 'test',
   387 |                     s.vector_embedding = $vector,
   388 |                     s.embedding_version = 'jina-embeddings-v4',
   389 |                     s.embedding_provider = 'jina-ai',
   390 |                     s.embedding_timestamp = datetime(),
```

### tests/test_phase7c_schema_v2_1.py:390 — Use of canonical 'embedding_timestamp' (info)
```text
   388 |                     s.embedding_version = 'jina-embeddings-v4',
   389 |                     s.embedding_provider = 'jina-ai',
   390 |                     s.embedding_timestamp = datetime(),
   391 |                     s.embedding_dimensions = 1024
   392 |                 """,
```

### tests/test_phase7c_schema_v2_1.py:391 — Use of canonical 'embedding_dimensions' (info)
```text
   389 |                     s.embedding_provider = 'jina-ai',
   390 |                     s.embedding_timestamp = datetime(),
   391 |                     s.embedding_dimensions = 1024
   392 |                 """,
   393 |                 vector=test_vector,
```

### tests/test_phase7c_schema_v2_1.py:399 — Presence of :Section label (info)
```text
   397 |             result = session.run(
   398 |                 """
   399 |                 MATCH (s:Section {id: 'test-complete-section'})
   400 |                 RETURN s, size(s.vector_embedding) as dims
   401 |                 """
```

### tests/test_phase7c_schema_v2_1.py:400 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   398 |                 """
   399 |                 MATCH (s:Section {id: 'test-complete-section'})
   400 |                 RETURN s, size(s.vector_embedding) as dims
   401 |                 """
   402 |             )
```

### tests/test_phase7c_schema_v2_1.py:424 — Presence of :Section label (info)
```text
   422 |             session.run(
   423 |                 """
   424 |                 MERGE (s:Section {id: $id})
   425 |                 SET s.text = 'Test text',
   426 |                     s.document_id = 'test-doc',
```

### tests/test_phase7c_schema_v2_1.py:432 — Use of canonical 'embedding_version' (info)
```text
   430 |                     s.order = 0,
   431 |                     s.tokens = 10
   432 |                 // Deliberately omitting: vector_embedding, embedding_version, etc.
   433 |                 """,
   434 |                 id=test_id,
```

### tests/test_phase7c_schema_v2_1.py:432 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   430 |                     s.order = 0,
   431 |                     s.tokens = 10
   432 |                 // Deliberately omitting: vector_embedding, embedding_version, etc.
   433 |                 """,
   434 |                 id=test_id,
```

### tests/test_phase7c_schema_v2_1.py:438 — Presence of :Section label (info)
```text
   436 | 
   437 |             # Verify this incomplete section exists (DB allowed it)
   438 |             result = session.run("MATCH (s:Section {id: $id}) RETURN s", id=test_id)
   439 |             assert result.single() is not None, "Incomplete section should exist in DB"
   440 | 
```

### tests/test_phase7c_schema_v2_1.py:445 — Presence of :Section label (info)
```text
   443 |             result = session.run(
   444 |                 """
   445 |                 MATCH (s:Section {id: $id})
   446 |                 WHERE s.vector_embedding IS NULL
   447 |                    OR s.embedding_version IS NULL
```

### tests/test_phase7c_schema_v2_1.py:446 — Use of canonical 'vector_embedding' (Neo4j) (info)
```text
   444 |                 """
   445 |                 MATCH (s:Section {id: $id})
   446 |                 WHERE s.vector_embedding IS NULL
   447 |                    OR s.embedding_version IS NULL
   448 |                    OR s.embedding_provider IS NULL
```

### tests/test_phase7c_schema_v2_1.py:447 — Use of canonical 'embedding_version' (info)
```text
   445 |                 MATCH (s:Section {id: $id})
   446 |                 WHERE s.vector_embedding IS NULL
   447 |                    OR s.embedding_version IS NULL
   448 |                    OR s.embedding_provider IS NULL
   449 |                 RETURN count(s) as incomplete_count
```

### tests/test_phase7c_schema_v2_1.py:462 — Presence of :Section label (info)
```text
   460 | 
   461 |             # Cleanup test data
   462 |             session.run("MATCH (s:Section {id: $id}) DETACH DELETE s", id=test_id)
   463 | 
   464 | 
```

### tests/test_phase7e_phase0.py:511 — Presence of :Section label (info)
```text
   509 |             # Get a document with sections
   510 |             result = session.run("""
   511 |                 MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
   512 |                 WITH d, count(s) as section_count, sum(s.token_count) as section_sum
   513 |                 WHERE section_count > 0
```

### tests/test_tokenizer_service.py:15 — Reference to canonical model v3 (info)
```text
    13 | # Set test environment before importing tokenizer service
    14 | os.environ["TOKENIZER_BACKEND"] = "hf"
    15 | os.environ["HF_TOKENIZER_ID"] = "jinaai/jina-embeddings-v3"
    16 | os.environ["HF_CACHE"] = "/tmp/hf-test-cache"
    17 | os.environ["TRANSFORMERS_OFFLINE"] = "true"  # CRITICAL: Prevent network calls
```

### requirements.txt:74 — Use of tiktoken (NON-canonical for Jina v3) (warn)
```text
    72 | 
    73 | # Phase 7C Hotfix: Tokenizer service for accurate token counting
    74 | # CRITICAL: Use jina-embeddings-v3 tokenizer (XLM-RoBERTa), NOT tiktoken (OpenAI)
    75 | transformers>=4.43.0  # HuggingFace tokenizer (primary backend)
    76 | tokenizers>=0.15.0    # Fast Rust backend for transformers
```

### scripts/backfill_document_tokens.py:53 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    51 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
    52 |         WITH d, count(s) as section_count
    53 |         RETURN d.id as doc_id,
    54 |                d.token_count as current_token_count,
    55 |                section_count
```

### scripts/backfill_document_tokens.py:56 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    54 |                d.token_count as current_token_count,
    55 |                section_count
    56 |         ORDER BY doc_id
    57 |         """
    58 | 
```

### scripts/backfill_document_tokens.py:120 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   118 |         logger.info(f"Documents to update: {len(missing_docs)}")
   119 |         for doc in missing_docs[:5]:  # Show first 5
   120 |             logger.info(f"  - {doc['doc_id']}: sections={doc['section_count']}")
   121 |         if len(missing_docs) > 5:
   122 |             logger.info(f"  ... and {len(missing_docs) - 5} more")
```

### scripts/backfill_document_tokens.py:141 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   139 |         WITH d, sum(s.tokens) AS section_tokens
   140 |         SET d.token_count = section_tokens
   141 |         RETURN d.id as doc_id,
   142 |                d.token_count as new_token_count
   143 |         """
```

### scripts/baseline_distribution_analysis.py:51 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    49 |         query = """
    50 |         MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
    51 |         RETURN d.id as doc_id,
    52 |                s.id as section_id,
    53 |                s.title as heading,
```

### scripts/baseline_distribution_analysis.py:52 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    50 |         MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
    51 |         RETURN d.id as doc_id,
    52 |                s.id as section_id,
    53 |                s.title as heading,
    54 |                s.level as level,
```

### scripts/baseline_distribution_analysis.py:131 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   129 | 
   130 |         for section in sections:
   131 |             by_doc[section['doc_id']].append(section)
   132 | 
   133 |         doc_stats = []
```

### scripts/baseline_distribution_analysis.py:134 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   132 | 
   133 |         doc_stats = []
   134 |         for doc_id, doc_sections in by_doc.items():
   135 |             token_counts = [s['token_count'] for s in doc_sections if s['token_count']]
   136 | 
```

### scripts/baseline_distribution_analysis.py:143 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   141 | 
   142 |             doc_stats.append({
   143 |                 'doc_id': doc_id,
   144 |                 'section_count': len(doc_sections),
   145 |                 'total_tokens': sum(token_counts),
```

### scripts/baseline_distribution_analysis.py:143 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   141 | 
   142 |             doc_stats.append({
   143 |                 'doc_id': doc_id,
   144 |                 'section_count': len(doc_sections),
   145 |                 'total_tokens': sum(token_counts),
```

### scripts/baseline_distribution_analysis.py:162 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   160 | 
   161 |         for section in sections:
   162 |             by_doc[section['doc_id']].append(section)
   163 | 
   164 |         h2_groups = []
```

### scripts/baseline_distribution_analysis.py:166 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   164 |         h2_groups = []
   165 | 
   166 |         for doc_id, doc_sections in by_doc.items():
   167 |             # Sort by position
   168 |             sorted_sections = sorted(doc_sections, key=lambda s: s.get('position', 0))
```

### scripts/baseline_distribution_analysis.py:182 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   180 |                         tokens = sum(s['token_count'] for s in current_group if s['token_count'])
   181 |                         h2_groups.append({
   182 |                             'doc_id': doc_id,
   183 |                             'h2_heading': current_h2,
   184 |                             'section_count': len(current_group),
```

### scripts/baseline_distribution_analysis.py:182 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   180 |                         tokens = sum(s['token_count'] for s in current_group if s['token_count'])
   181 |                         h2_groups.append({
   182 |                             'doc_id': doc_id,
   183 |                             'h2_heading': current_h2,
   184 |                             'section_count': len(current_group),
```

### scripts/baseline_distribution_analysis.py:199 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   197 |                 tokens = sum(s['token_count'] for s in current_group if s['token_count'])
   198 |                 h2_groups.append({
   199 |                     'doc_id': doc_id,
   200 |                     'h2_heading': current_h2,
   201 |                     'section_count': len(current_group),
```

### scripts/baseline_distribution_analysis.py:199 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   197 |                 tokens = sum(s['token_count'] for s in current_group if s['token_count'])
   198 |                 h2_groups.append({
   199 |                     'doc_id': doc_id,
   200 |                     'h2_heading': current_h2,
   201 |                     'section_count': len(current_group),
```

### scripts/baseline_distribution_analysis.py:282 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   280 |             for doc in by_doc[:10]:  # Top 10
   281 |                 lines.append(
   282 |                     f"| {doc['doc_id'][:30]} | {doc['section_count']} | "
   283 |                     f"{doc['total_tokens']:,} | {doc['avg']:.0f} | "
   284 |                     f"{doc['p50']:.0f} | {doc['p90']:.0f} | {doc['p95']:.0f} |"
```

### scripts/dev/seed_minimal_graph.py:214 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   212 |             session.run(
   213 |                 """
   214 |                 MATCH (d:Document {id: $doc_id})
   215 |                 MATCH (s:Section {id: $sec_id})
   216 |                 MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
```

### scripts/dev/seed_minimal_graph.py:218 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   216 |                 MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
   217 |             """,
   218 |                 doc_id=section["document_id"],
   219 |                 sec_id=section["id"],
   220 |                 order=section["order"],
```

### scripts/neo4j/create_schema_v2_1.cypher:69 — Neo4j vector dimensions (warn)
```text
    67 | OPTIONS {
    68 |   indexConfig: {
    69 |     `vector.dimensions`: 1024,
    70 |     `vector.similarity_function`: 'cosine'
    71 |   }
```

### scripts/neo4j/create_schema_v2_1.cypher:80 — Neo4j vector dimensions (warn)
```text
    78 | OPTIONS {
    79 |   indexConfig: {
    80 |     `vector.dimensions`: 1024,
    81 |     `vector.similarity_function`: 'cosine'
    82 |   }
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:257 — Neo4j vector dimensions (warn)
```text
   255 | OPTIONS {
   256 |   indexConfig: {
   257 |     `vector.dimensions`: 1024,
   258 |     `vector.similarity_function`: 'cosine'
   259 |   }
```

### scripts/neo4j/create_schema_v2_1_complete.cypher:268 — Neo4j vector dimensions (warn)
```text
   266 | OPTIONS {
   267 |   indexConfig: {
   268 |     `vector.dimensions`: 1024,
   269 |     `vector.similarity_function`: 'cosine'
   270 |   }
```

### scripts/perf/test_traversal_latency.py:43 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    41 |             data = resp.json()
    42 |             evidence = data.get("answer_json", {}).get("evidence", [])
    43 |             section_ids = [ev["section_id"] for ev in evidence if ev.get("section_id")]
    44 |             return section_ids[:count]
    45 |     except Exception as e:
```

### scripts/perf/test_traversal_latency.py:43 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    41 |             data = resp.json()
    42 |             evidence = data.get("answer_json", {}).get("evidence", [])
    43 |             section_ids = [ev["section_id"] for ev in evidence if ev.get("section_id")]
    44 |             return section_ids[:count]
    45 |     except Exception as e:
```

### scripts/run_baseline_queries.py:109 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   107 |                 result_items.append({
   108 |                     'rank': i + 1,
   109 |                     'section_id': hit.get('node_id'),
   110 |                     'score': float(hit.get('score', 0.0)),
   111 |                     'document_id': hit.get('document_id'),
```

### scripts/validate_token_accounting.py:66 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    64 |         MATCH (d:Document)
    65 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
    66 |         WITH d.id AS doc_id,
    67 |              d.token_count AS doc_tokens,
    68 |              sum(s.tokens) AS section_tokens_sum,
```

### scripts/validate_token_accounting.py:71 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    69 |              count(s) AS section_count
    70 |         WHERE doc_tokens IS NOT NULL AND doc_tokens > 0
    71 |         WITH doc_id,
    72 |              doc_tokens,
    73 |              section_tokens_sum,
```

### scripts/validate_token_accounting.py:80 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    78 |                 ELSE abs(1.0 * (doc_tokens - section_tokens_sum) / doc_tokens)
    79 |              END AS error_rate
    80 |         RETURN doc_id,
    81 |                doc_tokens,
    82 |                section_tokens_sum,
```

### scripts/validate_token_accounting.py:152 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   150 |             for i, doc in enumerate(violations[:10], 1):  # Show first 10
   151 |                 logger.warning(
   152 |                     f"  {i}. {doc['doc_id']}: "
   153 |                     f"doc_tokens={doc['doc_tokens']}, "
   154 |                     f"section_sum={doc['section_tokens_sum']}, "
```

### scripts/validate_token_accounting.py:182 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   180 |         WHERE d.token_count IS NULL OR d.token_count = 0
   181 |         OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   182 |         RETURN d.id as doc_id,
   183 |                d.token_count as token_count,
   184 |                count(s) as section_count
```

### scripts/validate_token_accounting.py:185 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   183 |                d.token_count as token_count,
   184 |                count(s) as section_count
   185 |         ORDER BY doc_id
   186 |         """
   187 | 
```

### scripts/validate_token_accounting.py:248 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   246 |             logger.warning("  python scripts/backfill_document_tokens.py --execute")
   247 |             for doc in missing[:5]:
   248 |                 logger.warning(f"  - {doc['doc_id']}")
   249 |             if len(missing) > 5:
   250 |                 logger.warning(f"  ... and {len(missing) - 5} more")
```

### src/ingestion/auto/orchestrator.py:836 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   834 |                 continue
   835 | 
   836 |             # Convert section_id to UUID
   837 |             point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section["id"]))
   838 | 
```

### src/ingestion/auto/orchestrator.py:874 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   872 | 
   873 |                 query = """
   874 |                 MATCH (s:Section {id: $section_id})
   875 |                 SET s.vector_embedding = $embedding,
   876 |                     s.embedding_version = $version
```

### src/ingestion/auto/orchestrator.py:880 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   878 |                 session.run(
   879 |                     query,
   880 |                     section_id=section["id"],
   881 |                     embedding=embedding,
   882 |                     version=self.config.embedding.version,
```

### src/ingestion/auto/orchestrator.py:892 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   890 |             for section in sections:
   891 |                 query = """
   892 |                 MATCH (s:Section {id: $section_id})
   893 |                 SET s.embedding_version = $version
   894 |                 """
```

### src/ingestion/auto/orchestrator.py:897 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   895 |                 session.run(
   896 |                     query,
   897 |                     section_id=section["id"],
   898 |                     version=self.config.embedding.version,
   899 |                 )
```

### src/ingestion/build_graph.py:228 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   226 |              count(DISTINCT q) + count(DISTINCT a) as provenance_count
   227 | 
   228 |         RETURN s.id as section_id,
   229 |                provenance_count,
   230 |                CASE
```

### src/ingestion/build_graph.py:248 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   246 | 
   247 |         # Step 2: Separate orphans by action
   248 |         to_delete = [o["section_id"] for o in orphans if o["action"] == "delete"]
   249 |         to_mark_stale = [
   250 |             o["section_id"] for o in orphans if o["action"] == "mark_stale"
```

### src/ingestion/build_graph.py:250 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   248 |         to_delete = [o["section_id"] for o in orphans if o["action"] == "delete"]
   249 |         to_mark_stale = [
   250 |             o["section_id"] for o in orphans if o["action"] == "mark_stale"
   251 |         ]
   252 | 
```

### src/ingestion/build_graph.py:362 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   360 | 
   361 |         for m in mentions:
   362 |             if "section_id" in m and "entity_id" in m:
   363 |                 # Standard Section→Entity MENTIONS relationship
   364 |                 section_entity_rels.append(m)
```

### src/ingestion/build_graph.py:396 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   394 |             query = """
   395 |             UNWIND $mentions as m
   396 |             MATCH (s:Section {id: m.section_id})
   397 |             MATCH (e {id: m.entity_id})
   398 |             MERGE (s)-[r:MENTIONS {entity_id: m.entity_id}]->(e)
```

### src/ingestion/build_graph.py:721 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   719 |         document_id = document.get("id") or section.get("document_id")
   720 | 
   721 |         # Convert section_id (SHA-256 hex string) to UUID for Qdrant compatibility
   722 |         # Use UUID5 with a namespace to ensure deterministic mapping
   723 |         point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))
```

### src/ingestion/build_graph.py:737 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   735 |             vector=embedding,
   736 |             payload={
   737 |                 "node_id": node_id,  # Original section_id for matching
   738 |                 "node_label": label,
   739 |                 "document_id": document_id,
```

### src/ingestion/extract/__init__.py:33 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    31 | 
    32 |     for section in sections:
    33 |         logger.debug("Extracting entities from section", section_id=section["id"])
    34 | 
    35 |         # Extract commands
```

### src/ingestion/extract/commands.py:28 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    26 | 
    27 |     text = section["text"]
    28 |     section_id = section["id"]
    29 | 
    30 |     # Pattern 1: CLI commands in code blocks
```

### src/ingestion/extract/commands.py:33 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    31 |     for code_block in section.get("code_blocks", []):
    32 |         commands_from_code, mentions_from_code = _extract_from_code_block(
    33 |             code_block, section_id, text
    34 |         )
    35 |         commands.extend(commands_from_code)
```

### src/ingestion/extract/commands.py:39 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    37 | 
    38 |     # Pattern 2: Inline code with command-like patterns
    39 |     inline_commands, inline_mentions = _extract_inline_commands(text, section_id)
    40 |     commands.extend(inline_commands)
    41 |     mentions.extend(inline_mentions)
```

### src/ingestion/extract/commands.py:44 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    42 | 
    43 |     # Pattern 3: Command documentation patterns (e.g., "The `weka` command...")
    44 |     doc_commands, doc_mentions = _extract_documented_commands(text, section_id)
    45 |     commands.extend(doc_commands)
    46 |     mentions.extend(doc_mentions)
```

### src/ingestion/extract/commands.py:54 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    52 |     logger.debug(
    53 |         "Extracted commands",
    54 |         section_id=section_id,
    55 |         commands_count=len(commands),
    56 |         mentions_count=len(mentions),
```

### src/ingestion/extract/commands.py:54 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    52 |     logger.debug(
    53 |         "Extracted commands",
    54 |         section_id=section_id,
    55 |         commands_count=len(commands),
    56 |         mentions_count=len(mentions),
```

### src/ingestion/extract/commands.py:63 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    61 | 
    62 | def _extract_from_code_block(
    63 |     code_block: str, section_id: str, full_text: str
    64 | ) -> Tuple[List[Dict], List[Dict]]:
    65 |     """Extract commands from code blocks."""
```

### src/ingestion/extract/commands.py:84 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    82 |                     cmd["name"],
    83 |                     cmd["full_command"],
    84 |                     section_id,
    85 |                 )
    86 |                 commands.append(command_entity)
```

### src/ingestion/extract/commands.py:92 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    90 |                 if span:
    91 |                     mention = _create_mention(
    92 |                         section_id,
    93 |                         command_entity["id"],
    94 |                         span,
```

### src/ingestion/extract/commands.py:103 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   101 | 
   102 | def _extract_inline_commands(
   103 |     text: str, section_id: str
   104 | ) -> Tuple[List[Dict], List[Dict]]:
   105 |     """Extract commands from inline code (backticks)."""
```

### src/ingestion/extract/commands.py:120 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   118 |                     cmd["name"],
   119 |                     cmd["full_command"],
   120 |                     section_id,
   121 |                 )
   122 |                 commands.append(command_entity)
```

### src/ingestion/extract/commands.py:126 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   124 |                 span = (match.start(), match.end())
   125 |                 mention = _create_mention(
   126 |                     section_id,
   127 |                     command_entity["id"],
   128 |                     span,
```

### src/ingestion/extract/commands.py:137 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   135 | 
   136 | def _extract_documented_commands(
   137 |     text: str, section_id: str
   138 | ) -> Tuple[List[Dict], List[Dict]]:
   139 |     """Extract commands from documentation patterns."""
```

### src/ingestion/extract/commands.py:158 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   156 |                     command_name,
   157 |                     command_name,
   158 |                     section_id,
   159 |                 )
   160 |                 commands.append(command_entity)
```

### src/ingestion/extract/commands.py:164 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   162 |                 span = (match.start(1), match.end(1))
   163 |                 mention = _create_mention(
   164 |                     section_id,
   165 |                     command_entity["id"],
   166 |                     span,
```

### src/ingestion/extract/commands.py:263 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   261 | 
   262 | def _create_mention(
   263 |     section_id: str, entity_id: str, span: Tuple[int, int], confidence: float
   264 | ) -> Dict:
   265 |     """Create a MENTIONS relationship."""
```

### src/ingestion/extract/commands.py:267 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   265 |     """Create a MENTIONS relationship."""
   266 |     return {
   267 |         "section_id": section_id,
   268 |         "entity_id": entity_id,
   269 |         "confidence": confidence,
```

### src/ingestion/extract/commands.py:267 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   265 |     """Create a MENTIONS relationship."""
   266 |     return {
   267 |         "section_id": section_id,
   268 |         "entity_id": entity_id,
   269 |         "confidence": confidence,
```

### src/ingestion/extract/commands.py:272 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   270 |         "start": span[0],
   271 |         "end": span[1],
   272 |         "source_section_id": section_id,
   273 |     }
   274 | 
```

### src/ingestion/extract/configs.py:26 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    24 | 
    25 |     text = section["text"]
    26 |     section_id = section["id"]
    27 | 
    28 |     # Pattern 1: Configuration file patterns
```

### src/ingestion/extract/configs.py:29 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    27 | 
    28 |     # Pattern 1: Configuration file patterns
    29 |     file_configs, file_mentions = _extract_config_files(text, section_id)
    30 |     configurations.extend(file_configs)
    31 |     mentions.extend(file_mentions)
```

### src/ingestion/extract/configs.py:34 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    32 | 
    33 |     # Pattern 2: Configuration parameters (key=value, key: value)
    34 |     param_configs, param_mentions = _extract_config_parameters(text, section_id)
    35 |     configurations.extend(param_configs)
    36 |     mentions.extend(param_mentions)
```

### src/ingestion/extract/configs.py:39 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    37 | 
    38 |     # Pattern 3: Environment variables
    39 |     env_configs, env_mentions = _extract_env_variables(text, section_id)
    40 |     configurations.extend(env_configs)
    41 |     mentions.extend(env_mentions)
```

### src/ingestion/extract/configs.py:46 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    44 |     for code_block in section.get("code_blocks", []):
    45 |         code_configs, code_mentions = _extract_from_config_code(
    46 |             code_block, section_id, text
    47 |         )
    48 |         configurations.extend(code_configs)
```

### src/ingestion/extract/configs.py:57 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    55 |     logger.debug(
    56 |         "Extracted configurations",
    57 |         section_id=section_id,
    58 |         configs_count=len(configurations),
    59 |         mentions_count=len(mentions),
```

### src/ingestion/extract/configs.py:57 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    55 |     logger.debug(
    56 |         "Extracted configurations",
    57 |         section_id=section_id,
    58 |         configs_count=len(configurations),
    59 |         mentions_count=len(mentions),
```

### src/ingestion/extract/configs.py:65 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    63 | 
    64 | 
    65 | def _extract_config_files(text: str, section_id: str) -> Tuple[List[Dict], List[Dict]]:
    66 |     """Extract configuration file references."""
    67 |     configurations = []
```

### src/ingestion/extract/configs.py:95 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    93 | 
    94 |                 span = (match.start(1), match.end(1))
    95 |                 mention = _create_mention(section_id, config_entity["id"], span, 0.85)
    96 |                 mentions.append(mention)
    97 | 
```

### src/ingestion/extract/configs.py:102 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   100 | 
   101 | def _extract_config_parameters(
   102 |     text: str, section_id: str
   103 | ) -> Tuple[List[Dict], List[Dict]]:
   104 |     """Extract configuration parameters."""
```

### src/ingestion/extract/configs.py:128 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   126 | 
   127 |                 span = (match.start(1), match.end(1))
   128 |                 mention = _create_mention(section_id, config_entity["id"], span, 0.75)
   129 |                 mentions.append(mention)
   130 | 
```

### src/ingestion/extract/configs.py:134 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   132 | 
   133 | 
   134 | def _extract_env_variables(text: str, section_id: str) -> Tuple[List[Dict], List[Dict]]:
   135 |     """Extract environment variables."""
   136 |     configurations = []
```

### src/ingestion/extract/configs.py:158 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   156 | 
   157 |                 span = (match.start(1), match.end(1))
   158 |                 mention = _create_mention(section_id, config_entity["id"], span, 0.9)
   159 |                 mentions.append(mention)
   160 | 
```

### src/ingestion/extract/configs.py:165 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   163 | 
   164 | def _extract_from_config_code(
   165 |     code_block: str, section_id: str, full_text: str
   166 | ) -> Tuple[List[Dict], List[Dict]]:
   167 |     """Extract configuration keys from YAML/JSON code blocks."""
```

### src/ingestion/extract/configs.py:189 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   187 |                 if span:
   188 |                     mention = _create_mention(
   189 |                         section_id, config_entity["id"], span, 0.8
   190 |                     )
   191 |                     mentions.append(mention)
```

### src/ingestion/extract/configs.py:208 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   206 |                 if span:
   207 |                     mention = _create_mention(
   208 |                         section_id, config_entity["id"], span, 0.8
   209 |                     )
   210 |                     mentions.append(mention)
```

### src/ingestion/extract/configs.py:258 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   256 | 
   257 | def _create_mention(
   258 |     section_id: str, entity_id: str, span: Tuple[int, int], confidence: float
   259 | ) -> Dict:
   260 |     """Create a MENTIONS relationship."""
```

### src/ingestion/extract/configs.py:262 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   260 |     """Create a MENTIONS relationship."""
   261 |     return {
   262 |         "section_id": section_id,
   263 |         "entity_id": entity_id,
   264 |         "confidence": confidence,
```

### src/ingestion/extract/configs.py:262 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   260 |     """Create a MENTIONS relationship."""
   261 |     return {
   262 |         "section_id": section_id,
   263 |         "entity_id": entity_id,
   264 |         "confidence": confidence,
```

### src/ingestion/extract/configs.py:267 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   265 |         "start": span[0],
   266 |         "end": span[1],
   267 |         "source_section_id": section_id,
   268 |     }
   269 | 
```

### src/ingestion/extract/procedures.py:103 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   101 |         mentions.append(
   102 |             {
   103 |                 "section_id": section["id"],
   104 |                 "entity_id": proc_id,
   105 |                 "entity_label": "Procedure",
```

### src/ingestion/extract/procedures.py:127 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   125 |         mentions.append(
   126 |             {
   127 |                 "section_id": section["id"],
   128 |                 "entity_id": step_info["id"],
   129 |                 "entity_label": "Step",
```

### src/ingestion/extract/procedures.py:155 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   153 |         logger.info(
   154 |             "Extracted procedure from section",
   155 |             section_id=section["id"],
   156 |             procedures_found=len(procedures),
   157 |             steps_count=len(steps),
```

### src/ingestion/parsers/html.py:201 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   199 |     text = "\n\n".join(section_data["content_elements"])
   200 |     normalized_text = _normalize_text(text)
   201 |     section_id = _compute_section_id(
   202 |         source_uri, section_data["anchor"], normalized_text
   203 |     )
```

### src/ingestion/parsers/html.py:208 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   206 | 
   207 |     return {
   208 |         "id": section_id,
   209 |         "document_id": _compute_document_id(source_uri),
   210 |         "level": section_data["level"],
```

### src/ingestion/parsers/markdown.py:234 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   232 | 
   233 |     # Compute deterministic section ID using content-coupled approach
   234 |     section_id = _section_id(source_uri, anchor, checksum)
   235 | 
   236 |     # Count tokens (simple whitespace split for now)
```

### src/ingestion/parsers/markdown.py:240 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   238 | 
   239 |     return {
   240 |         "id": section_id,
   241 |         "document_id": _compute_document_id(source_uri),
   242 |         "level": section_data["level"],
```

### src/ingestion/parsers/notion.py:193 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   191 |     text = "\n\n".join(section_data["content_elements"])
   192 |     normalized_text = _normalize_text(text)
   193 |     section_id = _compute_section_id(
   194 |         source_uri, section_data["anchor"], normalized_text
   195 |     )
```

### src/ingestion/parsers/notion.py:200 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   198 | 
   199 |     return {
   200 |         "id": section_id,
   201 |         "document_id": _compute_document_id(source_uri),
   202 |         "level": section_data["level"],
```

### src/mcp_server/query_service.py:282 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   280 |                 retrieved_sections = [
   281 |                     {
   282 |                         "section_id": result.node_id,
   283 |                         "rank": idx + 1,
   284 |                         "score_vec": getattr(result, "vector_score", 0.0),
```

### src/providers/tokenizer_service.py:5 — Use of tiktoken (NON-canonical for Jina v3) (warn)
```text
     3 | 
     4 | CRITICAL: This module uses the EXACT tokenizer for jina-embeddings-v3 (XLM-RoBERTa family).
     5 | DO NOT use tiktoken or cl100k_base - those are for OpenAI models and will give wrong counts.
     6 | 
     7 | Dual-backend architecture:
```

### src/providers/tokenizer_service.py:383 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   381 |         self,
   382 |         text: str,
   383 |         section_id: Optional[str] = None,
   384 |     ) -> List[Dict[str, Any]]:
   385 |         """
```

### src/providers/tokenizer_service.py:397 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   395 |         Args:
   396 |             text: Text to split (may be large)
   397 |             section_id: Optional parent section identifier
   398 | 
   399 |         Returns:
```

### src/providers/tokenizer_service.py:404 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   402 |                 {
   403 |                     'text': str,              # Chunk text content
   404 |                     'chunk_index': int,       # 0-based chunk index
   405 |                     'total_chunks': int,      # Total chunks created
   406 |                     'token_count': int,       # Exact token count for this chunk
```

### src/providers/tokenizer_service.py:432 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   430 |                 {
   431 |                     "text": text,
   432 |                     "chunk_index": 0,
   433 |                     "total_chunks": 1,
   434 |                     "token_count": token_count,
```

### src/providers/tokenizer_service.py:439 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   437 |                     "overlap_end": False,
   438 |                     "integrity_hash": self.compute_integrity_hash(text),
   439 |                     "parent_section_id": section_id,
   440 |                 }
   441 |             ]
```

### src/providers/tokenizer_service.py:456 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   454 |         chunks = []
   455 |         start_idx = 0
   456 |         chunk_index = 0
   457 | 
   458 |         while start_idx < total_tokens:
```

### src/providers/tokenizer_service.py:471 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   469 |             chunk = {
   470 |                 "text": chunk_text,
   471 |                 "chunk_index": chunk_index,
   472 |                 "total_chunks": 0,  # Will update after loop
   473 |                 "token_count": len(chunk_tokens),
```

### src/providers/tokenizer_service.py:471 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   469 |             chunk = {
   470 |                 "text": chunk_text,
   471 |                 "chunk_index": chunk_index,
   472 |                 "total_chunks": 0,  # Will update after loop
   473 |                 "token_count": len(chunk_tokens),
```

### src/providers/tokenizer_service.py:478 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   476 |                 "overlap_end": end_idx < total_tokens,
   477 |                 "integrity_hash": self.compute_integrity_hash(chunk_text),
   478 |                 "parent_section_id": section_id,
   479 |             }
   480 | 
```

### src/providers/tokenizer_service.py:489 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   487 |                 start_idx = end_idx
   488 | 
   489 |             chunk_index += 1
   490 | 
   491 |         # Update total_chunks in all chunks
```

### src/query/hybrid_search.py:614 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   612 |         # Uses MENTIONS relationship from Section to entities
   613 |         focus_query = """
   614 |         UNWIND $section_ids AS section_id
   615 |         MATCH (s:Section {id: section_id})-[:MENTIONS]->(e)
   616 |         WHERE e.id IN $focused_entity_ids
```

### src/query/hybrid_search.py:615 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   613 |         focus_query = """
   614 |         UNWIND $section_ids AS section_id
   615 |         MATCH (s:Section {id: section_id})-[:MENTIONS]->(e)
   616 |         WHERE e.id IN $focused_entity_ids
   617 |         RETURN s.id AS section_id, count(DISTINCT e) AS focus_hits, collect(DISTINCT e.id) AS matched_entities
```

### src/query/hybrid_search.py:617 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   615 |         MATCH (s:Section {id: section_id})-[:MENTIONS]->(e)
   616 |         WHERE e.id IN $focused_entity_ids
   617 |         RETURN s.id AS section_id, count(DISTINCT e) AS focus_hits, collect(DISTINCT e.id) AS matched_entities
   618 |         """
   619 | 
```

### src/query/hybrid_search.py:632 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   630 | 
   631 |                 for record in result:
   632 |                     focus_counts[record["section_id"]] = record["focus_hits"]
   633 |                     matched_entities_map[record["section_id"]] = record[
   634 |                         "matched_entities"
```

### src/query/hybrid_search.py:633 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   631 |                 for record in result:
   632 |                     focus_counts[record["section_id"]] = record["focus_hits"]
   633 |                     matched_entities_map[record["section_id"]] = record[
   634 |                         "matched_entities"
   635 |                     ]
```

### src/query/response_builder.py:39 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    37 |     """Evidence supporting an answer."""
    38 | 
    39 |     section_id: Optional[str] = None
    40 |     node_id: Optional[str] = None
    41 |     node_label: Optional[str] = None
```

### src/query/response_builder.py:262 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   260 |                 supporting_section_ids = []
   261 |                 for ev in evidence[:5]:
   262 |                     # Prefer section_id, fallback to node_id if it's a Section
   263 |                     if ev.section_id:
   264 |                         supporting_section_ids.append(ev.section_id)
```

### src/query/response_builder.py:263 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   261 |                 for ev in evidence[:5]:
   262 |                     # Prefer section_id, fallback to node_id if it's a Section
   263 |                     if ev.section_id:
   264 |                         supporting_section_ids.append(ev.section_id)
   265 |                     elif ev.node_label == "Section" and ev.node_id:
```

### src/query/response_builder.py:264 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   262 |                     # Prefer section_id, fallback to node_id if it's a Section
   263 |                     if ev.section_id:
   264 |                         supporting_section_ids.append(ev.section_id)
   265 |                     elif ev.node_label == "Section" and ev.node_id:
   266 |                         supporting_section_ids.append(ev.node_id)
```

### src/query/response_builder.py:339 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   337 |             metadata = result.metadata
   338 | 
   339 |             # Determine section_id and node_id
   340 |             section_id = None
   341 |             node_id = result.node_id
```

### src/query/response_builder.py:340 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   338 | 
   339 |             # Determine section_id and node_id
   340 |             section_id = None
   341 |             node_id = result.node_id
   342 | 
```

### src/query/response_builder.py:344 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   342 | 
   343 |             if result.node_label == "Section":
   344 |                 section_id = result.node_id
   345 |             else:
   346 |                 # For non-sections, try to get section from metadata
```

### src/query/response_builder.py:347 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   345 |             else:
   346 |                 # For non-sections, try to get section from metadata
   347 |                 section_id = metadata.get("section_id")
   348 | 
   349 |             # Mode-specific evidence extraction
```

### src/query/response_builder.py:347 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   345 |             else:
   346 |                 # For non-sections, try to get section from metadata
   347 |                 section_id = metadata.get("section_id")
   348 | 
   349 |             # Mode-specific evidence extraction
```

### src/query/response_builder.py:362 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   360 |                 evidence_list.append(
   361 |                     Evidence(
   362 |                         section_id=section_id,
   363 |                         node_id=node_id,
   364 |                         node_label=result.node_label,
```

### src/query/response_builder.py:362 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   360 |                 evidence_list.append(
   361 |                     Evidence(
   362 |                         section_id=section_id,
   363 |                         node_id=node_id,
   364 |                         node_label=result.node_label,
```

### src/query/response_builder.py:402 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   400 |                 evidence_list.append(
   401 |                     Evidence(
   402 |                         section_id=section_id,
   403 |                         node_id=node_id,
   404 |                         node_label=result.node_label,
```

### src/query/response_builder.py:402 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   400 |                 evidence_list.append(
   401 |                     Evidence(
   402 |                         section_id=section_id,
   403 |                         node_id=node_id,
   404 |                         node_label=result.node_label,
```

### src/query/response_builder.py:723 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   721 |                 if ev.snippet:
   722 |                     lines.append(f"> {ev.snippet}\n")
   723 |                 if ev.section_id:
   724 |                     lines.append(f"**Section:** `{ev.section_id}`\n")
   725 |                 if ev.path:
```

### src/query/response_builder.py:724 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   722 |                     lines.append(f"> {ev.snippet}\n")
   723 |                 if ev.section_id:
   724 |                     lines.append(f"**Section:** `{ev.section_id}`\n")
   725 |                 if ev.path:
   726 |                     lines.append(f"**Path:** {' → '.join(ev.path)}\n")
```

### src/query/session_tracker.py:323 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   321 |             retrieved_sections: List of retrieved sections with scores
   322 |                 Format: [{
   323 |                     section_id: str,
   324 |                     rank: int,
   325 |                     score_vec: float,
```

### src/query/session_tracker.py:340 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   338 |         MATCH (q:Query {query_id: $query_id})
   339 |         UNWIND $sections as section
   340 |         MATCH (s:Section {id: section.section_id})
   341 |         MERGE (q)-[r:RETRIEVED]->(s)
   342 |         ON CREATE SET
```

### src/query/session_tracker.py:429 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   427 |                 MATCH (a:Answer {answer_id: $answer_id})
   428 |                 UNWIND $citations as citation
   429 |                 MATCH (s:Section {id: citation.section_id})
   430 |                 CREATE (a)-[r:SUPPORTED_BY {
   431 |                     rank: citation.rank,
```

### src/query/session_tracker.py:438 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   436 | 
   437 |                 citations = [
   438 |                     {"section_id": sec_id, "rank": idx + 1}
   439 |                     for idx, sec_id in enumerate(supporting_section_ids)
   440 |                 ]
```

### src/query/templates/explain.cypher:33 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    31 | OPTIONAL MATCH (concept)-[:RELATED_TO]->(related:Concept)
    32 | RETURN concept,
    33 |        collect(DISTINCT {section_id: sec.id, title: sec.title, document_id: sec.document_id}) AS mentioned_in,
    34 |        collect(DISTINCT ex) AS examples,
    35 |        collect(DISTINCT related) AS related_concepts
```

### tests/fixtures/baseline_query_set.yaml:9 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
     7 | #   - category: Type of query (config, procedure, troubleshooting, complex)
     8 | #   - token_estimate: Approximate token count (for expansion logic)
     9 | #   - judgments: Graded relevance (section_id -> 0/1/2) for nDCG
    10 | #     - 2 = highly relevant (complete answer)
    11 | #     - 1 = partially relevant (related content)
```

### tests/integration/test_phase7c_integration.py:443 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   441 |         retrieved_sections = [
   442 |             {
   443 |                 "section_id": "test-section-1",
   444 |                 "rank": 1,
   445 |                 "score_vec": 0.95,
```

### tests/integration/test_phase7c_integration.py:452 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   450 |             },
   451 |             {
   452 |                 "section_id": "test-section-2",
   453 |                 "rank": 2,
   454 |                 "score_vec": 0.88,
```

### tests/integration/test_phase7c_integration.py:482 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   480 |                         s.embedding_dimensions = 1024
   481 |                     """,
   482 |                     id=sec["section_id"],
   483 |                     text=f"Test content for {sec['section_id']}",
   484 |                     vector=test_vector,
```

### tests/integration/test_phase7c_integration.py:483 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   481 |                     """,
   482 |                     id=sec["section_id"],
   483 |                     text=f"Test content for {sec['section_id']}",
   484 |                     vector=test_vector,
   485 |                 )
```

### tests/integration/test_phase7c_integration.py:498 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   496 |                     RETURN count(r) as count,
   497 |                            collect({
   498 |                                section_id: s.id,
   499 |                                rank: r.rank,
   500 |                                score_combined: r.score_combined
```

### tests/integration/test_phase7c_integration.py:591 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   589 |                            a.tokens_used as tokens,
   590 |                            count(c) as citation_count,
   591 |                            collect({section_id: s.id, rank: c.rank}) as citations
   592 |                     ORDER BY c.rank
   593 |                     """,
```

### tests/integration/test_phase7c_integration.py:608 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   606 |                 assert citations[0]["rank"] == 1
   607 |                 assert citations[1]["rank"] == 2
   608 |                 assert citations[0]["section_id"] == test_section_ids[0]
   609 |                 assert citations[1]["section_id"] == test_section_ids[1]
   610 | 
```

### tests/integration/test_phase7c_integration.py:609 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   607 |                 assert citations[1]["rank"] == 2
   608 |                 assert citations[0]["section_id"] == test_section_ids[0]
   609 |                 assert citations[1]["section_id"] == test_section_ids[1]
   610 | 
   611 |             print("\n✅ Answer created with citations")
```

### tests/integration/test_phase7c_integration.py:715 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   713 |             session.run(
   714 |                 """
   715 |                 MERGE (d:Document {id: $doc_id})
   716 |                 SET d.title = 'Test NFS Document',
   717 |                     d.source_uri = 'test-nfs.md'
```

### tests/integration/test_phase7c_integration.py:719 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   717 |                     d.source_uri = 'test-nfs.md'
   718 |                 """,
   719 |                 doc_id=test_doc_id,
   720 |             )
   721 | 
```

### tests/integration/test_phase7c_integration.py:735 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   733 |             # Create sections mentioning NFS
   734 |             for i in range(3):
   735 |                 section_id = f"test-section-nfs-{i}-{uuid.uuid4()}"
   736 |                 test_section_ids.append(section_id)
   737 | 
```

### tests/integration/test_phase7c_integration.py:736 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   734 |             for i in range(3):
   735 |                 section_id = f"test-section-nfs-{i}-{uuid.uuid4()}"
   736 |                 test_section_ids.append(section_id)
   737 | 
   738 |                 test_vector = [0.1 + i * 0.01] * 1024  # Slightly different vectors
```

### tests/integration/test_phase7c_integration.py:742 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   740 |                 session.run(
   741 |                     """
   742 |                     MERGE (s:Section:Chunk {id: $section_id})
   743 |                     SET s.text = $text,
   744 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:744 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   742 |                     MERGE (s:Section:Chunk {id: $section_id})
   743 |                     SET s.text = $text,
   744 |                         s.document_id = $doc_id,
   745 |                         s.level = 1,
   746 |                         s.title = $title,
```

### tests/integration/test_phase7c_integration.py:756 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   754 |                         s.embedding_dimensions = 1024
   755 |                     WITH s
   756 |                     MATCH (d:Document {id: $doc_id})
   757 |                     MERGE (d)-[:HAS_SECTION]->(s)
   758 |                     WITH s
```

### tests/integration/test_phase7c_integration.py:762 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   760 |                     MERGE (s)-[:MENTIONS {
   761 |                         confidence: 0.9,
   762 |                         source_section_id: $section_id
   763 |                     }]->(e)
   764 |                     """,
```

### tests/integration/test_phase7c_integration.py:765 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   763 |                     }]->(e)
   764 |                     """,
   765 |                     section_id=section_id,
   766 |                     text=f"Section {i} discussing NFS configuration",
   767 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:765 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   763 |                     }]->(e)
   764 |                     """,
   765 |                     section_id=section_id,
   766 |                     text=f"Section {i} discussing NFS configuration",
   767 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:767 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   765 |                     section_id=section_id,
   766 |                     text=f"Section {i} discussing NFS configuration",
   767 |                     doc_id=test_doc_id,
   768 |                     title=f"NFS Section {i}",
   769 |                     order=i,
```

### tests/integration/test_phase7c_integration.py:775 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   773 | 
   774 |         yield {
   775 |             "doc_id": test_doc_id,
   776 |             "entity_id": test_entity_id,
   777 |             "section_ids": test_section_ids,
```

### tests/integration/test_phase7c_integration.py:784 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   782 |             session.run(
   783 |                 """
   784 |                 MATCH (d:Document {id: $doc_id})
   785 |                 OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   786 |                 DETACH DELETE d, s
```

### tests/integration/test_phase7c_integration.py:790 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   788 |                 DETACH DELETE e
   789 |                 """,
   790 |                 doc_id=test_doc_id,
   791 |                 entity_id=test_entity_id,
   792 |             )
```

### tests/integration/test_phase7c_integration.py:910 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   908 |             session.run(
   909 |                 """
   910 |                 MERGE (d:Document {id: $doc_id})
   911 |                 SET d.title = 'Test Orphan Document'
   912 |                 """,
```

### tests/integration/test_phase7c_integration.py:913 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   911 |                 SET d.title = 'Test Orphan Document'
   912 |                 """,
   913 |                 doc_id=test_doc_id,
   914 |             )
   915 | 
```

### tests/integration/test_phase7c_integration.py:917 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   915 | 
   916 |             for i in range(3):
   917 |                 section_id = f"test-orphan-section-{i}-{uuid.uuid4()}"
   918 |                 test_vector = [0.1] * 1024
   919 | 
```

### tests/integration/test_phase7c_integration.py:922 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   920 |                 session.run(
   921 |                     """
   922 |                     MERGE (s:Section:Chunk {id: $section_id})
   923 |                     SET s.text = $text,
   924 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:924 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   922 |                     MERGE (s:Section:Chunk {id: $section_id})
   923 |                     SET s.text = $text,
   924 |                         s.document_id = $doc_id,
   925 |                         s.level = 1,
   926 |                         s.title = $title,
```

### tests/integration/test_phase7c_integration.py:936 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   934 |                         s.embedding_dimensions = 1024
   935 |                     WITH s
   936 |                     MATCH (d:Document {id: $doc_id})
   937 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
   938 |                     """,
```

### tests/integration/test_phase7c_integration.py:939 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   937 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
   938 |                     """,
   939 |                     section_id=section_id,
   940 |                     text=f"Section {i} content",
   941 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:939 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   937 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
   938 |                     """,
   939 |                     section_id=section_id,
   940 |                     text=f"Section {i} content",
   941 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:941 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   939 |                     section_id=section_id,
   940 |                     text=f"Section {i} content",
   941 |                     doc_id=test_doc_id,
   942 |                     title=f"Section {i}",
   943 |                     order=i,
```

### tests/integration/test_phase7c_integration.py:952 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   950 |                 result = session.run(
   951 |                     """
   952 |                     MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   953 |                     RETURN collect(s.id) as section_ids
   954 |                     ORDER BY s.order
```

### tests/integration/test_phase7c_integration.py:956 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   954 |                     ORDER BY s.order
   955 |                     """,
   956 |                     doc_id=test_doc_id,
   957 |                 )
   958 |                 all_section_ids = result.single()["section_ids"]
```

### tests/integration/test_phase7c_integration.py:996 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   994 |                 session.run(
   995 |                     """
   996 |                     MATCH (d:Document {id: $doc_id})
   997 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
   998 |                     DETACH DELETE d, s
```

### tests/integration/test_phase7c_integration.py:1000 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   998 |                     DETACH DELETE d, s
   999 |                     """,
  1000 |                     doc_id=test_doc_id,
  1001 |                 )
  1002 | 
```

### tests/integration/test_phase7c_integration.py:1015 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1013 |             session.run(
  1014 |                 """
  1015 |                 MERGE (d:Document {id: $doc_id})
  1016 |                 SET d.title = 'Test Stale Document'
  1017 |                 """,
```

### tests/integration/test_phase7c_integration.py:1018 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1016 |                 SET d.title = 'Test Stale Document'
  1017 |                 """,
  1018 |                 doc_id=test_doc_id,
  1019 |             )
  1020 | 
```

### tests/integration/test_phase7c_integration.py:1022 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1020 | 
  1021 |             for i in range(2):
  1022 |                 section_id = f"test-stale-section-{i}-{uuid.uuid4()}"
  1023 |                 section_ids.append(section_id)
  1024 |                 test_vector = [0.1] * 1024
```

### tests/integration/test_phase7c_integration.py:1023 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1021 |             for i in range(2):
  1022 |                 section_id = f"test-stale-section-{i}-{uuid.uuid4()}"
  1023 |                 section_ids.append(section_id)
  1024 |                 test_vector = [0.1] * 1024
  1025 | 
```

### tests/integration/test_phase7c_integration.py:1028 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1026 |                 session.run(
  1027 |                     """
  1028 |                     MERGE (s:Section:Chunk {id: $section_id})
  1029 |                     SET s.text = $text,
  1030 |                         s.document_id = $doc_id,
```

### tests/integration/test_phase7c_integration.py:1030 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1028 |                     MERGE (s:Section:Chunk {id: $section_id})
  1029 |                     SET s.text = $text,
  1030 |                         s.document_id = $doc_id,
  1031 |                         s.level = 1,
  1032 |                         s.title = $title,
```

### tests/integration/test_phase7c_integration.py:1042 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1040 |                         s.embedding_dimensions = 1024
  1041 |                     WITH s
  1042 |                     MATCH (d:Document {id: $doc_id})
  1043 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
  1044 |                     """,
```

### tests/integration/test_phase7c_integration.py:1045 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1043 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
  1044 |                     """,
  1045 |                     section_id=section_id,
  1046 |                     text=f"Section {i} content",
  1047 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:1045 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1043 |                     MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
  1044 |                     """,
  1045 |                     section_id=section_id,
  1046 |                     text=f"Section {i} content",
  1047 |                     doc_id=test_doc_id,
```

### tests/integration/test_phase7c_integration.py:1047 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1045 |                     section_id=section_id,
  1046 |                     text=f"Section {i} content",
  1047 |                     doc_id=test_doc_id,
  1048 |                     title=f"Section {i}",
  1049 |                     order=i,
```

### tests/integration/test_phase7c_integration.py:1072 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1070 |                     })
  1071 |                     WITH q
  1072 |                     MATCH (sec:Section {id: $section_id})
  1073 |                     CREATE (q)-[:RETRIEVED {
  1074 |                         rank: 1,
```

### tests/integration/test_phase7c_integration.py:1080 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
  1078 |                     """,
  1079 |                     session_id=test_session_id,
  1080 |                     section_id=section_ids[1],  # Section 1 has provenance
  1081 |                 )
  1082 | 
```

### tests/integration/test_phase7c_integration.py:1129 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1127 |                 session.run(
  1128 |                     """
  1129 |                     MATCH (d:Document {id: $doc_id})
  1130 |                     OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
  1131 |                     DETACH DELETE d, s
```

### tests/integration/test_phase7c_integration.py:1136 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
  1134 |                     DETACH DELETE sess, q
  1135 |                     """,
  1136 |                     doc_id=test_doc_id,
  1137 |                     session_id=test_session_id,
  1138 |                 )
```

### tests/integration/test_session_tracking.py:230 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   228 |         retrieved_sections = [
   229 |             {
   230 |                 "section_id": "sec-nfs-config-1",
   231 |                 "rank": 1,
   232 |                 "score_vec": 0.95,
```

### tests/integration/test_session_tracking.py:239 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   237 |             },
   238 |             {
   239 |                 "section_id": "sec-nfs-config-2",
   240 |                 "rank": 2,
   241 |                 "score_vec": 0.87,
```

### tests/integration/test_session_tracking.py:535 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   533 |             [
   534 |                 {
   535 |                     "section_id": "sec-1",
   536 |                     "rank": 1,
   537 |                     "score_combined": 0.95,
```

### tests/p1_t3_test.py:126 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   124 |     with neo4j_driver.session() as session:
   125 |         # Create a test document
   126 |         doc_id = hashlib.sha256(b"test_doc_1").hexdigest()
   127 |         result = session.run(
   128 |             """
```

### tests/p1_t3_test.py:136 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   134 |             RETURN d
   135 |             """,
   136 |             id=doc_id,
   137 |             uri="test://document/1",
   138 |             title="Test Document",
```

### tests/p1_t3_test.py:144 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   142 | 
   143 |         # Cleanup
   144 |         session.run("MATCH (d:Document {id: $id}) DELETE d", id=doc_id)
   145 | 
   146 | 
```

### tests/p1_t3_test.py:153 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   151 |     with neo4j_driver.session() as session:
   152 |         # Create a test section
   153 |         section_id = hashlib.sha256(b"test_section_1").hexdigest()
   154 |         result = session.run(
   155 |             """
```

### tests/p1_t3_test.py:157 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   155 |             """
   156 |             MERGE (s:Section {id: $id})
   157 |             SET s.document_id = $doc_id,
   158 |                 s.level = 1,
   159 |                 s.title = $title,
```

### tests/p1_t3_test.py:163 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   161 |             RETURN s
   162 |             """,
   163 |             id=section_id,
   164 |             doc_id="test_doc",
   165 |             title="Test Section",
```

### tests/p1_t3_test.py:164 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   162 |             """,
   163 |             id=section_id,
   164 |             doc_id="test_doc",
   165 |             title="Test Section",
   166 |             text="This is a test section.",
```

### tests/p1_t3_test.py:172 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   170 | 
   171 |         # Cleanup
   172 |         session.run("MATCH (s:Section {id: $id}) DELETE s", id=section_id)
```

### tests/p2_t3_test.py:336 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   334 |         """Test search with filters."""
   335 |         # This assumes the seeded data has document_id
   336 |         doc_id = search_engine.vector_store.search(
   337 |             search_engine.embedder.encode("test").tolist(), k=1
   338 |         )[0].get("document_id")
```

### tests/p2_t3_test.py:340 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   338 |         )[0].get("document_id")
   339 | 
   340 |         if doc_id:
   341 |             results = search_engine.search(
   342 |                 "configuration", k=5, filters={"document_id": doc_id}
```

### tests/p2_t3_test.py:342 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   340 |         if doc_id:
   341 |             results = search_engine.search(
   342 |                 "configuration", k=5, filters={"document_id": doc_id}
   343 |             )
   344 | 
```

### tests/p2_t3_test.py:347 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   345 |             # All results should be from the same document
   346 |             for r in results.results:
   347 |                 assert r.metadata.get("document_id") == doc_id or doc_id in str(
   348 |                     r.metadata
   349 |                 )
```

### tests/p2_t3_test.py:347 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   345 |             # All results should be from the same document
   346 |             for r in results.results:
   347 |                 assert r.metadata.get("document_id") == doc_id or doc_id in str(
   348 |                     r.metadata
   349 |                 )
```

### tests/p2_t4_test.py:87 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    85 | 
    86 |     def test_evidence_has_section_id(self, sample_ranked_results):
    87 |         """Test that section evidence includes section_id."""
    88 |         builder = ResponseBuilder()
    89 |         evidence = builder._extract_evidence(sample_ranked_results[:1])
```

### tests/p2_t4_test.py:92 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    90 | 
    91 |         # First result is a Section
    92 |         assert evidence[0].section_id == "section-1"
    93 |         assert evidence[0].node_label == "Section"
    94 | 
```

### tests/p3_t2_test.py:54 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
    52 |         # Verify mention structure
    53 |         for mention in command_mentions:
    54 |             assert "section_id" in mention
    55 |             assert "entity_id" in mention
    56 |             assert "confidence" in mention
```

### tests/p3_t2_test.py:263 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   261 |         for mention in mentions:
   262 |             assert mention["source_section_id"]
   263 |             assert mention["source_section_id"] == mention["section_id"]
   264 | 
   265 |         # All mentions must have valid spans
```

### tests/p3_t3_test.py:101 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
    99 |             result = session.run(
   100 |                 """
   101 |                 MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   102 |                 RETURN count(s) as section_count
   103 |                 """,
```

### tests/p3_t3_test.py:104 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   102 |                 RETURN count(s) as section_count
   103 |                 """,
   104 |                 doc_id=sample_document["document"]["id"],
   105 |             )
   106 |             first_section_count = result.single()["section_count"]
```

### tests/p3_t3_test.py:130 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   128 |             result = session.run(
   129 |                 """
   130 |                 MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
   131 |                 RETURN count(s) as section_count
   132 |                 """,
```

### tests/p3_t3_test.py:133 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   131 |                 RETURN count(s) as section_count
   132 |                 """,
   133 |                 doc_id=sample_document["document"]["id"],
   134 |             )
   135 |             second_section_count = result.single()["section_count"]
```

### tests/p3_t3_test.py:167 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   165 |                 """
   166 |                 MATCH (s:Section)-[m:MENTIONS]->(e)
   167 |                 WHERE s.document_id = $doc_id
   168 |                 RETURN m.confidence as confidence,
   169 |                        m.start as start,
```

### tests/p3_t3_test.py:174 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   172 |                 LIMIT 10
   173 |                 """,
   174 |                 doc_id=sample_document["document"]["id"],
   175 |             )
   176 | 
```

### tests/p3_t3_test.py:217 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   215 |                 """
   216 |                 MATCH (s:Section)
   217 |                 WHERE s.document_id = $doc_id
   218 |                 RETURN count(s) as count
   219 |                 """,
```

### tests/p3_t3_test.py:220 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   218 |                 RETURN count(s) as count
   219 |                 """,
   220 |                 doc_id=sample_document["document"]["id"],
   221 |             )
   222 |             graph_count = result.single()["count"]
```

### tests/p3_t3_test.py:249 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   247 |                     """
   248 |                     MATCH (s:Section)
   249 |                     WHERE s.document_id = $doc_id
   250 |                       AND s.vector_embedding IS NOT NULL
   251 |                     RETURN count(s) as count
```

### tests/p3_t3_test.py:253 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   251 |                     RETURN count(s) as count
   252 |                     """,
   253 |                     doc_id=sample_document["document"]["id"],
   254 |                 )
   255 |                 vector_count = result.single()["count"]
```

### tests/p3_t3_test.py:279 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   277 |                     """
   278 |                     MATCH (s:Section)
   279 |                     WHERE s.document_id = $doc_id
   280 |                     RETURN s.embedding_version as version
   281 |                     LIMIT 1
```

### tests/p3_t3_test.py:283 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   281 |                     LIMIT 1
   282 |                     """,
   283 |                     doc_id=sample_document["document"]["id"],
   284 |                 )
   285 |                 record = result.single()
```

### tests/p3_t4_integration_test.py:118 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   116 |                 """
   117 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   118 |                 RETURN s.id as section_id, s.checksum as checksum, s.title as title
   119 |                 ORDER BY s.order
   120 |                 """,
```

### tests/p3_t4_integration_test.py:152 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   150 |                 """
   151 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   152 |                 RETURN s.id as section_id, s.checksum as checksum, s.title as title
   153 |                 ORDER BY s.order
   154 |                 """,
```

### tests/p3_t4_integration_test.py:308 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   306 |                 MATCH (doc:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
   307 |                 WHERE s.embedding_version = $emb_version
   308 |                 RETURN s.id as section_id
   309 |                 """,
   310 |                 uri=source_uri,
```

### tests/p3_t4_integration_test.py:313 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   311 |                 emb_version=config.embedding.version,
   312 |             )
   313 |             section_ids = [r["section_id"] for r in result]
   314 | 
   315 |         total_sections = len(section_ids)
```

### tests/p3_t4_test.py:141 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   139 |                 """
   140 |                 MATCH (s:Section)
   141 |                 WHERE s.document_id = $doc_id
   142 |                 RETURN s.id as id, s.updated_at as updated_at
   143 |                 """,
```

### tests/p3_t4_test.py:144 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   142 |                 RETURN s.id as id, s.updated_at as updated_at
   143 |                 """,
   144 |                 doc_id=document["id"],
   145 |             )
   146 |             # Store timestamps to verify unchanged sections keep their timestamps
```

### tests/p4_t2_perf_test.py:289 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   287 |         {
   288 |             "name": "section_lookup",
   289 |             "query": "MATCH (s:Section) WHERE s.document_id = $doc_id RETURN s",
   290 |             "params": {"doc_id": "perf-doc-1"},
   291 |             "optimize": {"label": "Section", "property": "document_id"},
```

### tests/p4_t2_perf_test.py:290 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   288 |             "name": "section_lookup",
   289 |             "query": "MATCH (s:Section) WHERE s.document_id = $doc_id RETURN s",
   290 |             "params": {"doc_id": "perf-doc-1"},
   291 |             "optimize": {"label": "Section", "property": "document_id"},
   292 |         },
```

### tests/p4_t4_test.py:261 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   259 | 
   260 |         # Create similar queries (same pattern)
   261 |         base_query = "MATCH (s:Section {document_id: $doc_id}) RETURN s LIMIT 10"
   262 | 
   263 |         for i in range(8):
```

### tests/test_phase7c_dual_write.py:195 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   193 |         import uuid
   194 | 
   195 |         section_id = "test-dual-write-section-1"
   196 |         point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section_id))
   197 | 
```

### tests/test_phase7c_dual_write.py:196 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   194 | 
   195 |         section_id = "test-dual-write-section-1"
   196 |         point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section_id))
   197 | 
   198 |         legacy_point = qdrant_client.retrieve(
```

### tests/test_phase7c_dual_write.py:207 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   205 |             len(legacy_point[0].vector) == 384
   206 |         ), "Legacy collection has wrong dimensions"
   207 |         assert legacy_point[0].payload["node_id"] == section_id
   208 |         assert legacy_point[0].payload["embedding_dimensions"] == 384
   209 |         assert legacy_point[0].payload["embedding_provider"] == "sentence-transformers"
```

### tests/test_phase7c_dual_write.py:219 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   217 |         assert len(new_point) == 1, "Section not found in new collection"
   218 |         assert len(new_point[0].vector) == 1024, "New collection has wrong dimensions"
   219 |         assert new_point[0].payload["node_id"] == section_id
   220 |         assert new_point[0].payload["embedding_dimensions"] == 1024
   221 |         # Provider may be jina-ai or ollama depending on ENV
```

### tests/test_phase7c_dual_write.py:267 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   265 |             result = session.run(
   266 |                 """
   267 |                 MATCH (s:Section {id: $section_id})
   268 |                 RETURN s.embedding_version as version,
   269 |                        s.embedding_provider as provider,
```

### tests/test_phase7c_dual_write.py:274 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   272 |                        s.embedding_task as task
   273 |                 """,
   274 |                 section_id="test-metadata-section",
   275 |             )
   276 | 
```

### tests/test_phase7c_ingestion.py:106 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   104 |                 result = session.run(
   105 |                     """
   106 |                     MATCH (s {id: $section_id})
   107 |                     RETURN labels(s) as labels
   108 |                     """,
```

### tests/test_phase7c_ingestion.py:109 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   107 |                     RETURN labels(s) as labels
   108 |                     """,
   109 |                     section_id=section["id"],
   110 |                 )
   111 |                 record = result.single()
```

### tests/test_phase7c_ingestion.py:136 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   134 |                 result = session.run(
   135 |                     """
   136 |                     MATCH (s:Section {id: $section_id})
   137 |                     RETURN s.vector_embedding as embedding,
   138 |                            s.embedding_version as version,
```

### tests/test_phase7c_ingestion.py:144 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   142 |                            s.embedding_task as task
   143 |                     """,
   144 |                     section_id=section["id"],
   145 |                 )
   146 |                 record = result.single()
```

### tests/test_phase7c_schema_v2_1.py:234 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   232 | 
   233 |             # Check for chunk_embeddings_v2 (Chunk label, 1024-D)
   234 |             chunk_index = next(
   235 |                 (
   236 |                     idx
```

### tests/test_phase7c_schema_v2_1.py:242 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   240 |                 None,
   241 |             )
   242 |             assert chunk_index is not None, "chunk_embeddings_v2 index not found"
   243 |             assert "Chunk" in chunk_index["labels"]
   244 | 
```

### tests/test_phase7c_schema_v2_1.py:243 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   241 |             )
   242 |             assert chunk_index is not None, "chunk_embeddings_v2 index not found"
   243 |             assert "Chunk" in chunk_index["labels"]
   244 | 
   245 |             # Note: Actual dimension validation requires inspecting index config
```

### tests/test_phase7e_phase0.py:514 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   512 |                 WITH d, count(s) as section_count, sum(s.token_count) as section_sum
   513 |                 WHERE section_count > 0
   514 |                 RETURN d.doc_id as doc_id,
   515 |                        d.token_count as doc_tokens,
   516 |                        section_sum,
```

### tests/test_phase7e_phase0.py:514 — Use of non-canonical 'doc_id' (should be 'document_id') (warn)
```text
   512 |                 WITH d, count(s) as section_count, sum(s.token_count) as section_sum
   513 |                 WHERE section_count > 0
   514 |                 RETURN d.doc_id as doc_id,
   515 |                        d.token_count as doc_tokens,
   516 |                        section_sum,
```

### tests/test_tokenizer_service.py:274 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   272 |         """Test splitting when text is under limit."""
   273 |         text = "Short text that doesn't need splitting."
   274 |         chunks = service.split_to_chunks(text, section_id="test_section")
   275 | 
   276 |         assert len(chunks) == 1
```

### tests/test_tokenizer_service.py:278 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   276 |         assert len(chunks) == 1
   277 |         assert chunks[0]["text"] == text
   278 |         assert chunks[0]["chunk_index"] == 0
   279 |         assert chunks[0]["total_chunks"] == 1
   280 |         assert chunks[0]["overlap_start"] is False
```

### tests/test_tokenizer_service.py:293 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   291 |         assert token_count > service.target_tokens, f"Text has {token_count} tokens, need >{service.target_tokens} for splitting"
   292 | 
   293 |         chunks = service.split_to_chunks(text, section_id="large_section")
   294 | 
   295 |         # Should create multiple chunks
```

### tests/test_tokenizer_service.py:300 — Use of non-canonical 'chunk_index' (should be 'order') (warn)
```text
   298 |         # Verify chunk metadata
   299 |         for i, chunk in enumerate(chunks):
   300 |             assert chunk["chunk_index"] == i
   301 |             assert chunk["total_chunks"] == len(chunks)
   302 |             assert chunk["token_count"] <= 100  # Should not exceed target
```

### tests/test_tokenizer_service.py:494 — Use of non-canonical 'section_id' (should be 'id') (warn)
```text
   492 | 
   493 |         # Split it
   494 |         chunks = service.split_to_chunks(large_text, section_id="cli_ref")
   495 | 
   496 |         # Should create multiple chunks
```
