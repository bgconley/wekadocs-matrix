# Hybrid RAG v2.2 – End‑to‑End Production Validation (Customized Spec)

This document adapts the generic end‑to‑end test plan to the WekaDocs Matrix v2.2 architecture. It is tailored to our concrete components (StructuredChunker, Neo4j v2.2 schema, Qdrant `chunks_multi` with named vectors, HybridRetriever + QdrantWeightedSearcher, Jina‑v3 embeddings/tokenizer, and our observability model). It defines what to validate, how to drive the pipeline, acceptance thresholds, artifacts, and the exact test modules we will author (pending approval).

Important: This spec aligns with our canonical docs:
- `docs/hybrid-rag-v2_2-architecture.md` (system flow)
- `docs/hybrid-rag-v2_2-spec.md` (single source of truth)
- `docs/hybrid-rag-v2_2-testing.md` (mechanics + commands)

We explicitly exclude behaviors we have not implemented (e.g., real semantic enricher, reranker), while leaving hooks and opt‑in tests ready to activate later.

---

## 0) Scope & Assumptions (WekaDocs v2.2)

Goal. Validate, on a production document slice, that our v2.2 hybrid RAG stack:
- Ingests Markdown reliably; preserves headings, tables, code fences.
- Assembles chunks deterministically using `StructuredChunker` (microdoc flags, stubs, sliding splits).
- Writes Neo4j nodes/relationships following the v2.2 schema (no APOC; typed relationships guarded by `IS NOT NULL`).
- Embeds with Jina v3 (1024‑D) and upserts named vectors to Qdrant `chunks_multi` (`content`, `title`, reserved `entity`).
- Executes BM25 + multi‑vector ANN + adjacency expansion; enforces budget; reports metrics.
- (Optional) Generates answers from contexts; validates citation fidelity.

Assumptions & constraints.
- Env uses containerized services; local access is `bolt://localhost:7687` (Neo4j), `http://localhost:6333` (Qdrant), `localhost:6379` (Redis) in test runs.
- Embeddings: Jina v3 API or our deterministic test embedder (for offline runs). Tokenization: HuggingFace Jina v3 tokenizer.
- Context budget: 4500 tokens.
- Reranker and semantic enrichment are present as hooks but disabled by default.

Data policy.
- Use a read‑only snapshot under a distinct `doc_tag` (e.g., `prod-e2e-<date>`). Preserve snapshot manifest (commit hash, doc_ids).

---

## 1) Production Data Sampling (Weka‑specific)

We will ingest a representative sample labeled with a unique `doc_tag` to isolate test traffic:
- Stratify by doc type (guides, API refs, cookbooks, release notes, FAQs, tables, code‑heavy pages).
- Size: 300–1,000 docs (≥50k tokens) for stable metrics.
- Diversity targets (aspirational; measured via parsed front‑matter/structure):
  - ≥20% with YAML front‑matter, ≥20% tables, ≥20% code blocks, ≥20% deep headings (H3+), ≥10% cross‑doc links.

Artifact: `reports/e2e-prod/snapshots/<snapshot_id>.json` with `{snapshot_id, commit, timestamp, doc_tag, doc_ids[]}`.

Acceptance: snapshot recorded; sample mix computed; all docs carry the test `doc_tag`.

---

## 2) Ingestion (Markdown → Sections)

Driver: our ingestion harness uses `GraphBuilder` (`src/ingestion/build_graph.py`) with `StructuredChunker` configured via `IngestionConfig.chunk_assembly`.

Validate parsing of:
- YAML front‑matter (`title`, `tags`, `product`, `version`, `updated_at`, `owners`) when present.
- Headings (H1–H6), paragraphs/lists, tables, code fences (language), callouts; links (anchors and cross‑doc); images (alt text only).

Tests (acceptance):
- Parsed document coverage: ≥ 99.5% success.
- Character coverage: parsed_char_sum / raw_char_sum ≥ 0.98.
- Structural parity: heading count diff per doc ≤ 2%.
- Front‑matter extraction: required keys present when available (≥ 99%).

Artifacts: persisted via builder staging logs and per‑doc section arrays.

Notes: Markdown parser implementation is part of our internal ingestion; we only assert outcomes observable in sections and metadata produced by `build_graph`.

---

## 3) Chunking (StructuredChunker)

Strategy (implemented):
- Block‑anchored, heading‑aware grouping (H1/H2 anchors + sibling/descendent within block).
- Token budget/overlap: deterministic `split_to_chunks` when needed; microdoc annotations when doc total tokens below threshold; emission of microdoc stub to guarantee adjacency.
- Deterministic IDs via `generate_chunk_id(document_id, original_section_ids)`.

Tests (acceptance):
- No code fence splitting (best‑effort check by scanning chunk text for unbalanced fences across boundaries; warn only).
- Microdoc flags correct: if doc under threshold – all chunks `doc_is_microdoc=True` and stub emitted.
- Split metadata: presence of `chunk_index`, `total_chunks`, overlap fields; sliding order monotonic.
- Chunk size distribution is healthy for our defaults: 5th–95th percentile within [300, 7900] tokens (wide—matches our provider max), median in expected band for the corpus.

Artifacts: payload fields (`is_combined`, `is_split`, `boundaries_json`, `doc_is_microdoc`, `is_microdoc_stub`, `token_count`, `_citation_units`).

---

## 4) Embedding & Upsert (Qdrant)

Vectors produced (named vectors):
- `content`: v_text(chunk.text) using Jina v3 (1024‑D)
- `title`: v_title (heading/title‑focused embedding) using same provider
- `entity`: reserved; may mirror `content` until semantics are enabled

Quality tests:
- Determinism: with deterministic embedder, identical inputs → identical bytes hash.
- Sanity: 0 NaN vectors; all dims 1024; named vector dict present in `PointStruct.vector`.
- Versioning: payload includes provider/model identifiers and tokenizer version when available.

Acceptance: 100% chunks embedded; 0 NaN; Qdrant collection `chunks_multi` present and vectors conform to named‑vector schema.

---

## 5) Metadata Extraction & Indexing

From front‑matter/structure, store:
- `document_id`, `doc_id` (both), `doc_tag`, `version`, `lang`, `tenant`, `heading`, hashes (`text_hash`, `shingle_hash`), `token_count`.
- Optional semantic placeholders: `semantic_metadata` (entities/topics summary when enrichment enabled – disabled by default).

Tests:
- Types correct (dates parsed; numeric fields numeric).
- Coverage: required keys present per our schema; `(doc_id, id)` unique across chunks.

---

## 6) Graph & Vector Alignment (Neo4j v2.2)

Graph model (implemented subset):
- Nodes: `Document`, `Section`, `Chunk` (with `document_id` and `doc_id`), plus metadata nodes (SchemaVersion, RelationshipTypesMarker).
- Edges: `NEXT_CHUNK` (document order). Optional typed relationships (SAME_HEADING, CHILD_OF, PREV/NEXT sibling, etc.) built post‑ingest (no APOC; `IS NOT NULL` guards).

Tests:
- Orphans: `%chunks_without_graph_node = 0` (every Qdrant point should correlate to a `Chunk` node via node_id/document_id).
- Multiplicity: 1:1 mapping between stored chunk and graph chunk node.
- Adjacency: `NEXT_CHUNK` chain complete for each document; microdoc documents still have at least one adjacency by stub.

---

## 7) Query Processing (Filters)

We leverage `doc_tag` and `version` filters inside retrieval to scope the test slice. No special pre‑processing (spelling/acronyms) is asserted at this time.

---

## 8) Hybrid Retrieval (Implemented Path)

Candidate generation (as implemented):
1. BM25 via Neo4j Full‑Text Index (chunk text/title)
2. Multi‑vector ANN via `QdrantWeightedSearcher` on `content` and `title` (and `entity` as a placeholder)
3. Adjacency expansion (NEXT_CHUNK) – optional typed expansion is gated by config

Fusion/metrics:
- RRF or weighted fusion depending on config; capture metrics: `seed_count`, `seed_gated`, `fusion_time_ms`, vector fields used, expansion stats, microdoc stats.

Tests:
- Hybrid vs branches: hybrid Recall@K ≥ max(BM25, ANN) on a small labeled sample (if gold spans available); otherwise, structural sanity checks: non‑zero fused scores, vector score preservation, filters applied.
- Expansion sanity: `expansion_count > 0` when adjacency exists and config enabled; `expansion_cap_hit` behavior.
- Budgeting: `total_tokens ≤ 4500`, microdoc extras not outranking primaries.

Note: reranker is a hook and remains disabled—tests assert `metrics["reranker_applied"] == False`.

---

## 9) Stitching & Context Assembly

Policy (implemented):
- Keep chunks in fused order; apply doc continuity boost; append microdoc extras at the end; enforce token budget via tokenizer.

Tests:
- Offsets/IDs preserved (chunk ids and headings present).
- No stubs in final output; microdoc counts reported in metrics.

---

## 10) (Optional) Generation & Fidelity

Optional for now. If executed:
- Generate answers strictly from assembled context; ensure per‑sentence citations exist and match cited spans.
- Fidelity checker (planned harness): sentence‑to‑span alignment via embeddings + LCS; numeric/entity consistency; coverage and unsupported rates.

---

## 11) Metrics & Acceptance (Per Stage)

- Ingestion/Chunking: coverage ≥ 99.5%; char coverage ≥ 98%; microdoc/stub invariants.
- Embedding/Qdrant: 0 NaNs; named vectors dims match; payload conforms; determinism (when using deterministic embedder).
- Retrieval: metrics present (`seed_gated`, `microdoc_*`, `expansion_*`, `fusion_method`, `vector_fields`); budgets enforced; non‑zero fused.
- (Optional) Recall/QoR: if gold available, Hybrid uplift ≥ +6 pts Recall@20 over best single branch for navigational/factoid; graph uplift for multi‑hop ≥ +2 pts.
- Latency: keep under dev‑mode guardrails (p50 pre‑gen ≤ 500 ms suggested; not enforced in test suite).

---

## 12) Execution Plan (Weka Harness)

1. Snapshot: copy production docs, assign unique `doc_tag`, record manifest (`reports/e2e-prod/snapshots/…`).
2. Ingest: use `GraphBuilder` entrypoint to parse, assemble (StructuredChunker), write Neo4j, compute embeddings, upsert Qdrant; typed relationship builders run after upsert.
3. Retrieval: run HybridRetriever with filters `{doc_tag: <snapshot_tag>}`; capture metrics + ranked chunks; (optional) LLM generation.
4. Evidence: archive test logs, snapshot manifests, and per‑test JSON traces (planned) under `reports/e2e-prod/runs/<date>/`.

---

## 13) Proposed Test Suite (Files & Logic) – Pending Approval

We will create a new suite under `tests/e2e_v22_prod/` (names shown below; we will not create files until approved):

- `conftest.py`
  - Provides fixtures:
    - `prod_snapshot_manifest` (load snapshot doc list and expected tags)
    - `graph_builder` (instantiates builder with StructuredChunker + config)
    - `retriever` (HybridRetriever with ANN weights, ef, top‑k; filters seeded with `doc_tag`)
    - `qdrant_client`, `neo4j_driver`, `tokenizer`

- `test_prod_ingestion_markdown.py`
  - Validates parse coverage, char coverage, structural parity, front‑matter extraction tallies computed over the ingested set. Aggregates per‑doc metrics.

- `test_prod_chunking_invariants.py`
  - Asserts microdoc flags and stub emission; split metadata presence/order; distribution of chunk sizes; best‑effort code‑fence boundary checks.

- `test_prod_vectors_qdrant.py`
  - Scans `chunks_multi` for points with `doc_tag`: validates named vectors present, correct dims (1024); payload keys (doc_id/document_id/heading/token_count/text_hash) and absence of NaNs.

- `test_prod_graph_alignment.py`
  - For a random sample of chunk ids: asserts presence of corresponding `Chunk` nodes; validates `NEXT_CHUNK` chain continuity per document; zero orphans.

- `test_prod_retrieval_hybrid.py`
  - Issues a small set of curated queries for the snapshot; validates non‑empty fused results; metrics present (`seed_gated`, `expansion_count`, `microdoc_present`/`used`, `fusion_method`, `vector_fields`), budget enforcement, and no stubs in final results.
  - If a miniature gold subset is provided, compute Recall@k uplift (optional assertions gated by availability).

- (Optional) `test_prod_generation_fidelity.py`
  - If LLM generation is enabled, runs fidelity checker heuristics (coverage, unsupported rate) on a tiny labeled set.

Artifacts: we will write logs to `reports/e2e-prod/runs/<date>/` and (optionally) per‑request JSON traces enumerating candidates, fusion, expansion, and final context.

---

## 14) Configuration for Test Runs

Recommended overrides for local container access:

```bash
export NEO4J_URI=bolt://localhost:7687
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export REDIS_HOST=localhost
export REDIS_PASSWORD=…
```

Retriever defaults (example):

```yaml
retrieval:
  ann:
    weights: { content: 1.0, title: 0.35 }
    top_k_per_field: 120
    ef: 256
  graph:
    enabled: true
    same_doc_only: true
    undirected: true
    max_hops: 1  # adjacency only by default
    limit_per_seed: 200
```

We will pass a unique `doc_tag` to isolate snapshot data and to scope retrieval.

---

## 15) Acceptance / Go–No–Go (v2.2)

- Ingestion/Chunking tests pass with coverage & parity thresholds.
- Qdrant named‑vector conformance (content/title[/entity] @ 1024‑D), payload sanity (required fields), and zero NaNs.
- Graph alignment shows zero orphans; NEXT_CHUNK chains complete; typed builders execute without violating guards.
- Retrieval tests confirm metrics + budgets + expansion behavior; optional hybrid uplift (if gold available) meets target on the sample.
- (Optional) Generation fidelity meets thresholds on the labeled subset.

Failing critical gates yields no‑go; we will collect traces and file issues by stage.

---

## 16) Next Steps (Pending Your Approval)

- Approve this spec and the proposed file layout under `tests/e2e_v22_prod/`.
- Provide the production snapshot or doc set and the `doc_tag` to apply during ingest (we will add a snapshot manifest helper to generate the doc list + tag).
- (Optional) Provide a small gold set (queries + supporting spans) to enable Recall@k checks.
- After approval, we will scaffold the tests (without running them) and wire fixtures to your snapshot path and doc_tag. Once configured, we can run the suite and archive artifacts.
