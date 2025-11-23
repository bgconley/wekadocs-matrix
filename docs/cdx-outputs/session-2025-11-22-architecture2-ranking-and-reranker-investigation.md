# WekaDocs Matrix – Architecture 2 Ranking & Reranker Investigation Notes

_Session date: 2025-11-22 → 2025-11-23_

This document captures the full working context of the Architecture 2
("cascading waterfall") ranking implementation and the subsequent
investigation into reranker behaviour and infrastructure wiring. It is
intended as a durable, highly detailed reference for future work on the
hybrid retrieval, reranking, and ranking pipeline in WekaDocs Matrix.

The focus areas for this session were:

- Implementing the Architecture 2 ranking blend of recall, rerank, and
  graph signals.
- Ensuring BGE-M3 multi-vector retrieval (dense, sparse, ColBERT) uses a
  magnitude-preserving fusion rather than pure rank-based RRF.
- Integrating the BGE cross-encoder reranker in a way that treats it as
  a feature, not a dictator, and adding a veto mechanism.
- Propagating semantic scores from seed chunks to graph neighbors.
- Debugging surprising reranker outputs (very negative logits) and the
  interaction with our veto and blend logic.
- Untangling infrastructure issues around the local embedding service
  and reranker service when called from inside the `mcp-server`
  container.

The document is structured as follows:

1. Prior baseline and Architecture 2 intent.
2. Code-level changes implemented in this session.
3. Reranker behaviour and scoring anomalies.
4. Embedding and reranker service connectivity from Docker.
5. Current behaviour of the end-to-end smoke test.
6. Files touched and specific modifications.
7. What works vs what is broken.
8. Next steps and open questions.

## 1. Baseline Architecture and Architecture 2 Intent

Before this session, the retrieval stack had already undergone a
significant migration toward BGE-M3 embeddings and Qdrant’s
multi-vector collections, but ranking and reranking still bore the
legacy of a Phase 7E design built around BM25 plus RRF.

Key baseline elements were:

- **Embedding profile**: `bge_m3` with dense, sparse, and ColBERT
  vectors, driven by `BAAI/bge-m3` and a local BGE-M3 service.
- **Qdrant collection**: `chunks_multi_bge_m3`, a multi-vector
  collection with named dense vectors, ColBERT late-interaction vectors
  under `late-interaction`, and a sparse field `text-sparse`.
- **Hybrid retrieval**: BM25 from Neo4j plus vector search from Qdrant,
  with fusion typically done by RRF at various points.
- **Reranker integration**: a cross-encoder reranker (originally Jina,
  later a local BGE reranker service) that was allowed to dominate
  ranking by overriding vector scores.

Several architectural issues were already known:

- Over-reliance on RRF, which discards magnitude and collapses strong vs
  weak matches into similar positions.
- Treating reranker score as the primary or only semantic signal once it
  existed, effectively discarding the useful consensus among dense,
  sparse, and ColBERT.
- Graph neighbors (added via enrichment) starting with weak or no
  semantic scores, making them second-class citizens even when they were
  contextually important.
- A normalisation layer (`_normalize_score` in `src/query/ranking.py`)
  that tried to juggle many score types, including ColBERT
  late-interaction, but sometimes clamped or distorted them.

Architecture 2, the "cascading waterfall" design, was articulated to
address these issues:

- **Recall**: Perform multi-vector recall with BGE-M3 (dense, sparse,
  ColBERT) in Qdrant, using a magnitude-preserving fusion so that
  `vector_score` represents a meaningful strength and consensus signal.
- **Rerank**: Apply a cross-encoder reranker to the strongest seeds,
  obtaining `rerank_score` logits that reflect semantic compatibility
  between the query and each chunk.
- **Fusion**: Blend recall and rerank scores into a single semantic
  score, rather than letting the reranker overwrite recall.
- **Graph**: Propagate semantic strength from seeds to graph neighbors
  and incorporate graph distance as an additional feature.

The high-level formula that guided the implementation is:

- `FinalScore = W_recall * Norm(S_recall) + W_rerank * Sigmoid(S_rerank)
  + W_graph * S_graph`

with typical starting weights of 0.4 / 0.4 / 0.2 for recall, rerank, and
graph contributions.

## 2. Configuration and Model Changes

### 2.1 HybridSearchConfig extensions

The `HybridSearchConfig` model in `src/shared/config.py` was extended to
support Architecture 2’s blend and veto semantics. New fields:

- `semantic_recall_weight: float = 0.4`
- `semantic_rerank_weight: float = 0.4`
- `semantic_graph_weight: float = 0.2`
- `reranker_veto_threshold: float = 0.2`
- `graph_propagation_decay: float = 0.85`

These fields control how the Ranker blends recall, rerank, and graph
signals, and how aggressively the reranker can veto candidates.

The relevant configuration file `config/development.yaml` was updated to
set these for the development profile under `search.hybrid`:

- `mode: "bge_reranker"` (already present from prior work).
- `method: "weighted"` – switches fusion away from RRF toward a
  magnitude-preserving weighted fusion.
- `semantic_recall_weight: 0.4`
- `semantic_rerank_weight: 0.4`
- `semantic_graph_weight: 0.2`
- `reranker_veto_threshold: 0.2`
- `graph_propagation_decay: 0.85`

The reranker configuration in both the model and dev config was aligned
with the desired BGE reranker behaviour:

- `RerankerConfig.max_tokens_per_pair` increased from 800 to 1024 to
  give the cross-encoder more room to see meaningful context.
- `config/development.yaml` updated to match this.

### 2.2 Reranker provider configuration

The system already supported a `bge-reranker-service` provider via
`BGERerankerServiceProvider` in
`src/providers/rerank/local_bge_service.py`. The provider is configured
by:

- `model: "BAAI/bge-reranker-v2-m3"`.
- `base_url`: now set via `RERANKER_BASE_URL`, defaulting to
  `http://127.0.0.1:9001` but overridden to
  `http://host.docker.internal:9001` from inside `mcp-server` when
  running in Docker.

The provider uses a simple HTTP POST to `/v1/rerank` and expects a
payload:

```json
{
  "query": "...",
  "documents": ["doc text 1", "doc text 2", ...],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

The response is parsed as:

- `results`: list of `{index, score, document}` entries.

Instrumentation was added to log:

- Provider name, model, document count, top_k.
- First few scores.
- Snippets of the query and first couple of documents.

This logging was essential for debugging reranker behaviour later in the
session.

## 2. Code-Level Changes Implemented

This section documents the concrete code changes made to move toward
Architecture 2. The emphasis is on the ranking and retrieval pipeline,
config knobs, and the wiring of the BGE reranker service.

### 2.1 Configuration Model Extensions

File: `src/shared/config.py`

The `HybridSearchConfig` and `RerankerConfig` Pydantic models were
extended to support Architecture 2 controls and more precise reranker
behaviour.

- `RerankerConfig`:
  - `max_tokens_per_pair` increased from `800` to `1024` to allow
    slightly longer document contexts in rerank pairs.
- `HybridSearchConfig` gained new fields:
  - `semantic_recall_weight: float = 0.4`
  - `semantic_rerank_weight: float = 0.4`
  - `semantic_graph_weight: float = 0.2`
  - `reranker_veto_threshold: float = 0.2`
  - `graph_propagation_decay: float = 0.85`

These provide a configurable way to tune the blend between recall,
rerank, and graph, as well as a veto mechanism for very low rerank
probabilities and a decay factor for semantic propagation to neighbors.

### 2.2 Development Configuration Updates

File: `config/development.yaml`

The development config was adjusted to reflect our new default mode and
weights during experimentation.

- `search.hybrid.mode` set to `"bge_reranker"` to select the new
  vector-only + cross-encoder reranker mode (BM25 disabled).
- `search.hybrid.method` set to `"weighted"` so that client-side fusion
  uses a magnitude-preserving approach rather than pure RRF.
- The reranker block updated:
  - `max_tokens_per_pair: 1024`.
- New knobs surfaced under `search.hybrid`:
  - `semantic_recall_weight: 0.4`
  - `semantic_rerank_weight: 0.4`
  - `semantic_graph_weight: 0.2`
  - `reranker_veto_threshold: 0.2`
  - `graph_propagation_decay: 0.85`

BM25 remains available for legacy mode but is disabled in
`bge_reranker` mode to simplify analysis of vector + rerank dynamics.

### 2.3 ChunkResult & Graph Propagation

File: `src/query/hybrid_retrieval.py`

The `ChunkResult` dataclass, which carries retrieval metadata throughout
the hybrid pipeline, was extended with a new field:

- `inherited_score: Optional[float] = None`

This is used to store a propagated semantic score inherited from a seed
chunk, allowing graph neighbors to carry more semantic weight when
ranking.

Graph enrichment was updated in
`HybridRetriever._fetch_graph_neighbors`:

- Previously, neighbors received a `graph_score` based on
  `1.0 / (graph_distance + 1)` and a fused score that combined the
  source fused score and graph score with a simple weighted formula.
- Now, we compute a semantic source score as the maximum of:
  - `source_chunk.fused_score`
  - `source_chunk.vector_score`
  - `source_chunk.bm25_score`
  - plus, if present, `sigmoid(source_chunk.rerank_score)`.
- A propagated score is computed as:
  - `propagated = source_score * graph_propagation_decay`.
- For each neighbor:
  - `chunk.inherited_score = propagated`
  - `chunk.fused_score` is set to at least `propagated`.
  - `chunk.vector_score` is set to `chunk.fused_score`.
  - `chunk.vector_score_kind` is set to `"graph_propagated"` if not
    already set.

This ensures that graph neighbors start with a meaningful semantic
signal rather than zero, so they can compete with seeds if the graph
relationship is strong.

### 2.4 Qdrant Multi-Vector Fusion Changes

File: `src/query/hybrid_retrieval.py` (class `QdrantMultiVectorRetriever`)

The legacy `_fuse_rankings` method implemented RRF over per-field
rankings, losing magnitude information. It was changed to implement a
simple weighted, magnitude-preserving fusion:

- Inputs: `rankings: Dict[str, List[Tuple[str, float]]]` mapping vector
  field name (e.g., `content`, `title`, `entity`, sparse field) to a
  list of `(point_id, score)` pairs.
- We now compute:
  - `max_by_field[vector_name] = max(score)` for each field, ignoring
    `None` scores.
  - For each `(pid, raw_score)` in each field:
    - Normalize `normalized = raw_score / max_score` if `max_score > 0`
      else `0.0`.
    - Accumulate: `fused[pid] += weight * normalized`, where
      `weight = self.field_weights.get(vector_name, 1.0)`.

The result is a fused score in a roughly [0,1] range, with different
vector fields contributing according to configured field weights.

In `_search_legacy`, the computed `fused_score` is now passed as both
`fused_score` and `vector_score` to `_chunk_from_payload`, and the
fusion method is tagged as `"weighted"`. This avoids subtle situations
where vector scores would reflect only the primary dense field while the
fused score represented a multi-field mix.

The Query API path (`_search_via_query_api`) was also adjusted:

- `Prefetch` now uses `Fusion.DBSF` when available (falling back to
  `Fusion.RRF` if DBSF is not exposed) to encourage
  magnitude-preserving fusion on the Qdrant side.
- Returned `ScoredPoint` scores are treated as fused vector scores, with
  `vector_score_kind = "similarity"` and `fusion_method = "weighted"`.

### 2.5 HybridRetriever Fusion and Rerank Flow

File: `src/query/hybrid_retrieval.py` (class `HybridRetriever`)

#### Fusion method selection and BM25 gating

- `self.fusion_method` now reflects the `HybridSearchConfig.method`
  (`"rrf"` vs `"weighted"`).
- In `bge_reranker` mode, BM25 is disabled entirely; only vector results
  are used as seeds. In legacy mode, BM25 may still be consulted and
  fused.

#### Vector-only path in bge_reranker mode

In `HybridRetriever.retrieve`:

- If `hybrid_mode == "bge_reranker"`, BM25 is skipped, and we rely
  solely on the multi-vector Qdrant retriever.
- Vector results are taken as fused results; for each chunk:
  - If `fused_score` is `None`, it is set to `vector_score`.
  - `fusion_method` is set to `"weighted"` to reflect the new fusion.

#### Reranker candidate construction and instrumentation

`_apply_reranker` now constructs richer candidate texts and logs
meaningful samples:

- Candidate text construction:
  - `text_body = chunk.text.strip()`.
  - `heading = chunk.heading.strip()`.
  - If both exist: `text = f"{heading}\n\n{text_body}"`.
  - Else: whichever is non-empty, or the candidate is skipped if both
    are empty and token_count is zero.
- Candidates are passed to the reranker service as:
  - `{ "id": chunk.chunk_id, "text": text, "original_result": chunk }`.

After reranking, we log sample payloads:

- Up to three reranked payload entries are sampled.
- We log:
  - A query snippet (first 200 characters).
  - A list of doc snippets (first 200 characters of each sampled
    candidate text).
  - The rerank scores for those candidates.

This log is crucial for verifying that the reranker is being fed
well-formed query and document text, and for debugging surprising
logits.

### 2.6 Ranking Pipeline Changes

File: `src/query/ranking.py`

The `Ranker` class is responsible for converting raw search results into
ranked outputs with a variety of features. It was significantly updated
for Architecture 2.

#### RankingFeatures expanded

The `RankingFeatures` dataclass gained additional fields to reflect the
new blend logic:

- `recall_score: float = 0.0`
- `rerank_score: float = 0.0` (now the post-sigmoid probability or
  inherited semantic, not the raw logit)
- `inherited_score: float = 0.0` (propagated semantic from seeds)

These are tracked alongside existing fields like `semantic_score`,
`graph_distance_score`, `recency_score`, `entity_priority_score`, and
`coverage_score`.

#### Ranker configuration

`Ranker.__init__` now reads Architecture 2 weights from config:

- `semantic_recall_weight`
- `semantic_rerank_weight`
- `semantic_graph_weight`
- `reranker_veto_threshold`

It still preserves older weight fields used for non-semantic features
(e.g., recency, entity priority, coverage) but the main semantic blend
now uses the explicit recall/rerank/graph weights.

#### Max vector score precomputation

In `Ranker.rank`, we now compute:

- `max_scores_by_kind`: the maximum vector score per `vector_score_kind`
  for batch-relative normalisation (used especially for ColBERT and
  similar kinds).
- `max_vector_score`: the maximum vector score overall, as a fallback
  for normalisation when a specific kind is not available.

These values are passed into `_extract_features` to calibrate recall
scores.

#### Feature extraction and semantic blend

`_extract_features` now:

1. Reads raw metadata:
   - `vector_score`, `vector_score_kind`, `bm25_score`, `rerank_score`,
     `inherited_score` from the `SearchResult` metadata.
2. Computes a raw recall score:
   - `recall_raw = vector_score if available else result.score`.
   - `recall_norm` computed via `_normalize_score` using
     `vector_score_kind` when available, or via division by
     `max_vector_score` otherwise.
3. Computes a rerank probability:
   - If `rerank_score` is present, interpret it as a logit and compute
     `rerank_prob = sigmoid(rerank_score)`.
   - Else, if `inherited_score` is present, treat that as an already
     normalised semantic value in [0,1].
4. Applies a veto mechanism:
   - If `rerank_score` is present and `rerank_prob <
     reranker_veto_threshold`, semantic_score is forced to 0.0.
5. Computes semantic_score under Architecture 2:
   - If not vetoed:
     - `semantic_score = semantic_recall_weight * recall_norm +
       semantic_rerank_weight * rerank_prob`.

The graph distance feature is still computed via `_distance_score`, but
we now fold it into the semantic+graph composite used for the final
score:

- `semantic_with_graph = semantic_score + semantic_graph_weight *
  graph_distance_score`.

Finally, the `final_score` is computed as:

- If veto is active: `final_score = 0.0`.
- Else, if the fused method was `"rrf"`, we keep the original fused
  score as primary and add a tiny epsilon-weighted version of our new
  semantic+extras for tie-breaking.
- Otherwise, `final_score = semantic_with_graph + extras`, where extras
  include recency, entity priority, coverage, and any entity-focus
  bonuses.

This structure keeps the Architecture 2 blend at the centre of ranking
while preserving some backwards compatibility for RRF-based flows.

### 2.7 Reranker Service Client Logging

File: `src/providers/rerank/local_bge_service.py`

The `BGERerankerServiceProvider` was instrumented to facilitate deeper
inspection of reranker behaviour.

- After calling `/v1/rerank`, we now parse the response JSON and log a
  debug entry with:
  - Provider name and model.
  - Document count and `top_k`.
  - Number of results.
  - The scores of the first few results.
  - A query snippet (first 200 characters).
  - Snippets of the first two documents (first 200 characters).

This logging is intentionally narrow (limited to a few docs and
characters) so that it provides visibility without flooding logs. It is
crucial for verifying whether negative logits are associated with
unexpected text formatting or content.


## 3. Reranker Behaviour and Scoring Anomalies

With the above plumbing in place, we began investigating the behaviour
of the BGE reranker (`BAAI/bge-reranker-v2-m3`) when applied to Weka
documentation chunks.

### 3.1 Direct Probes vs In-Pipeline Calls

Direct tests against the reranker service, outside of the MCP server and
Docker environment, showed that the model could produce sensible
logits for domain-specific Weka content.

Example:

- Query: "What are the prerequisites for storage expansion?"
- Documents: several excerpts from the WekaDocs Matrix documentation on
  dynamic provisioning, persistent volumes, and heterogeneous systems.
- Results: logits in a healthy positive range for clearly relevant
  documents, e.g. +2.5, +3.0, etc., indicating that the model understood
  the domain material and query.

However, when the reranker was exercised via the MCP server and the
hybrid retrieval pipeline, we repeatedly observed strongly negative
logits, such as:

- `[ -6.87, -7.95, -3.91 ]` for three Weka chunks related to storage
  expansion.

Applying the logistic sigmoid function to such logits yields very small
probabilities:

- `sigmoid(-6.87) ≈ 0.0010`
- `sigmoid(-7.95) ≈ 0.00035`
- `sigmoid(-3.91) ≈ 0.0195`

When combined with the Architecture 2 veto threshold (0.2), all of
these results would be treated as vetoed, effectively zeroing their
semantic contribution. This is why evidence confidences in some earlier
runs appeared as 0.00 when the reranker was active.

### 3.2 Hypotheses for Negative Logits

Given that the same model could produce positive logits in direct tests,
while producing negative logits in the pipeline, we considered several
hypotheses:

1. **Text formatting mismatch**:
   - Perhaps the text slices being sent from the pipeline were too short
     or stripped of crucial headings.
   - Or perhaps markdown or templating artefacts were confusing the
     model, effectively making the chunks look less semantically aligned
     with the query.
2. **Query mismatch**:
   - The pipeline query may differ subtly from the test query (e.g.,
     including additional context or formatting) and lead to different
     semantics.
3. **Truncation or tokenisation issues**:
   - Earlier limits at 800 tokens per pair might have truncated away the
     most relevant parts of the chunk.
4. **Score bias and calibration**:
   - It is possible that this reranker checkpoint tends to produce
     negative logits even for moderately relevant matches, and the
     decision boundary is not near 0.

To explore these, we first improved the way text is constructed for
candidates (heading + body) and increased `max_tokens_per_pair` to
1024. We also added logging of query/document snippets to verify the
actual text being sent.

### 3.3 Discussion of Bias and Veto Threshold

A separate manual analysis suggested that adding a positive bias
(e.g. `+2.0`) to rerank logits could be helpful, shifting moderately
relevant pairs into a stronger regime while leaving clear negatives far
into the rejection zone. For example:

- A logit of +2.5 produces `sigmoid(2.5) ≈ 0.92`, already quite high.
- Adding a bias to weak positives (e.g., from +0.5 to +2.5) can push
  them above a veto threshold like 0.2.

However, in this working session we **did not** apply any bias in code.
All observed logits and probabilities were raw outputs from the
reranker, and the behaviour of the veto mechanism is based purely on
those raw logits.

Our initial `reranker_veto_threshold` of 0.2 assumes that the reranker’s
sigmoid outputs for reasonable matches should fall comfortably above
0.2. With the negative logits we observed in the pipeline, this
threshold effectively zeroes out many candidates. Once we are confident
that the input text and query formatting are correct, this threshold may
need to be revisited.

For now, the key conclusion is that the reranker is **not** inherently
broken; the negative logits reflect either

- a mismatch between query and chunk as seen by the model, or
- an imbalance in how we package context for the model in the pipeline.

The Architecture 2 pipeline is therefore operating correctly according
to its specification: low rerank probabilities are being vetoed.


## 4. Embedding & Reranker Service Connectivity in Docker

During this session we also ran into, and partially resolved, a series
of infrastructure issues related to calling the local embedding service
and reranker service from within the `mcp-server` Docker container.

### 4.1 Services and Ports

We are dealing with two local HTTP services:

1. **Embedding service** (BGE-M3 encoder)
   - Intended to run on the host machine, typically at port 9000.
   - Used by `src/providers/embeddings/bge_m3_service.py` via the
     `BGE_M3_API_URL` environment variable.
2. **Reranker service** (BGE cross-encoder reranker)
   - Intended to run on the host machine at port 9001.
   - Used by `BGERerankerServiceProvider` via a base URL, from
     `RERANKER_BASE_URL` or defaulting to `http://127.0.0.1:9001` if not
     overridden.

Direct tests from the host showed both services functioning when called
via `http://127.0.0.1:<port>`, including `/healthz` and `/v1/rerank`.
The problems arose specifically when calling these services from inside
`mcp-server`.

### 4.2 host.docker.internal and IPv6

Inside Docker, `host.docker.internal` is a special hostname that
resolves to the host machine. During investigation we observed that,
after a certain point in time, `host.docker.internal` resolved to an
IPv6-only address inside containers:

- `getent hosts host.docker.internal` returned
  `fdc4:f303:9324::254 host.docker.internal`.

If a service is listening only on IPv4 (e.g. `127.0.0.1:9000`), an IPv6
route to the host will fail. This led to `Network is unreachable` errors
when attempting to call `http://host.docker.internal:9000` from
`mcp-server`.

To force an IPv4 mapping inside the container, we updated
`docker-compose.yml` for `mcp-server`:

```yaml
extra_hosts:
  - "host.docker.internal:192.168.65.2"
```

This ensured that `/etc/hosts` inside the container maps
`host.docker.internal` to the Docker host gateway’s IPv4 address. After
this change, `getent ahosts host.docker.internal` correctly showed
IPv4 addresses.

### 4.3 Binding Services on the Host

Even with `host.docker.internal` mapped to an IPv4 address, services
must be listening on an address that the container can reach. We
confirmed the following behaviours:

- Starting the embedding service as
  `python -m embedding_service --host 0.0.0.0 --port 9000` still resulted
  in Uvicorn reporting `http://127.0.0.1:9000`. This indicates that the
  module entrypoint hard-coded or overrode the host argument.
- Starting the embedding service via uvicorn directly, as
  `uvicorn embedding_service.api.app:app --host 0.0.0.0 --port 9000`, is
  the more reliable way to ensure it binds to all interfaces (0.0.0.0).
- Reranker service on 9001 was already listening on IPv4 and was
  reachable from `mcp-server` using
  `RERANKER_BASE_URL=http://host.docker.internal:9001`.

Despite these adjustments, repeated tests showed that the embedding
service was still not reachable from inside the `mcp-server` container,
indicating that either:

- the service had not actually bound to 0.0.0.0 as expected, or
- there was some host-level restriction (routing or firewall) blocking
  connections from `192.168.65.2:9000` to the host process.

Given that the reranker on 9001 remained reachable, and the direct tests
against the embedding service from the host continued to succeed, the
most plausible explanation is that the embedder process was, at various
points, still bound only to 127.0.0.1.

This part of the debugging is not fully resolved in this session; the
key takeaway is that:

- Reranker connectivity is functional when `RERANKER_BASE_URL` points to
  `http://host.docker.internal:9001`.
- The embedding service must be started with an explicit uvicorn command
  that binds to 0.0.0.0 to be reachable from `mcp-server`.


## 5. Current Smoke Test Behaviour

The canonical smoke test is `scripts/smoke_test_query.py`, which runs a
complete query through the MCP server:

- It initialises connections to Neo4j, Qdrant, and Redis.
- It constructs a `QueryService` and executes a search for a standard
  query, such as "What are the prerequisites for storage expansion?".
- It prints answer markdown and a structured JSON representation of the
  response.

During this session, we saw multiple smoke test runs with different
behaviours depending on service reachability and configuration.

### 5.1 Vector-Only Without Reranker

When the reranker service was unreachable or disabled, the pipeline
behaved as follows:

- Hybrid search used Qdrant multi-vector recall with the new weighted
  fusion.
- BM25 was disabled in `bge_reranker` mode, so only vector recall
  contributed to seeds.
- Ranking used vector-based semantic scores plus graph and recency, but
  without rerank contributions.
- Evidence confidences were moderate, typically around 0.3, reflecting
  the vector + graph blend.

### 5.2 With Reranker Reachable

When the reranker service at `http://host.docker.internal:9001` was
reachable, and embeddings were functioning, the pipeline logs showed:

- The reranker successfully reranking a handful of top seeds (e.g., 4
  candidates) with non-zero latency.
- However, in prior runs before improving text construction,
  rerank_scores were strongly negative, leading to near-zero semantic
  probabilities.
- With the veto threshold at 0.2, these probabilities triggered the
  veto logic, zeroing semantic scores and creating counterintuitive
  confidences.

At the end of this session, the full combination of

- Qdrant multi-vector recall,
- local embedding service,
- reranker service, and
- Architecture 2 ranking

had not yet been validated in a fully healthy state due to the
lingering connectivity issues with the embedding service.

Nevertheless, the structural pieces (fusion, candidate construction,
rerank integration, ranking blend, and propagation) are all in place.
