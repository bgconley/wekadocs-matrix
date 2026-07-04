# Refactor Roadmap

## Phase 0: Freeze And Measure

- Keep source code unchanged until the audit findings are accepted.
- Add a resolved-config diagnostic command/report that prints effective domain, embedding, sparse, ColBERT, reranker, chunker, graph, and MCP profile.
- Create a legacy manifest for stale alternates and scripts.
- Capture the current dirty worktree as a separate decision: accept it as baseline, revert it intentionally, or move it to a cleanup branch. Do not mix that choice with Nutanix conversion.

## Phase 1: Domain Extraction

- Introduce a domain profile schema without changing retrieval behavior.
- Move MCP instructions, reranker instructions, embedding query instructions, NER labels, and domain examples into `domains/weka.yaml`.
- Add `domains/nutanix.yaml` with placeholder fields and explicit TODOs.
- Wire startup to load a domain profile and fail if domain instructions are missing.
- Add a domain-residue check that fails when `DOMAIN=nutanix` and active runtime prompts/config contain `WEKA`, `WekaDocs`, or `wekadocs`.

## Phase 2: Nutanix Corpus And Metadata

- Add a Nutanix corpus source directory or external ingest manifest.
- Define Nutanix doc taxonomy: product, version, component, doc type, command/API entities, troubleshooting categories.
- Update doc_tag/category extraction to use domain rules, not WEKA path assumptions.

## Phase 3: Retrieval Quality Gates

- Add per-query retrieval trace output: dense/sparse/title/entity/ColBERT/graph/reranker/fallback states.
- Convert fail-open behavior into visible degradation metadata.
- Build Nutanix golden queries and acceptance criteria.

## Phase 4: Module Decomposition

- Split `AtomicIngestionCoordinator` into parse/chunk/enrich/embed/write/link stages.
- Split `HybridRetriever` into query plan, candidate generation, graph expansion, rerank, and response/citation assembly.
- Delete or quarantine confirmed legacy modules after fresh tests prove no active imports.
- Convert direct fallback branches into explicit `RetrievalTrace` / `IngestionTrace` events so availability tradeoffs are observable.

## Phase 5: Runtime Rename/Cleanup

- Rename service/container/telemetry namespaces only after domain extraction proves behavior.
- Retire WEKA-specific env examples or move them under `examples/weka/`.
- Replace old root context clutter with an index of historical artifacts.

## Phase 6: Release Proof

- Run fresh Nutanix ingest on a representative corpus.
- Run golden retrieval/citation/evidence tests.
- Compare fallback/degradation metrics.
- Produce install-from-scratch proof from a clean checkout/container volume.
