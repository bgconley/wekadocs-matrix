# Target Architecture

## Goal

Keep the mature RAG machinery, but separate domain configuration from retrieval/ingestion infrastructure so WEKA and Nutanix are explicit domain packs instead of hardcoded assumptions.

## Proposed Layers

1. `domain/`
   - Domain profile: `id`, display name, product taxonomy, source corpus roots, entity labels, exclusion rules, instructions, golden queries.
   - WEKA profile becomes legacy/reference.
   - Nutanix profile becomes target.

2. `ingestion/`
   - Stage contracts: parse -> normalize metadata -> chunk -> enrich entities/references -> embed -> validate -> write -> post-commit linking.
   - `AtomicIngestionCoordinator` becomes a thin orchestrator over stage objects.

3. `retrieval/`
   - Query planning and provider selection are separated from execution.
   - A retrieval trace object records every signal/fallback/score transformation.

4. `mcp/`
   - Tools remain domain-neutral.
   - Instructions are loaded from domain profile at server startup.

5. `config/`
   - One resolved runtime config report that prints the effective embedding/rerank/chunking/graph/tool profile after env overrides.

6. `evals/`
   - Domain-specific golden query suite.
   - Citation faithfulness and retrieval coverage checks.
   - Fallback/degradation gates.

## Target Nutanix Runtime

- Service names and OTEL namespace can be generic (`rag-*`) or explicitly Nutanix (`nutanix-rag-*`), but should no longer say WEKA.
- Model instructions should mention Nutanix only through the domain profile, not hardcoded inside provider profiles.
- Corpus tags should reflect Nutanix product families, versions, doc type, and source URL path.
- Existing WEKA corpus can stay as historical fixture only if isolated.
