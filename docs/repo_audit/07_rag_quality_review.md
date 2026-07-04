# RAG Quality Review

## Strengths

- Retrieval has multiple complementary signals: dense, sparse, doc-title sparse, title sparse, entity sparse, ColBERT late interaction, structural graph expansion, cross-document `RELATED_TO`, references, and reranking.
- Ingestion enforces strict anti-truncation behavior for oversize embedding inputs, which is the right bias for evidence integrity.
- The MCP evidence tools preserve passage IDs, scratch payloads, excerpt reads, and quote/excerpt workflows, which is better than one-shot answer generation.
- Neo4j/Qdrant writes are coordinated in an atomic saga: Qdrant success is required before Neo4j commit when Qdrant is primary/dual-write.

## Quality Risks

1. Domain contamination: WEKA-specific query/rerank/entity instructions will bias retrieval and reranking for Nutanix terms.
2. Config drift: the system can silently run a different embedding/rerank stack than a reader expects from any single file.
3. Fail-open behavior: reranker/ColBERT/sparse/Query API fallbacks often preserve candidates rather than failing closed, which protects availability but hides quality regressions.
4. Missing Nutanix evals: no target-domain golden set, adversarial query set, citation-quality rubric, or corpus coverage checklist exists.
5. Graph quality depends on entity extraction and cross-doc linking. Structural/entity priors are powerful but can amplify domain-specific residue.
6. Existing tests are not reliable proof. They may assert old paths or stale behavior.

## Cross-Doc Linking Note

The most recent external re-evaluation of `src/services/cross_doc_linking.py` is directionally better than the first review: most alleged critical bugs were false positives. The remaining meaningful concerns are operational quality issues, especially ColBERT fail-open behavior when source/target vectors are missing and silent/debug-only prior failures. For this repo-audit, those are medium quality/observability risks, not blockers to understanding the active architecture.

## Nutanix Acceptance Bar

Before claiming Nutanix readiness, require:

- A Nutanix corpus inventory and ingestion proof.
- Domain-neutralized runtime prompts and model instructions.
- Nutanix entity/metadata taxonomy.
- Golden queries with expected supporting passages.
- Citation audits for at least installation, cluster operations, troubleshooting, APIs/CLI, and version-specific docs.
- A fail-open observability report that states when sparse/ColBERT/reranker/graph fell back.
