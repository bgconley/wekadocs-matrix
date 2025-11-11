# Hybrid Retrieval Pipeline – Full Evaluation (2025‑11‑02)

## Executive Summary
- Current hybrid retrieval blends BM25 (Neo4j full‑text) + vector (Qdrant) with RRF fusion, optional adjacency expansion, and stitched context assembly.
- Recent regression pack runs still show cross‑document citation mixing and within‑document ordering glitches. We implemented two production‑safe improvements (dominant‑document gating and citation sorting), which improve coherence but do not fully resolve all cases.
- Root causes concentrate in: (1) lack of document scoping in retrieval, (2) fusion/seed ordering heuristics preferring citation richness without doc‑continuity, and (3) heading normalization/sorting edge cases.
- Recommended P0 improvements (production‑safe): add doc‑scoping (doc_tag) into both engines and fuse logic, strengthen document‑continuity constraints before assembly, and retain both parent “Step” and deeper “Detail” citations when they are informative for order.

## Architecture Overview (as‑built)
- BM25 (Neo4j full‑text): Indexed across `Chunk|CitationUnit` on `[text, heading]`. See `src/query/hybrid_retrieval.py:103` and search pipeline at `src/query/hybrid_retrieval.py:235`.
- Vector (Qdrant): Embeds query, searches collection `chunks`, optional field‑equality filter passthrough. See `src/query/hybrid_retrieval.py:421`.
- Fusion: RRF by default (k=60) or weighted fusion with score normalization. See RRF at `src/query/hybrid_retrieval.py:748`.
- Expansion: Optional 1‑hop `NEXT_CHUNK` neighbors for first N seeds with down‑weighted scores. See `src/query/hybrid_retrieval.py:981`.
- Hydration: Adds citation labels to vector‑only winners via `CitationUnit -> IN_CHUNK`, with fallbacks to `boundaries_json` or `original_section_ids`. See `src/query/hybrid_retrieval.py:820`.
- Context budget enforcement: Token‑bounded grouping by `parent_section_id`, maintains intra‑section order. See `src/query/hybrid_retrieval.py:1059`.
- Stitched Markdown and citations rendering: `ContextAssembler` assembles ordered text and citation block; recent fix sorts labels. See `src/query/context_assembly.py:335`.
- Dominant‑doc gating (new): Applied after hybrid retrieval, before assembly. See `src/mcp_server/query_service.py:388`.

## Strengths
- RRF fusion is robust under diverse score distributions; weighted fusion is available when normalization is acceptable.
- BM25 includes `CitationUnit` hits and boosts CU scores (`CITATIONUNIT_BOOST`), increasing the chance of meaningful headings/citations.
- Hydration path covers both presence/absence of CUs and leverages boundaries + original sections for fallbacks.
- Expansion is bounded and never outranks the seed chunk, which avoids runaway graph traversal.
- SLO metrics and observability are present for retrieval p95 and expansion rate.

## Weaknesses and Brittle Areas
1) Cross‑Document Mixing (dominant in failed regressions)
- Symptom: Top stitched citations are from the wrong document when the query embeds a hint (e.g., `REGPACK‑01`).
- Technical cause:
  - No doc‑scoping dimension in either engine beyond arbitrary `filters` (only `embedding_version` is set by default).
  - Full‑text analyzer likely tokenizes `REGPACK-01 → [regpack, 01]`, providing weak discriminative power; RRF then blends hits from several docs, and later ordering by citation richness can bias toward unrelated docs.
- Evidence:
  - Failures where 1st citation is from another case (“Bootstrap” or API endpoint) despite a specific case query.
  - Code: BM25 `search()` applies only equality filters to resolved chunk properties, no doc‑tag filter. `src/query/hybrid_retrieval.py:251-257`.
  - Vector `search()` supports equality filters but is called without a doc identifier. `src/query/hybrid_retrieval.py:453-476`.

2) Seed Ordering Heuristic Prefers Citation Richness over Doc Continuity
- Symptom: Fusion seeds and final sort favor chunks with more citation labels, which can displace the correct doc when that doc has fewer labels in early ranks.
- Technical cause:
  - Sorting by `(len(citation_labels), fused_score)` in multiple stages risks doc jitter when different docs are interleaved. See `src/query/hybrid_retrieval.py:704-713` and again at `src/query/hybrid_retrieval.py:1011-1019`.

3) Within‑Document Ordering Edge Cases (deep nesting)
- Symptom: Child detail headings (e.g., H4 “Detail A…”) can appear before the parent Step 1 citation.
- Technical cause:
  - Prior to the recent fix, labels were not consistently normalized/sorted. While we now sort `(order asc, title asc)` at render (`src/query/context_assembly.py:335-367`), upstream CU hydration filters to `max_level` (`_assign_normalized_citations`) can exclude parent headings in favor of deepest level, which isn’t always desirable for ordered Step citations. See `src/query/hybrid_retrieval.py:874-899`.

4) Dominant‑Doc Gating Location and Strength
- Symptom: Post‑retrieval gating (in `QueryService`) helps but arrives too late to influence expansion/hydration trade‑offs.
- Technical cause:
  - Gating occurs after `HybridRetriever.retrieve()` returns (`src/mcp_server/query_service.py:388-421`). Expansion/hydration were already computed on possibly mixed seeds, and return ordering is already biased.

5) BM25 Index Shape and Analyzer
- Symptom: Mixed hits across `Chunk|CitationUnit` and analyzer behavior may favor popular headings from unrelated docs (e.g., “Install Widget Pro — Quickstart”).
- Technical cause:
  - Index targets two labels and two properties with default analyzers. No per‑doc family boost or scoped query constructs; no hard facet/field for doc family.

6) Expansion Adds Neighbors Without Doc Check
- Symptom: In mixed scenarios, expansion may introduce more neighbors from the wrong doc once its seeds get in. (Down‑weighting mitigates, but seed set composition controls bias.)
- Technical cause:
  - Expansion uses seed chunk ids; no doc‑continuity constraint. See `src/query/hybrid_retrieval.py:1030-1054`.

## Fit‑for‑Purpose Assessment
- For general “find best related sections across a corpus” queries: good baseline (RRF + hydration + budget) with solid observability.
- For QA‑style or test‑fixture scenarios where the query is intended to target one document/family: current pipeline is brittle without an explicit scoping signal. In practice, real users often imply scope (“in X doc” or via UI context), so adding doc scoping greatly improves reliability without harming general use.

## Recommendations (Production‑Safe)
P0 – Document Scoping + Continuity
- Add a `doc_tag` dimension at ingest time to chunks/CitationUnits (e.g., parse from source URI, repo path, or metadata). Expose it as `doc_tag` field in Qdrant payload and a property in Neo4j.
- Accept `filters["doc_tag"]` from callers and pass to both BM25 and Vector retrievers so both engines are constrained. Code touchpoints:
  - BM25 filter: `src/query/hybrid_retrieval.py:251-257` (ensure `chunk.doc_tag = $filter_doc_tag`).
  - Qdrant filter: `src/query/hybrid_retrieval.py:453-476` (add `FieldCondition(key="doc_tag", match=MatchValue(value=...))`).
- If no explicit filter is passed, compute a dominant doc at the fusion stage (not after return):
  - Move the dominant‑doc gating logic inside `HybridRetriever.retrieve()` after seeds are formed and hydrated (right before expansion) to influence all subsequent steps.
  - Replace the current post‑retrieval gating in `QueryService` (`src/mcp_server/query_service.py:388-421`) with the in‑retriever gating (or keep both, but the in‑retriever one should be authoritative).

P0 – Seed Sorting With Doc Continuity Weight
- After fusion + hydration, sort seeds primarily by `(doc_id_mode, fused_score)` where `doc_id_mode` promotes chunks from the primary doc. Keep `(len(citation_labels))` as a tiebreaker, not the primary key. See `src/query/hybrid_retrieval.py:704-713` and `1011-1019`.

P0 – Citation Label Strategy
- In `_assign_normalized_citations`, don’t drop parent Step headings when deeper levels exist. Keep both the top‑level Step N and the first detail to preserve the expected ordering signal. See `src/query/hybrid_retrieval.py:874-899`.

P1 – Expansion Doc Guard
- Constrain `_bounded_expansion` to neighbors from the primary doc when doc gating is in effect. See `src/query/hybrid_retrieval.py:1030-1054`.

P1 – BM25 Analyzer and Indexing Strategy
- Consider adding a keyword field (e.g., `doc_tag`) to the full‑text query to scope touchpoints more cleanly.
- Reevaluate CU vs Chunk weighting (`CITATIONUNIT_BOOST`) for your corpus; instrument pass@k by doc family.

P1 – Observability & Telemetry
- Add per‑query doc_id distribution and a doc dominance score to metrics so you can quickly see cross‑doc mixing.
- Emit a fuse‑stage metric: fraction of seeds from primary doc; use it to alert when below a threshold.

## Concrete Action Plan
1. Ingest doc_tag
- During ingestion, stamp chunks and CUs with `doc_tag` (e.g., derived from source URI prefixes like `tests://regression_pack/REGPACK-XX`).
- Store doc_tag in Neo4j nodes and Qdrant payload (ensure collection schema supports it).

2. Pass doc_tag from QueryService
- When query pattern includes a known tag, add `filters["doc_tag"]` and pass to retriever; otherwise, leave unset and rely on in‑retriever dominance gating.

3. Move dominance gating into retriever
- Apply right after seed formation/hydration and before expansion to influence neighbor inclusion and final ranking.

4. Adjust seed/final sort keys
- Sort by `(doc_continuity_weight, fused_score, len(citation_labels))` where `doc_continuity_weight` is 1 for primary doc, 0 otherwise.

5. Update citation normalization
- Keep parent “Step N” and first “Detail …” when both exist; avoid max‑level filtering that removes parent signal.

6. Add expansion doc guard (when gated)
- Filter neighbors to `document_id == primary_doc_id` before adding to results.

7. Expand metrics
- Add per‑run doc distribution metrics and a new `hybrid_primary_doc_fraction` gauge.

## Code References
- BM25 full‑text search (filters only on resolved chunk): `src/query/hybrid_retrieval.py:235`.
- Vector search + filters passthrough: `src/query/hybrid_retrieval.py:421`.
- Fusion (RRF): `src/query/hybrid_retrieval.py:748`.
- Seed sorting by citation richness: `src/query/hybrid_retrieval.py:704` and `1011`.
- Hydration of missing citations and max‑level filter: `src/query/hybrid_retrieval.py:820`, `874-899`.
- Expansion (neighbors): `src/query/hybrid_retrieval.py:981`.
- Context budget enforcement and grouping: `src/query/hybrid_retrieval.py:1059`.
- New dominant‑doc gating location (currently post‑retrieval): `src/mcp_server/query_service.py:388`.
- Citation rendering sort (recent fix): `src/query/context_assembly.py:335`.

## Risks & Corner Cases
- Highly similar parallel documents (e.g., duplicated manuals or versions) may still require an explicit scoping signal; dominance gating alone can be ambiguous.
- Very short queries (“install”) will under‑specify intent; add UI‑level context signals where possible.
- Documents with inconsistent heading levels (missing or out‑of‑order) can reduce the effectiveness of order‑based sorting; keep fallbacks (boundaries/original_sections) and surface a “heading_integrity” flag in diagnostics.

## Test Strategy (to validate improvements)
- Golden tests for doc_tag filtering: ensure only the intended doc’s chunks appear in seeds and expansion.
- Multi‑doc ambiguity tests (no tag): assert dominance gating either keeps a single doc when dominance exists, or falls back to multi‑doc with clear diagnostics when not.
- Deep‑nest order tests: require Step N before Detail A consistently.
- Expansion guard tests: when gated, neighbors remain within the primary doc.

---
This assessment favors minimal, production‑safe changes that reduce brittleness without narrowing the system’s utility for broad queries. Adopting doc scoping and moving dominance control into the retriever will address the majority of failure modes while keeping fusion and expansion strengths intact.
