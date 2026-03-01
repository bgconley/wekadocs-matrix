# Retrieval Diagnostics — fe3f515d-74cc-4c8d-9e78-3ee8e7dfce8a

- timestamp: 2026-01-20T03:23:30.645266Z
- transport: stdio
- tool: kb_search
- session_id: server-df282477-d9fd-41be-aa5f-59e489381c2f
- trace: None/None
- scope: project_id=wekadocs-matrix env=development doc_tag=None

## Timings (ms)
- bm25: 0.0
- vector_search: 104.65645790100098
- fusion: 0.03457069396972656
- rerank: 0.6618499755859375
- graph_expansion: 4.999637603759766
- total: 2698.9965438842773

## Counts
- candidates_initial: 30
- candidates_post_filter: 10
- candidates_post_dedupe: 10
- returned: 2
- dropped: dedupe=3 veto=0 scope_mismatch=0

## Top Results
| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | 74f7f45310a8d736693388991ade85da6862147ac229e5e6c1b6eaada3cd78ed_chunk_1_7fb99ba8c33ecea4 | additional-protocols | reranked | 0.01777049180327869 | 0.0 | 0.0 | 399 |
| 2 | 1ebce2ff481201cd1dfde382517a1d06c18d09890f083d5a7865c20736d5b14a_chunk_17_3d49f02414f426a7 | additional-protocols | reranked | 0.016444444444444446 | 0.0 | 0.0 | 16 |

## Budgets
- response_bytes: 3282
- tokens_estimate: 1523
- partial: False
- limit_reason: none
