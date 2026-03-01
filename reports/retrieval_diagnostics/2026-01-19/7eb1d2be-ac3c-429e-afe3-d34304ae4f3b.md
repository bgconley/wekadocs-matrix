# Retrieval Diagnostics — 7eb1d2be-ac3c-429e-afe3-d34304ae4f3b

- timestamp: 2026-01-19T23:13:18.703279Z
- transport: stdio
- tool: kb_search
- session_id: server-2fa11f73-74db-4410-9af7-a3626cbae9b1
- trace: None/None
- scope: project_id=wekadocs-matrix env=development doc_tag=None

## Timings (ms)
- bm25: 0.0
- vector_search: 109.18092727661133
- fusion: 0.05459785461425781
- rerank: 0.7567405700683594
- graph_expansion: 5.3195953369140625
- total: 3220.4341888427734

## Counts
- candidates_initial: 30
- candidates_post_filter: 12
- candidates_post_dedupe: 12
- returned: 2
- dropped: dedupe=3 veto=0 scope_mismatch=0

## Top Results
| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | 74f7f45310a8d736693388991ade85da6862147ac229e5e6c1b6eaada3cd78ed_chunk_1_7fb99ba8c33ecea4 | additional-protocols | reranked | 0.018032786885245903 | 0.0 | 0.0 | 399 |
| 2 | 95ac7b9cd6eaf52089d881e3b7d68b5190456f1603ddb884b5ed684eff539b5a_chunk_0_c0e1d1c4a59abbf3 | additional-protocols | reranked | 0.016451612903225808 | 0.0 | 0.0 | 208 |

## Budgets
- response_bytes: 3471
- tokens_estimate: 1537
- partial: False
- limit_reason: none
