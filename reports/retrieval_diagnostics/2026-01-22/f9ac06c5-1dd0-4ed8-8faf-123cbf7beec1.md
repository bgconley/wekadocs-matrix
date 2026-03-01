# Retrieval Diagnostics — f9ac06c5-1dd0-4ed8-8faf-123cbf7beec1

- timestamp: 2026-01-22T03:16:58.506872Z
- transport: stdio
- tool: kb_retrieve_evidence
- session_id: server-2341d57f-96b3-4552-a5da-00eb904fa5a1
- trace: None/None
- scope: project_id=wekadocs-matrix env=development doc_tag=None

## Timings (ms)
- bm25: 0.0
- vector_search: 107.09786415100098
- fusion: 0.05555152893066406
- rerank: 8.892536163330078
- graph_expansion: 85.19315719604492
- total: 2686.3534450531006

## Counts
- candidates_initial: 60
- candidates_post_filter: 17
- candidates_post_dedupe: 17
- returned: 7
- dropped: dedupe=3 veto=0 scope_mismatch=0

## Top Results
| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | 5269b8caf1a267bd69e9266df5884bb1c817391d077805b935fa3fffe821a077_chunk_6_0f4b2923043e8dd6 | additional-protocols | reranked | 0.016972034715525556 | 0.0 | 0.0 | 110 |
| 2 | dc432ee5e95caa145ee1fa46087717aa37cee5fd1901688fa73fbb0be73bc2b5_chunk_0_5194b5c6484b475f | additional-protocols | reranked | 0.01635673624288425 | 0.0 | 0.0 | 55 |
| 3 | 1ebce2ff481201cd1dfde382517a1d06c18d09890f083d5a7865c20736d5b14a_chunk_40_f26def30f64b9c54 | additional-protocols | reranked | 0.016209150326797386 | 0.0 | 0.0 | 60 |
| 4 | 4e6a558a33d423117077d325fac378ac1eeacea6b738954045d038bfce4d86ea_chunk_2_da96af72a53a19ee | README | reranked | 0.015955882352941177 | 0.0 | 0.0 | 311 |
| 5 | a56d2ef7bda727517dfb1dd93cb8288e49fc0a2d038b375f2b7544a61eac5b52_chunk_1_2b56f8bb2750d613 | additional-protocols | reranked | 0.015258467023172906 | 0.0 | 0.0 | 133 |
| 6 | 9660a88fd0db08e30e202d0fe60b171acb7e142c1c528712c2097622191f3b1b_chunk_0_b79a105d9ec96532 | planning-and-installation | reranked | 0.015030728709394205 | 0.0 | 0.0 | 113 |
| 7 | c9f0021dcafaf10402bd0e00ecaa898748b3b104443d4b5b758277f8476100bc_chunk_0_ead49647b32d105d | weka-system-overview | reranked | 0.01480968858131488 | 0.0 | 0.0 | 242 |

## Budgets
- response_bytes: 6166
- tokens_estimate: 2586
- partial: False
- limit_reason: none
