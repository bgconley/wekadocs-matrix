# Retrieval Diagnostics — d1d6cac4-ffa9-4a2b-9435-e0c0a3eeec6f

- timestamp: 2026-01-19T23:12:57.438472Z
- transport: stdio
- tool: kb_retrieve_evidence
- session_id: server-2fa11f73-74db-4410-9af7-a3626cbae9b1
- trace: None/None
- scope: project_id=wekadocs-matrix env=development doc_tag=None

## Timings (ms)
- bm25: 0.0
- vector_search: 110.71228981018066
- fusion: 0.05507469177246094
- rerank: 2.6154518127441406
- graph_expansion: 5.02467155456543
- total: 3002.4685859680176

## Counts
- candidates_initial: 60
- candidates_post_filter: 17
- candidates_post_dedupe: 17
- returned: 6
- dropped: dedupe=4 veto=0 scope_mismatch=0

## Top Results
| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | 150dbf583fef35688e64e2ce569c52b36889a9e036e4c5646e59613033f081df_chunk_0_7fd83422dfb52eaa | getting-started-with-weka | reranked | 0.016624879459980715 | 0.0 | 0.0 | 99 |
| 2 | 5269b8caf1a267bd69e9266df5884bb1c817391d077805b935fa3fffe821a077_chunk_6_0f4b2923043e8dd6 | additional-protocols | reranked | 0.016698292220113854 | 0.0 | 0.0 | 110 |
| 3 | dc432ee5e95caa145ee1fa46087717aa37cee5fd1901688fa73fbb0be73bc2b5_chunk_0_5194b5c6484b475f | additional-protocols | reranked | 0.016097105508870214 | 0.0 | 0.0 | 55 |
| 4 | 1ebce2ff481201cd1dfde382517a1d06c18d09890f083d5a7865c20736d5b14a_chunk_40_f26def30f64b9c54 | additional-protocols | reranked | 0.015955882352941177 | 0.0 | 0.0 | 60 |
| 5 | 9660a88fd0db08e30e202d0fe60b171acb7e142c1c528712c2097622191f3b1b_chunk_0_b79a105d9ec96532 | planning-and-installation | reranked | 0.015601809954751134 | 0.0 | 0.0 | 113 |
| 6 | 4e6a558a33d423117077d325fac378ac1eeacea6b738954045d038bfce4d86ea_chunk_2_da96af72a53a19ee | README | reranked | 0.015258467023172906 | 0.0 | 0.0 | 311 |

## Budgets
- response_bytes: 5979
- tokens_estimate: 2535
- partial: False
- limit_reason: none
