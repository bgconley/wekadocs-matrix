You're absolutely right - let me display the complete task list with ALL tasks for every day. Here's the full implementation checklist:

  📋 Complete Golden Plan Task List - Phase 7-9 Implementation

  📅 WEEK 1: Foundation

  Phase 7a: Fix E1-E7 + Safety + Metrics (Days 1-2)

  Day 1 (Monday) - Foundation & Safety

  - Remove APOC dependency from traversal.py (5 min)
  - Update relationship whitelist to MENTIONS|HAS_SECTION
  - Add EXPLAIN-plan guard (reject Expand(All), unlabeled) (30 min)
  - Add driver timeout 1500ms to Neo4j connections (10 min)
  - Create neo/explain_guard.py with validation logic
  - Define 20-query golden set from real user queries (1 hour)
  - Document golden set in /docs/golden-set-queries.md
  - Freeze graph schema for all phases (30 min)
  - Document frozen schema in /docs/api-contracts.md
  - Create feature flags: FEATURE_VERBOSITY_ENABLED, FEATURE_GRAPH_MODE_ENABLED (1 hour)
  - Add feature flag configuration to config/defaults.yaml
  - Start metrics collection (verbosity_total, response_bytes, graph_nodes)
  - Create /tests/e2e/test_golden_set.py test file
  - Run baseline tests with current implementation
  - Document baseline metrics in /reports/phase-7/baseline.csv

  Day 2 (Tuesday) - Verbosity Implementation

  - Fix existing response_builder.py verbosity logic
  - Implement snippet mode (200-char limit)
  - Implement full mode (16KB/2000 tokens limit)
  - Implement graph mode (1-hop, 50 nodes max)
  - Add truncation logic at paragraph boundaries
  - Update query_service.py to handle verbosity parameter
  - Wire verbosity through MCP layer in stdio_server.py
  - Test with Claude Desktop - snippet mode
  - Test with Claude Desktop - full mode
  - Test with Claude Desktop - graph mode
  - Run golden set with all 3 modes
  - Measure completeness improvement
  - Document bottlenecks found
  - Create performance comparison report
  - Commit changes with conventional commits

  🔍 Gate 7a Checklist (End of Day 2)

  - Verbosity modes pass schema validation
  - EXPLAIN guard & timeouts active (1500ms)
  - Golden set completeness ≥30% improvement over snippet
  - P95 latency: snippet≤200ms, full≤350ms, graph≤450ms
  - Zero unbounded queries (all pass EXPLAIN guard)
  - Provenance: 100% of assertions traceable to section IDs
  - Feature flags documented and working
  - Metrics collecting successfully
  - All tests passing
  - Performance report generated

  🚀 DECISION POINT #1
  - IF completeness ≥80% AND P95 targets met → STOP HERE (Week 1 Victory!)
  - IF completeness ≥30% but <80% → Continue to Phase 7b
  - IF completeness <30% → Debug root cause before continuing

  ---
  Phase 7b: traverse_relationships Tool (Days 3-4)

  Day 3 (Wednesday) - TraversalService Core

  - Create src/services/traversal/ directory structure
  - Create traversal/init.py
  - Create traversal/bfs.py with bounded BFS implementation
  - Implement fixed depth traversal (max_depth=3)
  - Implement server-enforced caps (MAX_NODES=100)
  - Add whitelist validation for relationships
  - Add whitelist validation for labels (Section|Entity)
  - Implement deterministic ordering (depth, type, id)
  - Add ID-based deduplication logic
  - Create traversal/service.py orchestrator
  - Add circuit breaker pattern (60s disable after 2 timeouts)
  - Implement hard timeout (2000ms)
  - Add node count monitoring
  - Write unit tests for BFS logic
  - Test depth limits and node caps

  Day 4 (Thursday) - MCP Integration & Testing

  - Register traverse_relationships in stdio_server.py
  - Define MCP tool schema (frozen from 7a)
  - Add parameters: seed_ids, max_depth, rel_types
  - Implement response format: nodes + edges + budget + explain
  - Wire tool through routers.py
  - Add validation in validation.py
  - Test traverse from known section ID
  - Verify returns ≥1 relevant entity + ≥2 sections
  - Measure response time for depth=2
  - Measure response time for depth=3
  - Test circuit breaker (force 2 timeouts)
  - Verify 60s disable window activates
  - Run traverse on golden set seeds
  - Document traversal patterns found
  - Create Gate 7b performance report

  🔍 Gate 7b Checklist (End of Day 4)

  - traverse_relationships schema frozen & documented
  - BFS caps enforced (MAX_NODES=100, MAX_DEPTH=3)
  - Deterministic ordering verified (stable sort)
  - Circuit breaker tested (60s disable after 2 timeouts)
  - Depth=2 traversal returns ≥1 entity + ≥2 sections
  - P95 depth=2 ≤350ms
  - P95 depth=3 ≤600ms
  - Multi-turn exploration works in Claude Desktop
  - Schema matches 7a graph output exactly
  - All tests passing

  Combined 7a+7b Assessment
  - Does graph exploration + full text solve the problem?
  - IF YES (completeness ≥60%) → Continue to 7c
  - IF NO → Continue to 7c anyway (need baselines)

  ---
  Phase 7c: Performance Baselines (Day 5)

  Day 5 (Friday) - Metrics & Decision

  - Publish metrics already collecting since Day 1
  - Create scripts/perf/test_verbosity_baseline.py
  - Run 30 queries × 3 modes test suite
  - Generate P50/P95/P99 for each mode
  - Document latency distribution
  - Identify bottlenecks (driver, Cypher, vector search, I/O)
  - Create CSV report: /reports/phase-7/performance.csv
  - Create Markdown summary: /reports/phase-7/summary.md
  - Calculate cost per query (even if $0)
  - Document estimated token/compute costs
  - Review Gate 7a results
  - Review Gate 7b results
  - Consolidate all Phase 7 artifacts
  - Make Go/No-Go decision
  - Plan Week 2 based on results

  🚀 DECISION POINT #2 - End of Week 1
  - IF completeness ≥80% → SHIP Phase 7 only (Week 1 Victory!)
  - IF completeness ≥60% → Continue but consider reduced scope
  - IF completeness <60% → Must continue to Phase 8

  ---
  📅 WEEK 2: Intelligence Layer

  Phase 8a: Provider Abstraction (Monday-Tuesday)

  Monday - Provider Interfaces

  - Create src/providers/ directory structure
  - Create providers/init.py
  - Create providers/base/embedding.py with EmbeddingProvider protocol
  - Create providers/base/reranker.py with Reranker protocol
  - Define embed() method signature
  - Define rerank() method signature
  - Create providers/implementations/openai.py (current baseline)
  - Implement OpenAI embedding with existing logic
  - Create providers/implementations/huggingface.py (fallback)
  - Implement HuggingFace local embeddings
  - Create config/providers.yaml configuration
  - Add default provider settings
  - Add fallback chain configuration
  - Add timeout/retry policies

  Tuesday - Caching & Testing

  - Implement Redis caching layer
  - Use SHA256(normalize(text)+model_version) for keys
  - Set TTL: 7 days for embeddings, 1 day for reranking
  - Add cache hit/miss metrics
  - Create A/B testing framework
  - Add feature flag: PROVIDER_EXPERIMENT_ENABLED
  - Implement traffic split configuration
  - Add comparison metrics collection
  - Create fallback routing table (jina→openai→hf)
  - Implement per-provider circuit breakers
  - Add telemetry parity (same metrics for all providers)
  - Run same 30 test queries with both providers
  - Compare relevance scores, latency, cost
  - Verify fallback chain works on timeout
  - Test cache warmup process

  🔍 Gate 8a Checklist

  - Provider interfaces defined and documented
  - OpenAI baseline implementation working
  - HuggingFace fallback operational
  - Redis cache operational
  - Cache hit rate ≥60% on second run
  - Fallback chain tested (timeout → fallback)
  - A/B harness results documented
  - Cost/latency/quality metrics compared
  - All providers have identical metric dimensions
  - Circuit breakers working independently

  ---
  Phase 8b: Jina Integration with Dual-Index (Wednesday-Thursday)

  Wednesday - Jina Provider & Dual-Index Setup

  - Create providers/implementations/jina.py
  - Implement jina-embeddings-v3 (768D)
  - Implement jina-reranker-v3
  - Add error handling for Jina API
  - Add circuit breaker for Jina
  - Create parallel vector fields in schema
  - Add docs_embed_openai field (384D)
  - Add docs_embed_jina field (768D)
  - Build separate HNSW index for OpenAI
  - Build separate HNSW index for Jina
  - Implement routing logic by provider flag
  - Add version tracking fields
  - Store embedding_provider per vector
  - Store embedding_model per vector
  - Store embedding_version per vector

  Thursday - Migration & Testing

  - Create backfill monitoring dashboard
  - Implement rolling batch job for backfill
  - Start dual-write process
  - Monitor backfill progress
  - Verify 80% threshold before switching
  - Create A/B test with 10% traffic
  - Monitor Jina performance
  - Increase to 50% traffic if stable
  - Document rollback procedure
  - Test rollback via config switch
  - Implement batch embedding requests
  - Add connection pooling
  - Make async where beneficial
  - Run NDCG comparison on golden set
  - Measure P95 embedding latency

  🔍 Gate 8b Checklist

  - Dual indexes online (docs_embed_openai, docs_embed_jina)
  - Routing logic verified and working
  - Backfill ≥80% complete
  - NDCG@10 ≥0.75 (improvement ≥15%)
  - P95 embedding latency ≤150ms (batch size 16)
  - Provider errors <0.5% after warmup
  - Version tracking operational
  - Canary tested (10%→50%)
  - Rollback procedure documented and tested
  - Cost within budget constraints

  🚀 DECISION POINT #3
  - IF Jina NDCG ≥15% better → Make Jina default
  - IF Jina comparable → Keep OpenAI (simpler)
  - IF Jina worse → Keep OpenAI, investigate why

  ---
  📅 WEEK 3: Optimization

  Phase 8c: Frontier Gating (Monday-Tuesday)

  Monday - Frontier Scoring Implementation

  - Create src/services/traversal/frontier.py
  - Implement scoring function: αsemantic + βmetadata - γ*novelty
  - Add delta threshold logic (only expand if improvement > τ)
  - Implement top-k selection per depth level
  - Add expected gain calculation
  - Implement novelty penalty (MMR-style)
  - Add budget-aware expansion logic
  - Define stop conditions (converged, timeout, budget)
  - Create config/traversal.yaml
  - Set weight parameters (α, β, γ)
  - Set delta threshold (τ = 0.05 default)
  - Set top-k per level (20 default)
  - Add novelty decay factor

  Tuesday - Testing & Optimization

  - Implement budget-aware scoring (remaining budget consideration)
  - Add cold-start fallback (BFS if embeddings unavailable)
  - Set explain.reason='fallback_bfs' for fallback
  - Add audit trail logging
  - Log pruned_count
  - Log expanded_count
  - Log top 3 'would-be' expansions with scores
  - Test frontier gating on golden set
  - Measure node expansion reduction
  - Verify NDCG maintained
  - Test P95 traversal depth=3
  - Tune parameters if needed
  - Document optimal settings
  - Create explainability report
  - Verify scoring breakdown works

  🔍 Gate 8c Checklist

  - Frontier gating operational
  - Node expansions reduced ≥40%
  - NDCG maintained within 2% of baseline
  - P95 traversal depth=3 ≤300ms with gating on
  - Explainability fields populated
  - Cold-start fallback tested
  - Audit trail logging working
  - Parameter tuning documented
  - Budget-aware stopping working
  - All stop conditions tested

  ---
  Phase 8d: Safety Hardening (Wednesday-Thursday)

  Wednesday - EXPLAIN Validation

  - Create src/neo/validation.py
  - Implement EXPLAIN plan analysis
  - Add rejection for Expand(All) patterns
  - Add estimated row/cost thresholds
  - Add label scan detection
  - Implement label-whitelist enforcement (Section|Entity only)
  - Add depth syntax checks (reject *.. unbounded)
  - Implement driver-level timeout
  - Add query-level timeout
  - Add connection pool timeout
  - Implement cascading timeout strategy
  - Cache EXPLAIN results by query template
  - Monitor EXPLAIN latency overhead

  Thursday - Rate Limiting & Degradation

  - Implement per-IP rate limits for graph mode
  - Add per-user budgets for traverse tool
  - Create budget tracking system
  - Implement graceful degradation to snippet mode
  - Set diagnostics.degraded=true when degraded
  - Add audit logging for all denials
  - Log user/session budget consumption
  - Test rate limiting with concurrent requests
  - Verify degradation path works
  - Test budget enforcement
  - Document rate limit configuration
  - Create monitoring dashboard for denials
  - Run full safety test suite
  - Verify 100% queries bounded

  🔍 Gate 8d Checklist

  - 100% of dynamic Cypher calls pass EXPLAIN guard
  - Rate-limit denials <1% of requests
  - Degradation to snippet mode tested
  - Budget tracking operational
  - All denials logged with user/session info
  - Timeout cascade working (driver→query→pool)
  - Label whitelist enforced
  - Depth syntax validated
  - EXPLAIN cache operational
  - Safety dashboard created

  🚀 DECISION POINT #4 - End of Week 3
  - IF retrieval quality acceptable → SHIP Phase 7+8 (3 weeks total)
  - IF quality still insufficient → Continue to Phase 9a
  - Document what's still missing for decision

  ---
  📅 WEEK 4: Schema Evaluation (OPTIONAL)

  Phase 9a: DocRAG Schema Bridge

  Monday-Wednesday - Schema Prototype

  - Check if Phase 8 solved quality issues
  - IF NOT, create test environment for schema
  - Create migrations/add_chunk_labels.cypher
  - Add :Chunk label to Section nodes (dual-label)
  - Make migration idempotent
  - Create dry_run=true mode
  - Test migration counts only (no execution)
  - Create new relationship types plan
  - Keep existing: MENTIONS, HAS_SECTION
  - Add: HAS_CHUNK, HAS_TOPIC, RELATED_TO
  - Design bridge edges (read-only to ops graph)
  - Create Session/Query/Answer node types
  - Add TTL for ephemeral nodes
  - Create cleanup job (daily cron)

  Thursday-Friday - Testing & Comparison

  - Run migration in test environment
  - Update query templates for DocRAG
  - Maintain backward compatibility
  - Create canary queries (old vs new)
  - Run A/B test on golden set
  - Measure retrieval improvement
  - Check P95 latency regression
  - Test provenance tracking (chunk→claim)
  - Create rollback script
  - Test rollback procedure
  - Document migration process
  - Make Go/No-Go decision

  🚀 DECISION POINT #5
  - IF schema improves retrieval >10% → Implement in production
  - IF improvement <10% → Skip to Phase 9b
  - Document decision rationale

  ---
  📅 WEEK 5: Production Readiness

  Phase 9b: Observability & Production Readiness

  Monday-Tuesday - Comprehensive Metrics

  - Add all Phase 7 spec counters/histograms
  - Create custom Grafana dashboards
  - Configure alert rules for anomalies
  - Set up SLO monitoring (P50<200ms, P95<500ms)
  - Create unified trace spans
  - Add spans: embed, vector_search, rerank
  - Add spans: graph_expand, explain_guard, neo4j_run
  - Implement baggage propagation
  - Set sampling strategy (1% baseline, 100% on error)
  - Configure Prometheus exporters
  - Test metric collection
  - Verify dashboard visibility

  Wednesday-Thursday - Load Testing

  - Set up load test environment
  - Configure 100 concurrent users simulation
  - Create mixed workload (70% snippet, 20% full, 10% graph)
  - Run load test for 1 hour
  - Identify bottlenecks
  - Document capacity limits
  - Create capacity planning report
  - Test auto-scaling if applicable
  - Measure resource utilization
  - Check for memory leaks
  - Verify connection pooling

  Friday - Documentation

  - Write API documentation
  - Create operational runbook
  - Write troubleshooting guide
  - Create performance tuning guide
  - Document all feature flags
  - Create architecture diagrams
  - Write deployment guide
  - Document rollback procedures
  - Create incident response playbook

  ---
  📅 WEEK 6: Production Rollout

  Phase 9c: Production Rollout

  Monday-Tuesday - Pre-Production Checklist

  - Configure all feature flags (default-safe)
  - Set PHASE7_VERBOSITY_ENABLED=true
  - Set PHASE8_PROVIDERS_ENABLED=false
  - Set PHASE8_FRONTIER_ENABLED=false
  - Set PHASE9_DOCRAG_SCHEMA=false (if applicable)
  - Create runtime config endpoint
  - Implement error budget SLO
  - Configure auto-reduction on budget burn >25%
  - Test all rollback procedures
  - Verify monitoring dashboards
  - Check alert configurations
  - Review security (no credential leaks)
  - Complete team training
  - Define support escalation path

  Wednesday - Canary Deployment

  - Deploy to 10% of traffic
  - Monitor for 24 hours
  - Check error rates (<0.1%)
  - Verify P95 latency within targets
  - Monitor cost metrics
  - If stable, increase to 50%
  - Monitor for 48 hours
  - Check all SLOs

  Thursday-Friday - Full Rollout

  - If 50% stable, go to 100%
  - Monitor for 1 week
  - Daily SLO reviews
  - Check for P0/P1 incidents
  - Monitor user satisfaction
  - Verify rollback capability (<5 min)
  - Send customer communication
  - Update on-call schedule
  - Prepare post-mortem process
  - Document lessons learned

  🔍 Final Production Gate Checklist

  - All SLOs met for 7 consecutive days
  - No P0/P1 incidents
  - Error rate consistently <0.1%
  - P95 latency within all targets
  - Cost tracking operational and within budget
  - All monitoring green
  - Rollback tested in production
  - Documentation complete
  - Team trained on operations
  - Customer communications sent

  🎯 FINAL SHIP DECISION
  - All gates passed → Mark as GA (Generally Available)
  - Celebrate successful 6-week implementation!
  - Schedule retrospective
  - Plan next phase enhancements

  ---
  📊 Task Count Summary

  - Phase 7a: 30 tasks + 10 gate checks
  - Phase 7b: 30 tasks + 10 gate checks
  - Phase 7c: 15 tasks + 3 decision checks
  - Phase 8a: 29 tasks + 10 gate checks
  - Phase 8b: 30 tasks + 10 gate checks + 3 decisions
  - Phase 8c: 28 tasks + 10 gate checks
  - Phase 8d: 27 tasks + 10 gate checks + 3 decisions
  - Phase 9a: 27 tasks + 3 decisions
  - Phase 9b: 32 tasks
  - Phase 9c: 32 tasks + 10 final gate checks + 4 ship decisions

  Total: ~280 implementation tasks + ~70 validation checks + 5 major decision points

  ---
  ★ Insight ─────────────────────────────────────
  1. The dual-index tasks in Phase 8b are the riskiest - 15 specific tasks that must be executed in order to prevent embedding dimension mismatch failures.

  2. Day 1 has only 15 tasks but they're foundational - These safety and measurement tasks prevent weeks of debugging later.

  3. The golden set definition (Task #6 on Day 1) influences everything - All subsequent validation depends on these 20 queries being representative of
  real usage.
  ─────────────────────────────────────────────────

  This complete list is stored in memory as golden-plan-task-list-20251020 and can be used to track daily progress through the implementation.
