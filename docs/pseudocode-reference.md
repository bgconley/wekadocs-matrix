# 4) `/docs/pseudocode-reference.md` — **Pseudocode Reference (v2, canonical)**

This is the consolidated pseudocode for **every phase/task (1.1 → 5.4)**: templates‑first NL→Cypher, plan‑gated validator, hybrid retrieval, provenance‑first ingestion, configurable vectors, versioned caches, OTel, chaos tests, DR drills, and phase gates.

# WekaDocs GraphRAG MCP — **Pseudocode Reference (v2)**

## Global conventions used in all phases

```pseudocode
CONFIG := load_yaml("config/*.yaml")  // embedding, search.vector.primary, limits, security, etc.

ID := {
  // Deterministic ids: stable across re-ingest
  node(section)      -> sha256(source_uri + "#" + anchor + normalize(text))
  node(entity)       -> sha256(namespace + ":" + canonical_name)
  rel(u,v,rel_type)  -> sha256(u.id + "->" + rel_type + "->" + v.id + ("@" + source_section_id?))
}

VERSIONS := {
  schema_version       := read_or_init_schema_version()
  embedding_version    := CONFIG.embedding.version
}

CACHE_KEY(prefix, params) :=
  (VERSIONS.schema_version + ":" + VERSIONS.embedding_version + ":" + prefix + ":" + hash(params))

EMBEDDER := create_embedder(CONFIG.embedding.model_name, dims=CONFIG.embedding.dims)

VECTOR_SOT :=
  if CONFIG.search.vector.primary == "qdrant" then QDRANT else NEO4J_VECTOR

OTEL := init_tracer(service="weka-mcp")
```

---

## Phase 1 – Core Infrastructure

### **Task 1.1 – Docker environment setup**

```pseudocode
procedure compose_stack()
  write docker-compose.yml with services: mcp-server, neo4j, qdrant (optional), redis, ingestion-worker
  for each service:
    set healthcheck, CPU/mem limits, volumes, restart: unless-stopped
  set secrets via docker secrets or k8s secrets (NOT plain env)
  add network "weka-net"
  ensure neo4j heap/pagecache not duplicated across env vars

procedure verify_stack()
  run "docker compose up -d"
  wait_until_healthy(all_services)
  assert tcp_connect("neo4j:7687"), http_ok("qdrant:6333/health"), redis_ping("redis:6379")
  restart_all()
  assert data_persisted()

compose_stack()
verify_stack()
```



---

### **Task 1.2 – MCP server foundation**

```pseudocode
procedure start_mcp_server()
  app := FastAPI()
  attach_middleware(request_id, structured_logging, OTEL_tracing)
  pools := {
    neo4j := init_neo4j_pool(CONFIG.neo4j),
    qdrant := init_qdrant_client(CONFIG.qdrant) if VECTOR_SOT==QDRANT or dual_write,
    redis := init_redis(CONFIG.redis)
  }

  register_routes(app):
    GET /health -> return {status:"ok"}
    GET /ready  -> check pools healthy
    GET /metrics -> prometheus_exporter

    POST /mcp:
      request := parse_mcp_request()
      switch request.method:
        case "initialize": return mcp_initialize()
        case "tools/list": return list_tools()
        case "tools/call": return call_tool(request.params)
        case "completion": return completion(request.params)
        default: error(UnknownMethod)

procedure mcp_initialize()
  ensure_protocol_compat(params.protocolVersion)
  await pools.neo4j.ping()
  return {capabilities:{tools:true,sampling:false,resources:false}}

procedure list_tools()
  return registry.describe_all_tools()

procedure call_tool(args)
  require_auth_and_rate_limit()
  key := CACHE_KEY("tool", args)
  if redis.get(key) -> cached: return cached
  result := registry.execute(args.name, args.arguments, pools)
  redis.setex(key, ttl=3600, value=result)
  return result

procedure graceful_shutdown()
  close_all(pools)
```



---

### **Task 1.3 – Database schema initialization**

```pseudocode
procedure create_graph_schema()
  // Core provenance-first nodes
  run_cypher("""
    CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
    CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section)  REQUIRE s.id IS UNIQUE;
    CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.document_id, s.anchor);
  """)

  // Domain nodes (Command, Configuration, Procedure, Error, Concept, Example, Step, Parameter, Component)
  for label in DOMAIN_LABELS:
    run_cypher("CREATE CONSTRAINT IF NOT EXISTS FOR (n:"+label+") REQUIRE n.id IS UNIQUE;")
    for idx_prop in INDEXED_PROPS[label]:
      run_cypher("CREATE INDEX IF NOT EXISTS FOR (n:"+label+") ON (n."+idx_prop+");")

  // Vector indexes: use configured dims/similarity; do NOT hard-code 384
  for vec_label in VECTORIZED_LABELS:  // e.g., Section + select Entities
    call_neo4j_vector_create(
      name=vec_label+"_embeddings",
      label=vec_label,
      property="vector_embedding",
      dims=CONFIG.embedding.dims,
      similarity=CONFIG.embedding.similarity
    )

  // schema_version node
  ensure_singleton_schema_version(VERSIONS.schema_version)
```



---

### **Task 1.4 – Security layer**

```pseudocode
middleware require_auth_and_rate_limit()
  jwt := parse_and_verify_JWT(request.headers.Authorization)
  bucket := "ratelimit:"+jwt.sub
  if not token_bucket_allow(bucket, burst=CONFIG.security.rate_limit.burst, rate=CONFIG.security.rate_limit.per_minute):
    return http_429()

middleware audit_log()
  write_audit({client:jwt.sub, method:request.method, path:request.path, correlation_id})

validator := CypherValidator(
  regex_guards = {...},           // prohibit DELETE/SET/MERGE/DROP, etc.
  depth_regex = r"\*(\d+)\.\.(\d+)",   // correct pattern
  maxDepth = CONFIG.limits.max_depth,
  maxRows  = CONFIG.limits.max_results,
  timeoutMs= CONFIG.limits.cypher_timeout_ms
)

procedure execute_safe_cypher(query, params)
  validated := validator.validate(query, params)       // structure + params
  plan := neo4j.EXPLAIN(validated.query, validated.params)
  if plan.estimatedExpansions > THRESHOLDS.expansions or plan.labelScans > THRESHOLDS.scans:
    raise ValidationError("Unsafe plan")
  return neo4j.RUN(validated.query, validated.params, timeout=CONFIG.limits.cypher_timeout_ms)
```



---

## Phase 2 – Query Processing Engine

### **Task 2.1 – NL → Cypher translation**

```pseudocode
procedure plan_query(nl_query)
  intent := classify_intent(nl_query)          // search | traverse | compare | troubleshoot | explain
  entities := link_entities(nl_query)          // resolve to node labels/ids when possible

  if TEMPLATE_LIBRARY.has(intent, entities):
    cypher, params := TEMPLATE_LIBRARY.render(intent, entities, nl_query)
  else:
    draft := LLM_propose_cypher(nl_query, entities)    // optional; never executed directly
    cypher, params := normalize_and_parameterize(draft)

  return {intent, cypher, params}

procedure normalize_and_parameterize(draft)
  // ensure all literals become $params; inject LIMIT early; cap variable-length paths
  params := extract_literals(draft)
  cypher := rewrite_literals_to_params(draft, params)
  cypher := enforce_limits_and_depth(cypher, CONFIG.limits)
  return cypher, params
```



---

### **Task 2.2 – Cypher validation system**

```pseudocode
class CypherValidator:
  init(regex_guards, depth_regex, maxDepth, maxRows, timeoutMs)
  method validate(query, params):
    q := strip_comments_and_normalize_whitespace(query)
    assert_no_forbidden_keywords(q, ["DELETE","MERGE","SET","DROP","GRANT","REVOKE"])
    assert_all_user_inputs_are_params(q, params)
    depth := parse_max_variable_length(depth_regex, q) default 1
    if depth > maxDepth: raise ValidationError("DepthExceeded")
    if not has_limit(q): q := append_limit(q, maxRows)
    return {query:q, params}

procedure run_validated(cypher, params)
  v := validator.validate(cypher, params)
  plan := neo4j.EXPLAIN(v.query, v.params)
  if plan.nodeByLabelScans > ALLOWED or plan.expandAll > ALLOWED:
    raise ValidationError("PlanTooExpensive")
  return neo4j.RUN(v.query, v.params, timeout=CONFIG.limits.cypher_timeout_ms)
```

*(Fixes traversal regex bug and adds EXPLAIN plan checks.)*

---

### **Task 2.3 – Hybrid search**

```pseudocode
procedure hybrid_search(text_query, filters, K=20)
  // Step 1: semantic seed on Sections (+ optional Entities)
  qvec := EMBEDDER.encode(text_query)
  seeds := if VECTOR_SOT==QDRANT then
             qdrant.topK(collection="sections", vector=qvec, K=K, payload_filters=filters)
           else
             neo4j.vector_query(label="Section", property="vector_embedding", vector=qvec, K=K, filters)

  // Step 2: controlled graph expansion (1..H hops; typed edges only)
  H := clamp(CONFIG.search.max_hops, 1, 2)
  expanded := []
  for seed in seeds.topN( min(10, K) ):
    expanded += run_validated("""
      MATCH (s:Section {id: $sid})
      OPTIONAL MATCH path=(s)-[:MENTIONS|:CONTAINS_STEP|:HAS_PARAMETER|:REQUIRES|:AFFECTS*1..$H]->(n)
      RETURN DISTINCT n, length(path) AS dist
      ORDER BY dist ASC LIMIT $L
    """, {sid:seed.section_id, H: H, L: 50})

  // Step 3: optional connecting paths between top entities
  bridges := run_validated("""
      UNWIND $ids AS a
      UNWIND $ids AS b
      WITH DISTINCT a,b WHERE a<b
      MATCH p=shortestPath( (x {id:a})-[*..3]-(y {id:b}) )
      RETURN nodes(p) as nodes, length(p) as len LIMIT 30
  """, {ids: seeds.ids.topN(5)})

  // Step 4: rank + dedupe
  results := rank_merge(seeds, expanded, bridges, recency_boost=true)
  return results.topN(K)
```



---

### **Task 2.4 – Response generation**

```pseudocode
procedure build_response(query, intent, retrieval)
  evidence := collect_top_evidence(retrieval, limit=5)   // prefer Section → include anchor + snippet
  answer_md := render_markdown(intent, retrieval, evidence)
  answer_json := {
    answer: extract_structured(intent, retrieval),
    evidence: [{section_id, node_id?, path?}...],
    confidence: estimate_confidence(retrieval, evidence),
    diagnostics: {ranking_features, timing}
  }
  return {answer_markdown: answer_md, answer_json}

procedure render_markdown(intent, retrieval, evidence)
  switch intent:
    case "procedure": return md_procedure(retrieval.steps, retrieval.proc_meta)
    case "troubleshoot": return md_troubleshoot(retrieval.errors, retrieval.procedures)
    case "compare": return md_compare(retrieval.pairs)
    default: return md_factual_topK(retrieval.topK, evidence)

procedure estimate_confidence(retrieval, evidence)
  return bounded( 0.3*top_semantic_score + 0.2*evidence_strength + 0.2*coverage + 0.3*path_coherence )
```

*(Adds dual outputs + evidence/confidence promised in spec.)*

---

## Phase 3 – Ingestion Pipeline

### **Task 3.1 – Multi‑format parser**

```pseudocode
procedure parse_document(source_uri, raw_bytes)
  fmt := detect_format(source_uri, raw_bytes)   // markdown | html | notion
  text := decode(raw_bytes)
  doc := { id: sha256(source_uri), source_uri, source_type:fmt }

  sections := []
  if fmt == "markdown":
    ast := md_to_ast(text)
    sections := walk_headings_and_blocks(ast) -> list of {
      level, title, anchor, order, text, code_blocks[], tables[]
    }
  else if fmt == "html":
    dom := html_parse(text)
    sections := extract_sections_from_html(dom)
  else if fmt == "notion":
    blocks := notion_fetch(page_id=source_uri)
    sections := notion_blocks_to_sections(blocks)

  for s in sections:
    s.id := ID.node(section=s)        // sha256(source_uri + "#" + anchor + normalized_text)
    s.tokens := count_tokens(s.text)
    s.checksum := sha256(s.text)

  return { Document: doc, Sections: sections }
```



---

### **Task 3.2 – Entity extraction**

```pseudocode
procedure extract_entities(sections)
  entities := {}    // keyed by canonical id
  mentions := []    // (Section -> Entity) with {confidence, spans}

  for section in sections:
    // Rules + light NLP
    commands := regex_cli_commands(section.text, section.code_blocks)
    configs  := regex_config_params(section.text)
    errors   := regex_error_codes(section.text)
    concepts := nlp_extract_concepts(section.text)
    procedures, steps := detect_procedure_and_steps(section)

    for each detected entity e:
      eid := ID.node(entity=e)
      upsert entities[eid] := canonicalize(e)
      mentions.append( {section_id: section.id, entity_id: eid, spans, confidence} )

  return {entities, mentions}
```

*(Simple, high‑precision first pass; provenance kept via MENTIONS.)*

---

### **Task 3.3 – Graph construction**

```pseudocode
procedure upsert_graph(doc, sections, entities, mentions)
  tx := neo4j.begin()

  MERGE (d:Document {id: doc.id})
    SET d += {source_uri, source_type, title, version, checksum, last_edited, updated_at:now()}

  for s in sections batch 1000:
    MERGE (sec:Section {id:s.id})
      SET sec += {document_id:doc.id, level, title, anchor, order, text, tokens, checksum, updated_at:now()}
    MERGE (d)-[:HAS_SECTION {order:s.order}]->(sec)

  for e in entities.values batch 1000:
    MERGE (n:LabelFor(e) {id:e.id})
      SET n += e.properties, n.updated_at = now()

  for m in mentions batch 2000:
    MATCH (sec:Section {id:m.section_id}), (n {id:m.entity_id})
    MERGE (sec)-[r:MENTIONS {entity_id:m.entity_id}]->(n)
      SET r += {confidence:m.confidence, start:m.spans.start, end:m.spans.end, source_section_id:sec.id, updated_at:now()}

  tx.commit()

  // Embeddings + vector upsert (primary store)
  for sec in sections if changed(sec.checksum):
    vec := EMBEDDER.encode( title_trail(sec) + "\n\n" + sec.text )
    if VECTOR_SOT==QDRANT:
      qdrant.upsert(collection="sections",
                    id=sec.id, vector=vec,
                    payload={node_id:sec.id, node_label:"Section", document_id:doc.id, updated_at:now(), embedding_version:VERSIONS.embedding_version})
    else:
      neo4j.run("MATCH (s:Section {id:$id}) SET s.vector_embedding=$vec, s.embedding_version=$v", {id:sec.id, vec:vec, v:VERSIONS.embedding_version})
```

*(Idempotent MERGEs; provenance; configurable vectors; dual‑store optional via feature flag.)*

---

### **Task 3.4 – Incremental update**

```pseudocode
procedure incremental_ingest(document_uri)
  parsed := parse_document(document_uri, fetch(document_uri))
  diffs := diff_against_graph(parsed.Sections by checksum)

  // Stage changes
  for sec in diffs.added_or_modified:
    CREATE (:Section_Staged { ...same props..., staged:true })

  // Swap labels atomically
  tx := neo4j.begin()
    for each staged in Section_Staged:
      DETACH DELETE old Section with same id   // or relabel: (old:Section)->(old:Section_Old)
      SET staged:Section REMOVE staged:Section_Staged
  tx.commit()

  // Re-embed changed & adjacent entities only
  enqueue_reembed(diffs.affected_ids)

procedure nightly_reconciliation()
  // Compare graph Sections vs vector store by embedding_version
  graph_ids := neo4j.list_ids(label="Section", where=embedding_version==VERSIONS.embedding_version)
  vector_ids := VECTOR_SOT.list_ids(collection="sections", where=embedding_version==VERSIONS.embedding_version)
  missing := setdiff(graph_ids, vector_ids)
  if missing.size/graph_ids.size > 0.005: alert("drift")
  for id in missing: reembed_and_upsert(id)
```

*(Implements safe staged updates + drift repair.)*

---

## Phase 4 – Advanced Query Features

### **Task 4.1 – Complex query patterns**

```pseudocode
TEMPLATES.advanced := {
  dependency_chain(component):
    """
    MATCH path=(c:Component {name:$component})-[:DEPENDS_ON*1..5]->(t)
    RETURN path, length(path) AS depth ORDER BY depth
    """,

  impact_assessment(config_name):
    """
    MATCH (cfg:Configuration {name:$config_name})
    MATCH (cfg)-[:AFFECTS*1..3]->(affected)
    OPTIONAL MATCH (affected)-[:CRITICAL_FOR]->(svc)
    RETURN cfg, affected, svc, CASE WHEN svc IS NULL THEN 'NORMAL' ELSE 'CRITICAL' END AS impact
    """,

  troubleshooting(error_code):
    """
    MATCH (e:Error {code:$error_code})
    MATCH p=(e)<-[:RESOLVES]-(proc:Procedure)-[:CONTAINS_STEP*]->(step:Step)
    OPTIONAL MATCH (step)-[:EXECUTES]->(cmd:Command)
    RETURN e, proc, step, cmd ORDER BY step.order
    """
}

procedure execute_template(name, params)
  cypher := TEMPLATES.advanced[name]
  return run_validated(cypher, params)
```

*(Codifies complex but safe, pre‑approved patterns.)*

---

### **Task 4.2 – Query optimization**

```pseudocode
procedure optimize_hot_queries()
  slow := metrics.top_slowest_cypher(N=50)
  for q in slow:
    plan := neo4j.EXPLAIN(q)
    suggestions := derive_index_and_rewrite_hints(plan)
    if suggestions.indexes:
      open_PR("add_indexes.cypher", suggestions.indexes)
    if suggestions.rewrite:
      patch_template(q.template_id, suggestions.rewrite)

procedure plan_cache()
  fingerprint := hash(template_id + sorted(param_names))
  if not cache.exists("plan:"+fingerprint):
    compiled := neo4j.PREPARE(template)
    cache.set("plan:"+fingerprint, compiled)
```

*(Formalizes ongoing plan analysis + caching.)*

---

### **Task 4.3 – Caching & performance**

```pseudocode
L1 := in_proc_cache(max_bytes=100MB)
L2 := redis

procedure cached(key_prefix, params, compute_fn)
  key := CACHE_KEY(key_prefix, params)
  if L1.get(key) -> v: return v
  if L2.get(key) -> v: L1.put(key, v); return v
  v := compute_fn()
  L1.put(key, v); L2.setex(key, ttl=3600, value=v)
  return v

procedure warm_top_intents()
  for intent in analytics.top_intents(N=20):
    cached("intent_warm", {intent}, () => run_intent(intent.sample_query))
```

*(Adds version‑prefixed cache keys to avoid stale data after model/schema change.)*

---

### **Task 4.4 – Learning & adaptation**

```pseudocode
procedure log_feedback(query_id, rating, notes, missed_entities[])
  store({query_id, rating, notes, missed_entities, ts:now()})

procedure weekly_model_update()
  usage := fetch_usage_logs()
  patterns := mine_common_query_patterns(usage)
  for p in patterns.new_templates:
    TEMPLATE_LIBRARY.add(p.intent, p.template)
  ranking_weights := fit_weights_to_feedback(usage, target_metric="NDCG")
  save_ranking_weights(ranking_weights)
```

*(Closes the loop for continual improvement.)*

---

## Phase 5 – Integration & Deployment

### **Task 5.1 – External systems**

```pseudocode
// Notion (example)
procedure notion_sync_loop()
  while true:
    events := notion.poll_changes(since=state.last_cursor)
    for ev in events:
      enqueue_ingest(ev.page_url)
    state.last_cursor := events.next_cursor
    sleep(CONFIG.connectors.notion.poll_interval)

// Webhooks handler
POST /webhooks/notion:
  if verify_signature(request): enqueue_ingest(request.body.page_url)

// Backpressure & circuit breaker
if neo4j.cpu > 0.8 or qdrant.p95 > 200ms: reduce_ingestion_concurrency()
```



---

### **Task 5.2 – Monitoring & observability**

```pseudocode
metrics := {
  mcp_latency_histogram, cypher_timing, cache_hit_rate, vector_latency,
  ingestion_queue_lag, reconciliation_drift
}

tracing := OTEL.enable_in(FastAPI, neo4j_driver, qdrant_client)
dashboards := grafana.import("wekadocs.json")

alerts := {
  "p99_latency_gt_2s" -> page_oncall(),
  "error_rate_gt_1pct"-> page_oncall(),
  "reconciliation_drift_gt_0.5pct" -> ticket_to_data_eng()
}
```

*(Meets SLOs and visibility stated in spec.)*

---

### **Task 5.3 – Testing framework**

```pseudocode
suite unit_tests():
  assert validator.blocks_injection_patterns()
  assert validator.enforces_params_and_limits()
  assert schema_creation_is_idempotent()

suite integration_tests():
  ingest(golden_docs)
  assert deterministic_graph_snapshot()
  assert hybrid_search_returns_expected_nodes()

suite e2e_tests():
  q := "How to resolve error E123?"
  r := mcp_completion(q)
  assert r.answer_json.evidence not empty
  assert 0.0 <= r.answer_json.confidence <= 1.0

suite perf_tests():
  warm_caches()
  run_load(QPS=100) -> assert P95 < 500ms

suite chaos_tests():
  kill_service("qdrant") -> system_degrades_but_serves_graph_only()
  simulate_neo4j_backpressure() -> ingestion_backs_off()
```



---

### **Task 5.4 – Production deployment**

```pseudocode
pipeline CI_CD():
  on push:
    run unit + integration + e2e + security scans
    build image with immutable tag
    publish to registry
    deploy to staging -> run smoke & perf
    canary_prod(5%) for 60m with SLO guards
    if ok: rollout 25%->50%->100%
    else: auto_rollback()

backup_and_DR():
  hourly snapshots neo4j + qdrant; daily full backups
  replicate to secondary region
  quarterly restore_drill(target RTO=1h, RPO=15m)
```

*(Formalizes blue/green + canary + DR targets.)*

---

## Appendix A — Core data model pseudocode (for reference)

```pseudocode
// Nodes
Document {id, source_uri, source_type, title, version, checksum, last_edited}
Section  {id, document_id, level, title, anchor, order, text, tokens, checksum, vector_embedding?, embedding_version?}
Command | Configuration | Procedure | Error | Concept | Example | Step | Parameter | Component
// Common props on domain nodes: {id, name|title|term..., description?, updated_at, vector_embedding?, embedding_version?}

// Relationships (all with provenance)
(:Document)-[:HAS_SECTION{order}]->(:Section)
(:Section)-[:MENTIONS{confidence, start, end, source_section_id}]->(:Entity)
Derived edges w/ provenance: REQUIRES | AFFECTS | RESOLVES | CONTAINS_STEP{order} | EXECUTES | RELATED_TO | HAS_PARAMETER
```



---

## Appendix B — Example MCP tool shapes (thin wrappers)

```pseudocode
tool search_documentation(query, filters):
  results := hybrid_search(query, filters)
  return build_response(query, "search", results)

tool traverse_relationships(start_id, rel_types, max_depth):
  cypher := """
    MATCH (n {id:$start})-[r:%REL_TYPES%*1..$D]->(m)
    RETURN n,r,m LIMIT $L
  """
  cypher := expand_rel_types(cypher, rel_types)
  rows := run_validated(cypher, {start:start_id, D:max_depth, L:CONFIG.limits.max_results})
  return build_response("traverse", "explain", rows)

tool troubleshoot_error(error_code):
  rows := execute_template("troubleshooting", {error_code})
  return build_response("troubleshoot", "troubleshoot", rows)
```

---




---

# 5) PR stubs & file scaffolds (per phase/task)

> Use the script below to scaffold directories & placeholder files.
> Each task has: **code stub**, **test stub (no mocks)**, and **report hook**.

### 5.1 Scaffold script (run once)

```bash
#!/usr/bin/env bash
set -euo pipefail

declare -A TASKS=(
  [p1_t1]="src/platform/compose/ docker/ scripts/test/phase1/"
  [p1_t2]="src/mcp_server/ src/shared/observability/"
  [p1_t3]="scripts/neo4j/ src/shared/schema.py"
  [p1_t4]="src/mcp_server/security/ src/shared/audit/"
  [p2_t1]="src/query/planner.py src/query/templates/"
  [p2_t2]="src/mcp_server/validation.py"
  [p2_t3]="src/query/hybrid_search.py src/query/ranking.py"
  [p2_t4]="src/query/response_builder.py"
  [p3_t1]="src/ingestion/parsers/"
  [p3_t2]="src/ingestion/extract/"
  [p3_t3]="src/ingestion/build_graph.py"
  [p3_t4]="src/ingestion/{incremental.py,reconcile.py}"
  [p4_t1]="src/query/templates/advanced/"
  [p4_t2]="src/ops/optimizer.py"
  [p4_t3]="src/shared/cache.py src/ops/warmers/"
  [p4_t4]="src/learning/"
  [p5_t1]="src/connectors/"
  [p5_t2]="deploy/monitoring/"
  [p5_t3]="tests/ ci/ .github/workflows/"
  [p5_t4]="deploy/k8s/ deploy/helm/ ci/cd/"
)

for t in "${!TASKS[@]}"; do
  IFS=' ' read -r -a paths <<< "${TASKS[$t]}"
  for p in "${paths[@]}"; do mkdir -p "$p"; done
  touch "tests/${t}_test.py" "src/${t}_README.md"
done

mkdir -p reports/phase-{1,2,3,4,5}
echo "Scaffold complete."
```

### 5.2 File stubs (illustrative subset)

* `src/mcp_server/main.py` — MCP entrypoint (tools, health, metrics).
* `src/mcp_server/validation.py` — validator (regex + `EXPLAIN` gates).
* `src/query/planner.py` — templates‑first NL→Cypher planner.
* `src/query/templates/{search.cypher, traverse.cypher, ...}`
* `src/query/hybrid_search.py` — vector+graph retrieval.
* `src/query/ranking.py` — multi‑signal ranker.
* `src/query/response_builder.py` — Markdown + JSON with evidence.
* `src/ingestion/parsers/{markdown.py,html.py,notion.py}`
* `src/ingestion/extract/{commands.py,configs.py,procedures.py,...}`
* `src/ingestion/build_graph.py`, `src/ingestion/{incremental.py,reconcile.py}`
* `scripts/neo4j/create_schema.cypher`
* `src/shared/{schema.py,cache.py,config.py,observability/tracing.py,audit/logger.py}`

### 5.3 Test stubs (NO MOCKS) & reports harness

* `tests/p1_t1_compose_test.py` — brings up compose, asserts health; writes JUnit & `summary.json`.
* `tests/p2_t2_validator_negative_test.py` — executes malicious queries vs live Neo4j; expects blocks.
* `tests/p2_t3_hybrid_perf_test.py` — Locust/k6 wrapper to produce P95 metrics CSV.
* `tests/p3_t3_idempotency_test.py` — ingest twice; assert stable counts.
* `tests/p4_t3_cache_rotation_test.py` — rotate embedding version; assertions on cache invalidation.
* `tests/p5_t3_chaos_test.py` — kill vector service; verify degraded operation.

**Helper:** `/scripts/test/run_phase.sh`
Runs tests for a phase, emits `/reports/phase-N/junit.xml` & `/reports/phase-N/summary.json`:

```bash
#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:?phase number required (1..5)}"
pytest -q --maxfail=1 --junitxml="reports/phase-${PHASE}/junit.xml" \
  -k "p${PHASE}_" \
  | tee "reports/phase-${PHASE}/pytest.out"

python scripts/test/summarize.py --phase "${PHASE}" \
  --junit "reports/phase-${PHASE}/junit.xml" \
  --out "reports/phase-${PHASE}/summary.json"
```

**`scripts/test/summarize.py` (schema writer skeleton):**

```python
import json, sys, argparse, time, subprocess, hashlib, os
p=argparse.ArgumentParser(); p.add_argument("--phase"); p.add_argument("--junit"); p.add_argument("--out"); a=p.parse_args()
summary={"phase":a.phase,"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
         "results":[], "metrics":{}, "artifacts":["junit.xml"]}
# minimal example: parse JUnit for pass/fail counts (left as exercise to implement fully)
with open(a.out,"w") as f: json.dump(summary,f,indent=2)
print(f"Wrote {a.out}")
```

### 5.4 CI workflow & PR template

**`.github/workflows/ci.yml`** (phase‑aware):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: docker compose up -d
      - run: make test-phase-1
      - run: make test-phase-2
      - run: make test-phase-3
      - run: make test-phase-4
      - run: make test-phase-5
      - uses: actions/upload-artifact@v4
        with:
          name: reports
          path: reports/
```

**`Makefile`** (key targets):

```makefile
PHASE?=1
up:
\tdocker compose up -d
down:
\tdocker compose down -v
test-phase-%: up
\tbash scripts/test/run_phase.sh $* || (echo "Phase $* failed" && exit 1)
```

**`.github/pull_request_template.md`**

```
# PR Title (Phase X.Y): <feature>

## Scope
- [ ] Implements task X.Y exactly as per /docs/implementation-plan.md
- [ ] No mocks used in tests; hits live stack

## Tests & Artifacts
- [ ] Added tests under tests/pX_tY_*.py
- [ ] Attached /reports/phase-X/junit.xml and summary.json
- [ ] For perf tasks: attached perf CSV and plots

## Phase Gate
- [ ] Meets DoD and Gate criteria for Phase X
```

---

## How to use this pack

1. Documents located under `/docs/` as named.
2. Run the **scaffold script** to create stubs and folders.
3. Implement each task; for every PR, include **reports** for that phase.
4. Advance only when the **Phase Gate** is met (CI must pass, artifacts uploaded).
5. Share `/reports/phase-*/summary.json` & `junit.xml` with me to **verify success**.
