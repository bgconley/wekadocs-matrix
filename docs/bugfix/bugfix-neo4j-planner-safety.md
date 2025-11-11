# Neo4j Vector Safety & Planner Guardrails Bugfix Log

## Problem Summary (ExplainGuard context)
ExplainGuard originally rejected valid traversals because only `MENTIONS`/`HAS_SECTION` edges were whitelisted. The remediation (central schema allow‑list) proved the need for tighter-but-accurate guardrails elsewhere. During this review we discovered three adjacent flaws:
1. **Neo4j vector lookups** crashed (`AttributeError`) and were open to Cypher injection because they interpolated filters directly and treated `get_config()` like a dict.
2. **Template fallback** always returned zero rows; `_build_params` injected an empty `section_ids` list into every plan, and the fallback Cypher filtered on `IN []`.
3. **Constraint rewriting** overwrote template-defined hop bounds, broadening traversals in `_inject_limits_and_constraints`, undermining ExplainGuard’s intent of bounded exploration.

## Proposed vs Executed Plan
- **Plan:**
  1. Parameterize Neo4j vector queries, reference `config.embedding.version` via attributes, and sanitize filter keys.
  2. Make planner fallbacks respect missing `section_ids` (treat as “match all”), and stop defaulting to `[]`.
  3. Update `_inject_limits_and_constraints` to only clamp ranges exceeding validator limits, preserving author-selected minimums.
  4. Add unit tests for the Neo4j vector store, planner fallback, and range rewrites.
- **Execution:** Implemented exactly as planned. No deviations were necessary after code inspection.

## Files Touched
1. `src/query/hybrid_search.py` – safe Neo4j vector search with sanitized parameters.
2. `src/query/planner.py` – smarter parameter defaults, safe fallback, constraint clamping.
3. `tests/p2_t1_test.py` – new planner regression cases.
4. `tests/query/test_vector_store.py` – new coverage for Neo4j parameterization / config usage.

## Neo4j / Qdrant Schema Impact
None. The work is limited to query and planner code; no schema migrations, collection changes, or index updates are required.

## Bootstrap Guidance for New Installs
1. Install / update dependencies normally (no new packages required).
2. Re-run `pytest tests/p2_t1_test.py tests/query/test_vector_store.py` to confirm planner/vector guardrails behave as expected.
3. No data reimport/reindex is needed; simply redeploy the code.

## Full Diffs

### `src/query/hybrid_search.py`
```diff
@@
-Phase 7C: Integrated with reranking provider for post-ANN refinement.
-"""
-
-import time
+Phase 7C: Integrated with reranking provider for post-ANN refinement.
+"""
+
+import re
+import time
@@
-    def search(
-        self, vector: List[float], k: int, filters: Optional[Dict] = None
-    ) -> List[Dict[str, Any]]:
-        """Search Neo4j vector index for top-k vectors."""
-        # Build WHERE clause from filters
-        where_clauses = []
-        if filters:
-            for key, value in filters.items():
-                if isinstance(value, str):
-                    where_clauses.append(f"node.{key} = '{value}'")
-                else:
-                    where_clauses.append(f"node.{key} = {value}")
-
-        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""
-
-        query = f"""
-        CALL db.index.vector.queryNodes($index_name, $k, $vector)
-        YIELD node, score
-        WHERE node.embedding_version = $embedding_version{where_clause}
-        RETURN node.id AS id, score, node.document_id AS document_id,
-               labels(node)[0] AS node_label, properties(node) AS metadata
-        LIMIT $k
-        """
-
-        with self.driver.session() as session:
-            result = session.run(
-                query,
-                index_name=self.index_name,
-                k=k,
-                vector=vector,
-                embedding_version=get_config()
-                .get("embedding", {})
-                .get("version", "v1"),
-            )
+    def search(
+        self, vector: List[float], k: int, filters: Optional[Dict] = None
+    ) -> List[Dict[str, Any]]:
+        """Search Neo4j vector index for top-k vectors."""
+        config = get_config()
+
+        where_clauses = ["node.embedding_version = $embedding_version"]
+        params: Dict[str, Any] = {
+            "index_name": self.index_name,
+            "k": k,
+            "vector": vector,
+            "embedding_version": config.embedding.version,
+        }
+
+        if filters:
+            dedupe: Dict[str, int] = {}
+            for key, value in filters.items():
+                safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key)
+                dedupe.setdefault(safe_key, 0)
+                dedupe[safe_key] += 1
+                suffix = dedupe[safe_key]
+                param_name = f"filter_{safe_key}_{suffix}"
+                where_clauses.append(f"node.{safe_key} = ${param_name}")
+                params[param_name] = value
+
+        where_clause = " AND ".join(where_clauses)
+
+        query = """
+        CALL db.index.vector.queryNodes($index_name, $k, $vector)
+        YIELD node, score
+        WHERE {where_clause}
+        RETURN node.id AS id, score, node.document_id AS document_id,
+               labels(node)[0] AS node_label, properties(node) AS metadata
+        LIMIT $k
+        """.format(where_clause=where_clause)
+
+        with self.driver.session() as session:
+            result = session.run(query, **params)
```

### `src/query/planner.py`
```diff
@@
-        # For search intent, add empty section_ids if not provided
-        if "section_ids" not in params:
-            params["section_ids"] = []
+        # Allow optional section scoping; None indicates "match all"
+        if "section_ids" not in params:
+            params["section_ids"] = None
@@
-        # Ensure variable-length patterns use parameter
-        cypher = re.sub(r"\*(\d+)\.\.(\d+)", r"*1..$max_hops", cypher)
+        max_depth = getattr(self.config.validator, "max_depth", 3)
+
+        def _range_rewrite(match: re.Match) -> str:
+            min_hops = int(match.group(1))
+            max_hops = int(match.group(2))
+
+            if max_hops <= max_depth:
+                return match.group(0)
+
+            if min_hops <= 1:
+                return "*1..$max_hops"
+
+            return f"*{min_hops}..$max_hops"
+
+        cypher = re.sub(r"\*(\d+)\.\.(\d+)", _range_rewrite, cypher)
@@
-        MATCH (s:Section)
-        WHERE s.id IN $section_ids
+        MATCH (s:Section)
+        WITH s, coalesce($section_ids, []) AS allowed_ids
+        WHERE size(allowed_ids) = 0 OR s.id IN allowed_ids
         RETURN s
```

### `tests/p2_t1_test.py`
```diff
@@
     def test_parameterization_no_literals(self):
         planner = QueryPlanner()
         plan = planner.plan("find section with id abc123")
@@
         assert len(dangerous_patterns) == 0 or all(
             "section_ids" not in p for p in dangerous_patterns
         )
+
+    def test_fallback_plan_handles_missing_section_ids(self):
+        planner = QueryPlanner()
+        params = planner._build_params({})
+        plan = planner._fallback_plan("random", "search", params)
+
+        assert "coalesce($section_ids, [])" in plan.cypher
+        assert plan.params["section_ids"] is None
+
+    def test_inject_limits_preserves_lower_bounds(self):
+        planner = QueryPlanner()
+        params = {
+            "limit": 10,
+            "max_hops": planner.config.validator.max_depth,
+        }
+        cypher = "MATCH p=()-[:REL*2..10]->() RETURN p"
+        updated = planner._inject_limits_and_constraints(cypher, params)
+
+        assert "*2..$max_hops" in updated
+        assert "LIMIT $limit" in updated
+
+    def test_inject_limits_leaves_tight_bounds(self):
+        planner = QueryPlanner()
+        params = {
+            "limit": 5,
+            "max_hops": planner.config.validator.max_depth,
+        }
+        cypher = "MATCH p=()-[:REL*0..1]->() RETURN p"
+        updated = planner._inject_limits_and_constraints(cypher, params)
+
+        assert "*0..1" in updated
```

### `tests/query/test_vector_store.py`
```diff
+import pytest
+
+from src.query.hybrid_search import Neo4jVectorStore
+
+
+class DummySession:
+    ... (see file for full helper) ...
+
+def test_neo4j_vector_store_parameterizes_filters():
+    ...
+
+def test_neo4j_vector_store_uses_embedding_version_from_config():
+    ...
```
*(See file for complete helper implementations; included in repository.)*

## Outstanding Concerns & Regression Checks
- **Regressions covered:** Added planner tests and brand-new vector store tests; existing ingestion/ExplainGuard suites remain untouched.
- **Manual verification:** No runtime regression expected; only query construction changed.
- **Follow-ups:** None mandatory, but consider extending `_inject_limits_and_constraints` to cover unbounded `*..` patterns in future phases.

## Verification
- `pytest tests/p2_t1_test.py tests/query/test_vector_store.py -q`
- All relevant tests passed locally.
