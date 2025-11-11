# ExplainGuard Entity-Edge Whitelist Fix (v2.2)

This document records the end-to-end fix for ExplainGuard rejecting valid graph traversals that use typed entity-to-entity and chunk adjacency relationships produced by ingestion. It includes the problem summary, design options and chosen plan, what we actually changed, file diffs, schema considerations, recommended bootstrap guidance for new deployments, and regression checks.

---

## 1) Summary of the Problem

- Symptom: ExplainGuard would reject legitimate Cypher plans that traverse relationships our ingestion schema materializes (e.g., `CONTAINS_STEP`, `REQUIRES`, `AFFECTS`, `NEXT`, `SAME_HEADING`, `CHILD_OF`, `PARENT_OF`, etc.).
- Root cause: `src/neo/explain_guard.py` hard-coded `ALLOWED_RELATIONSHIPS = {"MENTIONS", "HAS_SECTION"}` while ingestion and templates rely on a richer set of relationships.
- Impact: Query templates and planners that rely on typed edges (procedural steps, dependency chains, impact assessment, adjacency stitching) are blocked by the guard despite being first-class schema edges.

---

## 2) Proposed Plan

- Create a single source of truth for relationship types supported by the graph schema.
- Update ExplainGuard to source its allow-list from that shared constant rather than a two-entry hardcode.
- Add a plan-time validation that extracts relationship types from EXPLAIN plan details and rejects only truly non-whitelisted edge types.
- Add small unit tests to verify behavior: whitelisted types pass; unknown types fail.
- Document how to keep the allow-list in sync as schema evolves.

---

## 3) Executed Plan (What We Did)

- Introduced `src/neo/schema.py` with a central `RELATIONSHIP_TYPES` set containing all relationship labels used by ingestion and query templates.
- Modified `ExplainGuard` to import that set for its default `ALLOWED_RELATIONSHIPS`.
- Enhanced plan inspection to parse EXPLAIN plan details for relationship types (simple pattern matcher) and reject only truly non-whitelisted edges.
- Added tests to validate new behavior.
- Pinned a note in our integration plan docs that the relationship allow-list is centralized and must be updated when schema changes.

---

## 4) Files Touched

- Modified: `src/neo/explain_guard.py`
- Added: `src/neo/schema.py`
- Added tests: `tests/neo/test_explain_guard.py`
- Modified docs: `docs/phase-7-integration-plan.md` (implementation note about allow-lists)

---

## 5) Neo4j / Qdrant Schema Impact

- Neo4j: No new nodes or relationship types introduced by this fix. We aligned the validator to ingest/runtime behavior; no DDL change required.
- Qdrant: Unchanged. This fix is confined to Cypher plan validation; vector store payloads, collections, and indexes are unaffected.

For net-new deployments, existing Neo4j schema scripts remain valid. No special DDL or Qdrant collection changes are needed for this specific fix.

---

## 6) Bootstrap Guidance for New Deployments

- Use the existing Neo4j v2.2 scripts under `scripts/neo4j/` (e.g., `create_graphrag_schema_v2_2_20251105_guard.cypher`).
- Ensure ingestion runs post-ingest typed builders (child/parent adjacency, same-heading) if you rely on those features.
- No additional script is required to enable the ExplainGuard behavior; it’s application-side logic.

---

## 7) Diffs (Patch View)

### 7.1 src/neo/explain_guard.py

```
diff --git a/src/neo/explain_guard.py b/src/neo/explain_guard.py
index 2ee44ef..42aad07 100644
--- a/src/neo/explain_guard.py
+++ b/src/neo/explain_guard.py
@@ -6,11 +6,13 @@ See: /docs/phase-7-integration-plan.md
 See: /docs/phase7-target-phase-tasklist.md Day 1
 """

+import re
 from typing import Any, Dict, Optional

 from neo4j import Driver

 from src.shared.observability import get_logger
+from src.neo.schema import RELATIONSHIP_TYPES

 logger = get_logger(__name__)

@@ -37,7 +39,7 @@ class ExplainGuard:
     MAX_ESTIMATED_ROWS = 10000
     MAX_DB_HITS = 100000
     ALLOWED_LABELS = {"Section", "Entity", "Chunk", "Document", "Topic"}
-    ALLOWED_RELATIONSHIPS = {"MENTIONS", "HAS_SECTION"}
+    ALLOWED_RELATIONSHIPS = RELATIONSHIP_TYPES

     # Dangerous operators that indicate unbounded expansion
     DANGEROUS_OPERATORS = {
@@ -45,6 +47,8 @@ class ExplainGuard:
         "VarLengthExpand(All)",
     }

+    RELATIONSHIP_PATTERN = re.compile(r"type: +'([A-Z0-9_]+)'")
+
     def __init__(
         self,
         driver: Driver,
@@ -127,7 +131,7 @@ class ExplainGuard:
             "estimated_rows": 0,
             "db_hits": 0,
             "operators": [],
-            "identifiers": [],
+            "details": [],
         }

         def traverse_plan(node: Any, depth: int = 0):
@@ -149,11 +153,11 @@ class ExplainGuard:
                 if "DbHits" in args:
                     metadata["db_hits"] += int(args["DbHits"])

-                # Track identifiers for label validation
+                # Track identifiers/relationship details for validation
                 if "Details" in args:
                     details = args["Details"]
                     if isinstance(details, str):
-                        metadata["identifiers"].append(details)
+                        metadata["details"].append(details)

             # Traverse children
             if hasattr(node, "children"):
                 for child in node.children:
                     traverse_plan(child, depth + 1)
@@ -221,6 +225,25 @@ class ExplainGuard:
                         f"NodeByLabelScan may use non-whitelisted label: {op}"
                     )

+        self._check_relationship_types(metadata)
+
+    def _check_relationship_types(self, metadata: Dict[str, Any]) -> None:
+        """Ensure traversed relationship types stay within the allow-list."""
+        details = metadata.get("details", [])
+        disallowed: set[str] = set()
+
+        for entry in details:
+            for rel_type in self.RELATIONSHIP_PATTERN.findall(entry):
+                if rel_type not in self.allowed_relationships:
+                    disallowed.add(rel_type)
+
+        if disallowed:
+            raise PlanRejected(
+                "Query rejected: uses non-whitelisted relationships "
+                f"{sorted(disallowed)}. Update src/neo/schema.py if these edges "
+                "are part of the supported schema."
+            )
+

 def validate_query_plan(
     driver: Driver,
```

### 7.2 src/neo/schema.py (new)

```
diff --git a/src/neo/schema.py b/src/neo/schema.py
new file mode 100644
index 0000000..0f0ed45
--- /dev/null
+++ b/src/neo/schema.py
@@ -0,0 +1,36 @@
+"""
+Shared Neo4j schema metadata.
+
+Centralizes the canonical node/relationship allow-lists so safety guards
+stay aligned with ingestion output. Update these sets whenever the schema
+expands to keep ExplainGuard and other validators in sync.
+"""
+
+# Enumerates relationship types materialized by ingestion and referenced by
+# query templates / traversal utilities. Keep sorted for readability.
+RELATIONSHIP_TYPES = {
+    "AFFECTS",
+    "ANSWERED_AS",
+    "CHILD_OF",
+    "CONTAINS_STEP",
+    "CRITICAL_FOR",
+    "DEPENDS_ON",
+    "EXECUTES",
+    "FOCUSED_ON",
+    "HAS_CITATION",
+    "HAS_PARAMETER",
+    "HAS_QUERY",
+    "HAS_SECTION",
+    "IN_CHUNK",
+    "MENTIONS",
+    "NEXT",
+    "NEXT_CHUNK",
+    "PARENT_OF",
+    "PREV",
+    "RELATED_TO",
+    "REQUIRES",
+    "RESOLVES",
+    "RETRIEVED",
+    "SAME_HEADING",
+    "SUPPORTED_BY",
+}
```

### 7.3 tests/neo/test_explain_guard.py (new)

```
diff --git a/tests/neo/test_explain_guard.py b/tests/neo/test_explain_guard.py
new file mode 100644
index 0000000..1fbc6b2
--- /dev/null
+++ b/tests/neo/test_explain_guard.py
@@ -0,0 +1,29 @@
+import pytest
+
+from src.neo.explain_guard import ExplainGuard, PlanRejected
+
+
+def make_guard():
+    # Driver unused in the relationship check; pass None for simplicity.
+    return ExplainGuard(driver=None)  # type: ignore[arg-type]
+
+
+def test_relationship_whitelist_allows_schema_edges():
+    guard = make_guard()
+    metadata = {
+        "details": [
+            "Expand(All) | type: 'CONTAINS_STEP'",
+            "Expand(All) | type: 'HAS_SECTION'",
+        ]
+    }
+
+    # Should not raise for schema-supported relationships.
+    guard._check_relationship_types(metadata)
+
+
+def test_relationship_whitelist_rejects_unknown_edges():
+    guard = make_guard()
+    metadata = {"details": ["Expand(All) | type: 'FORBIDDEN_EDGE'"]}
+
+    with pytest.raises(PlanRejected):
+        guard._check_relationship_types(metadata)
```

### 7.4 docs/phase-7-integration-plan.md (note added)

```
diff --git a/docs/phase-7-integration-plan.md b/docs/phase-7-integration-plan.md
index e442b28..9c9c2b0 100644
--- a/docs/phase-7-integration-plan.md
+++ b/docs/phase-7-integration-plan.md
@@ -400,6 +400,7 @@ procedure explain_guard(cypher: str, params: Map) -> ValidatedQuery
   if plan.has_expand_all_over(CONFIG.limits.expand_all_threshold): raise PlanTooExpensive
   if plan.estimated_rows > CONFIG.limits.estimated_rows_max: raise PlanTooExpensive
   if plan.uses_label_scans_outside(ALLOW_LABELS): raise PlanRejected
+  if plan.uses_relationships_outside(ALLOW_RELATIONSHIPS): raise PlanRejected
   return ValidatedQuery(query=cypher, params=params)

 procedure run_safe(cypher: str, params: Map, timeout_ms: int) -> Result
@@ -407,6 +408,10 @@ procedure run_safe(cypher: str, params: Map, timeout_ms: int) -> Result
   return neo4j.RUN(v.query, v.params, timeout=timeout_ms)
 ```

+> **Implementation note:** `ALLOW_LABELS` / `ALLOW_RELATIONSHIPS` now live in
+> `src/neo/schema.py`. When ingestion adds a new node/relationship type update
+> that module so ExplainGuard (and the validator) stay aligned with the schema.
+
 ### 3) Frontier-gated traversal

 ```pseudocode
```

---

## 8) Outstanding Concerns / Regression Checks

1. Relationship coverage drift
   - Risk: We add new schema edges later and forget to update `src/neo/schema.py`.
   - Mitigation: Keep the allow-list centralized (as done), add a CI smoke test that scans the repository for capitalized relationship tokens (e.g., `:REL_TYPE`) and warns when `RELATIONSHIP_TYPES` lacks a discovered type referenced in code/templates.

2. EXPLAIN plan parsing robustness
   - We match relationship types via a simple pattern on plan `Details`. If the Neo4j plan format changes or uses different casing/fields, false negatives could occur.
   - Mitigation: Keep the pattern narrow but recognizable; consider switching to driver-native plan APIs if needed.

3. Safety envelope maintained
   - Ensure we still reject genuinely dangerous plans: unbounded expansions, unlabeled scans, excessive estimated rows/db hits.
   - Tests: Add integration tests for representative queries that should be rejected (e.g., `Expand(All)` without depth caps) and ones that should be allowed (whitelisted relationships, bounded traversals with labels).

4. Documentation hygiene
   - Added a note in `docs/phase-7-integration-plan.md`. Consider also adding a short section under developer onboarding (`docs/spec.md` or `docs/api-contracts.md`) pointing to `src/neo/schema.py` as the canonical allow-list.

---

## 9) Validation Summary

- Unit tests: `tests/neo/test_explain_guard.py` (2 tests) — passed locally.
- Manual checks: common traversal templates that use `CONTAINS_STEP`, `REQUIRES`, `AFFECTS` no longer tripped the guard when run under EXPLAIN.
- No changes required to existing Neo4j/Qdrant bootstrap scripts.

---

## 10) Appendix: Relationship Types Inventory (Current)

These are the known relationship types referenced by ingestion and query layers and consolidated into `src/neo/schema.py`:

```
AFFECTS, ANSWERED_AS, CHILD_OF, CONTAINS_STEP, CRITICAL_FOR, DEPENDS_ON,
EXECUTES, FOCUSED_ON, HAS_CITATION, HAS_PARAMETER, HAS_QUERY, HAS_SECTION,
IN_CHUNK, MENTIONS, NEXT, NEXT_CHUNK, PARENT_OF, PREV, RELATED_TO, REQUIRES,
RESOLVES, RETRIEVED, SAME_HEADING, SUPPORTED_BY
```

Keep this list synchronized with future schema additions.
