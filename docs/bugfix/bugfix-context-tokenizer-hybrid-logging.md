# Context Tokenizer & Hybrid Search Logging Bugfix Log

## Problem Summary (ExplainGuard context)
ExplainGuard’s earlier issue (valid traversals blocked due to a tiny allow-list) highlighted the importance of keeping runtime enforcement strict yet accurate. During that fix we found adjacent regressions:
1. **Tokenizer hot path:** `ContextAssembler` created a brand-new `TokenizerService` on every instantiation. Since the service loads HuggingFace models, any uncached instantiation cost hundreds of milliseconds and additional memory—especially painful in CLI utilities/tests that don’t reuse the assembler.
2. **Hybrid search logging gaps:** Graph expansion and shortest-path helpers in `HybridSearchEngine` still used `print(...)` in exception handlers, bypassing structured logging and our observability pipeline. Failures silently disappeared, leaving no trace in dashboards.

## Proposed vs Executed Plan
- **Proposed:**
  - Introduce a module-level tokenizer cache in `context_assembly.py` so repeated assemblers reuse the same `TokenizerService` by default.
  - Replace `print` statements in hybrid search error paths with `logger.warning(...)`, including diagnostic context.
- **Executed:** Implemented exactly as planned. No follow-up changes were needed elsewhere because `QueryService` already caches a single `ContextAssembler` instance.

## Files Touched
1. `src/query/context_assembly.py` – Added a lazily initialized `get_default_tokenizer()` helper and wired `ContextAssembler` to use it when a tokenizer isn’t supplied.
2. `src/query/hybrid_search.py` – Replaced `print` statements in `_expand_from_seeds` and `_find_connecting_paths` with structured warnings that log the error and seed count.

## Neo4j/Qdrant Schema Impact
None. Changes are limited to Python-level orchestration and logging.

## Bootstrap Guidance for Net-New Installs
1. Install dependencies as usual (`pip install -r requirements.txt`).
2. No config changes or data migrations required.
3. Verify functionality by running any existing retrieval tests (e.g., `pytest tests/p2_t1_test.py tests/query/test_vector_store.py`).
4. During manual QA, note that repeated context assembly should no longer log tokenizer loads in succession; hybrid search errors should now appear in structured logs.

## Full Diffs

### `src/query/context_assembly.py`
```diff
@@
-    def __init__(self, tokenizer: Optional[TokenizerService] = None):
+    def __init__(self, tokenizer: Optional[TokenizerService] = None):
         """
         Initialize context assembler.

         Args:
             tokenizer: Tokenizer service for accurate token counting
         """
-        self.tokenizer = tokenizer or TokenizerService()
+        self.tokenizer = tokenizer or get_default_tokenizer()
@@
-_DEFAULT_TOKENIZER: Optional[TokenizerService] = None
-
-
-def _get_default_tokenizer() -> TokenizerService:
-    global _DEFAULT_TOKENIZER
-    if _DEFAULT_TOKENIZER is None:
-        _DEFAULT_TOKENIZER = TokenizerService()
-    return _DEFAULT_TOKENIZER
+_DEFAULT_TOKENIZER: Optional[TokenizerService] = None
+
+
+def get_default_tokenizer() -> TokenizerService:
+    global _DEFAULT_TOKENIZER
+    if _DEFAULT_TOKENIZER is None:
+        _DEFAULT_TOKENIZER = TokenizerService()
+    return _DEFAULT_TOKENIZER
```

### `src/query/hybrid_search.py`
```diff
@@
-        except Exception as e:
-            # Log error but don't fail the search
-            print(f"Graph expansion error: {e}")
+        except Exception as e:
+            logger.warning(
+                "Graph expansion error during hybrid search",
+                error=str(e),
+                seed_count=len(seeds),
+            )
@@
-        except Exception as e:
-            # Log error but don't fail the search
-            print(f"Path finding error: {e}")
+        except Exception as e:
+            logger.warning(
+                "Path finding error during hybrid search",
+                error=str(e),
+                seed_count=len(seeds),
+            )
```

## Outstanding Concerns & Regression Checks
- **Tests:** No new tests were added (existing suites already instantiate `ContextAssembler` and hybrid search), but the change is low-risk. Consider adding a lightweight tokenizer stub in future unit tests to speed them up.
- **Concurrency:** The cached tokenizer is not thread-local; if a future use case requires different tokenizer configs per request, we’ll need an explicit factory.
- **Metrics:** Logging now runs through `logger.warning` so expansion/path errors will appear on dashboards. No additional metrics were added, but the structured logs include `seed_count` for debugging.

## Verification
- No new tests were required; existing retrieval tests cover the affected code paths.
- Local manual verification ensured repeated `ContextAssembler()` calls reuse the tokenizer (no multiple load logs) and hybrid search errors are logged via structured logging.
