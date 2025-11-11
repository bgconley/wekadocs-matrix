# Tokenizer Caching, Hybrid Search Logging, and Phase‑6 Backup Cleanup

## Problem Summary (ExplainGuard context)
- ExplainGuard previously blocked valid traversals due to tight allow‑lists. While that was fixed, adjacent issues emerged: Context assembly created a new tokenizer per request, hybrid search error paths used bare `print()` calls (bypassing structured logs), and obsolete Phase‑6 auto-ingest backups lingered.

## Proposed vs Executed Plan
- **Proposed**: Cache tokenizer instances, replace `print` with structured logging, and remove dead backups.
- **Executed**: Implemented a module-level `get_default_tokenizer()` hook, swapped logging in `_expand_from_seeds` and `_find_connecting_paths`, and deleted `.backup` files.

## Files Touched
- `src/query/context_assembly.py`
- `src/query/hybrid_search.py`
- `src/ingestion/auto/service.py.backup` (deleted)
- `src/ingestion/auto/watcher.py.backup` (deleted)

## Neo4j/Qdrant Schema Impact
None.

## Bootstrap Guidance
1. Standard dependency install.
2. No new config.
3. Redeploy only.

## Diffs
*(summaries instead of full diffs)*
- Context assembler now uses `get_default_tokenizer()` to reuse a shared tokenizer.
- Hybrid search exception handlers call `logger.warning(..., seed_count=...)`.
- Removed obsolete `.backup` service/watcher files.

## Outstanding Concerns & Regression Checks
- No new tests were added; existing suites cover affected paths.
- Tokenizer cache uses a simple module-level singleton; future extensions may require thread-specific behavior.
