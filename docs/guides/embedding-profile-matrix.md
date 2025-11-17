# Embedding Profile Matrix (Phases 1–3)

Phase 1–3 of the embedding provider modularization program are now complete. This document captures the canonical state of the profile manifest, registry-driven provider factory, and ingestion/retrieval wiring so engineers can reason about the system without digging through historical specs.

## Phase 1 – Profile Catalog & Settings

- **Manifest** – All supported profiles live in `config/embedding_profiles.yaml`. Each entry contains the provider slug, model id, dims, tokenizer backend/model, capability flags, and a list of required environment variables.
- **`EmbeddingSettings`** – `src/providers/settings.py` materializes a profile into run-time settings (provider, dims, tokenizer metadata, capabilities, namespace behavior). Settings respect deterministic environment precedence: the active profile is authoritative and legacy `EMBEDDINGS_*` overrides log warnings.
- **Strict provenance** – Every ingestion/retrieval payload records `embedding_version`, `embedding_provider`, and `embedding_profile`, allowing hard filters when swapping profiles or running dual-write migrations.
- **Quick checks** – Use `python scripts/verify_providers.py` to render the profile matrix (provider/model/dims/tokenizer/capabilities) and highlight missing env requirements before attempting ingestion or CI runs.

## Phase 2 – Provider Registry & Canonical BGE Service

- **Registry-driven factory** – `ProviderFactory` is the single entry point for embedders. Supported providers today are `jina-ai`, `sentence-transformers`, and the canonical `bge-m3-service` integration. Aliases like `bge_m3` or `st_minilm` all normalize through this registry.
- **Capability flags** – The factory emits telemetry via `build_embedding_telemetry`, ensuring downstream systems consume `supports_dense`, `supports_sparse`, `supports_colbert`, etc. without branching on provider strings.
- **BGE-M3 service integration** – The BGEM3 service exposes REST endpoints at `BGE_M3_API_URL`. The Python EmbeddingClient is vendored in `src/clients`, so no path hacks are required. Set `EMBEDDINGS_PROFILE=bge_m3` and `BGE_M3_API_URL` to use it.
- **Operational requirements** – When `EMBEDDINGS_PROFILE=bge_m3`, export `BGE_M3_API_URL`. `JINA_API_KEY` remains mandatory for `jina_v3`, while `st_minilm` requires no external secrets.

## Phase 3 – Pipeline Wiring & Metadata Enforcement

- **Ingestion + reconciliation** – `GraphBuilder`, the auto-ingestion worker, and reconciliation jobs now request embedders exclusively via `ProviderFactory` so they obey the active profile’s dims/capabilities. Dual-write scaffolding remains for archival reference but defaults to the 1024-D collections.
- **Index + schema awareness** – `IndexRegistry` and the Neo4j/Qdrant schema helpers read from `EmbeddingSettings`, stamping the correct dims, similarity metric, and namespace hints for each profile. `EMBEDDING_NAMESPACE_MODE=profile` provides deterministic per-profile collection/index names when needed.
- **Strict-mode drift guard** – `EMBEDDING_STRICT_MODE` enforces that ingestion will fail fast if an existing section carries a different `embedding_version`, preventing mixed vectors inside a collection.
- **Tokenizer parity** – The tokenizer service and any consumers pull backend/model identifiers straight from `EmbeddingSettings`, ensuring both ingestion and query-side truncation match the selected profile.

## Verification Workflow

1. **Render the matrix**
   ```bash
   python scripts/verify_providers.py
   ```
   Confirms provider -> model/dims/task/tokenizer plus missing env vars.

2. **Run profile ingestion/retrieval smoke**
   ```bash
   export RUN_PROFILE_MATRIX_INTEGRATION=1
   export JINA_API_KEY=...            # for jina_v3
   export BGE_M3_API_URL=...          # for bge_m3
   export
   pytest tests/integration/test_profile_ingestion_retrieval.py -k profile
   ```
   Hits real providers, re-ingests a synthetic document per profile, and verifies both Qdrant metadata and `QdrantMultiVectorRetriever`.

3. **Ping raw providers only**
   ```bash
   RUN_PROFILE_MATRIX_INTEGRATION=1 pytest tests/integration/test_profile_matrix_integration.py
   ```
   Minimal health check that each provider returns a vector with the expected dimensionality.

Both suites auto-skip if required env vars are missing, so run them locally before enabling CI automation.

## CI Coverage

The GitHub Actions workflow now includes a `profile-matrix-smoke` job that runs on a nightly schedule (and via manual `workflow_dispatch`). It:

- Checks out the repo and installs dependencies.
- Spins up Qdrant via Docker for the ingestion/retrieval tests.
- Requires the following secrets to be configured in the repo:
  - `JINA_API_KEY`
  - `BGE_M3_API_URL`
  - `BGE_M3_CLIENT_REPO` (Git URL for cloning the canonical `bge-m3-custom` repo during the run)
  - Optional `BGE_M3_CLIENT_REF` for pinning the client commit.
- Runs the profile matrix integration tests with the vendored client and `RUN_PROFILE_MATRIX_INTEGRATION=1`.
- Executes `tests/integration/test_profile_matrix_integration.py` and `tests/integration/test_profile_ingestion_retrieval.py`.

If any secret is missing, the job short-circuits with a notice so that regular CI (unit/phase tests) still passes. Configure the secrets and the job will provide continuous coverage for real-provider swaps without manual intervention.

## Configure CI Secrets

Set the required GitHub Actions secrets before enabling the nightly smoke job:

1. Ensure you have the GitHub CLI (`gh`) authenticated with repo admin privileges.
2. Create/update each secret. Values can be provided interactively, piped from files, or read from environment variables.

```bash
# Minimal secrets
gh secret set JINA_API_KEY           # prompts for value
gh secret set BGE_M3_API_URL --body "https://bge-service.example.com"
gh secret set BGE_M3_CLIENT_REPO --body "git@github.com:wekadocs/bge-m3-custom.git"

# Optional pin to a commit/tag for deterministic client revisions
gh secret set BGE_M3_CLIENT_REF --body "refs/tags/v1.2.3"
```

Tips:
- For self-hosted BGE services, prefer HTTPS endpoints that are reachable from GitHub-hosted runners or expose them through a secure tunnel.
- If you keep secrets in a `.env` file locally, `gh secret set -f .env` can import them in bulk; double-check the file contains only the keys you intend to publish.
- Use `gh secret list` to verify that the repository now exposes the required variables.

## Manual Dispatch & Monitoring

While the `profile-matrix-smoke` job runs nightly, you can trigger it on demand after updating infrastructure/secrets:

```bash
# Trigger ci.yml manually (defaults to workflow_dispatch)
gh workflow run ci.yml

# Watch latest execution (includes profile-matrix-smoke job)
gh run watch

# Or open the Actions tab in the GitHub UI and inspect the Real Provider Smoke job logs
```

Troubleshooting checklist:
- Ensure `RUN_PROFILE_MATRIX_INTEGRATION` shows up in the job logs (it is injected automatically when `profile-matrix-smoke` runs).
- Confirm that the “Clone canonical BGE-M3 client” step fetched the repo/commit you expect.
- If tests skip, check the notice emitted by the “Validate integration secrets” step—missing secrets cause an early exit.
- Qdrant startup issues typically mean port collisions (job uses `-p 6333:6333`) or rate limits, so rerun the workflow or clean up conflicting jobs.

## Cutover Runbook (Strict Mode)

Use this sequence whenever you swap embedding profiles in production:

1. **Pre-flight checks**
   ```cypher
   MATCH (s:Section)
   WHERE s.embedding_version <> $CURRENT_VERSION
   RETURN count(*) AS drift, collect(s.id)[0..5] AS sample;
   ```
   - Replace `$CURRENT_VERSION` with the target profile version (e.g., `jina-embeddings-v3`). Running this before strict mode blocks ingestion helps surface lingering backlog.
2. **Enable dual-write (optional)**
   ```bash
   echo "DUAL_WRITE_1024D=true" >> .env
   docker-compose restart ingestion-worker
   ```
   - Dual-write is only required if you still have legacy 384-D collections in service. Fresh installs can skip this and keep the flag `false`.
3. **Toggle strict mode intentionally**
   ```bash
   # Default is true; override temporarily only if you must ingest during cleanup
   export EMBEDDING_STRICT_MODE=true
   ```
   - If you disable strict mode to clear drift, re-enable it immediately after validation so GraphBuilder will block future mismatches. The runtime logs and Prometheus counter `embedding_profile_guard_events_total` show whether the guard is blocking, warning, or clean.
4. **Cutover + verification**
   - Re-ingest or reconcile using the new profile.
   - Run the `profile-matrix-smoke` workflow (see above) to confirm real-provider tests pass.
   - Re-run the Cypher drift query; expect zero rows.
5. **Rollback**
   - Toggle `DUAL_WRITE_1024D=false` (or restore the prior embedding profile env vars) and restart the workers to fall back.
   - Clear any failed ingestion jobs and rerun with the previous profile.

Document the outcome in Serena (summary + observations) so the strict-mode guardrails remain auditable.

## Phase 6+ Outlook

With phases 1–5 delivered and real-provider coverage automated, the next milestones focus on rollout and observability:

1. **Rollout guardrails (Phase 6)** – finalize EMBEDDING_STRICT_MODE defaults, bake profile/namespace metadata into telemetry dashboards, and document the cutover sequence (dual-write toggle, verification queries, rollback switch).
2. **Diagnostics + alerting (Phase 7)** – hook the profile matrix results into monitoring (e.g., fail CI if a provider drifts), surface BGE/Jina latency + success-rate metrics, and ensure HybridRetriever logs emit per-profile budgets.
3. **Doc polish** – consolidate the migration/runbook content referenced in Serena (phases 1–3) into this guide, so engineers can onboard without chasing historical notes.

Capture decisions/test evidence in Serena memories as we progress so future sessions inherit a clear provenance trail.
