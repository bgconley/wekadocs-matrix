# Embedding Profile Rollout & Safety (Phase 6)

Phase 6 of the embedding provider modularization program introduces **controlled profile swaps** for staging/production. This document explains the new flags, how overrides are gated by environment, and the concrete rollout/rollback steps.

## Runtime flags & behavior

Embedding configuration is still sourced from:

- `config/<env>.yaml` → `embedding.profile` (baseline profile for the environment).
- `config/embedding_profiles.yaml` → manifest of all supported profiles.
- `EMBEDDINGS_PROFILE` → optional runtime override of the active profile.

New Phase 6 flags in `Settings` (`src/shared/config.py:525`):

- `EMBEDDING_PROFILE_SWAPPABLE` → `embedding_profile_swappable: bool`
  - Default: `false`.
  - When `true` in a **non-dev/test** env, allows `EMBEDDINGS_PROFILE` to override the baseline profile.
- `EMBEDDING_PROFILE_EXPERIMENT` → `embedding_profile_experiment: Optional[str]`
  - When set to a profile name (e.g. `bge_m3`), permits an override **only if** `EMBEDDINGS_PROFILE` matches that profile.

`apply_embedding_profile` (`src/shared/config.py:631`) now enforces:

- It resolves a **baseline profile** from `config.embedding.profile` or manifest defaults.
- It applies a runtime override from `EMBEDDINGS_PROFILE` only when:
  - Environment is `development`/`dev`/`test`, **or**
  - `EMBEDDING_PROFILE_SWAPPABLE=true`, **or**
  - `EMBEDDING_PROFILE_EXPERIMENT=<profile>` and `EMBEDDINGS_PROFILE=<profile>`.
- In any other case, a conflicting override in non-dev/test will raise a **startup error** with a clear message instead of silently swapping profiles.

Provider telemetry (`src/providers/factory.py:289`) includes:

- `embedding_namespace_mode`
- `embedding_strict_mode`
- `embedding_profile_swappable`
- `embedding_profile_experiment`

Use these labels in metrics/logs to confirm the rollout state of each environment.

## Rollout checklist (switching profiles safely)

Assume you are moving from `jina_v3` → `bge_m3` as the primary profile.

1. **Prepare the manifest and secrets**
   - Ensure `config/embedding_profiles.yaml` has a valid `bge_m3` entry (dims, tokenizer, requirements).
   - Export provider secrets for the target profile (e.g. `BGE_M3_API_URL`).
   - Verify the matrix:
     ```bash
     python scripts/verify_providers.py
     ```

2. **Verify in development**
   - In `development` env:
     ```bash
     export EMBEDDINGS_PROFILE=bge_m3
     make run-dev  # or equivalent
     ```
   - Dev always allows overrides; confirm ingestion/retrieval succeed and Profile Matrix tests are green:
     ```bash
     export RUN_PROFILE_MATRIX_INTEGRATION=1
     pytest tests/integration/test_profile_ingestion_retrieval.py -k profile
     ```

3. **Stage the new profile (staging)**
   - In staging (e.g. `ENV=staging` / `settings.env=staging`):
     ```bash
     export EMBEDDING_PROFILE_SWAPPABLE=true
     export EMBEDDINGS_PROFILE=bge_m3
     ```
   - Restart the MCP/query services.
   - Re-ingest the target corpus with the new profile (using your Phase 6 ingestion/orchestrator flow).
   - Run regression + quality suites and compare metrics (latency, success rate, relevance) against the previous profile.

4. **Lock in the profile**
   - Once staging looks good:
     - Update `config/staging.yaml` (and eventually `config/production.yaml`) to set:
       ```yaml
       embedding:
         profile: "bge_m3"
       ```
     - Remove `EMBEDDINGS_PROFILE` and set:
       ```bash
       export EMBEDDING_PROFILE_SWAPPABLE=false
       unset EMBEDDING_PROFILE_EXPERIMENT
       ```
   - This makes `bge_m3` the **baseline** profile and prevents accidental runtime swaps.

5. **Production rollout**
   - Repeat the staging steps in production:
     1. Temporarily enable:
        ```bash
        export EMBEDDING_PROFILE_SWAPPABLE=true
        export EMBEDDINGS_PROFILE=bge_m3
        ```
     2. Re-ingest into the new collections/indexes (respecting namespace/suffix rules from `EMBEDDING_NAMESPACE_MODE`).
     3. Run Phase 6 post-ingest verification (alignment checks, sample queries, dashboards).
   - When metrics look good, update `config/production.yaml` to `profile: "bge_m3"` and remove the env overrides as in step 4.

## Experiment / canary rollout

To run a targeted experiment instead of a broad swap:

1. Leave the baseline profile as-is in YAML (for example, `jina_v3`).
2. On a **single canary instance** (or a dedicated experiment environment), set:
   ```bash
   export EMBEDDING_PROFILE_EXPERIMENT=bge_m3
   export EMBEDDINGS_PROFILE=bge_m3
   ```
3. Restart only that instance and re-ingest a scoped dataset (or reuse existing vectors if compatible).
4. Route a slice of traffic to the canary and compare metrics vs. the baseline instance.

Notes:

- If `EMBEDDINGS_PROFILE` is set to anything other than `EMBEDDING_PROFILE_EXPERIMENT` in a non-dev/test env, startup will fail fast.
- This ensures experiments are **explicitly named** and cannot be enabled by typoed overrides.

## Rollback runbook

If a new profile degrades quality or reliability:

1. **Stop new writes**
   - Disable ingestion jobs targeting the experimental profile.
   - If you used `EMBEDDING_PROFILE_SWAPPABLE` in staging/prod, immediately:
     ```bash
     unset EMBEDDINGS_PROFILE
     export EMBEDDING_PROFILE_SWAPPABLE=false
     unset EMBEDDING_PROFILE_EXPERIMENT
     ```

2. **Restore baseline**
   - Ensure `config/<env>.yaml` still points to the previous stable profile (e.g. `jina_v3`).
   - Restart services so `apply_embedding_profile` resolves only the baseline profile.

3. **Validate recovery**
   - Re-run smoke tests and key queries to confirm behavior matches the pre-change baseline.
   - Inspect provider configuration logs:
     - Look for `Profile Swappable: False` and an empty `Profile Experiment` in the startup banner (`src/providers/factory.py:315`).

4. **Post-mortem**
   - Capture metrics and logs for the failed profile.
   - Update `config/embedding_profiles.yaml` and this runbook with any new constraints or lessons learned.

With these controls in place, profile changes are:

- **Safe by default** in staging/prod (overrides rejected unless explicitly allowed).
- **Easy to experiment with** via `EMBEDDING_PROFILE_EXPERIMENT`.
- **Auditable** through provider telemetry and logs.
