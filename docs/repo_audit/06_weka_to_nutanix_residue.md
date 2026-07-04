# WEKA To Nutanix Residue

## Summary

WEKA residue is active, not merely historical. The repo name and many docs can remain historical, but active prompts, model instructions, environment names, service names, and corpus assumptions must be domain-extracted before this can be trusted for Nutanix.

## Active Runtime Residue

| Surface | Examples | Severity |
|---|---|---|
| MCP instructions | `src/mcp_server/mcp_app.py`, `src/mcp_server/mcp_tools.py` describe a WEKA documentation knowledge base | High |
| Reranker instructions | `config/development.yaml` reranker instruction and `instructions_by_type` are explicitly WEKA-focused | High |
| Embedding query instruction | `config/embedding_profiles.yaml` `qwen3_0_6b.query_instruction` says WEKA distributed file system | High |
| NER/entity labels | `config/development.yaml` GLiNER labels/examples are WEKA technical taxonomy | High |
| Compose/service naming | `weka-net`, `weka-neo4j`, `weka-qdrant`, `weka-mcp-server`, `weka-ingestion-worker`, OTEL service names | Medium |
| Env files | `.env.docker`, `.env.production`, `.env.apply-schema` point to `weka-*` hostnames | Medium |
| Corpus and reports | `data/ingest/*`, `docs/wekadocs50_combined.md`, context reports | Historical/input |

## Nutanix Gaps

- No obvious Nutanix corpus under `data/`, `docs/`, or config paths.
- No Nutanix domain entity schema, product taxonomy, or retrieval prompts.
- No Nutanix golden query set or evaluation expectations.
- No Nutanix deployment naming/profile.

## Migration Principle

Do not mechanically replace `weka` with `nutanix`. The active system should gain a domain layer:

- `domain.id`: `weka` or `nutanix`.
- `domain.name`: display name.
- domain instructions for MCP/evidence behavior.
- embedding/reranker task instructions.
- NER/entity labels and exclusion rules.
- corpus tags and doc category mapping.
- golden queries and acceptance criteria.

Then WEKA can become a legacy domain pack and Nutanix can be introduced deliberately.
