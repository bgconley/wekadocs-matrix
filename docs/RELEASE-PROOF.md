# Release Proof

This runbook captures the install-from-scratch proof for the Nutanix docs
matrix. It is split into offline checks and live proof because live proof
requires the model stack, Qdrant, Neo4j, Redis, and the ingestion worker.

## Prerequisites

- Branch: `wip/weka-to-nutanix-migration`
- Docker compose services available locally or on the configured GPU host.
- Required secrets exported: `NEO4J_PASSWORD`, `REDIS_PASSWORD`, `JWT_SECRET`.
- Nutanix corpus available at `${NUTANIX_CORPUS_PATH:-data/ingest/nutanix}`.

## Offline Checks

```bash
python -m py_compile scripts/smoke_test_golden_queries.py \
  scripts/run_canonical_retrieval_benchmark.py \
  scripts/eval/check_evidence_citations.py
pytest tests/test_evidence_citation_scoring.py -q
bash -n scripts/release_proof.sh
bash scripts/release_proof.sh
NEO4J_PASSWORD=x REDIS_PASSWORD=x JWT_SECRET=x docker compose config >/dev/null
```

## Live Proof

Run only after the user confirms services are available:

```bash
bash scripts/release_proof.sh --live
```

Artifacts land under `artifacts/release-proof/<timestamp>/` unless
`RELEASE_PROOF_ARTIFACT_DIR` is set.

## Acceptance Bars

- Golden retrieval hit rate: record the measured rate. The first Nutanix live
  run is calibration; do not weaken the bar silently.
- Citation/evidence pass rate: default `CITATION_MIN_PASS_RATE=0.8`.
- Completion report must record ingest counts, retrieval rate, citation rate,
  fallback/degradation events, artifact paths, and any calibrated threshold.

## Backup / Restore

Use the existing scripts under `deploy/scripts/` before destructive live runs
when preserving a prior local corpus matters.
