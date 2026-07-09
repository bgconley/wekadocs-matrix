#!/usr/bin/env bash
set -euo pipefail

LIVE=0
if [[ "${1:-}" == "--live" ]]; then
  LIVE=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--live]" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="${RELEASE_PROOF_ARTIFACT_DIR:-artifacts/release-proof/${STAMP}}"
CORPUS_PATH="${NUTANIX_CORPUS_PATH:-data/ingest/nutanix}"
GOLDEN_TOP_K="${GOLDEN_TOP_K:-10}"
CITATION_MIN_PASS_RATE="${CITATION_MIN_PASS_RATE:-0.8}"

run() {
  echo "+ $*"
  if [[ "$LIVE" == "1" ]]; then
    "$@"
  fi
}

if [[ "$LIVE" != "1" ]]; then
  echo "DRY RUN: pass --live to execute service-touching commands."
fi

run mkdir -p "$ARTIFACT_DIR"

run docker compose up -d
run python scripts/reset_datastores.py
run scripts/ingestctl ingest "$CORPUS_PATH" --tag nutanix --json

run python scripts/smoke_test_golden_queries.py \
  --queries tests/fixtures/golden_query_set.yaml \
  --top-k "$GOLDEN_TOP_K" \
  --report "$ARTIFACT_DIR/golden.json"

run python scripts/run_canonical_retrieval_benchmark.py \
  --queries tests/fixtures/canonical_retrieval_benchmark.yaml \
  --report "$ARTIFACT_DIR/canonical.json"

run python scripts/eval/check_evidence_citations.py \
  --input "$ARTIFACT_DIR/canonical.json" \
  --output "$ARTIFACT_DIR/citations.json" \
  --min-pass-rate "$CITATION_MIN_PASS_RATE"

echo "Release proof artifacts: $ARTIFACT_DIR"
