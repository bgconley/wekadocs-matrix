# Canonical Retrieval Benchmark

This benchmark freezes 10 retrieval prompts that are grounded in the current WEKA docs corpus.

Purpose:
- compare retrieval changes without prompt drift
- separate query-quality problems from retrieval regressions
- keep graph-debugging runs comparable across profiles and deployments

Artifacts:
- Query set: `tests/fixtures/canonical_retrieval_benchmark.yaml`
- Runner: `scripts/run_canonical_retrieval_benchmark.py`

Rules:
- Prompt text is part of the contract. Do not paraphrase benchmark prompts.
- Acceptance is defined per query in the YAML.
- If a prompt is no longer representative, add a new benchmark version instead of silently editing `v1`.

Usage:
```bash
python scripts/run_canonical_retrieval_benchmark.py
python scripts/run_canonical_retrieval_benchmark.py --base-url http://10.25.0.50:8000
python scripts/run_canonical_retrieval_benchmark.py --compare reports/retrieval_benchmarks/previous.json
```

Evaluation model:
- `heading_contains`: checks whether one or more expected headings appear within top N
- `doc_tag_equals`: checks whether results from an expected doc dominate top N
- `doc_tag_max_count`: limits known noise docs within top N

Interpretation:
- A failed query does not automatically mean the system is unusable.
- It does mean the change should not be described as an improvement without qualification.
- Use the compare mode to measure whether a change moved the same frozen prompt in the right direction.
