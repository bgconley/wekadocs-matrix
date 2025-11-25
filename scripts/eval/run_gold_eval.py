"""
CLI to evaluate retrieval against a gold set with human-readable doc_tags.

Gold YAML schema (list of cases):
  - query: "text"
    expected_doc_tags:
      - "monitor-the-weka-cluster_snapshot-management"
      - "operation-guide_events_events-list"

This script resolves doc_tags -> document_ids via Qdrant payloads, runs the
QueryService, and computes simple metrics (Recall@3, Hit@3, MRR).
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

import yaml
from qdrant_client import QdrantClient

# Query pipeline (uses project settings)
from src.mcp_server.query_service import get_query_service
from src.shared.connections import close_connections, initialize_connections
from tests.eval.id_resolver import resolve_doc_ids_for_tag
from tests.eval.metrics import hit_at_k, mrr, recall_at_k


def load_gold(path: Path) -> List[Dict]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Gold file must be a list of cases")
    return data


async def run_eval(args):
    await initialize_connections()
    qs = get_query_service()

    client = QdrantClient(url=args.qdrant_url)
    gold_cases = load_gold(Path(args.gold_file))

    stats = []
    for case in gold_cases:
        query = case.get("query")
        tags = case.get("expected_doc_tags") or []
        if not query or not tags:
            continue

        expected_ids: List[str] = []
        for tag in tags:
            expected_ids.extend(resolve_doc_ids_for_tag(client, args.collection, tag))
        expected_ids = list(dict.fromkeys(expected_ids))

        resp = qs.search(query=query, top_k=args.top_k, verbosity="graph")
        retrieved_ids = [c.document_id for c in resp.results]

        r3 = recall_at_k(retrieved_ids, expected_ids, k=3)
        h3 = hit_at_k(retrieved_ids, expected_ids, k=3)
        rr = mrr(retrieved_ids, expected_ids)
        stats.append({"query": query, "recall@3": r3, "hit@3": h3, "mrr": rr})

    await close_connections()

    # Output
    for s in stats:
        print(
            f"{s['query'][:60]}... | recall@3={s['recall@3']:.2f} "
            f"hit@3={s['hit@3']:.2f} mrr={s['mrr']:.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-file", required=True)
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="chunks_multi_bge_m3")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    sys.exit(main())
