
Absolutely—here are both deliverables, ready to drop into your repo:

- **A tiny fusion A/B harness** that compares **RRF vs. weighted fusion** (vector + BM25), computes **Hit@k, MRR@k, nDCG@k**, and emits a Markdown + CSV report.
    
- **A Redis invalidation pack** with two options:
    
    - **Pattern-scan deletes** (works with any existing keys).
        
    - **Epoch-based keys** (preferred, O(1) invalidation by bumping a counter—no scans).
        
    

---

## **1) Fusion A/B Harness (RRF vs. Weighted)**

  

> **Path:** tools/fusion_ab.py

> **Deps:** PyYAML (and qdrant-client only if you enable the optional live Qdrant mode)

```
#!/usr/bin/env python3
# tools/fusion_ab.py
"""
Fusion A/B harness: RRF vs Weighted fusion for vector + BM25 results.

Usage (offline JSONL inputs):
  python tools/fusion_ab.py \
    --queries tests/fixtures/baseline_query_set.yaml \
    --vector-jsonl results/vector.jsonl \
    --bm25-jsonl results/bm25.jsonl \
    --method rrf --topk 10 --rrf-k 60 \
    --out-md reports/fusion_ab_results.md --out-csv reports/fusion_ab_results.csv

Optional: Live Qdrant for vector, HTTP endpoint for BM25:
  python tools/fusion_ab.py \
    --queries tests/fixtures/baseline_query_set.yaml \
    --qdrant-url http://localhost:6333 --qdrant-collection chunks \
    --embed-provider jina --embed-model jina-embeddings-v3 --jina-api-key $JINA_API_KEY \
    --bm25-endpoint http://localhost:8080/search \
    --method weighted --alpha 0.6 \
    --out-md reports/fusion_ab_results.md

Inputs
------
queries YAML item:
- id: q1
  text: "How to set MTU?"
  # Either binary golds...
  gold_chunk_ids: ["chunk_a1b2c3", "chunk_d4e5f6"]
  # ...or graded relevance (preferred for nDCG)
  judgments:
    chunk_a1b2c3: 2   # 2=highly relevant, 1=partially relevant, 0=irrelevant
    chunk_zzzzzz: 1

vector.jsonl / bm25.jsonl: one JSON per line:
{"query_id":"q1","results":[{"id":"chunk_a1b2c3","score":0.83},{"id":"chunk_d4e5f6","score":0.72}]}

Outputs
-------
- Markdown summary with macro metrics per fusion method
- CSV with per-query metrics
"""

import argparse, json, math, os, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import yaml
from statistics import mean

# ---------- Data models ----------
@dataclass
class QueryItem:
    id: str
    text: str
    judgments: Dict[str, int]   # id -> relevance grade (0/1/2). If empty, fallback to gold set.
    gold_set: set               # set of relevant chunk_ids (binary golds)

@dataclass
class ResultItem:
    id: str
    score: float

# ---------- IO helpers ----------
def load_queries(path: str) -> List[QueryItem]:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    out = []
    for q in data:
        judg = q.get("judgments") or {}
        gold = set(q.get("gold_chunk_ids") or [])
        out.append(QueryItem(id=q["id"], text=q["text"], judgments=judg, gold_set=gold))
    return out

def load_jsonl(path: str) -> Dict[str, List[ResultItem]]:
    out: Dict[str, List[ResultItem]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["query_id"]
            results = [ResultItem(id=r["id"], score=float(r["score"])) for r in obj["results"]]
            out[qid] = results
    return out

# ---------- Normalization & fusion ----------
def minmax(scores: List[ResultItem]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = [r.score for r in scores]
    lo, hi = min(vals), max(vals)
    if math.isclose(hi, lo):
        return {r.id: 0.0 for r in scores}
    return {r.id: (r.score - lo) / (hi - lo) for r in scores}

def to_ranks(ids_in_order: List[str]) -> Dict[str, int]:
    # 1-based rank
    return {cid: i + 1 for i, cid in enumerate(ids_in_order)}

def fuse_rrf(vec_ranks: Dict[str, int], bm25_ranks: Dict[str, int], k: int = 60) -> Dict[str, float]:
    ids = set(vec_ranks) | set(bm25_ranks)
    fused = {}
    for cid in ids:
        rv = 1 / (k + vec_ranks.get(cid, 10_000))
        rb = 1 / (k + bm25_ranks.get(cid, 10_000))
        fused[cid] = rv + rb
    return fused

def fuse_weighted(vec_norm: Dict[str, float], bm25_norm: Dict[str, float], alpha: float = 0.6) -> Dict[str, float]:
    ids = set(vec_norm) | set(bm25_norm)
    fused = {}
    for cid in ids:
        fused[cid] = alpha * vec_norm.get(cid, 0.0) + (1 - alpha) * bm25_norm.get(cid, 0.0)
    return fused

# ---------- Metrics ----------
def dcg(gains: List[int]) -> float:
    return sum((g / math.log2(i + 2)) for i, g in enumerate(gains))

def ndcg_at_k(ranked_ids: List[str], judgments: Dict[str, int], k: int) -> float:
    if not judgments:
        return float("nan")
    gains = [judgments.get(cid, 0) for cid in ranked_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    if not ideal or sum(ideal) == 0:
        return float("nan")
    return dcg(gains) / dcg(ideal)

def hit_at_k(ranked_ids: List[str], gold: set, k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0

def mrr_at_k(ranked_ids: List[str], gold: set, k: int) -> float:
    if not gold:
        return float("nan")
    for i, cid in enumerate(ranked_ids[:k], 1):
        if cid in gold:
            return 1.0 / i
    return 0.0

# ---------- Optional live adapters ----------
def qdrant_vector_search_adapter(qdrant_url, api_key, collection, query_texts, embed_model, embed_provider, jina_api_key, top_k=20):
    """
    Returns: dict[query_id] -> List[ResultItem]
    NOTE: Requires qdrant-client and an embedding provider; kept minimal on purpose.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        import requests
    except Exception as e:
        raise RuntimeError("Install qdrant-client to use live adapter") from e

    def embed(texts: List[str]) -> List[List[float]]:
        if embed_provider == "jina":
            r = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {jina_api_key}", "Content-Type": "application/json"},
                json={"input": texts, "model": embed_model, "encoding_format": "float"},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            return [d["embedding"] for d in data["data"]]
        raise RuntimeError("Only provider=jina implemented in this tiny adapter")

    client = QdrantClient(url=qdrant_url, api_key=api_key or None)
    vectors = embed([qt for _, qt in query_texts])
    out = {}
    for (qid, _), vec in zip(query_texts, vectors):
        hits = client.search(collection_name=collection, query_vector=("content", vec), limit=top_k)
        out[qid] = [ResultItem(id=h.payload["id"], score=float(h.score)) for h in hits]
    return out

def http_bm25_adapter(endpoint: str, query_texts, top_k=50):
    """
    Minimal HTTP adapter: GET {endpoint}?q=<query>&k=<top_k> -> [{"id":..., "score":...}]
    Returns: dict[query_id] -> List[ResultItem]
    """
    import requests, urllib.parse
    out = {}
    for qid, text in query_texts:
        url = f"{endpoint}?q={urllib.parse.quote(text)}&k={top_k}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        items = r.json()
        out[qid] = [ResultItem(id=i["id"], score=float(i.get("score", 1.0))) for i in items]
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--vector-jsonl")
    ap.add_argument("--bm25-jsonl")
    ap.add_argument("--method", choices=["rrf", "weighted"], default="rrf")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", default=None)

    # Optional live modes
    ap.add_argument("--qdrant-url")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--qdrant-collection", default="chunks")
    ap.add_argument("--embed-provider", choices=["jina"], default=None)
    ap.add_argument("--embed-model", default="jina-embeddings-v3")
    ap.add_argument("--jina-api-key", default=None)
    ap.add_argument("--bm25-endpoint", default=None)

    args = ap.parse_args()
    queries = load_queries(args.queries)

    # Load results: offline or live
    vector_by_q: Dict[str, List[ResultItem]] = {}
    bm25_by_q: Dict[str, List[ResultItem]] = {}

    if args.vector-jsonl:
        vector_by_q = load_jsonl(args.vector-jsonl)
    elif args.qdrant_url and args.embed_provider:
        vector_by_q = qdrant_vector_search_adapter(
            args.qdrant_url, args.qdrant_api_key, args.qdrant_collection,
            [(q.id, q.text) for q in queries],
            args.embed_model, args.embed_provider, args.jina_api_key,
            top_k=max(2*args.topk, 20)
        )
    else:
        print("ERROR: Provide --vector-jsonl or live Qdrant options.", file=sys.stderr)
        sys.exit(2)

    if args.bm25-jsonl:
        bm25_by_q = load_jsonl(args.bm25-jsonl)
    elif args.bm25_endpoint:
        bm25_by_q = http_bm25_adapter(args.bm25_endpoint, [(q.id, q.text) for q in queries], top_k=max(2*args.topk, 50))
    else:
        print("ERROR: Provide --bm25-jsonl or --bm25-endpoint.", file=sys.stderr)
        sys.exit(2)

    # Evaluate
    rows = []
    agg = {"hit1": [], "hit3": [], "hit5": [], "mrr10": [], "ndcg10": []}
    for q in queries:
        vec = vector_by_q.get(q.id, [])
        kwd = bm25_by_q.get(q.id, [])

        # Prepare ranks/norms
        vec_ids = [r.id for r in vec]
        kwd_ids = [r.id for r in kwd]
        vec_ranks = to_ranks(vec_ids)
        kwd_ranks = to_ranks(kwd_ids)
        vec_norm = minmax(vec)
        kwd_norm = minmax(kwd)

        # Fuse
        if args.method == "rrf":
            fused_scores = fuse_rrf(vec_ranks, kwd_ranks, k=args.rrf_k)
        else:
            fused_scores = fuse_weighted(vec_norm, kwd_norm, alpha=args.alpha)

        ranked_ids = [cid for cid, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)]
        top_ids = ranked_ids[:args.topk]

        # Metrics (prefer graded judgments; fallback to gold set)
        if q.judgments:
            ndcg10 = ndcg_at_k(ranked_ids, q.judgments, k=min(10, len(ranked_ids)))
            # Convert graded judgments to binary gold for Hit/MRR: any grade >0 is relevant
            gold = {cid for cid, g in q.judgments.items() if g > 0}
        else:
            ndcg10 = float("nan")
            gold = q.gold_set

        hit1 = hit_at_k(ranked_ids, gold, 1)
        hit3 = hit_at_k(ranked_ids, gold, 3)
        hit5 = hit_at_k(ranked_ids, gold, 5)
        mrr10 = mrr_at_k(ranked_ids, gold, 10)

        for key, val in [("hit1", hit1), ("hit3", hit3), ("hit5", hit5), ("mrr10", mrr10), ("ndcg10", ndcg10)]:
            if not (isinstance(val, float) and math.isnan(val)):
                agg[key].append(val)

        rows.append({
            "query_id": q.id,
            "hit@1": f"{hit1:.3f}",
            "hit@3": f"{hit3:.3f}",
            "hit@5": f"{hit5:.3f}",
            "mrr@10": f"{mrr10:.3f}",
            "ndcg@10": ("NA" if math.isnan(ndcg10) else f"{ndcg10:.3f}")
        })

    # Write Markdown
    md = []
    md.append(f"# Fusion A/B Results — method={args.method.upper()}")
    md.append("")
    def avg(xs): return float("nan") if not xs else mean(xs)
    md.append(f"- **Hit@1**: {avg(agg['hit1']):.3f}")
    md.append(f"- **Hit@3**: {avg(agg['hit3']):.3f}")
    md.append(f"- **Hit@5**: {avg(agg['hit5']):.3f}")
    md.append(f"- **MRR@10**: {avg(agg['mrr10']):.3f}")
    ndcg_vals = [x for x in agg["ndcg10"] if not math.isnan(x)]
    md.append(f"- **nDCG@10**: {avg(ndcg_vals):.3f}" if ndcg_vals else "- **nDCG@10**: NA")
    md.append("")
    md.append("## Per-query")
    md.append("| query_id | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['query_id']} | {r['hit@1']} | {r['hit@3']} | {r['hit@5']} | {r['mrr@10']} | {r['ndcg@10']} |")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["query_id","hit@1","hit@3","hit@5","mrr@10","ndcg@10"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"Wrote {args.out_md}" + (f" and {args.out_csv}" if args.out_csv else ""))

if __name__ == "__main__":
    main()
```

### **Minimal JSONL generators (optional)**

  

If you need to dump results for the harness:

- **Vector via Qdrant** → tools/dump_qdrant_vector_jsonl.py (10–15 lines; similar to the live adapter).
    
- **BM25 via your HTTP search** → tools/dump_bm25_jsonl.py.
    

  

_(Ask if you want me to add them; the harness already supports live modes.)_

---

## **2) Redis Invalidation Pack**

  

Two patterns:

  

### **A) Pattern‑scan deletes (works with any keys)**

  

> **Path:** tools/redis_invalidation.py

```
#!/usr/bin/env python3
# tools/redis_invalidation.py
"""
Pattern-scan invalidation for doc- and chunk-scoped caches.

Recommended patterns if you follow the key scheme below:
- {ns}:search:doc:{document_id}:*
- {ns}:fusion:doc:{document_id}:*
- {ns}:answer:doc:{document_id}:*
- {ns}:vector:chunk:{id}:*
- {ns}:bm25:doc:{document_id}:*
- custom extra patterns are supported

Usage:
  python tools/redis_invalidation.py --redis-url redis://localhost:6379/0 \
     --namespace rag:v1 --doc-id doc_abc123 \
     --chunks chunk_a1b2c3 chunk_d4e5f6 \
     --extra-pattern "{ns}:render:doc:{document_id}:*"
"""
import argparse, os
import redis

def scan_delete(r: "redis.Redis", pattern: str, batch=1000) -> int:
    deleted = 0
    for key in r.scan_iter(match=pattern, count=batch):
        deleted += r.delete(key)
    return deleted

def invalidate(redis_url: str, namespace: str, document_id: str,
               chunk_ids: list[str] | None = None,
               extra_patterns: list[str] | None = None) -> int:
    r = redis.Redis.from_url(redis_url)
    chunk_ids = chunk_ids or []
    patterns = [
        f"{namespace}:search:doc:{document_id}:*",
        f"{namespace}:bm25:doc:{document_id}:*",
        f"{namespace}:fusion:doc:{document_id}:*",
        f"{namespace}:answer:doc:{document_id}:*",
    ] + [f"{namespace}:vector:chunk:{cid}:*" for cid in chunk_ids]

    if extra_patterns:
        for p in extra_patterns:
            patterns.append(p.format(ns=namespace, document_id=document_id))

    total = 0
    for p in set(patterns):
        total += scan_delete(r, p)
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--namespace", default=os.getenv("CACHE_NS", "rag:v1"))
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--chunks", nargs="*", default=[])
    ap.add_argument("--extra-pattern", action="append", default=[])
    args = ap.parse_args()

    deleted = invalidate(args.redis_url, args.namespace, args.document_id, args.chunks, args.extra_pattern)
    print(f"Deleted {deleted} keys for doc={args.document_id}")

if __name__ == "__main__":
    main()
```

**Where to call it:** after a successful re‑ingest/upsert of a document, pass the document_id and the set of chunk_ids you just wrote.

---

### **B) Epoch‑based keys (preferred; O(1) invalidation)**

  

Instead of deleting old keys, **bake an epoch** into every cache key. Invalidate by bumping a counter—no scans, no deletes.

  

**Key scheme (examples):**

- Search/fusion cache key:
    

```
{ns}:fusion:doc:{document_id}:epoch:{doc_epoch}:q:{sha1(query_text)}
```

-   
    
- Chunk cache key:
    

```
{ns}:vector:chunk:{id}:epoch:{chunk_epoch}
```

  

  

**Epoch store:**

- HSET {ns}:doc_epoch {document_id} <int>
    
- HSET {ns}:chunk_epoch {id} <int>
    

  

**Lookup when caching:**

```
import hashlib, redis
r = redis.Redis.from_url(REDIS_URL)
ns = "rag:v1"

def doc_epoch(document_id: str) -> str:
    v = r.hget(f"{ns}:doc_epoch", document_id)
    return (v.decode() if v else "0")

def fusion_cache_key(document_id: str, query_text: str) -> str:
    e = doc_epoch(document_id)
    qh = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:16]
    return f"{ns}:fusion:doc:{document_id}:epoch:{e}:q:{qh}"
```

**Invalidate after re‑ingest (bump epoch):**

  

> **Path:** tools/redis_epoch_bump.py

```
#!/usr/bin/env python3
import argparse, os, redis

def bump_doc_epoch(r, ns: str, document_id: str) -> int:
    # Atomically increment
    return r.hincrby(f"{ns}:doc_epoch", document_id, 1)

def bump_chunk_epochs(r, ns: str, chunk_ids: list[str]) -> int:
    pipe = r.pipeline()
    for cid in chunk_ids:
        pipe.hincrby(f"{ns}:chunk_epoch", cid, 1)
    res = pipe.execute()
    return sum(int(x) for x in res)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL","redis://localhost:6379/0"))
    ap.add_argument("--namespace", default=os.getenv("CACHE_NS","rag:v1"))
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--chunks", nargs="*", default=[])
    args = ap.parse_args()

    r = redis.Redis.from_url(args.redis_url)
    de = bump_doc_epoch(r, args.namespace, args.document_id)
    ce = bump_chunk_epochs(r, args.namespace, args.chunks)
    print(f"Bumped doc_epoch to {de} and {ce} chunk_epoch entries")

if __name__ == "__main__":
    main()
```

**Why this is better:** you avoid SCAN sweeps entirely; old keys naturally die when their epoch is no longer used for reads/writes. You can still set TTLs to reclaim space faster.

---

## **3) Recommended Minimal Key Namespace (so both snippets “just work”)**

  

If you don’t already have a scheme, adopt this (or map your keys to it):

```
{ns} = "rag:v1"

# Query/result layer
{ns}:search:doc:{document_id}:q:{sha1(query)}                -> raw search output (optional)
{ns}:bm25:doc:{document_id}:q:{sha1(query)}                  -> bm25-only results (optional)
{ns}:fusion:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}     -> fused result ids & scores
{ns}:answer:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}     -> stitched context blob / final answer cache

# Unit caches
{ns}:vector:chunk:{id}:epoch:{chunk_epoch}        -> vector or re-ranking features
{ns}:chunk:meta:{id}                              -> light metadata (if you cache it)

# Epoch state
{ns}:doc_epoch                                         -> hash: document_id -> epoch int
{ns}:chunk_epoch                                       -> hash: id -> epoch int
```

- **Pattern-scan deletes** will target the search, fusion, answer, and vector spaces for a doc/chunk.
    
- **Epoch-based** will just **bump** doc_epoch and/or chunk_epoch.
    

---

## **4) Wire-up: where to call invalidation**

  

After your **Neo4j + Qdrant upserts** complete for a document:

```
# pseudo in your ingestion command
from tools.redis_invalidation import invalidate as redis_invalidate   # if using pattern-scan
# OR
# from tools.redis_epoch_bump import bump_doc_epoch, bump_chunk_epochs

updated_chunk_ids = [c.id for c in chunks]  # combined/split chunks you just wrote

if CACHE_INVALIDATION_MODE == "scan":
    deleted = redis_invalidate(REDIS_URL, "rag:v1", document_id, updated_chunk_ids)
    logger.info(f"cache invalidation (scan) deleted={deleted} document_id={document_id}")
else:
    # epoch bump (preferred)
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    bump_doc_epoch(r, "rag:v1", document_id)
    bump_chunk_epochs(r, "rag:v1", updated_chunk_ids)
    logger.info(f"cache invalidation (epoch) bumped for document_id={document_id}, chunks={len(updated_chunk_ids)}")
```

> If you serve multi-document queries, consider also **global epoch** ({ns}:global_epoch) you bump when a **large batch** re‑ingest completes, and include it in the cache key too. That gives you a single knob to invalidate everything.

---

## **5) Add these to** 

## **requirements.txt**

```
PyYAML>=6.0.2
redis>=5.0.6
qdrant-client>=1.8.2  # only if you plan to use live vector mode in the harness
```

---

## **6) Example commands**

  

**RRF run (offline files):**

```
python tools/fusion_ab.py \
  --queries tests/fixtures/baseline_query_set.yaml \
  --vector-jsonl results/vector.jsonl \
  --bm25-jsonl results/bm25.jsonl \
  --method rrf --rrf-k 60 --topk 10 \
  --out-md reports/fusion_rrf.md --out-csv reports/fusion_rrf.csv
```

**Weighted run (live Qdrant + HTTP BM25):**

```
python tools/fusion_ab.py \
  --queries tests/fixtures/baseline_query_set.yaml \
  --qdrant-url http://localhost:6333 --qdrant-collection chunks \
  --embed-provider jina --embed-model jina-embeddings-v3 --jina-api-key $JINA_API_KEY \
  --bm25-endpoint http://localhost:8080/search \
  --method weighted --alpha 0.6 --topk 10 \
  --out-md reports/fusion_weighted.md
```

---

### **That’s it**

  

This gives you:

- A **repeatable A/B harness** to validate RRF vs weighted fusion on your query set (and to tune alpha, rrf_k, and topk).
    
- A **robust cache invalidation** strategy you can adopt immediately (pattern‑scan) and a **future‑proof epoch scheme** you can migrate to with minimal code changes.
    

  

If you want me to also add the **Qdrant/BM25 JSONL dumpers**, I can include those next so your team can produce offline inputs with one command.