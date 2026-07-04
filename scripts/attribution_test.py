#!/usr/bin/env python3
"""Controlled 3-run attribution test. Run ON the GPU node."""

import http.client
import json
import os
import subprocess
import time

QUERIES = [
    ("1-META", "how is metadata managed and architected on a weka cluster"),
    (
        "2-SIZE",
        "How do I appropriately size the weka drives, compute, and frontends containers?",
    ),
    ("3-CLI", "weka cluster run command --force"),
    ("4-PROC", "how to install WEKA cluster"),
    ("5-REF", "what is WEKA deduplication"),
]

REPO = "/home/bgconley/wekadocs-matrix"
YAML = REPO + "/config/development.yaml"
OUT_DIR = REPO + "/reports/retrieval_diagnostics/2026-03-07-attribution"
os.makedirs(OUT_DIR, exist_ok=True)


def sed(pattern, replacement):
    subprocess.run(
        ["sed", "-i", "s|{}|{}|".format(pattern, replacement), YAML], check=True
    )


def restart_and_wait():
    subprocess.run(
        ["docker", "compose", "restart", "mcp-server"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    for attempt in range(20):
        time.sleep(2)
        try:
            c = http.client.HTTPConnection("localhost", 8000, timeout=3)
            c.request("GET", "/health")
            r = c.getresponse()
            if r.status == 200:
                r.read()
                print("  Server ready after {}s".format((attempt + 1) * 2))
                return
        except Exception:
            pass
    raise RuntimeError("Server did not start")


def mcp_call(question):
    conn = http.client.HTTPConnection("localhost", 8000, timeout=120)
    hdrs = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    init = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "attr", "version": "1.0"},
            },
        }
    )
    conn.request("POST", "/_mcp/", init, hdrs)
    r = conn.getresponse()
    sid = r.getheader("mcp-session-id")
    r.read()
    hdrs["mcp-session-id"] = sid
    conn.request(
        "POST",
        "/_mcp/",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        hdrs,
    )
    conn.getresponse().read()
    conn.request(
        "POST",
        "/_mcp/",
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "kb_retrieve_evidence",
                    "arguments": {"question": question, "top_k": 20, "max_quotes": 10},
                },
            }
        ),
        hdrs,
    )
    return json.loads(conn.getresponse().read().decode())


def save_traces(run_name, profile_val, spec_val):
    run_dir = OUT_DIR + "/" + run_name
    os.makedirs(run_dir, exist_ok=True)
    res = subprocess.run(
        ["docker", "logs", "weka-mcp-server"], capture_output=True, text=True
    )
    traces = {}
    for line in res.stdout.split("\n"):
        try:
            obj = json.loads(line.strip())
            cid = obj.get("correlation_id")
            if cid:
                traces.setdefault(cid, []).append(obj)
        except Exception:
            pass
    saved = 0
    for cid, evts in traces.items():
        for e in evts:
            if e.get("event") == "retrieval_started":
                qry = e.get("query", "")
                for _, q in QUERIES:
                    if q[:25].lower() in qry.lower():
                        slug = qry.replace(" ", "_")[:40]
                        p = "{}/{}_{}.json".format(run_dir, cid[:8], slug)
                        if not os.path.exists(p):
                            with open(p, "w") as f:
                                json.dump(
                                    {
                                        "correlation_id": cid,
                                        "run": run_name,
                                        "profile": profile_val,
                                        "specificity": spec_val,
                                        "events": evts,
                                    },
                                    f,
                                    indent=2,
                                )
                            saved += 1
                        break
                break
    print("  Saved {} traces to {}/".format(saved, run_dir))


def run_queries(run_name):
    for tag, q in QUERIES:
        r = mcp_call(q)
        sc = r["result"]["structuredContent"]
        print("\n  {}:".format(tag))
        for i, qt in enumerate(sc["quotes"], 1):
            h = (qt.get("heading") or qt.get("title") or "")[:55]
            d = qt.get("doc_tag", "")
            c = qt.get("confidence") or "?"
            print("    #{:2d}  conf={:8}  doc={:30}  {}".format(i, str(c), d, h))


# --- Run 1: Legacy (profile=null) ---
print("\n" + "=" * 60)
print("RUN 1: Legacy (profile=null)")
print("=" * 60)
# Comment out profile line
sed('    profile: "graph_assisted"', "    # profile: null")
sed("    profile_overrides:", "    # profile_overrides:")
sed(
    "      use_specificity_adjustment: true",
    "      # use_specificity_adjustment: false",
)
restart_and_wait()
run_queries("run1_legacy")
save_traces("run1_legacy", None, False)

# --- Run 2: graph_assisted, specificity OFF ---
print("\n" + "=" * 60)
print("RUN 2: graph_assisted, specificity OFF")
print("=" * 60)
# Restore profile, set specificity false
sed("    # profile: null", '    profile: "graph_assisted"')
sed("    # profile_overrides:", "    profile_overrides:")
sed(
    "      # use_specificity_adjustment: false",
    "      use_specificity_adjustment: false",
)
restart_and_wait()
run_queries("run2_graph_no_spec")
save_traces("run2_graph_no_spec", "graph_assisted", False)

# --- Run 3: graph_assisted, specificity ON ---
print("\n" + "=" * 60)
print("RUN 3: graph_assisted, specificity ON")
print("=" * 60)
sed("      use_specificity_adjustment: false", "      use_specificity_adjustment: true")
restart_and_wait()
run_queries("run3_graph_with_spec")
save_traces("run3_graph_with_spec", "graph_assisted", True)

# --- Restore ---
subprocess.run(["git", "checkout", YAML], cwd=REPO, check=True)
restart_and_wait()
print("\nRestored to committed config (graph_assisted + specificity).")
