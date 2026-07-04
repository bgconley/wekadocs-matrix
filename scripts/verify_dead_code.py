#!/usr/bin/env python3
"""Verify dead code claims from reports/codepath-map.txt against actual imports."""

import os
import re

# ── Extract claims ──
with open("reports/codepath-map.txt") as f:
    raw = f.read()

dead_modules = re.findall(r"\[DEAD\]\s+(src\.[\w.]+)", raw)
dead_functions = re.findall(r"\[DEAD\]\s+(src\.[\w.]+(?:\.[\w.]+)?)", raw)

# De-duplicate and separate modules from functions
dead_modules_set = set()
dead_functions_set = set()
for token in dead_functions:
    parts = token.split(".")
    # Heuristic: if the last component has an uppercase first letter, treat as class.method
    # Otherwise treat as module (if it's all lowercase or ends in py)
    if any(c.isupper() for c in parts[-1] if c != "_"):
        dead_functions_set.add(token)
    else:
        dead_modules_set.add(token)

# Remove functions from module list (subsumption)
for mod in list(dead_modules_set):
    if any(f.startswith(mod + ".") for f in dead_functions_set):
        dead_modules_set.discard(mod)

print(f"Modules to verify: {len(dead_modules_set)}")
print(f"Functions to verify: {len(dead_functions_set)}")


# ── Search for imports ──
def find_imports(path: str) -> list[str]:
    """Return files that import `path` (dotted or slash form)."""
    results = []
    dotted = path.replace(".", "/")
    base = dotted.split("/")[-1]
    for root, dirs, files in os.walk(path="src/"):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as fh:
                        content = fh.read()
                    # Match import patterns
                    if re.search(
                        rf"import\s+src\.{dotted.replace('/', '.')}|from\s+src\.{dotted.replace('/', '.')}",
                        content,
                    ):
                        results.append(fpath)
                    # Match "from src.module import ClassName"
                    if re.search(r"from\s+src\.([\w.]+)\s+import", content):
                        pass  # already caught
                except:
                    pass
    return results


print("\n--- MODULE VERIFICATION ---")
for mod in sorted(dead_modules_set):
    refs = find_imports(mod)
    status = "LIVE" if refs else "DEAD"
    print(f"[{status}] {mod}")
    if refs:
        for r in refs:
            print(f"  └─ {r}")

print("\n--- FUNCTION VERIFICATION (sample) ---")
# For functions, check if they appear in any non-DEAD context
for func in sorted(dead_functions_set)[:50]:
    dotted = func.replace(".", "/")
    found_in = []
    for root, dirs, files in os.walk("src/"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as fh:
                        content = fh.read()
                    if re.search(re.escape(func.split(".")[-1]), content):
                        found_in.append(fpath)
                except:
                    pass
    # Check tests too
    test_refs = []
    if os.path.exists("tests"):
        for root, dirs, files in os.walk("tests"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath) as fh:
                            content = fh.read()
                        if re.search(re.escape(func.split(".")[-1]), content):
                            test_refs.append(fpath)
                    except:
                        pass
    status = "MAYBE-LIVE" if found_in or test_refs else "DEAD"
    print(f"[{status}] {func}")
    if found_in:
        for r in found_in[:3]:
            print(f"  └─ src: {r}")
    if test_refs:
        for r in test_refs[:3]:
            print(f"  └─ test: {r}")
