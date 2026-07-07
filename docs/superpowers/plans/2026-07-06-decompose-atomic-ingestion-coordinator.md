# Decompose AtomicIngestionCoordinator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> This is **Slice C** of the Refactor-Roadmap wrap-up (`docs/repo_audit/10_refactor_roadmap.md`, Phase 4a: "Split AtomicIngestionCoordinator into parse/chunk/enrich/embed/write/link stages; convert direct fallback branches into explicit IngestionTrace events").
>
> **DEPENDENCY: run Slice B (legacy quarantine) first.** Slice B removes `patched/atomic_patched_v3.py` (a 3,859-LOC orphan duplicate of this class), without which GitNexus symbol resolution for `AtomicIngestionCoordinator` is ambiguous.
>
> **PREREQUISITE: re-index GitNexus before starting.** The index is stale (indexed at `cbdb11b`, HEAD is well ahead). Run `npx gitnexus analyze` first, or every `gitnexus impact` step below reports against old code. Do this AFTER Slice B removes `patched/` so the re-index is collision-free.

**Goal:** Turn `src/ingestion/atomic.py` (2,417 lines) into a thin orchestrator by lifting each inline stage into `src/ingestion/stages/*`, with **zero behavior change per commit**, guarded by an offline characterization pin.

**Architecture:** Strangler-fig decomposition mirroring the successful `3fa1bb9` precedent (which already extracted the Neo4j/Qdrant writers with backward-compat re-exports). Extract **leaf-first** — `link → parse → chunk → enrich → write → embed` — keeping the coordinator's public signature frozen and a re-export for every moved symbol. A new offline pin (fake Neo4j/Qdrant + stub embedder) is the fast inner-loop gate; the live docker integration suite is the outer gate.

**Tech Stack:** Python 3.11, pytest with fakes, existing `Neo4jWriter`/`QdrantWriter`/`chunk_assembler`/`parsers` collaborators, GitNexus.

**Execution note:** Create/overwrite files with the Write tool and edit with Edit (apply_patch-style), not shell heredoc redirection. **Commit messages must pass gitlint** (enforced at `commit-msg`): title ≤72 chars with a `(pN.N)` numeric scope, a blank line, then a `Phase N.N — <recap>` body whose lines are ≤80 chars. The `git commit`/`Commit:` examples are shown title-first — append the `-m "Phase 4.0 — …"` body when committing.

---

## Critical Context (read before starting)

- **There is NO offline behavior pin today.** `tests/test_atomic_ingestion.py` (1,152 LOC) was deleted in `3fa1bb9` and never replaced. Every remaining coordinator test needs **live** Neo4j+Qdrant(+GLiNER) via docker (`tests/conftest.py` line 1: "NO MOCKS"). **Task 0 builds the offline pin and is a blocking prerequisite for all extraction.**
- **Risk is MEDIUM, not the LOW GitNexus reports.** The sole runtime caller (`src/ingestion/worker.py:231`) constructs the class dynamically, so the static graph shows `impactedCount: 0`. In reality this is the *entire ingest path*, writing to two stores transactionally with compensation — a regression corrupts the Neo4j/Qdrant consistency invariant this module exists to guarantee.
- **Frozen public contract (must NOT change):**
  1. `AtomicIngestionCoordinator(neo4j_driver, qdrant_client, config, *, validate_before_commit=True, strict_mode=None)`
  2. `.ingest_document_atomic(source_uri, content, format="markdown", *, embedding_model=None, embedding_version=None) -> AtomicIngestionResult`
  3. `AtomicIngestionResult` shape + `.to_dict()` + `.stats` keys
  4. module-level `ingest_document_atomic(...) -> dict`
  5. the 6 backward-compat re-exports at `atomic.py:113-130` (`Neo4jWriter`, `QdrantWriter`, `ALLOWED_ENTITY_RELATIONSHIP_TYPES`, `IngestionValidator`, `SagaContext`, `ValidationResult`) — tests import these from `src.ingestion.atomic`.
- **#1 correctness trap — shared mutable `sections` state across stages.** Sections dicts are mutated in place and carry cross-stage keys: `_mentions` (set in orchestrator ~480-560, read in embed ~1710 and write ~2100), `_embedding_text` (set in enrich, read/stripped in embed ~1335), `was_truncated`/`token_count` (embed sets, orchestrator Phase 3b @735 re-reads, write re-reads). Extraction must pass these forward explicitly or preserve the in-place-mutation contract; a silent drop desyncs the Neo4j payload from the Qdrant vectors.
- **Do NOT** attempt to decouple `_compute_embeddings` from `builder` privates (`builder._build_section_text_for_embedding`, `builder.embedder`, `builder.embedding_plan/settings/dims`) in this slice — keep `builder` as a parameter. A later phase can define an `Embedder` protocol.
- **Lazy imports are load-bearing.** Collaborators are imported *inside* methods to avoid import cycles (see `src/ingestion/__init__.py`). New `stages/*` modules must replicate lazy-import discipline.

---

## File Structure

Create a new stage package (does not exist yet):

- Create: `src/ingestion/stages/__init__.py` (re-exports stage entrypoints)
- Create: `src/ingestion/stages/trace.py` (`IngestionTrace`, `IngestionTraceEvent` — modeled on `src/mcp_server/retrieval_trace.py:71`)
- Create: `src/ingestion/stages/link.py`
- Create: `src/ingestion/stages/parse.py`
- Create: `src/ingestion/stages/chunk.py`
- Create: `src/ingestion/stages/enrich.py`
- Create: `src/ingestion/stages/write.py`
- Create: `src/ingestion/stages/embed.py`

Modify (progressively slim to a facade):

- Modify: `src/ingestion/atomic.py` — keep `AtomicIngestionResult`, `__init__`, the public `ingest_document_atomic` orchestrator (becomes a ~120-line stage-call sequence), the module convenience fn, and the 6 re-exports.

Test:

- Create: `tests/ingestion/test_atomic_characterization.py` (the offline pin)

### Explicit Out Of Scope

- Do not change any of the 5 frozen contract items.
- Do not decouple `builder` privates (keep `builder` param in `embed.py`).
- Do not edit `patched/atomic_patched_v3.py` (Slice B removes it).
- Do not fix retrieval/query code; this is ingest-side only.

---

## Task 0: Build The Offline Characterization Pin (BLOCKING PREREQUISITE)

**Files:**
- Create: `tests/ingestion/test_atomic_characterization.py`

- [ ] **Step 1: Write the offline pin test**

Author a deterministic orchestration pin for the current `ingest_document_atomic`
facade. The test must use the real public method, but monkeypatch the three
internal stages (`_prepare_ingestion`, `_compute_embeddings`,
`_execute_atomic_saga`) so it proves the coordinator contract without docker,
Neo4j, Qdrant, live embedders, or fragile fake-driver archaeology.

This is intentionally not an import/missing-module RED. During extraction, the
test should fail only when the stage order, shared mutable `sections` contract,
embedding handoff, or `AtomicIngestionResult` shape regresses.

```python
# tests/ingestion/test_atomic_characterization.py
"""Offline behavior pin for AtomicIngestionCoordinator orchestration."""
from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.atomic import AtomicIngestionCoordinator
from src.shared.config import get_config


class DummyDriver:
    def session(self, **_):
        raise AssertionError("offline pin must not open Neo4j sessions")


class DummyQdrant:
    pass


def test_atomic_ingestion_orchestrates_prepare_embed_saga_and_result_contract(monkeypatch):
    config = get_config().model_copy(deep=True)
    coordinator = AtomicIngestionCoordinator(
        DummyDriver(),
        DummyQdrant(),
        config,
        validate_before_commit=False,
    )
    calls = []

    document = {
        "id": "doc-nutanix-files",
        "doc_id": "doc-nutanix-files",
        "title": "Nutanix Files snapshots",
        "total_tokens": 12,
    }
    sections = [
        {
            "id": "chunk-1",
            "document_id": "doc-nutanix-files",
            "text": "Nutanix Files supports NFS and SMB shares.",
            "token_count": 12,
            "original_section_ids": ["source-section-1"],
            "_mentions": [{"entity_id": "gliner-files", "source": "gliner"}],
        }
    ]
    entities = {
        "struct-files": {"id": "struct-files", "name": "Nutanix Files"},
        "gliner-files": {"id": "gliner-files", "name": "Nutanix Files"},
    }
    mentions = [
        {
            "section_id": "source-section-1",
            "entity_id": "struct-files",
            "source": "structural",
        }
    ]
    references = [{"source_chunk_id": "chunk-1", "target_hint": "Prism Central"}]
    builder = SimpleNamespace(name="fake-builder")

    def fake_prepare(
        source_uri,
        content,
        fmt,
        *,
        embedding_model=None,
        embedding_version=None,
    ):
        calls.append(("prepare", source_uri, fmt, embedding_model, embedding_version))
        assert "Nutanix Files" in content
        return {
            "document": document,
            "sections": sections,
            "entities": entities,
            "mentions": mentions,
            "references": references,
            "builder": builder,
        }

    def fake_compute(doc, stage_sections, stage_entities, stage_builder):
        calls.append(("embed", doc["id"], [s["id"] for s in stage_sections]))
        assert stage_builder is builder
        assert stage_entities is entities
        merged_ids = {m["entity_id"] for m in stage_sections[0]["_mentions"]}
        assert merged_ids == {"gliner-files", "struct-files"}
        return {
            "sections": {
                "chunk-1": {
                    "content": [0.1, 0.2, 0.3],
                    "sparse": {"indices": [1], "values": [0.5]},
                    "colbert": [[0.1, 0.2]],
                }
            },
            "stats": {"sparse_coverage": 1.0},
        }

    def fake_saga(**kwargs):
        calls.append(("saga", kwargs["document"]["id"], len(kwargs["sections"])))
        assert kwargs["references"] is references
        assert kwargs["embeddings"]["sections"]["chunk-1"]["content"] == [0.1, 0.2, 0.3]
        return {
            "success": True,
            "stats": {
                "sections_upserted": 1,
                "entities_upserted": 2,
                "vectors_upserted": 1,
                "cross_doc_edges": 0,
            },
        }

    monkeypatch.setattr(coordinator, "_prepare_ingestion", fake_prepare)
    monkeypatch.setattr(coordinator, "_compute_embeddings", fake_compute)
    monkeypatch.setattr(coordinator, "_execute_atomic_saga", fake_saga)

    result = coordinator.ingest_document_atomic(
        "nutanixdocs://files/snapshots",
        "# Nutanix Files\n\nNFS and SMB snapshot workflow.",
        embedding_model="test-model",
        embedding_version="test-version",
    )

    assert result.success is True
    assert result.document_id == "doc-nutanix-files"
    assert result.neo4j_committed is True
    assert result.qdrant_committed is True
    assert result.stats["sections_upserted"] == 1
    assert result.stats["vectors_upserted"] == 1
    assert calls == [
        (
            "prepare",
            "nutanixdocs://files/snapshots",
            "markdown",
            "test-model",
            "test-version",
        ),
        ("embed", "doc-nutanix-files", ["chunk-1"]),
        ("saga", "doc-nutanix-files", 1),
    ]
```

> This pin deliberately stops at the coordinator boundary. It was derived after
> a full read of `src/ingestion/atomic.py`: the public method owns prepare →
> mention merge/filter → embed → token-total adjustment → saga → result shaping.
> The extraction series can add deeper stage-specific tests, but this first pin
> must stay fast and deterministic so every commit can re-run it.

- [ ] **Step 2: Run the pin against current code and snapshot expectations**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
pytest tests/ingestion/test_atomic_characterization.py -q
```

Expected: PASS. If it fails, adjust the fakes/fixture until it faithfully exercises the current `ingest_document_atomic` and passes — this green run is the baseline every later task must preserve.

- [ ] **Step 3: Record the live outer gate**

Confirm the two live end-to-end tests exist and record how to run them (docker required):

```bash
rg -l "ingest_document_atomic|AtomicIngestionCoordinator" tests/integration/test_phase1_entity_edges.py tests/integration/test_gliner_ingestion_flow.py
```

Note in the task log: these run only with Neo4j+Qdrant(+GLiNER) up (`SKIP_INTEGRATION_TESTS`/`@pytest.mark.live`). Run them once **before** Task 2 and once **after** Task 9 as the outer gate.

- [ ] **Step 4: Commit the pin alone**

```bash
git add tests/ingestion/test_atomic_characterization.py
git commit -m "test(p4.0): offline characterization pin for atomic ingestion" \
  -m "Phase 4.0 — offline pin (fake Neo4j/Qdrant + stub embedder) for atomic ingest."
```

---

## Task 1: Scaffold `stages/` + IngestionTrace (no extraction yet)

**Files:**
- Create: `src/ingestion/stages/__init__.py`, `src/ingestion/stages/trace.py`

- [ ] **Step 1: Add `IngestionTrace` modeled on the retrieval trace**

Create `stages/trace.py` with `IngestionTraceEvent` (dataclass: `stage`, `kind`, `message`, `data`) and `IngestionTrace` (accumulates events, `.add_event(...)`, `.to_dict()`), mirroring `src/mcp_server/retrieval_trace.py`'s builder shape. No wiring into `atomic.py` yet.

- [ ] **Step 2: Empty stage-package init + run the pin**

`stages/__init__.py` re-exports whatever stage entrypoints exist so far (initially just trace). Run:

```bash
pytest tests/ingestion/test_atomic_characterization.py -q
python -m compileall -q src/ingestion
```

Expected: pin still PASS (nothing wired yet), compile clean.

- [ ] **Step 3: Commit**

```bash
git add src/ingestion/stages/__init__.py src/ingestion/stages/trace.py
git commit -m "feat(p4.0): scaffold ingestion stages package + IngestionTrace" \
  -m "Phase 4.0 — add stages/ package + IngestionTrace; no extraction wired yet."
```

---

## Tasks 2–7: Extract Stages Leaf-First

**Every extraction task follows the same 5 steps** (this is the strangler loop):

1. **GitNexus impact** on the method being moved: `npx gitnexus impact <method> --direction upstream --include-tests --repo wekadocs-matrix` — report blast radius (expected LOW/internal; the real gate is the pin).
2. **Extract** the method body into the new `stages/<name>.py` behind the stable interface below; leave a thin delegating call in `atomic.py` (or call the stage fn directly from the orchestrator). Preserve the shared-`sections` mutation contract exactly.
3. **Re-run the offline pin** — must stay green (byte-for-byte behavior).
4. **`compileall` + ruff** on `src/ingestion/`.
5. **Commit** (one stage per commit) using the two-`-m` form — `-m "<title>"` then `-m "Phase 4.0 — <recap>"` — since gitlint B8 requires a `Phase 4.0` body line (an empty body fails the commit-msg hook).

- [ ] **Task 2 — Extract LINK.** Move `_create_cross_doc_links` (259-389) + `_get_document_count` (248-257) → `stages/link.py::create_cross_doc_links(neo4j, qdrant, config, document, sections, embeddings, *, trace) -> LinkStats`. Least-coupled (post-commit, never fails ingest). Commit (title / body): `refactor(p4.0): extract link stage` / `Phase 4.0 — extract link stage to stages/link.py; pin stays green`.

- [ ] **Task 3 — Extract PARSE.** Move `_prepare_ingestion` parse+doc_tag/category/scope block (855-925) → `stages/parse.py::parse_document(source_uri, content, format, config) -> ParsedDoc`. Thin over `parsers/`. Commit (title / body): `refactor(p4.0): extract parse stage` / `Phase 4.0 — extract parse stage to stages/parse.py; pin stays green`.

- [ ] **Task 4 — Extract CHUNK.** Move the assemble + token-totals block (987-1010) → `stages/chunk.py::assemble_chunks(document, sections, config) -> list[section]`. Thin over `chunk_assembler.get_chunk_assembler`. Commit (title / body): `refactor(p4.0): extract chunk stage` / `Phase 4.0 — extract chunk stage to stages/chunk.py; pin stays green`.

- [ ] **Task 5 — Extract ENRICH (trickiest).** Consolidate the ENRICH logic currently smeared across 3 methods: `extract_entities`+references+GLiNER (924-1030), orchestrator mention merge/dedup/filter (480-560), and `entity_id_to_name` (1067-1108) → `stages/enrich.py` with `extract_and_enrich(...)` + `merge_section_mentions(...)`. **Extend the pin** to assert mention counts and entity-prune parity before/after. Commit (title / body): `refactor(p4.0): extract enrich stage` / `Phase 4.0 — extract enrich stage to stages/enrich.py; pin stays green`.

- [ ] **Task 6 — Extract WRITE.** Move `_execute_atomic_saga` (1981-2334) → `stages/write.py::execute_saga(*, document, sections, entities, mentions, references, embeddings, neo4j_writer, qdrant_writer, config, trace) -> SagaResult`, injecting `self.neo4j_writer`/`self.qdrant_writer`. **Preserve the deferred-commit ordering exactly:** Neo4j tx open → writes → Qdrant upsert → commit Neo4j *after* Qdrant → compensation (rollback Neo4j / delete Qdrant points, 2258-2330) with `written_qdrant_points`/`written_neo4j_chunks` bookkeeping. Run the pin **and the live integration gate**. Commit (title / body): `refactor(p4.0): extract write saga stage` / `Phase 4.0 — extract write saga to stages/write.py; commit order preserved`.

- [ ] **Task 7 — Extract EMBED (the prize, do last).** Move all of `_compute_embeddings` (1042-1979), including nested `_strip_embedding_context` (1317) and `_embed_sparse_safe` (1543), → `stages/embed.py::compute_embeddings(document, sections, entities, builder, config, *, trace) -> EmbeddingBundle`. **Keep `builder` as a param** (reaches builder privates — behavior-preserving). Run the pin **and the live gate**. Commit (title / body): `refactor(p4.0): extract embed stage` / `Phase 4.0 — extract embed stage to stages/embed.py; builder kept as param`.

---

## Task 8: Wire IngestionTrace (behavior-additive — isolate from extraction)

**Files:** Modify each `stages/*.py` + `atomic.py`.

- [ ] **Step 1: Replace silent fallbacks with trace events (keep the logs)**

For each enumerated fallback branch, add `trace.add_event(stage=..., kind="fallback", message=..., data=...)` **alongside** the existing `logger.warning` (belt-and-suspenders). The branches: sparse/doc_title/title/entity-sparse/colbert None-placeholder insertions (embed 1602/1634/1650/1726/1744), `non_stub_content_chunk_missing_sparse_vector` (~1912), `embedding_dims=1024` default (1184), doc_title-from-id (1264), `doc_tag_extraction_fallback` (908), `gliner_enrichment_failed_non_blocking` (1021), `entity_missing_name_field_using_fallback` (1091), cross-doc `skipped/error` returns (300-389), `_get_document_count` except→0 (256), and the saga compensation branches.

- [ ] **Step 2: Attach the trace to the result**

Set `AtomicIngestionResult.stats["trace"] = trace.to_dict()`. Extend the offline pin to assert the trace field is present and records at least one event on a fallback-triggering fixture.

- [ ] **Step 3: Run pin + compile + commit**

```bash
pytest tests/ingestion/test_atomic_characterization.py -q
git commit -am "feat(p4.0): emit IngestionTrace events for ingest fallbacks" \
  -m "Phase 4.0 — emit IngestionTrace events over silent fallbacks; attach to result."
```

> This is the ONLY task that intentionally adds observable output. Do not fold it into Tasks 2–7 (those stay pure "no behavior change").

---

## Task 9: Slim The Facade + Re-export Audit

**Files:** Modify `src/ingestion/atomic.py`

- [ ] **Step 1: Confirm `atomic.py` is now a thin sequencer**

`ingest_document_atomic` should read as an ordered stage-call sequence (~120 lines). Verify the file shrank substantially:

```bash
wc -l src/ingestion/atomic.py   # expect well under ~900 (from 2417)
```

- [ ] **Step 2: Verify the 6 backward-compat re-exports still resolve**

```bash
python -c "from src.ingestion.atomic import Neo4jWriter, QdrantWriter, ALLOWED_ENTITY_RELATIONSHIP_TYPES, IngestionValidator, SagaContext, ValidationResult; print('re-exports OK')"
```

Expected: `re-exports OK`. Add `# noqa: F401` (or an `__all__`) to the re-exports rather than deleting them — this fixes the pre-existing `F401 ALLOWED_ENTITY_RELATIONSHIP_TYPES` ruff error at `atomic.py:117` without breaking test importers.

- [ ] **Step 3: Full offline suite + live outer gate + ruff**

```bash
pytest tests/ingestion tests/unit/test_phase2_schema_cleanup.py tests/test_cleanup_validation.py -q
ruff check src/ingestion/atomic.py src/ingestion/stages
python -m compileall -q src/ingestion
# Live outer gate (docker required): pytest tests/integration/test_phase1_entity_edges.py tests/integration/test_gliner_ingestion_flow.py -q
npx gitnexus detect-changes --scope staged --repo wekadocs-matrix
```

Expected: offline tests pass, ruff clean on touched files, compile clean; run the live gate once when services are available.

- [ ] **Step 4: Commit + push**

```bash
git commit -am "refactor(p4.0): slim AtomicIngestionCoordinator to a stage facade" \
  -m "Phase 4.0 — reduce atomic.py to a stage sequencer; re-exports intact."
git push origin wip/weka-to-nutanix-migration
```

---

## Acceptance Criteria

- `src/ingestion/atomic.py` is a thin orchestrator (stage-call sequence + result contract + re-exports), materially smaller than 2,417 lines.
- Each of `parse/chunk/enrich/embed/write/link` logic lives in `src/ingestion/stages/*` behind a stable function interface.
- The 5 frozen public-contract items are unchanged; the 6 re-exports still import from `src.ingestion.atomic`.
- The offline characterization pin passed after **every** extraction commit (no behavior change), and the live integration gate passed once before and once after the series.
- `IngestionTrace` events replace silent ingest fallbacks (Task 8), attached at `result.stats["trace"]`.
- The pre-existing `F401` ruff error in `atomic.py` is resolved via `# noqa`/`__all__`, not by deleting re-exports.
- Branch pushed.

## Self-Review

- Strangler discipline: pin → impact → extract-behind-interface → re-run pin → commit, one stage per commit, leaf-first.
- Behavior additivity is isolated to Task 8; Tasks 2–7 are pure moves.
- The #1 trap (shared `sections` mutation of `_mentions`/`_embedding_text`/`was_truncated`) is called out per relevant stage.
- Dependency on Slice B (remove `patched/atomic_patched_v3.py`) is stated up front.
- Live-service coupling is honest: the offline pin is the inner loop; the docker suite is the outer gate.
