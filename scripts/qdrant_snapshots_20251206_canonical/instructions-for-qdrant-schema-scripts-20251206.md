Schema Snapshot System Explained

  1. scripts/qdrant_schema_snapshot.py - The Tool

  This is a CLI tool that manages Qdrant collection schemas. It has four commands:

  | Command  | Purpose                              | Idempotent? |
  |----------|--------------------------------------|-------------|
  | snapshot | Capture live schema → JSON file      | Yes         |
  | restore  | Apply JSON file → Qdrant collection  | Yes         |
  | validate | Check if live schema matches JSON    | Yes         |
  | diff     | Show differences between live & JSON | Yes         |

  Usage Examples:

  # Capture current schema (creates/updates the JSON snapshot)
  python scripts/qdrant_schema_snapshot.py snapshot \
    --collection chunks_multi_<dense_profile> --host 127.0.0.1

  # Restore schema (idempotent - safe to run multiple times)
  python scripts/qdrant_schema_snapshot.py restore \
    --collection chunks_multi_<dense_profile> --host 127.0.0.1

  # Validate current schema matches snapshot
  python scripts/qdrant_schema_snapshot.py validate \
    --collection chunks_multi_<dense_profile> --host 127.0.0.1

  # Show what's different
  python scripts/qdrant_schema_snapshot.py diff \
    --collection chunks_multi_<dense_profile> --host 127.0.0.1

  ★ Insight ─────────────────────────────────────
  Idempotency is key: The restore command checks if the collection exists with the correct schema first. If it matches, it does nothing. If it differs, it warns you and requires --force to recreate. This
  prevents accidental data loss.
  ─────────────────────────────────────────────────

  ---
  2. scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/chunks_multi_bge_m3_schema.json - The Snapshot

  This is a point-in-time capture of the Qdrant collection schema in JSON format. It contains:

  ├── vectors_config          # Dense vectors (content, title, doc_title, late-interaction)
  │   ├── size: 1024         # Dimensions
  │   ├── distance: Cosine   # Similarity metric
  │   └── multivector_config # For ColBERT (late-interaction only)
  │
  ├── sparse_vectors_config   # Sparse vectors (text-sparse, doc_title-sparse, etc.)
  │   └── index.on_disk: true
  │
  ├── hnsw_config             # HNSW index parameters (m=48, ef_construct=256)
  │
  └── payload_indexes         # 24 indexed payload fields (doc_id, tenant, etc.)

  What it captures:
  - 4 dense vectors: content, title, doc_title, late-interaction
  - 4 sparse vectors: text-sparse, doc_title-sparse, title-sparse, entity-sparse
  - HNSW config: m=48, ef_construct=256 (high-quality index settings)
  - 24 payload indexes: Fields that are indexed for filtering

  What it does NOT capture:
  - Actual vector data (points)
  - Payload values
  - Collection statistics (these are transient)

  ---
  How They Work Together

  ┌─────────────────┐        snapshot         ┌─────────────────────────┐
  │  Live Qdrant    │  ──────────────────────►│  JSON Snapshot File     │
  │  Collection     │                         │  (version controlled)   │
  └─────────────────┘        restore          └─────────────────────────┘
          ▲          ◄──────────────────────────────────┘
          │
          │  validate/diff
          └────────────────► Match? ✅ or Differences? ❌

  Workflow:
  1. Development: Make schema changes, then snapshot to capture
  2. Version Control: Commit the JSON file with code changes
  3. Deployment: Run restore to ensure schema matches (idempotent)
  4. Disaster Recovery: If collection corrupted, restore --force rebuilds it

  ★ Insight ─────────────────────────────────────
  Why this matters:
  - Schema changes are now code-reviewable (JSON in git)
  - New environments get identical schemas via restore
  - Disaster recovery is one command away
  - Environment drift is detectable via validate
