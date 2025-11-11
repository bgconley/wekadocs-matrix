# Repo Scan Summary

- Extracted to: `/mnt/data/repo`
- File count (scanned types): 369
- Stack guess: {"has_python": true, "has_node": false, "has_go": false, "has_java": false, "has_rust": false, "has_docker": true, "has_requirements": false, "has_packagejson": false, "has_cypher": true, "has_sql": false}
- .env-like files: 0

## Keyword hits (first 50 files)

### docker-compose.yml
- **\bQdrant\b** → `ec.md §2 (Architecture) # See: /docs/implementation-plan.md → Task 1.1 DoD & Tests  version: '3.8'  networks:   weka-net:     driver: bridge     name: weka-net  volumes:   neo4j-data:   neo4j-logs:   qdrant-data:   redis-data:   hf-cache:  …`
- **\bQdrant\b** → `healthcheck:       test: ["CMD", "cypher-shell", "-u", "${NEO4J_USER:-neo4j}", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]       interval: 10s       timeout: 5s       retries: 5       start_period: 30s    qdrant:     image: qdrant/qdrant:v1.7.4 …`
- **\bQdrant\b** → `test: ["CMD", "cypher-shell", "-u", "${NEO4J_USER:-neo4j}", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]       interval: 10s       timeout: 5s       retries: 5       start_period: 30s    qdrant:     image: qdrant/qdrant:v1.7.4     container_name:…`
### config/feature_flags.json
- **\bhybrid\b** → `{   "flags": {     "hybrid_search_v2": {       "enabled": false,       "rollout_percentage": 0,       "description": "New hybrid search algorithm with improved ranking"     },     "advanced_caching": {       "enabled": true,       "rollout_…`
- **\bQdrant\b** → `"description": "Advanced L1+L2 caching with version prefixes"     },     "dual_vector_write": {       "enabled": false,       "rollout_percentage": 0,       "description": "Write vectors to both Qdrant and Neo4j (for migration)"     },     …`
- **\bqdrant\b** → `"description": "Advanced L1+L2 caching with version prefixes"     },     "dual_vector_write": {       "enabled": false,       "rollout_percentage": 0,       "description": "Write vectors to both Qdrant and Neo4j (for migration)"     },     …`
### config/development.yaml
- **\bhybrid\b** → `collection_name: "weka_sections_v2"  # Phase 7C: New 1024-D collection       use_grpc: false       timeout: 30     neo4j:       index_name: "section_embeddings_v2"  # Phase 7C: New 1024-D index    # Hybrid search settings   hybrid:     vect…`
- **\bhybrid\b** → `ections_v2"  # Phase 7C: New 1024-D collection       use_grpc: false       timeout: 30     neo4j:       index_name: "section_embeddings_v2"  # Phase 7C: New 1024-D index    # Hybrid search settings   hybrid:     vector_weight: 0.7  # Weight…`
- **\bQdrant\b** → `al.query    # Performance settings   batch_size: 32   max_sequence_length: 8192  # Jina v3 supports longer sequences  # Search configuration search:   # Vector search settings   vector:     primary: "qdrant"     dual_write: false  # Phase 7…`
### scripts/apply_complete_schema_v2_1.py
- **\bNeo4j\b** → `nts  logger = get_logger(__name__)   def apply_complete_v2_1_schema():     """Apply complete v2.1 schema to clean database."""      logger.info("Starting complete v2.1 schema application")      # Get Neo4j connection     manager = get_conne…`
- **\bNeo4j\b** → `ma application")      # Get Neo4j connection     manager = get_connection_manager()     driver = manager.get_neo4j_driver()      # Read complete v2.1 schema     schema_path = Path(__file__).parent / "neo4j" / "create_schema_v2_1_complete.cy…`
- **\bneo4j\b** → `nts  logger = get_logger(__name__)   def apply_complete_v2_1_schema():     """Apply complete v2.1 schema to clean database."""      logger.info("Starting complete v2.1 schema application")      # Get Neo4j connection     manager = get_conne…`
### scripts/run_baseline_queries.py
- **\bhybrid_search\b** → `ath for imports sys.path.insert(0, str(Path(__file__).parent.parent))  from src.providers.embeddings.jina import JinaEmbeddingProvider from src.providers.factory import ProviderFactory from src.query.hybrid_search import QdrantVectorStore f…`
- **\bhybrid\b** → `y_set.yaml     python scripts/run_baseline_queries.py --report reports/phase-7e/baseline-queries.json     python scripts/run_baseline_queries.py --top-k 10  Features: - Executes queries using current hybrid search - Captures latency, result…`
- **\bQdrant\b** → `te_embedding_provider()         logger.info(f"Using embedding provider: {embedding_provider.__class__.__name__}")                  # Get collection name         collection_name = config.search.vector.qdrant.collection_name         logger.in…`
### scripts/backfill_document_tokens.py
- **\bNeo4j\b** → `ror handling with rollback - Idempotent execution """  import argparse import json import logging import sys from datetime import datetime from pathlib import Path from typing import Dict, List  from neo4j import Driver  # Add src to path f…`
- **\bNeo4j\b** → `nfo("=" * 80)     logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")     logger.info(f"Report: {args.report if args.report else 'None'}")     logger.info("")          try:         # Get Neo4j connection         conn_manager = …`
- **\bneo4j\b** → `ror handling with rollback - Idempotent execution """  import argparse import json import logging import sys from datetime import datetime from pathlib import Path from typing import Dict, List  from neo4j import Driver  # Add src to path f…`
### scripts/test_jina_integration.py
- **\bjina\b** → `#!/usr/bin/env python3 """ Test Jina AI embedding provider integration. Validates Phase 7C.7 implementation with real Jina v4 API. """  import os import sys from pathlib import Path  # Add project root to path sys.path.insert(0, str(Pat`
- **\bjina\b** → `#!/usr/bin/env python3 """ Test Jina AI embedding provider integration. Validates Phase 7C.7 implementation with real Jina v4 API. """  import os import sys from pathlib import Path  # Add project root to path sys.path.insert(0, str(Path(__…`
- **\bjina\b** → `st 1: Verify ProviderFactory creates JinaEmbeddingProvider correctly."""     print("\n" + "=" * 70)     print("TEST 1: Provider Factory Integration")     print("=" * 70)      # Override config to use Jina     config = get_config()     confi…`
### scripts/validate_token_accounting.py
- **\bNeo4j\b** → `ical summary - JSON export for analysis """  import argparse import json import logging import sys from datetime import datetime from pathlib import Path from typing import Dict, List, Optional  from neo4j import Driver  # Add src to path f…`
- **\bNeo4j\b** → `TokenAccountingValidator:     """Validates Document.token_count matches section sums"""      def __init__(self, driver: Driver, threshold: float = 0.01):         """         Args:             driver: Neo4j driver             threshold: Maxi…`
- **\bNeo4j\b** → `s.threshold:.2%}")     logger.info(f"Fail on violation: {args.fail_on_violation}")     logger.info(f"Report: {args.report if args.report else 'None'}")     logger.info("")          try:         # Get Neo4j connection         conn_manager = …`
### scripts/cleanup-databases.py
- **\bQdrant\b** → `p Script - Surgical Data Deletion with Schema Preservation  This script performs surgical cleanup of all test/development data from: - Neo4j (nodes + relationships, preserves constraints + indexes) - Qdrant (vectors, preserves collection sc…`
- **\bQdrant\b** → `--dry-run           Show what would be deleted without actually deleting     --redis-db N        Redis database number to clean (default: 1)     --skip-neo4j        Skip Neo4j cleanup     --skip-qdrant       Skip Qdrant cleanup     --skip-r…`
- **\bQdrant\b** → `Show what would be deleted without actually deleting     --redis-db N        Redis database number to clean (default: 1)     --skip-neo4j        Skip Neo4j cleanup     --skip-qdrant       Skip Qdrant cleanup     --skip-redis        Skip Red…`
### scripts/README-cleanup.md
- **\bQdrant\b** → `# Purpose  Performs surgical cleanup of test/development data from all databases while **preserving schemas**:  - **Neo4j:** Deletes all nodes and relationships, preserves constraints and indexes - **Qdrant:** Deletes all vector collections…`
- **\bQdrant\b** → `schemas auto-recreate on next use - **Redis:** Flushes specified database (default: db=1 for tests)  ## Safety Features  ✅ **Schema Preservation** - Neo4j constraints and indexes are NEVER deleted - Qdrant collection schemas auto-recreate -…`
- **\bQdrant\b** → `leanup-databases.py --dry-run  # Quiet mode (minimal output) python scripts/cleanup-databases.py --quiet ```  ### Advanced Usage  ```bash # Clean only Neo4j python scripts/cleanup-databases.py --skip-qdrant --skip-redis  # Clean only Qdrant…`
### scripts/QUICKSTART-cleanup.md
- **\bQdrant\b** → `tandard cleanup** | `python scripts/cleanup-databases.py` | | **Preview only** | `python scripts/cleanup-databases.py --dry-run` | | **Clean Neo4j only** | `python scripts/cleanup-databases.py --skip-qdrant --skip-redis` | | **Clean Qdrant …`
- **\bQdrant\b** → `ts/cleanup-databases.py` | | **Preview only** | `python scripts/cleanup-databases.py --dry-run` | | **Clean Neo4j only** | `python scripts/cleanup-databases.py --skip-qdrant --skip-redis` | | **Clean Qdrant only** | `python scripts/cleanup-…`
- **\bQdrant\b** → `ip-qdrant --skip-redis` | | **Clean Qdrant only** | `python scripts/cleanup-databases.py --skip-neo4j --skip-redis` | | **Clean Redis only** | `python scripts/cleanup-databases.py --skip-neo4j --skip-qdrant` | | **Quiet mode** | `python scr…`
### scripts/test_jina_payload_limits.py
- **\bjina\b** → `#!/usr/bin/env python3 """ Test Jina API payload size limits to validate hypothesis.  This script tests: 1. Small payload (should succeed) 2. Large payload ~160KB (should fail with 400)  Expected behavior: - Small: HTTP 200 - Large: HTT`
- **\bjina\b** → `d succeed) 2. Large payload ~160KB (should fail with 400)  Expected behavior: - Small: HTTP 200 - Large: HTTP 400 Bad Request (payload too large) """  import json import os import sys import httpx  # Jina API configuration JINA_API_KEY = os…`
- **\bjina\b** → `t json import os import sys import httpx  # Jina API configuration JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi") API_URL = "https://api.jina.ai/v1/embeddings"  def test_small_p…`
### scripts/baseline_distribution_analysis.py
- **\bNeo4j\b** → `and Markdown """  import argparse import json import logging import sys from collections import defaultdict from datetime import datetime from pathlib import Path from typing import Dict, List  from neo4j import Driver  # Add src to path fo…`
- **\bNeo4j\b** → `logger.info(f"JSON report: {args.report if args.report else 'None'}")     logger.info(f"Markdown report: {args.markdown if args.markdown else 'None'}")     logger.info("")          try:         # Get Neo4j connection         conn_manager = …`
- **\bneo4j\b** → `and Markdown """  import argparse import json import logging import sys from collections import defaultdict from datetime import datetime from pathlib import Path from typing import Dict, List  from neo4j import Driver  # Add src to path fo…`
### scripts/init_schema.py
- **\bNeo4j\b** → `#!/usr/bin/env python3 """Initialize Neo4j schema with config-driven vector indexes"""  import sys from pathlib import Path  # Add project root to path project_root = Path(__file__).parent.parent sys.path.insert(0, str(project_root))  from …`
- **\bNeo4j\b** → `from src.shared.schema import create_schema  # noqa: E402   def main():     config = get_config()      manager = get_connection_manager()     driver = manager.get_neo4j_driver()      print("Creating Neo4j schema with config-driven vector in…`
- **\bneo4j\b** → `#!/usr/bin/env python3 """Initialize Neo4j schema with config-driven vector indexes"""  import sys from pathlib import Path  # Add project root to path project_root = Path(__file__).parent.parent sys.path.insert(0, str(project_root))  from …`
### scripts/verify_providers.py
- **\brerank\b** → `nt(f"  Dimensions: {embed_provider.dims}")         print(f"  Provider: {embed_provider.provider_name}")     except Exception as e:         print(f"  ❌ Failed: {e}")         return 1          # Create rerank provider     print("\nRerank Prov…`
- **\bjina\b** → `#!/usr/bin/env python3 """Quick script to verify Jina provider configuration."""  import os import sys  # Add src to path sys.path.insert(0, '/app/src')  from src.providers.factory import create_embedding_provider, create_rerank_provider  d…`
- **\bembedd** → `ovider  def main():     print("=" * 60)     print("PROVIDER CONFIGURATION VERIFICATION")     print("=" * 60)          # Check environment variables     print("\nEnvironment Variables:")     print(f"  EMBEDDINGS_MODEL: {os.getenv('EMBEDDINGS…`
### scripts/neo4j/create_schema_v2_1_complete.cypher
- **\bhybrid\b** → `node creation // - See: src/ingestion/build_graph.py for validation logic // // REQUIRED FIELDS (enforced in application code): // Section nodes: //   - vector_embedding (List<Float>) - CRITICAL for hybrid search //   - embedding_version (S…`
- **\bNeo4j\b** → `on) // ============================================================================ // COMMUNITY EDITION NOTE: // Property existence constraints (REQUIRE property IS NOT NULL) are not // supported in Neo4j Community Edition. They require En…`
- **\bNeo4j\b** → `lidation // Pattern: Singleton node with version metadata // Idempotent: MERGE ensures only one version node exists // // METADATA FIELDS: // - version: Schema version identifier (v2.1) // - edition: Neo4j edition (community or enterprise) …`
### scripts/neo4j/create_schema_v2_1.cypher
- **\bjina\b** → `1024-D (UPDATED from stub) // ============================================================================ // Purpose: Semantic search on Section/Chunk nodes // CRITICAL: Dimensions set to 1024-D for Jina v4 (not 384-D or 768-D) // Idempote…`
- **\bjina\b** → `y',     sv.updated_at = datetime(),     sv.description = 'Phase 7C: 1024-D vectors, dual-labeling, session tracking (Community Edition)',     sv.vector_dimensions = 1024,     sv.embedding_provider = 'jina-ai',     sv.embedding_model = 'jina…`
- **\bjina\b** → `sv.description = 'Phase 7C: 1024-D vectors, dual-labeling, session tracking (Community Edition)',     sv.vector_dimensions = 1024,     sv.embedding_provider = 'jina-ai',     sv.embedding_model = 'jina-embeddings-v4',     sv.validation_note …`
### scripts/neo4j/create_schema.cypher
- **\bNeo4j\b** → `// Implements Phase 1, Task 1.3 (Database schema initialization) // See: /docs/spec.md §3 (Data model) // See: /docs/implementation-plan.md → Task 1.3 DoD & Tests // Neo4j schema creation - idempotent, can be run multiple times  // ========…`
- **\bneo4j\b** → `// Implements Phase 1, Task 1.3 (Database schema initialization) // See: /docs/spec.md §3 (Data model) // See: /docs/implementation-plan.md → Task 1.3 DoD & Tests // Neo4j schema creation - idempotent, can be run multiple times  // ========…`
- **\bembedd** → `=================================== // NOTE: Vector indexes will be created programmatically via schema.py // because they require config-driven dimensions and similarity metrics. // This ensures the embedding configuration is the single so…`
### scripts/test/summarize_phase3.py
- **\bQdrant\b** → `"name": "Graph Construction & Embeddings",                 "tests_passed": 0,                 "tests_total": 0,                 "status": "not_run",                 "notes": "Requires live Neo4j/Qdrant - run separately",             },     …`
- **\bQdrant\b** → `"name": "Incremental Updates & Reconciliation",                 "tests_passed": 0,                 "tests_total": 0,                 "status": "not_run",                 "notes": "Requires live Neo4j/Qdrant - run separately",             },…`
- **\bqdrant\b** → `"name": "Graph Construction & Embeddings",                 "tests_passed": 0,                 "tests_total": 0,                 "status": "not_run",                 "notes": "Requires live Neo4j/Qdrant - run separately",             },     …`
### scripts/test/check_phase3_metrics.py
- **\bvector\b** → `#!/usr/bin/env python3 """Ensure critical Phase 3 metrics (vector parity & idempotence) remain green.  Reads the summary.json emitted by scripts/test/run_phase.sh and verifies that key gatekeeper tests passed. If any required test is missin…`
### scripts/test/debug_explain.py
- **\bNeo4j\b** → `#!/usr/bin/env python3 """Debug script to inspect Neo4j EXPLAIN plan structure."""  import os  from neo4j import GraphDatabase   def main():     uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")     user = os.getenv("NEO4J_USER", "neo4j…`
- **\bNeo4j\b** → `#!/usr/bin/env python3 """Debug script to inspect Neo4j EXPLAIN plan structure."""  import os  from neo4j import GraphDatabase   def main():     uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")     user = os.getenv("NEO4J_USER", "neo4j…`
- **\bNeo4j\b** → `ipt to inspect Neo4j EXPLAIN plan structure."""  import os  from neo4j import GraphDatabase   def main():     uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")     user = os.getenv("NEO4J_USER", "neo4j")     pwd = os.getenv("NEO4J_PASSW…`
### scripts/dev/seed_minimal_graph.py
- **\bQdrant\b** → `#!/usr/bin/env python3 """ Seed Minimal Graph for Phase 2 Testing Creates deterministic test data in Neo4j and Qdrant. NO MOCKS - uses real services. """  import hashlib import os import sys from pathlib import Path  # Add src to path sys.p…`
- **\bQdrant\b** → `os.getenv("NEO4J_USER", "neo4j")     neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")      driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))      # Connect to Qdrant if primary     vector_primary = c…`
- **\bQdrant\b** → `hDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))      # Connect to Qdrant if primary     vector_primary = config.search.vector.primary     qdrant_client = None     if vector_primary == "qdrant":         qdrant_host = os.getenv…`
### scripts/eval/run_eval.py
- **\bjina\b** → `#!/usr/bin/env python3 """ Phase 7C.9 - Quality Validation Script  Establishes quality baseline for Jina v4 @ 1024-D deployment. Calculates NDCG, MRR, latency metrics and validates quality gates.  Usage:     python scripts/eval/run_eval.py …`
- **\bjina\b** → `{             "meta": {                 "timestamp": datetime.utcnow().isoformat(),                 "phase": "7C",                 "task": "7C.9",                 "description": "Quality baseline for Jina v4 @ 1024-D deployment",           …`
- **\bembedd** → `f.query_service = query_service or QueryService()         self.config = get_config()         self.settings = get_settings()          logger.info("Quality Evaluator initialized")         logger.info(f"Embedding provider: {self.config.embeddi…`
### scripts/perf/test_verbosity_latency.py
- **\bQuery\b** → `t performance issues",     "Explain WekaFS architecture",     "How do I upgrade the system?", ]  MCP_ENDPOINT = "http://localhost:8000/mcp/tools/call" WARMUP_REQUESTS = 5 TEST_ITERATIONS = 10  # Each query run 10 times = 50 total requests p…`
- **\bQuery\b** → `try:             requests.post(                 MCP_ENDPOINT,                 json={                     "name": "search_documentation",                     "arguments": {                         "query": QUERIES[i % len(QUERIES)],         …`
- **\bQuery\b** → `print(f"Warmup request {i+1} failed: {e}")      print(f"Running {len(QUERIES) * TEST_ITERATIONS} test requests...")      # Test requests     for iteration in range(TEST_ITERATIONS):         for query in QUERIES:             try:            …`
### scripts/perf/test_traversal_latency.py
- **\bQuery\b** → `section IDs from search results."""     try:         resp = requests.post(             MCP_ENDPOINT,             json={                 "name": "search_documentation",                 "arguments": {"query": "cluster configuration", "top_k":…`
### src/connectors/RUNBOOK.md
- **\bQdrant\b** → `health endpoints  ---  ## Architecture  ``` External System (GitHub/Notion)     ↓ (polling/webhook) Connector (with Circuit Breaker)     ↓ Ingestion Queue (Redis)     ↓ Ingestion Worker     ↓ Neo4j + Qdrant ```  **Components:** - `BaseConne…`
- **\bQdrant\b** → `lling paused  **Diagnosis**: ```python stats = queue.get_stats() # Check: size, usage_pct ```  **Resolution**: 1. Check ingestion worker is running 2. Scale ingestion workers if needed 3. Check Neo4j/Qdrant performance 4. Temporary: increas…`
- **\bqdrant\b** → `health endpoints  ---  ## Architecture  ``` External System (GitHub/Notion)     ↓ (polling/webhook) Connector (with Circuit Breaker)     ↓ Ingestion Queue (Redis)     ↓ Ingestion Worker     ↓ Neo4j + Qdrant ```  **Components:** - `BaseConne…`
### src/ingestion/build_graph.py
- **\bhybrid\b** → `count=count,             )          return total_sections      def _remove_missing_sections(         self, session, document_id: str, valid_section_ids: List[str]     ) -> int:         """         Hybrid orphan section cleanup strategy.    …`
- **\bhybrid\b** → `:                     raise ValueError(                         f"Section {section['id']} missing REQUIRED vector_embedding. "                         "Ingestion blocked - embeddings are mandatory in hybrid system."                     )   …`
- **\bQdrant\b** → `dual_write = config.search.vector.dual_write          # Phase 7C.7: Fresh start with 1024-D (Session 06-08)         # No dual-write complexity - starting fresh with Jina v4 @ 1024-D          # Ensure Qdrant collections exist if using Qdrant…`
### src/ingestion/api.py
- **\bembedd** → `rt Any, Dict, Optional  from src.ingestion.build_graph import ingest_document as build_graph_ingest   def ingest_document(     source_uri: str,     content: str,     fmt: str = "markdown",     *,     embedding_model: Optional[str] = None,  …`
- **\bembedd** → `build_graph import ingest_document as build_graph_ingest   def ingest_document(     source_uri: str,     content: str,     fmt: str = "markdown",     *,     embedding_model: Optional[str] = None,     embedding_version: Optional[str] = None,…`
- **\bembedd** → `ional[str] = None, ) -> Dict[str, Any]:     """     Synchronous façade used by integration tests.     Delegates to the full graph-building ingestion pipeline while allowing     optional overrides for embedding model metadata.     """     re…`
### src/ingestion/reconcile.py
- **\bQdrant\b** → `class DriftStats:     embedding_version: str     graph_count: int     vector_count: int     extras_removed: int     missing_backfilled: int     drift_pct: float   class Reconciler:     """     Keeps Qdrant strictly in sync with graph Sectio…`
- **\bQdrant\b** → `self.neo4j = neo4j_driver          # Handle backwards compatibility for different call signatures         if isinstance(config_or_qdrant, QdrantClient):             # Old signature: Reconciler(neo4j, qdrant, ...)             self.qdrant = c…`
- **\bQdrant\b** → `# Handle backwards compatibility for different call signatures         if isinstance(config_or_qdrant, QdrantClient):             # Old signature: Reconciler(neo4j, qdrant, ...)             self.qdrant = config_or_qdrant             self.co…`
### src/ingestion/incremental.py
- **\bQdrant\b** → `_client: QdrantClient = None,         collection_name: str = "weka_sections",         embedding_version: str = "v1",     ):         self.neo4j = neo4j_driver         self.config = config         self.qdrant = qdrant_client         self.coll…`
- **\bQdrant\b** → `llection = collection_name         self.version = embedding_version         if config and hasattr(config, "search") and hasattr(config.search, "vector"):             if hasattr(config.search.vector, "qdrant"):                 self.collectio…`
- **\bQdrant\b** → `on         if config and hasattr(config, "search") and hasattr(config.search, "vector"):             if hasattr(config.search.vector, "qdrant"):                 self.collection = config.search.vector.qdrant.collection_name             self.…`
### src/ingestion/parsers/html.py
- **\bembedd** → `text,         "tokens": tokens,         "checksum": checksum,         "code_blocks": section_data["code_blocks"],         "tables": section_data["tables"],         "vector_embedding": None,         "embedding_version": None,     }   def _co…`
### src/ingestion/parsers/markdown.py
- **\bembedd** → `r": order,         "text": text,         "tokens": tokens,         "checksum": checksum,         "code_blocks": section_data["code_blocks"],         "tables": section_data["tables"],         # Vector embedding fields (populated later)      …`
- **\bembedd** → `checksum,         "code_blocks": section_data["code_blocks"],         "tables": section_data["tables"],         # Vector embedding fields (populated later)         "vector_embedding": None,         "embedding_version": None,     }   def _co…`
- **\bvector\b** → `"order": order,         "text": text,         "tokens": tokens,         "checksum": checksum,         "code_blocks": section_data["code_blocks"],         "tables": section_data["tables"],         # Vector embedding fields (populated later) …`
### src/ingestion/parsers/notion.py
- **\bembedd** → `text,         "tokens": tokens,         "checksum": checksum,         "code_blocks": section_data["code_blocks"],         "tables": section_data["tables"],         "vector_embedding": None,         "embedding_version": None,     }   def _co…`
### src/ingestion/auto/backpressure.py
- **\bQdrant\b** → `""" Phase 6, Task 6.1: Back-Pressure Monitoring  Monitors Neo4j CPU and Qdrant latency to prevent overwhelming downstream systems.  Triggers pause when: - Neo4j CPU > 80% - Qdrant P95 latency > 200ms  See: /docs/coder-guidance-phase6.md → 6…`
- **\bQdrant\b** → `""" Phase 6, Task 6.1: Back-Pressure Monitoring  Monitors Neo4j CPU and Qdrant latency to prevent overwhelming downstream systems.  Triggers pause when: - Neo4j CPU > 80% - Qdrant P95 latency > 200ms  See: /docs/coder-guidance-phase6.md → 6…`
- **\bQdrant\b** → `Initialize back-pressure monitor          Args:             neo4j_uri: Neo4j connection URI             neo4j_user: Neo4j username             neo4j_password: Neo4j password             qdrant_host: Qdrant hostname             qdrant_port: …`
### src/ingestion/auto/verification.py
- **\bhybrid_search\b** → `sample queries.  See: /docs/pseudocode-phase6.md → Task 6.4 See: /docs/implementation-plan-phase-6.md → Task 6.4 """  from typing import Dict, List, Optional  from neo4j import Driver  from src.query.hybrid_search import HybridSearchEngine …`
- **\bhybrid\b** → `o sample queries configured for tag", tag=tag)             return []          answers = []         for q in queries[:3]:  # Limit to 3 queries to avoid slowdown             try:                 # Run hybrid search                 results = …`
- **\bQdrant\b** → `)                 record = result.single()                 graph_count = record["graph_count"] if record else 0              # Count vectors in primary store             if self.vector_primary == "qdrant" and self.qdrant_client:            …`
### src/ingestion/auto/__init__.py
- **\bvector\b** → `hitecture: - Watchers monitor FS/S3/HTTP for new documents - Orchestrator runs resumable FSM through ingestion stages - Progress events stream to Redis for CLI consumption - Verification checks graph/vector alignment and sample queries - Re…`
### src/ingestion/auto/README.md
- **\bQdrant\b** → `er` — HTTP endpoint polling - `WatcherManager` — Manages multiple watchers from config  ### 3. `backpressure.py` — Resource Monitoring  **Monitors:** - Neo4j CPU (heuristic based on active queries) - Qdrant P95 latency (from Prometheus metr…`
- **\bQdrant\b** → `- Qdrant P95 latency (from Prometheus metrics)  **Behavior:** - Signals pause when thresholds exceeded - Resumes automatically when pressure clears  **Thresholds (configurable):** - Neo4j CPU > 80% - Qdrant P95 > 200ms  ### 4. `service.py` …`
- **\bQdrant\b** → `s in queue - `ingest_watchers_count` — Number of active watchers - `ingest_backpressure_paused` — 0 (running) or 1 (paused) - `ingest_neo4j_cpu` — Estimated Neo4j CPU (0-1) - `ingest_qdrant_p95_ms` — Qdrant P95 latency (ms)  ## Configuratio…`
### src/ingestion/auto/cli.py
- **\bembedd** → `rels_added', 0)}")              print("\nVectors:")             vectors = report.get("vector", {})             print(f"  Sections indexed: {vectors.get('sections_indexed', 0)}")             print(f"  Embedding version: {vectors.get('embeddi…`
- **\bembedd** → `rint("\nVectors:")             vectors = report.get("vector", {})             print(f"  Sections indexed: {vectors.get('sections_indexed', 0)}")             print(f"  Embedding version: {vectors.get('embedding_version', 'N/A')}")           …`
- **\bvector\b** → `print(f"  Nodes added: {graph.get('nodes_added', 0)}")             print(f"  Relationships added: {graph.get('rels_added', 0)}")              print("\nVectors:")             vectors = report.get("vector", {})             print(f"  Sections …`
### src/ingestion/auto/orchestrator.py
- **\bQdrant\b** → `redis_client: Redis client for state/progress             neo4j_driver: Neo4j driver for graph operations             config: Application configuration             qdrant_client: Optional Qdrant client for vectors         """         self.r…`
- **\bQdrant\b** → `n configuration             qdrant_client: Optional Qdrant client for vectors         """         self.redis = redis_client         self.neo4j = neo4j_driver         self.config = config         self.qdrant = qdrant_client          # Initia…`
- **\bQdrant\b** → `"""         tracker.advance(JobStage.GRAPHING, "Building graph")          start_time = time.time()          # Initialize graph builder         builder = GraphBuilder(self.neo4j, self.config, self.qdrant)          # Check if this is an incre…`
### src/ingestion/auto/progress.py
- **\bembedd** → `class JobStage(str, Enum):     """Job processing stages with deterministic ordering."""      PENDING = "PENDING"     PARSING = "PARSING"     EXTRACTING = "EXTRACTING"     GRAPHING = "GRAPHING"     EMBEDDING = "EMBEDDING"     VECTORS = "VECT…`
- **\bembedd** → `tage(str, Enum):     """Job processing stages with deterministic ordering."""      PENDING = "PENDING"     PARSING = "PARSING"     EXTRACTING = "EXTRACTING"     GRAPHING = "GRAPHING"     EMBEDDING = "EMBEDDING"     VECTORS = "VECTORS"     P…`
- **\bembedd** → `OR"   # Stage weights for progress calculation (total = 100%) STAGE_WEIGHTS = {     JobStage.PENDING: 0,     JobStage.PARSING: 10,     JobStage.EXTRACTING: 15,     JobStage.GRAPHING: 25,     JobStage.EMBEDDING: 20,     JobStage.VECTORS: 15,…`
### src/ingestion/auto/report.py
- **\bQdrant\b** → `"sections_total": 0,             "documents_total": 0,         }      def _get_vector_stats(self) -> Dict:         """Get vector store stats."""         try:             if self.vector_primary == "qdrant" and self.qdrant_client:            …`
- **\bQdrant\b** → `ts(self) -> Dict:         """Get vector store stats."""         try:             if self.vector_primary == "qdrant" and self.qdrant_client:                 collection_name = self.config.search.vector.qdrant.collection_name                 c…`
- **\bQdrant\b** → `collection_name = self.config.search.vector.qdrant.collection_name                 coll_info = self.qdrant_client.get_collection(collection_name)                  return {                     "sot": "qdrant",                     "sections_i…`
### src/ingestion/extract/configs.py
- **\bembedd** → `"description": description,         "category": category,         "introduced_in": None,         "deprecated_in": None,         "updated_at": None,         "vector_embedding": None,         "embedding_version": None,     }   def _create_men…`
### src/ingestion/extract/commands.py
- **\bembedd** → `e,         "description": full_command,         "category": "cli",         "introduced_in": None,         "deprecated_in": None,         "updated_at": None,         "vector_embedding": None,         "embedding_version": None,     }   def _c…`
### src/neo/__init__.py
- **\bNeo4j\b** → `""" Neo4j utilities and query safety guards. Phase 7a: EXPLAIN-plan validation and performance hardening. """  from .explain_guard import (     ExplainGuard,     PlanRejected,     PlanTooExpensive,     validat`
- **\bneo4j\b** → `""" Neo4j utilities and query safety guards. Phase 7a: EXPLAIN-plan validation and performance hardening. """  from .explain_guard import (     ExplainGuard,     PlanRejected,     PlanTooExpensive,     validat`
- **\bQuery\b** → `""" Neo4j utilities and query safety guards. Phase 7a: EXPLAIN-plan validation and performance hardening. """  from .explain_guard import (     ExplainGuard,     PlanRejected,     PlanTooExpensive,     validate_query_plan, )  __a`
### src/neo/explain_guard.py
- **\bNeo4j\b** → `""" EXPLAIN-plan validation guard for Neo4j queries. Phase 7a: Reject expensive/unbounded queries before execution.  See: /docs/phase-7-integration-plan.md See: /docs/phase7-target-phase-tasklist.md Day 1 """  from typing import Any, Dict, …`
- **\bNeo4j\b** → `ase 7a: Reject expensive/unbounded queries before execution.  See: /docs/phase-7-integration-plan.md See: /docs/phase7-target-phase-tasklist.md Day 1 """  from typing import Any, Dict, Optional  from neo4j import Driver  from src.shared.obs…`
- **\bNeo4j\b** → `violates safety constraints."""      pass   class PlanTooExpensive(PlanRejected):     """Raised when a query plan exceeds cost/row thresholds."""      pass   class ExplainGuard:     """     Validates Neo4j query plans using EXPLAIN before e…`
### src/learning/suggestions.py
- **\bNeo4j\b** → `templates and performance indexes. """  import logging import re from collections import Counter, defaultdict from dataclasses import dataclass from typing import Any, Dict, List, Optional, Set  from neo4j import Driver  from .feedback impo…`
- **\bNeo4j\b** → `indexes."""      def __init__(         self,         driver: Driver,         feedback_collector: FeedbackCollector,     ):         """Initialize suggestion engine.          Args:             driver: Neo4j driver for index analysis          …`
- **\bneo4j\b** → `templates and performance indexes. """  import logging import re from collections import Counter, defaultdict from dataclasses import dataclass from typing import Any, Dict, List, Optional, Set  from neo4j import Driver  from .feedback impo…`
### src/learning/feedback.py
- **\bNeo4j\b** → `k for ranking weight tuning and template mining. """  import json import uuid from dataclasses import dataclass, field from datetime import datetime from typing import Any, Dict, List, Optional  from neo4j import Driver   @dataclass class Q…`
- **\bNeo4j\b** → `s FeedbackCollector:     """Collects and persists query feedback for learning."""      def __init__(self, driver: Driver):         """Initialize feedback collector.          Args:             driver: Neo4j driver for persistence         """…`
- **\bNeo4j\b** → `driver: Neo4j driver for persistence         """         self.driver = driver         self._ensure_schema()      def _ensure_schema(self) -> None:         """Ensure feedback schema exists in Neo4j."""         with self.driver.session() as s…`
### src/learning/__init__.py
- **\bQuery\b** → `"""Learning and adaptation module for Phase 4 Task 4.4.  This module implements: - Feedback collection (query → result → rating) - Ranking weight optimization (NDCG-based) - Template and index suggestion mining """  from .feedback import Fe…`
### src/learning/ranking_tuner.py
- **\bQuery\b** → `feedbacks: List[QueryFeedback],     ) -> List[Tuple[Dict[str, float], List[float]]]:         """Extract (features, relevance) pairs from feedback.          Args:             feedbacks: List of rated query feedbacks          Returns:        …`
- **\bQuery\b** → `g is None or not fb.result_ids:                 continue              # Extract feature vectors for each result             if not fb.ranking_features:                 continue              # Use the query rating as relevance for all return…`
- **\bQuery\b** → `= []         for features, relevance in training_data:             # Compute scores with current weights             # Note: In real scenario, features would be per-result,             # here we use query-level features as proxy            …`
### src/providers/factory.py
- **\brerank\b** → `""" Provider factory for ENV-selectable embedding and rerank providers. Phase 7C, Task 7C.1: Factory pattern for docker-compose friendly configuration.  Supported providers: - Embedding: jina-ai, ollama, sentence-transformers - Rerank: jina…`
- **\brerank\b** → `-selectable embedding and rerank providers. Phase 7C, Task 7C.1: Factory pattern for docker-compose friendly configuration.  Supported providers: - Embedding: jina-ai, ollama, sentence-transformers - Rerank: jina-ai, noop  Configuration via…`
- **\brerank\b** → `anker model  See .env.example for full configuration options. """  import logging import os from typing import Optional  from src.providers.embeddings.base import EmbeddingProvider from src.providers.rerank.base import RerankProvider  logge…`
... (truncated) ...
