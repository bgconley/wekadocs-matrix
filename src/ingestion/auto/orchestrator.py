# Implements Phase 6, Task 6.2 (Orchestrator - resumable, idempotent job processing)
# See: /docs/app-spec-phase6.md
# See: /docs/implementation-plan-phase-6.md → Task 6.2
# See: /docs/pseudocode-phase6.md → Task 6.2

"""
Orchestrator for auto-ingestion jobs.

Implements a resumable state machine that processes documents through the full
Phase 3 pipeline: parse → extract → graph → embed → vectors → reconcile → report.

State is persisted to Redis after each stage for crash recovery.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import redis
from neo4j import Driver
from sentence_transformers import SentenceTransformer

from src.ingestion.build_graph import GraphBuilder
from src.ingestion.extract import extract_entities
from src.ingestion.incremental import IncrementalUpdater
from src.ingestion.parsers.html import parse_html
from src.ingestion.parsers.markdown import parse_markdown
from src.ingestion.parsers.notion import parse_notion
from src.ingestion.reconcile import Reconciler
from src.shared.config import Config
from src.shared.observability import get_logger

from .progress import JobStage, ProgressTracker
from .report import ReportGenerator
from .verification import PostIngestVerifier

logger = get_logger(__name__)


@dataclass
class JobState:
    """
    Job state persisted to Redis.

    After each stage completes, state is saved to allow resumption.
    """

    job_id: str
    source_uri: str
    checksum: str
    tag: str
    status: str  # PENDING, PARSING, EXTRACTING, etc.
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    # Stage completion flags for resume logic
    stages_completed: List[str] = field(default_factory=list)

    # Intermediate artifacts (for resume)
    document_id: Optional[str] = None
    document: Optional[Dict] = None
    sections: Optional[List[Dict]] = None
    entities: Optional[Dict] = None
    mentions: Optional[List[Dict]] = None

    # Stats
    stats: Dict = field(default_factory=dict)


class Orchestrator:
    """
    Orchestrates auto-ingestion jobs through Phase 3 pipeline.

    Features:
    - Resumable: State persisted after each stage
    - Idempotent: Deterministic IDs and MERGE semantics
    - Observable: Progress events emitted to Redis streams
    - Safe: Handles errors gracefully, supports retry
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        neo4j_driver: Driver,
        config: Config,
        qdrant_client=None,
    ):
        """
        Initialize orchestrator.

        Args:
            redis_client: Redis client for state/progress
            neo4j_driver: Neo4j driver for graph operations
            config: Application configuration
            qdrant_client: Optional Qdrant client for vectors
        """
        self.redis = redis_client
        self.neo4j = neo4j_driver
        self.config = config
        self.qdrant = qdrant_client

        # Initialize embedder lazily
        self.embedder: Optional[SentenceTransformer] = None

        logger.info("Orchestrator initialized")

    def process_job(self, job_id: str) -> Dict:
        """
        Process a job through the full pipeline.

        Supports resume from any stage if job was interrupted.

        Args:
            job_id: Job ID to process

        Returns:
            Final job stats
        """
        # Load or initialize job state
        state = self._load_state(job_id)
        if not state:
            raise ValueError(f"Job {job_id} not found in queue")

        # Initialize progress tracker
        tracker = ProgressTracker(self.redis, job_id)

        logger.info(
            "Processing job",
            job_id=job_id,
            source_uri=state.source_uri,
            current_status=state.status,
        )

        try:
            # Mark job as started
            if not state.started_at:
                state.started_at = time.time()
                self._save_state(state)

            # Execute pipeline stages
            state = self._execute_pipeline(state, tracker)

            # Mark as complete
            state.status = JobStage.DONE.value
            state.completed_at = time.time()
            self._save_state(state)

            tracker.complete("Job completed successfully")

            logger.info(
                "Job completed",
                job_id=job_id,
                duration_seconds=state.completed_at - state.started_at,
                stats=state.stats,
            )

            return state.stats

        except Exception as exc:
            # Handle error
            error_msg = str(exc)
            state.status = JobStage.ERROR.value
            state.error = error_msg
            state.completed_at = time.time()
            self._save_state(state)

            tracker.error(error_msg)

            logger.error(
                "Job failed",
                job_id=job_id,
                error=error_msg,
                exc_info=True,
            )

            raise

    def _execute_pipeline(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Execute pipeline stages with resume support.

        Args:
            state: Current job state
            tracker: Progress tracker

        Returns:
            Updated job state
        """
        # Stage 1: PARSING
        if JobStage.PARSING.value not in state.stages_completed:
            state = self._stage_parsing(state, tracker)

        # Stage 2: EXTRACTING
        if JobStage.EXTRACTING.value not in state.stages_completed:
            state = self._stage_extracting(state, tracker)

        # Stage 3: GRAPHING
        if JobStage.GRAPHING.value not in state.stages_completed:
            state = self._stage_graphing(state, tracker)

        # Stage 4: EMBEDDING
        if JobStage.EMBEDDING.value not in state.stages_completed:
            state = self._stage_embedding(state, tracker)

        # Stage 5: VECTORS
        if JobStage.VECTORS.value not in state.stages_completed:
            state = self._stage_vectors(state, tracker)

        # Stage 6: POSTCHECKS
        if JobStage.POSTCHECKS.value not in state.stages_completed:
            state = self._stage_postchecks(state, tracker)

        # Stage 7: REPORTING
        if JobStage.REPORTING.value not in state.stages_completed:
            state = self._stage_reporting(state, tracker)

        return state

    def _stage_parsing(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 1: Parse document into Document + Sections.

        Uses Phase 3.1 parsers (markdown, html, notion).
        """
        tracker.advance(JobStage.PARSING, f"Parsing document: {state.source_uri}")

        start_time = time.time()

        # Determine format from source_uri
        source_path = Path(state.source_uri)
        ext = source_path.suffix.lower()

        # Read content
        content = self._read_source(state.source_uri)

        # Parse based on format
        if ext in [".md", ".markdown"]:
            result = parse_markdown(state.source_uri, content)
        elif ext in [".html", ".htm"]:
            result = parse_html(state.source_uri, content)
        elif ext == ".json":  # Notion export
            result = parse_notion(json.loads(content))
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Extract parsed data
        document = result["Document"]
        sections = result["Sections"]

        # Save to state
        state.document = document
        state.sections = sections
        state.document_id = document["id"]
        state.status = JobStage.PARSING.value
        state.stages_completed.append(JobStage.PARSING.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["parsing"] = {
            "duration_ms": duration_ms,
            "sections_parsed": len(sections),
        }

        self._save_state(state)

        tracker.advance(
            JobStage.PARSING,
            f"Parsed {len(sections)} sections",
            details={"sections": len(sections)},
        )

        logger.debug(
            "Parsing complete",
            job_id=state.job_id,
            sections=len(sections),
            duration_ms=duration_ms,
        )

        return state

    def _stage_extracting(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 2: Extract entities from sections.

        Uses Phase 3.2 extractors (commands, configs, procedures).
        """
        tracker.advance(JobStage.EXTRACTING, "Extracting entities")

        start_time = time.time()

        # Extract entities
        entities, mentions = extract_entities(state.sections)

        # Save to state
        state.entities = entities
        state.mentions = mentions
        state.status = JobStage.EXTRACTING.value
        state.stages_completed.append(JobStage.EXTRACTING.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["extracting"] = {
            "duration_ms": duration_ms,
            "entities_extracted": len(entities),
            "mentions_created": len(mentions),
        }

        self._save_state(state)

        tracker.advance(
            JobStage.EXTRACTING,
            f"Extracted {len(entities)} entities",
            details={"entities": len(entities), "mentions": len(mentions)},
        )

        logger.debug(
            "Extraction complete",
            job_id=state.job_id,
            entities=len(entities),
            mentions=len(mentions),
            duration_ms=duration_ms,
        )

        return state

    def _stage_graphing(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 3: Upsert graph nodes and relationships.

        Uses Phase 3.3 GraphBuilder with incremental updates.
        """
        tracker.advance(JobStage.GRAPHING, "Building graph")

        start_time = time.time()

        # Initialize graph builder
        builder = GraphBuilder(self.neo4j, self.config, self.qdrant)

        # Check if this is an incremental update
        updater = IncrementalUpdater(
            self.neo4j,
            self.config,
            self.qdrant,
            collection_name=self.config.search.vector.qdrant.collection_name,
            embedding_version=self.config.embedding.version,
        )

        # Compute diff to detect changes
        diff = updater.compute_diff(state.document_id, state.sections)

        if diff["total_changes"] == 0:
            # No changes - skip graph operations
            logger.info(
                "No changes detected - skipping graph upsert",
                job_id=state.job_id,
                document_id=state.document_id,
            )

            state.status = JobStage.GRAPHING.value
            state.stages_completed.append(JobStage.GRAPHING.value)

            state.stats["graphing"] = {
                "duration_ms": 0,
                "sections_upserted": 0,
                "entities_upserted": 0,
                "mentions_created": 0,
                "incremental": True,
                "no_changes": True,
            }

            self._save_state(state)

            tracker.advance(JobStage.GRAPHING, "No changes detected")

            return state

        # Upsert to graph
        graph_stats = builder.upsert_document(
            state.document, state.sections, state.entities, state.mentions
        )

        # Save to state
        state.status = JobStage.GRAPHING.value
        state.stages_completed.append(JobStage.GRAPHING.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["graphing"] = {
            "duration_ms": duration_ms,
            "sections_upserted": graph_stats.get("sections_upserted", 0),
            "entities_upserted": graph_stats.get("entities_upserted", 0),
            "mentions_created": graph_stats.get("mentions_created", 0),
            "incremental": diff["total_changes"] < len(state.sections),
            "changes": diff["total_changes"],
        }

        self._save_state(state)

        tracker.advance(
            JobStage.GRAPHING,
            f"Upserted {graph_stats.get('sections_upserted', 0)} sections",
            details=graph_stats,
        )

        logger.debug(
            "Graphing complete",
            job_id=state.job_id,
            stats=graph_stats,
            duration_ms=duration_ms,
        )

        return state

    def _stage_embedding(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 4: Compute embeddings for sections.

        Embeddings are computed but not yet upserted to vector store.
        This stage can be rerun without side effects.
        """
        tracker.advance(JobStage.EMBEDDING, "Computing embeddings")

        start_time = time.time()

        # Initialize embedder lazily
        if not self.embedder:
            logger.info(
                "Loading embedding model", model=self.config.embedding.embedding_model
            )
            self.embedder = SentenceTransformer(self.config.embedding.embedding_model)

        # Compute embeddings for sections
        embeddings_computed = 0
        for section in state.sections:
            # Build text for embedding (title + content)
            title = section.get("title", "")
            text = section.get("text", "")
            text_to_embed = f"{title}\n\n{text}" if title else text

            # Compute embedding
            embedding = self.embedder.encode(text_to_embed).tolist()

            # Store in section (not yet persisted to vector store)
            section["vector_embedding"] = embedding
            section["embedding_version"] = self.config.embedding.version
            embeddings_computed += 1

        # Save to state
        state.status = JobStage.EMBEDDING.value
        state.stages_completed.append(JobStage.EMBEDDING.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["embedding"] = {
            "duration_ms": duration_ms,
            "embeddings_computed": embeddings_computed,
        }

        self._save_state(state)

        tracker.advance(
            JobStage.EMBEDDING,
            f"Computed {embeddings_computed} embeddings",
            details={"embeddings": embeddings_computed},
        )

        logger.debug(
            "Embedding complete",
            job_id=state.job_id,
            embeddings=embeddings_computed,
            duration_ms=duration_ms,
        )

        return state

    def _stage_vectors(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 5: Upsert embeddings to vector store.

        Handles both Qdrant and Neo4j vector stores based on config.
        """
        tracker.advance(JobStage.VECTORS, "Upserting vectors")

        start_time = time.time()

        vectors_upserted = 0

        # Determine vector store strategy
        primary = self.config.search.vector.primary
        dual_write = self.config.search.vector.dual_write

        # Purge existing vectors for this document to prevent drift
        self._purge_existing_vectors(state.document_id, state.source_uri)

        # Upsert to primary store
        if primary == "qdrant" and self.qdrant:
            vectors_upserted = self._upsert_to_qdrant(
                state.sections, state.document, state.document_id
            )

            # Dual write to Neo4j if enabled
            if dual_write:
                self._set_embedding_metadata_in_neo4j(state.sections)
        else:
            # Neo4j primary
            vectors_upserted = self._upsert_to_neo4j(state.sections)

            # Dual write to Qdrant if enabled
            if dual_write and self.qdrant:
                self._upsert_to_qdrant(
                    state.sections, state.document, state.document_id
                )

        # Save to state
        state.status = JobStage.VECTORS.value
        state.stages_completed.append(JobStage.VECTORS.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["vectors"] = {
            "duration_ms": duration_ms,
            "vectors_upserted": vectors_upserted,
            "primary_store": primary,
            "dual_write": dual_write,
        }

        self._save_state(state)

        tracker.advance(
            JobStage.VECTORS,
            f"Upserted {vectors_upserted} vectors to {primary}",
            details={"vectors": vectors_upserted, "store": primary},
        )

        logger.debug(
            "Vector upsert complete",
            job_id=state.job_id,
            vectors=vectors_upserted,
            primary=primary,
            duration_ms=duration_ms,
        )

        return state

    def _stage_postchecks(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 6: Post-ingestion checks (reconciliation, parity).

        Ensures graph ↔ vector alignment.
        """
        tracker.advance(JobStage.POSTCHECKS, "Running post-ingestion checks")

        start_time = time.time()

        # Run reconciliation if enabled
        reconciliation_stats = {}
        if self.config.ingestion.reconciliation.enabled and self.qdrant:
            reconciler = Reconciler(self.neo4j, self.config, self.qdrant)
            reconciliation_stats = reconciler.reconcile()

        # Save to state
        state.status = JobStage.POSTCHECKS.value
        state.stages_completed.append(JobStage.POSTCHECKS.value)

        # Update stats
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["postchecks"] = {
            "duration_ms": duration_ms,
            "reconciliation": reconciliation_stats,
        }

        self._save_state(state)

        tracker.advance(
            JobStage.POSTCHECKS,
            "Post-checks complete",
            details=reconciliation_stats,
        )

        logger.debug(
            "Post-checks complete",
            job_id=state.job_id,
            reconciliation=reconciliation_stats,
            duration_ms=duration_ms,
        )

        return state

    def _stage_reporting(self, state: JobState, tracker: ProgressTracker) -> JobState:
        """
        Stage 7: Generate ingestion report with Task 6.4 verification.

        Uses PostIngestVerifier for drift calculation and sample queries,
        and ReportGenerator for complete JSON/MD reporting.
        """
        tracker.advance(JobStage.REPORTING, "Generating report with verification")

        start_time = time.time()

        # Task 6.4 Verification
        verifier = PostIngestVerifier(
            driver=self.neo4j,
            config=self.config,
            qdrant_client=self.qdrant,
        )
        verification_result = verifier.verify_ingestion(
            job_id=state.job_id,
            parsed={"Document": state.document, "Sections": state.sections or []},
            tag=state.tag,
        )

        # Task 6.4 Report Generation
        report_gen = ReportGenerator(
            driver=self.neo4j,
            config=self.config,
            qdrant_client=self.qdrant,
        )

        # Build timings from state.stats
        timings_ms = {
            "parse": state.stats.get("parsing", {}).get("duration_ms", 0),
            "extract": state.stats.get("extracting", {}).get("duration_ms", 0),
            "graph": state.stats.get("graphing", {}).get("duration_ms", 0),
            "embed": state.stats.get("embedding", {}).get("duration_ms", 0),
        }

        # Generate complete report
        report = report_gen.generate_report(
            job_id=state.job_id,
            tag=state.tag,
            parsed={"Document": state.document, "Sections": state.sections or []},
            verdict=verification_result,
            timings=timings_ms,
        )

        # Write report to disk (use job_id as directory for test compatibility)
        output_dir = f"reports/ingest/{state.job_id}"
        report_paths = report_gen.write_report(report, output_dir=output_dir)

        # Save to state (store report path for CLI access)
        state.status = JobStage.REPORTING.value
        state.stages_completed.append(JobStage.REPORTING.value)

        # Update stats with verification results and report path
        duration_ms = int((time.time() - start_time) * 1000)
        state.stats["reporting"] = {
            "duration_ms": duration_ms,
            "report_json_path": str(report_paths["json"]),
            "report_md_path": str(report_paths["markdown"]),
            "ready_for_queries": verification_result.get("ready", False),
            "drift_pct": verification_result.get("drift_pct", 0.0),
        }

        self._save_state(state)

        tracker.advance(
            JobStage.REPORTING,
            f"Report saved to {report_paths['json']}",
            details={
                "json_path": str(report_paths["json"]),
                "md_path": str(report_paths["markdown"]),
                "ready": verification_result.get("ready", False),
                "drift_pct": verification_result.get("drift_pct", 0.0),
            },
        )

        logger.info(
            "Reporting complete",
            job_id=state.job_id,
            report_json_path=str(report_paths["json"]),
            ready_for_queries=verification_result.get("ready", False),
            drift_pct=verification_result.get("drift_pct", 0.0),
            duration_ms=duration_ms,
        )

        return state

    def _load_state(self, job_id: str) -> Optional[JobState]:
        """Load job state from Redis."""
        state_key = f"ingest:state:{job_id}"
        data = self.redis.hgetall(state_key)

        if not data:
            return None

        # Decode and deserialize
        state_dict = {}
        for key, value in data.items():
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            value_str = value.decode("utf-8") if isinstance(value, bytes) else value

            # Deserialize JSON fields
            if key_str in [
                "stages_completed",
                "document",
                "sections",
                "entities",
                "mentions",
                "stats",
            ]:
                try:
                    state_dict[key_str] = json.loads(value_str)
                except json.JSONDecodeError:
                    state_dict[key_str] = None
            elif key_str in ["created_at", "updated_at", "started_at", "completed_at"]:
                try:
                    state_dict[key_str] = (
                        float(value_str) if value_str != "None" else None
                    )
                except ValueError:
                    state_dict[key_str] = None
            else:
                state_dict[key_str] = value_str

        # Ensure required fields with defaults (replace None values too)
        if (
            "stages_completed" not in state_dict
            or state_dict["stages_completed"] is None
        ):
            state_dict["stages_completed"] = []
        if "stats" not in state_dict or state_dict["stats"] is None:
            state_dict["stats"] = {}
        if "document" not in state_dict or state_dict["document"] is None:
            state_dict["document"] = {}
        if "sections" not in state_dict or state_dict["sections"] is None:
            state_dict["sections"] = []
        if "entities" not in state_dict or state_dict["entities"] is None:
            state_dict["entities"] = []
        if "mentions" not in state_dict or state_dict["mentions"] is None:
            state_dict["mentions"] = []

        return JobState(**state_dict)

    def _save_state(self, state: JobState):
        """Save job state to Redis."""
        state_key = f"ingest:state:{state.job_id}"

        state.updated_at = time.time()

        # Serialize to dict
        state_dict = {
            "job_id": state.job_id,
            "source_uri": state.source_uri,
            "checksum": state.checksum,
            "tag": state.tag,
            "status": state.status,
            "created_at": str(state.created_at),
            "updated_at": str(state.updated_at),
            "started_at": str(state.started_at) if state.started_at else "None",
            "completed_at": str(state.completed_at) if state.completed_at else "None",
            "error": state.error or "",
            "stages_completed": json.dumps(state.stages_completed),
            "document_id": state.document_id or "",
            "document": json.dumps(state.document) if state.document else "null",
            "sections": json.dumps(state.sections) if state.sections else "null",
            "entities": json.dumps(state.entities) if state.entities else "null",
            "mentions": json.dumps(state.mentions) if state.mentions else "null",
            "stats": json.dumps(state.stats),
        }

        # Save to Redis
        self.redis.hset(state_key, mapping=state_dict)

        # Set TTL (7 days)
        self.redis.expire(state_key, 7 * 24 * 60 * 60)

    def _read_source(self, source_uri: str) -> str:
        """Read content from source URI."""
        # Handle file:// URIs
        if source_uri.startswith("file://"):
            path = source_uri[7:]  # Strip file://
        else:
            path = source_uri

        # Read file
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _purge_existing_vectors(self, document_id: str, source_uri: str):
        """Purge existing vectors for document to prevent drift."""
        if not self.qdrant:
            return

        collection_name = self.config.search.vector.qdrant.collection_name

        filter_must = [
            {"key": "node_label", "match": {"value": "Section"}},
            {
                "key": "embedding_version",
                "match": {"value": self.config.embedding.version},
            },
        ]

        if source_uri:
            filter_must.append({"key": "source_uri", "match": {"value": source_uri}})
        else:
            filter_must.append({"key": "document_id", "match": {"value": document_id}})

        try:
            self.qdrant.delete(
                collection_name=collection_name,
                points_selector={"filter": {"must": filter_must}},
                wait=True,
            )
            logger.debug(
                "Purged existing vectors",
                collection=collection_name,
                document_id=document_id,
            )
        except Exception as exc:
            logger.warning("Failed to purge existing vectors", error=str(exc))

    def _upsert_to_qdrant(
        self, sections: List[Dict], document: Dict, document_id: str
    ) -> int:
        """Upsert section embeddings to Qdrant."""
        import uuid

        from qdrant_client.models import PointStruct

        collection_name = self.config.search.vector.qdrant.collection_name
        source_uri = document.get("source_uri", "")
        document_uri = Path(source_uri).name if source_uri else source_uri

        points = []
        for section in sections:
            embedding = section.get("vector_embedding")
            if not embedding:
                continue

            # Convert section_id to UUID
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section["id"]))

            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload={
                    "node_id": section["id"],  # Original section ID
                    "node_label": "Section",
                    "document_id": document_id,
                    "document_uri": document_uri,
                    "source_uri": source_uri,
                    "title": section.get("title"),
                    "anchor": section.get("anchor"),
                    "embedding_version": self.config.embedding.version,
                },
            )
            points.append(point)

        if points:
            self.qdrant.upsert(collection_name=collection_name, points=points)

        return len(points)

    def _upsert_to_neo4j(self, sections: List[Dict]) -> int:
        """Upsert section embeddings to Neo4j."""
        with self.neo4j.session() as session:
            for section in sections:
                embedding = section.get("vector_embedding")
                if not embedding:
                    continue

                query = """
                MATCH (s:Section {id: $section_id})
                SET s.vector_embedding = $embedding,
                    s.embedding_version = $version
                """
                session.run(
                    query,
                    section_id=section["id"],
                    embedding=embedding,
                    version=self.config.embedding.version,
                )

        return len([s for s in sections if s.get("vector_embedding")])

    def _set_embedding_metadata_in_neo4j(self, sections: List[Dict]):
        """Set embedding_version metadata in Neo4j without storing vectors."""
        with self.neo4j.session() as session:
            for section in sections:
                query = """
                MATCH (s:Section {id: $section_id})
                SET s.embedding_version = $version
                """
                session.run(
                    query,
                    section_id=section["id"],
                    version=self.config.embedding.version,
                )

    def _build_report(self, state: JobState) -> Dict:
        """Build ingestion report."""
        # Calculate totals
        total_duration_ms = sum(
            stage_stats.get("duration_ms", 0) for stage_stats in state.stats.values()
        )

        # Calculate drift percentage
        reconciliation = state.stats.get("postchecks", {}).get("reconciliation", {})
        drift_pct = reconciliation.get("drift_pct", 0.0)

        # Determine readiness
        ready_for_queries = (
            drift_pct <= 0.005  # Drift < 0.5%
            and state.stats.get("vectors", {}).get("vectors_upserted", 0) > 0
        )

        return {
            "job_id": state.job_id,
            "tag": state.tag,
            "source_uri": state.source_uri,
            "checksum": state.checksum,
            "document_id": state.document_id,
            "sections": len(state.sections) if state.sections else 0,
            "entities": len(state.entities) if state.entities else 0,
            "mentions": len(state.mentions) if state.mentions else 0,
            "drift_pct": drift_pct,
            "ready_for_queries": ready_for_queries,
            "timings_ms": state.stats,
            "total_duration_ms": total_duration_ms,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
        }

    def _render_report_markdown(self, report: Dict) -> str:
        """Render report as Markdown."""
        md = f"""# Ingestion Report

**Job ID:** `{report['job_id']}`
**Tag:** `{report['tag']}`
**Source:** `{report['source_uri']}`

## Summary

- **Sections:** {report['sections']}
- **Entities:** {report['entities']}
- **Mentions:** {report['mentions']}
- **Drift:** {report['drift_pct']:.2%}
- **Ready for Queries:** {'✅ Yes' if report['ready_for_queries'] else '❌ No'}

## Timings

"""

        for stage, stats in report["timings_ms"].items():
            duration = stats.get("duration_ms", 0)
            md += f"- **{stage.capitalize()}:** {duration}ms\n"

        md += f"\n**Total Duration:** {report['total_duration_ms']}ms\n"

        return md
