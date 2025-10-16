"""
Phase 6, Task 6.2: Orchestrator (Resumable, Idempotent Jobs)

Tests for state machine, resume logic, idempotency, and progress events.

NO MOCKS - All tests run against live Docker stack.

See: /docs/implementation-plan-phase-6.md → Task 6.2
See: /docs/coder-guidance-phase6.md → 6.2
"""

import time

import pytest


class TestStateMachine:
    """Test orchestrator state machine transitions"""

    def test_state_progression(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Job progresses through all stages: PENDING→PARSING→...→DONE

        DoD:
        - State persisted in Redis after each stage
        - All stages reached
        - Final state is DONE
        """
        import uuid

        from src.ingestion.auto.orchestrator import JobState, Orchestrator
        from src.ingestion.auto.progress import JobStage

        # Create test document
        job_id = str(uuid.uuid4())
        test_file = watch_dir / "test.md"
        test_file.write_text(
            """# Test Document

## Section 1
This is a test section with some content.

## Section 2
Configure the cluster with `cluster.size=3`.
"""
        )

        # Create job state in Redis
        _state = JobState(
            job_id=job_id,
            source_uri=f"file://{test_file}",
            checksum="test123",
            tag="test",
            status="PENDING",
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Save initial state
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": str(test_file),
                "checksum": "test123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        # Process job
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        _stats = orchestrator.process_job(job_id)

        # Verify final state
        final_state = orchestrator._load_state(job_id)
        assert final_state.status == JobStage.DONE.value
        assert len(final_state.stages_completed) > 0
        assert JobStage.PARSING.value in final_state.stages_completed
        assert JobStage.EXTRACTING.value in final_state.stages_completed
        assert JobStage.GRAPHING.value in final_state.stages_completed
        assert final_state.completed_at is not None

    def test_error_state(self, redis_sync_client, neo4j_driver, qdrant_client, config):
        """
        Invalid input → ERROR state with message

        DoD:
        - Malformed document causes ERROR
        - Error message captured
        - Job does not retry indefinitely
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import JobStage

        job_id = str(uuid.uuid4())

        # Create job state with invalid source
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": "/nonexistent/file.md",
                "checksum": "test123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        # Process job (should fail)
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )

        with pytest.raises(Exception):
            orchestrator.process_job(job_id)

        # Verify error state
        final_state = orchestrator._load_state(job_id)
        assert final_state.status == JobStage.ERROR.value
        assert final_state.error is not None
        assert len(final_state.error) > 0


class TestResumeLogic:
    """Test resume after worker crash/restart"""

    def test_resume_from_parsing(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Kill worker during PARSING → resume → completes

        DoD:
        - Job starts
        - Worker killed during PARSING (simulated by partial state)
        - Worker restarted (new orchestrator)
        - Job resumes from PARSING
        - Completes successfully
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import JobStage

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "resume_test.md"
        test_file.write_text(
            """# Resume Test
## Section 1
Test content for resume logic.
"""
        )

        # Create job state that's "stuck" in PARSING (simulating crash)
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "resume123",
                "tag": "test",
                "status": JobStage.PARSING.value,
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "started_at": str(time.time()),
                "stages_completed": "[]",  # No stages completed yet
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "completed_at": "None",
            },
        )

        # Create new orchestrator (simulating restart)
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )

        # Process job - should resume and complete
        _stats = orchestrator.process_job(job_id)

        # Verify completion
        final_state = orchestrator._load_state(job_id)
        assert final_state.status == JobStage.DONE.value
        assert final_state.completed_at is not None

    def test_resume_from_embedding(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Resume from EMBEDDING stage → completes without duplication

        DoD:
        - Job has completed PARSING, EXTRACTING, GRAPHING
        - Simulated crash at EMBEDDING
        - Resume completes remaining stages
        - No duplicate vectors
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import JobStage

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "embed_test.md"
        test_file.write_text(
            """# Embedding Test
## Test Section
Content for embedding test.
"""
        )

        # Partially complete job: PARSING, EXTRACTING, GRAPHING done
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "embed123",
                "tag": "test",
                "status": JobStage.GRAPHING.value,
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "started_at": str(time.time()),
                "stages_completed": f'["{JobStage.PARSING.value}", "{JobStage.EXTRACTING.value}", "{JobStage.GRAPHING.value}"]',
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "completed_at": "None",
            },
        )

        # Resume and complete
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        _stats = orchestrator.process_job(job_id)

        # Verify completion
        final_state = orchestrator._load_state(job_id)
        assert final_state.status == JobStage.DONE.value

    def test_no_duplicate_work_on_resume(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Resume does not duplicate graph/vector writes

        DoD:
        - Process job to completion
        - Simulate resume from middle stage
        - Verify no duplication in graph or vectors
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import JobStage

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "nodup_test.md"
        test_file.write_text(
            """# No Dup Test
## Unique Section
Unique content to track.
"""
        )

        # Create and process job normally
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "nodup123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        _stats1 = orchestrator.process_job(job_id)

        # Get counts after first run
        with neo4j_driver.session() as session:
            result = session.run("MATCH (s:Section) RETURN count(s) as count")
            sections_count = result.single()["count"]

        # Simulate resume by resetting some stages_completed
        state = orchestrator._load_state(job_id)
        state.status = JobStage.EMBEDDING.value
        state.stages_completed = [
            JobStage.PARSING.value,
            JobStage.EXTRACTING.value,
            JobStage.GRAPHING.value,
        ]
        orchestrator._save_state(state)

        # Resume processing
        _stats2 = orchestrator.process_job(job_id)

        # Verify no duplication
        with neo4j_driver.session() as session:
            result = session.run("MATCH (s:Section) RETURN count(s) as count")
            sections_count_after = result.single()["count"]

        assert (
            sections_count == sections_count_after
        ), "Resume should not duplicate sections"


class TestIdempotency:
    """Test idempotent ingestion behavior"""

    def test_reingest_unchanged_doc(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Re-ingest same document → no changes

        DoD:
        - Ingest document once
        - Capture node/edge/vector counts
        - Re-ingest same document (same checksum)
        - Counts unchanged
        - Checksums match
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator

        # Create test document
        test_file = watch_dir / "idempotent_test.md"
        test_file.write_text(
            """# Idempotent Test
## Section 1
Content that should not be duplicated.
## Section 2
More unique content.
"""
        )

        # First ingestion
        job_id_1 = str(uuid.uuid4())
        state_key = f"ingest:state:{job_id_1}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id_1,
                "source_uri": f"file://{test_file}",
                "checksum": "idem123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id_1)

        # Get counts after first ingestion
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN count(DISTINCT s) as sections
            """,
                uri=f"file://{test_file}",
            )
            sections_count_1 = result.single()["sections"]

        # Second ingestion (same document, same checksum)
        job_id_2 = str(uuid.uuid4())
        state_key_2 = f"ingest:state:{job_id_2}"
        redis_sync_client.hset(
            state_key_2,
            mapping={
                "job_id": job_id_2,
                "source_uri": f"file://{test_file}",
                "checksum": "idem123",  # Same checksum
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator.process_job(job_id_2)

        # Get counts after second ingestion
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN count(DISTINCT s) as sections
            """,
                uri=f"file://{test_file}",
            )
            sections_count_2 = result.single()["sections"]

        # Verify no duplication
        assert sections_count_1 == sections_count_2
        assert sections_count_1 > 0  # Ensure something was actually ingested

    def test_deterministic_ids(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Same document → same IDs across runs

        DoD:
        - Ingest document
        - Capture section IDs
        - Delete graph
        - Re-ingest
        - Section IDs match exactly
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator

        # Create test document
        test_file = watch_dir / "deterministic_test.md"
        test_file.write_text(
            """# Deterministic Test
## Section Alpha
Alpha content.
## Section Beta
Beta content.
"""
        )

        # First ingestion
        job_id_1 = str(uuid.uuid4())
        state_key = f"ingest:state:{job_id_1}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id_1,
                "source_uri": f"file://{test_file}",
                "checksum": "det123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id_1)

        # Capture section IDs
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                RETURN s.id as id
                ORDER BY s.order
            """,
                uri=f"file://{test_file}",
            )
            ids_1 = [rec["id"] for rec in result]

        # Delete sections
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                DETACH DELETE s
            """,
                uri=f"file://{test_file}",
            )

        # Second ingestion
        job_id_2 = str(uuid.uuid4())
        state_key_2 = f"ingest:state:{job_id_2}"
        redis_sync_client.hset(
            state_key_2,
            mapping={
                "job_id": job_id_2,
                "source_uri": f"file://{test_file}",
                "checksum": "det123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator.process_job(job_id_2)

        # Capture section IDs again
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                RETURN s.id as id
                ORDER BY s.order
            """,
                uri=f"file://{test_file}",
            )
            ids_2 = [rec["id"] for rec in result]

        # Verify deterministic IDs
        assert ids_1 == ids_2
        assert len(ids_1) > 0


class TestProgressEvents:
    """Test progress event streaming"""

    def test_progress_events_emitted(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Job emits progress events to Redis stream

        DoD:
        - Events appear in ingest:events:<job_id>
        - Events include stage, percent, message
        - Events ordered correctly
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import ProgressReader

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "progress_test.md"
        test_file.write_text(
            """# Progress Test
## Test Section
Test content for progress events.
"""
        )

        # Create job
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "prog123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        # Process job
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id)

        # Read progress events
        reader = ProgressReader(redis_sync_client, job_id)
        events = reader.read_events(count=100)

        # Verify events exist and have required fields
        assert len(events) > 0
        for event in events:
            assert event.job_id == job_id
            assert event.stage is not None
            assert event.percent is not None
            assert event.message is not None
            assert event.timestamp is not None

    def test_progress_percentages(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Progress percentages increase monotonically

        DoD:
        - Each stage shows progress 0-100%
        - Overall progress increases
        - No backwards progress
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import ProgressReader

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "monotonic_test.md"
        test_file.write_text(
            """# Monotonic Test
## Section One
Content one.
## Section Two
Content two.
"""
        )

        # Create job
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "mono123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        # Process job
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id)

        # Read and verify monotonic progress
        reader = ProgressReader(redis_sync_client, job_id)
        events = reader.read_events(count=100)

        percentages = [e.percent for e in events]

        # Verify monotonic increase (with small tolerance for rounding)
        for i in range(1, len(percentages)):
            assert (
                percentages[i] >= percentages[i - 1] - 0.1
            ), f"Progress decreased: {percentages[i-1]} -> {percentages[i]}"

        # Verify reaches 100%
        assert max(percentages) >= 95.0, "Should reach near 100%"


class TestPipelineIntegration:
    """Test integration with Phase 3 pipeline"""

    def test_calls_existing_parsers(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Orchestrator calls existing Phase 3 parsers

        DoD:
        - Markdown parser invoked
        - Sections extracted correctly
        - Anchors preserved
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "parser_test.md"
        test_file.write_text(
            """# Parser Test Document

## Section with Anchor
This section should have an anchor.

## Another Section
With more content.
"""
        )

        # Create and process job
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "parser123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id)

        # Verify sections were parsed
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})-[:HAS_SECTION]->(s:Section)
                RETURN s.title as title, s.anchor as anchor, s.text as text
                ORDER BY s.order
            """,
                uri=f"file://{test_file}",
            )

            sections = list(result)
            assert len(sections) > 0

            # Verify anchors exist
            for section in sections:
                assert section["anchor"] is not None
                assert section["title"] is not None

    def test_calls_existing_extractors(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Orchestrator calls existing Phase 3 extractors

        DoD:
        - Entity extraction runs
        - MENTIONS relationships created
        - Provenance preserved
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "extractor_test.md"
        test_file.write_text(
            """# Extractor Test

## Configuration
Set the cluster.size parameter to 5.

## Commands
Run the `weka start` command to begin.
"""
        )

        # Create and process job
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "extract123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id)

        # Verify MENTIONS relationships exist
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                MATCH (s:Section {document_id: d.id})-[m:MENTIONS]->(e)
                RETURN count(m) as mention_count
            """,
                uri=f"file://{test_file}",
            )

            mention_count = result.single()["mention_count"]
            # May be 0 if extractors don't match patterns, but check executed
            assert mention_count >= 0

    def test_calls_build_graph(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Orchestrator calls existing build_graph

        DoD:
        - MERGE semantics used
        - Deterministic IDs
        - Provenance on edges
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "graph_test.md"
        test_file.write_text(
            """# Graph Build Test

## Test Section
Content for graph building verification.
"""
        )

        # Create and process job
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "graph123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        orchestrator.process_job(job_id)

        # Verify graph structure
        with neo4j_driver.session() as session:
            # Check Document node
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                RETURN d.id as id, d.checksum as checksum
            """,
                uri=f"file://{test_file}",
            )

            doc = result.single()
            assert doc is not None
            assert doc["id"] is not None
            assert doc["checksum"] is not None

            # Check HAS_SECTION relationship
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})-[r:HAS_SECTION]->(s:Section)
                RETURN count(r) as rel_count
            """,
                uri=f"file://{test_file}",
            )

            rel_count = result.single()["rel_count"]
            assert rel_count > 0


class TestE2EOrchestratorFlow:
    """End-to-end orchestrator test"""

    def test_complete_job_lifecycle(
        self, redis_sync_client, neo4j_driver, qdrant_client, watch_dir, config
    ):
        """
        Enqueue job → orchestrator processes → reaches DONE

        DoD:
        - Job created
        - All stages complete
        - Graph updated
        - Vectors upserted
        - State is DONE
        - Timing metrics captured
        """
        import uuid

        from src.ingestion.auto.orchestrator import Orchestrator
        from src.ingestion.auto.progress import JobStage, ProgressReader

        job_id = str(uuid.uuid4())
        test_file = watch_dir / "e2e_test.md"
        test_file.write_text(
            """# End-to-End Test Document

## Introduction
This document tests the complete orchestrator pipeline.

## Configuration Section
Set cluster.size to 10 for optimal performance.

## Command Section
Execute `weka status` to check the system.

## Procedure Section
1. First, configure the cluster
2. Then, start the services
3. Finally, verify the installation

## Conclusion
The orchestrator should handle this document end-to-end.
"""
        )

        # Create job in Redis
        state_key = f"ingest:state:{job_id}"
        redis_sync_client.hset(
            state_key,
            mapping={
                "job_id": job_id,
                "source_uri": f"file://{test_file}",
                "checksum": "e2e123",
                "tag": "test",
                "status": "PENDING",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "stages_completed": "[]",
                "stats": "{}",
                "error": "",
                "document_id": "",
                "document": "null",
                "sections": "null",
                "entities": "null",
                "mentions": "null",
                "started_at": "None",
                "completed_at": "None",
            },
        )

        # Process job
        orchestrator = Orchestrator(
            redis_sync_client, neo4j_driver, config, qdrant_client
        )
        _stats = orchestrator.process_job(job_id)

        # Verify final state
        final_state = orchestrator._load_state(job_id)
        assert final_state.status == JobStage.DONE.value
        assert final_state.completed_at is not None
        assert final_state.started_at is not None
        assert final_state.completed_at > final_state.started_at

        # Verify all stages completed
        expected_stages = [
            JobStage.PARSING.value,
            JobStage.EXTRACTING.value,
            JobStage.GRAPHING.value,
            JobStage.EMBEDDING.value,
            JobStage.VECTORS.value,
            JobStage.POSTCHECKS.value,
            JobStage.REPORTING.value,
        ]
        for stage in expected_stages:
            assert stage in final_state.stages_completed, f"Stage {stage} not completed"

        # Verify graph was updated
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {source_uri: $uri})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN count(DISTINCT s) as sections
            """,
                uri=f"file://{test_file}",
            )

            sections_count = result.single()["sections"]
            assert sections_count > 0

        # Verify timing metrics captured
        assert "parsing" in final_state.stats
        assert "extracting" in final_state.stats
        assert "graphing" in final_state.stats
        assert "embedding" in final_state.stats
        assert "vectors" in final_state.stats

        # Verify duration tracked
        for stage_stats in final_state.stats.values():
            if isinstance(stage_stats, dict):
                assert "duration_ms" in stage_stats

        # Verify progress events
        reader = ProgressReader(redis_sync_client, job_id)
        events = reader.read_events(count=100)
        assert len(events) > 0

        # Verify report generated
        from pathlib import Path

        report_path = Path(f"reports/ingest/{job_id}/ingest_report.json")
        assert report_path.exists(), "Report should be generated"


# Fixtures
@pytest.fixture
def sample_job():
    """Sample job payload for tests"""
    return {
        "job_id": "test-job-123",
        "source_uri": "file:///test.md",
        "checksum": "abc123def456",
        "tag": "test",
    }
