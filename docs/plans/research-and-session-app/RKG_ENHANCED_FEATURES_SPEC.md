# Research Knowledge Graph (RKG) - Enhanced Features Specification

## Document Overview

This specification extends the core RKG system with:
1. **Missing Core Features** - Entity extraction, incremental sync, export, analytics, contradiction detection
2. **LangFuse Integration** - LLM observability, tracing, prompt management
3. **NewRelic Integration** - Application monitoring, distributed tracing, logs-in-context
4. **Infisical Integration** - Secrets management with Pro/Free tier feature flags

---

# Part 1: Missing Core Features

## 1.1 Entity Extraction Pipeline

### Overview
Automatic Named Entity Recognition (NER) to extract people, technologies, organizations, and concepts from documents, linking them as first-class nodes in the Neo4j graph.

### Technology Selection: GLiNER + spaCy Hybrid

**Primary**: GLiNER (Generalist Lightweight Named Entity Recognition)
- Zero-shot entity extraction for any entity type
- Bidirectional transformer encoder with parallel extraction
- 350M parameters, runs efficiently on CPU
- Custom entity types without retraining

**Secondary**: spaCy en_core_web_trf
- Transformer-based pipeline for standard NER
- High accuracy on OntoNotes entities (PERSON, ORG, GPE, PRODUCT, etc.)
- Fast inference with GPU acceleration

### Entity Types Schema

```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class EntityType(str, Enum):
    PERSON = "person"           # People, researchers, authors
    ORGANIZATION = "organization"  # Companies, institutions
    TECHNOLOGY = "technology"   # Frameworks, libraries, tools
    CONCEPT = "concept"         # AI/ML concepts, methodologies
    PRODUCT = "product"         # Software products, APIs
    LOCATION = "location"       # Geographic locations
    EVENT = "event"             # Conferences, releases
    METRIC = "metric"           # Performance metrics, benchmarks
    CUSTOM = "custom"           # User-defined entity types

class ExtractedEntity(BaseModel):
    """Entity extracted from document text."""
    text: str                   # Surface form as found in text
    entity_type: EntityType
    normalized_name: str        # Canonical form for linking
    confidence: float           # Extraction confidence 0-1
    start_char: int             # Character offset in source
    end_char: int
    context: str                # Surrounding text snippet

class EntityExtractionResult(BaseModel):
    """Result of entity extraction on a document."""
    document_id: str
    entities: List[ExtractedEntity]
    extraction_model: str       # "gliner" or "spacy"
    processing_time_ms: float
```

### GLiNER Integration

```python
# src/rkg_mcp/extraction/gliner_extractor.py

from gliner import GLiNER
from typing import List, Dict, Any
import asyncio
from functools import lru_cache

class GLiNEREntityExtractor:
    """Zero-shot entity extraction using GLiNER."""

    # Custom entity labels for research knowledge domain
    DEFAULT_LABELS = [
        "person", "researcher", "author",
        "organization", "company", "institution",
        "technology", "framework", "library", "tool", "api",
        "concept", "methodology", "algorithm",
        "product", "software", "service",
        "metric", "benchmark", "performance measure",
        "event", "conference", "release"
    ]

    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        threshold: float = 0.5,
        custom_labels: List[str] | None = None
    ):
        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold
        self.labels = custom_labels or self.DEFAULT_LABELS

    async def extract_entities(
        self,
        text: str,
        labels: List[str] | None = None,
        threshold: float | None = None
    ) -> List[ExtractedEntity]:
        """Extract entities from text using GLiNER."""
        labels = labels or self.labels
        threshold = threshold or self.threshold

        # Run in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None,
            lambda: self.model.predict_entities(text, labels, threshold=threshold)
        )

        entities = []
        for pred in predictions:
            entity = ExtractedEntity(
                text=pred["text"],
                entity_type=self._map_label_to_type(pred["label"]),
                normalized_name=self._normalize_entity_name(pred["text"]),
                confidence=pred["score"],
                start_char=pred["start"],
                end_char=pred["end"],
                context=self._extract_context(text, pred["start"], pred["end"])
            )
            entities.append(entity)

        return self._deduplicate_entities(entities)

    def _map_label_to_type(self, label: str) -> EntityType:
        """Map GLiNER label to EntityType enum."""
        label_lower = label.lower()
        mappings = {
            ("person", "researcher", "author"): EntityType.PERSON,
            ("organization", "company", "institution"): EntityType.ORGANIZATION,
            ("technology", "framework", "library", "tool", "api"): EntityType.TECHNOLOGY,
            ("concept", "methodology", "algorithm"): EntityType.CONCEPT,
            ("product", "software", "service"): EntityType.PRODUCT,
            ("metric", "benchmark", "performance measure"): EntityType.METRIC,
            ("event", "conference", "release"): EntityType.EVENT,
        }
        for keys, entity_type in mappings.items():
            if label_lower in keys:
                return entity_type
        return EntityType.CUSTOM

    def _normalize_entity_name(self, text: str) -> str:
        """Normalize entity name for consistent linking."""
        # Lowercase, strip whitespace, collapse spaces
        normalized = " ".join(text.lower().split())
        return normalized

    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract surrounding context for entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.normalized_name, entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())
```

### spaCy Integration (Fallback/Enhancement)

```python
# src/rkg_mcp/extraction/spacy_extractor.py

import spacy
from typing import List
import asyncio

class SpaCyEntityExtractor:
    """Standard NER extraction using spaCy transformer model."""

    SPACY_TO_ENTITY_TYPE = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORGANIZATION,
        "GPE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
        "PRODUCT": EntityType.PRODUCT,
        "EVENT": EntityType.EVENT,
        "WORK_OF_ART": EntityType.CONCEPT,
        "LAW": EntityType.CONCEPT,
    }

    def __init__(self, model_name: str = "en_core_web_trf"):
        self.nlp = spacy.load(model_name)

    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self.nlp, text)

        entities = []
        for ent in doc.ents:
            entity_type = self.SPACY_TO_ENTITY_TYPE.get(ent.label_, EntityType.CUSTOM)
            entities.append(ExtractedEntity(
                text=ent.text,
                entity_type=entity_type,
                normalized_name=ent.text.lower().strip(),
                confidence=0.85,  # spaCy doesn't provide confidence scores
                start_char=ent.start_char,
                end_char=ent.end_char,
                context=text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
            ))
        return entities
```

### Hybrid Extractor with Entity Linking

```python
# src/rkg_mcp/extraction/hybrid_extractor.py

from typing import List, Dict, Optional
from .gliner_extractor import GLiNEREntityExtractor
from .spacy_extractor import SpaCyEntityExtractor

class HybridEntityExtractor:
    """
    Combines GLiNER (zero-shot) with spaCy (transformer NER)
    for comprehensive entity extraction.
    """

    def __init__(
        self,
        use_gliner: bool = True,
        use_spacy: bool = True,
        gliner_threshold: float = 0.5,
        merge_strategy: str = "union"  # "union" | "intersection" | "gliner_primary"
    ):
        self.gliner = GLiNEREntityExtractor(threshold=gliner_threshold) if use_gliner else None
        self.spacy = SpaCyEntityExtractor() if use_spacy else None
        self.merge_strategy = merge_strategy

    async def extract_entities(
        self,
        text: str,
        custom_labels: List[str] | None = None
    ) -> EntityExtractionResult:
        """Extract entities using hybrid approach."""
        import time
        start_time = time.time()

        gliner_entities = []
        spacy_entities = []

        if self.gliner:
            gliner_entities = await self.gliner.extract_entities(text, labels=custom_labels)

        if self.spacy:
            spacy_entities = await self.spacy.extract_entities(text)

        # Merge based on strategy
        if self.merge_strategy == "union":
            merged = self._merge_union(gliner_entities, spacy_entities)
        elif self.merge_strategy == "intersection":
            merged = self._merge_intersection(gliner_entities, spacy_entities)
        else:  # gliner_primary
            merged = gliner_entities or spacy_entities

        processing_time = (time.time() - start_time) * 1000

        return EntityExtractionResult(
            document_id="",  # Set by caller
            entities=merged,
            extraction_model="hybrid" if self.gliner and self.spacy else "gliner" if self.gliner else "spacy",
            processing_time_ms=processing_time
        )

    def _merge_union(
        self,
        gliner: List[ExtractedEntity],
        spacy: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Merge entities from both extractors, deduplicating by normalized name."""
        all_entities = gliner + spacy
        seen = {}
        for entity in all_entities:
            key = (entity.normalized_name, entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())

    def _merge_intersection(
        self,
        gliner: List[ExtractedEntity],
        spacy: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Only keep entities found by both extractors."""
        gliner_keys = {(e.normalized_name, e.entity_type) for e in gliner}
        spacy_keys = {(e.normalized_name, e.entity_type) for e in spacy}
        common_keys = gliner_keys & spacy_keys

        # Return GLiNER version (usually more precise)
        return [e for e in gliner if (e.normalized_name, e.entity_type) in common_keys]
```

### Neo4j Entity Storage

```cypher
// Entity node creation/update
MERGE (e:Entity {normalized_name: $normalized_name, entity_type: $entity_type})
ON CREATE SET
    e.id = randomUUID(),
    e.created_at = datetime(),
    e.mention_count = 1,
    e.first_seen = $document_id,
    e.surface_forms = [$text]
ON MATCH SET
    e.mention_count = e.mention_count + 1,
    e.surface_forms = CASE
        WHEN NOT $text IN e.surface_forms
        THEN e.surface_forms + $text
        ELSE e.surface_forms
    END

// Link entity to document
MATCH (d:Document {id: $document_id})
MATCH (e:Entity {normalized_name: $normalized_name, entity_type: $entity_type})
MERGE (d)-[r:MENTIONS]->(e)
ON CREATE SET
    r.confidence = $confidence,
    r.first_mention_char = $start_char,
    r.context = $context,
    r.created_at = datetime()
ON MATCH SET
    r.mention_count = coalesce(r.mention_count, 1) + 1

// Create entity co-occurrence relationships
MATCH (d:Document {id: $document_id})-[:MENTIONS]->(e1:Entity)
MATCH (d)-[:MENTIONS]->(e2:Entity)
WHERE e1.id < e2.id  // Avoid duplicates
MERGE (e1)-[r:CO_OCCURS_WITH]->(e2)
ON CREATE SET r.count = 1, r.documents = [$document_id]
ON MATCH SET
    r.count = r.count + 1,
    r.documents = CASE
        WHEN NOT $document_id IN r.documents
        THEN r.documents + $document_id
        ELSE r.documents
    END
```

### Background Processing Pipeline

```python
# src/rkg_mcp/extraction/pipeline.py

import asyncio
from typing import AsyncGenerator
from ..storage.neo4j import Neo4jStorage
from ..storage.qdrant import QdrantStorage

class EntityExtractionPipeline:
    """Background pipeline for entity extraction on ingested documents."""

    def __init__(
        self,
        extractor: HybridEntityExtractor,
        neo4j: Neo4jStorage,
        batch_size: int = 10,
        max_concurrent: int = 4
    ):
        self.extractor = extractor
        self.neo4j = neo4j
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_document(self, document_id: str, content: str) -> EntityExtractionResult:
        """Extract entities from a single document."""
        async with self.semaphore:
            result = await self.extractor.extract_entities(content)
            result.document_id = document_id

            # Store entities in Neo4j
            for entity in result.entities:
                await self.neo4j.create_entity(
                    document_id=document_id,
                    entity=entity
                )

            return result

    async def process_backlog(self) -> AsyncGenerator[EntityExtractionResult, None]:
        """Process documents that haven't been entity-extracted yet."""
        # Query for documents without entity extraction
        unprocessed = await self.neo4j.get_documents_without_entities()

        for batch in self._batch(unprocessed, self.batch_size):
            tasks = [
                self.process_document(doc["id"], doc["content"])
                for doc in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, EntityExtractionResult):
                    yield result

    def _batch(self, items, size):
        for i in range(0, len(items), size):
            yield items[i:i + size]
```

---

## 1.2 Incremental Session Sync

### Overview
Real-time monitoring of Claude Code and OpenAI Codex session directories for automatic ingestion of new research sessions.

### File Watcher Implementation

```python
# src/rkg_mcp/sync/watcher.py

import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from typing import Callable, Awaitable, Set
import logging

logger = logging.getLogger(__name__)

class SessionFileHandler(FileSystemEventHandler):
    """Handle filesystem events for session files."""

    def __init__(
        self,
        callback: Callable[[Path], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        patterns: Set[str] = {"*.jsonl"}
    ):
        self.callback = callback
        self.loop = loop
        self.patterns = patterns
        self._debounce_tasks: dict[Path, asyncio.Task] = {}
        self._debounce_delay = 2.0  # Wait for file to finish writing

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_callback(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_callback(Path(event.src_path))

    def _schedule_callback(self, path: Path):
        """Debounced callback scheduling."""
        if not any(path.match(p) for p in self.patterns):
            return

        # Cancel existing task for this path
        if path in self._debounce_tasks:
            self._debounce_tasks[path].cancel()

        # Schedule new debounced task
        async def debounced_callback():
            await asyncio.sleep(self._debounce_delay)
            try:
                await self.callback(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
            finally:
                self._debounce_tasks.pop(path, None)

        task = asyncio.run_coroutine_threadsafe(
            debounced_callback(),
            self.loop
        )
        self._debounce_tasks[path] = task


class IncrementalSessionSync:
    """
    Watches session directories for new/modified files
    and triggers automatic ingestion.
    """

    DEFAULT_WATCH_PATHS = [
        Path.home() / ".claude" / "projects",
        Path.home() / ".codex" / "sessions",
    ]

    def __init__(
        self,
        ingestion_callback: Callable[[Path], Awaitable[None]],
        watch_paths: list[Path] | None = None,
        recursive: bool = True
    ):
        self.callback = ingestion_callback
        self.watch_paths = watch_paths or self.DEFAULT_WATCH_PATHS
        self.recursive = recursive
        self.observer = Observer()
        self._running = False

    async def start(self):
        """Start watching session directories."""
        loop = asyncio.get_event_loop()
        handler = SessionFileHandler(self.callback, loop)

        for watch_path in self.watch_paths:
            if watch_path.exists():
                self.observer.schedule(
                    handler,
                    str(watch_path),
                    recursive=self.recursive
                )
                logger.info(f"Watching: {watch_path}")
            else:
                logger.warning(f"Watch path does not exist: {watch_path}")

        self.observer.start()
        self._running = True
        logger.info("Incremental sync started")

    async def stop(self):
        """Stop watching."""
        if self._running:
            self.observer.stop()
            self.observer.join()
            self._running = False
            logger.info("Incremental sync stopped")

    async def scan_existing(self) -> list[Path]:
        """Scan for existing session files that haven't been ingested."""
        existing_files = []
        for watch_path in self.watch_paths:
            if watch_path.exists():
                existing_files.extend(watch_path.rglob("*.jsonl"))
        return existing_files
```

### Sync State Management

```python
# src/rkg_mcp/sync/state.py

from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

class FileSyncState(BaseModel):
    """Track sync state for a single file."""
    path: str
    last_modified: datetime
    last_synced: datetime
    last_position: int  # Byte offset for incremental reading
    checksum: str
    session_id: str | None = None
    error: str | None = None

class SyncStateManager:
    """Persistent state management for incremental sync."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state: Dict[str, FileSyncState] = {}
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self._state = {
                k: FileSyncState(**v)
                for k, v in data.items()
            }

    def _save_state(self):
        """Persist state to disk."""
        data = {k: v.model_dump(mode="json") for k, v in self._state.items()}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2, default=str))

    def get_file_state(self, path: Path) -> FileSyncState | None:
        """Get sync state for a file."""
        return self._state.get(str(path))

    def update_file_state(
        self,
        path: Path,
        last_position: int,
        session_id: str | None = None,
        error: str | None = None
    ):
        """Update sync state for a file."""
        import hashlib

        checksum = hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else ""

        self._state[str(path)] = FileSyncState(
            path=str(path),
            last_modified=datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else datetime.now(),
            last_synced=datetime.now(),
            last_position=last_position,
            checksum=checksum,
            session_id=session_id,
            error=error
        )
        self._save_state()

    def needs_sync(self, path: Path) -> bool:
        """Check if file needs to be synced."""
        state = self.get_file_state(path)
        if not state:
            return True

        if not path.exists():
            return False

        current_mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return current_mtime > state.last_modified
```

---

## 1.3 Export Capabilities

### Overview
Export knowledge graph subsets for sharing, backup, and migration.

### Export Formats

```python
# src/rkg_mcp/export/formats.py

from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ExportFormat(str, Enum):
    JSON = "json"           # Full JSON export
    JSONL = "jsonl"         # Streaming JSON Lines
    MARKDOWN = "markdown"   # Human-readable markdown
    CYPHER = "cypher"       # Neo4j Cypher statements
    CSV = "csv"             # Tabular export

class ExportScope(BaseModel):
    """Define what to export."""
    projects: List[str] | None = None       # Filter by project names
    sessions: List[str] | None = None       # Specific session IDs
    date_from: datetime | None = None
    date_to: datetime | None = None
    entity_types: List[str] | None = None   # Filter by entity types
    include_documents: bool = True
    include_insights: bool = True
    include_entities: bool = True
    include_embeddings: bool = False        # Large, usually exclude
    max_documents: int | None = None

class ExportMetadata(BaseModel):
    """Metadata for export package."""
    export_id: str
    created_at: datetime
    scope: ExportScope
    format: ExportFormat
    document_count: int
    insight_count: int
    entity_count: int
    session_count: int
    rkg_version: str
    checksum: str
```

### Export Implementation

```python
# src/rkg_mcp/export/exporter.py

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator
import zipfile
import io

class KnowledgeGraphExporter:
    """Export knowledge graph data in various formats."""

    def __init__(self, neo4j: Neo4jStorage, qdrant: QdrantStorage):
        self.neo4j = neo4j
        self.qdrant = qdrant

    async def export_to_json(
        self,
        scope: ExportScope,
        output_path: Path
    ) -> ExportMetadata:
        """Export to JSON with full structure."""
        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Fetch data based on scope
        documents = await self._fetch_documents(scope)
        sessions = await self._fetch_sessions(scope)
        insights = await self._fetch_insights(scope)
        entities = await self._fetch_entities(scope) if scope.include_entities else []

        export_data = {
            "metadata": {
                "export_id": export_id,
                "created_at": datetime.now().isoformat(),
                "scope": scope.model_dump(mode="json"),
            },
            "sessions": sessions,
            "documents": documents,
            "insights": insights,
            "entities": entities,
            "relationships": await self._fetch_relationships(scope)
        }

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        # Generate checksum
        checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()

        return ExportMetadata(
            export_id=export_id,
            created_at=datetime.now(),
            scope=scope,
            format=ExportFormat.JSON,
            document_count=len(documents),
            insight_count=len(insights),
            entity_count=len(entities),
            session_count=len(sessions),
            rkg_version="1.0.0",
            checksum=checksum
        )

    async def export_to_markdown(
        self,
        scope: ExportScope,
        output_dir: Path
    ) -> ExportMetadata:
        """Export to human-readable markdown files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        sessions = await self._fetch_sessions(scope)

        for session in sessions:
            session_dir = output_dir / session["project"] / session["id"]
            session_dir.mkdir(parents=True, exist_ok=True)

            # Session overview
            overview = f"""# Session: {session['id']}

**Project**: {session['project']}
**Interface**: {session['agentic_interface']}
**Started**: {session['started_at']}
**Ended**: {session.get('ended_at', 'Ongoing')}

## Documents Captured

"""
            documents = await self._fetch_session_documents(session["id"])
            for doc in documents:
                overview += f"### {doc['title']}\n\n"
                overview += f"**Source**: {doc['source_url']}\n"
                overview += f"**Type**: {doc['source_type']}\n\n"
                overview += f"{doc.get('content_preview', '')}\n\n"
                overview += "---\n\n"

            # Write session file
            (session_dir / "README.md").write_text(overview)

            # Write individual documents
            for doc in documents:
                doc_file = session_dir / f"{doc['id']}.md"
                doc_file.write_text(doc.get('content', ''))

        # Create index
        index = self._create_markdown_index(sessions, output_dir)
        (output_dir / "INDEX.md").write_text(index)

        return ExportMetadata(
            export_id=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            scope=scope,
            format=ExportFormat.MARKDOWN,
            document_count=sum(len(await self._fetch_session_documents(s["id"])) for s in sessions),
            insight_count=0,  # Not included in markdown export
            entity_count=0,
            session_count=len(sessions),
            rkg_version="1.0.0",
            checksum=""
        )

    async def export_to_cypher(
        self,
        scope: ExportScope,
        output_path: Path
    ) -> ExportMetadata:
        """Export as Cypher statements for Neo4j import."""
        statements = []

        # Export nodes
        sessions = await self._fetch_sessions(scope)
        for session in sessions:
            statements.append(self._session_to_cypher(session))

        documents = await self._fetch_documents(scope)
        for doc in documents:
            statements.append(self._document_to_cypher(doc))

        entities = await self._fetch_entities(scope)
        for entity in entities:
            statements.append(self._entity_to_cypher(entity))

        # Export relationships
        relationships = await self._fetch_relationships(scope)
        for rel in relationships:
            statements.append(self._relationship_to_cypher(rel))

        output_path.write_text("\n".join(statements))

        return ExportMetadata(
            export_id=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            scope=scope,
            format=ExportFormat.CYPHER,
            document_count=len(documents),
            insight_count=0,
            entity_count=len(entities),
            session_count=len(sessions),
            rkg_version="1.0.0",
            checksum=hashlib.sha256(output_path.read_bytes()).hexdigest()
        )

    def _session_to_cypher(self, session: Dict) -> str:
        """Convert session to MERGE statement."""
        props = json.dumps(session)
        return f"MERGE (s:Session {{id: '{session['id']}'}}) SET s += {props};"

    def _document_to_cypher(self, doc: Dict) -> str:
        """Convert document to MERGE statement."""
        props = {k: v for k, v in doc.items() if k not in ['content']}  # Exclude large content
        return f"MERGE (d:Document {{id: '{doc['id']}'}}) SET d += {json.dumps(props)};"

    def _entity_to_cypher(self, entity: Dict) -> str:
        """Convert entity to MERGE statement."""
        return f"MERGE (e:Entity {{normalized_name: '{entity['normalized_name']}', entity_type: '{entity['entity_type']}'}}) SET e += {json.dumps(entity)};"

    def _relationship_to_cypher(self, rel: Dict) -> str:
        """Convert relationship to MERGE statement."""
        return f"MATCH (a {{id: '{rel['from_id']}'}}), (b {{id: '{rel['to_id']}'}}) MERGE (a)-[:{rel['type']}]->(b);"
```

### Import Implementation

```python
# src/rkg_mcp/export/importer.py

class KnowledgeGraphImporter:
    """Import knowledge graph data from various formats."""

    def __init__(self, neo4j: Neo4jStorage, qdrant: QdrantStorage, embedder: EmbeddingProvider):
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.embedder = embedder

    async def import_from_json(
        self,
        input_path: Path,
        merge_strategy: str = "skip_existing"  # "skip_existing" | "overwrite" | "merge"
    ) -> Dict[str, int]:
        """Import from JSON export."""
        data = json.loads(input_path.read_text())

        stats = {
            "sessions_imported": 0,
            "documents_imported": 0,
            "entities_imported": 0,
            "insights_imported": 0,
            "skipped": 0,
            "errors": 0
        }

        # Import sessions
        for session in data.get("sessions", []):
            try:
                if merge_strategy == "skip_existing":
                    exists = await self.neo4j.session_exists(session["id"])
                    if exists:
                        stats["skipped"] += 1
                        continue

                await self.neo4j.create_session(session)
                stats["sessions_imported"] += 1
            except Exception as e:
                logger.error(f"Error importing session {session['id']}: {e}")
                stats["errors"] += 1

        # Import documents (with re-embedding)
        for doc in data.get("documents", []):
            try:
                if merge_strategy == "skip_existing":
                    exists = await self.neo4j.document_exists(doc["id"])
                    if exists:
                        stats["skipped"] += 1
                        continue

                # Re-generate embeddings
                if doc.get("content"):
                    embedding = await self.embedder.embed_text(doc["content"], "document")
                    await self.qdrant.upsert_document(doc, embedding)

                await self.neo4j.create_document(doc)
                stats["documents_imported"] += 1
            except Exception as e:
                logger.error(f"Error importing document {doc['id']}: {e}")
                stats["errors"] += 1

        # Import entities
        for entity in data.get("entities", []):
            try:
                await self.neo4j.create_entity_node(entity)
                stats["entities_imported"] += 1
            except Exception as e:
                stats["errors"] += 1

        # Import relationships
        for rel in data.get("relationships", []):
            try:
                await self.neo4j.create_relationship(rel)
            except Exception as e:
                stats["errors"] += 1

        return stats
```

---

## 1.4 Analytics & Insights Dashboard

### Overview
Metrics and visualizations for research activity, coverage gaps, and knowledge trends.

### Analytics Data Models

```python
# src/rkg_mcp/analytics/models.py

from pydantic import BaseModel
from datetime import datetime, date
from typing import List, Dict, Optional

class ResearchActivityMetrics(BaseModel):
    """Daily/weekly research activity metrics."""
    period: str  # "day" | "week" | "month"
    start_date: date
    end_date: date
    total_sessions: int
    total_documents: int
    total_insights: int
    documents_by_source: Dict[str, int]  # source_type -> count
    documents_by_interface: Dict[str, int]  # agentic_interface -> count
    top_projects: List[Dict[str, Any]]  # [{name, document_count, insight_count}]

class EntityCoverage(BaseModel):
    """Entity coverage analysis."""
    entity_type: str
    total_count: int
    documents_with_entity: int
    avg_mentions_per_doc: float
    top_entities: List[Dict[str, Any]]  # [{name, mention_count, document_count}]
    trending_entities: List[Dict[str, Any]]  # Entities with recent growth

class KnowledgeGaps(BaseModel):
    """Identify gaps in research coverage."""
    orphan_documents: int  # Documents not linked to entities
    unconfirmed_insights: int  # Insights without supporting sources
    stale_topics: List[Dict[str, Any]]  # Topics not researched recently
    missing_relationships: List[Dict[str, Any]]  # Potential links not established

class InsightTrends(BaseModel):
    """Trends in insights and conclusions."""
    period: str
    total_insights: int
    by_type: Dict[str, int]  # insight_type -> count
    confidence_distribution: Dict[str, int]  # confidence_bucket -> count
    contradictions_detected: int
    supported_insights: int  # Insights with multiple sources
```

### Analytics Engine

```python
# src/rkg_mcp/analytics/engine.py

from datetime import datetime, timedelta
from typing import Dict, List, Any

class AnalyticsEngine:
    """Compute analytics and metrics from knowledge graph."""

    def __init__(self, neo4j: Neo4jStorage):
        self.neo4j = neo4j

    async def get_activity_metrics(
        self,
        period: str = "week",
        project_filter: str | None = None
    ) -> ResearchActivityMetrics:
        """Calculate research activity metrics."""

        # Determine date range
        now = datetime.now()
        if period == "day":
            start = now - timedelta(days=1)
        elif period == "week":
            start = now - timedelta(weeks=1)
        else:  # month
            start = now - timedelta(days=30)

        # Query Neo4j for metrics
        query = """
        MATCH (s:Session)-[:CAPTURED]->(d:Document)
        WHERE s.started_at >= datetime($start_date)
        """ + (f" AND s.project = '{project_filter}'" if project_filter else "") + """
        WITH s, d
        OPTIONAL MATCH (s)-[:PRODUCED]->(i:Insight)
        RETURN
            count(DISTINCT s) as total_sessions,
            count(DISTINCT d) as total_documents,
            count(DISTINCT i) as total_insights,
            collect(DISTINCT d.source_type) as source_types,
            collect(DISTINCT s.agentic_interface) as interfaces
        """

        result = await self.neo4j.execute_query(query, {"start_date": start.isoformat()})
        row = result[0]

        # Get documents by source type
        docs_by_source = await self._count_by_field("Document", "source_type", start)
        docs_by_interface = await self._count_by_field("Session", "agentic_interface", start)

        # Get top projects
        top_projects = await self._get_top_projects(start, limit=10)

        return ResearchActivityMetrics(
            period=period,
            start_date=start.date(),
            end_date=now.date(),
            total_sessions=row["total_sessions"],
            total_documents=row["total_documents"],
            total_insights=row["total_insights"],
            documents_by_source=docs_by_source,
            documents_by_interface=docs_by_interface,
            top_projects=top_projects
        )

    async def get_entity_coverage(self) -> List[EntityCoverage]:
        """Analyze entity coverage across documents."""
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
        WITH e.entity_type as entity_type,
             count(DISTINCT e) as total_entities,
             count(DISTINCT d) as docs_with_entity,
             sum(CASE WHEN d IS NOT NULL THEN 1 ELSE 0 END) as total_mentions
        RETURN entity_type, total_entities, docs_with_entity, total_mentions
        ORDER BY total_entities DESC
        """

        results = await self.neo4j.execute_query(query)

        coverage = []
        for row in results:
            # Get top entities for this type
            top_entities = await self._get_top_entities_by_type(row["entity_type"])
            trending = await self._get_trending_entities(row["entity_type"])

            coverage.append(EntityCoverage(
                entity_type=row["entity_type"],
                total_count=row["total_entities"],
                documents_with_entity=row["docs_with_entity"],
                avg_mentions_per_doc=row["total_mentions"] / max(row["docs_with_entity"], 1),
                top_entities=top_entities,
                trending_entities=trending
            ))

        return coverage

    async def identify_knowledge_gaps(self) -> KnowledgeGaps:
        """Identify gaps and areas needing attention."""

        # Orphan documents (no entity links)
        orphan_query = """
        MATCH (d:Document)
        WHERE NOT (d)-[:MENTIONS]->(:Entity)
        RETURN count(d) as orphan_count
        """
        orphan_result = await self.neo4j.execute_query(orphan_query)

        # Unconfirmed insights (low source count)
        unconfirmed_query = """
        MATCH (i:Insight)
        OPTIONAL MATCH (i)-[:DERIVED_FROM]->(d:Document)
        WITH i, count(d) as source_count
        WHERE source_count < 2
        RETURN count(i) as unconfirmed_count
        """
        unconfirmed_result = await self.neo4j.execute_query(unconfirmed_query)

        # Stale topics (entities not mentioned recently)
        stale_query = """
        MATCH (e:Entity)<-[:MENTIONS]-(d:Document)<-[:CAPTURED]-(s:Session)
        WITH e, max(s.started_at) as last_mentioned
        WHERE last_mentioned < datetime() - duration('P30D')
        RETURN e.normalized_name as name, e.entity_type as type, last_mentioned
        ORDER BY last_mentioned ASC
        LIMIT 20
        """
        stale_results = await self.neo4j.execute_query(stale_query)

        return KnowledgeGaps(
            orphan_documents=orphan_result[0]["orphan_count"],
            unconfirmed_insights=unconfirmed_result[0]["unconfirmed_count"],
            stale_topics=[dict(r) for r in stale_results],
            missing_relationships=[]  # Implement relationship prediction
        )

    async def _get_top_projects(self, since: datetime, limit: int = 10) -> List[Dict]:
        """Get top projects by activity."""
        query = """
        MATCH (p:Project)<-[:BELONGS_TO]-(s:Session)
        WHERE s.started_at >= datetime($since)
        OPTIONAL MATCH (s)-[:CAPTURED]->(d:Document)
        OPTIONAL MATCH (s)-[:PRODUCED]->(i:Insight)
        RETURN p.name as name,
               count(DISTINCT d) as document_count,
               count(DISTINCT i) as insight_count,
               count(DISTINCT s) as session_count
        ORDER BY document_count DESC
        LIMIT $limit
        """
        results = await self.neo4j.execute_query(query, {
            "since": since.isoformat(),
            "limit": limit
        })
        return [dict(r) for r in results]
```

### Analytics MCP Tools

```python
# src/rkg_mcp/tools/analytics.py

@mcp.tool()
async def rkg_get_activity_summary(
    period: str = "week",
    project: str | None = None
) -> Dict[str, Any]:
    """
    Get research activity summary for a time period.

    Args:
        period: Time period - "day", "week", or "month"
        project: Optional project name to filter by

    Returns:
        Activity metrics including session counts, document counts,
        and breakdowns by source type and interface.
    """
    metrics = await analytics_engine.get_activity_metrics(period, project)
    return metrics.model_dump()

@mcp.tool()
async def rkg_identify_gaps() -> Dict[str, Any]:
    """
    Identify knowledge gaps and areas needing attention.

    Returns:
        Analysis of orphan documents, unconfirmed insights,
        stale topics, and potential missing relationships.
    """
    gaps = await analytics_engine.identify_knowledge_gaps()
    return gaps.model_dump()

@mcp.tool()
async def rkg_get_entity_stats(entity_type: str | None = None) -> List[Dict[str, Any]]:
    """
    Get entity coverage statistics.

    Args:
        entity_type: Optional filter by entity type

    Returns:
        Coverage metrics for each entity type including
        top entities and trending entities.
    """
    coverage = await analytics_engine.get_entity_coverage()
    if entity_type:
        coverage = [c for c in coverage if c.entity_type == entity_type]
    return [c.model_dump() for c in coverage]
```

---

## 1.5 Contradiction Detection

### Overview
Automatically detect and flag insights that conflict with each other across sessions.

### Contradiction Detection Engine

```python
# src/rkg_mcp/analysis/contradiction.py

from typing import List, Tuple, Dict, Any
from pydantic import BaseModel
from enum import Enum

class ContradictionType(str, Enum):
    DIRECT = "direct"           # Explicit opposite claims
    STATISTICAL = "statistical"  # Conflicting numbers/metrics
    TEMPORAL = "temporal"        # Claims invalidated by time
    SCOPE = "scope"              # Different scope leading to apparent conflict
    UNCERTAIN = "uncertain"      # Potential contradiction, needs review

class ContradictionPair(BaseModel):
    """A pair of potentially contradicting insights."""
    insight_a_id: str
    insight_b_id: str
    insight_a_content: str
    insight_b_content: str
    contradiction_type: ContradictionType
    confidence: float  # 0-1 confidence in contradiction
    explanation: str
    detected_at: datetime
    resolved: bool = False
    resolution: str | None = None

class ContradictionDetector:
    """
    Detect contradictions between insights using
    semantic similarity and LLM verification.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        qdrant: QdrantStorage,
        neo4j: Neo4jStorage,
        similarity_threshold: float = 0.8,
        llm_verify: bool = True
    ):
        self.embedder = embedder
        self.qdrant = qdrant
        self.neo4j = neo4j
        self.similarity_threshold = similarity_threshold
        self.llm_verify = llm_verify

    async def detect_contradictions(
        self,
        insight_id: str | None = None
    ) -> List[ContradictionPair]:
        """
        Detect contradictions for a specific insight or across all insights.

        Strategy:
        1. Find semantically similar insights (high cosine similarity)
        2. Filter to insights about the same entities/topics
        3. Use LLM to verify if they actually contradict
        """

        if insight_id:
            insights = [await self._get_insight(insight_id)]
        else:
            insights = await self._get_all_insights()

        contradictions = []

        for insight in insights:
            # Find similar insights
            similar = await self._find_similar_insights(
                insight["content"],
                insight["id"],
                threshold=self.similarity_threshold
            )

            for candidate in similar:
                # Check if they're about the same topic
                shared_entities = await self._get_shared_entities(
                    insight["id"],
                    candidate["id"]
                )

                if not shared_entities:
                    continue  # Different topics, not a contradiction

                # Verify contradiction with LLM
                if self.llm_verify:
                    is_contradiction, explanation, conf = await self._verify_contradiction(
                        insight["content"],
                        candidate["content"],
                        shared_entities
                    )
                else:
                    is_contradiction = True
                    explanation = "High semantic similarity on same topic"
                    conf = similar["score"]

                if is_contradiction:
                    contradiction_type = self._classify_contradiction(
                        insight["content"],
                        candidate["content"]
                    )

                    contradictions.append(ContradictionPair(
                        insight_a_id=insight["id"],
                        insight_b_id=candidate["id"],
                        insight_a_content=insight["content"],
                        insight_b_content=candidate["content"],
                        contradiction_type=contradiction_type,
                        confidence=conf,
                        explanation=explanation,
                        detected_at=datetime.now()
                    ))

        return self._deduplicate_contradictions(contradictions)

    async def _find_similar_insights(
        self,
        content: str,
        exclude_id: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find semantically similar insights."""
        embedding = await self.embedder.embed_text(content, "query")

        results = await self.qdrant.search(
            collection="session_insights",
            query_vector=embedding.vector,
            limit=50,
            score_threshold=threshold,
            filter={"insight_id": {"$ne": exclude_id}}
        )

        return [
            {"id": r.payload["insight_id"], "content": r.payload["content"], "score": r.score}
            for r in results
        ]

    async def _get_shared_entities(
        self,
        insight_a_id: str,
        insight_b_id: str
    ) -> List[str]:
        """Find entities mentioned in both insights' source documents."""
        query = """
        MATCH (i1:Insight {id: $insight_a})-[:DERIVED_FROM]->(d1:Document)-[:MENTIONS]->(e:Entity)
        MATCH (i2:Insight {id: $insight_b})-[:DERIVED_FROM]->(d2:Document)-[:MENTIONS]->(e)
        RETURN DISTINCT e.normalized_name as entity
        """
        results = await self.neo4j.execute_query(query, {
            "insight_a": insight_a_id,
            "insight_b": insight_b_id
        })
        return [r["entity"] for r in results]

    async def _verify_contradiction(
        self,
        content_a: str,
        content_b: str,
        shared_entities: List[str]
    ) -> Tuple[bool, str, float]:
        """Use LLM to verify if two statements contradict."""
        prompt = f"""Analyze if these two statements contradict each other.
They both relate to: {', '.join(shared_entities)}

Statement A: {content_a}

Statement B: {content_b}

Respond in JSON format:
{{
    "is_contradiction": true/false,
    "explanation": "brief explanation",
    "confidence": 0.0-1.0
}}

Consider:
- Different time frames may not be contradictions
- Different contexts/scopes may not be contradictions
- Complementary information is not a contradiction
"""
        # This would call the LLM API
        # For now, return placeholder
        return False, "", 0.0

    def _classify_contradiction(
        self,
        content_a: str,
        content_b: str
    ) -> ContradictionType:
        """Classify the type of contradiction."""
        # Simple heuristics - could be enhanced with LLM
        if any(word in content_a.lower() for word in ["not", "never", "won't", "isn't"]):
            if any(word in content_b.lower() for word in ["is", "will", "does", "always"]):
                return ContradictionType.DIRECT

        # Check for numerical contradictions
        import re
        nums_a = set(re.findall(r'\d+\.?\d*', content_a))
        nums_b = set(re.findall(r'\d+\.?\d*', content_b))
        if nums_a and nums_b and not nums_a.intersection(nums_b):
            return ContradictionType.STATISTICAL

        return ContradictionType.UNCERTAIN

    def _deduplicate_contradictions(
        self,
        contradictions: List[ContradictionPair]
    ) -> List[ContradictionPair]:
        """Remove duplicate contradiction pairs (A-B and B-A)."""
        seen = set()
        unique = []

        for c in contradictions:
            key = tuple(sorted([c.insight_a_id, c.insight_b_id]))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique
```

### Neo4j Contradiction Storage

```cypher
// Create contradiction relationship
MATCH (i1:Insight {id: $insight_a_id})
MATCH (i2:Insight {id: $insight_b_id})
MERGE (i1)-[r:CONTRADICTS]->(i2)
SET r.contradiction_type = $type,
    r.confidence = $confidence,
    r.explanation = $explanation,
    r.detected_at = datetime(),
    r.resolved = false

// Query for unresolved contradictions
MATCH (i1:Insight)-[r:CONTRADICTS]->(i2:Insight)
WHERE r.resolved = false
OPTIONAL MATCH (i1)-[:DERIVED_FROM]->(d1:Document)<-[:CAPTURED]-(s1:Session)
OPTIONAL MATCH (i2)-[:DERIVED_FROM]->(d2:Document)<-[:CAPTURED]-(s2:Session)
RETURN i1, i2, r,
       collect(DISTINCT s1.project) as projects_a,
       collect(DISTINCT s2.project) as projects_b
ORDER BY r.confidence DESC

// Resolve contradiction
MATCH (i1:Insight {id: $insight_a_id})-[r:CONTRADICTS]->(i2:Insight {id: $insight_b_id})
SET r.resolved = true,
    r.resolution = $resolution,
    r.resolved_at = datetime()
```

### Contradiction MCP Tools

```python
@mcp.tool()
async def rkg_detect_contradictions(
    insight_id: str | None = None,
    min_confidence: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Detect contradicting insights in the knowledge graph.

    Args:
        insight_id: Optional specific insight to check
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        List of contradiction pairs with explanations
    """
    contradictions = await contradiction_detector.detect_contradictions(insight_id)
    return [
        c.model_dump()
        for c in contradictions
        if c.confidence >= min_confidence
    ]

@mcp.tool()
async def rkg_resolve_contradiction(
    insight_a_id: str,
    insight_b_id: str,
    resolution: str
) -> Dict[str, str]:
    """
    Mark a contradiction as resolved with explanation.

    Args:
        insight_a_id: First insight ID
        insight_b_id: Second insight ID
        resolution: Explanation of how contradiction was resolved

    Returns:
        Confirmation of resolution
    """
    await neo4j.resolve_contradiction(insight_a_id, insight_b_id, resolution)
    return {
        "status": "resolved",
        "insight_a": insight_a_id,
        "insight_b": insight_b_id,
        "resolution": resolution
    }
```

---

# Part 2: Observability & Monitoring Integrations

## 2.1 LangFuse Integration

### Overview
LangFuse provides LLM-specific observability including prompt tracing, cost tracking, and evaluation. We integrate it for monitoring all LLM interactions including embedding generation, reranking, and any future LLM-based features.

### Installation & Configuration

```python
# requirements.txt additions
langfuse>=2.0.0

# .env configuration
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000  # Self-hosted
# LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Cloud (EU)
# LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # Cloud (US)
```

### LangFuse Client Singleton

```python
# src/rkg_mcp/observability/langfuse_client.py

from langfuse import get_client, Langfuse
from typing import Optional
from functools import lru_cache
import os

class LangFuseManager:
    """
    Singleton manager for LangFuse client with feature flag support.
    """

    _instance: Optional["LangFuseManager"] = None
    _client: Optional[Langfuse] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize LangFuse client based on feature flags."""
        from ..config import settings

        if not settings.langfuse_enabled:
            self._client = None
            return

        try:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url,
                enabled=True,
                debug=settings.debug
            )

            # Verify connection
            if self._client.auth_check():
                logger.info("LangFuse client initialized successfully")
            else:
                logger.warning("LangFuse auth check failed")
                self._client = None

        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}")
            self._client = None

    @property
    def client(self) -> Optional[Langfuse]:
        return self._client

    @property
    def enabled(self) -> bool:
        return self._client is not None

@lru_cache()
def get_langfuse() -> LangFuseManager:
    return LangFuseManager()
```

### LangFuse Tracing Decorators

```python
# src/rkg_mcp/observability/tracing.py

from functools import wraps
from typing import Callable, Any
import time
from contextlib import contextmanager

class LangFuseTracer:
    """Tracing utilities for LangFuse integration."""

    def __init__(self):
        self.langfuse = get_langfuse()

    @contextmanager
    def trace_span(
        self,
        name: str,
        metadata: dict | None = None,
        input_data: Any = None
    ):
        """Context manager for tracing a span."""
        if not self.langfuse.enabled:
            yield None
            return

        with self.langfuse.client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata,
            input=input_data
        ) as span:
            try:
                yield span
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise

    @contextmanager
    def trace_generation(
        self,
        name: str,
        model: str,
        input_data: Any = None,
        model_parameters: dict | None = None
    ):
        """Context manager for tracing an LLM generation."""
        if not self.langfuse.enabled:
            yield None
            return

        with self.langfuse.client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_data,
            model_parameters=model_parameters
        ) as generation:
            start_time = time.time()
            try:
                yield generation
            finally:
                generation.update(
                    completion_start_time=start_time
                )

    def trace_function(
        self,
        name: str | None = None,
        capture_input: bool = True,
        capture_output: bool = True
    ) -> Callable:
        """Decorator for tracing function execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = name or func.__name__
                input_data = {"args": args, "kwargs": kwargs} if capture_input else None

                with self.trace_span(span_name, input_data=input_data) as span:
                    result = await func(*args, **kwargs)
                    if span and capture_output:
                        span.update(output=result)
                    return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or func.__name__
                input_data = {"args": args, "kwargs": kwargs} if capture_input else None

                with self.trace_span(span_name, input_data=input_data) as span:
                    result = func(*args, **kwargs)
                    if span and capture_output:
                        span.update(output=result)
                    return result

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

tracer = LangFuseTracer()
```

### Instrumented Embedding Provider

```python
# src/rkg_mcp/embeddings/voyage_instrumented.py

from .voyage import VoyageEmbeddingProvider
from ..observability.tracing import tracer, get_langfuse

class InstrumentedVoyageEmbeddingProvider(VoyageEmbeddingProvider):
    """Voyage embedding provider with LangFuse tracing."""

    async def embed_text(
        self,
        text: str,
        input_type: str = "document"
    ) -> EmbeddingResult:
        """Embed text with LangFuse tracing."""
        langfuse = get_langfuse()

        with tracer.trace_generation(
            name="voyage_embed_text",
            model=self.model_name,
            input_data={"text_length": len(text), "input_type": input_type},
            model_parameters={"model": self.model_name}
        ) as generation:
            result = await super().embed_text(text, input_type)

            if generation:
                generation.update(
                    output={"vector_dim": len(result.vector)},
                    usage={
                        "input": len(text.split()),  # Approximate token count
                        "total": len(text.split())
                    }
                )

            return result

    async def embed_batch(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[EmbeddingResult]:
        """Batch embed with LangFuse tracing."""
        with tracer.trace_generation(
            name="voyage_embed_batch",
            model=self.model_name,
            input_data={"batch_size": len(texts), "input_type": input_type},
            model_parameters={"model": self.model_name}
        ) as generation:
            results = await super().embed_batch(texts, input_type)

            if generation:
                total_tokens = sum(len(t.split()) for t in texts)
                generation.update(
                    output={"vectors_generated": len(results)},
                    usage={"input": total_tokens, "total": total_tokens}
                )

            return results
```

### LangFuse Docker Compose Service

```yaml
# docker-compose.langfuse.yml
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@langfuse-db:5432/langfuse  # pragma: allowlist secret
      - NEXTAUTH_SECRET=${LANGFUSE_NEXTAUTH_SECRET}
      - SALT=${LANGFUSE_SALT}
      - NEXTAUTH_URL=http://localhost:3000
      - TELEMETRY_ENABLED=${LANGFUSE_TELEMETRY_ENABLED:-true}
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
    depends_on:
      langfuse-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - rkg-network

  langfuse-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rkg-network

volumes:
  langfuse_postgres_data:

networks:
  rkg-network:
    external: true
```

### LangFuse Prompt Management

```python
# src/rkg_mcp/prompts/managed.py

from langfuse import Langfuse

class PromptManager:
    """Manage prompts via LangFuse for versioning and A/B testing."""

    def __init__(self):
        self.langfuse = get_langfuse()
        self._cache: dict[str, str] = {}

    async def get_prompt(
        self,
        name: str,
        version: int | None = None,
        variables: dict | None = None
    ) -> str:
        """
        Fetch prompt from LangFuse with optional variable substitution.
        Falls back to local prompts if LangFuse is disabled.
        """
        if not self.langfuse.enabled:
            return self._get_local_prompt(name, variables)

        try:
            prompt = self.langfuse.client.get_prompt(
                name=name,
                version=version,
                type="text"
            )

            template = prompt.compile(**variables) if variables else prompt.prompt
            return template

        except Exception as e:
            logger.warning(f"Failed to fetch prompt '{name}' from LangFuse: {e}")
            return self._get_local_prompt(name, variables)

    def _get_local_prompt(self, name: str, variables: dict | None = None) -> str:
        """Fallback to local prompt templates."""
        local_prompts = {
            "contradiction_detection": """
Analyze if these two statements contradict each other.
They both relate to: {entities}

Statement A: {statement_a}
Statement B: {statement_b}

Respond in JSON format with is_contradiction, explanation, and confidence.
""",
            "entity_extraction": """
Extract named entities from the following text.
Focus on: people, organizations, technologies, concepts.

Text: {text}

Return entities as JSON array with text, type, and confidence.
""",
        }

        template = local_prompts.get(name, "")
        if variables:
            template = template.format(**variables)
        return template
```

---

## 2.2 NewRelic Integration

### Overview
NewRelic provides APM, distributed tracing, and log management for the entire infrastructure stack including the MCP server, storage backends, and macOS app backend.

### Installation & Configuration

```python
# requirements.txt additions
newrelic>=9.0.0

# newrelic.ini configuration file
[newrelic]
license_key = ${NEW_RELIC_LICENSE_KEY}
app_name = RKG-MCP-Server
monitor_mode = true
log_level = info
audit_log_file = /var/log/newrelic/newrelic-python-agent-audit.log
proxy_host =
proxy_port =
high_security = false
transaction_tracer.enabled = true
transaction_tracer.transaction_threshold = apdex_f
transaction_tracer.record_sql = obfuscated
transaction_tracer.stack_trace_threshold = 0.5
transaction_tracer.explain_enabled = true
transaction_tracer.explain_threshold = 0.5
error_collector.enabled = true
error_collector.capture_events = true
error_collector.ignore_errors =
browser_monitoring.auto_instrument = false
thread_profiler.enabled = true
distributed_tracing.enabled = true
span_events.enabled = true
application_logging.enabled = true
application_logging.forwarding.enabled = true
application_logging.forwarding.max_samples_stored = 10000
application_logging.metrics.enabled = true
application_logging.local_decorating.enabled = true
```

### NewRelic Agent Wrapper

```python
# src/rkg_mcp/observability/newrelic_agent.py

import newrelic.agent
from typing import Callable, Any, Optional
from functools import wraps
import os

class NewRelicAgent:
    """
    NewRelic APM integration with feature flag support.
    """

    _initialized: bool = False

    @classmethod
    def initialize(cls, config_file: str | None = None):
        """Initialize NewRelic agent."""
        from ..config import settings

        if not settings.newrelic_enabled:
            logger.info("NewRelic disabled via feature flag")
            return

        if cls._initialized:
            return

        config_file = config_file or settings.newrelic_config_file

        try:
            newrelic.agent.initialize(config_file)
            cls._initialized = True
            logger.info("NewRelic agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NewRelic: {e}")

    @classmethod
    @property
    def enabled(cls) -> bool:
        return cls._initialized

    @staticmethod
    def background_task(name: str | None = None, group: str = "Task"):
        """Decorator for background task tracing."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not NewRelicAgent.enabled:
                    return await func(*args, **kwargs)

                task_name = name or func.__name__
                with newrelic.agent.BackgroundTask(
                    newrelic.agent.application(),
                    name=task_name,
                    group=group
                ):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not NewRelicAgent.enabled:
                    return func(*args, **kwargs)

                task_name = name or func.__name__
                with newrelic.agent.BackgroundTask(
                    newrelic.agent.application(),
                    name=task_name,
                    group=group
                ):
                    return func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    @staticmethod
    def trace_function(name: str | None = None):
        """Decorator for function tracing."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            @newrelic.agent.function_trace(name=name or func.__name__)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            @wraps(func)
            @newrelic.agent.function_trace(name=name or func.__name__)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            if not NewRelicAgent.enabled:
                return func

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    @staticmethod
    def record_custom_event(event_type: str, params: dict):
        """Record a custom event to NewRelic."""
        if NewRelicAgent.enabled:
            newrelic.agent.record_custom_event(event_type, params)

    @staticmethod
    def record_custom_metric(name: str, value: float):
        """Record a custom metric to NewRelic."""
        if NewRelicAgent.enabled:
            newrelic.agent.record_custom_metric(name, value)

    @staticmethod
    def add_custom_attribute(key: str, value: Any):
        """Add custom attribute to current transaction."""
        if NewRelicAgent.enabled:
            newrelic.agent.add_custom_attribute(key, value)

    @staticmethod
    def notice_error(error: Exception | None = None):
        """Record an error to NewRelic."""
        if NewRelicAgent.enabled:
            newrelic.agent.notice_error(error)
```

### NewRelic Logging Integration

```python
# src/rkg_mcp/observability/logging.py

import logging
import sys
from typing import Optional
import newrelic.agent

class NewRelicLogFormatter(logging.Formatter):
    """
    Log formatter that adds NewRelic linking metadata.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Add NewRelic linking metadata
        if NewRelicAgent.enabled:
            linking_metadata = newrelic.agent.get_linking_metadata()
            for key, value in linking_metadata.items():
                setattr(record, key.replace(".", "_"), value)

        return super().format(record)

class NewRelicLogHandler(logging.Handler):
    """
    Custom handler that forwards logs to NewRelic.
    """

    def emit(self, record: logging.LogRecord):
        if not NewRelicAgent.enabled:
            return

        try:
            # NewRelic automatically captures logs with forwarding enabled
            # This handler adds custom attributes
            newrelic.agent.record_log_event(
                message=self.format(record),
                level=record.levelname,
                timestamp=record.created * 1000
            )
        except Exception:
            self.handleError(record)

def setup_logging(
    level: str = "INFO",
    enable_newrelic: bool = True,
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup logging with NewRelic integration.
    """
    logger = logging.getLogger("rkg_mcp")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(NewRelicLogFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)

    # NewRelic handler
    if enable_newrelic and NewRelicAgent.enabled:
        nr_handler = NewRelicLogHandler()
        nr_handler.setFormatter(NewRelicLogFormatter(
            "%(message)s"
        ))
        logger.addHandler(nr_handler)

    return logger
```

### Distributed Tracing Middleware

```python
# src/rkg_mcp/observability/middleware.py

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import newrelic.agent
import time

class NewRelicMiddleware(BaseHTTPMiddleware):
    """
    Middleware for NewRelic transaction tracing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not NewRelicAgent.enabled:
            return await call_next(request)

        # Start transaction
        transaction = newrelic.agent.current_transaction()

        # Add request attributes
        newrelic.agent.add_custom_attribute("http.method", request.method)
        newrelic.agent.add_custom_attribute("http.url", str(request.url))
        newrelic.agent.add_custom_attribute("http.client_ip", request.client.host if request.client else "unknown")

        # Extract distributed tracing headers
        for header in ["traceparent", "tracestate", "newrelic"]:
            if header in request.headers:
                newrelic.agent.add_custom_attribute(f"dt.{header}", request.headers[header])

        start_time = time.time()

        try:
            response = await call_next(request)

            # Record response attributes
            newrelic.agent.add_custom_attribute("http.status_code", response.status_code)

            return response

        except Exception as e:
            newrelic.agent.notice_error()
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            newrelic.agent.record_custom_metric("Custom/RequestDuration", duration_ms)
```

### Instrumented Storage Clients

```python
# src/rkg_mcp/storage/qdrant_instrumented.py

from .qdrant import QdrantStorage
from ..observability.newrelic_agent import NewRelicAgent
import newrelic.agent

class InstrumentedQdrantStorage(QdrantStorage):
    """Qdrant storage with NewRelic instrumentation."""

    @newrelic.agent.function_trace(name="qdrant_search")
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        **kwargs
    ):
        NewRelicAgent.add_custom_attribute("qdrant.collection", collection)
        NewRelicAgent.add_custom_attribute("qdrant.limit", limit)

        start_time = time.time()
        try:
            result = await super().search(collection, query_vector, limit, **kwargs)
            NewRelicAgent.record_custom_metric(
                f"Custom/Qdrant/SearchResults/{collection}",
                len(result)
            )
            return result
        finally:
            duration_ms = (time.time() - start_time) * 1000
            NewRelicAgent.record_custom_metric(
                f"Custom/Qdrant/SearchDuration/{collection}",
                duration_ms
            )

    @newrelic.agent.function_trace(name="qdrant_upsert")
    async def upsert(self, collection: str, points: list, **kwargs):
        NewRelicAgent.add_custom_attribute("qdrant.collection", collection)
        NewRelicAgent.add_custom_attribute("qdrant.points_count", len(points))

        return await super().upsert(collection, points, **kwargs)

# Similar instrumentation for Neo4j
class InstrumentedNeo4jStorage(Neo4jStorage):
    """Neo4j storage with NewRelic instrumentation."""

    @newrelic.agent.database_trace(
        "neo4j",
        None,
        "execute_query",
        newrelic.agent.DatastoreTrace
    )
    async def execute_query(self, query: str, params: dict | None = None):
        NewRelicAgent.add_custom_attribute("neo4j.query_length", len(query))
        return await super().execute_query(query, params)
```

### NewRelic Infrastructure Agent for Docker

```yaml
# docker-compose.newrelic.yml
services:
  newrelic-infra:
    image: newrelic/infrastructure:latest
    cap_add:
      - SYS_PTRACE
    network_mode: host
    pid: host
    privileged: true
    volumes:
      - /:/host:ro
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - NRIA_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
      - NRIA_DISPLAY_NAME=rkg-infrastructure
      - NRIA_LOG_LEVEL=info
    restart: unless-stopped

  # NewRelic log forwarder
  newrelic-logs:
    image: newrelic/newrelic-fluentbit-output:latest
    environment:
      - LOG_LEVEL=info
      - NR_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf:ro
    restart: unless-stopped
```

---

## 2.3 Infisical Integration

### Overview
Infisical provides centralized secrets management with runtime fetching, audit logging, and access controls. We support both Pro tier (with RBAC, versioning, rotation) and Free tier (basic secrets) via feature flags.

### Infisical Client with Tier Support

```python
# src/rkg_mcp/secrets/infisical_client.py

from infisical_sdk import InfisicalSDKClient
from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum
import os

class InfisicalTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class SecretValue(BaseModel):
    """Wrapper for secret values with metadata."""
    key: str
    value: str
    version: int | None = None
    environment: str
    path: str
    last_fetched: datetime

class InfisicalManager:
    """
    Infisical secrets manager with Pro/Free tier feature flags.

    Pro tier features:
    - Secret versioning
    - Secret rotation
    - RBAC
    - Audit log access
    - Point-in-time recovery

    Free tier features:
    - Basic secret storage
    - Environment-based access
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        host: str = "https://app.infisical.com",
        project_id: str | None = None,
        environment: str = "dev",
        tier: InfisicalTier = InfisicalTier.FREE
    ):
        self.client_id = client_id or os.getenv("INFISICAL_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("INFISICAL_CLIENT_SECRET")
        self.host = host
        self.project_id = project_id or os.getenv("INFISICAL_PROJECT_ID")
        self.environment = environment
        self.tier = tier
        self._client: InfisicalSDKClient | None = None
        self._secret_cache: Dict[str, SecretValue] = {}
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize Infisical client."""
        if not all([self.client_id, self.client_secret, self.project_id]):
            logger.warning("Infisical credentials not configured, using environment variables")
            return

        try:
            self._client = InfisicalSDKClient(host=self.host)
            self._client.auth.universal_auth.login(
                self.client_id,
                self.client_secret
            )
            logger.info(f"Infisical client initialized (tier: {self.tier})")
        except Exception as e:
            logger.error(f"Failed to initialize Infisical: {e}")
            self._client = None

    async def get_secret(
        self,
        key: str,
        path: str = "/",
        version: int | None = None,
        use_cache: bool = True
    ) -> str | None:
        """
        Fetch a secret value.

        Args:
            key: Secret key name
            path: Secret path (default: root)
            version: Specific version (Pro tier only)
            use_cache: Whether to use cached value
        """
        cache_key = f"{self.environment}:{path}:{key}"

        # Check cache
        if use_cache and cache_key in self._secret_cache:
            cached = self._secret_cache[cache_key]
            if (datetime.now() - cached.last_fetched).seconds < self._cache_ttl:
                return cached.value

        # Fallback to environment variables if client not initialized
        if not self._client:
            return os.getenv(key)

        try:
            # Pro tier: support versioning
            if self.tier == InfisicalTier.PRO and version is not None:
                secret = self._client.secrets.get_secret_by_name(
                    secret_name=key,
                    project_id=self.project_id,
                    environment_slug=self.environment,
                    secret_path=path,
                    version=version
                )
            else:
                secret = self._client.secrets.get_secret_by_name(
                    secret_name=key,
                    project_id=self.project_id,
                    environment_slug=self.environment,
                    secret_path=path
                )

            # Cache the value
            self._secret_cache[cache_key] = SecretValue(
                key=key,
                value=secret.secretValue,
                version=getattr(secret, 'version', None),
                environment=self.environment,
                path=path,
                last_fetched=datetime.now()
            )

            return secret.secretValue

        except Exception as e:
            logger.error(f"Failed to fetch secret '{key}': {e}")
            # Fallback to environment variable
            return os.getenv(key)

    async def get_secrets_by_path(
        self,
        path: str = "/"
    ) -> Dict[str, str]:
        """Fetch all secrets at a path."""
        if not self._client:
            return {}

        try:
            secrets = self._client.secrets.list_secrets(
                project_id=self.project_id,
                environment_slug=self.environment,
                secret_path=path
            )
            return {s.secretKey: s.secretValue for s in secrets}
        except Exception as e:
            logger.error(f"Failed to list secrets at '{path}': {e}")
            return {}

    # Pro tier features
    async def get_secret_versions(
        self,
        key: str,
        path: str = "/",
        limit: int = 10
    ) -> list[Dict[str, Any]]:
        """
        Get version history for a secret.
        Pro tier only.
        """
        if self.tier != InfisicalTier.PRO:
            raise FeatureNotAvailableError("Secret versioning requires Pro tier")

        if not self._client:
            return []

        try:
            versions = self._client.secrets.get_secret_versions(
                secret_name=key,
                project_id=self.project_id,
                environment_slug=self.environment,
                secret_path=path,
                limit=limit
            )
            return [
                {
                    "version": v.version,
                    "value": v.secretValue,
                    "created_at": v.createdAt,
                    "created_by": v.createdBy
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get versions for '{key}': {e}")
            return []

    async def rotate_secret(
        self,
        key: str,
        new_value: str,
        path: str = "/"
    ) -> bool:
        """
        Rotate a secret to a new value.
        Pro tier only.
        """
        if self.tier != InfisicalTier.PRO:
            raise FeatureNotAvailableError("Secret rotation requires Pro tier")

        if not self._client:
            return False

        try:
            self._client.secrets.update_secret_by_name(
                secret_name=key,
                secret_value=new_value,
                project_id=self.project_id,
                environment_slug=self.environment,
                secret_path=path
            )

            # Invalidate cache
            cache_key = f"{self.environment}:{path}:{key}"
            self._secret_cache.pop(cache_key, None)

            logger.info(f"Secret '{key}' rotated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secret '{key}': {e}")
            return False

    def clear_cache(self):
        """Clear the secret cache."""
        self._secret_cache.clear()

class FeatureNotAvailableError(Exception):
    """Raised when a Pro tier feature is accessed on Free tier."""
    pass
```

### Settings Integration

```python
# src/rkg_mcp/config/settings.py

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with Infisical integration."""

    # Infisical configuration
    infisical_enabled: bool = True
    infisical_tier: str = "free"  # "free" | "pro"
    infisical_client_id: Optional[str] = None
    infisical_client_secret: Optional[str] = None
    infisical_project_id: Optional[str] = None
    infisical_host: str = "https://app.infisical.com"
    infisical_environment: str = "dev"

    # These will be fetched from Infisical at runtime
    # Defined here as fallbacks
    voyage_api_key: Optional[str] = None
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None

    # Observability
    langfuse_enabled: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: str = "http://localhost:3000"

    newrelic_enabled: bool = True
    newrelic_license_key: Optional[str] = None
    newrelic_config_file: str = "newrelic.ini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

async def load_secrets_from_infisical(settings: Settings) -> Settings:
    """Load secrets from Infisical, overriding environment variables."""
    if not settings.infisical_enabled:
        return settings

    infisical = InfisicalManager(
        client_id=settings.infisical_client_id,
        client_secret=settings.infisical_client_secret,
        host=settings.infisical_host,
        project_id=settings.infisical_project_id,
        environment=settings.infisical_environment,
        tier=InfisicalTier(settings.infisical_tier)
    )

    await infisical.initialize()

    # Fetch secrets and override settings
    secret_mappings = {
        "VOYAGE_API_KEY": "voyage_api_key",  # pragma: allowlist secret
        "NEO4J_PASSWORD": "neo4j_password",  # pragma: allowlist secret
        "LANGFUSE_PUBLIC_KEY": "langfuse_public_key",
        "LANGFUSE_SECRET_KEY": "langfuse_secret_key",  # pragma: allowlist secret
        "NEW_RELIC_LICENSE_KEY": "newrelic_license_key",
    }

    for secret_key, setting_attr in secret_mappings.items():
        value = await infisical.get_secret(secret_key)
        if value:
            setattr(settings, setting_attr, value)

    return settings
```

### Infisical Docker Compose Service

```yaml
# docker-compose.infisical.yml
services:
  infisical:
    image: infisical/infisical:latest
    ports:
      - "8080:8080"
    environment:
      - ENCRYPTION_KEY=${INFISICAL_ENCRYPTION_KEY}
      - JWT_SIGNUP_SECRET=${INFISICAL_JWT_SIGNUP_SECRET}
      - JWT_REFRESH_SECRET=${INFISICAL_JWT_REFRESH_SECRET}
      - JWT_AUTH_SECRET=${INFISICAL_JWT_AUTH_SECRET}
      - JWT_SERVICE_SECRET=${INFISICAL_JWT_SERVICE_SECRET}
      - JWT_MFA_SECRET=${INFISICAL_JWT_MFA_SECRET}
      - MONGO_URL=mongodb://infisical-mongo:27017/infisical
      - REDIS_URL=redis://infisical-redis:6379
      - SITE_URL=http://localhost:8080
      - SMTP_HOST=${INFISICAL_SMTP_HOST:-}
      - SMTP_PORT=${INFISICAL_SMTP_PORT:-587}
      - SMTP_USERNAME=${INFISICAL_SMTP_USERNAME:-}
      - SMTP_PASSWORD=${INFISICAL_SMTP_PASSWORD:-}
      - SMTP_FROM_ADDRESS=${INFISICAL_SMTP_FROM:-noreply@example.com}
    depends_on:
      - infisical-mongo
      - infisical-redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rkg-network

  infisical-mongo:
    image: mongo:6.0
    volumes:
      - infisical_mongo_data:/data/db
    networks:
      - rkg-network

  infisical-redis:
    image: redis:7-alpine
    volumes:
      - infisical_redis_data:/data
    networks:
      - rkg-network

volumes:
  infisical_mongo_data:
  infisical_redis_data:
```

---

# Part 3: Complete Docker Infrastructure

## 3.1 Combined Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Core Storage
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rkg-network

  neo4j:
    image: neo4j:5.25-community
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-password}
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - rkg-network

  # MCP Server
  mcp-server:
    build:
      context: ./rkg-mcp-server
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      # Storage
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}

      # Infisical (will fetch other secrets)
      - INFISICAL_ENABLED=${INFISICAL_ENABLED:-true}
      - INFISICAL_TIER=${INFISICAL_TIER:-free}
      - INFISICAL_CLIENT_ID=${INFISICAL_CLIENT_ID}
      - INFISICAL_CLIENT_SECRET=${INFISICAL_CLIENT_SECRET}
      - INFISICAL_PROJECT_ID=${INFISICAL_PROJECT_ID}
      - INFISICAL_HOST=${INFISICAL_HOST:-http://infisical:8080}
      - INFISICAL_ENVIRONMENT=${INFISICAL_ENVIRONMENT:-dev}

      # Observability
      - LANGFUSE_ENABLED=${LANGFUSE_ENABLED:-true}
      - LANGFUSE_BASE_URL=http://langfuse:3000
      - NEWRELIC_ENABLED=${NEWRELIC_ENABLED:-true}
      - NEW_RELIC_CONFIG_FILE=/app/newrelic.ini
      - NEW_RELIC_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
      - NEW_RELIC_APP_NAME=rkg-mcp-server

      # Feature flags
      - ENTITY_EXTRACTION_ENABLED=${ENTITY_EXTRACTION_ENABLED:-true}
      - INCREMENTAL_SYNC_ENABLED=${INCREMENTAL_SYNC_ENABLED:-true}
      - CONTRADICTION_DETECTION_ENABLED=${CONTRADICTION_DETECTION_ENABLED:-true}
    volumes:
      - ${HOME}/.claude:/app/claude_sessions:ro
      - ${HOME}/.codex:/app/codex_sessions:ro
      - ./newrelic.ini:/app/newrelic.ini:ro
    depends_on:
      qdrant:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rkg-network

  # Observability Stack
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@langfuse-db:5432/langfuse  # pragma: allowlist secret
      - NEXTAUTH_SECRET=${LANGFUSE_NEXTAUTH_SECRET}
      - SALT=${LANGFUSE_SALT}
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      langfuse-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rkg-network

  langfuse-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rkg-network

  # Secrets Management
  infisical:
    image: infisical/infisical:latest
    ports:
      - "8080:8080"
    environment:
      - ENCRYPTION_KEY=${INFISICAL_ENCRYPTION_KEY}
      - JWT_SIGNUP_SECRET=${INFISICAL_JWT_SIGNUP_SECRET}
      - JWT_REFRESH_SECRET=${INFISICAL_JWT_REFRESH_SECRET}
      - JWT_AUTH_SECRET=${INFISICAL_JWT_AUTH_SECRET}
      - JWT_SERVICE_SECRET=${INFISICAL_JWT_SERVICE_SECRET}
      - JWT_MFA_SECRET=${INFISICAL_JWT_MFA_SECRET}
      - MONGO_URL=mongodb://infisical-mongo:27017/infisical
      - REDIS_URL=redis://infisical-redis:6379
      - SITE_URL=http://localhost:8080
    depends_on:
      - infisical-mongo
      - infisical-redis
    networks:
      - rkg-network

  infisical-mongo:
    image: mongo:6.0
    volumes:
      - infisical_mongo_data:/data/db
    networks:
      - rkg-network

  infisical-redis:
    image: redis:7-alpine
    volumes:
      - infisical_redis_data:/data
    networks:
      - rkg-network

volumes:
  qdrant_data:
  neo4j_data:
  neo4j_logs:
  langfuse_postgres_data:
  infisical_mongo_data:
  infisical_redis_data:

networks:
  rkg-network:
    driver: bridge
```

---

# Part 4: Feature Flags & Configuration

## 4.1 Unified Feature Flag System

```python
# src/rkg_mcp/config/feature_flags.py

from pydantic import BaseModel
from typing import Dict, Any
from enum import Enum

class FeatureFlag(str, Enum):
    # Core features
    ENTITY_EXTRACTION = "entity_extraction"
    INCREMENTAL_SYNC = "incremental_sync"
    EXPORT_IMPORT = "export_import"
    ANALYTICS = "analytics"
    CONTRADICTION_DETECTION = "contradiction_detection"

    # Observability
    LANGFUSE = "langfuse"
    NEWRELIC = "newrelic"

    # Secrets management
    INFISICAL = "infisical"
    INFISICAL_PRO = "infisical_pro"

    # Storage features
    SPARSE_VECTORS = "sparse_vectors"
    RERANKING = "reranking"

class FeatureFlags(BaseModel):
    """Feature flag configuration."""

    flags: Dict[FeatureFlag, bool] = {
        # Core features - enabled by default
        FeatureFlag.ENTITY_EXTRACTION: True,
        FeatureFlag.INCREMENTAL_SYNC: True,
        FeatureFlag.EXPORT_IMPORT: True,
        FeatureFlag.ANALYTICS: True,
        FeatureFlag.CONTRADICTION_DETECTION: True,

        # Observability - enabled by default
        FeatureFlag.LANGFUSE: True,
        FeatureFlag.NEWRELIC: True,

        # Secrets - Infisical enabled, Pro features require upgrade
        FeatureFlag.INFISICAL: True,
        FeatureFlag.INFISICAL_PRO: False,  # Set to True with Pro license

        # Storage features
        FeatureFlag.SPARSE_VECTORS: True,
        FeatureFlag.RERANKING: True,
    }

    def is_enabled(self, flag: FeatureFlag) -> bool:
        return self.flags.get(flag, False)

    def enable(self, flag: FeatureFlag):
        self.flags[flag] = True

    def disable(self, flag: FeatureFlag):
        self.flags[flag] = False

    def set_from_env(self):
        """Override flags from environment variables."""
        import os
        for flag in FeatureFlag:
            env_key = f"FEATURE_{flag.value.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                self.flags[flag] = env_value.lower() in ("true", "1", "yes")

# Global singleton
_feature_flags: FeatureFlags | None = None

def get_feature_flags() -> FeatureFlags:
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
        _feature_flags.set_from_env()
    return _feature_flags

def require_feature(flag: FeatureFlag):
    """Decorator to require a feature flag."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not get_feature_flags().is_enabled(flag):
                raise FeatureDisabledError(f"Feature '{flag.value}' is disabled")
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not get_feature_flags().is_enabled(flag):
                raise FeatureDisabledError(f"Feature '{flag.value}' is disabled")
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class FeatureDisabledError(Exception):
    """Raised when a disabled feature is accessed."""
    pass
```

---

# Part 5: Application Naming

## Top 10 Proposed Names

Based on the system's purpose (capturing ephemeral research from agentic coding sessions into a persistent, searchable knowledge graph), here are 10 naming suggestions:

| Rank | Name | Rationale |
|------|------|-----------|
| 1 | **Mnemos** | Greek for "memory" - captures the essence of persistent research memory |
| 2 | **ResearchVault** | Conveys secure, permanent storage of research knowledge |
| 3 | **KnowledgeWeave** | Emphasizes the interconnected graph nature of the knowledge |
| 4 | **SessionMind** | Focuses on the session-capture and intelligent retrieval aspects |
| 5 | **GraphRecall** | Combines graph database with the recall/retrieval function |
| 6 | **AgenticArchive** | Highlights the agentic coding origin of the captured research |
| 7 | **NexusKG** | "Nexus" (connection point) + "KG" (Knowledge Graph) |
| 8 | **ResearchLoom** | Evokes weaving together disparate research threads |
| 9 | **EphemeralEdge** | Captures the transformation from ephemeral to permanent |
| 10 | **Cognium** | Derived from "cognition" - suggests intelligent knowledge management |

### Name Recommendations by Use Case

**For Enterprise/Professional Use**: ResearchVault, NexusKG, AgenticArchive
- These names convey professionalism and security

**For Developer/Technical Audience**: Mnemos, GraphRecall, Cognium
- These names are more memorable and technically evocative

**For Broader Appeal**: KnowledgeWeave, SessionMind, ResearchLoom
- These names are intuitive and descriptive

### My Top Recommendation: **Mnemos**

Mnemos (μνῆμος) is the Greek root for "memory" (same root as "mnemonic"). It's:
- Short and memorable
- Technically relevant (knowledge preservation)
- Unique and not commonly used
- Works well as a product name: "Mnemos MCP Server", "Mnemos Explorer"
- Easy to pronounce across languages

---

# Summary

This specification extends the RKG system with:

1. **Entity Extraction Pipeline** - GLiNER + spaCy hybrid for automatic NER
2. **Incremental Session Sync** - File watching with debounced callbacks
3. **Export Capabilities** - JSON, Markdown, Cypher export formats
4. **Analytics Dashboard** - Research activity metrics and gap analysis
5. **Contradiction Detection** - Semantic similarity + LLM verification
6. **LangFuse Integration** - LLM observability with self-hosted deployment
7. **NewRelic Integration** - APM, distributed tracing, logs-in-context
8. **Infisical Integration** - Secrets management with Pro/Free tier support
9. **Unified Feature Flags** - Consistent enable/disable across all features
10. **Complete Docker Infrastructure** - Combined compose with all services

All integrations follow the modular provider pattern from the original specification and include proper error handling, feature flags, and graceful degradation.
