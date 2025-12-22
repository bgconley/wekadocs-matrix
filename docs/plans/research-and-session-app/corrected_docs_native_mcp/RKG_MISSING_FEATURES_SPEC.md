# RKG Missing Features Specification
## Research-Informed Implementation Guide

*This document provides detailed, research-backed specifications for the five core features identified as gaps in the original RKG architecture.*

---

## Table of Contents

1. [Entity Extraction Pipeline](#1-entity-extraction-pipeline)
2. [Incremental Session Sync](#2-incremental-session-sync)
3. [Export Capabilities](#3-export-capabilities)
4. [Analytics & Insights Dashboard](#4-analytics--insights-dashboard)
5. [Contradiction Detection](#5-contradiction-detection)
6. [Integration Patterns](#6-integration-patterns)
7. [Implementation Priority & Dependencies](#7-implementation-priority--dependencies)

---

## 1. Entity Extraction Pipeline

### 1.1 Research Foundation

Based on Neo4j's Information Extraction Pipeline architecture, the entity extraction system follows a 4-stage pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Coreference   │───▶│     Named       │───▶│  Relationship   │───▶│   Knowledge     │
│   Resolution    │    │    Entity       │    │   Extraction    │    │     Graph       │
│                 │    │  Recognition    │    │                 │    │   Population    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Insight**: Entity linking/wikification solves the disambiguation problem where multiple text forms should map to a single graph node.

### 1.2 Architecture

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PRODUCT = "product"
    LOCATION = "location"
    EVENT = "event"
    METRIC = "metric"
    API = "api"
    FRAMEWORK = "framework"
    CUSTOM = "custom"

class RelationType(Enum):
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    AUTHORED_BY = "AUTHORED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    COMPETES_WITH = "COMPETES_WITH"
    PART_OF = "PART_OF"
    SUCCESSOR_OF = "SUCCESSOR_OF"
    USES = "USES"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"

@dataclass
class ExtractedEntity:
    """Entity extracted from text with metadata."""
    text: str                          # Original mention text
    entity_type: EntityType
    confidence: float                  # 0.0-1.0 extraction confidence
    start_char: int                    # Character offset in source
    end_char: int
    normalized_name: str               # Canonical form for deduplication
    wikipedia_id: Optional[str] = None # Entity linking result
    wikidata_qid: Optional[str] = None # Wikidata identifier
    aliases: list[str] = field(default_factory=list)
    context_sentence: Optional[str] = None

    @property
    def canonical_id(self) -> str:
        """Generate stable ID for graph node matching."""
        key = f"{self.entity_type.value}:{self.normalized_name.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

@dataclass
class ExtractedRelationship:
    """Knowledge triplet extracted from text."""
    subject: ExtractedEntity
    predicate: RelationType
    object: ExtractedEntity
    confidence: float
    source_sentence: str
    extraction_method: str  # "rule_based", "rebel", "cooccurrence"
```

### 1.3 Hybrid Extraction Strategy

Combine multiple extraction methods for comprehensive coverage:

#### 1.3.1 GLiNER (Zero-Shot NER)

```python
from gliner import GLiNER

class GLiNERExtractor:
    """Zero-shot NER using GLiNER for flexible entity types."""

    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1"):
        self.model = GLiNER.from_pretrained(model_name)

        # Technical domain entity types
        self.entity_labels = [
            "person", "organization", "technology", "programming language",
            "framework", "library", "api", "database", "cloud service",
            "file format", "protocol", "algorithm", "data structure",
            "software tool", "operating system", "hardware"
        ]

    def extract(self, text: str, threshold: float = 0.5) -> list[ExtractedEntity]:
        """Extract entities with configurable confidence threshold."""
        entities = self.model.predict_entities(
            text,
            self.entity_labels,
            threshold=threshold
        )

        return [
            ExtractedEntity(
                text=ent["text"],
                entity_type=self._map_label(ent["label"]),
                confidence=ent["score"],
                start_char=ent["start"],
                end_char=ent["end"],
                normalized_name=self._normalize(ent["text"])
            )
            for ent in entities
        ]

    def _map_label(self, label: str) -> EntityType:
        """Map GLiNER labels to EntityType enum."""
        mapping = {
            "programming language": EntityType.TECHNOLOGY,
            "framework": EntityType.FRAMEWORK,
            "library": EntityType.TECHNOLOGY,
            "api": EntityType.API,
            "database": EntityType.TECHNOLOGY,
            "cloud service": EntityType.PRODUCT,
            "software tool": EntityType.PRODUCT,
        }
        return mapping.get(label, EntityType.CONCEPT)

    def _normalize(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        # Handle common variations
        text = text.strip().lower()
        # Expand common abbreviations
        abbreviations = {
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "k8s": "kubernetes",
            "tf": "terraform",
            "aws": "amazon web services",
        }
        return abbreviations.get(text, text)
```

#### 1.3.2 REBEL (Relationship Extraction)

```python
from transformers import pipeline

class REBELExtractor:
    """End-to-end relationship extraction using REBEL model.

    REBEL extracts knowledge triplets in format:
    <triplet> subject <subj> object <obj> relation

    Based on: https://github.com/Babelscape/rebel
    Trained on 200+ relation types from Wikipedia ontology.
    """

    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto"
        )

    def extract_triplets(self, text: str) -> list[ExtractedRelationship]:
        """Extract knowledge triplets from text."""
        # Generate triplets
        output = self.pipe(
            text,
            max_length=512,
            num_beams=5,
            num_return_sequences=3,
            decoder_start_token_id=0
        )

        triplets = []
        for seq in output:
            triplets.extend(self._parse_rebel_output(seq["generated_text"], text))

        return triplets

    def _parse_rebel_output(self, output: str, source: str) -> list[ExtractedRelationship]:
        """Parse REBEL output format to ExtractedRelationship objects."""
        triplets = []

        # REBEL format: <triplet> subj <subj> obj <obj> relation
        import re
        pattern = r'<triplet>\s*(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+?)(?=<triplet>|$)'

        for match in re.finditer(pattern, output):
            subject_text, object_text, relation = match.groups()

            # Create entity objects
            subject = ExtractedEntity(
                text=subject_text.strip(),
                entity_type=EntityType.CONCEPT,  # Default, refine with NER
                confidence=0.8,
                start_char=0,  # Would need alignment for precise offsets
                end_char=0,
                normalized_name=subject_text.strip().lower()
            )

            obj = ExtractedEntity(
                text=object_text.strip(),
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
                start_char=0,
                end_char=0,
                normalized_name=object_text.strip().lower()
            )

            triplets.append(ExtractedRelationship(
                subject=subject,
                predicate=self._map_relation(relation.strip()),
                object=obj,
                confidence=0.8,
                source_sentence=source[:500],
                extraction_method="rebel"
            ))

        return triplets

    def _map_relation(self, relation: str) -> RelationType:
        """Map REBEL relations to RelationType enum."""
        # Map common REBEL relations to our types
        mapping = {
            "developer": RelationType.AUTHORED_BY,
            "founded by": RelationType.AUTHORED_BY,
            "instance of": RelationType.PART_OF,
            "subclass of": RelationType.PART_OF,
            "part of": RelationType.PART_OF,
            "uses": RelationType.USES,
            "dependency": RelationType.DEPENDS_ON,
        }
        return mapping.get(relation.lower(), RelationType.RELATED_TO)
```

#### 1.3.3 Rule-Based Co-occurrence Extraction

```python
import spacy
from collections import defaultdict

class CooccurrenceExtractor:
    """Extract relationships based on entity co-occurrence in sentences.

    This approach infers relationships based on entities appearing
    in the same sentence, leveraging grammatical dependencies.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")  # Transformer model

    def extract_cooccurrences(
        self,
        text: str,
        entities: list[ExtractedEntity],
        window_size: int = 1  # Sentences
    ) -> list[ExtractedRelationship]:
        """Extract relationships from entity co-occurrences."""
        doc = self.nlp(text)
        relationships = []

        # Map entities to sentences
        entity_sentences = defaultdict(list)
        for sent_idx, sent in enumerate(doc.sents):
            for entity in entities:
                if entity.start_char >= sent.start_char and entity.end_char <= sent.end_char:
                    entity_sentences[sent_idx].append(entity)

        # Find co-occurring pairs
        for sent_idx, sent_entities in entity_sentences.items():
            if len(sent_entities) < 2:
                continue

            # Check entities within window
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # Determine relationship based on grammatical structure
                    rel_type = self._infer_relationship(
                        doc, ent1, ent2, list(doc.sents)[sent_idx]
                    )

                    relationships.append(ExtractedRelationship(
                        subject=ent1,
                        predicate=rel_type,
                        object=ent2,
                        confidence=0.6,  # Lower confidence for co-occurrence
                        source_sentence=str(list(doc.sents)[sent_idx]),
                        extraction_method="cooccurrence"
                    ))

        return relationships

    def _infer_relationship(
        self,
        doc,
        ent1: ExtractedEntity,
        ent2: ExtractedEntity,
        sentence
    ) -> RelationType:
        """Infer relationship type from grammatical structure."""
        # Look for specific dependency patterns
        for token in sentence:
            # Subject-verb-object patterns
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb = token.lemma_.lower()

                # Map verbs to relationship types
                if verb in {"use", "utilize", "employ", "leverage"}:
                    return RelationType.USES
                elif verb in {"depend", "require", "need"}:
                    return RelationType.DEPENDS_ON
                elif verb in {"implement", "provide", "offer"}:
                    return RelationType.IMPLEMENTS
                elif verb in {"compete", "rival"}:
                    return RelationType.COMPETES_WITH

        return RelationType.CO_OCCURS_WITH
```

### 1.4 Entity Pipeline Orchestrator

```python
import asyncio
from typing import Protocol
from concurrent.futures import ThreadPoolExecutor

class EntityExtractor(Protocol):
    """Protocol for entity extractors."""
    def extract(self, text: str) -> list[ExtractedEntity]: ...

class EntityExtractionPipeline:
    """Orchestrates multi-stage entity extraction.

    Pipeline stages:
    1. Coreference resolution (replace pronouns)
    2. Named entity recognition (GLiNER + spaCy)
    3. Entity linking (Wikipedia disambiguation)
    4. Relationship extraction (REBEL + co-occurrence)
    5. Graph population (Neo4j)
    """

    def __init__(
        self,
        gliner_extractor: GLiNERExtractor,
        rebel_extractor: REBELExtractor,
        cooccurrence_extractor: CooccurrenceExtractor,
        neo4j_client,
        max_workers: int = 4
    ):
        self.gliner = gliner_extractor
        self.rebel = rebel_extractor
        self.cooccurrence = cooccurrence_extractor
        self.neo4j = neo4j_client
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_workers)

    async def process_document(
        self,
        document_id: str,
        text: str,
        source_metadata: dict
    ) -> dict:
        """Full extraction pipeline for a document."""
        async with self._semaphore:
            # Stage 1: Coreference resolution
            resolved_text = await self._resolve_coreferences(text)

            # Stage 2: Named entity recognition (parallel)
            entities = await asyncio.gather(
                self._extract_gliner(resolved_text),
                self._extract_spacy(resolved_text)
            )
            merged_entities = self._merge_entities(entities[0], entities[1])

            # Stage 3: Entity linking (parallel batch)
            linked_entities = await self._link_entities(merged_entities)

            # Stage 4: Relationship extraction (parallel)
            relationships = await asyncio.gather(
                self._extract_rebel(resolved_text),
                self._extract_cooccurrence(resolved_text, linked_entities)
            )
            all_relationships = relationships[0] + relationships[1]

            # Stage 5: Graph population
            await self._populate_graph(
                document_id,
                linked_entities,
                all_relationships,
                source_metadata
            )

            return {
                "document_id": document_id,
                "entities_extracted": len(linked_entities),
                "relationships_extracted": len(all_relationships),
                "entities": [e.text for e in linked_entities],
                "relationship_types": list(set(r.predicate.value for r in all_relationships))
            }

    async def _resolve_coreferences(self, text: str) -> str:
        """Replace pronouns with their referents.

        Example: "Elon Musk founded Tesla. He is also CEO of SpaceX."
        Becomes: "Elon Musk founded Tesla. Elon Musk is also CEO of SpaceX."
        """
        # Use coreferee or neuralcoref for resolution
        # Simplified implementation - production would use proper library
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._resolve_coreferences_sync,
            text
        )

    def _resolve_coreferences_sync(self, text: str) -> str:
        """Synchronous coreference resolution."""
        # In production, use:
        # - coreferee (spaCy extension)
        # - fastcoref
        # - neuralcoref (legacy)
        return text  # Placeholder

    async def _extract_gliner(self, text: str) -> list[ExtractedEntity]:
        """Run GLiNER extraction in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.gliner.extract,
            text
        )

    async def _extract_spacy(self, text: str) -> list[ExtractedEntity]:
        """Run spaCy NER in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._spacy_extract_sync,
            text
        )

    def _spacy_extract_sync(self, text: str) -> list[ExtractedEntity]:
        """Synchronous spaCy extraction."""
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(text)

        return [
            ExtractedEntity(
                text=ent.text,
                entity_type=self._spacy_to_entity_type(ent.label_),
                confidence=0.9,
                start_char=ent.start_char,
                end_char=ent.end_char,
                normalized_name=ent.text.lower()
            )
            for ent in doc.ents
        ]

    def _spacy_to_entity_type(self, label: str) -> EntityType:
        """Map spaCy labels to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "MONEY": EntityType.METRIC,
            "PERCENT": EntityType.METRIC,
            "CARDINAL": EntityType.METRIC,
        }
        return mapping.get(label, EntityType.CONCEPT)

    def _merge_entities(
        self,
        gliner_entities: list[ExtractedEntity],
        spacy_entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Merge and deduplicate entities from multiple sources."""
        seen = {}
        merged = []

        # Combine all entities
        all_entities = gliner_entities + spacy_entities

        for entity in all_entities:
            key = entity.canonical_id

            if key in seen:
                # Keep higher confidence version
                if entity.confidence > seen[key].confidence:
                    seen[key] = entity
            else:
                seen[key] = entity

        return list(seen.values())

    async def _link_entities(
        self,
        entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Link entities to Wikipedia/Wikidata for disambiguation.

        Entity linking solves problems like:
        - "Apple" -> Apple Inc. vs apple (fruit)
        - "Python" -> Python (programming language) vs Python (snake)
        """
        # In production, use:
        # - REL (Radboud Entity Linker)
        # - BLINK (Facebook)
        # - mGENRE (multilingual)

        # For now, return entities as-is
        # TODO: Implement entity linking
        return entities

    async def _extract_rebel(self, text: str) -> list[ExtractedRelationship]:
        """Run REBEL extraction in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.rebel.extract_triplets,
            text
        )

    async def _extract_cooccurrence(
        self,
        text: str,
        entities: list[ExtractedEntity]
    ) -> list[ExtractedRelationship]:
        """Run co-occurrence extraction in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.cooccurrence.extract_cooccurrences,
            text,
            entities
        )

    async def _populate_graph(
        self,
        document_id: str,
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelationship],
        metadata: dict
    ) -> None:
        """Populate Neo4j graph with extracted knowledge."""
        async with self.neo4j.session() as session:
            # Create/merge entity nodes
            for entity in entities:
                await session.run("""
                    MERGE (e:Entity {canonical_id: $canonical_id})
                    ON CREATE SET
                        e.name = $name,
                        e.type = $type,
                        e.created_at = datetime(),
                        e.aliases = $aliases
                    ON MATCH SET
                        e.aliases = CASE
                            WHEN NOT $name IN e.aliases
                            THEN e.aliases + $name
                            ELSE e.aliases
                        END,
                        e.mention_count = coalesce(e.mention_count, 0) + 1

                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[m:MENTIONS]->(e)
                    ON CREATE SET
                        m.first_seen = datetime(),
                        m.count = 1,
                        m.confidence = $confidence
                    ON MATCH SET
                        m.count = m.count + 1,
                        m.confidence = CASE
                            WHEN $confidence > m.confidence
                            THEN $confidence
                            ELSE m.confidence
                        END
                """, {
                    "canonical_id": entity.canonical_id,
                    "name": entity.normalized_name,
                    "type": entity.entity_type.value,
                    "aliases": entity.aliases + [entity.text],
                    "doc_id": document_id,
                    "confidence": entity.confidence
                })

            # Create relationship edges
            for rel in relationships:
                await session.run("""
                    MATCH (s:Entity {canonical_id: $subj_id})
                    MATCH (o:Entity {canonical_id: $obj_id})
                    MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(o)
                    ON CREATE SET
                        r.created_at = datetime(),
                        r.source_doc = $doc_id,
                        r.confidence = $confidence,
                        r.extraction_method = $method,
                        r.evidence = [$evidence]
                    ON MATCH SET
                        r.evidence = r.evidence + $evidence,
                        r.confidence = CASE
                            WHEN $confidence > r.confidence
                            THEN $confidence
                            ELSE r.confidence
                        END
                """, {
                    "subj_id": rel.subject.canonical_id,
                    "obj_id": rel.object.canonical_id,
                    "rel_type": rel.predicate.value,
                    "doc_id": document_id,
                    "confidence": rel.confidence,
                    "method": rel.extraction_method,
                    "evidence": rel.source_sentence[:500]
                })
```

### 1.5 Neo4j Schema for Entities

```cypher
-- Entity node constraints and indexes
CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE;

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases];

-- Mention relationship index
CREATE INDEX mention_confidence IF NOT EXISTS
FOR ()-[m:MENTIONS]-() ON (m.confidence);
```

---

## 2. Incremental Session Sync

### 2.1 Research Foundation

Based on Watchdog best practices for file monitoring:
- Rate-limiting with `collections.deque` to ignore events within 1s window
- Debouncing with 2s delay to wait for file completion
- State management with last_modified, last_position, checksum
- Incremental reading from last byte offset

### 2.2 Architecture

```python
import asyncio
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

@dataclass
class FileState:
    """Persistent state for incremental file processing."""
    path: str
    last_modified: float
    last_position: int  # Byte offset for incremental reading
    checksum: str       # Content hash for change detection
    last_processed: datetime
    line_count: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "last_modified": self.last_modified,
            "last_position": self.last_position,
            "checksum": self.checksum,
            "last_processed": self.last_processed.isoformat(),
            "line_count": self.line_count
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileState":
        return cls(
            path=data["path"],
            last_modified=data["last_modified"],
            last_position=data["last_position"],
            checksum=data["checksum"],
            last_processed=datetime.fromisoformat(data["last_processed"]),
            line_count=data.get("line_count", 0)
        )

@dataclass
class SyncConfig:
    """Configuration for incremental sync."""
    watch_directories: list[str] = field(default_factory=lambda: [
        os.path.expanduser("~/.claude"),
        os.path.expanduser("~/.codex"),
        os.path.expanduser("~/.cursor"),
    ])
    file_patterns: list[str] = field(default_factory=lambda: [
        "*.jsonl",
        "*.json",
        "session_*.txt"
    ])
    debounce_seconds: float = 2.0
    rate_limit_seconds: float = 1.0
    state_file: str = os.path.expanduser("~/.rkg/sync_state.json")
    max_concurrent_processing: int = 4

class RateLimitedHandler(FileSystemEventHandler):
    """File system handler with rate limiting and debouncing.

    Implements best practices from research:
    - Rate-limiting: Track event timestamps, ignore events within window
    - Debouncing: Wait for file completion before processing
    """

    def __init__(
        self,
        config: SyncConfig,
        on_file_ready: Callable[[str], None]
    ):
        self.config = config
        self.on_file_ready = on_file_ready

        # Rate limiting: track recent events per file
        self._event_times: dict[str, deque] = {}
        self._rate_limit_window = timedelta(seconds=config.rate_limit_seconds)

        # Debouncing: track pending files and their timers
        self._pending_files: dict[str, asyncio.Task] = {}
        self._loop = asyncio.get_event_loop()

    def _should_process(self, path: str) -> bool:
        """Check if file matches patterns and passes rate limit."""
        # Check file pattern
        from fnmatch import fnmatch
        if not any(fnmatch(os.path.basename(path), p) for p in self.config.file_patterns):
            return False

        # Rate limiting check
        now = datetime.now()
        if path not in self._event_times:
            self._event_times[path] = deque(maxlen=10)

        events = self._event_times[path]

        # Remove old events outside window
        while events and (now - events[0]) > self._rate_limit_window:
            events.popleft()

        # Check if we've had too many recent events
        if len(events) >= 3:  # Max 3 events per rate limit window
            return False

        events.append(now)
        return True

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._schedule_processing(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._schedule_processing(event.src_path)

    def _schedule_processing(self, path: str):
        """Schedule file processing with debounce delay."""
        # Cancel existing timer for this file
        if path in self._pending_files:
            self._pending_files[path].cancel()

        # Schedule new processing after debounce delay
        async def delayed_process():
            await asyncio.sleep(self.config.debounce_seconds)
            self.on_file_ready(path)

        self._pending_files[path] = self._loop.create_task(delayed_process())

class IncrementalSyncManager:
    """Manages incremental synchronization of session files.

    Key features:
    - Watches multiple directories (Claude, Codex, Cursor)
    - Incremental reading from last position
    - Persistent state across restarts
    - Semaphore-controlled concurrency
    """

    def __init__(
        self,
        config: SyncConfig,
        document_processor: Callable[[str, dict], asyncio.coroutine]
    ):
        self.config = config
        self.document_processor = document_processor

        self._state: dict[str, FileState] = {}
        self._observer: Optional[Observer] = None
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        self._running = False

        self._load_state()

    def _load_state(self):
        """Load persistent state from disk."""
        if os.path.exists(self.config.state_file):
            try:
                with open(self.config.state_file, 'r') as f:
                    data = json.load(f)
                    self._state = {
                        k: FileState.from_dict(v)
                        for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load sync state: {e}")
                self._state = {}

    def _save_state(self):
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self.config.state_file), exist_ok=True)
        with open(self.config.state_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self._state.items()},
                f,
                indent=2
            )

    async def start(self):
        """Start watching directories for changes."""
        self._running = True

        # Create handler
        handler = RateLimitedHandler(
            self.config,
            on_file_ready=lambda p: asyncio.create_task(self._process_file(p))
        )

        # Start observer
        self._observer = Observer()
        for directory in self.config.watch_directories:
            if os.path.exists(directory):
                self._observer.schedule(handler, directory, recursive=True)
                print(f"Watching: {directory}")

        self._observer.start()

        # Initial scan of existing files
        await self._initial_scan()

    async def stop(self):
        """Stop watching and save state."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        self._save_state()

    async def _initial_scan(self):
        """Scan watched directories for existing files."""
        from fnmatch import fnmatch

        tasks = []
        for directory in self.config.watch_directories:
            if not os.path.exists(directory):
                continue

            for root, _, files in os.walk(directory):
                for filename in files:
                    if any(fnmatch(filename, p) for p in self.config.file_patterns):
                        filepath = os.path.join(root, filename)
                        tasks.append(self._process_file(filepath))

        if tasks:
            await asyncio.gather(*tasks)

    async def _process_file(self, filepath: str):
        """Process a file incrementally."""
        async with self._processing_semaphore:
            try:
                stat = os.stat(filepath)
                current_mtime = stat.st_mtime

                # Check if file needs processing
                if filepath in self._state:
                    state = self._state[filepath]
                    if state.last_modified >= current_mtime:
                        return  # No changes
                else:
                    state = FileState(
                        path=filepath,
                        last_modified=0,
                        last_position=0,
                        checksum="",
                        last_processed=datetime.min
                    )

                # Read incrementally from last position
                new_content = await self._read_incremental(filepath, state.last_position)

                if not new_content:
                    return

                # Process new lines
                new_lines = new_content.strip().split('\n')
                for line in new_lines:
                    if not line.strip():
                        continue

                    try:
                        # Assume JSONL format
                        entry = json.loads(line)
                        await self.document_processor(filepath, entry)
                        state.line_count += 1
                    except json.JSONDecodeError:
                        # Handle non-JSON content
                        await self.document_processor(filepath, {"raw": line})
                        state.line_count += 1

                # Update state
                state.last_modified = current_mtime
                state.last_position = stat.st_size
                state.checksum = hashlib.md5(new_content.encode()).hexdigest()
                state.last_processed = datetime.now()

                self._state[filepath] = state
                self._save_state()

                print(f"Processed {len(new_lines)} new lines from {filepath}")

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    async def _read_incremental(self, filepath: str, position: int) -> str:
        """Read file content from last position."""
        loop = asyncio.get_event_loop()

        def read_sync():
            with open(filepath, 'r') as f:
                f.seek(position)
                return f.read()

        return await loop.run_in_executor(None, read_sync)

    def get_sync_status(self) -> dict:
        """Get current sync status for all tracked files."""
        return {
            "tracked_files": len(self._state),
            "total_lines_processed": sum(s.line_count for s in self._state.values()),
            "files": [
                {
                    "path": s.path,
                    "lines": s.line_count,
                    "last_processed": s.last_processed.isoformat()
                }
                for s in sorted(
                    self._state.values(),
                    key=lambda x: x.last_processed,
                    reverse=True
                )[:10]
            ]
        }
```

### 2.3 Integration with RKG Pipeline

```python
class RKGSessionSync:
    """Integrates incremental sync with RKG processing pipeline."""

    def __init__(
        self,
        entity_pipeline: EntityExtractionPipeline,
        qdrant_client,
        neo4j_client,
        voyage_client
    ):
        self.entity_pipeline = entity_pipeline
        self.qdrant = qdrant_client
        self.neo4j = neo4j_client
        self.voyage = voyage_client

        self.sync_manager = IncrementalSyncManager(
            config=SyncConfig(),
            document_processor=self._process_session_entry
        )

    async def _process_session_entry(self, filepath: str, entry: dict):
        """Process a single session entry through RKG pipeline."""
        # Extract session metadata
        session_id = self._extract_session_id(filepath)
        interface = self._detect_interface(filepath)

        # Handle different entry types
        if "content" in entry:
            await self._process_content(entry, session_id, interface)
        elif "tool_use" in entry or "tool_result" in entry:
            await self._process_tool_interaction(entry, session_id, interface)
        elif "search_results" in entry:
            await self._process_search_results(entry, session_id, interface)

    async def _process_content(self, entry: dict, session_id: str, interface: str):
        """Process text content entry."""
        content = entry["content"]

        # Generate embedding
        embedding = await self.voyage.embed(
            content,
            model="voyage-3-large",
            input_type="document"
        )

        # Store in Qdrant
        doc_id = hashlib.sha256(
            f"{session_id}:{content[:100]}".encode()
        ).hexdigest()[:16]

        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": doc_id,
                "vector": {"dense": embedding},
                "payload": {
                    "session_id": session_id,
                    "interface": interface,
                    "source_type": "session_transcript",
                    "content": content[:10000],
                    "created_at": datetime.now().isoformat()
                }
            }]
        )

        # Run entity extraction
        await self.entity_pipeline.process_document(
            document_id=doc_id,
            text=content,
            source_metadata={
                "session_id": session_id,
                "interface": interface
            }
        )

    def _extract_session_id(self, filepath: str) -> str:
        """Extract session ID from filepath."""
        # Handle different naming conventions
        filename = os.path.basename(filepath)
        # Example: session_2024-01-15_14-30-00.jsonl
        if filename.startswith("session_"):
            return filename.replace("session_", "").replace(".jsonl", "")
        return hashlib.md5(filepath.encode()).hexdigest()[:12]

    def _detect_interface(self, filepath: str) -> str:
        """Detect agentic interface from filepath."""
        if ".claude" in filepath:
            return "claude_code"
        elif ".codex" in filepath:
            return "openai_codex"
        elif ".cursor" in filepath:
            return "cursor"
        return "unknown"
```

---

## 3. Export Capabilities

### 3.1 Research Foundation

Based on Neo4j APOC export procedures and best practices:
- GraphML for visualization tools (Gephi, CytoScape)
- Mixed property types export as STRING
- POINT/temporal values formatted as STRING
- Labels exported alphabetically
- Round-trip support with `readLabels: true`, `storeNodeIds: true`

### 3.2 Architecture

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
import json
import csv
from datetime import datetime
import hashlib

class ExportFormat(Enum):
    JSON = "json"           # Full structured export
    JSONL = "jsonl"         # Streaming line-delimited
    MARKDOWN = "markdown"   # Human-readable
    CYPHER = "cypher"       # Neo4j import statements
    CSV = "csv"             # Tabular data
    GRAPHML = "graphml"     # Graph visualization tools

@dataclass
class ExportScope:
    """Defines what to export."""
    projects: Optional[list[str]] = None      # Filter by project
    sessions: Optional[list[str]] = None      # Filter by session
    date_from: Optional[datetime] = None      # Date range start
    date_to: Optional[datetime] = None        # Date range end
    entity_types: Optional[list[str]] = None  # Filter entity types
    include_embeddings: bool = False          # Include vector data
    include_metadata: bool = True             # Include full metadata
    max_documents: Optional[int] = None       # Limit results

@dataclass
class ExportMetadata:
    """Metadata about an export."""
    export_id: str
    format: ExportFormat
    scope: ExportScope
    created_at: datetime
    document_count: int
    entity_count: int
    relationship_count: int
    file_size_bytes: int
    checksum: str
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "export_id": self.export_id,
            "format": self.format.value,
            "scope": {
                "projects": self.scope.projects,
                "sessions": self.scope.sessions,
                "date_from": self.scope.date_from.isoformat() if self.scope.date_from else None,
                "date_to": self.scope.date_to.isoformat() if self.scope.date_to else None,
            },
            "created_at": self.created_at.isoformat(),
            "counts": {
                "documents": self.document_count,
                "entities": self.entity_count,
                "relationships": self.relationship_count
            },
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "version": self.version
        }

class KnowledgeGraphExporter:
    """Export knowledge graph data in various formats.

    Supports:
    - Full exports for backup
    - Filtered exports for sharing
    - Format conversion for visualization tools
    - Round-trip import/export
    """

    def __init__(self, neo4j_client, qdrant_client):
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client

    async def export(
        self,
        format: ExportFormat,
        scope: ExportScope,
        output_path: str
    ) -> ExportMetadata:
        """Export knowledge graph data."""
        # Fetch data based on scope
        documents = await self._fetch_documents(scope)
        entities = await self._fetch_entities(scope)
        relationships = await self._fetch_relationships(scope)

        # Export in requested format
        exporters = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.JSONL: self._export_jsonl,
            ExportFormat.MARKDOWN: self._export_markdown,
            ExportFormat.CYPHER: self._export_cypher,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.GRAPHML: self._export_graphml,
        }

        await exporters[format](documents, entities, relationships, output_path, scope)

        # Calculate metadata
        file_size = os.path.getsize(output_path)
        with open(output_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        return ExportMetadata(
            export_id=hashlib.sha256(f"{output_path}:{datetime.now()}".encode()).hexdigest()[:16],
            format=format,
            scope=scope,
            created_at=datetime.now(),
            document_count=len(documents),
            entity_count=len(entities),
            relationship_count=len(relationships),
            file_size_bytes=file_size,
            checksum=checksum
        )

    async def _fetch_documents(self, scope: ExportScope) -> list[dict]:
        """Fetch documents from Qdrant based on scope."""
        # Build filter
        filters = []

        if scope.projects:
            filters.append({
                "key": "project",
                "match": {"any": scope.projects}
            })

        if scope.sessions:
            filters.append({
                "key": "session_id",
                "match": {"any": scope.sessions}
            })

        if scope.date_from:
            filters.append({
                "key": "created_at",
                "range": {"gte": scope.date_from.isoformat()}
            })

        if scope.date_to:
            filters.append({
                "key": "created_at",
                "range": {"lte": scope.date_to.isoformat()}
            })

        # Query Qdrant
        results = await self.qdrant.scroll(
            collection_name="research_documents",
            filter={"must": filters} if filters else None,
            limit=scope.max_documents or 10000,
            with_vectors=scope.include_embeddings
        )

        return [
            {
                "id": point.id,
                "payload": point.payload,
                "vector": point.vector if scope.include_embeddings else None
            }
            for point in results[0]
        ]

    async def _fetch_entities(self, scope: ExportScope) -> list[dict]:
        """Fetch entities from Neo4j based on scope."""
        # Build Cypher query
        where_clauses = []
        params = {}

        if scope.entity_types:
            where_clauses.append("e.type IN $entity_types")
            params["entity_types"] = scope.entity_types

        if scope.projects:
            where_clauses.append("""
                EXISTS {
                    MATCH (e)<-[:MENTIONS]-(d:Document)
                    WHERE d.project IN $projects
                }
            """)
            params["projects"] = scope.projects

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            MATCH (e:Entity)
            {where_clause}
            RETURN e {{
                .canonical_id,
                .name,
                .type,
                .aliases,
                .mention_count,
                .created_at
            }} AS entity
            LIMIT $limit
        """
        params["limit"] = scope.max_documents or 10000

        async with self.neo4j.session() as session:
            result = await session.run(query, params)
            return [record["entity"] async for record in result]

    async def _fetch_relationships(self, scope: ExportScope) -> list[dict]:
        """Fetch relationships from Neo4j based on scope."""
        query = """
            MATCH (s:Entity)-[r]->(o:Entity)
            RETURN {
                source: s.canonical_id,
                target: o.canonical_id,
                type: type(r),
                confidence: r.confidence,
                evidence: r.evidence
            } AS relationship
            LIMIT $limit
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, {"limit": scope.max_documents or 50000})
            return [record["relationship"] async for record in result]

    async def _export_json(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as single JSON file."""
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "scope": {
                    "projects": scope.projects,
                    "sessions": scope.sessions
                }
            },
            "documents": documents,
            "entities": entities,
            "relationships": relationships
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    async def _export_jsonl(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as streaming JSONL (line-delimited JSON)."""
        with open(output_path, 'w') as f:
            # Write metadata header
            f.write(json.dumps({
                "type": "metadata",
                "exported_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }) + "\n")

            # Write documents
            for doc in documents:
                f.write(json.dumps({"type": "document", "data": doc}, default=str) + "\n")

            # Write entities
            for entity in entities:
                f.write(json.dumps({"type": "entity", "data": entity}, default=str) + "\n")

            # Write relationships
            for rel in relationships:
                f.write(json.dumps({"type": "relationship", "data": rel}, default=str) + "\n")

    async def _export_markdown(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as human-readable Markdown."""
        lines = [
            "# Knowledge Graph Export",
            f"\nExported: {datetime.now().isoformat()}",
            f"\n## Summary",
            f"- Documents: {len(documents)}",
            f"- Entities: {len(entities)}",
            f"- Relationships: {len(relationships)}",
            "\n---\n",
            "## Entities\n"
        ]

        # Group entities by type
        by_type = {}
        for e in entities:
            t = e.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in sorted(by_type.items()):
            lines.append(f"\n### {entity_type.title()} ({len(type_entities)})\n")
            for e in type_entities[:50]:  # Limit per type
                lines.append(f"- **{e.get('name', 'Unknown')}**")
                if e.get('aliases'):
                    lines.append(f"  - Aliases: {', '.join(e['aliases'][:5])}")
                if e.get('mention_count'):
                    lines.append(f"  - Mentions: {e['mention_count']}")

        lines.append("\n---\n")
        lines.append("## Key Relationships\n")

        # Show top relationships
        for rel in relationships[:100]:
            lines.append(
                f"- {rel.get('source', '?')} → "
                f"**{rel.get('type', 'RELATED_TO')}** → "
                f"{rel.get('target', '?')}"
            )

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    async def _export_cypher(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as Cypher statements for Neo4j import."""
        lines = [
            "// RKG Knowledge Graph Export",
            f"// Generated: {datetime.now().isoformat()}",
            "// Import with: cat export.cypher | cypher-shell -u neo4j -p password",
            "",
            "// Create constraints",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
            "",
            "// Create entities"
        ]

        for entity in entities:
            props = json.dumps(entity, default=str)
            lines.append(f"MERGE (e:Entity {{canonical_id: '{entity.get('canonical_id', '')}'}}) SET e += {props};")

        lines.append("\n// Create relationships")

        for rel in relationships:
            lines.append(
                f"MATCH (s:Entity {{canonical_id: '{rel.get('source', '')}'}}), "
                f"(o:Entity {{canonical_id: '{rel.get('target', '')}'}}) "
                f"MERGE (s)-[r:{rel.get('type', 'RELATED_TO')}]->(o) "
                f"SET r.confidence = {rel.get('confidence', 0.5)};"
            )

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    async def _export_csv(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as CSV files (creates multiple files)."""
        base_path = output_path.replace('.csv', '')

        # Export entities
        with open(f"{base_path}_entities.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'canonical_id', 'name', 'type', 'mention_count', 'aliases'
            ])
            writer.writeheader()
            for e in entities:
                writer.writerow({
                    'canonical_id': e.get('canonical_id', ''),
                    'name': e.get('name', ''),
                    'type': e.get('type', ''),
                    'mention_count': e.get('mention_count', 0),
                    'aliases': '|'.join(e.get('aliases', []))
                })

        # Export relationships
        with open(f"{base_path}_relationships.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'source', 'target', 'type', 'confidence'
            ])
            writer.writeheader()
            for r in relationships:
                writer.writerow({
                    'source': r.get('source', ''),
                    'target': r.get('target', ''),
                    'type': r.get('type', ''),
                    'confidence': r.get('confidence', 0.5)
                })

    async def _export_graphml(
        self,
        documents: list[dict],
        entities: list[dict],
        relationships: list[dict],
        output_path: str,
        scope: ExportScope
    ):
        """Export as GraphML for visualization tools (Gephi, CytoScape).

        Note: Based on Neo4j APOC research:
        - Mixed property types → exported as STRING
        - Labels exported alphabetically
        """
        import xml.etree.ElementTree as ET

        # Create GraphML structure
        graphml = ET.Element('graphml', {
            'xmlns': 'http://graphml.graphdrawing.org/xmlns',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        })

        # Define keys (attributes)
        keys = [
            ('name', 'node', 'string'),
            ('type', 'node', 'string'),
            ('mention_count', 'node', 'int'),
            ('rel_type', 'edge', 'string'),
            ('confidence', 'edge', 'double'),
        ]

        for key_id, for_type, attr_type in keys:
            ET.SubElement(graphml, 'key', {
                'id': key_id,
                'for': for_type,
                'attr.name': key_id,
                'attr.type': attr_type
            })

        # Create graph
        graph = ET.SubElement(graphml, 'graph', {
            'id': 'RKG',
            'edgedefault': 'directed'
        })

        # Add nodes
        for entity in entities:
            node = ET.SubElement(graph, 'node', {
                'id': entity.get('canonical_id', '')
            })

            for key, value in [
                ('name', entity.get('name', '')),
                ('type', entity.get('type', '')),
                ('mention_count', str(entity.get('mention_count', 0)))
            ]:
                data = ET.SubElement(node, 'data', {'key': key})
                data.text = str(value)

        # Add edges
        for i, rel in enumerate(relationships):
            edge = ET.SubElement(graph, 'edge', {
                'id': f"e{i}",
                'source': rel.get('source', ''),
                'target': rel.get('target', '')
            })

            rel_type = ET.SubElement(edge, 'data', {'key': 'rel_type'})
            rel_type.text = rel.get('type', 'RELATED_TO')

            confidence = ET.SubElement(edge, 'data', {'key': 'confidence'})
            confidence.text = str(rel.get('confidence', 0.5))

        # Write to file
        tree = ET.ElementTree(graphml)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
```

### 3.3 Import with Merge Strategies

```python
class ImportStrategy(Enum):
    SKIP_EXISTING = "skip_existing"   # Don't update existing records
    OVERWRITE = "overwrite"           # Replace existing records
    MERGE = "merge"                   # Merge properties, keep newest

class KnowledgeGraphImporter:
    """Import knowledge graph data with configurable merge strategies."""

    def __init__(self, neo4j_client, qdrant_client, voyage_client):
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client
        self.voyage = voyage_client

    async def import_data(
        self,
        input_path: str,
        strategy: ImportStrategy = ImportStrategy.MERGE,
        re_embed: bool = True
    ) -> dict:
        """Import knowledge graph data.

        Args:
            input_path: Path to import file (JSON or JSONL)
            strategy: How to handle existing records
            re_embed: Whether to regenerate embeddings (recommended for consistency)
        """
        # Detect format
        if input_path.endswith('.jsonl'):
            data = self._read_jsonl(input_path)
        else:
            data = self._read_json(input_path)

        stats = {
            "documents_imported": 0,
            "entities_imported": 0,
            "relationships_imported": 0,
            "skipped": 0,
            "errors": 0
        }

        # Import documents
        for doc in data.get("documents", []):
            try:
                await self._import_document(doc, strategy, re_embed)
                stats["documents_imported"] += 1
            except Exception as e:
                print(f"Error importing document: {e}")
                stats["errors"] += 1

        # Import entities
        for entity in data.get("entities", []):
            try:
                await self._import_entity(entity, strategy)
                stats["entities_imported"] += 1
            except Exception as e:
                print(f"Error importing entity: {e}")
                stats["errors"] += 1

        # Import relationships
        for rel in data.get("relationships", []):
            try:
                await self._import_relationship(rel, strategy)
                stats["relationships_imported"] += 1
            except Exception as e:
                print(f"Error importing relationship: {e}")
                stats["errors"] += 1

        return stats

    async def _import_document(
        self,
        doc: dict,
        strategy: ImportStrategy,
        re_embed: bool
    ):
        """Import a single document."""
        doc_id = doc.get("id")
        payload = doc.get("payload", {})

        # Check if exists
        existing = await self.qdrant.retrieve(
            collection_name="research_documents",
            ids=[doc_id]
        )

        if existing and strategy == ImportStrategy.SKIP_EXISTING:
            return

        # Re-generate embedding if requested
        if re_embed and payload.get("content"):
            vector = await self.voyage.embed(
                payload["content"],
                model="voyage-3-large",
                input_type="document"
            )
        else:
            vector = doc.get("vector")

        # Upsert to Qdrant
        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": doc_id,
                "vector": {"dense": vector},
                "payload": payload
            }]
        )

    async def _import_entity(self, entity: dict, strategy: ImportStrategy):
        """Import a single entity to Neo4j."""
        if strategy == ImportStrategy.SKIP_EXISTING:
            merge_clause = "ON CREATE SET e += $props"
        elif strategy == ImportStrategy.OVERWRITE:
            merge_clause = "SET e = $props"
        else:  # MERGE
            merge_clause = """
                ON CREATE SET e = $props
                ON MATCH SET
                    e.aliases = CASE
                        WHEN e.aliases IS NULL THEN $props.aliases
                        ELSE [x IN e.aliases WHERE NOT x IN $props.aliases] + $props.aliases
                    END,
                    e.mention_count = CASE
                        WHEN e.mention_count > coalesce($props.mention_count, 0)
                        THEN e.mention_count
                        ELSE $props.mention_count
                    END
            """

        query = f"""
            MERGE (e:Entity {{canonical_id: $canonical_id}})
            {merge_clause}
        """

        async with self.neo4j.session() as session:
            await session.run(query, {
                "canonical_id": entity.get("canonical_id"),
                "props": entity
            })

    async def _import_relationship(self, rel: dict, strategy: ImportStrategy):
        """Import a single relationship to Neo4j."""
        query = """
            MATCH (s:Entity {canonical_id: $source})
            MATCH (o:Entity {canonical_id: $target})
            MERGE (s)-[r:RELATES_TO {type: $type}]->(o)
            ON CREATE SET r.confidence = $confidence, r.imported_at = datetime()
            ON MATCH SET r.confidence = CASE
                WHEN $confidence > r.confidence THEN $confidence
                ELSE r.confidence
            END
        """

        async with self.neo4j.session() as session:
            await session.run(query, {
                "source": rel.get("source"),
                "target": rel.get("target"),
                "type": rel.get("type", "RELATED_TO"),
                "confidence": rel.get("confidence", 0.5)
            })

    def _read_json(self, path: str) -> dict:
        """Read JSON export file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _read_jsonl(self, path: str) -> dict:
        """Read JSONL export file."""
        data = {"documents": [], "entities": [], "relationships": []}

        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entry_type = entry.get("type")
                if entry_type == "document":
                    data["documents"].append(entry.get("data", {}))
                elif entry_type == "entity":
                    data["entities"].append(entry.get("data", {}))
                elif entry_type == "relationship":
                    data["relationships"].append(entry.get("data", {}))

        return data
```

---

## 4. Analytics & Insights Dashboard

### 4.1 Research Foundation

Based on knowledge graph analytics patterns:
- Activity metrics: sessions, documents, insights by source/interface/project
- Entity coverage: mention counts, top entities, trending over time
- Knowledge gaps: orphan documents, unconfirmed insights, stale topics
- Graph statistics: node counts, relationship density, common patterns

### 4.2 Metrics Data Models

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

@dataclass
class TimeRange:
    """Time range for analytics queries."""
    start: datetime
    end: datetime

    @classmethod
    def last_n_days(cls, n: int) -> "TimeRange":
        end = datetime.now()
        start = end - timedelta(days=n)
        return cls(start=start, end=end)

    @classmethod
    def last_week(cls) -> "TimeRange":
        return cls.last_n_days(7)

    @classmethod
    def last_month(cls) -> "TimeRange":
        return cls.last_n_days(30)

@dataclass
class ResearchActivityMetrics:
    """Metrics about research activity over time."""
    time_range: TimeRange
    total_sessions: int
    total_documents: int
    total_insights: int
    documents_by_source: dict[str, int]      # brave_search, firecrawl, etc.
    documents_by_interface: dict[str, int]   # claude_code, codex, etc.
    documents_by_project: dict[str, int]
    daily_activity: list[dict]               # [{date, documents, insights}]
    peak_activity_hour: int                  # 0-23
    avg_documents_per_session: float

@dataclass
class EntityCoverageMetrics:
    """Metrics about entity extraction and coverage."""
    total_entities: int
    entities_by_type: dict[str, int]
    top_entities: list[dict]                 # [{name, type, mentions}]
    trending_entities: list[dict]            # Entities with recent growth
    orphan_entities: int                     # Entities with no relationships
    avg_mentions_per_entity: float
    entity_growth_rate: float                # New entities per day

@dataclass
class KnowledgeGapMetrics:
    """Metrics identifying gaps in the knowledge graph."""
    orphan_documents: int                    # Documents with no entities
    unlinked_entities: int                   # Entities with no relationships
    stale_topics: list[dict]                 # Topics not updated recently
    low_confidence_insights: int             # Insights below threshold
    missing_entity_types: list[str]          # Expected types with no entries
    coverage_by_topic: dict[str, float]      # Topic -> coverage score

@dataclass
class InsightTrendMetrics:
    """Metrics about insight generation and quality."""
    total_insights: int
    insights_by_type: dict[str, int]         # fact, opinion, question, etc.
    confidence_distribution: dict[str, int]  # {high: n, medium: n, low: n}
    contradiction_count: int
    confirmed_insights: int
    insights_per_session: float
    top_insight_sources: list[dict]

@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph structure."""
    total_nodes: int
    total_relationships: int
    node_counts_by_label: dict[str, int]
    relationship_counts_by_type: dict[str, int]
    avg_relationships_per_node: float
    max_node_degree: int
    graph_density: float
    connected_components: int
    largest_component_size: int
```

### 4.3 Analytics Engine

```python
class RKGAnalyticsEngine:
    """Computes analytics and insights from the knowledge graph."""

    def __init__(self, neo4j_client, qdrant_client):
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client

    async def get_research_activity(
        self,
        time_range: Optional[TimeRange] = None
    ) -> ResearchActivityMetrics:
        """Get research activity metrics."""
        time_range = time_range or TimeRange.last_month()

        # Query Qdrant for document counts
        filter_conditions = {
            "must": [{
                "key": "created_at",
                "range": {
                    "gte": time_range.start.isoformat(),
                    "lte": time_range.end.isoformat()
                }
            }]
        }

        # Get total counts
        total_docs = await self.qdrant.count(
            collection_name="research_documents",
            count_filter=filter_conditions
        )

        # Get breakdown by source
        docs_by_source = {}
        for source in ["brave_search", "firecrawl", "session_transcript", "insight"]:
            count = await self.qdrant.count(
                collection_name="research_documents",
                count_filter={
                    "must": [
                        filter_conditions["must"][0],
                        {"key": "source_type", "match": {"value": source}}
                    ]
                }
            )
            docs_by_source[source] = count.count

        # Get breakdown by interface from Neo4j
        docs_by_interface = await self._query_interface_counts(time_range)

        # Get daily activity
        daily_activity = await self._query_daily_activity(time_range)

        # Calculate peak hour
        peak_hour = await self._calculate_peak_hour(time_range)

        # Get session count
        session_count = await self._query_session_count(time_range)

        return ResearchActivityMetrics(
            time_range=time_range,
            total_sessions=session_count,
            total_documents=total_docs.count,
            total_insights=docs_by_source.get("insight", 0),
            documents_by_source=docs_by_source,
            documents_by_interface=docs_by_interface,
            documents_by_project={},  # TODO: Implement
            daily_activity=daily_activity,
            peak_activity_hour=peak_hour,
            avg_documents_per_session=total_docs.count / max(session_count, 1)
        )

    async def get_entity_coverage(
        self,
        time_range: Optional[TimeRange] = None
    ) -> EntityCoverageMetrics:
        """Get entity coverage metrics."""
        async with self.neo4j.session() as session:
            # Total entities
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = await result.single()
            total_entities = record["count"]

            # Entities by type
            result = await session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
            """)
            entities_by_type = {
                record["type"]: record["count"]
                async for record in result
            }

            # Top entities by mentions
            result = await session.run("""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type,
                       coalesce(e.mention_count, 0) as mentions
                ORDER BY mentions DESC
                LIMIT 20
            """)
            top_entities = [
                {"name": r["name"], "type": r["type"], "mentions": r["mentions"]}
                async for r in result
            ]

            # Trending entities (recent growth)
            result = await session.run("""
                MATCH (e:Entity)<-[m:MENTIONS]-(d:Document)
                WHERE d.created_at > datetime() - duration('P7D')
                WITH e, count(m) as recent_mentions
                WHERE recent_mentions > 2
                RETURN e.name as name, e.type as type, recent_mentions
                ORDER BY recent_mentions DESC
                LIMIT 10
            """)
            trending_entities = [
                {"name": r["name"], "type": r["type"], "recent_mentions": r["recent_mentions"]}
                async for r in result
            ]

            # Orphan entities (no relationships)
            result = await session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATES_TO]-() AND NOT (e)<-[:RELATES_TO]-()
                RETURN count(e) as count
            """)
            record = await result.single()
            orphan_entities = record["count"]

            # Average mentions
            result = await session.run("""
                MATCH (e:Entity)
                RETURN avg(coalesce(e.mention_count, 0)) as avg_mentions
            """)
            record = await result.single()
            avg_mentions = record["avg_mentions"] or 0

            return EntityCoverageMetrics(
                total_entities=total_entities,
                entities_by_type=entities_by_type,
                top_entities=top_entities,
                trending_entities=trending_entities,
                orphan_entities=orphan_entities,
                avg_mentions_per_entity=avg_mentions,
                entity_growth_rate=0  # TODO: Calculate
            )

    async def get_knowledge_gaps(self) -> KnowledgeGapMetrics:
        """Identify gaps in the knowledge graph."""
        # Orphan documents (no entities extracted)
        orphan_docs = await self.qdrant.count(
            collection_name="research_documents",
            count_filter={
                "must_not": [{"has_id": []}]  # TODO: Better filter
            }
        )

        async with self.neo4j.session() as session:
            # Unlinked entities
            result = await session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATES_TO]-() AND NOT (e)<-[:RELATES_TO]-()
                RETURN count(e) as count
            """)
            record = await result.single()
            unlinked = record["count"]

            # Stale topics (entities not mentioned in 30 days)
            result = await session.run("""
                MATCH (e:Entity)<-[m:MENTIONS]-(d:Document)
                WITH e, max(d.created_at) as last_mention
                WHERE last_mention < datetime() - duration('P30D')
                RETURN e.name as name, e.type as type, last_mention
                ORDER BY last_mention ASC
                LIMIT 20
            """)
            stale_topics = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "last_mention": r["last_mention"].isoformat() if r["last_mention"] else None
                }
                async for r in result
            ]

            # Low confidence insights
            low_confidence = await self.qdrant.count(
                collection_name="research_documents",
                count_filter={
                    "must": [
                        {"key": "source_type", "match": {"value": "insight"}},
                        {"key": "confidence", "range": {"lt": 0.5}}
                    ]
                }
            )

            return KnowledgeGapMetrics(
                orphan_documents=0,  # TODO: Implement
                unlinked_entities=unlinked,
                stale_topics=stale_topics,
                low_confidence_insights=low_confidence.count,
                missing_entity_types=[],  # TODO: Compare expected vs actual
                coverage_by_topic={}  # TODO: Implement
            )

    async def get_graph_statistics(self) -> GraphStatistics:
        """Get statistics about the knowledge graph structure."""
        async with self.neo4j.session() as session:
            # Node counts by label
            result = await session.run("""
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n)
                    WHERE label IN labels(n)
                    RETURN count(n) as count
                }
                RETURN label, count
            """)
            node_counts = {r["label"]: r["count"] async for r in result}
            total_nodes = sum(node_counts.values())

            # Relationship counts by type
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """)
            rel_counts = {r["type"]: r["count"] async for r in result}
            total_rels = sum(rel_counts.values())

            # Average degree
            result = await session.run("""
                MATCH (n)
                WITH n, size((n)-[]-()) as degree
                RETURN avg(degree) as avg_degree, max(degree) as max_degree
            """)
            record = await result.single()
            avg_degree = record["avg_degree"] or 0
            max_degree = record["max_degree"] or 0

            # Graph density
            # density = 2 * edges / (nodes * (nodes - 1))
            density = 0
            if total_nodes > 1:
                density = 2 * total_rels / (total_nodes * (total_nodes - 1))

            # Connected components (using GDS if available)
            # Simplified: count nodes with no connections
            result = await session.run("""
                MATCH (n)
                WHERE NOT (n)-[]-()
                RETURN count(n) as isolated
            """)
            record = await result.single()
            isolated = record["isolated"]

            return GraphStatistics(
                total_nodes=total_nodes,
                total_relationships=total_rels,
                node_counts_by_label=node_counts,
                relationship_counts_by_type=rel_counts,
                avg_relationships_per_node=avg_degree,
                max_node_degree=max_degree,
                graph_density=density,
                connected_components=1,  # TODO: Implement with GDS
                largest_component_size=total_nodes - isolated
            )

    async def _query_interface_counts(self, time_range: TimeRange) -> dict[str, int]:
        """Query document counts by agentic interface."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (s:Session)-[:CONTAINS]->(d:Document)
                WHERE s.created_at >= $start AND s.created_at <= $end
                RETURN s.interface as interface, count(d) as count
            """, {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            })
            return {r["interface"]: r["count"] async for r in result}

    async def _query_daily_activity(self, time_range: TimeRange) -> list[dict]:
        """Query daily document counts."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (d:Document)
                WHERE d.created_at >= $start AND d.created_at <= $end
                WITH date(d.created_at) as day, count(d) as doc_count
                RETURN day, doc_count
                ORDER BY day
            """, {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            })
            return [
                {"date": str(r["day"]), "documents": r["doc_count"]}
                async for r in result
            ]

    async def _calculate_peak_hour(self, time_range: TimeRange) -> int:
        """Calculate the hour with most activity."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (d:Document)
                WHERE d.created_at >= $start AND d.created_at <= $end
                WITH d.created_at.hour as hour, count(d) as count
                RETURN hour
                ORDER BY count DESC
                LIMIT 1
            """, {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            })
            record = await result.single()
            return record["hour"] if record else 14  # Default to 2 PM

    async def _query_session_count(self, time_range: TimeRange) -> int:
        """Query unique session count."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (s:Session)
                WHERE s.created_at >= $start AND s.created_at <= $end
                RETURN count(s) as count
            """, {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            })
            record = await result.single()
            return record["count"]
```

### 4.4 Dashboard API Endpoints

```python
from fastapi import APIRouter, Query
from datetime import datetime
from typing import Optional

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/activity")
async def get_activity_metrics(
    days: int = Query(default=30, ge=1, le=365),
    analytics: RKGAnalyticsEngine = Depends(get_analytics_engine)
):
    """Get research activity metrics."""
    time_range = TimeRange.last_n_days(days)
    metrics = await analytics.get_research_activity(time_range)
    return metrics

@router.get("/entities")
async def get_entity_metrics(
    analytics: RKGAnalyticsEngine = Depends(get_analytics_engine)
):
    """Get entity coverage metrics."""
    return await analytics.get_entity_coverage()

@router.get("/gaps")
async def get_knowledge_gaps(
    analytics: RKGAnalyticsEngine = Depends(get_analytics_engine)
):
    """Get knowledge gap analysis."""
    return await analytics.get_knowledge_gaps()

@router.get("/graph")
async def get_graph_statistics(
    analytics: RKGAnalyticsEngine = Depends(get_analytics_engine)
):
    """Get graph structure statistics."""
    return await analytics.get_graph_statistics()

@router.get("/summary")
async def get_analytics_summary(
    days: int = Query(default=7, ge=1, le=365),
    analytics: RKGAnalyticsEngine = Depends(get_analytics_engine)
):
    """Get combined analytics summary."""
    time_range = TimeRange.last_n_days(days)

    activity = await analytics.get_research_activity(time_range)
    entities = await analytics.get_entity_coverage()
    gaps = await analytics.get_knowledge_gaps()
    graph = await analytics.get_graph_statistics()

    return {
        "time_range": {
            "start": time_range.start.isoformat(),
            "end": time_range.end.isoformat()
        },
        "highlights": {
            "total_documents": activity.total_documents,
            "total_entities": entities.total_entities,
            "total_relationships": graph.total_relationships,
            "knowledge_gaps": gaps.unlinked_entities + gaps.low_confidence_insights
        },
        "activity": activity,
        "entities": entities,
        "gaps": gaps,
        "graph": graph
    }
```

---

## 5. Contradiction Detection

### 5.1 Research Foundation

Based on RefChecker research:
- Fine-grained detection at knowledge triplet level
- 3-stage pipeline: Claim Extractor → Hallucination Checker → Aggregation
- Classification: Entailment, Contradiction, Neutral (NLI-based)
- Aggregation rules: Strict, Soft, Major

### 5.2 Architecture

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import asyncio

class ContradictionType(Enum):
    DIRECT = "direct"           # Explicit factual contradiction
    STATISTICAL = "statistical" # Conflicting numbers/metrics
    TEMPORAL = "temporal"       # Outdated vs current info
    SCOPE = "scope"             # Different contexts/qualifications
    UNCERTAIN = "uncertain"     # Possible contradiction, needs review

class ContradictionStatus(Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

@dataclass
class KnowledgeTriplet:
    """Fine-grained knowledge unit for contradiction checking.

    Based on RefChecker's triplet extraction approach.
    """
    subject: str
    predicate: str
    object: str
    source_text: str
    source_document_id: str
    confidence: float

    def to_text(self) -> str:
        """Convert triplet to natural language."""
        return f"{self.subject} {self.predicate} {self.object}"

@dataclass
class ContradictionCandidate:
    """A potential contradiction between two insights."""
    insight_a_id: str
    insight_b_id: str
    triplet_a: KnowledgeTriplet
    triplet_b: KnowledgeTriplet
    contradiction_type: ContradictionType
    confidence: float
    explanation: str
    shared_entities: list[str]
    detection_method: str  # "semantic", "entity_overlap", "llm"

@dataclass
class Contradiction:
    """Confirmed contradiction with resolution tracking."""
    id: str
    candidates: list[ContradictionCandidate]
    status: ContradictionStatus
    final_type: ContradictionType
    resolution_explanation: Optional[str] = None
    resolved_by: Optional[str] = None  # user_id or "auto"
    resolved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

class ClaimExtractor:
    """Extract knowledge triplets from text.

    Based on RefChecker's claim extraction approach.
    """

    def __init__(self, rebel_extractor: REBELExtractor):
        self.rebel = rebel_extractor

    async def extract_triplets(
        self,
        text: str,
        document_id: str
    ) -> list[KnowledgeTriplet]:
        """Extract knowledge triplets from text."""
        relationships = await asyncio.to_thread(
            self.rebel.extract_triplets, text
        )

        return [
            KnowledgeTriplet(
                subject=rel.subject.text,
                predicate=rel.predicate.value,
                object=rel.object.text,
                source_text=rel.source_sentence,
                source_document_id=document_id,
                confidence=rel.confidence
            )
            for rel in relationships
        ]

class ContradictionChecker:
    """Check for contradictions using NLI and LLM.

    Based on RefChecker's multi-method checking approach.
    """

    def __init__(
        self,
        qdrant_client,
        neo4j_client,
        voyage_client,
        anthropic_client
    ):
        self.qdrant = qdrant_client
        self.neo4j = neo4j_client
        self.voyage = voyage_client
        self.anthropic = anthropic_client

        # Load NLI model for fast initial filtering
        from transformers import pipeline
        self.nli_model = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device_map="auto"
        )

    async def check_triplet(
        self,
        triplet: KnowledgeTriplet,
        min_similarity: float = 0.7
    ) -> list[ContradictionCandidate]:
        """Check a triplet for contradictions against existing knowledge."""
        candidates = []

        # Step 1: Semantic search for related content
        related_docs = await self._semantic_search(triplet, min_similarity)

        # Step 2: Entity-based filtering
        related_by_entity = await self._entity_search(triplet)

        # Combine and deduplicate
        all_related = self._merge_related(related_docs, related_by_entity)

        # Step 3: NLI-based filtering (fast)
        nli_candidates = await self._nli_filter(triplet, all_related)

        # Step 4: LLM verification (slow but accurate)
        for candidate in nli_candidates:
            if candidate.confidence > 0.6:
                verified = await self._llm_verify(triplet, candidate)
                if verified:
                    candidates.append(verified)

        return candidates

    async def _semantic_search(
        self,
        triplet: KnowledgeTriplet,
        min_similarity: float
    ) -> list[dict]:
        """Find semantically similar content."""
        # Embed the triplet text
        embedding = await self.voyage.embed(
            triplet.to_text(),
            model="voyage-3-large",
            input_type="query"
        )

        # Search Qdrant
        results = await self.qdrant.search(
            collection_name="research_documents",
            query_vector=("dense", embedding),
            limit=20,
            score_threshold=min_similarity
        )

        return [
            {
                "id": r.id,
                "content": r.payload.get("content", ""),
                "score": r.score
            }
            for r in results
            if r.id != triplet.source_document_id  # Exclude source
        ]

    async def _entity_search(self, triplet: KnowledgeTriplet) -> list[dict]:
        """Find content mentioning the same entities."""
        # Extract entities from triplet
        entities = [triplet.subject, triplet.object]

        async with self.neo4j.session() as session:
            result = await session.run("""
                UNWIND $entities as entity_name
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(entity_name)
                   OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower(entity_name))
                MATCH (e)<-[:MENTIONS]-(d:Document)
                WHERE d.id <> $source_id
                RETURN DISTINCT d.id as id, d.content as content
                LIMIT 20
            """, {
                "entities": entities,
                "source_id": triplet.source_document_id
            })

            return [
                {"id": r["id"], "content": r["content"], "score": 0.8}
                async for r in result
            ]

    def _merge_related(
        self,
        semantic: list[dict],
        entity: list[dict]
    ) -> list[dict]:
        """Merge and deduplicate related documents."""
        seen = {}
        for doc in semantic + entity:
            doc_id = doc["id"]
            if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                seen[doc_id] = doc
        return list(seen.values())

    async def _nli_filter(
        self,
        triplet: KnowledgeTriplet,
        related: list[dict]
    ) -> list[ContradictionCandidate]:
        """Use NLI model for fast contradiction filtering."""
        candidates = []
        triplet_text = triplet.to_text()

        for doc in related:
            # Run NLI inference
            result = await asyncio.to_thread(
                self.nli_model,
                f"{doc['content'][:500]} [SEP] {triplet_text}",
                return_all_scores=True
            )

            # Check for contradiction label
            scores = {r["label"]: r["score"] for r in result[0]}

            if scores.get("CONTRADICTION", 0) > 0.5:
                candidates.append(ContradictionCandidate(
                    insight_a_id=triplet.source_document_id,
                    insight_b_id=doc["id"],
                    triplet_a=triplet,
                    triplet_b=KnowledgeTriplet(
                        subject="",
                        predicate="",
                        object="",
                        source_text=doc["content"][:500],
                        source_document_id=doc["id"],
                        confidence=doc["score"]
                    ),
                    contradiction_type=ContradictionType.UNCERTAIN,
                    confidence=scores["CONTRADICTION"],
                    explanation="NLI model detected potential contradiction",
                    shared_entities=[],
                    detection_method="nli"
                ))

        return candidates

    async def _llm_verify(
        self,
        triplet: KnowledgeTriplet,
        candidate: ContradictionCandidate
    ) -> Optional[ContradictionCandidate]:
        """Use LLM for detailed contradiction verification."""
        prompt = f"""Analyze whether these two statements contradict each other.

Statement A (from document {triplet.source_document_id}):
"{triplet.to_text()}"
Context: "{triplet.source_text[:300]}"

Statement B (from document {candidate.insight_b_id}):
"{candidate.triplet_b.source_text[:500]}"

Respond with a JSON object:
{{
    "is_contradiction": true/false,
    "contradiction_type": "direct" | "statistical" | "temporal" | "scope" | "uncertain",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of why these do or don't contradict",
    "shared_entities": ["list", "of", "shared", "entities"]
}}

Only respond with the JSON object, no other text."""

        response = await self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            import json
            result = json.loads(response.content[0].text)

            if result.get("is_contradiction"):
                return ContradictionCandidate(
                    insight_a_id=candidate.insight_a_id,
                    insight_b_id=candidate.insight_b_id,
                    triplet_a=triplet,
                    triplet_b=candidate.triplet_b,
                    contradiction_type=ContradictionType(result["contradiction_type"]),
                    confidence=result["confidence"],
                    explanation=result["explanation"],
                    shared_entities=result.get("shared_entities", []),
                    detection_method="llm"
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

class ContradictionAggregator:
    """Aggregate and store contradictions.

    Based on RefChecker's aggregation rules:
    - Strict: Any contradiction → flagged
    - Soft: Ratio-based threshold
    - Major: Voting across multiple checks
    """

    def __init__(self, neo4j_client, aggregation_mode: str = "soft"):
        self.neo4j = neo4j_client
        self.aggregation_mode = aggregation_mode
        self.soft_threshold = 0.5  # For soft mode

    async def aggregate_and_store(
        self,
        candidates: list[ContradictionCandidate]
    ) -> list[Contradiction]:
        """Aggregate candidates and store confirmed contradictions."""
        if not candidates:
            return []

        # Group by document pair
        by_pair = {}
        for c in candidates:
            key = tuple(sorted([c.insight_a_id, c.insight_b_id]))
            if key not in by_pair:
                by_pair[key] = []
            by_pair[key].append(c)

        contradictions = []

        for pair, pair_candidates in by_pair.items():
            if self._should_flag(pair_candidates):
                # Select best candidate
                best = max(pair_candidates, key=lambda x: x.confidence)

                contradiction = Contradiction(
                    id=hashlib.sha256(
                        f"{pair[0]}:{pair[1]}".encode()
                    ).hexdigest()[:16],
                    candidates=pair_candidates,
                    status=ContradictionStatus.DETECTED,
                    final_type=best.contradiction_type
                )

                # Store in Neo4j
                await self._store_contradiction(contradiction)
                contradictions.append(contradiction)

        return contradictions

    def _should_flag(self, candidates: list[ContradictionCandidate]) -> bool:
        """Determine if candidates should be flagged as contradiction."""
        if self.aggregation_mode == "strict":
            # Any contradiction → flag
            return any(c.confidence > 0.5 for c in candidates)

        elif self.aggregation_mode == "soft":
            # Ratio-based
            high_confidence = sum(1 for c in candidates if c.confidence > 0.7)
            return high_confidence / len(candidates) > self.soft_threshold

        elif self.aggregation_mode == "major":
            # Majority voting
            high_confidence = sum(1 for c in candidates if c.confidence > 0.6)
            return high_confidence > len(candidates) / 2

        return False

    async def _store_contradiction(self, contradiction: Contradiction):
        """Store contradiction in Neo4j."""
        async with self.neo4j.session() as session:
            # Get the best candidate for primary details
            best = max(contradiction.candidates, key=lambda x: x.confidence)

            await session.run("""
                MATCH (a:Document {id: $doc_a_id})
                MATCH (b:Document {id: $doc_b_id})

                MERGE (a)-[r:CONTRADICTS]->(b)
                SET r.id = $id,
                    r.type = $type,
                    r.confidence = $confidence,
                    r.explanation = $explanation,
                    r.status = $status,
                    r.created_at = datetime(),
                    r.shared_entities = $shared_entities
            """, {
                "doc_a_id": best.insight_a_id,
                "doc_b_id": best.insight_b_id,
                "id": contradiction.id,
                "type": contradiction.final_type.value,
                "confidence": best.confidence,
                "explanation": best.explanation,
                "status": contradiction.status.value,
                "shared_entities": best.shared_entities
            })

class ContradictionDetectionPipeline:
    """Full pipeline for contradiction detection.

    Orchestrates:
    1. Triplet extraction
    2. Contradiction checking
    3. Aggregation and storage
    """

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        checker: ContradictionChecker,
        aggregator: ContradictionAggregator
    ):
        self.extractor = claim_extractor
        self.checker = checker
        self.aggregator = aggregator

    async def process_document(
        self,
        document_id: str,
        content: str
    ) -> list[Contradiction]:
        """Process a document for contradictions."""
        # Extract triplets
        triplets = await self.extractor.extract_triplets(content, document_id)

        # Check each triplet
        all_candidates = []
        for triplet in triplets:
            candidates = await self.checker.check_triplet(triplet)
            all_candidates.extend(candidates)

        # Aggregate and store
        return await self.aggregator.aggregate_and_store(all_candidates)

    async def get_contradictions(
        self,
        status: Optional[ContradictionStatus] = None,
        limit: int = 50
    ) -> list[dict]:
        """Get contradictions from the graph."""
        async with self.aggregator.neo4j.session() as session:
            where_clause = ""
            if status:
                where_clause = f"WHERE r.status = '{status.value}'"

            result = await session.run(f"""
                MATCH (a:Document)-[r:CONTRADICTS]->(b:Document)
                {where_clause}
                RETURN {{
                    id: r.id,
                    type: r.type,
                    confidence: r.confidence,
                    explanation: r.explanation,
                    status: r.status,
                    doc_a: {{id: a.id, title: a.title}},
                    doc_b: {{id: b.id, title: b.title}},
                    shared_entities: r.shared_entities,
                    created_at: r.created_at
                }} AS contradiction
                ORDER BY r.created_at DESC
                LIMIT $limit
            """, {"limit": limit})

            return [r["contradiction"] async for r in result]

    async def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str,
        resolved_by: str = "user"
    ):
        """Mark a contradiction as resolved."""
        async with self.aggregator.neo4j.session() as session:
            await session.run("""
                MATCH ()-[r:CONTRADICTS {id: $id}]->()
                SET r.status = 'resolved',
                    r.resolution_explanation = $resolution,
                    r.resolved_by = $resolved_by,
                    r.resolved_at = datetime()
            """, {
                "id": contradiction_id,
                "resolution": resolution,
                "resolved_by": resolved_by
            })
```

---

## 6. Integration Patterns

### 6.1 MCP Tool Definitions

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

# Entity Extraction Tool
ENTITY_EXTRACT_TOOL = Tool(
    name="rkg_extract_entities",
    description="""Extract entities and relationships from text.

    Extracts: people, organizations, technologies, concepts, products,
    locations, events, metrics, APIs, frameworks.

    Also extracts relationships between entities using REBEL model.""",
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract entities from"
            },
            "document_id": {
                "type": "string",
                "description": "Optional document ID for linking"
            },
            "extract_relationships": {
                "type": "boolean",
                "default": True,
                "description": "Whether to extract relationships"
            }
        },
        "required": ["text"]
    }
)

# Sync Status Tool
SYNC_STATUS_TOOL = Tool(
    name="rkg_sync_status",
    description="Get status of incremental session synchronization",
    inputSchema={
        "type": "object",
        "properties": {}
    }
)

# Export Tool
EXPORT_TOOL = Tool(
    name="rkg_export",
    description="""Export knowledge graph data.

    Formats: json, jsonl, markdown, cypher, csv, graphml

    Filters: projects, sessions, date range, entity types""",
    inputSchema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "jsonl", "markdown", "cypher", "csv", "graphml"],
                "default": "json"
            },
            "projects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by project names"
            },
            "sessions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by session IDs"
            },
            "include_embeddings": {
                "type": "boolean",
                "default": False
            }
        }
    }
)

# Analytics Tool
ANALYTICS_TOOL = Tool(
    name="rkg_analytics",
    description="Get analytics and insights about the knowledge graph",
    inputSchema={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["activity", "entities", "gaps", "graph", "summary"],
                "default": "summary"
            },
            "days": {
                "type": "integer",
                "default": 30,
                "description": "Time range in days"
            }
        }
    }
)

# Contradiction Tool
CONTRADICTIONS_TOOL = Tool(
    name="rkg_contradictions",
    description="""Get or check contradictions in the knowledge graph.

    Actions:
    - list: Get existing contradictions
    - check: Check text for contradictions against existing knowledge
    - resolve: Mark a contradiction as resolved""",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "check", "resolve"],
                "default": "list"
            },
            "text": {
                "type": "string",
                "description": "Text to check for contradictions (for 'check' action)"
            },
            "contradiction_id": {
                "type": "string",
                "description": "Contradiction ID (for 'resolve' action)"
            },
            "resolution": {
                "type": "string",
                "description": "Resolution explanation (for 'resolve' action)"
            },
            "status_filter": {
                "type": "string",
                "enum": ["detected", "confirmed", "resolved", "false_positive"]
            }
        }
    }
)
```

### 6.2 Hybrid Search Integration

```python
class HybridSearchEngine:
    """Combines Qdrant vector search with Neo4j graph traversal.

    Based on research on Qdrant sparse vectors and Neo4j + vector integration.
    """

    def __init__(self, qdrant_client, neo4j_client, voyage_client):
        self.qdrant = qdrant_client
        self.neo4j = neo4j_client
        self.voyage = voyage_client

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_reranking: bool = True,
        expand_by_entities: bool = True
    ) -> list[dict]:
        """Perform hybrid search combining dense, sparse, and graph traversal."""

        # Step 1: Generate dense embedding
        dense_vector = await self.voyage.embed(
            query,
            model="voyage-3-large",
            input_type="query"
        )

        # Step 2: Generate sparse vector (BM25-style)
        sparse_vector = await self._generate_sparse_vector(query)

        # Step 3: Prefetch from both indexes
        dense_results = await self.qdrant.search(
            collection_name="research_documents",
            query_vector=("dense", dense_vector),
            limit=limit * 3,  # Overfetch for fusion
            with_payload=True
        )

        sparse_results = await self.qdrant.search(
            collection_name="research_documents",
            query_vector=("sparse", sparse_vector),
            limit=limit * 3,
            with_payload=True
        )

        # Step 4: Reciprocal Rank Fusion (RRF)
        fused = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight,
            sparse_weight
        )

        # Step 5: Entity expansion (optional)
        if expand_by_entities:
            fused = await self._expand_by_entities(fused, query)

        # Step 6: Reranking (optional)
        if use_reranking:
            fused = await self._rerank(query, fused)

        return fused[:limit]

    async def _generate_sparse_vector(self, text: str) -> dict:
        """Generate sparse vector using BM25-style tokenization.

        In production, use SPLADE or learned sparse encoder.
        """
        # Simple BM25-style sparse vector
        import re
        from collections import Counter

        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Count term frequencies
        tf = Counter(tokens)

        # Create sparse vector (indices = token hashes, values = TF)
        # In production, use proper vocabulary mapping
        indices = [hash(t) % 30000 for t in tf.keys()]
        values = list(tf.values())

        return {"indices": indices, "values": values}

    def _reciprocal_rank_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        dense_weight: float,
        sparse_weight: float,
        k: int = 60
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion.

        RRF Score = Σ (weight / (k + rank))
        """
        scores = {}
        payloads = {}

        # Score dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
            payloads[doc_id] = result.payload

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
            payloads[doc_id] = result.payload

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {
                "id": doc_id,
                "score": scores[doc_id],
                "payload": payloads[doc_id]
            }
            for doc_id in sorted_ids
        ]

    async def _expand_by_entities(
        self,
        results: list[dict],
        query: str
    ) -> list[dict]:
        """Expand results by finding related documents through entity graph."""
        # Extract entity IDs from top results
        top_doc_ids = [r["id"] for r in results[:5]]

        async with self.neo4j.session() as session:
            # Find documents connected through shared entities
            result = await session.run("""
                UNWIND $doc_ids as doc_id
                MATCH (d:Document {id: doc_id})-[:MENTIONS]->(e:Entity)
                    <-[:MENTIONS]-(related:Document)
                WHERE related.id NOT IN $doc_ids
                WITH related, count(DISTINCT e) as shared_entities
                WHERE shared_entities >= 2
                RETURN related.id as id, shared_entities
                ORDER BY shared_entities DESC
                LIMIT 10
            """, {"doc_ids": top_doc_ids})

            # Add graph-expanded results with lower base score
            expanded_ids = set()
            async for record in result:
                if record["id"] not in {r["id"] for r in results}:
                    expanded_ids.add(record["id"])

            if expanded_ids:
                # Fetch payloads for expanded results
                expanded_points = await self.qdrant.retrieve(
                    collection_name="research_documents",
                    ids=list(expanded_ids),
                    with_payload=True
                )

                for point in expanded_points:
                    results.append({
                        "id": point.id,
                        "score": 0.3,  # Lower score for graph-expanded
                        "payload": point.payload,
                        "expansion": "entity_graph"
                    })

        return results

    async def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using Voyage reranker."""
        if not results:
            return results

        documents = [r["payload"].get("content", "")[:2000] for r in results]

        reranked = await self.voyage.rerank(
            query=query,
            documents=documents,
            model="rerank-2.5",
            top_k=len(results)
        )

        # Reorder results based on reranking
        reranked_results = []
        for item in reranked.results:
            result = results[item.index]
            result["rerank_score"] = item.relevance_score
            result["score"] = (result["score"] + item.relevance_score) / 2  # Blend scores
            reranked_results.append(result)

        return sorted(reranked_results, key=lambda x: x["score"], reverse=True)
```

---

## 7. Implementation Priority & Dependencies

### 7.1 Recommended Implementation Order

```
Phase 1: Foundation (Week 1-2)
├── Entity Extraction Pipeline
│   ├── GLiNER setup and integration
│   ├── spaCy transformer model
│   ├── Entity deduplication logic
│   └── Neo4j schema and indexes
│
└── Incremental Sync
    ├── Watchdog file monitoring
    ├── State persistence
    ├── Debounce/rate limiting
    └── Session file parsing

Phase 2: Storage & Retrieval (Week 3-4)
├── Hybrid Search
│   ├── Sparse vector generation
│   ├── RRF fusion implementation
│   └── Entity-based expansion
│
└── REBEL Integration
    ├── Model setup
    ├── Triplet parsing
    └── Relationship graph population

Phase 3: Analysis (Week 5-6)
├── Analytics Engine
│   ├── Activity metrics
│   ├── Entity coverage
│   ├── Knowledge gaps
│   └── Graph statistics
│
└── Export/Import
    ├── JSON/JSONL formats
    ├── GraphML for visualization
    ├── Cypher for Neo4j
    └── Import with merge strategies

Phase 4: Advanced (Week 7-8)
├── Contradiction Detection
│   ├── NLI model setup
│   ├── Claim extraction
│   ├── LLM verification
│   └── Resolution tracking
│
└── Dashboard API
    ├── FastAPI endpoints
    ├── WebSocket for real-time
    └── Swift app integration
```

### 7.2 Dependency Graph

```
                    ┌─────────────────┐
                    │   GLiNER NER    │
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌─────────┐           ┌─────────────┐          ┌─────────────┐
│ Entity  │           │   REBEL     │          │ Coreference │
│ Pipeline│◀──────────│ Extraction  │──────────│ Resolution  │
└────┬────┘           └─────────────┘          └─────────────┘
     │
     │  ┌─────────────┐
     ├──│ Incremental │
     │  │    Sync     │
     │  └─────────────┘
     │
     ▼
┌─────────────────┐     ┌─────────────────┐
│  Neo4j Graph    │◀────│  Contradiction  │
│   Population    │     │   Detection     │
└────────┬────────┘     └─────────────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐          ┌─────────────┐
│    Analytics    │          │   Export    │
│     Engine      │          │   System    │
└─────────────────┘          └─────────────┘
         │                          │
         └──────────┬───────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │  Dashboard API  │
            └─────────────────┘
```

### 7.3 Resource Requirements

| Component | CPU | Memory | GPU | Storage |
|-----------|-----|--------|-----|---------|
| GLiNER | Medium | 2GB | Optional (2x faster) | 500MB |
| REBEL | High | 4GB | Recommended | 1.5GB |
| spaCy (trf) | Medium | 1GB | Optional | 500MB |
| NLI Model | Medium | 1GB | Optional | 400MB |
| Coreference | Low | 500MB | No | 200MB |

**Total Recommended**: 4 CPU cores, 16GB RAM, Optional GPU (RTX 3060+)

---

## Appendix A: Neo4j Schema Reference

```cypher
// Full schema for missing features

// Entity nodes
CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE;

CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases];

// Sync state
CREATE CONSTRAINT sync_state_path IF NOT EXISTS
FOR (s:SyncState) REQUIRE s.path IS UNIQUE;

// Contradictions
CREATE INDEX contradiction_status IF NOT EXISTS FOR ()-[c:CONTRADICTS]-() ON (c.status);
CREATE INDEX contradiction_type IF NOT EXISTS FOR ()-[c:CONTRADICTS]-() ON (c.type);

// Analytics indexes
CREATE INDEX document_created IF NOT EXISTS FOR (d:Document) ON (d.created_at);
CREATE INDEX session_created IF NOT EXISTS FOR (s:Session) ON (s.created_at);
```

---

*Document Version: 1.0.0*
*Based on research conducted December 2024*
*Sources: Neo4j Information Extraction Pipeline, REBEL, RefChecker, Watchdog, Neo4j APOC, Qdrant Sparse Vectors*
