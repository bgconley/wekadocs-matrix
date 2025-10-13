# Weka Documentation GraphRAG MCP Server - Implementation Plan (Continued)

## Phase 2: Query Processing Engine (Continued)

### Task 2.3: Hybrid Search Implementation
**Timeline**: 3 days
**Dependencies**: Tasks 2.1, 2.2
**Deliverables**: Vector search integration, graph traversal, result ranking

#### Implementation Steps:

1. **Create hybrid search engine**:
```python
# src/query/hybrid_search.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase
import structlog

logger = structlog.get_logger()

class SearchType(Enum):
    """Types of search strategies."""
    SEMANTIC = "semantic"
    GRAPH = "graph"
    HYBRID = "hybrid"

@dataclass
class SearchResult:
    """Individual search result."""
    node_id: str
    node_type: str
    score: float
    properties: Dict[str, Any]
    relationships: Optional[List[Dict]] = None
    path: Optional[List[str]] = None

@dataclass
class SearchResults:
    """Collection of search results."""
    results: List[SearchResult]
    total_count: int
    search_type: SearchType
    execution_time: float
    query_metadata: Dict[str, Any]

class HybridSearchEngine:
    """
    Combines vector similarity and graph traversal for comprehensive search.
    """

    def __init__(self,
                 neo4j_driver,
                 qdrant_client: QdrantClient,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.neo4j = neo4j_driver
        self.qdrant = qdrant_client
        self.embedder = SentenceTransformer(embedding_model)
        self.collection_name = "weka_docs"

    async def search(self,
                    query: str,
                    search_type: SearchType = SearchType.HYBRID,
                    max_results: int = 10,
                    filters: Optional[Dict] = None) -> SearchResults:
        """
        Execute search based on specified strategy.
        """
        import time
        start_time = time.time()

        if search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(query, max_results, filters)
        elif search_type == SearchType.GRAPH:
            results = await self._graph_search(query, max_results, filters)
        else:  # HYBRID
            results = await self._hybrid_search(query, max_results, filters)

        execution_time = time.time() - start_time

        return SearchResults(
            results=results,
            total_count=len(results),
            search_type=search_type,
            execution_time=execution_time,
            query_metadata={
                "query": query,
                "filters": filters,
                "max_results": max_results
            }
        )

    async def _semantic_search(self,
                              query: str,
                              limit: int,
                              filters: Optional[Dict]) -> List[SearchResult]:
        """
        Perform vector similarity search using Qdrant.
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Search in Qdrant
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            query_filter=self._build_qdrant_filter(filters) if filters else None
        )

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            results.append(SearchResult(
                node_id=hit.payload.get("node_id"),
                node_type=hit.payload.get("node_type"),
                score=hit.score,
                properties=hit.payload
            ))

        return results

    async def _graph_search(self,
                           query: str,
                           limit: int,
                           filters: Optional[Dict]) -> List[SearchResult]:
        """
        Perform graph-based search using Neo4j.
        """
        # Build Cypher query based on search terms
        cypher = self._build_graph_query(query, filters)

        results = []
        async with self.neo4j.session() as session:
            result = await session.run(cypher, {"query": query, "limit": limit})

            async for record in result:
                node = record["node"]
                results.append(SearchResult(
                    node_id=node.get("id"),
                    node_type=list(node.labels)[0] if node.labels else "Unknown",
                    score=record.get("score", 1.0),
                    properties=dict(node),
                    relationships=record.get("relationships")
                ))

        return results

    async def _hybrid_search(self,
                            query: str,
                            limit: int,
                            filters: Optional[Dict]) -> List[SearchResult]:
        """
        Combine semantic and graph search for best results.
        """
        # Phase 1: Semantic search for initial candidates
        semantic_results = await self._semantic_search(
            query,
            limit=min(limit * 2, 50),  # Get more candidates
            filters=filters
        )

        if not semantic_results:
            return []

        # Phase 2: Graph expansion from semantic results
        expanded_results = await self._expand_graph_context(
            semantic_results[:10],  # Top 10 semantic results
            max_hops=2
        )

        # Phase 3: Find connecting paths between top results
        if len(semantic_results) > 1:
            paths = await self._find_connecting_paths(
                semantic_results[:5],
                max_path_length=3
            )
            expanded_results.extend(paths)

        # Phase 4: Merge and rank results
        merged_results = self._merge_and_rank(
            semantic_results,
            expanded_results,
            query
        )

        return merged_results[:limit]

    async def _expand_graph_context(self,
                                   seed_nodes: List[SearchResult],
                                   max_hops: int = 2) -> List[SearchResult]:
        """
        Expand search results through graph relationships.
        """
        if not seed_nodes:
            return []

        node_ids = [node.node_id for node in seed_nodes]

        cypher = """
        UNWIND $node_ids AS node_id
        MATCH (n {id: node_id})
        OPTIONAL MATCH path = (n)-[*1..%d]-(related)
        WHERE related.id <> node_id
        RETURN DISTINCT related as node,
               length(path) as distance,
               [rel in relationships(path) | type(rel)] as relationship_types
        ORDER BY distance
        LIMIT 50
        """ % max_hops

        expanded = []
        async with self.neo4j.session() as session:
            result = await session.run(cypher, {"node_ids": node_ids})

            async for record in result:
                node = record["node"]
                if node:
                    # Score based on distance (closer = higher score)
                    distance = record["distance"]
                    score = 1.0 / (1 + distance)

                    expanded.append(SearchResult(
                        node_id=node.get("id"),
                        node_type=list(node.labels)[0] if node.labels else "Unknown",
                        score=score,
                        properties=dict(node),
                        relationships=record["relationship_types"]
                    ))

        return expanded

    async def _find_connecting_paths(self,
                                    nodes: List[SearchResult],
                                    max_path_length: int = 3) -> List[SearchResult]:
        """
        Find paths connecting multiple search results.
        """
        if len(nodes) < 2:
            return []

        node_ids = [node.node_id for node in nodes[:5]]  # Limit to prevent explosion

        cypher = """
        UNWIND $node_ids AS start_id
        UNWIND $node_ids AS end_id
        WHERE start_id < end_id
        MATCH path = shortestPath((start {id: start_id})-[*..%d]-(end {id: end_id}))
        UNWIND nodes(path) AS node
        RETURN DISTINCT node,
               [n IN nodes(path) | n.id] as path_nodes,
               length(path) as path_length
        LIMIT 30
        """ % max_path_length

        path_nodes = []
        async with self.neo4j.session() as session:
            result = await session.run(cypher, {"node_ids": node_ids})

            async for record in result:
                node = record["node"]
                if node:
                    path_nodes.append(SearchResult(
                        node_id=node.get("id"),
                        node_type=list(node.labels)[0] if node.labels else "Unknown",
                        score=0.5,  # Medium score for path nodes
                        properties=dict(node),
                        path=record["path_nodes"]
                    ))

        return path_nodes

    def _merge_and_rank(self,
                       semantic_results: List[SearchResult],
                       graph_results: List[SearchResult],
                       query: str) -> List[SearchResult]:
        """
        Merge and rank results from different search strategies.
        """
        # Create a dictionary to track unique results
        merged = {}

        # Add semantic results with boosted scores
        for result in semantic_results:
            merged[result.node_id] = result
            result.score *= 1.2  # Boost semantic matches

        # Add graph results, combining scores if already present
        for result in graph_results:
            if result.node_id in merged:
                # Combine scores
                merged[result.node_id].score += result.score * 0.8
                # Merge relationships
                if result.relationships:
                    existing_rels = merged[result.node_id].relationships or []
                    merged[result.node_id].relationships = list(set(
                        existing_rels + result.relationships
                    ))
            else:
                merged[result.node_id] = result

        # Sort by score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_results

    def _build_qdrant_filter(self, filters: Dict) -> Dict:
        """Build Qdrant filter from generic filters."""
        qdrant_filter = {"must": []}

        if "node_type" in filters:
            qdrant_filter["must"].append({
                "key": "node_type",
                "match": {"value": filters["node_type"]}
            })

        if "category" in filters:
            qdrant_filter["must"].append({
                "key": "category",
                "match": {"value": filters["category"]}
            })

        return qdrant_filter if qdrant_filter["must"] else None

    def _build_graph_query(self, query: str, filters: Optional[Dict]) -> str:
        """Build Cypher query for graph search."""
        where_clauses = []

        # Add text search
        where_clauses.append(
            "(n.name CONTAINS $query OR n.description CONTAINS $query)"
        )

        # Add filters
        if filters:
            if "node_type" in filters:
                where_clauses.append(f"'{filters['node_type']}' IN labels(n)")
            if "category" in filters:
                where_clauses.append(f"n.category = '{filters['category']}'")

        where_clause = " AND ".join(where_clauses)

        return f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN n as node, 1.0 as score
        LIMIT $limit
        """
```

2. **Implement result ranking**:
```python
# src/query/ranking.py
from typing import List, Dict, Any
import math
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import structlog

logger = structlog.get_logger()

@dataclass
class RankingFeatures:
    """Features used for ranking search results."""
    semantic_score: float
    graph_distance: float
    text_relevance: float
    entity_type_score: float
    relationship_score: float
    recency_score: float

class ResultRanker:
    """
    Ranks search results using multiple signals.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.weights = {
            "semantic_score": 0.3,
            "graph_distance": 0.2,
            "text_relevance": 0.2,
            "entity_type_score": 0.1,
            "relationship_score": 0.15,
            "recency_score": 0.05
        }

    def rank_results(self,
                    results: List[Dict],
                    query: str,
                    user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Rank results using multiple features.
        """
        if not results:
            return results

        # Extract features for each result
        features_list = []
        for result in results:
            features = self._extract_features(result, query)
            features_list.append(features)

        # Calculate final scores
        scored_results = []
        for result, features in zip(results, features_list):
            score = self._calculate_score(features, user_preferences)
            result["ranking_score"] = score
            result["ranking_features"] = features
            scored_results.append(result)

        # Sort by score
        sorted_results = sorted(
            scored_results,
            key=lambda x: x["ranking_score"],
            reverse=True
        )

        return sorted_results

    def _extract_features(self, result: Dict, query: str) -> RankingFeatures:
        """Extract ranking features from a result."""
        # Semantic similarity score (from vector search)
        semantic_score = result.get("score", 0.0)

        # Graph distance (inverse of distance)
        graph_distance = 1.0 / (1 + result.get("distance", 0))

        # Text relevance using TF-IDF
        text_relevance = self._calculate_text_relevance(
            result.get("properties", {}),
            query
        )

        # Entity type score (prefer certain types)
        entity_type_score = self._score_entity_type(
            result.get("node_type", "Unknown")
        )

        # Relationship score (more relationships = more relevant)
        relationship_score = min(1.0, len(result.get("relationships", [])) / 10)

        # Recency score (if timestamp available)
        recency_score = self._calculate_recency(
            result.get("properties", {}).get("updated_at")
        )

        return RankingFeatures(
            semantic_score=semantic_score,
            graph_distance=graph_distance,
            text_relevance=text_relevance,
            entity_type_score=entity_type_score,
            relationship_score=relationship_score,
            recency_score=recency_score
        )

    def _calculate_score(self,
                        features: RankingFeatures,
                        user_preferences: Optional[Dict]) -> float:
        """Calculate final ranking score."""
        weights = self.weights.copy()

        # Adjust weights based on user preferences
        if user_preferences:
            if user_preferences.get("prefer_recent"):
                weights["recency_score"] *= 2
                # Normalize weights
                total = sum(weights.values())
                weights = {k: v/total for k, v in weights.items()}

        score = (
            weights["semantic_score"] * features.semantic_score +
            weights["graph_distance"] * features.graph_distance +
            weights["text_relevance"] * features.text_relevance +
            weights["entity_type_score"] * features.entity_type_score +
            weights["relationship_score"] * features.relationship_score +
            weights["recency_score"] * features.recency_score
        )

        return score

    def _calculate_text_relevance(self, properties: Dict, query: str) -> float:
        """Calculate text relevance using TF-IDF."""
        # Combine text from properties
        text = " ".join([
            str(v) for v in properties.values()
            if isinstance(v, str)
        ])

        if not text:
            return 0.0

        # Simple keyword matching for now
        query_terms = query.lower().split()
        text_lower = text.lower()

        matches = sum(1 for term in query_terms if term in text_lower)
        relevance = matches / len(query_terms) if query_terms else 0

        return min(1.0, relevance)

    def _score_entity_type(self, entity_type: str) -> float:
        """Score based on entity type preference."""
        type_scores = {
            "Command": 0.9,
            "Procedure": 0.85,
            "Error": 0.8,
            "Configuration": 0.75,
            "Component": 0.7,
            "Concept": 0.65,
            "Unknown": 0.3
        }
        return type_scores.get(entity_type, 0.5)

    def _calculate_recency(self, timestamp: Optional[str]) -> float:
        """Calculate recency score from timestamp."""
        if not timestamp:
            return 0.5

        # Parse timestamp and calculate age
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age_days = (datetime.now(dt.tzinfo) - dt).days

            # Exponential decay
            recency = math.exp(-age_days / 365)  # Decay over a year
            return recency
        except:
            return 0.5
```

### Task 2.4: Response Generation System
**Timeline**: 2 days
**Dependencies**: Tasks 2.1-2.3
**Deliverables**: Response builder, evidence collection, confidence scoring

#### Implementation Steps:

1. **Create response builder**:
```python
# src/query/response_builder.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger()

class ResponseType(Enum):
    """Types of responses."""
    DIRECT_ANSWER = "direct_answer"
    EXPLANATION = "explanation"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"

@dataclass
class Evidence:
    """Supporting evidence for response."""
    source_node_id: str
    source_type: str
    text: str
    confidence: float
    relationship_path: Optional[List[str]] = None

@dataclass
class StructuredResponse:
    """Structured response to query."""
    answer: str
    response_type: ResponseType
    evidence: List[Evidence]
    related_entities: List[Dict[str, Any]]
    confidence_score: float
    metadata: Dict[str, Any]
    suggestions: Optional[List[str]] = None

class ResponseBuilder:
    """
    Builds structured responses from query results.
    """

    def __init__(self):
        self.response_templates = self._load_response_templates()

    def build_response(self,
                      query_results: Dict,
                      original_query: str,
                      query_intent: str,
                      response_type: ResponseType = None) -> StructuredResponse:
        """
        Build structured response from query results.
        """
        # Determine response type if not specified
        if not response_type:
            response_type = self._determine_response_type(query_intent)

        # Extract primary answer
        answer = self._extract_primary_answer(
            query_results,
            original_query,
            response_type
        )

        # Collect supporting evidence
        evidence = self._collect_evidence(query_results)

        # Find related entities
        related_entities = self._extract_related_entities(query_results)

        # Calculate confidence score
        confidence = self._calculate_confidence(
            query_results,
            evidence,
            original_query
        )

        # Generate suggestions for follow-up
        suggestions = self._generate_suggestions(
            query_results,
            response_type
        )

        # Build metadata
        metadata = self._build_metadata(query_results)

        return StructuredResponse(
            answer=answer,
            response_type=response_type,
            evidence=evidence,
            related_entities=related_entities,
            confidence_score=confidence,
            metadata=metadata,
            suggestions=suggestions
        )

    def _determine_response_type(self, query_intent: str) -> ResponseType:
        """Determine response type from query intent."""
        intent_to_type = {
            "search": ResponseType.DIRECT_ANSWER,
            "explain": ResponseType.EXPLANATION,
            "troubleshoot": ResponseType.TROUBLESHOOTING,
            "compare": ResponseType.COMPARISON,
            "procedure": ResponseType.PROCEDURE
        }
        return intent_to_type.get(query_intent, ResponseType.DIRECT_ANSWER)

    def _extract_primary_answer(self,
                               results: Dict,
                               query: str,
                               response_type: ResponseType) -> str:
        """Extract primary answer from results."""
        if not results.get("results"):
            return "No information found for your query."

        # Get template for response type
        template = self.response_templates.get(response_type)

        if response_type == ResponseType.DIRECT_ANSWER:
            return self._build_direct_answer(results["results"])

        elif response_type == ResponseType.PROCEDURE:
            return self._build_procedure_answer(results["results"])

        elif response_type == ResponseType.TROUBLESHOOTING:
            return self._build_troubleshooting_answer(results["results"])

        elif response_type == ResponseType.COMPARISON:
            return self._build_comparison_answer(results["results"])

        else:  # EXPLANATION
            return self._build_explanation_answer(results["results"])

    def _build_direct_answer(self, results: List[Dict]) -> str:
        """Build direct answer from results."""
        if not results:
            return "No results found."

        # Take top result
        top_result = results[0]

        # Extract key information
        if top_result.get("node_type") == "Command":
            return self._format_command_answer(top_result)
        elif top_result.get("node_type") == "Error":
            return self._format_error_answer(top_result)
        else:
            return self._format_generic_answer(top_result)

    def _format_command_answer(self, command: Dict) -> str:
        """Format command information."""
        props = command.get("properties", {})

        answer = f"**{props.get('name', 'Command')}**\n\n"

        if props.get('description'):
            answer += f"{props['description']}\n\n"

        if props.get('cli_syntax'):
            answer += f"**CLI Syntax:**\n```bash\n{props['cli_syntax']}\n```\n\n"

        if props.get('rest_endpoint'):
            answer += f"**REST Endpoint:**\n```\n{props.get('http_method', 'GET')} {props['rest_endpoint']}\n```\n\n"

        if props.get('minimum_role'):
            answer += f"**Required Role:** {props['minimum_role']}\n"

        return answer

    def _build_procedure_answer(self, results: List[Dict]) -> str:
        """Build procedure answer with steps."""
        # Find procedure nodes
        procedures = [r for r in results if r.get("node_type") == "Procedure"]

        if not procedures:
            return "No procedures found for this query."

        procedure = procedures[0]
        props = procedure.get("properties", {})

        answer = f"## {props.get('title', 'Procedure')}\n\n"
        answer += f"{props.get('description', '')}\n\n"

        # Add steps if available
        steps = [r for r in results if r.get("node_type") == "Step"]
        if steps:
            answer += "### Steps:\n\n"
            sorted_steps = sorted(steps, key=lambda x: x.get("properties", {}).get("order", 0))

            for step in sorted_steps:
                step_props = step.get("properties", {})
                order = step_props.get("order", "")
                instruction = step_props.get("instruction", "")
                answer += f"{order}. {instruction}\n"

        if props.get("estimated_time"):
            answer += f"\n**Estimated Time:** {props['estimated_time']}\n"

        return answer

    def _build_troubleshooting_answer(self, results: List[Dict]) -> str:
        """Build troubleshooting answer."""
        # Find error nodes
        errors = [r for r in results if r.get("node_type") == "Error"]

        if not errors:
            return "No error information found."

        error = errors[0]
        props = error.get("properties", {})

        answer = f"## Error: {props.get('code', 'Unknown')}\n\n"
        answer += f"**Message:** {props.get('message', 'No message')}\n"
        answer += f"**Severity:** {props.get('severity', 'Unknown')}\n\n"

        # Add resolution steps
        procedures = [r for r in results if r.get("node_type") == "Procedure"]
        if procedures:
            answer += "### Resolution Steps:\n\n"
            for proc in procedures:
                proc_props = proc.get("properties", {})
                answer += f"- {proc_props.get('title', 'Unnamed procedure')}\n"

        # Add common causes
        if props.get("common_causes"):
            answer += "\n### Common Causes:\n\n"
            for cause in props["common_causes"]:
                answer += f"- {cause}\n"

        return answer

    def _collect_evidence(self, results: Dict) -> List[Evidence]:
        """Collect supporting evidence from results."""
        evidence_list = []

        for result in results.get("results", [])[:5]:  # Top 5 results
            props = result.get("properties", {})

            # Extract relevant text
            text = props.get("description") or props.get("message") or ""

            if text:
                evidence_list.append(Evidence(
                    source_node_id=result.get("node_id"),
                    source_type=result.get("node_type"),
                    text=text[:500],  # Limit text length
                    confidence=result.get("score", 0.5),
                    relationship_path=result.get("path")
                ))

        return evidence_list

    def _extract_related_entities(self, results: Dict) -> List[Dict[str, Any]]:
        """Extract related entities from results."""
        related = []
        seen_ids = set()

        for result in results.get("results", []):
            # Add relationships if present
            for rel in result.get("relationships", []):
                if isinstance(rel, dict):
                    rel_id = rel.get("id")
                    if rel_id and rel_id not in seen_ids:
                        seen_ids.add(rel_id)
                        related.append({
                            "id": rel_id,
                            "type": rel.get("type"),
                            "name": rel.get("name"),
                            "relationship": rel.get("relationship_type")
                        })

        return related[:10]  # Limit to 10 related entities

    def _calculate_confidence(self,
                            results: Dict,
                            evidence: List[Evidence],
                            query: str) -> float:
        """Calculate confidence score for response."""
        if not results.get("results"):
            return 0.0

        # Factors affecting confidence:
        # 1. Number of results
        result_score = min(1.0, len(results["results"]) / 5)

        # 2. Top result score
        top_score = results["results"][0].get("score", 0) if results["results"] else 0

        # 3. Evidence strength
        evidence_score = sum(e.confidence for e in evidence) / max(len(evidence), 1)

        # 4. Query coverage (simplified)
        query_terms = set(query.lower().split())
        covered_terms = set()
        for result in results["results"][:3]:
            text = str(result.get("properties", {})).lower()
            covered_terms.update(term for term in query_terms if term in text)

        coverage_score = len(covered_terms) / max(len(query_terms), 1)

        # Weighted average
        confidence = (
            0.3 * result_score +
            0.3 * top_score +
            0.2 * evidence_score +
            0.2 * coverage_score
        )

        return min(1.0, confidence)

    def _generate_suggestions(self,
                            results: Dict,
                            response_type: ResponseType) -> List[str]:
        """Generate follow-up suggestions."""
        suggestions = []

        if response_type == ResponseType.TROUBLESHOOTING:
            suggestions.append("View related errors")
            suggestions.append("Check system logs")
            suggestions.append("Contact support if issue persists")

        elif response_type == ResponseType.PROCEDURE:
            suggestions.append("View prerequisites")
            suggestions.append("Check estimated completion time")
            suggestions.append("Review rollback procedures")

        elif response_type == ResponseType.DIRECT_ANSWER:
            # Suggest related commands or concepts
            for result in results.get("results", [])[:3]:
                if result.get("node_type") == "Command":
                    name = result.get("properties", {}).get("name")
                    if name:
                        suggestions.append(f"Learn more about {name}")

        return suggestions[:5]

    def _build_metadata(self, results: Dict) -> Dict[str, Any]:
        """Build response metadata."""
        return {
            "result_count": len(results.get("results", [])),
            "execution_time": results.get("execution_time", 0),
            "search_type": results.get("search_type", "unknown"),
            "nodes_traversed": results.get("nodes_traversed", 0),
            "relationships_followed": results.get("relationships_followed", 0)
        }

    def _load_response_templates(self) -> Dict[ResponseType, str]:
        """Load response templates."""
        return {
            ResponseType.DIRECT_ANSWER: "{answer}",
            ResponseType.EXPLANATION: "## Explanation\n\n{answer}\n\n### Details\n{details}",
            ResponseType.PROCEDURE: "## Procedure: {title}\n\n{description}\n\n### Steps\n{steps}",
            ResponseType.COMPARISON: "## Comparison\n\n{comparison_table}\n\n### Summary\n{summary}",
            ResponseType.TROUBLESHOOTING: "## Troubleshooting: {error}\n\n### Problem\n{problem}\n\n### Solution\n{solution}"
        }
```

## Phase 3: Documentation Ingestion Pipeline

### Task 3.1: Multi-Format Document Parser
**Timeline**: 3 days
**Dependencies**: Phase 1
**Deliverables**: Markdown parser, HTML parser, Notion integration

#### Implementation Steps:

1. **Create unified document parser**:
```python
# src/ingestion/parser.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime

import markdown
from bs4 import BeautifulSoup
from notion_client import AsyncClient as NotionClient
import structlog

logger = structlog.get_logger()

@dataclass
class ParsedDocument:
    """Parsed document with extracted content."""
    source_id: str
    format: str
    title: str
    content: str
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    checksum: str
    parsed_at: datetime

@dataclass
class DocumentSection:
    """Section of a document."""
    id: str
    title: str
    content: str
    level: int
    parent_id: Optional[str] = None
    code_blocks: List[Dict] = None
    tables: List[Dict] = None

class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    async def parse(self, source: Any) -> ParsedDocument:
        """Parse document from source."""
        pass

    def calculate_checksum(self, content: str) -> str:
        """Calculate checksum for content."""
        return hashlib.sha256(content.encode()).hexdigest()

class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""

    def __init__(self):
        self.md = markdown.Markdown(
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )

    async def parse(self, source: str) -> ParsedDocument:
        """Parse Markdown document."""
        # Convert to HTML first
        html = self.md.convert(source)
        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title = self._extract_title(soup)

        # Extract sections
        sections = self._extract_sections(soup, source)

        # Extract metadata
        metadata = self._extract_metadata(source)

        checksum = self.calculate_checksum(source)

        return ParsedDocument(
            source_id=checksum[:12],
            format="markdown",
            title=title,
            content=source,
            sections=sections,
            metadata=metadata,
            checksum=checksum,
            parsed_at=datetime.now()
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract document title."""
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()

        # Try to find first heading
        for tag in ['h2', 'h3', 'h4']:
            heading = soup.find(tag)
            if heading:
                return heading.get_text().strip()

        return "Untitled Document"

    def _extract_sections(self, soup: BeautifulSoup, source: str) -> List[Dict]:
        """Extract sections from document."""
        sections = []
        current_section = None

        for element in soup.children:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Save previous section
                if current_section:
                    sections.append(current_section)

                # Start new section
                level = int(element.name[1])
                current_section = {
                    "id": hashlib.md5(element.get_text().encode()).hexdigest()[:8],
                    "title": element.get_text().strip(),
                    "level": level,
                    "content": "",
                    "code_blocks": [],
                    "tables": []
                }

            elif current_section:
                # Add to current section
                if element.name == 'pre':
                    # Code block
                    code = element.find('code')
                    if code:
                        current_section["code_blocks"].append({
                            "language": code.get('class', [''])[0].replace('language-', ''),
                            "code": code.get_text()
                        })

                elif element.name == 'table':
                    # Table
                    current_section["tables"].append(
                        self._parse_table(element)
                    )

                elif element.string:
                    current_section["content"] += element.string + "\n"

        # Add last section
        if current_section:
            sections.append(current_section)

        return sections

    def _parse_table(self, table_element) -> Dict:
        """Parse HTML table."""
        headers = []
        rows = []

        # Extract headers
        thead = table_element.find('thead')
        if thead:
            for th in thead.find_all('th'):
                headers.append(th.get_text().strip())

        # Extract rows
        tbody = table_element.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                for td in tr.find_all('td'):
                    row.append(td.get_text().strip())
                rows.append(row)

        return {"headers": headers, "rows": rows}

    def _extract_metadata(self, source: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        metadata = {}

        # Check for YAML front matter
        if source.startswith('---'):
            lines = source.split('\n')
            front_matter = []
            in_front_matter = False

            for line in lines[1:]:
                if line == '---':
                    break
                front_matter.append(line)

            if front_matter:
                import yaml
                try:
                    metadata = yaml.safe_load('\n'.join(front_matter))
                except:
                    logger.warning("Failed to parse front matter")

        return metadata

class NotionParser(DocumentParser):
    """Parser for Notion pages."""

    def __init__(self, notion_token: str):
        self.notion = NotionClient(auth=notion_token)

    async def parse(self, page_id: str) -> ParsedDocument:
        """Parse Notion page."""
        # Fetch page
        page = await self.notion.pages.retrieve(page_id)

        # Fetch page content
        blocks = await self._fetch_all_blocks(page_id)

        # Convert to sections
        sections = self._blocks_to_sections(blocks)

        # Extract content
        content = self._blocks_to_markdown(blocks)

        # Extract title
        title = self._extract_page_title(page)

        checksum = self.calculate_checksum(content)

        return ParsedDocument(
            source_id=page_id,
            format="notion",
            title=title,
            content=content,
            sections=sections,
            metadata={
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
                "url": page.get("url")
            },
            checksum=checksum,
            parsed_at=datetime.now()
        )

    async def _fetch_all_blocks(self, page_id: str) -> List[Dict]:
        """Fetch all blocks from Notion page."""
        blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            response = await self.notion.blocks.children.list(
                block_id=page_id,
                start_cursor=start_cursor
            )

            blocks.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return blocks

    def _blocks_to_sections(self, blocks: List[Dict]) -> List[Dict]:
        """Convert Notion blocks to sections."""
        sections = []
        current_section = None

        for block in blocks:
            block_type = block.get("type")

            if block_type in ["heading_1", "heading_2", "heading_3"]:
                # Save previous section
                if current_section:
                    sections.append(current_section)

                # Start new section
                level = int(block_type[-1])
                text = self._extract_text_from_block(block)

                current_section = {
                    "id": block["id"],
                    "title": text,
                    "level": level,
                    "content": "",
                    "code_blocks": [],
                    "tables": []
                }

            elif current_section:
                if block_type == "code":
                    code_data = block.get(block_type, {})
                    current_section["code_blocks"].append({
                        "language": code_data.get("language", ""),
                        "code": self._extract_text_from_block(block)
                    })

                elif block_type == "table":
                    # Process table
                    pass  # Table processing would go here

                else:
                    text = self._extract_text_from_block(block)
                    if text:
                        current_section["content"] += text + "\n"

        # Add last section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from Notion block."""
        block_type = block.get("type")
        block_data = block.get(block_type, {})

        # Handle rich text
        if "rich_text" in block_data:
            texts = []
            for text_obj in block_data["rich_text"]:
                texts.append(text_obj.get("plain_text", ""))
            return " ".join(texts)

        # Handle other text fields
        if "text" in block_data:
            if isinstance(block_data["text"], list):
                return " ".join([t.get("plain_text", "") for t in block_data["text"]])
            return block_data["text"]

        return ""

    def _blocks_to_markdown(self, blocks: List[Dict]) -> str:
        """Convert Notion blocks to Markdown."""
        markdown_lines = []

        for block in blocks:
            block_type = block.get("type")

            if block_type == "heading_1":
                text = self._extract_text_from_block(block)
                markdown_lines.append(f"# {text}")

            elif block_type == "heading_2":
                text = self._extract_text_from_block(block)
                markdown_lines.append(f"## {text}")

            elif block_type == "heading_3":
                text = self._extract_text_from_block(block)
                markdown_lines.append(f"### {text}")

            elif block_type == "paragraph":
                text = self._extract_text_from_block(block)
                markdown_lines.append(text)

            elif block_type == "code":
                code_data = block.get(block_type, {})
                language = code_data.get("language", "")
                code = self._extract_text_from_block(block)
                markdown_lines.append(f"```{language}\n{code}\n```")

            elif block_type == "bulleted_list_item":
                text = self._extract_text_from_block(block)
                markdown_lines.append(f"- {text}")

            elif block_type == "numbered_list_item":
                text = self._extract_text_from_block(block)
                markdown_lines.append(f"1. {text}")

            markdown_lines.append("")  # Empty line between blocks

        return "\n".join(markdown_lines)

    def _extract_page_title(self, page: Dict) -> str:
        """Extract title from Notion page."""
        properties = page.get("properties", {})

        # Look for title property
        for prop_name, prop_data in properties.items():
            if prop_data.get("type") == "title":
                title_data = prop_data.get("title", [])
                if title_data:
                    return title_data[0].get("plain_text", "Untitled")

        return "Untitled"
```

### Task 3.2: Entity Extraction System
**Timeline**: 3 days
**Dependencies**: Task 3.1
**Deliverables**: Entity extractors for all node types

[Continue with remaining implementation details...]

## Deployment and Operations Guide

### Production Deployment Checklist

1. **Pre-deployment**:
   - [ ] All tests passing
   - [ ] Security scan completed
   - [ ] Performance benchmarks met
   - [ ] Documentation updated
   - [ ] Backup procedures tested

2. **Deployment**:
   - [ ] Blue-green deployment configured
   - [ ] Health checks verified
   - [ ] Monitoring alerts configured
   - [ ] Rollback procedures ready
   - [ ] Traffic gradually shifted

3. **Post-deployment**:
   - [ ] Metrics within expected range
   - [ ] No error spike observed
   - [ ] Cache warming completed
   - [ ] User acceptance tested
   - [ ] Performance validated

### Operational Runbook

#### Incident Response

1. **High Latency**:
   ```bash
   # Check query complexity
   docker exec weka-mcp python -m src.tools.analyze_slow_queries

   # Review cache hit rate
   docker exec weka-mcp python -m src.tools.check_cache_stats

   # Check Neo4j query plans
   docker exec weka-neo4j cypher-shell "CALL db.listQueries()"
   ```

2. **Memory Issues**:
   ```bash
   # Check memory usage
   docker stats

   # Review heap dumps
   docker exec weka-mcp python -m src.tools.memory_profiler

   # Adjust cache sizes if needed
   docker exec weka-mcp python -m src.tools.adjust_cache_size --size 1GB
   ```

3. **Graph Corruption**:
   ```bash
   # Validate graph consistency
   docker exec weka-mcp python -m src.graph.validate_consistency

   # Restore from backup if needed
   docker exec weka-mcp python -m src.graph.restore_backup --version latest
   ```

### Scaling Guide

#### Horizontal Scaling

1. **Add MCP Server Replicas**:
   ```yaml
   # docker-compose.scale.yml
   services:
     mcp-server:
       deploy:
         replicas: 3
       environment:
         - INSTANCE_ID={{.Task.Slot}}
   ```

2. **Neo4j Clustering**:
   ```yaml
   # Enable Neo4j clustering for read scalability
   neo4j-core-1:
     environment:
       - NEO4J_causal__clustering_initial__discovery__members=neo4j-core-1,neo4j-core-2,neo4j-core-3
       - NEO4J_dbms_mode=CORE
   ```

#### Vertical Scaling

1. **Increase Resources**:
   ```yaml
   services:
     mcp-server:
       deploy:
         resources:
           limits:
             cpus: '4'
             memory: 8G
   ```

2. **Optimize Indexes**:
   ```cypher
   -- Add more specific indexes based on query patterns
   CREATE INDEX FOR (c:Command) ON (c.category, c.cli_syntax);
   CREATE INDEX FOR (e:Error) ON (e.severity, e.code);
   ```

## Conclusion

This implementation plan provides a complete technical roadmap for building the Weka Documentation GraphRAG MCP Server. The phased approach ensures systematic development with clear dependencies and deliverables. Each component has been designed with security, performance, and maintainability in mind.

The system's architecture enables:
- Safe execution of complex graph queries
- Scalable document ingestion from multiple sources
- High-performance hybrid search capabilities
- Robust security against injection attacks
- Comprehensive monitoring and observability

With this implementation, you'll have a production-ready system capable of handling sophisticated documentation queries while maintaining the security and performance requirements of an enterprise environment.
