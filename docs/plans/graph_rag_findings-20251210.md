 Prescriptive GraphRAG Recommendations for WekaDocs-Matrix

  Based on comprehensive research from Neo4j, Microsoft Research, GraphRAG.com, Gradient Flow, and academic papers, here are the specific changes needed to make your graph effective for GraphRAG.

  ---
  Executive Summary: The Two-Layer Graph Model

  The industry consensus is clear: effective GraphRAG requires two interconnected graph layers:

  ★ Insight ─────────────────────────────────────
  The "Lexical Graph + Domain Graph" pattern is the gold standard:

  1. LEXICAL GRAPH: Document → Section/Chunk structure (provenance)
  2. DOMAIN GRAPH: Extracted entities and their relationships (knowledge)
  3. BRIDGE: HAS_ENTITY / MENTIONS relationships connecting them

  Your graph has both layers but they're disconnected islands.
  ─────────────────────────────────────────────────

  ---
  Prescriptive Change #1: Add MENTIONS / HAS_ENTITY Relationships

  This is the #1 priority fix. The canonical GraphRAG pattern requires explicit relationships linking chunks to the entities they mention.

  Current State (Broken)

  (Section) ──────────────── [NO CONNECTION] ──────────────── (Configuration)
  (Section) ──────────────── [NO CONNECTION] ──────────────── (Command)
  (Section) ──────────────── [NO CONNECTION] ──────────────── (Procedure)

  Target State (Per GraphRAG.com Best Practice)

  // The standard pattern from graphrag.com/reference/knowledge-graph/lexical-graph-extracted-entities
  (Document)-[:HAS_SECTION]->(Chunk)-[:MENTIONS]->(Entity)
                                    └-[:MENTIONS]->(Configuration)
                                    └-[:MENTIONS]->(Command)
                                    └-[:MENTIONS]->(Procedure)

  Implementation Cypher

  // Create MENTIONS relationships with provenance metadata
  CREATE (chunk:Chunk)-[:MENTIONS {
    confidence: 0.95,           // Extraction confidence score
    extraction_method: 'gliner', // How it was extracted
    character_offset: 245,       // Where in the text
    created_at: datetime()
  }]->(entity:Entity)

  Required Schema Changes

  // Add indexes for traversal performance
  CREATE INDEX entity_mentions_idx FOR ()-[r:MENTIONS]->() ON (r.confidence);
  CREATE INDEX chunk_entity_idx FOR (c:Chunk)-[:MENTIONS]->(e:Entity) ON (c.document_id, e.name);

  ---
  Prescriptive Change #2: Add Section Hierarchy with NEXT_CHUNK and PARENT_SECTION

  Microsoft GraphRAG and Neo4j best practices emphasize document structure for context assembly.

  Target Pattern (From Neo4j GraphRAG Manifesto)

  // Hierarchical section relationships
  (Section {level: 1})-[:HAS_SUBSECTION]->(Section {level: 2})
  (Section {level: 2})-[:HAS_SUBSECTION]->(Section {level: 3})

  // Sequential chunk navigation (critical for context windowing)
  (Chunk {order: 0})-[:NEXT_CHUNK]->(Chunk {order: 1})-[:NEXT_CHUNK]->(Chunk {order: 2})

  Migration Query

  // Create NEXT_CHUNK relationships for sequential navigation
  MATCH (c1:Chunk), (c2:Chunk)
  WHERE c1.document_id = c2.document_id
    AND c2.order = c1.order + 1
  MERGE (c1)-[:NEXT_CHUNK]->(c2);

  // Create hierarchical PARENT_SECTION relationships
  MATCH (parent:Section), (child:Section)
  WHERE parent.document_id = child.document_id
    AND parent.level < child.level
    AND parent.order < child.order
    AND NOT EXISTS {
      MATCH (closer:Section)
      WHERE closer.document_id = parent.document_id
        AND closer.level < child.level
        AND closer.level > parent.level
        AND closer.order < child.order
        AND closer.order > parent.order
    }
  MERGE (child)-[:PARENT_SECTION]->(parent);

  Why This Matters

  ★ Insight ─────────────────────────────────────
  Context assembly is the core of RAG. When you retrieve a chunk,
  you often need:
  - Previous/next chunks for continuity
  - Parent section for hierarchical context
  - Sibling sections for related information

  Without NEXT_CHUNK: You must sort by order at query time (slow)
  With NEXT_CHUNK: Single graph traversal (fast, O(1) per hop)
  ─────────────────────────────────────────────────

  ---
  Prescriptive Change #3: Implement the Microsoft GraphRAG Community Detection Pattern

  Microsoft's breakthrough insight: Cluster entities into communities and generate hierarchical summaries for "global" questions.

  Target Architecture

                      ┌─────────────────────────────┐
                      │     Community Level 2       │
                      │  "WEKA Cloud Deployment"    │
                      │     (summary: 500 tokens)   │
                      └──────────────┬──────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
      ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
      │  Community L1   │  │  Community L1   │  │  Community L1   │
      │ "AWS Setup"     │  │ "Azure Setup"   │  │ "GCP Setup"     │
      │ (summary)       │  │ (summary)       │  │ (summary)       │
      └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
               │                    │                    │
          ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
          ▼         ▼          ▼         ▼          ▼         ▼
      (Entity)  (Entity)   (Entity)  (Entity)   (Entity)  (Entity)

  Implementation with Neo4j GDS

  // Step 1: Project the entity graph
  CALL gds.graph.project(
    'entity-graph',
    ['Configuration', 'Command', 'Procedure', 'Step'],
    {
      RELATED_TO: {orientation: 'UNDIRECTED'},
      CONTAINS_STEP: {orientation: 'UNDIRECTED'}
    }
  );

  // Step 2: Run Leiden community detection (hierarchical)
  CALL gds.leiden.write('entity-graph', {
    writeProperty: 'community_id',
    includeIntermediateCommunities: true,  // Creates hierarchy
    intermediateCommunitiesWriteProperty: 'community_hierarchy'
  });

  // Step 3: Create Community nodes
  MATCH (e:Entity)
  WITH DISTINCT e.community_id AS community_id, collect(e) AS members
  CREATE (c:Community {
    id: 'community_' + toString(community_id),
    level: 0,
    member_count: size(members)
  })
  WITH c, members
  UNWIND members AS member
  CREATE (member)-[:BELONGS_TO]->(c);

  // Step 4: Generate community summaries (via LLM)
  // Store as community.summary property

  Community Summary Schema

  CREATE (c:Community {
    id: STRING,
    level: INTEGER,              // 0 = leaf, higher = more abstract
    summary: STRING,             // LLM-generated summary
    summary_embedding: LIST<FLOAT>,  // For community-level search
    member_count: INTEGER,
    key_entities: LIST<STRING>,  // Top entities by centrality
    created_at: DATETIME
  })

  ---
  Prescriptive Change #4: Fix Procedure-Section Provenance

  Your CONTAINS_STEP relationships have source_section_id but the IDs don't resolve. This breaks provenance tracing.

  Migration Strategy

  // Option A: Create direct DEFINED_IN relationships
  MATCH (p:Procedure)
  // Find the section that best matches the procedure by title/content
  MATCH (s:Section)
  WHERE s.heading CONTAINS p.title OR s.text CONTAINS p.title
  WITH p, s,
       apoc.text.sorensenDiceSimilarity(p.description, s.text) AS similarity
  ORDER BY similarity DESC
  LIMIT 1
  MERGE (p)-[:DEFINED_IN {confidence: similarity}]->(s);

  // Option B: Fix the source_section_id values
  // (requires understanding why they don't match)
  MATCH (p:Procedure)-[r:CONTAINS_STEP]->(step:Step)
  WHERE r.source_section_id IS NOT NULL
  MATCH (s:Section)
  WHERE s.id = r.source_section_id OR s.heading = r.source_section_id
  MERGE (step)-[:EXTRACTED_FROM]->(s);

  ---
  Prescriptive Change #5: Implement the Hybrid Retrieval Pattern

  The gold standard architecture from Gradient Flow and Neo4j:

  ┌──────────────────────────────────────────────────────────────────┐
  │                        QUERY FLOW                                 │
  ├──────────────────────────────────────────────────────────────────┤
  │  1. VECTOR SEARCH (Qdrant)                                       │
  │     → Find top-k semantically similar chunks                      │
  │     → Return chunk IDs + scores                                   │
  │                                                                   │
  │  2. ENTITY EXTRACTION (from query)                                │
  │     → Extract entities mentioned in the user question             │
  │     → Match to graph entities                                     │
  │                                                                   │
  │  3. GRAPH EXPANSION (Neo4j)                                       │
  │     → From retrieved chunks: traverse MENTIONS → Entity           │
  │     → From matched entities: traverse relationships               │
  │     → Expand to related chunks via MENTIONS                       │
  │                                                                   │
  │  4. CONTEXT ASSEMBLY                                              │
  │     → Merge vector results + graph expansion                      │
  │     → Use NEXT_CHUNK for continuity                               │
  │     → Include entity relationship context                         │
  │                                                                   │
  │  5. RE-RANKING (optional)                                         │
  │     → Score by graph centrality (PageRank)                        │
  │     → Score by path distance to query entities                    │
  │     → RRF fusion of all signals                                   │
  └──────────────────────────────────────────────────────────────────┘

  Neo4j Query for Graph-Enhanced Retrieval

  // Given chunk IDs from vector search, expand via graph
  WITH $vector_chunk_ids AS seed_chunks
  MATCH (c:Chunk) WHERE c.id IN seed_chunks

  // Get entities mentioned in these chunks
  OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)

  // Get related entities (1-2 hops)
  OPTIONAL MATCH (e)-[r:RELATED_TO|CONTAINS_STEP*1..2]-(related:Entity)

  // Get other chunks that mention these entities
  OPTIONAL MATCH (related)<-[:MENTIONS]-(other_chunk:Chunk)
  WHERE NOT other_chunk.id IN seed_chunks

  // Get sequential context
  OPTIONAL MATCH (c)-[:NEXT_CHUNK*1..2]-(neighbor:Chunk)

  // Get parent section for hierarchy context
  OPTIONAL MATCH (c)-[:PARENT_SECTION]->(parent:Section)

  RETURN DISTINCT
    c AS seed_chunk,
    collect(DISTINCT e) AS mentioned_entities,
    collect(DISTINCT related) AS related_entities,
    collect(DISTINCT other_chunk)[0..3] AS graph_expanded_chunks,
    collect(DISTINCT neighbor) AS sequential_context,
    parent AS parent_section

  ---
  Prescriptive Change #6: Add Entity Resolution and Deduplication

  Your 1,121 Configuration entities and 292 Command entities likely have duplicates or near-duplicates.

  Implementation Pattern

  // Find similar entities using embedding similarity (if you add embeddings)
  MATCH (e1:Configuration), (e2:Configuration)
  WHERE e1.id < e2.id
    AND gds.similarity.cosine(e1.embedding, e2.embedding) > 0.9
  MERGE (e1)-[:SAME_AS {similarity: gds.similarity.cosine(e1.embedding, e2.embedding)}]->(e2);

  // Or use string similarity for names
  MATCH (e1:Configuration), (e2:Configuration)
  WHERE e1.id < e2.id
    AND apoc.text.sorensenDiceSimilarity(e1.name, e2.name) > 0.85
  MERGE (e1)-[:SAME_AS {similarity: apoc.text.sorensenDiceSimilarity(e1.name, e2.name)}]->(e2);

  Canonical Entity Pattern

  // Create canonical entities that merge duplicates
  MATCH (e:Configuration)-[:SAME_AS*]-(related:Configuration)
  WITH e, collect(DISTINCT related) + [e] AS cluster
  ORDER BY size(cluster) DESC
  WITH head(collect(cluster)) AS largest_cluster
  UNWIND largest_cluster AS member
  WITH largest_cluster[0] AS canonical, member
  WHERE member <> canonical
  // Transfer all MENTIONS to canonical
  MATCH (chunk)-[r:MENTIONS]->(member)
  MERGE (chunk)-[:MENTIONS]->(canonical)
  DELETE r;

  ---
  Target Schema Summary

  After implementing these changes, your graph should have this structure:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         LEXICAL LAYER                                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  (Document)──[:HAS_SECTION]──>(Section/Chunk)                           │
  │                                    │                                     │
  │                         [:NEXT_CHUNK]  [:PARENT_SECTION]                │
  │                                    │         │                           │
  │                                    ▼         ▼                           │
  │                           (Section/Chunk)──(parent Section)             │
  └─────────────────────────────────────────────────────────────────────────┘
                                       │
                                [:MENTIONS]  ← THE CRITICAL BRIDGE
                                       │
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         DOMAIN LAYER                                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  (Configuration)──[:RELATED_TO]──>(Command)                             │
  │         │                            │                                   │
  │         └───[:USED_IN]───>(Procedure)──[:CONTAINS_STEP]──>(Step)        │
  │                                │                                         │
  │                         [:DEFINED_IN]                                    │
  │                                │                                         │
  │                                ▼                                         │
  │                           (Section) ← PROVENANCE LINK                   │
  └─────────────────────────────────────────────────────────────────────────┘
                                       │
                               [:BELONGS_TO]
                                       │
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       COMMUNITY LAYER (Optional but powerful)            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  (Community L0)──[:PARENT_COMMUNITY]──>(Community L1)──>...             │
  │       │                                                                  │
  │       └── summary, summary_embedding, key_entities                      │
  └─────────────────────────────────────────────────────────────────────────┘

  ---
  Implementation Priority Order

  | Priority | Change                       | Effort | Impact                                 |
  |----------|------------------------------|--------|----------------------------------------|
  | 🔴 P0    | Add MENTIONS relationships   | Medium | Critical - Fixes entity orphaning      |
  | 🔴 P0    | Fix Procedure provenance     | Low    | Critical - Enables procedure retrieval |
  | 🟡 P1    | Add NEXT_CHUNK relationships | Low    | High - Enables context windowing       |
  | 🟡 P1    | Add section hierarchy        | Medium | High - Enables hierarchical context    |
  | 🟢 P2    | Community detection          | High   | Medium - Enables global queries        |
  | 🟢 P2    | Entity resolution            | Medium | Medium - Improves entity quality       |

  ---
  ★ Insight ─────────────────────────────────────
  The research is unanimous: GraphRAG outperforms vector-only RAG
  by 3x on accuracy (Data.world study) and uses 26-97% fewer tokens
  (Microsoft study). But this only works when:

  1. Entities are CONNECTED to their source chunks (MENTIONS)
  2. Chunks are CONNECTED to each other (NEXT_CHUNK, hierarchy)
  3. Entities are CONNECTED to other entities (domain relationships)

  Your graph has good bones but broken bridges. Fix the MENTIONS
  relationships first - everything else builds on that foundation.
  ─────────────────────────────────────────────────

  Would you like me to:
  1. Generate the complete migration scripts for these changes?
  2. Examine your ingestion code to add MENTIONS during entity extraction?
  3. Design the hybrid retrieval query that leverages this new structure?
