cypher-shell -u neo4j -p '<password>' \
  --param document_id="DOC-12345" \
  --param tenant="acme" \
  --param window=6 \
  --param rebuild=true \
  -f post_ingest_relationships_v2_2.cypher


# to run on all docs: cypher-shell -u neo4j -p '<password>' -f post_ingest_relationships_v2_2.cypher
