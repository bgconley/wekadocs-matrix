# Test Document for Auto-Ingestion

This is a smoke test for the auto-ingestion pipeline.

## Section 1: Testing Vector Embedding

Testing vector embedding and graph construction with meaningful content.

## Section 2: Commands

Here's a sample command for testing entity extraction:

```bash
weka cluster status
weka fs list
```

## Section 3: Configuration

Sample configuration parameters:

- `max_connections`: Maximum number of connections
- `timeout_seconds`: Request timeout in seconds

## Section 4: Procedure

To test the system:

1. Drop a markdown file into the watched directory
2. Verify it gets enqueued
3. Check that the worker processes it
4. Validate the document appears in Neo4j
5. Verify vectors are created in Qdrant
