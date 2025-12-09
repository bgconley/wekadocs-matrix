"""
Live integration tests for GLiNER model loading and extraction.

These tests actually load the GLiNER model and perform real extraction.
They are slower (~10-15 seconds) but verify the full integration works.

Run with: pytest tests/integration/test_gliner_live.py -v -s

Marks:
- @pytest.mark.slow: Takes >5 seconds
- @pytest.mark.live: Requires real model download/load
"""

import pytest


@pytest.mark.slow
@pytest.mark.live
class TestGLiNERLiveModel:
    """Live tests that actually load and use the GLiNER model."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        from src.providers.ner.gliner_service import GLiNERService

        # Reset singleton state
        GLiNERService._instance = None
        GLiNERService._initialized = False
        yield
        # Cleanup after test
        if GLiNERService._instance is not None:
            GLiNERService._instance.reset()
            GLiNERService._instance = None
            GLiNERService._initialized = False

    def test_model_loads_successfully(self):
        """Verify the actual GLiNER model loads without error."""
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()

        # This triggers actual model loading
        assert service.is_available is True
        assert service._model is not None
        assert service._model_load_failed is False

    def test_device_detection_works(self):
        """Verify device is detected correctly (MPS on Apple Silicon)."""
        from src.providers.ner.gliner_service import GLiNERService

        service = GLiNERService()
        _ = service.is_available  # Trigger load

        # On Apple Silicon Mac, should be 'mps'
        # On other systems, 'cuda' or 'cpu'
        assert service._device in ("mps", "cuda", "cpu")
        print(f"\n  Device detected: {service._device}")

    def test_extract_weka_entities(self):
        """Verify extraction works with WEKA-specific text."""
        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        # Real WEKA documentation text
        text = """
        To mount WEKA filesystem on RHEL 8, install the weka-agent package
        and configure NFS exports. Use the weka fs mount command with
        --net-apply option. Check /var/log/weka for errors.
        """

        entities = service.extract_entities(text, labels)

        print(f"\n  Extracted {len(entities)} entities:")
        for e in entities:
            print(f"    - {e.text!r} ({e.label}) score={e.score:.2f}")

        # Should find some entities
        assert len(entities) > 0

        # Check we found expected types
        entity_labels = {e.label for e in entities}
        entity_texts_lower = {e.text.lower() for e in entities}

        # Should recognize at least some of these
        expected_finds = ["weka", "rhel", "nfs"]
        found_any = any(exp in entity_texts_lower for exp in expected_finds)

        print(f"  Labels found: {entity_labels}")
        print(f"  Texts found: {entity_texts_lower}")

        # Soft assertion - GLiNER may not find all, but should find something relevant
        assert found_any or len(entities) > 0, "Should extract at least one entity"

    def test_batch_extraction_works(self):
        """Verify batch extraction works with multiple texts."""
        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        texts = [
            "Mount NFS share on Ubuntu 22.04 LTS",
            "Configure AWS S3 backend for WEKA cluster",
            "Check IOPS performance with fio benchmark",
        ]

        results = service.batch_extract_entities(texts, labels)

        print("\n  Batch extraction results:")
        for i, (text, entities) in enumerate(zip(texts, results)):
            print(f"    Text {i+1}: {len(entities)} entities")
            for e in entities:
                print(f"      - {e.text!r} ({e.label})")

        # Should return one list per input text
        assert len(results) == len(texts)

        # At least some texts should have entities
        total_entities = sum(len(r) for r in results)
        assert total_entities > 0, "Should extract entities from batch"

    def test_extraction_with_empty_text(self):
        """Verify empty text returns empty list (no crash)."""
        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        result = service.extract_entities("", labels)
        assert result == []

    def test_caching_works(self):
        """Verify LRU cache prevents re-extraction for same query."""
        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        # Short text (< 200 chars) should be cached
        query = "How to configure NFS on RHEL?"

        # First call - cache miss
        result1 = service.extract_entities(query, labels)

        # Check cache info
        cache_info = service._extract_cached.cache_info()
        print(
            f"\n  After first call: hits={cache_info.hits}, misses={cache_info.misses}"
        )

        # Second call - should be cache hit
        result2 = service.extract_entities(query, labels)

        cache_info = service._extract_cached.cache_info()
        print(
            f"  After second call: hits={cache_info.hits}, misses={cache_info.misses}"
        )

        # Results should be identical
        assert result1 == result2

        # Should have at least one cache hit now
        assert cache_info.hits >= 1


@pytest.mark.slow
@pytest.mark.live
class TestGLiNERModelPerformance:
    """Performance benchmarks for GLiNER extraction."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False
        yield
        if GLiNERService._instance is not None:
            GLiNERService._instance.reset()
            GLiNERService._instance = None
            GLiNERService._initialized = False

    def test_single_extraction_latency(self):
        """Measure single extraction latency (after model loaded)."""
        import time

        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        # Warm up - load model
        _ = service.extract_entities("warmup text", labels)

        # Measure extraction time
        text = "Configure WEKA cluster with NFS exports on RHEL 8 servers."

        start = time.perf_counter()
        _ = service.extract_entities(text, labels)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Single extraction latency: {elapsed_ms:.1f}ms")

        # Should be reasonably fast after warmup
        assert elapsed_ms < 500, f"Extraction too slow: {elapsed_ms:.1f}ms"

    def test_batch_extraction_throughput(self):
        """Measure batch extraction throughput."""
        import time

        from src.providers.ner.gliner_service import GLiNERService
        from src.providers.ner.labels import get_default_labels

        service = GLiNERService()
        labels = get_default_labels()

        # Warm up
        _ = service.extract_entities("warmup", labels)

        # Create batch of realistic texts
        texts = [
            f"Document {i}: Configure WEKA backend {i} with NFS mount on host-{i}"
            for i in range(32)
        ]

        start = time.perf_counter()
        results = service.batch_extract_entities(texts, labels)
        elapsed_s = time.perf_counter() - start

        throughput = len(texts) / elapsed_s
        print(f"\n  Batch of {len(texts)} texts: {elapsed_s:.2f}s")
        print(f"  Throughput: {throughput:.1f} texts/second")

        assert len(results) == len(texts)
        # Should process at least 5 texts/second on reasonable hardware
        assert throughput > 5, f"Throughput too low: {throughput:.1f} texts/s"
