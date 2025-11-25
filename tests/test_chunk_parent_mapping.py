from src.shared.chunk_utils import canonicalize_parent_ids


def test_canonicalize_parent_ids_updates_to_chunk_ids():
    chunks = [
        {
            "id": "chunk-root",
            "original_section_ids": ["orig-root"],
            "parent_section_id": None,
        },
        {
            "id": "child-chunk",
            "original_section_ids": ["orig-child"],
            "parent_section_id": "orig-root",
        },
    ]

    remapped, missing = canonicalize_parent_ids(chunks)

    assert remapped == 1
    assert missing == 0
    assert chunks[1]["parent_section_id"] == "chunk-root"
    assert chunks[1]["parent_chunk_id"] == "chunk-root"
    assert chunks[1]["parent_section_original_id"] == "orig-root"


def test_canonicalize_parent_ids_handles_missing_parent_gracefully():
    chunks = [
        {
            "id": "orphan-chunk",
            "original_section_ids": ["orig-orphan"],
            "parent_section_id": "ghost-parent",
        }
    ]

    remapped, missing = canonicalize_parent_ids(chunks)

    assert remapped == 0
    assert missing == 1
    assert chunks[0]["parent_section_id"] is None
    assert chunks[0]["parent_chunk_id"] is None
    assert chunks[0]["parent_section_original_id"] == "ghost-parent"


def test_canonicalize_parent_ids_supports_combined_chunks():
    chunks = [
        {
            "id": "combined-parent",
            "original_section_ids": ["orig-a", "orig-b"],
            "parent_section_id": None,
        },
        {
            "id": "child-chunk",
            "original_section_ids": ["orig-child"],
            "parent_section_id": "orig-b",
        },
    ]

    remapped, missing = canonicalize_parent_ids(chunks)

    assert remapped == 1
    assert missing == 0
    assert chunks[1]["parent_section_id"] == "combined-parent"
    assert chunks[1]["parent_section_original_id"] == "orig-b"
