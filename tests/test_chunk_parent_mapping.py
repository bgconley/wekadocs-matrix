from src.shared.chunk_utils import remap_parent_section_ids


def test_remap_parent_ids_updates_to_chunk_ids():
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

    remapped, missing = remap_parent_section_ids(chunks)

    assert remapped == 1
    assert missing == 0
    assert chunks[1]["parent_section_id"] == "chunk-root"


def test_remap_parent_ids_handles_missing_parent_gracefully():
    chunks = [
        {
            "id": "orphan-chunk",
            "original_section_ids": ["orig-orphan"],
            "parent_section_id": "ghost-parent",
        }
    ]

    remapped, missing = remap_parent_section_ids(chunks)

    assert remapped == 0
    assert missing == 1
    assert chunks[0]["parent_section_id"] is None


def test_remap_parent_ids_supports_combined_chunks():
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

    remapped, missing = remap_parent_section_ids(chunks)

    assert remapped == 1
    assert missing == 0
    assert chunks[1]["parent_section_id"] == "combined-parent"
