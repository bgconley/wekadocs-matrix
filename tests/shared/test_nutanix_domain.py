"""Tests for the Nutanix domain authority used by runtime surfaces."""

from src.shared.domain import get_domain_config


def test_domain_config_exposes_nutanix_runtime_names():
    domain = get_domain_config()

    assert domain.product_name == "Nutanix"
    assert domain.app_name == "nutanix-docs-matrix"
    assert domain.corpus_namespace == "nutanix"
    assert domain.uri_scheme == "nutanixdocs"
    assert domain.cache_prefix == "nutanix:cache:v1"
    assert domain.otel_service_prefix == "nutanix"


def test_domain_config_covers_full_nutanix_taxonomy():
    domain = get_domain_config()
    inventory = " ".join(domain.product_taxonomy)

    for required in (
        "Nutanix Cloud Platform",
        "NCP",
        "Nutanix Cloud Infrastructure",
        "NCI",
        "AOS",
        "AHV",
        "Prism Central",
        "Nutanix Central",
        "Nutanix Unified Storage",
        "NUS",
        "Nutanix Files",
        "Nutanix Objects",
        "Nutanix Volumes",
        "Nutanix Data Lens",
        "Nutanix Cloud Manager",
        "NCM",
        "Nutanix Database Service",
        "NDB",
        "Nutanix Kubernetes Platform",
        "NKP",
        "Nutanix Data Services for Kubernetes",
        "NDK",
        "Nutanix Cloud Clusters",
        "NC2",
        "Nutanix Government Cloud Clusters",
        "GC2",
        "Nutanix Enterprise AI",
        "NAI",
        "Nutanix Flow",
        "Nutanix Disaster Recovery",
        "Move",
    ):
        assert required in inventory


def test_domain_config_prompts_are_nutanix_only():
    domain = get_domain_config()
    prompt_text = " ".join(
        [
            domain.query_instruction,
            domain.reranker_instruction,
            *domain.reranker_instructions_by_type.values(),
            *domain.ner_labels,
        ]
    )

    assert "Nutanix" in prompt_text
    legacy_upper = "W" + "EKA"
    legacy_lower = "we" + "ka"
    assert legacy_upper not in prompt_text
    assert legacy_lower not in prompt_text
