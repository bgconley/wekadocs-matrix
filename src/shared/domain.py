# =============================================================================
# @status: ACTIVE
# @called-by: config.py, mcp_app.py, mcp_tools.py, query_service.py
# =============================================================================
"""Nutanix domain authority for runtime prompts and naming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DomainConfig:
    product_name: str
    app_name: str
    corpus_namespace: str
    uri_scheme: str
    cache_prefix: str
    otel_service_prefix: str
    product_taxonomy: Tuple[str, ...]
    query_instruction: str
    reranker_instruction: str
    reranker_instructions_by_type: Dict[str, str]
    ner_labels: Tuple[str, ...]


NUTANIX_TAXONOMY: Tuple[str, ...] = (
    "Nutanix Cloud Platform",
    "NCP",
    "Nutanix Central",
    "Prism Central",
    "Prism Element",
    "Prism",
    "Nutanix Cloud Infrastructure",
    "NCI",
    "AOS Storage",
    "AOS",
    "AHV Virtualization",
    "AHV",
    "NCI Data",
    "NCI-VDI",
    "NCI-Edge",
    "NCI with External Storage",
    "Foundation",
    "Foundation Central",
    "LCM",
    "NCC",
    "Nutanix Cloud Clusters",
    "NC2",
    "Nutanix Government Cloud Clusters",
    "GC2",
    "NC2 on AWS",
    "NC2 on Azure",
    "NC2 on Google Cloud",
    "NC2 on OVHcloud",
    "Nutanix Unified Storage",
    "NUS",
    "Files Storage",
    "Nutanix Files",
    "Objects Storage",
    "Nutanix Objects",
    "Volumes Block Storage",
    "Nutanix Volumes",
    "Nutanix Data Lens",
    "NDL",
    "CSI",
    "COSI",
    "S3",
    "SMB",
    "NFS",
    "iSCSI",
    "Nutanix Cloud Manager",
    "NCM",
    "Intelligent Operations",
    "Self-Service",
    "Cost Governance",
    "Security Central",
    "NCM-Edge",
    "Nutanix Kubernetes Platform",
    "NKP",
    "Nutanix Data Services for Kubernetes",
    "NDK",
    "Cloud Native AOS",
    "Multicloud Kubernetes",
    "fleet management",
    "GitOps",
    "service mesh",
    "observability",
    "backup and restore",
    "Nutanix Database Service",
    "NDB",
    "Nutanix Enterprise AI",
    "NAI",
    "Agent Gateway",
    "inference management",
    "model deployment",
    "model routing",
    "AI services",
    "Nutanix Flow",
    "Nutanix Disaster Recovery",
    "Move",
    "microsegmentation",
    "overlay networking",
    "security compliance",
    "RBAC",
    "IAM",
    "runbooks",
)


QUERY_INSTRUCTION = (
    "Instruct: Given a technical query about Nutanix documentation, retrieve "
    "relevant passages covering Nutanix Cloud Platform, NCP, NCI, AOS, AHV, "
    "Prism, Nutanix Central, NUS, Nutanix Files, Nutanix Objects, Nutanix "
    "Volumes, NCM, NDB, NKP, NC2, NAI, Flow, Move, disaster recovery, "
    "networking, security, lifecycle operations, configuration, administration, "
    "troubleshooting, performance, APIs, and CLI commands\nQuery: "
)


RERANKER_INSTRUCTION = (
    "Judge whether the document is relevant to the search query about Nutanix "
    "technical documentation. Match the specificity of the query: if it asks "
    "about NCP, NCI, AOS, AHV, Prism, NUS, Files, Objects, Volumes, NCM, NDB, "
    "NKP, NC2, NAI, Flow, Move, disaster recovery, networking, security, "
    "lifecycle operations, APIs, or CLI commands, prioritize documentation "
    "about that exact product, component, or workflow over broad overviews. "
    "Answer only yes or no."
)


RERANKER_INSTRUCTIONS_BY_TYPE: Dict[str, str] = {
    "subsystem_architecture": (
        "Judge whether the document is relevant to the query about Nutanix "
        "platform or subsystem architecture, including NCP, NCI, AOS, AHV, "
        "Prism, NUS, NCM, NDB, NKP, NC2, NAI, Flow, disaster recovery, "
        "storage services, networking, security, or lifecycle internals. "
        "Prioritize deep technical documentation about the specific subsystem "
        "over general overview pages. Answer only yes or no."
    ),
    "resource_sizing": (
        "Judge whether the document is relevant to the query about Nutanix "
        "resource sizing, capacity planning, configuration maximums, licensing "
        "tiers, node counts, CPU, memory, storage, performance, GPU resources, "
        "cloud capacity, or cluster requirements. Prioritize sizing tables, "
        "limits, formulas, and deployment requirements. Answer only yes or no."
    ),
}


NER_LABELS: Tuple[str, ...] = (
    "PRODUCT (e.g. Nutanix Cloud Platform, NCI, NCM, NUS, NDB, NKP, NC2, NAI)",
    "COMPONENT (e.g. AOS, AHV, Prism Central, Files, Objects, Volumes, Flow)",
    "PLATFORM (e.g. on-premises, AWS, Azure, Google Cloud, OVHcloud, edge)",
    "VERSION (e.g. AOS 7.3, PC 2024.x, NAI 2.7)",
    "COMMAND (e.g. ncli, acli, kubectl, nutanix command-line operations)",
    "API (e.g. Prism v4 API, REST endpoint, category API)",
    "PROTOCOL (e.g. NFS, SMB, S3, iSCSI, CSI, COSI)",
    "STORAGE_CONCEPT (e.g. storage container, snapshot, replication, tiering)",
    "CLOUD_PROVIDER (e.g. AWS, Azure, Google Cloud, OVHcloud)",
    "DEPLOYMENT_MODEL (e.g. NCI, NC2, GC2, NCI-Edge, NCI-VDI)",
    "ERROR (e.g. alert, error code, failed task, health check failure)",
    "METRIC (e.g. IOPS, latency, throughput, CPU, memory, usable TiB)",
    "PROCEDURE_STEP (e.g. click Save, run the command, create a cluster)",
)


_DOMAIN_CONFIG = DomainConfig(
    product_name="Nutanix",
    app_name="nutanix-docs-matrix",
    corpus_namespace="nutanix",
    uri_scheme="nutanixdocs",
    cache_prefix="nutanix:cache:v1",
    otel_service_prefix="nutanix",
    product_taxonomy=NUTANIX_TAXONOMY,
    query_instruction=QUERY_INSTRUCTION,
    reranker_instruction=RERANKER_INSTRUCTION,
    reranker_instructions_by_type=RERANKER_INSTRUCTIONS_BY_TYPE,
    ner_labels=NER_LABELS,
)


def get_domain_config() -> DomainConfig:
    return _DOMAIN_CONFIG
