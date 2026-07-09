from dataclasses import replace

from docpipe.clean import (
    RunMeta,
    balance_fences,
    clean_document,
    demote_extra_h1s,
    ensure_title_h1,
    first_h1_text,
    resolve_title,
)
from docpipe.contract import validate_contract
from docpipe.fences import iter_lines_with_fence_state
from docpipe.manifest import DocRecord

_BASE = DocRecord(
    pdf_path="x.pdf",
    rel_path="x.pdf",
    sha256="deadbeef",
    size_bytes=10,
    page_count=2,
    slug="ahv-admin-guide",
    title=None,
    doc_version="7.5",
    product="AOS",
)


def _rec(**kw) -> DocRecord:
    return replace(_BASE, **kw)


def test_prepend_h1_when_missing():
    out = ensure_title_h1("some text\n## Section", "My Title")
    assert out.startswith("# My Title\n")
    assert "## Section" in out


def test_keep_existing_leading_h1():
    out = ensure_title_h1("# Existing\n\nbody", "Ignored")
    assert out.startswith("# Existing")
    assert "# Ignored" not in out


def test_single_h1_enforced():
    assert demote_extra_h1s("# A\n# B\n# C") == "# A\n## B\n## C"


def test_resolve_title_prefers_body_h1():
    assert (
        resolve_title("# NFS Networking\n\nx", _rec(title="Embedded Title"))
        == "NFS Networking"
    )
    assert (
        resolve_title("no heading here", _rec(title="Embedded Title"))
        == "Embedded Title"
    )
    assert resolve_title("no heading", _rec(title=None)) == "Ahv Admin Guide"


def test_balance_fences_closes_before_heading():
    out = balance_fences("## A\n\n```text\nsome code\n## B\n\nmore")
    assert out.count("```") == 2
    lines = out.splitlines()
    assert "```" in lines[: lines.index("## B")]  # closed before B


def test_balance_fences_closes_at_eof():
    out = balance_fences("intro\n```bash\ncmd")
    assert out.count("```") == 2
    assert out.rstrip().endswith("```")


def test_balance_fences_ignores_shell_comment():
    body = "```bash\n# a shell comment\nls\n```\n\ntext"
    assert balance_fences(body) == body  # single-# is not a heading; nothing changes


# --- Cluster 1: fence-awareness (findings #2, #3, #22) -------------------------


def test_demote_extra_h1s_skips_fenced_comments():
    # Finding #2: a '#' comment inside a code fence must NOT be demoted to '##'
    # (which would leak the command out of the block and inject a false heading).
    body = "# Title\n\n```bash\n# a comment\ncmd\n```\n\n# Second"
    out = demote_extra_h1s(body).splitlines()
    assert "# a comment" in out  # unchanged, still fenced content
    assert "## a comment" not in out
    assert "## Second" in out  # a real second H1 is still demoted


def test_balance_fences_ignores_h2_comment_in_balanced_block():
    # Finding #3: a '##' comment inside a properly-closed fence must not trigger
    # a premature close. Balanced (even) fence count => leave the doc untouched.
    body = (
        "## Section\n\n```bash\n## configure interface\n"
        "ip addr add 1.2.3.4/24 dev eth0\n```\n"
    )
    assert balance_fences(body) == body


def test_first_h1_text_skips_fenced_comment():
    # Finding #22: a '#' comment in a leading code block must not become the title.
    body = (
        "```bash\n# Deploy the production cluster\n"
        "ncli cluster create name=prod\n```\n\n## Overview\nbody"
    )
    assert first_h1_text(body) is None
    assert first_h1_text("# Real Title\n\n```bash\n# c\n```") == "Real Title"


def test_clean_document_preserves_fenced_comments_end_to_end():
    # Finding #2/#3 together, through the real clean_document pipeline order
    # (demote_extra_h1s -> balance_fences): the fenced block stays intact.
    body = (
        "## Host Networking\n\n```bash\n# create the vlan-backed network\n"
        "acli net.create vlan0 vlan=0\n# verify it\nacli net.list\n```\n"
    )
    md, _ = clean_document(body, _rec(), RunMeta(model_id="m", endpoint="e", dpi=200))
    assert md.count("```") == 2  # exactly one intact block
    assert "# create the vlan-backed network" in md  # comment kept as-is
    assert "## create the vlan-backed network" not in md
    assert "acli net.create vlan0 vlan=0" in md  # command still inside code


def test_clean_document_fences_unfenced_cli_command_groups():
    md, _title = clean_document(
        "# AHV Commands\n\n"
        "Use the following commands.\n\n"
        "ncli storage list\n"
        "ncli storage get name=default\n\n"
        "nutanix@cvm$ manage_ovs show_bridges\n"
        "<acropolis> ovs-vsctl show\n"
        "ncli> cluster info\n"
        "$ ssh nutanix@example\n\n"
        "A price like $5 is prose, not a shell prompt.\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=200),
    )
    body = md.split("---\n", 2)[2]

    assert "```bash\nncli storage list\nncli storage get name=default\n```" in body
    assert (
        "```bash\n"
        "nutanix@cvm$ manage_ovs show_bridges\n"
        "<acropolis> ovs-vsctl show\n"
        "ncli> cluster info\n"
        "$ ssh nutanix@example\n"
        "```" in body
    )
    assert "A price like $5 is prose" in body


def test_clean_document_does_not_refence_existing_cli_blocks():
    md, _title = clean_document(
        "# AHV Commands\n\n```bash\nncli storage list\n```\n\nDone.",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=200),
    )
    body = md.split("---\n", 2)[2]

    assert body.count("```bash") == 1
    assert body.count("```") == 2
    assert "```bash\n```bash" not in body


def test_clean_document_lifts_indented_fences_to_contract_safe_top_level():
    md, _title = clean_document(
        "# AHV Maintenance\n\n"
        "1. Run the status command.\n\n"
        "    ```bash\n"
        "    nutanix@cvm$ cluster status\n"
        "    ```\n\n"
        "    The output appears after the command.\n\n"
        "        - **Name**: cluster status\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert "\n```bash\nnutanix@cvm$ cluster status\n```\n" in md
    assert "\n    ```" not in md
    assert "\nThe output appears after the command." in md
    assert "\n- **Name**: cluster status" in md


def test_clean_document_lifts_fenced_shell_comments_before_h1_demote():
    md, _title = clean_document(
        "# Nutanix Kubernetes Engine Guide\n\n"
        "6. Verify that services have started on all etcd nodes.\n\n"
        "    ```bash\n"
        "    # export ETCD_IP_0=<replace with etcd 0 IP address>\n"
        "    # etcdctl -w table endpoint status\n"
        "    ```\n\n"
        "# Deleting a Private Registry\n\n"
        "Body.\n",
        _rec(slug="nutanix-kubernetes-engine-guide"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    h1s = [
        line
        for line, in_fence in iter_lines_with_fence_state(md)
        if not in_fence and line.startswith("# ")
    ]
    assert h1s == ["# Nutanix Kubernetes Engine Guide"]
    assert "```bash\n# export ETCD_IP_0=<replace with etcd 0 IP address>" in md
    assert "\n## Deleting a Private Registry" in md


def test_clean_document_lifts_blockquoted_fence_commands_to_top_level():
    md, _title = clean_document(
        "# NKE Deployment\n\n"
        "> - To list the Prism Element UUID, run the following command.\n"
        ">\n"
        '> ```bashnutanix@CVM:~$ ncli cluster info |grep "Cluster Uuid"\n'
        "> ```\n",
        _rec(slug="nutanix-kubernetes-engine-guide"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert '```bash\nnutanix@CVM:~$ ncli cluster info |grep "Cluster Uuid"\n```' in md
    assert "> ```" not in md


def test_clean_document_outdents_blocks_revealed_after_fence_repair():
    md, _title = clean_document(
        "# AHV Repair\n\n"
        "```text\n"
        "output that forgot to close\n\n"
        "## Follow Up\n\n"
        "    a. AHV host IP address.\n\n"
        "        Check the AHV host IP address.\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert "\na. AHV host IP address." in md
    assert "\nCheck the AHV host IP address." in md


def test_clean_document_collapses_duplicated_fences_before_outdenting():
    md, _title = clean_document(
        "# AHV Startup\n\n"
        "```bash\n"
        "```bash\n"
        "nutanix@cvm$ cluster start\n"
        "```\n"
        "```\n\n"
        "    The cluster services start after the command.\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert md.count("```bash") == 1
    assert "\n```bash\nnutanix@cvm$ cluster start\n```\n" in md
    assert "\nThe cluster services start after the command." in md


def test_clean_document_repairs_indented_code_blocks_after_malformed_fences():
    md, _title = clean_document(
        "# AHV Startup\n\n"
        "```text\n"
        "open output\n"
        "```bash\n"
        "nutanix@cvm$ cluster start\n"
        "```\n\n"
        "    a. Continue with the next verification step.\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert "\na. Continue with the next verification step." in md


def test_clean_document_converges_after_nested_fence_reveals_later_indents():
    md, _title = clean_document(
        "# AHV Maintenance\n\n"
        "```\n"
        "service output\n"
        "```bash\n"
        "nutanix@cvm$ sudo shutdown -P now\n"
        "```\n\n"
        "    b. Ping each CVM (ping `cvm_ip_addr`) to verify shutdown.\n"
        "6. Shut down each node in the cluster.\n\n"
        "    a. Log on to the IPMI web console of each node.\n\n"
        "    b. Under **Remote Control** > **Power Control**, select **Power Off Server - Orderly Shutdown**.\n\n"
        "    > **Note:** The IPMI web console layout can change.\n\n"
        "    c. Ping each host (`ping hypervisor_ip_addr`) to verify shutdown.\n\n"
        "7. Complete the maintenance activity.\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert "\na. Log on to the IPMI web console of each node." in md
    assert "\n> **Note:** The IPMI web console layout can change." in md


def test_clean_document_completes_pipe_table_rows():
    md, _title = clean_document(
        "# NC2 Guide\n\n"
        "| Revision Date | Revision Description\n"
        "| :--- | :--- |\n"
        "| June 15, 2026 | Updated deployment guidance. |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| Revision Date | Revision Description |" in md
    assert "| June 15, 2026 | Updated deployment guidance. |" in md


def test_clean_document_normalizes_extra_separator_cells():
    md, _title = clean_document(
        "# AOS Guide\n\n"
        "| Module or Service | Module name | Disabled | Enabled |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| API Audit | api_audit | api_audit.log | api_audit.log |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| :--- | :--- | :--- | :--- |\n" in md
    assert "| :--- | :--- | :--- | :--- | :--- |" not in md


def test_clean_document_pads_short_table_rows_to_header_width():
    md, _title = clean_document(
        "# AHV Guide\n\n"
        "| Parameter | Description | Values | Values |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| Name | Displays the policy name. | (name) |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| Name | Displays the policy name. | (name) |  |" in md


def test_clean_document_merges_surplus_table_cells_to_expected_column():
    md, _title = clean_document(
        "# NKE Guide\n\n"
        "| Service | Requests | Limits | Replicas |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| Prometheus | CPU: 110m | CPU: 600m | Memory: 240Mi | 1 overall |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| Prometheus | CPU: 110m | CPU: 600m \\| Memory: 240Mi | 1 overall |" in md


def test_clean_document_repairs_malformed_separator_rows():
    md, _title = clean_document(
        "# NKE Guide\n\n"
        "| Parameter | Description |\n"
        "| :--- | : | : : |\n"
        "| Name | The name of the node. |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| :--- | :--- |" in md
    assert "| :--- | : | : : |" not in md


def test_clean_document_repairs_colon_only_separator_rows():
    md, _title = clean_document(
        "# NKE Guide\n\n"
        "| Parameter | Description |\n"
        "| : : | : : |\n"
        "| Name | The name of the node. |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| :--- | :--- |" in md
    assert "| : : | : : |" not in md


def test_clean_document_adds_header_for_headerless_parameter_rows():
    md, _title = clean_document(
        "# NAI Guide\n\n"
        "| naiAgent.agentImage.image | NAI Agent app image name | docker.io/nutanix/nai-agent-app |\n"
        "| naiAgent.agentImage.tag | NAI Agent app image tag | v2.7.0 |\n",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "| Key | Description | Default Value |" in md
    assert "| :--- | :--- | :--- |" in md


def test_clean_document_strips_leading_table_of_contents_noise():
    md, _title = clean_document(
        "# NC2 Azure Guide\n\n"
        "Cloud Clusters (NC2) Hosted\n"
        "July 6, 2026\n\n"
        "# Contents\n\n"
        "**Setting Up Azure Tenant** ........................................ 43\n"
        "- Creating an App Registration .................................... 45\n"
        "  - Creating an Azure Custom Role For Primary Subscription ......... "
        "|   |   |   |   |   |...|...||---||---|||||||||| || || ||\n"
        "Creating a New Client Secret                       # # # # # # #\n\n"
        "NC2 Licensing and Billing.......................................... 56\n\n"
        "## NC2 Cluster Management..........................................128\n\n"
        "- NC2 Management Console........................................... 128\n\n"
        "## ABOUT NC2 ON AZURE DEPLOYMENT AND USER GUIDE\n\n"
        "This guide covers deploying Nutanix Cloud Clusters on Azure.\n",
        _rec(slug="nutanix-cloud-clusters-azure-4"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert validate_contract(md).ok
    assert "## Contents" not in md
    assert "||---|" not in md
    assert "# # # # # # #" not in md
    assert (
        "NC2 Cluster Management..........................................128" not in md
    )
    assert "## ABOUT NC2 ON AZURE DEPLOYMENT AND USER GUIDE" in md
    assert "This guide covers deploying Nutanix Cloud Clusters on Azure." in md


def test_clean_document_strips_heading_entries_inside_leading_toc():
    md, _title = clean_document(
        "# AHV Administration Guide\n\n"
        "AHV 11.0\n"
        "July 1, 2026\n\n"
        "NUTANIX\n\n"
        "## Contents\n\n"
        "## AHV Overview\n\n"
        "- Storage Overview................................................................7\n"
        "AHV Turbo........................................................................8\n\n"
        "## Node Management\n\n"
        "Controller VM Access............................................................17\n\n"
        "## AHV Overview\n\n"
        "AHV is the Nutanix hypervisor for running user VMs.\n",
        _rec(slug="ahv-admin-guide-v11-0"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "## Contents" not in md
    assert (
        "Storage Overview................................................................7"
        not in md
    )
    assert "## Node Management" not in md
    assert "## AHV Overview" in md
    assert "AHV is the Nutanix hypervisor for running user VMs." in md


def test_clean_document_strips_short_page_refs_inside_leading_toc():
    md, _title = clean_document(
        "# Nutanix Enterprise AI Guide\n\n"
        "Nutanix Enterprise AI 2.7\n"
        "June 18, 2026\n\n"
        "## Contents\n\n"
        "**About this Publication** . . . . . . . . . . . . . . . . . . 6\n"
        "Nutanix Enterprise AI Overview. 7\n"
        "Deploy Nutanix Enterprise AI. 21\n"
        "**Viewing the Dashboard in Nutanix Enterprise AI** ........... **133**\n"
        "Dashboard Widgets................................................ **133**\n\n"
        "## About this Publication\n\n"
        "This publication describes Nutanix Enterprise AI.\n",
        _rec(slug="nutanix-enterprise-ai-v2-7"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "## Contents" not in md
    assert "Nutanix Enterprise AI Overview. 7" not in md
    assert (
        "Dashboard Widgets................................................ **133**"
        not in md
    )
    assert "## About this Publication" in md
    assert "This publication describes Nutanix Enterprise AI." in md


def test_clean_document_strips_wrapped_plain_heading_entries_inside_leading_toc():
    md, _title = clean_document(
        "# Nutanix Enterprise AI Guide\n\n"
        "Nutanix Enterprise AI 2.7\n"
        "June 18, 2026\n\n"
        "## Contents\n\n"
        "**Getting Started with Nutanix Enterprise AI** . . . . . . . . . . 7\n"
        "Configuring OpenTelemetry Collector to view Nutanix Enterprise AI\n"
        "   Metrics................................................................249\n"
        "Nutanix Enterprise AI Support Bundle......................................261\n"
        "Copyright................................................................266\n\n"
        "## ABOUT THIS PUBLICATION\n\n"
        "This document provides information about Nutanix Enterprise AI.\n",
        _rec(slug="nutanix-enterprise-ai-v2-7"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "## Contents" not in md
    assert "Configuring OpenTelemetry Collector to view Nutanix Enterprise AI" not in md
    assert (
        "Metrics................................................................249"
        not in md
    )
    assert (
        "Copyright................................................................266"
        not in md
    )
    assert "## ABOUT THIS PUBLICATION" in md
    assert "This document provides information about Nutanix Enterprise AI." in md


def test_clean_document_keeps_real_contents_section_without_toc_evidence():
    md, _title = clean_document(
        "# Admin Guide\n\n"
        "## Contents\n\n"
        "This section describes package contents and supported files.\n\n"
        "## Install\n\n"
        "Install the package from the downloaded archive.\n",
        _rec(slug="admin-guide"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "## Contents" in md
    assert "This section describes package contents and supported files." in md
    assert "## Install" in md


def test_clean_document_keeps_real_contents_section_with_wide_table():
    md, _title = clean_document(
        "# Package Guide\n\n"
        "## Contents\n\n"
        "| Name | Type | Default | Minimum | Maximum | Units | Required | Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| cache_size | integer | 8 | 1 | 64 | GiB | yes | Runtime cache size. |\n\n"
        "## Install\n\n"
        "Install the package from the downloaded archive.\n",
        _rec(slug="package-guide"),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="oxcart", dpi=218),
    )

    assert "## Contents" in md
    assert (
        "| Name | Type | Default | Minimum | Maximum | Units | Required | Notes |" in md
    )
    assert (
        "| cache_size | integer | 8 | 1 | 64 | GiB | yes | Runtime cache size. |" in md
    )
    assert "## Install" in md


def test_clean_document_contract():
    md, title = clean_document(
        "# Hello World\n\nbody\n\n\n\n\nmore",
        _rec(),
        RunMeta(model_id="qwen36-27b-fp8-oxcart", endpoint="blackbird", dpi=200),
    )
    assert title == "Hello World"
    assert md.startswith("---\n")  # front matter at byte 0
    assert 'title: "Hello World"' in md
    assert 'version: "7.5"' in md
    assert 'sha256: "deadbeef"' in md
    assert "pipeline_version:" in md
    body = md.split("---\n", 2)[2]
    assert body.lstrip().startswith("# Hello World")  # body opens with the single H1
    assert sum(1 for ln in md.splitlines() if ln.startswith("# ")) == 1
    assert "\n\n\n" not in md  # blank runs collapsed


def test_clean_document_emits_parser_consumed_last_edited():
    run = RunMeta(
        model_id="qwen36-27b-fp8-oxcart",
        endpoint="oxcart",
        dpi=200,
        extracted_at="2026-07-08T02:50:00Z",
    )

    md, _title = clean_document("# Hello World\n\nbody", _rec(), run)

    assert 'last_edited: "2026-07-08T02:50:00Z"' in md
