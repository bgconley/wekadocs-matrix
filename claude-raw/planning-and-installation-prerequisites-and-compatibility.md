<!-- ============================================ -->
<!-- File 2/259: planning-and-installation_prerequisites-and-compatibility.md -->
<!-- ============================================ -->

---
description:
---

# Prerequisites and compatibility

Note: **Important:** The versions mentioned on the prerequisites and compatibility page apply to the WEKA system's **latest minor version** (4.4.**X**). For information on new features and supported prerequisites released with each minor version, refer to the relevant release notes available at get.weka.io.
Check the release notes for details about any updates or changes accompanying the latest releases.

Note: In certain instances, WEKA collaborates with Strategic Server Partners to conduct platform qualifications alongside complementary components. If you have any inquiries, contact your designated WEKA representative.

## Minimal server configuration for a WEKA cluster

The minimal configuration for a new WEKA cluster installation is **8 servers**. This ensures optimal performance, resilience, and scalability for most deployments.

Note: For cloud-based installations, WEKA supports a minimal configuration of **6 servers** to accommodate the unique requirements of cloud environments.

## CPU

 | CPU family/architecture | Supported on backends | Supported on clients |
 | --- | --- | --- |
 | 2013 Intel® Core™ processor family and later | 👍Dual-socket | 👍Dual-socket |
 | AMD EPYC™ processor families 2nd (Rome), 3rd (Milan-X), and 4th (Genoa) Generations | 👍Single-socket | 👍 Single-socket and dual-socket |
 | Aarch64 |  | 👍Nvidia Grace |

Note: The following requirements must be met:
* AES is enabled.
* Secure Boot is disabled.
* AVX2 is enabled.

## Memory

* Sufficient memory to support the WEKA system needs as described in [memory requirements](../bare-metal/planning-a-weka-system-installation#memory-resource-planning).
* More memory support for the OS kernel or any other application.

## Operating system

Note: WEKA will support upcoming releases of the operating systems in the lists within one quarter (three months) of their respective General Availability (GA) dates.

* **Rocky Linux:**
  * 9.4, 9.3, 9.2, 9.1, 9.0
  * 8.10, 8.9, 8.8, 8.7, 8.6
* **RHEL:**
  * 9.4, 9.3, 9.2, 9.1, 9.0
  * 8.10, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1, 8.0
* **CentOS:**
  * 8.5, 8.4, 8.3, 8.2, 8.1, 8.0
* **Ubuntu:**
  * 24.04
  * 22.04
  * 20.04
  * 18.04
* **Amazon Linux:**
  * AMI 2018.03
  * AMI 2017.09
* **Amazon Linux 2 LTS** (formerly Amazon Linux 2 LTS 17.12)
  * Latest update package that was tested: 5.10.176-157.645.amzn2.x86_64

* **Rocky Linux:**
  * 9.5, 9.4, 9.3, 9.2, 9.1, 9.0
  * 8.10, 8.9, 8.8, 8.7, 8.6
* **RHEL:**
  * 9.4, 9.3, 9.2, 9.1, 9.0
  * 8.10, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1, 8.0
* **CentOS:**
  * 8.5, 8.4, 8.3, 8.2, 8.1, 8.0
* **Ubuntu:**
  * 24.04
  * 22.04
  * 20.04
  * 18.04
* **Amazon Linux:**
  * AMI 2018.03
  * AMI 2017.09
* **Amazon Linux 2 LTS** (formerly Amazon Linux 2 LTS 17.12)
  * Latest update package that was tested: 5.10.176-157.645.amzn2.x86_64
* **SLES:**
  * 15 DP6
  * 15 SP5
  * 15 SP4
  * 15 SP2
  * 12 SP5
* **Oracle Linux:**
  * 9
  * 8.9
* **Debian:**
  * 12 (with Linux kernel 6.6)
  * 10
* **AlmaLinux OS:**
  * 9.4
  * 8.10
* **Proxmox Virtual Environment**:
  * 8.2
  * 8.14

The following kernel versions are supported:

* 6.8
* 6.0 to 6.5
* 5.3 to 5.19
* 4.4.0-1106 to 4.19
* 3.10

Note: - Kernels 5.15 and higher are not supported with Amazon Linux operating systems.
- It is recommended to turn off auto kernel updates, so it will not get upgraded to an unsupported version.
- Confirm that both the kernel version and the operating system version are listed as supported, as these are distinct components with their own compatibility considerations.
- For clarity, the range of supported versions is inclusive.

#### General

* All WEKA servers must be synchronized in date/time (NTP recommended)
* A watchdog driver should be installed in /dev/watchdog (hardware watchdog recommended); search the WEKA knowledge base in the WEKA support portal for more information and how-to articles.
* If using `mlocate` or alike, it's advisable to exclude `wekafs` from `updatedb` filesystems lists; search the WEKA knowledge base in the WEKA support portal for more information and how-to articles.

#### SELinux

* SELinux is supported in both `permissive` and `enforcing` modes.
  * `The targeted` policy is supported.
  * The `mls` policy is not supported yet.

Note: - To set the SELinux security context for files,  use the `-o acl` in the mount command, and define the `wekafs` to use extended attributes in the SELinux policy configuration (`fs_use_xattr`).
- The maximum size for the Extended Attributes (xattr) is limited to 1024. This attribute is crucial in supporting Access Control Lists (ACL) and Alternate Data Streams (ADS) in SMB. Given its finite capacity, exercise caution when using ACLs and ADS on a filesystem using SELinux.

#### cgroups

* WEKA backends and clients that serve protocols must be deployed on a supported OS with **cgroupsV1**.
* **cgroupsV2** is supported on backends and clients, but not in deployments with protocol clusters.

Note: As of version 4.3.2, RHEL 7.X and CentOS 7.X are no longer supported due to their end-of-life status. If you need assistance upgrading your operating system, contact the [Customer Success Team](../../support/getting-support-for-your-weka-system#contact-customer-success-team) for guidance.

## WEKA installation directory

* **WEKA installation directory**:
  * The WEKA installation directory must be set to `/opt/weka`.
  * Use a direct path; symbolic links (symlinks) are not supported.
  * The `/opt/weka` directory is critical for proper WEKA operation. If it resides on shared storage, the storage must be highly available.
* **Boot drive minimum requirements**:
  * Capacity: NVMe SSD with 960 GB capacity
  * Durability: 1 DWPD (Drive Writes Per Day)
  * Write throughput: 1 GB/s
* **Boot drive considerations**:
  * Do not share the boot drive.
  * Do not mount using NFS.
  * Do not use a RAM drive remotely.
  * If two boot drives are available:
    * It is recommended to dedicate one boot drive for the OS and the other for the `/opt/weka` directory.
    * Do not use software RAID to have two boot drives.
* **Software required space**:
  * Ensure that at least 26 GB is available for the WEKA system installation.
  * Allocate an additional 10 GB per core used by WEKA.
* **Filesystem requirement**:
  * Set a separate filesystem on a separate partition for `/opt/weka`.

## Networking

Adhere to the following considerations when choosing the adapters:

* **LACP****:**  LACP is supported when bonding ports from dual-port Mellanox NICs into a single Mellanox device but is not compatible when using Virtual Functions (VFs).
* **MTU**\
  It is recommended to set the MTU to at least 4k on the NICs of WEKA cluster servers and the connected switches.
* **Jumbo Frames**\
  If any network connection, irrespective of whether it’s InfiniBand or Ethernet, on a given backend possess the capability to transmit frames exceeding 4 KB in size, it is mandatory for all network connections used directly by WEKA on that same backend to have the ability to transmit frames of at least 4 KB.
* **IOMMU** **support**\
  WEKA automatically detects and enables IOMMU for the server and PCI devices. Manual enablement is not required.

Note: When the Linux operating system is configured with `iommu=1`, IOMMU is enabled system-wide, and all PCI devices operate under IOMMU control. It is not possible to selectively exclude specific PCI devices from IOMMU when this mode is active.

* **Shared networking**\
  Shared networking (also known as single IP) allows a single IP address to be assigned to the Physical Function (PF) and shared across multiple Virtual Functions (VFs). This means that a single IP can be shared by every WEKA process on that server, while still being available to the host operating system.
*   **SR-IOV VF**

    Single Root I/O Virtualization Virtual Functions enable direct hardware access for virtual machines, improving network performance by reducing CPU overhead.

Note: Shared networking configuration for NIC models:
* NVIDIA NICs: When implementing Shared Networking (Single IP), Virtual Functions (VFs) are not required.
* Broadcom NICs: VFs must be configured when deploying Shared Networking architecture.

*   **Mixed networks**

    A mixed network configuration refers to a setup where a WEKA cluster connects to both InfiniBand and Ethernet networks.

    Certain features and configurations are not supported in mixed network setups. Review the following limitations and supported settings:

    * **Non-supported features in mixed networks:**
      * RDMA
      * VLAN
      * IPv6
    * **Supported MTU settings in mixed networks:**
      * Ethernet (9000) + InfiniBand (4K)
    * **Non-supported MTU settings in mixed networks:**
      * Ethernet (1500) + InfiniBand (4K)
      * Ethernet (9000) + InfiniBand (2K)
*   **Routed network**

    Enables communication between subnets using Layer 3 routing, allowing WEKA clusters to span multiple network segments.
*   **HA (High Availability)**

    Ensures system uptime through redundant components and automatic failover.
*   **RX Interrupts**

    Receive interrupts that notify the CPU when network packets arrive, critical for optimizing network processing performance.
* **IP addressing for dataplane NICs**\
  Exclusively use static IP addressing. DHCP is not supported for dataplane NICs.
*   **WEKA peer connectivity requires NAT-free networking**

    WEKA requires visibility and connectivity to all peers, without interference from networking technologies like network address translation, or NAT.

**Related topics**

### Supported network adapters <a href="#networking-ethernet" id="networking-ethernet"></a>

The following table provides the supported network adapters along with their supported features for backends and clients, and clients-only.

#### Supported network adapters for backends and clients

 | Adapter | Protocol | Supported features |
 | --- | --- | --- |
 | Amazon ENA | Ethernet | SR-IOV VF |
 | Broadcom BCM957508-P2100GDual-port (2x100Gb/s)Single-port (1x200Gb/s | Ethernet | Shared networkingSR-IOV VFHARouted network |
 | Broadcom BCM957608-P2200GDual-port (2x200Gb/s)Single-port (1x400Gb/s | Ethernet | Shared networkingSR-IOV VFHARouted network |
 | NVIDIA Mellanox CX-7 single-port | InfiniBand | Shared networkingRX interruptsRDMAHAPKEYIOMMU |
 | NVIDIA Mellanox CX-7 dual-port | InfiniBand | Shared networkingRX interruptsRDMAHAPKEYIOMMU |
 | NVIDIA Mellanox CX-7-ETH single-port | Ethernet | Shared networkingRDMAHARouted network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-7-ETH dual-port | Ethernet | LACPShared networkingRDMAHARouted network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-6 LX | Ethernet | Shared networkingRDMARX interruptsHARouted network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-6 DX | Ethernet | LACPShared networkingRX interruptsRDMAHARouted network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-6 | Ethernet InfiniBand | Mixed networksShared networkingRX interruptsRDMAHAIOMMU |
 | NVIDIA Mellanox CX-5 EX | Ethernet InfiniBand | Mixed networksRDMAHAPKEY (IB only)IOMMU |
 | NVIDIA Mellanox CX-5 BF | Ethernet | Mixed networksRDMAHAIOMMU |
 | NVIDIA Mellanox CX-5 | Ethernet InfiniBand | Mixed networksRX interruptsRDMAHAPKEY (IB only)Routed network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-4 LX | Ethernet InfiniBand | Mixed networksRX interruptsHARouted network (ETH only)IOMMU |
 | NVIDIA Mellanox CX-4 | Ethernet InfiniBand | Mixed networksRX interruptsHARouted network (ETH only)IOMMU |
 | VirtIO | Ethernet | HARouted network |

#### Supported network adapters for clients-only

The following network adapters support Ethernet and SRIOV VF for clients only:

* Intel X540
* Intel X550-T1 (avoid using this adapter in a single client connected to multiple clusters)
* Intel X710
* Intel X710-DA2
* Intel XL710
* Intel XL710-Q2
* Intel XXV710
* Intel 82599ES
* Intel 82599

### Ethernet drivers and configurations

*   **Supported Mellanox OFED versions for the Ethernet NICs:**

    * 24.04-0.7.0.0
    * 23.10-0.5.5.0
    * 23.04-1.1.3.0
    * 5.9-0.5.6.0
    * 5.8-1.1.2.1 LTS
    * 5.8-3.0.7.0
    * 5.7-1.0.2.0
    * 5.6-2.0.9.0
    * 5.6-1.0.3.3
    * 5.4-3.5.8.0 LTS
    * 5.4-3.4.0.0 LTS
    * 5.1-2.6.2.0
    * 5.1-2.5.8.0

    **Note:** Subsequent OFED minor versions are expected to be compatible with Nvidia hardware due to Nvidia's commitment to backwards compatibility.
* **Supported ENA drivers:**
  * 1.0.2 - 2.0.2
  * A current driver from an official OS repository is recommended
* **Supported ixgbevf drivers:**
  * 3.2.2 - 4.1.2
  * A current driver from an official OS repository is recommended
* **Supported Broadcom drivers**:
  * 228: Minimum required for 100/200 Gbps 57508 NIC
  * 231: Minimum required for 200/400 Gbps 57608 NIC

* **Ethernet speeds:**
  * 400 GbE / 200 GbE / 100 GbE / 50GbE / 40 GbE / 25 GbE / 10 GbE.
* **NICs bonding:**
  * Supports bonding dual ports on the same NVIDIA Mellanox NIC using mode 4 (LACP) to enhance redundancy and performance.
* **IEEE 802.1Q VLAN encapsulation:**
  * Supports VLAN tagging with a single VLAN tag on NVIDIA Mellanox NICs.
* **VXLAN:**
  * Virtual Extensible LANs are not supported.
* **DPDK backends and clients using NICs supporting shared networking (single IP):**
  * Require one IP address per client for both management and data plane.
  * SR-IOV enabled is not required.
* **DPDK backends clients using NICs supporting non-shared networking:**
  * IP address for management: One per NIC (configured before WEKA installation).
  * IP address for data plane: One per [WEKA core](../bare-metal/planning-a-weka-system-installation#cpu-resource-planning) in each server (applied during cluster initialization).
  * Virtual Functions (VFs):
    * Ensure the device supports a maximum number of VFs greater than the number of physical cores on the server.
    * Set the number of VFs to match the cores you intend to dedicate to WEKA.
    * Note that some BIOS configurations may be necessary.
  * SR-IOV: Enabled in BIOS.
* **UDP clients:**
  * Use a single IP address for all purposes.

Note: When assigning a network device to the WEKA system, no other application can create VFs on that device.

### InfiniBand drivers and configurations <a href="#networking-infiniband" id="networking-infiniband"></a>

WEKA supports the following Mellanox OFED versions for the InfiniBand adapters:

* 24.04-0.7.0.0
* 23.10-0.5.5.0
* 23.04-1.1.3.0
* 5.9-0.5.6.0
* 5.8-1.1.2.1 LTS
* 5.8-3.0.7.0
* 5.7-1.0.2.0
* 5.6-2.0.9.0
* 5.6-1.0.3.3
* 5.4-3.5.8.0 LTS
* 5.4-3.4.0.0 LTS
* 5.1-2.6.2.0
* 5.1-2.5.8.0

**Note:** Subsequent OFED minor versions are expected to be compatible with Nvidia hardware due to Nvidia's commitment to backwards compatibility.

WEKA supports the following InfiniBand configurations:

* InfiniBand speeds: Determined by the InfiniBand adapter supported speeds (FDR / EDR / HDR / NDR).
* Subnet manager: Configured to 4092.
* One WEKA system IP address for management and data plane.
* PKEYs: One partition key is supported by WEKA.
* Redundant InfiniBand ports can be used for both HA and higher bandwidth.

Note: If it is necessary to change PKEYs, contact the [Customer Success Team](../../support/getting-support-for-your-weka-system#contacting-weka-technical-support-team).

### Required ports

When configuring firewall ingress and egress rules the following access must be allowed.

Note: Right-scroll the table to view all columns.

 | Purpose | Source | Target | Target Ports | Protocol | Comments |
 | --- | --- | --- | --- | --- | --- |
 | WEKA server traffic for bare-metal deployments | All WEKA backend IPs | All WEKA backend IPs | 14000-14100 (drives)14200-14300 (frontend)14300-14400 (compute) | TCP and UDPTCP and UDPTCP and UDP | These ports are the default for the Resources Generator for the first three containers. You can customize the ports. |
 | WEKA client traffic | Client host IPs | All WEKA backend IPs | 14000-14100 (drives)14300-14400 (compute) | TCP and UDPTCP and UDP | These ports are the default. You can customize the ports. |
 | WEKA backend to client traffic | All WEKA backend IPs | Client host IPs | 14000-14100 (frontend) | TCP and UDP | These ports are the default. You can customize the ports. |
 | WEKA SSH management traffic | All WEKA backend IPs | All WEKA backend IPs | 22 | TCP |  |
 | WEKA server traffic for cloud deployments | All WEKA backend IPs | All WEKA backend IPs | 14000-14100 (drives)15000-15100 (compute)16000-16100 (frontend) | TCP and UDPTCP and UDPTCP and UDP | These ports are the default. You can customize the ports. |
 | WEKA client traffic (on cloud) | Client host IPs | All WEKA backend IPs | 14000-14100 (drives)15000-15100 (compute) | TCP and UDPTCP and UDP | These ports are the default. You can customize the ports. |
 | WEKA backend to client traffic (on cloud) | All WEKA backend IPs | Client host IPs | 14000-14100 (frontend) | TCP and UDP | These ports are the default. You can customize the ports. |
 | WEKA GUI access | Admin workstation IPs | All WEKA management IPs | 14000 | TCP | User web browser IP |
 | NFS | NFS client IPs | WEKA NFS backend IPs | 2049<mountd port> | TCP and UDPTCP and UDP | You can set the mountd port using the command: weka nfs global-config set --mountd-port |
 | NFSv3 (used for locking) | NFS client IPs | WEKA NFS backend IPs | 46999 (status monitor)47000 (lock manager) | TCP and UDP |  |
 | SMB/SMB-W | SMB client IPs | WEKA SMB backend IPs | 139445 | TCPTCP |  |
 | SMB-W | All WEKA SMB-W backend IPs | All WEKA SMB-W backend IPs | 2224 | TCP | This port is required for internal clustering processes. |
 | SMB/SMB-W | WEKA SMB backend IPs | All Domain Controllers for the selected Active Directory Domain | 8838946463632683269 | TCP and UDPTCP and UDPTCP and UDPTCP and UDPTCP and UDPTCP and UDP | These ports are required for SMB/SMB-W to use Active Directory as the identity source. Furthermore, every Domain Controller within the selected AD domain must be accessible from the WEKA SMB servers. |
 | SMB/SMB-W | WEKA SMB backend IPs | DNS servers | 53 | TCP and UDP |  |
 | S3 | S3 client IPs | WEKA S3 backend IPs | 9000 | TCP | This port is the default. You can customize the port. |
 | wekatester | All WEKA backend IPs | All WEKA backend IPs | 85019090 | TCPTCP | Port 8501 is used by wekanetperf. |
 | WEKA Management Station | User web browser IP | WEKA Management Station IP | 80 <LWH>443 <LWH>3000 <mon>7860 <admin UI>8760 <deploy>8090 <snap>8501 <mgmt>9090 <mgmt>9091 <mon>9093 <alerts> | HTTPHTTPSTCPTCPTCPTCPTCPTCPTCP |  |
 | Cloud WEKA Home, Local WEKA Home | All WEKA backend IPs | Cloud WEKA Home or Local WEKA Home | 80443 | HTTPHTTPS | Open according to the directions in the deployment scenario:- WEKA server IPs to CWH or LWH.- LWH to CWH (if forwarding data from LWH to CWH) |
 | Troubleshooting by the Customer Success Team (CST) | All WEKA backend IPs | CST remote access | 40004001 | TCPTCP |  |
 | Traces remote viewer | All WEKA backend IPs | CST remote access | 443 | TCP |  |
 | KMS: Hashicorp Vault | All WEKA backend IPs | Hashicorp Vault server | 82008201 | TCPTCP | Default vault ports: 8200 is configurable for client requests, while 8201 (base_port+1) handles internal cluster communication. |
 | KMS: KMIP | All WEKA backend IPs | KMIP server | 5696 | TCP | The default KMIP port, 5696, is configurable. Per the KMIP specification, servers must use this port when operating with the TTLV encoding format. |

## HA

See #high-availability

## SSDs

* The SSDs must support PLP (Power Loss Protection).
* WEKA system storage must be dedicated, and partitioning is not supported.
* The supported drive capacity is up to 30 TB.
* The ratio between the cluster's smallest and the largest SSD capacity must not exceed 8:1.

Note: To get the best performance, ensure TRIM) is supported by the device and enabled in the operating system.

## Object store

* API must be S3 compatible:
  * GET
    * Including byte-range support with expected performance gain when fetching partial objects
  * PUT
    * Supports any byte size of up to 65 MiB
  * DELETE
* Data Consistency: Amazon S3 consistency model:
  * GET after a single PUT is strongly consistent
  * Multiple PUTs are eventually consistent

### Certified object stores

* Amazon S3
  * S3 Standard
  * S3 Intelligent-Tiering
  *   These storage classes are ideal for remote buckets where data is written once and accessed in critical situations, such as during disaster recovery:

      * S3 Standard-IA
      * S3 One Zone-IA
      * S3 Glacier Instant Retrieval

      Remember, retrieval times, minimum storage periods, and potential charges due to object compaction may apply. If unsure, use S3 Intelligent-Tiering.
* Azure Blob Storage
* Google Cloud Storage (GCS)
* Cloudian HyperStore (version 7.3)
* Dell EMC ECS (version 3.5)
* Dell PowerScale S3 (version 9.8.0.0)
* HCP Classic V9.2 and up (with versioned buckets only)
* HCP for Cloud-Scale V2.x
* IBM Cloud Object Storage System (version 3.14.7)
* Lenovo MagnaScale (version 3.0)
* Quantum ActiveScale (version 5.5.1)
* Red Hat Ceph Storage (version 5.0)
* Scality RING with S3 connector (version 8.5)
* Scality RING with WEKA connector (version 9.5)
* Scality Artesca (version 1.5.2)
* SwiftStack (version 6.30)
* WEKA S3

## Virtual Machines

This section outlines the use of virtual machines (VMs) with WEKA, covering backends, clients, VMware platforms, and cloud environments. While VMs can be used in certain configurations, there are specific limitations and best practices to follow.

### Backends

Virtual machines may be used as backends for internal training purposes only and are not recommended for production environments.

WEKA provides best-effort support for backends deployed on virtual machines, but full support is not guaranteed. Additionally, WEKA does not guarantee support for components or configurations outside of our documented and supported cloud environments, and performance may vary.

### Clients

Virtual Machines (VMs) can be used as clients. Ensure the following prerequisites are met for each client type:

* **UDP clients**:
  * Reserve CPU resources and dedicate a core to the client to prevent CPU starvation of the WEKA process.
  * Ensure the root filesystem supports a 3K IOPS load for the WEKA client.
* **DPDK clients**:
  * Meet all the requirements for UDP clients.
  * Additionally, verify that the virtual platform (hypervisor, NICs, CPUs, and their respective versions) fully supports DPDK and the required virtual network drivers.

### **VMware platform (**&#x63;lient only)

When using **vmxnet3** devices, do not enable the SR-IOV feature, because it disables the vMotion functionality. Each frontend process requires a dedicated **vmxnet3** device and IP address, with an additional device and IP for each client VM to support the management process.

Core dedication is required when using **vmxnet3** devices.

### VMs and instances on cloud environments

Refer to the cloud deployment sections for the most up-to-date list of supported virtual machines and instances in various cloud environments.

**Related topics**

AWS:

Azure:

GCP:

\
**Related information**

For additional information and how-to articles, search the WEKA Knowledge Base in the WEKA support portal or contact the [Customer Success Team](../../support/getting-support-for-your-weka-system#contacting-weka-technical-support-team).

## KMS

**Supported KMS types:**

* **KMIP-compliant KMS****:** Supports protocol versions 1.2+ and 2.x. Only TTLV is supported as the messaging protocol. Supports commercial solutions such as Thales CipherTrust Manager.
* **HashiCorp Vault****:** Supports versions 1.1.5 to 1.14.x.
