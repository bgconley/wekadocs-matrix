#!/usr/bin/env python3
"""Generate a truly massive test document that exceeds 8192 tokens."""

content = """# Massive Reference Document - Complete API and Configuration Specification

This document tests Phase 1 truncation handling with genuinely oversized content that exceeds 8192 tokens.
Target: ~15,000 tokens to thoroughly test truncation behavior.

## Section 1: Complete API Reference

### REST API Endpoints Documentation

"""

# Generate extensive API documentation (100 endpoints)
for i in range(1, 101):
    content += f"""
#### Endpoint {i}: /api/v1/resource{i}

**Description**: This endpoint provides comprehensive access to resource{i} with full CRUD operations, authentication, and rate limiting.

**GET /api/v1/resource{i}**
- Description: Retrieve resource{i} by ID or list all available resources with pagination support
- Authentication: Required (Bearer token in Authorization header)
- Rate Limit: 1000 requests per hour per user
- Request Parameters:
  - id (optional): Resource unique identifier in UUID v4 format
  - limit (optional): Number of results to return (range: 1-100, default: 20)
  - offset (optional): Pagination offset for result sets (default: 0)
  - filter (optional): JSON-formatted filter criteria for advanced queries
  - sort (optional): Sort field and direction (field:asc or field:desc)
- Response Codes: 200 OK, 400 Bad Request, 401 Unauthorized, 404 Not Found, 429 Rate Limit Exceeded
- Response Body: JSON array of resource{i} objects with pagination metadata

**POST /api/v1/resource{i}**
- Description: Create a new resource{i} instance with validation and persistence
- Authentication: Required (Bearer token plus 'write' permission scope)
- Rate Limit: 100 requests per hour per user (stricter for write operations)
- Request Headers: Content-Type must be application/json
- Request Body: JSON object containing all required fields for resource{i} creation
- Response Codes: 201 Created, 400 Bad Request, 401 Unauthorized, 403 Forbidden
- Response Body: Created resource{i} object including generated ID

**PUT /api/v1/resource{i}/{{id}}**
- Description: Update an existing resource{i} instance (full replacement semantics)
- Authentication: Required with appropriate permissions
- Request Body: Complete JSON object representing the updated resource state

**DELETE /api/v1/resource{i}/{{id}}**
- Description: Permanently delete resource{i} instance (irreversible operation)
- Authentication: Required with 'delete' permission
- Response Codes: 204 No Content, 401 Unauthorized, 404 Not Found

"""

# Add extensive configuration section
content += """

## Section 2: System Configuration Parameters

This comprehensive section documents all configuration parameters available in the system.

### Configuration Categories

"""

categories = [
    "Performance",
    "Security",
    "Networking",
    "Storage",
    "Monitoring",
    "Authentication",
]
param_num = 1

for category in categories:
    content += f"""
### Category: {category} Configuration
"""
    for i in range(1, 26):  # 25 params per category = 150 total
        param_type = ["string", "integer", "boolean", "float", "array"][param_num % 5]
        content += f"""
#### Parameter {param_num}: {category.lower()}_setting_{i}

**Full Name**: system.{category.lower()}.setting_{i}
**Type**: {param_type}
**Default**: {'true' if param_num % 2 == 0 else 'false' if param_type == 'boolean' else str(param_num * 100)}
**Description**: This parameter controls critical {category.lower()} functionality for subsystem {i}. Proper configuration is essential for optimal system operation affecting resource allocation, performance characteristics, security posture, and operational reliability.
**Impact**: {'High - directly affects throughput' if i % 3 == 0 else 'Medium - affects efficiency' if i % 3 == 1 else 'Low - minimal impact'}
**Tuning**: For production environments, adjust this parameter based on workload characteristics, available system resources, and observed performance metrics. Start with default values and make incremental adjustments while monitoring system behavior.
**Related**: {category.lower()}_setting_{i-1 if i > 1 else 25}, {category.lower()}_setting_{i+1 if i < 25 else 1}

"""
        param_num += 1

# Add troubleshooting section
content += """

## Section 3: Comprehensive Troubleshooting Guide

### Common Issues and Resolutions

"""

issues = [
    ("High Latency", "System response time exceeds acceptable thresholds"),
    ("Low Throughput", "Data transfer rates below expected performance"),
    ("Connection Failures", "Clients unable to establish connections to services"),
    ("Authentication Errors", "Users experiencing login or permission issues"),
    ("Data Corruption", "Integrity check failures or inconsistent data states"),
    ("Memory Exhaustion", "Out of memory errors and system slowdowns"),
    ("CPU Saturation", "Processors running at 100 percent utilization"),
    ("Network Congestion", "Packet loss and high network latency"),
    ("Storage Full", "Filesystem capacity warnings and write failures"),
    ("Service Crashes", "Unexpected service terminations and restarts"),
]

for idx, (issue, description) in enumerate(issues, 1):
    content += f"""
### Issue {idx}: {issue}

**Description**: {description}

**Symptoms**:
- Performance degradation noticeable to end users
- Error messages in system logs indicating {issue.lower()}
- Monitoring alerts triggered for conditions related to {issue.lower()}
- Client applications reporting timeouts or failures
- System metrics showing abnormal patterns

**Diagnostic Steps**:
1. Gather system metrics and logs from all affected components
2. Identify root cause through correlation of metrics with symptoms
3. Implement remediation based on root cause analysis
4. Monitor system during and after remediation
5. Document the issue and resolution for future reference

**Common Root Causes**:
- Misconfigured parameters in system configuration
- Resource exhaustion due to workload spikes or capacity issues
- Hardware failures or performance degradation
- Software bugs memory leaks or inefficient algorithms
- External dependencies unavailable or performing poorly
- Network infrastructure issues affecting connectivity
- Insufficient capacity provisioning for current demand
- Competing workload interference and resource contention

**Resolution Procedures**:
1. Immediate mitigation: Reduce load or failover to backup systems
2. Collect diagnostics: Save all relevant logs and metrics for analysis
3. Apply fix: Based on identified root cause implement appropriate solution
4. Verify resolution: Confirm metrics return to normal acceptable ranges
5. Post-incident review: Analyze what happened why and how to prevent recurrence
6. Preventive measures: Implement monitoring capacity improvements or process changes

**Prevention Strategies**:
- Implement comprehensive monitoring with appropriate alert thresholds
- Configure alerts for early warning of degrading system conditions
- Establish capacity planning processes and conduct regular reviews
- Maintain detailed runbooks for common operational procedures
- Conduct regular system health checks scheduled maintenance activities
- Keep systems updated with latest patches security updates and versions
- Document all configuration changes in your change management system
- Train operations team on troubleshooting procedures and best practices

"""

# Write file
output_path = (
    "/Users/brennanconley/vibecode/wekadocs-matrix/data/ingest/test-truly-massive.md"
)
with open(output_path, "w") as f:
    f.write(content)

print(f"✅ Generated document: {output_path}")
print(f"   Characters: {len(content):,}")
print(f"   Estimated tokens: ~{len(content) // 3:,} (if ~3 chars/token)")
print("   Target: Exceed 8192 tokens for truncation testing")
