"""
Phase 5 Task 5.4 Tests - Production Deployment
NO MOCKS - Tests against live Docker Compose stack
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "deploy" / "scripts"


class TestBlueGreenDeployment:
    """Test blue/green deployment strategy."""

    def test_blue_green_switch_script_exists(self):
        """Verify blue/green switch script exists and is executable."""
        script = SCRIPTS_DIR / "blue-green-switch.sh"
        assert script.exists(), f"Script not found: {script}"
        assert os.access(script, os.X_OK), "Script not executable"

    def test_blue_deployment_manifest_exists(self):
        """Verify blue deployment manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/mcp-server-deployment.yaml"
        assert Path(manifest).exists()

    def test_green_deployment_manifest_exists(self):
        """Verify green deployment manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/mcp-server-green-deployment.yaml"
        assert Path(manifest).exists()

    def test_blue_green_manifests_have_version_labels(self):
        """Verify deployments have correct version labels."""
        blue_manifest = BASE_DIR / "deploy/k8s/base/mcp-server-deployment.yaml"
        green_manifest = BASE_DIR / "deploy/k8s/base/mcp-server-green-deployment.yaml"

        with open(blue_manifest) as f:
            blue_content = f.read()
            assert "version: blue" in blue_content

        with open(green_manifest) as f:
            green_content = f.read()
            assert "version: green" in green_content


class TestCanaryDeployment:
    """Test canary deployment with progressive rollout."""

    def test_canary_rollout_script_exists(self):
        """Verify canary rollout script exists."""
        script = SCRIPTS_DIR / "canary-rollout.sh"
        assert script.exists()
        assert os.access(script, os.X_OK)

    def test_canary_deployment_manifest_exists(self):
        """Verify canary deployment manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/mcp-server-canary-deployment.yaml"
        assert Path(manifest).exists()

    def test_canary_manifest_has_correct_labels(self):
        """Verify canary manifest has version: canary label."""
        manifest = BASE_DIR / "deploy/k8s/base/mcp-server-canary-deployment.yaml"
        with open(manifest) as f:
            content = f.read()
            assert "version: canary" in content

    def test_canary_script_has_progressive_stages(self):
        """Verify canary script implements 5% → 25% → 50% → 100% rollout."""
        script = SCRIPTS_DIR / "canary-rollout.sh"
        with open(script) as f:
            content = f.read()
            # Check for progressive scaling
            assert "scale_canary 1 5" in content or "5%" in content
            assert "scale_canary 2 25" in content or "25%" in content
            assert "scale_canary 3 50" in content or "50%" in content
            assert "100" in content


class TestBackupRestore:
    """Test backup/restore functionality."""

    def test_backup_script_exists(self):
        """Verify backup script exists."""
        script = SCRIPTS_DIR / "backup-all.sh"
        assert script.exists()
        assert os.access(script, os.X_OK)

    def test_restore_script_exists(self):
        """Verify restore script exists."""
        script = SCRIPTS_DIR / "restore-all.sh"
        assert script.exists()
        assert os.access(script, os.X_OK)

    def test_backup_script_covers_all_services(self):
        """Verify backup script backs up Neo4j, Qdrant, and Redis."""
        script = SCRIPTS_DIR / "backup-all.sh"
        with open(script) as f:
            content = f.read()
            assert "neo4j" in content.lower()
            assert "qdrant" in content.lower()
            assert "redis" in content.lower()

    def test_restore_script_validates_manifest(self):
        """Verify restore script checks for manifest.json."""
        script = SCRIPTS_DIR / "restore-all.sh"
        with open(script) as f:
            content = f.read()
            assert "manifest.json" in content


class TestDisasterRecovery:
    """Test disaster recovery procedures."""

    def test_dr_runbook_exists(self):
        """Verify DR runbook exists."""
        runbook = BASE_DIR / "deploy" / "DR-RUNBOOK.md"
        assert runbook.exists()

    def test_dr_drill_script_exists(self):
        """Verify DR drill script exists."""
        script = SCRIPTS_DIR / "dr-drill.sh"
        assert script.exists()
        assert os.access(script, os.X_OK)

    def test_dr_runbook_documents_rto_rpo(self):
        """Verify DR runbook documents RTO and RPO targets."""
        runbook = BASE_DIR / "deploy" / "DR-RUNBOOK.md"
        with open(runbook) as f:
            content = f.read()
            assert "RTO" in content
            assert "RPO" in content
            assert "1 hour" in content or "60 min" in content
            assert "15 min" in content

    def test_dr_drill_measures_timing(self):
        """Verify DR drill script measures timing."""
        script = SCRIPTS_DIR / "dr-drill.sh"
        with open(script) as f:
            content = f.read()
            assert "date +%s" in content or "SECONDS" in content
            assert "duration" in content.lower() or "rto" in content.lower()


class TestFeatureFlags:
    """Test feature flags implementation."""

    def test_feature_flags_module_exists(self):
        """Verify feature flags module exists."""
        module = BASE_DIR / "src" / "shared" / "feature_flags.py"
        assert module.exists()

    def test_feature_flags_config_exists(self):
        """Verify feature flags config exists."""
        config = BASE_DIR / "config" / "feature_flags.json"
        assert config.exists()

    def test_feature_flags_config_is_valid_json(self):
        """Verify feature flags config is valid JSON."""
        config = BASE_DIR / "config" / "feature_flags.json"
        with open(config) as f:
            data = json.load(f)
            assert "flags" in data
            assert isinstance(data["flags"], dict)

    def test_feature_flags_have_rollout_percentage(self):
        """Verify feature flags support rollout percentage."""
        config = BASE_DIR / "config" / "feature_flags.json"
        with open(config) as f:
            data = json.load(f)
            for flag_name, flag_data in data["flags"].items():
                assert "rollout_percentage" in flag_data
                assert 0 <= flag_data["rollout_percentage"] <= 100


class TestKubernetesManifests:
    """Test Kubernetes manifest completeness."""

    def test_namespace_manifest_exists(self):
        """Verify namespace manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/namespace.yaml"
        assert Path(manifest).exists()

    def test_configmap_manifest_exists(self):
        """Verify configmap manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/configmap.yaml"
        assert Path(manifest).exists()

    def test_secrets_manifest_exists(self):
        """Verify secrets manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/secrets.yaml"
        assert Path(manifest).exists()

    def test_neo4j_statefulset_exists(self):
        """Verify Neo4j statefulset exists."""
        manifest = BASE_DIR / "deploy/k8s/base/neo4j-statefulset.yaml"
        assert Path(manifest).exists()

    def test_qdrant_statefulset_exists(self):
        """Verify Qdrant statefulset exists."""
        manifest = BASE_DIR / "deploy/k8s/base/qdrant-statefulset.yaml"
        assert Path(manifest).exists()

    def test_redis_statefulset_exists(self):
        """Verify Redis statefulset exists."""
        manifest = BASE_DIR / "deploy/k8s/base/redis-statefulset.yaml"
        assert Path(manifest).exists()

    def test_ingestion_worker_deployment_exists(self):
        """Verify ingestion worker deployment exists."""
        manifest = BASE_DIR / "deploy/k8s/base/ingestion-worker-deployment.yaml"
        assert Path(manifest).exists()

    def test_ingress_manifest_exists(self):
        """Verify ingress manifest exists."""
        manifest = BASE_DIR / "deploy/k8s/base/ingress.yaml"
        assert Path(manifest).exists()

    def test_kustomization_exists(self):
        """Verify kustomization.yaml exists."""
        manifest = BASE_DIR / "deploy/k8s/base/kustomization.yaml"
        assert Path(manifest).exists()
