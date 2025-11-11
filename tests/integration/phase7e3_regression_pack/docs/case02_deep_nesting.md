# Deploy Pipelines for DataFlow

DocTag: REGPACK-02

## End-to-End Pipeline

### Step 1: Provision Resources

- Create a VPC, subnets, IAM roles.
- Choose region and zones.

#### Detail A: Set Parameter X

- Set `PIPELINE_PARAM_X=true`.
- Ensure **Parameter X** is visible in diagnostics.

### Step 2: Deploy Orchestrator

- Apply Helm chart for the scheduler.
- Wait for health checks to pass.

#### Detail B: Enable Autoscaling

- Configure HorizontalPodAutoscaler.
- Target CPU 70%.
