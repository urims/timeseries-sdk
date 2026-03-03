---
name: operating-sagemaker
description: Provides expert guidance on using AWS SageMaker for ML workflows including training jobs, experiments, model registry, endpoints, pipelines, and monitoring — specifically for time series forecasting teams. Use when setting up SageMaker training jobs, deploying prediction endpoints, configuring spot training, building SageMaker Pipelines, using Feature Store, setting up Model Monitor, or integrating SageMaker with MLflow. Trigger on "SageMaker", "AWS training job", "SageMaker endpoint", "spot instances", "SageMaker Pipelines", "model registry AWS", "SageMaker Studio", "container for SageMaker", "deploying on AWS", "ECR container", "SageMaker processing job". Do not use for general cloud architecture without SageMaker. Do not use for training configuration that isn't SageMaker-specific — use `training-ts-models` instead.
---

# Operating SageMaker

## SageMaker Workflow for Time Series Teams

```
Local development → Container build → Training job → Model registry → Endpoint
     ↑                                      ↓
     └──── Monitor + retrain ←── Model Monitor
```

## Task Router

Select the workflow that matches your task:

**Setting up a training job** → see [references/training-jobs.md](references/training-jobs.md)
**Deploying an endpoint** → see [references/endpoints.md](references/endpoints.md)
**Building a pipeline** → see [references/pipelines.md](references/pipelines.md)

## Quick Reference: Common Commands

```python
# Estimator setup (PyTorch example for time series)
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.1.0",
    py_version="py310",
    use_spot_instances=True,          # 60-70% cost savings
    max_wait=7200,                    # 2x max_run for spot
    max_run=3600,
    hyperparameters={
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    metric_definitions=[
        {"Name": "val_loss", "Regex": "val_loss: ([0-9\\.]+)"},
        {"Name": "val_mase", "Regex": "val_mase: ([0-9\\.]+)"},
    ],
)
```

## Cost Optimization Checklist

- [ ] Using spot instances for training (saves 60–70%)
- [ ] Instance type matches workload (don't default to p3.2xlarge)
- [ ] Auto-scaling configured for endpoints
- [ ] S3 lifecycle policies on training artifacts
- [ ] Processing jobs for batch inference (cheaper than real-time endpoints)

## Common Pitfalls

- **Missing metric definitions**: SageMaker can't track metrics it can't parse. Always define regex patterns that match your logging format exactly.
- **Spot interruption without checkpointing**: Training loses all progress. Always implement checkpoint saving to S3 and resumption logic.
- **Oversized inference instance**: Start with ml.t3.medium for testing, scale up based on actual latency requirements.
- **Forgetting to delete endpoints**: Endpoints bill by the hour even with zero traffic. Use auto-scaling with min_instance_count=0 or delete after testing.
