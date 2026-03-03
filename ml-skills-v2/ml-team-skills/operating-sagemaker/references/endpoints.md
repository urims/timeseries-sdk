# SageMaker Endpoints

## Contents
- Real-time vs batch inference
- Endpoint configuration
- Auto-scaling
- Multi-model endpoints
- A/B testing deployments

## Choosing Inference Mode

| Mode | When to use | Latency | Cost model |
|------|-------------|---------|------------|
| Real-time endpoint | <100ms response needed, continuous traffic | Low | Per-hour (always on) |
| Serverless inference | Sporadic traffic, cold start acceptable | Medium | Per-request |
| Batch transform | Bulk forecasts, no latency requirement | N/A | Per-job |
| Async inference | Large payloads, minutes-scale response OK | High | Per-hour + queue |

For most time series forecasting teams: **batch transform** for scheduled forecasts, **real-time endpoint** for on-demand queries.

## Real-Time Endpoint Configuration

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data=f"s3://{bucket}/models/{model_name}/model.tar.gz",
    role=role,
    framework_version="2.1.0",
    py_version="py310",
    entry_point="inference.py",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t3.medium",  # Start small, scale based on load
)
```

## Auto-Scaling

```python
client = boto3.client("application-autoscaling")

client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=4,
)

client.put_scaling_policy(
    PolicyName="invocations-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

## Cleanup Reminder

Endpoints bill continuously. After testing, always delete:
```python
predictor.delete_endpoint()
```
