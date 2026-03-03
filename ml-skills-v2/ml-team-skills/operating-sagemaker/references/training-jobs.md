# SageMaker Training Jobs

## Contents
- Instance type selection
- Spot training setup
- Distributed training
- Custom containers
- Checkpointing
- Debugging training failures

## Instance Type Selection

| Workload | Recommended Instance | Notes |
|----------|---------------------|-------|
| Small models, prototyping | ml.g4dn.xlarge | Single T4 GPU, good cost/performance |
| Medium models (TFT, N-HiTS) | ml.g5.xlarge | A10G GPU, 24GB VRAM |
| Large models, multi-GPU | ml.g5.12xlarge | 4x A10G, distributed training |
| CPU-only (LightGBM baselines) | ml.m5.2xlarge | 8 vCPU, 32GB RAM |
| Hyperparameter tuning | ml.g4dn.xlarge | Maximize parallelism over instance size |

## Spot Training Setup

Spot instances save 60–70% but can be interrupted. Always pair with checkpointing:

```python
estimator = PyTorch(
    # ...
    use_spot_instances=True,
    max_wait=7200,          # Maximum total time including interruptions
    max_run=3600,           # Maximum training time
    checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{experiment_name}",
    checkpoint_local_path="/opt/ml/checkpoints",
)
```

In your training script, implement checkpoint save/resume:

```python
# Save checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, os.path.join(path, "checkpoint.pt"))

# Resume from checkpoint (if spot instance resumes)
checkpoint_path = os.path.join("/opt/ml/checkpoints", "checkpoint.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
```

## Custom Containers for Time Series

When your dependencies exceed what SageMaker's built-in containers provide:

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN pip install --no-cache-dir \
    neuralforecast==1.7.0 \
    gluonts==0.14.0 \
    pytorch-lightning==2.1.0

COPY src/ /opt/ml/code/
ENV SAGEMAKER_PROGRAM train.py
```

Build and push:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker build -t ts-training .
docker tag ts-training $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ts-training:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ts-training:latest
```

## Debugging Training Failures

Check in this order:
1. **CloudWatch logs**: `/aws/sagemaker/TrainingJobs` — look for the actual Python traceback
2. **Instance metrics**: CPU/GPU utilization, memory — OOM shows as sudden stop at 100% memory
3. **Data access**: Verify S3 paths are correct and IAM role has access
4. **Container startup**: If the job fails in "Starting", the container likely has a dependency issue
