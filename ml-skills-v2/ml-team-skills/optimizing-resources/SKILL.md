---
name: optimizing-resources
description: Optimizes ML workflows, training, and inference for constrained compute, memory, time, or data budgets. Provides strategies for small GPU memory, limited training data, tight inference latency, cost budgets, CPU-only environments, and edge deployment. Use when operating under any resource constraint that affects model selection, training strategy, or deployment architecture. Trigger on "out of memory", "OOM", "too slow", "reduce cost", "small dataset", "CPU only", "optimize memory", "quantization", "pruning", "distillation", "few-shot", "low data regime", "edge deployment", "cheap alternative", "limited GPU", "optimize inference", "budget constraint", "cheaper instance". Do not use for general architecture selection — use `designing-ts-architectures`. Do not use for SageMaker instance selection — use `operating-sagemaker`.
---

# Optimizing Resources

## Resource Constraint Triage

Identify the binding constraint first — the optimization strategy depends on which resource is scarce:

| Constraint | Symptoms | Primary Strategy |
|-----------|----------|-----------------|
| **GPU memory** | OOM errors, batch size = 1 | Gradient accumulation, mixed precision, model pruning |
| **Compute time** | Training takes days | Smaller model, fewer epochs with better LR schedule, early stopping |
| **Inference latency** | >100ms response time | Quantization, distillation, simpler architecture |
| **Training data** | <500 samples per series | Data augmentation, transfer learning, foundation models |
| **Budget** | Cloud costs too high | Spot instances, smaller instances, batch inference |
| **No GPU** | CPU-only environment | LightGBM baselines, DLinear, quantized models |

## GPU Memory Optimization

Apply in this order — each builds on the previous:

1. **Mixed precision training** (free, ~30% memory savings):
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       output = model(input)
       loss = criterion(output, target)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Gradient accumulation** (free, trades time for memory):
   ```python
   accumulation_steps = 4  # Effective batch = batch_size * accumulation_steps
   for i, (input, target) in enumerate(dataloader):
       loss = model(input, target) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Gradient checkpointing** (~30% memory savings, ~20% slower):
   ```python
   from torch.utils.checkpoint import checkpoint
   # Apply to memory-heavy layers
   ```

4. **Model reduction**: Reduce hidden_size, n_layers, attention heads

## Low Data Strategies

When you have fewer than 500 time steps per series:

1. **Foundation models zero-shot** (Chronos, TimesFM) — no training needed
2. **Transfer learning**: Pre-train on related public datasets, fine-tune on yours
3. **Data augmentation for time series**:
   - Jittering: Add small Gaussian noise
   - Scaling: Multiply by random factor (0.8–1.2)
   - Window slicing: Sample sub-windows as separate training examples
   - Magnitude warping: Smooth random curve multiplication
4. **Simpler models**: N-HiTS or DLinear often beat complex models on small data

## Cost Optimization Decision Tree

```
Is this training or inference?
├── Training:
│   ├── Use spot instances (60–70% savings)
│   ├── Right-size GPU (g4dn.xlarge before p3.2xlarge)
│   └── Reduce experiment count — use facilitating-experiments to be systematic
└── Inference:
    ├── Batch inference where possible (cheapest)
    ├── Serverless for sporadic traffic
    ├── Auto-scale real-time endpoints to zero when idle
    └── Quantize model (INT8) for 2–4x throughput on same hardware
```

## Quality Checklist

- [ ] Binding constraint correctly identified
- [ ] Optimization strategy matches the constraint (not a generic "try everything")
- [ ] Performance impact measured (accuracy trade-off quantified)
- [ ] Fallback plan if optimization isn't sufficient
