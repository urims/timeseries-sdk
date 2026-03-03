---
name: training-ts-models
description: Guides training of deep learning models for time series forecasting — covering data pipelines, loss functions, optimization, regularization, hyperparameter tuning, and debugging training runs. Use when configuring training loops, diagnosing training problems, selecting loss functions, building time series dataloaders, setting up mixed precision training, running hyperparameter searches, scheduling learning rates, implementing early stopping, or debugging training instabilities. Trigger on "train the model", "loss diverging", "hyperparameter tuning", "optimizer", "learning rate", "overfitting", "batch size", "dataloader", "training loop", "fine-tuning", "transfer learning for time series", "loss not converging", "NaN loss", "training is slow". Do not use for architecture selection — use `designing-ts-architectures`. Do not use for SageMaker-specific configuration — use `operating-sagemaker`.
---

# Training Time Series Models

## Pre-Training Checklist

Before writing any training code, verify these items. Skipping them is the most common cause of wasted compute:

```
□ Data quality checks passed (run guarding-data-quality first)
□ Architecture selected with ADR (run designing-ts-architectures first)
□ Train/val/test split uses temporal ordering (no random split)
□ No future information leaks into training features
□ Baseline metrics established (at minimum: seasonal naive)
□ Reproducibility seeds set — run scripts/set_seeds.py
□ Experiment tracked — run facilitating-experiments first
```

## Training Configuration

### Loss Function Selection

| Task | Recommended Loss | When to use alternatives |
|------|-----------------|------------------------|
| Point forecasting | MSE / Huber | Huber if outliers present |
| Quantile forecasting | Quantile Loss | Pinball loss for specific quantiles |
| Probabilistic | Negative Log-Likelihood | CRPS for distribution calibration |
| Skewed distributions | Asymmetric losses | Weighted quantile loss |
| Intermittent demand | RMSSE or custom zero-inflated | Tweedie loss |

### Optimizer Defaults

Start with these unless you have a reason not to:
- **Optimizer**: AdamW (weight_decay=0.01)
- **Learning rate**: 1e-3 (adjust via scheduler, not manual tuning)
- **Scheduler**: OneCycleLR (max_lr=1e-3, pct_start=0.3)
- **Gradient clipping**: max_norm=1.0

### Dataloader for Time Series

Time series dataloaders require windowed sampling with proper temporal alignment. Run `scripts/validate_dataloader.py` after setup to verify no leakage:

```python
# Validation command
python scripts/validate_dataloader.py \
    --dataset path/to/data.parquet \
    --lookback 168 \
    --horizon 24 \
    --frequency hourly
```

## Debugging Training Problems

### Loss Not Converging

1. Check learning rate — try 10x lower, then 10x higher
2. Verify data normalization (per-series z-score for time series)
3. Check for NaN in inputs: `torch.isnan(x).any()`
4. Reduce model complexity — does a simpler model learn?
5. Inspect a single batch manually: are targets reasonable?

### Overfitting

1. Add dropout (0.1–0.3) between layers
2. Reduce model capacity (hidden_size, n_layers)
3. Increase training data via augmentation (jitter, scaling, window slicing)
4. Apply weight decay (1e-2 to 1e-4)
5. Use early stopping with patience of 5–10 epochs on validation loss

### Training Instabilities (NaN / Inf)

1. Lower learning rate by 10x
2. Enable gradient clipping (max_norm=1.0)
3. Check for division by zero in custom losses
4. Use mixed precision carefully — some operations need float32
5. Inspect the specific batch that caused the issue

## Feedback Loop

After each training run:

1. Log metrics to experiment tracker
2. Compare against baseline and previous best
3. If validation loss improved: save checkpoint with full metadata
4. If not: diagnose using the debugging section above
5. Record the finding in the experiment log before the next run

## Quality Checklist

- [ ] Pre-training checklist completed
- [ ] Loss function matches the task and data characteristics
- [ ] Validation uses temporal split (not random)
- [ ] Training metrics logged with reproducibility metadata
- [ ] At least one comparison against baseline included in results
