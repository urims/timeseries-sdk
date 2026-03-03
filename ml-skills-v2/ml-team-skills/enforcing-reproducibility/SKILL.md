---
name: enforcing-reproducibility
description: Enforces reproducibility best practices across all ML workflows — from data versioning to model artifacts, ensuring any experiment can be re-run and any result traced back to its exact inputs. Use when setting up a new project, reviewing code for reproducibility gaps, creating model cards, configuring DVC or MLflow, or auditing whether results can be reproduced. Trigger on "reproducibility", "reproduce this experiment", "model card", "data versioning", "DVC", "artifact tracking", "environment setup", "requirements.txt", "Docker", "seed", "determinism", "can you prove this", "what changed between versions". Do not use for experiment design — use `facilitating-experiments`. Do not use for Git workflow conventions unrelated to ML reproducibility.
---

# Enforcing Reproducibility

## Why This Matters

Non-technical stakeholders will ask "can you prove this?" or "what changed between versions?" Reproducibility is how you answer with confidence.

## Reproducibility Checklist

Every experiment and model delivery must pass these checks:

```
Code:
  □ All random seeds fixed — run scripts/set_seeds.py (from training-ts-models)
  □ Deterministic mode enabled where possible
  □ Config externalized (no magic numbers in code)
  □ Environment pinned (requirements.txt or conda env)
  □ Code versioned in Git (commit hash logged)

Data:
  □ Raw data versioned (DVC or S3 versioning)
  □ Train/val/test split logic is deterministic
  □ Preprocessing steps logged with checksum of output
  □ No manual data edits (all transformations in code)

Model:
  □ Best checkpoint saved with all metadata
  □ Model card written (see template below)
  □ Hyperparameter config saved as YAML artifact
  □ Training metrics logged to experiment tracker

Results:
  □ Predictions include model version and generation timestamp
  □ Evaluation metrics computed from saved artifacts (not live)
  □ Results table includes error bars (mean ± std)
```

## Environment Pinning

Run `scripts/pin_environment.py` to capture the current environment:

```bash
python scripts/pin_environment.py --output reproducibility_metadata.json
```

This records Python version, package versions, CUDA version, git commit, and whether the repo has uncommitted changes.

For containerized training, use a locked Dockerfile:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
```

## Model Card Template

Every model that reaches validation must have a model card:

```markdown
# Model Card: [Model Name] v[Version]

## Model Details
- Architecture: [e.g., TFT]
- Version: v2.1
- Trained: [date]
- Owner: [team/person]
- Git commit: [hash]

## Intended Use
- Primary: [e.g., Forecast weekly demand for SKUs in Region X]
- Out-of-scope: [e.g., Not for new products with <3 months history]

## Training Data
- Source: [S3 path + DVC hash]
- Period: [start] to [end]
- Series count: [N]
- Frequency: [daily/weekly/etc]

## Performance
| Metric | Value | Baseline |
|--------|-------|----------|
| MASE | 0.87 | 1.00 |
| Coverage 90% | 91% | — |

Evaluation: [backtest strategy and fold count]

## Limitations
- [Known weakness 1 with quantification]
- [Known weakness 2]

## Reproducibility
- Config: [path]
- Checkpoint: [path]
- Environment: [hash of requirements-lock.txt]
- Reproduce with: `make reproduce-v2.1`
```

## Quality Checklist

- [ ] Reproducibility checklist fully completed
- [ ] Environment pinned and recorded
- [ ] Model card written for any model reaching validation
- [ ] All artifacts stored with traceable paths
- [ ] A team member can reproduce the result from the recorded metadata alone
