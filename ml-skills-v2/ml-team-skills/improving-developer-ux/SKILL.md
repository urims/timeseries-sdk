---
name: improving-developer-ux
description: Designs excellent developer experience (DX) for ML tools, SDKs, notebooks, internal tools, and CLI interfaces. Covers Python SDK design, CLI argument patterns, notebook UX, error message quality, logging output, configuration schemas, and internal tooling usability. Use when designing interfaces, tooling, or workflows that developers interact with — focusing on making the right thing easy and the wrong thing hard. Trigger on "developer experience", "DX", "SDK design", "CLI design", "notebook design", "error messages", "logging format", "config design", "internal tool", "make it easier to use", "onboarding experience", "confusing interface", "better error message". Do not use for API schema design — use `designing-apis`. Do not use for stakeholder-facing communications — use `communicating-to-stakeholders`.
---

# Improving Developer UX

## Core Principle: Pit of Success

Design so the right thing is easy and the wrong thing is hard (or impossible). Developers should fall into correct usage without reading documentation.

## DX Patterns

### Error Messages

Error messages are the primary teaching interface. Every error should answer three questions:

```
1. What happened?    → "Config validation failed"
2. Why?              → "Field 'horizon' must be a positive integer, got: -7"
3. How to fix it?    → "Set horizon to the number of forecast steps (e.g., horizon=24)"
```

Bad: `ValueError: invalid value`
Good: `ValueError: horizon must be a positive integer, got -7. Set horizon to the number of future time steps to predict (e.g., horizon=24 for daily data predicting one month ahead).`

### Configuration Design

```python
# Good: Defaults that work, overrides that are obvious
config = ForecastConfig(
    # Required — no defaults for critical choices
    model="tft",
    horizon=24,

    # Optional — sensible defaults
    batch_size=64,           # Works on most GPUs
    learning_rate=1e-3,      # Good starting point for AdamW
    max_epochs=100,          # Early stopping will handle actual count
    seed=42,                 # Reproducible by default
)
```

Principles:
- Required parameters have no defaults (force the developer to think)
- Optional parameters have battle-tested defaults
- Every parameter has a comment explaining why this default

### CLI Design

```bash
# Good: Progressive disclosure of complexity
# Simple case (most common)
forecast train --data sales.csv --model tft

# With options (power user)
forecast train --data sales.csv --model tft --horizon 24 --gpu 0 --seed 42

# Full config (advanced)
forecast train --config experiment.yaml
```

### Notebook UX

- First cell should work with `pip install` + `import`
- Show a result within 3 cells
- Use progress bars for anything >5 seconds
- Print human-readable summaries, not raw tensors
- Include a "restart and run all" test before sharing

### Logging

```python
# Good: Structured, scannable, actionable
[2025-01-15 10:30:00] INFO  Training started | model=tft | series=487 | horizon=24
[2025-01-15 10:30:15] INFO  Epoch 1/100 | train_loss=0.342 | val_loss=0.298 | lr=1e-4
[2025-01-15 10:31:02] WARN  Val loss plateaued for 3 epochs | Consider reducing LR
[2025-01-15 10:35:00] INFO  Early stopping at epoch 47 | best_val_loss=0.198 (epoch 44)
```

Format: `[timestamp] LEVEL context | key=value | key=value`

## Quality Checklist

- [ ] Error messages answer what, why, and how to fix
- [ ] Configuration has sensible defaults with comments
- [ ] Most common use case requires fewest arguments
- [ ] Progress indicators for operations >5 seconds
- [ ] Logging is structured and scannable
