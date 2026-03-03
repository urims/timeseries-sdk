---
name: designing-ts-architectures
description: Guides selection, design, and justification of deep learning architectures for time series forecasting problems. Covers univariate and multivariate forecasting, probabilistic output, multi-step prediction, hierarchical forecasting, and anomaly detection on temporal data. Use when choosing or designing a neural network for time series tasks, comparing architecture candidates, or writing architecture decision records. Trigger on "which model should I use", "architecture for forecasting", "LSTM vs Transformer", "design the model", "TFT", "N-BEATS", "N-HiTS", "PatchTST", "iTransformer", "Chronos", "TimesFM", "DLinear", "DeepAR", "architecture decision record", or any request about DL model design for time series. Do not use for training configuration — use `training-ts-models`. Do not use for teaching concepts — use `teaching-timeseries-dl`.
---

# Designing Time Series Architectures

## Architecture Selection Process

Follow this decision tree before recommending any architecture:

### Step 1: Characterize the Problem

```
□ Forecast horizon: short (≤24), medium (25–168), long (>168 steps)
□ Frequency: sub-hourly / hourly / daily / weekly / monthly
□ Series count: single / small batch (<100) / large scale (>1000)
□ Covariates: none / known-future / past-only / static metadata
□ Output type: point / quantile / full distribution
□ Interpretability requirement: none / partial / full
□ Latency constraint: batch / near-real-time (<1s) / real-time (<100ms)
□ Data volume: <1K / 1K–100K / >100K steps per series
```

### Step 2: Shortlist Architectures

| Scenario | Primary | Alternative |
|----------|---------|-------------|
| Short horizon, high interpretability | TFT | N-HiTS |
| Long horizon, many series | PatchTST / iTransformer | TimesNet |
| Zero-shot / few-shot | Chronos, TimesFM, MOIRAI | Lag-Llama |
| Univariate, fast iteration | N-BEATS / N-HiTS | DLinear |
| Probabilistic output | TFT / DeepAR | TimeGrad |
| Multivariate cross-series | iTransformer | Crossformer |
| Limited data (<500 steps) | N-HiTS + augmentation | Prophet + residuals |
| Production / latency-sensitive | DLinear / NLinear | Optimized N-HiTS |

For detailed architecture comparison (inductive biases, implementation libraries, failure modes, hyperparameter ranges), see [references/architectures.md](references/architectures.md).

### Step 3: Produce an Architecture Decision Record

Always produce an ADR when recommending an architecture:

```markdown
## Architecture Decision Record

**Decision**: [Architecture Name]
**Date**: [date]
**Status**: Proposed / Accepted / Deprecated

### Context
[Problem description, data characteristics, constraints]

### Decision Drivers
- [e.g., interpretability required by stakeholder]
- [e.g., multivariate with 500 series]

### Considered Options
1. [Option A] — rejected because [reason]
2. [Option B] — rejected because [reason]
3. **[Chosen]** — selected because [reason]

### Consequences
- Positive: [list]
- Negative / risks: [list]
- Mitigation: [list]

### Success Metrics
- Primary: [metric + threshold, e.g., MASE < 1.2]
- Secondary: [metric + threshold]
- Guardrails: [metric + threshold, e.g., P90 coverage > 88%]
```

## Common Anti-Patterns

- **Overcomplicated default**: Defaulting to Transformers for short univariate series — linear models often win
- **Ignoring seasonality**: Not accounting for multiple seasonal periods
- **Leaking future covariates**: Using features at time t unavailable in production
- **Wrong loss for the task**: MSE for skewed distributions → use quantile or CRPS loss
- **Benchmark-less decisions**: Always compare against Seasonal Naïve baseline and a classical model (ETS/ARIMA)

## Baseline Hierarchy

Before any DL model, establish these baselines in order. Each step must justify added complexity with measurable gains:

1. **Naive**: Last value / seasonal naive
2. **Statistical**: ETS, ARIMA, TBATS
3. **ML**: LightGBM with lag features
4. **Simple DL**: DLinear / NLinear
5. **Target architecture**: The complex DL model
