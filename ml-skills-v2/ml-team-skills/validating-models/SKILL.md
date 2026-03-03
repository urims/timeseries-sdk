---
name: validating-models
description: Designs and executes rigorous validation frameworks for time series forecasting models, including backtesting strategies, metric selection, stakeholder acceptance criteria, production monitoring loops, and feedback integration. Use when setting up model evaluation, designing backtests, interpreting forecast metrics, building feedback collection systems, or deciding whether a model is ready for production. Trigger on "validate the model", "backtest", "evaluation strategy", "which metric to use", "is the model good enough", "production monitoring", "feedback loop", "model drift", "compare models", "evaluation framework", "acceptance criteria", "MASE", "coverage", "CRPS". Do not use for training debugging — use `training-ts-models`. Do not use for explaining validation results to stakeholders — use `explaining-for-stakeholders`.
---

# Validating Models

## Validation Philosophy

Validation answers one question: *"Should we trust this model to make business decisions?"* Metrics alone don't answer this — context does.

## Validation Workflow

Follow this sequence. Run `scripts/run_backtest.py` to automate steps 1–3:

```
1. Select metrics → 2. Design backtest → 3. Run evaluation → 4. Interpret results → 5. Get stakeholder sign-off
     ↑                                                                                        │
     └──────────────── Feedback loop: adjust thresholds, retrain, re-validate ←──────────────┘
```

### Step 1: Metric Selection

| Business Need | Primary Metric | Why |
|--------------|----------------|-----|
| Accuracy of point forecasts | MASE | Scale-independent, compares to naive baseline |
| Direction matters more than magnitude | MASE + directional accuracy | Catches "right trend, wrong scale" |
| Uncertainty quantification | Coverage (90%) + CRPS | Coverage checks calibration, CRPS rewards sharpness |
| Cost of over/under prediction differs | Weighted quantile loss | Asymmetric penalty matches business cost |
| Intermittent demand | RMSSE | Handles zeros better than MAPE |

Avoid MAPE for time series — it's undefined at zero and biased toward underprediction.

### Step 2: Backtest Design

Time series cross-validation uses expanding or sliding windows. Never use random splits.

```
Expanding window (preferred when data is limited):
|--train--|--val--|
|----train----|--val--|
|-------train-------|--val--|

Sliding window (preferred when stationarity is questionable):
   |--train--|--val--|
      |--train--|--val--|
         |--train--|--val--|
```

Run `scripts/run_backtest.py` with your configuration:

```bash
python scripts/run_backtest.py \
    --model-checkpoint path/to/model.pt \
    --data path/to/data.parquet \
    --strategy expanding \
    --n-folds 5 \
    --horizon 24 \
    --metrics mase,coverage_90,crps
```

### Step 3: Acceptance Criteria

Define these before running validation — never after seeing results:

```yaml
acceptance_criteria:
  hard_requirements:          # Must pass or model is rejected
    mase: "< 1.0"            # Must beat naive baseline
    coverage_90: ">= 0.85"   # Prediction intervals must be calibrated
  soft_requirements:          # Should pass, deviations need justification
    mase: "< 0.85"
    directional_accuracy: ">= 0.60"
  guardrails:                 # Red flags that halt deployment
    max_error_any_series: "< 5x median error"
    nan_predictions: "== 0"
```

### Step 4: Production Monitoring

After deployment, monitor continuously for:
- **Data drift**: Distribution of incoming features vs training data
- **Prediction drift**: Distribution of outputs shifting unexpectedly
- **Performance degradation**: Actual vs predicted, computed on a rolling window

## Quality Checklist

- [ ] Metrics match the business cost structure
- [ ] Backtest uses temporal splits (not random)
- [ ] Acceptance criteria defined before seeing results
- [ ] Results include comparison to baseline
- [ ] Stakeholder sign-off documented
- [ ] Production monitoring plan in place
