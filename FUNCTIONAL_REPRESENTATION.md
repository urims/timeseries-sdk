# Functional Representation — TimeseriesSystem SDK v0.1.0

## What Was Built

A Python SDK (`tssdk`) for time series deep learning experimentation, designed
around the ml-team-skills framework. The SDK transforms raw tabular data
(date × series_id × features) into DL-ready tensors and provides scaffolding
for reproducible experiment iteration.

## Data Characteristics Discovered

```
Source: timeseries_sample.csv
Format: (date:YYYYMM, ts_id:str, actual_cost_paid:float, contract_value:float)
Shape: 360 rows, 5 series, 72 monthly timesteps each (2020-01 → 2025-12)
Nulls: LP4 6039T38G16 513 has 100% null contract_value (dropped)
       999 6006T10P39 513 has 2 leading nulls (dropped with LP4 under drop_series)
Scale heterogeneity: Series range from ~2.7 to ~140,000 (per-series z-score required)
Spread: actual_cost_paid - contract_value ranges from -447 to +1347
Usable series after cleanup: 3 (with drop_series strategy)
```

## Modules Created

```
tssdk/
├── __init__.py          → Pipeline (convenience wrapper), public API re-exports
├── config.py            → TimeseriesConfig, ExperimentConfig (validated dataclasses)
├── data/
│   ├── loader.py        → load() → CSV/Parquet → validated DataFrame + YYYYMM parsing
│   ├── preprocessor.py  → Preprocessor (fit_transform/transform/inverse_transform)
│   │                      Per-series z-score, null handling, short series filtering
│   ├── windower.py      → Windower → (X_encoder, X_decoder, Y) numpy arrays
│   │                      Temporal train/val/test splits, context-aware val/test windows
│   └── validator.py     → validate() → shape, NaN, Inf, variance checks
├── models/
│   ├── base.py          → BaseTimeseriesModel (abstract: build, get_model, describe)
│   └── constrained_lstm.py → ConstrainedLSTM (encoder-decoder + LogSumExp floor)
├── training/
│   ├── runner.py        → TrainingRunner (train with early stopping, evaluate, predict)
│   └── metrics.py       → MAE, RMSE, MASE, constraint_violation_rate
├── experiment/
│   └── tracker.py       → ExperimentTracker (hypothesis → metrics → decision → JSON)
└── utils/
    ├── seeds.py         → set_all_seeds() (Python, NumPy, TF)
    └── logging.py       → get_logger() → [timestamp] LEVEL name | key=value
```

## Architecture Validation: simplest_lstm_constrained.py

### Issues Found and Fixed

1. **build_model() didn't return model** — now returns via `self.model` in OOP wrapper.
2. **constrained_loss referenced undefined `floor_value`** — removed; MSE loss used with
   constraint layer handling the floor guarantee.
3. **main() entangled with file I/O** — separated into SDK pipeline vs. model definition.
4. **No per-series normalization** — added Preprocessor with per-series z-score.
5. **No reproducibility seeds** — added `set_all_seeds()`.
6. **Output shape bug** — `reduce_logsumexp` with stacked (batch, H, 1, 2) produced
   (batch, H, 1, 1). Fixed by squeezing inputs before stacking, then expanding.
7. **Model oversized for data** — reduced hidden sizes from 128/64/64 to 64/32/32
   (33,953 params) for 3-series × 72-step dataset.

### Architecture Compliance (per designing-ts-architectures skill)

- ✓ Encoder-decoder with state transfer (context bridge via enc_h, enc_c)
- ✓ Known-future covariates handled via decoder input
- ✓ Differentiable constraint (LogSumExp smooth max)
- ✓ Alpha-lag business logic preserved
- ✓ No future leakage in encoder input
- ✓ Temporal ordering enforced in data pipeline (not model)
- ⚠ Baseline hierarchy not yet complete (only naive computed; need statistical + ML baselines)

## Experiment EXP-001 Results

```
Decision: ITERATE
Hypothesis: Constrained LSTM beats naive baseline on test split
Result: Model MAE=0.436 vs Naive MAE=0.029 (on normalized scale)
Root cause: Only 3 test windows (very small dataset), high val loss plateau
Next steps:
  - EXP-002: Try scaling_method="none" to avoid normalization artifacts
  - EXP-003: Try forward_fill null_strategy to retain more series
  - EXP-004: Compare against DLinear baseline
  - EXP-005: Increase data volume (more series)
```

## SDK Usage Contract

```python
# Minimal (2 lines to prepared data):
from tssdk import TimeseriesConfig, Pipeline
config = TimeseriesConfig(target_col="actual_cost_paid", covariate_cols=["contract_value"])
splits = Pipeline(config).prepare("data.csv")

# Full experiment:
from tssdk import TimeseriesConfig, Pipeline, ConstrainedLSTM, TrainingRunner, ExperimentTracker, set_all_seeds
config = TimeseriesConfig(target_col="actual_cost_paid", covariate_cols=["contract_value"])
set_all_seeds(config.seed)
splits = Pipeline(config).prepare("data.csv")
model_builder = ConstrainedLSTM(config)
model_builder.build()
runner = TrainingRunner(config)
history = runner.train(model_builder.get_model(), splits["train"], splits["val"])
metrics = runner.evaluate(model_builder.get_model(), splits["test"])
```

## Extension Points for Next Prompt

1. **New architectures**: Subclass `BaseTimeseriesModel`, implement `build/get_model/describe`.
2. **New data sources**: `loader.py` already handles CSV and Parquet.
3. **New metrics**: Add to `training/metrics.py`, update `compute_all_metrics`.
4. **Hyperparameter search**: Wrap `TrainingRunner.train()` with Optuna/Ray Tune.
5. **Production serving**: Add `model.export()` → SavedModel/ONNX.

## Skills Applied

| Skill | Where Applied |
|-------|--------------|
| `improving-developer-ux` | Config validation, error messages, sensible defaults, structured logging |
| `designing-ts-architectures` | ADR in constrained_lstm.py, baseline hierarchy, anti-pattern avoidance |
| `teaching-timeseries-dl` | Docstrings, comments explaining encoder-decoder, constraint layer |
| `facilitating-experiments` | ExperimentTracker (hypothesis → decision), EXP-001 lifecycle |
| `training-ts-models` | Temporal splits, early stopping, LR scheduling, pre-training checklist |
