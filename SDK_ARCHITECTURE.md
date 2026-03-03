# TimeseriesSystem SDK — Architecture Document

## Purpose

A Python SDK for time series deep learning experimentation. Transforms raw tabular data
(date × series_id × features) into DL-ready tensors, and provides the scaffolding to
iterate across architectures, experiments, and configurations reproducibly.

## Design Principles (from `improving-developer-ux`)

1. **Pit of Success** — correct usage requires fewer arguments than incorrect usage.
2. **Progressive Disclosure** — simple case: 2 lines. Power user: full config object.
3. **Error Messages** answer: What happened? Why? How to fix it?
4. **Sensible Defaults** — every optional parameter has a battle-tested default with a comment.

## SDK Module Map

```
tssdk/
├── __init__.py               ← Public API re-exports
├── config.py                 ← Dataclass configs (TimeseriesConfig, SplitConfig, etc.)
├── data/
│   ├── __init__.py
│   ├── loader.py             ← Raw CSV/Parquet → cleaned DataFrame
│   ├── preprocessor.py       ← Scaling, null handling, date parsing
│   ├── windower.py           ← Sliding window → (encoder, decoder, target) arrays
│   └── validator.py          ← Temporal leakage checks, shape assertions
├── models/
│   ├── __init__.py
│   ├── base.py               ← Abstract base for all architectures
│   └── constrained_lstm.py   ← Encoder-decoder LSTM with floor constraint
├── training/
│   ├── __init__.py
│   ├── runner.py             ← Train loop, callbacks, checkpointing
│   └── metrics.py            ← MASE, MAE, RMSE, coverage, constraint violation %
├── experiment/
│   ├── __init__.py
│   └── tracker.py            ← Experiment logging (hypothesis → result → decision)
└── utils/
    ├── __init__.py
    ├── seeds.py              ← Reproducibility seed setter
    └── logging.py            ← Structured logging [timestamp] LEVEL key=value
```

## Data Flow

```
Raw CSV/Parquet
    │
    ▼
┌─────────────────────────────────┐
│  Loader                         │  Read file, validate columns, parse dates
│  loader.load("data.csv")       │  → pd.DataFrame (date, ts_id, features...)
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Preprocessor                   │  Per-series scaling, null imputation,
│  preprocessor.fit_transform()  │  date → ordinal, feature engineering
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Windower                       │  Sliding window extraction
│  windower.create_windows()     │  → (X_encoder, X_decoder, Y) np arrays
│                                 │  Temporal train/val/test split
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Validator                      │  Shape checks, leakage detection,
│  validator.validate()          │  constraint feasibility
└─────────────┬───────────────────┘
              │
              ▼
        Model.fit()
```

## Experiment Lifecycle (from `facilitating-experiments`)

```
1. Hypothesis   →  ExperimentTracker.start(hypothesis="...")
2. Config       →  TimeseriesConfig(model="constrained_lstm", ...)
3. Data prep    →  Pipeline: load → preprocess → window → validate
4. Train        →  Runner.train(model, data, config)
5. Evaluate     →  Runner.evaluate() → metrics dict
6. Decide       →  ExperimentTracker.conclude(decision="accept|reject|iterate")
```

## Architecture Validation Checklist (from `designing-ts-architectures`)

The `constrained_lstm.py` architecture was reviewed against expert standards:

- [x] Encoder-decoder with state transfer (context bridge)
- [x] Known-future covariates handled via decoder input
- [x] Differentiable constraint (LogSumExp smooth max)
- [x] Alpha-lag business logic preserved
- [x] Loss function appropriate (MSE + optional constraint penalty)
- [x] No future leakage in encoder input
- [x] Temporal ordering enforced in data pipeline (not model)

Issues found and fixed:
1. `build_model()` didn't return the model object
2. `constrained_loss` referenced undefined `floor_value` — fixed to use layer output
3. `main()` was entangled with file I/O — separated into SDK pipeline
4. No per-series normalization — added to preprocessor
5. No reproducibility seeds — added via `utils.seeds`

## Baseline Hierarchy (from `designing-ts-architectures`)

Before trusting the constrained LSTM, establish baselines:
1. **Naive**: last value repeated
2. **Seasonal Naive**: same month last year
3. **Linear**: DLinear or simple regression
4. **Constrained LSTM**: the target architecture

## Configuration Contract

```python
@dataclass
class TimeseriesConfig:
    # ── Required (no defaults — force the developer to think)
    target_col: str
    covariate_cols: list[str]

    # ── Architecture
    encoder_length: int = 24       # Lookback window (T)
    decoder_length: int = 6        # Forecast horizon (H)
    alpha_lag: int = 3             # Contract-to-sell delay
    margin: float = 2.0            # Minimum margin above floor

    # ── Training
    batch_size: int = 32           # Works on most GPUs
    learning_rate: float = 1e-3    # Good starting point for Adam
    max_epochs: int = 100          # Early stopping handles actual count
    patience: int = 10             # Epochs without improvement before stop
    validation_ratio: float = 0.2  # Last 20% of time for validation

    # ── Reproducibility
    seed: int = 42                 # Reproducible by default
```
