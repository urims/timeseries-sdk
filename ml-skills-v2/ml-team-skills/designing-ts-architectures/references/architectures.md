# Architecture Deep Dives

## Contents
- TFT (Temporal Fusion Transformer)
- N-BEATS / N-HiTS
- PatchTST / iTransformer
- Foundation models (Chronos, TimesFM, MOIRAI)
- DLinear / NLinear
- DeepAR

---

## TFT (Temporal Fusion Transformer)

**Inductive bias**: Multi-horizon forecasting with variable selection, temporal attention, and static enrichment. Designed for interpretability via attention-based feature importance.

**Best for**: Short-to-medium horizons where stakeholders require explainable forecasts with known future covariates (holidays, promotions, planned events).

**Libraries**: PyTorch Forecasting (`TemporalFusionTransformer`), GluonTS, NeuralForecast

**Common failure modes**:
- Overfits on small series counts (<50 series) without aggressive dropout
- Variable selection can become unstable with highly correlated features
- Slow training compared to N-HiTS for equivalent accuracy on simple problems

**Recommended hyperparameter ranges**:
- hidden_size: 16–160 (start at 32)
- attention_head_size: 1–4
- dropout: 0.1–0.3
- learning_rate: 1e-4 to 1e-2 (use OneCycleLR)
- max_encoder_length: 2–4× forecast horizon

---

## N-BEATS / N-HiTS

**Inductive bias**: Purely backward-looking. Learns basis expansion coefficients via fully connected stacks. N-HiTS adds multi-rate sampling for long horizons.

**Best for**: Univariate or low-covariate problems where fast iteration matters. Excellent baseline that's hard to beat with more complex models on clean data.

**Libraries**: NeuralForecast (`NBEATS`, `NHITS`), Darts

**Common failure modes**:
- No native covariate support in N-BEATS (N-HiTS has partial support via exogenous)
- Stack count too high → overfitting; too low → underfitting (3–5 stacks typical)
- Interpretable mode constrains capacity; generic mode often wins on accuracy

**Recommended hyperparameter ranges**:
- n_stacks: 3–5 (interpretable) or 2–3 (generic)
- n_blocks per stack: 1–3
- hidden_size: 128–512
- learning_rate: 1e-4 to 5e-3

---

## PatchTST / iTransformer

**Inductive bias**: PatchTST patches the time dimension (like ViT patches images), enabling efficient self-attention over long sequences. iTransformer inverts the standard approach — applies attention across variate dimension, not time.

**Best for**: Long-horizon forecasting with many series. PatchTST excels at capturing long-range temporal dependencies. iTransformer handles multivariate cross-series correlations.

**Libraries**: NeuralForecast, HuggingFace (PatchTST), custom implementations

**Common failure modes**:
- PatchTST: Patch size too large relative to seasonality → misses patterns
- iTransformer: Performance degrades when series have very different scales
- Both: Require more data than N-HiTS to realize their advantage

**Recommended hyperparameter ranges**:
- patch_len: 8–24 (align with seasonal period fractions)
- stride: patch_len // 2 (50% overlap default)
- d_model: 64–256
- n_heads: 4–8
- n_layers: 2–4

---

## Foundation Models (Chronos, TimesFM, MOIRAI)

**Inductive bias**: Pre-trained on massive corpora of time series data. Designed for zero-shot or few-shot forecasting without task-specific training.

**Best for**: Cold-start problems, rapid prototyping, situations where you lack training data or compute for task-specific models. Good as strong baselines.

**Libraries**: Chronos (Amazon, HuggingFace), TimesFM (Google), MOIRAI (Salesforce)

**Common failure modes**:
- Struggle with domain-specific patterns not represented in pre-training data
- Quantile outputs may be miscalibrated for your specific domain
- Inference latency can be high for large models (Chronos-large)
- Limited ability to incorporate exogenous covariates (Chronos has none)

**When to fine-tune vs. use zero-shot**:
- Zero-shot: <500 data points, rapid prototyping, baseline comparison
- Fine-tune: >5K data points, domain-specific patterns, production deployment

---

## DLinear / NLinear

**Inductive bias**: Simple linear projection from lookback window to forecast horizon. NLinear adds normalization to handle distribution shift.

**Best for**: Strong baseline that's embarrassingly hard to beat. Production inference at <1ms latency. Situations where model complexity must be justified.

**Libraries**: NeuralForecast, simple custom implementation (~20 lines of PyTorch)

**Common failure modes**:
- Cannot capture complex nonlinear patterns (by design)
- No probabilistic output without modification
- Lookback window selection is critical — too short misses patterns, too long adds noise

---

## DeepAR

**Inductive bias**: Autoregressive RNN that models the full predictive distribution. Generates probabilistic forecasts natively via sampling.

**Best for**: Probabilistic forecasting where you need well-calibrated uncertainty estimates. Particularly strong with many similar time series (global model).

**Libraries**: GluonTS (`DeepAREstimator`), PyTorch Forecasting

**Common failure modes**:
- Autoregressive generation means errors compound at long horizons
- Slow inference due to sequential sampling
- Requires careful handling of missing values in the training window
