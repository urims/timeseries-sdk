---
name: guarding-data-quality
description: Detects, diagnoses, and handles data quality issues in time series datasets — including missing values, outliers, structural breaks, leakage, stationarity problems, and distribution drift. Runs validation checks that must pass before any model training begins. Use when working with raw or processed time series data before training or evaluation, or when debugging unexpected model behavior that might stem from data issues. Trigger on "data quality", "missing values", "outliers", "data validation", "check the data", "data issues", "anomalies in data", "stockout", "sensor failure", "data drift", "stationarity", "leakage check", or at the start of any new modeling project. Do not use for model validation metrics — use `validating-models`. Do not use for feature engineering decisions — use `designing-ts-architectures` for architecture-level decisions.
---

# Guarding Data Quality

## Principle

A model trained on bad data is confidently wrong. Run this skill before any training.

## Data Quality Checklist

Run `scripts/check_data_quality.py` on your dataset, then review results:

```bash
python scripts/check_data_quality.py \
    --data path/to/data.parquet \
    --time-column timestamp \
    --target-column sales \
    --frequency daily
```

The script checks for the issues below. If any check fails, address it before training.

### Check 1: Completeness

| Issue | Detection | Recommended Handling |
|-------|-----------|---------------------|
| Missing timestamps | Gaps in time index | Fill with explicit NaN rows, then impute |
| Missing values | NaN count per column | Forward-fill for <3 gaps, model-based for longer |
| Truncated series | Series shorter than lookback + horizon | Exclude or use foundation model zero-shot |

### Check 2: Outliers

| Type | Detection | Handling |
|------|-----------|---------|
| Point outliers | >4 IQR from rolling median | Investigate → clamp or remove if data error |
| Level shifts | CUSUM or structural break test | Segment the series at the break point |
| Impossible values | Negative demand, >24h in a day | Domain constraint validation |

The key question for every outlier: is this a data error or a real event? Stockouts, promotions, and system outages are real — don't remove them without domain confirmation.

### Check 3: Temporal Integrity

- Consistent frequency (no unexpected gaps or duplicates)
- Timezone consistency across all series
- No future data leaking into historical records

### Check 4: Distribution Properties

- Stationarity test (ADF) on the target — non-stationary series need differencing or appropriate model design
- Seasonal decomposition — identify all seasonal periods
- Distribution of train vs validation — significant shift indicates a problem

## Handling Missing Data in Time Series

Priority order (most to least preferred):
1. **Forward-fill** for gaps ≤ natural frequency (e.g., 1–2 missing daily values)
2. **Seasonal interpolation** for gaps spanning seasonal patterns
3. **Model-based imputation** for longer gaps (but flag these predictions as imputed)
4. **Drop the series** if >20% of values are missing

Never use simple mean imputation — it destroys temporal structure.

## Quality Checklist

- [ ] All checks from `scripts/check_data_quality.py` pass or have documented exceptions
- [ ] Outlier handling decisions are recorded with reasoning
- [ ] Missing data strategy is documented
- [ ] No temporal leakage confirmed
- [ ] Data version tagged (DVC hash or S3 version)
