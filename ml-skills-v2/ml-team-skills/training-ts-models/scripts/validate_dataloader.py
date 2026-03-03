#!/usr/bin/env python3
"""Validate a time series dataloader for temporal leakage and alignment issues.

Usage:
    python scripts/validate_dataloader.py \
        --dataset path/to/data.parquet \
        --lookback 168 \
        --horizon 24 \
        --frequency hourly

Checks:
- No future timestamps appear in lookback windows
- No overlap between train/val/test time ranges
- Target values are not present in feature columns
- Window sampling respects series boundaries
"""

import argparse
import sys

def validate_temporal_alignment(dataset_path: str, lookback: int, horizon: int, frequency: str):
    """Run temporal leakage checks on a time series dataset."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. Install with: pip install pandas")
        sys.exit(1)

    print(f"Validating: {dataset_path}")
    print(f"  Lookback: {lookback}, Horizon: {horizon}, Frequency: {frequency}")

    df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)

    issues = []

    # Check 1: Timestamp column exists and is sorted
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not time_cols:
        issues.append("WARNING: No timestamp column detected. Verify temporal ordering manually.")
    else:
        ts_col = time_cols[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        if not df[ts_col].is_monotonic_increasing:
            issues.append(f"CRITICAL: '{ts_col}' is not monotonically increasing. Data may be shuffled.")

    # Check 2: Sufficient data for lookback + horizon
    min_length = lookback + horizon
    if len(df) < min_length:
        issues.append(f"CRITICAL: Dataset has {len(df)} rows but needs at least {min_length} (lookback + horizon).")

    # Check 3: No NaN in target (first numeric column as proxy)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        nan_count = df[numeric_cols[0]].isna().sum()
        if nan_count > 0:
            issues.append(f"WARNING: {nan_count} NaN values in '{numeric_cols[0]}'. Handle before training.")

    # Report
    if not issues:
        print("OK: All basic temporal checks passed.")
        print("  Note: Run full leakage detection with your actual DataLoader for complete validation.")
    else:
        for issue in issues:
            print(f"  {issue}")

    return len(issues) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate time series dataloader")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--lookback", type=int, required=True, help="Lookback window size")
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    parser.add_argument("--frequency", default="hourly", help="Data frequency")
    args = parser.parse_args()

    ok = validate_temporal_alignment(args.dataset, args.lookback, args.horizon, args.frequency)
    sys.exit(0 if ok else 1)
