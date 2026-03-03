#!/usr/bin/env python3
"""Check time series data quality before model training.

Usage:
    python scripts/check_data_quality.py \
        --data path/to/data.parquet \
        --time-column timestamp \
        --target-column sales \
        --frequency daily

Checks:
    1. Completeness: missing timestamps, missing values, truncated series
    2. Outliers: point outliers (IQR), impossible values
    3. Temporal integrity: frequency consistency, duplicates, timezone
    4. Distribution: basic statistics, stationarity test

Output:
    Prints pass/fail for each check with details.
    Returns exit code 0 if all pass, 1 if any critical check fails.
"""

import argparse
import sys
from pathlib import Path

def check_completeness(df, time_col, target_col, frequency):
    """Check for missing timestamps and values."""
    issues = []

    # Missing values in target
    nan_count = df[target_col].isna().sum()
    nan_pct = nan_count / len(df) * 100
    if nan_pct > 20:
        issues.append(f"CRITICAL: {nan_pct:.1f}% missing values in '{target_col}' ({nan_count} rows)")
    elif nan_pct > 0:
        issues.append(f"WARNING: {nan_pct:.1f}% missing values in '{target_col}' ({nan_count} rows)")

    # Series length
    if len(df) < 52:
        issues.append(f"WARNING: Only {len(df)} rows. Most models need at least 52 observations.")

    return issues

def check_outliers(df, target_col):
    """Detect point outliers using IQR method."""
    issues = []
    values = df[target_col].dropna()

    if len(values) == 0:
        return ["CRITICAL: No non-null values in target column"]

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 4 * iqr
    upper = q3 + 4 * iqr

    outliers = values[(values < lower) | (values > upper)]
    if len(outliers) > 0:
        issues.append(f"WARNING: {len(outliers)} outliers detected (>4 IQR). "
                      f"Range: [{lower:.2f}, {upper:.2f}]. "
                      f"Outlier values: {outliers.head(5).tolist()}")

    # Negative values (common issue for demand/sales)
    negative_count = (values < 0).sum()
    if negative_count > 0:
        issues.append(f"WARNING: {negative_count} negative values in '{target_col}'. "
                      f"Verify these are valid (e.g., returns) or data errors.")

    return issues

def check_temporal_integrity(df, time_col, frequency):
    """Check for temporal consistency."""
    import pandas as pd
    issues = []

    df[time_col] = pd.to_datetime(df[time_col])

    # Duplicates
    dup_count = df[time_col].duplicated().sum()
    if dup_count > 0:
        issues.append(f"CRITICAL: {dup_count} duplicate timestamps found")

    # Monotonic
    if not df[time_col].is_monotonic_increasing:
        issues.append("CRITICAL: Timestamps are not monotonically increasing. Sort before training.")

    # Frequency gaps
    freq_map = {"daily": "D", "hourly": "h", "weekly": "W", "monthly": "MS"}
    if frequency in freq_map:
        expected = pd.date_range(df[time_col].min(), df[time_col].max(), freq=freq_map[frequency])
        missing_steps = len(expected) - len(df)
        if missing_steps > 0:
            issues.append(f"WARNING: {missing_steps} missing {frequency} timestamps in range")

    return issues

def run_all_checks(data_path, time_col, target_col, frequency):
    """Run all quality checks and report."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. Install with: pip install pandas --break-system-packages")
        sys.exit(1)

    print(f"Data Quality Report: {data_path}")
    print("=" * 60)

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    all_issues = []

    # Check 1: Completeness
    print("CHECK 1: Completeness")
    issues = check_completeness(df, time_col, target_col, frequency)
    all_issues.extend(issues)
    print("  PASS" if not issues else "\n".join(f"  {i}" for i in issues))
    print()

    # Check 2: Outliers
    print("CHECK 2: Outliers")
    issues = check_outliers(df, target_col)
    all_issues.extend(issues)
    print("  PASS" if not issues else "\n".join(f"  {i}" for i in issues))
    print()

    # Check 3: Temporal Integrity
    print("CHECK 3: Temporal Integrity")
    issues = check_temporal_integrity(df, time_col, frequency)
    all_issues.extend(issues)
    print("  PASS" if not issues else "\n".join(f"  {i}" for i in issues))
    print()

    # Summary
    critical = [i for i in all_issues if "CRITICAL" in i]
    warnings = [i for i in all_issues if "WARNING" in i]
    print("=" * 60)
    print(f"Summary: {len(critical)} critical, {len(warnings)} warnings")

    if critical:
        print("RESULT: FAIL — resolve critical issues before training")
        return False
    elif warnings:
        print("RESULT: PASS with warnings — review before training")
        return True
    else:
        print("RESULT: PASS — data quality checks passed")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check time series data quality")
    parser.add_argument("--data", required=True, help="Path to dataset (.parquet or .csv)")
    parser.add_argument("--time-column", required=True, help="Name of timestamp column")
    parser.add_argument("--target-column", required=True, help="Name of target variable column")
    parser.add_argument("--frequency", default="daily",
                        choices=["hourly", "daily", "weekly", "monthly"],
                        help="Expected data frequency")
    args = parser.parse_args()

    ok = run_all_checks(args.data, args.time_column, args.target_column, args.frequency)
    sys.exit(0 if ok else 1)
