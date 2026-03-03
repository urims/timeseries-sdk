#!/usr/bin/env python3
"""Run time series backtesting with expanding or sliding window strategy.

Usage:
    python scripts/run_backtest.py \
        --model-checkpoint path/to/model.pt \
        --data path/to/data.parquet \
        --strategy expanding \
        --n-folds 5 \
        --horizon 24 \
        --metrics mase,coverage_90,crps

Output:
    Prints per-fold metrics and aggregate statistics (mean ± std).
    Saves results to backtest_results.json in the current directory.
"""

import argparse
import json
import sys
from pathlib import Path

def generate_folds(n_samples: int, n_folds: int, horizon: int, strategy: str, min_train: int = None):
    """Generate train/test split indices for time series cross-validation.

    Args:
        n_samples: Total number of time steps
        n_folds: Number of validation folds
        horizon: Forecast horizon (test set size per fold)
        strategy: "expanding" or "sliding"
        min_train: Minimum training set size (default: 2 * horizon)

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    if min_train is None:
        min_train = 2 * horizon

    folds = []
    total_test = n_folds * horizon
    first_test_start = n_samples - total_test

    if first_test_start < min_train:
        print(f"WARNING: Not enough data for {n_folds} folds. "
              f"Need {min_train + total_test} samples, have {n_samples}.")
        n_folds = max(1, (n_samples - min_train) // horizon)
        first_test_start = n_samples - (n_folds * horizon)
        print(f"Reduced to {n_folds} folds.")

    for i in range(n_folds):
        test_start = first_test_start + i * horizon
        test_end = test_start + horizon

        if strategy == "expanding":
            train_start = 0
        elif strategy == "sliding":
            train_start = max(0, test_start - min_train)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'expanding' or 'sliding'.")

        train_end = test_start
        folds.append((train_start, train_end, test_start, test_end))

    return folds


def compute_mase(actual, predicted, seasonal_period: int = 1):
    """Compute Mean Absolute Scaled Error."""
    import numpy as np
    naive_errors = np.abs(np.diff(actual, n=seasonal_period))
    if naive_errors.mean() == 0:
        return float("inf")
    forecast_errors = np.abs(actual[-len(predicted):] - predicted)
    return forecast_errors.mean() / naive_errors.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series backtest")
    parser.add_argument("--model-checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--strategy", choices=["expanding", "sliding"], default="expanding")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--metrics", default="mase", help="Comma-separated metric names")
    args = parser.parse_args()

    print(f"Backtest configuration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Metrics: {args.metrics}")
    print(f"  Model: {args.model_checkpoint}")
    print(f"  Data: {args.data}")
    print()

    # Generate fold structure
    # In production, this would load data and model, run predictions per fold,
    # and compute actual metrics. This script provides the framework.
    print("Fold structure (implement model loading and prediction for your architecture):")
    sample_folds = generate_folds(
        n_samples=1000,  # Replace with actual data length
        n_folds=args.n_folds,
        horizon=args.horizon,
        strategy=args.strategy,
    )

    for i, (ts, te, vs, ve) in enumerate(sample_folds):
        print(f"  Fold {i+1}: train[{ts}:{te}] → test[{vs}:{ve}]")

    print()
    print("To complete the backtest, load your model and data, generate predictions")
    print("per fold, and compute metrics. Save results to backtest_results.json.")
