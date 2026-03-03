"""
tssdk.training.metrics — Forecasting evaluation metrics.

Includes standard metrics (MAE, RMSE, MASE) and domain-specific
metrics (constraint violation rate, coverage).
"""

import numpy as np
from typing import Dict, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    MASE < 1 means the model beats seasonal naive.

    Args:
        y_true: Actual values (flattened).
        y_pred: Predicted values (flattened).
        y_train: Training series for computing naive error.
        seasonality: Seasonal period (1 = naive, 12 = yearly for monthly).
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale < 1e-10:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def constraint_violation_rate(
    y_pred: np.ndarray,
    floor: np.ndarray,
    margin: float = 0.0,
) -> float:
    """Percentage of predictions that violate the floor constraint.

    Args:
        y_pred: Predicted values.
        floor: Floor values (shifted contract).
        margin: Required margin above floor.

    Returns:
        Violation rate in [0, 1]. Target: 0.0.
    """
    violations = y_pred < (floor + margin - 1e-6)  # Small tolerance
    return float(np.mean(violations))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    floor: Optional[np.ndarray] = None,
    margin: float = 0.0,
    seasonality: int = 12,
) -> Dict[str, float]:
    """Compute all relevant metrics in one call.

    Returns:
        Dict with keys: mae, rmse, mase (if y_train), violation_rate (if floor).
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

    if y_train is not None and len(y_train) > seasonality:
        metrics["mase"] = mase(y_true, y_pred, y_train, seasonality)

    if floor is not None:
        metrics["constraint_violation_rate"] = constraint_violation_rate(
            y_pred, floor, margin
        )

    return metrics
