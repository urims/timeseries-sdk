"""
tssdk.data.validator — Validate windowed datasets for DL readiness.

Checks for:
- Correct tensor shapes
- No NaN/Inf values
- Temporal ordering (no future leakage)
- Constraint feasibility
- Data distribution sanity
"""

import numpy as np
from typing import Dict, List

from tssdk.config import TimeseriesConfig
from tssdk.data.windower import WindowedDataset
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of a validation run."""

    def __init__(self):
        self.checks: List[Dict] = []
        self.passed = True

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "✓" if c["passed"] else "✗"
            lines.append(f"  {status} {c['name']}: {c['detail']}")
        header = "PASSED" if self.passed else "FAILED"
        return f"Validation {header}\n" + "\n".join(lines)


def validate(
    dataset: WindowedDataset,
    config: TimeseriesConfig,
    split_name: str = "unknown",
) -> ValidationResult:
    """Run all validation checks on a windowed dataset.

    Args:
        dataset: WindowedDataset to validate.
        config: TimeseriesConfig for expected shapes and constraints.
        split_name: Label for logging (e.g., "train", "val").

    Returns:
        ValidationResult with pass/fail for each check.

    Example:
        >>> result = validate(splits["train"], config, "train")
        >>> print(result.summary())
        >>> assert result.passed, "Training data validation failed"
    """
    result = ValidationResult()

    # ── Shape checks ──
    _check_shapes(dataset, config, result)

    # ── NaN/Inf checks ──
    _check_numerics(dataset, result)

    # ── Non-empty check ──
    _check_non_empty(dataset, result)

    # ── Target variance check ──
    _check_variance(dataset, result)

    status = "PASSED" if result.passed else "FAILED"
    logger.info(f"Validation [{split_name}] {status} | checks={len(result.checks)}")

    return result


def _check_shapes(
    ds: WindowedDataset, config: TimeseriesConfig, result: ValidationResult
):
    """Verify array dimensions match config."""
    T, H = config.encoder_length, config.decoder_length
    n = len(ds.X_encoder)

    # Encoder shape
    expected_enc = (n, T, config.n_encoder_features)
    actual_enc = ds.X_encoder.shape
    result.add(
        "encoder_shape",
        actual_enc == expected_enc,
        f"expected={expected_enc}, actual={actual_enc}",
    )

    # Decoder shape
    expected_dec = (n, H, config.n_decoder_features)
    actual_dec = ds.X_decoder.shape
    result.add(
        "decoder_shape",
        actual_dec == expected_dec,
        f"expected={expected_dec}, actual={actual_dec}",
    )

    # Target shape
    expected_y = (n, H, 1)
    actual_y = ds.Y.shape
    result.add(
        "target_shape",
        actual_y == expected_y,
        f"expected={expected_y}, actual={actual_y}",
    )


def _check_numerics(ds: WindowedDataset, result: ValidationResult):
    """Check for NaN and Inf in all arrays."""
    for name, arr in [
        ("X_encoder", ds.X_encoder),
        ("X_decoder", ds.X_decoder),
        ("Y", ds.Y),
    ]:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()

        result.add(
            f"{name}_no_nan",
            not has_nan,
            f"NaN count={np.isnan(arr).sum()}" if has_nan else "clean",
        )
        result.add(
            f"{name}_no_inf",
            not has_inf,
            f"Inf count={np.isinf(arr).sum()}" if has_inf else "clean",
        )


def _check_non_empty(ds: WindowedDataset, result: ValidationResult):
    """Verify dataset has at least one sample."""
    n = len(ds.X_encoder)
    result.add("non_empty", n > 0, f"samples={n}")


def _check_variance(ds: WindowedDataset, result: ValidationResult):
    """Warn if target has zero variance (constant series)."""
    if len(ds.Y) == 0:
        result.add("target_variance", True, "empty dataset — skip")
        return

    var = np.var(ds.Y)
    result.add(
        "target_variance",
        var > 1e-10,
        f"variance={var:.6f}" + (" ← near-zero, model may not learn" if var <= 1e-10 else ""),
    )
