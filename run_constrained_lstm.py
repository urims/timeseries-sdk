"""
run_constrained_lstm.py — First experiment: Constrained LSTM on timeseries_sample.csv

Experiment: EXP-001
Hypothesis: The constrained encoder-decoder LSTM can learn the relationship between
            actual_cost_paid and contract_value with 0% constraint violations,
            while achieving MAE < naive baseline on the test set.

This script demonstrates the full SDK workflow:
    1. Configure → 2. Prepare data → 3. Build model → 4. Train → 5. Evaluate → 6. Record

Usage:
    python run_constrained_lstm.py
"""

import sys
import os
import numpy as np

# ── Add SDK to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tssdk import (
    TimeseriesConfig,
    Pipeline,
    ConstrainedLSTM,
    TrainingRunner,
    ExperimentTracker,
    compute_all_metrics,
    set_all_seeds,
)
from tssdk.utils.sdk_logging import get_logger

logger = get_logger("run_constrained_lstm")


def main():
    # ═══════════════════════════════════════════════
    # 1. CONFIGURE
    # ═══════════════════════════════════════════════
    config = TimeseriesConfig(
        # Required — the team must specify these
        target_col="actual_cost_paid",
        covariate_cols=["contract_value"],

        # Column identifiers
        date_col="date",
        series_id_col="ts_id",

        # Architecture (tuned for this dataset)
        encoder_length=24,          # 24 months lookback
        decoder_length=6,           # 6 months forecast
        alpha_lag=3,                # Contract affects cost 3 months later
        margin=0.0,                 # No minimum margin for first experiment
        constraint_temp=5.0,        # Smooth constraint sharpness

        # Encoder/Decoder sizing (smaller for limited data)
        encoder_hidden_1=64,        # Reduced from 128 — only 5 series
        encoder_hidden_2=32,        # Reduced from 64
        decoder_hidden=32,          # Match encoder output

        # Training
        batch_size=16,              # Small batches for small dataset
        learning_rate=1e-3,
        max_epochs=100,
        patience=15,                # More patience for small data
        validation_ratio=0.15,
        test_ratio=0.10,

        # Preprocessing
        scaling_method="per_series",    # Critical: series have very different scales
        null_strategy="drop_series",    # LP4 series is 100% null on contract — drop it
        min_series_length=30,           # Need enough for T+H windows

        # Reproducibility
        seed=42,
    )

    # ── Set seeds ──
    set_all_seeds(config.seed)
    logger.info(f"Seeds set to {config.seed}")

    # ═══════════════════════════════════════════════
    # 2. PREPARE DATA
    # ═══════════════════════════════════════════════
    data_path = os.path.join(os.path.dirname(__file__), "timeseries_sample.csv")

    if not os.path.exists(data_path):
        # Try project directory
        data_path = "/mnt/project/timeseries_sample.csv"

    logger.info(f"Loading data from: {data_path}")

    pipe = Pipeline(config)
    splits = pipe.prepare(data_path, verbose=True)

    print(f"\n{'='*60}")
    print(f"Data Summary:")
    print(f"  Train: {splits['train'].X_encoder.shape[0]} windows")
    print(f"  Val:   {splits['val'].X_encoder.shape[0]} windows")
    print(f"  Test:  {splits['test'].X_encoder.shape[0]} windows")
    print(f"  Encoder shape: {splits['train'].X_encoder.shape}")
    print(f"  Decoder shape: {splits['train'].X_decoder.shape}")
    print(f"  Target shape:  {splits['train'].Y.shape}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════
    # 3. BUILD MODEL
    # ═══════════════════════════════════════════════
    model_builder = ConstrainedLSTM(config)
    model_builder.build()
    model = model_builder.get_model()
    model.summary()

    # ═══════════════════════════════════════════════
    # 4. START EXPERIMENT
    # ═══════════════════════════════════════════════
    tracker = ExperimentTracker(output_dir="experiments")
    tracker.start(
        experiment_id="EXP-001",
        hypothesis=(
            "The constrained encoder-decoder LSTM can learn the actual_cost_paid ↔ "
            "contract_value relationship with 0% constraint violations, while "
            "achieving MAE lower than naive baseline on the test split."
        ),
        config={k: str(v) for k, v in config.__dict__.items()},
        model_description=model_builder.describe(),
    )

    # ═══════════════════════════════════════════════
    # 5. TRAIN
    # ═══════════════════════════════════════════════
    runner = TrainingRunner(config)
    train_result = runner.train(
        model=model,
        train_data=splits["train"],
        val_data=splits["val"],
        checkpoint_dir="experiments/EXP-001/checkpoints",
    )

    # ═══════════════════════════════════════════════
    # 6. EVALUATE
    # ═══════════════════════════════════════════════
    test_metrics = runner.evaluate(model, splits["test"])

    # ── Compute naive baseline ──
    naive_preds = splits["test"].X_encoder[:, -1, 0:1]  # Last known target value
    naive_preds = np.tile(naive_preds, (1, config.decoder_length)).reshape(-1)
    naive_mae = float(np.mean(np.abs(splits["test"].Y.flatten() - naive_preds)))

    test_metrics["naive_mae"] = naive_mae
    test_metrics["improvement_vs_naive"] = (naive_mae - test_metrics["mae"]) / naive_mae

    print(f"\n{'='*60}")
    print(f"Results — EXP-001: Constrained LSTM")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"\n  Naive MAE:            {naive_mae:.6f}")
    print(f"  Model MAE:            {test_metrics['mae']:.6f}")
    print(f"  Improvement vs naive: {test_metrics['improvement_vs_naive']:.1%}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════
    # 7. CONCLUDE EXPERIMENT
    # ═══════════════════════════════════════════════
    tracker.log_metrics(test_metrics)

    if test_metrics["mae"] < naive_mae:
        decision = "accept"
        notes = f"Model MAE ({test_metrics['mae']:.4f}) beats naive ({naive_mae:.4f})"
    else:
        decision = "iterate"
        notes = f"Model MAE ({test_metrics['mae']:.4f}) ≥ naive ({naive_mae:.4f}) — needs tuning"

    experiment_path = tracker.conclude(decision=decision, notes=notes)
    print(f"Experiment record saved: {experiment_path}")


if __name__ == "__main__":
    main()
