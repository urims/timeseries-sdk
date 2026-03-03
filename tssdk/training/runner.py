"""
tssdk.training.runner — Training loop with callbacks, early stopping, and logging.

Wraps Keras model.fit() with structured logging, early stopping,
and experiment-aware checkpointing.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
from pathlib import Path

from tssdk.config import TimeseriesConfig
from tssdk.data.windower import WindowedDataset
from tssdk.training.metrics import compute_all_metrics
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


class TrainingRunner:
    """Run training with structured logging and early stopping.

    Usage:
        runner = TrainingRunner(config)
        history = runner.train(model, splits["train"], splits["val"])
        metrics = runner.evaluate(model, splits["test"])
    """

    def __init__(self, config: TimeseriesConfig):
        self.config = config

    def train(
        self,
        model: tf.keras.Model,
        train_data: WindowedDataset,
        val_data: Optional[WindowedDataset] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the model with early stopping and logging.

        Args:
            model: Compiled Keras model.
            train_data: Training WindowedDataset.
            val_data: Validation WindowedDataset (recommended).
            checkpoint_dir: Directory for saving best model. If None, no checkpointing.

        Returns:
            Dict with 'history' (Keras history) and 'best_epoch'.
        """
        c = self.config

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_data else "loss",
                patience=c.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_data else "loss",
                factor=0.5,
                patience=c.patience // 2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(Path(checkpoint_dir) / "best_model.keras"),
                    monitor="val_loss" if val_data else "loss",
                    save_best_only=True,
                    verbose=0,
                )
            )

        # Prepare validation data
        validation_data = None
        if val_data and len(val_data.X_encoder) > 0:
            validation_data = (
                [val_data.X_encoder, val_data.X_decoder],
                val_data.Y,
            )

        logger.info(
            f"Training started | "
            f"train_samples={len(train_data.X_encoder)} | "
            f"val_samples={len(val_data.X_encoder) if val_data else 0} | "
            f"epochs={c.max_epochs} | batch_size={c.batch_size} | lr={c.learning_rate}"
        )

        history = model.fit(
            [train_data.X_encoder, train_data.X_decoder],
            train_data.Y,
            epochs=c.max_epochs,
            batch_size=c.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = (
            np.argmin(history.history.get("val_loss", history.history["loss"])) + 1
        )

        logger.info(
            f"Training complete | "
            f"best_epoch={best_epoch} | "
            f"final_loss={history.history['loss'][-1]:.6f}"
        )

        return {"history": history.history, "best_epoch": best_epoch}

    def evaluate(
        self,
        model: tf.keras.Model,
        test_data: WindowedDataset,
        preprocessor=None,
    ) -> Dict[str, float]:
        """Evaluate model on test data and compute all metrics.

        Args:
            model: Trained Keras model.
            test_data: Test WindowedDataset.
            preprocessor: Optional Preprocessor for inverse-transforming predictions.

        Returns:
            Dict of metric name → value.
        """
        if len(test_data.X_encoder) == 0:
            logger.warning("Test data is empty — skipping evaluation")
            return {}

        y_pred = model.predict(
            [test_data.X_encoder, test_data.X_decoder],
            verbose=0,
        )
        y_true = test_data.Y

        metrics = compute_all_metrics(
            y_true=y_true.flatten(),
            y_pred=y_pred.flatten(),
        )

        logger.info(
            f"Evaluation | " +
            " | ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        )

        return metrics

    def predict(
        self,
        model: tf.keras.Model,
        dataset: WindowedDataset,
    ) -> np.ndarray:
        """Generate predictions for a dataset.

        Args:
            model: Trained model.
            dataset: WindowedDataset to predict on.

        Returns:
            Predictions array with shape (samples, H, 1).
        """
        return model.predict(
            [dataset.X_encoder, dataset.X_decoder],
            verbose=0,
        )
