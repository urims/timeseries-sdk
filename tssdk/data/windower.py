"""
tssdk.data.windower — Sliding window extraction for encoder-decoder architectures.

Converts a preprocessed DataFrame into numpy arrays shaped for DL models:
  - X_encoder: (samples, T, n_features)  — historical context
  - X_decoder: (samples, H, n_covariates) — known future covariates
  - Y:         (samples, H, 1)            — future target values

Handles temporal train/val/test splitting (NEVER random).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

from tssdk.config import TimeseriesConfig
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


class WindowedDataset(NamedTuple):
    """Container for windowed encoder-decoder arrays."""
    X_encoder: np.ndarray   # (samples, T, n_encoder_features)
    X_decoder: np.ndarray   # (samples, H, n_decoder_features)
    Y: np.ndarray           # (samples, H, 1)
    series_ids: np.ndarray  # (samples,) — which series each sample came from
    timestamps: np.ndarray  # (samples,) — forecast origin timestamp for each sample


class Windower:
    """Extract sliding windows from timeseries DataFrames.

    Usage:
        windower = Windower(config)
        splits = windower.create_splits(df)
        # splits["train"].X_encoder.shape → (n_train, T, 2)
        # splits["val"].X_encoder.shape   → (n_val, T, 2)
        # splits["test"].X_encoder.shape  → (n_test, T, 2)

    Temporal split strategy (from training-ts-models skill):
    - Train: earliest data up to cutoff_train
    - Val:   cutoff_train to cutoff_val
    - Test:  cutoff_val to end
    Windows are extracted within each split; no window crosses split boundaries.
    """

    def __init__(self, config: TimeseriesConfig):
        self.config = config
        self.T = config.encoder_length
        self.H = config.decoder_length

    def create_windows(self, df: pd.DataFrame) -> WindowedDataset:
        """Extract all possible sliding windows from a DataFrame.

        Args:
            df: Preprocessed DataFrame sorted by (series_id, date).

        Returns:
            WindowedDataset with encoder, decoder, and target arrays.
        """
        config = self.config
        all_enc, all_dec, all_y, all_ids, all_ts = [], [], [], [], []

        for series_id, group in df.groupby(config.series_id_col):
            group = group.sort_values(config.date_col).reset_index(drop=True)

            # Build feature matrices
            target = group[config.target_col].values.reshape(-1, 1)
            covariates = group[config.covariate_cols].values
            dates = group[config.date_col].values

            # Encoder features: [target, covariates]
            encoder_features = np.concatenate([target, covariates], axis=1)

            n = len(group)
            window_size = self.T + self.H

            if n < window_size:
                logger.warning(
                    f"Series '{series_id}' too short for windowing | "
                    f"length={n} < T+H={window_size} — skipping"
                )
                continue

            # Slide the window
            for start in range(n - window_size + 1):
                enc_end = start + self.T
                dec_end = enc_end + self.H

                # Encoder: T steps of [target, covariates]
                all_enc.append(encoder_features[start:enc_end])

                # Decoder: H steps of [covariates only] (known future)
                all_dec.append(covariates[enc_end:dec_end])

                # Target: H steps of [target only]
                all_y.append(target[enc_end:dec_end])

                # Metadata
                all_ids.append(series_id)
                all_ts.append(dates[enc_end - 1])  # Forecast origin

        if not all_enc:
            raise ValueError(
                f"No valid windows could be extracted. "
                f"Check that your series are long enough: need ≥ {self.T + self.H} timesteps "
                f"(encoder_length={self.T} + decoder_length={self.H})."
            )

        dataset = WindowedDataset(
            X_encoder=np.array(all_enc, dtype=np.float32),
            X_decoder=np.array(all_dec, dtype=np.float32),
            Y=np.array(all_y, dtype=np.float32),
            series_ids=np.array(all_ids),
            timestamps=np.array(all_ts),
        )

        logger.info(
            f"Windowed | samples={len(dataset.X_encoder)} | "
            f"encoder={dataset.X_encoder.shape} | "
            f"decoder={dataset.X_decoder.shape} | "
            f"target={dataset.Y.shape}"
        )

        return dataset

    def create_splits(
        self, df: pd.DataFrame
    ) -> Dict[str, WindowedDataset]:
        """Create temporal train/val/test splits, then window each.

        Split strategy: per-series temporal ordering.
        Train windows come from the earliest portion of each series,
        val from the middle, test from the end. No window crosses boundaries.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            Dict with keys "train", "val", "test", each a WindowedDataset.
        """
        config = self.config
        train_dfs, val_dfs, test_dfs = [], [], []

        for series_id, group in df.groupby(config.series_id_col):
            group = group.sort_values(config.date_col).reset_index(drop=True)
            n = len(group)

            # Temporal cutoffs
            test_size = max(1, int(n * config.test_ratio))
            val_size = max(1, int(n * config.validation_ratio))
            train_size = n - val_size - test_size

            if train_size < self.T + self.H:
                logger.warning(
                    f"Series '{series_id}' train split too short for windowing | "
                    f"train_size={train_size} < T+H={self.T + self.H} — skipping series"
                )
                continue

            train_dfs.append(group.iloc[:train_size])
            val_dfs.append(group.iloc[train_size:train_size + val_size])
            test_dfs.append(group.iloc[train_size + val_size:])

        splits = {}
        for name, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
            if not dfs:
                logger.warning(f"No data for '{name}' split")
                splits[name] = WindowedDataset(
                    X_encoder=np.array([], dtype=np.float32).reshape(0, self.T, config.n_encoder_features),
                    X_decoder=np.array([], dtype=np.float32).reshape(0, self.H, config.n_decoder_features),
                    Y=np.array([], dtype=np.float32).reshape(0, self.H, 1),
                    series_ids=np.array([]),
                    timestamps=np.array([]),
                )
                continue

            combined = pd.concat(dfs, ignore_index=True)

            # For val/test: we need encoder context that extends back into train
            # So we use a "context-aware" approach: include preceding T steps
            if name in ("val", "test"):
                splits[name] = self._create_windows_with_context(
                    combined, df, name
                )
            else:
                splits[name] = self.create_windows(combined)

        logger.info(
            f"Splits created | "
            f"train={len(splits['train'].X_encoder)} | "
            f"val={len(splits['val'].X_encoder)} | "
            f"test={len(splits['test'].X_encoder)}"
        )

        return splits

    def _create_windows_with_context(
        self,
        split_df: pd.DataFrame,
        full_df: pd.DataFrame,
        split_name: str,
    ) -> WindowedDataset:
        """Create windows for val/test with encoder context from preceding data.

        For val/test splits, the first few windows need encoder history that
        extends before the split boundary. We grab that context from the full df.
        """
        config = self.config
        all_enc, all_dec, all_y, all_ids, all_ts = [], [], [], [], []

        for series_id, group in split_df.groupby(config.series_id_col):
            # Get the full series to provide context
            full_series = full_df[
                full_df[config.series_id_col] == series_id
            ].sort_values(config.date_col).reset_index(drop=True)

            # Find where this split starts in the full series
            split_dates = group[config.date_col].values
            if len(split_dates) == 0:
                continue

            first_split_date = split_dates[0]
            full_dates = full_series[config.date_col].values
            split_start_idx = np.searchsorted(full_dates, first_split_date)

            # We need at least T steps before the first split date for context
            context_start = max(0, split_start_idx - self.T)

            # Work with the context-extended portion
            extended = full_series.iloc[context_start:].reset_index(drop=True)

            target = extended[config.target_col].values.reshape(-1, 1)
            covariates = extended[config.covariate_cols].values
            dates = extended[config.date_col].values
            encoder_features = np.concatenate([target, covariates], axis=1)

            n = len(extended)
            # Find the offset where actual split data starts
            offset = split_start_idx - context_start

            # Windows must have their forecast origin within the split
            for start in range(max(0, offset - self.T), n - self.T - self.H + 1):
                enc_end = start + self.T
                dec_end = enc_end + self.H

                # Only include if the forecast origin is within the split
                origin_date = dates[enc_end - 1]
                if origin_date < first_split_date:
                    continue

                all_enc.append(encoder_features[start:enc_end])
                all_dec.append(covariates[enc_end:dec_end])
                all_y.append(target[enc_end:dec_end])
                all_ids.append(series_id)
                all_ts.append(origin_date)

        if not all_enc:
            return WindowedDataset(
                X_encoder=np.array([], dtype=np.float32).reshape(0, self.T, config.n_encoder_features),
                X_decoder=np.array([], dtype=np.float32).reshape(0, self.H, config.n_decoder_features),
                Y=np.array([], dtype=np.float32).reshape(0, self.H, 1),
                series_ids=np.array([]),
                timestamps=np.array([]),
            )

        return WindowedDataset(
            X_encoder=np.array(all_enc, dtype=np.float32),
            X_decoder=np.array(all_dec, dtype=np.float32),
            Y=np.array(all_y, dtype=np.float32),
            series_ids=np.array(all_ids),
            timestamps=np.array(all_ts),
        )
