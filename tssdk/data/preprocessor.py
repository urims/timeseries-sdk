"""
tssdk.data.preprocessor — Scale, impute, and prepare timeseries DataFrames.

Handles per-series normalization (critical for heterogeneous series),
null imputation, and series filtering. Maintains scaler state for
inverse transforms at prediction time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from tssdk.config import TimeseriesConfig
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScalerState:
    """Stores per-series scaling parameters for inverse transform."""
    means: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    method: str = "per_series"


class Preprocessor:
    """Stateful preprocessor: fit on train, transform train+val+test.

    Usage:
        prep = Preprocessor(config)
        df_scaled = prep.fit_transform(df_train)       # Fit and transform
        df_val_scaled = prep.transform(df_val)          # Transform only
        df_original = prep.inverse_transform(df_scaled) # Back to original scale

    The preprocessor:
    1. Handles nulls according to config.null_strategy
    2. Filters series shorter than config.min_series_length
    3. Scales features per-series (z-score) or globally
    4. Stores scaler state for inverse transform
    """

    def __init__(self, config: TimeseriesConfig):
        self.config = config
        self.scaler_state = ScalerState(method=config.scaling_method)
        self._is_fitted = False
        self._feature_cols = [config.target_col] + config.covariate_cols

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on data and return transformed copy.

        Args:
            df: DataFrame with date, series_id, target, and covariate columns.

        Returns:
            Scaled DataFrame with nulls handled and short series removed.
        """
        df = self._handle_nulls(df)
        df = self._filter_short_series(df)
        df = self._fit_scalers(df)
        df = self._apply_scaling(df)
        self._is_fitted = True

        n_series = df[self.config.series_id_col].nunique()
        logger.info(f"Preprocessor fit | series={n_series} | method={self.config.scaling_method}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using already-fitted scaler.

        Raises:
            RuntimeError: If fit_transform hasn't been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor not fitted. Call fit_transform() on training data first, "
                "then use transform() on validation/test data."
            )
        df = self._handle_nulls(df)
        df = self._apply_scaling(df)
        return df

    def inverse_transform(
        self,
        values: np.ndarray,
        series_id: str,
        column: str,
    ) -> np.ndarray:
        """Convert scaled values back to original scale.

        Args:
            values: Scaled numpy array.
            series_id: Which series these values belong to.
            column: Which feature column (e.g., 'actual_cost_paid').

        Returns:
            Values in original scale.
        """
        if self.config.scaling_method == "none":
            return values

        if self.config.scaling_method == "per_series":
            mean = self.scaler_state.means[series_id][column]
            std = self.scaler_state.stds[series_id][column]
        else:  # global
            mean = self.scaler_state.means["__global__"][column]
            std = self.scaler_state.stds["__global__"][column]

        std = std if std > 1e-8 else 1.0  # Avoid division by near-zero
        return values * std + mean

    # ── Private methods ──

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values per config strategy."""
        df = df.copy()
        strategy = self.config.null_strategy
        id_col = self.config.series_id_col

        null_counts = df[self._feature_cols].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls == 0:
            return df

        if strategy == "drop_series":
            # Drop entire series that have ANY null in covariates
            series_with_nulls = df.loc[
                df[self._feature_cols].isnull().any(axis=1), id_col
            ].unique()
            before = df[id_col].nunique()
            df = df[~df[id_col].isin(series_with_nulls)]
            after = df[id_col].nunique()
            logger.info(
                f"Null handling (drop_series) | dropped={before - after} series | "
                f"remaining={after}"
            )

        elif strategy == "forward_fill":
            df[self._feature_cols] = df.groupby(id_col)[self._feature_cols].ffill()
            # Backfill any remaining leading nulls
            df[self._feature_cols] = df.groupby(id_col)[self._feature_cols].bfill()
            remaining = df[self._feature_cols].isnull().sum().sum()
            if remaining > 0:
                logger.warning(
                    f"Forward+back fill left {remaining} nulls. "
                    f"These series may be entirely null — consider drop_series strategy."
                )

        elif strategy == "interpolate":
            df[self._feature_cols] = (
                df.groupby(id_col)[self._feature_cols]
                .apply(lambda g: g.interpolate(method="linear").ffill().bfill())
            )

        logger.info(f"Null handling ({strategy}) | nulls_before={total_nulls}")
        return df

    def _filter_short_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove series shorter than min_series_length."""
        id_col = self.config.series_id_col
        min_len = self.config.min_series_length

        lengths = df.groupby(id_col).size()
        valid_ids = lengths[lengths >= min_len].index
        dropped = len(lengths) - len(valid_ids)

        if dropped > 0:
            logger.info(
                f"Filtered short series | dropped={dropped} | "
                f"min_length={min_len} | remaining={len(valid_ids)}"
            )

        return df[df[id_col].isin(valid_ids)].reset_index(drop=True)

    def _fit_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute scaling parameters from training data."""
        method = self.config.scaling_method
        id_col = self.config.series_id_col

        if method == "none":
            return df

        if method == "per_series":
            for series_id, group in df.groupby(id_col):
                self.scaler_state.means[series_id] = {}
                self.scaler_state.stds[series_id] = {}
                for col in self._feature_cols:
                    self.scaler_state.means[series_id][col] = group[col].mean()
                    self.scaler_state.stds[series_id][col] = group[col].std(ddof=0)

        elif method == "global":
            self.scaler_state.means["__global__"] = {}
            self.scaler_state.stds["__global__"] = {}
            for col in self._feature_cols:
                self.scaler_state.means["__global__"][col] = df[col].mean()
                self.scaler_state.stds["__global__"][col] = df[col].std(ddof=0)

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score scaling using fitted parameters."""
        if self.config.scaling_method == "none":
            return df

        df = df.copy()
        id_col = self.config.series_id_col

        if self.config.scaling_method == "per_series":
            for series_id, idx in df.groupby(id_col).groups.items():
                if series_id not in self.scaler_state.means:
                    logger.warning(
                        f"Series '{series_id}' not seen during fit — skipping scaling. "
                        f"This series was likely not in the training data."
                    )
                    continue
                for col in self._feature_cols:
                    mean = self.scaler_state.means[series_id][col]
                    std = self.scaler_state.stds[series_id][col]
                    std = std if std > 1e-8 else 1.0
                    df.loc[idx, col] = (df.loc[idx, col] - mean) / std

        elif self.config.scaling_method == "global":
            for col in self._feature_cols:
                mean = self.scaler_state.means["__global__"][col]
                std = self.scaler_state.stds["__global__"][col]
                std = std if std > 1e-8 else 1.0
                df[col] = (df[col] - mean) / std

        return df
