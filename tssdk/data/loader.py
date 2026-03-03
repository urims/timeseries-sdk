"""
tssdk.data.loader — Load raw timeseries files into validated DataFrames.

Supports CSV and Parquet. Validates required columns exist and
parses YYYYMM integer dates into proper datetime objects.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from tssdk.config import TimeseriesConfig
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


def load(
    path: str,
    config: TimeseriesConfig,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a timeseries file and validate its structure.

    Args:
        path: Path to CSV or Parquet file.
        config: TimeseriesConfig specifying column names and requirements.
        columns: Optional subset of columns to load. If None, loads all.

    Returns:
        pd.DataFrame with validated columns and parsed dates.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If required columns are missing from the file.

    Example:
        >>> config = TimeseriesConfig(target_col="actual_cost_paid", covariate_cols=["contract_value"])
        >>> df = load("data.csv", config)
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {path}. "
            f"Check the path and ensure the file exists."
        )

    # ── Read file ──
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    elif filepath.suffix in (".csv", ".tsv"):
        sep = "\t" if filepath.suffix == ".tsv" else ","
        df = pd.read_csv(path, usecols=columns, sep=sep)
    else:
        raise ValueError(
            f"Unsupported file format: '{filepath.suffix}'. "
            f"Supported formats: .csv, .tsv, .parquet"
        )

    # ── Validate required columns ──
    required_cols = [config.date_col, config.series_id_col, config.target_col] + config.covariate_cols
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Update your TimeseriesConfig or check your data file."
        )

    # ── Parse YYYYMM integer dates ──
    df = _parse_dates(df, config.date_col)

    # ── Sort by series and time ──
    df = df.sort_values([config.series_id_col, config.date_col]).reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} rows | "
        f"series={df[config.series_id_col].nunique()} | "
        f"date_range={df[config.date_col].min()} → {df[config.date_col].max()}"
    )

    return df


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert YYYYMM integer dates to datetime.

    Handles both integer formats (202001) and already-parsed datetimes.
    """
    df = df.copy()
    sample = df[date_col].iloc[0]

    if isinstance(sample, (int, np.integer)):
        # YYYYMM format → datetime
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str), format="%Y%m"
        )
        logger.info(f"Parsed YYYYMM integer dates → datetime")
    elif isinstance(sample, str):
        # Try common formats
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="%Y%m")
        except ValueError:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
        logger.info(f"Parsed string dates → datetime")
    # else: assume already datetime

    return df
