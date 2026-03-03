"""
tssdk.config — Configuration contracts for the Timeseries DL SDK.

Design: Required parameters have no defaults (force the developer to think).
        Optional parameters have battle-tested defaults with comments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimeseriesConfig:
    """Master configuration for a timeseries DL experiment.

    Usage (simple):
        config = TimeseriesConfig(
            target_col="actual_cost_paid",
            covariate_cols=["contract_value"],
        )

    Usage (power user):
        config = TimeseriesConfig(
            target_col="actual_cost_paid",
            covariate_cols=["contract_value"],
            encoder_length=36,
            decoder_length=12,
            alpha_lag=3,
        )
    """

    # ── Required — no defaults for critical choices ──
    target_col: str
    covariate_cols: List[str]

    # ── Column identifiers ──
    date_col: str = "date"                  # Column containing time index
    series_id_col: str = "ts_id"            # Column identifying each series

    # ── Architecture ──
    encoder_length: int = 24                # Lookback window (T timesteps)
    decoder_length: int = 6                 # Forecast horizon (H timesteps)
    alpha_lag: int = 3                      # Covariate-to-target delay (business rule)
    margin: float = 2.0                     # Minimum margin above constraint floor
    constraint_temp: float = 5.0            # LogSumExp sharpness (higher = harder max)

    # ── Encoder LSTM ──
    encoder_hidden_1: int = 128             # First LSTM layer hidden size
    encoder_hidden_2: int = 64              # Second LSTM layer hidden size (= decoder)

    # ── Decoder LSTM ──
    decoder_hidden: int = 64                # Decoder LSTM hidden size

    # ── Training ──
    batch_size: int = 32                    # Works on most GPUs
    learning_rate: float = 1e-3             # Good starting point for Adam
    max_epochs: int = 100                   # Early stopping handles actual count
    patience: int = 10                      # Epochs without val improvement → stop
    validation_ratio: float = 0.2           # Last 20% of time for validation
    test_ratio: float = 0.1                 # Last 10% of time for test (after val)

    # ── Preprocessing ──
    scaling_method: str = "per_series"      # "per_series" | "global" | "none"
    null_strategy: str = "forward_fill"     # "forward_fill" | "interpolate" | "drop_series"
    min_series_length: int = 12             # Minimum timesteps to include a series

    # ── Reproducibility ──
    seed: int = 42                          # Reproducible by default

    def __post_init__(self):
        """Validate configuration on creation — fail fast with helpful messages."""
        if self.encoder_length < 2:
            raise ValueError(
                f"encoder_length must be ≥ 2, got {self.encoder_length}. "
                f"Set encoder_length to the number of historical timesteps the model sees "
                f"(e.g., encoder_length=24 for 24 months of history)."
            )
        if self.decoder_length < 1:
            raise ValueError(
                f"decoder_length must be ≥ 1, got {self.decoder_length}. "
                f"Set decoder_length to the number of future timesteps to forecast."
            )
        if self.alpha_lag < 0:
            raise ValueError(
                f"alpha_lag must be ≥ 0, got {self.alpha_lag}. "
                f"alpha_lag represents the delay between covariate effect and target. "
                f"Set to 0 if covariates affect target immediately."
            )
        if not 0 < self.validation_ratio < 1:
            raise ValueError(
                f"validation_ratio must be between 0 and 1 (exclusive), got {self.validation_ratio}."
            )
        if self.scaling_method not in ("per_series", "global", "none"):
            raise ValueError(
                f"scaling_method must be 'per_series', 'global', or 'none', "
                f"got '{self.scaling_method}'. "
                f"'per_series' is recommended for heterogeneous series."
            )
        if self.null_strategy not in ("forward_fill", "interpolate", "drop_series"):
            raise ValueError(
                f"null_strategy must be 'forward_fill', 'interpolate', or 'drop_series', "
                f"got '{self.null_strategy}'."
            )

    @property
    def n_encoder_features(self) -> int:
        """Number of features per encoder timestep: target + covariates."""
        return 1 + len(self.covariate_cols)

    @property
    def n_decoder_features(self) -> int:
        """Number of features per decoder timestep: known-future covariates only."""
        return len(self.covariate_cols)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Following facilitating-experiments skill:
    every experiment starts with a hypothesis and ends with a decision.
    """
    experiment_id: str
    hypothesis: str
    control_description: str = ""
    treatment_description: str = ""
    success_metric: str = "MAE"
    success_threshold: Optional[float] = None
    notes: str = ""
