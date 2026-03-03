"""
tssdk — Time Series Deep Learning Experimentation SDK.

Quick start:
    from tssdk import TimeseriesConfig, Pipeline
    config = TimeseriesConfig(target_col="actual_cost_paid", covariate_cols=["contract_value"])
    pipe = Pipeline(config)
    splits = pipe.prepare("data.csv")
    # → splits["train"], splits["val"], splits["test"] ready for model.fit()

Full pipeline:
    from tssdk import TimeseriesConfig, Pipeline
    from tssdk.models import ConstrainedLSTM
    from tssdk.training import TrainingRunner
    from tssdk.experiment import ExperimentTracker
"""

from tssdk.config import TimeseriesConfig, ExperimentConfig
from tssdk.data import load, Preprocessor, Windower, WindowedDataset, validate
from tssdk.models import ConstrainedLSTM, BaseTimeseriesModel
from tssdk.training import TrainingRunner, compute_all_metrics
from tssdk.experiment import ExperimentTracker
from tssdk.utils import set_all_seeds, get_logger

__version__ = "0.1.0"


class Pipeline:
    """Convenience wrapper: load → preprocess → window → validate in one call.

    Usage:
        config = TimeseriesConfig(target_col="actual_cost_paid", covariate_cols=["contract_value"])
        pipe = Pipeline(config)
        splits = pipe.prepare("timeseries_sample.csv")
    """

    def __init__(self, config: TimeseriesConfig):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.windower = Windower(config)

    def prepare(self, path: str, verbose: bool = True):
        """Full data preparation pipeline.

        Args:
            path: Path to CSV or Parquet file.
            verbose: Print validation summary.

        Returns:
            Dict with "train", "val", "test" WindowedDatasets.
        """
        # 1. Load
        df = load(path, self.config)

        # 2. Preprocess
        df_clean = self.preprocessor.fit_transform(df)

        # 3. Window + split
        splits = self.windower.create_splits(df_clean)

        # 4. Validate each split
        all_passed = True
        for name, dataset in splits.items():
            result = validate(dataset, self.config, split_name=name)
            if verbose:
                print(result.summary())
            if not result.passed:
                all_passed = False

        if not all_passed:
            print("\n⚠  Some validation checks failed — review above before training.")

        return splits


__all__ = [
    "TimeseriesConfig",
    "ExperimentConfig",
    "Pipeline",
    "load",
    "Preprocessor",
    "Windower",
    "WindowedDataset",
    "validate",
    "ConstrainedLSTM",
    "BaseTimeseriesModel",
    "TrainingRunner",
    "compute_all_metrics",
    "ExperimentTracker",
    "set_all_seeds",
    "get_logger",
]
