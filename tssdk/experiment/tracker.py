"""
tssdk.experiment.tracker — Experiment lifecycle management.

Every experiment starts with a hypothesis and ends with a decision.
No undirected exploration.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

from tssdk.config import ExperimentConfig
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """Complete record of a single experiment."""
    config: Dict[str, Any]
    hypothesis: str
    started_at: str = ""
    completed_at: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    model_description: Dict[str, Any] = field(default_factory=dict)
    decision: str = ""  # "accept" | "reject" | "iterate"
    notes: str = ""


class ExperimentTracker:
    """Track experiments with hypothesis-driven discipline.

    Usage:
        tracker = ExperimentTracker("experiments/")
        tracker.start(
            experiment_id="EXP-001",
            hypothesis="Constrained LSTM beats naive baseline by >10% MAE",
            config=config.__dict__,
        )
        # ... run training and evaluation ...
        tracker.log_metrics({"mae": 0.15, "rmse": 0.22})
        tracker.conclude(decision="accept", notes="MAE improved 25% vs naive")
    """

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[ExperimentRecord] = None
        self._experiment_id: str = ""

    def start(
        self,
        experiment_id: str,
        hypothesis: str,
        config: Dict[str, Any],
        model_description: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start a new experiment.

        Args:
            experiment_id: Unique identifier (e.g., "EXP-001").
            hypothesis: Falsifiable hypothesis statement.
            config: Full configuration dict for reproducibility.
            model_description: Architecture metadata from model.describe().
        """
        if not hypothesis:
            raise ValueError(
                "Every experiment must start with a hypothesis. "
                "Example: 'Adding weather covariates will reduce MAE by ≥5%'"
            )

        self._experiment_id = experiment_id
        self._current = ExperimentRecord(
            config=config,
            hypothesis=hypothesis,
            started_at=datetime.now().isoformat(),
            model_description=model_description or {},
        )

        logger.info(
            f"Experiment started | id={experiment_id} | "
            f"hypothesis={hypothesis[:80]}..."
        )

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics for the current experiment."""
        if self._current is None:
            raise RuntimeError(
                "No active experiment. Call start() before logging metrics."
            )
        self._current.metrics.update(metrics)
        logger.info(
            f"Metrics logged [{self._experiment_id}] | " +
            " | ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        )

    def conclude(self, decision: str, notes: str = "") -> str:
        """Conclude the experiment with a decision.

        Args:
            decision: One of "accept", "reject", "iterate".
            notes: Explanation of the decision.

        Returns:
            Path to saved experiment record.
        """
        if self._current is None:
            raise RuntimeError("No active experiment to conclude.")

        if decision not in ("accept", "reject", "iterate"):
            raise ValueError(
                f"Decision must be 'accept', 'reject', or 'iterate', got '{decision}'."
            )

        self._current.decision = decision
        self._current.notes = notes
        self._current.completed_at = datetime.now().isoformat()

        # Save to file
        out_path = self.output_dir / f"{self._experiment_id}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(self._current), f, indent=2, default=str)

        logger.info(
            f"Experiment concluded | id={self._experiment_id} | "
            f"decision={decision} | saved={out_path}"
        )

        self._current = None
        return str(out_path)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all recorded experiments."""
        records = []
        for path in sorted(self.output_dir.glob("*.json")):
            with open(path) as f:
                records.append(json.load(f))
        return records
