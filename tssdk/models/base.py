"""
tssdk.models.base — Abstract base class for all timeseries DL architectures.

Every architecture in the SDK implements this interface, ensuring
consistent usage across experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from tssdk.config import TimeseriesConfig


class BaseTimeseriesModel(ABC):
    """Abstract base for timeseries forecasting models.

    All models must implement:
    - build() → construct the model graph
    - get_model() → return the compiled model
    - describe() → return architecture metadata for experiment logging
    """

    def __init__(self, config: TimeseriesConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def build(self) -> None:
        """Construct and compile the model. Sets self.model."""
        ...

    @abstractmethod
    def get_model(self):
        """Return the compiled model object."""
        ...

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Return architecture metadata for experiment tracking.

        Returns:
            Dict with keys like 'name', 'params', 'encoder_layers', etc.
        """
        ...
