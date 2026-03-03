"""
tssdk.utils.seeds — Reproducibility seed setter.

Sets seeds for Python, NumPy, and TensorFlow to ensure
deterministic results across runs.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for all random number generators.

    Args:
        seed: Integer seed value. Default: 42.

    Sets: Python random, NumPy, TensorFlow, and PYTHONHASHSEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TF not installed — skip silently
