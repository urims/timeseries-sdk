"""
tssdk.utils.sdk_logging — Structured logging for ML pipelines.

Format: [timestamp] LEVEL  context | key=value | key=value
"""

import logging
import sys
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger for SDK modules.

    Args:
        name: Logger name (typically __name__).
        level: Logging level. Default: INFO.

    Returns:
        Configured logger with structured format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
