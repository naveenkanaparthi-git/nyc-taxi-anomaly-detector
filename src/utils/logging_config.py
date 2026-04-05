"""Structured JSON logging configuration."""

import logging
import sys

from pythonjsonlogger import jsonlogger


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure and return a JSON-formatted logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
