"""Structured Logging Infrastructure for CARE-ASR Module.

Why It Exists:
    Configures standard library logging with JSON and console formatters to ensure
    traceability across pipeline execution.

Teammate Dependencies:
    - All teammates consume structured log outputs for debugging and integration monitoring.

Imported By:
    - `care_asr.__init__`
    - `care_asr.config.settings`

TODOs:
    - Integrate structlog for advanced contextual log enrichment.
"""

import logging
import logging.config
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def setup_logger(config_path: Path | str = "logging.yaml", default_level: int = logging.INFO) -> logging.Logger:
    """Configures structured logging using a YAML configuration file.

    Args:
        config_path (Path | str): Path to the logging YAML configuration file.
        default_level (int): Fallback logging level if configuration file loading fails.

    Returns:
        logging.Logger: Configured logger instance for the care_asr module.

    Raises:
        FileNotFoundError: If the specified config file path does not exist.

    Examples:
        >>> log = setup_logger("logging.yaml")
        >>> log.info("Logger initialized successfully.")
    """
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
        logger.info(f"Logging configured successfully from {path}")
    else:
        logging.basicConfig(level=default_level)
        logger.warning(f"Logging config file not found at {path}. Using default basic configuration.")
    return logging.getLogger("care_asr")
