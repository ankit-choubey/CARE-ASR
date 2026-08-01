"""Pydantic Settings Configuration Manager for CARE-ASR Module.

Why It Exists:
    Loads application settings from environment variables (.env) and config.yaml,
    providing type-safe configuration objects across all module components.

Teammate Dependencies:
    - Internal developers use `get_settings()` for BioBERT model path and threshold lookups.

Imported By:
    - `care_asr.ner.extractor`
    - `care_asr.thresholds.threshold_engine`

TODOs:
    - Add validation for CUDA device availability when TORCH_DEVICE=cuda.
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore
import contextlib

import yaml

from care_asr.utils.exceptions import ThresholdConfigurationError

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings and operational threshold configuration.

    Attributes:
        environment (str): System environment (development, staging, production).
        log_level (str): Logging verbosity level.
        biobert_model_name_or_path (str): HuggingFace hub model name or local path.
        torch_device (str): Compute device for PyTorch execution (cuda or cpu).
        batch_size (int): BioBERT NER token classification inference batch size.
        config_file_path (Path): Path to the YAML configuration file.
    """

    environment: str = Field(default="development", description="Execution environment")
    log_level: str = Field(default="INFO", description="Logging level")
    biobert_model_name_or_path: str = Field(
        default="d4data/biomedical-ner-all", description="BioBERT model identifier"
    )
    torch_device: str = Field(default="cuda", description="PyTorch execution device")
    batch_size: int = Field(default=16, ge=1, description="NER inference batch size")
    config_file_path: Path = Field(
        default=Path("config.yaml"), description="YAML configuration path"
    )

    with contextlib.suppress(Exception):
        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", extra="ignore"
        )

    def load_yaml_config(self) -> dict:
        """Parses and returns the YAML threshold and model configuration dictionary.

        Returns:
            dict: Parsed YAML configuration dictionary containing category threshold rules.

        Raises:
            ThresholdConfigurationError: If the YAML file does not exist or fails parsing.

        Examples:
            >>> settings = get_settings()
            >>> config_dict = settings.load_yaml_config()
            >>> "categories" in config_dict
            True
        """
        if not self.config_file_path.exists():
            logger.error(f"Configuration file not found at {self.config_file_path}")
            raise ThresholdConfigurationError(
                f"Configuration file missing: {self.config_file_path}"
            )
        try:
            with open(self.config_file_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise ThresholdConfigurationError(f"YAML configuration parsing error: {e}") from e


@lru_cache
def get_settings() -> Settings:
    """Returns a cached singleton instance of the Settings model.

    Returns:
        Settings: Initialized settings object.

    Examples:
        >>> settings = get_settings()
        >>> settings.environment
        'development'
    """
    logger.info("Initializing application settings...")
    return Settings()
