"""Configuration Subpackage for CARE-ASR Module.

Why It Exists:
    Centralizes settings management, environment variable loading, and YAML threshold
    parsing using Pydantic Settings.

Teammate Dependencies:
    - Internal modules load settings for model paths, device allocation, and operational thresholds.

Imported By:
    - `care_asr.ner.extractor`
    - `care_asr.thresholds.threshold_engine`
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Add dynamic reloading capability for runtime threshold updates.
"""

from care_asr.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
