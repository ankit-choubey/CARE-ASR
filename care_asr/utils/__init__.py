"""Utility Package for CARE-ASR Module.

Why It Exists:
    Provides centralized logging infrastructure and exception handling hierarchies
    across all subpackages.

Teammate Dependencies:
    - Internal module developers and Mahi (Testing Lead).

Imported By:
    - `care_asr.config`
    - `care_asr.ner`
    - `care_asr.validation`
    - `care_asr.thresholds`
    - `care_asr.evaluation`

TODOs:
    - Add custom exception handlers for Pydantic schema validation failures.
"""

from care_asr.utils.exceptions import (
    AlignmentException,
    CandidateEvaluationError,
    CAREASRBaseException,
    ModelInferenceError,
    SchemaValidationError,
    ThresholdConfigurationError,
)
from care_asr.utils.logger import setup_logger

__all__ = [
    "CAREASRBaseException",
    "SchemaValidationError",
    "ModelInferenceError",
    "AlignmentException",
    "ThresholdConfigurationError",
    "CandidateEvaluationError",
    "setup_logger",
]
