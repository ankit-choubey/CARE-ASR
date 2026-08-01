"""Utils Package.

Provides logging formatters, custom exception hierarchies, and shared helpers.
"""

from care_asr.utils.exceptions import (
    AlignmentError,
    CandidateEvaluationError,
    CAREASRError,
    ModelInferenceError,
    SchemaValidationError,
    ThresholdConfigurationError,
)
from care_asr.utils.logger import setup_logger

__all__ = [
    "CAREASRError",
    "SchemaValidationError",
    "ModelInferenceError",
    "AlignmentError",
    "ThresholdConfigurationError",
    "CandidateEvaluationError",
    "setup_logger",
]
