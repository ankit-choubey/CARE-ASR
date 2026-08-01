"""Custom Exception Hierarchy for CARE-ASR Module.

Why It Exists:
    Defines domain-specific exception types to isolate errors in schema validation,
    BioBERT inference, span alignment, threshold evaluation, and candidate scoring.

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Catches explicit exceptions in automated test suites.
    - Ankit (Integration Lead): Uses exception types to trigger graceful pipeline fallbacks.

Imported By:
    - `care_asr.config.settings`
    - `care_asr.ner.extractor`
    - `care_asr.ner.span_aligner`
    - `care_asr.thresholds.threshold_engine`
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Add error code metadata attributes to exception classes.
"""

import logging

logger = logging.getLogger(__name__)


class CAREASRError(Exception):
    """Base exception for all CARE-ASR module errors.

    Args:
        message (str): Human-readable error description.
        trace_id (str | None): Unique transaction trace ID for log correlation.
    """

    def __init__(self, message: str, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id
        logger.error(f"[{trace_id or 'NO_TRACE'}] CAREASRError: {message}")


class SchemaValidationError(CAREASRError):
    """Raised when incoming JSON payload fails Pydantic schema validation."""

    pass


class ModelInferenceError(CAREASRError):
    """Raised when BioBERT model loading or PyTorch token classification fails."""

    pass


class InvalidCheckpointError(ModelInferenceError):
    """Raised when a model checkpoint id2label metadata contains unsupported entity taxonomy labels."""

    pass


class AlignmentError(CAREASRError):
    """Raised when subtoken character offsets cannot be mapped to ASR word boundaries."""

    pass


class ThresholdConfigurationError(CAREASRError):
    """Raised when category threshold settings are missing, invalid, or out-of-range."""

    pass


class CandidateEvaluationError(CAREASRError):
    """Raised when FAISS candidate scoring or composite utility computation encounters an invalid state."""

    pass
