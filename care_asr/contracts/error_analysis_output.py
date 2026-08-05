"""Error Analysis & Audit Output Contract Schema.

Why It Exists:
    Enforces runtime validation for offline diagnostic audit reports emitted to Mahi
    for system-level evaluation, category breakdown, and quality gate assertion.

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Consumes reports conforming to this schema to verify
      benchmark precision, recall, and F1 gains across MED, COND, ANA, TTP, and PHI.    Imported By:
        - `care_asr.evaluation.metrics_calculator`

    Serialization:
        - `audit_report_to_dict()` converts a report into a plain dict.
        - `save_audit_report()` writes a report to a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CategoryMetric(BaseModel):
    """Performance evaluation metrics for a single CARE-ASR entity category.

    Attributes:
        precision (float): Entity-level precision score in [0.0, 1.0].
        recall (float): Entity-level recall score in [0.0, 1.0].
        f1_score (float): Harmonic mean F1 score in [0.0, 1.0].
        support (int): Total ground-truth entity instances in benchmark dataset.
    """

    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    support: int = Field(..., ge=0, description="Ground truth sample count")


class CategoryBreakdown(BaseModel):
    """Metrics breakdown across the 5 official CARE-ASR entity categories.

    Attributes:
        MED (CategoryMetric): Metrics for Medication entities.
        COND (CategoryMetric): Metrics for Medical Condition entities.
        ANA (CategoryMetric): Metrics for Anatomical Site entities.
        TTP (CategoryMetric): Metrics for Test/Treatment/Procedure entities.
        PHI (CategoryMetric): Metrics for Protected Health Information entities.
    """

    MED: CategoryMetric = Field(..., description="Metrics for Medication entities")
    COND: CategoryMetric = Field(..., description="Metrics for Medical Condition entities")
    ANA: CategoryMetric = Field(..., description="Metrics for Anatomical Site entities")
    TTP: CategoryMetric = Field(..., description="Metrics for Test/Treatment/Procedure entities")
    PHI: CategoryMetric = Field(..., description="Metrics for PHI entities")


class ErrorTaxonomy(BaseModel):
    """Classification counts for recovery failure root causes.

    Attributes:
        phonetic_distortion_count (int): Acoustic distortion failures.
        oov_error_count (int): Out-of-vocabulary entity failures.
        ner_boundary_mismatch_count (int): BioBERT span boundary misalignment failures.
        retrieval_miss_count (int): Candidate missing from FAISS index.
    """

    phonetic_distortion_count: int = Field(..., ge=0, description="Phonetic distortion count")
    oov_error_count: int = Field(..., ge=0, description="OOV error count")
    ner_boundary_mismatch_count: int = Field(..., ge=0, description="NER boundary mismatch count")
    retrieval_miss_count: int = Field(..., ge=0, description="Retrieval candidate miss count")


class OverallMetrics(BaseModel):
    """Aggregate dataset-level recovery metrics.

    Attributes:
        raw_asr_f1 (float): F1 score of raw uncorrected Whisper ASR text.
        rectified_f1 (float): F1 score after candidate retrieval recovery.
        precision_gain (float): Absolute precision improvement (rectified - raw).
        recall_gain (float): Absolute recall improvement (rectified - raw).
    """

    raw_asr_f1: float = Field(..., ge=0.0, le=1.0, description="Raw ASR F1 score")
    rectified_f1: float = Field(..., ge=0.0, le=1.0, description="Rectified F1 score")
    precision_gain: float = Field(..., description="Precision gain delta")
    recall_gain: float = Field(..., description="Recall gain delta")


class FailedInstance(BaseModel):
    """Represents a single unrecovered entity instance for diagnostic review.

    Attributes:
        transcript_id (str): Unique transaction identifier.
        ground_truth (str): Ground-truth target entity text.
        predicted (str): Predicted or raw ASR entity text.
        failure_type (str): Categorized error taxonomy root cause.
    """

    transcript_id: str = Field(..., description="Transaction ID")
    ground_truth: str = Field(..., description="Ground truth text")
    predicted: str = Field(..., description="Predicted entity text")
    failure_type: str = Field(..., description="Categorized failure root cause")


class ErrorAnalysisAuditOutput(BaseModel):
    """Final audit report contract emitted to Mahi for QA evaluation.

    Attributes:
        batch_id (str): Unique benchmark evaluation run identifier.
        total_samples (int): Total number of evaluation transcript samples.
        overall_metrics (OverallMetrics): Aggregate performance gains.
        category_breakdown (CategoryBreakdown): Per-category metrics (MED, COND, ANA, TTP, PHI).
        error_taxonomy (ErrorTaxonomy): Root cause failure breakdown.
        failed_instances (list[FailedInstance]): Detailed list of unrecovered entity cases.
    """

    batch_id: str = Field(..., description="Evaluation run ID")
    total_samples: int = Field(..., ge=0, description="Total evaluation samples")
    overall_metrics: OverallMetrics = Field(..., description="Aggregate evaluation metrics")
    category_breakdown: CategoryBreakdown = Field(..., description="Per-category metric breakdown")
    error_taxonomy: ErrorTaxonomy = Field(..., description="Failure root cause taxonomy counts")
    failed_instances: list[FailedInstance] = Field(default_factory=list, description="Failed entity instances")


class NEREntity(BaseModel):
    """Represents a clinical entity extracted by BioBERT NER.

    Attributes:
        word (str): Entity text string.
        category (str): Entity category (MED, COND, ANA, TTP, PHI).
        start (int): Start token/word index.
        end (int): End token/word index.
        score (float): Confidence score in [0.0, 1.0].
    """

    word: str = Field(..., description="Entity text string")
    category: str = Field(default="MED", description="Entity category label")
    start: int = Field(default=0, ge=0, description="Start token index")
    end: int = Field(default=0, ge=0, description="End token index")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


def audit_report_to_dict(report: ErrorAnalysisAuditOutput) -> dict[str, Any]:
    """Converts an audit report into a plain JSON-serializable dictionary.

    Uses the Pydantic ``model_dump()`` representation so the contract remains
    the single source of truth for the serialized shape.

    Args:
        report (ErrorAnalysisAuditOutput): The audit report to serialize.

    Returns:
        dict[str, Any]: JSON-serializable report payload.
    """
    return report.model_dump()


def save_audit_report(
    report: ErrorAnalysisAuditOutput,
    output_path: Path | str,
) -> None:
    """Writes an audit report to a JSON file following project conventions.

    Parent directories are created automatically. The payload is written with
    ``indent=2`` and ``ensure_ascii=False`` to match existing evaluation output.

    Args:
        report (ErrorAnalysisAuditOutput): The audit report to serialize.
        output_path (Path | str): Destination file path.

    Raises:
        RuntimeError: If the file cannot be written or does not exist afterwards.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(audit_report_to_dict(report), f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write audit report to '{path}': {exc}") from exc

    if not path.is_file():
        raise RuntimeError(f"Audit report file was not created at '{path}'.")
