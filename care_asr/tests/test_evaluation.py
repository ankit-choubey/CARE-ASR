"""Unit Tests for ErrorAnalysisEngine & FailureTaxonomyClassifier.

Covers:
- Entity-level precision, recall, and F1 per category (MED, COND, ANA, TTP, PHI)
- Perfect, missing, and spurious predictions
- Empty prediction / ground-truth datasets
- Failed instance generation and error taxonomy aggregation
- Raw vs rectified overall metric calculation
- Audit report JSON serialization helpers
"""

import json
from pathlib import Path
from typing import Any

import pytest

from care_asr.contracts.error_analysis_output import (
    ErrorAnalysisAuditOutput,
    audit_report_to_dict,
    save_audit_report,
)
from care_asr.evaluation.metrics_calculator import ErrorAnalysisEngine
from care_asr.evaluation.taxonomy_classifier import FailureTaxonomyClassifier


def _gt_entity(text: str, category: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
    """Builds a ground-truth entity dict with deterministic character offsets."""
    return {
        "entity_text": text,
        "category": category,
        "start_char": start,
        "end_char": start + len(text) if end is None else end,
    }


def _pred_entity(
    text: str,
    category: str,
    rectified: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
    start: int = 0,
    end: int | None = None,
) -> dict[str, Any]:
    """Builds a predicted entity dict, optionally with rectified text and candidates."""
    entity: dict[str, Any] = {
        "entity_text": text,
        "category": category,
        "start_char": start,
        "end_char": start + len(text) if end is None else end,
    }
    if rectified is not None:
        entity["rectified_text"] = rectified
    if candidates is not None:
        entity["retrieval_candidates"] = candidates
    return entity


def _transcript(transcript_id: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds a transcript item carrying a list of entity dicts."""
    return {"transcript_id": transcript_id, "entities": entities}


def test_error_analysis_engine_initialization() -> None:
    """Tests ErrorAnalysisEngine instantiation."""
    engine = ErrorAnalysisEngine()
    assert engine is not None


def test_failure_taxonomy_classifier_interface() -> None:
    """Tests FailureTaxonomyClassifier static methods."""
    assert hasattr(FailureTaxonomyClassifier, "classify_failure")
    assert hasattr(FailureTaxonomyClassifier, "aggregate_taxonomy")


def test_perfect_prediction() -> None:
    """A fully recovered dataset yields perfect metrics and no failed instances."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("metformin", "MED")])]
    pred = [_transcript("t1", [_pred_entity("metformin", "MED", rectified="metformin")])]

    report = engine.generate_audit_report("batch-1", gt, pred)

    assert report.batch_id == "batch-1"
    assert report.total_samples == 1
    assert report.failed_instances == []

    med = report.category_breakdown.MED
    assert med.support == 1
    assert med.precision == 1.0
    assert med.recall == 1.0
    assert med.f1_score == 1.0

    for category in ("COND", "ANA", "TTP", "PHI"):
        metric = getattr(report.category_breakdown, category)
        assert metric.support == 0
        assert metric.precision == 0.0

    assert report.overall_metrics.raw_asr_f1 == 1.0
    assert report.overall_metrics.rectified_f1 == 1.0
    assert report.error_taxonomy.retrieval_miss_count == 0


def test_missing_prediction() -> None:
    """A ground-truth entity with no prediction is a false negative with a failed instance."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("metformin", "MED")])]
    pred: list[dict[str, Any]] = []

    report = engine.generate_audit_report("batch-1", gt, pred)

    med = report.category_breakdown.MED
    assert med.support == 1
    assert med.precision == 0.0
    assert med.recall == 0.0
    assert med.f1_score == 0.0

    assert len(report.failed_instances) == 1
    failure = report.failed_instances[0]
    assert failure.transcript_id == "t1"
    assert failure.ground_truth == "metformin"
    assert failure.predicted == ""
    assert failure.failure_type == "retrieval_miss"
    assert report.error_taxonomy.retrieval_miss_count == 1


def test_false_positive() -> None:
    """A spurious prediction yields zero precision and a failed instance for the miss."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("metformin", "MED", start=0)])]
    pred = [_transcript("t1", [_pred_entity("aspirin", "MED", start=10)])]

    report = engine.generate_audit_report("batch-1", gt, pred)

    med = report.category_breakdown.MED
    assert med.support == 1
    assert med.precision == 0.0
    assert med.recall == 0.0
    assert med.f1_score == 0.0

    assert len(report.failed_instances) == 1
    assert report.failed_instances[0].ground_truth == "metformin"
    assert report.failed_instances[0].failure_type == "retrieval_miss"


def test_mixed_categories() -> None:
    """Metrics are computed independently per category."""
    engine = ErrorAnalysisEngine()
    gt = [
        _transcript(
            "t1",
            [_gt_entity("metformin", "MED", start=0), _gt_entity("hypertension", "COND", start=10)],
        ),
        _transcript("t2", [_gt_entity("left ventricle", "ANA", start=0)]),
    ]
    pred = [
        _transcript(
            "t1",
            [
                _pred_entity("metformin", "MED", rectified="metformin", start=0),
                _pred_entity("hypertension", "COND", rectified="hypertension", start=10),
            ],
        ),
        _transcript("t2", []),
    ]

    report = engine.generate_audit_report("batch-1", gt, pred)

    assert report.total_samples == 2
    assert report.category_breakdown.MED.f1_score == 1.0
    assert report.category_breakdown.COND.f1_score == 1.0
    assert report.category_breakdown.ANA.support == 1
    assert report.category_breakdown.ANA.f1_score == 0.0

    assert len(report.failed_instances) == 1
    assert report.failed_instances[0].ground_truth == "left ventricle"


def test_empty_predictions() -> None:
    """Every ground-truth entity is a retrieval miss when nothing is predicted."""
    engine = ErrorAnalysisEngine()
    gt = [
        _transcript("t1", [_gt_entity("metformin", "MED")]),
        _transcript("t2", [_gt_entity("aspirin", "MED")]),
    ]
    pred: list[dict[str, Any]] = []

    report = engine.generate_audit_report("batch-1", gt, pred)

    assert report.total_samples == 2
    assert report.category_breakdown.MED.support == 2
    assert report.category_breakdown.MED.recall == 0.0
    assert len(report.failed_instances) == 2
    assert report.error_taxonomy.retrieval_miss_count == 2
    assert report.overall_metrics.rectified_f1 == 0.0


def test_empty_ground_truth() -> None:
    """With no ground truth, all metrics are zero and no instances fail."""
    engine = ErrorAnalysisEngine()
    gt: list[dict[str, Any]] = []
    pred = [_transcript("t1", [_pred_entity("aspirin", "MED", rectified="aspirin")])]

    report = engine.generate_audit_report("batch-1", gt, pred)

    assert report.total_samples == 0
    assert report.category_breakdown.MED.support == 0
    assert report.category_breakdown.MED.precision == 0.0
    assert report.failed_instances == []
    assert report.overall_metrics.raw_asr_f1 == 0.0
    assert report.overall_metrics.rectified_f1 == 0.0


def test_failed_instance_generation() -> None:
    """Missed entities inherit the confusion context for taxonomy classification."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("metformin", "MED", start=0)])]
    pred = [
        _transcript(
            "t1",
            [
                _pred_entity(
                    "metformun",
                    "MED",
                    rectified="metformun",
                    candidates=[{"concept_id": "C1", "canonical_name": "Metformin"}],
                    start=0,
                )
            ],
        )
    ]

    report = engine.generate_audit_report("batch-1", gt, pred)

    assert report.category_breakdown.MED.f1_score == 0.0

    assert len(report.failed_instances) == 1
    failure = report.failed_instances[0]
    assert failure.ground_truth == "metformin"
    assert failure.predicted == "metformun"
    assert failure.failure_type == "phonetic_distortion"
    assert report.error_taxonomy.phonetic_distortion_count == 1


def test_taxonomy_aggregation() -> None:
    """Multiple failure root causes are aggregated into the ErrorTaxonomy counts."""
    engine = ErrorAnalysisEngine()
    gt = [
        _transcript("t1", [_gt_entity("metformin", "MED", start=0)]),
        _transcript("t2", [_gt_entity("hypertension", "COND", start=0)]),
        _transcript("t3", [_gt_entity("levothyroxine", "MED", start=0)]),
    ]
    pred = [
        _transcript(
            "t1",
            [
                _pred_entity(
                    "metformun",
                    "MED",
                    rectified="metformun",
                    candidates=[{"concept_id": "C1"}],
                    start=0,
                )
            ],
        ),
        _transcript(
            "t2",
            [
                _pred_entity(
                    "severe hypertension",
                    "COND",
                    rectified="severe hypertension",
                    candidates=[{"concept_id": "C2"}],
                    start=0,
                    end=12,
                )
            ],
        ),
        _transcript("t3", []),
    ]

    report = engine.generate_audit_report("batch-1", gt, pred)

    taxonomy = report.error_taxonomy
    assert taxonomy.phonetic_distortion_count == 1
    assert taxonomy.ner_boundary_mismatch_count == 1
    assert taxonomy.retrieval_miss_count == 1
    assert taxonomy.oov_error_count == 0
    assert len(report.failed_instances) == 3


def test_overall_metric_calculation() -> None:
    """Raw and rectified overall metrics reflect recovery gains."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("Acetaminophen", "MED", start=0)])]
    pred = [
        _transcript(
            "t1",
            [
                _pred_entity(
                    "Tylenol",
                    "MED",
                    rectified="Acetaminophen",
                    candidates=[{"concept_id": "C1"}],
                    start=0,
                )
            ],
        )
    ]

    report = engine.generate_audit_report("batch-1", gt, pred)

    overall = report.overall_metrics
    assert overall.raw_asr_f1 == 0.0  # "Tylenol" does not match ground truth
    assert overall.rectified_f1 == 1.0  # "Acetaminophen" matches after recovery
    assert overall.precision_gain == 1.0
    assert overall.recall_gain == 1.0
    # Recovery succeeded, so no failed instances remain
    assert report.failed_instances == []
    assert report.error_taxonomy.retrieval_miss_count == 0


def _sample_report() -> ErrorAnalysisAuditOutput:
    """Builds a representative audit report through the engine."""
    engine = ErrorAnalysisEngine()
    gt = [_transcript("t1", [_gt_entity("metformin", "MED")])]
    pred = [_transcript("t1", [_pred_entity("metformin", "MED", rectified="metformin")])]
    return engine.generate_audit_report("batch-1", gt, pred)


def test_audit_report_to_dict_serialization() -> None:
    """audit_report_to_dict mirrors model_dump and round-trips through JSON."""
    report = _sample_report()

    payload = audit_report_to_dict(report)

    assert isinstance(payload, dict)
    assert payload == report.model_dump()
    assert payload["batch_id"] == "batch-1"
    assert payload["total_samples"] == 1
    assert payload["category_breakdown"]["MED"]["f1_score"] == 1.0
    assert payload["overall_metrics"]["rectified_f1"] == 1.0
    assert json.loads(json.dumps(payload)) == payload


def test_save_audit_report_writes_json(tmp_path: Path) -> None:
    """save_audit_report writes a JSON file that reads back identically."""
    report = _sample_report()
    output_path = tmp_path / "audit" / "report.json"

    save_audit_report(report, output_path)

    assert output_path.is_file()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == audit_report_to_dict(report)
    restored = ErrorAnalysisAuditOutput.model_validate(loaded)
    assert restored == report


def test_save_audit_report_accepts_str_path(tmp_path: Path) -> None:
    """A plain string output path is accepted."""
    report = _sample_report()
    output_path = str(tmp_path / "str_path" / "audit.json")

    save_audit_report(report, output_path)

    assert Path(output_path).is_file()


def test_save_audit_report_creates_parent_directories(tmp_path: Path) -> None:
    """Nested parent directories are created automatically."""
    report = _sample_report()
    output_path = tmp_path / "a" / "b" / "c" / "audit.json"

    save_audit_report(report, output_path)

    assert output_path.is_file()
    assert output_path.parent.is_dir()


def test_save_audit_report_invalid_path_raises(tmp_path: Path) -> None:
    """Write failures raise a descriptive RuntimeError."""
    report = _sample_report()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to write audit report"):
        save_audit_report(report, blocker / "audit.json")
