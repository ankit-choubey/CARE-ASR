"""Offline Error Analysis Engine & Audit Metrics Calculator.

Why It Exists:
    Calculates entity-level Precision, Recall, F1 gains, category breakdown (MED, COND, ANA, TTP, PHI),
    and generates ErrorAnalysisAuditOutput payloads for Mahi's QA suite.

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Consumes audit reports emitted by ErrorAnalysisEngine.
    - FailureTaxonomyClassifier classifies every unrecovered entity for the audit's error taxonomy.

Imported By:
    - Automated test runners and benchmark evaluation entrypoints.
"""

import logging
from collections import defaultdict
from typing import Any

from care_asr.contracts.error_analysis_output import (
    CategoryBreakdown,
    CategoryMetric,
    ErrorAnalysisAuditOutput,
    FailedInstance,
    OverallMetrics,
)
from care_asr.evaluation.taxonomy_classifier import FailureTaxonomyClassifier

logger = logging.getLogger(__name__)

_CATEGORIES = ("MED", "COND", "ANA", "TTP", "PHI")


def _normalize(text: str) -> str:
    """Lowercases, trims, and collapses whitespace for deterministic entity matching."""
    return " ".join(text.lower().strip().split())


def _precision(true_positives: int, false_positives: int) -> float:
    """Computes precision as tp / (tp + fp); 0.0 when there are no predictions."""
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator > 0 else 0.0


def _recall(true_positives: int, false_negatives: int) -> float:
    """Computes recall as tp / (tp + fn); 0.0 when there is no ground truth."""
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator > 0 else 0.0


def _f1_score(precision: float, recall: float) -> float:
    """Computes the harmonic mean of precision and recall; 0.0 when both are zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class ErrorAnalysisEngine:
    """Computes offline precision/recall/F1 metrics and error taxonomy audit reports."""

    def __init__(self) -> None:
        self.taxonomy_classifier = FailureTaxonomyClassifier()
        logger.info("ErrorAnalysisEngine initialized.")

    def generate_audit_report(
        self,
        batch_id: str,
        ground_truth_dataset: list[dict[str, Any]],
        predicted_outputs: list[dict[str, Any]],
    ) -> ErrorAnalysisAuditOutput:
        """Calculates benchmark metrics and produces an ErrorAnalysisAuditOutput payload.

        Ground-truth and predicted inputs are lists of transcript dictionaries:

        .. code-block:: python

            {
                "transcript_id": "t1",
                "entities": [
                    {"entity_text": "metformin", "category": "MED",
                     "start_char": 0, "end_char": 9},
                ],
            }

        Predicted entity dicts may additionally carry ``rectified_text`` (the final
        output after candidate recovery) and ``retrieval_candidates``. ``rectified_text``
        defaults to ``entity_text`` when absent. Entities with a category outside the
        five official CARE-ASR categories are logged and skipped.

        Args:
            batch_id (str): Unique benchmark evaluation run identifier.
            ground_truth_dataset (list[dict[str, Any]]): Ground-truth transcript annotations.
            predicted_outputs (list[dict[str, Any]]): Predicted validated entity outputs.

        Returns:
            ErrorAnalysisAuditOutput: Validated audit report contract emitted to Mahi.

        Examples:
            >>> engine = ErrorAnalysisEngine()
            >>> report = engine.generate_audit_report("run_001", gt_data, pred_data)
            >>> report.overall_metrics.rectified_f1 >= 0.0
            True
        """
        gt_entities = self._flatten_entities(ground_truth_dataset)
        pred_entities = self._flatten_entities(predicted_outputs)

        gt_by_category = {category: [e for e in gt_entities if e["category"] == category] for category in _CATEGORIES}
        pred_by_category = {
            category: [e for e in pred_entities if e["category"] == category] for category in _CATEGORIES
        }

        breakdown, unmatched_by_category = self._compute_category_metrics(gt_by_category, pred_by_category)

        pred_by_transcript: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in pred_entities:
            pred_by_transcript[entity["transcript_id"]].append(entity)

        failed_instances = self._collect_failed_instances(unmatched_by_category, pred_by_transcript)
        error_taxonomy = self.taxonomy_classifier.aggregate_taxonomy(
            [failure.model_dump() for failure in failed_instances]
        )
        overall_metrics = self._compute_overall_metrics(gt_by_category, pred_by_category)

        logger.info(
            f"Generated audit report '{batch_id}': {len(gt_entities)} ground-truth entities, "
            f"{len(pred_entities)} predicted entities, {len(failed_instances)} failed instances."
        )

        return ErrorAnalysisAuditOutput(
            batch_id=batch_id,
            total_samples=len(ground_truth_dataset),
            overall_metrics=overall_metrics,
            category_breakdown=breakdown,
            error_taxonomy=error_taxonomy,
            failed_instances=failed_instances,
        )

    def _flatten_entities(self, transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flattens transcript items into entity dicts carrying their transcript_id.

        Entities with a category outside the five official CARE-ASR categories are
        logged and skipped.
        """
        entities: list[dict[str, Any]] = []
        for item in transcripts:
            transcript_id = str(item.get("transcript_id", ""))
            for entity in item.get("entities", []):
                category = str(entity.get("category", ""))
                if category not in _CATEGORIES:
                    logger.warning(
                        f"Skipping entity with unknown category '{category}' " f"in transcript '{transcript_id}'."
                    )
                    continue
                entities.append({**entity, "transcript_id": transcript_id})
        return entities

    def _match_entities(
        self,
        gt_entities: list[dict[str, Any]],
        pred_entities: list[dict[str, Any]],
        text_key: str = "entity_text",
    ) -> tuple[int, int, int, list[dict[str, Any]]]:
        """Deterministically matches predicted entities to ground truth one-to-one.

        A predicted entity is a true positive when an unmatched ground-truth entity
        with the same normalized text exists; ``text_key`` selects which predicted
        field is compared (``rectified_text`` falls back to ``entity_text``).

        Returns:
            tuple[int, int, int, list[dict[str, Any]]]: (true_positives, false_positives,
                false_negatives, unmatched_ground_truth_entities).
        """
        by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in gt_entities:
            by_text[_normalize(str(entity.get("entity_text", "")))].append(entity)

        matched_ids: set[int] = set()
        true_positives = 0
        false_positives = 0

        for entity in pred_entities:
            raw = entity.get(text_key)
            if not raw:
                raw = entity.get("entity_text", "")
            key = _normalize(str(raw))

            bucket = by_text.get(key)
            if bucket is None:
                false_positives += 1
                continue

            match = next((candidate for candidate in bucket if id(candidate) not in matched_ids), None)
            if match is None:
                false_positives += 1
                continue

            matched_ids.add(id(match))
            true_positives += 1

        unmatched = [entity for entity in gt_entities if id(entity) not in matched_ids]
        return true_positives, false_positives, len(unmatched), unmatched

    def _compute_category_metrics(
        self,
        gt_by_category: dict[str, list[dict[str, Any]]],
        pred_by_category: dict[str, list[dict[str, Any]]],
    ) -> tuple[CategoryBreakdown, dict[str, list[dict[str, Any]]]]:
        """Computes per-category entity-level metrics against the rectified outputs.

        Returns:
            tuple[CategoryBreakdown, dict[str, list[dict[str, Any]]]]: The category
                breakdown and the unmatched ground-truth entities per category.
        """
        metrics: dict[str, CategoryMetric] = {}
        unmatched_by_category: dict[str, list[dict[str, Any]]] = {}

        for category in _CATEGORIES:
            true_positives, false_positives, false_negatives, unmatched = self._match_entities(
                gt_by_category[category], pred_by_category[category], text_key="rectified_text"
            )
            precision = _precision(true_positives, false_positives)
            recall = _recall(true_positives, false_negatives)
            metrics[category] = CategoryMetric(
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1_score=round(_f1_score(precision, recall), 4),
                support=len(gt_by_category[category]),
            )
            unmatched_by_category[category] = unmatched

        return CategoryBreakdown(**metrics), unmatched_by_category

    def _compute_overall_metrics(
        self,
        gt_by_category: dict[str, list[dict[str, Any]]],
        pred_by_category: dict[str, list[dict[str, Any]]],
    ) -> OverallMetrics:
        """Computes raw vs rectified overall entity-level metrics and their gains.

        The raw ASR baseline compares ``entity_text`` against ground truth; the
        rectified baseline compares ``rectified_text``. Gains are the rectified
        minus raw precision and recall deltas.
        """
        raw_tp = raw_fp = raw_fn = 0
        rect_tp = rect_fp = rect_fn = 0

        for category in _CATEGORIES:
            tp, fp, fn, _ = self._match_entities(
                gt_by_category[category], pred_by_category[category], text_key="entity_text"
            )
            raw_tp += tp
            raw_fp += fp
            raw_fn += fn

            tp, fp, fn, _ = self._match_entities(
                gt_by_category[category], pred_by_category[category], text_key="rectified_text"
            )
            rect_tp += tp
            rect_fp += fp
            rect_fn += fn

        raw_precision = _precision(raw_tp, raw_fp)
        raw_recall = _recall(raw_tp, raw_fn)
        rectified_precision = _precision(rect_tp, rect_fp)
        rectified_recall = _recall(rect_tp, rect_fn)

        return OverallMetrics(
            raw_asr_f1=round(_f1_score(raw_precision, raw_recall), 4),
            rectified_f1=round(_f1_score(rectified_precision, rectified_recall), 4),
            precision_gain=round(rectified_precision - raw_precision, 4),
            recall_gain=round(rectified_recall - raw_recall, 4),
        )

    def _collect_failed_instances(
        self,
        unmatched_by_category: dict[str, list[dict[str, Any]]],
        pred_by_transcript: dict[str, list[dict[str, Any]]],
    ) -> list[FailedInstance]:
        """Builds a FailedInstance for every unrecovered ground-truth entity.

        The predicted context is taken from the predicted entity in the same
        transcript that matches the missed entity's character offsets (falling back
        to normalized-text equality), then classified via FailureTaxonomyClassifier.
        """
        failed_instances: list[FailedInstance] = []
        for unmatched in unmatched_by_category.values():
            for gt_entity in unmatched:
                transcript_id = str(gt_entity.get("transcript_id", ""))
                confusion = self._find_confusion(gt_entity, pred_by_transcript.get(transcript_id, []))
                predicted_text = str(confusion.get("entity_text", "")) if confusion else ""
                candidates = confusion.get("retrieval_candidates", []) if confusion else []

                failure_type = self.taxonomy_classifier.classify_failure(
                    ground_truth=str(gt_entity.get("entity_text", "")),
                    predicted=predicted_text,
                    retrieval_candidates=candidates,
                )

                failed_instances.append(
                    FailedInstance(
                        transcript_id=transcript_id,
                        ground_truth=str(gt_entity.get("entity_text", "")),
                        predicted=predicted_text,
                        failure_type=failure_type,
                    )
                )
        return failed_instances

    def _find_confusion(self, gt_entity: dict[str, Any], pred_entities: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Finds the predicted entity that best explains a missed ground-truth entity.

        Prefers exact character-offset equality, then normalized-text equality. The
        text fallback is intentionally category-agnostic so cross-category confusions
        are still surfaced for taxonomy classification.
        """
        gt_start = gt_entity.get("start_char")
        gt_end = gt_entity.get("end_char")
        gt_text = _normalize(str(gt_entity.get("entity_text", "")))

        for entity in pred_entities:
            if entity.get("start_char") == gt_start and entity.get("end_char") == gt_end:
                return entity

        for entity in pred_entities:
            if _normalize(str(entity.get("entity_text", ""))) == gt_text:
                return entity

        return None
