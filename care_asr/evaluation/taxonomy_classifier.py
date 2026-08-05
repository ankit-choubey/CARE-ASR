"""Failure Taxonomy Classifier for Error Analysis.

Why It Exists:
    Categorizes unrecovered entity failures into root cause taxonomy buckets:
    Phonetic Distortion, Out-Of-Vocabulary (OOV), BioBERT Boundary Mismatch, or Retrieval Miss.

Teammate Dependencies:
    - Internal ErrorAnalysisEngine uses FailureTaxonomyClassifier during audit report generation.

Imported By:
    - `care_asr.evaluation.metrics_calculator`

Classification Rules (deterministic, using only the provided inputs):
    1. ``retrieval_miss``       - No retrieval candidates were returned for the span.
    2. ``boundary_mismatch``    - Predicted and ground truth differ but one contains the
                                  other (NER span boundary misalignment).
    3. ``phonetic_distortion``  - Predicted is a close spelling/acoustic variant of the
                                  ground truth (similarity ratio >= 0.6).
    4. ``oov_error``            - Everything else: a dissimilar out-of-vocabulary term
                                  with candidates present.
"""

import logging
from difflib import SequenceMatcher
from typing import Any

from care_asr.contracts.error_analysis_output import ErrorTaxonomy

logger = logging.getLogger(__name__)

PHONETIC_SIMILARITY_THRESHOLD = 0.6

CATEGORY_PHONETIC_DISTORTION = "phonetic_distortion"
CATEGORY_OOV_ERROR = "oov_error"
CATEGORY_BOUNDARY_MISMATCH = "boundary_mismatch"
CATEGORY_RETRIEVAL_MISS = "retrieval_miss"

#: Maps taxonomy category strings to ErrorTaxonomy count fields.
CATEGORY_TO_COUNT_FIELD: dict[str, str] = {
    CATEGORY_PHONETIC_DISTORTION: "phonetic_distortion_count",
    CATEGORY_OOV_ERROR: "oov_error_count",
    CATEGORY_BOUNDARY_MISMATCH: "ner_boundary_mismatch_count",
    CATEGORY_RETRIEVAL_MISS: "retrieval_miss_count",
}


class FailureTaxonomyClassifier:
    """Classifies entity recovery failures into defined root cause categories."""

    @staticmethod
    def classify_failure(
        ground_truth: str,
        predicted: str,
        retrieval_candidates: list[dict[str, Any]],
    ) -> str:
        """Classifies a single unrecovered entity instance into a failure taxonomy category.

        The classification is deterministic and uses only the provided inputs:

        1. If no retrieval candidates were returned, the failure is ``retrieval_miss``.
        2. Otherwise, if the normalized strings differ and one contains the other,
           the failure is ``boundary_mismatch`` (NER span boundary misalignment).
        3. Otherwise, if the strings are closely similar (SequenceMatcher ratio
           ``>= 0.6``), the failure is ``phonetic_distortion``.
        4. Otherwise (including exact or degenerate matches), the failure is ``oov_error``.

        Args:
            ground_truth (str): Ground truth clinical entity string.
            predicted (str): Raw or rectified predicted entity string.
            retrieval_candidates (list[dict[str, Any]]): Retrieved candidates for the span.

        Returns:
            str: Taxonomy category string (phonetic_distortion, oov_error,
                boundary_mismatch, retrieval_miss).

        Examples:
            >>> cat = FailureTaxonomyClassifier.classify_failure("metformin", "metformin", [])
            >>> isinstance(cat, str)
            True
        """
        gt = FailureTaxonomyClassifier._normalize(ground_truth)
        pred = FailureTaxonomyClassifier._normalize(predicted)

        if not retrieval_candidates:
            logger.info("Classified failure as 'retrieval_miss' (no retrieval candidates).")
            return CATEGORY_RETRIEVAL_MISS

        if gt and pred and gt != pred:
            if gt in pred or pred in gt:
                logger.info("Classified failure as 'boundary_mismatch' (span containment).")
                return CATEGORY_BOUNDARY_MISMATCH
            if FailureTaxonomyClassifier._similarity_ratio(gt, pred) >= PHONETIC_SIMILARITY_THRESHOLD:
                logger.info("Classified failure as 'phonetic_distortion' (similar spelling).")
                return CATEGORY_PHONETIC_DISTORTION

        logger.info("Classified failure as 'oov_error'.")
        return CATEGORY_OOV_ERROR

    @staticmethod
    def aggregate_taxonomy(failures: list[dict[str, Any]]) -> ErrorTaxonomy:
        """Aggregates taxonomy counts across a list of failure instances.

        Each failure dictionary may already carry a ``failure_type`` key; if it is
        absent, the failure is classified on the fly from ``ground_truth``,
        ``predicted``, and ``retrieval_candidates``. Unknown categories are logged
        and skipped.

        Args:
            failures (list[dict[str, Any]]): List of failure instance dictionaries.

        Returns:
            ErrorTaxonomy: Pydantic model populated with root cause failure counts.

        Examples:
            >>> counts = FailureTaxonomyClassifier.aggregate_taxonomy(failures_list)
            >>> isinstance(counts, ErrorTaxonomy)
            True
        """
        counts: dict[str, int] = dict.fromkeys(CATEGORY_TO_COUNT_FIELD.values(), 0)

        for failure in failures:
            category = failure.get("failure_type")
            if category is None:
                category = FailureTaxonomyClassifier.classify_failure(
                    ground_truth=str(failure.get("ground_truth", "")),
                    predicted=str(failure.get("predicted", "")),
                    retrieval_candidates=failure.get("retrieval_candidates", []),
                )

            count_field = CATEGORY_TO_COUNT_FIELD.get(category)
            if count_field is None:
                logger.warning(f"Skipping failure with unknown taxonomy category: {category}")
                continue

            counts[count_field] += 1

        return ErrorTaxonomy(**counts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercases, trims, and collapses whitespace in the input string."""
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _similarity_ratio(left: str, right: str) -> float:
        """Returns a SequenceMatcher similarity ratio in [0.0, 1.0] for two strings."""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return float(SequenceMatcher(None, left, right).ratio())
