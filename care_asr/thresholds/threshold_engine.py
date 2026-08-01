"""Category Threshold Engine.

Why It Exists:
    A configurable decision engine that determines whether a retrieved candidate
    is acceptable for a detected medical entity based on category-specific rules.
    It evaluates metrics without performing retrieval or ranking.

Teammate Dependencies:
    - Divya (Retrieval Lead): Uses this to validate semantic and phonetic FAISS outputs.
    - Mahi (Testing Lead): Verifies validation logic limits against thresholds.

Design Rationale:
    - O(1) Execution Time: No loops over candidates or thresholds are performed during evaluation.
      Dictionary lookups are constant time.
    - Separation of Concerns: Helper methods isolate individual metric checks.
    - Aggregated Failures: Does not short-circuit, so all failure reasons are returned.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from care_asr.config.settings import get_settings
from care_asr.utils.exceptions import ThresholdConfigurationError

logger = logging.getLogger(__name__)


class ThresholdResult(BaseModel):
    """Structured response container for candidate evaluation results."""

    accepted: bool
    rejection_reasons: list[str]
    category: str
    thresholds_used: dict[str, float]
    input_metrics: dict[str, float]
    decision_timestamp: datetime


class CategoryThresholdEngine:
    """Evaluates candidate acceptance based on category-specific thresholds.

    This engine caches configuration in memory and evaluates candidate metrics in O(1) time.
    It aggregates multiple rule violations rather than stopping at the first failure.
    """

    def __init__(self) -> None:
        """Initializes the threshold engine and validates configuration."""
        self.settings = get_settings()
        self.thresholds = self._load_thresholds()
        self._validate_configuration()

    def _load_thresholds(self) -> dict[str, Any]:
        """Loads the 'thresholds' block from config.yaml.

        Returns:
            dict: The dictionary of category threshold constraints.

        Raises:
            ThresholdConfigurationError: If the 'thresholds' block is missing.
        """
        yaml_config = self.settings.load_yaml_config()
        thresholds = yaml_config.get("thresholds")
        if thresholds is None:
            raise ThresholdConfigurationError("Missing 'thresholds' block in configuration.")
        return thresholds

    def _validate_configuration(self) -> None:
        """Validates that all required metrics exist and are numeric for each configured category.

        Raises:
            ThresholdConfigurationError: If any configuration value is missing or invalid.
        """
        required_keys = [
            "min_semantic_similarity",
            "max_phonetic_distance",
            "min_asr_confidence",
            "max_entropy",
        ]
        if not self.thresholds:
            raise ThresholdConfigurationError("Thresholds configuration is empty.")

        for category, rules in self.thresholds.items():
            for key in required_keys:
                if key not in rules:
                    raise ThresholdConfigurationError(
                        f"Category '{category}' is missing required threshold '{key}'."
                    )
                if not isinstance(rules[key], (int, float)):
                    raise ThresholdConfigurationError(
                        f"Threshold '{key}' for category '{category}' must be numeric."
                    )

    def _check_semantic_similarity(
        self, value: float, threshold: float, reasons: list[str]
    ) -> None:
        """Evaluates semantic similarity against the minimum threshold."""
        if value < threshold:
            reasons.append("semantic_similarity_below_threshold")

    def _check_phonetic_distance(self, value: float, threshold: float, reasons: list[str]) -> None:
        """Evaluates phonetic distance against the maximum threshold."""
        if value > threshold:
            reasons.append("phonetic_distance_above_threshold")

    def _check_asr_confidence(self, value: float, threshold: float, reasons: list[str]) -> None:
        """Evaluates ASR confidence against the minimum threshold."""
        if value < threshold:
            reasons.append("asr_confidence_below_threshold")

    def _check_entropy(self, value: float, threshold: float, reasons: list[str]) -> None:
        """Evaluates ASR entropy against the maximum threshold."""
        if value > threshold:
            reasons.append("entropy_above_threshold")

    def _build_result(
        self,
        accepted: bool,
        rejection_reasons: list[str],
        category: str,
        thresholds_used: dict[str, float],
        input_metrics: dict[str, float],
    ) -> ThresholdResult:
        """Constructs the structured response object."""
        return ThresholdResult(
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            category=category,
            thresholds_used=thresholds_used,
            input_metrics=input_metrics,
            decision_timestamp=datetime.now(UTC),
        )

    def evaluate_candidate_acceptance(
        self,
        category: str,
        semantic_similarity: float,
        phonetic_distance: float,
        asr_confidence: float,
        entropy: float,
    ) -> ThresholdResult:
        """Evaluates a retrieved candidate against all configured thresholds for a specific category.

        Executes in O(1) time since all rules are directly accessed via dictionary hash lookups.

        Args:
            category (str): The entity category (e.g., MED, COND).
            semantic_similarity (float): The semantic similarity score of the candidate.
            phonetic_distance (float): The phonetic distance of the candidate.
            asr_confidence (float): The aggregated ASR confidence of the source entity span.
            entropy (float): The max ASR entropy of the source entity span.

        Returns:
            ThresholdResult: Pydantic model containing acceptance flag and aggregate metadata.

        Raises:
            ThresholdConfigurationError: If the provided category is unknown.
        """
        start_time = time.perf_counter()

        if category not in self.thresholds:
            logger.error(f"Unknown category evaluated: {category}")
            raise ThresholdConfigurationError(f"Unknown category: {category}")

        rules = self.thresholds[category]
        rejection_reasons: list[str] = []

        # Evaluate all rules independently without short-circuiting
        self._check_semantic_similarity(
            semantic_similarity, rules["min_semantic_similarity"], rejection_reasons
        )
        self._check_phonetic_distance(
            phonetic_distance, rules["max_phonetic_distance"], rejection_reasons
        )
        self._check_asr_confidence(asr_confidence, rules["min_asr_confidence"], rejection_reasons)
        self._check_entropy(entropy, rules["max_entropy"], rejection_reasons)

        accepted = len(rejection_reasons) == 0

        thresholds_used = {
            "min_semantic_similarity": rules["min_semantic_similarity"],
            "max_phonetic_distance": rules["max_phonetic_distance"],
            "min_asr_confidence": rules["min_asr_confidence"],
            "max_entropy": rules["max_entropy"],
        }

        input_metrics = {
            "semantic_similarity": semantic_similarity,
            "phonetic_distance": phonetic_distance,
            "asr_confidence": asr_confidence,
            "entropy": entropy,
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Evaluated candidate for category '{category}'. Accepted: {accepted}. "
            f"Failed rules: {rejection_reasons}. Metrics: {input_metrics}. "
            f"Exec time: {elapsed_ms:.3f}ms"
        )

        return self._build_result(
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            category=category,
            thresholds_used=thresholds_used,
            input_metrics=input_metrics,
        )
