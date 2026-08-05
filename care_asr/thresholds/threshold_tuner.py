"""Category-Specific Threshold Tuner.

Why It Exists:
    Tunes category-specific acceptance thresholds by sweeping a grid of candidate
    values and scoring each combination against a set of observed candidate
    metrics, using the existing CategoryThresholdEngine runtime override API.

Teammate Dependencies:
    - Divya (Retrieval Lead): Uses tuning results to adjust retrieval acceptance
      strictness per category based on observed candidate metrics.
    - Mahi (Testing Lead): Verifies that tuning is deterministic and that invalid
      tuning runs never mutate engine state.

Design Rationale:
    - Reuses CategoryThresholdEngine entirely: candidate combinations are applied
      via ``update_category_thresholds()`` and scored via
      ``evaluate_candidate_acceptance()``, so no threshold storage or validation
      logic is duplicated.
    - Deterministic: combinations are generated in grid order and score ties keep
      the first encountered combination.
    - In-memory only: winning thresholds are applied as runtime overrides and
      nothing is persisted to disk.
"""

import itertools
import logging
from collections.abc import Iterator

from pydantic import BaseModel

from care_asr.thresholds.threshold_engine import CategoryThresholdEngine, ThresholdResult
from care_asr.utils.exceptions import ThresholdConfigurationError

logger = logging.getLogger(__name__)


class ThresholdTuningResult(BaseModel):
    """Structured summary of a completed tuning run for one category.

    Attributes:
        category: The entity category that was tuned (e.g., MED, COND).
        original_thresholds: The numeric thresholds in effect before tuning.
        tuned_thresholds: The winning numeric thresholds applied by the tuner.
        combinations_evaluated: Number of threshold combinations scored.
        best_score: Acceptance-rate score (fraction of accepted candidates) of the
            winning combination over the supplied candidate metrics.
    """

    category: str
    original_thresholds: dict[str, float]
    tuned_thresholds: dict[str, float]
    combinations_evaluated: int
    best_score: float


class ThresholdTuner:
    """Tunes category thresholds by sweeping a grid of candidate values.

    The tuner reuses the existing ``CategoryThresholdEngine`` for both applying
    candidate combinations (``update_category_thresholds``) and scoring them
    (``evaluate_candidate_acceptance``). It never introduces its own threshold
    storage or configuration source.
    """

    _METRIC_KEYS: tuple[str, ...] = (
        "semantic_similarity",
        "phonetic_distance",
        "asr_confidence",
        "entropy",
    )

    def __init__(self, engine: CategoryThresholdEngine | None = None) -> None:
        """Initializes the tuner around a threshold engine.

        Args:
            engine: The ``CategoryThresholdEngine`` to tune. Defaults to a fresh
                engine loaded from the project configuration.
        """
        self.engine = engine if engine is not None else CategoryThresholdEngine()

    def tune(
        self,
        category: str,
        candidate_metrics: list[dict[str, float]],
        grid: dict[str, list[float]],
    ) -> ThresholdTuningResult:
        """Tunes a single category and applies the winning thresholds.

        Every combination in the Cartesian product of the grid values is applied
        through ``update_category_thresholds`` and scored with
        ``evaluate_candidate_acceptance`` over the supplied candidate metrics.
        The combination with the highest acceptance rate wins; ties keep the
        first combination encountered. The winning combination is then applied to
        the engine as a runtime override.

        Args:
            category: The entity category to tune (e.g., MED, COND).
            candidate_metrics: Observed candidate metrics used for scoring. Each
                entry must contain the keys ``semantic_similarity``,
                ``phonetic_distance``, ``asr_confidence``, and ``entropy``.
            grid: Candidate threshold values keyed by threshold name, e.g.
                ``{"min_semantic_similarity": [0.80, 0.85]}``.

        Returns:
            ThresholdTuningResult: Structured summary of the tuning run.

        Raises:
            ThresholdConfigurationError: If the category is unknown or a grid
                combination is invalid for the engine.
            ValueError: If the candidate list or grid is empty, or a candidate
                metric dict is malformed.
        """
        if category not in self.engine.thresholds:
            logger.error("Unknown category passed to tuner: %s", category)
            raise ThresholdConfigurationError(f"Unknown category: {category}")

        self._validate_candidate_metrics(candidate_metrics)
        self._validate_grid(grid)

        original_thresholds = {
            key: float(self.engine.thresholds[category][key]) for key in CategoryThresholdEngine._REQUIRED_KEYS
        }

        # _validate_grid guarantees at least one combination in the product.
        combinations = list(self._generate_combinations(grid))

        best_score = -1.0
        best_combination: dict[str, float] = {}
        for combination in combinations:
            try:
                self.engine.update_category_thresholds(category, combination)
            except ThresholdConfigurationError:
                # The engine validates before mutating, so state is already
                # unchanged; restore explicitly to stay robust to future ordering.
                self.engine.update_category_thresholds(category, original_thresholds)
                raise
            score = self._score_combination(category, candidate_metrics)
            if score > best_score:
                best_score = score
                best_combination = combination

        self.engine.update_category_thresholds(category, best_combination)

        tuned_thresholds = {
            key: float(self.engine.thresholds[category][key]) for key in CategoryThresholdEngine._REQUIRED_KEYS
        }

        logger.info(
            "Tuned category '%s': best score %.4f over %d combination(s); " "thresholds %s -> %s",
            category,
            best_score,
            len(combinations),
            original_thresholds,
            tuned_thresholds,
        )

        return ThresholdTuningResult(
            category=category,
            original_thresholds=original_thresholds,
            tuned_thresholds=tuned_thresholds,
            combinations_evaluated=len(combinations),
            best_score=best_score,
        )

    def run_grid(
        self,
        candidate_metrics_by_category: dict[str, list[dict[str, float]]],
        grid: dict[str, list[float]],
    ) -> list[ThresholdTuningResult]:
        """Tunes multiple categories against the same grid.

        Categories are processed in sorted order for deterministic output. All
        inputs are validated before any category is tuned, so an invalid input
        raises without partially mutating engine state.

        Args:
            candidate_metrics_by_category: Candidate metrics keyed by category.
            grid: Candidate threshold values keyed by threshold name.

        Returns:
            list[ThresholdTuningResult]: One summary per category, in sorted
                category order.

        Raises:
            ThresholdConfigurationError: If any category is unknown.
            ValueError: If the input is empty or a candidate metric dict is
                malformed.
        """
        if not candidate_metrics_by_category:
            raise ValueError("No categories supplied for grid tuning.")

        self._validate_grid(grid)
        for category, candidate_metrics in candidate_metrics_by_category.items():
            if category not in self.engine.thresholds:
                logger.error("Unknown category passed to tuner: %s", category)
                raise ThresholdConfigurationError(f"Unknown category: {category}")
            self._validate_candidate_metrics(candidate_metrics)

        results = [
            self.tune(category, candidate_metrics_by_category[category], grid)
            for category in sorted(candidate_metrics_by_category)
        ]
        return results

    def _generate_combinations(self, grid: dict[str, list[float]]) -> Iterator[dict[str, float]]:
        """Yields the Cartesian product of grid values as threshold dicts."""
        keys = list(grid.keys())
        value_lists = [grid[key] for key in keys]
        for values in itertools.product(*value_lists):
            yield dict(zip(keys, values, strict=True))

    def _score_combination(
        self,
        category: str,
        candidate_metrics: list[dict[str, float]],
    ) -> float:
        """Scores a currently-applied threshold combination.

        The score is the fraction of candidate metrics accepted by the engine
        under the thresholds currently applied to the category.

        Returns:
            float: Acceptance rate in the range [0.0, 1.0].
        """
        accepted = 0
        for metrics in candidate_metrics:
            result: ThresholdResult = self.engine.evaluate_candidate_acceptance(
                category,
                semantic_similarity=metrics["semantic_similarity"],
                phonetic_distance=metrics["phonetic_distance"],
                asr_confidence=metrics["asr_confidence"],
                entropy=metrics["entropy"],
            )
            if result.accepted:
                accepted += 1
        return accepted / len(candidate_metrics)

    def _validate_candidate_metrics(self, candidate_metrics: list[dict[str, float]]) -> None:
        """Validates the structure of the candidate metrics input.

        This validates the tuner's own input format; threshold value validation
        remains owned by CategoryThresholdEngine.

        Raises:
            ValueError: If the list is empty or any entry is malformed.
        """
        if not candidate_metrics:
            raise ValueError("No candidate metrics provided; cannot score threshold combinations.")

        for index, metrics in enumerate(candidate_metrics):
            missing = [key for key in self._METRIC_KEYS if key not in metrics]
            if missing:
                raise ValueError(f"Candidate metrics at index {index} missing required keys: {missing}")
            for key in self._METRIC_KEYS:
                value = metrics[key]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Metric '{key}' at index {index} must be numeric, " f"got {type(value).__name__}."
                    )

    def _validate_grid(self, grid: dict[str, list[float]]) -> None:
        """Validates that the grid has at least one key with candidate values.

        Threshold-key and value validation is delegated to
        CategoryThresholdEngine when each combination is applied.

        Raises:
            ValueError: If the grid is empty or any key has no candidate values.
        """
        if not grid:
            raise ValueError("Grid must contain at least one threshold key.")

        for key, values in grid.items():
            if not values:
                raise ValueError(f"Grid key '{key}' has an empty list of candidate values.")
