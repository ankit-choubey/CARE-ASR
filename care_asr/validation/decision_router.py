"""Candidate Recovery Decision Router.

Why It Exists:
    Determines whether a detected entity span requires retrieval candidate recovery based on
    primary Whisper ASR logit entropy and category risk levels (MED, COND, ANA, TTP, PHI).

Teammate Dependencies:
    - Internal CandidateEvaluator uses DecisionRouter to flag entity spans for FAISS recovery.

Imported By:
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Add dynamic entropy threshold adjustments for noisy clinical acoustic environments.
"""

import logging

from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.utils.exceptions import ThresholdConfigurationError

logger = logging.getLogger(__name__)


class DecisionRouter:
    """Routes entity spans to candidate recovery based on ASR entropy and category risk rules.

    Threshold values are read from the in-memory ``thresholds`` dictionary cached by
    ``CategoryThresholdEngine``, which loads the ``thresholds`` block of ``config.yaml``
    once through ``get_settings()``. The router never re-loads configuration, so every
    configured category is guaranteed to expose ``min_asr_confidence`` and ``max_entropy``.

    Args:
        threshold_engine (CategoryThresholdEngine): Category threshold engine instance.
    """

    def __init__(self, threshold_engine: CategoryThresholdEngine) -> None:
        self.threshold_engine = threshold_engine
        logger.info("DecisionRouter initialized with CategoryThresholdEngine.")

    def should_trigger_recovery(
        self,
        category: str,
        asr_confidence: float,
        asr_entropy: float,
    ) -> bool:
        """Determines whether an entity span requires retrieval candidate recovery.

        Recovery is triggered when the primary ASR confidence falls below the
        category's ``min_asr_confidence`` threshold OR the primary ASR entropy
        exceeds the category's ``max_entropy`` threshold. Boundary equality does
        not trigger recovery.

        Args:
            category (str): Official entity category (MED, COND, ANA, TTP, PHI).
            asr_confidence (float): Primary ASR word confidence score in [0.0, 1.0].
            asr_entropy (float): Primary Whisper frame-level logit entropy score.

        Returns:
            bool: True if retrieval candidate recovery should be triggered for this span.

        Raises:
            ThresholdConfigurationError: If the category is not configured.

        Examples:
            >>> router = DecisionRouter(threshold_engine)
            >>> router.should_trigger_recovery("MED", 0.50, 0.85)
            True
        """
        thresholds = self.threshold_engine.thresholds

        if category not in thresholds:
            logger.error(f"Unknown category evaluated for recovery decision: {category}")
            raise ThresholdConfigurationError(f"Unknown category: {category}")

        rules = thresholds[category]
        # thresholds is dict[str, Any]; engine validated numerics, so casts are safe and keep mypy happy
        min_asr_confidence = float(rules["min_asr_confidence"])
        max_entropy = float(rules["max_entropy"])

        should_recover = asr_confidence < min_asr_confidence or asr_entropy > max_entropy

        logger.info(
            f"Recovery decision for category '{category}': "
            f"asr_confidence={asr_confidence:.4f} (min={min_asr_confidence}), "
            f"asr_entropy={asr_entropy:.4f} (max={max_entropy}) -> "
            f"trigger_recovery={should_recover}"
        )
        return should_recover
