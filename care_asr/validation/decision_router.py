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

logger = logging.getLogger(__name__)


class DecisionRouter:
    """Routes entity spans to candidate recovery based on ASR entropy and category risk rules.

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
        """Determines if an entity span's ASR entropy exceeds category risk tolerance.

        Args:
            category (str): Official entity category (MED, COND, ANA, TTP, PHI).
            asr_confidence (float): Primary ASR word confidence score.
            asr_entropy (float): Primary Whisper frame-level logit entropy score.

        Returns:
            bool: True if retrieval candidate recovery should be triggered for this span.

        Examples:
            >>> router = DecisionRouter(threshold_engine)
            >>> router.should_trigger_recovery("MED", 0.50, 0.85)
            True
        """
        pass
