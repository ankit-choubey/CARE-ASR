"""Category-Specific Thresholding Subpackage for CARE-ASR Module.

Why It Exists:
    Provides dynamic, category-tailored acceptance threshold evaluation for the 5 official
    CARE-ASR entity categories (MED, COND, ANA, TTP, PHI).

Teammate Dependencies:
    - Candidate Evaluator uses threshold engine to verify FAISS candidates.

Imported By:
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Implement dynamic threshold tuning interface for online reinforcement learning.
"""

from care_asr.thresholds.threshold_engine import CategoryThresholdEngine

__all__ = ["CategoryThresholdEngine"]
