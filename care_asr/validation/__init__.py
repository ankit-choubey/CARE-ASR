"""Retrieval Candidate Validation Subpackage for CARE-ASR Module.

Why It Exists:
    Evaluates FAISS semantic and phonetic candidates against primary Whisper ASR logit entropy
    and category-specific thresholds, producing validated and scored candidates for Ankit's pipeline.

Teammate Dependencies:
    - Ankit (Integration Lead): Consumes `ValidatedCandidatesOutput` from `CandidateEvaluator`.
    - Divya (FAISS Lead): Provides `RetrievalCandidatesInput` evaluated by this subpackage.

Imported By:
    - Main pipeline integration scripts.

TODOs:
    - Add hybrid candidate deduplication for candidates appearing in both semantic and phonetic FAISS.
"""

from care_asr.validation.candidate_evaluator import CandidateEvaluator
from care_asr.validation.decision_router import DecisionRouter

__all__ = ["CandidateEvaluator", "DecisionRouter"]
