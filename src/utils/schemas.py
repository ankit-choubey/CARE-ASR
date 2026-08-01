"""
Canonical Schema Interface.
Re-exports Pydantic contracts from care_asr.contracts to maintain single source of truth.
"""

from care_asr.contracts.asr_input import ASRInput, TokenScore, Transcript, WordTimestamp
from care_asr.contracts.error_analysis_output import ErrorAnalysisOutput, NEREntity
from care_asr.contracts.retrieval_input import RetrievalCandidate, RetrievalInput
from care_asr.contracts.validated_output import CorrectionOutput, ValidatedCandidatesOutput

__all__ = [
    "ASRInput",
    "TokenScore",
    "WordTimestamp",
    "Transcript",
    "RetrievalCandidate",
    "RetrievalInput",
    "CorrectionOutput",
    "ValidatedCandidatesOutput",
    "ErrorAnalysisOutput",
    "NEREntity",
]
