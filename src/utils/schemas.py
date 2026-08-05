"""
Canonical Schema Interface.
Re-exports Pydantic contracts from care_asr.contracts to maintain single source of truth.
"""

from care_asr.contracts.asr_input import ASRTranscriptInput, TokenScore, Transcript, WordTimestamp
from care_asr.contracts.error_analysis_output import ErrorAnalysisAuditOutput, NEREntity
from care_asr.contracts.retrieval_input import RetrievalCandidate, RetrievalCandidatesInput
from care_asr.contracts.validated_output import CorrectionOutput, ValidatedCandidatesOutput

__all__ = [
    "ASRTranscriptInput",
    "TokenScore",
    "WordTimestamp",
    "Transcript",
    "RetrievalCandidate",
    "RetrievalCandidatesInput",
    "CorrectionOutput",
    "ValidatedCandidatesOutput",
    "ErrorAnalysisAuditOutput",
    "NEREntity",
]
