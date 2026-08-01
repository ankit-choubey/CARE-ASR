"""Data Interface Contracts Package for CARE-ASR Module.

Why It Exists:
    Defines Pydantic v2 schemas enforcing strict interface contracts for all incoming
    and outgoing JSON payloads between teammates.

Teammate Dependencies:
    - Ankit (ASR Lead): Produces `ASRTranscriptInput`, consumes `ValidatedCandidatesOutput`.
    - Divya (FAISS Lead): Produces `RetrievalCandidatesInput`.
    - Mahi (Testing Lead): Consumes `ErrorAnalysisAuditOutput`.

Imported By:
    - `care_asr.ner.extractor`
    - `care_asr.validation.candidate_evaluator`
    - `care_asr.evaluation.metrics_calculator`

TODOs:
    - Add custom Pydantic validators for character offset boundaries.
"""

from care_asr.contracts.asr_input import ASRTranscriptInput, WordAlignment
from care_asr.contracts.error_analysis_output import (
    CategoryMetric,
    ErrorAnalysisAuditOutput,
    ErrorTaxonomy,
)
from care_asr.contracts.retrieval_input import (
    EntityQuery,
    PhoneticCandidate,
    RetrievalCandidatesInput,
    SemanticCandidate,
)
from care_asr.contracts.validated_output import (
    AppliedThresholds,
    DetectedEntity,
    ValidatedCandidate,
    ValidatedCandidatesOutput,
)

__all__ = [
    "ASRTranscriptInput",
    "WordAlignment",
    "RetrievalCandidatesInput",
    "EntityQuery",
    "SemanticCandidate",
    "PhoneticCandidate",
    "ValidatedCandidatesOutput",
    "DetectedEntity",
    "ValidatedCandidate",
    "AppliedThresholds",
    "ErrorAnalysisAuditOutput",
    "CategoryMetric",
    "ErrorTaxonomy",
]
