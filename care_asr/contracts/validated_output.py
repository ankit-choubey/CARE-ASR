"""Validated Output Contract Schema.

Why It Exists:
    Enforces runtime validation for the final structured output emitted by this module
    to Ankit's pipeline integration layer. Contains detected entities, primary ASR entropy,
    and ranked, threshold-checked retrieval candidates.

Teammate Dependencies:
    - Ankit (Integration Lead): Consumes payloads conforming to this schema to perform
      final transcript text substitution and clinical EHR record population.

Imported By:
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Add utility function to extract top-ranked candidate for quick text substitution.
"""

import logging
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CategoryType = Literal["MED", "COND", "ANA", "TTP", "PHI"]
CandidateSourceType = Literal["SEMANTIC_FAISS", "PHONETIC_FAISS", "HYBRID"]


class AppliedThresholds(BaseModel):
    """Category operational thresholds applied during evaluation.

    Attributes:
        min_asr_conf_threshold (float): Minimum ASR confidence required.
        min_sim_threshold (float): Minimum semantic similarity score required.
        max_phonetic_dist_threshold (float): Maximum phonetic edit distance permitted.
    """

    min_asr_conf_threshold: float = Field(..., description="Minimum ASR confidence threshold")
    min_sim_threshold: float = Field(..., description="Minimum semantic similarity threshold")
    max_phonetic_dist_threshold: float = Field(..., description="Maximum phonetic distance threshold")


class ValidatedCandidate(BaseModel):
    """Represents a validated and scored retrieval candidate for an entity span.

    Attributes:
        candidate_rank (int): Rank index (1-based, 1 is highest composite utility).
        canonical_name (str): Standardized medical concept name.
        cui (str): UMLS Concept Unique Identifier.
        candidate_source (CandidateSourceType): Source index (SEMANTIC_FAISS, PHONETIC_FAISS, HYBRID).
        composite_utility_score (float): Weighted score combining entropy loss, semantic similarity, and phonetic proximity.
        semantic_similarity (float | None): Cosine similarity score if retrieved via semantic FAISS.
        phonetic_distance (float | None): Edit distance if retrieved via phonetic search.
        passes_category_threshold (bool): True if candidate satisfies all category operational thresholds.
    """

    candidate_rank: int = Field(..., ge=1, description="Candidate rank (1-indexed)")
    canonical_name: str = Field(..., description="Canonical concept name")
    cui: str = Field(..., description="UMLS CUI code")
    candidate_source: CandidateSourceType = Field(..., description="Retrieval source engine")
    composite_utility_score: float = Field(..., description="Weighted composite utility score")
    semantic_similarity: float | None = Field(default=None, description="Semantic cosine similarity")
    phonetic_distance: float | None = Field(default=None, description="Phonetic edit distance")
    passes_category_threshold: bool = Field(..., description="True if satisfies category thresholds")


class DetectedEntity(BaseModel):
    """Represents a detected clinical entity span with primary ASR confidence and candidate options.

    Attributes:
        entity_id (str): Unique entity identifier.
        category (CategoryType): Official CARE-ASR category (MED, COND, ANA, TTP, PHI).
        original_text (str): Entity text string extracted from raw transcript.
        start_char (int): 0-based start character offset.
        end_char (int): 0-based end character offset.
        primary_asr_confidence (float): Whisper ASR confidence score in [0.0, 1.0].
        asr_entropy (float): Primary logit entropy score from Whisper ASR.
        requires_recovery (bool): True if high entropy or low confidence triggers retrieval recovery.
        validated_candidates (list[ValidatedCandidate]): Ranked list of validated retrieval candidates.
        applied_thresholds (AppliedThresholds): Operating thresholds applied for this entity category.
    """

    entity_id: str = Field(..., description="Unique entity identifier")
    category: CategoryType = Field(..., description="Official CARE-ASR category")
    original_text: str = Field(..., description="Original text string from transcript")
    start_char: int = Field(..., ge=0, description="Start character offset")
    end_char: int = Field(..., ge=0, description="End character offset")
    primary_asr_confidence: float = Field(..., ge=0.0, le=1.0, description="Primary ASR confidence")
    asr_entropy: float = Field(..., ge=0.0, description="Primary ASR logit entropy")
    requires_recovery: bool = Field(..., description="True if retrieval candidate recovery is triggered")
    validated_candidates: list[ValidatedCandidate] = Field(default_factory=list, description="Ranked candidates")
    applied_thresholds: AppliedThresholds = Field(..., description="Threshold configuration applied")


class ProcessingMetadata(BaseModel):
    """Execution timing and processing telemetry metadata.

    Attributes:
        ner_execution_time_ms (float): BioBERT NER extraction execution time in milliseconds.
        validation_execution_time_ms (float): Candidate evaluation execution time in milliseconds.
        entities_count (int): Total number of clinical entities detected.
        high_entropy_entities_count (int): Number of entities requiring retrieval recovery.
    """

    ner_execution_time_ms: float = Field(..., ge=0.0, description="NER execution time in ms")
    validation_execution_time_ms: float = Field(..., ge=0.0, description="Validation execution time in ms")
    entities_count: int = Field(..., ge=0, description="Total entities detected")
    high_entropy_entities_count: int = Field(..., ge=0, description="High-entropy entities count")


class ValidatedCandidatesOutput(BaseModel):
    """Final output payload contract emitted to Ankit's integration pipeline.

    Attributes:
        transcript_id (UUID): Unique transaction identifier matching ASR payload.
        detected_entities (list[DetectedEntity]): List of detected entities with validated candidates.
        processing_metadata (ProcessingMetadata): Performance telemetry metadata.
    """

    transcript_id: UUID = Field(..., description="Unique transaction UUID")
    detected_entities: list[DetectedEntity] = Field(..., description="List of detected and validated entities")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing telemetry metadata")
