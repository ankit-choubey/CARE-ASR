"""
CARE-ASR Shared Data Object Definitions.

This module implements the binding interface contract defined in
docs/interface_contract.md. All pipeline modules communicate exclusively
through these immutable data objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


class EntityCategory(str, Enum):
    MEDICATION = "MED"
    CONDITION = "COND"
    ANATOMY = "ANA"
    PROCEDURE = "TTP"
    NON_MEDICAL = "NON"


class RetrievalSource(str, Enum):
    SEMANTIC = "SEMANTIC_FAISS"
    PHONETIC = "PHONETIC_METAPHONE"


@dataclass(frozen=True)
class AudioInput:
    """Input audio container for the CARE-ASR pipeline.

    Attributes:
        file_path: Absolute path to the WAV/FLAC/MP3 file (Optional if bytes provided).
        audio_bytes: Raw PCM audio bytes (16kHz mono).
        sample_rate: Audio sampling frequency in Hz (Default: 16000).
        speaker_id: Metadata tag for speaker/accent tracking.
        session_id: Unique identifier for the transaction run.
    """

    file_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    sample_rate: int = 16000
    speaker_id: Optional[str] = "unknown"
    session_id: str = field(default_factory=lambda: "session_0000")


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_time: float
    end_time: float
    token_indices: List[int]


@dataclass(frozen=True)
class Transcript:
    """ASR output container containing raw text, tokenization, and probability tensors.

    Attributes:
        raw_text: Full transcribed string produced by ASR decoder.
        tokens: List of subword token strings.
        token_ids: Tensor or list of vocabulary token IDs.
        logits: PyTorch tensor of shape (Sequence_Length, Vocab_Size) containing
            unnormalized log probability distributions.
        word_timings: Word-level timestamp alignments.
        language: Detected language code (e.g., 'en').
    """

    raw_text: str
    tokens: List[str]
    token_ids: List[int]
    logits: torch.Tensor
    word_timings: List[WordTiming]
    language: str = "en"


@dataclass(frozen=True)
class ConfidenceScore:
    """Uncertainty analysis output computed via Tsallis entropy gating.

    Attributes:
        token_entropies: List of float values representing Tsallis entropy per token.
        word_entropies: List of float values mapped to word-level boundaries.
        is_uncertain_token: Boolean list where True indicates entropy > tau_entropy.
        is_uncertain_word: Boolean list for word-level uncertainty flag.
        q_parameter: Tsallis entropy non-extensivity parameter q (Default: 0.5).
        threshold_tau: Entropy cutoff threshold tau used for flagging.
    """

    token_entropies: List[float]
    word_entropies: List[float]
    is_uncertain_token: List[bool]
    is_uncertain_word: List[bool]
    q_parameter: float = 0.5
    threshold_tau: float = 0.45


@dataclass(frozen=True)
class EntitySpan:
    """Tagged clinical entity span requiring potential retrieval post-processing.

    Attributes:
        text: Raw text string of the entity span.
        category: EntityCategory enum tag (MED, COND, ANA, TTP).
        start_char: Character start index in Transcript.raw_text.
        end_char: Character end index in Transcript.raw_text.
        start_word_idx: Word start index in Transcript.word_timings.
        end_word_idx: Word end index in Transcript.word_timings.
        mean_entropy: Aggregated Tsallis entropy across the entity span.
        requires_correction: True if mean_entropy > tau_entropy AND category != NON.
    """

    text: str
    category: EntityCategory
    start_char: int
    end_char: int
    start_word_idx: int
    end_word_idx: int
    mean_entropy: float
    requires_correction: bool


@dataclass(frozen=True)
class RetrievalCandidate:
    """Retrieved concept candidate from medical knowledge base.

    Attributes:
        concept_id: UMLS CUI or RxNorm RxCUI identifier string (e.g., 'C0025598').
        concept_name: Preferred clinical concept label (e.g., 'Metformin').
        source: RetrievalSource enum tag (SEMANTIC or PHONETIC).
        raw_score: Raw similarity score (Cosine sim for semantic, distance ratio for
            phonetic).
        rank: Rank position within its single-source retrieval list (1-indexed).
    """

    concept_id: str
    concept_name: str
    source: RetrievalSource
    raw_score: float
    rank: int


@dataclass(frozen=True)
class FusionCandidate:
    """Merged candidate concept after Reciprocal Rank Fusion (RRF).

    Attributes:
        concept_id: UMLS CUI or RxNorm RxCUI.
        concept_name: Preferred clinical concept string.
        rrf_score: Calculated RRF score value.
        semantic_rank: Rank from semantic retrieval (None if missing).
        phonetic_rank: Rank from phonetic retrieval (None if missing).
        final_rank: Unified rank order (1-indexed).
    """

    concept_id: str
    concept_name: str
    rrf_score: float
    semantic_rank: Optional[int]
    phonetic_rank: Optional[int]
    final_rank: int


@dataclass(frozen=True)
class CorrectionResult:
    """Final decision result for a single clinical entity span.

    Attributes:
        original_span: Original entity text string.
        proposed_span: Term proposed by LLM.
        final_span: Final output term (matches proposed_span if accepted,
            original_span if rejected).
        is_edited: True if an edit was accepted.
        rejection_reason: Explanation if edit was rejected by safety gate
            (None if accepted).
        levenshtein_distance: Character edit distance between original and final term.
        selected_candidate: FusionCandidate object if replacement occurred.
    """

    original_span: str
    proposed_span: str
    final_span: str
    is_edited: bool
    rejection_reason: Optional[str] = None
    levenshtein_distance: int = 0
    selected_candidate: Optional[FusionCandidate] = None


@dataclass(frozen=True)
class PipelineOutput:
    """Final pipeline container holding corrected transcript and complete execution
    trace.

    Attributes:
        session_id: Unique run session ID.
        raw_transcript: Original raw ASR transcript text.
        corrected_transcript: Final post-processed clinical transcript.
        entity_spans: List of all detected entity spans.
        corrections: List of CorrectionResult objects for processed entity spans.
        total_latency_seconds: Total pipeline execution time in seconds.
        latency_breakdown: Execution time breakdown per pipeline module.
    """

    session_id: str
    raw_transcript: str
    corrected_transcript: str
    entity_spans: List[EntitySpan]
    corrections: List[CorrectionResult]
    total_latency_seconds: float
    latency_breakdown: Dict[str, float]
