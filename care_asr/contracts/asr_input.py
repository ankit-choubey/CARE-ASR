"""ASR Transcript Input Contract Schema.

Why It Exists:
    Enforces runtime validation for raw transcriptions, word-level timestamps,
    confidence scores, and frame-level entropy emitted by Ankit's Whisper ASR module.

Teammate Dependencies:
    - Ankit (Whisper ASR Lead): Produces payloads conforming to this schema.

Imported By:
    - `care_asr.ner.extractor`
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Implement Pydantic field validators ensuring start_time <= end_time.
"""

import logging
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WordAlignment(BaseModel):
    """Represents a single word alignment object with ASR confidence and entropy.

    Attributes:
        word (str): Decoded word token text.
        start_time (float): Word start timestamp in seconds.
        end_time (float): Word end timestamp in seconds.
        confidence (float): Whisper ASR confidence score in range [0.0, 1.0].
        entropy (float): Whisper frame-level logit entropy score (primary confidence metric).
        start_char (int): 0-based start character offset in raw transcript.
        end_char (int): 0-based end character offset in raw transcript.
    """

    word: str = Field(..., description="Decoded word token text")
    start_time: float = Field(..., ge=0.0, description="Word start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="Word end timestamp in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Whisper ASR confidence score")
    entropy: float = Field(..., ge=0.0, description="Whisper frame-level logit entropy score")
    start_char: int = Field(..., ge=0, description="Start character offset")
    end_char: int = Field(..., ge=0, description="End character offset")


class ASRTranscriptInput(BaseModel):
    """Input payload contract emitted by Ankit's Whisper ASR pipeline.

    Attributes:
        transcript_id (UUID): Unique transaction identifier for audio recording.
        audio_duration_seconds (float): Total audio duration in seconds.
        raw_transcript (str): Complete raw transcript string decoded by Whisper ASR.
        words (list[WordAlignment]): Ordered list of word alignment objects.
    """

    transcript_id: UUID = Field(..., description="Unique transaction UUID")
    audio_duration_seconds: float = Field(..., ge=0.0, description="Audio duration in seconds")
    raw_transcript: str = Field(..., min_length=1, description="Raw text transcript")
    words: list[WordAlignment] = Field(..., description="List of word alignment objects")

    def validate_offsets(self) -> bool:
        """Validates that word character offsets lie within raw_transcript bounds.

        Checks every word's ``start_char``/``end_char`` against the length of
        ``raw_transcript`` and verifies ``start_char <= end_char``. Offsets are
        treated as 0-based half-open (``[start_char, end_char)``) character
        ranges, so ``end_char == len(raw_transcript)`` is permitted.

        Returns:
            bool: True if offsets are valid.

        Raises:
            ValueError: If any character offset is invalid or out of bounds.

        Examples:
            >>> payload = ASRTranscriptInput(...)
            >>> payload.validate_offsets()
            True
        """
        transcript_len = len(self.raw_transcript)
        for word in self.words:
            if word.start_char > word.end_char:
                raise ValueError(
                    f"Word '{word.word}' has start_char > end_char " f"({word.start_char} > {word.end_char})."
                )
            if word.end_char > transcript_len:
                raise ValueError(
                    f"Word '{word.word}' end_char {word.end_char} exceeds raw_transcript " f"length {transcript_len}."
                )
        return True


class TokenScore(BaseModel):
    """Represents a token decoding step score."""

    step: int = Field(..., description="Decoding step index")
    token_id: int = Field(..., description="Decoded token vocabulary ID")
    token: str = Field(..., description="Decoded token string")
    log_prob: float = Field(..., description="Log probability score")
    prob: float = Field(..., description="Probability score in [0.0, 1.0]")
    entropy: float = Field(0.0, description="Tsallis entropy score")


class WordTimestamp(BaseModel):
    """Represents word timestamp metadata."""

    word: str = Field(..., description="Word string")
    start: float = Field(..., description="Start timestamp")
    end: float = Field(..., description="End timestamp")


class Transcript(BaseModel):
    """Represents decoded transcript with per-token scores."""

    text: str = Field(..., description="Full decoded text string")
    token_scores: list[TokenScore] = Field(default_factory=list, description="Token scores")
    word_timestamps: list[WordTimestamp] = Field(default_factory=list, description="Word timestamps")
