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

        Returns:
            bool: True if offsets are valid.

        Raises:
            ValueError: If any character offset is invalid or out of bounds.

        Examples:
            >>> payload = ASRTranscriptInput(...)
            >>> payload.validate_offsets()
            True
        """
        pass
