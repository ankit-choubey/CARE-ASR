"""
Confidence score extraction utilities for CARE-ASR (S3/T1).

Provides helper functions for extracting and summarizing per-token
confidence scores from Whisper decoder outputs. The core extraction
logic lives in WhisperTranscriber.transcribe(); this module provides
standalone utilities for analysis and debugging.
"""

from __future__ import annotations

import numpy as np

from care_asr.contracts.asr_input import TokenScore, Transcript


def extract_low_confidence_tokens(
    transcript: Transcript,
    threshold: float = 0.5,
) -> list[TokenScore]:
    """Returns tokens with probability below the given threshold."""
    return [ts for ts in transcript.token_scores if ts.prob < threshold]


def mean_confidence(transcript: Transcript) -> float:
    """Returns the mean token probability across the transcript."""
    if not transcript.token_scores:
        return 0.0
    return float(np.mean([ts.prob for ts in transcript.token_scores]))


def confidence_summary(transcript: Transcript) -> dict[str, float]:
    """Returns a summary dict of confidence statistics."""
    if not transcript.token_scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
    probs = [ts.prob for ts in transcript.token_scores]
    return {
        "mean": float(np.mean(probs)),
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "std": float(np.std(probs)),
        "count": len(probs),
    }
