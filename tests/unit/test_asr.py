"""Unit tests for Whisper ASR transcriber interface and confidence score extractor (S3/T1)."""

import pytest
from care_asr.contracts.asr_input import TokenScore, Transcript


def test_transcript_creation():
    """Verifies Transcript Pydantic model construction."""
    t = Transcript(
        text="patient prescribed amoxicillin",
        token_scores=[
            TokenScore(step=0, token_id=1, token="patient", log_prob=-0.01, prob=0.99),
            TokenScore(step=1, token_id=2, token="prescribed", log_prob=-0.02, prob=0.98),
            TokenScore(step=2, token_id=3, token="amoxicillin", log_prob=-2.5, prob=0.08),
        ],
    )
    assert t.text == "patient prescribed amoxicillin"
    assert len(t.token_scores) == 3
    assert t.token_scores[2].prob < 0.5


def test_whisper_transcriber_importable():
    """Verifies WhisperTranscriber class is importable."""
    from src.asr.transcriber import WhisperTranscriber
    assert WhisperTranscriber is not None


def test_confidence_module_importable():
    """Verifies confidence module functions work as expected."""
    from src.asr.confidence import extract_low_confidence_tokens, mean_confidence, confidence_summary
    t = Transcript(
        text="test",
        token_scores=[TokenScore(step=0, token_id=1, token="test", log_prob=-0.5, prob=0.6)],
    )
    assert mean_confidence(t) == pytest.approx(0.6, abs=0.01)
    assert confidence_summary(t)["count"] == 1
    assert len(extract_low_confidence_tokens(t, threshold=0.5)) == 0
    assert len(extract_low_confidence_tokens(t, threshold=0.7)) == 1
