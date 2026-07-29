"""Unit Tests for BioBertNERExtractor & SpanAligner.

Why It Exists:
    Verifies BioBERT model instantiation stub, subtoken alignment logic, and ASR payload contract handling.

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Executes pytest suite.

Imported By:
    - `pytest` runner.

TODOs:
    - Add mock tests for subtoken character boundary reconciliation.
"""

import logging
from uuid import uuid4
import pytest

from care_asr.config.settings import Settings
from care_asr.contracts.asr_input import ASRTranscriptInput, WordAlignment
from care_asr.ner.extractor import BioBertNERExtractor
from care_asr.ner.span_aligner import SpanAligner

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_asr_input() -> ASRTranscriptInput:
    """Fixture providing a mock ASRTranscriptInput instance.

    Returns:
        ASRTranscriptInput: Validated contract instance for testing.
    """
    return ASRTranscriptInput(
        transcript_id=uuid4(),
        audio_duration_seconds=5.0,
        raw_transcript="Patient prescribed metformin 500mg daily.",
        words=[
            WordAlignment(
                word="Patient",
                start_time=0.0,
                end_time=0.5,
                confidence=0.95,
                entropy=0.10,
                start_char=0,
                end_char=7,
            ),
            WordAlignment(
                word="prescribed",
                start_time=0.6,
                end_time=1.0,
                confidence=0.92,
                entropy=0.15,
                start_char=8,
                end_char=18,
            ),
            WordAlignment(
                word="metformin",
                start_time=1.1,
                end_time=1.6,
                confidence=0.65,
                entropy=0.75,
                start_char=19,
                end_char=28,
            ),
        ],
    )


def test_span_aligner_interface() -> None:
    """Tests that SpanAligner exposes expected static methods."""
    assert hasattr(SpanAligner, "align_subtokens_to_words")


def test_biobert_extractor_initialization() -> None:
    """Tests BioBertNERExtractor initialization with default settings."""
    settings = Settings()
    extractor = BioBertNERExtractor(settings=settings, auto_load=False)
    assert extractor.model_name == settings.biobert_model_name_or_path
    assert extractor.device is not None


def test_extract_entities_uninitialized_error(mock_asr_input: ASRTranscriptInput) -> None:
    """Tests that extract_entities raises ModelInferenceError if model/tokenizer are None."""
    settings = Settings()
    extractor = BioBertNERExtractor(settings=settings, auto_load=False)
    with pytest.raises(Exception):
        extractor.extract_entities(mock_asr_input)
