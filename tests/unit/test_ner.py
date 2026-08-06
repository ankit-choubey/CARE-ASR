"""Unit tests for BioBERT NER extractor interface (T4)."""

import pytest
from care_asr.ner.extractor import BioBertNERExtractor


def test_extractor_class_exists():
    """Verifies BioBertNERExtractor is importable."""
    assert BioBertNERExtractor is not None


def test_extractor_init_no_crash():
    """Extractor init should not crash (lazy model loading)."""
    extractor = BioBertNERExtractor()
    assert extractor is not None


def test_uninitialized_extract_raises():
    """Calling extract without loading model should raise RuntimeError."""
    extractor = BioBertNERExtractor()
    with pytest.raises((RuntimeError, AttributeError)):
        extractor.extract("patient prescribed amoxicillin")
