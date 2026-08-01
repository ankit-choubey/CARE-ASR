"""
Unit tests for LLM Corrector parser (Task T7).
"""

import pytest

from src.correction.llm_corrector import LLMCorrector


@pytest.fixture
def corrector():
    c = LLMCorrector.__new__(LLMCorrector)
    c.cfg = {}
    return c


def test_parse_correct(corrector):
    """Verify parsing CORRECT schema output."""
    r = corrector._parse("CORRECT | amoxicillin", "amoxicilin", ["amoxicillin", "ampicillin"])
    assert r.label == "CORRECT"
    assert r.corrected_token == "amoxicillin"


def test_parse_unsure_keeps_original(corrector):
    """Verify parsing UNSURE schema output keeps original token."""
    r = corrector._parse("UNSURE", "cardigan", ["carvedilol"])
    assert r.label == "UNSURE"
    assert r.corrected_token == "cardigan"
