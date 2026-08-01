"""
Unit tests for UNSURE Safety Gate (Task T10).
"""

import pytest

from care_asr.contracts.validated_output import CorrectionOutput
from src.safety.unsure_gate import UnsureGate


@pytest.fixture
def gate():
    g = UnsureGate.__new__(UnsureGate)
    g.threshold = 0.5
    return g


def _co(orig: str, corr: str, label: str, conf: float) -> CorrectionOutput:
    return CorrectionOutput(original_token=orig, corrected_token=corr, label=label, confidence=conf)


def test_unsure_label_keeps_original(gate):
    """Verify UNSURE label forces fallback to original token."""
    r = gate.apply(_co("cardigan", "carvedilol", "UNSURE", 0.0))
    assert r.corrected_token == "cardigan"
    assert r.label == "UNSURE"


def test_low_confidence_triggers_fallback(gate):
    """Verify confidence below threshold forces fallback to original token."""
    r = gate.apply(_co("amoxicilin", "amoxicillin", "CORRECT", 0.3))
    assert r.corrected_token == "amoxicilin"
    assert r.label == "UNSURE"
