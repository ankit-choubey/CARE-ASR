"""
Integration test checkpoint T15 — Full pipeline with UNSURE refusal fallback.
"""

from care_asr.contracts.validated_output import CorrectionOutput
from src.pipeline.pipeline import CARPipeline
from src.safety.unsure_gate import UnsureGate


def test_t15_unsure_fallback_preserves_original():
    """Verify safety gate prevents hallucinated corrections when label is UNSURE."""
    p = CARPipeline()
    p.safety_gate = UnsureGate().apply
    p.corrector = lambda token, cands: CorrectionOutput(
        original_token=token,
        corrected_token="wrong_drug",
        label="UNSURE",
        confidence=0.0,
    )
    res = p.run("fake_audio.wav")
    assert "wrong_drug" not in res["corrected"]
