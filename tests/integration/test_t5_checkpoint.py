"""
Integration test checkpoint T5 — Stub pipeline verification.
"""

from src.pipeline.pipeline import CARPipeline


def test_pipeline_produces_corrected_string():
    """Verify E2E pipeline produces a non-empty corrected string."""
    p = CARPipeline()
    result = p.run("fake_audio.wav")
    assert isinstance(result["corrected"], str)
    assert len(result["corrected"]) > 0


def test_pipeline_attribution_contains_all_modules():
    """Verify attribution log captures trace across pipeline modules."""
    p = CARPipeline()
    log = []
    p.run("fake_audio.wav", attribution_log=log)
    modules = [e["module"] for e in log]
    assert "M1_ASR" in modules
    assert "M2_ENTROPY" in modules
    assert "M3_NER" in modules
