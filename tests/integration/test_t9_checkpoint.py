"""T9 second integration checkpoint — full pipeline with dual retrieval."""

from src.pipeline.pipeline import CARPipeline


def test_t9_full_pipeline_with_retrieval():
    """Runs CARPipeline end-to-end with stubs and checks output shape."""
    pipeline = CARPipeline()
    log = []
    result = pipeline.run("test_audio", attribution_log=log)
    assert "original" in result
    assert "corrected" in result
    assert "attribution" in result
    # Must have at least M1, M2, M3 entries
    modules = [entry["module"] for entry in log]
    assert "M1_ASR" in modules
    assert "M2_ENTROPY" in modules
    assert "M3_NER" in modules


def test_t9_attribution_has_latency():
    """Pipeline attribution log must include LATENCY entry."""
    pipeline = CARPipeline()
    log = []
    pipeline.run("test_audio", attribution_log=log)
    latency_entries = [e for e in log if e.get("module") == "LATENCY"]
    assert len(latency_entries) == 1
