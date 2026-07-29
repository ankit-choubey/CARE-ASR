import pytest

from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.utils.exceptions import ThresholdConfigurationError

@pytest.fixture
def engine():
    return CategoryThresholdEngine()

def test_valid_candidate(engine):
    # Base configuration for MED requires:
    # min_semantic_similarity=0.85, max_phonetic_distance=2.0
    # min_asr_confidence=0.75, max_entropy=0.45
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.80, 0.20)
    assert res.accepted is True
    assert len(res.rejection_reasons) == 0

def test_low_semantic_similarity(engine):
    res = engine.evaluate_candidate_acceptance("MED", 0.50, 1.0, 0.80, 0.20)
    assert res.accepted is False
    assert "semantic_similarity_below_threshold" in res.rejection_reasons

def test_high_entropy(engine):
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.80, 1.50)
    assert res.accepted is False
    assert "entropy_above_threshold" in res.rejection_reasons

def test_low_asr_confidence(engine):
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.10, 0.20)
    assert res.accepted is False
    assert "asr_confidence_below_threshold" in res.rejection_reasons

def test_high_phonetic_distance(engine):
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 5.0, 0.80, 0.20)
    assert res.accepted is False
    assert "phonetic_distance_above_threshold" in res.rejection_reasons

def test_multiple_simultaneous_failures(engine):
    # Fails all criteria
    res = engine.evaluate_candidate_acceptance("MED", 0.50, 5.0, 0.10, 1.50)
    assert res.accepted is False
    assert len(res.rejection_reasons) == 4
    expected_reasons = {
        "semantic_similarity_below_threshold",
        "phonetic_distance_above_threshold",
        "asr_confidence_below_threshold",
        "entropy_above_threshold"
    }
    assert set(res.rejection_reasons) == expected_reasons

def test_unknown_category(engine):
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        engine.evaluate_candidate_acceptance("INVALID", 0.90, 1.0, 0.80, 0.20)

def test_invalid_config(monkeypatch):
    from care_asr.config.settings import Settings
    
    class MockSettings(Settings):
        def load_yaml_config(self):
            return {"thresholds": {"MED": {}}} # missing all keys
            
    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)
    
    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        CategoryThresholdEngine()
        
def test_invalid_config_type(monkeypatch):
    from care_asr.config.settings import Settings
    
    class MockSettings(Settings):
        def load_yaml_config(self):
            return {"thresholds": {"MED": {
                "min_semantic_similarity": "NOT_NUMERIC",
                "max_phonetic_distance": 2.0,
                "min_asr_confidence": 0.75,
                "max_entropy": 0.45
            }}}
            
    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)
    
    with pytest.raises(ThresholdConfigurationError, match="must be numeric"):
        CategoryThresholdEngine()
