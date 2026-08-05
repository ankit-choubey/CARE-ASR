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
        "entropy_above_threshold",
    }
    assert set(res.rejection_reasons) == expected_reasons


def test_unknown_category(engine):
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        engine.evaluate_candidate_acceptance("INVALID", 0.90, 1.0, 0.80, 0.20)


def test_invalid_config(monkeypatch):
    from care_asr.config.settings import Settings

    class MockSettings(Settings):
        def load_yaml_config(self):
            return {"thresholds": {"MED": {}}}  # missing all keys

    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)

    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        CategoryThresholdEngine()


def test_invalid_config_type(monkeypatch):
    from care_asr.config.settings import Settings

    class MockSettings(Settings):
        def load_yaml_config(self):
            return {
                "thresholds": {
                    "MED": {
                        "min_semantic_similarity": "NOT_NUMERIC",
                        "max_phonetic_distance": 2.0,
                        "min_asr_confidence": 0.75,
                        "max_entropy": 0.45,
                    }
                }
            }

    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)

    with pytest.raises(ThresholdConfigurationError, match="must be numeric"):
        CategoryThresholdEngine()


def test_update_category_thresholds_valid(engine: CategoryThresholdEngine) -> None:
    """Runtime override applies and takes effect in subsequent evaluations."""
    engine.update_category_thresholds("MED", {"min_asr_confidence": 0.82})

    assert engine.thresholds["MED"]["min_asr_confidence"] == 0.82

    # 0.80 is now below the raised minimum and must be rejected
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.80, 0.20)
    assert res.accepted is False
    assert "asr_confidence_below_threshold" in res.rejection_reasons
    assert res.thresholds_used["min_asr_confidence"] == 0.82


def test_update_category_thresholds_partial(engine: CategoryThresholdEngine) -> None:
    """Only the supplied keys change; all other thresholds remain untouched."""
    engine.update_category_thresholds("MED", {"min_asr_confidence": 0.82})

    rules = engine.thresholds["MED"]
    assert rules["min_asr_confidence"] == 0.82
    assert rules["max_entropy"] == 0.45
    assert rules["min_semantic_similarity"] == 0.85
    assert rules["max_phonetic_distance"] == 2.0


def test_update_category_thresholds_unknown_category(engine: CategoryThresholdEngine) -> None:
    """Unknown categories raise ThresholdConfigurationError without mutating state."""
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        engine.update_category_thresholds("INVALID", {"min_asr_confidence": 0.82})

    assert "INVALID" not in engine.thresholds


def test_update_category_thresholds_invalid_key(engine: CategoryThresholdEngine) -> None:
    """Unknown threshold keys raise ThresholdConfigurationError without mutating state."""
    with pytest.raises(ThresholdConfigurationError, match="Unknown threshold key"):
        engine.update_category_thresholds("MED", {"min_asr_confidence_typo": 0.82})

    assert engine.thresholds["MED"]["min_asr_confidence"] == 0.75


def test_update_category_thresholds_invalid_value_type(engine: CategoryThresholdEngine) -> None:
    """Non-numeric threshold values raise the same error as engine initialization."""
    with pytest.raises(ThresholdConfigurationError, match="must be numeric"):
        engine.update_category_thresholds("MED", {"min_asr_confidence": "NOT_NUMERIC"})

    assert engine.thresholds["MED"]["min_asr_confidence"] == 0.75


def test_update_category_thresholds_validation_failure(engine: CategoryThresholdEngine) -> None:
    """Merged rules failing validation raise the missing-key error without applying."""
    # Simulate a category whose in-memory rules lost a required key
    del engine.thresholds["MED"]["min_asr_confidence"]

    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        engine.update_category_thresholds("MED", {"max_entropy": 0.50})
