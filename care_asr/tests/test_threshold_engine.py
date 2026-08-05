from typing import Any, cast

import pytest

from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.thresholds.threshold_tuner import ThresholdTuner, ThresholdTuningResult
from care_asr.utils.exceptions import ThresholdConfigurationError


@pytest.fixture
def engine() -> CategoryThresholdEngine:
    return CategoryThresholdEngine()


def test_valid_candidate(engine: CategoryThresholdEngine) -> None:
    # Base configuration for MED requires:
    # min_semantic_similarity=0.85, max_phonetic_distance=2.0
    # min_asr_confidence=0.75, max_entropy=0.45
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.80, 0.20)
    assert res.accepted is True
    assert len(res.rejection_reasons) == 0


def test_low_semantic_similarity(engine: CategoryThresholdEngine) -> None:
    res = engine.evaluate_candidate_acceptance("MED", 0.50, 1.0, 0.80, 0.20)
    assert res.accepted is False
    assert "semantic_similarity_below_threshold" in res.rejection_reasons


def test_high_entropy(engine: CategoryThresholdEngine) -> None:
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.80, 1.50)
    assert res.accepted is False
    assert "entropy_above_threshold" in res.rejection_reasons


def test_low_asr_confidence(engine: CategoryThresholdEngine) -> None:
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 1.0, 0.10, 0.20)
    assert res.accepted is False
    assert "asr_confidence_below_threshold" in res.rejection_reasons


def test_high_phonetic_distance(engine: CategoryThresholdEngine) -> None:
    res = engine.evaluate_candidate_acceptance("MED", 0.90, 5.0, 0.80, 0.20)
    assert res.accepted is False
    assert "phonetic_distance_above_threshold" in res.rejection_reasons


def test_multiple_simultaneous_failures(engine: CategoryThresholdEngine) -> None:
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


def test_unknown_category(engine: CategoryThresholdEngine) -> None:
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        engine.evaluate_candidate_acceptance("INVALID", 0.90, 1.0, 0.80, 0.20)


def test_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from care_asr.config.settings import Settings

    class MockSettings(Settings):
        def load_yaml_config(self) -> dict[str, Any]:
            return {"thresholds": {"MED": {}}}  # missing all keys

    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)

    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        CategoryThresholdEngine()


def test_invalid_config_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from care_asr.config.settings import Settings

    class MockSettings(Settings):
        def load_yaml_config(self) -> dict[str, Any]:
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
        engine.update_category_thresholds("MED", {"min_asr_confidence": cast(float, "NOT_NUMERIC")})

    assert engine.thresholds["MED"]["min_asr_confidence"] == 0.75


def test_update_category_thresholds_validation_failure(engine: CategoryThresholdEngine) -> None:
    """Merged rules failing validation raise the missing-key error without applying."""
    # Simulate a category whose in-memory rules lost a required key
    del engine.thresholds["MED"]["min_asr_confidence"]

    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        engine.update_category_thresholds("MED", {"max_entropy": 0.50})


def _candidate_metrics(*similarities: float) -> list[dict[str, float]]:
    """Builds valid candidate metric dicts that vary only in semantic similarity."""
    return [
        {
            "semantic_similarity": similarity,
            "phonetic_distance": 1.0,
            "asr_confidence": 0.80,
            "entropy": 0.20,
        }
        for similarity in similarities
    ]


@pytest.fixture
def tuner(engine: CategoryThresholdEngine) -> ThresholdTuner:
    return ThresholdTuner(engine)


def test_tune_success(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """Tuning keeps the grid combination with the best acceptance score."""
    metrics = _candidate_metrics(0.80, 0.82, 0.90)
    result: ThresholdTuningResult = tuner.tune("MED", metrics, {"min_semantic_similarity": [0.85, 0.80]})

    assert result.category == "MED"
    assert result.original_thresholds["min_semantic_similarity"] == 0.85
    assert result.tuned_thresholds["min_semantic_similarity"] == 0.80
    assert result.combinations_evaluated == 2
    assert result.best_score == 1.0


def test_tune_applies_runtime_override(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """The winning combination is applied to the engine as a runtime override."""
    tuner.tune("MED", _candidate_metrics(0.80, 0.90), {"min_semantic_similarity": [0.85, 0.80]})

    assert engine.thresholds["MED"]["min_semantic_similarity"] == 0.80
    res = engine.evaluate_candidate_acceptance("MED", 0.80, 1.0, 0.80, 0.20)
    assert res.accepted is True


def test_tune_unknown_category_no_mutation(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """Unknown categories raise ThresholdConfigurationError without mutating state."""
    before = dict(engine.thresholds)
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        tuner.tune("INVALID", _candidate_metrics(0.80), {"min_semantic_similarity": [0.80]})
    assert engine.thresholds == before


def test_tune_unknown_grid_key_no_mutation(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """Unknown grid keys raise ThresholdConfigurationError without mutating state."""
    before = dict(engine.thresholds["MED"])
    with pytest.raises(ThresholdConfigurationError, match="Unknown threshold key"):
        tuner.tune(
            "MED",
            _candidate_metrics(0.80),
            {"min_semantic_similarity_typo": [0.80, 0.85]},
        )
    assert engine.thresholds["MED"] == before


def test_tune_invalid_grid_value_no_mutation(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """A non-numeric grid value restores the engine to its original thresholds."""
    before = dict(engine.thresholds["MED"])
    with pytest.raises(ThresholdConfigurationError, match="must be numeric"):
        tuner.tune(
            "MED",
            _candidate_metrics(0.80),
            {"min_semantic_similarity": [0.80, cast(float, "NOT_NUMERIC")]},
        )
    assert engine.thresholds["MED"] == before


def test_run_grid_multiple_categories(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """run_grid tunes every category and reports one summary per category."""
    med_metrics = _candidate_metrics(0.80, 0.82, 0.90)
    cond_metrics = _candidate_metrics(0.77, 0.79, 0.90)
    results = tuner.run_grid(
        {"MED": med_metrics, "COND": cond_metrics},
        {"min_semantic_similarity": [0.78, 0.77]},
    )

    assert [r.category for r in results] == ["COND", "MED"]
    by_category = {r.category: r for r in results}
    assert by_category["MED"].tuned_thresholds["min_semantic_similarity"] == 0.78
    assert by_category["COND"].tuned_thresholds["min_semantic_similarity"] == 0.77
    assert engine.thresholds["MED"]["min_semantic_similarity"] == 0.78
    assert engine.thresholds["COND"]["min_semantic_similarity"] == 0.77


def test_tune_deterministic(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """Repeated tuning from identical state produces identical summaries."""
    metrics = _candidate_metrics(0.80, 0.82, 0.90)
    grid = {"min_semantic_similarity": [0.85, 0.80]}
    first: ThresholdTuningResult = tuner.tune("MED", metrics, grid)
    engine.update_category_thresholds("MED", first.original_thresholds)
    second: ThresholdTuningResult = tuner.tune("MED", metrics, grid)
    assert first.model_dump() == second.model_dump()


def test_tune_empty_candidate_list(tuner: ThresholdTuner, engine: CategoryThresholdEngine) -> None:
    """An empty candidate list raises ValueError without mutating state."""
    before = dict(engine.thresholds["MED"])
    with pytest.raises(ValueError, match="No candidate metrics"):
        tuner.tune("MED", [], {"min_semantic_similarity": [0.80, 0.85]})
    assert engine.thresholds["MED"] == before


def test_tune_empty_grid(tuner: ThresholdTuner) -> None:
    """An empty grid raises ValueError."""
    with pytest.raises(ValueError, match="at least one threshold key"):
        tuner.tune("MED", _candidate_metrics(0.80), {})
