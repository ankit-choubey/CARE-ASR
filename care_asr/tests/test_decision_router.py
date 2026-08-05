"""Unit tests for the Candidate Recovery DecisionRouter.

Covers:
- Recovery triggered by low ASR confidence and by high ASR entropy
- Non-recovery when all metrics are within category tolerance
- Boundary behavior at exact threshold values for MED, ANA, and PHI
- Unknown category error handling
- Invalid threshold configuration failing fast at engine construction
"""

from typing import Any

import pytest

from care_asr.config.settings import Settings
from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.utils.exceptions import ThresholdConfigurationError
from care_asr.validation.decision_router import DecisionRouter


@pytest.fixture
def router() -> DecisionRouter:
    engine = CategoryThresholdEngine()
    return DecisionRouter(engine)


def test_valid_recovery_case_low_confidence(router: DecisionRouter) -> None:
    """Recovery triggers when ASR confidence is below the category minimum."""
    # MED requires min_asr_confidence=0.75
    assert router.should_trigger_recovery("MED", 0.50, 0.10) is True


def test_valid_recovery_case_high_entropy(router: DecisionRouter) -> None:
    """Recovery triggers when ASR entropy exceeds the category maximum."""
    # MED permits max_entropy=0.45
    assert router.should_trigger_recovery("MED", 0.90, 0.80) is True


def test_valid_non_recovery_case(router: DecisionRouter) -> None:
    """No recovery when confidence and entropy are both within category tolerance."""
    assert router.should_trigger_recovery("MED", 0.90, 0.10) is False


def test_med_boundary(router: DecisionRouter) -> None:
    """MED boundaries (min_asr_confidence=0.75, max_entropy=0.45)."""
    # Exact threshold values do not trigger recovery
    assert router.should_trigger_recovery("MED", 0.75, 0.45) is False
    # Just below min_asr_confidence triggers recovery
    assert router.should_trigger_recovery("MED", 0.75 - 1e-9, 0.45) is True
    # Just above max_entropy triggers recovery
    assert router.should_trigger_recovery("MED", 0.75, 0.45 + 1e-9) is True


def test_ana_boundary(router: DecisionRouter) -> None:
    """ANA boundaries (min_asr_confidence=0.65, max_entropy=0.65)."""
    assert router.should_trigger_recovery("ANA", 0.65, 0.65) is False
    assert router.should_trigger_recovery("ANA", 0.64, 0.65) is True
    assert router.should_trigger_recovery("ANA", 0.65, 0.66) is True


def test_phi_boundary(router: DecisionRouter) -> None:
    """PHI boundaries (min_asr_confidence=0.85, max_entropy=0.30)."""
    assert router.should_trigger_recovery("PHI", 0.85, 0.30) is False
    assert router.should_trigger_recovery("PHI", 0.84, 0.30) is True
    assert router.should_trigger_recovery("PHI", 0.85, 0.31) is True


def test_unknown_category(router: DecisionRouter) -> None:
    """Unknown categories raise the shared ThresholdConfigurationError."""
    with pytest.raises(ThresholdConfigurationError, match="Unknown category: INVALID"):
        router.should_trigger_recovery("INVALID", 0.90, 0.10)


def test_invalid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Router construction fails fast when the thresholds block is malformed."""

    class MockSettings(Settings):
        def load_yaml_config(self) -> dict[str, Any]:
            return {"thresholds": {"MED": {}}}  # missing all required keys

    monkeypatch.setattr("care_asr.thresholds.threshold_engine.get_settings", MockSettings)

    with pytest.raises(ThresholdConfigurationError, match="missing required threshold"):
        DecisionRouter(CategoryThresholdEngine())
