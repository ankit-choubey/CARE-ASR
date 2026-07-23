"""
Unit tests for CARE-ASR Task T3 Tsallis uncertainty gate functions.

Covers:
- Threshold behavior and binary decision outputs
- Full sequence token gating and structured decision report
- TsallisUncertaintyGate class threshold setting and evaluation
"""

import numpy as np
import pytest
import torch

from care_asr.uncertainty.gate import (
    TsallisUncertaintyGate,
    gate_tokens,
    is_uncertain,
)


def test_is_uncertain_threshold_behavior():
    """Verify threshold evaluation function on scalar, tensor, and numpy values."""
    assert is_uncertain(0.8, threshold=0.5) is True
    assert is_uncertain(0.2, threshold=0.5) is False
    assert is_uncertain(0.5, threshold=0.5) is True

    entropies_tensor = torch.tensor([0.1, 0.4, 0.6, 0.9])
    flags_tensor = is_uncertain(entropies_tensor, threshold=0.5)
    assert flags_tensor.tolist() == [False, False, True, True]

    entropies_np = np.array([0.2, 0.7])
    flags_np = is_uncertain(entropies_np, threshold=0.5)
    assert flags_np.tolist() == [False, True]


def test_gate_tokens_report_structure():
    """Verify gate_tokens produces structured decision dictionary with required keys."""
    vocab_size = 100
    # Step 0: Confident token
    logits_step0 = torch.zeros((1, vocab_size))
    logits_step0[0, 0] = 30.0

    # Step 1: Uncertain token
    logits_step1 = torch.ones((1, vocab_size))

    token_scores = [logits_step0, logits_step1]
    report = gate_tokens(token_scores, threshold=0.5, alpha=1 / 3)

    assert isinstance(report, dict)
    assert "entropies" in report
    assert "uncertain_flags" in report
    assert "uncertain_indices" in report
    assert "threshold" in report
    assert "alpha" in report
    assert "overall_uncertain" in report

    assert len(report["entropies"]) == 2
    assert report["uncertain_flags"] == [False, True]
    assert report["uncertain_indices"] == [1]
    assert report["overall_uncertain"] is True


def test_tsallis_uncertainty_gate_class_and_dynamic_threshold():
    """Verify TsallisUncertaintyGate class methods and threshold tuning."""
    gate = TsallisUncertaintyGate(threshold=0.5, alpha=1 / 3)
    assert gate.threshold == 0.5

    # Update threshold
    gate.set_threshold(1.5)
    assert gate.threshold == 1.5

    # Test invalid threshold
    with pytest.raises(ValueError):
        gate.set_threshold(-0.1)

    # Evaluate sequence
    vocab_size = 50
    logits = torch.ones((1, vocab_size))
    result = gate.evaluate([logits])

    assert "entropies" in result
    assert result["threshold"] == 1.5
