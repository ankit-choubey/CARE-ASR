"""Unit tests for Tsallis entropy gate (T3)."""

import torch
from care_asr.uncertainty.tsallis_entropy import compute_tsallis_entropy
from care_asr.uncertainty.gate import gate_tokens, TsallisUncertaintyGate


def test_high_confidence_low_entropy():
    """High-confidence distribution should have low entropy."""
    p = torch.zeros(100)
    p[0] = 1.0
    ent = compute_tsallis_entropy(p, alpha=1/3)
    assert ent < 1.0


def test_uniform_high_entropy():
    """Uniform distribution should have high entropy."""
    p = torch.ones(100) / 100
    ent = compute_tsallis_entropy(p, alpha=1/3)
    assert ent > 1.0


def test_gate_tokens_output_shape():
    """gate_tokens should return expected dict keys."""
    scores = [torch.softmax(torch.randn(100), dim=0) for _ in range(3)]
    result = gate_tokens(scores, threshold=0.5, alpha=1/3)
    assert "entropies" in result
    assert "uncertain_flags" in result
    assert "uncertain_indices" in result
    assert len(result["entropies"]) == 3


def test_gate_class_evaluate():
    """TsallisUncertaintyGate.evaluate should work."""
    gate = TsallisUncertaintyGate(threshold=0.5, alpha=1/3)
    scores = [torch.softmax(torch.randn(100), dim=0)]
    result = gate.evaluate(scores)
    assert "overall_uncertain" in result
