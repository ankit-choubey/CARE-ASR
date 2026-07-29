"""
Unit tests for CARE-ASR Task T3 Tsallis entropy functions.

Covers:
- Highly confident vs uniform distribution entropy bounds
- Batch entropy calculation
- Empty input validation
- Invalid alpha validation
- Invalid probability validation
- Numerical stability
"""

import numpy as np
import pytest
import torch

from care_asr.uncertainty.tsallis_entropy import (
    compute_batch_entropy,
    compute_tsallis_entropy,
    softmax,
)


def test_1_highly_confident_distribution_low_entropy():
    """Verify that a highly confident distribution (e.g. 99% probability on one token) produces low entropy."""
    # 99.9% probability on token 0, negligible probability on others
    probs = np.array([0.999, 0.0005, 0.0005], dtype=np.float32)
    entropy = compute_tsallis_entropy(probs, alpha=1 / 3)

    assert isinstance(entropy, float)
    # Low entropy for confident distribution compared to uniform (0.238 vs 1.619)
    assert entropy < 0.3


def test_2_uniform_distribution_high_entropy():
    """Verify that a uniform distribution (equal uncertainty across all options) produces high entropy."""
    vocab_size = 100
    probs_uniform = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float32)
    entropy_uniform = compute_tsallis_entropy(probs_uniform, alpha=1 / 3)

    # Confident distribution for comparison
    probs_confident = np.zeros(vocab_size, dtype=np.float32)
    probs_confident[0] = 1.0
    entropy_confident = compute_tsallis_entropy(probs_confident, alpha=1 / 3)

    assert entropy_uniform > entropy_confident
    assert entropy_uniform > 2.0  # Significant entropy for uniform distribution over 100 tokens


def test_3_batch_entropy_calculation():
    """Verify batch entropy calculation across multiple decoder step logit tensors."""
    vocab_size = 500
    # Step 1: Confident logits
    logits_step1 = torch.zeros((1, vocab_size))
    logits_step1[0, 0] = 50.0

    # Step 2: Uncertain logits
    logits_step2 = torch.ones((1, vocab_size))

    scores_list = [logits_step1, logits_step2]
    batch_entropies = compute_batch_entropy(scores_list, alpha=1 / 3)

    assert isinstance(batch_entropies, torch.Tensor)
    assert len(batch_entropies) == 2
    assert batch_entropies[0].item() < batch_entropies[1].item()


def test_4_empty_input_validation():
    """Verify that empty inputs raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        softmax(torch.tensor([]))
    assert "empty" in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(np.array([]))
    assert "empty" in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        compute_batch_entropy([])
    assert "empty" in str(exc_info.value).lower()


def test_5_invalid_alpha_validation():
    """Verify that invalid alpha values (alpha <= 0 or alpha == 1.0) raise ValueError."""
    valid_probs = np.array([0.7, 0.3])

    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(valid_probs, alpha=0.0)
    assert "positive" in str(exc_info.value).lower() or "alpha" in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(valid_probs, alpha=-0.5)
    assert "positive" in str(exc_info.value).lower() or "alpha" in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(valid_probs, alpha=1.0)
    assert "1.0" in str(exc_info.value)


def test_6_invalid_probabilities_validation():
    """Verify that negative probabilities or distributions not summing to ~1.0 raise ValueError."""
    invalid_neg = np.array([-0.1, 1.1])
    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(invalid_neg)
    assert "negative" in str(exc_info.value).lower()

    invalid_sum = np.array([0.5, 0.9])
    with pytest.raises(ValueError) as exc_info:
        compute_tsallis_entropy(invalid_sum)
    assert "sum" in str(exc_info.value).lower()


def test_7_numerical_stability():
    """Verify numerical stability on extreme logits and tiny probability values without NaN/Inf."""
    extreme_logits = torch.tensor([[-1000.0, 1000.0, -500.0]])
    probs = softmax(extreme_logits)

    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()
    assert abs(torch.sum(probs).item() - 1.0) < 1e-4

    entropy = compute_tsallis_entropy(probs)
    assert not np.isnan(entropy)
    assert not np.isinf(entropy)
