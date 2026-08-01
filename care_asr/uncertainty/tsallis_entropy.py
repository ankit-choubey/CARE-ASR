"""
Tsallis Entropy Calculation Module for CARE-ASR (Task T3).

This module provides core mathematical functions to compute Tsallis entropy
from model logits or probability distributions.

--------------------------------------------------------------------------------
BEGINNER-FRIENDLY EXPLANATION:

1. WHAT IS ENTROPY?
   - Entropy measures "uncertainty" or "surprise" in a probability distribution.
   - High entropy means the model is unsure and spread its guess across many possible
     words (e.g. 5% chance for 'amoxicillin', 5% for 'ampicillin', 4% for 'aspirin').
   - Low entropy means the model is extremely confident in one word (e.g. 99% chance for 'the').

2. WHY CARE-ASR USES TSALLIS ENTROPY (alpha = 1/3):
   - Standard Shannon entropy treats all probabilities logarithmically.
   - Tsallis entropy H_alpha(P) = (1 / (alpha - 1)) * (1 - sum(P_i^alpha)) is a non-extensive
     generalization parameterized by alpha.
   - At alpha = 1/3, Tsallis entropy gives higher weight to rare acoustic/clinical terms in long-tail
     distributions, making it much more sensitive to subtle model hesitation on medical entities
     (like drug dosages or rare condition names).
--------------------------------------------------------------------------------
"""

import numpy as np
import torch


def softmax(logits: torch.Tensor | np.ndarray, dim: int = -1) -> torch.Tensor | np.ndarray:
    """
    Converts unnormalized model logits into a valid probability distribution.

    PURPOSE:
        Whisper outputs raw logit scores for every word in its vocabulary.
        To measure how certain the model is, we first transform logits into probabilities
        that lie between 0.0 and 1.0 and sum to exactly 1.0.

    INPUTS:
        logits (torch.Tensor or np.ndarray): Raw unnormalized logit values.
        dim (int): Dimension along which softmax is applied (default: -1).

    OUTPUTS:
        torch.Tensor or np.ndarray: Normalized probability values of the same shape.
    """
    if isinstance(logits, torch.Tensor):
        if logits.numel() == 0:
            raise ValueError("Cannot compute softmax on empty tensor.")
        # Subtract max logit for numerical stability to prevent overflow
        max_logits, _ = torch.max(logits, dim=dim, keepdim=True)
        exps = torch.exp(logits - max_logits)
        probs = exps / torch.sum(exps, dim=dim, keepdim=True)
        return probs
    elif isinstance(logits, np.ndarray):
        if logits.size == 0:
            raise ValueError("Cannot compute softmax on empty array.")
        max_logits = np.max(logits, axis=dim, keepdims=True)
        exps = np.exp(logits - max_logits)
        probs = exps / np.sum(exps, axis=dim, keepdims=True)
        return probs
    else:
        raise TypeError(
            f"Unsupported logits type: {type(logits)}. Expected torch.Tensor or np.ndarray."
        )


def _validate_probs(probs: torch.Tensor | np.ndarray) -> float:
    """Validates input probabilities and returns total sum."""
    if isinstance(probs, torch.Tensor):
        if probs.numel() == 0:
            raise ValueError("Cannot compute Tsallis entropy on empty probability tensor.")
        if torch.any(probs < 0.0):
            raise ValueError("Probability distribution contains negative values.")
        prob_sum = torch.sum(probs).item()
    elif isinstance(probs, np.ndarray):
        if probs.size == 0:
            raise ValueError("Cannot compute Tsallis entropy on empty probability array.")
        if np.any(probs < 0.0):
            raise ValueError("Probability distribution contains negative values.")
        prob_sum = float(np.sum(probs))
    else:
        raise TypeError(
            f"Unsupported probs type: {type(probs)}. Expected torch.Tensor or np.ndarray."
        )

    if abs(prob_sum - 1.0) > 1e-2:
        raise ValueError(f"Probabilities must sum to approximately 1.0 (got sum={prob_sum:.4f}).")
    return prob_sum


def compute_tsallis_entropy(
    probs: torch.Tensor | np.ndarray, alpha: float = 1 / 3, eps: float = 1e-12
) -> torch.Tensor | float:
    """
    Computes Tsallis non-extensive entropy H_alpha(P) for a probability distribution.

    PURPOSE:
        Calculates the exact uncertainty score for a token probability distribution.
        At alpha = 1/3, Tsallis entropy H_{1/3}(P) = 1.5 * (sum(P_i^{1/3}) - 1) detects
        model uncertainty even when top-1 probability appears moderately high.

    FORMULA:
        H_alpha(P) = (1 / (alpha - 1)) * (1 - sum_i(P_i^alpha))

    INPUTS:
        probs (torch.Tensor or np.ndarray): Probability distribution vector (must sum to ~1.0).
        alpha (float): Entropic parameter alpha (default: 1/3). Must be > 0 and != 1.0.
        eps (float): Numerical stability threshold for low probability values.

    OUTPUTS:
        float or torch.Tensor: Scalar Tsallis entropy value.
    """
    if alpha <= 0.0:
        raise ValueError(
            f"Alpha parameter must be strictly positive (alpha > 0). Got alpha={alpha}."
        )
    if abs(alpha - 1.0) < 1e-6:
        raise ValueError(
            "Alpha parameter cannot be exactly 1.0 for Tsallis entropy (use Shannon entropy limit)."
        )

    _validate_probs(probs)

    if isinstance(probs, torch.Tensor):
        safe_probs = torch.clamp(probs, min=eps)
        sum_p_alpha = torch.sum(torch.pow(safe_probs, alpha), dim=-1)
        entropy = (1.0 / (alpha - 1.0)) * (1.0 - sum_p_alpha)
        return entropy.item() if entropy.ndim == 0 else entropy
    else:
        safe_probs = np.clip(probs, a_min=eps, a_max=None)
        sum_p_alpha = np.sum(np.power(safe_probs, alpha), axis=-1)
        entropy = (1.0 / (alpha - 1.0)) * (1.0 - sum_p_alpha)
        return float(entropy) if np.ndim(entropy) == 0 else entropy


def compute_batch_entropy(
    scores: list[torch.Tensor] | torch.Tensor | np.ndarray, alpha: float = 1 / 3
) -> torch.Tensor | np.ndarray:
    """
    Computes Tsallis entropy across a sequence or batch of logit/probability tensors.

    PURPOSE:
        Processes multiple decoding steps (e.g. outputs.scores list from Whisper)
        and computes per-step entropy values in a single vectorized pass.

    INPUTS:
        scores (List[torch.Tensor], torch.Tensor, or np.ndarray):
            List of decoder step logit/prob tensors, or 2D/3D tensor of logits/probs.
        alpha (float): Entropic index (default: 1/3).

    OUTPUTS:
        torch.Tensor or np.ndarray: 1D array of per-step Tsallis entropy values.
    """
    if isinstance(scores, list):
        if len(scores) == 0:
            raise ValueError("Cannot compute batch entropy on an empty list of scores.")

        entropies = []
        for _step_idx, step_score in enumerate(scores):
            # Check if input is logits (max prob != 1.0 sum) or already normalized probs
            step_sum = (
                torch.sum(step_score).item()
                if isinstance(step_score, torch.Tensor)
                else np.sum(step_score)
            )
            prob_step = softmax(step_score) if abs(step_sum - 1.0) > 0.01 else step_score

            ent = compute_tsallis_entropy(prob_step, alpha=alpha)
            entropies.append(ent)

        return torch.tensor(entropies, dtype=torch.float32)

    elif isinstance(scores, (torch.Tensor, np.ndarray)):
        if (isinstance(scores, torch.Tensor) and scores.numel() == 0) or (
            isinstance(scores, np.ndarray) and scores.size == 0
        ):
            raise ValueError("Cannot compute batch entropy on empty array/tensor.")

        # Determine if logits need softmax
        prob_scores = (
            softmax(scores)
            if (
                (isinstance(scores, torch.Tensor) and abs(torch.sum(scores[0]).item() - 1.0) > 1e-2)
                or (isinstance(scores, np.ndarray) and abs(float(np.sum(scores[0])) - 1.0) > 1e-2)
            )
            else scores
        )

        if isinstance(prob_scores, torch.Tensor):
            safe_probs = torch.clamp(prob_scores, min=1e-12)
            sum_p_alpha = torch.sum(torch.pow(safe_probs, alpha), dim=-1)
            return (1.0 / (alpha - 1.0)) * (1.0 - sum_p_alpha)
        else:
            safe_probs = np.clip(prob_scores, a_min=1e-12, a_max=None)
            sum_p_alpha = np.sum(np.power(safe_probs, alpha), axis=-1)
            return (1.0 / (alpha - 1.0)) * (1.0 - sum_p_alpha)

    else:
        raise TypeError(f"Unsupported batch scores type: {type(scores)}.")
