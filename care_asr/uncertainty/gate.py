"""
Tsallis Uncertainty Gate Module for CARE-ASR (Task T3).

Provides decision gating functions to filter uncertain tokens based on
configurable entropy thresholds. Threshold configuration is kept separate
from entropy computation to support threshold tuning in Task T8.

--------------------------------------------------------------------------------
BEGINNER-FRIENDLY EXPLANATION:

1. HOW THE UNCERTAINTY GATE WORKS:
   - The gate acts as a traffic controller between Whisper-medium and Retrieval.
   - For every generated word, it looks at the Tsallis entropy computed by T3.
   - If entropy >= threshold:
       -> Mark token as UNCERTAIN (trigger clinical retrieval for this word/phrase).
   - If entropy < threshold:
       -> Mark token as CONFIDENT (pass Whisper output directly without retrieval).

2. WHY SEPARATE THE THRESHOLD?
   - Computing entropy is a fixed mathematical property of model predictions.
   - Deciding WHERE to draw the cutoff threshold is an empirical decision parameter.
   - Keeping the threshold configurable allows tuning optimal precision/recall in T8
     without rewriting core entropy logic.
--------------------------------------------------------------------------------
"""

from typing import Any

import numpy as np
import torch

from care_asr.uncertainty.tsallis_entropy import compute_batch_entropy


def is_uncertain(
    entropy: float | torch.Tensor | np.ndarray, threshold: float = 0.5
) -> bool | torch.Tensor | np.ndarray:
    """
    Evaluates whether entropy score(s) exceed the configurable decision threshold.

    PURPOSE:
        Converts continuous entropy values into a binary uncertainty decision
        (True = uncertain / trigger retrieval, False = confident / skip retrieval).

    INPUTS:
        entropy (float, torch.Tensor, or np.ndarray): Calculated Tsallis entropy value(s).
        threshold (float): Entropy cutoff threshold (default: 0.5).

    OUTPUTS:
        bool, torch.Tensor, or np.ndarray: Boolean flag or mask where True indicates uncertainty.
    """
    if isinstance(entropy, (int, float)):
        return float(entropy) >= float(threshold)
    elif isinstance(entropy, (torch.Tensor, np.ndarray)):
        return entropy >= float(threshold)
    else:
        raise TypeError(f"Unsupported entropy type: {type(entropy)}.")


def gate_tokens(
    token_scores: list[torch.Tensor] | torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    alpha: float = 1 / 3,
) -> dict[str, Any]:
    """
    High-level token gating function processing Whisper scores through the uncertainty gate.

    PURPOSE:
        Processes a full sequence of token scores, calculates Tsallis entropy for each step,
        applies threshold evaluation, and returns a detailed gating decision report.

    INPUTS:
        token_scores (List[torch.Tensor], torch.Tensor, or np.ndarray): Token logits or probabilities.
        threshold (float): Entropy decision threshold (default: 0.5).
        alpha (float): Tsallis entropic index (default: 1/3).

    OUTPUTS:
        Dict[str, Any]: Detailed decision summary dictionary containing:
            - "entropies": List of per-token float entropy values.
            - "uncertain_flags": List of boolean flags (True = uncertain).
            - "uncertain_indices": List of 0-based token step indices that exceeded threshold.
            - "threshold": Cutoff threshold used.
            - "alpha": Entropic index alpha used.
            - "overall_uncertain": True if any token step in sequence was marked uncertain.
    """
    batch_entropies = compute_batch_entropy(token_scores, alpha=alpha)

    if isinstance(batch_entropies, torch.Tensor):
        entropy_list = [float(e.item()) for e in batch_entropies]
    elif isinstance(batch_entropies, np.ndarray):
        entropy_list = [float(e) for e in batch_entropies]
    else:
        entropy_list = [float(batch_entropies)]

    flags = [e >= threshold for e in entropy_list]
    uncertain_indices = [idx for idx, flag in enumerate(flags) if flag]

    return {
        "entropies": entropy_list,
        "uncertain_flags": flags,
        "uncertain_indices": uncertain_indices,
        "threshold": float(threshold),
        "alpha": float(alpha),
        "overall_uncertain": len(uncertain_indices) > 0,
    }


class TsallisUncertaintyGate:
    """
    Configurable class instance encapsulating Tsallis uncertainty gating logic.

    Supports dynamic threshold tuning in Task T8 without modifying core entropy functions.
    """

    def __init__(self, threshold: float = 0.5, alpha: float = 1 / 3):
        """
        Initializes gate with threshold and entropic index.

        Args:
            threshold: Initial decision threshold (default: 0.5).
            alpha: Tsallis entropic parameter (default: 1/3).
        """
        self.threshold = threshold
        self.alpha = alpha

    def set_threshold(self, new_threshold: float) -> None:
        """Updates decision threshold for tuning in Task T8."""
        if new_threshold < 0:
            raise ValueError(f"Threshold must be non-negative. Got {new_threshold}.")
        self.threshold = new_threshold

    def evaluate(
        self, token_scores: list[torch.Tensor] | torch.Tensor | np.ndarray
    ) -> dict[str, Any]:
        """Runs gating evaluation using stored threshold and alpha settings."""
        return gate_tokens(token_scores, threshold=self.threshold, alpha=self.alpha)
