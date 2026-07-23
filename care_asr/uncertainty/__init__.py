"""
CARE-ASR Uncertainty Module: Tsallis Entropy Calculation and Uncertainty Gating.

Exposes Tsallis entropy functions and uncertainty gating logic to filter
unconfident model tokens for downstream clinical retrieval.
"""

from care_asr.uncertainty.gate import (
    TsallisUncertaintyGate,
    gate_tokens,
    is_uncertain,
)
from care_asr.uncertainty.tsallis_entropy import (
    compute_batch_entropy,
    compute_tsallis_entropy,
    softmax,
)

__all__ = [
    "softmax",
    "compute_tsallis_entropy",
    "compute_batch_entropy",
    "is_uncertain",
    "gate_tokens",
    "TsallisUncertaintyGate",
]
