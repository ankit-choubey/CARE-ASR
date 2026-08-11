"""
Tsallis entropy calculation.
"""
import torch

def tsallis_entropy(probs: torch.Tensor, alpha: float = 1/3) -> float:
    """Computes Tsallis non-extensive entropy for a probability distribution.
    
    Formula: H_q = (1 - sum(p^alpha)) / (alpha - 1)
    """
    if probs.dim() > 1:
        probs = probs.squeeze()
        
    return float((1.0 - (probs ** alpha).sum().item()) / (alpha - 1.0))
