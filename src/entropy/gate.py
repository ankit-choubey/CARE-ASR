"""
Tsallis Entropy Gate (M2 Module).
Flags tokens as uncertain if their Tsallis entropy exceeds a specified threshold.
"""

from __future__ import annotations

import yaml

from care_asr.contracts.asr_input import Transcript


class TsallisEntropyGate:
    """Entropy Gate using Tsallis non-extensive entropy.

    This replaces the max-probability baseline thresholding. Tokens with
    entropy greater than the threshold are flagged for downstream retrieval.
    """

    def __init__(self, config_path: str = "configs/entropy.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {"threshold": 1.5}

        self.threshold = float(cfg.get("threshold", 1.5))

    def __call__(self, transcript: Transcript) -> list[bool]:
        """Evaluates each token's entropy against the threshold.

        Returns:
            list[bool]: True if the token is uncertain (entropy > threshold).
        """
        return [ts.entropy > self.threshold for ts in transcript.token_scores]
