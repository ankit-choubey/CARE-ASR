"""
UNSURE Safety Gate — Refusal fallback mechanism.
If correction output label is UNSURE or confidence < threshold, keeps original token.
"""

from __future__ import annotations

import yaml

from care_asr.contracts.validated_output import CorrectionOutput


class UnsureGate:
    """Refusal safety gate ensuring unconfident corrections fall back to original token."""

    def __init__(self, config_path: str = "configs/safety.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {"unsure_threshold": 0.5, "fallback_policy": "original_token"}

        self.threshold = float(cfg.get("unsure_threshold", 0.5))

    def apply(self, correction: CorrectionOutput) -> CorrectionOutput:
        """Evaluates single correction output and returns fallback token if UNSURE or low confidence."""
        if correction.label == "UNSURE" or correction.confidence < self.threshold:
            return CorrectionOutput(
                original_token=correction.original_token,
                corrected_token=correction.original_token,  # Fallback to original token
                label="UNSURE",
                confidence=correction.confidence,
            )
        return correction

    def batch_apply(self, corrections: list[CorrectionOutput]) -> list[CorrectionOutput]:
        """Applies safety gate across a list of correction outputs."""
        return [self.apply(c) for c in corrections]
