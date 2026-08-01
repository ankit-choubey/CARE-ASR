"""
End-to-end CARE-ASR pipeline orchestrator.
Connects M1 ASR -> M2 Entropy Gate -> M3 NER -> M4 Dual Retrieval -> M5 Fusion -> M6 LLM Corrector -> M7 Safety Gate.
"""

from __future__ import annotations

from typing import Any

from src.fusion.rrf import reciprocal_rank_fusion
from src.pipeline.stubs import (
    stub_corrector,
    stub_entropy_gate,
    stub_ner,
    stub_phonetic_retrieve,
    stub_semantic_retrieve,
    stub_transcriber,
)


class CARPipeline:
    """Master CARE-ASR pipeline orchestrating all 8 modular processing layers."""

    def __init__(self) -> None:
        self.transcriber = stub_transcriber
        self.entropy_gate = stub_entropy_gate
        self.ner = stub_ner
        self.semantic_retrieve = stub_semantic_retrieve
        self.phonetic_retrieve = stub_phonetic_retrieve
        self.corrector = stub_corrector
        self.safety_gate = None

    def run(self, audio_input: Any, attribution_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Runs end-to-end CARE-ASR correction pipeline across input audio."""
        if attribution_log is None:
            attribution_log = []

        transcript = self.transcriber(audio_input)
        attribution_log.append({"module": "M1_ASR", "text": transcript.text})

        uncertain_flags = self.entropy_gate(transcript)
        attribution_log.append({"module": "M2_ENTROPY", "uncertain_count": sum(uncertain_flags)})

        entities = self.ner(transcript)
        entity_words = {e.word.lower() for e in entities}
        attribution_log.append({"module": "M3_NER", "entity_count": len(entities)})

        words = transcript.text.split()
        corrected_words = list(words)

        for i, (word, is_uncertain) in enumerate(zip(words, uncertain_flags, strict=False)):
            if not (is_uncertain and word.lower() in entity_words):
                continue

            semantic_candidates = self.semantic_retrieve(word)
            phonetic_candidates = self.phonetic_retrieve(word)
            attribution_log.append(
                {
                    "module": "M4_RETRIEVAL",
                    "token": word,
                    "semantic_top1": (semantic_candidates[0].candidate if semantic_candidates else None),
                    "phonetic_top1": (phonetic_candidates[0].candidate if phonetic_candidates else None),
                }
            )

            fused = reciprocal_rank_fusion([semantic_candidates, phonetic_candidates])
            attribution_log.append(
                {
                    "module": "M5_FUSION",
                    "fused_top1": fused[0].candidate if fused else None,
                }
            )

            correction = self.corrector(word, fused)
            if self.safety_gate is not None:
                correction = self.safety_gate(correction)

            attribution_log.append(
                {
                    "module": "M6M7_CORRECT_GATE",
                    "label": correction.label,
                    "corrected": correction.corrected_token,
                }
            )

            if correction.label != "UNSURE":
                corrected_words[i] = correction.corrected_token

        return {
            "original": transcript.text,
            "corrected": " ".join(corrected_words),
            "attribution": attribution_log,
        }
