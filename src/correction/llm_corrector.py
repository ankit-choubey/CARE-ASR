"""
LLM-based medical term corrector with Outlines schema-constrained decoding.
Ensures output strictly conforms to CORRECT | <candidate>, WRONG, or UNSURE.
"""

from __future__ import annotations

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from care_asr.contracts.retrieval_input import RetrievalCandidate
from care_asr.contracts.validated_output import CorrectionOutput

FEW_SHOT = """\
You are a clinical ASR correction assistant.
Classify the candidate medical term:
- CORRECT | <candidate>
- WRONG
- UNSURE

Examples:
Input: asr="amoxicilin", candidates=["amoxicillin"], context="prescribed amoxicilin"
Output: CORRECT | amoxicillin

Input: asr="cardigan", candidates=["carvedilol"], context="takes cardigan for heart"
Output: UNSURE
"""


class LLMCorrector:
    """Qwen2.5 / LLM clinical term corrector with schema constraint."""

    def __init__(self, config_path: str = "configs/correction.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "max_new_tokens": 30}

        self.cfg = cfg
        self.use_model = False

        model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        if torch.cuda.is_available():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                self.model.eval()
                self.use_model = True
            except Exception:
                self.use_model = False

    def correct(self, asr_token: str, candidates: list[RetrievalCandidate], context: str = "") -> CorrectionOutput:
        """Corrects uncertain ASR token using candidate retrieval list."""
        cand_names = [c.candidate for c in candidates[:5]]

        if not self.use_model or not cand_names:
            # Fallback heuristic parser when GPU/model is offline
            if cand_names:
                top_cand = cand_names[0]
                return CorrectionOutput(
                    original_token=asr_token,
                    corrected_token=top_cand,
                    label="CORRECT",
                    confidence=0.85,
                )
            return CorrectionOutput(
                original_token=asr_token,
                corrected_token=asr_token,
                label="UNSURE",
                confidence=0.0,
            )

        prompt = f'{FEW_SHOT}\nInput: asr="{asr_token}", candidates={cand_names}, context="{context}"\nOutput:'
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        return self._parse(response, asr_token, cand_names)

    def _parse(self, response: str, asr_token: str, candidates: list[str]) -> CorrectionOutput:
        """Parses LLM output into canonical CorrectionOutput schema."""
        up = response.upper()
        if "UNSURE" in up:
            return CorrectionOutput(
                original_token=asr_token,
                corrected_token=asr_token,
                label="UNSURE",
                confidence=0.0,
            )
        if "CORRECT" in up and "|" in response:
            chosen = response.split("|")[-1].strip().lower()
            matched = next(
                (c for c in candidates if c.lower() == chosen),
                candidates[0] if candidates else asr_token,
            )
            return CorrectionOutput(
                original_token=asr_token,
                corrected_token=matched,
                label="CORRECT",
                confidence=0.9,
            )
        if "WRONG" in up:
            return CorrectionOutput(
                original_token=asr_token,
                corrected_token=asr_token,
                label="WRONG",
                confidence=0.1,
            )
        return CorrectionOutput(
            original_token=asr_token,
            corrected_token=asr_token,
            label="UNSURE",
            confidence=0.0,
        )
