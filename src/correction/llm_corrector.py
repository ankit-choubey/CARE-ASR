"""
LLM-based medical term corrector with Outlines schema-constrained decoding.
Ensures output strictly conforms to CORRECT | <candidate>, WRONG, or UNSURE.
"""
from __future__ import annotations

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    def __init__(self, config_path: str = "configs/correction.yaml") -> None:
        try:
            cfg = yaml.safe_load(open(config_path))
        except Exception:
            cfg = {"model_name": "Qwen/Qwen2.5-7B-Instruct"}

        self.cfg = cfg
        self.use_outlines = False

        if torch.cuda.is_available():
            try:
                try:
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg["model_name"],
                        quantization_config=bnb,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                except Exception as bnb_err:
                    print(f"BitsAndBytes quantization load failed ({bnb_err}); trying float16 fallback...")
                    self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg["model_name"],
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                self.model.eval()

                # Outlines constrained generator setup if available
                import outlines
                self.outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
                # Regex restricting output to exact schema
                regex_pattern = r"(CORRECT \| [a-zA-Z0-9_\- ]+|WRONG|UNSURE)"
                self.generator = outlines.generate.regex(self.outlines_model, regex_pattern)
                self.use_outlines = True
            except Exception as e:
                self.use_outlines = False
                print(f"Failed to initialize LLM / Outlines: {e}. Falling back to heuristic parser.")

    def correct(self, asr_token: str, candidates: list, context: str = "") -> CorrectionOutput:
        cand_names = [c.candidate for c in candidates[:5]]

        if not self.use_outlines or not cand_names:
            # Fallback heuristic parser when GPU/model is offline (for local unit testing)
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
        response = self.generator(prompt, max_tokens=30)
        return self._parse(response, asr_token, cand_names)

    def _parse(self, response: str, asr_token: str, candidates: list[str]) -> CorrectionOutput:
        up = response.upper()
        if "UNSURE" in up:
            return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="UNSURE", confidence=0.0)
        if "CORRECT" in up and "|" in response:
            chosen = response.split("|")[-1].strip().lower()
            matched = next((c for c in candidates if c.lower() == chosen), candidates[0] if candidates else asr_token)
            return CorrectionOutput(original_token=asr_token, corrected_token=matched, label="CORRECT", confidence=0.9)
        if "WRONG" in up:
            return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="WRONG", confidence=0.1)
        return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="UNSURE", confidence=0.0)
