"""
Whisper ASR wrapper — audio array -> Transcript with token scores.
S3 / T1 module. Runs on GPU/CPU for audio evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from care_asr.contracts.asr_input import TokenScore, Transcript
from src.entropy.tsallis import tsallis_entropy


def _pick_device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperTranscriber:
    """Wrapper class for HuggingFace Whisper inference and token score extraction."""

    def __init__(self, config_path: str = "configs/asr.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {
                "model_name": "openai/whisper-medium",
                "device": "auto",
                "language": "en",
                "max_new_tokens": 448,
            }

        self.device = _pick_device(cfg.get("device", "auto"))
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_name = cfg.get("model_name", "openai/whisper-medium")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.cfg = cfg

    def transcribe(self, audio_array: Any, sample_rate: int = 16_000) -> Transcript:
        """Transcribes audio array into Transcript object with token log probabilities."""
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            result = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                language=self.cfg.get("language", "en"),
                max_new_tokens=self.cfg.get("max_new_tokens", 440),
            )

        sequences = result.sequences[0]
        text = self.processor.decode(sequences, skip_special_tokens=True)

        token_scores: list[TokenScore] = []
        for step, score_tensor in enumerate(result.scores):
            if step >= len(sequences) - 1:
                break
            token_id = int(sequences[step + 1].item())
            log_probs = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            probs = torch.exp(log_probs)
            
            log_p = float(log_probs[token_id].item())
            entropy_val = tsallis_entropy(probs)
            
            token_scores.append(
                TokenScore(
                    step=step,
                    token_id=token_id,
                    token=self.processor.decode([token_id]),
                    log_prob=log_p,
                    prob=float(probs[token_id].item()),
                    entropy=entropy_val,
                )
            )

        return Transcript(text=text, token_scores=token_scores, word_timestamps=[])
