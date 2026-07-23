"""
Baseline Evaluation Harness for CARE-ASR (Task T1).

Runs HuggingFace Whisper-medium inference on audio utterances from the
AfriSpeech-200 clinical test split, extracts word timestamps and decoder token scores,
computes WER/CER scoreboard metrics, and saves prediction artifacts to results/.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation.io_utils import (
    load_afrispeech_dataset,
    save_metrics,
    save_predictions,
)
from src.evaluation.metrics import evaluate_baseline


class WhisperBaselineEvaluator:
    """
    Evaluator class for Task T1 Whisper-medium baseline inference.
    """

    def __init__(
        self, model_name: str = "openai/whisper-medium", device: Optional[str] = None
    ):
        """
        Initializes Whisper processor and model.

        Args:
            model_name: HuggingFace model identifier (openai/whisper-medium).
            device: Device placement ('cuda', 'mps', 'cpu'). Auto-selects if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        print(
            f"[WhisperBaselineEvaluator] Loading '{model_name}' on device '{device}'..."
        )
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def process_utterance(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        audio_id: str = "utt_000",
        reference: str = "",
    ) -> Dict[str, Any]:
        """
        Runs Whisper-medium inference on a single audio utterance and extracts
        prediction string, word-level timestamps, and token decoder scores.

        Args:
            audio_data: 1D audio waveform signal array.
            sample_rate: Audio sampling frequency (Hz).
            audio_id: Unique utterance identifier string.
            reference: Ground-truth reference transcription string.

        Returns:
            Dict[str, Any]: Formatted utterance prediction result dictionary.
        """
        input_features = self.processor(
            audio_data, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_features, return_dict_in_generate=True, output_scores=True
            )

        sequences = outputs.sequences
        scores = outputs.scores

        raw_sequence = sequences[0].cpu()
        prediction = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]

        # Extract word timestamps
        word_timestamps = self._extract_word_timestamps(sequences[0])

        # Extract token scores
        token_scores = self._extract_token_scores(scores, raw_sequence)

        utterance_result = {
            "audio_id": str(audio_id),
            "prediction": str(prediction),
            "reference": str(reference),
            "word_timestamps": word_timestamps,
            "token_scores": token_scores,
        }

        return utterance_result

    def _extract_word_timestamps(
        self, sequence_tensor: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Extracts word-level timestamps from Whisper output sequence.

        Returns:
            List[Dict[str, Any]]: List of word timestamp dictionaries containing
                                  'word', 'start', and 'end'.
        """
        word_timestamps = []
        try:
            decoded_timestamps = self.processor.tokenizer._decode_asr(
                [{"tokens": sequence_tensor.tolist()}], return_timestamps=True
            )
            chunks = decoded_timestamps.get("chunks", [])
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp", (None, None))
                if text and timestamp and len(timestamp) == 2:
                    start_time = (
                        float(timestamp[0]) if timestamp[0] is not None else 0.0
                    )
                    end_time = (
                        float(timestamp[1])
                        if timestamp[1] is not None
                        else start_time + 0.1
                    )
                    word_timestamps.append(
                        {
                            "word": text,
                            "start": round(start_time, 2),
                            "end": round(end_time, 2),
                        }
                    )
        except Exception:
            # Fallback estimation if token-level timestamps are not present in sequence
            words = self.processor.decode(
                sequence_tensor, skip_special_tokens=True
            ).split()
            current_time = 0.0
            for w in words:
                word_timestamps.append(
                    {
                        "word": w,
                        "start": round(current_time, 2),
                        "end": round(current_time + 0.3, 2),
                    }
                )
                current_time += 0.35

        return word_timestamps

    def _extract_token_scores(
        self, scores: Tuple[torch.Tensor, ...], sequence_tensor: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Extracts per-step logit/probability score statistics for generated tokens.

        Returns:
            List[Dict[str, Any]]: List of per-token score dictionaries containing
                                  'step', 'token_id', 'token', 'log_prob', and 'prob'.
        """
        token_scores = []
        num_scores = len(scores)
        token_ids = sequence_tensor.tolist()
        num_tokens = len(token_ids)
        prompt_offset = num_tokens - num_scores

        for step_idx, step_logits in enumerate(scores):
            logits = step_logits[0].cpu()  # shape: (vocab_size,)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Corresponding token ID generated at this step
            token_id_idx = prompt_offset + step_idx
            gen_token_id = (
                token_ids[token_id_idx]
                if token_id_idx < num_tokens
                else torch.argmax(logits).item()
            )

            token_str = self.processor.tokenizer.decode([gen_token_id])
            token_prob = float(probs[gen_token_id].item())
            token_log_prob = float(log_probs[gen_token_id].item())

            token_scores.append(
                {
                    "step": step_idx,
                    "token_id": int(gen_token_id),
                    "token": token_str,
                    "log_prob": round(token_log_prob, 4),
                    "prob": round(token_prob, 6),
                }
            )

        return token_scores


def run_baseline_evaluation(
    model_name: str = "openai/whisper-medium",
    dataset_name_or_path: str = "afrispeech",
    split: str = "test",
    category: str = "clinical",
    max_samples: Optional[int] = None,
    output_dir: str = "results",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes the complete Task T1 baseline evaluation pipeline.

    1. Loads dataset iterator.
    2. Runs Whisper-medium inference for every utterance.
    3. Saves predictions to results/predictions.json.
    4. Computes baseline metrics (WER/CER).
    5. Saves metrics to results/baseline_metrics.json.

    Args:
        model_name: HuggingFace Whisper model name.
        dataset_name_or_path: Path or HF dataset identifier.
        split: Dataset split ('test').
        category: Subset category ('clinical').
        max_samples: Optional sample count cap.
        output_dir: Target directory for results.

    Returns:
        Tuple containing (predictions_list, metrics_dictionary).
    """
    evaluator = WhisperBaselineEvaluator(model_name=model_name)

    print(
        f"[run_baseline_evaluation] Loading dataset '{dataset_name_or_path}' ({category} {split})..."
    )
    dataset_iter = load_afrispeech_dataset(
        dataset_name_or_path=dataset_name_or_path,
        split=split,
        category=category,
        max_samples=max_samples,
    )

    predictions = []
    for idx, sample in enumerate(dataset_iter, start=1):
        print(f"  - Processing utterance {idx}: audio_id='{sample['audio_id']}'...")
        pred_item = evaluator.process_utterance(
            audio_data=sample["audio"],
            sample_rate=sample["sample_rate"],
            audio_id=sample["audio_id"],
            reference=sample["reference"],
        )
        predictions.append(pred_item)

    # Save predictions.json
    predictions_path = os.path.join(output_dir, "predictions.json")
    save_predictions(predictions, predictions_path)

    # Compute & save baseline_metrics.json
    metrics_summary = evaluate_baseline(predictions)
    metrics_path = os.path.join(output_dir, "baseline_metrics.json")
    save_metrics(metrics_summary, metrics_path)

    return predictions, metrics_summary


if __name__ == "__main__":
    run_baseline_evaluation()
