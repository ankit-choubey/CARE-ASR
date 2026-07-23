"""
Input/Output utilities for CARE-ASR evaluation harness.

Handles loading audio/reference datasets (e.g., AfriSpeech-200 clinical test split),
saving prediction outputs, metrics, and enforcing JSON schema validity.
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional
import numpy as np


REQUIRED_PREDICTION_KEYS = {
    "audio_id",
    "prediction",
    "reference",
    "word_timestamps",
    "token_scores",
}


def validate_prediction_schema(pred_item: Dict[str, Any]) -> bool:
    """
    Validates that an utterance prediction dictionary contains all required keys
    as specified in Task T1 schema.

    Args:
        pred_item: Prediction dictionary for a single utterance.

    Returns:
        bool: True if schema is valid, raises ValueError otherwise.
    """
    missing_keys = REQUIRED_PREDICTION_KEYS - set(pred_item.keys())
    if missing_keys:
        raise ValueError(
            f"Prediction dictionary is missing required keys: {sorted(list(missing_keys))}"
        )
    return True


def load_afrispeech_dataset(
    dataset_name_or_path: str = "afrispeech",
    split: str = "test",
    category: str = "clinical",
    max_samples: Optional[int] = None,
    use_dummy_fallback: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Loads and yields audio samples from AfriSpeech-200 clinical test split
    or fallback test dataset.

    Each yielded dictionary contains:
        - audio_id: Unique identifier string for the utterance.
        - audio: 1D numpy float array of audio samples.
        - sample_rate: Audio sampling frequency in Hz (typically 16000).
        - reference: Ground-truth reference transcription string.

    Args:
        dataset_name_or_path: HuggingFace dataset name or local path.
        split: Dataset split ('test', 'validation', 'train').
        category: Subset filter ('clinical', 'general', etc.).
        max_samples: Optional limit on number of samples to yield.
        use_dummy_fallback: If True, falls back gracefully to dummy samples if network/local files unavailable.

    Yields:
        Dict[str, Any]: Utterance dictionary.
    """
    count = 0

    # Try loading from local path if directory exists
    if os.path.exists(dataset_name_or_path):
        import soundfile as sf

        transcripts_file = os.path.join(
            dataset_name_or_path, f"{split}_transcripts.json"
        )

        if os.path.exists(transcripts_file):
            with open(transcripts_file, "r", encoding="utf-8") as f:
                transcripts = json.load(f)

            for item in transcripts:
                if max_samples is not None and count >= max_samples:
                    break
                audio_path = item.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    audio_data, sr = sf.read(audio_path)
                    yield {
                        "audio_id": item.get("audio_id", f"sample_{count:04d}"),
                        "audio": np.array(audio_data, dtype=np.float32),
                        "sample_rate": sr,
                        "reference": item.get("reference", ""),
                    }
                    count += 1
            if count > 0:
                return

    # Try loading via HuggingFace datasets library
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name_or_path, name=category, split=split)
        for idx, sample in enumerate(ds):
            if max_samples is not None and count >= max_samples:
                break

            audio_info = sample.get("audio", {})
            audio_array = audio_info.get("array", np.zeros(16000, dtype=np.float32))
            sr = audio_info.get("sampling_rate", 16000)
            audio_id = sample.get(
                "audio_id", sample.get("id", f"afrispeech_{split}_{idx:04d}")
            )
            reference = sample.get(
                "transcript", sample.get("text", sample.get("reference", ""))
            )

            yield {
                "audio_id": str(audio_id),
                "audio": np.array(audio_array, dtype=np.float32),
                "sample_rate": sr,
                "reference": str(reference),
            }
            count += 1
        if count > 0:
            return
    except Exception as e:
        print(f"[load_afrispeech_dataset] HuggingFace dataset load note: {e}")

    # Fallback synthetic dataset generator for unit testing / offline execution
    if use_dummy_fallback:
        dummy_data = [
            (
                "clinical_utt_001",
                "the patient presents with acute hypertension and elevated fever",
            ),
            (
                "clinical_utt_002",
                "prescribed amoxicillin five hundred milligrams orally twice daily",
            ),
            (
                "clinical_utt_003",
                "electrocardiogram reveals normal sinus rhythm with no ST elevation",
            ),
        ]
        sample_rate = 16000
        duration_sec = 2.5
        t = np.linspace(
            0, duration_sec, int(sample_rate * duration_sec), endpoint=False
        )

        for idx, (utt_id, ref_text) in enumerate(dummy_data):
            if max_samples is not None and count >= max_samples:
                break
            # Generate a distinct multi-tone audio signal
            freq = 440.0 + idx * 100.0
            audio_signal = 0.4 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            yield {
                "audio_id": utt_id,
                "audio": audio_signal,
                "sample_rate": sample_rate,
                "reference": ref_text,
            }
            count += 1


def save_predictions(predictions: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves predictions list to JSON file after schema validation.

    Args:
        predictions: List of utterance prediction dictionaries.
        output_path: Target JSON file path.
    """
    for pred in predictions:
        validate_prediction_schema(pred)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(
        f"[save_predictions] Successfully saved {len(predictions)} predictions to '{output_path}'."
    )


def load_predictions(input_path: str) -> List[Dict[str, Any]]:
    """
    Loads predictions list from a JSON file and validates schema.

    Args:
        input_path: Path to predictions JSON file.

    Returns:
        List[Dict[str, Any]]: Loaded prediction list.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    for pred in predictions:
        validate_prediction_schema(pred)
    return predictions


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Saves metrics summary dictionary to JSON file.

    Args:
        metrics: Dictionary containing baseline evaluation metrics.
        output_path: Target JSON file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[save_metrics] Successfully saved evaluation metrics to '{output_path}'.")
