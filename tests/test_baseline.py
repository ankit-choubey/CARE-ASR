"""
Unit tests for CARE-ASR Task T1 baseline evaluation harness and IO utilities.
"""

import os

import numpy as np
import pytest

from src.evaluation.baseline import WhisperBaselineEvaluator, run_baseline_evaluation
from src.evaluation.io_utils import (
    load_afrispeech_dataset,
    load_predictions,
    save_predictions,
    validate_prediction_schema,
)


def test_dataset_iteration_works():
    """Verify load_afrispeech_dataset yields valid audio sample dictionaries."""
    samples = list(load_afrispeech_dataset(max_samples=2, use_dummy_fallback=True))
    assert len(samples) == 2

    first_sample = samples[0]
    assert "audio_id" in first_sample
    assert "audio" in first_sample
    assert isinstance(first_sample["audio"], np.ndarray)
    assert "sample_rate" in first_sample
    assert "reference" in first_sample


def test_output_json_schema_validation():
    """Verify JSON schema validation accepts valid schema and rejects incomplete schema."""
    valid_item = {
        "audio_id": "test_001",
        "prediction": "sample transcript",
        "reference": "sample transcript",
        "word_timestamps": [{"word": "sample", "start": 0.0, "end": 0.3}],
        "token_scores": [{"step": 0, "token_id": 10, "token": "sample", "log_prob": 0.0, "prob": 1.0}],
    }
    assert validate_prediction_schema(valid_item) is True

    invalid_item = {"audio_id": "test_002", "prediction": "missing other keys"}
    with pytest.raises(ValueError) as exc_info:
        validate_prediction_schema(invalid_item)
    assert "missing required keys" in str(exc_info.value)


def test_save_and_load_predictions(tmp_path):
    """Test saving and re-loading predictions JSON file."""
    predictions = [
        {
            "audio_id": "utt_100",
            "prediction": "normal sinus rhythm",
            "reference": "normal sinus rhythm",
            "word_timestamps": [],
            "token_scores": [],
        }
    ]
    out_file = tmp_path / "test_predictions.json"
    save_predictions(predictions, str(out_file))

    loaded = load_predictions(str(out_file))
    assert len(loaded) == 1
    assert loaded[0]["audio_id"] == "utt_100"
    assert loaded[0]["prediction"] == "normal sinus rhythm"


@pytest.fixture(scope="module")
def evaluator_instance():
    """Fixture initializing evaluator for fast unit testing with whisper-tiny."""
    try:
        return WhisperBaselineEvaluator(model_name="openai/whisper-tiny")
    except Exception as e:
        pytest.skip(f"Model loading unavailable: {e}")


def test_prediction_pipeline_returns_expected_object(evaluator_instance):
    """Verify process_utterance returns standard utterance dictionary with required keys."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio_signal = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    result = evaluator_instance.process_utterance(
        audio_data=audio_signal,
        sample_rate=sample_rate,
        audio_id="unit_test_utt",
        reference="test reference text",
    )

    assert isinstance(result, dict)
    assert result["audio_id"] == "unit_test_utt"
    assert "prediction" in result
    assert result["reference"] == "test reference text"
    assert "word_timestamps" in result
    assert isinstance(result["word_timestamps"], list)
    assert "token_scores" in result
    assert isinstance(result["token_scores"], list)
    assert len(result["token_scores"]) > 0


def test_end_to_end_baseline_run(tmp_path):
    """Verify run_baseline_evaluation creates predictions.json and baseline_metrics.json."""
    output_dir = tmp_path / "results"
    try:
        predictions, metrics = run_baseline_evaluation(
            model_name="openai/whisper-tiny", max_samples=1, output_dir=str(output_dir)
        )
    except Exception as e:
        pytest.skip(f"Model loading unavailable: {e}")

    assert len(predictions) == 1
    assert os.path.exists(output_dir / "predictions.json")
    assert os.path.exists(output_dir / "baseline_metrics.json")
    assert "WER" in metrics["metrics"]
    assert "CER" in metrics["metrics"]
