"""
Unit tests for CARE-ASR Task T1 metrics module.
"""

import pytest

from src.evaluation.metrics import (
    compute_category_recall,
    compute_cer,
    compute_mwer,
    compute_wer,
    evaluate_baseline,
    normalize_text,
)


def test_normalize_text():
    """Verify text normalization (lowercasing and trimming)."""
    assert normalize_text("  Acute Hypertension  ") == "acute hypertension"
    assert normalize_text("") == ""


def test_compute_wer_simple_synthetic():
    """Test WER calculation on simple synthetic examples."""
    ref = ["the patient presents with acute fever"]
    pred = ["the patient presents with high fever"]

    # 1 substitution ("acute" -> "high") out of 6 words = 1/6 ≈ 0.1667
    wer_val = compute_wer(predictions=pred, references=ref)
    assert 0.15 < wer_val < 0.20

    # Exact match = 0.0 WER
    assert compute_wer(predictions=ref, references=ref) == 0.0


def test_compute_cer_simple_synthetic():
    """Test CER calculation on simple synthetic examples."""
    ref = ["fever"]
    pred = ["fever"]
    assert compute_cer(predictions=pred, references=ref) == 0.0

    pred_sub = ["favor"]
    cer_val = compute_cer(predictions=pred_sub, references=ref)
    assert cer_val > 0.0


def test_placeholder_mwer_interface_exists():
    """Verify placeholder compute_mwer interface raises NotImplementedError."""
    dummy_predictions = [
        {
            "audio_id": "1",
            "prediction": "a",
            "reference": "a",
            "word_timestamps": [],
            "token_scores": [],
        }
    ]
    with pytest.raises(NotImplementedError) as exc_info:
        compute_mwer(predictions=dummy_predictions, entity_spans=None)
    assert "M-WER requires clinical entity spans produced by Task T4" in str(
        exc_info.value
    )


def test_placeholder_category_recall_interface_exists():
    """Verify placeholder compute_category_recall interface raises NotImplementedError."""
    dummy_predictions = [
        {
            "audio_id": "1",
            "prediction": "a",
            "reference": "a",
            "word_timestamps": [],
            "token_scores": [],
        }
    ]
    with pytest.raises(NotImplementedError) as exc_info:
        compute_category_recall(
            predictions=dummy_predictions, ground_truth_entities=None
        )
    assert "Per-category Recall requires medical entity span ground truth" in str(
        exc_info.value
    )


def test_evaluate_baseline():
    """Test evaluate_baseline dictionary output format."""
    predictions = [
        {
            "audio_id": "utt_1",
            "prediction": "patient has mild hypertension",
            "reference": "patient has acute hypertension",
            "word_timestamps": [],
            "token_scores": [],
        }
    ]
    summary = evaluate_baseline(predictions)
    assert "metrics" in summary
    assert "WER" in summary["metrics"]
    assert "CER" in summary["metrics"]
    assert summary["metrics"]["M-WER"] == "RESERVED_FOR_T4 (NotImplementedError)"
    assert (
        summary["metrics"]["category_recall"] == "RESERVED_FOR_T4 (NotImplementedError)"
    )
