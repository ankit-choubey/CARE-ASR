"""
Metrics calculation module for CARE-ASR evaluation harness.

Computes baseline WER (Word Error Rate) and CER (Character Error Rate).
Reserves placeholder interfaces for M-WER (Medical WER) and per-category Recall
to be implemented in Task T4 when entity spans become available.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import jiwer


def normalize_text(text: str) -> str:
    """
    Standard text normalization for WER/CER evaluation.
    Lowercases text, strips leading/trailing whitespace, and removes redundant spaces.

    Args:
        text: Input string.

    Returns:
        str: Normalized string.
    """
    if not text:
        return ""
    # Standard lowercase and space collapse
    normalized = " ".join(text.lower().strip().split())
    return normalized


def compute_wer(
    predictions: Union[List[Dict[str, Any]], List[str]],
    references: Optional[List[str]] = None,
) -> float:
    """
    Computes Word Error Rate (WER) using jiwer across prediction and reference pairs.

    Args:
        predictions: List of prediction dictionaries (containing 'prediction' and 'reference')
                     or list of prediction strings.
        references: Optional list of reference strings if predictions is a list of strings.

    Returns:
        float: Calculated WER value (0.0 to 1.0+).
    """
    preds_text, refs_text = _extract_texts(predictions, references)
    if not preds_text or not refs_text:
        return 0.0

    # Normalize text strings
    norm_preds = [normalize_text(p) for p in preds_text]
    norm_refs = [normalize_text(r) for r in refs_text]

    return float(jiwer.wer(reference=norm_refs, hypothesis=norm_preds))


def compute_cer(
    predictions: Union[List[Dict[str, Any]], List[str]],
    references: Optional[List[str]] = None,
) -> float:
    """
    Computes Character Error Rate (CER) using jiwer across prediction and reference pairs.

    Args:
        predictions: List of prediction dictionaries (containing 'prediction' and 'reference')
                     or list of prediction strings.
        references: Optional list of reference strings if predictions is a list of strings.

    Returns:
        float: Calculated CER value (0.0 to 1.0+).
    """
    preds_text, refs_text = _extract_texts(predictions, references)
    if not preds_text or not refs_text:
        return 0.0

    norm_preds = [normalize_text(p) for p in preds_text]
    norm_refs = [normalize_text(r) for r in refs_text]

    return float(jiwer.cer(reference=norm_refs, hypothesis=norm_preds))


def compute_mwer(
    predictions: List[Dict[str, Any]], entity_spans: Optional[List[Any]] = None
) -> float:
    """
    Placeholder interface for Medical-WER (M-WER).

    M-WER evaluates error rates specifically on clinical entity spans (e.g. dosages,
    diagnoses, anatomy, medications). M-WER cannot be computed in Task T1 because entity
    spans are produced during Task T4.

    Args:
        predictions: Utterance predictions list.
        entity_spans: Clinical entity span annotations.

    Raises:
        NotImplementedError: Always raised in T1 to preserve interface contract for T4.
    """
    raise NotImplementedError(
        "M-WER requires clinical entity spans produced by Task T4. "
        "This interface is reserved for T4 implementation."
    )


def compute_category_recall(
    predictions: List[Dict[str, Any]], ground_truth_entities: Optional[List[Any]] = None
) -> Dict[str, float]:
    """
    Placeholder interface for per-category medical entity Recall.

    Category Recall measures extraction accuracy across clinical categories (e.g.,
    Medication, Anatomy, Symptom, Procedure). Cannot be computed in Task T1.

    Args:
        predictions: Utterance predictions list.
        ground_truth_entities: Ground truth medical entity lists by category.

    Raises:
        NotImplementedError: Always raised in T1 to preserve interface contract for T4.
    """
    raise NotImplementedError(
        "Per-category Recall requires medical entity span ground truth produced by Task T4. "
        "This interface is reserved for T4 implementation."
    )


def evaluate_baseline(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs baseline evaluation suite over all predictions and constructs
    the official baseline scoreboard summary dictionary.

    Args:
        predictions: List of utterance prediction dictionaries.

    Returns:
        Dict[str, Any]: Baseline metrics dictionary.
    """
    wer_score = compute_wer(predictions)
    cer_score = compute_cer(predictions)

    metrics_summary = {
        "dataset": "AfriSpeech-200 clinical test split",
        "num_samples": len(predictions),
        "metrics": {
            "WER": round(wer_score, 4),
            "CER": round(cer_score, 4),
            "M-WER": "RESERVED_FOR_T4 (NotImplementedError)",
            "category_recall": "RESERVED_FOR_T4 (NotImplementedError)",
        },
        "status": "T1 Baseline Evaluation Completed Successfully",
    }

    return metrics_summary


def _extract_texts(
    predictions: Union[List[Dict[str, Any]], List[str]],
    references: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Helper to parse prediction and reference texts from list of dicts or strings."""
    if (
        isinstance(predictions, list)
        and len(predictions) > 0
        and isinstance(predictions[0], dict)
    ):
        preds_text = [str(item.get("prediction", "")) for item in predictions]
        refs_text = [str(item.get("reference", "")) for item in predictions]
    else:
        preds_text = [str(p) for p in predictions]
        refs_text = [str(r) for r in (references or [])]
    return preds_text, refs_text
