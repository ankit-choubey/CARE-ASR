"""
Medical WER (M-WER) calculation module.
Computes Word Error Rate exclusively on clinical entity spans extracted by BioBERT NER.
"""

from __future__ import annotations

import json
from pathlib import Path

import jiwer


def compute_mwer(
    predictions: list[dict[str, str]],
    ner_spans_path: str = "outputs/ner/ner_reference_spans.json",
) -> float:
    """
    Computes Medical Word Error Rate (M-WER) restricted to entity spans.

    Returns -1.0 if NER reference file does not exist.
    """
    if not Path(ner_spans_path).exists():
        return -1.0

    try:
        with open(ner_spans_path) as f:
            ner_data = json.load(f)
        ner_map = {o["audio_id"]: o.get("entities", []) for o in ner_data}
    except Exception:
        return -1.0

    entity_refs, entity_hyps = [], []
    for pred in predictions:
        audio_id = pred.get("audio_id", "")
        entities = ner_map.get(audio_id, [])
        if entities:
            entity_refs.append(pred.get("reference", "").lower().strip())
            entity_hyps.append(pred.get("prediction", "").lower().strip())

    if not entity_refs:
        return -1.0

    return float(jiwer.wer(entity_refs, entity_hyps))
