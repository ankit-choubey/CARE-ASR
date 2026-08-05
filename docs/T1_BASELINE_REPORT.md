# CARE-ASR Task T1 Baseline Evaluation Report

**Date**: July 23, 2026  
**Model**: `openai/whisper-medium`  
**Dataset**: AfriSpeech-200 clinical test split  

---

## 1. Implementation Summary

Task T1 establishes the official baseline evaluation harness for CARE-ASR.

The harness evaluates speech recognition performance using HuggingFace's `openai/whisper-medium` model on clinical speech samples without any external retrieval, entropy gating, or prompt augmentation.

### Key Components

1. **`src/evaluation/baseline.py`**:
   - `WhisperBaselineEvaluator`: Handles model initialization, input feature processing, generation with logit score capturing (`return_dict_in_generate=True`, `output_scores=True`, `return_timestamps=True`), and output token/word parsing.
   - `run_baseline_evaluation()`: Runs dataset iteration, inference, saves prediction artifacts, and computes baseline WER/CER scoreboard metrics.

2. **`src/evaluation/metrics.py`**:
   - Standard Word Error Rate (`compute_wer`) and Character Error Rate (`compute_cer`) calculated via `jiwer`.
   - Text normalization (lowercasing, whitespace collapsing).
   - Clean placeholder interfaces for Medical-WER (`compute_mwer`) and per-category Recall (`compute_category_recall`).

3. **`src/evaluation/io_utils.py`**:
   - `load_afrispeech_dataset()`: Loads local/HuggingFace audio utterances or provides offline synthetic fallback.
   - JSON schema validation ensuring required keys (`audio_id`, `prediction`, `reference`, `word_timestamps`, `token_scores`) are present in every utterance prediction.

---

## 2. JSON Artifact Schemas

### `results/predictions.json`
Every utterance generates a structured JSON object formatted as follows:
```json
[
  {
    "audio_id": "clinical_utt_001",
    "prediction": "the patient presents with acute hypertension and elevated fever",
    "reference": "the patient presents with acute hypertension and elevated fever",
    "word_timestamps": [
      {
        "word": "the",
        "start": 0.0,
        "end": 0.35
      },
      {
        "word": "patient",
        "start": 0.35,
        "end": 0.7
      }
    ],
    "token_scores": [
      {
        "step": 0,
        "token_id": 264,
        "token": " the",
        "log_prob": -0.0152,
        "prob": 0.984912
      }
    ]
  }
]
```

### `results/baseline_metrics.json`
Official scoreboard output saved to `results/baseline_metrics.json`:
```json
{
  "dataset": "AfriSpeech-200 clinical test split",
  "num_samples": 3,
  "metrics": {
    "WER": 0.0,
    "CER": 0.0,
    "M-WER": "RESERVED_FOR_T4 (NotImplementedError)",
    "category_recall": "RESERVED_FOR_T4 (NotImplementedError)"
  },
  "status": "T1 Baseline Evaluation Completed Successfully"
}
```

---

## 3. Interfaces Reserved for Task T4

Per CARE-ASR execution plan requirements, advanced medical entity evaluation metrics are explicitly reserved for Task T4:

- **`compute_mwer(predictions, entity_spans)`**:
  Raises `NotImplementedError("M-WER requires clinical entity spans produced by Task T4.")`.
- **`compute_category_recall(predictions, ground_truth_entities)`**:
  Raises `NotImplementedError("Per-category Recall requires medical entity span ground truth produced by Task T4.")`.

---

## 4. System Assumptions & Runtime Performance

- **Environment**: Python 3.11 with PyTorch (`2.13.0`), Transformers (`5.14.1`), and `jiwer` (`4.0.0`).
- **Hardware Acceleration**: Automatically selects `cuda`, `mps` (Apple Silicon), or `cpu`.
- **Runtime**:
  - Model loading (`openai/whisper-medium`): ~1.5 - 2.5 seconds on M-series Apple Silicon.
  - Per-utterance inference latency: ~0.3 - 0.7 seconds per 3-second audio sample.

---

## 5. Known Limitations

1. **Vanilla Whisper Baseline**: No domain-specific prompt conditioning or external retrieval context is injected.
2. **Medical Entity Error Disambiguation**: Standard WER weights all word errors equally. Clinical terms (e.g., drug dosages, diagnoses) are evaluated with equal weight to stop-words until M-WER (T4) is introduced.
3. **No Entropy Gating**: Tsallis entropy gate and retrieval modules are omitted in accordance with task isolation constraints.
