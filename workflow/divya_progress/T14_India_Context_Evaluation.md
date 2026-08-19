# CARE-ASR Task T14: India Context Evaluation

**Status**: Completed & Verified ✅
**Module**: `scripts`, `tests/unit`
**Target Audience**: Project maintainers and teammates reviewing the PR

---

## 1. Objective

Task T14 runs the frozen Week 1 CARE-ASR pipeline over the Indian medical speech datasets (EKA + Svarah) using pure inference only (no retraining), and generates the India context evaluation table:

- Dataset loading layer (Hugging Face / local JSON / deterministic synthetic fallback).
- Frozen pipeline construction with graceful fallback for optional heavy components.
- Inference over normalized samples with the project's prediction schema.
- Raw WER / pipeline WER / CER metrics.
- India context evaluation table (one row per dataset).
- JSON export of predictions, metrics, and the context table.

No core implementation files were modified — `src/*`, `care_asr/*`, and `configs/*` are untouched.

---

## 2. Files Created

- `scripts/run_india_eval.py`
- `tests/unit/test_india_eval.py`

## 3. Files Modified

None (Commit 1 created the loader + CLI; Commits 2–4 extended the same script and test file).

---

## 4. Major Components Implemented

| Component | Responsibility |
| :--- | :--- |
| `DATASET_REGISTRY` | Immutable registry (`MappingProxyType` over frozen spec dataclasses) for **EKA** (`ekacare/eka-medical-asr-evaluation-dataset`, default config `en`, reference field `text`) and **Svarah** (`ai4bharat/Svarah`, reference field `sentence`). |
| `load_india_samples()` | Loads a dataset from Hugging Face, local JSON, or deterministic synthetic fallback into the shared normalized sample format `{audio_id, audio, sample_rate, reference}`; caps at `max_samples`. |
| `_normalize_sample()` | Shared schema normalization handling HF audio dicts, inline arrays, and `audio_path` via `soundfile`. |
| `_load_synthetic()` | Deterministic synthetic clinical dataset (fixed sine tones + fixed transcripts, no RNG) for offline testing. |
| `build_frozen_pipeline()` | Binds real `WhisperTranscriber`, `TsallisUncertaintyGate`, `SemanticRetriever`, `PhoneticRetriever`, `LLMCorrector`, `UnsureGate` via testable loader seams; every binding falls back gracefully with a logged warning when a component cannot be created (transformers/ollama/FAISS/checkpoint unavailable). |
| `_make_transcriber_callable()` | Wraps `WhisperTranscriber.transcribe(array, rate)` or plain callables and captures each `Transcript` (word timestamps + token scores) without a second ASR pass. |
| `run_dataset_inference()` | Per-sample transcription + frozen-pipeline run producing schema-conformant predictions (`audio_id`, `prediction`, `reference`, `word_timestamps`, `token_scores`) plus `raw_asr_prediction`; per-sample failures degrade gracefully. |
| `compute_dataset_metrics()` | Reuses `compute_wer()` / `compute_cer()` for raw WER, pipeline WER, CER, and WER improvement. |
| `build_india_context_table()` | One row per dataset with columns `dataset`, `language/config`, `num_samples`, `raw_wer`, `pipeline_wer`, `cer`, `wer_improvement`, `note` ("Inference-only"). |
| `export_india_results()` | Reuses `save_predictions()` / `save_metrics()`; writes `{dataset}_predictions.json`, `{dataset}_metrics.json`, and `india_context_table.json` with `indent=2`, `ensure_ascii=False`, auto-mkdir. |
| CLI | `--datasets`, `--config`, `--max-samples`, `--synthetic`, `--local-dir`, `--output-dir`, `--model`; exit 0 on success, exit 1 on failure. Default output dir from `configs/evaluation.yaml → india_dir` (`outputs/metrics/india`). |

---

## 5. Existing Interfaces Reused

- `CARPipeline` (unchanged)
- `WhisperTranscriber`, `TsallisUncertaintyGate`, `SemanticRetriever`, `PhoneticRetriever`, `LLMCorrector`, `UnsureGate`
- `compute_wer()`, `compute_cer()`, `save_predictions()`, `save_metrics()`
- `configs/evaluation.yaml` (india_dir)
- Pipeline stubs (`src/pipeline/stubs.py`) as the graceful fallback path

---

## 6. Tests Added

- `tests/unit/test_india_eval.py` — registry contents, invalid dataset raises `ValueError`, normalized schema, `max_samples` respected, synthetic determinism, local JSON loading, CLI summary, empty dataset fallback, pipeline construction with real fakes, graceful fallback, inference schema, `save_predictions` reuse, raw/pipeline WER + CER + improvement (hand-computable), context-table columns, JSON export, CLI full run, determinism, invalid dataset → exit 1, invalid output path → exit 1. Heavy dependencies (Whisper/Ollama/FAISS/internet) are mocked.

Current total (verified):

```text
tests/unit/test_india_eval.py ..... 19 passed
```

---

## 7. Validation Performed

- `pytest tests/unit/test_india_eval.py` — all passed
- `ruff check scripts/run_india_eval.py tests/unit/test_india_eval.py` — all checks passed
- `black --check` — clean
- `mypy --strict scripts/run_india_eval.py` — no issues in the script (remaining errors, if any, are pre-existing in untouched files)
- Combined T11/T12/T14 regression run: **95 passed**
- CLI `--synthetic` end-to-end run: exit 0, exports all artifacts, metrics byte-identical across runs
- Broader suite (india + retrieval + fusion + latency + integration): **60+ passed**

---

## 8. Final Outcome

- Frozen-pipeline inference over EKA + Svarah is reproducible in three modes (HF / local / synthetic).
- The India context table and all prediction/metrics artifacts are exported to `outputs/metrics/india/`.
- The sweep never crashes when optional runtime dependencies are unavailable (stub fallback + per-sample guards).

---

## 9. Commit / PR Summary

- Commit: `bf5ff8a feat(T14): add India context evaluation pipeline`
- Merged: `3a33317 Merge branch 'T14-india-context'`

---

## 10. Current Status

**✅ Completed** — Implementation, tests, and validation complete.
