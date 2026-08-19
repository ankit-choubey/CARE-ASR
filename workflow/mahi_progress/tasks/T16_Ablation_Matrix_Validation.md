# T16_Ablation_Matrix_Validation

## Status: COMPLETED

## Summary
Cross-validated the 6-mode ablation matrix, per-category breakdown, and accent-stratified WER/FDR benchmark metrics against ground truth transcripts.

## Checkpoints
- [x] Verified 6 ablation modes: Whisper Baseline, Dual Retrieval, Gated Only, No Phonetic, No LLM, and Full CARE-ASR
- [x] Audited 105 samples across African, Indian, and Mixed accent categories
- [x] Verified category-level scores across 12 clinical specialties
- [x] Confirmed mathematical consistency of results in `results/ablation_table.json` and `README.md`
