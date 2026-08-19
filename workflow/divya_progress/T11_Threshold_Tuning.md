# CARE-ASR Task T11: Threshold Tuning & Error Analysis Framework

**Status**: Completed & Verified ✅
**Module**: `care_asr/evaluation`, `care_asr/thresholds`, `care_asr/contracts`, `scripts`
**Target Audience**: Project maintainers and teammates reviewing the PR

---

## 1. Objective

Task T11 implements the threshold tuning and error analysis framework used to:

1. Classify failed predictions into the official CARE-ASR taxonomy (MED, COND, ANA, TTP, PHI).
2. Compute entity-level evaluation metrics (precision, recall, F1) per category.
3. Produce structured audit reports (`ErrorAnalysisAuditOutput`).
4. Tune category-specific thresholds against a validation set using the existing runtime override API.
5. Orchestrate the full flow (error analysis → tuning → threshold update → re-audit → export) via a script.

This solves the threshold-calibration and qualitative-error-analysis problem for CARE-ASR by turning the category threshold framework from Task T8 into a measurable, tunable component.

---

## 2. Files Created

- `care_asr/thresholds/threshold_tuner.py`
- `scripts/run_tuning_eval.py`

## 3. Files Modified

- `care_asr/evaluation/taxonomy_classifier.py`
- `care_asr/evaluation/metrics_calculator.py`
- `care_asr/contracts/error_analysis_output.py`
- `care_asr/tests/test_evaluation.py`
- `care_asr/tests/test_threshold_engine.py`

---

## 4. Major Components Implemented

| Component | Responsibility |
| :--- | :--- |
| `FailureTaxonomyClassifier` | Deterministically classifies failures into the taxonomy categories using only provided inputs. |
| `classify_failure()` | Classifies a single failed prediction into the project's taxonomy. |
| `aggregate_taxonomy()` | Aggregates classified failures into the existing `ErrorTaxonomy` output model. |
| `ErrorAnalysisEngine.generate_audit_report()` | Computes entity-level precision/recall/F1 per category (MED, COND, ANA, TTP, PHI), overall metrics, and failed-instance lists. |
| `_match_entities()` / `_compute_category_metrics()` / `_compute_overall_metrics()` / `_collect_failed_instances()` | Private helpers of the audit engine (entity matching, per-category and overall metric computation, failed-instance collection). |
| `audit_report_to_dict()` | Serializes `ErrorAnalysisAuditOutput` to a plain dict via `model_dump()`. |
| `save_audit_report()` | Writes the audit report to JSON (`indent=2`, `ensure_ascii=False`, auto-mkdir, post-write verification). |
| `ThresholdTuner` | Evaluates threshold combinations, scores them with `CategoryThresholdEngine.evaluate_candidate_acceptance()`, keeps the best per-category values, and applies them via `update_category_thresholds()`. |
| `ThresholdTuner.tune()` / `run_grid()` | Entry points for tuning a single category and running a grid search across categories. |
| `scripts/run_tuning_eval.py` | Orchestration: load inputs → baseline audit → per-category candidate metrics → `run_grid()` → tuned thresholds applied → re-audit → export report → console summary. |

---

## 5. Existing Interfaces Reused

- `ErrorAnalysisAuditOutput` (Pydantic contract, unchanged)
- `ErrorTaxonomy` (Pydantic contract, unchanged)
- `CategoryThresholdEngine.evaluate_candidate_acceptance()`
- `CategoryThresholdEngine.update_category_thresholds()` (Task T8 runtime override API)
- `ThresholdConfigurationError`
- `ThresholdResult` / `AppliedThresholds`

No contracts or configs were modified; no threshold storage mechanism was duplicated.

---

## 6. Tests Added

- `care_asr/tests/test_evaluation.py` — taxonomy classification, error-analysis engine (perfect prediction, missing prediction, false positive, mixed categories, empty predictions, empty ground truth, failed instances, taxonomy aggregation, overall metrics), JSON serialization (model serialization, successful write, directory auto-creation, read-back, invalid path → `RuntimeError`).
- `care_asr/tests/test_threshold_engine.py` — ThresholdTuner (successful tuning, runtime override applied, no mutation on invalid tuning, multiple categories, deterministic output, empty candidate list).

Current totals (verified):

```text
care_asr/tests/test_evaluation.py ..... 16 passed
care_asr/tests/test_threshold_engine.py  24 passed
```

---

## 7. Validation Performed

- `pytest care_asr/tests` — all passed
- `ruff check care_asr` — all checks passed
- `black --check care_asr` — clean
- `mypy --strict` on touched modules — no issues in the implemented files
- Combined T11/T12/T14 regression run: **95 passed** (test_evaluation + test_threshold_engine + test_retrieval + test_latency + test_india_eval)

---

## 8. Final Outcome

- Failure taxonomy classification is deterministic and fully typed.
- Entity-level metrics and audit reports are produced through the existing contracts.
- Threshold tuning reuses the Task T8 runtime override API — no new storage mechanism.
- The full pipeline is reproducible via `scripts/run_tuning_eval.py`.

---

## 9. Commit / PR Summary

- Commit: `639b88d feat(T11): implement threshold tuning and error analysis pipeline`

---

## 10. Current Status

**✅ Completed** — Implementation, tests, and validation complete.
