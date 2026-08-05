# CARE-ASR: BioBERT NER & Retrieval Candidate Validation Engine

## Overview
This repository contains the official production skeleton for the **BioBERT Named Entity Recognition (NER), Medical Entity Schema Definition, Retrieval Candidate Validation, Category-Specific Thresholding, and Error Analysis Engine** module of **CARE-ASR** (Confidence-Aware Retrieval-Augmented Clinical Entity Recovery).

---

## Official CARE-ASR Entity Taxonomy
This module enforces the 5 official CARE-ASR clinical entity categories:
- **`MED`**: Medications, drugs, chemical compounds, brand/generic names.
- **`COND`**: Medical conditions, diseases, diagnoses, symptoms, disorders.
- **`ANA`**: Anatomical sites, body structures, organs, sub-structures.
- **`TTP`**: Tests, Treatments, Procedures, lab panels, surgical interventions.
- **`PHI`**: Protected Health Information (patient identifiers, dates, locations, clinician names).

---

## Module Ownership & Boundary Matrix

| Module / Package | Primary Owner | Consumed By | Produced For |
| :--- | :--- | :--- | :--- |
| `care_asr.contracts` | Lead Architect (User) | Ankit, Divya, Mahi | Pydantic Interface Contracts |
| `care_asr.config` | Lead Architect (User) | All Internal Modules | Pydantic Settings & YAML Config |
| `care_asr.ner` | Lead Architect (User) | Ankit Integration | BioBERT Entity Extraction |
| `care_asr.thresholds` | Lead Architect (User) | Candidate Evaluator | Category Threshold Engine |
| `care_asr.validation` | Lead Architect (User) | Ankit Integration | Candidate Validation & Scoring |
| `care_asr.evaluation` | Lead Architect (User) | Mahi (QA Lead) | Error Taxonomy & F1 Audit Reports |
| `care_asr.utils` | Lead Architect (User) | All Internal Modules | Logging & Custom Exceptions |
| `care_asr.tests` | Lead Architect (User) | Mahi (QA Lead) | Internal & Integration Tests |

---

## Teammate Handoff Interfaces

- **Ankit (ASR & Integration Lead)**:
  - Inputs: Sends `ASRTranscriptInput` (`care_asr.contracts.asr_input`).
  - Outputs: Receives `ValidatedCandidatesOutput` (`care_asr.contracts.validated_output`).
- **Divya (FAISS & Retrieval Lead)**:
  - Inputs: Sends `RetrievalCandidatesInput` (`care_asr.contracts.retrieval_input`).
- **Mahi (Testing & QA Lead)**:
  - Outputs: Receives `ErrorAnalysisAuditOutput` (`care_asr.contracts.error_analysis_output`).

---

## Installation & Setup

1. Ensure Python 3.11+ is installed.
2. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
5. Run tests:
   ```bash
   pytest
   ```

---

## Repository Structure

```
care_asr/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   ├── asr_input.py
│   ├── retrieval_input.py
│   ├── validated_output.py
│   └── error_analysis_output.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── ner/
│   ├── __init__.py
│   ├── extractor.py
│   └── span_aligner.py
├── thresholds/
│   ├── __init__.py
│   └── threshold_engine.py
├── validation/
│   ├── __init__.py
│   ├── candidate_evaluator.py
│   └── decision_router.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics_calculator.py
│   └── taxonomy_classifier.py
├── utils/
│   ├── __init__.py
│   ├── exceptions.py
│   └── logger.py
└── tests/
    ├── __init__.py
    ├── test_ner_extractor.py
    ├── test_candidate_evaluator.py
    ├── test_threshold_engine.py
    └── test_evaluation.py
```

---

## Dataset, Index Generation & Evaluation Utilities

All dataset acquisition, index construction, evaluation, tuning, and benchmarking in this repository is driven by reproducible Python scripts under `scripts/`. Each script is configured through the YAML files in `configs/` and can be run from the repository root.

| Script | Purpose | Input(s) | Output(s) | When to use | Required for reproduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `scripts/build_semantic_index.py` | Builds the semantic FAISS index over medical concepts using RxNorm + Bio_ClinicalBERT. | HuggingFace `nishanth-augustai/rxnorm_data`; checkpoint `emilyalsentzer/Bio_ClinicalBERT`. | `data/indices/faiss_umls.index`, `data/indices/cui_mapping.json`. | Once, before any semantic retrieval or evaluation. | **Yes** |
| `scripts/build_phonetic_index.py` | Builds the phonetic FAISS index over audio utterances using HuBERT embeddings. | HuggingFace `facebook/hubert-base-ls960`; AfriSpeech-200 test set (`intronhealth/afrispeech-200`). | Phonetic FAISS index (`data/indices/faiss_phonetic.index`), utterance metadata (`data/indices/utterance_metadata.json`). | Once, before any phonetic retrieval or evaluation. | **Yes** |
| `scripts/run_ner_extraction.py` | Generates reference NER annotations (MED, COND, ANA, TTP, PHI) used for entity-level evaluation. | Reference transcripts; BioBERT checkpoint `d4data/biomedical-ner-all`. | `data/processed/afrispeech_reference_ner_tags.json`. | Before M-WER / error-analysis evaluation. | **Yes** |
| `scripts/run_eval.py` | Runs the complete Week-1 evaluation pipeline across all 6 ablation modes. | Saved AfriSpeech dataset (`--data-path`, via `load_from_disk`); `openai/whisper-medium`. | `{mode}_predictions.json`, `{mode}_metrics.json` per mode. | For the main ablation / evaluation scoreboard. | **Yes** |
| `scripts/run_india_eval.py` | Runs inference on the India medical datasets (EKA + Svarah) using the frozen pipeline. | HuggingFace `ekacare/eka-medical-asr-evaluation-dataset` (config `en`) and `ai4bharat/Svarah`; local JSON or `--synthetic`. | India context table, `{dataset}_predictions.json`, `{dataset}_metrics.json` under `outputs/metrics/india/`. | For the India context evaluation sweep. | **Yes** |
| `scripts/run_tuning_eval.py` | Runs category-specific threshold tuning and generates error-analysis audit reports. | Ground-truth + predictions JSON (`--ground-truth`, `--predictions`). | Audit report JSON under `outputs/audit_reports/`. | After evaluation, when thresholds need tuning. | **Yes** |
| `scripts/run_latency_benchmark.py` | Benchmarks retrieval latency, embedding-cache efficiency, and batching. | Local dataset JSON or `--synthetic`. | `outputs/latency_reports/latency_benchmark.json`. | To validate latency claims after batching/caching changes. | No (diagnostic) |

### Generated Artifacts

| Artifact | Producer | Description |
| :--- | :--- | :--- |
| `data/indices/faiss_umls.index` | `scripts/build_semantic_index.py` | Semantic FAISS `IndexFlatIP` over ClinicalBERT concept embeddings; consumed by `src/retrieval/semantic.py`. |
| `data/indices/cui_mapping.json` | `scripts/build_semantic_index.py` | Position-to-concept mapping for the semantic index. |
| Phonetic FAISS index | `scripts/build_phonetic_index.py` | Phonetic `IndexFlatIP` over HuBERT utterance embeddings; consumed by `src/retrieval/phonetic.py`. |
| Utterance metadata | `scripts/build_phonetic_index.py` | Position-to-utterance metadata JSON for the phonetic index. |
| `outputs/audit_reports/` | `scripts/run_tuning_eval.py` | Threshold-tuning and error-analysis audit reports. |
| `outputs/latency_reports/` | `scripts/run_latency_benchmark.py` | Latency benchmark JSON reports. |
| `outputs/metrics/india/` | `scripts/run_india_eval.py` | India context table plus EKA/Svarah predictions and metrics. |
| Evaluation prediction JSON files | `scripts/run_eval.py` | Per-utterance predictions per ablation mode (`{mode}_predictions.json`). |
| Evaluation metrics JSON files | `scripts/run_eval.py` | Per-mode WER / unsure-rate metrics (`{mode}_metrics.json`). |

### Reproducing the Project

Run the following in order from the repository root:

1. **Download datasets** — fetch AfriSpeech-200, EKA, and Svarah from Hugging Face (or provide local copies).
2. **Build the semantic index** — `python scripts/build_semantic_index.py`
3. **Build the phonetic index** — `python scripts/build_phonetic_index.py`
4. **Generate NER reference annotations** — `python scripts/run_ner_extraction.py`
5. **Run evaluation** — `python scripts/run_eval.py --mode <baseline|naive_correction|dual_retrieval|entropy_gated|thresholded|unsure_gate> --data-path <path>`
6. **Run India evaluation** — `python scripts/run_india_eval.py --datasets eka svarah`
7. **Run threshold tuning** — `python scripts/run_tuning_eval.py --ground-truth <path> --predictions <path> --output outputs/audit_reports/run_001_audit_report.json`
8. **Run latency benchmark** — `python scripts/run_latency_benchmark.py --dataset <path>`

> **Note:** This repository intentionally uses Python scripts instead of Jupyter notebooks. Dataset downloading, preprocessing, index construction, evaluation, tuning, and benchmarking are all reproducible using the provided scripts.

---

## Code Quality & Tooling
- **Linter**: `ruff check .`
- **Formatter**: `black --check .`
- **Type Checker**: `mypy care_asr`
