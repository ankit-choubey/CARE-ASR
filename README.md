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

## Code Quality & Tooling
- **Linter**: `ruff check .`
- **Formatter**: `black --check .`
- **Type Checker**: `mypy care_asr`

---

## Dataset Preparation & External Resources

This section documents every external dataset, index, and generated artifact used by the project, so contributors can reproduce them from scratch.

### AfriSpeech-200

- **What it is**: The primary ASR evaluation corpus — African-accented English clinical speech (test split used throughout).
- **Why it is needed**: Ground-truth audio + transcripts for baseline WER/CER evaluation and for phonetic index generation.
- **Where it comes from**: Hugging Face — `intronhealth/afrispeech-200`, configuration `all`, split `test`.
- **Script**: `scripts/download_afrispeech.py` (use `--save-to-disk` to persist; `--overwrite` to replace an existing copy).
- **Input(s)**: Hugging Face dataset `intronhealth/afrispeech-200` (`all` / `test`).
- **Output location**: `data/raw/afrispeech` (via `Dataset.save_to_disk()`).
- **Commit vs regenerate**: **Regenerate** — raw audio is large and not committed; `run_eval.py` reloads it locally via `load_from_disk()`.
- **Note**: the runtime `datasets` version must support the dataset's loading script (older `datasets` releases work; very recent ones reject the legacy `afrispeech-200.py` loader).

### RxNorm

- **What it is**: Medical concept vocabulary (RxNorm drugs) used to build the semantic retrieval index.
- **Why it is needed**: Supplies the clinical concepts that the semantic retriever searches over.
- **Where it comes from**: Hugging Face — `nishanth-augustai/rxnorm_data` (train split, filtered to English non-suppressed records).
- **Script**: `scripts/build_semantic_index.py` (encodes concepts with `emilyalsentzer/Bio_ClinicalBERT`, builds an `IndexFlatIP` FAISS index).
- **Input(s)**: HF `nishanth-augustai/rxnorm_data`; HF checkpoint `emilyalsentzer/Bio_ClinicalBERT`.
- **Output location**: `data/indices/faiss_umls.index`, `data/indices/cui_mapping.json`.
- **Commit vs regenerate**: **Committed** (both files are tracked) but fully **regenerable** via the script.

### Phonetic Index

- **What it is**: FAISS index over HuBERT audio embeddings of AfriSpeech-200 utterances, used for phonetic (sound-based) retrieval.
- **Why it is needed**: Recovers medical terms that sound like mistranscribed ASR spans — the phonetic half of dual retrieval.
- **Where it comes from**: Local generation — HuBERT (`facebook/hubert-base-ls960`) embeddings computed over the AfriSpeech-200 test set.
- **Script**: `scripts/build_phonetic_index.py` (delegates to the builders in `src/retrieval/phonetic.py`).
- **Input(s)**: HF `facebook/hubert-base-ls960`; AfriSpeech-200 audio (`intronhealth/afrispeech-200`).
- **Output location**: `data/indices/faiss_phonetic.index`, `data/indices/utterance_metadata.json`.
- **Commit vs regenerate**: **Regenerate** — not currently committed.
- **Note**: `PhoneticRetriever` reads `phonetic_index.faiss`, `phonetic_labels.json`, and `medical_vocab.json` from `data/indices/`; verify these names align with the builder's outputs before running evaluation.

### NER Reference Data

- **What it is**: Reference NER annotations (MED, COND, ANA, TTP, PHI) over reference transcripts.
- **Why it is needed**: Ground truth for entity-level evaluation (M-WER / error analysis).
- **Where it comes from**: Local generation via BioBERT NER inference.
- **Script**: `scripts/run_ner_extraction.py` (uses `d4data/biomedical-ner-all` through `care_asr/ner/extractor.py`).
- **Input(s)**: Reference transcripts; HF checkpoint `d4data/biomedical-ner-all`.
- **Output location**: `data/processed/afrispeech_reference_ner_tags.json`.
- **Commit vs regenerate**: **Regenerate** — currently runs on mock transcripts; real annotations should be produced over the real AfriSpeech test set.

### India Evaluation Datasets

- **What they are**: Two Indian-medical ASR datasets used for the India context evaluation sweep (T14).
- **Why they are needed**: Validate the frozen pipeline on Indian English / Hindi clinical speech.
- **Where they come from**: Hugging Face — **EKA Medical ASR Evaluation Dataset** (`ekacare/eka-medical-asr-evaluation-dataset`, default config `en`) and **Svarah** (`ai4bharat/Svarah`).
- **Script**: `scripts/run_india_eval.py` downloads them automatically (also supports local JSON via `--local-dir` and offline synthetic data via `--synthetic`).
- **Input(s)**: HF datasets above (or local JSON / synthetic).
- **Output location**: `outputs/metrics/india/` — `india_context_table.json`, `{dataset}_predictions.json`, `{dataset}_metrics.json`.
- **Commit vs regenerate**: **Regenerate** — outputs are transient evaluation artifacts.

---

## Data & Index Generation Pipeline

Regenerate everything from scratch by running these steps in order from the repository root:

1. **Download AfriSpeech** — `python scripts/download_afrispeech.py --save-to-disk`
2. **Build Semantic Index** — `python scripts/build_semantic_index.py`
3. **Build Phonetic Index** — `python scripts/build_phonetic_index.py`
4. **Generate NER References** — `python scripts/run_ner_extraction.py`
5. **Run Evaluation** — `python scripts/run_eval.py --mode <baseline|naive_correction|dual_retrieval|entropy_gated|thresholded|unsure_gate> --data-path <afrispeech_dir>`
6. **Run Threshold Tuning** — `python scripts/run_tuning_eval.py --ground-truth <gt.json> --predictions <preds.json> --output outputs/audit_reports/run_001_audit_report.json`
7. **Run Latency Benchmark** — `python scripts/run_latency_benchmark.py --dataset <dataset.json>`
8. **Run India Evaluation** — `python scripts/run_india_eval.py --datasets eka svarah` (add `--synthetic` for an offline run)

---

## For Contributors

- **Scripts that download datasets**: `scripts/download_afrispeech.py` (AfriSpeech-200), `scripts/run_india_eval.py` (EKA + Svarah, auto-downloaded from Hugging Face).
- **Scripts that build indexes**: `scripts/build_semantic_index.py` (semantic FAISS + CUI mapping), `scripts/build_phonetic_index.py` (phonetic FAISS + utterance metadata).
- **Scripts that generate evaluation artifacts**: `scripts/run_ner_extraction.py` (NER reference tags), `scripts/run_eval.py` (ablation predictions/metrics), `scripts/run_tuning_eval.py` (threshold tuning + audit reports), `scripts/run_latency_benchmark.py` (latency reports), `scripts/run_india_eval.py` (India context table + predictions/metrics).
- **Generated files required for integration**: `data/indices/faiss_umls.index` and `data/indices/cui_mapping.json` — the semantic index consumed by `src/retrieval/semantic.py`; currently committed so the pipeline works out of the box.
- **Files that can safely be regenerated instead of committed**: raw datasets (`data/raw/`), the phonetic index + metadata (`data/indices/faiss_phonetic.index`, `utterance_metadata.json`), NER reference tags (`data/processed/`), and all `outputs/` artifacts (audit reports, latency reports, India metrics).
