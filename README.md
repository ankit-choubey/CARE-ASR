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
