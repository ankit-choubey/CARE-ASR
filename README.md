# CARE-ASR: Confidence-Aware Retrieval-Augmented Clinical Entity Recovery for Accented and Code-Mixed Medical Speech Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-261237.svg)](https://github.com/astral-sh/ruff)

> **CARE-ASR** is a modular, confidence-aware post-processing framework designed to recover misrecognized medical entities (medications, conditions, anatomy, procedures) in accented and code-mixed clinical Speech-to-Text outputs without requiring end-to-end model retraining.

---

## 1. Project Introduction

Automatic Speech Recognition (ASR) systems like OpenAI's Whisper demonstrate remarkable general performance, yet routinely fail in specialized clinical settings—especially when processing non-native accents, regional pronunciations, and code-mixed clinical narratives. In medical documentation, misrecognizing a drug dosage or disease name (e.g., transcribing *"Metformin"* as *"Met-formin"* or *"chloroquine"* as *"clear queen"*) can severely disrupt clinical workflows and downstream EHR indexing.

**CARE-ASR** addresses this challenge through a lightweight, zero-retraining post-correction pipeline. By combining **Tsallis Entropy uncertainty gating**, **hybrid semantic-phonetic retrieval (ClinicalBERT + FAISS + Double Metaphone)** over UMLS/RxNorm knowledge bases, and **Safety-Gated LLM post-editing**, CARE-ASR selectively pinpoints and repairs mistranscribed medical terms while preserving non-medical context.

---

## 2. Problem Statement

General-purpose ASR models face two critical bottlenecks in global clinical settings:
1. **Acoustic & Phonetic Degradation**: Accented speech introduces phonetic variations unobserved in standard training datasets, causing acoustic confusions.
2. **Out-of-Vocabulary (OOV) & Rare Medical Terms**: Specialist drug names, anatomical terms, and brand names are frequently substituted with common english word approximations.

Existing end-to-end fine-tuning approaches require massive annotated audio datasets and compute budgets. CARE-ASR solves this by acting as an **entropy-driven, retrieval-augmented post-processor** that operates directly on standard Whisper decoder confidence metrics and external medical knowledge bases.

---

## 3. Pipeline Diagram

```
                     +---------------------------+
                     |    Input Audio Stream     |
                     +---------------------------+
                                   |
                                   v
                     +---------------------------+
                     |   Whisper ASR Decoder     |
                     +---------------------------+
                                   |
                                   v
                     +---------------------------+
                     |   Tsallis Entropy Gate    |
                     +---------------------------+
                                   |
                         [Low Confidence Spans]
                                   v
                     +---------------------------+
                     |   BioBERT NER Tagger      |
                     +---------------------------+
                                   |
                                   v
               +-------------------+-------------------+
               |                                       |
               v                                       v
   +-----------------------+               +-----------------------+
   |  Semantic Retrieval   |               |  Phonetic Retrieval   |
   | (ClinicalBERT + FAISS)|               |  (Double Metaphone)   |
   +-----------------------+               +-----------------------+
               |                                       |
               +-------------------+-------------------+
                                   |
                                   v
                     +---------------------------+
                     | Reciprocal Rank Fusion    |
                     |       (RRF Candidate)     |
                     +---------------------------+
                                   |
                                   v
                     +---------------------------+
                     | LLM Post-Correction (Llama)|
                     +---------------------------+
                                   |
                                   v
                     +---------------------------+
                     |   Medical Safety Gate     |
                     +---------------------------+
                                   |
                                   v
                     +---------------------------+
                     | Final Clinical Transcript |
                     +---------------------------+
```

---

## 4. Repository Structure

```
CARE-ASR/
│
├── README.md                      # Project introduction, setup, & quick start
├── README_ARCHITECTURE.md         # Visual & structural architectural breakdown
├── CONTRIBUTING.md                # Contribution guidelines & coding standards
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation metadata format
├── CHANGELOG.md                   # Version history & release notes
├── SECURITY.md                    # Security vulnerability reporting policy
├── CODE_OF_CONDUCT.md             # Contributor Covenant Code of Conduct
├── pyproject.toml                 # Dependencies, tool configs (Black, Ruff, Pytest)
├── .gitignore                     # ML/Python ignore configuration
│
├── .github/                       # GitHub configurations
│   ├── CODEOWNERS                 # File/module ownership mapping
│   ├── PULL_REQUEST_TEMPLATE.md   # Pull request submission template
│   ├── ISSUE_TEMPLATE/            # Standardized issue templates
│   └── workflows/
│       └── tests.yml              # CI/CD workflow (linting, tests)
│
├── docs/                          # Comprehensive technical documentation
│   ├── architecture.md            # Modular pipeline design (~400 lines)
│   ├── interface_contract.md      # Data schemas & shared objects (~500 lines)
│   ├── api_reference.md           # Python module API documentation
│   ├── execution_notes.md         # S1-S3 & T1-T18 execution plan mapping
│   └── meeting_notes.md           # Governance & decision alignment logs
│
├── configs/                       # Yaml configuration files
│   ├── pipeline.yaml              # Core pipeline hyperparameters
│   ├── model.yaml                 # Model endpoints & quantization configs
│   ├── retrieval.yaml             # FAISS, BM25, & RRF settings
│   └── eval.yaml                  # Metrics & scoring configurations
│
├── requirements/                  # Modular requirement specifications
│   ├── base.txt                   # Core runtime dependencies
│   ├── dev.txt                    # Development, formatting, & linting tools
│   └── test.txt                   # Testing framework dependencies
│
├── src/                           # Source code modules
│   ├── care_asr/
│   │   ├── core/                  # Engine & pipeline orchestration
│   │   ├── transcriber/           # Whisper model wrapping & logit extraction
│   │   ├── confidence/            # Tsallis entropy computation
│   │   ├── ner/                   # BioBERT entity recognition
│   │   ├── retrieval/             # Semantic & phonetic FAISS search
│   │   ├── fusion/                # Reciprocal Rank Fusion engine
│   │   ├── correction/            # LLM prompt construction & inference
│   │   └── safety/                # Fallback, edit-distance, & sanity gate
│   └── utils/                     # Logging, IO, & formatting utilities
│
├── tests/                         # Pytest suite
│   ├── unit/                      # Isolated unit tests per module
│   └── integration/               # End-to-end pipeline tests
│
└── data/                          # Dataset scripts, indices, & ground truth
```

---

## 5. Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (Recommended for FAISS & LLM inference)
- `ffmpeg` installed on system path

### Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/ankit-choubey/CARE-ASR.git
cd CARE-ASR

# Create python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -e .
```

---

## 6. Quick Start

### Basic Python Usage

```python
from care_asr.core.pipeline import CareAsrPipeline
from care_asr.transcriber.whisper import WhisperTranscriber

# Initialize CARE-ASR pipeline
pipeline = CareAsrPipeline.from_config("configs/pipeline.yaml")

# Run pipeline on clinical audio clip
audio_path = "demo/sample_clinical_accented.wav"
output = pipeline.run(audio_path)

print(f"Raw ASR Transcript:     {output.raw_transcript}")
print(f"Corrected Transcript:   {output.corrected_transcript}")
print(f"Entities Recovered:     {len(output.recovered_entities)}")
```

---

## 7. Team Members

- **Ankit Choubey** - *Integration Lead & ML Core Engineer*
- **Mahi Nandini** - *Testing, Evaluation & Interface QA*
- **Aarth** - *Data Lead & Domain Taxonomy Specialist*
- **Divya** - *Preprocessing Lead & Retrieval Engine Engineer*

---

## 8. Tech Stack

- **Speech Recognition**: OpenAI Whisper (`whisper-medium` / `whisper-large-v3`)
- **Uncertainty Estimation**: PyTorch, Tsallis Entropy Gating
- **Entity Recognition**: Hugging Face Transformers (`BioBERT` / `ClinicalBERT`)
- **Semantic Search**: FAISS (Facebook AI Similarity Search), Sentence-Transformers
- **Phonetic Search**: `abydos`, Double Metaphone algorithm
- **Correction Engine**: Ollama / vLLM, `Llama-3.1-8B-Instruct`
- **Quality & Safety**: Levenshtein edit-distance guards, Python 3.10+, Pytest, Ruff, Black

---

## 9. Results (Placeholder)

| Metric | Raw Whisper Medium | CARE-ASR (Proposed) | Improvement (%) |
| :--- | :---: | :---: | :---: |
| **Overall WER (%)** | 18.4% | *TBD* | *TBD* |
| **Medical Entity WER (M-WER %)** | 34.2% | *TBD* | *TBD* |
| **Medication Name Recall (%)** | 62.1% | *TBD* | *TBD* |
| **Anatomical Term Recall (%)** | 68.5% | *TBD* | *TBD* |
| **P95 Latency (s)** | 1.2s | *TBD* | *TBD* |

*Note: Benchmark evaluation on AfriSpeech-200 clinical test split currently in progress.*

---

## 10. Citation

If you use CARE-ASR in your research, please cite our repository:

```bibtex
@software{choubey2026careasr,
  author = {Choubey, Ankit and Nandini, Mahi and Aarth and Divya},
  title = {CARE-ASR: Confidence-Aware Retrieval-Augmented Clinical Entity Recovery for Accented and Code-Mixed Medical Speech Recognition},
  year = {2026},
  url = {https://github.com/ankit-choubey/CARE-ASR}
}
```

---

## 11. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
