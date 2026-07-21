# CARE-ASR System Architecture Specification

## 1. Executive Architecture Overview

CARE-ASR (Confidence-Aware Retrieval-Augmented Clinical Entity Recovery) is a modular post-processing framework designed to recover mistranscribed clinical entities from Speech-to-Text outputs without fine-tuning the underlying ASR model. The architecture is engineered around the principle of strict decoupled modularity: every stage communicates exclusively via immutable data contracts.

```
+-----------------------------------------------------------------------------------+
|                                  Audio Input                                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 1. Transcriber Module (Whisper API / Local Model)                                 |
|    - Outputs: Raw Transcript + Token Logits + Per-token Softmax Distributions    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. Confidence Estimator (Tsallis Entropy Gate)                                   |
|    - Computes non-extensive q-entropy over token probability distributions        |
|    - Thresholds high-uncertainty word spans (H_q > tau_entropy)                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. Clinical Named Entity Recognition (BioBERT NER Tagger)                          |
|    - Categorizes low-confidence spans (MED, COND, ANA, TTP)                      |
|    - Bypasses non-medical uncertain spans                                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------+ +--------------------------------------------+
| 4. Semantic Retrieval Engine       | | 5. Phonetic Retrieval Engine             |
|    - ClinicalBERT Dense Embeddings | |    - Double Metaphone + Levenshtein        |
|    - FAISS Index over UMLS/RxNorm  | |    - Phonetic Hash Index over Medical Vocab|
+------------------------------------+ +--------------------------------------------+
                                     \   /
                                      v v
+-----------------------------------------------------------------------------------+
| 6. Hybrid Candidate Fusion (Reciprocal Rank Fusion - RRF)                        |
|    - Combines semantic similarity ranks and phonetic distance scores              |
|    - Outputs Top-K unified retrieval candidates                                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 7. LLM Correction Engine (Llama-3.1-8B-Instruct)                                  |
|    - Constrained prompt template with sentence context + top retrieval candidates|
|    - Generates candidate replacement term                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 8. Medical Safety & Fallback Gate                                                 |
|    - Character Levenshtein distance check (d_edit <= threshold)                   |
|    - Prevents hallucinated substitutions; enforces UNSURE fallback                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 9. Pipeline Output Assembler                                                      |
|    - Reconstructs corrected clinical transcript + audit trail                     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Design Philosophy & Modularity Contract

### 2.1 Why Modularity Matters (Professor's Directive #1)
In clinical speech processing pipelines, monolithic architectures fail because swapping a single component (e.g., upgrading Whisper-medium to Whisper-large, replacing BioBERT with ClinicalBERT, or altering FAISS to BM25) breaks downstream execution. CARE-ASR enforces strict **Interface Isolation**:

1. **Zero Direct Inter-Module Imports**: Stage $N$ never calls functions or accesses internals of Stage $N-2$. All communication occurs through strongly typed data objects defined in [interface_contract.md](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/docs/interface_contract.md).
2. **Feature Plug-and-Play**: Any stage (e.g., Phonetic Retrieval) can be toggled OFF or swapped via `configs/pipeline.yaml` without changing code in surrounding modules.
3. **Fail-Safe Fallbacks**: If any module fails or returns low-confidence output, the system gracefully degrades to the raw Whisper transcript for that specific token span.

---

## 3. Detailed Module Specifications

---

### Module 1: Transcriber (`src/care_asr/transcriber/`)
* **Responsibility**: Load audio input, execute ASR inference, and extract token-level logits alongside transcript text.
* **Why This Exists**: Standard ASR pipelines output plain text, discarding confidence distributions. CARE-ASR requires raw logit access to perform uncertainty estimation.
* **Input**: `AudioInput` (File path or raw WAV byte buffer, sample rate 16kHz).
* **Output**: `Transcript` (Raw transcript text, list of token strings, timestamp alignments, and $T \times V$ tensor of raw logits).
* **Key Dependencies**: `openai-whisper`, `torch`, `torchaudio`.

---

### Module 2: Confidence Estimator (`src/care_asr/confidence/`)
* **Responsibility**: Compute Tsallis non-extensive entropy over token probability distributions to identify mistranscription uncertainty.
* **Why This Exists**: Shannon entropy treats all probability tail events linearly. Tsallis entropy with parameter $q \neq 1$ accentuates uncertainty in fat-tailed probability distributions common in accented/code-mixed speech.
* **Mathematical Formulation**:
  $$H_q(P) = \frac{1}{q - 1} \left( 1 - \sum_{i=1}^{V} p_i^q \right)$$
* **Input**: `Transcript` object containing per-token logits.
* **Output**: `ConfidenceScore` object mapping each token/word span to its Tsallis entropy score and a binary boolean flag `is_uncertain` ($\text{Entropy} > \tau_{\text{entropy}}$).

---

### Module 3: Clinical Entity Recognizer (`src/care_asr/ner/`)
* **Responsibility**: Filter low-confidence token spans and tag clinical entity types.
* **Why This Exists**: Unsupervised correction across general english words causes unwanted edits. By isolating medical categories (Medication, Condition, Anatomy, Procedure/Test), correction budget is reserved strictly for domain terms.
* **Entity Categories**:
  - `MED`: Medications & Dosages (e.g., *Metformin*, *50mg*)
  - `COND`: Diseases, Symptoms, & Conditions (e.g., *Hypertension*, *Dyspnea*)
  - `ANA`: Anatomical Locations (e.g., *Left Ventricle*, *Subclavian*)
  - `TTP`: Test, Treatment, & Procedures (e.g., *Echocardiogram*)
* **Input**: `Transcript` + `ConfidenceScore`.
* **Output**: `EntitySpan` objects containing entity text, category, character offsets, and uncertainty status.

---

### Module 4: Semantic Retrieval Engine (`src/care_asr/retrieval/semantic.py`)
* **Responsibility**: Dense vector search over UMLS (Unified Medical Language System) and RxNorm concept embeddings.
* **Why This Exists**: Captures semantic equivalences and contextual synonyms for misrecognized terms.
* **Mechanism**: Encodes uncertain entity spans into 768-dimensional embeddings using `ClinicalBERT` and queries a pre-indexed `FAISS` vector database (`IndexFlatIP` / `IndexIVFFlat`).
* **Input**: Entity span text + surrounding sentence context.
* **Output**: List of top-$K$ semantic candidate concepts with cosine similarity scores.

---

### Module 5: Phonetic Retrieval Engine (`src/care_asr/retrieval/phonetic.py`)
* **Responsibility**: Retrieve medical terms matching the acoustic/phonetic sound of the misrecognized transcript span.
* **Why This Exists**: Accented ASR errors are predominantly acoustic (words that sound similar but are spelled differently). Semantic models miss pure phonetic substitutions.
* **Mechanism**: Generates Double Metaphone codes for input spans and calculates weighted Levenshtein phonetic distance against a dictionary of 100,000+ medical terms.
* **Input**: Entity span text string.
* **Output**: List of top-$K$ phonetic candidate concepts with phonetic similarity scores.

---

### Module 6: Candidate Fusion (`src/care_asr/fusion/`)
* **Responsibility**: Merge ranked candidate lists from Semantic and Phonetic retrieval engines into a single unified candidate list.
* **Why This Exists**: Neither semantic nor phonetic retrieval alone is sufficient; fusion balances acoustic likelihood with semantic context.
* **Mechanism**: Reciprocal Rank Fusion (RRF):
  $$RRF\_Score(d) = \sum_{m \in \{sem, phon\}} \frac{1}{k + r_m(d)}$$
  where $k=60$ and $r_m(d)$ is the rank of candidate $d$ in retrieval mode $m$.
* **Input**: Semantic Candidates + Phonetic Candidates.
* **Output**: `FusionCandidate` list sorted by RRF score.

---

### Module 7: LLM Post-Correction Engine (`src/care_asr/correction/`)
* **Responsibility**: Select the best replacement term or leave original unchanged given context and RRF candidates.
* **Why This Exists**: Large Language Models excel at contextual reasoning and sentence fluency, choosing the most clinically plausible term among candidates.
* **Model**: Quantized `Llama-3.1-8B-Instruct` run locally via Ollama / vLLM.
* **Input**: Original sentence context, entity span, and Top-5 `FusionCandidate` items.
* **Output**: Proposed replacement string or `UNSURE`.

---

### Module 8: Medical Safety & Fallback Gate (`src/care_asr/safety/`)
* **Responsibility**: Perform deterministic sanity checks before applying LLM edits.
* **Why This Exists**: LLMs can hallucinate non-existent medical terms or make drastic substitutions that alter clinical meaning.
* **Sanity Checks**:
  1. **Levenshtein Threshold**: Reject edits where normalized edit distance between original and proposed term exceeds $\delta_{\text{max}} = 0.45$.
  2. **Category Preserving**: Reject proposed terms that switch entity category (e.g., replacing a medication with an anatomical term).
  3. **UNSURE Trigger**: Fall back to raw Whisper transcript if safety checks fail.
* **Input**: Original entity span, proposed replacement, entity category.
* **Output**: `CorrectionResult` (Final approved term, edit accepted boolean, safety flag).

---

### Module 9: Pipeline Output Assembler (`src/care_asr/core/`)
* **Responsibility**: Assemble corrected tokens back into fluent text and generate complete audit telemetry.
* **Why This Exists**: Downstream clinical systems require both the final clean transcript and an auditable log of what was changed and why.
* **Input**: Original `Transcript` + List of `CorrectionResult` objects.
* **Output**: `PipelineOutput` containing full corrected transcript, raw transcript, entity recovery logs, and execution latency trace.

---

## 4. Architectural Differentiation: Flowchart vs. Architecture (Directive #2)

| Dimension | Flowchart View (Process View) | Architectural View (Structural View) |
| :--- | :--- | :--- |
| **Primary Focus** | Sequential step-by-step data flow and conditional branching. | Component contracts, class boundaries, memory layouts, and data types. |
| **Target Audience** | Domain clinicians, project managers, evaluators. | Software engineers, ML integrators, system architects. |
| **Error Handling** | Shows visual decision diamonds (e.g., *Is Entropy > Threshold?*). | Specifies exception classes, fallback objects, and thread safety. |
| **Artifact Location** | `README_ARCHITECTURE.md` (Diagrams) | `docs/architecture.md` (This document) |

---

## 5. Dataset Strategy: Dual-Path Framework (Directive #3)

To guarantee execution feasibility, CARE-ASR accommodates two data paths:
- **Path A (Standard Benchmarks)**: Primary evaluation using the AfriSpeech-200 clinical speech split and Medical Speech Translation benchmark datasets.
- **Path B (Custom Synthetic / Clinical Dataset)**: Backup pipeline using Whisper text-to-speech accented synthesis combined with noisy medical entity injection scripts located in `data/scripts/`.

---

## 6. Documentation Justification & Governance (Directive #4)

Every section in this architectural document explicitly outlines **Why This Exists**. This guarantees that all team members (Ankit, Mahi, Aarth, Divya) understand the architectural necessity of each pipeline stage rather than treating specs as opaque instructions.
