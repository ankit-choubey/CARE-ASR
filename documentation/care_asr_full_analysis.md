# CARE-ASR Project State: Comprehensive Honest Analysis

> **This document tells you exactly where you are, where the holes are, and what needs to happen next. No inflation, no vagueness.**

---

## 1. What CARE-ASR Is Actually Building (Per Final Doc)

The project is a **8-module clinical ASR post-correction pipeline** for accented/code-mixed medical speech. Here's the exact architecture the design document specifies:

| Module | What It Does | Input Contract | Output Contract |
| :--- | :--- | :--- | :--- |
| **M1: ASR Backbone** | Whisper-medium transcribes audio → text + per-token logit scores | Raw audio waveform | Text tokens + per-token confidence scores |
| **M2: Uncertainty Gate** | Tsallis entropy (α=1/3) → boolean uncertain flag per token | Per-token logit scores | Boolean flag per token |
| **M3: Clinical NER** | BioBERT/d4data → MED / COND / ANA / TTP / PHI category tags | Text tokens | Boolean flag + category label per token |
| **M4: Dual Retrieval** | Semantic (ClinicalBERT + FAISS over UMLS/RxNorm) + Phonetic (Whisper encoder hidden states + FAISS) | Flagged uncertain token + context | Ranked candidate lists (two independent channels) |
| **M5: Fusion** | RRF merges two ranked lists → single top-K list | Two ranked lists | One ranked list (top-K) |
| **M6: Correction Engine** | Qwen2.5-7B-Instruct (primary) / Llama-3.1-8B (upgrade) → picks one candidate | Top-K candidates + ASR context | One candidate + confidence score |
| **M7: Safety Gate** | CORRECT / WRONG / UNSURE three-class gate; UNSURE → keep original token | One candidate + confidence | CORRECT/WRONG/UNSURE + final text |
| **M8: Output Assertion** | Per-layer attribution logging, M-WER/FDR metrics, ablation scoring | Final transcript + trace | Metrics report + attribution log |

**Target Datasets**: AfriSpeech-200 (primary), EKA + Svarah (India inference sweep)  
**Target Claims**: M-WER improvement over raw Whisper, UNSURE fallback rate, ablation table (baseline → naive → dual retrieval → entropy-gated → thresholded → UNSURE), False Discovery Rate

---

## 2. Real Progress: Who Did What, What Actually Exists

### 2.1 Ankit + Mahi Work (Your Stream)

#### ✅ DONE — S3: Whisper Logit Probe
- **File**: `care_asr/probes/whisper_scores_probe.py` (40 lines, real implementation)
- **Tests**: `tests/test_whisper_probe.py` — **4 tests, all PASS**
- Extracts `output_scores` from Whisper decoder, confirms tensor shape
- **Quality**: Good. Confirmed working.

#### ✅ DONE — T1: Baseline Harness  
- **Files**: `src/evaluation/baseline.py` (63 lines), `src/evaluation/metrics.py` (35 lines), `src/evaluation/io_utils.py` (45 lines)
- **Tests**: `tests/test_baseline.py` — **5 tests PASS**, `tests/test_metrics.py` — **6 tests PASS**
- **Quality Issue**: `results/predictions.json` contains only 3 synthetic utterances, two of which produced garbage ("BEEEEEE...") output. WER = 1.0 in `baseline_metrics.json`. This is because the baseline ran on **synthetic placeholder audio, not real AfriSpeech-200 data.**
- **M-WER**: Correctly reserved as `NotImplementedError` pending T4. ✓
- **Data**: `data/raw/afrispeech/` is **EMPTY** — real AfriSpeech-200 has NOT been downloaded.

#### ✅ DONE — T3: Tsallis Entropy Gate
- **Files**: `care_asr/uncertainty/tsallis_entropy.py` (199 lines), `care_asr/uncertainty/gate.py` (136 lines)
- **Tests**: `tests/test_tsallis_entropy.py` — **7 tests PASS**, `tests/test_uncertainty_gate.py` — **3 tests PASS**
- Mathematical implementation verified correct (α=1/3 formula matches Checks doc)
- `TsallisUncertaintyGate` class with threshold=0.5, dynamic threshold support
- **Quality**: Solid. Best-quality module in the repo.

#### ❌ MISSING — T7: Real Correction Step (LLM Wiring)
- `src/correction/llm_corrector.py` — **0 bytes, empty file**
- No Qwen2.5 / Llama prompting code exists anywhere
- No CORRECT/WRONG/UNSURE three-way classification implemented
- This is the most critical unbuilt module.

#### ❌ MISSING — T10: UNSURE Fallback Gate
- `src/safety/unsure_gate.py` — **0 bytes, empty file**
- The safest + most defensible claim of the whole project (per Checks doc: "no competing system implements a refusal mechanism") has **no implementation**

#### ❌ MISSING — T5 Integration Checkpoint
- `tests/integration/test_t5_checkpoint.py` — **0 bytes, empty**
- `src/pipeline/stubs.py` — **0 bytes, empty**
- `src/pipeline/pipeline.py` — **0 bytes, empty**
- There is **no connected pipeline** even in stub form

---

### 2.2 Divya + Aarth Work

#### ✅ DONE — T2: Semantic FAISS Index Build
- **Script**: `scripts/build_semantic_index.py` (497 lines, full implementation)
- **Progress doc**: `divya_progress/T2_Semantic_Index_Builder.md` — thorough
- Uses `emilyalsentzer/Bio_ClinicalBERT` + `nishanth-augustai/rxnorm_data` from HuggingFace
- CUI mapping generated: `data/indices/cui_mapping.json` (864KB, 40,000 concepts)
- **Critical Gap**: `data/indices/faiss_umls.index` — **DOES NOT EXIST** on disk
  - The CUI mapping was saved, but the actual FAISS `.index` binary file was never committed / not present
  - This means semantic retrieval cannot currently run

#### ✅ DONE — T4: NER Entity Tagging Pipeline
- **Files**: `care_asr/ner/extractor.py` (408 lines), `care_asr/ner/span_aligner.py` (123 lines)
- **Script**: `scripts/run_ner_extraction.py` (77 lines)
- Uses `d4data/biomedical-ner-all` model (full taxonomy: MED/COND/ANA/TTP/PHI)
- Taxonomy mapping properly configured in `config.yaml`
- `SpanAligner` — O(N+M) two-pointer word alignment ✓
- **Tests**: `care_asr/tests/test_ner_extractor.py` — **3 tests PASS** (after our fix today)
- **Gap**: Script is written but has **not been run** on actual AfriSpeech data (data dir empty). NER output files do not exist in `outputs/`.
- **NER → M-WER bridge**: No code yet connects NER output to M-WER scorer (T1's reserved slot)

#### ✅ DONE — T8: Category Threshold Engine
- **Files**: `care_asr/thresholds/threshold_engine.py` (211 lines), `care_asr/validation/candidate_evaluator.py` (274 lines)
- **Tests**: `care_asr/tests/test_threshold_engine.py` — **9 tests PASS**; `care_asr/tests/test_candidate_evaluator.py` — **5 tests PASS**
- Category-specific thresholds configurable per MED/COND/ANA/TTP/PHI
- **Quality**: Good architecture, well tested. Ready for wiring.

#### ❌ MISSING — T6: Phonetic Retrieval Engine
- `src/retrieval/phonetic.py` — **0 bytes, empty**
- `scripts/build_phonetic_index.py` — **0 bytes, empty**
- `data/embeddings/phonetic/` — **EMPTY directory**
- `data/indices/medical_vocab.json` — **DOES NOT EXIST**
- Phonetic channel of dual retrieval is **completely unbuilt**

#### ❌ MISSING — T6 Phonetic Index Data
- Per the plan: "mine REAL drug-name pronunciations from AfriSpeech-200 using T4's entity tags"
- HuBERT model referenced in `models/hubert/` — **EMPTY directory**
- No audio segments extracted, no HuBERT embeddings generated

---

### 2.3 Shared / Integration Work

#### ⚠️ PARTIAL — Data Contracts (Pydantic schemas)
- `care_asr/contracts/asr_input.py` (77 lines) ✓
- `care_asr/contracts/retrieval_input.py` (101 lines) ✓  
- `care_asr/contracts/validated_output.py` (142 lines) ✓
- `care_asr/contracts/error_analysis_output.py` (126 lines) ✓
- `src/utils/schemas.py` (212 lines) — SEPARATE schema set (parallel, possibly inconsistent)
- **Issue**: Two parallel schema/contract systems exist (`care_asr/contracts/` and `src/utils/schemas.py`). These need to be reconciled before wiring.

#### ❌ MISSING — T5: Pipeline Integration (Stubs)
- `src/pipeline/pipeline.py` — **empty**
- No connected E2E pipeline exists
- Cannot run end-to-end even on dummy data

#### ❌ MISSING — T9: Second Integration Checkpoint
- `tests/integration/test_t9_checkpoint.py` — **empty**
- No dual-retrieval + correction pipeline run completed

#### ❌ MISSING — RRF Fusion
- `src/fusion/rrf.py` — **0 bytes, empty**
- The 20-line RRF implementation that the plan calls "the simplest step" is not written

#### ❌ MISSING — Semantic Retrieval Query Module
- `src/retrieval/semantic.py` — **0 bytes, empty**
- FAISS index was built offline (build script exists), but query code doesn't exist

#### ❌ MISSING — Demo App
- `demo/app.py` — **0 bytes, empty**
- No Gradio or CLI demo exists

#### ❌ MISSING — India Evaluation Scripts
- `scripts/run_india_eval.py` — **0 bytes, empty**
- `outputs/metrics/india/` — **empty directory**
- `data/raw/india/` — **empty directory**

#### ❌ MISSING — M-WER / Category Recall Implementation
- `src/evaluation/mwer.py` — **0 bytes, empty**
- `src/evaluation/wer.py` — **0 bytes, empty**
- `care_asr/evaluation/metrics_calculator.py` has `ErrorAnalysisEngine` class (55 lines) but no M-WER logic

---

## 3. Empty Files / Folders — Who Was Supposed to Fill Them

| File/Folder | Who | What It Needs |
| :--- | :--- | :--- |
| `src/asr/transcriber.py` | Ankit | Whisper inference wrapper using HF Transformers — loads model, processes audio, returns tokens + logits |
| `src/asr/confidence.py` | Ankit | Token-level confidence score extractor (bridges S3 probe → T3 entropy gate) |
| `src/retrieval/semantic.py` | Divya | FAISS query function using ClinicalBERT embedding of uncertain span |
| `src/retrieval/phonetic.py` | Divya | Double Metaphone / HuBERT phonetic search against phonetic index |
| `scripts/build_phonetic_index.py` | Divya | Build HuBERT-based phonetic FAISS index from drug-name audio |
| `src/fusion/rrf.py` | Divya / Ankit | 20-line RRF implementation merging two ranked lists |
| `src/correction/llm_corrector.py` | Ankit + Mahi | Qwen2.5-7B prompting with CORRECT/WRONG/UNSURE constraint |
| `src/safety/unsure_gate.py` | Ankit + Mahi | Three-way classifier gate — UNSURE → fallback to original token |
| `src/ner/tagger.py` | Aarth | Thin wrapper connecting M3 NER extractor to pipeline interface |
| `src/pipeline/stubs.py` | Ankit | Stub-wired pipeline returning correctly-shaped fake output |
| `src/pipeline/pipeline.py` | Ankit | Full connected E2E pipeline orchestrator |
| `src/evaluation/mwer.py` | Ankit + Aarth | Medical WER implementation using NER entity spans |
| `src/evaluation/wer.py` | Ankit | WER computation (currently in `metrics.py`, needs standalone module) |
| `scripts/run_baseline.py` | Ankit | Script to run baseline eval on real AfriSpeech-200 |
| `scripts/download_afrispeech.py` | Ankit | Download AfriSpeech-200 dataset from HuggingFace |
| `scripts/run_eval.py` | Ankit | Full pipeline evaluation script |
| `scripts/run_india_eval.py` | Aarth + Divya | India (EKA + Svarah) inference sweep |
| `demo/app.py` | Mahi | Gradio web interface for live demo |
| `tests/integration/test_t5_checkpoint.py` | Mahi | E2E stub pipeline test |
| `tests/integration/test_t9_checkpoint.py` | Mahi + Ankit | First real ablation number test |
| `tests/integration/test_t15_checkpoint.py` | Mahi | Full pipeline with UNSURE + India sweep |
| `tests/unit/test_retrieval.py` | Mahi | Semantic + phonetic retrieval unit tests |
| `tests/unit/test_fusion.py` | Mahi | RRF fusion unit tests |
| `tests/unit/test_correction.py` | Mahi | LLM corrector unit tests |
| `tests/unit/test_safety.py` | Mahi | UNSURE gate unit tests |
| `configs/asr.yaml` | Ankit | Whisper model name, device, generation settings |
| `configs/entropy.yaml` | Ankit | α=1/3, threshold value |
| `configs/ner.yaml` | Aarth | Model path, taxonomy mapping reference |
| `configs/correction.yaml` | Ankit | LLM endpoint, model name, prompt template |
| `configs/fusion.yaml` | Divya | RRF k constant, weights |
| `configs/safety.yaml` | Ankit | UNSURE threshold, fallback policy |
| `configs/evaluation.yaml` | Ankit | Metrics settings, output dir |
| `data/raw/afrispeech/` | Ankit | Download AfriSpeech-200 using `scripts/download_afrispeech.py` |
| `data/raw/india/` | Aarth / Divya | EKA + Svarah inference data |
| `data/indices/faiss_umls.index` | Divya | Run `python scripts/build_semantic_index.py` to generate |
| `data/indices/medical_vocab.json` | Divya | Medical vocabulary for phonetic search |
| `data/embeddings/phonetic/` | Divya | HuBERT embeddings for drug-name audio clips |
| `models/whisper/`, `models/biobert/`, `models/hubert/`, `models/qwen/` | Team | HuggingFace model cache dirs — populated automatically on first load |
| `notebooks/ankit/`, `mahi/`, `aarth/`, `divya/` | Each person | Exploration notebooks (lower priority, optional) |
| `outputs/metrics/baseline/`, `ablation/`, `india/` | Team | Populated by evaluation scripts after running |

---

## 4. Overall Task Completion Status (Honest Accounting)

| Task | Plan Definition | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **S1** Interface Lock | Data contracts defined | ✅ Done | `care_asr/contracts/`, `src/utils/schemas.py` |
| **S2** Scope + Env | Configs and env working | ✅ Done | `pyproject.toml`, `.env.example`, `config.yaml` |
| **S3** Whisper Probe | Logit extraction confirmed | ✅ Done | 4 tests pass |
| **T1** Baseline Harness | WER/M-WER on AfriSpeech | ⚠️ Partial | Runs but on **synthetic data only**, real AfriSpeech not downloaded, WER = 1.0 on garbage output |
| **T2** Semantic Index | ClinicalBERT + FAISS built | ⚠️ Partial | Build script done ✓, CUI mapping done ✓, **FAISS .index file missing** |
| **T3** Entropy Gate | Tsallis implemented + tested | ✅ Done | 10 tests pass, solid |
| **T4** NER Pipeline | BioBERT tagging on AfriSpeech | ⚠️ Partial | Code done ✓, **never run on real data**, output files missing |
| **T5** First Integration | Stub pipeline E2E works | ❌ Missing | All pipeline files empty |
| **T6** Phonetic Retrieval | HuBERT + phonetic FAISS | ❌ Missing | Script empty, index missing, vocab missing |
| **T7** LLM Correction | Qwen2.5 CORRECT/WRONG/UNSURE | ❌ Missing | `llm_corrector.py` empty |
| **T8** Category Thresholds | Per-category threshold engine | ✅ Done | 9 + 5 tests pass |
| **T9** 2nd Integration | Dual retrieval + correction E2E | ❌ Missing | No pipeline exists |
| **T10** UNSURE Fallback | Three-way safety gate | ❌ Missing | `unsure_gate.py` empty |
| **T11** Error Analysis | Category recall from NER output | ⚠️ Partial | `ErrorAnalysisEngine` class exists, not wired to data |
| **T12** Latency Pass | Batching + latency measurement | ❌ Missing | No profiling code exists |
| **T13** QLoRA (optional) | Conditional fine-tune | ❌ Skip | Explicitly optional per plan |
| **T14** India Sweep | EKA + Svarah inference | ❌ Missing | Script empty, data missing |
| **T15** 3rd Integration | Full pipeline + India | ❌ Missing | Everything upstream missing |
| **T16** Ablation Table | All 6 rows with real numbers | ❌ Missing | No numbers exist yet |
| **T17** Demo + Report | Gradio demo + writeup | ❌ Missing | `demo/app.py` empty |
| **T18** Final Freeze | All verified and locked | ❌ Missing | — |

**Honest Score: 6 of 21 tasks complete (28.5%), 3 tasks partial (14%), 11 tasks missing (52%)**

The previous assessment saying "21/21 100% complete" was **incorrect** — it confused file existence with file content. Every file in `src/` (except `evaluation/` and `utils/`) is **0 bytes empty**.

---

## 5. Critical Path to Completion — Exact Priority Order

### PRIORITY 1: Get Real Data Running (Unblocks Everything)

**Ankit — Do this first:**
```bash
# Download AfriSpeech-200 and run real baseline
python scripts/download_afrispeech.py
python scripts/run_baseline.py  # Write this script
```
Until AfriSpeech is downloaded and real WER numbers exist, the entire project has no ground truth to improve on.

**Divya — Run the semantic index builder:**
```bash
python scripts/build_semantic_index.py
# This will generate data/indices/faiss_umls.index (the missing file)
```

---

### PRIORITY 2: Ankit + Mahi — Build T5 (Stub Pipeline)

This is the minimum viable integration checkpoint. It doesn't need real models — just correctly-shaped fake outputs flowing end-to-end. Write `src/pipeline/stubs.py` and `tests/integration/test_t5_checkpoint.py`.

---

### PRIORITY 3: Build the 4 Missing Core Modules (in dependency order)

1. **`src/asr/transcriber.py`** — Whisper wrapper (Ankit, 1 day)
2. **`src/retrieval/semantic.py`** — FAISS query function (Divya, ~2 hours after index exists)
3. **`src/fusion/rrf.py`** — 20-line RRF (anyone, 1 hour)
4. **`src/correction/llm_corrector.py`** — Qwen2.5 prompting (Ankit + Mahi, 1-2 days)
5. **`src/safety/unsure_gate.py`** — Three-way gate (Ankit + Mahi, 1 day)

---

### PRIORITY 4: Divya — T6 Phonetic Retrieval

The phonetic channel is entirely unbuilt. Options per the Checks doc:
- Use Whisper encoder hidden states (easier, no extra model download)
- Or HuBERT (more accurate but extra 380MB model)

The plan says: "use frozen Whisper encoder hidden states, not separately trained HuBERT" — this is the simpler path.

---

### PRIORITY 5: Connect M-WER (Aarth + Ankit)

Run NER script on real AfriSpeech → use output spans to implement `src/evaluation/mwer.py` → plug into `baseline_metrics.json`. Without M-WER, there is no "improvement" to claim.

---

## 6. Quality Issues Found

| Issue | Severity | Location |
| :--- | :--- | :--- |
| `results/predictions.json` has `"BEEEE..."` garbage for 2/3 utterances | High | T1 baseline ran on bad synthetic audio |
| `data/indices/faiss_umls.index` missing | High | T2 incomplete — retrieval cannot run |
| Two parallel schema systems (`care_asr/contracts/` vs `src/utils/schemas.py`) | Medium | Will cause import confusion when wiring |
| All `src/` module files (transcriber, retrieval, fusion, correction, safety, pipeline) empty | Critical | No runnable pipeline exists |
| All `configs/*.yaml` except 4 are empty | Medium | Pipeline cannot load config at runtime |
| No real AfriSpeech data downloaded | High | All baseline numbers are synthetic placeholders |
| `ruff` shows trailing whitespace in `scripts/run_ner_extraction.py` | Low | Minor style issue |

---

## 7. Checks Doc Claims — Verified Implementation Status

The `CARE-ASR Checks.docx` specifies what must be demonstrable by the end. Here's where each stands:

| Checks Doc Claim | Implementation Status |
| :--- | :--- |
| Tsallis entropy gate (α=1/3) — "Include as core V1" | ✅ Implemented, tested |
| BioBERT NER boundary detector — "Include" | ✅ Implemented, tested |
| Dual FAISS retrieval (Semantic + Phonetic) — "Include" | ⚠️ Semantic script exists, FAISS file missing; Phonetic entirely missing |
| RRF fusion — "Include, 20-line implementation" | ❌ Not written |
| Llama/Qwen CORRECT/WRONG/UNSURE gate — "Include, strongest claim" | ❌ Not written |
| Clinician-verified gold subset (100-200 utterances) — "Non-negotiable" | ❌ Not started |
| Per-layer attribution logging — "Non-negotiable" | ❌ Not implemented |
| Medical-term false discovery rate — "Include" | ❌ Not implemented |
| Ablation table (6 rows, all real numbers) | ❌ No real numbers exist yet |

---

## 8. Context Summary for External LLM (What Happened, What Exists, What's Needed)

**What CARE-ASR is**: A modular post-processing pipeline that corrects Whisper ASR output for accented/code-mixed clinical speech. 8 modules: ASR → Entropy Gate → NER → Dual Retrieval (semantic + phonetic) → RRF Fusion → LLM Correction → Safety Gate (UNSURE fallback) → Output/Metrics.

**What is fully working**:
- Tsallis entropy computation and gate (`care_asr/uncertainty/`) — tested, correct math
- BioBERT NER extractor (`care_asr/ner/`) — tested, not yet run on real data
- Category threshold engine (`care_asr/thresholds/`) — tested
- Candidate evaluator (`care_asr/validation/`) — tested
- Semantic index BUILD script (`scripts/build_semantic_index.py`) — but output `.index` file missing
- Data contracts / Pydantic schemas (`care_asr/contracts/`) — defined
- Baseline eval harness (`src/evaluation/`) — runs but on placeholder data only
- Whisper logit probe (`care_asr/probes/`) — confirmed working

**What is entirely unbuilt (all empty files)**:
- `src/asr/transcriber.py` — Whisper wrapper
- `src/retrieval/semantic.py` — FAISS semantic query
- `src/retrieval/phonetic.py` — Phonetic search
- `src/fusion/rrf.py` — RRF fusion (20 lines needed)
- `src/correction/llm_corrector.py` — Qwen2.5 CORRECT/WRONG/UNSURE
- `src/safety/unsure_gate.py` — Three-way safety gate
- `src/pipeline/pipeline.py` — End-to-end orchestrator
- `demo/app.py` — Gradio demo
- All `tests/integration/*.py` — Integration checkpoints
- All `tests/unit/test_retrieval/fusion/correction/safety.py` — Retrieval/fusion/correction tests
- `src/evaluation/mwer.py` + `wer.py` — Medical WER implementation

**Data that doesn't exist yet**:
- AfriSpeech-200 audio (`data/raw/afrispeech/` — empty)
- FAISS semantic index (`data/indices/faiss_umls.index` — missing, build script exists)
- Phonetic FAISS index and medical vocab (`data/indices/medical_vocab.json` — missing)
- India evaluation data (`data/raw/india/` — empty)
- Real baseline numbers (current WER=1.0 on garbage synthetic audio)

**LLM / Model selection** (per Checks doc & Final doc):
- Correction LLM: **Qwen2.5-7B-Instruct** is PRIMARY (not Llama, not OpenBioLLM)
- NER: `d4data/biomedical-ner-all` (not BC5CDR BioBERT — this was swapped)
- Phonetic encoder: Whisper encoder hidden states (NOT HuBERT, per Checks doc simplification)
- Semantic encoder: `emilyalsentzer/Bio_ClinicalBERT`
- Entropy parameter: α = 1/3 (confirmed correct in code ✓)

**Tests**: 48 tests pass across `care_asr/tests/` and `tests/`. But the `tests/unit/` suite (retrieval, fusion, correction, safety) is entirely empty. Integration test files are empty.

**Key divergences between code and plan**:
1. The plan uses `Qwen2.5-7B` as primary LLM — code has no LLM module at all
2. Plan says phonetic retrieval uses Whisper encoder states — code has `scripts/build_phonetic_index.py` empty
3. Plan's T5 checkpoint requires stub-wired E2E pipeline — pipeline files are empty
4. Ablation table (T16) requires 6 rows of real numbers — zero real numbers exist yet
