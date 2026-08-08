# Aarth's Progress Report (End-to-End)

## Overview
This document tracks the progress, implementation details, and outcomes for the tasks assigned to **Aarth** under the CARE-ASR Final Execution Plan. 

### Role Mapping
- **Role:** Data Lead (Reports to Ankit, manages Divya)
- **Primary Domain:** NLP entity mapping, validation algorithms, threshold logic.

---

## 1. T4: NER Entity Tagging (Completed)
**Goal:** Build and run the BioBERT NER pipeline to produce MED/COND/ANA/TTP/PHI reference spans required for the M-WER scoreboard.

**Implementation Details:**
- Created `BioBertNERExtractor` to perform token classification over raw transcripts.
- Integrated `d4data/biomedical-ner-all` model from HuggingFace.
- Configured a `taxonomy_mapping` in `config.yaml` to elegantly handle all 83 potential model tags, mapping only relevant ones to CARE-ASR categories (MED, COND, ANA, TTP, PHI) and ignoring unrelated tags (e.g., ACTIVITY, COLOR).
- Developed `SpanAligner` to precisely map subtoken character offsets to Whisper ASR word boundaries (start_time, end_time).
- Created and executed `scripts/run_ner_extraction.py` to run the pipeline over test data and export `afrispeech_reference_ner_tags.json`.

**Outcome:** NER tags generated successfully and handed off to Ankit for the T1 Baseline Harness and to Divya for the T6 Phonetic extraction.

---

## 2. T8: Category Thresholds & Decision Engine (Completed Programmatically)
**Goal:** Implement the logic to reject/accept retrieved candidates based on semantic, phonetic, and confidence scores across different categories.

**Implementation Details:**
- Developed the `CategoryThresholdEngine` to load distinct, strict acceptance rules per medical category (e.g., PHI requires 0.92 semantic similarity, whereas COND requires 0.78).
- Implemented `CandidateEvaluator` to rank the raw FAISS retrieval candidates outputted by Divya's pipeline.
- Established the integration point for Ankit's Tsallis entropy gate.

**Outcome:** The code foundation is 100% complete and tested. The exact numeric thresholds are currently drafted using baseline estimations.

---

## 3. Technical Highlights & Engineering Choices
- **HuggingFace Pipeline Engineering:** Safely implemented PyTorch GPU inference with automatic CPU fallback logic to prevent Out-Of-Memory pipeline crashes. 
- **O(N+M) Span Alignment:** Designed the character-to-word aligner (`SpanAligner`) to operate in linear time without nested loops, successfully converting raw BioBERT character bounds into timestamped Whisper audio boundaries even during partial overlap or multi-word span situations.
- **Robust Exception Handling:** Implemented a unified error inheritance tree (`CAREASRError`) ensuring any model label mismatches or schema validation failures halt the CI/CD pipeline immediately rather than passing silent corrupted data to Divya's retrieval nodes.
- **Data Contract Enforcement:** Integrated rigid Pydantic models (`ASRTranscriptInput`, `WordAlignment`) to guarantee type safety between the ASR baseline boundaries and the downstream semantic mapping.
- **Dynamic Configuration:** Decoupled all medical taxonomy definitions and numeric threshold limits into a central `config.yaml` to allow for rapid ablation tuning without requiring code redeployments.

---

## 4. Pending Collaborations & Next Steps
- **T2 Validation:** Awaiting Divya's semantic FAISS index to validate it using 5 known drug names.
- **T6 Validation:** Awaiting Divya's phonetic FAISS index to validate it with known misheard transcriptions.
- **T11 Threshold Tuning:** Awaiting Ankit's first ablation numbers (T9) to empirically tune the draft thresholds in `config.yaml`.
- **T14 India Eval Consolidation:** Awaiting Divya's execution over EKA + Svarah datasets to merge into a final results table.
- **T16 Ablation Table Freeze:** Must sit with Ankit to lock the final results matrix.
