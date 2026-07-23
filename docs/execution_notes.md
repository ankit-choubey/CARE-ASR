# CARE-ASR Execution Plan & Task Mapping (S1..S3, T1..T18)

This document maps all execution tasks (S1 through S3 setup tasks, and T1 through T18 implementation tasks) derived directly from the verified `CARE-ASR_Execution_Plan.docx`.

---

## 1. Team Role & Ownership Mapping

| Source Role                                   | Responsibility Scope                                                                      | Team Member                                       |
| :-------------------------------------------- | :---------------------------------------------------------------------------------------- | :------------------------------------------------ |
| **Person C** (Integration & Infra Lead) | Baseline harness, WER/M-WER scoring, pipeline orchestration, continuous daily merges.     | **Ankit Choubey**                           |
| **Person A** (ML Research Lead)         | Tsallis entropy gating, few-shot/QLoRA LLM correction, UNSURE safety fallbacks.           | **Ankit Choubey** & **Mahi Nandini**  |
| **Testing Lead** (QA / Verification)    | Pytest unit tests, ablation checkpoints, latency benching, interface contract validation. | **Mahi Nandini**                            |
| **Person B** (Retrieval Engineer)       | Dense ClinicalBERT FAISS index, Double Metaphone index, RRF fusion engine.                | **Divya** (Lead), **Aarth** (Support) |
| **Person D** (Domain / Data Specialist) | BioBERT NER entity tagging, category thresholds, error analysis, benchmark evaluation.    | **Aarth** (Lead), **Divya** (Support) |

---

## 2. Complete Task Execution Flow (S1..S3 -> T1..T18)

```
[S1: Interface Lock] ──► [S2: Scope Lock] ──► [S3: Logit Probe]
                                                     │
                                                     ▼
                                          [T1: Baseline Harness]
                                                     │
                      +------------------------------+------------------------------+
                      │                                                             │
                      ▼                                                             ▼
         [T2: Semantic FAISS Index]                                      [T3: Entropy Gate Impl]
                      │                                                             │
                      ▼                                                             ▼
         [T4: NER Tagging Pipeline]                                      [T5: Baseline + Entropy Integration]
                      │                                                             │
                      +------------------------------+------------------------------+
                                                     │
                                                     ▼
                                          [T6: Phonetic Matcher]
                                                     │
                                                     ▼
                                          [T7: RRF Fusion Engine]
                                                     │
                                                     ▼
                                          [T8: Candidate Retriever Test]
                                                     │
                                                     ▼
                                          [T9: LLM Post-Correction Wire]
                                                     │
                                                     ▼
                                          [T10: Safety Gate & Fallbacks]
                                                     │
                                                     ▼
                                          [T11: End-to-End Pipeline Integration]
                                                     │
                                                     ▼
                                          [T12: Evaluation Harness & WER/M-WER]
                                                     │
                                                     ▼
                                          [T13: Ablation Study Suite]
                                                     │
                                                     ▼
                                          [T14: Latency & Memory Profiling]
                                                     │
                                                     ▼
                                          [T15: Error Analysis & Category Recall]
                                                     │
                                                     ▼
                                          [T16: Demo Script & CLI Interface]
                                                     │
                                                     ▼
                                          [T17: Final Report & Paper Drafting]
                                                     │
                                                     ▼
                                          [T18: Project Lock & Code Release]
```

---

## 3. Comprehensive Task Specifications

### Phase 0: Setup Tasks (Day 1)

* **S1: Interface-First Meeting & Data Contract Lock**

  - **Owner**: Whole Team
  - **Dependencies**: None
  - **Deliverable**: Lock exact function signatures and dataclasses in `docs/interface_contract.md`.
* **S2: Scope & Environment Verification**

  - **Owner**: Whole Team
  - **Dependencies**: S1
  - **Deliverable**: Verify PyTorch, CUDA, FAISS, and HuggingFace environment access across GPU nodes.
* **S3: Whisper Logit Extraction Probe**

  - **Owner**: Ankit
  - **Dependencies**: S1
  - **Deliverable**: Verify per-token logit distribution extraction from `whisper-medium` decoder outputs on 1 real audio clip.

---

### Phase 1: Core Foundation & Retrieval (Days 1 - 4)

* **T1: Baseline ASR Evaluation Harness**

  - **Owner**: Ankit
  - **Dependencies**: S3
  - **Deliverable**: Run standard Whisper-medium on AfriSpeech-200 clinical split; record initial WER and M-WER baseline.
* **T2: Semantic FAISS Vector Index Construction**

  - **Owner**: Divya (Lead), Aarth (Support)
  - **Dependencies**: S1
  - **Deliverable**: Build 768-dim ClinicalBERT embedding index over RxNorm and UMLS concepts using FAISS.
* **T3: Tsallis Entropy Gate Implementation**

  - **Owner**: Ankit + Mahi
  - **Dependencies**: T1, S3
  - **Deliverable**: Implement $H_q(P)$ formula, unit-test on logit outputs from T1 baseline.
* **T4: NER Entity Tagging Pipeline**

  - **Owner**: Aarth (Lead), Divya (Support)
  - **Dependencies**: T1
  - **Deliverable**: Annotate AfriSpeech clinical test set with BioBERT to produce ground-truth entity spans (MED, COND, ANA, TTP).
* **T5: Baseline + Entropy Gate Integration Milestone**

  - **Owner**: Ankit + Mahi
  - **Dependencies**: T3, T4
  - **Deliverable**: Merge entropy gate with NER tagger; verify low-confidence medical entity detection recall.

---

### Phase 2: Hybrid Retrieval & Fusion (Days 4 - 7)

* **T6: Phonetic Search Engine Implementation**

  - **Owner**: Divya
  - **Dependencies**: T2
  - **Deliverable**: Implement Double Metaphone encoder and Levenshtein distance dictionary search.
* **T7: Reciprocal Rank Fusion (RRF) Engine**

  - **Owner**: Divya
  - **Dependencies**: T2, T6
  - **Deliverable**: Combine semantic and phonetic candidate lists into unified `FusionCandidate` output.
* **T8: Retrieval Unit Test & Recall Verification**

  - **Owner**: Mahi
  - **Dependencies**: T7
  - **Deliverable**: Pytest suite verifying Top-5 candidate recall on synthetic medical mistranscriptions.

---

### Phase 3: LLM Correction & Safety (Days 7 - 9)

* **T9: LLM Post-Correction Wiring**

  - **Owner**: Ankit
  - **Dependencies**: T7, T8
  - **Deliverable**: Prompt design and API connection to local Qwen2.5-7B-Instruct(few-shot inference).
* **T10: Medical Safety Gate & Fallback Logic**

  - **Owner**: Mahi + Ankit
  - **Dependencies**: T9
  - **Deliverable**: Implement Levenshtein ratio check ($d \le 0.45$) and UNSURE fallback triggers.
* **T11: End-to-End CARE-ASR Integration**

  - **Owner**: Ankit
  - **Dependencies**: T10
  - **Deliverable**: Connect Modules 1 through 9 into unified `CareAsrPipeline`.

---

### Phase 4: Evaluation, Benchmarking & Release (Days 9 - 12)

* **T12: Full System Benchmark & WER/M-WER Scoring**

  - **Owner**: Ankit + Mahi
  - **Dependencies**: T11
  - **Deliverable**: Compute final WER, M-WER, and Category Recall across AfriSpeech test splits.
* **T13: Comprehensive Ablation Study Suite**

  - **Owner**: Mahi
  - **Dependencies**: T12
  - **Deliverable**: Execute ablations (No-Entropy, No-Phonetic, No-Safety) to prove contribution of each module.
* **T14: Latency & Memory Profiling**

  - **Owner**: Ankit
  - **Dependencies**: T11
  - **Deliverable**: Profile P50/P95/P99 latency and VRAM usage.
* **T15: Error Analysis & Category Recall Breakdown**

  - **Owner**: Aarth
  - **Dependencies**: T12
  - **Deliverable**: Quantitative analysis of recovered vs missed clinical terms per entity type.
* **T16: Interactive Demo & CLI Tooling**

  - **Owner**: Mahi + Divya
  - **Dependencies**: T11
  - **Deliverable**: CLI script `python -m care_asr.demo` for real-time wav processing.
* **T17: Final Research Paper & Documentation Assembly**

  - **Owner**: Aarth + Whole Team
  - **Dependencies**: T12, T13, T15
  - **Deliverable**: Complete paper draft and documentation sync.
* **T18: Code Freeze & Version 0.1.0 Release**

  - **Owner**: Whole Team
  - **Dependencies**: T16, T17
  - **Deliverable**: Final tag release `v0.1.0`.
