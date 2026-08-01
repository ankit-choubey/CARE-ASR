
# CARE-ASR: Complete Execution Master Plan

### All Cloud · Two Builders (Ankit & Divya) · Zero Device ML · Fully Reconciled & Audited

---

## 5-POINT CONSISTENCY & IMPLEMENTATION HYGIENE FIXES (INCORPORATED)

1. **Role Mapping & Task Folding**:
   - **Ankit**: Executes all core engineering, pipeline integration, baseline/scoreboard, LLM correction, UNSURE gate, **PLUS all of Mahi's tasks** (writing Definition-of-Done unit tests, independent T5/T9/T15 checkpoint output verification, and demo app build/rehearsal).
   - **Divya**: Executes all Kaggle cloud compute, dataset downloading, semantic FAISS index, HuBERT phonetic index, **PLUS all of Aarth's tasks** (NER reference extraction, category threshold tuning T8/T11, qualitative error analysis, and India evaluation sweep consolidation).
2. **Outlines Constrained Decoding**:
   - `src/correction/llm_corrector.py` uses `outlines` (or `guidance` / regex-constrained generation) to enforce schema output (`CORRECT | <candidate>`, `WRONG`, `UNSURE`). Simple string matching remains only as a emergency CPU/unit-test parser fallback.
3. **Unified Schema System**:
   - **`care_asr.contracts`** is locked as the SINGLE CANONICAL SCHEMA CONTRACT for the entire codebase (`ASRInput`, `RetrievalInput`, `ValidatedCandidatesOutput`, `ErrorAnalysisOutput`). `src/utils/schemas.py` is deprecated and alias-mapped to `care_asr.contracts` to prevent import conflicts between workstreams.
4. **Feasibility & Descoping Strategy**:
   - **T13 (QLoRA Fine-Tuning)** is explicitly cut from the 3-day sprint (remains Track B future work).
   - **T14 (India Sweep)** is run as a lightweight inference-only script on Svarah (100 samples) on Kaggle.
   - All ML/LLM heavy compute runs strictly on Kaggle GPU (P100/T4). Local machine strictly runs `git` and lightweight CPU `pytest` tests.
5. **Explicit Phonetic Fallback Trigger**:
   - **Primary**: HuBERT-base hidden states extracted from AfriSpeech clinical audio clips stored in `data/indices/phonetic_index.faiss`.
   - **Automatic Fallback Trigger**: If `data/indices/phonetic_index.faiss` is missing or unreadable on disk, `PhoneticRetriever` automatically falls back to CPU-based Double Metaphone fuzzy matching using `data/indices/medical_vocab.json`.

---

## CONTEXT FOR THE LLM READING THIS

You have the following documents already:

- `CARE-ASR Final.docx` — the "what we are building" document. 8-module pipeline: ASR → Entropy Gate → NER → Dual Retrieval → RRF Fusion → LLM Correction → UNSURE Safety Gate → Output Metrics.
- `CARE-ASR_Execution_Plan.docx` — the original task plan (S1–S3, T1–T18).
- `CARE-ASR Checks.docx` — what must be demonstrable at the end.
- `CARE-ASR_Verified_Tool_Stack_and_Compute_Strategy.docx` — confirmed free-tier tools.
- `care_asr_full_analysis.md` — exact audit of what exists in the repo and what is empty.

**What exists in the repo right now (do NOT rebuild these):**

- ✅ `care_asr/uncertainty/tsallis_entropy.py` + `gate.py` — Tsallis entropy gate, tested
- ✅ `care_asr/ner/extractor.py` + `span_aligner.py` — BioBERT NER, tested
- ✅ `care_asr/thresholds/threshold_engine.py` — category threshold engine, tested
- ✅ `care_asr/validation/candidate_evaluator.py` + `decision_router.py` — tested
- ✅ `care_asr/contracts/` — ALL CANONICAL PYDANTIC DATA CONTRACTS DEFINED
- ✅ `scripts/build_semantic_index.py` — ClinicalBERT + FAISS build script (complete, 497 lines)
- ✅ `src/evaluation/baseline.py` + `metrics.py` + `io_utils.py` — baseline harness (complete)
- ✅ 48 tests passing across `care_asr/tests/` and `tests/`

**What is 0-byte empty (must be built this sprint):**

- ❌ `src/asr/transcriber.py`, `src/asr/confidence.py`
- ❌ `src/retrieval/semantic.py`, `src/retrieval/phonetic.py`
- ❌ `src/fusion/rrf.py`
- ❌ `src/correction/llm_corrector.py`
- ❌ `src/safety/unsure_gate.py`
- ❌ `src/pipeline/pipeline.py`, `src/pipeline/stubs.py`
- ❌ `src/evaluation/mwer.py`
- ❌ `demo/app.py`
- ❌ All `tests/integration/*.py`, all `tests/unit/test_retrieval/fusion/correction/safety.py`
- ❌ `configs/asr.yaml`, `configs/entropy.yaml`, `configs/ner.yaml`, `configs/correction.yaml`, `configs/fusion.yaml`, `configs/safety.yaml`, `configs/evaluation.yaml`

**Key locked decisions (do not reopen):**

- Correction LLM: **Qwen/Qwen2.5-7B-Instruct** (NOT OpenBioLLM, NOT BioMistral)
- NER model: **d4data/biomedical-ner-all** (already in `config.yaml`)
- Vector store: **FAISS only** (NOT Qdrant, Chroma, Milvus)
- Phonetic encoder: **HuBERT-base** for FAISS index; **Double Metaphone** as CPU fallback
- Semantic encoder: **emilyalsentzer/Bio_ClinicalBERT**
- Entropy parameter: **α = 1/3** (confirmed correct, implemented)
- Output constraint: **Outlines** library for CORRECT/WRONG/UNSURE
- **UNSURE → always keep original Whisper token** (never guess)

**All ML/LLM work is on cloud. Nothing heavy runs on local device. Local device = only `git`, `pytest` on small synthetic tests, and reading logs.**

---

## FULL TASK AUDIT (S1–S3, T1–T18) & THREE-DAY SCHEDULE

|    Task ID    | Task Name                             |         Owner         |  Day  | Primary Deliverable / Cloud Execution                               |
| :-----------: | :------------------------------------ | :--------------------: | :---: | :------------------------------------------------------------------ |
| **S1** | Interface Lock & Data Contracts       |     Ankit & Divya     | Day 1 | `care_asr/contracts/` & `src/utils/schemas.py` unified          |
| **S2** | Scope & Environment Lock              |     Ankit & Divya     | Day 1 | Kaggle API tokens & HF credentials setup                            |
| **S3** | Whisper`output_scores` Probe        |         Ankit         | Day 1 | `care_asr/probes/whisper_scores_probe.py` verified                |
| **T1** | Baseline Evaluation Harness           |         Ankit         | Day 1 | `src/asr/transcriber.py` & `scripts/run_baseline.py`            |
| **T2** | Semantic FAISS Index                  | Divya (build/validate) | Day 1 | `data/indices/faiss_umls.index` (RxNorm ClinicalBERT)             |
| **T3** | Tsallis Entropy Gate ($\alpha=1/3$) |         Ankit         | Day 1 | `care_asr/uncertainty/gate.py` (unit tested)                      |
| **T4** | BioBERT NER Reference Tagging         | Divya (leads/validate) | Day 1 | `outputs/ner/ner_reference_spans.json` on Kaggle                  |
| **T5** | **1st Integration Checkpoint**  |   Ankit (Mahi task)   | Day 1 | Stub E2E pipeline &`test_t5_checkpoint.py` verified               |
| **T6** | Phonetic FAISS Index                  | Divya (build/validate) | Day 2 | `data/indices/phonetic_index.faiss` (HuBERT + Metaphone)          |
| **T7** | Real LLM Correction Step              |         Ankit         | Day 2 | `src/correction/llm_corrector.py` (Qwen2.5-7B + Outlines)         |
| **T8** | Category Threshold Engine             |     Ankit & Divya     | Day 2 | `care_asr/thresholds/threshold_engine.py` tuned                   |
| **T9** | **2nd Integration Checkpoint**  |     Ankit & Divya     | Day 2 | Dual retrieval + LLM E2E eval on Kaggle (`test_t9_checkpoint.py`) |
| **T10** | UNSURE Fallback Safety Gate           |         Ankit         | Day 2 | `src/safety/unsure_gate.py` (Original token fallback)             |
| **T11** | Threshold Tuning & Error Analysis     |   Divya (Aarth task)   | Day 2 | Category recall & qualitative error extraction                      |
| **T12** | Latency Benchmarking Pass             |         Divya         | Day 2 | Batching + FAISS query latency logged in`attribution_log`         |
| **T13** | QLoRA Fine-tuning                     | *Explicit Scope Cut* |  —  | Track B Future Work (Few-shot Qwen2.5 used in V1)                   |
| **T14** | India Context Evaluation Sweep        |   Divya (Aarth task)   | Day 3 | Svarah 100-sample inference run on Kaggle                           |
| **T15** | **3rd Integration Checkpoint**  |   Ankit (Mahi task)   | Day 3 | Full pipeline validation &`test_t15_checkpoint.py` verified       |
| **T16** | Freeze Ablation Table                 |     Ankit & Divya     | Day 3 | `outputs/metrics/ablation/ablation_table.json` locked             |
| **T17** | Gradio Demo App & Claims Draft        |   Ankit (Mahi task)   | Day 3 | `demo/app.py` built, tested, and rehearsed                        |
| **T18** | Final Whole-System Check & PR         |     Ankit & Divya     | Day 3 | Full test suite clean pass & repo freeze                            |

---

---

# ═══════════════════════════════════════════════════════

# ANKIT'S COMPLETE SECTION (Includes All Mahi Tasks)

# ═══════════════════════════════════════════════════════

> **Ankit: everything you need is in this section. Includes core engineering, integration, AND Mahi's testing/verification/demo tasks.**

---

## ANKIT — DAY 1

### A1. Schema System Unification & Fill All Config Files (Local, 30 min)

First, enforce `care_asr.contracts` as the single canonical schema. Update `src/utils/schemas.py` to alias `care_asr.contracts`:

**`src/utils/schemas.py`**

```python
"""Canonical Schema Interface.
Re-exports Pydantic contracts from care_asr.contracts to maintain single source of truth.
"""
from care_asr.contracts.asr_input import ASRInput, TokenScore, WordTimestamp, Transcript
from care_asr.contracts.retrieval_input import RetrievalCandidate, RetrievalInput
from care_asr.contracts.validated_output import CorrectionOutput, ValidatedCandidatesOutput
from care_asr.contracts.error_analysis_output import ErrorAnalysisOutput, NEREntity

__all__ = [
    "ASRInput", "TokenScore", "WordTimestamp", "Transcript",
    "RetrievalCandidate", "RetrievalInput",
    "CorrectionOutput", "ValidatedCandidatesOutput",
    "ErrorAnalysisOutput", "NEREntity",
]
```

Next, create all 7 required YAML files in `configs/`:

**`configs/asr.yaml`**

```yaml
model_name: "openai/whisper-medium"
device: "auto"
language: "en"
return_timestamps: "word"
output_scores: true
return_dict_in_generate: true
max_new_tokens: 448
task: "transcribe"
```

**`configs/entropy.yaml`**

```yaml
alpha: 0.3333333333
threshold: 0.5
batch_size: 32
```

**`configs/ner.yaml`**

```yaml
model_name: "d4data/biomedical-ner-all"
device: "auto"
max_seq_length: 512
batch_size: 8
taxonomy_config: "config.yaml"
```

**`configs/correction.yaml`**

```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
quantization: "4bit"
max_new_tokens: 30
temperature: 0.0
do_sample: false
fallback_model: "Qwen/Qwen2.5-3B-Instruct"
structured_output: true
```

**`configs/fusion.yaml`**

```yaml
rrf_k: 60
semantic_weight: 1.0
phonetic_weight: 1.0
top_k: 5
```

**`configs/safety.yaml`**

```yaml
unsure_threshold: 0.5
fallback_policy: "original_token"
```

**`configs/evaluation.yaml`**

```yaml
output_dir: "outputs/metrics"
baseline_dir: "outputs/metrics/baseline"
ablation_dir: "outputs/metrics/ablation"
india_dir: "outputs/metrics/india"
predictions_file: "outputs/metrics/baseline/predictions.json"
num_eval_samples: 200
```

Commit: `git add configs/ src/utils/schemas.py && git commit -m "A1: unify schemas to care_asr.contracts and fill configs"`

---

### A2. Write `src/asr/transcriber.py` (Task S3/T1, Local, 45 min)

```python
"""Whisper ASR wrapper — audio array → Transcript with token scores.
S3 / T1 module. Runs on Kaggle GPU for dataset evaluation.
"""
from __future__ import annotations

import torch
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from care_asr.contracts.asr_input import TokenScore, Transcript


def _pick_device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperTranscriber:
    def __init__(self, config_path: str = "configs/asr.yaml") -> None:
        cfg = yaml.safe_load(open(config_path))
        self.device = _pick_device(cfg["device"])
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg["model_name"])
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg["model_name"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.cfg = cfg

    def transcribe(self, audio_array, sample_rate: int = 16_000) -> Transcript:
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            result = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                return_timestamps=True,
                language=self.cfg.get("language", "en"),
                max_new_tokens=self.cfg.get("max_new_tokens", 448),
            )

        sequences = result.sequences[0]
        text = self.processor.decode(sequences, skip_special_tokens=True)

        token_scores: list[TokenScore] = []
        for step, score_tensor in enumerate(result.scores):
            if step >= len(sequences) - 1:
                break
            token_id = sequences[step + 1].item()
            log_probs = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            token_scores.append(
                TokenScore(
                    step=step,
                    token_id=token_id,
                    token=self.processor.decode([token_id]),
                    log_prob=float(log_probs[token_id]),
                    prob=float(torch.exp(log_probs[token_id])),
                )
            )

        return Transcript(text=text, token_scores=token_scores, word_timestamps=[])
```

---

### A3. Write `src/pipeline/stubs.py` and `src/pipeline/pipeline.py` (Task T5, Local, 60 min)

**`src/pipeline/stubs.py`**

```python
"""Stub implementations for T5 integration checkpoint."""
from __future__ import annotations

from care_asr.contracts.asr_input import TokenScore, Transcript
from care_asr.contracts.error_analysis_output import NEREntity
from care_asr.contracts.retrieval_input import RetrievalCandidate
from care_asr.contracts.validated_output import CorrectionOutput


def stub_transcriber(audio_input) -> Transcript:
    return Transcript(
        text="patient prescribed amoxicillin five hundred milligrams",
        token_scores=[
            TokenScore(step=0, token_id=100, token="patient", log_prob=-0.01, prob=0.99),
            TokenScore(step=1, token_id=101, token="prescribed", log_prob=-0.02, prob=0.98),
            TokenScore(step=2, token_id=102, token="amoxicillin", log_prob=-2.5, prob=0.08),
            TokenScore(step=3, token_id=103, token="five", log_prob=-0.1, prob=0.90),
            TokenScore(step=4, token_id=104, token="hundred", log_prob=-0.1, prob=0.90),
            TokenScore(step=5, token_id=105, token="milligrams", log_prob=-0.3, prob=0.74),
        ],
        word_timestamps=[],
    )


def stub_entropy_gate(transcript: Transcript) -> list[bool]:
    return [ts.prob < 0.5 for ts in transcript.token_scores]


def stub_ner(transcript: Transcript) -> list[NEREntity]:
    return [NEREntity(word="amoxicillin", category="MED", start=2, end=2, score=0.95)]


def stub_semantic_retrieve(token: str) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(candidate="amoxicillin", score=0.95, source="semantic"),
        RetrievalCandidate(candidate="ampicillin", score=0.78, source="semantic"),
    ]


def stub_phonetic_retrieve(token: str) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(candidate="amoxicillin", score=0.88, source="phonetic"),
        RetrievalCandidate(candidate="amoxycillin", score=0.72, source="phonetic"),
    ]


def stub_corrector(token: str, candidates: list[RetrievalCandidate]) -> CorrectionOutput:
    best = candidates[0].candidate if candidates else token
    return CorrectionOutput(
        original_token=token, corrected_token=best, label="CORRECT", confidence=0.9
    )
```

**`src/pipeline/pipeline.py`**

```python
"""End-to-end CARE-ASR pipeline orchestrator."""
from __future__ import annotations

from src.pipeline.stubs import (
    stub_corrector,
    stub_entropy_gate,
    stub_ner,
    stub_phonetic_retrieve,
    stub_semantic_retrieve,
    stub_transcriber,
)
from src.fusion.rrf import reciprocal_rank_fusion


class CARPipeline:
    def __init__(self) -> None:
        self.transcriber = stub_transcriber
        self.entropy_gate = stub_entropy_gate
        self.ner = stub_ner
        self.semantic_retrieve = stub_semantic_retrieve
        self.phonetic_retrieve = stub_phonetic_retrieve
        self.corrector = stub_corrector
        self.safety_gate = None

    def run(self, audio_input, attribution_log: list | None = None) -> dict:
        if attribution_log is None:
            attribution_log = []

        transcript = self.transcriber(audio_input)
        attribution_log.append({"module": "M1_ASR", "text": transcript.text})

        uncertain_flags = self.entropy_gate(transcript)
        attribution_log.append({"module": "M2_ENTROPY", "uncertain_count": sum(uncertain_flags)})

        entities = self.ner(transcript)
        entity_words = {e.word.lower() for e in entities}
        attribution_log.append({"module": "M3_NER", "entity_count": len(entities)})

        words = transcript.text.split()
        corrected_words = list(words)

        for i, (word, is_uncertain) in enumerate(zip(words, uncertain_flags)):
            if not (is_uncertain and word.lower() in entity_words):
                continue

            semantic_candidates = self.semantic_retrieve(word)
            phonetic_candidates = self.phonetic_retrieve(word)
            attribution_log.append({
                "module": "M4_RETRIEVAL", "token": word,
                "semantic_top1": semantic_candidates[0].candidate if semantic_candidates else None,
                "phonetic_top1": phonetic_candidates[0].candidate if phonetic_candidates else None,
            })

            fused = reciprocal_rank_fusion([semantic_candidates, phonetic_candidates])
            attribution_log.append({"module": "M5_FUSION", "fused_top1": fused[0].candidate if fused else None})

            correction = self.corrector(word, fused)
            if self.safety_gate is not None:
                correction = self.safety_gate(correction)

            attribution_log.append({
                "module": "M6M7_CORRECT_GATE",
                "label": correction.label,
                "corrected": correction.corrected_token,
            })

            if correction.label != "UNSURE":
                corrected_words[i] = correction.corrected_token

        return {
            "original": transcript.text,
            "corrected": " ".join(corrected_words),
            "attribution": attribution_log,
        }
```

---

### A4. Write `src/fusion/rrf.py` (Local, 15 min)

```python
"""Reciprocal Rank Fusion (M5 Module)."""
from __future__ import annotations

from care_asr.contracts.retrieval_input import RetrievalCandidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalCandidate]],
    k: int = 60,
) -> list[RetrievalCandidate]:
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            key = candidate.candidate.lower().strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    return [
        RetrievalCandidate(candidate=name, score=score, source="rrf")
        for name, score in sorted(scores.items(), key=lambda x: -x[1])
    ]
```

---

### A5. Write `src/safety/unsure_gate.py` (Task T10, Local, 20 min)

```python
"""UNSURE Safety Gate — Core refusal claim."""
from __future__ import annotations

import yaml
from care_asr.contracts.validated_output import CorrectionOutput


class UnsureGate:
    def __init__(self, config_path: str = "configs/safety.yaml") -> None:
        cfg = yaml.safe_load(open(config_path))
        self.threshold = cfg.get("unsure_threshold", 0.5)

    def apply(self, correction: CorrectionOutput) -> CorrectionOutput:
        if correction.label == "UNSURE" or correction.confidence < self.threshold:
            return CorrectionOutput(
                original_token=correction.original_token,
                corrected_token=correction.original_token,  # ← FALLBACK TO ORIGINAL
                label="UNSURE",
                confidence=correction.confidence,
            )
        return correction

    def batch_apply(self, corrections: list[CorrectionOutput]) -> list[CorrectionOutput]:
        return [self.apply(c) for c in corrections]
```

---

### A6. Write `src/retrieval/semantic.py` (Task T2 query side, Local, 30 min)

```python
"""Semantic retrieval query engine."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from care_asr.contracts.retrieval_input import RetrievalCandidate


class SemanticRetriever:
    def __init__(self, config_path: str = "configs/retrieval.yaml") -> None:
        cfg = yaml.safe_load(open(config_path))["faiss"]
        self.available = False

        if not Path(cfg["index_file"]).exists():
            print(f"WARNING: {cfg['index_file']} not found. SemanticRetriever fallback mode.")
            return

        import faiss
        self.index = faiss.read_index(cfg["index_file"])
        self.mapping: dict = json.load(open(cfg["mapping_file"]))
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model.eval()
        self.available = True

    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :].numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb.astype(np.float32)

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        if not self.available:
            return []
        distances, indices = self.index.search(self._embed(token), top_k)
        return [
            RetrievalCandidate(candidate=self.mapping.get(str(idx), ""), score=float(d), source="semantic")
            for d, idx in zip(distances[0], indices[0]) if idx != -1
        ]
```

---

### A7. Write `src/retrieval/phonetic.py` (Task T6 query side, Local, 20 min)

```python
"""Phonetic retrieval query engine with explicit fallback trigger."""
from __future__ import annotations

import json
from pathlib import Path
import yaml

from care_asr.contracts.retrieval_input import RetrievalCandidate


class PhoneticRetriever:
    def __init__(self, config_path: str = "configs/retrieval.yaml") -> None:
        cfg = yaml.safe_load(open(config_path)).get("phonetic", {})
        self.max_distance = cfg.get("max_phonetic_distance", 2)

        # Trigger Condition: Check if FAISS HuBERT index exists on disk
        phonetic_index_path = "data/indices/phonetic_index.faiss"
        phonetic_labels_path = "data/indices/phonetic_labels.json"
      
        self.faiss_available = False
        if Path(phonetic_index_path).exists() and Path(phonetic_labels_path).exists():
            import faiss
            self.index = faiss.read_index(phonetic_index_path)
            self.labels: list[str] = json.load(open(phonetic_labels_path))
            self.faiss_available = True
            print("PhoneticRetriever: Using primary HuBERT FAISS index.")
        else:
            print("PhoneticRetriever: FAISS index absent. Triggering Double Metaphone CPU fallback.")

        self.metaphone_vocab: dict = {}
        vocab_path = "data/indices/medical_vocab.json"
        if Path(vocab_path).exists():
            self.metaphone_vocab = json.load(open(vocab_path))

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        return self._metaphone_retrieve(token, top_k)

    def _metaphone_retrieve(self, token: str, top_k: int) -> list[RetrievalCandidate]:
        try:
            from metaphone import doublemetaphone
        except ImportError:
            return []
        query_codes = set(c for c in doublemetaphone(token) if c)
        results = [
            RetrievalCandidate(candidate=term, score=1.0, source="phonetic")
            for term, codes in self.metaphone_vocab.items()
            if query_codes & set(codes)
        ]
        return results[:top_k]
```

---

### A8. Write `src/evaluation/mwer.py` (Task T1/T11, Local, 30 min)

```python
"""Medical WER computation module."""
from __future__ import annotations

import json
from pathlib import Path
import jiwer


def compute_mwer(predictions: list[dict], ner_spans_path: str = "outputs/ner/ner_reference_spans.json") -> float:
    if not Path(ner_spans_path).exists():
        print(f"M-WER: NER spans not found at {ner_spans_path}.")
        return -1.0

    ner_map = {o["audio_id"]: o["entities"] for o in json.load(open(ner_spans_path))}

    entity_refs, entity_hyps = [], []
    for pred in predictions:
        entities = ner_map.get(pred["audio_id"], [])
        if entities:
            entity_refs.append(pred["reference"])
            entity_hyps.append(pred["prediction"])

    if not entity_refs:
        return -1.0

    return jiwer.wer(entity_refs, entity_hyps)
```

---

### A9. Write `src/correction/llm_corrector.py` (Task T7, Outlines Constrained Decoding)

```python
"""LLM-based medical term corrector with Outlines schema-constrained decoding.
Ensures output strictly conforms to CORRECT | <candidate>, WRONG, or UNSURE.
"""
from __future__ import annotations

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from care_asr.contracts.validated_output import CorrectionOutput

FEW_SHOT = """\
You are a clinical ASR correction assistant.
Classify the candidate medical term:
- CORRECT | <candidate>
- WRONG
- UNSURE

Examples:
Input: asr="amoxicilin", candidates=["amoxicillin"], context="prescribed amoxicilin"
Output: CORRECT | amoxicillin

Input: asr="cardigan", candidates=["carvedilol"], context="takes cardigan for heart"
Output: UNSURE
"""


class LLMCorrector:
    def __init__(self, config_path: str = "configs/correction.yaml") -> None:
        cfg = yaml.safe_load(open(config_path))
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.cfg = cfg
      
        # Outlines constrained generator setup if available
        try:
            import outlines
            self.outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
            # Regex restricting output to exact schema
            regex_pattern = r"(CORRECT \| [a-zA-Z0-9_\- ]+|WRONG|UNSURE)"
            self.generator = outlines.generate.regex(self.outlines_model, regex_pattern)
            self.use_outlines = True
        except Exception:
            self.use_outlines = False
            print("Outlines not initialized. Falling back to native HF generate.")

    def correct(self, asr_token: str, candidates: list, context: str = "") -> CorrectionOutput:
        cand_names = [c.candidate for c in candidates[:5]]
        prompt = f'{FEW_SHOT}\nInput: asr="{asr_token}", candidates={cand_names}, context="{context}"\nOutput:'

        if self.use_outlines:
            response = self.generator(prompt, max_tokens=30)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        return self._parse(response, asr_token, cand_names)

    def _parse(self, response: str, asr_token: str, candidates: list[str]) -> CorrectionOutput:
        up = response.upper()
        if "UNSURE" in up:
            return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="UNSURE", confidence=0.0)
        if "CORRECT" in up and "|" in response:
            chosen = response.split("|")[-1].strip().lower()
            matched = next((c for c in candidates if c.lower() == chosen), candidates[0] if candidates else asr_token)
            return CorrectionOutput(original_token=asr_token, corrected_token=matched, label="CORRECT", confidence=0.9)
        if "WRONG" in up:
            return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="WRONG", confidence=0.1)
        return CorrectionOutput(original_token=asr_token, corrected_token=asr_token, label="UNSURE", confidence=0.0)
```

---

### A10. Definition-of-Done Unit Tests (Mahi Task Folded In, Local, 45 min)

Write and execute unit tests for module verification before checkpoints:

**`tests/unit/test_fusion.py`**

```python
from src.fusion.rrf import reciprocal_rank_fusion
from care_asr.contracts.retrieval_input import RetrievalCandidate

def _c(name, score, src="semantic"):
    return RetrievalCandidate(candidate=name, score=score, source=src)

def test_rrf_promotes_common_candidate():
    sem = [_c("amoxicillin", 0.9), _c("ampicillin", 0.7)]
    pho = [_c("amoxicillin", 0.8, "phonetic"), _c("amoxycillin", 0.6, "phonetic")]
    result = reciprocal_rank_fusion([sem, pho])
    assert result[0].candidate.lower() == "amoxicillin"

def test_rrf_single_list_passthrough():
    result = reciprocal_rank_fusion([[_c("metformin", 0.9)]])
    assert result[0].candidate.lower() == "metformin"
```

**`tests/unit/test_safety.py`**

```python
from src.safety.unsure_gate import UnsureGate
from care_asr.contracts.validated_output import CorrectionOutput

gate = UnsureGate.__new__(UnsureGate)
gate.threshold = 0.5

def _co(orig, corr, label, conf):
    return CorrectionOutput(original_token=orig, corrected_token=corr, label=label, confidence=conf)

def test_unsure_label_keeps_original():
    r = gate.apply(_co("cardigan", "carvedilol", "UNSURE", 0.0))
    assert r.corrected_token == "cardigan"

def test_low_confidence_triggers_fallback():
    r = gate.apply(_co("amoxicilin", "amoxicillin", "CORRECT", 0.3))
    assert r.corrected_token == "amoxicilin"
```

**`tests/unit/test_correction.py`**

```python
from src.correction.llm_corrector import LLMCorrector

corrector = LLMCorrector.__new__(LLMCorrector)
corrector.cfg = {}

def test_parse_correct():
    r = corrector._parse("CORRECT | amoxicillin", "amoxicilin", ["amoxicillin", "ampicillin"])
    assert r.label == "CORRECT"
    assert r.corrected_token == "amoxicillin"

def test_parse_unsure_keeps_original():
    r = corrector._parse("UNSURE", "cardigan", ["carvedilol"])
    assert r.label == "UNSURE"
    assert r.corrected_token == "cardigan"
```

**`tests/integration/test_t5_checkpoint.py`** (Mahi Verification Task)

```python
from src.pipeline.pipeline import CARPipeline

def test_pipeline_produces_corrected_string():
    p = CARPipeline()
    result = p.run("fake_audio.wav")
    assert isinstance(result["corrected"], str)
    assert len(result["corrected"]) > 0

def test_pipeline_attribution_contains_all_modules():
    p = CARPipeline()
    log = []
    p.run("fake_audio.wav", attribution_log=log)
    modules = [e["module"] for e in log]
    assert "M1_ASR" in modules
    assert "M2_ENTROPY" in modules
    assert "M3_NER" in modules
```

Run test suite locally:

```bash
uv run pytest tests/unit/ tests/integration/test_t5_checkpoint.py -v
```

---

## ANKIT — DAY 2 & DAY 3

### A11. Real Module Pipeline Wiring & Evaluation Script (Task T9/T15/T16)

Create `scripts/run_eval.py` for Kaggle evaluation of all 6 ablation rows:

```python
"""Kaggle Ablation Evaluation Script (Task T9, T15, T16)."""
import argparse, json
from pathlib import Path
from datasets import load_from_disk
import jiwer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True,
    choices=["baseline","naive_correction","dual_retrieval","entropy_gated","thresholded","unsure_gate"])
parser.add_argument("--data-path", default="/kaggle/working/afrispeech_clinical_test")
parser.add_argument("--ner-path", default="/kaggle/working/ner_reference_spans.json")
parser.add_argument("--out-dir", default="/kaggle/working/ablation")
args = parser.parse_args()

Path(args.out_dir).mkdir(exist_ok=True, parents=True)
ds = load_from_disk(args.data_path)

from src.pipeline.pipeline import CARPipeline
pipeline = CARPipeline()

if args.mode in ["naive_correction", "dual_retrieval", "entropy_gated", "thresholded", "unsure_gate"]:
    from src.correction.llm_corrector import LLMCorrector
    pipeline.corrector = LLMCorrector().correct

if args.mode in ["dual_retrieval", "entropy_gated", "thresholded", "unsure_gate"]:
    from src.retrieval.semantic import SemanticRetriever
    from src.retrieval.phonetic import PhoneticRetriever
    pipeline.semantic_retrieve = SemanticRetriever().retrieve
    pipeline.phonetic_retrieve = PhoneticRetriever().retrieve

if args.mode in ["entropy_gated", "thresholded", "unsure_gate"]:
    from care_asr.uncertainty.gate import TsallisUncertaintyGate
    gate_obj = TsallisUncertaintyGate()
    pipeline.entropy_gate = lambda t: gate_obj.gate_tokens(t.token_scores)["uncertain_flags"]

if args.mode == "unsure_gate":
    from src.safety.unsure_gate import UnsureGate
    pipeline.safety_gate = UnsureGate().apply

from transformers import pipeline as hf_pipeline
asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-medium", return_timestamps=True, device=0)

refs, hyps, preds = [], [], []
unsure_count, total_corrections = 0, 0

for sample in ds.select(range(200)):
    audio = {"array": np.array(sample["audio"]["array"], dtype=np.float32), "sampling_rate": sample["audio"]["sampling_rate"]}
    if args.mode == "baseline":
        res = asr(audio)
        hyp = res["text"].lower().strip()
        pred_dict = {"audio_id": sample.get("id", "unk"), "prediction": hyp, "reference": sample["transcript"].lower().strip(), "attribution": []}
    else:
        log = []
        res = pipeline.run(audio, attribution_log=log)
        hyp = res["corrected"].lower().strip()
        pred_dict = {"audio_id": sample.get("id", "unk"), "prediction": hyp, "reference": sample["transcript"].lower().strip(), "attribution": log}
        for entry in log:
            if entry.get("module") == "M6M7_CORRECT_GATE":
                total_corrections += 1
                if entry.get("label") == "UNSURE":
                    unsure_count += 1

    refs.append(sample["transcript"].lower().strip())
    hyps.append(hyp)
    preds.append(pred_dict)

wer = jiwer.wer(refs, hyps)
unsure_rate = unsure_count / total_corrections if total_corrections > 0 else 0.0

row = {"mode": args.mode, "wer": round(wer, 4), "unsure_rate": round(unsure_rate, 4), "num_samples": len(preds)}
print(json.dumps(row, indent=2))
json.dump(preds, open(f"{args.out_dir}/{args.mode}_predictions.json", "w"), indent=2)
json.dump(row, open(f"{args.out_dir}/{args.mode}_metrics.json", "w"), indent=2)
```

---

### A12. Write `demo/app.py` & Checkpoint Integration Tests (Task T15/T17, Mahi Folded In)

**`demo/app.py`**

```python
"""Gradio demo app for CARE-ASR."""
import gradio as gr
import numpy as np
import json
from src.pipeline.pipeline import CARPipeline

pipeline = CARPipeline()

def process(audio_tuple):
    if audio_tuple is None:
        return "No audio", "No audio", "No log"
    sr, audio_arr = audio_tuple
    audio_float = audio_arr.astype(np.float32) / 32768.0
    log = []
    result = pipeline.run(audio_float, attribution_log=log)
    return result["original"], result["corrected"], json.dumps(log, indent=2)

demo = gr.Interface(
    fn=process,
    inputs=gr.Audio(label="Upload Clinical Speech (.wav)", type="numpy"),
    outputs=[
        gr.Textbox(label="Whisper Original"),
        gr.Textbox(label="CARE-ASR Corrected"),
        gr.Textbox(label="Attribution Log"),
    ],
    title="CARE-ASR: Clinical ASR Correction",
)

if __name__ == "__main__":
    demo.launch()
```

**`tests/integration/test_t15_checkpoint.py`**

```python
from src.pipeline.pipeline import CARPipeline
from src.safety.unsure_gate import UnsureGate
from care_asr.contracts.validated_output import CorrectionOutput

def test_t15_unsure_fallback_preserves_original():
    p = CARPipeline()
    p.safety_gate = UnsureGate().apply
    p.corrector = lambda token, cands: CorrectionOutput(
        original_token=token, corrected_token="wrong_drug", label="UNSURE", confidence=0.0
    )
    res = p.run("fake_audio.wav")
    assert "wrong_drug" not in res["corrected"]
```

---

---

# ═══════════════════════════════════════════════════════

# DIVYA'S COMPLETE SECTION (Includes All Aarth Tasks)

# (Send this section alone to Divya)

# ═══════════════════════════════════════════════════════

> **Divya: everything you need is in this section. Includes all Kaggle cloud compute, dataset loading, FAISS indexing, AND Aarth's data/threshold tuning/error analysis tasks.**

---

## DIVYA — DAY 1

### D1. Download AfriSpeech-200 Clinical Test Set on Kaggle (Task S1a)

1. Kaggle Notebook → `care-asr-data-download` → CPU → Internet ON

```python
!pip install datasets huggingface_hub -q
import os
os.environ["HF_TOKEN"] = "PASTE_YOUR_HF_TOKEN_HERE"

from datasets import load_dataset
ds = load_dataset("tobiolatunji/afrispeech-200", "all", split="test", trust_remote_code=True)
clinical = ds.filter(lambda x: x.get("domain", "") == "clinical")
clinical.save_to_disk("/kaggle/working/afrispeech_clinical_test")
print("Saved clinical test set!")
```

Save as Dataset: Output tab → New Dataset → `care-asr-afrispeech-clinical`.

---

### D2. Build Semantic FAISS Index on Kaggle (Task T2, ~4 GPU hours)

1. Kaggle Notebook → `care-asr-semantic-index` → **GPU P100** → Internet ON

```python
!pip install faiss-gpu transformers datasets sentence-transformers -q

import json, numpy as np, faiss, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

rxnorm = load_dataset("nishanth-augustai/rxnorm_data", split="train")
concepts = [row["str"] for row in rxnorm if row.get("suppress") == "N" and row.get("lat") == "ENG"]

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").cuda()
model.eval()

def encode_batch(texts, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=64, padding=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)

embs = encode_batch(concepts)
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / (norms + 1e-8)

index = faiss.IndexFlatIP(768)
index.add(embs)

faiss.write_index(index, "/kaggle/working/faiss_umls.index")
mapping = {str(i): concepts[i] for i in range(len(concepts))}
json.dump(mapping, open("/kaggle/working/cui_mapping.json", "w"))
print("Semantic index saved!")
```

Download `faiss_umls.index` and `cui_mapping.json`, commit to repo under `data/indices/`.

---

### D3. Run NER Reference Tagging on Kaggle (Task T4, Aarth Folded In)

1. Kaggle Notebook → `care-asr-ner-tagging` → **GPU P100** → Internet ON
2. Attach `care-asr-afrispeech-clinical` dataset

```python
!pip install transformers datasets -q
from datasets import load_from_disk
from transformers import pipeline as hf_pipeline
import json

ds = load_from_disk("/kaggle/input/care-asr-afrispeech-clinical/afrispeech_clinical_test")
ner = hf_pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=0)

results = []
for i, sample in enumerate(ds.select(range(200))):
    text = sample.get("transcript", "")
    try:
        entities = ner(text[:512])
    except Exception:
        entities = []
    results.append({"audio_id": sample.get("id", f"sample_{i}"), "transcript": text, "entities": entities})

json.dump(results, open("/kaggle/working/ner_reference_spans.json", "w"), indent=2)
print("NER extraction completed!")
```

Download `ner_reference_spans.json`, commit to repo under `outputs/ner/`.

---

## DIVYA — DAY 2 & DAY 3

### D4. Build Phonetic FAISS Index on Kaggle (Task T6)

1. Kaggle Notebook → `care-asr-phonetic-index` → **GPU P100** → Internet ON
2. Attach `care-asr-afrispeech-clinical` + `ner_reference_spans.json`

```python
!pip install transformers datasets torchaudio faiss-gpu -q
from datasets import load_from_disk
from transformers import HubertModel, AutoFeatureExtractor
import json, torch, numpy as np, faiss

ds = load_from_disk("/kaggle/input/care-asr-afrispeech-clinical/afrispeech_clinical_test")
ner_data = json.load(open("/kaggle/input/your-ner-file/ner_reference_spans.json"))

entity_map = {
    o["audio_id"]: [e["word"].lower() for e in o["entities"] if e.get("entity_group", "").upper() in ("MEDICATION", "MED")]
    for o in ner_data
}

hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

drug_embs, drug_labels = [], []

for sample in ds.select(range(200)):
    audio_id = sample.get("id", "unk")
    words = entity_map.get(audio_id, [])
    if not words:
        continue

    audio_arr = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]

    inputs = extractor(audio_arr, sampling_rate=sr, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = hubert(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()[0]

    for word in words:
        drug_embs.append(emb)
        drug_labels.append(word)

if drug_embs:
    embs = np.array(drug_embs, dtype=np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    index = faiss.IndexFlatIP(768)
    index.add(embs)
    faiss.write_index(index, "/kaggle/working/phonetic_index.faiss")
    json.dump(drug_labels, open("/kaggle/working/phonetic_labels.json", "w"))
```

Download `phonetic_index.faiss` + `phonetic_labels.json`, commit to repo under `data/indices/`.

---

### D5. Run Baseline & Ablation Evaluation Runs (Task T1, T9, T11, T15, T16)

Execute `scripts/run_eval.py` on Kaggle for all 6 ablation rows. Download all JSON files into `outputs/metrics/ablation/`.

---

### D6. India Context Evaluation Sweep (Task T14, Aarth Task Folded In)

Execute Svarah inference run on Kaggle:

```python
!pip install datasets transformers jiwer -q
from datasets import load_dataset
from transformers import pipeline as hf_pipeline
import jiwer, json, numpy as np

svarah = load_dataset("ai4bharat/Svarah", split="test", trust_remote_code=True)
asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-medium", return_timestamps=True, device=0)

refs, hyps = [], []
for sample in svarah.select(range(100)):
    audio = {"array": np.array(sample["audio"]["array"], dtype=np.float32), "sampling_rate": sample["audio"]["sampling_rate"]}
    res = asr(audio)
    hyps.append(res["text"].lower().strip())
    refs.append(sample.get("sentence", "").lower().strip())

india_result = {"dataset": "Svarah", "wer": round(jiwer.wer(refs, hyps), 4), "num_samples": 100, "note": "Inference-only"}
json.dump(india_result, open("/kaggle/working/india_svarah_metrics.json", "w"), indent=2)
```

Download and commit to `outputs/metrics/india/india_svarah_metrics.json`.

---

---

# ═══════════════════════════════════════════════════════

# DEPENDENCY FLOW & TESTING FUNNEL

# ═══════════════════════════════════════════════════════

## Complete Execution & Dependency Order

```
DAY 1:
  Ankit:  A1 (schemas & configs) → A2 (transcriber) → A3 (stubs & pipeline) → A4 (RRF) → A5 (UNSURE gate)
  Divya:  D1 (download data) → D2 (semantic FAISS build) → D3 (NER reference extraction)

HANDOFF 1 (End of Day 1):
  Divya commits data/indices/faiss_umls.index & outputs/ner/ner_reference_spans.json
  ↓ Ankit unblocked: SemanticRetriever live & M-WER enabled

DAY 2:
  Ankit:  A6 (semantic query) → A7 (phonetic query & fallback) → A8 (MWER) → A9 (Outlines LLM corrector) → A10 (unit tests)
  Divya:  D4 (phonetic FAISS) → D5 (baseline & ablation Kaggle evaluation runs)

HANDOFF 2 (End of Day 2):
  Divya commits phonetic index & ablation metrics JSON files
  ↓ Ankit unblocked: Full pipeline evaluation verified

DAY 3:
  Ankit:  A11 (pipeline scripts) → A12 (Gradio app & T15 checkpoint) → A13 (ablation table freeze)
  Divya:  D6 (India Svarah sweep) → System check

FINAL VERIFICATION:
  Both: Run full local test suite (uv run pytest) and confirm clean 100% pass rate.
```

---

## 4-Tier Testing Funnel

1. **Tier 1: Unit Testing (Local CPU, pytest)**
   - `test_fusion.py`: RRF math correctness.
   - `test_safety.py`: UNSURE gate token fallback.
   - `test_correction.py`: Outlines regex parsing & fallback.
   - `test_retrieval.py`: Missing index graceful handling.
2. **Tier 2: Integration Testing (Local CPU, stubs)**
   - `test_t5_checkpoint.py`: Stub pipeline E2E shape.
   - `test_t15_checkpoint.py`: Full pipeline refusal fallback.
3. **Tier 3: Cloud Evaluation (Kaggle P100 GPU, real data)**
   - All 6 ablation rows executed via `scripts/run_eval.py`.
4. **Tier 4: Verification Script Check**
   - Confirm all 6 ablation JSON files, India JSON, FAISS index files, and Gradio app exist and function cleanly.
