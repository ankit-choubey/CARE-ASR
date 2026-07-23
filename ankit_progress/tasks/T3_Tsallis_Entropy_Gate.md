# CARE-ASR Task T3: Tsallis Entropy Gate

**Status**: Completed & Verified  
**Module**: `care_asr.uncertainty`  
**Target Branch**: `ankit`  

---

## 1. Beginner-Friendly Explanation of Entropy & Why Task T3 Exists

### What is Entropy?
Imagine taking an exam with a multiple-choice question.
- If you are 100% sure option **(A)** is correct, your uncertainty is **zero**.
- If you have no idea and guess equally between **(A), (B), (C), and (D)**, your uncertainty is **maximum**.

In Automatic Speech Recognition (ASR), **entropy** is a mathematical formula that calculates how uncertain a machine learning model (like Whisper-medium) is when predicting the next word from speech audio.

### Why Does Uncertainty Matter?
Automatic speech recognition models struggle with clinical and accented speech (e.g. AfriSpeech dataset). When Whisper mishears a complex medical drug name (like *amoxicillin* or *hydrochlorothiazide*), it spreads its probability across multiple tokens instead of concentrating it on one word.

### Why Should Retrieval Only Happen for Uncertain Words?
Querying external clinical knowledge bases (like FAISS vector databases or BioBERT/ClinicalBERT) for every single word is computationally expensive and slow.
- Common words like `"the"`, `"and"`, `"patient"` have near-zero uncertainty and do not need retrieval.
- Rare or misheard clinical terms have high uncertainty.
The **Uncertainty Gate** acts as an intelligent traffic filter: it lets clear words pass through instantly and routes only uncertain words to the retrieval pipeline.

### Why CARE-ASR Uses Tsallis Entropy ($\alpha = 1/3$)
Standard Shannon entropy uses logarithmic scaling ($-\sum P_i \ln P_i$). In large vocabularies ($V = 51,865$ in Whisper), Shannon entropy can obscure subtle uncertainty when top-1 probability appears moderately high.

Tsallis entropy is a non-extensive generalization:
$$H_\alpha(P) = \frac{1}{\alpha - 1} \left( 1 - \sum_{i=1}^V P_i^\alpha \right)$$

Setting entropic index $\alpha = 1/3$ amplifies sensitivity to long-tail probability distributions. This makes CARE-ASR exceptionally sensitive to minor model hesitation on clinical entities.

### How Module T3 Fits in the Pipeline
```
[ Audio Input ] 
       │
       ▼
[ Whisper-medium ] ──(decoder logit scores)──► [ T3: Tsallis Entropy Gate ]
                                                        │
                                   ┌────────────────────┴────────────────────┐
                                   │                                         │
                             Entropy < Threshold                     Entropy >= Threshold
                                   │                                         │
                                   ▼                                         ▼
                           [ Pass Transcription ]                   [ T5: Clinical Retrieval ]
```

---

## 2. Mathematical Intuition

For probability vector $P = [P_1, P_2, \dots, P_V]$ where $\sum P_i = 1.0$:
- At $\alpha = 1/3$, $\alpha - 1 = -2/3$, giving:
$$H_{1/3}(P) = \frac{3}{2} \left( \sum_{i=1}^V P_i^{1/3} - 1 \right)$$

- **Confident Case**: $P = [1.0, 0.0, \dots, 0.0] \implies \sum P_i^{1/3} = 1.0 \implies H_{1/3}(P) = 0.0$.
- **Uniform Case**: $P_i = 1/V \implies \sum P_i^{1/3} = V \cdot V^{-1/3} = V^{2/3} \implies H_{1/3}(P) = 1.5 (V^{2/3} - 1) \gg 0$.

---

## 3. Project Structure & Files Created

```
CARE-ASR/
├── care_asr/
│   └── uncertainty/
│       ├── __init__.py
│       ├── tsallis_entropy.py
│       └── gate.py
├── tests/
│   ├── test_tsallis_entropy.py
│   └── test_uncertainty_gate.py
└── ankit_progress/
    ├── tasks/
    │   └── T3_Tsallis_Entropy_Gate.md
    └── prompts/
        └── T3_PR_DESCRIPTION.md
```

### Files Created

1. **[care_asr/uncertainty/tsallis_entropy.py](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/care_asr/uncertainty/tsallis_entropy.py)**: Core numerical functions (`softmax`, `compute_tsallis_entropy`, `compute_batch_entropy`).
2. **[care_asr/uncertainty/gate.py](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/care_asr/uncertainty/gate.py)**: Decision gating functions and class (`is_uncertain`, `gate_tokens`, `TsallisUncertaintyGate`).
3. **[care_asr/uncertainty/__init__.py](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/care_asr/uncertainty/__init__.py)**: Package initializer exposing public API.
4. **[tests/test_tsallis_entropy.py](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/tests/test_tsallis_entropy.py)**: Unit test suite for math and stability.
5. **[tests/test_uncertainty_gate.py](file:///Users/theankit/Documents/AK/Projects/CARE-ASR/tests/test_uncertainty_gate.py)**: Unit test suite for threshold decisions and gating reports.

---

## 4. Implemented Functions & Interfaces

| Function / Class | Purpose | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `softmax()` | Converts unnormalized logits to probabilities | `logits: Tensor/ndarray`, `dim: int` | `probs: Tensor/ndarray` |
| `compute_tsallis_entropy()` | Computes Tsallis entropy for probability vector | `probs`, `alpha: float=1/3`, `eps: float=1e-12` | `entropy: float/Tensor` |
| `compute_batch_entropy()` | Computes per-token entropy across decoder steps | `scores: List/Tensor/ndarray`, `alpha` | `entropies: Tensor/ndarray` |
| `is_uncertain()` | Checks if entropy score exceeds threshold | `entropy`, `threshold: float=0.5` | `bool / Tensor / ndarray` |
| `gate_tokens()` | Full sequence gating and structured report | `token_scores`, `threshold`, `alpha` | `Dict[str, Any]` decision report |
| `TsallisUncertaintyGate` | Object instance with dynamic threshold tuning | `threshold: float`, `alpha: float` | Gate object |

---

## 5. Unit Tests & Execution Results

All 21 test cases passed cleanly locally and in CI environment:

```
============================= test session starts ==============================
platform darwin -- Python 3.11.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/theankit/Documents/AK/Projects/CARE-ASR
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, anyio-4.14.2
collected 21 items

tests/test_baseline.py .....                                             [ 23%]
tests/test_metrics.py ......                                             [ 52%]
tests/test_tsallis_entropy.py .......                                   [ 85%]
tests/test_uncertainty_gate.py ...                                       [100%]

============================== 21 passed in 23.14s ==============================
```

---

## 6. Git Workflow & Commit Hash

```bash
git checkout main
git pull origin main
git checkout ankit
git merge main
git add .
git commit -m "T3: Implement Tsallis entropy gate"
git push origin ankit
```

**Generated Commit Hash**: `aecbc1b4cc677edaaa572bc2e74e51a2b64bcb82` (short `aecbc1b`)

---

## 7. Known Limitations & Next Dependency (Task T5)

- **Threshold Independence**: Threshold $0.5$ is a initial default; dynamic threshold calibration across accent/clinical categories will be performed in Task T8.
- **Next Dependency (Task T5)**: Module T3 outputs `uncertain_indices` and `uncertain_flags` which will be consumed by Task T5 (Clinical Entity Retrieval Engine).

---

## 8. Merge Readiness Checklist

- [x] Code follows Black + Ruff format guidelines.
- [x] Full docstrings with beginner-friendly explanations.
- [x] All 21 pytest unit tests pass cleanly.
- [x] No breaking contract changes to prior modules (T1/S3).
