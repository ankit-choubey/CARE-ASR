# CARE-ASR Task T2: Semantic Index Builder

**Status**: Completed & Verified ✅  
**Module**: `scripts`  
**Target Audience**: Project maintainers and teammates reviewing the PR  

---

## 0. Post-Task Update (T12)

Task T12 (Latency Optimization) extended the runtime consumer `src/retrieval/semantic.py` with:

- `retrieve_many()` — batched FAISS search over many query tokens in a single `index.search()`.
- `_embed_batch()` — batched ClinicalBERT embedding (one tokenizer call + one forward pass).
- A bounded per-instance query-embedding cache (`faiss.embedding_cache_maxsize` in `configs/retrieval.yaml`, additive keys only).
- Lazy `transformers` imports so the module stays importable when the install is broken.

These changes do NOT alter the T2 index builder (`scripts/build_semantic_index.py`) or its artifacts (`faiss_umls.index`, `cui_mapping.json`). Retrieval results are identical; only latency improved.

---

## 1. Objective

Task T2 implements the semantic index builder used to convert structured clinical concepts into dense vector representations and store them in a searchable FAISS index.

This solves the retrieval preparation problem in CARE-ASR by building the offline artifacts needed for semantic concept lookup during downstream clinical retrieval.

---

## 2. Files Modified

- `scripts/build_semantic_index.py`

---

## 3. Functions Implemented

The following functions were fully implemented in `scripts/build_semantic_index.py`:

| Function | Responsibility |
| :--- | :--- |
| `load_configs()` | Loads retrieval configuration required by the semantic indexing pipeline. |
| `load_clinical_bert()` | Loads the `emilyalsentzer/Bio_ClinicalBERT` model and tokenizer for clinical concept embeddings. |
| `load_concepts()` | Loads RxNorm concepts from the Hugging Face dataset and prepares them for indexing. |
| `encode_concepts()` | Encodes concept text into dense embeddings using ClinicalBERT with batching support. |
| `build_faiss_index()` | Builds the FAISS similarity index from normalized concept embeddings using `IndexFlatIP`. |
| `save_index()` | Saves the generated FAISS index artifact to disk. |
| `save_mapping()` | Saves the concept-to-index mapping as JSON for later retrieval use. |
| `main()` | Orchestrates the full end-to-end semantic index build pipeline. |

---

## 4. Dataset Used

- Hugging Face dataset: `nishanth-augustai/rxnorm_data`
- Clinical embedding model: `emilyalsentzer/Bio_ClinicalBERT`

The RxNorm dataset is used as the source of clinical concepts to index. Bio_ClinicalBERT is used to generate domain-specific embeddings so concept retrieval is grounded in clinical language rather than general-purpose text representations.

---

## 5. Implementation Details

- Loads retrieval configuration
- Loads Bio_ClinicalBERT
- Loads RxNorm concepts from Hugging Face
- Filters English and active concepts
- Encodes concepts using ClinicalBERT
- Uses masked mean pooling
- Applies L2 normalization
- Builds FAISS `IndexFlatIP`
- Saves FAISS index
- Saves concept mapping JSON
- Uses batching
- Supports CPU and CUDA automatically
- Includes runtime validation
- Includes descriptive `RuntimeError` handling
- Includes progress logging
- Uses `tqdm` progress bars

---

## 6. Improvements Made During Development

- Added automatic CPU/GPU device selection
- Added explicit tensor device movement
- Replaced FAISS index creation with `IndexFlatIP` implementation
- Removed temporary development dataset limit before final version
- Added CLI progress messages
- Improved docstrings
- Fixed import ordering
- Preserved full type hints
- Preserved validation checks

---

## 7. Validation Performed

- Python syntax check passed
- Ruff checks passed
- End-to-end semantic index pipeline executed
- FAISS index generated successfully
- Mapping JSON generated successfully
- Offline project tests executed

Pytest result:

```text
25 passed in 42.83s
```

---

## 8. Generated Artifacts

The pipeline generates:

- `data/indices/faiss_umls.index`
- `data/indices/cui_mapping.json`

---

## 9. Commands Used

```bash
python scripts/build_semantic_index.py
pytest
ruff check .
```
