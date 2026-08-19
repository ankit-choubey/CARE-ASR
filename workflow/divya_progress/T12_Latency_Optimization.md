# CARE-ASR Task T12: Latency Optimization

**Status**: Completed & Verified ✅
**Module**: `src/retrieval`, `src/pipeline`, `scripts`
**Target Audience**: Project maintainers and teammates reviewing the PR

---

## 1. Objective

Task T12 is a latency pass on the fused retrieval layer:

1. Batch FAISS queries (semantic + phonetic) so a single index search serves many tokens.
2. Cache query embeddings / phonetic encodings so repeated tokens are never re-encoded.
3. Instrument the pipeline (entropy gate, semantic retrieval, phonetic retrieval, fusion) with timing.
4. Provide an end-to-end benchmark that validates the latency claim under real measurement.

All existing public interfaces are preserved; this is a pure performance + instrumentation pass with no behavior changes.

---

## 2. Files Created

- `src/retrieval/latency.py`
- `scripts/run_latency_benchmark.py`
- `tests/unit/test_latency.py`

## 3. Files Modified

- `src/retrieval/semantic.py`
- `src/retrieval/phonetic.py`
- `src/pipeline/pipeline.py`
- `configs/retrieval.yaml` (additive keys only)
- `tests/unit/test_retrieval.py`

---

## 4. Major Components Implemented

| Component | Responsibility |
| :--- | :--- |
| `SemanticRetriever.retrieve_many()` | Batched FAISS search: deduplicates identical tokens, reuses cached embeddings, performs ONE `index.search()` for the batch, maps results back to input order. |
| `SemanticRetriever._embed_batch()` | Embeds a batch with one tokenizer call + one transformer forward pass (masked mean pooling, matching index construction). |
| Bounded embedding cache (`_embed_cache`) | Per-instance LRU cache keyed by normalized query token; configurable via `faiss.embedding_cache_maxsize`. |
| Lazy transformers imports | Module stays importable even when the `transformers` install is broken; imports happen at `SemanticRetriever.__init__()` with the same `RuntimeError` contract. |
| `PhoneticRetriever.retrieve_many()` | Batched phonetic lookup: deduplicates identical tokens, reuses cached Double Metaphone encodings, preserves order and duplicates. |
| Shared `DoubleMetaphone` instance | One instance per `PhoneticRetriever` instead of per-query construction. |
| Bounded encoding cache (`_encoding_cache`) | Instance-scoped LRU of normalized token → metaphone codes; configurable via `phonetic.encoding_cache_maxsize` (default 1000). |
| `LatencyStats` (`src/retrieval/latency.py`) | Reusable timing helper (`start`/`stop`/`record`/`summary`) used by the pipeline and benchmark. |
| Pipeline instrumentation | M2 entropy-gate latency, M4 semantic/phonetic/total retrieval latency, M5 fusion latency; timing appended to `attribution_log` as `gate_latency_ms`, `semantic_retrieval_latency_ms`, `phonetic_retrieval_latency_ms`, `retrieval_latency_ms`, `fusion_latency_ms`. |
| `scripts/run_latency_benchmark.py` | End-to-end benchmark: load dataset → run gate + retrieval + fusion → aggregate statistics (mean/median/p50/p90/p95/min/max, cache hit rates, totals) → console report + JSON export to `outputs/latency_reports/` (`indent=2`, `ensure_ascii=False`). |

---

## 5. Existing Interfaces Reused / Preserved

- `SemanticRetriever.retrieve()` / `PhoneticRetriever.retrieve()` — preserved; now delegate to `retrieve_many()`.
- `CARPipeline.run()` — signature, return schema, and correction/fusion logic unchanged.
- `RetrievalCandidate` contract — unchanged.
- `retrieve_many()` is used when available, with automatic fallback to sequential `retrieve()` for compatibility.

---

## 6. Tests Added

- `tests/unit/test_retrieval.py` — retrieve_many equals sequential retrieve, duplicate tokens embed/encode once, cache hit avoids recomputation, empty input, ordering preserved, duplicate outputs preserved, top_k respected, retrieve() backward compatibility, cache eviction, cache size respected, batching.
- `tests/unit/test_latency.py` — timing helper records values, attribution log contains required fields, retrieve_many path used when available, retrieve fallback works, non-negative timings, pipeline output unchanged, deterministic repeated runs.

Current totals (verified):

```text
tests/unit/test_retrieval.py ..... 22 passed
tests/unit/test_latency.py ....... 14 passed
```

---

## 7. Validation Performed

- `pytest tests/unit/test_retrieval.py tests/unit/test_latency.py` — all passed
- `ruff check src/retrieval tests/unit/test_retrieval.py tests/unit/test_latency.py src/pipeline` — all checks passed
- `black --check` — clean
- `mypy --strict` on touched modules — no issues in the implemented files
- Combined T11/T12/T14 regression run: **95 passed**
- `python scripts/run_latency_benchmark.py --help` — works
- Benchmark run on a small synthetic dataset — deterministic output, JSON report exported

---

## 8. Final Outcome

- Batching and caching reduce repeated embedding/encoding and FAISS searches while preserving identical retrieval results.
- Latency instrumentation is attached to the pipeline without changing any existing attribution entries.
- The benchmark validates the gating latency claim under real measurement.

---

## 9. Commit / PR Summary

- Commits:
  - `df96e33` / `5781491` — feat(T12): add batched semantic retrieval with embedding cache
  - `6be95f9` — style: format phonetic retriever with Black
  - `04d2363` — feat(T12): implement latency optimization and benchmarking
- PRs: `#10`, `#11` (merged into main via `d98ea49`)

---

## 10. Current Status

**✅ Completed** — Implementation, tests, and validation complete.
